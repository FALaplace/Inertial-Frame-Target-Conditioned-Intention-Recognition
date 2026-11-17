from __future__ import annotations

import json
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

from analytic_layer import FEATURE_NAMES, build_feature_sample, infer_counterpart_is_region
from orbit_core import TrajectoryPoint, apply_impulse_maneuver, parse_epoch


LABEL_KEYS = tuple(f"c{i}" for i in range(8))

ANALYTIC_SEQUENCE_FEATURES = (
    "time_offset_s",
    "distance_km",
    "relative_speed_km_s",
    "range_rate_km_s",
    "closing_speed_km_s",
    "actor_radius_km",
    "counterpart_radius_km",
    "altitude_gap_km",
    "plane_angle_deg",
    "surface_distance_km",
    "elevation_deg",
    "access_flag",
)

MANEUVER_SEQUENCE_FEATURES = (
    "maneuver_flag",
    "time_from_maneuver_s",
    "delta_v_x_km_s",
    "delta_v_y_km_s",
    "delta_v_z_km_s",
)

MANEUVER_GLOBAL_FEATURES = (
    "has_maneuver",
    "delta_v_norm_km_s",
    "delta_v_sigma_km_s",
    "maneuver_confidence",
)

ANALYTIC_GLOBAL_FEATURES = (
    "min_distance_km",
    "relative_speed_at_min_km_s",
    "dwell_700_s",
    "dwell_1200_s",
    "min_surface_distance_km",
    "access_duration_s",
    "access_pass_count",
    "max_elevation_deg",
    "maneuver_approach_gain_km",
    "coverage_gain_s",
)

ERROR_FEATURES = (
    "orbit_position_sigma",
    "orbit_velocity_sigma",
    "maneuver_time_sigma",
)


def iter_json_samples(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Stream the project's one-sample-per-line JSON array without loading it whole."""
    if not path.is_file():
        raise FileNotFoundError(path)

    sample_index = 0
    opened = False
    closed = False
    with path.open("r", encoding="utf-8") as fp:
        for line_number, raw_line in enumerate(fp, start=1):
            text = raw_line.strip()
            if not text:
                continue
            if text == "[":
                opened = True
                continue
            if text == "]":
                closed = True
                break
            if not opened:
                raise ValueError(f"Expected '[' before sample data at line {line_number}.")
            if text.endswith(","):
                text = text[:-1]
            try:
                sample = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid sample JSON at line {line_number}.") from exc
            if not isinstance(sample, dict):
                raise ValueError(f"Sample {sample_index} must be a JSON object.")
            yield sample_index, sample
            sample_index += 1

    if not opened or not closed:
        raise ValueError("Dataset must be a complete JSON array.")


def load_label_overrides(dataset_path: Path) -> dict[int, dict[str, float]]:
    """Load accepted sidecar labels produced by the daily labeling workflow."""
    directory = dataset_path.parent
    paths = []
    pilot_path = directory / "soft_labels_pilot_50_subagent.json"
    if pilot_path.is_file():
        paths.append(pilot_path)
    paths.extend(sorted(directory.glob("soft_labels_batch_*_subagent_*.json")))

    labels: dict[int, dict[str, float]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Label sidecar must contain a JSON list: {path}")
        is_pilot = path == pilot_path
        for row in payload:
            if not isinstance(row, dict):
                raise ValueError(f"Invalid label row in {path}")
            if not is_pilot and row.get("review_status") != "accepted":
                raise ValueError(f"Unaccepted label row found in final sidecar: {path}")
            index = int(row["sample_index"])
            y = _validate_soft_label(row.get("y"), index)
            if index in labels and labels[index] != y:
                raise ValueError(f"Conflicting sidecar labels for sample {index}.")
            labels[index] = y
    return labels


def iter_training_data(
    dataset_path: Path,
    sample_indices: set[int] | None = None,
    allow_sidecar_labels: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield selected labeled samples.

    ``allow_sidecar_labels=True`` preserves the historical labeling workflow.
    Every frozen formal experiment must pass ``False`` so the embedded ``y``
    and the dataset SHA-256 define the supervision source by themselves.
    """
    label_overrides = load_label_overrides(dataset_path) if allow_sidecar_labels else {}
    remaining = None if sample_indices is None else set(sample_indices)
    for sample_index, sample in iter_json_samples(dataset_path):
        if remaining is not None and sample_index not in remaining:
            continue
        source_y = sample.get("y")
        override_y = label_overrides.get(sample_index)
        if source_y is not None and override_y is not None:
            if _validate_soft_label(source_y, sample_index) != override_y:
                raise ValueError(f"Main-file and sidecar labels conflict at sample {sample_index}.")
        y = source_y if source_y is not None else override_y
        if y is None:
            continue
        labeled_sample = dict(sample)
        labeled_sample["y"] = y
        if remaining is not None:
            remaining.remove(sample_index)
        yield prepare_model_input(labeled_sample, sample_index)
    if remaining:
        raise ValueError(f"Requested sample indices were not found: {sorted(remaining)[:10]}")


def load_training_data(
    dataset_path: Path,
    limit: int | None = None,
    sample_indices: set[int] | None = None,
    allow_sidecar_labels: bool = True,
) -> list[dict[str, Any]]:
    """Materialize labeled samples; formal runs use embedded labels only."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None.")
    iterator = iter_training_data(
        dataset_path,
        sample_indices=sample_indices,
        allow_sidecar_labels=allow_sidecar_labels,
    )
    return list(iterator if limit is None else islice(iterator, limit))


def prepare_model_input(sample: dict[str, Any], sample_index: int = 0) -> dict[str, Any]:
    """Convert one raw actor-counterpart sample into model-ready NumPy arrays."""
    actor_trajectory = _trajectory(sample, "actor_trajectory", sample_index)
    target_trajectory = _trajectory(sample, "counterpart_trajectory", sample_index)
    _validate_time_grid(actor_trajectory, target_trajectory, sample_index)

    counterfactual_trajectory, counterfactual_valid = _counterfactual_actor(
        actor_trajectory,
        sample.get("maneuver_events", []),
        sample_index,
    )
    actor_factual = _trajectory_array(actor_trajectory)
    actor_counterfactual = _trajectory_array(counterfactual_trajectory)
    target = _trajectory_array(target_trajectory)

    factual_analytic = _analytic_array(sample, sample_index)
    if counterfactual_valid:
        counterfactual_sample = dict(sample)
        counterfactual_sample["actor_trajectory"] = counterfactual_trajectory
        counterfactual_sample["maneuver_events"] = []
        counterfactual_analytic = _analytic_array(counterfactual_sample, sample_index)
    else:
        counterfactual_analytic = factual_analytic.copy()

    maneuver_sequence, maneuver_global = _maneuver_arrays(
        actor_trajectory,
        sample.get("maneuver_events", []),
        sample_index,
    )
    is_region = infer_counterpart_is_region(target_trajectory)
    error_features, error_mask = _error_arrays(sample.get("error_summary", {}))
    analytic_global = _global_analytic_array(
        factual_analytic,
        counterfactual_analytic,
        maneuver_sequence,
        counterfactual_valid,
    )

    model_input = {
        "sample_index": np.int64(sample_index),
        "actor_factual": actor_factual,
        "actor_counterfactual": actor_counterfactual,
        "target": target,
        "maneuver_sequence": maneuver_sequence,
        "analytic_factual": factual_analytic,
        "analytic_counterfactual": counterfactual_analytic,
        "analytic_difference": factual_analytic - counterfactual_analytic,
        "analytic_global": analytic_global,
        "maneuver_global": maneuver_global,
        "error_features": error_features,
        "error_mask": error_mask,
        "counterpart_type": np.asarray(
            [0.0, 1.0] if is_region else [1.0, 0.0],
            dtype=np.float32,
        ),
        "counterfactual_valid": np.asarray(
            [1.0 if counterfactual_valid else 0.0],
            dtype=np.float32,
        ),
        "class_mask": np.asarray(
            [1, 1, 0, 0, 0, 1, 1, 1]
            if is_region
            else [1, 1, 1, 1, 1, 1, 0, 0],
            dtype=np.float32,
        ),
        "y": _soft_label_array(sample.get("y"), sample_index),
    }
    validate_model_input(model_input)
    return model_input


def validate_model_input(model_input: dict[str, Any]) -> None:
    """Run the smallest structural check needed before passing data to a model."""
    sequence_length = model_input["actor_factual"].shape[0]
    expected_shapes = {
        "actor_factual": (sequence_length, 6),
        "actor_counterfactual": (sequence_length, 6),
        "target": (sequence_length, 6),
        "maneuver_sequence": (sequence_length, len(MANEUVER_SEQUENCE_FEATURES)),
        "analytic_factual": (sequence_length, len(ANALYTIC_SEQUENCE_FEATURES)),
        "analytic_counterfactual": (sequence_length, len(ANALYTIC_SEQUENCE_FEATURES)),
        "analytic_difference": (sequence_length, len(ANALYTIC_SEQUENCE_FEATURES)),
        "analytic_global": (len(ANALYTIC_GLOBAL_FEATURES),),
        "maneuver_global": (len(MANEUVER_GLOBAL_FEATURES),),
        "error_features": (len(ERROR_FEATURES),),
        "error_mask": (len(ERROR_FEATURES),),
        "counterpart_type": (2,),
        "counterfactual_valid": (1,),
        "class_mask": (8,),
    }
    for key, shape in expected_shapes.items():
        value = model_input[key]
        if value.shape != shape:
            raise ValueError(f"{key} has shape {value.shape}; expected {shape}.")
        if not np.isfinite(value).all():
            raise ValueError(f"{key} contains a non-finite value.")

    if model_input["counterfactual_valid"][0] == 0.0:
        if not np.array_equal(model_input["actor_factual"], model_input["actor_counterfactual"]):
            raise ValueError("No-maneuver factual and counterfactual trajectories must match.")
        if not np.array_equal(model_input["analytic_difference"], np.zeros_like(model_input["analytic_difference"])):
            raise ValueError("No-maneuver analytic difference must be zero.")

    y = model_input["y"]
    if y is not None:
        if y.shape != (8,) or not np.isfinite(y).all():
            raise ValueError("y must be a finite eight-class vector.")
        if np.any(y < 0.0) or np.any(y > 1.0) or not np.isclose(y.sum(), 1.0):
            raise ValueError("y values must be in [0, 1] and sum to 1.")
        if np.any(y[model_input["class_mask"] == 0.0] != 0.0):
            raise ValueError("y violates the counterpart-type class mask.")


def print_model_input(model_input: dict[str, Any]) -> None:
    """Print shapes and representative values without dumping 181 full rows."""
    print("ANALYTIC_SEQUENCE_FEATURES =", ANALYTIC_SEQUENCE_FEATURES)
    print("ANALYTIC_GLOBAL_FEATURES =", ANALYTIC_GLOBAL_FEATURES)
    print("MANEUVER_SEQUENCE_FEATURES =", MANEUVER_SEQUENCE_FEATURES)
    print("ERROR_FEATURES =", ERROR_FEATURES)
    print()
    for key, value in model_input.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            if value.ndim <= 1:
                print(value)
            else:
                print("first =", value[0])
                print("last  =", value[-1])
        else:
            print(f"{key}: {value}")
        print()


def _trajectory(sample: dict[str, Any], key: str, sample_index: int) -> list[dict[str, Any]]:
    trajectory = sample.get(key)
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError(f"Sample {sample_index} has an invalid {key}.")
    return trajectory


def _validate_time_grid(
    actor: list[dict[str, Any]],
    target: list[dict[str, Any]],
    sample_index: int,
) -> None:
    if len(actor) != len(target):
        raise ValueError(f"Sample {sample_index} has mismatched trajectory lengths.")
    for step, (actor_point, target_point) in enumerate(zip(actor, target)):
        if actor_point.get("t") != target_point.get("t"):
            raise ValueError(f"Sample {sample_index} has a time-grid mismatch at step {step}.")


def _trajectory_array(trajectory: list[dict[str, Any]]) -> np.ndarray:
    rows = []
    for point in trajectory:
        r = point.get("r")
        v = point.get("v")
        if not isinstance(r, list) or not isinstance(v, list) or len(r) != 3 or len(v) != 3:
            raise ValueError("Each trajectory point must contain three-component r and v values.")
        rows.append([*r, *v])
    array = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Trajectory contains a non-finite value.")
    return array


def _counterfactual_actor(
    factual: list[dict[str, Any]],
    maneuver_events: Any,
    sample_index: int,
) -> tuple[list[dict[str, Any]], bool]:
    events = maneuver_events or []
    if not isinstance(events, list):
        raise ValueError(f"Sample {sample_index} maneuver_events must be a list.")
    if not events:
        return factual, False
    if len(events) != 1:
        raise ValueError(f"Sample {sample_index} must contain at most one maneuver event.")

    points = [
        TrajectoryPoint(
            t=parse_epoch(str(point["t"])),
            r_km=tuple(float(value) for value in point["r"]),
            v_km_s=tuple(float(value) for value in point["v"]),
        )
        for point in factual
    ]
    event = events[0]
    maneuver_time = parse_epoch(str(event["t_maneuver"]))
    if maneuver_time < points[0].t or maneuver_time > points[-1].t:
        raise ValueError(f"Sample {sample_index} maneuver time is outside the trajectory window.")
    delta_v = event.get("delta_v")
    if not isinstance(delta_v, list) or len(delta_v) != 3:
        raise ValueError(f"Sample {sample_index} delta_v must contain three values.")
    inverse_delta_v = tuple(-float(value) for value in delta_v)
    # ponytail: inverse the injected impulse; use original-TLE SGP4 only if this
    # two-body counterfactual approximation becomes a measured error source.
    counterfactual = apply_impulse_maneuver(points, maneuver_time, inverse_delta_v)
    return [point.to_json() for point in counterfactual], True


def _analytic_array(sample: dict[str, Any], sample_index: int) -> np.ndarray:
    feature_sample = build_feature_sample(sample, sample_index)
    indices = [FEATURE_NAMES.index(name) for name in ANALYTIC_SEQUENCE_FEATURES]
    return np.asarray(feature_sample["sequence"], dtype=np.float32)[:, indices]


def _maneuver_arrays(
    trajectory: list[dict[str, Any]],
    maneuver_events: Any,
    sample_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    sequence = np.zeros((len(trajectory), len(MANEUVER_SEQUENCE_FEATURES)), dtype=np.float32)
    events = maneuver_events or []
    if not events:
        return sequence, np.zeros(len(MANEUVER_GLOBAL_FEATURES), dtype=np.float32)
    if not isinstance(events, list) or len(events) != 1:
        raise ValueError(f"Sample {sample_index} must contain at most one maneuver event.")

    event = events[0]
    maneuver_time = parse_epoch(str(event["t_maneuver"]))
    times = [parse_epoch(str(point["t"])) for point in trajectory]
    event_index = min(range(len(times)), key=lambda i: abs((times[i] - maneuver_time).total_seconds()))
    delta_v = np.asarray(event["delta_v"], dtype=np.float32)
    if delta_v.shape != (3,):
        raise ValueError(f"Sample {sample_index} delta_v must contain three values.")

    sequence[:, 1] = [(time - maneuver_time).total_seconds() for time in times]
    sequence[event_index, 0] = 1.0
    sequence[event_index, 2:] = delta_v
    global_features = np.asarray(
        [
            1.0,
            float(event.get("delta_v_norm_km_s", np.linalg.norm(delta_v))),
            float(event.get("delta_v_sigma") or 0.0),
            float(event.get("confidence") or 0.0),
        ],
        dtype=np.float32,
    )
    return sequence, global_features


def _error_arrays(error_summary: Any) -> tuple[np.ndarray, np.ndarray]:
    summary = error_summary or {}
    if not isinstance(summary, dict):
        raise ValueError("error_summary must be a JSON object.")
    values = []
    mask = []
    for name in ERROR_FEATURES:
        value = summary.get(name)
        values.append(0.0 if value is None else float(value))
        mask.append(0.0 if value is None else 1.0)
    return np.asarray(values, dtype=np.float32), np.asarray(mask, dtype=np.float32)


def _global_analytic_array(
    factual: np.ndarray,
    counterfactual: np.ndarray,
    maneuver_sequence: np.ndarray,
    counterfactual_valid: bool,
) -> np.ndarray:
    index = {name: i for i, name in enumerate(ANALYTIC_SEQUENCE_FEATURES)}
    time_offsets = factual[:, index["time_offset_s"]]
    step_seconds = float(np.median(np.diff(time_offsets))) if len(time_offsets) > 1 else 0.0
    distance = factual[:, index["distance_km"]]
    counterfactual_distance = counterfactual[:, index["distance_km"]]
    closest_index = int(np.argmin(distance))
    access = factual[:, index["access_flag"]] >= 0.5
    counterfactual_access = counterfactual[:, index["access_flag"]] >= 0.5
    pass_count = int(np.count_nonzero(access & ~np.r_[False, access[:-1]]))

    approach_gain = 0.0
    coverage_gain = 0.0
    if counterfactual_valid:
        event_index = int(np.argmax(maneuver_sequence[:, 0]))
        approach_gain = float(
            np.min(counterfactual_distance[event_index:]) - np.min(distance[event_index:])
        )
        coverage_gain = float(
            (np.count_nonzero(access) - np.count_nonzero(counterfactual_access)) * step_seconds
        )

    return np.asarray(
        [
            float(distance[closest_index]),
            float(factual[closest_index, index["relative_speed_km_s"]]),
            float(np.count_nonzero(distance <= 700.0) * step_seconds),
            float(np.count_nonzero(distance <= 1200.0) * step_seconds),
            float(np.min(factual[:, index["surface_distance_km"]])),
            float(np.count_nonzero(access) * step_seconds),
            float(pass_count),
            float(np.max(factual[:, index["elevation_deg"]])),
            approach_gain,
            coverage_gain,
        ],
        dtype=np.float32,
    )


def _validate_soft_label(y: Any, sample_index: int) -> dict[str, float]:
    if not isinstance(y, dict) or tuple(y) != LABEL_KEYS:
        raise ValueError(f"Sample {sample_index} y must contain ordered keys c0-c7.")
    values = {key: float(y[key]) for key in LABEL_KEYS}
    array = np.asarray(list(values.values()), dtype=np.float64)
    if not np.isfinite(array).all() or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"Sample {sample_index} y contains an invalid probability.")
    if not np.isclose(array.sum(), 1.0):
        raise ValueError(f"Sample {sample_index} y must sum to 1.")
    return values


def _soft_label_array(y: Any, sample_index: int) -> np.ndarray | None:
    if y is None:
        return None
    values = _validate_soft_label(y, sample_index)
    return np.asarray([values[key] for key in LABEL_KEYS], dtype=np.float32)


if __name__ == "__main__":
    dataset_path = (
        Path(__file__).resolve().parents[1]
        / "dataset"
        / "x_samples_frozen_18000_v1.json"
    )
    first_index, first_sample = next(iter_json_samples(dataset_path))
    first_model_input = prepare_model_input(first_sample, first_index)
    print(f"Dataset: {dataset_path}")
    print_model_input(first_model_input)
