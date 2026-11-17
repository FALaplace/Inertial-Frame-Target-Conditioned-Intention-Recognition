"""Phase 3 B1: analytic shallow RandomForest soft-label baseline.

This baseline consumes only fixed-length summaries derived from the existing
``data_processing.prepare_model_input`` output.  It deliberately does not use
raw actor/target trajectories and is independent of the human blind-test data.
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from data_processing import (
    ANALYTIC_SEQUENCE_FEATURES,
    ERROR_FEATURES,
    MANEUVER_GLOBAL_FEATURES,
)


BASELINE_ID = "B1_analytic_random_forest"
ARTIFACT_SCHEMA_VERSION = 1
ANALYTIC_STATISTICS = ("mean", "std", "min", "max")
OUTPUT_DIM = 8

# Fixed, JSON-serializable Phase 3 configuration.  ``random_state`` is supplied
# explicitly by fit() so every formal run records its seed.
RF_CONFIG: dict[str, Any] = {
    "n_estimators": 300,
    "criterion": "squared_error",
    "max_depth": 18,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "bootstrap": True,
    "n_jobs": 1,
}

SATELLITE_CLASS_MASK = np.asarray(
    [1, 1, 1, 1, 1, 1, 0, 0], dtype=np.float64
)
GROUND_REGION_CLASS_MASK = np.asarray(
    [1, 1, 0, 0, 0, 1, 1, 1], dtype=np.float64
)

FEATURE_NAMES = tuple(
    f"{source}_{statistic}_{feature}"
    for source in ("analytic_factual", "analytic_difference")
    for statistic in ANALYTIC_STATISTICS
    for feature in ANALYTIC_SEQUENCE_FEATURES
) + tuple(
    [
        *(f"maneuver_global_{name}" for name in MANEUVER_GLOBAL_FEATURES),
        *(f"observed_error_{name}" for name in ERROR_FEATURES),
        *(f"error_mask_{name}" for name in ERROR_FEATURES),
        "counterpart_type_satellite",
        "counterpart_type_ground_region",
    ]
)


def _finite_array(
    model_input: Mapping[str, Any],
    key: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    value = np.asarray(model_input[key], dtype=np.float64)
    if value.shape != expected_shape:
        raise ValueError(f"{key} has shape {value.shape}; expected {expected_shape}.")
    if not np.isfinite(value).all():
        raise ValueError(f"{key} contains a non-finite value.")
    return value


def _counterpart_class_mask(model_input: Mapping[str, Any]) -> np.ndarray:
    counterpart_type = _finite_array(model_input, "counterpart_type", (2,))
    if np.array_equal(counterpart_type, np.asarray([1.0, 0.0])):
        return SATELLITE_CLASS_MASK
    if np.array_equal(counterpart_type, np.asarray([0.0, 1.0])):
        return GROUND_REGION_CLASS_MASK
    raise ValueError("counterpart_type must be [1, 0] or [0, 1].")


def extract_feature_vector(model_input: Mapping[str, Any]) -> np.ndarray:
    """Extract the deterministic 108-dimensional B1 analytic feature vector."""
    analytic_width = len(ANALYTIC_SEQUENCE_FEATURES)
    factual = np.asarray(model_input["analytic_factual"], dtype=np.float64)
    difference = np.asarray(model_input["analytic_difference"], dtype=np.float64)
    if (
        factual.ndim != 2
        or factual.shape[0] == 0
        or factual.shape[1] != analytic_width
        or difference.shape != factual.shape
    ):
        raise ValueError(
            "analytic_factual and analytic_difference must be non-empty arrays "
            f"with shape [T, {analytic_width}]."
        )
    if not np.isfinite(factual).all() or not np.isfinite(difference).all():
        raise ValueError("Analytic sequences must contain only finite values.")

    pooled = []
    for sequence in (factual, difference):
        pooled.extend(
            (
                sequence.mean(axis=0),
                sequence.std(axis=0),
                sequence.min(axis=0),
                sequence.max(axis=0),
            )
        )

    maneuver_global = _finite_array(
        model_input,
        "maneuver_global",
        (len(MANEUVER_GLOBAL_FEATURES),),
    )
    error_features = _finite_array(
        model_input, "error_features", (len(ERROR_FEATURES),)
    )
    error_mask = _finite_array(model_input, "error_mask", (len(ERROR_FEATURES),))
    counterpart_type = _finite_array(model_input, "counterpart_type", (2,))
    _counterpart_class_mask(model_input)

    vector = np.concatenate(
        (
            *pooled,
            maneuver_global,
            error_features * error_mask,
            error_mask,
            counterpart_type,
        )
    )
    if vector.shape != (len(FEATURE_NAMES),):
        raise AssertionError("B1 feature names and extraction order are inconsistent.")
    return vector.astype(np.float32, copy=False)


def extract_feature_matrix(
    model_inputs: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    if not model_inputs:
        raise ValueError("At least one model input is required.")
    return np.stack(
        [extract_feature_vector(model_input) for model_input in model_inputs]
    )


def _targets(model_inputs: Sequence[Mapping[str, Any]]) -> np.ndarray:
    targets = np.stack(
        [np.asarray(model_input.get("y"), dtype=np.float64) for model_input in model_inputs]
    )
    if targets.shape != (len(model_inputs), OUTPUT_DIM):
        raise ValueError("Every training input must contain an eight-dimensional y.")
    if (
        not np.isfinite(targets).all()
        or np.any(targets < 0.0)
        or not np.allclose(targets.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)
    ):
        raise ValueError("Training targets must be finite non-negative distributions.")
    masks = np.stack([_counterpart_class_mask(item) for item in model_inputs])
    if np.any(targets[masks == 0.0] != 0.0):
        raise ValueError("A training target violates its counterpart-type class mask.")
    return targets


def fit(
    model_inputs: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> RandomForestRegressor:
    """Fit B1 on prepared training inputs and their embedded soft labels."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer.")
    features = extract_feature_matrix(model_inputs)
    targets = _targets(model_inputs)
    model = RandomForestRegressor(random_state=int(seed), **RF_CONFIG)
    model.fit(features, targets)
    return model


def predict(
    model: RandomForestRegressor,
    model_inputs: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Predict valid eight-class distributions with the Phase 2 class mask."""
    raw_predictions = np.asarray(
        model.predict(extract_feature_matrix(model_inputs)), dtype=np.float64
    )
    if raw_predictions.shape != (len(model_inputs), OUTPUT_DIM):
        raise ValueError(
            f"RandomForest output has shape {raw_predictions.shape}; expected "
            f"({len(model_inputs)}, {OUTPUT_DIM})."
        )
    if not np.isfinite(raw_predictions).all():
        raise ValueError("RandomForest produced a non-finite prediction.")

    masks = np.stack([_counterpart_class_mask(item) for item in model_inputs])
    probabilities = np.maximum(raw_predictions, 0.0) * masks
    totals = probabilities.sum(axis=1, keepdims=True)
    empty_rows = totals[:, 0] <= 0.0
    if np.any(empty_rows):
        probabilities[empty_rows] = masks[empty_rows]
        totals = probabilities.sum(axis=1, keepdims=True)
    return (probabilities / totals).astype(np.float32)


def model_metadata(model: RandomForestRegressor) -> dict[str, Any]:
    """Return comparable model-size metadata for this non-neural baseline."""
    estimators = getattr(model, "estimators_", None)
    if not estimators:
        raise ValueError("The RandomForestRegressor has not been fitted.")
    parameters = model.get_params(deep=False)
    mismatches = {
        name: parameters.get(name)
        for name, expected in RF_CONFIG.items()
        if parameters.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"RandomForest differs from the frozen B1 config: {mismatches}.")
    seed = parameters["random_state"]
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("A fitted B1 model must record an integer random_state.")
    return {
        "baseline_id": BASELINE_ID,
        "model_type": type(model).__name__,
        "feature_count": len(FEATURE_NAMES),
        "output_count": int(model.n_outputs_),
        "tree_count": len(estimators),
        "total_node_count": int(sum(tree.tree_.node_count for tree in estimators)),
        "total_leaf_count": int(
            sum(np.count_nonzero(tree.tree_.children_left == -1) for tree in estimators)
        ),
        "serialized_model_size_bytes": len(
            pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        ),
        "parameter_count": None,
        "parameter_count_note": "Tree ensembles do not have a neural parameter count.",
        "random_seed": int(seed),
        "config": {**RF_CONFIG, "random_state": int(seed)},
    }


def save(model: RandomForestRegressor, path: Path) -> Path:
    """Atomically save a fitted B1 artifact. Only load trusted pickle files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "baseline_id": BASELINE_ID,
        "feature_names": FEATURE_NAMES,
        "metadata": model_metadata(model),
        "model": model,
    }
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def load(path: Path) -> RandomForestRegressor:
    """Load and validate a trusted local B1 artifact."""
    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Invalid B1 artifact payload.")
    if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported B1 artifact schema version.")
    if payload.get("baseline_id") != BASELINE_ID:
        raise ValueError("Artifact is not the Phase 3 B1 baseline.")
    if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("Artifact feature order differs from the current B1 definition.")
    model = payload.get("model")
    if not isinstance(model, RandomForestRegressor):
        raise ValueError("Artifact does not contain a RandomForestRegressor.")
    model_metadata(model)
    return model


def _demo() -> None:
    """Small synthetic check for pooling, fitting, masking, and persistence."""
    rng = np.random.default_rng(4)
    model_inputs = []
    for index in range(12):
        is_region = index % 2 == 1
        class_mask = (
            GROUND_REGION_CLASS_MASK if is_region else SATELLITE_CLASS_MASK
        )
        y = rng.random(OUTPUT_DIM) * class_mask
        y /= y.sum()
        model_inputs.append(
            {
                "analytic_factual": rng.normal(size=(7, len(ANALYTIC_SEQUENCE_FEATURES))),
                "analytic_difference": rng.normal(
                    size=(7, len(ANALYTIC_SEQUENCE_FEATURES))
                ),
                "maneuver_global": rng.normal(size=len(MANEUVER_GLOBAL_FEATURES)),
                "error_features": rng.normal(size=len(ERROR_FEATURES)),
                "error_mask": np.asarray([1.0, 0.0, 1.0]),
                "counterpart_type": np.asarray(
                    [0.0, 1.0] if is_region else [1.0, 0.0]
                ),
                "y": y,
            }
        )

    model = fit(model_inputs, seed=17)
    probabilities = predict(model, model_inputs)
    masks = np.stack([_counterpart_class_mask(item) for item in model_inputs])
    assert extract_feature_matrix(model_inputs).shape == (12, 108)
    assert probabilities.shape == (12, OUTPUT_DIM)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(probabilities[masks == 0.0] == 0.0)
    assert model_metadata(model)["random_seed"] == 17

    from tempfile import TemporaryDirectory

    with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
        artifact_path = save(model, Path(directory) / "b1.pkl")
        restored = load(artifact_path)
        assert np.array_equal(probabilities, predict(restored, model_inputs))


if __name__ == "__main__":
    _demo()
    print("Phase 3 B1 synthetic check passed.")
