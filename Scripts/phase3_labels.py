"""Build audited Phase 3 training-label overlays from frozen assets.

This module never reads or writes human blind-test labels.  Every public builder
returns an in-memory mapping and keeps validation/test targets untouched.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


IDEA_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = IDEA_DIR / "generated_dataset" / "sgp4_large_maneuver_1min"
LABEL_DIR = IDEA_DIR / "experiment_data" / "labels" / "phase1_label_freeze_v1"

MAIN_LABELS_PATH = LABEL_DIR / "labels_main_clean_10400.json"
TRAIN_AUGMENTED_LABELS_PATH = LABEL_DIR / "labels_train_augmented_10400.json"
FRONT_RULES_PATH = (
    DATASET_DIR
    / "multi_agent_optimized_1000"
    / "multiagent_raw_labels_00000_00999.json"
)
REMAINDER_RULES_DIR = DATASET_DIR / "multi_agent_optimized_10400" / "rules"
SPLIT_MANIFEST_PATH = (
    IDEA_DIR
    / "experiment_data"
    / "splits"
    / "actor_70_15_15_calibration_train_v1__labels_v5_clean.json"
)

EXPECTED_SAMPLE_COUNT = 10_400
EXPECTED_SAMPLE_INDICES = frozenset(range(EXPECTED_SAMPLE_COUNT))
CLASS_KEYS = tuple(f"c{index}" for index in range(8))
CLASS_NAMES = tuple(f"C{index}" for index in range(8))
COUNTERPART_TYPES = frozenset({"satellite", "ground_region"})
RULE_IDS = (
    "geometry_dynamics",
    "counterfactual_event",
    "temporal_behavior",
    "uncertainty_conservative",
    "skeptical_alternative",
    "human_geometry",
)
LABEL_IDS = (
    "v5_clean",
    "hard_argmax_v5",
    "fixed_rule_geometry_dynamics",
    "unweighted_six_rule_mean",
    "v5_train_aug132",
)

MAIN_LABEL_VERSION = "multiagent_human_calibrated_v5_clean_20260817"
TRAIN_AUGMENTED_VERSION = (
    "multiagent_human_calibrated_v5_aug188_train_only_20260817"
)

Label = dict[str, float]
LabelMap = dict[int, Label]


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"Expected a JSON object list: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(labels: Mapping[int, Mapping[str, float]]) -> str:
    payload = [
        {"sample_index": index, "y": dict(labels[index])}
        for index in sorted(labels)
    ]
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_counterpart_type(value: object, sample_index: int) -> str:
    counterpart_type = str(value)
    if counterpart_type not in COUNTERPART_TYPES:
        raise ValueError(
            f"Invalid counterpart_type at sample {sample_index}: {counterpart_type!r}"
        )
    return counterpart_type


def _validate_y(
    value: object,
    sample_index: int,
    counterpart_type: str,
    source: Path,
) -> Label:
    if not isinstance(value, dict) or set(value) != set(CLASS_KEYS):
        raise ValueError(
            f"Label {sample_index} in {source} must contain exactly c0-c7."
        )
    y = {key: float(value[key]) for key in CLASS_KEYS}
    if any(not math.isfinite(probability) or not 0.0 <= probability <= 1.0 for probability in y.values()):
        raise ValueError(f"Invalid probability at sample {sample_index} in {source}.")
    if not math.isclose(sum(y.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Probabilities do not sum to one at sample {sample_index} in {source}.")
    invalid = (6, 7) if counterpart_type == "satellite" else (2, 3, 4)
    if any(y[CLASS_KEYS[index]] != 0.0 for index in invalid):
        raise ValueError(f"Class-mask violation at sample {sample_index} in {source}.")
    return y


def _assert_complete(mapping: Mapping[int, object], source: Path | str) -> None:
    indices = set(mapping)
    if indices != EXPECTED_SAMPLE_INDICES:
        missing = sorted(EXPECTED_SAMPLE_INDICES - indices)[:10]
        extra = sorted(indices - EXPECTED_SAMPLE_INDICES)[:10]
        raise ValueError(
            f"{source} must cover sample_index 0-10399 exactly; "
            f"missing={missing}, extra={extra}."
        )


def _load_frozen_labels(
    path: Path,
    expected_version: str,
) -> tuple[LabelMap, dict[int, str]]:
    labels: LabelMap = {}
    counterpart_types: dict[int, str] = {}
    for row in _read_rows(path):
        sample_index = int(row.get("sample_index", -1))
        if sample_index in labels:
            raise ValueError(f"Duplicate sample_index {sample_index} in {path}.")
        if row.get("label_version") != expected_version:
            raise ValueError(f"Unexpected label_version at sample {sample_index} in {path}.")
        counterpart_type = _validate_counterpart_type(
            row.get("counterpart_type"), sample_index
        )
        y = _validate_y(row.get("y"), sample_index, counterpart_type, path)
        dominant = CLASS_NAMES[max(range(8), key=lambda index: y[CLASS_KEYS[index]])]
        if row.get("dominant_label") != dominant:
            raise ValueError(f"dominant_label mismatch at sample {sample_index} in {path}.")
        labels[sample_index] = y
        counterpart_types[sample_index] = counterpart_type
    _assert_complete(labels, path)
    return labels, counterpart_types


def _load_rule_views(
    counterpart_types: Mapping[int, str],
    requested_rule_ids: Iterable[str] = RULE_IDS,
) -> dict[str, LabelMap]:
    requested = tuple(requested_rule_ids)
    if not requested or len(requested) != len(set(requested)) or not set(requested) <= set(RULE_IDS):
        raise ValueError("requested_rule_ids must be unique members of RULE_IDS.")
    views: dict[str, LabelMap] = {rule_id: {} for rule_id in requested}
    for row in _read_rows(FRONT_RULES_PATH):
        sample_index = int(row.get("sample_index", -1))
        if sample_index not in range(1000):
            raise ValueError(f"Unexpected front-rule sample_index {sample_index}.")
        if any(sample_index in view for view in views.values()):
            raise ValueError(f"Duplicate sample_index {sample_index} in {FRONT_RULES_PATH}.")
        if _validate_counterpart_type(row.get("counterpart_type"), sample_index) != counterpart_types[sample_index]:
            raise ValueError(f"Counterpart type mismatch at sample {sample_index}.")
        agents = row.get("agents")
        if not isinstance(agents, dict) or set(agents) != set(RULE_IDS):
            raise ValueError(f"Front-rule agents mismatch at sample {sample_index}.")
        for rule_id in requested:
            agent = agents[rule_id]
            if not isinstance(agent, dict):
                raise ValueError(f"Invalid {rule_id} row at sample {sample_index}.")
            views[rule_id][sample_index] = _validate_y(
                agent.get("y"),
                sample_index,
                counterpart_types[sample_index],
                FRONT_RULES_PATH,
            )

    for rule_id in requested:
        path = REMAINDER_RULES_DIR / f"{rule_id}.json"
        for row in _read_rows(path):
            sample_index = int(row.get("sample_index", -1))
            if sample_index < 1000 or sample_index >= EXPECTED_SAMPLE_COUNT:
                raise ValueError(f"Unexpected remainder sample_index {sample_index} in {path}.")
            if sample_index in views[rule_id]:
                raise ValueError(f"Duplicate sample_index {sample_index} in {path}.")
            if bool(row.get("abstain", False)):
                raise ValueError(f"Rule {rule_id} abstained at sample {sample_index}.")
            views[rule_id][sample_index] = _validate_y(
                row.get("y"),
                sample_index,
                counterpart_types[sample_index],
                path,
            )
        _assert_complete(views[rule_id], rule_id)
    return views


def _hard_argmax(labels: Mapping[int, Mapping[str, float]]) -> LabelMap:
    result: LabelMap = {}
    for sample_index, y in labels.items():
        dominant = max(range(8), key=lambda index: y[CLASS_KEYS[index]])
        result[sample_index] = {
            key: float(index == dominant) for index, key in enumerate(CLASS_KEYS)
        }
    return result


def _unweighted_mean(views: Mapping[str, LabelMap]) -> LabelMap:
    result: LabelMap = {}
    for sample_index in range(EXPECTED_SAMPLE_COUNT):
        result[sample_index] = {
            key: sum(views[rule_id][sample_index][key] for rule_id in RULE_IDS)
            / len(RULE_IDS)
            for key in CLASS_KEYS
        }
    return result


def build_label_by_sample_index(label_id: str) -> LabelMap:
    """Build and strictly validate one complete 10,400-label version."""
    if label_id not in LABEL_IDS:
        raise ValueError(f"Unknown label_id={label_id!r}; expected one of {LABEL_IDS}.")

    main, counterpart_types = _load_frozen_labels(
        MAIN_LABELS_PATH, MAIN_LABEL_VERSION
    )
    if label_id == "v5_clean":
        labels = main
    elif label_id == "hard_argmax_v5":
        labels = _hard_argmax(main)
    elif label_id == "v5_train_aug132":
        labels, augmented_types = _load_frozen_labels(
            TRAIN_AUGMENTED_LABELS_PATH, TRAIN_AUGMENTED_VERSION
        )
        if augmented_types != counterpart_types:
            raise ValueError("Main and train-augmented counterpart types differ.")
        changed = {index for index in labels if labels[index] != main[index]}
        if len(changed) != 132:
            raise ValueError(f"Expected 132 augmented labels, found {len(changed)}.")
    else:
        requested_rules = (
            ("geometry_dynamics",)
            if label_id == "fixed_rule_geometry_dynamics"
            else RULE_IDS
        )
        views = _load_rule_views(counterpart_types, requested_rules)
        labels = (
            views["geometry_dynamics"]
            if label_id == "fixed_rule_geometry_dynamics"
            else _unweighted_mean(views)
        )

    _assert_complete(labels, label_id)
    for sample_index, y in labels.items():
        _validate_y(y, sample_index, counterpart_types[sample_index], Path(label_id))
    return {index: dict(labels[index]) for index in range(EXPECTED_SAMPLE_COUNT)}


def _validated_partitions(
    train_sample_indices: Iterable[int],
    validation_indices: Iterable[int],
    test_indices: Iterable[int],
) -> tuple[set[int], set[int], set[int]]:
    partitions: list[set[int]] = []
    for name, values in (
        ("train", train_sample_indices),
        ("validation", validation_indices),
        ("test", test_indices),
    ):
        items = [int(value) for value in values]
        if len(items) != len(set(items)):
            raise ValueError(f"{name} contains duplicate sample indices.")
        current = set(items)
        if not current <= EXPECTED_SAMPLE_INDICES:
            raise ValueError(f"{name} contains an out-of-range sample index.")
        partitions.append(current)

    train, validation, test = partitions
    if train & validation or train & test or validation & test:
        raise ValueError("Train, validation, and test indices must be disjoint.")
    if train | validation | test != EXPECTED_SAMPLE_INDICES:
        raise ValueError("Train, validation, and test must cover sample_index 0-10399 exactly.")
    return train, validation, test


def build_train_label_overlay(
    label_id: str,
    train_sample_indices: Iterable[int],
    validation_indices: Iterable[int],
    test_indices: Iterable[int],
) -> LabelMap:
    """Return labels for train only; validation and test can never be overridden."""
    train, validation, test = _validated_partitions(
        train_sample_indices, validation_indices, test_indices
    )
    labels = build_label_by_sample_index(label_id)
    overlay = {index: labels[index] for index in sorted(train)}
    if set(overlay) & (validation | test) or set(overlay) != train:
        raise RuntimeError("Train-label overlay crossed a frozen split boundary.")

    if label_id == "v5_train_aug132":
        main = build_label_by_sample_index("v5_clean")
        changed = {index for index in labels if labels[index] != main[index]}
        if len(changed) != 132 or not changed <= train:
            raise ValueError("The 132 augmented labels must be train-only.")
    return overlay


def _source_paths(label_id: str) -> tuple[Path, ...]:
    if label_id in {"v5_clean", "hard_argmax_v5"}:
        return (MAIN_LABELS_PATH,)
    if label_id == "v5_train_aug132":
        return (MAIN_LABELS_PATH, TRAIN_AUGMENTED_LABELS_PATH)
    if label_id == "fixed_rule_geometry_dynamics":
        return (
            MAIN_LABELS_PATH,
            FRONT_RULES_PATH,
            REMAINDER_RULES_DIR / "geometry_dynamics.json",
        )
    if label_id == "unweighted_six_rule_mean":
        return (
            MAIN_LABELS_PATH,
            FRONT_RULES_PATH,
            *(REMAINDER_RULES_DIR / f"{rule_id}.json" for rule_id in RULE_IDS),
        )
    raise ValueError(f"Unknown label_id={label_id!r}; expected one of {LABEL_IDS}.")


def _derivation(label_id: str) -> str:
    return {
        "v5_clean": "frozen_v5_clean_soft_labels",
        "hard_argmax_v5": "one_hot_argmax_of_frozen_v5_clean_train_labels",
        "fixed_rule_geometry_dynamics": "geometry_dynamics_rule_view",
        "unweighted_six_rule_mean": "per_class_arithmetic_mean_of_six_rule_views",
        "v5_train_aug132": "frozen_train_only_uncertainty_augmentation_132_rows",
    }[label_id]


def _provenance(label_id: str, labels: Mapping[int, Mapping[str, float]]) -> dict[str, Any]:
    return {
        "label_id": label_id,
        "sample_count": len(labels),
        "label_map_sha256": _canonical_sha256(labels),
        "derivation": _derivation(label_id),
        "sources": [
            {
                "path": path.resolve().relative_to(IDEA_DIR.resolve()).as_posix(),
                "sha256": _sha256_file(path),
            }
            for path in _source_paths(label_id)
        ],
    }


def label_asset_provenance(label_id: str) -> dict[str, Any]:
    """Return hashes and derivation metadata for one complete label mapping."""
    labels = build_label_by_sample_index(label_id)
    return _provenance(label_id, labels)


def describe_train_label_overlay(
    label_id: str,
    train_sample_indices: Iterable[int],
    validation_indices: Iterable[int],
    test_indices: Iterable[int],
) -> dict[str, Any]:
    """Return provenance plus train-only dominant counts and soft class mass."""
    train, validation, test = _validated_partitions(
        train_sample_indices, validation_indices, test_indices
    )
    labels = build_label_by_sample_index(label_id)
    overlay = {index: labels[index] for index in sorted(train)}
    if set(overlay) & (validation | test):
        raise RuntimeError("Train-label overlay crossed a frozen split boundary.")
    if label_id == "v5_train_aug132":
        main = build_label_by_sample_index("v5_clean")
        changed = {index for index in labels if labels[index] != main[index]}
        if len(changed) != 132 or not changed <= train:
            raise ValueError("The 132 augmented labels must be train-only.")

    dominant = Counter(
        CLASS_NAMES[max(range(8), key=lambda index: y[CLASS_KEYS[index]])]
        for y in overlay.values()
    )
    soft_mass = {
        key: sum(y[key] for y in overlay.values())
        for key in CLASS_KEYS
    }
    return {
        **_provenance(label_id, labels),
        "train": {
            "sample_count": len(overlay),
            "overlay_sha256": _canonical_sha256(overlay),
            "dominant_counts": {name: dominant[name] for name in CLASS_NAMES},
            "soft_mass": {
                CLASS_NAMES[index]: float(soft_mass[key])
                for index, key in enumerate(CLASS_KEYS)
            },
        },
        "leakage_guard": {
            "overlay_index_count": len(overlay),
            "validation_overlap": 0,
            "test_overlap": 0,
        },
    }


def validate_real_assets() -> dict[str, dict[str, Any]]:
    """Run the read-only integrity check against the frozen actor split."""
    manifest = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    split_indices = {
        name: [
            int(row["sample_index"])
            for row in manifest["samples"]
            if row["split"] == name
        ]
        for name in ("train", "validation", "test")
    }
    return {
        label_id: describe_train_label_overlay(
            label_id,
            split_indices["train"],
            split_indices["validation"],
            split_indices["test"],
        )
        for label_id in LABEL_IDS
    }


if __name__ == "__main__":
    audits = validate_real_assets()
    print(
        json.dumps(
            {
                label_id: {
                    "label_map_sha256": audit["label_map_sha256"],
                    "train": audit["train"],
                    "leakage_guard": audit["leakage_guard"],
                }
                for label_id, audit in audits.items()
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


__all__ = [
    "LABEL_IDS",
    "RULE_IDS",
    "build_label_by_sample_index",
    "build_train_label_overlay",
    "describe_train_label_overlay",
    "label_asset_provenance",
    "validate_real_assets",
]
