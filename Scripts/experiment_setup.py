"""Freeze and validate the formal actor-isolated experiment split.

Run this file directly in the IDE before formal training.  It joins the
optimized labels and candidate metadata by their authoritative 0-based row
index, protects all calibration actors from validation/test leakage, and
writes one deterministic split manifest under ``idea 1/experiment_data/splits``.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from analytic_layer import infer_counterpart_is_region
from data_processing import iter_json_samples


IDEA_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = IDEA_DIR / "generated_dataset" / "sgp4_large_maneuver_1min"
DATASET_PATH = DATASET_DIR / "x_samples_frozen_v5_clean.json"
CANDIDATE_SUMMARY_PATH = DATASET_DIR / "candidate_match_summary.json"
DATASET_MANIFEST_PATH = DATASET_DIR / "dataset_manifest.json"
HUMAN_LABELS_PATH = DATASET_DIR / "human_labels.json"

DATASET_VERSION = "sgp4_large_maneuver_1min_10400_v2_clean"
LABEL_VERSION = "multiagent_human_calibrated_v5_clean_20260817"
SPLIT_ID = "actor_70_15_15_calibration_train_v1"
BASE_SPLIT_MANIFEST_PATH = IDEA_DIR / "experiment_data" / "splits" / f"{SPLIT_ID}.json"
LABEL_MANIFEST_PATH = (
    IDEA_DIR / "experiment_data" / "labels" / "phase1_label_freeze_v1" / "label_manifest.json"
)
SPLIT_MANIFEST_PATH = (
    IDEA_DIR / "experiment_data" / "splits" / f"{SPLIT_ID}__labels_v5_clean.json"
)

BASE_SEED = 42
SEED_CANDIDATES = 512
SPLIT_NAMES = ("train", "validation", "test")
CLASS_NAMES = tuple(f"C{index}" for index in range(8))
CLASS_KEYS = tuple(f"c{index}" for index in range(8))
COUNTERPART_TYPES = ("satellite", "ground_region")


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"Expected a JSON object list: {path}")
    return payload


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(IDEA_DIR.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _actor_sort_key(actor_id: str) -> tuple[int, int | str]:
    return (0, int(actor_id)) if actor_id.isdigit() else (1, actor_id)


def _validate_soft_label(y: Any, sample_index: int, counterpart_type: str) -> str:
    if not isinstance(y, dict) or set(y) != set(CLASS_KEYS):
        raise ValueError(f"Sample {sample_index} must contain exactly c0-c7.")
    probabilities = [float(y[key]) for key in CLASS_KEYS]
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError(f"Sample {sample_index} has an invalid soft label.")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Sample {sample_index} soft label does not sum to one.")
    invalid = (6, 7) if counterpart_type == "satellite" else (2, 3, 4)
    if any(probabilities[index] != 0.0 for index in invalid):
        raise ValueError(f"Sample {sample_index} violates the counterpart class mask.")
    return CLASS_NAMES[max(range(8), key=probabilities.__getitem__)]


def _load_sample_metadata() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Join candidate metadata and optimized labels with row-order safeguards."""
    summary = _read_json_list(CANDIDATE_SUMMARY_PATH)
    samples: list[dict[str, Any]] = []

    for sample_index, sample in iter_json_samples(DATASET_PATH):
        if sample_index >= len(summary):
            raise ValueError("Optimized dataset contains more rows than candidate summary.")
        candidate = summary[sample_index]
        actor_id = str(candidate.get("actor_norad_cat_id", ""))
        counterpart_type = str(candidate.get("counterpart_type", ""))
        if not actor_id or counterpart_type not in COUNTERPART_TYPES:
            raise ValueError(f"Invalid candidate metadata at sample {sample_index}.")

        sample_score = float(sample.get("matching_score"))
        candidate_score = float(candidate.get("matching_score"))
        if not math.isclose(sample_score, candidate_score, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"Candidate/data row mismatch at sample {sample_index}: matching_score.")
        maneuver_count = len(sample.get("maneuver_events") or [])
        if maneuver_count != int(candidate.get("maneuver_count", -1)):
            raise ValueError(f"Candidate/data row mismatch at sample {sample_index}: maneuver_count.")
        inferred_type = (
            "ground_region"
            if infer_counterpart_is_region(sample.get("counterpart_trajectory", []))
            else "satellite"
        )
        if inferred_type != counterpart_type:
            raise ValueError(f"Candidate/data row mismatch at sample {sample_index}: counterpart_type.")

        score_metrics = candidate.get("score_metrics") or {}
        position_sigma = float(score_metrics.get("uncertainty_orbit_position_sigma_km", 0.0))
        samples.append(
            {
                "sample_index": sample_index,
                "actor": str(candidate.get("actor", "")),
                "actor_norad_cat_id": actor_id,
                "counterpart": str(candidate.get("counterpart", "")),
                "counterpart_type": counterpart_type,
                "maneuver_count": maneuver_count,
                "maneuver_presence": "maneuver" if maneuver_count else "non_maneuver",
                "dominant_class": _validate_soft_label(
                    sample.get("y"), sample_index, counterpart_type
                ),
                "orbit_position_sigma_km": position_sigma,
            }
        )

    if len(samples) != len(summary):
        raise ValueError(
            f"Dataset/summary length mismatch: {len(samples)} != {len(summary)}."
        )
    return samples, summary


def _calibration_actor_ids(
    samples: list[dict[str, Any]],
    summary: list[dict[str, Any]],
) -> tuple[set[str], list[int]]:
    human_labels = _read_json_list(HUMAN_LABELS_PATH)
    actor_ids: set[str] = set()
    sample_indices: list[int] = []
    seen_indices: set[int] = set()
    for row in human_labels:
        sample_index = int(row.get("sample_index", -1))
        if sample_index < 0 or sample_index >= len(samples) or sample_index in seen_indices:
            raise ValueError(f"Invalid or duplicate human sample_index: {sample_index}")
        seen_indices.add(sample_index)
        sample_indices.append(sample_index)
        candidate = summary[sample_index]
        for key in ("actor", "actor_norad_cat_id", "counterpart", "counterpart_type"):
            if str(row.get(key, "")) != str(candidate.get(key, "")):
                raise ValueError(f"Human/candidate row mismatch at sample {sample_index}: {key}.")
        actor_ids.add(str(candidate["actor_norad_cat_id"]))
    return actor_ids, sorted(sample_indices)


def _actor_profiles(samples: Iterable[Mapping[str, Any]]) -> dict[str, Counter[str]]:
    profiles: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        profile = profiles[str(sample["actor_norad_cat_id"])]
        profile[f"class:{sample['dominant_class']}"] += 1
        profile[f"type:{sample['counterpart_type']}"] += 1
        profile[f"maneuver:{sample['maneuver_presence']}"] += 1
    return dict(profiles)


def _sum_profiles(
    actor_ids: Iterable[str],
    profiles: Mapping[str, Counter[str]],
) -> Counter[str]:
    total: Counter[str] = Counter()
    for actor_id in actor_ids:
        total.update(profiles[actor_id])
    return total


def _distribution_score(
    assignments: Mapping[str, set[str]],
    profiles: Mapping[str, Counter[str]],
) -> float:
    global_counts = _sum_profiles(profiles.keys(), profiles)
    global_total = sum(global_counts[f"class:{name}"] for name in CLASS_NAMES)
    feature_groups = (
        tuple(f"class:{name}" for name in CLASS_NAMES),
        tuple(f"type:{name}" for name in COUNTERPART_TYPES),
        ("maneuver:maneuver", "maneuver:non_maneuver"),
    )
    score = 0.0
    for split_name in SPLIT_NAMES:
        counts = _sum_profiles(assignments[split_name], profiles)
        split_total = sum(counts[f"class:{name}"] for name in CLASS_NAMES)
        for keys in feature_groups:
            score += sum(
                abs(counts[key] / split_total - global_counts[key] / global_total)
                for key in keys
            ) / len(keys)
        if split_name != "train":
            missing_weight = 2.0 if split_name == "test" else 1.0
            score += missing_weight * sum(
                1.0
                for name in CLASS_NAMES
                if global_counts[f"class:{name}"] and not counts[f"class:{name}"]
            )
    return score


def _choose_actor_assignment(
    actor_ids: set[str],
    calibration_actor_ids: set[str],
    profiles: Mapping[str, Counter[str]],
) -> tuple[dict[str, str], int, float, dict[str, int]]:
    actor_count = len(actor_ids)
    validation_count = round(actor_count * 0.15)
    test_count = round(actor_count * 0.15)
    train_count = actor_count - validation_count - test_count
    available = sorted(actor_ids - calibration_actor_ids, key=_actor_sort_key)
    if len(available) < validation_count + test_count:
        raise ValueError("Too many calibration actors to form validation and test splits.")

    best: tuple[float, int, dict[str, set[str]]] | None = None
    for seed in range(BASE_SEED, BASE_SEED + SEED_CANDIDATES):
        shuffled = available.copy()
        random.Random(seed).shuffle(shuffled)
        validation = set(shuffled[:validation_count])
        test = set(shuffled[validation_count : validation_count + test_count])
        train = actor_ids - validation - test
        assignments = {"train": train, "validation": validation, "test": test}
        score = _distribution_score(assignments, profiles)
        if best is None or (score, seed) < (best[0], best[1]):
            best = (score, seed, assignments)

    if best is None:
        raise RuntimeError("No split candidate was generated.")
    score, selected_seed, actor_sets = best
    if len(actor_sets["train"]) != train_count:
        raise RuntimeError("Unexpected train actor count.")
    actor_assignment = {
        actor_id: split_name
        for split_name, split_actor_ids in actor_sets.items()
        for actor_id in split_actor_ids
    }
    return (
        actor_assignment,
        selected_seed,
        score,
        {"train": train_count, "validation": validation_count, "test": test_count},
    )


def _load_frozen_actor_assignment(
    actor_ids: set[str],
    profiles: Mapping[str, Counter[str]],
) -> tuple[dict[str, str], int, float, dict[str, int]]:
    """Reuse the Phase 0 actor assignment after the Phase 1 label freeze."""
    parent = _read_json(BASE_SPLIT_MANIFEST_PATH)
    validate_split_manifest(parent, expected_sample_count=len(parent["samples"]))
    actor_assignment = {
        str(row["actor_norad_cat_id"]): str(row["split"])
        for row in parent["actors"]
    }
    if set(actor_assignment) != actor_ids:
        raise ValueError("Parent split actor IDs do not match the frozen dataset.")
    actor_sets = {
        split_name: {
            actor_id
            for actor_id, assigned_split in actor_assignment.items()
            if assigned_split == split_name
        }
        for split_name in SPLIT_NAMES
    }
    score = _distribution_score(actor_sets, profiles)
    counts = {name: len(actor_sets[name]) for name in SPLIT_NAMES}
    return actor_assignment, int(parent["config"]["selected_seed"]), score, counts


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _build_audit(
    samples: list[dict[str, Any]],
    actor_assignment: Mapping[str, str],
    balance_score: float,
) -> tuple[dict[str, Any], tuple[float, float]]:
    train_errors = [
        float(sample["orbit_position_sigma_km"])
        for sample in samples
        if actor_assignment[str(sample["actor_norad_cat_id"])] == "train"
    ]
    error_thresholds = (_quantile(train_errors, 1.0 / 3.0), _quantile(train_errors, 2.0 / 3.0))
    audit: dict[str, Any] = {
        "balance_score": balance_score,
        "actor_counts": {name: 0 for name in SPLIT_NAMES},
        "sample_counts": {name: 0 for name in SPLIT_NAMES},
        "dominant_label_counts": {
            name: {class_name: 0 for class_name in CLASS_NAMES} for name in SPLIT_NAMES
        },
        "counterpart_type_counts": {
            name: {kind: 0 for kind in COUNTERPART_TYPES} for name in SPLIT_NAMES
        },
        "maneuver_presence_counts": {
            name: {"maneuver": 0, "non_maneuver": 0} for name in SPLIT_NAMES
        },
        "error_bin_thresholds_from_train_km": {
            "low_max": error_thresholds[0],
            "medium_max": error_thresholds[1],
        },
        "error_bin_counts": {
            name: {"low": 0, "medium": 0, "high": 0} for name in SPLIT_NAMES
        },
    }
    dominant_class_actors = {
        split_name: {class_name: set() for class_name in CLASS_NAMES}
        for split_name in SPLIT_NAMES
    }
    for split_name in actor_assignment.values():
        audit["actor_counts"][split_name] += 1
    for sample in samples:
        split_name = actor_assignment[str(sample["actor_norad_cat_id"])]
        audit["sample_counts"][split_name] += 1
        audit["dominant_label_counts"][split_name][sample["dominant_class"]] += 1
        dominant_class_actors[split_name][sample["dominant_class"]].add(
            str(sample["actor_norad_cat_id"])
        )
        audit["counterpart_type_counts"][split_name][sample["counterpart_type"]] += 1
        audit["maneuver_presence_counts"][split_name][sample["maneuver_presence"]] += 1
        sigma = float(sample["orbit_position_sigma_km"])
        error_bin = "low" if sigma <= error_thresholds[0] else "medium" if sigma <= error_thresholds[1] else "high"
        audit["error_bin_counts"][split_name][error_bin] += 1
    audit["dominant_class_actor_counts"] = {
        split_name: {
            class_name: len(actor_ids)
            for class_name, actor_ids in class_actors.items()
        }
        for split_name, class_actors in dominant_class_actors.items()
    }
    audit["missing_dominant_classes"] = {
        split_name: [
            class_name
            for class_name, count in audit["dominant_label_counts"][split_name].items()
            if count == 0
        ]
        for split_name in SPLIT_NAMES
    }
    return audit, error_thresholds


def validate_split_manifest(
    manifest: Mapping[str, Any],
    expected_sample_count: int | None = None,
) -> None:
    """Reject duplicate samples, actor leakage, or calibration leakage."""
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported split manifest schema_version.")
    rows = manifest.get("samples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Split manifest must contain sample rows.")
    if expected_sample_count is None:
        expected_sample_count = int(manifest["dataset"]["sample_count"])
    indices = [int(row["sample_index"]) for row in rows]
    if len(indices) != expected_sample_count or set(indices) != set(range(expected_sample_count)):
        raise ValueError("Split manifest must cover every 0-based sample_index exactly once.")

    actor_to_split: dict[str, str] = {}
    sample_counts = Counter()
    for row in rows:
        split_name = str(row["split"])
        actor_id = str(row["actor_norad_cat_id"])
        if split_name not in SPLIT_NAMES:
            raise ValueError(f"Unknown split: {split_name}")
        prior = actor_to_split.setdefault(actor_id, split_name)
        if prior != split_name:
            raise ValueError(f"Actor leakage detected for {actor_id}.")
        sample_counts[split_name] += 1
    if any(sample_counts[name] == 0 for name in SPLIT_NAMES):
        raise ValueError("All three splits must contain samples.")

    declared_actors = {
        str(row["actor_norad_cat_id"]): str(row["split"])
        for row in manifest.get("actors", [])
    }
    if declared_actors != actor_to_split:
        raise ValueError("Actor-level and sample-level split assignments differ.")
    audit = manifest.get("audit", {})
    if {name: int(audit["sample_counts"][name]) for name in SPLIT_NAMES} != dict(sample_counts):
        raise ValueError("Manifest sample-count audit is inconsistent.")
    computed_actor_counts = Counter(actor_to_split.values())
    if {
        name: int(audit["actor_counts"][name]) for name in SPLIT_NAMES
    } != dict(computed_actor_counts):
        raise ValueError("Manifest actor-count audit is inconsistent.")

    calibration_actors = set(manifest["calibration"]["actor_norad_cat_ids"])
    if len(calibration_actors) != int(manifest["calibration"]["actor_count"]):
        raise ValueError("Calibration actor count is inconsistent.")
    violations = sorted(
        actor_id for actor_id in calibration_actors if actor_to_split.get(actor_id) != "train"
    )
    if violations:
        raise ValueError(f"Calibration actors outside train: {violations}")


def load_split_manifest(path: Path = SPLIT_MANIFEST_PATH) -> dict[str, Any]:
    manifest = _read_json(path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Split manifest must be a JSON object: {path}")
    validate_split_manifest(manifest)
    return manifest


def load_split_indices(
    path: Path = SPLIT_MANIFEST_PATH,
) -> tuple[dict[str, Any], dict[str, list[int]]]:
    manifest = load_split_manifest(path)
    split_indices = {name: [] for name in SPLIT_NAMES}
    for row in manifest["samples"]:
        split_indices[str(row["split"])].append(int(row["sample_index"]))
    for indices in split_indices.values():
        indices.sort()
    return manifest, split_indices


def verify_manifest_dataset(
    manifest: Mapping[str, Any],
    dataset_path: Path = DATASET_PATH,
    verify_hash: bool = True,
) -> None:
    dataset = manifest["dataset"]
    if dataset_path.stat().st_size != int(dataset["size_bytes"]):
        raise ValueError("Dataset size differs from the frozen split manifest.")
    if verify_hash and sha256_file(dataset_path) != str(dataset["sha256"]):
        raise ValueError("Dataset SHA-256 differs from the frozen split manifest.")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def generate_split_manifest(output_path: Path = SPLIT_MANIFEST_PATH) -> dict[str, Any]:
    samples, summary = _load_sample_metadata()
    calibration_actors, human_sample_indices = _calibration_actor_ids(samples, summary)
    actor_ids = {str(sample["actor_norad_cat_id"]) for sample in samples}
    profiles = _actor_profiles(samples)
    actor_assignment, selected_seed, balance_score, target_actor_counts = (
        _load_frozen_actor_assignment(actor_ids, profiles)
    )
    audit, error_thresholds = _build_audit(samples, actor_assignment, balance_score)

    manifest_samples = []
    for sample in samples:
        actor_id = str(sample["actor_norad_cat_id"])
        sigma = float(sample["orbit_position_sigma_km"])
        error_bin = "low" if sigma <= error_thresholds[0] else "medium" if sigma <= error_thresholds[1] else "high"
        manifest_samples.append(
            {
                **sample,
                "split": actor_assignment[actor_id],
                "error_bin": error_bin,
                "is_calibration_actor": actor_id in calibration_actors,
            }
        )

    dataset_manifest = _read_json(DATASET_MANIFEST_PATH)
    if int(dataset_manifest.get("samples", -1)) != len(samples):
        raise ValueError("dataset_manifest.json sample count is stale.")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "split_id": SPLIT_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "version": DATASET_VERSION,
            "label_version": LABEL_VERSION,
            "path": _relative_path(DATASET_PATH),
            "sample_count": len(samples),
            "size_bytes": DATASET_PATH.stat().st_size,
            "sha256": sha256_file(DATASET_PATH),
        },
        "sources": {
            "parent_split_manifest": {
                "path": _relative_path(BASE_SPLIT_MANIFEST_PATH),
                "sha256": sha256_file(BASE_SPLIT_MANIFEST_PATH),
            },
            "phase1_label_manifest": {
                "path": _relative_path(LABEL_MANIFEST_PATH),
                "sha256": sha256_file(LABEL_MANIFEST_PATH),
            },
            "candidate_match_summary": {
                "path": _relative_path(CANDIDATE_SUMMARY_PATH),
                "sha256": sha256_file(CANDIDATE_SUMMARY_PATH),
            },
            "dataset_manifest": {
                "path": _relative_path(DATASET_MANIFEST_PATH),
                "sha256": sha256_file(DATASET_MANIFEST_PATH),
            },
            "human_labels": {
                "path": _relative_path(HUMAN_LABELS_PATH),
                "sha256": sha256_file(HUMAN_LABELS_PATH),
            },
        },
        "config": {
            "group_key": "actor_norad_cat_id",
            "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "target_actor_counts": target_actor_counts,
            "base_seed": BASE_SEED,
            "selected_seed": selected_seed,
            "seed_candidates": SEED_CANDIDATES,
            "selection_inputs": ["dominant_class", "counterpart_type", "maneuver_presence"],
            "rare_class_missing_penalty": {"validation": 1.0, "test": 2.0},
            "assignment_source": "frozen parent Phase 0 manifest",
            "reselected_after_label_freeze": False,
        },
        "calibration": {
            "policy": "all samples from human-calibrated actors are train-only",
            "human_label_count": len(human_sample_indices),
            "human_sample_indices": human_sample_indices,
            "actor_count": len(calibration_actors),
            "actor_norad_cat_ids": sorted(calibration_actors, key=_actor_sort_key),
        },
        "actors": [
            {
                "actor_norad_cat_id": actor_id,
                "split": actor_assignment[actor_id],
                "is_calibration_actor": actor_id in calibration_actors,
            }
            for actor_id in sorted(actor_ids, key=_actor_sort_key)
        ],
        "audit": audit,
        "samples": manifest_samples,
    }
    validate_split_manifest(manifest, expected_sample_count=len(samples))

    if output_path.is_file():
        existing = _read_json(output_path)
        existing_comparable = dict(existing)
        existing_comparable.pop("generated_at", None)
        new_comparable = dict(manifest)
        new_comparable.pop("generated_at", None)
        if existing_comparable == new_comparable:
            return existing
    _atomic_write_json(output_path, manifest)
    return manifest


def main() -> None:
    manifest = generate_split_manifest()
    audit = manifest["audit"]
    print(f"Split manifest: {SPLIT_MANIFEST_PATH}")
    print(f"Dataset SHA-256: {manifest['dataset']['sha256']}")
    print(f"Selected seed: {manifest['config']['selected_seed']}")
    print(f"Actors: {audit['actor_counts']}")
    print(f"Samples: {audit['sample_counts']}")
    print(f"Calibration actors: {manifest['calibration']['actor_count']} (train-only)")


if __name__ == "__main__":
    main()
