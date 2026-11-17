"""Freeze Phase 1 labels and export the independent human blind-test batch.

Run this file directly in the ``intention-ir`` environment.  It does not train
any model.  Historical v3/v4 files are read-only inputs; every Phase 1 artifact
is written to a new, versioned path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


IDEA_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = IDEA_DIR / "generated_dataset" / "sgp4_large_maneuver_1min"
PILOT_DIR = DATASET_DIR / "multi_agent_pilot_300"
FRONT_DIR = DATASET_DIR / "multi_agent_optimized_1000"
REMAINDER_DIR = DATASET_DIR / "multi_agent_optimized_10400"
EXPERIMENTS_DIR = IDEA_DIR / "experiment_data"

HISTORICAL_DATASET_PATH = DATASET_DIR / "x_samples_optimized.json"
MAIN_DATASET_PATH = DATASET_DIR / "x_samples_frozen_v5_clean.json"
CANDIDATE_SUMMARY_PATH = DATASET_DIR / "candidate_match_summary.json"
HUMAN_LABELS_PATH = DATASET_DIR / "human_labels.json"
PARENT_SPLIT_PATH = (
    EXPERIMENTS_DIR / "splits" / "actor_70_15_15_calibration_train_v1.json"
)

FRONT_LABELS_PATH = (
    FRONT_DIR / "optimized_labels_00000_00999_multiagent_2026-08-10.json"
)
FRONT_RAW_RULES_PATH = FRONT_DIR / "multiagent_raw_labels_00000_00999.json"
FRONT_RICH_PATH = FRONT_DIR / "rich_samples_00000_00999.json"
REMAINDER_LABELS_PATH = (
    REMAINDER_DIR / "optimized_labels_01000_10399_multiagent_2026-08-11.json"
)
PERTURBATIONS_PATH = (
    REMAINDER_DIR / "uncertainty_perturbations_01000_10399_2026-08-11.json"
)

FREEZE_ID = "phase1_label_freeze_v1"
FREEZE_DATE = "2026-08-17"
MAIN_LABEL_VERSION = "multiagent_human_calibrated_v5_clean_20260817"
TRAIN_AUGMENTED_VERSION = "multiagent_human_calibrated_v5_aug188_train_only_20260817"
HISTORICAL_LABEL_VERSION = "multiagent_human_calibrated_v4_aug188_all_20260811"

LABEL_DIR = EXPERIMENTS_DIR / "labels" / FREEZE_ID
MAIN_LABELS_PATH = LABEL_DIR / "labels_main_clean_10400.json"
TRAIN_AUGMENTED_LABELS_PATH = LABEL_DIR / "labels_train_augmented_10400.json"
FRONT_AUDIT_PATH = LABEL_DIR / "front1000_reaggregation_audit.json"
LABEL_MANIFEST_PATH = LABEL_DIR / "label_manifest.json"
LABEL_REPORT_PATH = LABEL_DIR / "phase1_label_freeze_report.md"

BLIND_ID = "phase1_blind_test_v1"
BLIND_SEED = 20260817
BLIND_SAMPLE_COUNT = 200
BLIND_DIR = EXPERIMENTS_DIR / "human_annotation" / BLIND_ID
BLIND_DATASET_PATH = BLIND_DIR / "unlabeled_x_samples.json"
BLIND_SUMMARY_PATH = BLIND_DIR / "candidate_match_summary.json"
BLIND_MANIFEST_PATH = BLIND_DIR / "selection_manifest.json"
BLIND_INSTRUCTIONS_PATH = BLIND_DIR / "annotation_instructions.md"

CLASS_KEYS = tuple(f"c{index}" for index in range(8))
CLASS_NAMES = tuple(f"C{index}" for index in range(8))
SPLIT_NAMES = ("train", "validation", "test")


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"Expected a JSON object list: {path}")
    return payload


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_json_array_lines_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        output.write("[\n")
        for index, row in enumerate(rows):
            if index:
                output.write(",\n")
            output.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            )
        output.write("\n]\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(IDEA_DIR.resolve()).as_posix()


def validate_y(y: object, counterpart_type: str, sample_index: int) -> dict[str, float]:
    if not isinstance(y, dict) or set(y) != set(CLASS_KEYS):
        raise ValueError(f"Sample {sample_index}: y must contain exactly c0-c7.")
    result = {key: float(y[key]) for key in CLASS_KEYS}
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in result.values()):
        raise ValueError(f"Sample {sample_index}: invalid probability.")
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Sample {sample_index}: probabilities do not sum to one.")
    invalid = ("c6", "c7") if counterpart_type == "satellite" else ("c2", "c3", "c4")
    if any(result[key] != 0.0 for key in invalid):
        raise ValueError(f"Sample {sample_index}: counterpart class mask violation.")
    return result


def dominant(y: Mapping[str, float]) -> str:
    return f"C{max(range(8), key=lambda index: float(y[f'c{index}']))}"


def indexed_rows(rows: list[dict[str, Any]], expected_count: int) -> dict[int, dict[str, Any]]:
    result = {int(row["sample_index"]): row for row in rows}
    if len(result) != expected_count or sorted(result) != list(range(expected_count)):
        raise ValueError(f"Expected one row for every sample index 0-{expected_count - 1}.")
    return result


def audit_front_reaggregation() -> dict[str, Any]:
    """Prove that the archived first batch is bit-identical under the final formula."""
    if str(PILOT_DIR) not in sys.path:
        sys.path.insert(0, str(PILOT_DIR))
    from compare_agents import js as distribution_js
    from optimize_first_1000 import (
        VALID_CLASSES,
        aggregate,
        calibration_disagreement_thresholds,
        calibration_weights,
        normalize,
        run_agents,
    )

    samples = {int(row["sample_index"]): row for row in read_json_rows(FRONT_RICH_PATH)}
    if sorted(samples) != list(range(1000)):
        raise ValueError("Front rich samples do not cover 0-999.")
    rerun_agents = run_agents(samples)
    archived_agents = indexed_rows(read_json_rows(FRONT_RAW_RULES_PATH), 1000)
    rule_mismatches: list[dict[str, Any]] = []
    for index in range(1000):
        for name, rerun_by_index in rerun_agents.items():
            archived = archived_agents[index]["agents"][name]
            rerun = rerun_by_index[index]
            if any(
                archived[key] != rerun[key]
                for key in ("y", "dominant_label", "confidence")
            ):
                rule_mismatches.append({"sample_index": index, "rule": name})

    archive_dir = DATASET_DIR / "label_archive"
    source_rows = read_json_rows(archive_dir / "soft_labels_pilot_50_subagent.json")
    source_rows.extend(
        read_json_rows(archive_dir / "soft_labels_batch_00050_01049_subagent_2026-07-14.json")
    )
    source = {
        int(row["sample_index"]): row
        for row in source_rows
        if 0 <= int(row["sample_index"]) < 1000
    }
    if sorted(source) != list(range(1000)):
        raise ValueError("Archived source labels do not cover 0-999.")

    current = indexed_rows(read_json_rows(FRONT_LABELS_PATH), 1000)
    human = {
        int(row["sample_index"]): row["y"]
        for row in read_json_rows(HUMAN_LABELS_PATH)
        if int(row["sample_index"]) < 1000
    }
    weights = calibration_weights()
    thresholds = calibration_disagreement_thresholds()
    y_mismatches: list[int] = []
    metadata_mismatches: list[int] = []
    js_values: list[float] = []
    for index in range(1000):
        aggregated_y, details = aggregate(
            index,
            samples[index],
            source[index]["y"],
            rerun_agents,
            weights,
            thresholds,
        )
        human_override = index in human
        final_y = (
            normalize(human[index], VALID_CLASSES[samples[index]["counterpart_type"]])
            if human_override
            else aggregated_y
        )
        details.update(
            {
                "human_override": human_override,
                "original_dominant_label": dominant(source[index]["y"]),
                "dominant_changed": dominant(final_y) != dominant(source[index]["y"]),
                "js_from_original": round(distribution_js(final_y, source[index]["y"]), 6),
            }
        )
        if human_override:
            details.update(
                {
                    "dominant_guard_applied": False,
                    "strong_consensus_override": False,
                    "training_sample_weight": 1.0,
                    "review_candidate": False,
                }
            )
        if final_y != current[index]["y"]:
            y_mismatches.append(index)
        if details != current[index]["optimization"]:
            metadata_mismatches.append(index)
        js_values.append(distribution_js(final_y, current[index]["y"]))

    if rule_mismatches or y_mismatches or metadata_mismatches:
        raise ValueError("Front-1000 final-formula equivalence audit failed.")
    audit = {
        "schema_version": 1,
        "audit_id": "front1000_final_formula_equivalence_v1",
        "audit_date": FREEZE_DATE,
        "sample_count": 1000,
        "rule_output_comparisons": 6000,
        "rule_output_mismatch_count": 0,
        "label_exact_match_count": 1000,
        "label_mismatch_count": 0,
        "optimization_metadata_exact_match_count": 1000,
        "optimization_metadata_mismatch_count": 0,
        "max_js_current_vs_recomputed": max(js_values),
        "conclusion": (
            "The first 1000 labels are bit-identical under the final shared aggregation "
            "formula; only provenance/version naming differed between batches."
        ),
        "calibration": {
            "reliability_fit_label_count": 100,
            "direct_human_override_count_full_dataset": 104,
            "front_human_override_count": len(human),
            "disagreement_median": thresholds[0],
            "disagreement_p90": thresholds[1],
        },
        "source_sha256": {
            relative_path(FRONT_RICH_PATH): sha256_file(FRONT_RICH_PATH),
            relative_path(FRONT_RAW_RULES_PATH): sha256_file(FRONT_RAW_RULES_PATH),
            relative_path(FRONT_LABELS_PATH): sha256_file(FRONT_LABELS_PATH),
        },
    }
    write_json_atomic(FRONT_AUDIT_PATH, audit)
    return audit


def build_label_versions() -> tuple[dict[str, Any], dict[int, dict[str, float]]]:
    front = indexed_rows(read_json_rows(FRONT_LABELS_PATH), 1000)
    remainder_rows = read_json_rows(REMAINDER_LABELS_PATH)
    remainder = {int(row["sample_index"]): row for row in remainder_rows}
    if len(remainder) != 9400 or sorted(remainder) != list(range(1000, 10400)):
        raise ValueError("Remainder labels do not cover 1000-10399.")
    historical = {**front, **remainder}

    perturbation_rows = read_json_rows(PERTURBATIONS_PATH)
    perturbations = {int(row["sample_index"]): row for row in perturbation_rows}
    if len(perturbations) != 188:
        raise ValueError("Expected 188 uncertainty perturbations.")
    human = {int(row["sample_index"]): row["y"] for row in read_json_rows(HUMAN_LABELS_PATH)}
    if set(perturbations).intersection(human):
        raise ValueError("Uncertainty perturbations overlap human overrides.")

    split_manifest = read_json(PARENT_SPLIT_PATH)
    split_by_index = {
        int(row["sample_index"]): str(row["split"])
        for row in split_manifest["samples"]
    }
    if sorted(split_by_index) != list(range(10400)):
        raise ValueError("Parent split does not cover all samples.")

    candidate_rows = read_json_rows(CANDIDATE_SUMMARY_PATH)
    if len(candidate_rows) != 10400:
        raise ValueError("Candidate summary does not cover all samples.")

    main_rows: list[dict[str, Any]] = []
    train_augmented_rows: list[dict[str, Any]] = []
    main_y: dict[int, dict[str, float]] = {}
    split_perturbations = Counter()
    split_dominant_changes = Counter()
    for index in range(10400):
        counterpart_type = str(candidate_rows[index]["counterpart_type"])
        historical_y = validate_y(historical[index]["y"], counterpart_type, index)
        perturbation = perturbations.get(index)
        if perturbation is not None:
            before_y = validate_y(perturbation["before_y"], counterpart_type, index)
            after_y = validate_y(perturbation["after_y"], counterpart_type, index)
            if historical_y != after_y:
                raise ValueError(f"Historical after_y mismatch at sample {index}.")
            clean_y = before_y
            split_name = split_by_index[index]
            split_perturbations[split_name] += 1
            if dominant(before_y) != dominant(after_y):
                split_dominant_changes[split_name] += 1
        else:
            clean_y = historical_y
            after_y = historical_y
        if index in human and clean_y != validate_y(human[index], counterpart_type, index):
            raise ValueError(f"Human override mismatch at sample {index}.")
        main_y[index] = clean_y
        common = {
            "sample_index": index,
            "counterpart_type": counterpart_type,
            "human_override": index in human,
        }
        main_rows.append(
            {
                **common,
                "label_version": MAIN_LABEL_VERSION,
                "y": clean_y,
                "dominant_label": dominant(clean_y),
                "uncertainty_augmentation": "none",
            }
        )
        train_y = after_y if perturbation is not None and split_by_index[index] == "train" else clean_y
        train_augmented_rows.append(
            {
                **common,
                "label_version": TRAIN_AUGMENTED_VERSION,
                "y": train_y,
                "dominant_label": dominant(train_y),
                "uncertainty_augmentation": (
                    "documented_rule_view_mixture"
                    if perturbation is not None and split_by_index[index] == "train"
                    else "none"
                ),
            }
        )

    changed_train_only = sum(
        main_rows[index]["y"] != train_augmented_rows[index]["y"]
        for index in range(10400)
    )
    if changed_train_only != split_perturbations["train"]:
        raise ValueError("Train-only augmentation differs outside the frozen train subset.")
    write_json_atomic(MAIN_LABELS_PATH, main_rows)
    write_json_atomic(TRAIN_AUGMENTED_LABELS_PATH, train_augmented_rows)
    summary = {
        "historical_perturbation_count": len(perturbations),
        "historical_dominant_change_count": sum(split_dominant_changes.values()),
        "perturbation_counts_by_split": dict(split_perturbations),
        "dominant_change_counts_by_split": dict(split_dominant_changes),
        "main_dominant_counts": dict(sorted(Counter(row["dominant_label"] for row in main_rows).items())),
        "train_augmented_dominant_counts": dict(
            sorted(Counter(row["dominant_label"] for row in train_augmented_rows).items())
        ),
        "train_augmented_difference_count": changed_train_only,
        "human_override_count": len(human),
    }
    return summary, main_y


def non_y_digest_update(digest: Any, sample: Mapping[str, Any]) -> None:
    payload = dict(sample)
    payload.pop("y", None)
    digest.update(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def write_clean_dataset(main_y: Mapping[int, dict[str, float]]) -> dict[str, Any]:
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_processing import iter_json_samples

    temporary = MAIN_DATASET_PATH.with_suffix(MAIN_DATASET_PATH.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    source_non_y = hashlib.sha256()
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        output.write("[\n")
        for index, sample in iter_json_samples(HISTORICAL_DATASET_PATH):
            non_y_digest_update(source_non_y, sample)
            sample["y"] = main_y[index]
            if index:
                output.write(",\n")
            output.write(
                json.dumps(sample, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            )
            count += 1
        output.write("\n]\n")
    if count != 10400:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Expected 10400 samples, found {count}.")

    target_non_y = hashlib.sha256()
    verified = 0
    for index, sample in iter_json_samples(temporary):
        non_y_digest_update(target_non_y, sample)
        if sample.get("y") != main_y[index]:
            raise ValueError(f"Clean dataset label mismatch at sample {index}.")
        verified += 1
    if verified != 10400 or source_non_y.digest() != target_non_y.digest():
        temporary.unlink(missing_ok=True)
        raise ValueError("Clean dataset non-y verification failed.")
    os.replace(temporary, MAIN_DATASET_PATH)
    return {
        "sample_count": verified,
        "non_y_sha256": target_non_y.hexdigest(),
        "sha256": sha256_file(MAIN_DATASET_PATH),
        "size_bytes": MAIN_DATASET_PATH.stat().st_size,
    }


def source_hashes() -> dict[str, str]:
    paths = [
        HISTORICAL_DATASET_PATH,
        CANDIDATE_SUMMARY_PATH,
        HUMAN_LABELS_PATH,
        PARENT_SPLIT_PATH,
        FRONT_LABELS_PATH,
        FRONT_RAW_RULES_PATH,
        FRONT_RICH_PATH,
        REMAINDER_LABELS_PATH,
        PERTURBATIONS_PATH,
        PILOT_DIR / "human_comparison_results.json",
    ]
    paths.extend(sorted((REMAINDER_DIR / "rules").glob("*.json")))
    return {relative_path(path): sha256_file(path) for path in paths}


def write_label_manifest(
    front_audit: Mapping[str, Any],
    label_summary: Mapping[str, Any],
    dataset_summary: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "freeze_id": FREEZE_ID,
        "freeze_date": FREEZE_DATE,
        "status": "frozen_before_independent_human_annotation",
        "evidence_boundary": (
            "Main and augmented labels are soft pseudo-label supervision, not human ground truth."
        ),
        "decision": {
            "main": (
                "Use deterministic aggregation. Restore before_y for all 188 historical "
                "rule-view mixtures so validation/test targets contain no injected randomness."
            ),
            "train_augmented_control": (
                "Apply after_y only to the 132 perturbation rows assigned to train; keep "
                "validation and test deterministic."
            ),
            "historical": "Keep the original v4 all-split augmented dataset unchanged for reproduction.",
            "decision_frozen_before_model_training": True,
        },
        "versions": {
            "main": MAIN_LABEL_VERSION,
            "train_augmented_control": TRAIN_AUGMENTED_VERSION,
            "historical": HISTORICAL_LABEL_VERSION,
        },
        "calibration": {
            "reliability_fit_label_count": 100,
            "direct_override_label_count": 104,
            "independent_test_use": False,
        },
        "front1000_equivalence": dict(front_audit),
        "uncertainty_augmentation": dict(label_summary),
        "main_dataset": {
            "path": relative_path(MAIN_DATASET_PATH),
            **dict(dataset_summary),
        },
        "outputs": {
            "main_labels": {
                "path": relative_path(MAIN_LABELS_PATH),
                "sha256": sha256_file(MAIN_LABELS_PATH),
            },
            "train_augmented_labels": {
                "path": relative_path(TRAIN_AUGMENTED_LABELS_PATH),
                "sha256": sha256_file(TRAIN_AUGMENTED_LABELS_PATH),
            },
            "front1000_audit": {
                "path": relative_path(FRONT_AUDIT_PATH),
                "sha256": sha256_file(FRONT_AUDIT_PATH),
            },
        },
        "sources": source_hashes(),
        "immutability_policy": (
            "Do not overwrite these files after training starts. Any label change requires a new "
            "version, manifest, dataset hash, and split-manifest file."
        ),
    }
    write_json_atomic(LABEL_MANIFEST_PATH, manifest)
    return manifest


def proportional_quotas(rows: list[dict[str, Any]], count: int) -> dict[tuple[str, str], int]:
    pool_counts = Counter(
        (str(row["counterpart_type"]), str(row["maneuver_presence"])) for row in rows
    )
    exact = {key: count * value / len(rows) for key, value in pool_counts.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remaining = count - sum(quotas.values())
    order = sorted(exact, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def weighted_stratum_choice(
    rng: random.Random,
    available: list[tuple[str, str]],
    remaining: Mapping[tuple[str, str], int],
) -> tuple[str, str]:
    total = sum(remaining[key] for key in available)
    position = rng.randrange(total)
    for key in sorted(available):
        position -= remaining[key]
        if position < 0:
            return key
    raise RuntimeError("Unable to choose a stratum.")


def select_blind_rows(split_manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    test_rows = [dict(row) for row in split_manifest["samples"] if row["split"] == "test"]
    if len(test_rows) != 1560 or len({row["actor_norad_cat_id"] for row in test_rows}) != 39:
        raise ValueError("Unexpected Phase 0 test pool.")
    quotas = proportional_quotas(test_rows, BLIND_SAMPLE_COUNT)
    remaining = dict(quotas)
    rng = random.Random(BLIND_SEED)
    by_actor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in test_rows:
        by_actor[str(row["actor_norad_cat_id"])].append(row)

    selected: list[dict[str, Any]] = []
    selected_indices: set[int] = set()
    actor_ids = sorted(by_actor)
    rng.shuffle(actor_ids)
    for actor_id in actor_ids:
        candidates_by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in by_actor[actor_id]:
            key = (str(row["counterpart_type"]), str(row["maneuver_presence"]))
            if remaining.get(key, 0) > 0:
                candidates_by_stratum[key].append(row)
        available = list(candidates_by_stratum)
        if not available:
            raise ValueError(f"No remaining blind-test quota for actor {actor_id}.")
        key = weighted_stratum_choice(rng, available, remaining)
        row = rng.choice(candidates_by_stratum[key])
        selected.append(row)
        selected_indices.add(int(row["sample_index"]))
        remaining[key] -= 1

    pool_by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in test_rows:
        index = int(row["sample_index"])
        if index not in selected_indices:
            pool_by_stratum[(str(row["counterpart_type"]), str(row["maneuver_presence"]))].append(row)
    for key in sorted(remaining):
        needed = remaining[key]
        additions = rng.sample(pool_by_stratum[key], needed)
        selected.extend(additions)
        selected_indices.update(int(row["sample_index"]) for row in additions)
    rng.shuffle(selected)

    selected_counts = Counter(
        (str(row["counterpart_type"]), str(row["maneuver_presence"])) for row in selected
    )
    if len(selected) != BLIND_SAMPLE_COUNT or len(selected_indices) != BLIND_SAMPLE_COUNT:
        raise ValueError("Blind-test selection has missing or duplicate samples.")
    if selected_counts != Counter(quotas):
        raise ValueError("Blind-test stratum quotas were not preserved.")
    if len({str(row["actor_norad_cat_id"]) for row in selected}) != 39:
        raise ValueError("Blind-test selection does not cover every test actor.")
    return selected, {
        "test_pool_count": len(test_rows),
        "test_actor_count": 39,
        "selected_count": len(selected),
        "selected_actor_count": 39,
        "pool_strata": {
            f"{key[0]}__{key[1]}": value
            for key, value in sorted(
                Counter((row["counterpart_type"], row["maneuver_presence"]) for row in test_rows).items()
            )
        },
        "selected_strata": {
            f"{key[0]}__{key[1]}": value for key, value in sorted(selected_counts.items())
        },
    }


def export_blind_dataset() -> dict[str, Any]:
    if (BLIND_DIR / "human_labels.json").exists():
        raise FileExistsError(
            "Blind annotations already exist; refusing to regenerate or reorder the annotation batch."
        )
    split_manifest = read_json(PARENT_SPLIT_PATH)
    selected, selection_summary = select_blind_rows(split_manifest)
    selected_indices = {int(row["sample_index"]) for row in selected}

    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_processing import iter_json_samples

    source_samples: dict[int, dict[str, Any]] = {}
    for index, sample in iter_json_samples(MAIN_DATASET_PATH):
        if index in selected_indices:
            source_samples[index] = sample
    if set(source_samples) != selected_indices:
        raise ValueError("Selected blind samples are missing from the main dataset.")

    candidate_rows = read_json_rows(CANDIDATE_SUMMARY_PATH)
    blind_samples: list[dict[str, Any]] = []
    blind_summaries: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for local_index, row in enumerate(selected):
        sample_index = int(row["sample_index"])
        source = source_samples[sample_index]
        blind_sample = {
            "sample_index": sample_index,
            "actor_trajectory": source["actor_trajectory"],
            "counterpart_trajectory": source["counterpart_trajectory"],
            "maneuver_events": source.get("maneuver_events", []),
        }
        forbidden = {
            "y",
            "matching_score",
            "score_components",
            "score_metrics",
            "dominant_label",
            "prediction",
        }
        if forbidden.intersection(blind_sample):
            raise ValueError(f"Blind sample {sample_index} contains a forbidden field.")
        metadata = candidate_rows[sample_index]
        for key in ("actor", "actor_norad_cat_id", "counterpart", "counterpart_type"):
            if str(row[key]) != str(metadata[key]):
                raise ValueError(f"Split/candidate mismatch at sample {sample_index}: {key}.")
        blind_summary = {
            "sample_index": sample_index,
            "actor": metadata["actor"],
            "actor_norad_cat_id": str(metadata["actor_norad_cat_id"]),
            "counterpart": metadata["counterpart"],
            "counterpart_type": metadata["counterpart_type"],
        }
        blind_samples.append(blind_sample)
        blind_summaries.append(blind_summary)
        manifest_rows.append(
            {
                "local_index": local_index,
                "display_index": local_index + 1,
                "sample_index": sample_index,
                **{key: blind_summary[key] for key in blind_summary if key != "sample_index"},
                "maneuver_presence": row["maneuver_presence"],
                "sample_sha256": canonical_sha256(blind_sample),
            }
        )

    write_json_array_lines_atomic(BLIND_DATASET_PATH, blind_samples)
    write_json_atomic(BLIND_SUMMARY_PATH, blind_summaries)
    manifest = {
        "schema_version": 1,
        "blind_test_id": BLIND_ID,
        "created_date": FREEZE_DATE,
        "status": "awaiting_human_annotation",
        "sample_count": BLIND_SAMPLE_COUNT,
        "selection_seed": BLIND_SEED,
        "source_split": "test",
        "source_split_id": split_manifest["split_id"],
        "selection_inputs": ["actor_norad_cat_id", "counterpart_type", "maneuver_presence"],
        "excluded_selection_inputs": [
            "pseudo_label",
            "matching_score",
            "rule_disagreement",
            "model_prediction",
        ],
        "sampling": {
            "method": "proportional type-by-maneuver strata with every test actor represented",
            **selection_summary,
        },
        "files": {
            "dataset": {
                "path": BLIND_DATASET_PATH.name,
                "sha256": sha256_file(BLIND_DATASET_PATH),
                "contains_y": False,
                "contains_matching_score": False,
            },
            "candidate_summary": {
                "path": BLIND_SUMMARY_PATH.name,
                "sha256": sha256_file(BLIND_SUMMARY_PATH),
                "contains_matching_score": False,
            },
        },
        "sources": {
            "main_dataset_sha256": sha256_file(MAIN_DATASET_PATH),
            "parent_split_manifest_sha256": sha256_file(PARENT_SPLIT_PATH),
            "candidate_summary_sha256": sha256_file(CANDIDATE_SUMMARY_PATH),
        },
        "samples": manifest_rows,
    }
    write_json_atomic(BLIND_MANIFEST_PATH, manifest)
    instructions = f"""# Phase 1 独立人工盲标说明

请在人工标注平台中只导入本目录的：

```text
{BLIND_DATASET_PATH.name}
```

本批次共 {BLIND_SAMPLE_COUNT} 条，来自冻结 test actors；文件已删除 `y`、`matching_score`、规则分歧和模型输出。平台会自动读取同目录的 `candidate_match_summary.json`，无需单独导入。

## 标注要求

- C0：无明显相关意图
- C1：自然接近 / 自然过境
- C2：目标抵近侦察
- C3：高速穿越 / 近距压迫
- C4：邻近驻留 / 伴飞占位
- C5：轨道预置 / 可达性增强
- C6：重点区域覆盖增强
- C7：覆盖范围扩展 / 任务区转移

请根据可见轨迹证据填写 C0-C7 概率并保证总和为 1。平台会根据卫星/地面目标自动禁用不适用类别。可以分多次完成；每次点击“保存标注”后，结果会自动写入本目录的 `human_labels.json`。

完成标准：界面进度显示 `200/200`。不要把其他目录中的 `soft_labels*.json` 或 `human_labels.json` 复制到本目录。
"""
    BLIND_INSTRUCTIONS_PATH.write_text(instructions, encoding="utf-8")
    return manifest


def write_report(
    front_audit: Mapping[str, Any],
    label_summary: Mapping[str, Any],
    dataset_summary: Mapping[str, Any],
    blind_manifest: Mapping[str, Any],
) -> None:
    lines = [
        "# Phase 1 监督信号冻结报告",
        "",
        f"冻结日期：`{FREEZE_DATE}`",
        f"主标签版本：`{MAIN_LABEL_VERSION}`",
        "",
        "## 已完成",
        "",
        f"- 前1,000条最终公式重算：{front_audit['label_exact_match_count']}/1000 完全一致。",
        "- 主版恢复188条不确定性混合前的确定性 `before_y`。",
        f"- 训练增强对照只改变 train 中 {label_summary['train_augmented_difference_count']} 条。",
        "- validation/test 的主版和训练增强对照均使用确定性标签。",
        f"- 主数据共 {dataset_summary['sample_count']} 条，非 `y` 内容保持一致。",
        f"- 独立盲标批次共 {blind_manifest['sample_count']} 条，覆盖39个 test actors。",
        "",
        "## 状态边界",
        "",
        "- 当前主标签仍是规则与人工校准形成的 soft pseudo-label，不是真实意图真值。",
        "- 104条旧人工标签用于校准/覆盖，不能作为独立 test。",
        "- Phase 1 目前停在 `awaiting_human_annotation`；盲标返回后才能冻结独立人工 test。",
        "",
    ]
    LABEL_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if (BLIND_DIR / "human_labels.json").exists():
        raise FileExistsError(
            "Blind annotations already exist; refusing to rewrite any frozen Phase 1 artifact."
        )
    print("[1/5] Auditing front-1000 final-formula equivalence...")
    front_audit = audit_front_reaggregation()
    print("[2/5] Building frozen main and train-only augmentation label versions...")
    label_summary, main_y = build_label_versions()
    print("[3/5] Writing and verifying the clean frozen dataset...")
    dataset_summary = write_clean_dataset(main_y)
    print("[4/5] Freezing label manifest and exporting the blind annotation batch...")
    write_label_manifest(front_audit, label_summary, dataset_summary)
    blind_manifest = export_blind_dataset()
    write_report(front_audit, label_summary, dataset_summary, blind_manifest)
    print("[5/5] Phase 1 pre-annotation artifacts passed all checks.")
    print(json.dumps({
        "label_version": MAIN_LABEL_VERSION,
        "main_dataset": str(MAIN_DATASET_PATH),
        "blind_dataset_to_import": str(BLIND_DATASET_PATH),
        "blind_sample_count": blind_manifest["sample_count"],
        "status": "awaiting_human_annotation",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
