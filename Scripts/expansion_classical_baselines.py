"""Refit the frozen analytic RF and type prior on prepared expansion inputs.

The caller loads Train + Validation only. RF history is indexed by tree count;
the deterministic type prior has one fitted point. Running this file executes
only a small synthetic check, never a dataset training run.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor

import phase3_rf_baseline as rf
from model_evaluate import compute_metrics
from phase2_experiment import _prior_probabilities


SEEDS = (7, 17, 42, 73, 101)
TREE_COUNTS = (50, 100, 150, 200, 250, 300)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        # Windows indexers can briefly hold a just-written history file open.
        for attempt in range(10):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.1)
    finally:
        temporary.unlink(missing_ok=True)


def _partition_check(
    samples: Sequence[Mapping[str, Any]],
    train_positions: Sequence[int],
    validation_positions: Sequence[int],
    split: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Check positional indexing against global sample IDs and actor groups."""
    positions = {}
    for name, values in (("train", train_positions), ("validation", validation_positions)):
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
               for value in values):
            raise ValueError(f"{name} positions must be integers.")
        array = np.asarray(values, dtype=np.int64)
        if array.ndim != 1 or len(array) == 0 or len(set(array.tolist())) != len(array):
            raise ValueError(f"{name} positions must be nonempty, unique, and one-dimensional.")
        if np.any(array < 0) or np.any(array >= len(samples)):
            raise ValueError(f"{name} position is outside the loaded sample list.")
        positions[name] = array
    train_set = set(positions["train"].tolist())
    validation_set = set(positions["validation"].tolist())
    if train_set & validation_set or train_set | validation_set != set(range(len(samples))):
        raise ValueError("Loaded inputs must contain exactly disjoint Train + Validation positions.")

    records = split["samples"]
    records_by_id = {int(row["sample_index"]): row for row in records}
    if len(records_by_id) != len(records):
        raise ValueError("Split manifest contains duplicate sample_index values.")
    actor_groups: dict[str, str] = {}
    for row in records:
        actor, group = str(row["actor_norad_cat_id"]), row["split"]
        if group not in ("train", "validation", "test"):
            raise ValueError(f"Unknown split: {group!r}.")
        if actor in actor_groups and actor_groups[actor] != group:
            raise ValueError(f"Actor {actor} appears in more than one split.")
        actor_groups[actor] = group
    sample_ids = [int(sample["sample_index"]) for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Loaded inputs contain duplicate sample_index values.")
    audit: dict[str, Any] = {"loaded_test_sample_count": 0, "actor_isolation_verified": True}
    for name, array in positions.items():
        actual = [sample_ids[int(position)] for position in array]
        expected = {index for index, row in records_by_id.items() if row["split"] == name}
        if set(actual) != expected:
            raise ValueError(f"Loaded {name} sample IDs differ from the complete manifest split.")
        audit[name] = {
            "sample_count": len(actual),
            "sample_indices_in_fit_order" if name == "train" else "sample_indices": actual,
            "actor_ids": sorted({str(records_by_id[index]["actor_norad_cat_id"]) for index in actual}),
        }
    return positions["train"], positions["validation"], audit


def _forest_probabilities(
    model: RandomForestRegressor, features: np.ndarray, masks: np.ndarray,
) -> np.ndarray:
    """Apply the existing RF probability convention to cached features."""
    probabilities = np.maximum(np.asarray(model.predict(features), dtype=np.float64), 0.0) * masks
    if probabilities.shape != masks.shape or not np.isfinite(probabilities).all():
        raise ValueError("RF predictions must be finite with shape [N, 8].")
    totals = probabilities.sum(axis=1, keepdims=True)
    empty = totals[:, 0] <= 0.0
    if empty.any():
        probabilities[empty] = masks[empty]
        totals = probabilities.sum(axis=1, keepdims=True)
    return (probabilities / totals).astype(np.float32)


def run_classical_baselines(
    samples: Sequence[Mapping[str, Any]],
    train_positions: Sequence[int],
    validation_positions: Sequence[int],
    split: Mapping[str, Any],
    output_root: str | Path,
    metadata: Mapping[str, Any],
    *,
    seeds: Sequence[int] = SEEDS,
) -> dict[str, Any]:
    """Train on prepared inputs, returning paths and final Train/Validation metrics.

    A completed ``output_root/classical`` is reused only after metadata, source,
    partition, and artifact-hash checks. Partial runs are never overwritten.
    Validation metrics do not select trees, features, or parameters.
    ``seeds`` defaults to the five old protocol seeds; a subset supports checks.
    """
    train, validation, partition = _partition_check(samples, train_positions, validation_positions, split)
    if (len(seeds) == 0 or len(set(seeds)) != len(seeds)
            or any(isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) for seed in seeds)):
        raise ValueError("Seeds must be a nonempty sequence of unique integers.")
    if TREE_COUNTS[-1] != rf.RF_CONFIG["n_estimators"]:
        raise ValueError("The fixed tree schedule no longer matches the old RF configuration.")
    script_dir = Path(__file__).resolve().parent
    source_names = (
        "expansion_classical_baselines.py", "phase3_rf_baseline.py",
        "data_processing.py", "phase2_experiment.py", "model_evaluate.py",
    )
    common = {
        "schema_version": 1,
        "experiment": dict(metadata),
        "rf_seeds": [int(seed) for seed in seeds],
        "fit_split": "train", "evaluation_splits": ["train", "validation"],
        "validation_used_for_fitting_or_selection": False,
        "targets": "caller_prepared_embedded_soft_y",
        "runtime": {"numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "source_sha256": {name: hashlib.sha256((script_dir / name).read_bytes()).hexdigest() for name in source_names},
        "loss_definition": {
            "soft_nll": "mean_i(-sum_c(y_ic * log(clip(p_ic, 1e-12, 1))))",
            "brier_score_sum_over_classes": "mean_i(sum_c((p_ic-y_ic)^2))",
            "train_loss_is_in_sample": True,
        },
    }
    root = Path(output_root).resolve() / "classical"
    if root.exists():
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("Existing classical output is partial; use a new output root.")
        completed = json.loads(manifest_path.read_text(encoding="utf-8"))
        if completed.get("status") != "complete" or any(completed.get(key) != value for key, value in common.items()):
            raise ValueError("Existing classical output is incomplete or its protocol/source/metadata differs.")
        expected_files = {"partition_audit.json"}
        expected_runs = [("counterpart_type_prior", None)] + [(rf.BASELINE_ID, int(seed)) for seed in seeds]
        if [(run["baseline_id"], run["seed"]) for run in completed["runs"]] != expected_runs:
            raise ValueError("Completed classical manifest has unexpected run identities.")
        for baseline_id, seed in expected_runs:
            relative = Path(baseline_id) if seed is None else Path(baseline_id) / f"seed_{seed}"
            expected_files.update((relative / name).as_posix() for name in ("model.pkl", "metadata.json", "history.json", "final_metrics.json"))
        if set(completed.get("artifact_sha256", {})) != expected_files:
            raise ValueError("Completed classical artifact inventory differs from the expected files.")
        for relative, expected_hash in completed["artifact_sha256"].items():
            artifact = root / relative
            if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"Completed classical artifact failed hash verification: {relative}.")
        if json.loads((root / "partition_audit.json").read_text(encoding="utf-8")) != partition:
            raise ValueError("Completed classical partition or fit order differs from the loaded inputs.")
        return {"output_dir": str(root), "manifest_path": str(manifest_path), **completed}

    features = rf.extract_feature_matrix(samples)
    targets = rf._targets(samples)
    masks = np.stack([rf._counterpart_class_mask(sample) for sample in samples])
    training_samples = [samples[int(position)] for position in train]
    priors, type_counts = _prior_probabilities(training_samples)
    # Slice once. Feature extraction is per sample, with no fitted normalization.
    train_features, train_targets = features[train], targets[train]
    # An interrupted run is retained for inspection; restarting needs a new root.
    root.mkdir(parents=True, exist_ok=False)
    _write_json(root / "partition_audit.json", partition)
    manifest = {**common, "status": "running", "runs": []}
    _write_json(root / "manifest.json", manifest)

    def metrics_by_split(probabilities: np.ndarray) -> dict[str, Any]:
        return {
            "train": compute_metrics(targets[train], probabilities[train]),
            "validation": compute_metrics(targets[validation], probabilities[validation]),
        }

    prior_root = root / "counterpart_type_prior"
    prior_root.mkdir()
    prior_artifact = {
        **common, "baseline_id": "counterpart_type_prior", "seed": None,
        "fit_sample_count": len(train), "counterpart_type_counts": type_counts,
        "probabilities": priors, "fit_method": "train_soft_label_mean_per_counterpart_type",
    }
    prior_predictions = np.asarray([
        priors["satellite" if int(np.asarray(sample["counterpart_type"]).argmax()) == 0 else "ground_region"]
        for sample in samples
    ], dtype=np.float64)
    prior_metrics = metrics_by_split(prior_predictions)
    prior_history = {
        "baseline_id": "counterpart_type_prior", "step_axis": "fit_count",
        "history_note": "One deterministic fit; this is a single point, not an epoch history.",
        "history": [{"fit_count": 1, **prior_metrics}],
    }
    with (prior_root / "model.pkl").open("xb") as handle:
        pickle.dump(prior_artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    _write_json(prior_root / "metadata.json", prior_artifact)
    _write_json(prior_root / "history.json", prior_history)
    _write_json(prior_root / "final_metrics.json", prior_metrics)
    manifest["runs"].append({"baseline_id": "counterpart_type_prior", "seed": None,
                             "output_dir": str(prior_root), "history_rows": 1, "final_metrics": prior_metrics})
    _write_json(root / "manifest.json", manifest)

    for seed in seeds:
        seed = int(seed)
        run_root = root / rf.BASELINE_ID / f"seed_{seed}"
        run_root.mkdir(parents=True)
        history = {"baseline_id": rf.BASELINE_ID, "seed": seed, "step_axis": "tree_count",
                   "tree_counts": list(TREE_COUNTS), "history": []}
        model = RandomForestRegressor(**rf.RF_CONFIG, random_state=seed, warm_start=True)
        cumulative_fit_seconds = 0.0
        for tree_count in TREE_COUNTS:
            model.set_params(n_estimators=tree_count)
            started = time.perf_counter()
            model.fit(train_features, train_targets)
            fit_seconds = time.perf_counter() - started
            cumulative_fit_seconds += fit_seconds
            metrics = metrics_by_split(_forest_probabilities(model, features, masks))
            history["history"].append({
                "tree_count": tree_count, "fit_seconds_this_stage": fit_seconds,
                "cumulative_fit_seconds": cumulative_fit_seconds, **metrics,
            })
            # Persist every completed stage, including losses before interruption.
            _write_json(run_root / "history.json", history)
        model.set_params(warm_start=False)
        model_path = run_root / "model.pkl"
        rf.save(model, model_path)
        run_metadata = {
            **common, **rf.model_metadata(model), "feature_names": list(rf.FEATURE_NAMES),
            "fit_sample_count": len(train), "training_mode": "refit_from_scratch",
            "history_axis": "tree_count", "tree_counts": list(TREE_COUNTS),
            "staging_method": "warm_start_same_order_same_seed_then_restore_warm_start_false",
            "final_selection": "fixed_300_trees_no_validation_selection",
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        }
        _write_json(run_root / "metadata.json", run_metadata)
        _write_json(run_root / "final_metrics.json", metrics)
        manifest["runs"].append({"baseline_id": rf.BASELINE_ID, "seed": seed,
                                 "output_dir": str(run_root), "history_rows": len(history["history"]),
                                 "final_metrics": metrics})
        _write_json(root / "manifest.json", manifest)
    manifest["status"] = "complete"
    manifest["artifact_sha256"] = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file() and path != root / "manifest.json"
    }
    _write_json(root / "manifest.json", manifest)
    return {"output_dir": str(root), "manifest_path": str(root / "manifest.json"), **manifest}


def _self_check() -> dict[str, Any]:
    """Small synthetic equivalence, leakage, history, and non-overwrite checks."""
    rng = np.random.default_rng(815)
    samples = []
    records = []
    for index in range(32):
        satellite = index % 2 == 0
        mask = rf.SATELLITE_CLASS_MASK if satellite else rf.GROUND_REGION_CLASS_MASK
        target = rng.uniform(size=8) * mask
        sample = {
            "sample_index": np.int64(index),
            "analytic_factual": rng.normal(size=(7, 12)).astype(np.float32),
            "analytic_difference": rng.normal(size=(7, 12)).astype(np.float32),
            "maneuver_global": rng.normal(size=len(rf.MANEUVER_GLOBAL_FEATURES)).astype(np.float32),
            "error_features": rng.normal(size=len(rf.ERROR_FEATURES)).astype(np.float32),
            "error_mask": np.asarray([1, 0, 1], dtype=np.float32),
            "counterpart_type": np.asarray([1, 0] if satellite else [0, 1], dtype=np.float32),
            "y": target / target.sum(),
        }
        samples.append(sample)
        records.append({"sample_index": index, "actor_norad_cat_id": str(1000 + index // 4),
                        "split": "train" if index < 24 else "validation"})
    # Test metadata may be present, but no Test model input is loaded.
    records.append({"sample_index": 40, "actor_norad_cat_id": "9999", "split": "test"})
    split = {"samples": records}
    train, validation = list(range(24)), list(range(24, 32))
    check_root = Path(__file__).resolve().parents[1] / "experiment_data/expansion_18000_v1/preflight/classical"
    check_root.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.mkdtemp(prefix="synthetic_", dir=check_root)
    root = Path(temporary)
    result = run_classical_baselines(samples, train, validation, split, root / "first", {"check": "synthetic"}, seeds=(17,))
    staged = rf.load(Path(result["output_dir"]) / rf.BASELINE_ID / "seed_17/model.pkl")
    one_shot = rf.fit(samples[:24], seed=17)
    np.testing.assert_array_equal(rf.predict(staged, samples), rf.predict(one_shot, samples))
    assert [tree.random_state for tree in staged.estimators_] == [tree.random_state for tree in one_shot.estimators_]
    assert staged.get_params() == one_shot.get_params()
    cached = _forest_probabilities(staged, rf.extract_feature_matrix(samples),
                                   np.stack([rf._counterpart_class_mask(sample) for sample in samples]))
    np.testing.assert_array_equal(cached, rf.predict(staged, samples))
    for run in result["runs"]:
        logged = json.loads((Path(run["output_dir"]) / "history.json").read_text(encoding="utf-8"))
        assert len(logged["history"]) == (1 if run["seed"] is None else 6)
        assert all("soft_nll" in row["train"] and "brier_score_sum_over_classes" in row["validation"]
                   for row in logged["history"])
    changed = [dict(sample) for sample in samples]
    for position in validation:
        allowed = np.flatnonzero(rf._counterpart_class_mask(changed[position]))
        changed[position]["y"] = np.zeros(8, dtype=np.float64)
        changed[position]["y"][allowed[-1]] = 1.0
    second = run_classical_baselines(changed, train, validation, split, root / "second", {"check": "changed_validation_y"}, seeds=(17,))
    second_model = rf.load(Path(second["output_dir"]) / rf.BASELINE_ID / "seed_17/model.pkl")
    np.testing.assert_array_equal(rf.predict(staged, samples), rf.predict(second_model, samples))
    first_prior = json.loads((Path(result["output_dir"]) / "counterpart_type_prior/metadata.json").read_text(encoding="utf-8"))
    second_prior = json.loads((Path(second["output_dir"]) / "counterpart_type_prior/metadata.json").read_text(encoding="utf-8"))
    assert first_prior["probabilities"] == second_prior["probabilities"]
    assert result["runs"][0]["final_metrics"]["validation"]["soft_nll"] != second["runs"][0]["final_metrics"]["validation"]["soft_nll"]
    bad_actor_split = json.loads(json.dumps(split))
    bad_actor_split["samples"][24]["actor_norad_cat_id"] = "1000"
    failures = 0
    for arguments in ((train, validation + [0], split), (train, validation[:-1], split),
                      (train, validation, bad_actor_split)):
        try:
            _partition_check(samples, *arguments)
        except ValueError:
            failures += 1
    assert failures == 3
    from unittest.mock import patch
    with patch.object(RandomForestRegressor, "fit", side_effect=AssertionError("Resume attempted fitting.")):
        resumed = run_classical_baselines(samples, train, validation, split, root / "first", {"check": "synthetic"}, seeds=(17,))
    assert resumed == result
    try:
        run_classical_baselines(samples, train, validation, split, root / "first", {}, seeds=(17,))
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched existing output was not protected.")
    damaged = Path(result["output_dir"]) / "counterpart_type_prior/history.json"
    damaged.write_text(damaged.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    try:
        run_classical_baselines(samples, train, validation, split, root / "first", {"check": "synthetic"}, seeds=(17,))
    except ValueError:
        pass
    else:
        raise AssertionError("Modified completed artifact was not rejected.")
    return {"status": "passed", "artifacts_dir": str(root), "synthetic_sample_count": 32, "rf_trees": 300,
            "staged_equals_one_shot": "exact probabilities, tree seeds, and final parameters",
            "changed_validation_targets_leave_models_unchanged": True,
            "overlap_missing_positions_actor_leak_rejected": True,
            "completed_resume_without_fit": True, "mismatch_and_tamper_rejected": True,
            "existing_outputs_protected": True, "rf_history_rows": 6, "prior_history_rows": 1}


if __name__ == "__main__":
    report = _self_check()
    _write_json(Path(report["artifacts_dir"]) / "selfcheck_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
