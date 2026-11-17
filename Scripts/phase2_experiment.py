"""Phase 2 core experiment orchestration without command-line arguments.

Directly running this file performs only the train/validation stage.  It never
loads the pseudo-test split or the in-progress blind human labels.  Final test
evaluation is deliberately exposed through ``phase2_final_evaluate.py`` so the
two gates cannot be crossed accidentally during model development.
"""

from __future__ import annotations

import json
import math
import statistics
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from data_processing import load_training_data
from experiment_setup import (
    DATASET_PATH,
    DATASET_VERSION,
    LABEL_VERSION,
    SPLIT_ID,
    SPLIT_MANIFEST_PATH,
    load_split_indices,
    sha256_file,
    verify_manifest_dataset,
)
from model import SUPPORTED_MODEL_VARIANTS
from model_evaluate import (
    CLASS_NAMES,
    PSEUDO_TARGET_SOURCE,
    evaluate_checkpoint,
    evaluate_probability_arrays,
    evaluation_output_paths,
)
from model_train import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    RUN_DATA_TAG,
    WEIGHT_DECAY,
    checkpoint_path_for,
    training_log_path_for,
    train_model,
)


IDEA_DIR = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = IDEA_DIR / "experiment_data"
PHASE2_ID = "phase2_core_v2"
PHASE2_PROTOCOL_PATH = EXPERIMENTS_DIR / "configs" / f"{PHASE2_ID}.json"

NEURAL_VARIANTS = ("factual_only", "full", "full_without_analytic")
PRIOR_VARIANT = "counterpart_type_prior"
PHASE2_SEEDS = (7, 17, 42, 73, 101)

LABEL_MANIFEST_PATH = (
    EXPERIMENTS_DIR / "labels" / "phase1_label_freeze_v1" / "label_manifest.json"
)
DATASET_DIR = DATASET_PATH.parent
DISAGREEMENT_SOURCES = (
    DATASET_DIR
    / "multi_agent_optimized_1000"
    / "optimized_labels_00000_00999_multiagent_2026-08-10.json",
    DATASET_DIR
    / "multi_agent_optimized_10400"
    / "optimized_labels_01000_10399_multiagent_2026-08-11.json",
)

BLIND_DIR = EXPERIMENTS_DIR / "human_annotation" / "phase1_blind_test_v1"
BLIND_SELECTION_PATH = BLIND_DIR / "selection_manifest.json"
BLIND_DATASET_PATH = BLIND_DIR / "unlabeled_x_samples.json"
BLIND_LIVE_LABELS_PATH = BLIND_DIR / "human_labels.json"
BLIND_FROZEN_LABELS_PATH = BLIND_DIR / "human_labels_frozen_v1.json"
BLIND_LABEL_MANIFEST_PATH = BLIND_DIR / "human_label_manifest_v1.json"
BLIND_LABEL_VERSION = "blind_human_frozen_v1"


def _run_stem(model_variant: str, random_seed: int | None) -> str:
    seed_tag = "seednone" if random_seed is None else f"seed{random_seed}"
    return f"{RUN_DATA_TAG}__{SPLIT_ID}__{model_variant}__{seed_tag}"


PRIOR_ARTIFACT_PATH = (
    EXPERIMENTS_DIR / "checkpoints" / f"{_run_stem(PRIOR_VARIANT, None)}.json"
)


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(IDEA_DIR.resolve()).as_posix()


def _load_disagreement_values() -> dict[int, float]:
    values: dict[int, float] = {}
    for path in DISAGREEMENT_SOURCES:
        rows = _read_json(path)
        if not isinstance(rows, list):
            raise ValueError(f"Disagreement source must be a JSON list: {path}")
        for row in rows:
            sample_index = int(row["sample_index"])
            value = float(row["optimization"]["generalized_js"])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"Invalid generalized_js at sample {sample_index}.")
            if sample_index in values:
                raise ValueError(f"Duplicate disagreement sample_index: {sample_index}")
            values[sample_index] = value
    if sorted(values) != list(range(10400)):
        raise ValueError("Disagreement sources do not cover sample_index 0-10399 exactly.")
    return values


def _protocol_payload() -> dict[str, Any]:
    if tuple(SUPPORTED_MODEL_VARIANTS) != (
        "full",
        "factual_only",
        "full_without_analytic",
    ):
        raise ValueError("model.py Phase 2 variants differ from the frozen protocol.")
    manifest, split_indices = load_split_indices(SPLIT_MANIFEST_PATH)
    verify_manifest_dataset(manifest, DATASET_PATH, verify_hash=True)
    disagreement = _load_disagreement_values()
    train_disagreement = [disagreement[index] for index in split_indices["train"]]
    threshold = float(statistics.median(train_disagreement))

    code_paths = (
        Path(__file__).resolve().with_name("data_processing.py"),
        Path(__file__).resolve().with_name("experiment_setup.py"),
        Path(__file__).resolve().with_name("model.py"),
        Path(__file__).resolve().with_name("model_train.py"),
        Path(__file__).resolve().with_name("model_evaluate.py"),
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("phase2_pseudo_evaluate.py"),
        Path(__file__).resolve().with_name("phase2_final_evaluate.py"),
        Path(__file__).resolve().with_name("phase2_figures.py"),
        Path(__file__).resolve().with_name("phase2_smoke_test.py"),
        Path(__file__).resolve().with_name("phase2_preflight.py"),
    )
    missing_code = [path for path in code_paths if not path.is_file()]
    if missing_code:
        raise FileNotFoundError(f"Phase 2 code is incomplete: {missing_code}")

    blind_selection = _read_json(BLIND_SELECTION_PATH)
    if int(blind_selection.get("sample_count", -1)) != 200:
        raise ValueError("Unexpected Phase 1 blind selection size.")
    return {
        "schema_version": 1,
        "phase2_id": PHASE2_ID,
        "status": "frozen_before_training",
        "dataset": {
            "path": _relative(DATASET_PATH),
            "version": DATASET_VERSION,
            "label_version": LABEL_VERSION,
            "sha256": sha256_file(DATASET_PATH),
        },
        "split": {
            "path": _relative(SPLIT_MANIFEST_PATH),
            "split_id": SPLIT_ID,
            "sha256": sha256_file(SPLIT_MANIFEST_PATH),
            "sample_counts": manifest["audit"]["sample_counts"],
            "actor_counts": manifest["audit"]["actor_counts"],
        },
        "labels": {
            "phase1_manifest_path": _relative(LABEL_MANIFEST_PATH),
            "phase1_manifest_sha256": sha256_file(LABEL_MANIFEST_PATH),
            "training_policy": "embedded_y_only_no_implicit_sidecars",
            "target_interpretation": "soft_pseudo_label_intention_hypothesis",
        },
        "models": {
            "deterministic_baseline": PRIOR_VARIANT,
            "neural_variants": list(NEURAL_VARIANTS),
            "neural_seeds": list(PHASE2_SEEDS),
            "expected_neural_run_count": len(NEURAL_VARIANTS) * len(PHASE2_SEEDS),
            "prior_run_count": 1,
            "class_mask_policy": "deterministic_from_counterpart_type_for_all_models",
            "factual_only_learned_inputs": ["actor_factual", "target"],
            "full_without_analytic_excludes": [
                "analytic_factual",
                "analytic_difference",
            ],
        },
        "training": {
            "run_data_tag": RUN_DATA_TAG,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW",
            "loss": "soft_cross_entropy",
            "selection_metric": "validation_soft_nll",
            "selection_mode": "min",
            "shuffle_policy": "variant_independent_torch_generator_seeded_by_run_seed",
            "normalization_policy": (
                "train actor_factual+target statistics shared across variants and applied "
                "to counterfactual; other statistics fit only for consumed inputs"
            ),
        },
        "analysis_strata": {
            "fixed": [
                "overall",
                "counterpart_type",
                "maneuver_presence",
                "train_derived_error_bin",
                "target_dominant_C0_C7",
                "label_disagreement_low_high",
            ],
            "label_disagreement_definition": "six_rule_view_generalized_js",
            "label_disagreement_threshold_source": "train_median_only",
            "label_disagreement_train_median": threshold,
            "label_disagreement_sources": [
                {
                    "path": _relative(path),
                    "sha256": sha256_file(path),
                }
                for path in DISAGREEMENT_SOURCES
            ],
        },
        "evaluation_policy": {
            "validation": "available during training and model selection",
            "pseudo_test": "run only after all checkpoints and config are frozen",
            "blind_human_test": "run only after exactly 200 labels are frozen",
            "test_results_must_not_change_training": True,
            "pseudo_and_blind_outputs_are_separate": True,
            "actor_bootstrap_seed": 20260818,
            "actor_bootstrap_iterations": 2000,
        },
        "blind_test_seal": {
            "selection_manifest_path": _relative(BLIND_SELECTION_PATH),
            "selection_manifest_sha256": sha256_file(BLIND_SELECTION_PATH),
            "unlabeled_dataset_path": _relative(BLIND_DATASET_PATH),
            "unlabeled_dataset_sha256": sha256_file(BLIND_DATASET_PATH),
            "sample_count": 200,
            "policy": "training stage never reads human_labels.json",
        },
        "code_sha256": {
            _relative(path): sha256_file(path)
            for path in code_paths
        },
    }


def freeze_phase2_protocol() -> Path:
    """Create the protocol once; reject silent code/data/config drift later."""
    payload = _protocol_payload()
    if PHASE2_PROTOCOL_PATH.is_file():
        existing = _read_json(PHASE2_PROTOCOL_PATH)
        comparable = dict(existing)
        comparable.pop("created_at", None)
        if comparable != payload:
            raise ValueError(
                "Existing Phase 2 protocol differs from current code/data. "
                "Create a new phase2_id instead of overwriting it."
            )
        return PHASE2_PROTOCOL_PATH
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(PHASE2_PROTOCOL_PATH, payload)
    return PHASE2_PROTOCOL_PATH


def _prior_probabilities(samples: Sequence[Mapping[str, Any]]) -> tuple[dict[str, list[float]], dict[str, int]]:
    sums = {
        "satellite": np.zeros(8, dtype=np.float64),
        "ground_region": np.zeros(8, dtype=np.float64),
    }
    counts = {name: 0 for name in sums}
    for sample in samples:
        counterpart_type = (
            "satellite"
            if int(np.asarray(sample["counterpart_type"]).argmax()) == 0
            else "ground_region"
        )
        sums[counterpart_type] += np.asarray(sample["y"], dtype=np.float64)
        counts[counterpart_type] += 1
    if any(count == 0 for count in counts.values()):
        raise ValueError("Both counterpart types must appear in the training split.")
    priors = {
        name: (sums[name] / counts[name]).tolist()
        for name in sums
    }
    if any(priors["satellite"][index] != 0.0 for index in (6, 7)):
        raise ValueError("Satellite prior violates the class mask.")
    if any(priors["ground_region"][index] != 0.0 for index in (2, 3, 4)):
        raise ValueError("Ground-region prior violates the class mask.")
    return priors, counts


def fit_prior_artifact(protocol_path: Path) -> Path:
    """Fit one deterministic counterpart-type prior from train soft labels only."""
    protocol_sha256 = sha256_file(protocol_path)
    if PRIOR_ARTIFACT_PATH.is_file():
        existing = _read_json(PRIOR_ARTIFACT_PATH)
        if existing["experiment"]["phase2_protocol_sha256"] != protocol_sha256:
            raise ValueError("Prior artifact belongs to another Phase 2 protocol.")
        return PRIOR_ARTIFACT_PATH

    manifest, split_indices = load_split_indices(SPLIT_MANIFEST_PATH)
    verify_manifest_dataset(manifest, DATASET_PATH, verify_hash=True)
    train_indices = split_indices["train"]
    samples = load_training_data(
        DATASET_PATH,
        sample_indices=set(train_indices),
        allow_sidecar_labels=False,
    )
    if {int(sample["sample_index"]) for sample in samples} != set(train_indices):
        raise ValueError("Prior fitting did not load the exact frozen train split.")
    priors, counts = _prior_probabilities(samples)
    payload = {
        "schema_version": 1,
        "artifact_type": "deterministic_counterpart_type_soft_prior",
        "experiment": {
            "phase2_id": PHASE2_ID,
            "run_id": _run_stem(PRIOR_VARIANT, None),
            "phase2_protocol_path": str(protocol_path.resolve()),
            "phase2_protocol_sha256": protocol_sha256,
            "dataset_version": DATASET_VERSION,
            "label_version": LABEL_VERSION,
            "dataset_sha256": manifest["dataset"]["sha256"],
            "split_id": SPLIT_ID,
            "split_manifest_path": str(SPLIT_MANIFEST_PATH.resolve()),
            "split_manifest_sha256": sha256_file(SPLIT_MANIFEST_PATH),
            "model_variant": PRIOR_VARIANT,
            "random_seed": None,
            "model_input_keys": [],
            "model_kwargs": {},
            "logits_definition": "log_of_probability_clipped_at_1e-12_for_json",
            "data_label_policy": "embedded_y_only_no_implicit_sidecars",
        },
        "fit": {
            "split": "train",
            "sample_count": len(samples),
            "counterpart_type_counts": counts,
            "probabilities": {
                name: {
                    class_name: float(values[index])
                    for index, class_name in enumerate(CLASS_NAMES)
                }
                for name, values in priors.items()
            },
        },
    }
    _atomic_write_json(PRIOR_ARTIFACT_PATH, payload)
    return PRIOR_ARTIFACT_PATH


def _validate_existing_metrics(
    metrics_path: Path,
    protocol_sha256: str,
    split_name: str,
    model_variant: str,
    target_source: str,
    random_seed: int | None,
) -> None:
    payload = _read_json(metrics_path)
    experiment = payload.get("experiment", {})
    model = payload.get("model", {})
    evaluation = payload.get("evaluation", {})
    output_stem = metrics_path.stem
    expected_run_stem = _run_stem(model_variant, random_seed)
    if experiment.get("phase2_protocol_sha256") != protocol_sha256:
        raise ValueError(f"Metric file uses another protocol: {metrics_path}")
    if experiment.get("model_variant") != model_variant:
        raise ValueError(f"Metric file uses another model variant: {metrics_path}")
    if experiment.get("random_seed") != random_seed:
        raise ValueError(f"Metric file uses another seed: {metrics_path}")
    if experiment.get("run_id") != expected_run_stem:
        raise ValueError(f"Metric file uses another training run: {metrics_path}")
    if model.get("variant") != model_variant or model.get("run_id") != output_stem:
        raise ValueError(f"Metric model/run metadata mismatch: {metrics_path}")
    if evaluation.get("run_id") != output_stem:
        raise ValueError(f"Metric evaluation run_id mismatch: {metrics_path}")
    if evaluation.get("split") != split_name:
        raise ValueError(f"Metric file uses another split: {metrics_path}")
    if evaluation.get("target_source") != target_source:
        raise ValueError(f"Metric file uses another target source: {metrics_path}")

    manifest, split_indices = load_split_indices(SPLIT_MANIFEST_PATH)
    manifest_rows = {int(row["sample_index"]): row for row in manifest["samples"]}
    if split_name == "blind_human":
        selection = _read_json(BLIND_SELECTION_PATH)
        expected_indices = [int(row["sample_index"]) for row in selection["samples"]]
    else:
        expected_indices = list(split_indices[split_name])
    expected_index_set = set(expected_indices)

    prediction_artifact = payload.get("prediction_artifact", {})
    prediction_path = evaluation_output_paths(output_stem)[0]
    recorded_prediction_path = Path(str(prediction_artifact.get("path", "")))
    if recorded_prediction_path.resolve() != prediction_path.resolve():
        raise ValueError(f"Metric file points to another prediction path: {metrics_path}")
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Metric file has no frozen predictions: {metrics_path}")
    if sha256_file(prediction_path) != prediction_artifact.get("sha256"):
        raise ValueError(f"Prediction hash differs from metric record: {prediction_path}")

    prediction_rows = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    prediction_indices = [int(row["sample_index"]) for row in prediction_rows]
    if len(prediction_indices) != len(set(prediction_indices)):
        raise ValueError(f"Duplicate prediction sample_index: {prediction_path}")
    if set(prediction_indices) != expected_index_set:
        raise ValueError(f"Prediction sample set differs from frozen split: {prediction_path}")
    if int(evaluation.get("sample_count", -1)) != len(expected_indices):
        raise ValueError(f"Metric sample count differs from frozen split: {metrics_path}")
    for row in prediction_rows:
        sample_index = int(row["sample_index"])
        manifest_row = manifest_rows[sample_index]
        if (
            row.get("run_id") != output_stem
            or row.get("split") != split_name
            or row.get("target_source") != target_source
            or str(row.get("actor_norad_cat_id"))
            != str(manifest_row["actor_norad_cat_id"])
        ):
            raise ValueError(f"Prediction row metadata mismatch: {prediction_path}")

    source_artifact = payload.get("artifact", {})
    if split_name != "blind_human":
        expected_source_path = (
            PRIOR_ARTIFACT_PATH
            if model_variant == PRIOR_VARIANT
            else checkpoint_path_for(model_variant, int(random_seed))
        )
        source_path = Path(str(source_artifact.get("path", "")))
        if source_path.resolve() != expected_source_path.resolve():
            raise ValueError(f"Metric file points to another source artifact: {metrics_path}")
        if not source_path.is_file() or sha256_file(source_path) != source_artifact.get(
            "sha256"
        ):
            raise ValueError(f"Source artifact hash differs: {source_path}")
    else:
        expected_test_predictions = evaluation_output_paths(
            f"{expected_run_stem}__test"
        )[0]
        source_prediction = Path(str(source_artifact.get("prediction_path", "")))
        if source_prediction.resolve() != expected_test_predictions.resolve():
            raise ValueError(f"Blind metric points to another test prediction file.")
        if source_artifact.get("prediction_sha256") != sha256_file(source_prediction):
            raise ValueError(f"Blind source prediction hash differs: {source_prediction}")
        label_manifest_path = Path(
            str(source_artifact.get("blind_label_manifest_path", ""))
        )
        if label_manifest_path.resolve() != BLIND_LABEL_MANIFEST_PATH.resolve():
            raise ValueError("Blind metric points to another label manifest.")
        if source_artifact.get("blind_label_manifest_sha256") != sha256_file(
            label_manifest_path
        ):
            raise ValueError("Blind label manifest hash differs.")
        label_manifest = _read_json(label_manifest_path)
        if (
            label_manifest.get("complete") is not True
            or label_manifest.get("label_version") != BLIND_LABEL_VERSION
            or set(map(int, label_manifest.get("sample_indices", [])))
            != expected_index_set
        ):
            raise ValueError("Blind label manifest content is inconsistent.")
        frozen_record = label_manifest["files"]["frozen_labels"]
        frozen_path = label_manifest_path.parent / str(frozen_record["path"])
        if sha256_file(frozen_path) != frozen_record["sha256"]:
            raise ValueError("Frozen blind-label hash differs from its manifest.")
        if source_artifact.get("blind_labels_sha256") != frozen_record["sha256"]:
            raise ValueError("Blind metric records another frozen-label hash.")


def evaluate_prior(
    split_name: str,
    protocol_path: Path,
) -> tuple[Path, Path]:
    if split_name not in {"validation", "test"}:
        raise ValueError("Prior evaluation is limited to validation or test.")
    artifact = _read_json(fit_prior_artifact(protocol_path))
    experiment = artifact["experiment"]
    protocol_sha256 = sha256_file(protocol_path)
    if experiment["phase2_protocol_sha256"] != protocol_sha256:
        raise ValueError("Prior artifact protocol hash mismatch.")

    manifest, split_indices = load_split_indices(SPLIT_MANIFEST_PATH)
    verify_manifest_dataset(manifest, DATASET_PATH, verify_hash=True)
    selected_indices = split_indices[split_name]
    samples = load_training_data(
        DATASET_PATH,
        sample_indices=set(selected_indices),
        allow_sidecar_labels=False,
    )
    sample_indices = np.asarray([int(sample["sample_index"]) for sample in samples])
    if set(sample_indices.tolist()) != set(selected_indices):
        raise ValueError("Prior evaluation did not load the exact frozen split.")
    targets = np.stack([np.asarray(sample["y"], dtype=np.float64) for sample in samples])
    prior_by_type = {
        name: np.asarray(
            [artifact["fit"]["probabilities"][name][class_name] for class_name in CLASS_NAMES],
            dtype=np.float64,
        )
        for name in ("satellite", "ground_region")
    }
    inference_started = time.perf_counter()
    probabilities = np.stack(
        [
            prior_by_type[
                "satellite"
                if int(np.asarray(sample["counterpart_type"]).argmax()) == 0
                else "ground_region"
            ]
            for sample in samples
        ]
    )
    inference_seconds = time.perf_counter() - inference_started
    logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    output_stem = f"{_run_stem(PRIOR_VARIANT, None)}__{split_name}"
    return evaluate_probability_arrays(
        targets=targets,
        probabilities=probabilities,
        logits=logits,
        sample_indices=sample_indices,
        manifest=manifest,
        split_name=split_name,
        target_source=PSEUDO_TARGET_SOURCE,
        output_stem=output_stem,
        experiment_metadata=experiment,
        model_metadata={
            "run_id": output_stem,
            "variant": PRIOR_VARIANT,
            "parameter_count": 0,
            "device": "cpu",
            "model_input_keys": [],
            "model_kwargs": {},
        },
        artifact_metadata={
            "type": "deterministic_prior_json",
            "path": str(PRIOR_ARTIFACT_PATH.resolve()),
            "sha256": sha256_file(PRIOR_ARTIFACT_PATH),
        },
        inference_seconds=inference_seconds,
        batch_size=None,
        protocol_path=protocol_path,
        overwrite_existing=False,
    )


def _training_is_complete(
    model_variant: str,
    random_seed: int,
    protocol_sha256: str,
) -> bool:
    checkpoint_path = checkpoint_path_for(model_variant, random_seed)
    log_path = training_log_path_for(model_variant, random_seed)
    if not checkpoint_path.exists() and not log_path.exists():
        return False
    if not checkpoint_path.is_file() or not log_path.is_file():
        raise FileExistsError(
            f"Incomplete run artifacts for {model_variant} seed {random_seed}; "
            "inspect them before any explicit retry."
        )
    log = _read_json(log_path)
    experiment = log.get("experiment", {})
    if log.get("status") != "completed":
        raise ValueError(f"Training log is not complete: {log_path}")
    if experiment.get("phase2_protocol_sha256") != protocol_sha256:
        raise ValueError(f"Training log belongs to another protocol: {log_path}")
    if experiment.get("model_variant") != model_variant:
        raise ValueError(f"Training log variant mismatch: {log_path}")
    if int(experiment.get("random_seed")) != random_seed:
        raise ValueError(f"Training log seed mismatch: {log_path}")
    if log["checkpoint"]["sha256"] != sha256_file(checkpoint_path):
        raise ValueError(f"Checkpoint hash differs from completed log: {checkpoint_path}")
    return True


def _record_failure(model_variant: str, random_seed: int, exc: BaseException) -> None:
    path = (
        EXPERIMENTS_DIR
        / "logs"
        / f"{_run_stem(model_variant, random_seed)}__failures.json"
    )
    failures = _read_json(path) if path.is_file() else []
    failures.append(
        {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "phase2_id": PHASE2_ID,
            "model_variant": model_variant,
            "random_seed": random_seed,
            "phase2_protocol_path": str(PHASE2_PROTOCOL_PATH.resolve()),
            "phase2_protocol_sha256": (
                sha256_file(PHASE2_PROTOCOL_PATH)
                if PHASE2_PROTOCOL_PATH.is_file()
                else None
            ),
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    )
    _atomic_write_json(path, failures)


def _ensure_neural_evaluation(
    model_variant: str,
    random_seed: int,
    split_name: str,
    protocol_sha256: str,
) -> tuple[Path, Path]:
    if split_name not in {"validation", "test"}:
        raise ValueError("Phase 2 neural evaluation is limited to validation/test.")
    checkpoint_path = checkpoint_path_for(model_variant, random_seed)
    output_stem = f"{checkpoint_path.stem}__{split_name}"
    predictions_path, metrics_path = evaluation_output_paths(output_stem)
    if predictions_path.is_file() and metrics_path.is_file():
        _validate_existing_metrics(
            metrics_path,
            protocol_sha256,
            split_name,
            model_variant,
            PSEUDO_TARGET_SOURCE,
            random_seed,
        )
        return predictions_path, metrics_path
    if predictions_path.exists() or metrics_path.exists():
        raise FileExistsError(f"Partial evaluation output exists for {output_stem}.")
    return evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        split_name=split_name,
        allow_frozen_test=split_name == "test",
        overwrite_existing=False,
    )


def train_validation_phase() -> None:
    """Train all neural runs and write validation predictions; never touch test."""
    protocol_path = freeze_phase2_protocol()
    protocol_sha256 = sha256_file(protocol_path)
    fit_prior_artifact(protocol_path)
    prior_stem = f"{_run_stem(PRIOR_VARIANT, None)}__validation"
    prior_predictions, prior_metrics = evaluation_output_paths(prior_stem)
    if prior_predictions.is_file() and prior_metrics.is_file():
        _validate_existing_metrics(
            prior_metrics,
            protocol_sha256,
            "validation",
            PRIOR_VARIANT,
            PSEUDO_TARGET_SOURCE,
            None,
        )
    elif prior_predictions.exists() or prior_metrics.exists():
        raise FileExistsError("Partial prior validation output exists.")
    else:
        evaluate_prior("validation", protocol_path)

    for model_variant in NEURAL_VARIANTS:
        for random_seed in PHASE2_SEEDS:
            if not _training_is_complete(model_variant, random_seed, protocol_sha256):
                try:
                    train_model(
                        model_variant=model_variant,
                        random_seed=random_seed,
                        phase2_protocol_path=protocol_path,
                        overwrite_existing=False,
                    )
                except BaseException as exc:
                    _record_failure(model_variant, random_seed, exc)
                    raise
            _ensure_neural_evaluation(
                model_variant,
                random_seed,
                "validation",
                protocol_sha256,
            )


def _assert_all_training_complete(protocol_path: Path) -> None:
    protocol_sha256 = sha256_file(protocol_path)
    if not PRIOR_ARTIFACT_PATH.is_file():
        raise FileNotFoundError("The deterministic prior has not been fitted.")
    prior_validation = evaluation_output_paths(
        f"{_run_stem(PRIOR_VARIANT, None)}__validation"
    )
    if not all(path.is_file() for path in prior_validation):
        raise FileNotFoundError("The deterministic prior validation output is incomplete.")
    _validate_existing_metrics(
        prior_validation[1],
        protocol_sha256,
        "validation",
        PRIOR_VARIANT,
        PSEUDO_TARGET_SOURCE,
        None,
    )
    for model_variant in NEURAL_VARIANTS:
        for random_seed in PHASE2_SEEDS:
            if not _training_is_complete(model_variant, random_seed, protocol_sha256):
                raise FileNotFoundError(
                    f"Missing completed run: {model_variant} seed {random_seed}"
                )
            validation_predictions, validation_metrics = evaluation_output_paths(
                f"{_run_stem(model_variant, random_seed)}__validation"
            )
            if not validation_predictions.is_file() or not validation_metrics.is_file():
                raise FileNotFoundError(
                    f"Incomplete validation evaluation for {model_variant} seed {random_seed}."
                )
            _validate_existing_metrics(
                validation_metrics,
                protocol_sha256,
                "validation",
                model_variant,
                PSEUDO_TARGET_SOURCE,
                random_seed,
            )


def pseudo_test_phase() -> None:
    """Run pseudo-test only after every Phase 2 run is frozen and complete."""
    protocol_path = freeze_phase2_protocol()
    protocol_sha256 = sha256_file(protocol_path)
    _assert_all_training_complete(protocol_path)

    prior_stem = f"{_run_stem(PRIOR_VARIANT, None)}__test"
    prior_predictions, prior_metrics = evaluation_output_paths(prior_stem)
    if prior_predictions.is_file() and prior_metrics.is_file():
        _validate_existing_metrics(
            prior_metrics,
            protocol_sha256,
            "test",
            PRIOR_VARIANT,
            PSEUDO_TARGET_SOURCE,
            None,
        )
    elif prior_predictions.exists() or prior_metrics.exists():
        raise FileExistsError("Partial prior pseudo-test output exists.")
    else:
        evaluate_prior("test", protocol_path)

    for model_variant in NEURAL_VARIANTS:
        for random_seed in PHASE2_SEEDS:
            _ensure_neural_evaluation(
                model_variant,
                random_seed,
                "test",
                protocol_sha256,
            )


def _validate_human_y(
    y: object,
    counterpart_type: str,
    sample_index: int,
) -> list[float]:
    expected = {f"c{index}" for index in range(8)}
    if not isinstance(y, dict) or set(y) != expected:
        raise ValueError(f"Blind label {sample_index} must contain exactly c0-c7.")
    values = [float(y[f"c{index}"]) for index in range(8)]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"Blind label {sample_index} contains an invalid probability.")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Blind label {sample_index} does not sum to one.")
    invalid = (6, 7) if counterpart_type == "satellite" else (2, 3, 4)
    if any(values[index] != 0.0 for index in invalid):
        raise ValueError(f"Blind label {sample_index} violates its counterpart class mask.")
    return values


def freeze_blind_human_labels() -> Path:
    """Freeze labels only when all 200 globally indexed rows are complete and valid."""
    protocol_path = freeze_phase2_protocol()
    _assert_all_training_complete(protocol_path)
    _assert_all_pseudo_test_outputs(protocol_path)
    selection = _read_json(BLIND_SELECTION_PATH)
    source_live_sha256 = sha256_file(BLIND_LIVE_LABELS_PATH)
    rows = _read_json(BLIND_LIVE_LABELS_PATH)
    if sha256_file(BLIND_LIVE_LABELS_PATH) != source_live_sha256:
        raise RuntimeError("Blind human_labels.json changed while it was being read.")
    if not isinstance(rows, list):
        raise ValueError("Blind human_labels.json must be a JSON list.")
    selected = {
        int(row["sample_index"]): row
        for row in selection["samples"]
    }
    if len(selection["samples"]) != 200 or len(selected) != 200:
        raise ValueError("Blind selection must contain 200 unique global sample indices.")
    labels: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or "sample_index" not in row:
            raise ValueError("Every blind label row requires sample_index.")
        sample_index = int(row["sample_index"])
        if sample_index in labels:
            raise ValueError(f"Duplicate blind label sample_index: {sample_index}")
        labels[sample_index] = row
    if set(labels) != set(selected) or len(labels) != 200:
        missing = sorted(set(selected) - set(labels))
        extra = sorted(set(labels) - set(selected))
        raise ValueError(
            "Blind annotation is not complete: "
            f"count={len(labels)}, missing={missing[:10]}, extra={extra[:10]}."
        )

    manifest, _ = load_split_indices(SPLIT_MANIFEST_PATH)
    split_rows = {int(row["sample_index"]): row for row in manifest["samples"]}
    frozen_rows: list[dict[str, Any]] = []
    for sample_index in sorted(labels):
        selection_row = selected[sample_index]
        split_row = split_rows[sample_index]
        if split_row["split"] != "test" or split_row["is_calibration_actor"]:
            raise ValueError(f"Blind label {sample_index} is not an independent test row.")
        counterpart_type = str(selection_row["counterpart_type"])
        if counterpart_type != str(split_row["counterpart_type"]):
            raise ValueError(f"Blind selection metadata drift at sample {sample_index}.")
        values = _validate_human_y(labels[sample_index].get("y"), counterpart_type, sample_index)
        annotation_source = labels[sample_index].get("annotation_source")
        review_status = labels[sample_index].get("review_status")
        if annotation_source != "human":
            raise ValueError(
                f"Blind label {sample_index} has unexpected annotation_source."
            )
        if review_status != "human_labeled":
            raise ValueError(f"Blind label {sample_index} is not marked complete.")
        frozen = dict(labels[sample_index])
        frozen["sample_index"] = sample_index
        frozen["y"] = {f"c{index}": values[index] for index in range(8)}
        frozen_rows.append(frozen)

    if sha256_file(BLIND_LIVE_LABELS_PATH) != source_live_sha256:
        raise RuntimeError("Blind human_labels.json changed during freeze validation.")

    if BLIND_FROZEN_LABELS_PATH.is_file():
        if _read_json(BLIND_FROZEN_LABELS_PATH) != frozen_rows:
            raise ValueError("Frozen blind labels already exist with different content.")
    else:
        _atomic_write_json(BLIND_FROZEN_LABELS_PATH, frozen_rows)

    label_manifest = {
        "schema_version": 1,
        "blind_test_id": selection["blind_test_id"],
        "label_version": BLIND_LABEL_VERSION,
        "complete": True,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": 200,
        "sample_indices": sorted(labels),
        "policy": "evaluation_only_never_train_calibrate_or_select",
        "files": {
            "source_live_labels": {
                "path": BLIND_LIVE_LABELS_PATH.name,
                "sha256_at_freeze": source_live_sha256,
            },
            "frozen_labels": {
                "path": BLIND_FROZEN_LABELS_PATH.name,
                "sha256": sha256_file(BLIND_FROZEN_LABELS_PATH),
            },
            "selection_manifest": {
                "path": BLIND_SELECTION_PATH.name,
                "sha256": sha256_file(BLIND_SELECTION_PATH),
            },
        },
    }
    if BLIND_LABEL_MANIFEST_PATH.is_file():
        existing_manifest = _read_json(BLIND_LABEL_MANIFEST_PATH)
        comparable_existing = dict(existing_manifest)
        comparable_existing.pop("frozen_at", None)
        comparable_new = dict(label_manifest)
        comparable_new.pop("frozen_at", None)
        if comparable_existing != comparable_new:
            raise ValueError("Blind label manifest already exists with different content.")
    else:
        _atomic_write_json(BLIND_LABEL_MANIFEST_PATH, label_manifest)
    return BLIND_LABEL_MANIFEST_PATH


def _prediction_paths_for_all_models(split_name: str) -> list[tuple[str, int | None, Path, Path]]:
    runs = [
        (
            PRIOR_VARIANT,
            None,
            *evaluation_output_paths(f"{_run_stem(PRIOR_VARIANT, None)}__{split_name}"),
        )
    ]
    runs.extend(
        (
            model_variant,
            random_seed,
            *evaluation_output_paths(f"{_run_stem(model_variant, random_seed)}__{split_name}"),
        )
        for model_variant in NEURAL_VARIANTS
        for random_seed in PHASE2_SEEDS
    )
    return runs


def _assert_all_pseudo_test_outputs(protocol_path: Path) -> None:
    """Verify every frozen pseudo-test pair before any human label is opened."""
    protocol_sha256 = sha256_file(protocol_path)
    for model_variant, random_seed, predictions_path, metrics_path in (
        _prediction_paths_for_all_models("test")
    ):
        if not predictions_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(
                f"Pseudo-test output is incomplete before blind evaluation: "
                f"{predictions_path}"
            )
        _validate_existing_metrics(
            metrics_path,
            protocol_sha256,
            "test",
            model_variant,
            PSEUDO_TARGET_SOURCE,
            random_seed,
        )


def blind_human_test_phase() -> None:
    """Score already-frozen test predictions against frozen blind human labels."""
    protocol_path = freeze_phase2_protocol()
    _assert_all_training_complete(protocol_path)
    _assert_all_pseudo_test_outputs(protocol_path)
    label_manifest_path = freeze_blind_human_labels()
    label_manifest = _read_json(label_manifest_path)
    frozen_labels = _read_json(BLIND_FROZEN_LABELS_PATH)
    selection = _read_json(BLIND_SELECTION_PATH)
    selected_order = [int(row["sample_index"]) for row in selection["samples"]]
    labels_by_index = {int(row["sample_index"]): row for row in frozen_labels}
    targets = np.asarray(
        [
            [labels_by_index[index]["y"][f"c{class_index}"] for class_index in range(8)]
            for index in selected_order
        ],
        dtype=np.float64,
    )
    manifest, _ = load_split_indices(SPLIT_MANIFEST_PATH)

    for model_variant, random_seed, predictions_path, metrics_path in _prediction_paths_for_all_models("test"):
        if not predictions_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(
                f"Pseudo-test predictions must be frozen before blind evaluation: {predictions_path}"
            )
        _validate_existing_metrics(
            metrics_path,
            sha256_file(protocol_path),
            "test",
            model_variant,
            PSEUDO_TARGET_SOURCE,
            random_seed,
        )
        prediction_rows = [
            json.loads(line)
            for line in predictions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_index = {int(row["sample_index"]): row for row in prediction_rows}
        if len(by_index) != len(prediction_rows):
            raise ValueError(f"Duplicate sample_index in test predictions: {predictions_path}")
        missing = sorted(set(selected_order) - set(by_index))
        if missing:
            raise ValueError(f"Test predictions miss blind samples: {missing[:10]}")
        probabilities = np.asarray(
            [
                [by_index[index]["probabilities"][class_name] for class_name in CLASS_NAMES]
                for index in selected_order
            ],
            dtype=np.float64,
        )
        logits = np.asarray(
            [
                [by_index[index]["logits"][class_name] for class_name in CLASS_NAMES]
                for index in selected_order
            ],
            dtype=np.float64,
        )
        source_metrics = _read_json(metrics_path)
        output_stem = f"{_run_stem(model_variant, random_seed)}__blind_human"
        blind_predictions_path, blind_metrics_path = evaluation_output_paths(output_stem)
        if blind_predictions_path.is_file() and blind_metrics_path.is_file():
            _validate_existing_metrics(
                blind_metrics_path,
                sha256_file(protocol_path),
                "blind_human",
                model_variant,
                BLIND_LABEL_VERSION,
                random_seed,
            )
            continue
        if blind_predictions_path.exists() or blind_metrics_path.exists():
            raise FileExistsError(f"Partial blind-human output exists for {output_stem}.")
        experiment = dict(source_metrics["experiment"])
        experiment.update(
            {
                "blind_label_version": BLIND_LABEL_VERSION,
                "blind_label_manifest_path": str(label_manifest_path.resolve()),
                "blind_label_manifest_sha256": sha256_file(label_manifest_path),
            }
        )
        blind_model_metadata = dict(source_metrics["model"])
        blind_model_metadata["run_id"] = output_stem
        evaluate_probability_arrays(
            targets=targets,
            probabilities=probabilities,
            logits=logits,
            sample_indices=selected_order,
            manifest=manifest,
            split_name="blind_human",
            target_source=BLIND_LABEL_VERSION,
            output_stem=output_stem,
            experiment_metadata=experiment,
            model_metadata=blind_model_metadata,
            artifact_metadata={
                "type": "frozen_test_predictions_plus_blind_labels",
                "prediction_path": str(predictions_path.resolve()),
                "prediction_sha256": sha256_file(predictions_path),
                "blind_label_manifest_path": str(label_manifest_path.resolve()),
                "blind_label_manifest_sha256": sha256_file(label_manifest_path),
                "blind_labels_sha256": label_manifest["files"]["frozen_labels"]["sha256"],
            },
            inference_seconds=None,
            batch_size=None,
            protocol_path=protocol_path,
            overwrite_existing=False,
        )


def final_evaluation_phase() -> None:
    """Freeze predictions first, then read/freeze labels and score blind-human."""
    protocol_path = freeze_phase2_protocol()
    _assert_all_training_complete(protocol_path)
    pseudo_test_phase()
    freeze_blind_human_labels()
    blind_human_test_phase()


def main() -> None:
    # IDE 中直接运行本文件：只训练并输出 validation，不读取任何 test 标签。
    train_validation_phase()


if __name__ == "__main__":
    main()
