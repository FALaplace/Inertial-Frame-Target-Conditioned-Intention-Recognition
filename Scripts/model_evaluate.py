"""Evaluate one formal checkpoint without refitting preprocessing statistics.

The safe default is validation.  Frozen test evaluation is orchestrated by
``phase2_pseudo_evaluate.py`` only after every run and the Phase 2 protocol
have been frozen; ``phase2_final_evaluate.py`` later reuses those predictions.
"""

from __future__ import annotations

import hashlib
import json
import time
from itertools import islice
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import Tensor
from torch.utils.data import DataLoader

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
from model import SUPPORTED_MODEL_VARIANTS, build_intention_model
from model_train import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    MODEL_INPUT_KEYS,
    NUM_WORKERS,
    IntentionDataset,
    NormalizationStats,
    runtime_environment,
    set_random_seed,
)


IDEA_DIR = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = IDEA_DIR / "experiment_data" / "predictions"
METRICS_DIR = IDEA_DIR / "experiment_data" / "metrics"
CLASS_NAMES = tuple(f"C{index}" for index in range(8))
PSEUDO_TARGET_SOURCE = "pseudo_v5_clean"


def _soft_distribution_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    if targets.shape != probabilities.shape or targets.ndim != 2 or targets.shape[1] != 8:
        raise ValueError("targets and probabilities must both have shape [N, 8].")
    if not np.isfinite(targets).all() or not np.isfinite(probabilities).all():
        raise ValueError("Metric inputs must be finite.")
    if np.any(targets < -1e-12) or np.any(targets > 1.0 + 1e-12):
        raise ValueError("Target distributions must stay within [0, 1].")
    if np.any(probabilities < -1e-12) or np.any(probabilities > 1.0 + 1e-12):
        raise ValueError("Predicted distributions must stay within [0, 1].")
    if not np.allclose(targets.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Target distributions must sum to one.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Predicted distributions must sum to one.")

    epsilon = 1e-12
    clipped = np.clip(probabilities, epsilon, 1.0)
    soft_nll_per_sample = -(targets * np.log(clipped)).sum(axis=1)
    brier_per_sample = np.square(probabilities - targets).sum(axis=1)
    l1_per_sample = np.abs(probabilities - targets).sum(axis=1)

    midpoint = 0.5 * (targets + probabilities)
    target_term = np.zeros_like(targets)
    target_mask = targets > 0.0
    target_term[target_mask] = targets[target_mask] * np.log(
        targets[target_mask] / midpoint[target_mask]
    )
    prediction_term = np.zeros_like(probabilities)
    prediction_mask = probabilities > 0.0
    prediction_term[prediction_mask] = probabilities[prediction_mask] * np.log(
        probabilities[prediction_mask] / midpoint[prediction_mask]
    )
    js_per_sample = 0.5 * (target_term + prediction_term).sum(axis=1)

    target_classes = targets.argmax(axis=1)
    predicted_classes = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        target_classes,
        predicted_classes,
        labels=np.arange(8),
        zero_division=0,
    )
    observed = support > 0
    balanced_accuracy = float(recall[observed].mean()) if observed.any() else 0.0

    confidence = probabilities.max(axis=1)
    correct = predicted_classes == target_classes
    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, 11)
    for bin_index in range(10):
        lower, upper = bin_edges[bin_index], bin_edges[bin_index + 1]
        in_bin = (confidence >= lower) & (
            confidence <= upper if bin_index == 9 else confidence < upper
        )
        if in_bin.any():
            ece += float(in_bin.mean()) * abs(
                float(correct[in_bin].mean()) - float(confidence[in_bin].mean())
            )

    return {
        "sample_count": int(len(targets)),
        "soft_nll": float(soft_nll_per_sample.mean()),
        "brier_score_sum_over_classes": float(brier_per_sample.mean()),
        "js_divergence": float(js_per_sample.mean()),
        "l1_distance": float(l1_per_sample.mean()),
        "dominant_label_agreement": float(correct.mean()),
        "balanced_accuracy": balanced_accuracy,
        "macro_f1_all_8_classes": float(f1.mean()),
        "dominant_hypothesis_ece_10_bins": ece,
        "confusion_matrix": confusion_matrix(
            target_classes, predicted_classes, labels=np.arange(8)
        ).tolist(),
        "per_class": {
            class_name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, class_name in enumerate(CLASS_NAMES)
        },
        "per_sample": {
            "soft_nll": soft_nll_per_sample,
            "brier": brier_per_sample,
            "js": js_per_sample,
            "l1": l1_per_sample,
        },
    }


def compute_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """Public metric helper used by evaluation and the Phase 0 smoke check."""
    metrics = _soft_distribution_metrics(targets, probabilities)
    metrics.pop("per_sample")
    return metrics


def _move_model_batch(
    batch: Mapping[str, Tensor],
    device: torch.device,
    model_input_keys: Sequence[str] = MODEL_INPUT_KEYS,
) -> tuple[dict[str, Tensor], Tensor]:
    model_batch = {
        key: batch[key].to(device, non_blocking=True)
        for key in model_input_keys
    }
    return model_batch, batch["y"].to(device, non_blocking=True)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        # Write the exact bytes that were hashed. Text-mode writes translate LF
        # to CRLF on Windows and would invalidate the recorded artifact SHA-256.
        temporary.write_bytes(text.encode("utf-8"))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def evaluation_output_paths(output_stem: str) -> tuple[Path, Path]:
    """Return the versioned prediction and metric paths for one evaluation."""
    return (
        PREDICTIONS_DIR / f"{output_stem}.jsonl",
        METRICS_DIR / f"{output_stem}.json",
    )


def _load_disagreement_values(
    protocol_path: Path | None,
    sample_indices: Sequence[int],
) -> tuple[np.ndarray, float] | None:
    """Load label disagreement only for reporting strata, never as model input."""
    if protocol_path is None:
        return None
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    analysis = protocol.get("analysis_strata")
    if not isinstance(analysis, dict):
        raise ValueError("Phase 2 protocol is missing analysis_strata.")
    threshold = float(analysis["label_disagreement_train_median"])
    values_by_index: dict[int, float] = {}
    for source in analysis["label_disagreement_sources"]:
        source_path = Path(str(source["path"]))
        if not source_path.is_absolute():
            source_path = IDEA_DIR / source_path
        if sha256_file(source_path) != str(source["sha256"]):
            raise ValueError(f"Label-disagreement source hash changed: {source_path}")
        rows = json.loads(source_path.read_text(encoding="utf-8"))
        for row in rows:
            values_by_index[int(row["sample_index"])] = float(
                row["optimization"]["generalized_js"]
            )
    missing = sorted(set(map(int, sample_indices)) - set(values_by_index))
    if missing:
        raise ValueError(f"Missing disagreement values for samples: {missing[:10]}")
    return (
        np.asarray([values_by_index[int(index)] for index in sample_indices]),
        threshold,
    )


def _stratum_masks(
    targets: np.ndarray,
    sample_indices: np.ndarray,
    manifest_rows: Mapping[int, Mapping[str, Any]],
    protocol_path: Path | None,
) -> dict[str, np.ndarray]:
    rows = [manifest_rows[int(index)] for index in sample_indices]
    counterpart_types = np.asarray([str(row["counterpart_type"]) for row in rows])
    maneuver_presence = np.asarray([str(row["maneuver_presence"]) for row in rows])
    error_bins = np.asarray([str(row["error_bin"]) for row in rows])
    target_classes = targets.argmax(axis=1)
    masks = {
        "overall": np.ones(len(targets), dtype=bool),
        "satellite": counterpart_types == "satellite",
        "ground_region": counterpart_types == "ground_region",
        "maneuver": maneuver_presence == "maneuver",
        "non_maneuver": maneuver_presence == "non_maneuver",
    }
    for error_bin in ("low", "medium", "high"):
        masks[f"error_{error_bin}"] = error_bins == error_bin
    for class_index, class_name in enumerate(CLASS_NAMES):
        masks[f"target_{class_name}"] = target_classes == class_index

    disagreement = _load_disagreement_values(protocol_path, sample_indices.tolist())
    if disagreement is not None:
        values, threshold = disagreement
        masks["label_disagreement_low"] = values <= threshold
        masks["label_disagreement_high"] = values > threshold
    return masks


def evaluate_probability_arrays(
    *,
    targets: np.ndarray,
    probabilities: np.ndarray,
    sample_indices: Sequence[int] | np.ndarray,
    manifest: Mapping[str, Any],
    split_name: str,
    target_source: str,
    output_stem: str,
    experiment_metadata: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
    logits: np.ndarray | None = None,
    artifact_metadata: Mapping[str, Any] | None = None,
    inference_seconds: float | None = None,
    batch_size: int | None = None,
    latency_benchmark: Mapping[str, Any] | None = None,
    protocol_path: Path | None = None,
    overwrite_existing: bool = False,
) -> tuple[Path, Path]:
    """Write uniform predictions and metrics for neural and prior models."""
    targets = np.asarray(targets, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    sample_indices_array = np.asarray(sample_indices, dtype=np.int64)
    if len(sample_indices_array) != len(targets):
        raise ValueError("sample_indices and targets must have the same length.")
    if len(set(sample_indices_array.tolist())) != len(sample_indices_array):
        raise ValueError("Evaluation sample_indices must be unique.")
    if logits is not None:
        logits = np.asarray(logits, dtype=np.float64)
        if logits.shape != probabilities.shape or not np.isfinite(logits).all():
            raise ValueError("logits must be finite and have shape [N, 8].")

    metric_bundle = _soft_distribution_metrics(targets, probabilities)
    per_sample_metrics = metric_bundle.pop("per_sample")
    manifest_rows = {
        int(row["sample_index"]): row
        for row in manifest["samples"]
    }
    missing = sorted(set(sample_indices_array.tolist()) - set(manifest_rows))
    if missing:
        raise ValueError(f"Evaluation samples are absent from the split manifest: {missing[:10]}")

    counterpart_types = np.asarray(
        [manifest_rows[int(index)]["counterpart_type"] for index in sample_indices_array]
    )
    satellite = counterpart_types == "satellite"
    ground_region = counterpart_types == "ground_region"
    for name, values in (("targets", targets), ("probabilities", probabilities)):
        if satellite.any() and not np.allclose(
            values[satellite][:, (6, 7)], 0.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"{name} violate the satellite class mask.")
        if ground_region.any() and not np.allclose(
            values[ground_region][:, (2, 3, 4)], 0.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"{name} violate the ground-region class mask.")

    predictions: list[dict[str, Any]] = []
    for position, sample_index in enumerate(sample_indices_array.tolist()):
        manifest_row = manifest_rows[sample_index]
        prediction = {
            "run_id": output_stem,
            "sample_index": sample_index,
            "actor_norad_cat_id": manifest_row["actor_norad_cat_id"],
            "split": split_name,
            "target_source": target_source,
            "counterpart_type": manifest_row["counterpart_type"],
            "has_maneuver": manifest_row["maneuver_presence"] == "maneuver",
            "error_bin": manifest_row["error_bin"],
            "target": {
                class_name: float(targets[position, class_index])
                for class_index, class_name in enumerate(CLASS_NAMES)
            },
            "probabilities": {
                class_name: float(probabilities[position, class_index])
                for class_index, class_name in enumerate(CLASS_NAMES)
            },
            "target_dominant": CLASS_NAMES[int(targets[position].argmax())],
            "predicted_dominant": CLASS_NAMES[int(probabilities[position].argmax())],
            "soft_nll": float(per_sample_metrics["soft_nll"][position]),
            "brier_score": float(per_sample_metrics["brier"][position]),
            "js_divergence": float(per_sample_metrics["js"][position]),
            "l1_distance": float(per_sample_metrics["l1"][position]),
        }
        if logits is not None:
            prediction["logits"] = {
                class_name: float(logits[position, class_index])
                for class_index, class_name in enumerate(CLASS_NAMES)
            }
        predictions.append(prediction)

    masks = _stratum_masks(
        targets,
        sample_indices_array,
        manifest_rows,
        protocol_path,
    )
    stratified_metrics = {
        name: compute_metrics(targets[mask], probabilities[mask])
        for name, mask in masks.items()
        if mask.any()
    }
    predictions_path, metrics_path = evaluation_output_paths(output_stem)
    if not overwrite_existing and (predictions_path.exists() or metrics_path.exists()):
        raise FileExistsError(
            "Evaluation output already exists. Use a new version or explicitly "
            "set overwrite_existing=True."
        )

    prediction_text = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for row in predictions
    )
    mean_inference_ms = (
        None
        if inference_seconds is None
        else 1000.0 * inference_seconds / max(len(predictions), 1)
    )
    metrics_payload = {
        "schema_version": 2,
        "prediction_artifact": {
            "path": str(predictions_path.resolve()),
            "sha256": hashlib.sha256(prediction_text.encode("utf-8")).hexdigest(),
        },
        "artifact": dict(artifact_metadata or {}),
        "experiment": dict(experiment_metadata),
        "model": dict(model_metadata),
        "evaluation": {
            "run_id": output_stem,
            "split": split_name,
            "target_source": target_source,
            "sample_count": len(predictions),
            "batch_size": batch_size,
            "inference_seconds": inference_seconds,
            "mean_inference_ms_per_sample": mean_inference_ms,
            "latency_benchmark": dict(latency_benchmark or {}),
        },
        "metrics": stratified_metrics,
    }
    _atomic_write_text(predictions_path, prediction_text)
    _atomic_write_text(
        metrics_path,
        json.dumps(metrics_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    return predictions_path, metrics_path


def benchmark_inference_latency(
    model: torch.nn.Module,
    dataset: IntentionDataset,
    device: torch.device,
    batch_sizes: Sequence[int] = (1, 32),
    warmup_batches: int = 3,
    measured_batches: int = 20,
) -> dict[str, Any]:
    """Measure warmed inference latency for the fixed batch sizes in E4/E10."""
    results: dict[str, Any] = {}
    with torch.inference_mode():
        for requested_batch_size in batch_sizes:
            loader = DataLoader(
                dataset,
                batch_size=requested_batch_size,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=device.type == "cuda",
            )
            batches = list(islice(loader, warmup_batches + measured_batches))
            if len(batches) <= warmup_batches:
                raise ValueError("Not enough evaluation samples for latency benchmarking.")
            for batch in batches[:warmup_batches]:
                model_batch, _ = _move_model_batch(
                    batch,
                    device,
                    model_input_keys=model.model_input_keys,
                )
                model(model_batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            sample_count = 0
            used_batches = batches[warmup_batches:]
            for batch in used_batches:
                model_batch, targets = _move_model_batch(
                    batch,
                    device,
                    model_input_keys=model.model_input_keys,
                )
                model(model_batch)
                sample_count += int(targets.shape[0])
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            results[f"batch_{requested_batch_size}"] = {
                "requested_batch_size": requested_batch_size,
                "warmup_batches": warmup_batches,
                "measured_batches": len(used_batches),
                "measured_samples": sample_count,
                "seconds": elapsed,
                "mean_ms_per_batch": 1000.0 * elapsed / len(used_batches),
                "mean_ms_per_sample": 1000.0 * elapsed / sample_count,
            }
    return results


def evaluate_checkpoint(
    checkpoint_path: Path = CHECKPOINT_PATH,
    dataset_path: Path = DATASET_PATH,
    split_manifest_path: Path = SPLIT_MANIFEST_PATH,
    split_name: str = "validation",
    batch_size: int = BATCH_SIZE,
    allow_frozen_test: bool = False,
    overwrite_existing: bool = False,
) -> tuple[Path, Path]:
    """Evaluate one frozen split and return prediction/metric output paths."""
    if split_name not in {"train", "validation", "test"}:
        raise ValueError("split_name must be train, validation, or test.")
    if split_name == "test" and not allow_frozen_test:
        raise PermissionError(
            "Frozen test evaluation is gated. Run phase2_pseudo_evaluate.py "
            "instead of calling evaluate_checkpoint(test) directly."
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Formal checkpoint not found: {checkpoint_path}. Run model_train.py first."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    experiment = checkpoint.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("Legacy checkpoint has no frozen Phase 0 experiment metadata.")
    expected = {
        "dataset_version": DATASET_VERSION,
        "label_version": LABEL_VERSION,
        "split_id": SPLIT_ID,
    }
    for key, value in expected.items():
        if experiment.get(key) != value:
            raise ValueError(f"Checkpoint {key} does not match the current formal experiment.")

    manifest, split_indices = load_split_indices(split_manifest_path)
    verify_manifest_dataset(manifest, dataset_path, verify_hash=True)
    if sha256_file(split_manifest_path) != experiment.get("split_manifest_sha256"):
        raise ValueError("Split manifest hash differs from the checkpoint.")
    if manifest["dataset"]["sha256"] != experiment.get("dataset_sha256"):
        raise ValueError("Dataset hash differs from the checkpoint.")
    model_variant = str(experiment.get("model_variant"))
    if model_variant not in SUPPORTED_MODEL_VARIANTS:
        raise ValueError(f"Unsupported checkpoint model_variant: {model_variant}")
    random_seed = experiment.get("random_seed")
    if random_seed is None:
        raise ValueError("Neural checkpoint is missing random_seed.")
    set_random_seed(int(random_seed))
    protocol_path: Path | None = None
    protocol_text = experiment.get("phase2_protocol_path")
    protocol_sha256 = experiment.get("phase2_protocol_sha256")
    if protocol_text is not None or protocol_sha256 is not None:
        if not isinstance(protocol_text, str) or not isinstance(protocol_sha256, str):
            raise ValueError("Checkpoint has incomplete Phase 2 protocol metadata.")
        protocol_path = Path(protocol_text)
        if sha256_file(protocol_path) != protocol_sha256:
            raise ValueError("Phase 2 protocol hash differs from the checkpoint.")

    selected_sample_indices = split_indices[split_name]
    samples = load_training_data(
        dataset_path,
        sample_indices=set(selected_sample_indices),
        allow_sidecar_labels=False,
    )
    if {int(sample["sample_index"]) for sample in samples} != set(selected_sample_indices):
        raise ValueError("Loaded evaluation samples do not match the split manifest.")

    normalization = checkpoint.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("Checkpoint is missing normalization statistics.")
    stats = NormalizationStats.from_checkpoint_dict(normalization)
    dataset = IntentionDataset(samples, list(range(len(samples))), stats)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    model_kwargs = experiment.get("model_kwargs", {})
    if not isinstance(model_kwargs, dict):
        raise ValueError("Checkpoint model_kwargs must be a dictionary.")
    model = build_intention_model(model_variant, **model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    probability_batches: list[np.ndarray] = []
    logit_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    sample_index_batches: list[np.ndarray] = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_start = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            model_batch, targets = _move_model_batch(
                batch,
                device,
                model_input_keys=model.model_input_keys,
            )
            outputs = model(model_batch)
            probability_batches.append(outputs["probabilities"].cpu().numpy())
            logit_batches.append(outputs["logits"].cpu().numpy())
            target_batches.append(targets.cpu().numpy())
            sample_index_batches.append(batch["sample_index"].numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_start

    probabilities = np.concatenate(probability_batches, axis=0)
    logits = np.concatenate(logit_batches, axis=0)
    targets = np.concatenate(target_batches, axis=0)
    sample_indices_array = np.concatenate(sample_index_batches).astype(np.int64)
    latency_benchmark = benchmark_inference_latency(model, dataset, device)

    output_stem = f"{checkpoint_path.stem}__{split_name}"
    predictions_path, metrics_path = evaluate_probability_arrays(
        targets=targets,
        probabilities=probabilities,
        logits=logits,
        sample_indices=sample_indices_array,
        manifest=manifest,
        split_name=split_name,
        target_source=PSEUDO_TARGET_SOURCE,
        output_stem=output_stem,
        experiment_metadata=experiment,
        model_metadata={
            "run_id": output_stem,
            "variant": model_variant,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "device": str(device),
            "model_input_keys": list(model.model_input_keys),
            "model_kwargs": dict(model.model_kwargs),
            "logits_definition": "raw_pre_mask_network_logits",
            "evaluation_runtime_environment": runtime_environment(device),
        },
        artifact_metadata={
            "type": "pytorch_checkpoint",
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256_file(checkpoint_path),
            "epoch": int(checkpoint["epoch"]),
        },
        inference_seconds=inference_seconds,
        batch_size=batch_size,
        latency_benchmark=latency_benchmark,
        protocol_path=protocol_path,
        overwrite_existing=overwrite_existing,
    )
    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")
    return predictions_path, metrics_path


if __name__ == "__main__":
    evaluate_checkpoint()
