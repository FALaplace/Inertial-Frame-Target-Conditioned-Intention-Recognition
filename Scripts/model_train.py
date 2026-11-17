"""Train the formal C0-C7 soft-label intention recognition model.

The data flow in this file is intentionally explicit:

    x_samples_frozen_v5_clean.json + frozen actor split manifest
        -> data_processing.load_training_data()
        -> actor-ID train/validation indices from the manifest
        -> fit normalization statistics on the training split only
        -> PyTorch Dataset / DataLoader (batch_size=32)
        -> IntentionRecognitionModel
        -> soft-label cross entropy
        -> validation loss and best checkpoint

The test split is never used in this file.  ``model_evaluate.py`` loads the
saved checkpoint and training-only normalization statistics for evaluation.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import sklearn
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

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
from model import (
    SUPPORTED_MODEL_VARIANTS,
    build_intention_model,
    soft_label_cross_entropy,
)


IDEA_DIR = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = IDEA_DIR / "experiment_data"
MODEL_VARIANT = "full"
RUN_DATA_TAG = "sgp4_1min_10400_labels_v5_clean_20260817__phase2_core_v2"

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VALIDATION_FRACTION = 0.2
RANDOM_SEED = 42
NUM_WORKERS = 0  # Windows 下数据已在内存中，单进程读取最稳定。


def checkpoint_path_for(model_variant: str, random_seed: int) -> Path:
    """Build a non-legacy checkpoint name from frozen experiment identifiers."""
    stem = f"{RUN_DATA_TAG}__{SPLIT_ID}__{model_variant}__seed{random_seed}"
    return EXPERIMENTS_DIR / "checkpoints" / f"{stem}.pt"


def training_log_path_for(model_variant: str, random_seed: int) -> Path:
    """Return the versioned, non-checkpoint training record path."""
    stem = f"{RUN_DATA_TAG}__{SPLIT_ID}__{model_variant}__seed{random_seed}"
    return EXPERIMENTS_DIR / "logs" / f"{stem}__training.json"


CHECKPOINT_PATH = checkpoint_path_for(MODEL_VARIANT, RANDOM_SEED)

# 这里只保留 model.py 实际读取的字段。class_mask 会由模型根据
# counterpart_type 生成，analytic_global 等当前版本未使用的字段不进 batch。
MODEL_INPUT_KEYS = (
    "actor_factual",
    "actor_counterfactual",
    "target",
    "maneuver_sequence",
    "analytic_factual",
    "analytic_difference",
    "maneuver_global",
    "error_features",
    "error_mask",
    "counterpart_type",
)


@dataclass(frozen=True)
class NormalizationStats:
    """Statistics fitted only on training samples and saved with the model."""

    trajectory_mean: np.ndarray
    trajectory_scale: np.ndarray
    analytic_mean: np.ndarray
    analytic_scale: np.ndarray
    analytic_difference_scale: np.ndarray
    maneuver_sequence_scale: np.ndarray
    maneuver_static_mean: np.ndarray
    maneuver_static_scale: np.ndarray
    error_mean: np.ndarray
    error_scale: np.ndarray

    def checkpoint_dict(self) -> dict[str, list[float]]:
        """Convert NumPy arrays into a checkpoint-friendly plain dictionary."""
        return {
            key: value.tolist()
            for key, value in asdict(self).items()
        }

    @classmethod
    def from_checkpoint_dict(
        cls,
        payload: Mapping[str, object],
    ) -> "NormalizationStats":
        """Restore the exact training-only statistics saved in a checkpoint."""
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        if set(payload) != expected:
            raise ValueError("Checkpoint normalization fields do not match model_train.py.")
        return cls(
            **{
                key: np.asarray(payload[key], dtype=np.float32)
                for key in expected
            }
        )


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    brier_score: float


def set_random_seed(seed: int) -> None:
    """Enable deterministic execution and seed every training RNG."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def runtime_environment(device: torch.device) -> dict[str, object]:
    """Record the software and accelerator context needed to audit a run."""
    try:
        sktime_version: str | None = version("sktime")
    except PackageNotFoundError:
        sktime_version = None
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "sktime_version": sktime_version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_type": device.type,
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor() or "cpu"
        ),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def peak_cpu_memory() -> tuple[int | None, str]:
    """Return process peak resident memory using the current platform API."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE,
            ctypes.c_void_p,
            wintypes.DWORD,
        )
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        succeeded = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        if not succeeded:
            return None, "windows_GetProcessMemoryInfo_failed"
        return int(counters.PeakWorkingSetSize), "windows_peak_working_set"

    try:
        import resource

        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        multiplier = 1 if platform.system() == "Darwin" else 1024
        return peak_rss * multiplier, "resource_ru_maxrss"
    except (ImportError, OSError, ValueError):
        return None, "unavailable"


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _actor_group_key(sample: Mapping[str, object]) -> str:
    """Use the exact factual actor trajectory as the actor-group proxy.

    The current raw JSON does not store ``actor_id``. Samples generated from
    the same actor reuse the same factual trajectory, so this fingerprint keeps
    those near-duplicate samples in one split.
    """
    actor = np.ascontiguousarray(sample["actor_factual"], dtype=np.float32)
    # ponytail: replace this proxy with actor_id when the dataset stores it.
    return hashlib.blake2b(actor.tobytes(), digest_size=16).hexdigest()


def split_by_actor_group(
    samples: list[dict[str, object]],
    validation_fraction: float = VALIDATION_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[list[int], list[int], int]:
    """Split complete actor groups instead of randomly splitting samples."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        group_to_indices[_actor_group_key(sample)].append(index)

    groups = list(group_to_indices)
    if len(groups) < 2:
        raise ValueError("At least two actor groups are required for train/validation split.")

    random.Random(seed).shuffle(groups)
    validation_group_count = max(1, round(len(groups) * validation_fraction))
    validation_groups = set(groups[:validation_group_count])

    train_indices: list[int] = []
    validation_indices: list[int] = []
    for group, indices in group_to_indices.items():
        destination = validation_indices if group in validation_groups else train_indices
        destination.extend(indices)

    if not train_indices or not validation_indices:
        raise ValueError("Train and validation splits must both contain samples.")
    if set(train_indices).intersection(validation_indices):
        raise RuntimeError("Train/validation sample overlap detected.")

    return sorted(train_indices), sorted(validation_indices), len(groups)


def positions_for_sample_indices(
    samples: list[dict[str, object]],
    sample_indices: Iterable[int],
) -> list[int]:
    """Map frozen file-level sample_index values to positions in a loaded subset."""
    position_by_sample_index = {
        int(sample["sample_index"]): position
        for position, sample in enumerate(samples)
    }
    requested = list(sample_indices)
    missing = sorted(set(requested) - set(position_by_sample_index))
    if missing:
        raise ValueError(f"Loaded data is missing manifest samples: {missing[:10]}")
    return [position_by_sample_index[index] for index in requested]


def _mean_and_scale(arrays: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate(tuple(arrays), axis=0)
    mean = values.mean(axis=0, dtype=np.float64)
    scale = values.std(axis=0, dtype=np.float64)
    return mean.astype(np.float32), np.maximum(scale, 1e-6).astype(np.float32)


def _root_mean_square_scale(arrays: Iterable[np.ndarray]) -> np.ndarray:
    """Scale sparse differences without centering, so exact zeros stay zero."""
    values = np.concatenate(tuple(arrays), axis=0)
    scale = np.sqrt(np.mean(np.square(values), axis=0, dtype=np.float64))
    return np.maximum(scale, 1e-6).astype(np.float32)


def fit_normalization_stats(
    samples: list[dict[str, object]],
    train_indices: list[int],
    model_variant: str = MODEL_VARIANT,
) -> NormalizationStats:
    """Fit only the fields consumed by one variant, using train samples only."""
    if model_variant not in SUPPORTED_MODEL_VARIANTS:
        raise ValueError(f"Unsupported model_variant: {model_variant}")
    train_samples = [samples[index] for index in train_indices]
    if not train_samples:
        raise ValueError("At least one training sample is required for normalization.")

    # 所有变体共享同一 Cartesian 标尺：只由事实 actor 与 target 拟合，
    # 再将它应用到 counterfactual，避免消融模型使用不同预处理口径。
    trajectory_keys = ["actor_factual", "target"]
    trajectory_mean, trajectory_scale = _mean_and_scale(
        np.asarray(sample[key], dtype=np.float32)
        for sample in train_samples
        for key in trajectory_keys
    )

    analytic_mean = np.zeros(12, dtype=np.float32)
    analytic_scale = np.ones(12, dtype=np.float32)
    analytic_difference_scale = np.ones(12, dtype=np.float32)
    if model_variant == "full":
        analytic_mean, analytic_scale = _mean_and_scale(
            np.asarray(sample["analytic_factual"], dtype=np.float32)
            for sample in train_samples
        )
        analytic_difference_scale = _root_mean_square_scale(
            np.asarray(sample["analytic_difference"], dtype=np.float32)
            for sample in train_samples
        )

    maneuver_sequence_scale = np.ones(4, dtype=np.float32)
    maneuver_static_mean = np.zeros(2, dtype=np.float32)
    maneuver_static_scale = np.ones(2, dtype=np.float32)
    if model_variant != "factual_only":
        maneuver_sequence_scale = _root_mean_square_scale(
            np.asarray(sample["maneuver_sequence"], dtype=np.float32)[:, 1:]
            for sample in train_samples
        )
        maneuver_static_mean, maneuver_static_scale = _mean_and_scale(
            np.asarray(sample["maneuver_global"], dtype=np.float32)[None, 2:4]
            for sample in train_samples
        )

    # 误差缺失时 error_mask=0；统计时只使用对应的有效值。
    error_mean = np.zeros(3, dtype=np.float32)
    error_scale = np.ones(3, dtype=np.float32)
    if model_variant != "factual_only":
        for feature_index in range(3):
            valid_values = np.asarray(
                [
                    np.asarray(sample["error_features"], dtype=np.float32)[feature_index]
                    for sample in train_samples
                    if np.asarray(sample["error_mask"], dtype=np.float32)[feature_index] > 0.5
                ],
                dtype=np.float32,
            )
            if valid_values.size:
                error_mean[feature_index] = valid_values.mean(dtype=np.float64)
                error_scale[feature_index] = max(
                    float(valid_values.std(dtype=np.float64)),
                    1e-6,
                )

    return NormalizationStats(
        trajectory_mean=trajectory_mean,
        trajectory_scale=trajectory_scale,
        analytic_mean=analytic_mean,
        analytic_scale=analytic_scale,
        analytic_difference_scale=analytic_difference_scale,
        maneuver_sequence_scale=maneuver_sequence_scale,
        maneuver_static_mean=maneuver_static_mean,
        maneuver_static_scale=maneuver_static_scale,
        error_mean=error_mean,
        error_scale=error_scale,
    )


def normalization_fitted_fields(model_variant: str) -> list[str]:
    fields = ["actor_factual", "target"]
    if model_variant != "factual_only":
        fields.extend(
            [
                "maneuver_sequence",
                "maneuver_global[2:4]",
                "error_features[error_mask]",
            ]
        )
    if model_variant == "full":
        fields.extend(["analytic_factual", "analytic_difference"])
    return fields


def normalize_model_sample(
    sample: Mapping[str, object],
    stats: NormalizationStats,
) -> dict[str, np.ndarray]:
    """Normalize one model sample without modifying data_processing output."""
    normalized = {
        key: np.asarray(sample[key], dtype=np.float32).copy()
        for key in MODEL_INPUT_KEYS
    }

    for key in ("actor_factual", "actor_counterfactual", "target"):
        normalized[key] = (
            normalized[key] - stats.trajectory_mean
        ) / stats.trajectory_scale

    normalized["analytic_factual"] = (
        normalized["analytic_factual"] - stats.analytic_mean
    ) / stats.analytic_scale
    normalized["analytic_difference"] /= stats.analytic_difference_scale

    # maneuver_flag 保持 0/1；其余稀疏通道只缩放、不减均值。
    normalized["maneuver_sequence"][:, 1:] /= stats.maneuver_sequence_scale
    normalized["maneuver_global"][2:4] = (
        normalized["maneuver_global"][2:4] - stats.maneuver_static_mean
    ) / stats.maneuver_static_scale

    normalized["error_features"] = (
        normalized["error_features"] - stats.error_mean
    ) / stats.error_scale
    normalized["error_features"] *= normalized["error_mask"]
    normalized["y"] = np.asarray(sample["y"], dtype=np.float32)
    return normalized


class IntentionDataset(Dataset[dict[str, Tensor]]):
    """Expose selected, normalized NumPy samples as PyTorch tensors."""

    def __init__(
        self,
        samples: list[dict[str, object]],
        indices: list[int],
        stats: NormalizationStats,
    ) -> None:
        self.samples = samples
        self.indices = indices
        self.stats = stats

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        sample = self.samples[self.indices[item]]
        normalized = normalize_model_sample(sample, self.stats)
        batch_item = {
            key: torch.from_numpy(normalized[key])
            for key in MODEL_INPUT_KEYS
        }
        batch_item["y"] = torch.from_numpy(normalized["y"])
        batch_item["sample_index"] = torch.as_tensor(
            int(sample["sample_index"]),
            dtype=torch.int64,
        )
        return batch_item


def build_dataloaders(
    samples: list[dict[str, object]],
    train_indices: list[int],
    validation_indices: list[int],
    stats: NormalizationStats,
    use_cuda: bool,
    random_seed: int = RANDOM_SEED,
) -> tuple[DataLoader[dict[str, Tensor]], DataLoader[dict[str, Tensor]]]:
    train_dataset = IntentionDataset(samples, train_indices, stats)
    validation_dataset = IntentionDataset(samples, validation_indices, stats)

    common = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": use_cuda,
    }
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(random_seed)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=shuffle_generator,
        **common,
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common)
    return train_loader, validation_loader


def _move_model_batch(
    batch: Mapping[str, Tensor],
    device: torch.device,
    model_input_keys: Iterable[str] = MODEL_INPUT_KEYS,
) -> tuple[dict[str, Tensor], Tensor]:
    """Move only the fields consumed by model.py and the target distribution."""
    model_batch = {
        key: batch[key].to(device, non_blocking=True)
        for key in model_input_keys
    }
    targets = batch["y"].to(device, non_blocking=True)
    return model_batch, targets


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> EpochMetrics:
    """Run one training epoch when optimizer is given, otherwise validate."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_brier = 0.0
    total_samples = 0
    use_amp = device.type == "cuda"

    with torch.set_grad_enabled(training):
        for batch in data_loader:
            model_input_keys = getattr(model, "model_input_keys", MODEL_INPUT_KEYS)
            model_batch, targets = _move_model_batch(
                batch,
                device,
                model_input_keys=model_input_keys,
            )
            batch_size = targets.shape[0]

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                outputs = model(model_batch)
                loss = soft_label_cross_entropy(outputs["masked_logits"], targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training/validation loss detected.")

            if training:
                if scaler is None:
                    raise RuntimeError("A GradScaler is required for training.")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            probabilities = outputs["probabilities"].detach()
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError("Non-finite model probabilities detected.")
            total_loss += loss.detach().item() * batch_size
            total_correct += (
                probabilities.argmax(dim=-1) == targets.argmax(dim=-1)
            ).sum().item()
            total_brier += (
                (probabilities - targets).square().sum(dim=-1)
            ).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("DataLoader produced no samples.")
    return EpochMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        brier_score=total_brier / total_samples,
    )


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    stats: NormalizationStats,
    split_sample_indices: Mapping[str, list[int]],
    train_metrics: EpochMetrics,
    validation_metrics: EpochMetrics,
    history: list[dict[str, object]],
    experiment_metadata: Mapping[str, object],
    random_seed: int,
    model_variant: str,
) -> None:
    """Atomically save everything needed to reproduce and resume training."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "normalization": stats.checkpoint_dict(),
                "split_sample_indices": dict(split_sample_indices),
                "train_metrics": asdict(train_metrics),
                "validation_metrics": asdict(validation_metrics),
                "history": history,
                "experiment": dict(experiment_metadata),
                "config": {
                    "batch_size": BATCH_SIZE,
                    "epochs": EPOCHS,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "random_seed": random_seed,
                    "model_variant": model_variant,
                },
            },
            temporary_path,
        )
        for attempt, delay_seconds in enumerate((0.1, 0.2, 0.4, 0.8, 1.6, None), 1):
            try:
                temporary_path.replace(path)
                break
            except PermissionError:
                if delay_seconds is None:
                    raise
                print(
                    "Checkpoint replace temporarily blocked by Windows; "
                    f"retrying {attempt}/5."
                )
                time.sleep(delay_seconds)
    finally:
        temporary_path.unlink(missing_ok=True)


def train_model(
    dataset_path: Path = DATASET_PATH,
    checkpoint_path: Path | None = None,
    split_manifest_path: Path = SPLIT_MANIFEST_PATH,
    model_variant: str = MODEL_VARIANT,
    random_seed: int = RANDOM_SEED,
    phase2_protocol_path: Path | None = None,
    overwrite_existing: bool = False,
) -> Path:
    """Train with the frozen manifest; never inspect or fit on test samples."""
    if model_variant not in SUPPORTED_MODEL_VARIANTS:
        raise ValueError(
            f"Unsupported model_variant={model_variant!r}; "
            f"expected one of {SUPPORTED_MODEL_VARIANTS}."
        )
    if checkpoint_path is None:
        checkpoint_path = checkpoint_path_for(model_variant, random_seed)
    training_log_path = training_log_path_for(model_variant, random_seed)
    existing_outputs = [
        path for path in (checkpoint_path, training_log_path) if path.exists()
    ]
    if existing_outputs and not overwrite_existing:
        raise FileExistsError(
            f"Training output already exists: {existing_outputs[0]}. "
            "Use a new seed/path or explicitly set overwrite_existing=True."
        )
    phase2_protocol_sha256: str | None = None
    if phase2_protocol_path is not None:
        if not phase2_protocol_path.is_file():
            raise FileNotFoundError(f"Phase 2 protocol not found: {phase2_protocol_path}")
        phase2_protocol_sha256 = sha256_file(phase2_protocol_path)

    manifest, split_sample_indices = load_split_indices(split_manifest_path)
    if manifest.get("split_id") != SPLIT_ID:
        raise ValueError("Unexpected split_id in split manifest.")
    dataset_metadata = manifest.get("dataset", {})
    if dataset_metadata.get("version") != DATASET_VERSION:
        raise ValueError("Unexpected dataset version in split manifest.")
    if dataset_metadata.get("label_version") != LABEL_VERSION:
        raise ValueError("Unexpected label version in split manifest.")
    verify_manifest_dataset(manifest, dataset_path, verify_hash=True)

    set_random_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_sample_indices = split_sample_indices["train"]
    validation_sample_indices = split_sample_indices["validation"]
    requested = set(train_sample_indices + validation_sample_indices)
    print(f"Loading frozen train/validation samples from: {dataset_path}")
    samples = load_training_data(
        dataset_path,
        sample_indices=requested,
        allow_sidecar_labels=False,
    )
    if len(samples) < 2:
        raise ValueError("At least two labeled samples are required.")

    train_positions = positions_for_sample_indices(samples, train_sample_indices)
    validation_positions = positions_for_sample_indices(samples, validation_sample_indices)
    stats = fit_normalization_stats(samples, train_positions, model_variant=model_variant)
    train_loader, validation_loader = build_dataloaders(
        samples,
        train_positions,
        validation_positions,
        stats,
        use_cuda=device.type == "cuda",
        random_seed=random_seed,
    )

    audit = manifest["audit"]
    print(
        f"Loaded {len(samples)} labeled samples from frozen actor groups: "
        f"train={len(train_positions)}, validation={len(validation_positions)}, "
        f"held-out test={len(split_sample_indices['test'])}."
    )
    print(f"Actors={audit['actor_counts']}")
    print(
        f"Device={device}, model_variant={model_variant}, seed={random_seed}, "
        f"batch_size={BATCH_SIZE}, epochs={EPOCHS}"
    )

    model = build_intention_model(model_variant).to(device)
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_validation_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, object]] = []
    training_started_at = datetime.now(timezone.utc).isoformat()
    training_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    experiment_metadata = {
        "run_id": checkpoint_path.stem,
        "experiment_id": (
            phase2_protocol_path.stem if phase2_protocol_path is not None else None
        ),
        "dataset_version": DATASET_VERSION,
        "label_version": LABEL_VERSION,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": str(dataset_metadata["sha256"]),
        "split_id": SPLIT_ID,
        "split_manifest_path": str(split_manifest_path.resolve()),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "split_selected_seed": int(manifest["config"]["selected_seed"]),
        "model_variant": model_variant,
        "random_seed": random_seed,
        "phase2_protocol_path": (
            str(phase2_protocol_path.resolve()) if phase2_protocol_path is not None else None
        ),
        "phase2_protocol_sha256": phase2_protocol_sha256,
        "model_parameter_count": total_parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "model_input_keys": list(getattr(model, "model_input_keys")),
        "model_kwargs": dict(getattr(model, "model_kwargs")),
        "normalization_fitted_fields": normalization_fitted_fields(model_variant),
        "normalization_policy": (
            "fit train actor_factual+target Cartesian statistics; apply the same "
            "statistics to counterfactual; fit other fields only when the variant uses them"
        ),
        "data_label_policy": "embedded_y_only_no_implicit_sidecars",
        "model_selection_metric": "validation_soft_nll",
        "model_selection_mode": "min",
        "test_access_policy": "never_loaded_by_model_train",
        "code_sha256": {
            "model.py": sha256_file(Path(__file__).resolve().with_name("model.py")),
            "model_train.py": sha256_file(Path(__file__).resolve()),
        },
        "runtime_environment": runtime_environment(device),
    }
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            scaler=scaler,
        )
        validation_metrics = run_epoch(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train": asdict(train_metrics),
                "validation": asdict(validation_metrics),
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
        )

        improved = validation_metrics.loss < best_validation_loss
        if improved:
            best_validation_loss = validation_metrics.loss
            best_epoch = epoch
            save_checkpoint(
                checkpoint_path,
                epoch,
                model,
                optimizer,
                stats,
                split_sample_indices,
                train_metrics,
                validation_metrics,
                history,
                experiment_metadata,
                random_seed,
                model_variant,
            )

        marker = " saved" if improved else ""
        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train loss={train_metrics.loss:.4f}, "
            f"acc={train_metrics.accuracy:.3f}, "
            f"brier={train_metrics.brier_score:.4f} | "
            f"val loss={validation_metrics.loss:.4f}, "
            f"acc={validation_metrics.accuracy:.3f}, "
            f"brier={validation_metrics.brier_score:.4f}{marker}"
        )

    training_seconds = time.perf_counter() - training_start
    peak_vram_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    peak_cpu_memory_bytes, cpu_memory_measurement = peak_cpu_memory()
    training_log = {
        "schema_version": 1,
        "status": "completed",
        "experiment": experiment_metadata,
        "config": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "random_seed": random_seed,
            "model_variant": model_variant,
        },
        "training": {
            "started_at": training_started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": training_seconds,
            "peak_vram_bytes": peak_vram_bytes,
            "peak_cpu_memory_bytes": peak_cpu_memory_bytes,
            "cpu_memory_measurement": cpu_memory_measurement,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "history": history,
        },
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256_file(checkpoint_path),
        },
    }
    _atomic_write_json(training_log_path, training_log)
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training log: {training_log_path}")
    return checkpoint_path


if __name__ == "__main__":
    # 单 run 调试也必须绑定已冻结协议，避免生成与 Phase 2 同名但不可审计的产物。
    # 完整 15-run 矩阵请直接运行 phase2_experiment.py。
    frozen_protocol = EXPERIMENTS_DIR / "configs" / "phase2_core_v2.json"
    if not frozen_protocol.is_file():
        raise FileNotFoundError(
            "Run phase2_preflight.py first to freeze phase2_core_v2.json."
        )
    train_model(phase2_protocol_path=frozen_protocol)
