"""Phase 3 的固定运行矩阵、命名规则和协议读取门禁。

本文件只定义实验，不执行训练或测试。Phase 2 的脚本和产物保持字节不变；
Phase 3 正式结果只写入当前 ``phase3_extended_v3`` 命名空间；此前的
``phase3_extended_v1/v2`` 前缀保留为只读审计记录。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from experiment_setup import (
    DATASET_PATH,
    DATASET_VERSION,
    LABEL_VERSION,
    SPLIT_ID,
    SPLIT_MANIFEST_PATH,
    sha256_file,
)
from model_train import BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY


IDEA_DIR = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = IDEA_DIR / "experiment_data"
PHASE3_ID = "phase3_extended_v3"
PHASE3_PROTOCOL_PATH = EXPERIMENTS_DIR / "configs" / f"{PHASE3_ID}.json"
SUPERSEDED_PHASE3_PROTOCOL_PATH = (
    EXPERIMENTS_DIR / "configs" / "phase3_extended_v2.json"
)
PARENT_PHASE2_PROTOCOL_PATH = (
    EXPERIMENTS_DIR / "configs" / "phase2_core_v2.json"
)
PHASE3_RUN_DATA_TAG = (
    "sgp4_1min_10400_labels_v5_clean_20260817__phase3_extended_v3"
)
PHASE3_SEEDS = (7, 17, 42, 73, 101)
CHECKPOINT_REPLACE_ATTEMPTS = 8
CHECKPOINT_RETRY_BASE_SECONDS = 0.1
CHECKPOINT_RETRY_MAX_SECONDS = 1.0


@dataclass(frozen=True)
class RunSpec:
    """One pre-registered Phase 3 training condition before seed expansion."""

    run_key: str
    comparison_group: str
    trainer: str
    model_variant: str
    train_label_id: str
    training_strategy: str
    normalization_variant: str

    def checkpoint_metadata(self) -> dict[str, str]:
        return asdict(self)


RUN_SPECS = (
    # E3: B1 and three raw-input neural backbones. B0/B2/P are reused from Phase 2.
    RunSpec(
        "b1_analytic_random_forest",
        "e3_unified_baselines",
        "random_forest",
        "analytic_random_forest",
        "v5_clean",
        "standard",
        "not_applicable_tree",
    ),
    RunSpec(
        "b3_bilstm_attention",
        "e3_unified_baselines",
        "torch",
        "bilstm_attention",
        "v5_clean",
        "standard",
        "factual_only",
    ),
    RunSpec(
        "b4_inception_time",
        "e3_unified_baselines",
        "torch",
        "inception_time",
        "v5_clean",
        "standard",
        "factual_only",
    ),
    RunSpec(
        "b5_lightweight_transformer",
        "e3_unified_baselines",
        "torch",
        "lightweight_transformer",
        "v5_clean",
        "standard",
        "factual_only",
    ),
    # E5: A0-A4 are unmasked controls; A1->A2 compares direct counterfactual
    # fusion with event-conditioned comparison rather than claiming pure addition.
    RunSpec(
        "a0_factual_only_no_mask",
        "e5_structure_ablation",
        "torch",
        "a0_factual_only_no_mask",
        "v5_clean",
        "standard",
        "factual_only",
    ),
    RunSpec(
        "a1_direct_counterfactual",
        "e5_structure_ablation",
        "torch",
        "a1_direct_counterfactual",
        "v5_clean",
        "standard",
        "full",
    ),
    RunSpec(
        "a2_event",
        "e5_structure_ablation",
        "torch",
        "a2_event",
        "v5_clean",
        "standard",
        "full",
    ),
    RunSpec(
        "a3_event_analytic",
        "e5_structure_ablation",
        "torch",
        "a3_event_analytic",
        "v5_clean",
        "standard",
        "full",
    ),
    RunSpec(
        "a4_event_analytic_static_no_mask",
        "e5_structure_ablation",
        "torch",
        "a4_event_analytic_static_no_mask",
        "v5_clean",
        "standard",
        "full",
    ),
    RunSpec(
        "a6_separate_encoders",
        "e5_structure_ablation",
        "torch",
        "a6_separate_encoders",
        "v5_clean",
        "standard",
        "full",
    ),
    RunSpec(
        "a6_parameter_matched_separate_encoders",
        "e5_structure_ablation",
        "torch",
        "a6_separate_encoders_parameter_matched",
        "v5_clean",
        "standard",
        "full",
    ),
    RunSpec(
        "without_maneuver_sequence",
        "e5_structure_ablation",
        "torch",
        "without_maneuver_sequence",
        "v5_clean",
        "standard",
        "full",
    ),
    RunSpec(
        "without_orbit_error_summary",
        "e5_structure_ablation",
        "torch",
        "without_orbit_error_summary",
        "v5_clean",
        "standard",
        "full",
    ),
    # E2: only train labels change; validation/test always use v5_clean.
    RunSpec(
        "e2_hard_argmax_full",
        "e2_label_sensitivity",
        "torch",
        "full",
        "hard_argmax_v5",
        "standard",
        "full",
    ),
    RunSpec(
        "e2_geometry_rule_full",
        "e2_label_sensitivity",
        "torch",
        "full",
        "fixed_rule_geometry_dynamics",
        "standard",
        "full",
    ),
    RunSpec(
        "e2_unweighted_six_rule_full",
        "e2_label_sensitivity",
        "torch",
        "full",
        "unweighted_six_rule_mean",
        "standard",
        "full",
    ),
    RunSpec(
        "e2_v5_train_aug132_full",
        "e2_label_sensitivity",
        "torch",
        "full",
        "v5_train_aug132",
        "standard",
        "full",
    ),
    # Section 9: all statistics are derived from train labels only.
    RunSpec(
        "i1_dominant_balanced_sampler_full",
        "class_imbalance",
        "torch",
        "full",
        "v5_clean",
        "dominant_balanced_sampler",
        "full",
    ),
    RunSpec(
        "i2_soft_class_weight_full",
        "class_imbalance",
        "torch",
        "full",
        "v5_clean",
        "soft_class_weight",
        "full",
    ),
    RunSpec(
        "i3_soft_focal_full",
        "class_imbalance",
        "torch",
        "full",
        "v5_clean",
        "soft_focal_gamma2",
        "full",
    ),
)


REUSED_PHASE2_RUNS = {
    "b0_counterpart_type_prior": "counterpart_type_prior",
    "b2_factual_only_tcn": "factual_only",
    "p_a5_a7_full_v5_clean": "full",
    "without_analytic": "full_without_analytic",
}


def validate_run_specs() -> None:
    keys = [spec.run_key for spec in RUN_SPECS]
    if len(keys) != len(set(keys)):
        raise ValueError("Phase 3 run_key values must be unique.")
    allowed_trainers = {"torch", "random_forest"}
    allowed_normalization = {
        "factual_only",
        "full_without_analytic",
        "full",
        "not_applicable_tree",
    }
    for spec in RUN_SPECS:
        if spec.trainer not in allowed_trainers:
            raise ValueError(f"Unsupported trainer in {spec.run_key}: {spec.trainer}")
        if spec.normalization_variant not in allowed_normalization:
            raise ValueError(
                f"Unsupported normalization variant in {spec.run_key}: "
                f"{spec.normalization_variant}"
            )
        if spec.trainer == "random_forest" and spec.normalization_variant != "not_applicable_tree":
            raise ValueError("RandomForest must declare normalization as not applicable.")
        if spec.trainer == "torch" and spec.normalization_variant == "not_applicable_tree":
            raise ValueError("Torch runs require a frozen train-only normalization variant.")


def get_run_spec(run_key: str) -> RunSpec:
    matches = [spec for spec in RUN_SPECS if spec.run_key == run_key]
    if len(matches) != 1:
        raise KeyError(f"Unknown Phase 3 run_key: {run_key}")
    return matches[0]


def run_stem(spec: RunSpec, random_seed: int) -> str:
    if random_seed not in PHASE3_SEEDS:
        raise ValueError(f"Seed {random_seed} is not frozen for Phase 3.")
    return (
        f"{PHASE3_RUN_DATA_TAG}__{SPLIT_ID}__{spec.run_key}__seed{random_seed}"
    )


def checkpoint_path_for(spec: RunSpec, random_seed: int) -> Path:
    suffix = ".pkl" if spec.trainer == "random_forest" else ".pt"
    return EXPERIMENTS_DIR / "checkpoints" / f"{run_stem(spec, random_seed)}{suffix}"


def training_log_path_for(spec: RunSpec, random_seed: int) -> Path:
    return (
        EXPERIMENTS_DIR
        / "logs"
        / f"{run_stem(spec, random_seed)}__training.json"
    )


def failure_log_path_for(spec: RunSpec, random_seed: int) -> Path:
    return (
        EXPERIMENTS_DIR
        / "logs"
        / f"{run_stem(spec, random_seed)}__failure.json"
    )


def evaluation_paths_for(
    spec: RunSpec,
    random_seed: int,
    split_name: str,
) -> tuple[Path, Path]:
    if split_name not in {"validation", "test"}:
        raise ValueError("Phase 3 comparison split must be validation or test.")
    stem = f"{run_stem(spec, random_seed)}__{split_name}"
    return (
        EXPERIMENTS_DIR / "predictions" / f"{stem}.jsonl",
        EXPERIMENTS_DIR / "metrics" / f"{stem}.json",
    )


def verify_reused_phase2_artifacts(protocol: dict[str, Any]) -> None:
    """Verify the exact non-blind Phase 2 files reused in final comparisons."""
    audit = protocol.get("parent_phase2", {}).get("artifact_audit", {})
    frozen_hashes = audit.get("artifact_sha256")
    if not isinstance(frozen_hashes, dict) or not frozen_hashes:
        raise ValueError("Phase 3 protocol has no reused Phase 2 artifact hash manifest.")
    if int(audit.get("artifact_count", -1)) != len(frozen_hashes):
        raise ValueError("Reused Phase 2 artifact count does not match its hash manifest.")
    for relative_path, expected_hash in frozen_hashes.items():
        path = IDEA_DIR / relative_path
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Reused Phase 2 artifact changed: {relative_path}")


def load_phase3_protocol(
    *,
    verify_code: bool = True,
    verify_reused_artifacts: bool = False,
) -> dict[str, Any]:
    if not PHASE3_PROTOCOL_PATH.is_file():
        raise FileNotFoundError(
            f"Phase 3 protocol is not frozen: {PHASE3_PROTOCOL_PATH}. "
            "Run phase3_preflight.py first."
        )
    protocol = json.loads(PHASE3_PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("phase3_id") != PHASE3_ID:
        raise ValueError("Unexpected Phase 3 protocol identifier.")
    if protocol.get("status") != "frozen_before_training":
        raise ValueError("Phase 3 protocol is not in frozen_before_training state.")
    if verify_code:
        for relative_path, expected_hash in protocol["code_sha256"].items():
            path = IDEA_DIR / relative_path
            if sha256_file(path) != expected_hash:
                raise ValueError(f"Phase 3 code changed after freeze: {relative_path}")
    if verify_reused_artifacts:
        verify_reused_phase2_artifacts(protocol)
    return protocol


validate_run_specs()


__all__ = [
    "BATCH_SIZE",
    "CHECKPOINT_REPLACE_ATTEMPTS",
    "CHECKPOINT_RETRY_BASE_SECONDS",
    "CHECKPOINT_RETRY_MAX_SECONDS",
    "DATASET_PATH",
    "DATASET_VERSION",
    "EPOCHS",
    "EXPERIMENTS_DIR",
    "IDEA_DIR",
    "LABEL_VERSION",
    "LEARNING_RATE",
    "PARENT_PHASE2_PROTOCOL_PATH",
    "PHASE3_ID",
    "PHASE3_PROTOCOL_PATH",
    "PHASE3_RUN_DATA_TAG",
    "PHASE3_SEEDS",
    "REUSED_PHASE2_RUNS",
    "RUN_SPECS",
    "RunSpec",
    "SPLIT_ID",
    "SPLIT_MANIFEST_PATH",
    "SUPERSEDED_PHASE3_PROTOCOL_PATH",
    "WEIGHT_DECAY",
    "checkpoint_path_for",
    "evaluation_paths_for",
    "failure_log_path_for",
    "get_run_spec",
    "load_phase3_protocol",
    "run_stem",
    "training_log_path_for",
    "verify_reused_phase2_artifacts",
]
