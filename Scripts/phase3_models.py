"""Phase 3 对比模型与结构消融模型。

本文件只新增 Phase 3 网络，不修改 Phase 2 已冻结的 ``model.py``。
所有模型统一接收 ``Mapping[str, Tensor]``，并返回与 Phase 2 相同的五个字段。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from model import (
    ANALYTIC_SEQUENCE_DIM,
    STATIC_INPUT_DIM,
    TRAJECTORY_PAIR_DIM,
    EventConditionedComparisonNetwork,
    FeatureFusionNetwork,
    FeatureSequenceEncoder,
    SharedTrajectoryEncoder,
    StaticFeatureNetwork,
)


SUPPORTED_PHASE3_MODEL_VARIANTS = (
    "bilstm_attention",
    "inception_time",
    "lightweight_transformer",
    "a0_factual_only_no_mask",
    "a1_direct_counterfactual",
    "a2_event",
    "a3_event_analytic",
    "a4_event_analytic_static_no_mask",
    "a6_separate_encoders",
    "a6_separate_encoders_parameter_matched",
    "without_maneuver_sequence",
    "without_orbit_error_summary",
    "without_class_mask",
)

_FACTUAL_ONLY_VARIANTS = {
    "bilstm_attention",
    "inception_time",
    "lightweight_transformer",
    "a0_factual_only_no_mask",
}
_DEDICATED_FACTUAL_ENCODER_VARIANTS = {
    "bilstm_attention",
    "inception_time",
    "lightweight_transformer",
    "a6_separate_encoders",
    "a6_separate_encoders_parameter_matched",
}
_SEPARATE_ENCODER_VARIANTS = {
    "a6_separate_encoders",
    "a6_separate_encoders_parameter_matched",
}
_EVENT_VARIANTS = {
    "a2_event",
    "a3_event_analytic",
    "a4_event_analytic_static_no_mask",
    "a6_separate_encoders",
    "a6_separate_encoders_parameter_matched",
    "without_maneuver_sequence",
    "without_orbit_error_summary",
    "without_class_mask",
}
_ANALYTIC_VARIANTS = {
    "a3_event_analytic",
    "a4_event_analytic_static_no_mask",
    "a6_separate_encoders",
    "a6_separate_encoders_parameter_matched",
    "without_maneuver_sequence",
    "without_orbit_error_summary",
    "without_class_mask",
}
_STATIC_VARIANTS = {
    "a4_event_analytic_static_no_mask",
    "a6_separate_encoders",
    "a6_separate_encoders_parameter_matched",
    "without_maneuver_sequence",
    "without_orbit_error_summary",
    "without_class_mask",
}
_NO_CLASS_MASK_VARIANTS = {
    "a0_factual_only_no_mask",
    "a1_direct_counterfactual",
    "a2_event",
    "a3_event_analytic",
    "a4_event_analytic_static_no_mask",
    "without_class_mask",
}


def _check_tensor_shape(name: str, value: Tensor, ndim: int, last_dim: int) -> None:
    """在模型入口给出可读的形状错误。"""
    if value.ndim != ndim or value.shape[-1] != last_dim:
        raise ValueError(
            f"{name} must have {ndim} dimensions and last dimension "
            f"{last_dim}; got {tuple(value.shape)}."
        )


class BiLSTMAttentionEncoder(nn.Module):
    """Bi-LSTM 编码后用可学习注意力汇聚整段事实轨迹。"""

    def __init__(
        self,
        input_dim: int = TRAJECTORY_PAIR_DIM,
        hidden_dim: int = 128,
        attention_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        recurrent_hidden_dim = max(1, hidden_dim // 2)
        recurrent_output_dim = recurrent_hidden_dim * 2
        self.input_dim = input_dim
        self.recurrent = nn.LSTM(
            input_size=input_dim,
            hidden_size=recurrent_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.Sequential(
            nn.Linear(recurrent_output_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )
        self.output_projection = nn.Sequential(
            nn.Linear(recurrent_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, sequence: Tensor) -> Tensor:
        _check_tensor_shape("sequence", sequence, ndim=3, last_dim=self.input_dim)
        encoded, _ = self.recurrent(sequence)
        attention_weights = torch.softmax(self.attention(encoded), dim=1)
        pooled = (attention_weights * encoded).sum(dim=1)
        return self.output_projection(pooled)


class _InceptionTimeBlock(nn.Module):
    """InceptionTime 的多尺度卷积分支与残差连接。"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bottleneck_dim: int,
        kernel_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        if output_dim % (len(kernel_sizes) + 1) != 0:
            raise ValueError("output_dim must be divisible by the number of branches.")
        if any(kernel_size < 1 or kernel_size % 2 == 0 for kernel_size in kernel_sizes):
            raise ValueError("InceptionTime kernel sizes must be positive odd integers.")

        branch_dim = output_dim // (len(kernel_sizes) + 1)
        self.bottleneck = nn.Conv1d(input_dim, bottleneck_dim, kernel_size=1)
        self.convolution_branches = nn.ModuleList(
            nn.Conv1d(
                bottleneck_dim,
                branch_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            for kernel_size in kernel_sizes
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(input_dim, branch_dim, kernel_size=1),
        )
        self.residual = (
            nn.Identity()
            if input_dim == output_dim
            else nn.Conv1d(input_dim, output_dim, kernel_size=1)
        )
        self.normalization = nn.BatchNorm1d(output_dim)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        bottleneck = self.bottleneck(x)
        branches = [branch(bottleneck) for branch in self.convolution_branches]
        branches.append(self.pool_branch(x))
        return self.activation(self.normalization(torch.cat(branches, dim=1)) + self.residual(x))


class InceptionTimeEncoder(nn.Module):
    """用多尺度 InceptionTime 块编码事实 actor-target 轨迹。"""

    def __init__(
        self,
        input_dim: int = TRAJECTORY_PAIR_DIM,
        hidden_dim: int = 128,
        bottleneck_dim: int = 32,
        num_blocks: int = 3,
        kernel_sizes: tuple[int, ...] = (9, 19, 39),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1.")
        self.input_dim = input_dim
        self.blocks = nn.Sequential(
            *(
                _InceptionTimeBlock(
                    input_dim=input_dim if block_index == 0 else hidden_dim,
                    output_dim=hidden_dim,
                    bottleneck_dim=bottleneck_dim,
                    kernel_sizes=kernel_sizes,
                )
                for block_index in range(num_blocks)
            )
        )
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, sequence: Tensor) -> Tensor:
        _check_tensor_shape("sequence", sequence, ndim=3, last_dim=self.input_dim)
        encoded = self.blocks(sequence.transpose(1, 2))
        pooled = torch.cat((encoded.mean(dim=-1), encoded.amax(dim=-1)), dim=-1)
        return self.output_projection(pooled)


class LightweightTransformerEncoder(nn.Module):
    """轻量 Transformer 编码器，使用无参数正弦位置编码。"""

    def __init__(
        self,
        input_dim: int = TRAJECTORY_PAIR_DIM,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _position_encoding(self, length: int, reference: Tensor) -> Tensor:
        positions = torch.arange(length, device=reference.device, dtype=reference.dtype)
        positions = positions.unsqueeze(1)
        frequencies = torch.arange(
            0,
            self.hidden_dim,
            2,
            device=reference.device,
            dtype=reference.dtype,
        )
        frequencies = torch.exp(frequencies * (-math.log(10_000.0) / self.hidden_dim))
        encoding = reference.new_zeros((length, self.hidden_dim))
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        return encoding.unsqueeze(0)

    def forward(self, sequence: Tensor) -> Tensor:
        _check_tensor_shape("sequence", sequence, ndim=3, last_dim=self.input_dim)
        encoded = self.input_projection(sequence)
        encoded = self.encoder(encoded + self._position_encoding(encoded.shape[1], encoded))
        pooled = torch.cat((encoded.mean(dim=1), encoded.amax(dim=1)), dim=-1)
        return self.output_projection(pooled)


class Phase3IntentionRecognitionModel(nn.Module):
    """按 Phase 3 变体组装对比模型或结构消融模型。"""

    def __init__(
        self,
        model_variant: str,
        trajectory_hidden_dim: int = 128,
        maneuver_hidden_dim: int = 32,
        event_hidden_dim: int = 64,
        analytic_hidden_dim: int = 96,
        static_hidden_dim: int = 32,
        fusion_hidden_dim: int = 128,
        dropout: float = 0.1,
        bilstm_attention_dim: int = 64,
        bilstm_layers: int = 1,
        inception_bottleneck_dim: int = 32,
        inception_blocks: int = 3,
        inception_kernel_sizes: tuple[int, ...] = (9, 19, 39),
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_feedforward_dim: int = 256,
    ) -> None:
        super().__init__()
        if model_variant not in SUPPORTED_PHASE3_MODEL_VARIANTS:
            raise ValueError(
                f"Unsupported model_variant={model_variant!r}; expected one of "
                f"{SUPPORTED_PHASE3_MODEL_VARIANTS}."
            )
        self.model_variant = model_variant
        self.class_mask_applied = model_variant not in _NO_CLASS_MASK_VARIANTS
        self.model_kwargs = {
            "trajectory_hidden_dim": trajectory_hidden_dim,
            "maneuver_hidden_dim": maneuver_hidden_dim,
            "event_hidden_dim": event_hidden_dim,
            "analytic_hidden_dim": analytic_hidden_dim,
            "static_hidden_dim": static_hidden_dim,
            "fusion_hidden_dim": fusion_hidden_dim,
            "dropout": dropout,
            "bilstm_attention_dim": bilstm_attention_dim,
            "bilstm_layers": bilstm_layers,
            "inception_bottleneck_dim": inception_bottleneck_dim,
            "inception_blocks": inception_blocks,
            "inception_kernel_sizes": tuple(inception_kernel_sizes),
            "transformer_heads": transformer_heads,
            "transformer_layers": transformer_layers,
            "transformer_feedforward_dim": transformer_feedforward_dim,
        }
        self.model_input_keys = self._input_keys_for(model_variant)

        if model_variant == "bilstm_attention":
            self.factual_trajectory_encoder = BiLSTMAttentionEncoder(
                hidden_dim=trajectory_hidden_dim,
                attention_dim=bilstm_attention_dim,
                num_layers=bilstm_layers,
                dropout=dropout,
            )
        elif model_variant == "inception_time":
            self.factual_trajectory_encoder = InceptionTimeEncoder(
                hidden_dim=trajectory_hidden_dim,
                bottleneck_dim=inception_bottleneck_dim,
                num_blocks=inception_blocks,
                kernel_sizes=tuple(inception_kernel_sizes),
                dropout=dropout,
            )
        elif model_variant == "lightweight_transformer":
            self.factual_trajectory_encoder = LightweightTransformerEncoder(
                hidden_dim=trajectory_hidden_dim,
                num_heads=transformer_heads,
                num_layers=transformer_layers,
                feedforward_dim=transformer_feedforward_dim,
                dropout=dropout,
            )
        elif model_variant in _SEPARATE_ENCODER_VARIANTS:
            self.factual_trajectory_encoder = SharedTrajectoryEncoder(
                hidden_dim=trajectory_hidden_dim,
                dropout=dropout,
            )
            self.counterfactual_trajectory_encoder = SharedTrajectoryEncoder(
                hidden_dim=trajectory_hidden_dim,
                dropout=dropout,
            )
        else:
            self.shared_trajectory_encoder = SharedTrajectoryEncoder(
                hidden_dim=trajectory_hidden_dim,
                dropout=dropout,
            )

        fusion_input_dim = trajectory_hidden_dim
        if model_variant == "a1_direct_counterfactual":
            fusion_input_dim += trajectory_hidden_dim
        if model_variant in _EVENT_VARIANTS:
            self.event_comparison_network = EventConditionedComparisonNetwork(
                trajectory_dim=trajectory_hidden_dim,
                maneuver_hidden_dim=maneuver_hidden_dim,
                output_dim=event_hidden_dim,
                dropout=dropout,
            )
            fusion_input_dim += event_hidden_dim
        if model_variant in _ANALYTIC_VARIANTS:
            self.feature_sequence_encoder = FeatureSequenceEncoder(
                hidden_dim=analytic_hidden_dim,
                dropout=dropout,
            )
            fusion_input_dim += analytic_hidden_dim
        if model_variant in _STATIC_VARIANTS:
            static_input_dim = (
                4
                if model_variant == "without_orbit_error_summary"
                else STATIC_INPUT_DIM
            )
            self.static_feature_network = StaticFeatureNetwork(
                input_dim=static_input_dim,
                hidden_dim=static_hidden_dim,
                dropout=dropout,
            )
            fusion_input_dim += static_hidden_dim

        self.feature_fusion_network = FeatureFusionNetwork(
            input_dim=fusion_input_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=dropout,
        )
        self.register_buffer(
            "counterpart_class_masks",
            torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 0, 0, 0, 1, 1, 1],
                ],
                dtype=torch.bool,
            ),
        )

    @staticmethod
    def _input_keys_for(model_variant: str) -> tuple[str, ...]:
        if model_variant in _FACTUAL_ONLY_VARIANTS:
            return ("actor_factual", "target", "counterpart_type")
        if model_variant == "a1_direct_counterfactual":
            return (
                "actor_factual",
                "actor_counterfactual",
                "target",
                "counterpart_type",
            )
        keys = [
            "actor_factual",
            "actor_counterfactual",
            "target",
        ]
        if model_variant != "without_maneuver_sequence":
            keys.append("maneuver_sequence")
        if model_variant in _ANALYTIC_VARIANTS:
            keys.extend(("analytic_factual", "analytic_difference"))
        if model_variant in _STATIC_VARIANTS:
            keys.append("maneuver_global")
            if model_variant != "without_orbit_error_summary":
                keys.extend(("error_features", "error_mask"))
        keys.append("counterpart_type")
        return tuple(keys)

    def _build_class_mask(self, counterpart_type: Tensor) -> Tensor:
        _check_tensor_shape("counterpart_type", counterpart_type, ndim=2, last_dim=2)
        return self.counterpart_class_masks[counterpart_type.argmax(dim=-1)]

    def _encode_factual(self, factual_pair: Tensor) -> Tensor:
        if self.model_variant in _DEDICATED_FACTUAL_ENCODER_VARIANTS:
            return self.factual_trajectory_encoder(factual_pair)
        return self.shared_trajectory_encoder(factual_pair)

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        actor_factual = batch["actor_factual"]
        target = batch["target"]
        counterpart_type = batch["counterpart_type"]
        _check_tensor_shape("actor_factual", actor_factual, ndim=3, last_dim=6)
        _check_tensor_shape("target", target, ndim=3, last_dim=6)

        factual_pair = torch.cat((actor_factual, target), dim=-1)
        h_factual = self._encode_factual(factual_pair)
        maneuver_gate = h_factual.new_zeros((h_factual.shape[0], 1))
        fused_parts = [h_factual]

        if self.model_variant not in _FACTUAL_ONLY_VARIANTS:
            actor_counterfactual = batch["actor_counterfactual"]
            _check_tensor_shape(
                "actor_counterfactual",
                actor_counterfactual,
                ndim=3,
                last_dim=6,
            )
            counterfactual_pair = torch.cat((actor_counterfactual, target), dim=-1)
            if self.model_variant in _SEPARATE_ENCODER_VARIANTS:
                h_counterfactual = self.counterfactual_trajectory_encoder(counterfactual_pair)
            else:
                h_counterfactual = self.shared_trajectory_encoder(counterfactual_pair)

            if self.model_variant == "a1_direct_counterfactual":
                fused_parts.append(h_counterfactual)
            else:
                if self.model_variant == "without_maneuver_sequence":
                    maneuver_global = batch["maneuver_global"]
                    _check_tensor_shape(
                        "maneuver_global", maneuver_global, ndim=2, last_dim=4
                    )
                    # Keep the full event-comparison architecture and only
                    # replace the detailed sequence by its global presence flag.
                    maneuver_sequence = actor_factual.new_zeros(
                        (actor_factual.shape[0], actor_factual.shape[1], 5)
                    )
                    maneuver_sequence[..., 0] = maneuver_global[:, 0:1]
                    h_event, maneuver_gate = self.event_comparison_network(
                        h_factual,
                        h_counterfactual,
                        maneuver_sequence,
                    )
                else:
                    h_event, maneuver_gate = self.event_comparison_network(
                        h_factual,
                        h_counterfactual,
                        batch["maneuver_sequence"],
                    )
                fused_parts.append(h_event)

        if self.model_variant in _ANALYTIC_VARIANTS:
            analytic_factual = batch["analytic_factual"]
            analytic_difference = batch["analytic_difference"]
            _check_tensor_shape(
                "analytic_factual",
                analytic_factual,
                ndim=3,
                last_dim=ANALYTIC_SEQUENCE_DIM,
            )
            _check_tensor_shape(
                "analytic_difference",
                analytic_difference,
                ndim=3,
                last_dim=ANALYTIC_SEQUENCE_DIM,
            )
            analytic_sequence = torch.cat((analytic_factual, analytic_difference), dim=-1)
            fused_parts.append(self.feature_sequence_encoder(analytic_sequence))

        if self.model_variant in _STATIC_VARIANTS:
            maneuver_global = batch["maneuver_global"]
            _check_tensor_shape("maneuver_global", maneuver_global, ndim=2, last_dim=4)
            if self.model_variant == "without_orbit_error_summary":
                static_features = torch.cat(
                    (maneuver_global[:, 2:4], counterpart_type),
                    dim=-1,
                )
            else:
                error_features = batch["error_features"]
                error_mask = batch["error_mask"]
                _check_tensor_shape("error_features", error_features, ndim=2, last_dim=3)
                _check_tensor_shape("error_mask", error_mask, ndim=2, last_dim=3)
                static_features = torch.cat(
                    (
                        maneuver_global[:, 2:4],
                        error_features,
                        error_mask,
                        counterpart_type,
                    ),
                    dim=-1,
                )
            fused_parts.append(self.static_feature_network(static_features))

        logits = self.feature_fusion_network(torch.cat(fused_parts, dim=-1))
        class_mask = self._build_class_mask(counterpart_type)
        if self.class_mask_applied:
            masked_logits = logits.masked_fill(
                ~class_mask,
                torch.finfo(logits.dtype).min,
            )
        else:
            masked_logits = logits
        probabilities = torch.softmax(masked_logits, dim=-1)
        return {
            "logits": logits,
            "masked_logits": masked_logits,
            "probabilities": probabilities,
            "class_mask": class_mask,
            "maneuver_gate": maneuver_gate,
        }


def build_phase3_model(
    model_variant: str,
    **model_kwargs: Any,
) -> Phase3IntentionRecognitionModel:
    """统一构造 Phase 3 对比模型和结构消融模型。

    三个 raw-input backbone 的默认规模在冻结前按 B2 factual-only TCN
    （267,016 参数）做近似匹配，避免把明显的容量差异误当作结构增益。
    """
    if model_variant == "bilstm_attention":
        model_kwargs.setdefault("trajectory_hidden_dim", 256)
    elif model_variant == "lightweight_transformer":
        model_kwargs.setdefault("transformer_feedforward_dim", 128)
    elif model_variant == "a6_separate_encoders_parameter_matched":
        # Capacity-matched companion for A6.  It complements (rather than
        # replaces) the same-width control because reducing width is itself a
        # controlled sensitivity change.
        model_kwargs.setdefault("trajectory_hidden_dim", 92)
    return Phase3IntentionRecognitionModel(
        model_variant=model_variant,
        **model_kwargs,
    )


__all__ = [
    "SUPPORTED_PHASE3_MODEL_VARIANTS",
    "BiLSTMAttentionEncoder",
    "InceptionTimeEncoder",
    "LightweightTransformerEncoder",
    "Phase3IntentionRecognitionModel",
    "build_phase3_model",
]
