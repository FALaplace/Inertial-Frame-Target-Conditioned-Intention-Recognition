"""事件条件下事实-反事实轨迹比较的意图识别模型。

本文件只负责网络结构，不负责读取 JSON、划分数据集或执行训练。
``data_processing.py`` 先生成单样本 NumPy 数组，后续 Dataset/DataLoader
需要把多个样本堆叠为以下带 batch 维度的 PyTorch Tensor：

    actor_factual          [B, N, 6]
    actor_counterfactual   [B, N, 6]
    target                 [B, N, 6]
    maneuver_sequence      [B, N, 5]
    analytic_factual       [B, N, 12]
    analytic_difference    [B, N, 12]
    maneuver_global        [B, 4]
    error_features         [B, 3]
    error_mask             [B, 3]
    counterpart_type       [B, 2]

当前首版采用 TCN 作为时序编码器，并明确分成五个可训练模块：

1. SharedTrajectoryEncoder：共享轨迹编码器；
2. EventConditionedComparisonNetwork：事件条件比较网络；
3. FeatureSequenceEncoder：解析特征序列编码器；
4. StaticFeatureNetwork：静态特征提取网络；
5. FeatureFusionNetwork：特征融合与分类网络。

为避免重复输入，首版不使用 ``analytic_global``、``analytic_counterfactual``、
``counterfactual_valid`` 和输入数据中预先保存的 ``class_mask``：

- 解析层使用事实特征与事实-反事实差分，共 24 维；
- 反事实有效门控 M 由 maneuver_sequence 的 maneuver_flag 推导；
- 类别约束由 counterpart_type 在模型输出端确定；
- matching_score 只用于候选筛选，从不进入该模型。

连续物理量应在后续 Dataset 中使用“仅由训练集计算”的统计量归一化。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


NUM_CLASSES = 8
SUPPORTED_MODEL_VARIANTS = (
    "full",
    "factual_only",
    "full_without_analytic",
)
TRAJECTORY_PAIR_DIM = 12
MANEUVER_SEQUENCE_DIM = 5
ANALYTIC_SEQUENCE_DIM = 12
ANALYTIC_INPUT_DIM = ANALYTIC_SEQUENCE_DIM * 2

# 静态输入由以下字段拼接：
# delta_v_sigma、maneuver_confidence、3 个误差值、3 个误差有效标志、2 维目标类型。
STATIC_INPUT_DIM = 2 + 3 + 3 + 2


def _check_tensor_shape(name: str, value: Tensor, ndim: int, last_dim: int) -> None:
    """给模型入口提供简短、可读的形状错误信息。"""
    if value.ndim != ndim or value.shape[-1] != last_dim:
        raise ValueError(
            f"{name} must have {ndim} dimensions and last dimension "
            f"{last_dim}; got {tuple(value.shape)}."
        )


class _TemporalResidualBlock(nn.Module):
    """TCN 内部残差块，不属于上面列出的五个概念子网络。"""

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class _TemporalConvEncoder(nn.Module):
    """将 [B, N, C] 时序压缩为一个 [B, H] 表征。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.input_projection = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            *(
                _TemporalResidualBlock(hidden_dim, dilation, dropout)
                for dilation in dilations
            )
        )
        self.pool_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )

    def forward(self, sequence: Tensor) -> Tensor:
        _check_tensor_shape("sequence", sequence, ndim=3, last_dim=self.input_dim)

        # Conv1d 使用 [B, C, N]，所以需要交换时间维和特征维。
        x = sequence.transpose(1, 2)
        x = self.blocks(self.input_projection(x))

        # mean 描述全时段平均行为，max 保留短时接近或机动等峰值证据。
        pooled = torch.cat((x.mean(dim=-1), x.amax(dim=-1)), dim=-1)
        return self.pool_projection(pooled)


class SharedTrajectoryEncoder(nn.Module):
    """子网络1：共享轨迹编码器。

    输入是一条 actor-target 轨迹对：

        factual_pair       = concat(actor_factual, target)        [B, N, 12]
        counterfactual_pair= concat(actor_counterfactual, target) [B, N, 12]

    总模型会用同一个实例分别编码两条轨迹，因此两次计算严格共享权重。
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = _TemporalConvEncoder(
            input_dim=TRAJECTORY_PAIR_DIM,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, trajectory_pair: Tensor) -> Tensor:
        return self.encoder(trajectory_pair)


class EventConditionedComparisonNetwork(nn.Module):
    """子网络2：事件条件比较网络。

    该网络完成三件事：

    1. 编码机动特征序列，得到 h_m；
    2. 从 maneuver_flag 推导 M，而不是读取独立 counterfactual_valid；
    3. 计算 M * (h_factual - h_counterfactual)，再与 h_m 融合。

    无机动样本的 M=0，最终 h_event 被严格置零。事实轨迹表征 h_factual
    仍会直接进入特征融合网络，因此无机动样本不会丢失主要轨迹信息。
    """

    def __init__(
        self,
        trajectory_dim: int = 128,
        maneuver_hidden_dim: int = 32,
        output_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.trajectory_dim = trajectory_dim
        self.maneuver_encoder = _TemporalConvEncoder(
            input_dim=MANEUVER_SEQUENCE_DIM,
            hidden_dim=maneuver_hidden_dim,
            dropout=dropout,
            dilations=(1, 2, 4),
        )
        self.comparison = nn.Sequential(
            nn.Linear(trajectory_dim + maneuver_hidden_dim + 1, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
        )

    def forward(
        self,
        h_factual: Tensor,
        h_counterfactual: Tensor,
        maneuver_sequence: Tensor,
    ) -> tuple[Tensor, Tensor]:
        _check_tensor_shape(
            "h_factual", h_factual, ndim=2, last_dim=self.trajectory_dim
        )
        _check_tensor_shape(
            "h_counterfactual",
            h_counterfactual,
            ndim=2,
            last_dim=self.trajectory_dim,
        )
        _check_tensor_shape(
            "maneuver_sequence",
            maneuver_sequence,
            ndim=3,
            last_dim=MANEUVER_SEQUENCE_DIM,
        )

        # maneuver_sequence[..., 0] 是 maneuver_flag。
        maneuver_gate = maneuver_sequence[..., 0].amax(dim=1, keepdim=True)
        maneuver_gate = maneuver_gate.clamp(min=0.0, max=1.0)

        h_maneuver = self.maneuver_encoder(maneuver_sequence)
        h_difference = maneuver_gate * (h_factual - h_counterfactual)

        comparison_input = torch.cat(
            (
                h_difference,
                maneuver_gate * h_maneuver,
                maneuver_gate,
            ),
            dim=-1,
        )
        h_event = maneuver_gate * self.comparison(comparison_input)
        return h_event, maneuver_gate


class FeatureSequenceEncoder(nn.Module):
    """子网络3：解析层特征序列编码器。

    输入由两部分组成：

        analytic_factual    [B, N, 12]  当前事实轨迹的物理解析证据
        analytic_difference [B, N, 12]  事实减反事实的解析证据变化

    二者拼接为 [B, N, 24]。不再额外输入 analytic_counterfactual，
    因为它已经能由 factual 和 difference 唯一确定。
    """

    def __init__(self, hidden_dim: int = 96, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = _TemporalConvEncoder(
            input_dim=ANALYTIC_INPUT_DIM,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, analytic_sequence: Tensor) -> Tensor:
        return self.encoder(analytic_sequence)


class StaticFeatureNetwork(nn.Module):
    """子网络4：静态特征提取网络。

    首版只保留无法从时序输入直接重复推导的信息：

    - delta_v_sigma 和 maneuver_confidence；
    - 轨道位置、轨道速度、机动时刻三类误差；
    - 对应的 error_mask，用于区分“误差为0”和“误差缺失”；
    - counterpart_type，即 satellite / ground_region。
    """

    def __init__(
        self,
        input_dim: int = STATIC_INPUT_DIM,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, static_features: Tensor) -> Tensor:
        _check_tensor_shape(
            "static_features", static_features, ndim=2, last_dim=self.input_dim
        )
        return self.network(static_features)


class FeatureFusionNetwork(nn.Module):
    """子网络5：特征融合与分类网络。

    拼接 h_factual、h_event、h_analytic 和 h_static 后输出 C0-C7 的
    8 维原始 logits。类别约束在总模型中紧接着执行，不属于可训练网络。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_CLASSES),
        )

    def forward(self, fused_features: Tensor) -> Tensor:
        _check_tensor_shape(
            "fused_features", fused_features, ndim=2, last_dim=self.input_dim
        )
        return self.network(fused_features)


class IntentionRecognitionModel(nn.Module):
    """按实验变体组合子网络，并输出受类别约束的意图概率分布。"""

    def __init__(
        self,
        trajectory_hidden_dim: int = 128,
        maneuver_hidden_dim: int = 32,
        event_hidden_dim: int = 64,
        analytic_hidden_dim: int = 96,
        static_hidden_dim: int = 32,
        fusion_hidden_dim: int = 128,
        dropout: float = 0.1,
        model_variant: str = "full",
    ) -> None:
        super().__init__()
        if model_variant not in SUPPORTED_MODEL_VARIANTS:
            raise ValueError(
                f"Unsupported model_variant={model_variant!r}; expected one of "
                f"{SUPPORTED_MODEL_VARIANTS}."
            )
        self.model_variant = model_variant
        self.model_kwargs = {
            "trajectory_hidden_dim": trajectory_hidden_dim,
            "maneuver_hidden_dim": maneuver_hidden_dim,
            "event_hidden_dim": event_hidden_dim,
            "analytic_hidden_dim": analytic_hidden_dim,
            "static_hidden_dim": static_hidden_dim,
            "fusion_hidden_dim": fusion_hidden_dim,
            "dropout": dropout,
        }
        if model_variant == "factual_only":
            self.model_input_keys = (
                "actor_factual",
                "target",
                "counterpart_type",
            )
        elif model_variant == "full_without_analytic":
            self.model_input_keys = (
                "actor_factual",
                "actor_counterfactual",
                "target",
                "maneuver_sequence",
                "maneuver_global",
                "error_features",
                "error_mask",
                "counterpart_type",
            )
        else:
            self.model_input_keys = (
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

        self.shared_trajectory_encoder = SharedTrajectoryEncoder(
            hidden_dim=trajectory_hidden_dim,
            dropout=dropout,
        )
        fusion_input_dim = trajectory_hidden_dim

        if model_variant != "factual_only":
            self.event_comparison_network = EventConditionedComparisonNetwork(
                trajectory_dim=trajectory_hidden_dim,
                maneuver_hidden_dim=maneuver_hidden_dim,
                output_dim=event_hidden_dim,
                dropout=dropout,
            )
            fusion_input_dim += event_hidden_dim

        if model_variant == "full":
            self.feature_sequence_encoder = FeatureSequenceEncoder(
                hidden_dim=analytic_hidden_dim,
                dropout=dropout,
            )
            # 保持 full 变体原有的融合维度和 state_dict 参数键。
            fusion_input_dim += analytic_hidden_dim

        if model_variant != "factual_only":
            self.static_feature_network = StaticFeatureNetwork(
                hidden_dim=static_hidden_dim,
                dropout=dropout,
            )
            fusion_input_dim += static_hidden_dim

        self.feature_fusion_network = FeatureFusionNetwork(
            input_dim=fusion_input_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=dropout,
        )

        # counterpart_type 顺序为 [satellite, ground_region]。
        # satellite 可用 C0-C5；ground_region 可用 C0、C1、C5、C6、C7。
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

    def _build_class_mask(self, counterpart_type: Tensor) -> Tensor:
        """根据 TargetType 生成确定性类别约束，不让网络学习该规则。"""
        _check_tensor_shape(
            "counterpart_type", counterpart_type, ndim=2, last_dim=2
        )
        counterpart_index = counterpart_type.argmax(dim=-1)
        return self.counterpart_class_masks[counterpart_index]

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """执行图中从五类输入到 C0-C7 概率分布的完整前向计算。

        返回：
        - logits：约束前的 8 维分类输出；
        - masked_logits：不适用类别已经被屏蔽的输出，训练损失使用它；
        - probabilities：经过类别约束和 softmax 的意图概率分布；
        - class_mask：本批样本实际使用的类别约束；
        - maneuver_gate：由 maneuver_flag 推导的 M，便于检查门控是否正确。
        """
        actor_factual = batch["actor_factual"]
        target = batch["target"]
        counterpart_type = batch["counterpart_type"]

        _check_tensor_shape("actor_factual", actor_factual, ndim=3, last_dim=6)
        _check_tensor_shape("target", target, ndim=3, last_dim=6)

        # 所有变体都只从事实 actor-target 轨迹对建立主表征。
        factual_pair = torch.cat((actor_factual, target), dim=-1)
        h_factual = self.shared_trajectory_encoder(factual_pair)

        if self.model_variant == "factual_only":
            # 该基线严格不读取机动、反事实、解析或误差输入。
            maneuver_gate = h_factual.new_zeros((h_factual.shape[0], 1))
            fused_features = h_factual
        else:
            # 事件与静态分支由 full 和 full_without_analytic 共同保留。
            actor_counterfactual = batch["actor_counterfactual"]
            maneuver_sequence = batch["maneuver_sequence"]
            maneuver_global = batch["maneuver_global"]
            error_features = batch["error_features"]
            error_mask = batch["error_mask"]

            _check_tensor_shape(
                "actor_counterfactual",
                actor_counterfactual,
                ndim=3,
                last_dim=6,
            )
            _check_tensor_shape(
                "maneuver_global", maneuver_global, ndim=2, last_dim=4
            )
            _check_tensor_shape(
                "error_features", error_features, ndim=2, last_dim=3
            )
            _check_tensor_shape("error_mask", error_mask, ndim=2, last_dim=3)

            counterfactual_pair = torch.cat(
                (actor_counterfactual, target), dim=-1
            )
            h_counterfactual = self.shared_trajectory_encoder(
                counterfactual_pair
            )
            h_event, maneuver_gate = self.event_comparison_network(
                h_factual,
                h_counterfactual,
                maneuver_sequence,
            )

            if self.model_variant == "full":
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
                analytic_sequence = torch.cat(
                    (analytic_factual, analytic_difference), dim=-1
                )
                h_analytic = self.feature_sequence_encoder(analytic_sequence)

            # maneuver_global 的顺序是：
            # [has_maneuver, delta_v_norm, delta_v_sigma, maneuver_confidence]。
            # full 保持原有的 event -> analytic -> static 调用顺序。
            static_features = torch.cat(
                (
                    maneuver_global[:, 2:4],
                    error_features,
                    error_mask,
                    counterpart_type,
                ),
                dim=-1,
            )
            h_static = self.static_feature_network(static_features)

            if self.model_variant == "full":
                fused_features = torch.cat(
                    (h_factual, h_event, h_analytic, h_static), dim=-1
                )
            else:
                fused_features = torch.cat(
                    (h_factual, h_event, h_static), dim=-1
                )

        logits = self.feature_fusion_network(fused_features)

        # 约束函数：先屏蔽目标类型不适用的类别，再执行 softmax。
        class_mask = self._build_class_mask(counterpart_type)
        masked_logits = logits.masked_fill(
            ~class_mask,
            torch.finfo(logits.dtype).min,
        )
        probabilities = torch.softmax(masked_logits, dim=-1)

        return {
            "logits": logits,
            "masked_logits": masked_logits,
            "probabilities": probabilities,
            "class_mask": class_mask,
            "maneuver_gate": maneuver_gate,
        }


def build_intention_model(
    model_variant: str = "full",
    **model_kwargs: Any,
) -> IntentionRecognitionModel:
    """统一构造 Phase 2 的完整模型和两个消融变体。"""
    return IntentionRecognitionModel(
        model_variant=model_variant,
        **model_kwargs,
    )


def soft_label_cross_entropy(masked_logits: Tensor, targets: Tensor) -> Tensor:
    """C0-C7 软伪标签使用的交叉熵损失。

    ``targets`` 是每行和为 1 的软概率分布，而不是单个整数类别编号。
    数据范围和类别约束已经由 ``data_processing.py`` 检查，这里只保留
    训练所需的最小计算。
    """
    if masked_logits.shape != targets.shape:
        raise ValueError(
            "masked_logits and targets must have the same shape; "
            f"got {tuple(masked_logits.shape)} and {tuple(targets.shape)}."
        )
    return -(targets * F.log_softmax(masked_logits, dim=-1)).sum(dim=-1).mean()


__all__ = [
    "SUPPORTED_MODEL_VARIANTS",
    "SharedTrajectoryEncoder",
    "EventConditionedComparisonNetwork",
    "FeatureSequenceEncoder",
    "StaticFeatureNetwork",
    "FeatureFusionNetwork",
    "IntentionRecognitionModel",
    "build_intention_model",
    "soft_label_cross_entropy",
]
