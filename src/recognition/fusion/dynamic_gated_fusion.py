#!/usr/bin/env python3
"""
创新点3: 动态门控跨模态融合 (Dynamic Gated Cross-Modal Fusion, DGCMF)

理论依据:
- 不同模态（姿态、外观、运动）在不同场景下可靠性不同
  - 遮挡时: 外观特征不可靠
  - 静止时: 运动特征不可靠
  - 侧脸时: 姿态特征可能有噪声
- 传统简单拼接或平均融合忽略了模态质量差异
- DGCMF通过质量感知门控动态调整模态权重

技术方案:
1. 质量评估网络: 预测每个模态的可靠性分数
2. 跨模态注意力: 建模模态间交互关系
3. 残差门控: 保留原始单模态信息，防止信息丢失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class ModalityQualityEstimator(nn.Module):
    """
    模态质量评估器

    为每个模态特征预测可靠性分数 [0, 1]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, dim] 或 [batch, seq_len, dim]

        Returns:
            [batch, 1] 或 [batch, seq_len, 1] 质量分数
        """
        return self.net(x)


class CrossModalAttention(nn.Module):
    """
    跨模态注意力模块

    使用一个模态作为Query，另一个模态作为Key/Value
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        quality_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, dim] 查询模态
            key_value: [batch, dim] 键值模态
            quality_mask: [batch, 1] 键值模态的质量掩码

        Returns:
            [batch, dim] 增强后的查询表示
        """
        batch_size = query.shape[0]

        # 扩展维度用于注意力计算
        q = query.unsqueeze(1)  # [batch, 1, dim]
        kv = key_value.unsqueeze(1)  # [batch, 1, dim]

        # 投影
        q = self.q_proj(q).view(batch_size, 1, self.num_heads, self.head_dim)
        k = self.k_proj(kv).view(batch_size, 1, self.num_heads, self.head_dim)
        v = self.v_proj(kv).view(batch_size, 1, self.num_heads, self.head_dim)

        # [batch, num_heads, 1, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch, heads, 1, 1]

        # 如果有质量掩码，调整注意力
        if quality_mask is not None:
            # 低质量时降低注意力权重
            attn = attn * quality_mask.view(batch_size, 1, 1, 1)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 输出
        out = torch.matmul(attn, v)  # [batch, heads, 1, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, 1, -1)
        out = self.out_proj(out).squeeze(1)  # [batch, dim]

        return self.norm(query + out)


class GatedFusionUnit(nn.Module):
    """
    门控融合单元

    动态融合两个特征，使用门控机制控制信息流
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

        self.transform = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor,
        weight1: Optional[torch.Tensor] = None,
        weight2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            feat1: [batch, dim] 第一个特征
            feat2: [batch, dim] 第二个特征
            weight1: [batch, 1] 第一个特征的权重
            weight2: [batch, 1] 第二个特征的权重

        Returns:
            [batch, dim] 融合后的特征
        """
        # 应用质量权重
        if weight1 is not None:
            feat1 = feat1 * weight1
        if weight2 is not None:
            feat2 = feat2 * weight2

        # 拼接
        combined = torch.cat([feat1, feat2], dim=-1)

        # 计算门控
        gate = self.gate(combined)

        # 变换并门控融合
        transformed = self.transform(combined)
        output = gate * transformed + (1 - gate) * (feat1 + feat2) / 2

        return output


class DynamicGatedCrossModalFusion(nn.Module):
    """
    动态门控跨模态融合 (DGCMF)

    融合姿态、外观、运动三种模态特征

    Args:
        pose_dim: 姿态特征维度
        appearance_dim: 外观特征维度
        motion_dim: 运动特征维度
        hidden_dim: 隐藏层维度 (统一投影维度)
        num_heads: 跨模态注意力头数
        dropout: Dropout比例
        use_quality_estimation: 是否使用质量估计
    """

    def __init__(
        self,
        pose_dim: int = 128,
        appearance_dim: int = 512,
        motion_dim: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_quality_estimation: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_quality_estimation = use_quality_estimation

        # ========== 1. 模态投影层 ==========
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.appearance_proj = nn.Sequential(
            nn.Linear(appearance_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.motion_proj = nn.Sequential(
            nn.Linear(motion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ========== 2. 质量评估网络 ==========
        if use_quality_estimation:
            self.pose_quality = ModalityQualityEstimator(hidden_dim, dropout=dropout)
            self.appearance_quality = ModalityQualityEstimator(hidden_dim, dropout=dropout)
            self.motion_quality = ModalityQualityEstimator(hidden_dim, dropout=dropout)

        # ========== 3. 跨模态注意力 ==========
        # 姿态 <- 外观
        self.pose_attend_appearance = CrossModalAttention(hidden_dim, num_heads, dropout)
        # 姿态 <- 运动
        self.pose_attend_motion = CrossModalAttention(hidden_dim, num_heads, dropout)
        # 外观 <- 姿态
        self.appearance_attend_pose = CrossModalAttention(hidden_dim, num_heads, dropout)
        # 运动 <- 姿态
        self.motion_attend_pose = CrossModalAttention(hidden_dim, num_heads, dropout)

        # ========== 4. 门控融合单元 ==========
        # 先融合姿态和外观
        self.fuse_pose_appearance = GatedFusionUnit(hidden_dim, dropout)
        # 再融合上一步结果和运动
        self.fuse_with_motion = GatedFusionUnit(hidden_dim, dropout)

        # ========== 5. 残差门控 ==========
        self.residual_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        # ========== 6. 最终输出层 ==========
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        pose_feat: torch.Tensor,
        appearance_feat: Optional[torch.Tensor] = None,
        motion_feat: Optional[torch.Tensor] = None,
        return_quality_scores: bool = False,
    ):
        """
        前向传播

        Args:
            pose_feat: [batch, pose_dim] 姿态特征 (必需)
            appearance_feat: [batch, appearance_dim] 外观特征 (可选)
            motion_feat: [batch, motion_dim] 运动特征 (可选)
            return_quality_scores: 是否返回质量分数

        Returns:
            如果 return_quality_scores=False:
                [batch, hidden_dim] 融合特征
            如果 return_quality_scores=True:
                (fused_feat, quality_dict)
        """
        batch_size = pose_feat.shape[0]
        device = pose_feat.device
        dtype = pose_feat.dtype

        # 1. 投影到统一维度
        pose = self.pose_proj(pose_feat)

        has_appearance = appearance_feat is not None
        has_motion = motion_feat is not None

        if has_appearance:
            appearance = self.appearance_proj(appearance_feat)
        else:
            appearance = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

        if has_motion:
            motion = self.motion_proj(motion_feat)
        else:
            motion = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

        # 2. 计算质量分数
        quality_scores = {}
        if self.use_quality_estimation:
            pose_q = self.pose_quality(pose)
            quality_scores['pose'] = pose_q

            if has_appearance:
                appearance_q = self.appearance_quality(appearance)
                quality_scores['appearance'] = appearance_q
            else:
                appearance_q = torch.zeros(batch_size, 1, device=device, dtype=dtype)
                quality_scores['appearance'] = appearance_q

            if has_motion:
                motion_q = self.motion_quality(motion)
                quality_scores['motion'] = motion_q
            else:
                motion_q = torch.zeros(batch_size, 1, device=device, dtype=dtype)
                quality_scores['motion'] = motion_q
        else:
            pose_q = torch.ones(batch_size, 1, device=device, dtype=dtype)
            appearance_q = torch.ones(batch_size, 1, device=device, dtype=dtype) if has_appearance else torch.zeros(batch_size, 1, device=device, dtype=dtype)
            motion_q = torch.ones(batch_size, 1, device=device, dtype=dtype) if has_motion else torch.zeros(batch_size, 1, device=device, dtype=dtype)

        # 保存原始姿态特征用于残差
        pose_original = pose.clone()

        # 3. 跨模态注意力
        if has_appearance:
            pose = self.pose_attend_appearance(pose, appearance, appearance_q)
            appearance = self.appearance_attend_pose(appearance, pose_original, pose_q)

        if has_motion:
            pose = self.pose_attend_motion(pose, motion, motion_q)
            motion = self.motion_attend_pose(motion, pose_original, pose_q)

        # 4. 门控融合
        if has_appearance:
            fused = self.fuse_pose_appearance(pose, appearance, pose_q, appearance_q)
        else:
            fused = pose

        if has_motion:
            fused = self.fuse_with_motion(fused, motion, None, motion_q)

        # 5. 残差门控
        residual_input = torch.cat([fused, pose_original], dim=-1)
        gate = self.residual_gate(residual_input)
        fused = gate * fused + (1 - gate) * pose_original

        # 6. 最终输出
        output = self.output_proj(fused)

        if return_quality_scores:
            return output, quality_scores
        return output

    def get_modality_weights(
        self,
        pose_feat: torch.Tensor,
        appearance_feat: Optional[torch.Tensor] = None,
        motion_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        获取各模态的权重分布（用于可视化分析）

        Returns:
            {'pose': [batch, 1], 'appearance': [batch, 1], 'motion': [batch, 1]}
        """
        _, quality_scores = self.forward(
            pose_feat, appearance_feat, motion_feat,
            return_quality_scores=True
        )

        # 归一化为权重
        total = sum(v for v in quality_scores.values())
        weights = {k: v / (total + 1e-8) for k, v in quality_scores.items()}

        return weights


# 别名
DGCMF = DynamicGatedCrossModalFusion


class SequenceDGCMF(nn.Module):
    """
    序列级DGCMF

    处理时序特征序列，对每个时间步进行跨模态融合
    """

    def __init__(
        self,
        pose_dim: int = 128,
        appearance_dim: int = 512,
        motion_dim: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dgcmf = DGCMF(
            pose_dim=pose_dim,
            appearance_dim=appearance_dim,
            motion_dim=motion_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        pose_seq: torch.Tensor,
        appearance_seq: Optional[torch.Tensor] = None,
        motion_seq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pose_seq: [batch, seq_len, pose_dim]
            appearance_seq: [batch, seq_len, appearance_dim] (可选)
            motion_seq: [batch, seq_len, motion_dim] (可选)

        Returns:
            [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = pose_seq.shape
        device = pose_seq.device

        outputs = []

        for t in range(seq_len):
            pose_t = pose_seq[:, t, :]
            appearance_t = appearance_seq[:, t, :] if appearance_seq is not None else None
            motion_t = motion_seq[:, t, :] if motion_seq is not None else None

            fused_t = self.dgcmf(pose_t, appearance_t, motion_t)
            outputs.append(fused_t)

        return torch.stack(outputs, dim=1)


if __name__ == '__main__':
    # 测试DGCMF
    print("Testing Dynamic Gated Cross-Modal Fusion (DGCMF)...")

    batch_size = 4
    pose_dim = 128
    appearance_dim = 512
    motion_dim = 64
    hidden_dim = 128

    # 创建DGCMF模块
    dgcmf = DGCMF(
        pose_dim=pose_dim,
        appearance_dim=appearance_dim,
        motion_dim=motion_dim,
        hidden_dim=hidden_dim,
        use_quality_estimation=True,
    )

    # 测试输入
    pose_feat = torch.randn(batch_size, pose_dim)
    appearance_feat = torch.randn(batch_size, appearance_dim)
    motion_feat = torch.randn(batch_size, motion_dim)

    # 测试完整三模态融合
    print("\n1. Full three-modality fusion:")
    output = dgcmf(pose_feat, appearance_feat, motion_feat)
    print(f"   Pose: {pose_feat.shape}")
    print(f"   Appearance: {appearance_feat.shape}")
    print(f"   Motion: {motion_feat.shape}")
    print(f"   Output: {output.shape}")

    # 测试带质量分数的输出
    output, quality = dgcmf(pose_feat, appearance_feat, motion_feat, return_quality_scores=True)
    print(f"   Quality scores: pose={quality['pose'].mean():.3f}, "
          f"appearance={quality['appearance'].mean():.3f}, "
          f"motion={quality['motion'].mean():.3f}")

    # 测试仅姿态
    print("\n2. Pose only:")
    output = dgcmf(pose_feat, None, None)
    print(f"   Output: {output.shape}")

    # 测试姿态+外观
    print("\n3. Pose + Appearance:")
    output = dgcmf(pose_feat, appearance_feat, None)
    print(f"   Output: {output.shape}")

    # 测试姿态+运动
    print("\n4. Pose + Motion:")
    output = dgcmf(pose_feat, None, motion_feat)
    print(f"   Output: {output.shape}")

    # 测试序列级DGCMF
    print("\n5. Sequence DGCMF:")
    seq_dgcmf = SequenceDGCMF(pose_dim, appearance_dim, motion_dim, hidden_dim)
    seq_len = 32
    pose_seq = torch.randn(batch_size, seq_len, pose_dim)
    appearance_seq = torch.randn(batch_size, seq_len, appearance_dim)
    motion_seq = torch.randn(batch_size, seq_len, motion_dim)
    output = seq_dgcmf(pose_seq, appearance_seq, motion_seq)
    print(f"   Sequence output: {output.shape}")

    # 测试模态权重可视化
    print("\n6. Modality weights:")
    weights = dgcmf.get_modality_weights(pose_feat, appearance_feat, motion_feat)
    print(f"   Weights: pose={weights['pose'].mean():.3f}, "
          f"appearance={weights['appearance'].mean():.3f}, "
          f"motion={weights['motion'].mean():.3f}")

    print("\nAll DGCMF tests passed!")
