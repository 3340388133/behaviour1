"""
Multi-Modal Fusion and End-to-End Classification

Combines features from different modalities and stages
for final suspicious gaze behavior classification.

创新点：
1. 多模态特征融合（姿态 + 追踪 + 外观）
2. 注意力加权的特征聚合
3. 端到端可微分的完整pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple


class MultiModalFusion(nn.Module):
    """
    Multi-Modal Feature Fusion Module

    创新点：融合来自不同来源的特征：
    - 姿态特征：头部角度变化
    - 追踪特征：位置、速度、Re-ID
    - 时序特征：LSTM/GRU输出

    Args:
        pose_dim: Dimension of pose features
        track_dim: Dimension of tracking features
        temporal_dim: Dimension of temporal features
        hidden_dim: Hidden dimension
        fusion_type: Fusion method ('concat', 'attention', 'bilinear')
    """

    def __init__(
        self,
        pose_dim: int = 256,
        track_dim: int = 128,
        temporal_dim: int = 512,
        hidden_dim: int = 256,
        fusion_type: str = "attention",
    ):
        super().__init__()

        self.fusion_type = fusion_type

        # Feature projections to common dimension
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
        )

        self.track_proj = nn.Sequential(
            nn.Linear(track_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
        )

        self.temporal_proj = nn.Sequential(
            nn.Linear(temporal_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
        )

        # Fusion mechanism
        if fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(hidden_dim),
            )
        elif fusion_type == "attention":
            self.attention = ModalityAttention(hidden_dim, num_modalities=3)
            self.fusion = nn.Linear(hidden_dim, hidden_dim)
        elif fusion_type == "bilinear":
            self.bilinear_pt = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
            self.bilinear_final = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
            self.fusion = nn.LayerNorm(hidden_dim)

        self.output_dim = hidden_dim

    def forward(
        self,
        pose_feat: torch.Tensor,
        track_feat: torch.Tensor,
        temporal_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pose_feat: Pose features (B, pose_dim)
            track_feat: Tracking features (B, track_dim)
            temporal_feat: Temporal features (B, temporal_dim)

        Returns:
            Fused features (B, hidden_dim)
        """
        # Project to common dimension
        pose = self.pose_proj(pose_feat)
        track = self.track_proj(track_feat)
        temporal = self.temporal_proj(temporal_feat)

        if self.fusion_type == "concat":
            fused = torch.cat([pose, track, temporal], dim=-1)
            output = self.fusion(fused)

        elif self.fusion_type == "attention":
            # Stack modalities
            stacked = torch.stack([pose, track, temporal], dim=1)  # (B, 3, D)
            output = self.attention(stacked)
            output = self.fusion(output)

        elif self.fusion_type == "bilinear":
            # Hierarchical bilinear fusion
            pt_fused = self.bilinear_pt(pose, track)
            output = self.bilinear_final(pt_fused, temporal)
            output = self.fusion(output)

        return output


class ModalityAttention(nn.Module):
    """
    Cross-modality attention for feature fusion.

    创新点：学习不同模态特征的重要性权重，
    自适应地融合多模态信息。
    """

    def __init__(self, dim: int, num_modalities: int = 3):
        super().__init__()

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.scale = dim ** -0.5

        # Modality-specific biases
        self.modality_bias = nn.Parameter(
            torch.zeros(num_modalities)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Stacked modality features (B, num_modalities, dim)

        Returns:
            Fused features (B, dim)
        """
        B, M, D = x.shape

        # Compute attention
        q = self.query(x.mean(dim=1, keepdim=True))  # (B, 1, D)
        k = self.key(x)
        v = self.value(x)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.modality_bias.view(1, 1, M)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).squeeze(1)

        return out


class SuspiciousGazeClassifier(nn.Module):
    """
    End-to-End Suspicious Gaze Behavior Classifier

    完整的分类pipeline，从原始特征到最终判定。

    创新点：
    1. 多尺度特征提取
    2. 多模态融合
    3. 可解释的注意力输出

    Args:
        pose_encoder: Pose sequence encoder
        temporal_model: Temporal modeling module
        num_classes: Number of classes (2: normal, suspicious)
        hidden_dim: Hidden dimension
    """

    def __init__(
        self,
        pose_dim: int = 6,
        track_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        # Pose feature extraction
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Tracking feature extraction
        self.track_encoder = nn.Sequential(
            nn.Linear(track_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Multi-modal fusion
        self.fusion = MultiModalFusion(
            pose_dim=hidden_dim,
            track_dim=hidden_dim,
            temporal_dim=hidden_dim,
            hidden_dim=hidden_dim,
            fusion_type="attention",
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Auxiliary heads for multi-task learning
        self.aux_pose_head = nn.Linear(hidden_dim, 3)  # Predict pose change
        self.aux_pattern_head = nn.Linear(hidden_dim, 3)  # Gaze pattern type

    def forward(
        self,
        pose_feat: torch.Tensor,
        track_feat: torch.Tensor,
        temporal_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pose_feat: Pose features (B, pose_dim) or (B, T, pose_dim)
            track_feat: Tracking features (B, track_dim)
            temporal_feat: Temporal features (B, temporal_dim)

        Returns:
            Dictionary with predictions and auxiliary outputs
        """
        # Handle sequence input
        if pose_feat.dim() == 3:
            pose_feat = pose_feat.mean(dim=1)

        # Encode features
        pose_encoded = self.pose_encoder(pose_feat)
        track_encoded = self.track_encoder(track_feat)

        # Fusion
        fused = self.fusion(pose_encoded, track_encoded, temporal_feat)

        # Main classification
        logits = self.classifier(fused)

        # Auxiliary predictions
        pose_change = self.aux_pose_head(fused)
        pattern = self.aux_pattern_head(fused)

        return {
            'logits': logits,
            'probs': F.softmax(logits, dim=-1),
            'pose_change': pose_change,
            'pattern': pattern,
            'features': fused,
        }


class BehaviorClassificationLoss(nn.Module):
    """
    Multi-task loss for behavior classification.

    Combines classification loss with auxiliary losses
    for improved feature learning.
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        pose_weight: float = 0.3,
        pattern_weight: float = 0.3,
        label_smoothing: float = 0.1,
    ):
        super().__init__()

        self.cls_weight = cls_weight
        self.pose_weight = pose_weight
        self.pattern_weight = pattern_weight

        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.pose_loss = nn.MSELoss()
        self.pattern_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            output: Model output dictionary
            target: Ground truth dictionary

        Returns:
            Loss dictionary
        """
        # Classification loss
        cls_loss = self.cls_loss(output['logits'], target['label'])

        # Pose change prediction loss
        pose_loss = torch.tensor(0.0, device=cls_loss.device)
        if 'pose_change' in target:
            pose_loss = self.pose_loss(output['pose_change'], target['pose_change'])

        # Pattern classification loss
        pattern_loss = torch.tensor(0.0, device=cls_loss.device)
        if 'pattern' in target:
            pattern_loss = self.pattern_loss(output['pattern'], target['pattern'])

        # Total loss
        total = (
            self.cls_weight * cls_loss +
            self.pose_weight * pose_loss +
            self.pattern_weight * pattern_loss
        )

        return {
            'total': total,
            'cls': cls_loss,
            'pose': pose_loss,
            'pattern': pattern_loss,
        }


class SuspiciousDetector(nn.Module):
    """
    Rule-based post-processing for suspicious behavior detection.

    Combines neural network predictions with heuristic rules
    for more robust detection.

    创新点：结合深度学习和规则的混合方法
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        min_gaze_change: float = 30.0,  # degrees
        min_frequency: int = 3,
        time_window: float = 5.0,  # seconds
    ):
        super().__init__()

        self.confidence_threshold = confidence_threshold
        self.min_gaze_change = min_gaze_change
        self.min_frequency = min_frequency
        self.time_window = time_window

    def forward(
        self,
        nn_probs: torch.Tensor,
        pose_sequence: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            nn_probs: Neural network class probabilities (B, 2)
            pose_sequence: Pose sequence (B, T, 3) - yaw, pitch, roll
            timestamps: Frame timestamps (B, T)

        Returns:
            Final detection results
        """
        B = nn_probs.shape[0]

        # Neural network confidence
        nn_suspicious = nn_probs[:, 1] > self.confidence_threshold

        # Rule-based checks
        rule_suspicious = []
        gaze_counts = []

        for i in range(B):
            poses = pose_sequence[i]  # (T, 3)
            times = timestamps[i]  # (T,)

            # Check for rapid gaze changes
            yaw_diff = torch.abs(poses[1:, 0] - poses[:-1, 0])
            large_changes = yaw_diff > self.min_gaze_change

            # Count changes within time window
            count = 0
            window_end = times[-1]
            window_start = window_end - self.time_window

            for t in range(len(times) - 1):
                if times[t] >= window_start and large_changes[t]:
                    count += 1

            gaze_counts.append(count)
            rule_suspicious.append(count >= self.min_frequency)

        rule_suspicious = torch.tensor(rule_suspicious, device=nn_probs.device)
        gaze_counts = torch.tensor(gaze_counts, device=nn_probs.device)

        # Combine NN and rules
        final_suspicious = nn_suspicious | rule_suspicious

        # Confidence score (combine NN prob and rule score)
        rule_score = (gaze_counts.float() / self.min_frequency).clamp(0, 1)
        final_confidence = 0.7 * nn_probs[:, 1] + 0.3 * rule_score

        return {
            'suspicious': final_suspicious,
            'confidence': final_confidence,
            'nn_suspicious': nn_suspicious,
            'rule_suspicious': rule_suspicious,
            'gaze_change_count': gaze_counts,
        }
