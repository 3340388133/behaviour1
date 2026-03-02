"""
WHENet: Wide-range Head pose Estimation Network

This module implements the baseline WHENet model for full-range
head pose estimation (-180° to 180° yaw angle).

Reference:
    WHENet: Real-time Fine-Grained Estimation for Wide Range Head Pose
    https://github.com/Ascend-Research/HeadPoseEstimation-WHENet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class EfficientNetBackbone(nn.Module):
    """
    EfficientNet backbone for feature extraction.

    Uses torchvision's EfficientNet implementation.
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        out_indices: Tuple[int, ...] = (2, 3, 4),
    ):
        super().__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            base_model = efficientnet_b0(weights=weights)
        else:
            base_model = efficientnet_b0(weights=None)

        # Extract feature stages
        self.features = base_model.features
        self.out_indices = out_indices

        # Channel numbers for EfficientNet-B0 stages
        # Stage 0: 32, Stage 1: 16, Stage 2: 24, Stage 3: 40,
        # Stage 4: 80, Stage 5: 112, Stage 6: 192, Stage 7: 320, Stage 8: 1280
        self.out_channels = [24, 40, 112]  # For indices 2, 3, 5

    def forward(self, x: torch.Tensor) -> list:
        """
        Extract multi-scale features.

        Args:
            x: Input image (B, 3, H, W)

        Returns:
            List of feature maps at specified indices
        """
        outputs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_indices:
                outputs.append(x)
        return outputs


class PoseRegressor(nn.Module):
    """
    Pose regression head for yaw, pitch, roll angles.

    Uses a combination of classification (binned angles) and
    regression (fine-grained offset) for improved accuracy.
    """

    def __init__(
        self,
        in_channels: int,
        num_bins: int = 66,
        angle_range: Tuple[float, float] = (-99, 99),
    ):
        super().__init__()
        self.num_bins = num_bins
        self.angle_range = angle_range

        # Classification head (coarse angle bins)
        self.fc_yaw = nn.Linear(in_channels, num_bins)
        self.fc_pitch = nn.Linear(in_channels, num_bins)
        self.fc_roll = nn.Linear(in_channels, num_bins)

        # Regression head (fine-grained offset)
        self.fc_yaw_reg = nn.Linear(in_channels, 1)
        self.fc_pitch_reg = nn.Linear(in_channels, 1)
        self.fc_roll_reg = nn.Linear(in_channels, 1)

        # Index tensor for soft-argmax
        idx_tensor = torch.arange(num_bins, dtype=torch.float32)
        self.register_buffer('idx_tensor', idx_tensor)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x: Feature vector (B, in_channels)

        Returns:
            Tuple of (yaw, pitch, roll) angles in degrees
            Each tensor has shape (B,)
        """
        # Classification logits
        yaw_cls = self.fc_yaw(x)
        pitch_cls = self.fc_pitch(x)
        roll_cls = self.fc_roll(x)

        # Soft-argmax for differentiable angle extraction
        yaw_soft = F.softmax(yaw_cls, dim=1)
        pitch_soft = F.softmax(pitch_cls, dim=1)
        roll_soft = F.softmax(roll_cls, dim=1)

        # Expected value (soft-argmax)
        yaw_expected = (yaw_soft * self.idx_tensor).sum(dim=1)
        pitch_expected = (pitch_soft * self.idx_tensor).sum(dim=1)
        roll_expected = (roll_soft * self.idx_tensor).sum(dim=1)

        # Convert to angle range
        bin_width = (self.angle_range[1] - self.angle_range[0]) / self.num_bins
        yaw = yaw_expected * bin_width + self.angle_range[0]
        pitch = pitch_expected * bin_width + self.angle_range[0]
        roll = roll_expected * bin_width + self.angle_range[0]

        # Add regression offset for fine-grained estimation
        yaw = yaw + self.fc_yaw_reg(x).squeeze(-1)
        pitch = pitch + self.fc_pitch_reg(x).squeeze(-1)
        roll = roll + self.fc_roll_reg(x).squeeze(-1)

        return yaw, pitch, roll, (yaw_cls, pitch_cls, roll_cls)


class WHENet(nn.Module):
    """
    WHENet: Wide-range Head pose Estimation Network

    This is the baseline model that estimates head pose angles
    (yaw, pitch, roll) from cropped head images.

    Args:
        backbone: Backbone network name
        pretrained: Whether to use pretrained weights
        num_bins: Number of angle bins for classification
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        num_bins: int = 66,
    ):
        super().__init__()

        # Backbone
        self.backbone = EfficientNetBackbone(
            model_name=backbone,
            pretrained=pretrained,
            out_indices=(4,),  # Only use the last stage
        )

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Feature channels from backbone
        in_channels = 112  # EfficientNet-B0 stage 4

        # Pose regressor
        self.regressor = PoseRegressor(
            in_channels=in_channels,
            num_bins=num_bins,
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input head image (B, 3, H, W)
               Recommended size: 224x224

        Returns:
            Tuple of (yaw, pitch, roll) angles in degrees
        """
        # Extract features
        features = self.backbone(x)[-1]

        # Global pooling
        pooled = self.gap(features).flatten(1)

        # Pose regression
        yaw, pitch, roll, cls_logits = self.regressor(pooled)

        return yaw, pitch, roll


class GeodesicLoss(nn.Module):
    """
    Geodesic Loss for rotation estimation.

    This loss measures the angular distance between predicted and
    ground truth rotations, which is more appropriate than MSE
    for angle regression.

    创新点：使用测地距离损失替代MSE，更好地处理角度的周期性
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted (yaw, pitch, roll) in degrees
            target: Ground truth (yaw, pitch, roll) in degrees

        Returns:
            Geodesic loss
        """
        yaw_p, pitch_p, roll_p = pred
        yaw_t, pitch_t, roll_t = target

        # Convert to radians
        yaw_p = yaw_p * math.pi / 180
        pitch_p = pitch_p * math.pi / 180
        roll_p = roll_p * math.pi / 180
        yaw_t = yaw_t * math.pi / 180
        pitch_t = pitch_t * math.pi / 180
        roll_t = roll_t * math.pi / 180

        # Compute rotation matrices
        R_pred = self._euler_to_rotation_matrix(yaw_p, pitch_p, roll_p)
        R_target = self._euler_to_rotation_matrix(yaw_t, pitch_t, roll_t)

        # Geodesic distance
        R_diff = torch.bmm(R_pred, R_target.transpose(1, 2))
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        trace = torch.clamp(trace, -1 + 1e-6, 3 - 1e-6)
        angle = torch.acos((trace - 1) / 2)

        if self.reduction == 'mean':
            return angle.mean()
        elif self.reduction == 'sum':
            return angle.sum()
        return angle

    def _euler_to_rotation_matrix(
        self,
        yaw: torch.Tensor,
        pitch: torch.Tensor,
        roll: torch.Tensor,
    ) -> torch.Tensor:
        """Convert Euler angles to rotation matrix."""
        B = yaw.shape[0]

        # Rotation matrices for each axis
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
        cos_r, sin_r = torch.cos(roll), torch.sin(roll)

        # Combined rotation matrix (Z-Y-X convention)
        zeros = torch.zeros_like(yaw)
        ones = torch.ones_like(yaw)

        R = torch.stack([
            cos_y * cos_p,
            cos_y * sin_p * sin_r - sin_y * cos_r,
            cos_y * sin_p * cos_r + sin_y * sin_r,
            sin_y * cos_p,
            sin_y * sin_p * sin_r + cos_y * cos_r,
            sin_y * sin_p * cos_r - cos_y * sin_r,
            -sin_p,
            cos_p * sin_r,
            cos_p * cos_r,
        ], dim=1).view(B, 3, 3)

        return R
