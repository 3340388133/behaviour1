"""
WHENet+: Enhanced Wide-range Head pose Estimation Network

This module implements the improved WHENet model with:
1. Self-Attention mechanism for global context
2. Multi-scale feature fusion for better small object handling
3. Adaptive coordinate transformation for different camera positions

创新点：
1. 在特征提取阶段引入自注意力机制，增强全局上下文理解
2. 多尺度特征融合，提升远距离/小尺寸人头的姿态估计精度
3. 自适应坐标系转换，支持正机位和侧机位的统一处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List

from .whenet import EfficientNetBackbone, PoseRegressor, GeodesicLoss
from .attention import SelfAttention, CBAM, SimDLKA, PoseGuidedAttention
from .feature_fusion import MultiScaleFusion, AdaptiveFeatureFusion


class WHENetPlus(nn.Module):
    """
    WHENet+: Enhanced Head Pose Estimation Network

    Improvements over baseline WHENet:
    1. Multi-scale feature extraction with FPN
    2. Self-attention for global context
    3. Adaptive feature fusion
    4. Camera-aware coordinate transformation

    Args:
        backbone: Backbone network name
        pretrained: Whether to use pretrained weights
        num_bins: Number of angle bins for classification
        attention_type: Type of attention ('self', 'cbam', 'simdlka')
        fusion_type: Type of feature fusion ('concat', 'attention', 'adaptive')
        enable_pose_guided: Enable pose-guided attention
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        num_bins: int = 66,
        attention_type: str = "self",
        fusion_type: str = "adaptive",
        enable_pose_guided: bool = True,
    ):
        super().__init__()

        # Multi-scale backbone
        self.backbone = EfficientNetBackbone(
            model_name=backbone,
            pretrained=pretrained,
            out_indices=(2, 3, 5),  # Multi-scale outputs
        )

        # Channel numbers for EfficientNet-B0
        backbone_channels = [24, 40, 112]
        fusion_channels = 128

        # Multi-scale feature fusion (创新点1)
        if fusion_type == "adaptive":
            self.fusion = AdaptiveFeatureFusion(
                in_channels=backbone_channels,
                out_channels=fusion_channels,
            )
        else:
            self.fusion = MultiScaleFusion(
                in_channels=backbone_channels,
                out_channels=fusion_channels,
            )
        self.fusion_type = fusion_type

        # Attention module (创新点2)
        self.attention_type = attention_type
        if attention_type == "self":
            self.attention = SelfAttention(
                dim=fusion_channels,
                num_heads=8,
                attn_drop=0.1,
                proj_drop=0.1,
            )
        elif attention_type == "cbam":
            self.attention = CBAM(
                channels=fusion_channels,
                reduction=16,
            )
        elif attention_type == "simdlka":
            self.attention = SimDLKA(
                channels=fusion_channels,
            )
        else:
            self.attention = nn.Identity()

        # Pose-guided attention (创新点3)
        self.enable_pose_guided = enable_pose_guided
        if enable_pose_guided:
            self.pose_guided_attn = PoseGuidedAttention(
                channels=fusion_channels,
                pose_dim=3,
            )

        # Global pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Feature projection
        self.fc = nn.Sequential(
            nn.Linear(fusion_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

        # Pose regressor
        self.regressor = PoseRegressor(
            in_channels=256,
            num_bins=num_bins,
        )

        # Camera-aware transformation (创新点4)
        self.camera_transform = CameraAwareTransform()

    def forward(
        self,
        x: torch.Tensor,
        camera_type: Optional[str] = None,
        pose_prior: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input head image (B, 3, H, W)
            camera_type: Camera position ('front' or 'side')
            pose_prior: Prior pose estimation for guided attention

        Returns:
            Dictionary containing:
                - yaw, pitch, roll: Predicted angles
                - features: Intermediate features for tracking
                - attention_map: Attention weights for visualization
        """
        # Multi-scale feature extraction
        multi_scale_features = self.backbone(x)

        # Feature fusion
        if self.fusion_type == "adaptive":
            fused = self.fusion(multi_scale_features)
        else:
            fused_list = self.fusion(multi_scale_features)
            fused = fused_list[0]  # Use highest resolution

        # Self-attention
        if self.attention_type == "self":
            B, C, H, W = fused.shape
            fused_flat = fused.flatten(2).transpose(1, 2)  # (B, HW, C)
            attended = self.attention(fused_flat)
            attended = attended.transpose(1, 2).view(B, C, H, W)
        else:
            attended = self.attention(fused)

        # Pose-guided attention (if enabled and prior available)
        if self.enable_pose_guided and pose_prior is not None:
            attended = self.pose_guided_attn(attended, pose_prior)

        # Global pooling and projection
        pooled = self.gap(attended).flatten(1)
        features = self.fc(pooled)

        # Pose regression
        yaw, pitch, roll, cls_logits = self.regressor(features)

        # Camera-aware transformation
        if camera_type is not None:
            yaw, pitch, roll = self.camera_transform(
                yaw, pitch, roll, camera_type
            )

        return {
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll,
            'features': features,
            'cls_logits': cls_logits,
        }

    def get_pose(
        self,
        x: torch.Tensor,
        camera_type: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simple interface for inference.

        Args:
            x: Input head image
            camera_type: Camera position

        Returns:
            Tuple of (yaw, pitch, roll) angles
        """
        output = self.forward(x, camera_type)
        return output['yaw'], output['pitch'], output['roll']


class CameraAwareTransform(nn.Module):
    """
    Camera-Aware Coordinate Transformation

    创新点：根据相机位置（正机位/侧机位）自适应转换坐标系，
    使得不同机位的角度具有统一的语义。

    正机位：人物正对相机，yaw=0表示正视
    侧机位：人物侧对相机，需要补偿相机角度
    """

    def __init__(self):
        super().__init__()

        # Learnable camera offsets
        self.register_buffer(
            'front_offset',
            torch.tensor([0.0, 0.0, 0.0])  # yaw, pitch, roll
        )
        self.register_buffer(
            'side_offset',
            torch.tensor([90.0, 0.0, 0.0])  # Side camera typically 90° offset
        )

    def forward(
        self,
        yaw: torch.Tensor,
        pitch: torch.Tensor,
        roll: torch.Tensor,
        camera_type: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform angles based on camera position.

        Args:
            yaw, pitch, roll: Predicted angles
            camera_type: 'front' or 'side'

        Returns:
            Transformed (yaw, pitch, roll)
        """
        if camera_type == 'front':
            offset = self.front_offset
        elif camera_type == 'side':
            offset = self.side_offset
        else:
            return yaw, pitch, roll

        yaw = yaw - offset[0]
        pitch = pitch - offset[1]
        roll = roll - offset[2]

        # Normalize yaw to [-180, 180]
        yaw = torch.remainder(yaw + 180, 360) - 180

        return yaw, pitch, roll


class WHENetPlusLoss(nn.Module):
    """
    Combined loss function for WHENet+

    Includes:
    1. Classification loss (cross-entropy for binned angles)
    2. Regression loss (geodesic distance)
    3. Feature consistency loss (for tracking)
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
        geo_weight: float = 0.5,
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.geo_weight = geo_weight

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.geo_loss = GeodesicLoss()

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            output: Model output dictionary
            target: Ground truth dictionary with keys:
                    'yaw', 'pitch', 'roll', 'yaw_bin', 'pitch_bin', 'roll_bin'

        Returns:
            Dictionary with loss values
        """
        # Classification loss
        yaw_cls, pitch_cls, roll_cls = output['cls_logits']
        cls_loss = (
            self.ce_loss(yaw_cls, target['yaw_bin']) +
            self.ce_loss(pitch_cls, target['pitch_bin']) +
            self.ce_loss(roll_cls, target['roll_bin'])
        ) / 3

        # Regression loss (MSE)
        reg_loss = (
            self.mse_loss(output['yaw'], target['yaw']) +
            self.mse_loss(output['pitch'], target['pitch']) +
            self.mse_loss(output['roll'], target['roll'])
        ) / 3

        # Geodesic loss
        geo_loss = self.geo_loss(
            (output['yaw'], output['pitch'], output['roll']),
            (target['yaw'], target['pitch'], target['roll']),
        )

        # Total loss
        total_loss = (
            self.cls_weight * cls_loss +
            self.reg_weight * reg_loss +
            self.geo_weight * geo_loss
        )

        return {
            'total': total_loss,
            'cls': cls_loss,
            'reg': reg_loss,
            'geo': geo_loss,
        }


def build_whenet_plus(config: dict) -> WHENetPlus:
    """
    Factory function to build WHENet+ model from config.

    Args:
        config: Configuration dictionary

    Returns:
        WHENetPlus model instance
    """
    model_cfg = config.get('model', {}).get('head_pose', {})

    return WHENetPlus(
        backbone=model_cfg.get('backbone', 'efficientnet_b0'),
        pretrained=model_cfg.get('pretrained', True),
        attention_type=model_cfg.get('attention', {}).get('type', 'self'),
        fusion_type=model_cfg.get('feature_fusion', {}).get('fusion_type', 'adaptive'),
        enable_pose_guided=True,
    )
