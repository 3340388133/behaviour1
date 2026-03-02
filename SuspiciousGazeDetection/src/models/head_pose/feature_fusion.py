"""
Multi-Scale Feature Fusion Module

This module implements various feature fusion strategies to combine
features from different scales and improve head pose estimation accuracy.

创新点：多尺度特征融合增强小目标（远距离人头）的检测能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class FeatureFusion(nn.Module):
    """
    Basic Feature Fusion Module

    Fuses features from multiple scales using various strategies.

    Args:
        in_channels: List of input channel numbers for each scale
        out_channels: Output channel number
        fusion_type: Fusion strategy ('concat', 'add', 'attention')
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        fusion_type: str = "concat",
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.num_scales = len(in_channels)

        # Channel alignment
        self.align_convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, 1, bias=False)
            for ch in in_channels
        ])
        self.align_bns = nn.ModuleList([
            nn.BatchNorm2d(out_channels)
            for _ in in_channels
        ])

        if fusion_type == "concat":
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(out_channels * self.num_scales, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        elif fusion_type == "attention":
            self.attention = ScaleAttention(out_channels, self.num_scales)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of feature maps from different scales
                      Each tensor has shape (B, C_i, H_i, W_i)

        Returns:
            Fused feature map (B, out_channels, H, W)
        """
        assert len(features) == self.num_scales

        # Get target size (use the largest feature map)
        target_size = features[0].shape[2:]

        # Align channels and resize
        aligned = []
        for i, feat in enumerate(features):
            x = self.align_convs[i](feat)
            x = self.align_bns[i](x)
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear',
                                 align_corners=False)
            aligned.append(x)

        # Fusion
        if self.fusion_type == "concat":
            fused = torch.cat(aligned, dim=1)
            fused = self.fusion_conv(fused)
        elif self.fusion_type == "add":
            fused = sum(aligned)
        elif self.fusion_type == "attention":
            fused = self.attention(aligned)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        return fused


class ScaleAttention(nn.Module):
    """
    Scale-wise Attention for Feature Fusion

    创新点：学习不同尺度特征的重要性权重，
    自适应地融合多尺度信息。

    Args:
        channels: Number of channels per scale
        num_scales: Number of scales to fuse
    """

    def __init__(self, channels: int, num_scales: int):
        super().__init__()
        self.num_scales = num_scales

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels * num_scales, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, num_scales),
            nn.Softmax(dim=1),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of aligned feature maps (B, C, H, W)

        Returns:
            Attention-weighted fused features
        """
        B = features[0].shape[0]

        # Global features from each scale
        global_feats = [self.global_pool(f).view(B, -1) for f in features]
        concat_feats = torch.cat(global_feats, dim=1)

        # Compute attention weights
        weights = self.fc(concat_feats)  # (B, num_scales)

        # Weighted sum
        stacked = torch.stack(features, dim=1)  # (B, num_scales, C, H, W)
        weights = weights.view(B, self.num_scales, 1, 1, 1)
        fused = (stacked * weights).sum(dim=1)

        return fused


class MultiScaleFusion(nn.Module):
    """
    Multi-Scale Feature Pyramid Fusion

    创新点：构建特征金字塔，实现自顶向下和自底向上的双向融合，
    增强不同尺度头部的检测能力。

    Args:
        in_channels: List of input channel numbers (from backbone stages)
        out_channels: Output channel number for each level
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
    ):
        super().__init__()
        self.num_levels = len(in_channels)

        # Lateral connections (1x1 conv for channel reduction)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, 1, bias=False)
            for ch in in_channels
        ])

        # Top-down pathway (3x3 conv after upsampling)
        self.td_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
            for _ in range(self.num_levels - 1)
        ])

        # Bottom-up pathway
        self.bu_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
            for _ in range(self.num_levels - 1)
        ])

        # Output convolutions
        self.out_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(self.num_levels)
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of feature maps from backbone stages
                      Ordered from high resolution to low resolution

        Returns:
            List of fused feature maps at each level
        """
        assert len(features) == self.num_levels

        # Lateral connections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i in range(self.num_levels - 1, 0, -1):
            size = laterals[i - 1].shape[2:]
            upsampled = F.interpolate(laterals[i], size=size, mode='nearest')
            laterals[i - 1] = laterals[i - 1] + self.td_convs[i - 1](upsampled)

        # Bottom-up pathway
        outputs = [laterals[0]]
        for i in range(self.num_levels - 1):
            downsampled = self.bu_convs[i](outputs[-1])
            outputs.append(laterals[i + 1] + downsampled)

        # Output convolutions
        outputs = [conv(f) for conv, f in zip(self.out_convs, outputs)]

        return outputs


class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive Feature Fusion with Spatial Attention

    创新点：结合空间注意力的自适应特征融合，
    在不同空间位置自动选择最优尺度的特征。

    Args:
        in_channels: List of input channels
        out_channels: Output channels
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
    ):
        super().__init__()
        self.num_scales = len(in_channels)

        # Channel projection
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            for ch in in_channels
        ])

        # Spatial attention for each scale
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(out_channels * self.num_scales, out_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, self.num_scales, 3, padding=1),
        )

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: Multi-scale feature maps

        Returns:
            Spatially adaptive fused features
        """
        target_size = features[0].shape[2:]

        # Project and resize
        aligned = []
        for i, feat in enumerate(features):
            x = self.projs[i](feat)
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear',
                                 align_corners=False)
            aligned.append(x)

        # Compute spatial attention weights
        concat = torch.cat(aligned, dim=1)
        attn = self.spatial_attn(concat)  # (B, num_scales, H, W)
        attn = F.softmax(attn, dim=1)

        # Weighted fusion
        stacked = torch.stack(aligned, dim=1)  # (B, num_scales, C, H, W)
        attn = attn.unsqueeze(2)  # (B, num_scales, 1, H, W)
        fused = (stacked * attn).sum(dim=1)

        return self.out_proj(fused)
