"""
Attention Mechanisms for Head Pose Estimation

This module implements various attention mechanisms to enhance
the feature extraction capability of the head pose estimation model.

Includes:
- SelfAttention: Multi-head self-attention
- CBAM: Convolutional Block Attention Module
- SimDLKA: Simple Dilated Large Kernel Attention (SOTA 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple


class SelfAttention(nn.Module):
    """
    Multi-head Self-Attention Module

    创新点：在头部姿态估计中引入自注意力机制，
    使模型能够关注图像中与头部姿态相关的关键区域。

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in QKV projection
        attn_drop: Attention dropout rate
        proj_drop: Projection dropout rate
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, C) or (B, C, H, W)

        Returns:
            Output tensor of same shape as input
        """
        input_shape = x.shape

        # Handle 2D feature maps
        if len(input_shape) == 4:
            B, C, H, W = input_shape
            x = rearrange(x, 'b c h w -> b (h w) c')

        B, N, C = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Restore original shape if needed
        if len(input_shape) == 4:
            x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x


class ChannelAttention(nn.Module):
    """Channel Attention Module for CBAM"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape

        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))

        attn = torch.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    """Spatial Attention Module for CBAM"""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.conv(concat))
        return x * attn


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)

    Sequential channel and spatial attention for CNNs.
    Ref: CBAM: Convolutional Block Attention Module (ECCV 2018)

    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio
        spatial_kernel: Kernel size for spatial attention
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel: int = 7,
    ):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SimDLKA(nn.Module):
    """
    Simple Dilated Large Kernel Attention (SimDLKA)

    创新点：2025年SOTA注意力机制，使用扩张卷积实现大感受野，
    在姿态估计中能更好地捕获头部与身体的空间关系。

    Ref: Integration of SimDLKA attention mechanism in YOLOv8 (PLOS One 2025)

    Args:
        channels: Number of input channels
        kernel_sizes: List of kernel sizes for multi-scale
        dilation_rates: List of dilation rates
    """

    def __init__(
        self,
        channels: int,
        kernel_sizes: Tuple[int, ...] = (5, 7, 9),
        dilation_rates: Tuple[int, ...] = (1, 2, 3),
    ):
        super().__init__()

        self.branches = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilation_rates):
            padding = (k + (k - 1) * (d - 1)) // 2
            branch = nn.Sequential(
                nn.Conv2d(channels, channels, k, padding=padding,
                         dilation=d, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            self.branches.append(branch)

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * len(kernel_sizes), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-scale feature extraction
        features = [branch(x) for branch in self.branches]
        multi_scale = torch.cat(features, dim=1)

        # Feature fusion
        fused = self.fusion(multi_scale)

        # Gating mechanism
        gate = self.gate(fused)

        return x + fused * gate


class PoseGuidedAttention(nn.Module):
    """
    Pose-Guided Attention Module

    创新点：利用先验姿态信息引导注意力分布，
    使模型更关注头部区域的关键特征点。

    Args:
        channels: Number of input channels
        pose_dim: Dimension of pose features (yaw, pitch, roll)
    """

    def __init__(self, channels: int, pose_dim: int = 3):
        super().__init__()

        self.pose_embed = nn.Sequential(
            nn.Linear(pose_dim, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid(),
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        pose_prior: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Feature map (B, C, H, W)
            pose_prior: Prior pose estimation (B, 3) - optional

        Returns:
            Attention-weighted feature map
        """
        B, C, H, W = x.shape

        # Spatial attention
        spatial_attn = self.spatial_conv(x)

        if pose_prior is not None:
            # Channel modulation based on pose
            pose_weight = self.pose_embed(pose_prior).view(B, C, 1, 1)
            x = x * pose_weight

        return x * spatial_attn
