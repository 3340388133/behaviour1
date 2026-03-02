"""
3D Convolutional Neural Networks for Behavior Classification

3D-CNN models for learning spatiotemporal features from
pose sequences represented as pseudo-images.

创新点：
1. 将姿态序列转换为伪图像，利用3D-CNN提取时空特征
2. 轻量级3D卷积设计，适合实时应用
3. 多尺度3D卷积核捕获不同时间跨度的模式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class Conv3DBlock(nn.Module):
    """Basic 3D convolution block with BN and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Optional[Tuple[int, int, int]] = None,
    ):
        super().__init__()

        if padding is None:
            padding = tuple(k // 2 for k in kernel_size)

        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class CNN3D(nn.Module):
    """
    3D Convolutional Neural Network for Pose Sequence Classification

    创新点：专门设计用于姿态序列分类的轻量级3D-CNN。
    输入：时序姿态特征图 (B, C, T, H, W) 或 (B, C, T)
    - C: 特征通道 (yaw, pitch, roll, etc.)
    - T: 时间维度
    - H, W: 空间维度（可选）

    Args:
        in_channels: Input channels (pose features)
        hidden_channels: List of hidden channel numbers
        num_classes: Number of output classes
        temporal_kernel: Temporal convolution kernel size
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: List[int] = [32, 64, 128],
        num_classes: int = 2,
        temporal_kernel: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.in_channels = in_channels

        # Build convolutional layers
        layers = []
        prev_channels = in_channels

        for i, out_ch in enumerate(hidden_channels):
            # 3D convolution (temporal + pseudo-spatial)
            layers.append(Conv3DBlock(
                prev_channels, out_ch,
                kernel_size=(temporal_kernel, 3, 3),
                stride=(1, 1, 1),
            ))

            # Temporal pooling (downsample time)
            if i < len(hidden_channels) - 1:
                layers.append(nn.MaxPool3d(
                    kernel_size=(2, 2, 2),
                    stride=(2, 2, 2),
                ))

            prev_channels = out_ch

        self.features = nn.Sequential(*layers)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_channels[-1], hidden_channels[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels[-1] // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor
               - If 3D (B, C, T): Treated as 1D temporal sequence
               - If 5D (B, C, T, H, W): Full 3D input

        Returns:
            Class logits (B, num_classes)
        """
        # Handle 1D temporal input
        if x.dim() == 3:
            B, C, T = x.shape
            # Add pseudo-spatial dimensions
            x = x.unsqueeze(-1).unsqueeze(-1)  # (B, C, T, 1, 1)
            x = x.expand(-1, -1, -1, 7, 7)     # (B, C, T, 7, 7)

        # Feature extraction
        features = self.features(x)

        # Global pooling
        pooled = self.global_pool(features).flatten(1)

        # Classification
        logits = self.classifier(pooled)

        return logits


class C3D(nn.Module):
    """
    C3D: Learning Spatiotemporal Features with 3D Convolutional Networks

    Classic C3D architecture adapted for pose sequence classification.

    Reference: Tran et al., "Learning Spatiotemporal Features with 3D
    Convolutional Networks", ICCV 2015
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 2,
    ):
        super().__init__()

        self.conv1 = Conv3DBlock(in_channels, 64, (3, 3, 3))
        self.pool1 = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))

        self.conv2 = Conv3DBlock(64, 128, (3, 3, 3))
        self.pool2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

        self.conv3a = Conv3DBlock(128, 256, (3, 3, 3))
        self.conv3b = Conv3DBlock(256, 256, (3, 3, 3))
        self.pool3 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

        self.conv4a = Conv3DBlock(256, 512, (3, 3, 3))
        self.conv4b = Conv3DBlock(512, 512, (3, 3, 3))
        self.pool4 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expand to 5D if needed
        if x.dim() == 3:
            x = x.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 16, 16)

        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3b(self.conv3a(x)))
        x = self.pool4(self.conv4b(self.conv4a(x)))

        x = self.global_pool(x).flatten(1)
        x = self.fc(x)

        return x


class R3D(nn.Module):
    """
    R3D: 3D ResNet for Action Recognition

    ResNet-style 3D CNN with residual connections.

    创新点：残差连接使得更深的3D网络成为可能，
    提升时空特征的学习能力。
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 2,
        layers: List[int] = [2, 2, 2, 2],
    ):
        super().__init__()

        self.in_planes = 64

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 64, (3, 7, 7), stride=(1, 2, 2),
                     padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        layers = [ResBlock3D(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(ResBlock3D(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 16, 16)

        x = self.conv1(x)
        x = self.pool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x).flatten(1)
        x = self.fc(x)

        return x


class ResBlock3D(nn.Module):
    """3D Residual Block."""

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(in_planes, planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class TemporalConv1D(nn.Module):
    """
    1D Temporal Convolution for Pose Sequences

    轻量级替代方案：当不需要空间维度时，
    使用1D卷积直接处理时序特征。

    Args:
        in_channels: Input feature channels
        hidden_channels: List of hidden channels
        num_classes: Number of output classes
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: List[int] = [64, 128, 256],
        num_classes: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()

        layers = []
        prev_ch = in_channels

        for out_ch in hidden_channels:
            layers.extend([
                nn.Conv1d(prev_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ])
            prev_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_channels[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) or (B, C, T)
        """
        if x.shape[-1] != x.shape[1]:
            x = x.transpose(1, 2)  # (B, C, T)

        x = self.features(x)
        x = self.global_pool(x).flatten(1)
        x = self.classifier(x)

        return x
