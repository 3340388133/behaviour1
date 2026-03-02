"""
FSA-Net PyTorch 实现 (简化版)
基于 Fine-Grained Structure Aggregation for Head Pose Estimation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SSRLayer(nn.Module):
    """Soft Stagewise Regression Layer"""
    def __init__(self, num_classes, stage_num, lambda_d):
        super().__init__()
        self.num_classes = num_classes
        self.stage_num = stage_num
        self.lambda_d = lambda_d

    def forward(self, x):
        # x: [B, stage_num, num_classes]
        a = x[:, 0, :]  # [B, num_classes]
        b = x[:, 1, :]
        c = x[:, 2, :]

        # Softmax
        a = F.softmax(a, dim=1)
        b = F.softmax(b, dim=1)
        c = F.softmax(c, dim=1)

        # Index tensor
        idx = torch.arange(self.num_classes, device=x.device, dtype=x.dtype)

        # Stage 1
        pred_a = (a * idx).sum(dim=1, keepdim=True)

        # Stage 2
        pred_b = (b * idx).sum(dim=1, keepdim=True)
        pred_b = pred_b / (self.num_classes / (1 + self.lambda_d))

        # Stage 3
        pred_c = (c * idx).sum(dim=1, keepdim=True)
        pred_c = pred_c / (self.num_classes / (1 + 2 * self.lambda_d))

        # Combine
        pred = pred_a + pred_b + pred_c
        return pred


class FSANet(nn.Module):
    """FSA-Net for Head Pose Estimation"""
    def __init__(self, num_classes=3, stage_num=3, lambda_d=1):
        super().__init__()

        # Feature extraction (simplified)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        # Regression heads for yaw, pitch, roll
        self.fc_yaw = nn.Linear(256, stage_num * num_classes)
        self.fc_pitch = nn.Linear(256, stage_num * num_classes)
        self.fc_roll = nn.Linear(256, stage_num * num_classes)

        self.num_classes = num_classes
        self.stage_num = stage_num
        self.ssr_yaw = SSRLayer(num_classes, stage_num, lambda_d)
        self.ssr_pitch = SSRLayer(num_classes, stage_num, lambda_d)
        self.ssr_roll = SSRLayer(num_classes, stage_num, lambda_d)

    def forward(self, x):
        # x: [B, 3, 64, 64]
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)

        # Yaw
        yaw = self.fc_yaw(feat)
        yaw = yaw.view(-1, self.stage_num, self.num_classes)
        yaw = self.ssr_yaw(yaw)

        # Pitch
        pitch = self.fc_pitch(feat)
        pitch = pitch.view(-1, self.stage_num, self.num_classes)
        pitch = self.ssr_pitch(pitch)

        # Roll
        roll = self.fc_roll(feat)
        roll = roll.view(-1, self.stage_num, self.num_classes)
        roll = self.ssr_roll(roll)

        return yaw, pitch, roll


def create_fsanet_model():
    """创建 FSA-Net 模型 (随机初始化)"""
    model = FSANet(num_classes=66, stage_num=3, lambda_d=1)
    return model
