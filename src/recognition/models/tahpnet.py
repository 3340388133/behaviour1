#!/usr/bin/env python3
"""
TAHPNet: Temporal-Aware Head Pose Network
时序感知头部姿态估计网络

创新点：
1. RepVGG backbone（轻量高效，可重参数化）
2. 时序平滑模块（GRU）：解决单帧估计的帧间抖动问题
3. 多任务学习：同时预测姿态 + 姿态变化率
4. 时序一致性损失：约束相邻帧的姿态平滑

论文贡献：
- 解决单帧姿态估计的帧间不连续问题
- 提升下游行为识别任务的准确率
- 提供轻量级实时部署方案
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


# ============== RepVGG Block ==============
class RepVGGBlock(nn.Module):
    """RepVGG 基础模块

    训练时：3x3 conv + 1x1 conv + identity（如果stride=1且通道相同）
    推理时：融合为单个 3x3 conv
    """

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, groups: int = 1, deploy: bool = False):
        super().__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups

        padding = 1

        if deploy:
            # 推理模式：单个融合卷积
            self.rbr_reparam = nn.Conv2d(
                in_channels, out_channels, kernel_size=3,
                stride=stride, padding=padding, groups=groups, bias=True
            )
        else:
            # 训练模式：三分支
            # Identity branch (only when stride=1 and in_channels=out_channels)
            self.rbr_identity = nn.BatchNorm2d(in_channels) \
                if in_channels == out_channels and stride == 1 else None

            # 3x3 branch
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          stride=stride, padding=padding, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )

            # 1x1 branch
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, padding=0, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.act(self.rbr_reparam(x))

        # 训练模式：三分支相加
        out = self.rbr_dense(x) + self.rbr_1x1(x)
        if self.rbr_identity is not None:
            out = out + self.rbr_identity(x)
        return self.act(out)

    def get_equivalent_kernel_bias(self):
        """获取融合后的卷积核和偏置"""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)

        # 1x1 卷积核填充为 3x3
        kernel1x1_padded = F.pad(kernel1x1, [1, 1, 1, 1])

        return kernel3x3 + kernel1x1_padded + kernelid, bias3x3 + bias1x1 + biasid

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0

        if isinstance(branch, nn.Sequential):
            # Conv + BN
            conv = branch[0]
            bn = branch[1]
            kernel = conv.weight
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
        else:
            # BN only (identity branch)
            kernel = torch.zeros(self.out_channels, self.in_channels // self.groups, 3, 3,
                               device=branch.weight.device)
            for i in range(self.out_channels):
                kernel[i, i % (self.in_channels // self.groups), 1, 1] = 1
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        """切换到推理模式"""
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=3,
            stride=self.stride, padding=1, groups=self.groups, bias=True
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        # 删除训练分支
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        self.deploy = True


# ============== RepVGG Backbone ==============
class RepVGGBackbone(nn.Module):
    """RepVGG-A0 轻量级 Backbone

    输入: [B, 3, 224, 224]
    输出: [B, 1280] 特征向量
    """

    # RepVGG-A0 配置
    num_blocks = [2, 4, 14, 1]
    width_multiplier = [0.75, 0.75, 0.75, 2.5]

    def __init__(self, deploy: bool = False):
        super().__init__()
        self.deploy = deploy

        # Stage 0: stem
        self.stage0 = RepVGGBlock(3, int(64 * self.width_multiplier[0]), stride=2, deploy=deploy)

        # Stage 1-4
        self.stage1 = self._make_stage(int(64 * self.width_multiplier[0]),
                                       int(64 * self.width_multiplier[0]),
                                       self.num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(64 * self.width_multiplier[0]),
                                       int(128 * self.width_multiplier[1]),
                                       self.num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(128 * self.width_multiplier[1]),
                                       int(256 * self.width_multiplier[2]),
                                       self.num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(256 * self.width_multiplier[2]),
                                       int(512 * self.width_multiplier[3]),
                                       self.num_blocks[3], stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.out_channels = int(512 * self.width_multiplier[3])  # 1280

    def _make_stage(self, in_channels: int, out_channels: int,
                    num_blocks: int, stride: int):
        layers = []
        layers.append(RepVGGBlock(in_channels, out_channels, stride=stride, deploy=self.deploy))
        for _ in range(1, num_blocks):
            layers.append(RepVGGBlock(out_channels, out_channels, stride=1, deploy=self.deploy))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        return x.flatten(1)

    def switch_to_deploy(self):
        """切换到推理模式"""
        for module in self.modules():
            if isinstance(module, RepVGGBlock):
                module.switch_to_deploy()
        self.deploy = True


# ============== 时序平滑模块 ==============
class TemporalSmoothingModule(nn.Module):
    """时序平滑模块 (GRU-based)

    输入: [B, T, 3] 单帧姿态序列 (yaw, pitch, roll)
    输出: [B, T, 3] 平滑后的姿态序列

    创新点：
    1. 双向 GRU 捕捉前后文信息
    2. 残差连接保留原始估计
    3. 可学习的平滑权重
    """

    def __init__(self, pose_dim: int = 3, hidden_dim: int = 64,
                 num_layers: int = 2, bidirectional: bool = True):
        super().__init__()
        self.pose_dim = pose_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # GRU 时序建模
        self.gru = nn.GRU(
            input_size=pose_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.1 if num_layers > 1 else 0
        )

        # 输出投影
        gru_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_proj = nn.Sequential(
            nn.Linear(gru_out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, pose_dim)
        )

        # 可学习的残差权重
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, pose_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pose_seq: [B, T, 3] 原始姿态序列
        Returns:
            smoothed: [B, T, 3] 平滑后的姿态序列
        """
        # GRU 前向
        gru_out, _ = self.gru(pose_seq)  # [B, T, hidden*2]

        # 投影回姿态空间
        refined = self.output_proj(gru_out)  # [B, T, 3]

        # 残差连接（保留原始估计的部分信息）
        alpha = torch.sigmoid(self.residual_weight)
        smoothed = alpha * pose_seq + (1 - alpha) * refined

        return smoothed


# ============== TAHPNet 主模型 ==============
class TAHPNet(nn.Module):
    """Temporal-Aware Head Pose Network

    时序感知头部姿态估计网络

    输入:
        - 图像序列: [B, T, 3, H, W] 或单帧 [B, 3, H, W]
    输出:
        - 姿态: [B, T, 3] (yaw, pitch, roll)
        - 姿态变化率（可选）: [B, T-1, 3]
    """

    def __init__(
        self,
        backbone: str = "repvgg_a0",
        hidden_dim: int = 64,
        temporal_layers: int = 2,
        bidirectional: bool = True,
        deploy: bool = False,
        pretrained_backbone: str = None,
    ):
        super().__init__()
        self.deploy = deploy

        # Backbone
        if backbone == "repvgg_a0":
            self.backbone = RepVGGBackbone(deploy=deploy)
            backbone_dim = self.backbone.out_channels  # 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # 加载预训练权重
        if pretrained_backbone:
            self._load_pretrained_backbone(pretrained_backbone)

        # 单帧姿态头
        self.pose_head = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 3)  # yaw, pitch, roll
        )

        # 时序平滑模块
        self.temporal_module = TemporalSmoothingModule(
            pose_dim=3,
            hidden_dim=hidden_dim,
            num_layers=temporal_layers,
            bidirectional=bidirectional
        )

        # 是否输出姿态变化率
        self.predict_velocity = True

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _load_pretrained_backbone(self, path: str):
        """加载预训练 backbone 权重"""
        state_dict = torch.load(path, map_location='cpu')
        # 只加载 backbone 部分
        backbone_dict = {k.replace('backbone.', ''): v
                        for k, v in state_dict.items() if k.startswith('backbone.')}
        if backbone_dict:
            self.backbone.load_state_dict(backbone_dict, strict=False)
            print(f"Loaded pretrained backbone from {path}")

    def forward_single_frame(self, x: torch.Tensor) -> torch.Tensor:
        """单帧前向传播

        Args:
            x: [B, 3, H, W]
        Returns:
            pose: [B, 3]
        """
        feat = self.backbone(x)  # [B, 1280]
        pose = self.pose_head(feat)  # [B, 3]
        return pose

    def forward(self, x: torch.Tensor, use_temporal: bool = True) -> Dict[str, torch.Tensor]:
        """前向传播

        Args:
            x: [B, T, 3, H, W] 视频序列 或 [B, 3, H, W] 单帧
            use_temporal: 是否使用时序平滑

        Returns:
            dict with:
                - 'pose': [B, T, 3] 或 [B, 3] 姿态
                - 'pose_raw': [B, T, 3] 原始单帧估计（如果 use_temporal=True）
                - 'velocity': [B, T-1, 3] 姿态变化率（可选）
        """
        # 检查输入维度
        if x.dim() == 4:
            # 单帧输入 [B, 3, H, W]
            pose = self.forward_single_frame(x)
            return {'pose': pose}

        # 序列输入 [B, T, 3, H, W]
        B, T, C, H, W = x.shape

        # 展平后通过 backbone
        x_flat = x.view(B * T, C, H, W)
        feat = self.backbone(x_flat)  # [B*T, 1280]

        # 单帧姿态估计
        pose_raw = self.pose_head(feat)  # [B*T, 3]
        pose_raw = pose_raw.view(B, T, 3)  # [B, T, 3]

        result = {'pose_raw': pose_raw}

        # 时序平滑
        if use_temporal and T > 1:
            pose_smoothed = self.temporal_module(pose_raw)  # [B, T, 3]
            result['pose'] = pose_smoothed
        else:
            result['pose'] = pose_raw

        # 计算姿态变化率
        if self.predict_velocity and T > 1:
            velocity = result['pose'][:, 1:, :] - result['pose'][:, :-1, :]  # [B, T-1, 3]
            result['velocity'] = velocity

        return result

    def switch_to_deploy(self):
        """切换到推理模式（重参数化）"""
        self.backbone.switch_to_deploy()
        self.deploy = True


# ============== 损失函数 ==============
class TAHPNetLoss(nn.Module):
    """TAHPNet 复合损失函数

    包含：
    1. 姿态回归损失 (L1/L2)
    2. 时序一致性损失（平滑约束）
    3. 速度监督损失（可选）
    """

    def __init__(
        self,
        pose_loss_weight: float = 1.0,
        smoothness_weight: float = 0.1,
        velocity_weight: float = 0.05,
        raw_loss_weight: float = 0.3,
    ):
        super().__init__()
        self.pose_loss_weight = pose_loss_weight
        self.smoothness_weight = smoothness_weight
        self.velocity_weight = velocity_weight
        self.raw_loss_weight = raw_loss_weight

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: torch.Tensor,
        target_velocity: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: 模型输出字典
            target: [B, T, 3] 真实姿态
            target_velocity: [B, T-1, 3] 真实速度（可选）

        Returns:
            loss dict
        """
        losses = {}

        # 1. 主姿态损失
        pose_pred = pred['pose']
        pose_loss = F.smooth_l1_loss(pose_pred, target)
        losses['pose_loss'] = pose_loss * self.pose_loss_weight

        # 2. 原始估计损失（辅助监督）
        if 'pose_raw' in pred:
            raw_loss = F.smooth_l1_loss(pred['pose_raw'], target)
            losses['raw_loss'] = raw_loss * self.raw_loss_weight

        # 3. 时序一致性损失（帧间差异最小化）
        if pose_pred.dim() == 3 and pose_pred.size(1) > 1:
            # 二阶差分（加速度）应该较小
            diff = pose_pred[:, 1:, :] - pose_pred[:, :-1, :]  # [B, T-1, 3]
            accel = diff[:, 1:, :] - diff[:, :-1, :]  # [B, T-2, 3]
            smoothness_loss = accel.abs().mean()
            losses['smoothness_loss'] = smoothness_loss * self.smoothness_weight

        # 4. 速度监督损失
        if target_velocity is not None and 'velocity' in pred:
            velocity_loss = F.smooth_l1_loss(pred['velocity'], target_velocity)
            losses['velocity_loss'] = velocity_loss * self.velocity_weight

        # 总损失
        losses['total'] = sum(losses.values())

        return losses


# ============== 便捷函数 ==============
def create_tahpnet(
    pretrained: str = None,
    deploy: bool = False,
    **kwargs
) -> TAHPNet:
    """创建 TAHPNet 模型

    Args:
        pretrained: 预训练权重路径
        deploy: 是否为推理模式
        **kwargs: 其他参数

    Returns:
        TAHPNet 模型
    """
    model = TAHPNet(deploy=deploy, **kwargs)

    if pretrained:
        state_dict = torch.load(pretrained, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded TAHPNet from {pretrained}")

    if deploy:
        model.switch_to_deploy()

    return model


# ============== 测试代码 ==============
if __name__ == "__main__":
    # 测试模型
    print("Testing TAHPNet...")

    model = TAHPNet()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 单帧测试
    x_single = torch.randn(2, 3, 224, 224)
    out_single = model(x_single)
    print(f"Single frame output: {out_single['pose'].shape}")  # [2, 3]

    # 序列测试
    x_seq = torch.randn(2, 16, 3, 224, 224)  # batch=2, seq_len=16
    out_seq = model(x_seq)
    print(f"Sequence output:")
    print(f"  - pose: {out_seq['pose'].shape}")  # [2, 16, 3]
    print(f"  - pose_raw: {out_seq['pose_raw'].shape}")  # [2, 16, 3]
    print(f"  - velocity: {out_seq['velocity'].shape}")  # [2, 15, 3]

    # 测试损失
    loss_fn = TAHPNetLoss()
    target = torch.randn(2, 16, 3)
    losses = loss_fn(out_seq, target)
    print(f"Losses: {[(k, v.item()) for k, v in losses.items()]}")

    # 测试推理模式
    model.switch_to_deploy()
    out_deploy = model(x_seq)
    print(f"Deploy mode output: {out_deploy['pose'].shape}")

    print("All tests passed!")
