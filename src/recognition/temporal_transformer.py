#!/usr/bin/env python3
"""
识别层：时序 Transformer 行为识别模型

创新点：
1. Transformer 时序建模（替代规则/LSTM）
2. Coordinate Attention 方向感知注意力
3. 多模态融合（姿态 + 外观 + 运动）
4. 不确定性感知的多任务学习
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention（创新点 2）

    相比 CBAM：
    - 更强的方向感知能力
    - 分别在 H/W 方向编码位置信息
    - 更适合几何回归任务
    """

    def __init__(self, in_channels: int, reduction: int = 32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mid_channels = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.SiLU(inplace=True)

        self.conv_h = nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, height, width]
        """
        identity = x
        n, c, h, w = x.size()

        # [n, c, h, 1]
        x_h = self.pool_h(x)
        # [n, c, 1, w] -> [n, c, w, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # concat along spatial dim
        y = torch.cat([x_h, x_w], dim=2)  # [n, c, h+w, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_h * a_w


class TemporalTransformerEncoder(nn.Module):
    """时序 Transformer 编码器"""

    def __init__(
        self,
        input_dim: int = 3,          # 姿态维度 (yaw, pitch, roll)
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer Norm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim] 姿态序列
            mask: [batch, seq_len] padding mask

        Returns:
            [batch, d_model] CLS token 表示
        """
        batch_size = x.size(0)

        # 投影到 d_model
        x = self.input_proj(x)  # [B, T, d_model]

        # 添加 [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, T+1, d_model]

        # 位置编码
        x = self.pos_encoder(x)

        # 处理 mask（如果有）
        if mask is not None:
            # 为 CLS token 添加 False（不 mask）
            cls_mask = torch.zeros(batch_size, 1, device=mask.device, dtype=torch.bool)
            mask = torch.cat([cls_mask, mask], dim=1)

        # Transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)

        # 返回 [CLS] token
        return x[:, 0, :]


class MultiModalFusion(nn.Module):
    """
    多模态融合模块（创新点 3）

    融合：姿态 + 外观 + 运动 特征
    """

    def __init__(
        self,
        pose_dim: int = 64,
        appearance_dim: int = 512,
        motion_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.pose_proj = nn.Linear(pose_dim, hidden_dim)
        self.appearance_proj = nn.Linear(appearance_dim, hidden_dim)
        self.motion_proj = nn.Linear(motion_dim, hidden_dim)

        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        pose_feat: torch.Tensor,
        appearance_feat: Optional[torch.Tensor] = None,
        motion_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pose_feat: [batch, pose_dim]
            appearance_feat: [batch, appearance_dim] (可选)
            motion_feat: [batch, motion_dim] (可选)

        Returns:
            [batch, hidden_dim]
        """
        pose = self.pose_proj(pose_feat)

        if appearance_feat is not None:
            appearance = self.appearance_proj(appearance_feat)
        else:
            appearance = torch.zeros_like(pose)

        if motion_feat is not None:
            motion = self.motion_proj(motion_feat)
        else:
            motion = torch.zeros_like(pose)

        # 简单拼接融合
        fused = torch.cat([pose, appearance, motion], dim=-1)
        return self.fusion(fused)


class SuspiciousBehaviorClassifier(nn.Module):
    """
    可疑行为识别完整模型

    整合所有创新点：
    1. 时序 Transformer
    2. Coordinate Attention（可选，用于图像特征）
    3. 多模态融合
    4. 不确定性感知损失
    """

    def __init__(
        self,
        # 姿态编码器参数
        pose_input_dim: int = 3,
        pose_d_model: int = 64,
        pose_nhead: int = 4,
        pose_num_layers: int = 2,
        # 多模态参数
        use_multimodal: bool = False,
        appearance_dim: int = 512,
        motion_dim: int = 32,
        # 分类器参数
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1,
        # 不确定性学习
        uncertainty_weighting: bool = True,
    ):
        super().__init__()

        self.use_multimodal = use_multimodal
        self.uncertainty_weighting = uncertainty_weighting

        # 姿态序列编码器（创新点 1）
        self.pose_encoder = TemporalTransformerEncoder(
            input_dim=pose_input_dim,
            d_model=pose_d_model,
            nhead=pose_nhead,
            num_layers=pose_num_layers,
            dropout=dropout,
        )

        # 多模态融合（创新点 3，可选）
        if use_multimodal:
            self.fusion = MultiModalFusion(
                pose_dim=pose_d_model,
                appearance_dim=appearance_dim,
                motion_dim=motion_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            classifier_input_dim = hidden_dim
        else:
            self.fusion = None
            classifier_input_dim = pose_d_model

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # 置信度回归头
        self.confidence_head = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # 不确定性权重（创新点 4）
        if uncertainty_weighting:
            self.log_sigma_cls = nn.Parameter(torch.zeros(1))
            self.log_sigma_conf = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        pose_seq: torch.Tensor,
        appearance_feat: Optional[torch.Tensor] = None,
        motion_feat: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pose_seq: [batch, seq_len, 3] 姿态序列 (yaw, pitch, roll)
            appearance_feat: [batch, appearance_dim] 外观特征（可选）
            motion_feat: [batch, motion_dim] 运动特征（可选）
            mask: [batch, seq_len] padding mask

        Returns:
            logits: [batch, num_classes] 分类 logits
            confidence: [batch, 1] 置信度
        """
        # 编码姿态序列
        pose_feat = self.pose_encoder(pose_seq, mask)  # [B, pose_d_model]

        # 多模态融合
        if self.use_multimodal and self.fusion is not None:
            feat = self.fusion(pose_feat, appearance_feat, motion_feat)
        else:
            feat = pose_feat

        # 分类
        logits = self.classifier(feat)
        confidence = self.confidence_head(feat)

        return logits, confidence

    def compute_loss(
        self,
        logits: torch.Tensor,
        confidence: torch.Tensor,
        labels: torch.Tensor,
        target_confidence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算损失（创新点 4：不确定性感知）

        Args:
            logits: [batch, num_classes]
            confidence: [batch, 1]
            labels: [batch] 分类标签
            target_confidence: [batch, 1] 目标置信度（可选）

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失详情
        """
        # 分类损失
        loss_cls = F.cross_entropy(logits, labels)

        # 置信度损失
        if target_confidence is not None:
            loss_conf = F.mse_loss(confidence.squeeze(), target_confidence.squeeze())
        else:
            # 用预测正确性作为目标置信度
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                target_conf = (pred == labels).float()
            loss_conf = F.binary_cross_entropy(confidence.squeeze(), target_conf)

        # 不确定性加权
        if self.uncertainty_weighting:
            sigma_cls = torch.exp(self.log_sigma_cls)
            sigma_conf = torch.exp(self.log_sigma_conf)

            loss_cls_weighted = loss_cls / (2 * sigma_cls ** 2) + self.log_sigma_cls
            loss_conf_weighted = loss_conf / (2 * sigma_conf ** 2) + self.log_sigma_conf

            total_loss = loss_cls_weighted + 0.5 * loss_conf_weighted
        else:
            total_loss = loss_cls + 0.5 * loss_conf

        loss_dict = {
            'loss_cls': loss_cls.item(),
            'loss_conf': loss_conf.item(),
            'total': total_loss.item(),
        }

        if self.uncertainty_weighting:
            loss_dict['sigma_cls'] = sigma_cls.item()
            loss_dict['sigma_conf'] = sigma_conf.item()

        return total_loss, loss_dict


# ==================== 基线模型对比 ====================

class LSTMBaseline(nn.Module):
    """LSTM 基线模型（用于消融实验对比）"""

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 6,
        dropout: float = 0.1,
        **kwargs,  # 忽略其他参数
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x: [batch, seq_len, input_dim]
        output, (h_n, _) = self.lstm(x)

        # 取最后时刻的隐状态
        # h_n: [num_layers*2, batch, hidden_dim]
        h_forward = h_n[-2, :, :]  # 最后一层前向
        h_backward = h_n[-1, :, :]  # 最后一层后向
        h = torch.cat([h_forward, h_backward], dim=-1)

        logits = self.classifier(h)
        confidence = torch.ones(x.size(0), 1, device=x.device) * 0.5

        return logits, confidence


class RuleBaseline(nn.Module):
    """
    规则基线模型（用于消融实验对比）

    6类行为分类：
    - 0: normal        正常行为
    - 1: glancing      频繁张望
    - 2: quick_turn    快速回头
    - 3: prolonged_watch 长时间观察
    - 4: looking_down  持续低头
    - 5: looking_up    持续抬头
    """

    def __init__(self, num_classes: int = 6, fps: float = 30.0, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.fps = fps

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: [batch, seq_len, 3] (yaw, pitch, roll in degrees)
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        device = x.device

        logits = torch.zeros(batch_size, self.num_classes, device=device)
        confidences = torch.zeros(batch_size, 1, device=device)

        for b in range(batch_size):
            yaw_seq = x[b, :, 0].cpu().numpy()
            pitch_seq = x[b, :, 1].cpu().numpy()

            label = self._classify_sequence(yaw_seq, pitch_seq, seq_len)
            logits[b, label] = 1.0
            confidences[b, 0] = 0.8 if label > 0 else 0.2

        return logits, confidences

    def _classify_sequence(self, yaw_seq, pitch_seq, seq_len):
        """基于规则分类单个序列"""
        import numpy as np

        window_05s = int(0.5 * self.fps)

        # 前向帧过滤
        front_facing = [y for y in yaw_seq if abs(y) < 120]
        if len(front_facing) < seq_len * 0.3:
            return 0

        # 类别2: quick_turn
        for i in range(seq_len - window_05s):
            if abs(yaw_seq[min(i + window_05s, seq_len-1)] - yaw_seq[i]) > 60:
                return 2
        for i in range(1, seq_len):
            if abs(yaw_seq[i] - yaw_seq[i-1]) > 40:
                return 2

        # 类别1: glancing
        direction_changes = 0
        prev_direction = 0
        for i in range(1, seq_len):
            diff = yaw_seq[i] - yaw_seq[i - 1]
            if abs(diff) < 3:
                continue
            curr_direction = 1 if diff > 0 else -1
            if prev_direction != 0 and curr_direction != prev_direction:
                direction_changes += 1
            prev_direction = curr_direction

        yaw_amplitude = max(front_facing) - min(front_facing) if front_facing else 0
        if direction_changes >= 2 and yaw_amplitude > 30:
            return 1

        # 类别5: looking_up
        looking_up_ratio = sum(1 for p in pitch_seq if p > 20) / seq_len
        if looking_up_ratio > 0.5:
            return 5

        # 类别4: looking_down
        looking_down_ratio = sum(1 for p in pitch_seq if p < -20) / seq_len
        if looking_down_ratio > 0.5:
            return 4

        # 类别3: prolonged_watch
        side_watch_ratio = sum(1 for y in yaw_seq if 30 < abs(y) < 120) / seq_len
        if side_watch_ratio > 0.5:
            return 3

        # 类别0: normal
        return 0


# ==================== 工具函数 ====================

def create_model(
    model_type: str = 'transformer',
    **kwargs
) -> nn.Module:
    """
    创建模型

    Args:
        model_type: 'transformer', 'lstm', 'rule'
    """
    if model_type == 'transformer':
        return SuspiciousBehaviorClassifier(**kwargs)
    elif model_type == 'lstm':
        return LSTMBaseline(**kwargs)
    elif model_type == 'rule':
        return RuleBaseline(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # 测试模型
    batch_size = 4
    seq_len = 32
    pose_dim = 3

    # 随机姿态序列
    pose_seq = torch.randn(batch_size, seq_len, pose_dim) * 30  # 模拟角度值

    # 测试 Transformer 模型
    print("Testing Transformer model...")
    model = SuspiciousBehaviorClassifier(
        use_multimodal=False,
        uncertainty_weighting=True,
    )
    logits, conf = model(pose_seq)
    print(f"  Input: {pose_seq.shape}")
    print(f"  Logits: {logits.shape}")
    print(f"  Confidence: {conf.shape}")

    # 测试损失计算
    labels = torch.randint(0, 2, (batch_size,))
    loss, loss_dict = model.compute_loss(logits, conf, labels)
    print(f"  Loss: {loss_dict}")

    # 测试 LSTM 基线
    print("\nTesting LSTM baseline...")
    lstm_model = LSTMBaseline()
    logits, conf = lstm_model(pose_seq)
    print(f"  Logits: {logits.shape}")

    # 测试规则基线
    print("\nTesting Rule baseline...")
    rule_model = RuleBaseline()
    logits, conf = rule_model(pose_seq)
    print(f"  Logits: {logits.shape}")

    print("\nAll tests passed!")
