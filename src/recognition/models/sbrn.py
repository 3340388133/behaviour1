#!/usr/bin/env python3
"""
可疑行为识别网络 (Suspicious Behavior Recognition Network, SBRN)

整合四个核心创新点:
1. PAPE - 周期感知位置编码
2. BPCL - 行为原型对比学习
3. DGCMF - 动态门控跨模态融合
4. CIAT - 类别不平衡自适应训练 (在训练器中实现)

架构:
输入 → 特征投影 → PAPE → Transformer → DGCMF → BPCL/分类头 → 输出
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

try:
    from ..position_encoding import PAPE, PAPETransformerEncoderLayer
    from ..fusion import DGCMF
    from ..contrastive import BPCL
except (ImportError, ValueError):
    from position_encoding import PAPE, PAPETransformerEncoderLayer
    from fusion import DGCMF
    from contrastive import BPCL


@dataclass
class SBRNConfig:
    """SBRN模型配置"""
    # 输入维度
    pose_input_dim: int = 3  # yaw, pitch, roll
    appearance_dim: int = 512  # 外观特征维度
    motion_dim: int = 64  # 运动特征维度

    # Transformer参数
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1

    # PAPE参数
    max_seq_len: int = 512
    periods: List[int] = None  # 默认 [15, 30, 60]
    use_relative_bias: bool = True

    # DGCMF参数
    use_multimodal: bool = True
    use_quality_estimation: bool = True

    # BPCL参数
    use_contrastive: bool = True
    num_prototypes_per_class: int = 3
    temperature: float = 0.07
    contrastive_margin: float = 0.5

    # 分类参数
    num_classes: int = 3  # normal, looking_around, unknown
    hidden_dim: int = 128

    # 不确定性学习
    uncertainty_weighting: bool = True

    def __post_init__(self):
        if self.periods is None:
            self.periods = [15, 30, 60]


class SuspiciousBehaviorRecognitionNetwork(nn.Module):
    """
    可疑行为识别网络 (SBRN)

    完整模型架构:
    1. 输入投影层
    2. PAPE位置编码
    3. Transformer Encoder (with relative position bias)
    4. DGCMF跨模态融合
    5. BPCL对比学习 + 分类头
    6. 置信度估计头
    """

    def __init__(self, config: Optional[SBRNConfig] = None):
        super().__init__()

        self.config = config if config else SBRNConfig()
        c = self.config

        # ========== 1. 输入投影层 ==========
        self.pose_proj = nn.Sequential(
            nn.Linear(c.pose_input_dim, c.d_model),
            nn.LayerNorm(c.d_model),
            nn.GELU(),
            nn.Dropout(c.dropout),
        )

        # ========== 2. PAPE位置编码 (创新点1) ==========
        self.pape = PAPE(
            d_model=c.d_model,
            max_len=c.max_seq_len,
            dropout=c.dropout,
            periods=c.periods,
            use_relative_bias=c.use_relative_bias,
            num_heads=c.nhead,
        )

        # ========== 3. [CLS] Token ==========
        self.cls_token = nn.Parameter(torch.randn(1, 1, c.d_model) * 0.02)

        # ========== 4. Transformer Encoder (with PAPE) ==========
        self.transformer_layers = nn.ModuleList([
            PAPETransformerEncoderLayer(
                d_model=c.d_model,
                nhead=c.nhead,
                dim_feedforward=c.dim_feedforward,
                dropout=c.dropout,
            )
            for _ in range(c.num_layers)
        ])

        self.transformer_norm = nn.LayerNorm(c.d_model)

        # ========== 5. DGCMF跨模态融合 (创新点3) ==========
        if c.use_multimodal:
            self.dgcmf = DGCMF(
                pose_dim=c.d_model,
                appearance_dim=c.appearance_dim,
                motion_dim=c.motion_dim,
                hidden_dim=c.hidden_dim,
                num_heads=c.nhead // 2,
                dropout=c.dropout,
                use_quality_estimation=c.use_quality_estimation,
            )
            classifier_input_dim = c.hidden_dim
        else:
            self.dgcmf = None
            classifier_input_dim = c.d_model

        # ========== 6. BPCL对比学习 (创新点2) ==========
        if c.use_contrastive:
            self.bpcl = BPCL(
                feature_dim=classifier_input_dim,
                num_classes=c.num_classes,
                num_prototypes_per_class=c.num_prototypes_per_class,
                temperature=c.temperature,
                margin=c.contrastive_margin,
            )

        # ========== 7. 分类头 ==========
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, c.hidden_dim),
            nn.LayerNorm(c.hidden_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_dim, c.num_classes),
        )

        # ========== 8. 置信度估计头 ==========
        self.confidence_head = nn.Sequential(
            nn.Linear(classifier_input_dim, c.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(c.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # ========== 9. 不确定性权重 ==========
        if c.uncertainty_weighting:
            self.log_sigma_cls = nn.Parameter(torch.zeros(1))
            self.log_sigma_conf = nn.Parameter(torch.zeros(1))
            self.log_sigma_cont = nn.Parameter(torch.zeros(1))

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(
        self,
        pose_seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        提取时序特征

        Args:
            pose_seq: [batch, seq_len, pose_input_dim]
            mask: [batch, seq_len] padding mask

        Returns:
            [batch, d_model] CLS token表示
        """
        batch_size, seq_len, _ = pose_seq.shape

        # 1. 投影
        x = self.pose_proj(pose_seq)  # [B, T, d_model]

        # 2. 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, T+1, d_model]

        # 更新mask
        if mask is not None:
            cls_mask = torch.zeros(batch_size, 1, device=mask.device, dtype=torch.bool)
            mask = torch.cat([cls_mask, mask], dim=1)

        # 3. PAPE位置编码
        x, relative_bias = self.pape(x, return_relative_bias=True)

        # 4. Transformer layers
        for layer in self.transformer_layers:
            x = layer(x, relative_bias=relative_bias, src_key_padding_mask=mask)

        x = self.transformer_norm(x)

        # 5. 返回CLS token
        return x[:, 0, :]  # [B, d_model]

    def forward(
        self,
        pose_seq: torch.Tensor,
        appearance_feat: Optional[torch.Tensor] = None,
        motion_feat: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            pose_seq: [batch, seq_len, pose_input_dim] 姿态序列
            appearance_feat: [batch, appearance_dim] 外观特征 (可选)
            motion_feat: [batch, motion_dim] 运动特征 (可选)
            mask: [batch, seq_len] padding mask
            return_features: 是否返回中间特征

        Returns:
            Dict:
                - 'logits': [batch, num_classes] 分类logits
                - 'confidence': [batch, 1] 置信度
                - 'features': [batch, hidden_dim] (如果return_features=True)
                - 'prototype_logits': [batch, num_classes] (如果use_contrastive=True)
                - 'quality_scores': Dict (如果use_multimodal=True)
        """
        outputs = {}

        # 1. 提取时序特征
        pose_feat = self.forward_features(pose_seq, mask)  # [B, d_model]

        # 2. 跨模态融合
        if self.dgcmf is not None:
            fused_feat, quality_scores = self.dgcmf(
                pose_feat, appearance_feat, motion_feat,
                return_quality_scores=True
            )
            outputs['quality_scores'] = quality_scores
        else:
            fused_feat = pose_feat

        # 3. BPCL对比学习logits
        if hasattr(self, 'bpcl'):
            prototype_logits = self.bpcl(fused_feat)
            outputs['prototype_logits'] = prototype_logits

        # 4. 分类
        classifier_logits = self.classifier(fused_feat)

        # 5. 融合logits
        if hasattr(self, 'bpcl'):
            logits = (prototype_logits + classifier_logits) / 2
        else:
            logits = classifier_logits

        outputs['logits'] = logits

        # 6. 置信度
        confidence = self.confidence_head(fused_feat)
        outputs['confidence'] = confidence

        # 7. 可选返回特征
        if return_features:
            outputs['features'] = fused_feat

        return outputs

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        target_confidence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算损失

        Args:
            outputs: forward()的输出
            labels: [batch] 类别标签
            target_confidence: [batch] 目标置信度 (可选)

        Returns:
            (total_loss, loss_dict)
        """
        logits = outputs['logits']
        confidence = outputs['confidence']
        features = outputs.get('features', None)

        loss_dict = {}

        # 1. 分类损失
        loss_cls = F.cross_entropy(logits, labels)
        loss_dict['loss_cls'] = loss_cls.item()

        # 2. 置信度损失
        if target_confidence is not None:
            loss_conf = F.mse_loss(confidence.squeeze(), target_confidence)
        else:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                target_conf = (pred == labels).float()
            loss_conf = F.binary_cross_entropy(confidence.squeeze(), target_conf)
        loss_dict['loss_conf'] = loss_conf.item()

        # 3. 对比学习损失
        if hasattr(self, 'bpcl') and features is not None:
            loss_contrastive, cont_dict = self.bpcl.compute_contrastive_loss(
                features, labels
            )
            loss_dict['loss_contrastive'] = loss_contrastive.item()
            loss_dict.update({f'cont_{k}': v for k, v in cont_dict.items()})
        else:
            loss_contrastive = torch.tensor(0.0, device=logits.device)

        # 4. 不确定性加权
        if self.config.uncertainty_weighting:
            sigma_cls = torch.exp(self.log_sigma_cls)
            sigma_conf = torch.exp(self.log_sigma_conf)
            sigma_cont = torch.exp(self.log_sigma_cont)

            loss_cls_w = loss_cls / (2 * sigma_cls ** 2) + self.log_sigma_cls
            loss_conf_w = loss_conf / (2 * sigma_conf ** 2) + self.log_sigma_conf
            loss_cont_w = loss_contrastive / (2 * sigma_cont ** 2) + self.log_sigma_cont

            total_loss = loss_cls_w + 0.5 * loss_conf_w + 0.3 * loss_cont_w

            loss_dict['sigma_cls'] = sigma_cls.item()
            loss_dict['sigma_conf'] = sigma_conf.item()
            loss_dict['sigma_cont'] = sigma_cont.item()
        else:
            total_loss = loss_cls + 0.5 * loss_conf + 0.3 * loss_contrastive

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict

    def predict(
        self,
        pose_seq: torch.Tensor,
        appearance_feat: Optional[torch.Tensor] = None,
        motion_feat: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测接口

        Returns:
            (predictions, confidences)
            - predictions: [batch] 预测类别
            - confidences: [batch] 置信度
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(pose_seq, appearance_feat, motion_feat, mask)
            predictions = outputs['logits'].argmax(dim=-1)
            confidences = outputs['confidence'].squeeze()
        return predictions, confidences

    def get_attention_weights(
        self,
        pose_seq: torch.Tensor,
        layer_idx: int = -1,
    ) -> torch.Tensor:
        """
        获取注意力权重（用于可视化）

        Args:
            pose_seq: [batch, seq_len, pose_input_dim]
            layer_idx: 层索引

        Returns:
            [batch, nhead, seq_len+1, seq_len+1] 注意力权重
        """
        batch_size, seq_len, _ = pose_seq.shape

        # 投影
        x = self.pose_proj(pose_seq)

        # 添加CLS
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # PAPE
        x, relative_bias = self.pape(x, return_relative_bias=True)

        # 获取指定层的注意力
        layer = self.transformer_layers[layer_idx]

        # 手动计算注意力
        q = layer.q_proj(x).view(batch_size, -1, layer.nhead, layer.head_dim).transpose(1, 2)
        k = layer.k_proj(x).view(batch_size, -1, layer.nhead, layer.head_dim).transpose(1, 2)

        scale = math.sqrt(layer.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if relative_bias is not None:
            attn_scores = attn_scores + relative_bias.unsqueeze(0)

        attn_weights = F.softmax(attn_scores, dim=-1)

        return attn_weights


# 别名
SBRN = SuspiciousBehaviorRecognitionNetwork


def create_sbrn(
    num_classes: int = 3,
    use_multimodal: bool = True,
    use_contrastive: bool = True,
    d_model: int = 128,
    num_layers: int = 4,
    **kwargs
) -> SBRN:
    """
    创建SBRN模型的便捷函数

    Args:
        num_classes: 类别数
        use_multimodal: 是否使用多模态融合
        use_contrastive: 是否使用对比学习
        d_model: 模型维度
        num_layers: Transformer层数

    Returns:
        SBRN模型
    """
    config = SBRNConfig(
        num_classes=num_classes,
        use_multimodal=use_multimodal,
        use_contrastive=use_contrastive,
        d_model=d_model,
        num_layers=num_layers,
        **kwargs
    )
    return SBRN(config)


if __name__ == '__main__':
    # 测试SBRN
    print("Testing Suspicious Behavior Recognition Network (SBRN)...")

    batch_size = 4
    seq_len = 32
    pose_dim = 3
    appearance_dim = 512
    motion_dim = 64
    num_classes = 3

    # 创建模型
    config = SBRNConfig(
        pose_input_dim=pose_dim,
        appearance_dim=appearance_dim,
        motion_dim=motion_dim,
        num_classes=num_classes,
        use_multimodal=True,
        use_contrastive=True,
    )
    model = SBRN(config)

    # 计算参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")

    # 测试输入
    pose_seq = torch.randn(batch_size, seq_len, pose_dim)
    appearance_feat = torch.randn(batch_size, appearance_dim)
    motion_feat = torch.randn(batch_size, motion_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    # 测试前向传播
    print("\n1. Forward pass (full multimodal):")
    outputs = model(pose_seq, appearance_feat, motion_feat, return_features=True)
    print(f"   Pose sequence: {pose_seq.shape}")
    print(f"   Appearance: {appearance_feat.shape}")
    print(f"   Motion: {motion_feat.shape}")
    print(f"   Output logits: {outputs['logits'].shape}")
    print(f"   Confidence: {outputs['confidence'].shape}")
    print(f"   Features: {outputs['features'].shape}")
    print(f"   Prototype logits: {outputs['prototype_logits'].shape}")
    print(f"   Quality scores: {list(outputs['quality_scores'].keys())}")

    # 测试损失计算
    print("\n2. Loss computation:")
    loss, loss_dict = model.compute_loss(outputs, labels)
    print(f"   Total loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"   {k}: {v:.4f}")

    # 测试仅姿态输入
    print("\n3. Pose-only forward:")
    outputs_pose_only = model(pose_seq)
    print(f"   Logits: {outputs_pose_only['logits'].shape}")

    # 测试预测接口
    print("\n4. Predict:")
    preds, confs = model.predict(pose_seq, appearance_feat, motion_feat)
    print(f"   Predictions: {preds}")
    print(f"   Confidences: {confs}")

    # 测试注意力权重
    print("\n5. Attention weights:")
    attn = model.get_attention_weights(pose_seq)
    print(f"   Attention shape: {attn.shape}")

    # 测试便捷创建函数
    print("\n6. create_sbrn():")
    model2 = create_sbrn(num_classes=3, use_multimodal=False, use_contrastive=False)
    outputs2 = model2(pose_seq)
    print(f"   Logits: {outputs2['logits'].shape}")

    # 测试梯度流
    print("\n7. Gradient flow:")
    model.train()
    outputs = model(pose_seq, appearance_feat, motion_feat, return_features=True)
    loss, _ = model.compute_loss(outputs, labels)
    loss.backward()

    # 检查梯度
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append((name, param.grad.norm().item()))

    print(f"   Parameters with gradients: {len(grad_norms)}")
    print(f"   Sample gradients:")
    for name, norm in grad_norms[:5]:
        print(f"     {name}: {norm:.6f}")

    print("\nAll SBRN tests passed!")
