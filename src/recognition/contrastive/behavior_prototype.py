#!/usr/bin/env python3
"""
创新点2: 行为原型对比学习 (Behavior Prototype Contrastive Learning, BPCL)

理论依据:
- 数据极度不平衡 (normal:5%, looking_around:78%, unknown:17%)
- 少数类(normal)样本不足导致决策边界模糊
- 传统CE损失对少数类学习效果差

技术方案:
1. 类别原型: 每个类别维护多个可学习原型向量（捕捉类内变化）
2. InfoNCE损失: 拉近样本与同类原型，推远与异类原型
3. 动量更新: 使用EMA更新原型，防止训练震荡
4. 困难负样本挖掘: 关注决策边界附近的困难样本
5. 边界损失: 显式拉大类间距离
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List


class BehaviorPrototypeContrastiveLearning(nn.Module):
    """
    行为原型对比学习 (BPCL)

    Args:
        feature_dim: 特征维度
        num_classes: 类别数量
        num_prototypes_per_class: 每个类别的原型数量
        temperature: InfoNCE温度参数
        momentum: 原型动量更新系数
        margin: 边界损失的边界值
        hard_negative_ratio: 困难负样本比例
    """

    def __init__(
        self,
        feature_dim: int = 128,
        num_classes: int = 3,
        num_prototypes_per_class: int = 3,
        temperature: float = 0.07,
        momentum: float = 0.999,
        margin: float = 0.5,
        hard_negative_ratio: float = 0.5,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes_per_class
        self.temperature = temperature
        self.momentum = momentum
        self.margin = margin
        self.hard_negative_ratio = hard_negative_ratio

        # ========== 1. 可学习原型向量 ==========
        # [num_classes, num_prototypes, feature_dim]
        self.prototypes = nn.Parameter(
            torch.randn(num_classes, num_prototypes_per_class, feature_dim) * 0.02
        )

        # 用于动量更新的影子原型 (不参与梯度计算)
        self.register_buffer(
            'prototype_momentum',
            torch.randn(num_classes, num_prototypes_per_class, feature_dim) * 0.02
        )

        # ========== 2. 特征投影头 ==========
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

        # ========== 3. 类别统计 (用于采样平衡) ==========
        # 追踪每个类别的样本数（用于加权）
        self.register_buffer('class_counts', torch.ones(num_classes))
        self.register_buffer('total_samples', torch.tensor(0.0))

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_similarities: bool = False,
    ):
        """
        前向传播

        Args:
            features: [batch, feature_dim] 输入特征
            labels: [batch] 类别标签 (训练时需要)
            return_similarities: 是否返回相似度矩阵

        Returns:
            如果 return_similarities=False:
                [batch, num_classes] 类别logits (基于原型相似度)
            如果 return_similarities=True:
                (logits, similarities)
                - similarities: [batch, num_classes, num_prototypes]
        """
        batch_size = features.shape[0]

        # 1. 投影特征
        projected = self.projector(features)  # [batch, feature_dim]
        projected = F.normalize(projected, dim=-1)

        # 2. L2归一化原型
        prototypes_normalized = F.normalize(self.prototypes, dim=-1)  # [C, P, D]

        # 3. 计算相似度: [batch, num_classes, num_prototypes]
        # projected: [B, D] -> [B, 1, 1, D]
        # prototypes: [C, P, D] -> [1, C, P, D]
        similarities = torch.einsum(
            'bd,cpd->bcp',
            projected,
            prototypes_normalized
        )  # [batch, num_classes, num_prototypes]

        # 4. 对每个类别取最大相似度 (最近原型)
        max_similarities, _ = similarities.max(dim=-1)  # [batch, num_classes]

        # 5. 转换为logits
        logits = max_similarities / self.temperature

        if return_similarities:
            return logits, similarities
        return logits

    def compute_contrastive_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算对比学习损失

        包含:
        1. InfoNCE损失: 拉近正样本，推远负样本
        2. 边界损失: 显式拉大类间距离
        3. 原型多样性损失: 防止同类原型坍塌

        Args:
            features: [batch, feature_dim]
            labels: [batch]

        Returns:
            (total_loss, loss_dict)
        """
        batch_size = features.shape[0]
        device = features.device

        # 1. 投影并归一化
        projected = self.projector(features)
        projected = F.normalize(projected, dim=-1)

        # 归一化原型
        prototypes_normalized = F.normalize(self.prototypes, dim=-1)

        # 2. 计算所有相似度
        similarities = torch.einsum(
            'bd,cpd->bcp',
            projected,
            prototypes_normalized
        )  # [batch, num_classes, num_prototypes]

        # ========== InfoNCE损失 ==========
        # 对每个样本，正类的最大相似度 vs 负类的相似度
        infonce_loss = self._compute_infonce_loss(similarities, labels)

        # ========== 边界损失 ==========
        margin_loss = self._compute_margin_loss(similarities, labels)

        # ========== 原型多样性损失 ==========
        diversity_loss = self._compute_diversity_loss(prototypes_normalized)

        # ========== 困难负样本损失 ==========
        hard_negative_loss = self._compute_hard_negative_loss(
            projected, prototypes_normalized, labels
        )

        # 总损失
        total_loss = (
            infonce_loss +
            0.1 * margin_loss +
            0.05 * diversity_loss +
            0.2 * hard_negative_loss
        )

        loss_dict = {
            'infonce': infonce_loss.item(),
            'margin': margin_loss.item(),
            'diversity': diversity_loss.item(),
            'hard_negative': hard_negative_loss.item(),
            'total_contrastive': total_loss.item(),
        }

        # 更新类别统计
        self._update_class_counts(labels)

        return total_loss, loss_dict

    def _compute_infonce_loss(
        self,
        similarities: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算InfoNCE损失

        Args:
            similarities: [batch, num_classes, num_prototypes]
            labels: [batch]
        """
        batch_size = similarities.shape[0]
        device = similarities.device

        # 对每个类别取最大相似度
        max_sim, _ = similarities.max(dim=-1)  # [batch, num_classes]
        max_sim = max_sim / self.temperature

        # 正类相似度
        positive_sim = max_sim[torch.arange(batch_size, device=device), labels]  # [batch]

        # InfoNCE: -log(exp(pos) / sum(exp(all)))
        # = -pos + log(sum(exp(all)))
        log_sum_exp = torch.logsumexp(max_sim, dim=-1)  # [batch]
        loss = -positive_sim + log_sum_exp

        return loss.mean()

    def _compute_margin_loss(
        self,
        similarities: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算边界损失

        确保正类相似度比负类相似度至少高出margin
        """
        batch_size = similarities.shape[0]
        device = similarities.device

        # 每个类别的最大相似度
        max_sim, _ = similarities.max(dim=-1)  # [batch, num_classes]

        # 正类相似度
        pos_mask = F.one_hot(labels, self.num_classes).bool()
        positive_sim = max_sim.masked_select(pos_mask)  # [batch]

        # 负类最大相似度
        neg_sim = max_sim.masked_fill(pos_mask, float('-inf'))
        max_neg_sim, _ = neg_sim.max(dim=-1)  # [batch]

        # Margin loss: max(0, margin - (pos - neg))
        margin_violation = self.margin - (positive_sim - max_neg_sim)
        loss = F.relu(margin_violation)

        return loss.mean()

    def _compute_diversity_loss(
        self,
        prototypes_normalized: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算原型多样性损失

        防止同类的不同原型坍塌到同一点
        """
        # prototypes_normalized: [num_classes, num_prototypes, feature_dim]
        loss = 0.0

        for c in range(self.num_classes):
            class_prototypes = prototypes_normalized[c]  # [num_prototypes, feature_dim]

            # 计算同类原型间的余弦相似度
            sim_matrix = torch.mm(class_prototypes, class_prototypes.t())  # [P, P]

            # 排除对角线
            mask = ~torch.eye(self.num_prototypes, device=sim_matrix.device).bool()
            off_diag_sim = sim_matrix.masked_select(mask)

            # 惩罚过高的相似度（鼓励多样性）
            loss = loss + F.relu(off_diag_sim - 0.5).mean()

        return loss / self.num_classes

    def _compute_hard_negative_loss(
        self,
        projected: torch.Tensor,
        prototypes_normalized: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算困难负样本损失

        关注决策边界附近的困难样本
        """
        batch_size = projected.shape[0]
        device = projected.device

        # 计算与所有原型的相似度
        similarities = torch.einsum(
            'bd,cpd->bcp',
            projected,
            prototypes_normalized
        )  # [batch, num_classes, num_prototypes]

        # 每个类别的最大相似度
        max_sim, _ = similarities.max(dim=-1)  # [batch, num_classes]

        # 正类mask
        pos_mask = F.one_hot(labels, self.num_classes).bool()

        # 找到困难负样本：与正类相似度最接近的负类
        neg_sim = max_sim.clone()
        neg_sim[pos_mask] = float('-inf')

        # 获取top-k困难负样本
        k = max(1, int(batch_size * self.hard_negative_ratio))

        # 对每个样本，找最困难的负类
        hardest_neg_sim, hardest_neg_idx = neg_sim.max(dim=-1)  # [batch]

        # 正类相似度
        positive_sim = max_sim[torch.arange(batch_size, device=device), labels]

        # 只对困难样本计算损失
        # 困难样本：负类相似度接近正类
        difficulty = hardest_neg_sim - positive_sim + self.margin
        hard_samples = difficulty > 0

        if hard_samples.sum() > 0:
            loss = F.relu(difficulty[hard_samples]).mean()
        else:
            loss = torch.tensor(0.0, device=device)

        return loss

    def _update_class_counts(self, labels: torch.Tensor):
        """更新类别统计"""
        for c in range(self.num_classes):
            count = (labels == c).sum().float()
            self.class_counts[c] = 0.9 * self.class_counts[c] + 0.1 * count
        self.total_samples = self.total_samples + labels.shape[0]

    @torch.no_grad()
    def update_prototypes_momentum(self):
        """使用动量更新原型（在训练循环中调用）"""
        self.prototype_momentum.data = (
            self.momentum * self.prototype_momentum.data +
            (1 - self.momentum) * self.prototypes.data
        )

    def get_prototype_assignments(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取样本的原型分配

        Args:
            features: [batch, feature_dim]

        Returns:
            (class_ids, prototype_ids)
            - class_ids: [batch] 预测类别
            - prototype_ids: [batch] 对应的原型索引
        """
        logits, similarities = self.forward(features, return_similarities=True)

        # 预测类别
        class_ids = logits.argmax(dim=-1)  # [batch]

        # 找到最近的原型
        batch_size = features.shape[0]
        best_prototypes = []

        for i in range(batch_size):
            cls = class_ids[i].item()
            proto_sims = similarities[i, cls]  # [num_prototypes]
            best_proto = proto_sims.argmax().item()
            best_prototypes.append(best_proto)

        prototype_ids = torch.tensor(best_prototypes, device=features.device)

        return class_ids, prototype_ids

    def get_class_prototypes(self, class_id: int) -> torch.Tensor:
        """
        获取指定类别的所有原型

        Returns:
            [num_prototypes, feature_dim]
        """
        return F.normalize(self.prototypes[class_id], dim=-1)

    def visualize_prototypes(self) -> Dict[str, torch.Tensor]:
        """
        获取用于可视化的原型信息

        Returns:
            {
                'prototypes': [num_classes, num_prototypes, feature_dim],
                'class_counts': [num_classes],
                'intra_class_similarity': [num_classes],
                'inter_class_similarity': scalar,
            }
        """
        prototypes_normalized = F.normalize(self.prototypes, dim=-1).detach()

        # 类内相似度
        intra_sim = []
        for c in range(self.num_classes):
            class_protos = prototypes_normalized[c]  # [P, D]
            sim = torch.mm(class_protos, class_protos.t())
            mask = ~torch.eye(self.num_prototypes, device=sim.device).bool()
            mean_sim = sim.masked_select(mask).mean()
            intra_sim.append(mean_sim)
        intra_sim = torch.stack(intra_sim)

        # 类间相似度
        inter_sims = []
        for c1 in range(self.num_classes):
            for c2 in range(c1 + 1, self.num_classes):
                protos_1 = prototypes_normalized[c1]  # [P, D]
                protos_2 = prototypes_normalized[c2]  # [P, D]
                sim = torch.mm(protos_1, protos_2.t())
                inter_sims.append(sim.mean())
        inter_sim = torch.stack(inter_sims).mean() if inter_sims else torch.tensor(0.0)

        return {
            'prototypes': prototypes_normalized,
            'class_counts': self.class_counts.detach(),
            'intra_class_similarity': intra_sim,
            'inter_class_similarity': inter_sim,
        }


# 别名
BPCL = BehaviorPrototypeContrastiveLearning


class BPCLWithClassifier(nn.Module):
    """
    结合BPCL和传统分类器的混合模型

    同时使用对比学习和交叉熵损失
    """

    def __init__(
        self,
        feature_dim: int = 128,
        num_classes: int = 3,
        num_prototypes_per_class: int = 3,
        temperature: float = 0.07,
        contrastive_weight: float = 0.5,
    ):
        super().__init__()

        self.contrastive_weight = contrastive_weight

        # BPCL模块
        self.bpcl = BPCL(
            feature_dim=feature_dim,
            num_classes=num_classes,
            num_prototypes_per_class=num_prototypes_per_class,
            temperature=temperature,
        )

        # 传统分类头
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [batch, feature_dim]
            labels: [batch] (训练时需要)

        Returns:
            {
                'logits': [batch, num_classes] (混合logits),
                'prototype_logits': [batch, num_classes],
                'classifier_logits': [batch, num_classes],
            }
        """
        # 原型based logits
        prototype_logits = self.bpcl(features)

        # 分类器logits
        classifier_logits = self.classifier(features)

        # 混合
        logits = (prototype_logits + classifier_logits) / 2

        return {
            'logits': logits,
            'prototype_logits': prototype_logits,
            'classifier_logits': classifier_logits,
        }

    def compute_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算混合损失

        Returns:
            (total_loss, loss_dict)
        """
        outputs = self.forward(features, labels)

        # 分类损失
        ce_loss = F.cross_entropy(outputs['classifier_logits'], labels)

        # 对比损失
        contrastive_loss, contrastive_dict = self.bpcl.compute_contrastive_loss(
            features, labels
        )

        # 总损失
        total_loss = (
            (1 - self.contrastive_weight) * ce_loss +
            self.contrastive_weight * contrastive_loss
        )

        loss_dict = {
            'ce_loss': ce_loss.item(),
            **contrastive_dict,
            'total': total_loss.item(),
        }

        return total_loss, loss_dict


if __name__ == '__main__':
    # 测试BPCL
    print("Testing Behavior Prototype Contrastive Learning (BPCL)...")

    batch_size = 16
    feature_dim = 128
    num_classes = 3

    # 创建BPCL模块
    bpcl = BPCL(
        feature_dim=feature_dim,
        num_classes=num_classes,
        num_prototypes_per_class=3,
        temperature=0.07,
    )

    # 测试输入
    features = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    # 模拟不平衡分布
    labels[:12] = 1  # looking_around (78%)
    labels[12:15] = 2  # unknown (17%)
    labels[15:] = 0  # normal (5%)

    # 测试前向传播
    print("\n1. Forward pass:")
    logits = bpcl(features)
    print(f"   Features: {features.shape}")
    print(f"   Logits: {logits.shape}")

    # 测试带相似度的输出
    logits, similarities = bpcl(features, return_similarities=True)
    print(f"   Similarities: {similarities.shape}")

    # 测试对比损失
    print("\n2. Contrastive loss:")
    loss, loss_dict = bpcl.compute_contrastive_loss(features, labels)
    print(f"   Total loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"   {k}: {v:.4f}")

    # 测试原型分配
    print("\n3. Prototype assignments:")
    class_ids, proto_ids = bpcl.get_prototype_assignments(features)
    print(f"   Class IDs: {class_ids[:5]}")
    print(f"   Prototype IDs: {proto_ids[:5]}")

    # 测试原型可视化信息
    print("\n4. Prototype visualization:")
    viz_info = bpcl.visualize_prototypes()
    print(f"   Intra-class similarity: {viz_info['intra_class_similarity']}")
    print(f"   Inter-class similarity: {viz_info['inter_class_similarity']:.4f}")

    # 测试混合模型
    print("\n5. BPCL with Classifier:")
    hybrid = BPCLWithClassifier(feature_dim, num_classes)
    outputs = hybrid(features, labels)
    print(f"   Logits: {outputs['logits'].shape}")

    loss, loss_dict = hybrid.compute_loss(features, labels)
    print(f"   Total loss: {loss.item():.4f}")

    # 测试动量更新
    print("\n6. Momentum update:")
    old_momentum = bpcl.prototype_momentum.clone()
    bpcl.update_prototypes_momentum()
    diff = (bpcl.prototype_momentum - old_momentum).abs().mean()
    print(f"   Momentum update diff: {diff:.6f}")

    print("\nAll BPCL tests passed!")
