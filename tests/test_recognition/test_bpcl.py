#!/usr/bin/env python3
"""
BPCL (行为原型对比学习) 单元测试
"""

import sys
from pathlib import Path
import pytest
import torch

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.recognition.contrastive import BPCL, BehaviorPrototypeContrastiveLearning
from src.recognition.contrastive.behavior_prototype import BPCLWithClassifier


class TestBPCL:
    """BPCL单元测试"""

    @pytest.fixture
    def bpcl(self):
        """创建BPCL实例"""
        return BPCL(
            feature_dim=128,
            num_classes=3,
            num_prototypes_per_class=3,
            temperature=0.07,
            momentum=0.999,
            margin=0.5,
        )

    @pytest.fixture
    def sample_features(self):
        """创建测试特征"""
        return torch.randn(16, 128)

    @pytest.fixture
    def sample_labels(self):
        """创建测试标签"""
        labels = torch.zeros(16, dtype=torch.long)
        labels[:12] = 1  # 多数类
        labels[12:15] = 2
        labels[15:] = 0  # 少数类
        return labels

    def test_output_shape(self, bpcl, sample_features):
        """测试输出形状"""
        logits = bpcl(sample_features)
        assert logits.shape == (16, 3)

    def test_similarities_shape(self, bpcl, sample_features):
        """测试相似度矩阵形状"""
        logits, sims = bpcl(sample_features, return_similarities=True)
        assert sims.shape == (16, 3, 3)  # [batch, num_classes, num_prototypes]

    def test_contrastive_loss_output(self, bpcl, sample_features, sample_labels):
        """测试对比损失输出"""
        loss, loss_dict = bpcl.compute_contrastive_loss(sample_features, sample_labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # 标量
        assert not torch.isnan(loss)

        expected_keys = ['infonce', 'margin', 'diversity', 'hard_negative', 'total_contrastive']
        for key in expected_keys:
            assert key in loss_dict

    def test_prototype_assignment(self, bpcl, sample_features):
        """测试原型分配"""
        class_ids, proto_ids = bpcl.get_prototype_assignments(sample_features)

        assert class_ids.shape == (16,)
        assert proto_ids.shape == (16,)
        assert (class_ids >= 0).all() and (class_ids < 3).all()
        assert (proto_ids >= 0).all() and (proto_ids < 3).all()

    def test_get_class_prototypes(self, bpcl):
        """测试获取类别原型"""
        for c in range(3):
            protos = bpcl.get_class_prototypes(c)
            assert protos.shape == (3, 128)

            # 检查归一化
            norms = protos.norm(dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_momentum_update(self, bpcl):
        """测试动量更新"""
        old_momentum = bpcl.prototype_momentum.clone()
        old_prototypes = bpcl.prototypes.clone()

        # 模拟训练：修改prototypes
        with torch.no_grad():
            bpcl.prototypes.data += torch.randn_like(bpcl.prototypes) * 0.1

        bpcl.update_prototypes_momentum()

        # 检查momentum更新
        expected = 0.999 * old_momentum + 0.001 * bpcl.prototypes.data
        assert torch.allclose(bpcl.prototype_momentum, expected, atol=1e-5)

    def test_visualize_prototypes(self, bpcl):
        """测试原型可视化信息"""
        viz_info = bpcl.visualize_prototypes()

        assert 'prototypes' in viz_info
        assert 'class_counts' in viz_info
        assert 'intra_class_similarity' in viz_info
        assert 'inter_class_similarity' in viz_info

        assert viz_info['prototypes'].shape == (3, 3, 128)
        assert viz_info['intra_class_similarity'].shape == (3,)

    def test_gradient_flow(self, bpcl, sample_features, sample_labels):
        """测试梯度流动"""
        sample_features.requires_grad = True

        loss, _ = bpcl.compute_contrastive_loss(sample_features, sample_labels)
        loss.backward()

        # 检查特征梯度
        assert sample_features.grad is not None

        # 检查原型梯度
        assert bpcl.prototypes.grad is not None


class TestBPCLWithClassifier:
    """BPCLWithClassifier测试"""

    @pytest.fixture
    def hybrid_model(self):
        """创建混合模型"""
        return BPCLWithClassifier(
            feature_dim=128,
            num_classes=3,
            num_prototypes_per_class=3,
            contrastive_weight=0.5,
        )

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        features = torch.randn(8, 128)
        labels = torch.randint(0, 3, (8,))
        return features, labels

    def test_forward_output(self, hybrid_model, sample_data):
        """测试前向传播输出"""
        features, labels = sample_data
        outputs = hybrid_model(features, labels)

        assert 'logits' in outputs
        assert 'prototype_logits' in outputs
        assert 'classifier_logits' in outputs

        assert outputs['logits'].shape == (8, 3)

    def test_compute_loss(self, hybrid_model, sample_data):
        """测试损失计算"""
        features, labels = sample_data
        loss, loss_dict = hybrid_model.compute_loss(features, labels)

        assert isinstance(loss, torch.Tensor)
        assert 'ce_loss' in loss_dict
        assert 'total' in loss_dict

    def test_gradient_flow(self, hybrid_model, sample_data):
        """测试梯度流动"""
        features, labels = sample_data
        features.requires_grad = True

        loss, _ = hybrid_model.compute_loss(features, labels)
        loss.backward()

        assert features.grad is not None


class TestBPCLEdgeCases:
    """BPCL边界情况测试"""

    def test_single_sample(self):
        """测试单样本"""
        bpcl = BPCL(feature_dim=64, num_classes=2, num_prototypes_per_class=2)
        features = torch.randn(1, 64)
        labels = torch.tensor([0])

        logits = bpcl(features)
        assert logits.shape == (1, 2)

        loss, _ = bpcl.compute_contrastive_loss(features, labels)
        assert not torch.isnan(loss)

    def test_all_same_class(self):
        """测试所有样本同类别"""
        bpcl = BPCL(feature_dim=64, num_classes=3)
        features = torch.randn(8, 64)
        labels = torch.ones(8, dtype=torch.long)

        loss, _ = bpcl.compute_contrastive_loss(features, labels)
        assert not torch.isnan(loss)

    def test_different_batch_sizes(self):
        """测试不同batch size"""
        bpcl = BPCL(feature_dim=64, num_classes=3)

        for batch_size in [1, 2, 4, 8, 16, 32]:
            features = torch.randn(batch_size, 64)
            labels = torch.randint(0, 3, (batch_size,))

            logits = bpcl(features)
            assert logits.shape == (batch_size, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
