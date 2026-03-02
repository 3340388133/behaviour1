#!/usr/bin/env python3
"""
DGCMF (动态门控跨模态融合) 单元测试
"""

import sys
from pathlib import Path
import pytest
import torch

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.recognition.fusion import DGCMF, DynamicGatedCrossModalFusion
from src.recognition.fusion.dynamic_gated_fusion import (
    ModalityQualityEstimator,
    CrossModalAttention,
    GatedFusionUnit,
    SequenceDGCMF,
)


class TestModalityQualityEstimator:
    """模态质量评估器测试"""

    @pytest.fixture
    def estimator(self):
        return ModalityQualityEstimator(input_dim=128, hidden_dim=64)

    def test_output_shape_2d(self, estimator):
        """测试2D输入输出形状"""
        x = torch.randn(8, 128)
        quality = estimator(x)
        assert quality.shape == (8, 1)

    def test_output_shape_3d(self, estimator):
        """测试3D输入输出形状"""
        x = torch.randn(8, 32, 128)
        quality = estimator(x)
        assert quality.shape == (8, 32, 1)

    def test_output_range(self, estimator):
        """测试输出范围 [0, 1]"""
        x = torch.randn(16, 128)
        quality = estimator(x)
        assert (quality >= 0).all() and (quality <= 1).all()


class TestCrossModalAttention:
    """跨模态注意力测试"""

    @pytest.fixture
    def attention(self):
        return CrossModalAttention(dim=128, num_heads=4)

    def test_output_shape(self, attention):
        """测试输出形状"""
        query = torch.randn(8, 128)
        key_value = torch.randn(8, 128)
        output = attention(query, key_value)
        assert output.shape == query.shape

    def test_with_quality_mask(self, attention):
        """测试带质量掩码"""
        query = torch.randn(8, 128)
        key_value = torch.randn(8, 128)
        quality_mask = torch.rand(8, 1)

        output = attention(query, key_value, quality_mask)
        assert output.shape == query.shape


class TestGatedFusionUnit:
    """门控融合单元测试"""

    @pytest.fixture
    def fusion_unit(self):
        return GatedFusionUnit(dim=128)

    def test_output_shape(self, fusion_unit):
        """测试输出形状"""
        feat1 = torch.randn(8, 128)
        feat2 = torch.randn(8, 128)
        output = fusion_unit(feat1, feat2)
        assert output.shape == (8, 128)

    def test_with_weights(self, fusion_unit):
        """测试带权重"""
        feat1 = torch.randn(8, 128)
        feat2 = torch.randn(8, 128)
        weight1 = torch.rand(8, 1)
        weight2 = torch.rand(8, 1)

        output = fusion_unit(feat1, feat2, weight1, weight2)
        assert output.shape == (8, 128)


class TestDGCMF:
    """DGCMF主模块测试"""

    @pytest.fixture
    def dgcmf(self):
        return DGCMF(
            pose_dim=128,
            appearance_dim=512,
            motion_dim=64,
            hidden_dim=128,
            num_heads=4,
            use_quality_estimation=True,
        )

    @pytest.fixture
    def sample_inputs(self):
        """创建测试输入"""
        batch_size = 8
        return {
            'pose': torch.randn(batch_size, 128),
            'appearance': torch.randn(batch_size, 512),
            'motion': torch.randn(batch_size, 64),
        }

    def test_full_multimodal(self, dgcmf, sample_inputs):
        """测试完整三模态融合"""
        output = dgcmf(
            sample_inputs['pose'],
            sample_inputs['appearance'],
            sample_inputs['motion'],
        )
        assert output.shape == (8, 128)

    def test_with_quality_scores(self, dgcmf, sample_inputs):
        """测试返回质量分数"""
        output, quality = dgcmf(
            sample_inputs['pose'],
            sample_inputs['appearance'],
            sample_inputs['motion'],
            return_quality_scores=True,
        )

        assert 'pose' in quality
        assert 'appearance' in quality
        assert 'motion' in quality

    def test_pose_only(self, dgcmf, sample_inputs):
        """测试仅姿态输入"""
        output = dgcmf(sample_inputs['pose'], None, None)
        assert output.shape == (8, 128)

    def test_pose_appearance(self, dgcmf, sample_inputs):
        """测试姿态+外观"""
        output = dgcmf(sample_inputs['pose'], sample_inputs['appearance'], None)
        assert output.shape == (8, 128)

    def test_pose_motion(self, dgcmf, sample_inputs):
        """测试姿态+运动"""
        output = dgcmf(sample_inputs['pose'], None, sample_inputs['motion'])
        assert output.shape == (8, 128)

    def test_get_modality_weights(self, dgcmf, sample_inputs):
        """测试获取模态权重"""
        weights = dgcmf.get_modality_weights(
            sample_inputs['pose'],
            sample_inputs['appearance'],
            sample_inputs['motion'],
        )

        assert 'pose' in weights
        assert 'appearance' in weights
        assert 'motion' in weights

    def test_gradient_flow(self, dgcmf, sample_inputs):
        """测试梯度流动"""
        for key in sample_inputs:
            sample_inputs[key].requires_grad = True

        output = dgcmf(
            sample_inputs['pose'],
            sample_inputs['appearance'],
            sample_inputs['motion'],
        )

        loss = output.sum()
        loss.backward()

        for key, tensor in sample_inputs.items():
            assert tensor.grad is not None, f"{key} has no gradient"

    def test_no_quality_estimation(self, sample_inputs):
        """测试不使用质量估计"""
        dgcmf = DGCMF(
            pose_dim=128,
            appearance_dim=512,
            motion_dim=64,
            use_quality_estimation=False,
        )

        output = dgcmf(
            sample_inputs['pose'],
            sample_inputs['appearance'],
            sample_inputs['motion'],
        )
        assert output.shape == (8, 128)


class TestSequenceDGCMF:
    """序列级DGCMF测试"""

    @pytest.fixture
    def seq_dgcmf(self):
        return SequenceDGCMF(
            pose_dim=128,
            appearance_dim=512,
            motion_dim=64,
            hidden_dim=128,
        )

    def test_sequence_output(self, seq_dgcmf):
        """测试序列输出"""
        batch_size = 4
        seq_len = 32

        pose_seq = torch.randn(batch_size, seq_len, 128)
        appearance_seq = torch.randn(batch_size, seq_len, 512)
        motion_seq = torch.randn(batch_size, seq_len, 64)

        output = seq_dgcmf(pose_seq, appearance_seq, motion_seq)
        assert output.shape == (batch_size, seq_len, 128)

    def test_sequence_pose_only(self, seq_dgcmf):
        """测试序列仅姿态"""
        pose_seq = torch.randn(4, 32, 128)
        output = seq_dgcmf(pose_seq, None, None)
        assert output.shape == (4, 32, 128)


class TestDGCMFEdgeCases:
    """DGCMF边界情况测试"""

    def test_single_sample(self):
        """测试单样本"""
        dgcmf = DGCMF(pose_dim=64, appearance_dim=128, motion_dim=32, hidden_dim=64)
        pose = torch.randn(1, 64)
        appearance = torch.randn(1, 128)
        motion = torch.randn(1, 32)

        output = dgcmf(pose, appearance, motion)
        assert output.shape == (1, 64)

    def test_different_batch_sizes(self):
        """测试不同batch size"""
        dgcmf = DGCMF(pose_dim=64, appearance_dim=128, motion_dim=32, hidden_dim=64)

        for batch_size in [1, 2, 4, 8, 16]:
            pose = torch.randn(batch_size, 64)
            appearance = torch.randn(batch_size, 128)
            motion = torch.randn(batch_size, 32)

            output = dgcmf(pose, appearance, motion)
            assert output.shape == (batch_size, 64)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
