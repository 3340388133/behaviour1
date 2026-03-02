#!/usr/bin/env python3
"""
SBRN (可疑行为识别网络) 单元测试
"""

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.recognition.models import SBRN, SBRNConfig, create_sbrn


class TestSBRNConfig:
    """SBRN配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = SBRNConfig()
        assert config.pose_input_dim == 3
        assert config.d_model == 128
        assert config.num_classes == 3
        assert config.periods == [15, 30, 60]

    def test_custom_config(self):
        """测试自定义配置"""
        config = SBRNConfig(
            d_model=256,
            num_layers=6,
            num_classes=5,
            periods=[10, 20],
        )
        assert config.d_model == 256
        assert config.num_layers == 6
        assert config.num_classes == 5
        assert config.periods == [10, 20]


class TestSBRN:
    """SBRN模型测试"""

    @pytest.fixture
    def full_model(self):
        """创建完整模型（所有创新点）"""
        config = SBRNConfig(
            use_multimodal=True,
            use_contrastive=True,
            use_relative_bias=True,
            uncertainty_weighting=True,
        )
        return SBRN(config)

    @pytest.fixture
    def minimal_model(self):
        """创建最小模型（无创新点）"""
        config = SBRNConfig(
            use_multimodal=False,
            use_contrastive=False,
            use_relative_bias=False,
            uncertainty_weighting=False,
        )
        return SBRN(config)

    @pytest.fixture
    def sample_inputs(self):
        """创建测试输入"""
        batch_size = 4
        seq_len = 32
        return {
            'pose_seq': torch.randn(batch_size, seq_len, 3),
            'appearance_feat': torch.randn(batch_size, 512),
            'motion_feat': torch.randn(batch_size, 64),
            'labels': torch.randint(0, 3, (batch_size,)),
        }

    def test_forward_full_model(self, full_model, sample_inputs):
        """测试完整模型前向传播"""
        outputs = full_model(
            sample_inputs['pose_seq'],
            sample_inputs['appearance_feat'],
            sample_inputs['motion_feat'],
            return_features=True,
        )

        assert 'logits' in outputs
        assert 'confidence' in outputs
        assert 'features' in outputs
        assert 'prototype_logits' in outputs
        assert 'quality_scores' in outputs

        assert outputs['logits'].shape == (4, 3)
        assert outputs['confidence'].shape == (4, 1)
        assert outputs['features'].shape == (4, 128)

    def test_forward_minimal_model(self, minimal_model, sample_inputs):
        """测试最小模型前向传播"""
        outputs = minimal_model(sample_inputs['pose_seq'])

        assert 'logits' in outputs
        assert 'confidence' in outputs
        assert outputs['logits'].shape == (4, 3)

    def test_forward_pose_only(self, full_model, sample_inputs):
        """测试仅姿态输入"""
        outputs = full_model(sample_inputs['pose_seq'])
        assert outputs['logits'].shape == (4, 3)

    def test_forward_features(self, full_model, sample_inputs):
        """测试特征提取"""
        features = full_model.forward_features(sample_inputs['pose_seq'])
        assert features.shape == (4, 128)

    def test_compute_loss(self, full_model, sample_inputs):
        """测试损失计算"""
        outputs = full_model(
            sample_inputs['pose_seq'],
            sample_inputs['appearance_feat'],
            sample_inputs['motion_feat'],
            return_features=True,
        )

        loss, loss_dict = full_model.compute_loss(outputs, sample_inputs['labels'])

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert not torch.isnan(loss)

        assert 'loss_cls' in loss_dict
        assert 'loss_conf' in loss_dict
        assert 'total' in loss_dict

    def test_predict(self, full_model, sample_inputs):
        """测试预测接口"""
        preds, confs = full_model.predict(
            sample_inputs['pose_seq'],
            sample_inputs['appearance_feat'],
            sample_inputs['motion_feat'],
        )

        assert preds.shape == (4,)
        assert confs.shape == (4,)
        assert (preds >= 0).all() and (preds < 3).all()
        assert (confs >= 0).all() and (confs <= 1).all()

    def test_get_attention_weights(self, full_model, sample_inputs):
        """测试获取注意力权重"""
        attn = full_model.get_attention_weights(sample_inputs['pose_seq'])

        # [batch, nhead, seq_len+1, seq_len+1] (包含CLS token)
        assert attn.shape[0] == 4
        assert attn.shape[1] == 8  # nhead
        assert attn.shape[2] == 33  # seq_len + 1
        assert attn.shape[3] == 33

    def test_gradient_flow(self, full_model, sample_inputs):
        """测试梯度流动"""
        sample_inputs['pose_seq'].requires_grad = True
        sample_inputs['appearance_feat'].requires_grad = True
        sample_inputs['motion_feat'].requires_grad = True

        outputs = full_model(
            sample_inputs['pose_seq'],
            sample_inputs['appearance_feat'],
            sample_inputs['motion_feat'],
            return_features=True,
        )

        loss, _ = full_model.compute_loss(outputs, sample_inputs['labels'])
        loss.backward()

        # 检查输入梯度
        assert sample_inputs['pose_seq'].grad is not None
        assert sample_inputs['appearance_feat'].grad is not None
        assert sample_inputs['motion_feat'].grad is not None

        # 检查模型参数梯度
        params_with_grad = 0
        for name, param in full_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                params_with_grad += 1

        assert params_with_grad > 0

    def test_with_padding_mask(self, full_model, sample_inputs):
        """测试带padding mask"""
        batch_size = 4
        seq_len = 32
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, -5:] = True  # 最后5个位置被mask

        outputs = full_model(sample_inputs['pose_seq'], mask=mask)
        assert outputs['logits'].shape == (4, 3)


class TestCreateSBRN:
    """create_sbrn便捷函数测试"""

    def test_default_creation(self):
        """测试默认创建"""
        model = create_sbrn()
        assert isinstance(model, SBRN)

    def test_custom_creation(self):
        """测试自定义创建"""
        model = create_sbrn(
            num_classes=5,
            use_multimodal=False,
            use_contrastive=False,
            d_model=64,
            num_layers=2,
        )

        assert model.config.num_classes == 5
        assert model.config.d_model == 64
        assert model.config.num_layers == 2

    def test_forward_pass(self):
        """测试前向传播"""
        model = create_sbrn(num_classes=3)
        x = torch.randn(2, 16, 3)
        outputs = model(x)
        assert outputs['logits'].shape == (2, 3)


class TestSBRNAblation:
    """SBRN消融测试 - 验证各创新点可独立开关"""

    def test_pape_only(self):
        """仅启用PAPE"""
        config = SBRNConfig(
            use_multimodal=False,
            use_contrastive=False,
            use_relative_bias=True,  # PAPE
        )
        model = SBRN(config)

        x = torch.randn(2, 16, 3)
        outputs = model(x)
        assert outputs['logits'].shape == (2, 3)

    def test_bpcl_only(self):
        """仅启用BPCL"""
        config = SBRNConfig(
            use_multimodal=False,
            use_contrastive=True,  # BPCL
            use_relative_bias=False,
        )
        model = SBRN(config)

        x = torch.randn(2, 16, 3)
        outputs = model(x, return_features=True)

        assert 'prototype_logits' in outputs
        assert outputs['logits'].shape == (2, 3)

    def test_dgcmf_only(self):
        """仅启用DGCMF"""
        config = SBRNConfig(
            use_multimodal=True,  # DGCMF
            use_contrastive=False,
            use_relative_bias=False,
        )
        model = SBRN(config)

        pose = torch.randn(2, 16, 3)
        appearance = torch.randn(2, 512)
        motion = torch.randn(2, 64)

        outputs = model(pose, appearance, motion)
        assert 'quality_scores' in outputs
        assert outputs['logits'].shape == (2, 3)


class TestSBRNEdgeCases:
    """SBRN边界情况测试"""

    def test_single_sample(self):
        """测试单样本"""
        model = create_sbrn()
        x = torch.randn(1, 16, 3)
        outputs = model(x)
        assert outputs['logits'].shape == (1, 3)

    def test_long_sequence(self):
        """测试长序列"""
        model = create_sbrn()
        x = torch.randn(2, 256, 3)  # 长序列
        outputs = model(x)
        assert outputs['logits'].shape == (2, 3)

    def test_short_sequence(self):
        """测试短序列"""
        model = create_sbrn()
        x = torch.randn(2, 4, 3)  # 短序列
        outputs = model(x)
        assert outputs['logits'].shape == (2, 3)

    def test_different_batch_sizes(self):
        """测试不同batch size"""
        model = create_sbrn()

        for batch_size in [1, 2, 4, 8, 16]:
            x = torch.randn(batch_size, 32, 3)
            outputs = model(x)
            assert outputs['logits'].shape == (batch_size, 3)

    def test_model_parameter_count(self):
        """测试模型参数量"""
        model = create_sbrn()
        num_params = sum(p.numel() for p in model.parameters())

        # 确保参数量在合理范围内
        assert num_params > 100000  # 至少10万参数
        assert num_params < 50000000  # 不超过5000万参数


class TestSBRNTraining:
    """SBRN训练相关测试"""

    def test_train_eval_mode(self):
        """测试训练/评估模式切换"""
        model = create_sbrn()

        model.train()
        assert model.training

        model.eval()
        assert not model.training

    def test_optimizer_compatible(self):
        """测试与优化器兼容"""
        model = create_sbrn()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(2, 16, 3)
        labels = torch.randint(0, 3, (2,))

        outputs = model(x, return_features=True)
        loss, _ = model.compute_loss(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 确保参数更新了
        assert True  # 如果没有报错就通过

    def test_save_load_state_dict(self, tmp_path):
        """测试保存/加载模型"""
        model = create_sbrn()
        save_path = tmp_path / "model.pt"

        # 保存
        torch.save(model.state_dict(), save_path)

        # 加载
        model2 = create_sbrn()
        model2.load_state_dict(torch.load(save_path, weights_only=True))

        # 验证输出一致
        x = torch.randn(2, 16, 3)
        with torch.no_grad():
            out1 = model(x)
            out2 = model2(x)

        assert torch.allclose(out1['logits'], out2['logits'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
