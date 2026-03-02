#!/usr/bin/env python3
"""
PAPE (周期感知位置编码) 单元测试
"""

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.recognition.position_encoding import PAPE, PeriodicAwarePositionalEncoding
from src.recognition.position_encoding.periodic_aware_pe import PAPETransformerEncoderLayer


class TestPAPE:
    """PAPE单元测试"""

    @pytest.fixture
    def pape(self):
        """创建PAPE实例"""
        return PAPE(
            d_model=128,
            max_len=512,
            periods=[15, 30, 60],
            use_relative_bias=True,
            num_heads=8,
            dropout=0.0,  # 测试时关闭dropout
        )

    @pytest.fixture
    def sample_input(self):
        """创建测试输入"""
        batch_size = 4
        seq_len = 32
        d_model = 128
        return torch.randn(batch_size, seq_len, d_model)

    def test_output_shape(self, pape, sample_input):
        """测试输出形状"""
        output = pape(sample_input)
        assert output.shape == sample_input.shape

    def test_relative_bias_shape(self, pape, sample_input):
        """测试相对位置偏置形状"""
        output, rel_bias = pape(sample_input, return_relative_bias=True)
        seq_len = sample_input.shape[1]
        assert rel_bias.shape == (8, seq_len, seq_len)

    def test_get_attention_bias(self, pape):
        """测试获取注意力偏置"""
        seq_len = 32
        bias = pape.get_attention_bias(seq_len)
        assert bias is not None
        assert bias.shape == (8, seq_len, seq_len)

    def test_no_relative_bias(self, sample_input):
        """测试不使用相对位置偏置"""
        pape_no_bias = PAPE(
            d_model=128,
            use_relative_bias=False,
        )
        output = pape_no_bias(sample_input)
        assert output.shape == sample_input.shape

        bias = pape_no_bias.get_attention_bias(32)
        assert bias is None

    def test_different_periods(self):
        """测试不同周期设置"""
        for periods in [[10], [10, 20], [10, 20, 40, 80]]:
            pape = PAPE(d_model=128, periods=periods)
            x = torch.randn(2, 16, 128)
            output = pape(x)
            assert output.shape == x.shape

    def test_gradient_flow(self, pape, sample_input):
        """测试梯度流动"""
        sample_input.requires_grad = True
        output = pape(sample_input)
        loss = output.sum()
        loss.backward()

        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()

    def test_periodic_encoding_computation(self, pape):
        """测试周期编码计算"""
        seq_len = 32
        device = torch.device('cpu')
        dtype = torch.float32

        periodic_enc = pape._compute_periodic_encoding(seq_len, device, dtype)

        # 检查形状
        expected_dim = len(pape.periods) * pape.period_dim
        assert periodic_enc.shape == (seq_len, expected_dim)

        # 检查数值范围（应该在合理范围内）
        assert not torch.isnan(periodic_enc).any()
        assert not torch.isinf(periodic_enc).any()


class TestPAPETransformerEncoderLayer:
    """PAPE Transformer Encoder Layer测试"""

    @pytest.fixture
    def layer(self):
        """创建层实例"""
        return PAPETransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            dropout=0.0,
        )

    @pytest.fixture
    def sample_input(self):
        """创建测试输入"""
        return torch.randn(4, 32, 128)

    def test_output_shape(self, layer, sample_input):
        """测试输出形状"""
        output = layer(sample_input)
        assert output.shape == sample_input.shape

    def test_with_relative_bias(self, layer, sample_input):
        """测试带相对位置偏置"""
        seq_len = sample_input.shape[1]
        rel_bias = torch.randn(8, seq_len, seq_len)
        output = layer(sample_input, relative_bias=rel_bias)
        assert output.shape == sample_input.shape

    def test_with_padding_mask(self, layer, sample_input):
        """测试带padding mask"""
        batch_size, seq_len = sample_input.shape[:2]
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, -5:] = True  # 最后5个位置被mask

        output = layer(sample_input, src_key_padding_mask=mask)
        assert output.shape == sample_input.shape

    def test_gradient_flow(self, layer, sample_input):
        """测试梯度流动"""
        sample_input.requires_grad = True
        output = layer(sample_input)
        loss = output.sum()
        loss.backward()

        assert sample_input.grad is not None

        # 检查层参数梯度
        for name, param in layer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} has no gradient"


class TestPAPEIntegration:
    """PAPE集成测试"""

    def test_full_pipeline(self):
        """测试完整流程"""
        batch_size = 4
        seq_len = 32
        d_model = 128
        num_layers = 2

        # 创建PAPE
        pape = PAPE(d_model=d_model, num_heads=8)

        # 创建多个PAPE层
        layers = nn.ModuleList([
            PAPETransformerEncoderLayer(d_model=d_model, nhead=8)
            for _ in range(num_layers)
        ])

        # 输入
        x = torch.randn(batch_size, seq_len, d_model)

        # 前向传播
        x, rel_bias = pape(x, return_relative_bias=True)

        for layer in layers:
            x = layer(x, relative_bias=rel_bias)

        assert x.shape == (batch_size, seq_len, d_model)

    def test_batch_consistency(self):
        """测试batch一致性 - 不同batch size应该产生一致的编码"""
        pape = PAPE(d_model=128, dropout=0.0)

        # 单样本
        x1 = torch.randn(1, 32, 128)
        out1 = pape(x1)

        # 多样本（重复相同输入）
        x2 = x1.repeat(4, 1, 1)
        out2 = pape(x2)

        # 检查输出一致性
        assert torch.allclose(out1[0], out2[0], atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
