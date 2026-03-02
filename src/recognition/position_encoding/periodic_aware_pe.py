#!/usr/bin/env python3
"""
创新点1: 周期感知位置编码 (Periodic-Aware Positional Encoding, PAPE)

理论依据:
- "四处张望"等可疑行为具有明显的周期性模式（如2秒内>=3次方向切换）
- 标准正弦位置编码仅编码绝对位置，无法显式捕捉周期性行为
- PAPE通过多尺度周期编码 + 相对位置偏置，显式建模行为的周期性特征

技术方案:
1. 标准正弦PE: 编码绝对时间位置
2. 可学习周期编码: 0.5s/1.0s/2.0s多尺度，捕捉不同频率的周期性行为
3. 相对位置偏置: 增强时序关系建模，适用于注意力机制
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class PeriodicAwarePositionalEncoding(nn.Module):
    """
    周期感知位置编码 (PAPE)

    融合三种位置信息:
    1. 标准正弦PE (绝对位置)
    2. 多尺度周期编码 (周期性行为)
    3. 相对位置偏置 (时序关系)

    Args:
        d_model: 模型维度
        max_len: 最大序列长度
        dropout: Dropout比例
        periods: 周期列表 (以帧数为单位)，默认 [15, 30, 60] 对应0.5s/1.0s/2.0s@30fps
        use_relative_bias: 是否使用相对位置偏置
        num_heads: 注意力头数 (用于相对位置偏置)
    """

    def __init__(
        self,
        d_model: int = 128,
        max_len: int = 512,
        dropout: float = 0.1,
        periods: Optional[list] = None,
        use_relative_bias: bool = True,
        num_heads: int = 8,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.use_relative_bias = use_relative_bias
        self.num_heads = num_heads

        # 默认周期: 0.5s, 1.0s, 2.0s @ 30fps
        self.periods = periods if periods is not None else [15, 30, 60]
        self.num_periods = len(self.periods)

        self.dropout = nn.Dropout(p=dropout)

        # ========== 1. 标准正弦位置编码 ==========
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # [max_len, d_model]

        # ========== 2. 可学习多尺度周期编码 ==========
        # 每个周期对应 d_model // num_periods 维度
        self.period_dim = d_model // (self.num_periods + 1)  # +1 给标准PE保留空间

        # 周期相位偏移 (可学习)
        self.period_phases = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, self.period_dim) * 0.02)
            for _ in self.periods
        ])

        # 周期幅度权重 (可学习)
        self.period_amplitudes = nn.ParameterList([
            nn.Parameter(torch.ones(1, 1, self.period_dim))
            for _ in self.periods
        ])

        # 融合投影层
        total_dim = d_model + self.num_periods * self.period_dim
        self.fusion_proj = nn.Linear(total_dim, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)

        # ========== 3. 相对位置偏置 ==========
        if use_relative_bias:
            # 相对位置偏置表: [-max_len+1, max_len-1] -> num_heads
            self.relative_bias_table = nn.Parameter(
                torch.zeros(2 * max_len - 1, num_heads)
            )
            nn.init.trunc_normal_(self.relative_bias_table, std=0.02)

            # 预计算相对位置索引
            coords = torch.arange(max_len)
            relative_coords = coords.unsqueeze(0) - coords.unsqueeze(1)  # [max_len, max_len]
            relative_coords = relative_coords + max_len - 1  # 偏移使索引非负
            self.register_buffer('relative_position_index', relative_coords)

    def _compute_periodic_encoding(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        计算多尺度周期编码

        Returns:
            [seq_len, num_periods * period_dim]
        """
        position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)

        periodic_encodings = []

        for i, period in enumerate(self.periods):
            # 计算周期位置: pos / period * 2π
            freq = 2 * math.pi / period
            phase = self.period_phases[i]  # [1, 1, period_dim]
            amplitude = self.period_amplitudes[i]  # [1, 1, period_dim]

            # 生成周期编码基
            dim_indices = torch.arange(self.period_dim, device=device, dtype=dtype)
            # 不同维度使用不同的相位偏移
            dim_phase = dim_indices / self.period_dim * math.pi

            # [seq_len, period_dim]
            periodic_enc = torch.sin(position * freq + dim_phase.unsqueeze(0))
            periodic_enc = periodic_enc + torch.cos(position * freq + dim_phase.unsqueeze(0))

            # 应用可学习的相位和幅度
            periodic_enc = periodic_enc.unsqueeze(0)  # [1, seq_len, period_dim]
            periodic_enc = periodic_enc * amplitude + phase
            periodic_enc = periodic_enc.squeeze(0)  # [seq_len, period_dim]

            periodic_encodings.append(periodic_enc)

        return torch.cat(periodic_encodings, dim=-1)  # [seq_len, num_periods * period_dim]

    def forward(
        self,
        x: torch.Tensor,
        return_relative_bias: bool = False,
    ):
        """
        前向传播

        Args:
            x: [batch, seq_len, d_model] 输入特征
            return_relative_bias: 是否返回相对位置偏置矩阵

        Returns:
            如果 return_relative_bias=False:
                [batch, seq_len, d_model] 添加位置编码后的特征
            如果 return_relative_bias=True:
                (features, relative_bias)
                - features: [batch, seq_len, d_model]
                - relative_bias: [num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # 1. 标准正弦PE
        sinusoidal_pe = self.pe[:seq_len, :].to(dtype)  # [seq_len, d_model]

        # 2. 多尺度周期编码
        periodic_pe = self._compute_periodic_encoding(seq_len, device, dtype)

        # 3. 融合所有位置编码
        combined_pe = torch.cat([sinusoidal_pe, periodic_pe], dim=-1)  # [seq_len, total_dim]
        fused_pe = self.fusion_proj(combined_pe)  # [seq_len, d_model]
        fused_pe = self.fusion_norm(fused_pe)

        # 添加到输入
        output = x + fused_pe.unsqueeze(0)
        output = self.dropout(output)

        if return_relative_bias and self.use_relative_bias:
            # 计算相对位置偏置
            relative_bias = self._get_relative_bias(seq_len)
            return output, relative_bias

        return output

    def _get_relative_bias(self, seq_len: int) -> torch.Tensor:
        """
        获取相对位置偏置矩阵

        Returns:
            [num_heads, seq_len, seq_len]
        """
        # 提取需要的相对位置索引
        relative_position_index = self.relative_position_index[:seq_len, :seq_len].contiguous()

        # 查表获取偏置
        relative_bias = self.relative_bias_table[relative_position_index.reshape(-1)]
        relative_bias = relative_bias.view(seq_len, seq_len, -1)
        relative_bias = relative_bias.permute(2, 0, 1).contiguous()  # [num_heads, seq_len, seq_len]

        return relative_bias

    def get_attention_bias(self, seq_len: int) -> Optional[torch.Tensor]:
        """
        获取用于注意力计算的偏置项

        可直接加到注意力分数上: attn_scores + bias

        Returns:
            [num_heads, seq_len, seq_len] 或 None
        """
        if not self.use_relative_bias:
            return None
        return self._get_relative_bias(seq_len)


# 别名
PAPE = PeriodicAwarePositionalEncoding


class PAPETransformerEncoderLayer(nn.Module):
    """
    集成PAPE的Transformer Encoder Layer

    在标准TransformerEncoderLayer基础上，支持相对位置偏置
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # 多头注意力 (手动实现以支持相对位置偏置)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization & Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()

    def forward(
        self,
        src: torch.Tensor,
        relative_bias: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src: [batch, seq_len, d_model]
            relative_bias: [num_heads, seq_len, seq_len] 相对位置偏置
            src_mask: 注意力掩码
            src_key_padding_mask: [batch, seq_len] padding掩码
        """
        # Self-attention with relative position bias
        src2 = self._self_attention(
            src, relative_bias, src_mask, src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

    def _self_attention(
        self,
        x: torch.Tensor,
        relative_bias: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """带相对位置偏置的自注意力"""
        batch_size, seq_len, _ = x.shape

        # QKV投影
        q = self.q_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim)

        # 转换维度: [batch, nhead, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 注意力分数: [batch, nhead, seq_len, seq_len]
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # 添加相对位置偏置
        if relative_bias is not None:
            attn_scores = attn_scores + relative_bias.unsqueeze(0)

        # 添加注意力掩码
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        # 添加padding掩码
        if key_padding_mask is not None:
            # [batch, 1, 1, seq_len]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # Softmax & Dropout
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 注意力输出
        attn_output = torch.matmul(attn_probs, v)  # [batch, nhead, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, seq_len, nhead, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, -1)  # [batch, seq_len, d_model]

        return self.out_proj(attn_output)


if __name__ == '__main__':
    # 测试PAPE
    print("Testing Periodic-Aware Positional Encoding (PAPE)...")

    batch_size = 4
    seq_len = 32
    d_model = 128

    # 创建PAPE模块
    pape = PAPE(
        d_model=d_model,
        max_len=512,
        periods=[15, 30, 60],  # 0.5s, 1.0s, 2.0s @ 30fps
        use_relative_bias=True,
        num_heads=8,
    )

    # 测试输入
    x = torch.randn(batch_size, seq_len, d_model)

    # 测试前向传播
    output = pape(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # 测试带相对位置偏置的输出
    output, rel_bias = pape(x, return_relative_bias=True)
    print(f"Relative bias shape: {rel_bias.shape}")

    # 测试注意力偏置
    attn_bias = pape.get_attention_bias(seq_len)
    print(f"Attention bias shape: {attn_bias.shape}")

    # 测试PAPE Transformer Layer
    print("\nTesting PAPE Transformer Encoder Layer...")
    layer = PAPETransformerEncoderLayer(d_model=d_model, nhead=8)

    output = layer(x, relative_bias=rel_bias)
    print(f"Layer output shape: {output.shape}")

    print("\nAll PAPE tests passed!")
