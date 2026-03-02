"""
Temporal Attention Mechanisms

Attention modules for analyzing time-series head pose data
and identifying suspicious gazing patterns.

创新点：
1. 时间注意力机制，自动关注异常时间节点
2. 多头注意力增强时序特征提取
3. 相对位置编码保持时间顺序信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class TemporalAttention(nn.Module):
    """
    Temporal Self-Attention Module

    创新点：对时序特征应用自注意力，自动学习哪些时间点
    的头部姿态变化最值得关注（可能是异常张望）。

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_relative_pos: Whether to use relative position encoding
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_relative_pos: bool = True,
        max_len: int = 512,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_relative_pos = use_relative_pos

        # Q, K, V projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

        # Relative position encoding
        if use_relative_pos:
            self.rel_pos_bias = nn.Parameter(
                torch.zeros(2 * max_len - 1, num_heads)
            )
            nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input sequence (B, T, dim)
            mask: Attention mask (B, T) - True for valid positions

        Returns:
            output: Attended features (B, T, dim)
            attn_weights: Attention weights (B, num_heads, T, T)
        """
        B, T, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)

        # Transpose for attention: (B, num_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        if self.use_relative_pos:
            rel_pos = self._get_relative_positions(T)
            attn = attn + rel_pos

        # Apply mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            attn = attn.masked_fill(~mask, float('-inf'))

        # Softmax
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = attn_weights @ v
        out = out.transpose(1, 2).reshape(B, T, -1)
        out = self.out_proj(out)

        return out, attn_weights

    def _get_relative_positions(self, length: int) -> torch.Tensor:
        """Compute relative position bias."""
        positions = torch.arange(length, device=self.rel_pos_bias.device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
        rel_pos = rel_pos + length - 1  # Shift to positive indices

        # Clip to valid range
        rel_pos = rel_pos.clamp(0, 2 * length - 2)

        bias = self.rel_pos_bias[rel_pos]  # (T, T, num_heads)
        return bias.permute(2, 0, 1).unsqueeze(0)  # (1, num_heads, T, T)


class TemporalTransformer(nn.Module):
    """
    Transformer encoder for temporal sequence modeling.

    创新点：完整的Transformer架构用于时序建模，
    相比RNN能更好地捕获长程依赖。

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout rate
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input sequence (B, T, dim)
            mask: Attention mask

        Returns:
            Encoded sequence (B, T, dim)
        """
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class TransformerBlock(nn.Module):
    """Single transformer block with attention and FFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = TemporalAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)

        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention with residual
        attn_out, _ = self.attn(self.norm1(x), mask)
        x = x + attn_out

        # FFN with residual
        x = x + self.mlp(self.norm2(x))

        return x


class GazePatternAttention(nn.Module):
    """
    Specialized attention for detecting gaze patterns.

    创新点：专门设计用于检测可疑张望行为的注意力模块，
    关注头部姿态角度的快速变化和重复模式。

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        window_size: Local attention window size
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        window_size: int = 10,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Global attention for overall pattern
        self.global_attn = TemporalAttention(dim, num_heads)

        # Local attention for rapid changes
        self.local_q = nn.Linear(dim, dim)
        self.local_k = nn.Linear(dim, dim)
        self.local_v = nn.Linear(dim, dim)

        # Pattern embedding
        self.pattern_embed = nn.Parameter(torch.randn(1, 3, dim))  # 3 patterns
        self.pattern_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # Output fusion
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Pose sequence features (B, T, dim)
            mask: Attention mask

        Returns:
            output: Enhanced features (B, T, dim)
            pattern_scores: Detected pattern scores (B, 3)
        """
        B, T, _ = x.shape

        # Global attention
        global_out, global_attn = self.global_attn(x, mask)

        # Local sliding window attention
        local_out = self._local_attention(x)

        # Fuse global and local
        fused = torch.cat([global_out, local_out], dim=-1)
        output = self.fusion(fused) + x  # Residual

        # Pattern matching
        pattern_embed = self.pattern_embed.expand(B, -1, -1)
        pooled = x.mean(dim=1, keepdim=True)  # (B, 1, dim)
        _, pattern_attn = self.pattern_attn(
            pooled, pattern_embed, pattern_embed
        )
        pattern_scores = pattern_attn.squeeze(1)  # (B, 3)

        return output, pattern_scores

    def _local_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Compute local sliding window attention."""
        B, T, _ = x.shape
        W = self.window_size

        # Pad sequence for sliding window
        pad_len = W // 2
        x_pad = F.pad(x, (0, 0, pad_len, pad_len))

        # Project to Q, K, V
        q = self.local_q(x)
        k = self.local_k(x_pad)
        v = self.local_v(x_pad)

        outputs = []
        for t in range(T):
            # Local window
            k_local = k[:, t:t + W]
            v_local = v[:, t:t + W]
            q_local = q[:, t:t + 1]

            # Attention in window
            attn = (q_local @ k_local.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = attn @ v_local

            outputs.append(out)

        return torch.cat(outputs, dim=1)
