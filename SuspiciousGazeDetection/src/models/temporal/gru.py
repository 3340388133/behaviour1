"""
GRU-based Temporal Models

Bidirectional GRU as a lighter alternative to LSTM.
GRU has fewer parameters while maintaining competitive performance.

创新点：
1. 双向GRU捕获前后文依赖
2. 相比LSTM参数更少，训练更快
3. 适合实时应用场景
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class BiGRU(nn.Module):
    """
    Bidirectional GRU for temporal sequence modeling.

    GRU优势：
    - 参数比LSTM少约25%
    - 训练速度更快
    - 在短序列上性能与LSTM相当

    Args:
        input_size: Input feature dimension
        hidden_size: GRU hidden dimension
        num_layers: Number of stacked GRU layers
        dropout: Dropout rate
        bidirectional: Whether to use bidirectional GRU
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input sequence (B, T, input_size)
            lengths: Sequence lengths for packed sequences

        Returns:
            output: Sequence output (B, T, hidden_size * num_directions)
            final: Final hidden state
        """
        B, T, _ = x.shape

        if lengths is not None:
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output_packed, h_n = self.gru(x_packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output_packed, batch_first=True, total_length=T
            )
        else:
            output, h_n = self.gru(x)

        output = self.layer_norm(output)
        output = self.dropout(output)

        # Final hidden state
        if self.bidirectional:
            final = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            final = h_n[-1]

        return output, final

    def get_output_size(self) -> int:
        """Get output feature dimension."""
        return self.hidden_size * self.num_directions


class ConvGRU(nn.Module):
    """
    Convolutional GRU for spatiotemporal modeling.

    创新点：在GRU中使用卷积操作，保持空间结构，
    适合处理视频帧序列的时空特征。

    Args:
        input_channels: Input feature channels
        hidden_channels: Hidden state channels
        kernel_size: Convolution kernel size
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # Reset gate
        self.conv_xr = nn.Conv2d(
            input_channels, hidden_channels, kernel_size, padding=padding
        )
        self.conv_hr = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size, padding=padding
        )

        # Update gate
        self.conv_xz = nn.Conv2d(
            input_channels, hidden_channels, kernel_size, padding=padding
        )
        self.conv_hz = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size, padding=padding
        )

        # Candidate hidden state
        self.conv_xn = nn.Conv2d(
            input_channels, hidden_channels, kernel_size, padding=padding
        )
        self.conv_hn = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size, padding=padding
        )

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W)
            h: Previous hidden state (B, hidden_channels, H, W)

        Returns:
            New hidden state
        """
        B, _, H, W = x.shape

        if h is None:
            h = torch.zeros(
                B, self.hidden_channels, H, W,
                device=x.device, dtype=x.dtype
            )

        # Reset gate
        r = torch.sigmoid(self.conv_xr(x) + self.conv_hr(h))

        # Update gate
        z = torch.sigmoid(self.conv_xz(x) + self.conv_hz(h))

        # Candidate hidden state
        n = torch.tanh(self.conv_xn(x) + r * self.conv_hn(h))

        # New hidden state
        h_new = (1 - z) * n + z * h

        return h_new


class ConvGRUSequence(nn.Module):
    """
    Convolutional GRU for processing video sequences.

    Args:
        input_channels: Input feature channels
        hidden_channels: Hidden state channels
        num_layers: Number of ConvGRU layers
        kernel_size: Convolution kernel size
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels
            self.cells.append(
                ConvGRU(in_ch, hidden_channels, kernel_size)
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input sequence (B, T, C, H, W)

        Returns:
            output: Sequence of hidden states (B, T, hidden_channels, H, W)
            final: Final hidden state (B, hidden_channels, H, W)
        """
        B, T, C, H, W = x.shape

        # Initialize hidden states
        states = [None] * self.num_layers
        outputs = []

        for t in range(T):
            input_t = x[:, t]

            for i, cell in enumerate(self.cells):
                states[i] = cell(input_t, states[i])
                input_t = states[i]

            outputs.append(states[-1])

        output = torch.stack(outputs, dim=1)
        final = states[-1]

        return output, final
