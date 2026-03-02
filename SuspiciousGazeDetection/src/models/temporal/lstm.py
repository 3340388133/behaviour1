"""
LSTM-based Temporal Models

Bidirectional LSTM for capturing temporal dependencies in
head pose sequences.

创新点：
1. 双向LSTM捕获前后文时序依赖
2. 多层堆叠增强表达能力
3. 残差连接防止梯度消失
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class LSTMCell(nn.Module):
    """
    Single LSTM cell with optional layer normalization.

    Args:
        input_size: Input feature dimension
        hidden_size: Hidden state dimension
        use_layer_norm: Whether to use layer normalization
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_layer_norm = use_layer_norm

        # Gates: input, forget, cell, output
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)

        if use_layer_norm:
            self.ln_cell = nn.LayerNorm(hidden_size)
            self.ln_hidden = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor (B, input_size)
            state: Previous (hidden, cell) state

        Returns:
            output: Hidden state (B, hidden_size)
            state: New (hidden, cell) state
        """
        B = x.shape[0]

        if state is None:
            h = torch.zeros(B, self.hidden_size, device=x.device)
            c = torch.zeros(B, self.hidden_size, device=x.device)
        else:
            h, c = state

        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)

        # Compute gates
        gates = self.gates(combined)
        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        g = torch.tanh(g)     # Cell candidate
        o = torch.sigmoid(o)  # Output gate

        # Update cell state
        c_new = f * c + i * g
        if self.use_layer_norm:
            c_new = self.ln_cell(c_new)

        # Update hidden state
        h_new = o * torch.tanh(c_new)
        if self.use_layer_norm:
            h_new = self.ln_hidden(h_new)

        return h_new, (h_new, c_new)


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM for temporal sequence modeling.

    创新点：
    1. 双向处理捕获完整时序上下文
    2. 残差连接增强梯度流动
    3. 层归一化稳定训练

    Args:
        input_size: Input feature dimension (e.g., pose features)
        hidden_size: LSTM hidden dimension
        num_layers: Number of stacked LSTM layers
        dropout: Dropout rate between layers
        bidirectional: Whether to use bidirectional LSTM
        use_residual: Whether to use residual connections
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_residual = use_residual
        self.num_directions = 2 if bidirectional else 1

        # Input projection (for residual connection)
        if use_residual and input_size != hidden_size * self.num_directions:
            self.input_proj = nn.Linear(
                input_size, hidden_size * self.num_directions
            )
        else:
            self.input_proj = None

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)

        # Dropout
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
            final: Final hidden state (B, hidden_size * num_directions)
        """
        B, T, _ = x.shape

        # Pack sequence if lengths provided
        if lengths is not None:
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output_packed, (h_n, c_n) = self.lstm(x_packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output_packed, batch_first=True, total_length=T
            )
        else:
            output, (h_n, c_n) = self.lstm(x)

        # Layer normalization
        output = self.layer_norm(output)

        # Residual connection
        if self.use_residual:
            if self.input_proj is not None:
                residual = self.input_proj(x)
            else:
                residual = x
            output = output + residual

        # Final hidden state (concatenate forward and backward)
        if self.bidirectional:
            final = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            final = h_n[-1]

        output = self.dropout(output)

        return output, final

    def get_output_size(self) -> int:
        """Get output feature dimension."""
        return self.hidden_size * self.num_directions


class StackedBiLSTM(nn.Module):
    """
    Stacked Bidirectional LSTM with inter-layer residuals.

    创新点：深层LSTM之间加入残差连接，
    每层都能直接接收原始信息，防止信息丢失。

    Args:
        input_size: Input feature dimension
        hidden_size: Hidden dimension
        num_layers: Number of BiLSTM layers
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size * 2
            self.layers.append(
                nn.LSTM(
                    in_size,
                    hidden_size,
                    batch_first=True,
                    bidirectional=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_size * 2))

        self.dropout = nn.Dropout(dropout)

        # Projection for first layer residual
        if input_size != hidden_size * 2:
            self.input_proj = nn.Linear(input_size, hidden_size * 2)
        else:
            self.input_proj = None

        self.output_size = hidden_size * 2

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input sequence (B, T, input_size)

        Returns:
            output: Final sequence output
            final: Final hidden state
        """
        for i, (lstm, norm) in enumerate(zip(self.layers, self.norms)):
            residual = x if i > 0 else (
                self.input_proj(x) if self.input_proj else x
            )

            x, (h_n, _) = lstm(x)
            x = norm(x)
            x = self.dropout(x)

            # Residual connection (except first layer if dims don't match)
            if i > 0 or self.input_proj is not None:
                x = x + residual

        # Final hidden state
        final = torch.cat([h_n[-2], h_n[-1]], dim=1)

        return x, final
