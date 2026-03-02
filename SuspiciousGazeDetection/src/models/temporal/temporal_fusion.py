"""
Temporal Feature Fusion Module

Combines RNN-based models with attention mechanisms for
comprehensive temporal modeling of head pose sequences.

创新点：
1. 多模型融合（LSTM + GRU + Attention）
2. 自适应权重学习
3. 层次化时序建模
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .lstm import BiLSTM
from .gru import BiGRU
from .temporal_attention import TemporalAttention, GazePatternAttention


class TemporalFusion(nn.Module):
    """
    Temporal Feature Fusion Network

    创新点：融合多种时序建模方法的优势：
    - BiLSTM: 长期依赖建模
    - BiGRU: 高效短期建模
    - Attention: 关键帧识别

    Args:
        input_size: Input feature dimension (pose features)
        hidden_size: Hidden dimension
        num_layers: Number of RNN layers
        dropout: Dropout rate
        fusion_type: How to fuse different models ('concat', 'attention', 'gate')
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        fusion_type: str = "attention",
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fusion_type = fusion_type

        # BiLSTM branch
        self.lstm = BiLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        # BiGRU branch
        self.gru = BiGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Temporal attention
        self.temporal_attn = TemporalAttention(
            dim=hidden_size * 2,  # Bidirectional
            num_heads=4,
            dropout=dropout,
        )

        # Fusion mechanism
        rnn_output_size = hidden_size * 2  # Bidirectional
        if fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(rnn_output_size * 3, rnn_output_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        elif fusion_type == "attention":
            self.fusion_attn = nn.MultiheadAttention(
                rnn_output_size, num_heads=4, dropout=dropout, batch_first=True
            )
            self.fusion = nn.Linear(rnn_output_size, rnn_output_size)
        elif fusion_type == "gate":
            self.gate_lstm = nn.Sequential(
                nn.Linear(rnn_output_size, 1),
                nn.Sigmoid(),
            )
            self.gate_gru = nn.Sequential(
                nn.Linear(rnn_output_size, 1),
                nn.Sigmoid(),
            )
            self.gate_attn = nn.Sequential(
                nn.Linear(rnn_output_size, 1),
                nn.Sigmoid(),
            )
            self.fusion = nn.Linear(rnn_output_size, rnn_output_size)

        self.output_size = rnn_output_size

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Pose sequence (B, T, input_size)
               Input features: [yaw, pitch, roll, x, y, scale, ...]
            lengths: Sequence lengths

        Returns:
            Dictionary containing:
                - sequence: Temporal features (B, T, output_size)
                - final: Aggregated features (B, output_size)
                - attn_weights: Attention weights for visualization
        """
        # BiLSTM
        lstm_out, lstm_final = self.lstm(x, lengths)

        # BiGRU
        gru_out, gru_final = self.gru(x, lengths)

        # Temporal attention on LSTM output
        attn_out, attn_weights = self.temporal_attn(lstm_out)

        # Fusion
        if self.fusion_type == "concat":
            # Simple concatenation and projection
            fused = torch.cat([lstm_out, gru_out, attn_out], dim=-1)
            output = self.fusion(fused)

        elif self.fusion_type == "attention":
            # Cross-attention fusion
            stacked = torch.stack([lstm_out, gru_out, attn_out], dim=2)
            B, T, N, D = stacked.shape
            stacked = stacked.view(B * T, N, D)
            query = lstm_out.view(B * T, 1, D)
            fused, _ = self.fusion_attn(query, stacked, stacked)
            output = self.fusion(fused.view(B, T, D)) + lstm_out

        elif self.fusion_type == "gate":
            # Gated fusion
            g_lstm = self.gate_lstm(lstm_out)
            g_gru = self.gate_gru(gru_out)
            g_attn = self.gate_attn(attn_out)

            # Normalize gates
            gates = F.softmax(torch.cat([g_lstm, g_gru, g_attn], dim=-1), dim=-1)

            # Weighted sum
            output = (
                gates[..., 0:1] * lstm_out +
                gates[..., 1:2] * gru_out +
                gates[..., 2:3] * attn_out
            )
            output = self.fusion(output)

        # Aggregate final representation
        final = output.mean(dim=1)  # Global average pooling

        return {
            'sequence': output,
            'final': final,
            'attn_weights': attn_weights,
        }


class HierarchicalTemporalModel(nn.Module):
    """
    Hierarchical Temporal Model

    创新点：层次化时序建模
    - 帧级：单帧姿态特征
    - 片段级：短时序模式（如单次张望）
    - 视频级：长时序模式（如频繁张望行为）

    Args:
        input_size: Pose feature dimension
        segment_size: Frames per segment
        hidden_size: Hidden dimension
    """

    def __init__(
        self,
        input_size: int,
        segment_size: int = 10,
        hidden_size: int = 256,
    ):
        super().__init__()

        self.segment_size = segment_size

        # Frame-level encoder
        self.frame_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
        )

        # Segment-level: process short clips
        self.segment_rnn = BiLSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
        )

        # Segment-level attention
        self.segment_attn = GazePatternAttention(
            dim=hidden_size,
            num_heads=4,
            window_size=segment_size,
        )

        # Video-level: process segments
        self.video_rnn = BiLSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
        )

        # Video-level attention
        self.video_attn = TemporalAttention(
            dim=hidden_size,
            num_heads=4,
        )

        self.output_size = hidden_size

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Pose sequence (B, T, input_size)

        Returns:
            Dictionary with hierarchical features
        """
        B, T, _ = x.shape

        # Frame-level encoding
        frame_features = self.frame_encoder(x)  # (B, T, hidden_size)

        # Segment-level processing
        # Reshape into segments
        num_segments = T // self.segment_size
        T_seg = num_segments * self.segment_size

        # Trim to fit segments
        frame_features_trim = frame_features[:, :T_seg]
        segments = frame_features_trim.view(
            B, num_segments, self.segment_size, -1
        )

        # Process each segment
        segment_features = []
        for i in range(num_segments):
            seg = segments[:, i]  # (B, segment_size, hidden_size)
            seg_out, seg_final = self.segment_rnn(seg)
            segment_features.append(seg_final)

        segment_features = torch.stack(segment_features, dim=1)  # (B, num_segments, hidden_size)

        # Segment attention for pattern detection
        segment_attended, pattern_scores = self.segment_attn(segment_features)

        # Video-level processing
        video_out, video_final = self.video_rnn(segment_attended)
        video_attended, video_attn = self.video_attn(video_out)

        # Final aggregation
        final = video_attended.mean(dim=1)

        return {
            'frame_features': frame_features,
            'segment_features': segment_attended,
            'video_features': video_attended,
            'final': final,
            'pattern_scores': pattern_scores,
            'video_attn': video_attn,
        }


class PoseSequenceEncoder(nn.Module):
    """
    Encoder for head pose sequences.

    Converts raw pose angles into rich temporal features
    suitable for behavior classification.

    Args:
        pose_dim: Dimension of pose input (yaw, pitch, roll, ...)
        hidden_size: Hidden dimension
        model_type: Type of temporal model ('fusion', 'hierarchical')
    """

    def __init__(
        self,
        pose_dim: int = 6,  # yaw, pitch, roll, x, y, scale
        hidden_size: int = 256,
        model_type: str = "fusion",
    ):
        super().__init__()

        self.pose_dim = pose_dim

        # Pose feature extraction
        self.pose_embed = nn.Sequential(
            nn.Linear(pose_dim, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # Pose derivative features (velocity, acceleration)
        self.derivative_embed = nn.Sequential(
            nn.Linear(pose_dim * 2, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(inplace=True),
        )

        # Temporal model
        if model_type == "fusion":
            self.temporal = TemporalFusion(
                input_size=hidden_size,
                hidden_size=hidden_size // 2,
            )
        else:
            self.temporal = HierarchicalTemporalModel(
                input_size=hidden_size,
                hidden_size=hidden_size,
            )

        self.output_size = hidden_size

    def forward(
        self,
        poses: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            poses: Pose sequence (B, T, pose_dim)
                   Format: [yaw, pitch, roll, x, y, scale]

        Returns:
            Encoded temporal features
        """
        B, T, _ = poses.shape

        # Compute derivatives (velocity and acceleration)
        velocity = poses[:, 1:] - poses[:, :-1]
        acceleration = velocity[:, 1:] - velocity[:, :-1]

        # Pad to match original length
        velocity = F.pad(velocity, (0, 0, 0, 1))
        acceleration = F.pad(acceleration, (0, 0, 0, 2))

        # Embed pose and derivatives
        pose_features = self.pose_embed(poses)
        derivative_features = self.derivative_embed(
            torch.cat([velocity, acceleration], dim=-1)
        )

        # Fuse features
        features = torch.cat([pose_features, derivative_features], dim=-1)
        features = self.feature_fusion(features)

        # Temporal modeling
        output = self.temporal(features)
        output['derivatives'] = {
            'velocity': velocity,
            'acceleration': acceleration,
        }

        return output
