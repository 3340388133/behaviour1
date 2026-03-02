"""
Head Pose Estimation Module

Models:
- WHENet: Wide-range Head pose Estimation Network (baseline)
- WHENetPlus: WHENet with attention and feature fusion (innovation)
- SelfAttention: Self-attention mechanism
- FeatureFusion: Multi-scale feature fusion
"""

from .whenet import WHENet
from .attention import SelfAttention, CBAM, SimDLKA
from .feature_fusion import FeatureFusion, MultiScaleFusion
from .whenet_plus import WHENetPlus

__all__ = [
    "WHENet",
    "WHENetPlus",
    "SelfAttention",
    "CBAM",
    "SimDLKA",
    "FeatureFusion",
    "MultiScaleFusion",
]
