"""
Object Tracking Module

Multi-object tracking with Re-ID for person tracking
in surveillance scenarios.

Includes:
- StrongSORT: Enhanced DeepSORT with stronger association
- JointModel: Joint tracking and pose estimation
"""

from .strong_sort import StrongSORTWrapper, TrackState
from .joint_model import JointTrackingPose

__all__ = [
    "StrongSORTWrapper",
    "TrackState",
    "JointTrackingPose",
]
