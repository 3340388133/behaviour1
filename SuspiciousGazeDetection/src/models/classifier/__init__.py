"""
Behavior Classification Module

Final classification stage for suspicious gaze detection.

Includes:
- CNN3D: 3D Convolutional Network for spatiotemporal features
- MultiModalFusion: Multi-modal feature fusion classifier
- SuspiciousGazeClassifier: End-to-end classifier
"""

from .cnn3d import CNN3D, C3D, R3D
from .fusion import MultiModalFusion, SuspiciousGazeClassifier

__all__ = [
    "CNN3D",
    "C3D",
    "R3D",
    "MultiModalFusion",
    "SuspiciousGazeClassifier",
]
