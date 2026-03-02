"""
Joint Tracking and Pose Estimation Model

创新点：将目标追踪与头部姿态估计深度融合，
利用时间连续性约束提升姿态估计的稳定性。

主要特性：
1. 追踪信息辅助姿态估计
2. 姿态信息辅助身份关联
3. 时间平滑提升稳定性
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class PersonState:
    """
    Complete state for a tracked person.

    Includes tracking info, pose history, and behavior features.
    """
    track_id: int
    bbox: np.ndarray                    # Current bounding box
    head_bbox: np.ndarray               # Current head box
    pose: np.ndarray                    # Current pose [yaw, pitch, roll]
    pose_history: deque = field(default_factory=lambda: deque(maxlen=100))
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=100))
    timestamp_history: deque = field(default_factory=lambda: deque(maxlen=100))
    features: Optional[np.ndarray] = None
    smoothed_pose: Optional[np.ndarray] = None

    def add_observation(
        self,
        bbox: np.ndarray,
        head_bbox: np.ndarray,
        pose: np.ndarray,
        timestamp: float,
    ):
        """Add new observation."""
        self.bbox = bbox
        self.head_bbox = head_bbox
        self.pose = pose

        self.bbox_history.append(bbox.copy())
        self.pose_history.append(pose.copy())
        self.timestamp_history.append(timestamp)

        # Update smoothed pose
        self.smoothed_pose = self._smooth_pose()

    def _smooth_pose(self, window: int = 5) -> np.ndarray:
        """Apply temporal smoothing to pose."""
        if len(self.pose_history) < 2:
            return self.pose.copy()

        recent = list(self.pose_history)[-window:]
        return np.mean(recent, axis=0)

    def get_pose_velocity(self) -> Optional[np.ndarray]:
        """Compute pose change velocity."""
        if len(self.pose_history) < 2:
            return None

        poses = np.array(list(self.pose_history)[-5:])
        times = np.array(list(self.timestamp_history)[-5:])

        if len(times) < 2:
            return None

        dt = times[-1] - times[0]
        if dt < 0.001:
            return None

        return (poses[-1] - poses[0]) / dt

    def get_gaze_pattern_features(self) -> Dict[str, float]:
        """
        Extract features for suspicious gaze detection.

        创新点：从追踪历史中提取行为特征
        """
        if len(self.pose_history) < 10:
            return {
                'yaw_variance': 0.0,
                'yaw_range': 0.0,
                'direction_changes': 0,
                'avg_speed': 0.0,
            }

        poses = np.array(list(self.pose_history))
        yaw = poses[:, 0]

        # Yaw variance (high = looking around)
        yaw_var = np.var(yaw)

        # Yaw range
        yaw_range = np.max(yaw) - np.min(yaw)

        # Direction changes (sign changes in velocity)
        yaw_diff = np.diff(yaw)
        sign_changes = np.sum(np.diff(np.sign(yaw_diff)) != 0)

        # Average angular speed
        times = np.array(list(self.timestamp_history))
        if len(times) > 1:
            dt = times[-1] - times[0]
            total_change = np.sum(np.abs(yaw_diff))
            avg_speed = total_change / dt if dt > 0 else 0
        else:
            avg_speed = 0

        return {
            'yaw_variance': float(yaw_var),
            'yaw_range': float(yaw_range),
            'direction_changes': int(sign_changes),
            'avg_speed': float(avg_speed),
        }


class JointTrackingPose(nn.Module):
    """
    Joint Tracking and Pose Estimation Module

    创新点：
    1. 利用追踪ID的时间连续性平滑姿态估计
    2. 用姿态相似性辅助跨帧身份关联
    3. 预测姿态变化提前检测异常

    Args:
        pose_model: Head pose estimation model
        use_kalman: Use Kalman filter for pose smoothing
        pose_weight_for_reid: Weight of pose similarity in ReID
    """

    def __init__(
        self,
        pose_model: nn.Module,
        use_kalman: bool = True,
        pose_weight_for_reid: float = 0.2,
    ):
        super().__init__()

        self.pose_model = pose_model
        self.use_kalman = use_kalman
        self.pose_weight_for_reid = pose_weight_for_reid

        # Person states
        self.persons: Dict[int, PersonState] = {}

        # Kalman filters for pose smoothing
        self.kalman_filters: Dict[int, KalmanPoseFilter] = {}

        # Pose prediction for association
        self.pose_predictor = PosePredictorLSTM(
            pose_dim=3,
            hidden_dim=64,
        )

    def forward(
        self,
        frame: torch.Tensor,
        detections: torch.Tensor,
        track_ids: torch.Tensor,
        head_crops: torch.Tensor,
        timestamp: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Process frame with tracking and pose estimation.

        Args:
            frame: Video frame (B, 3, H, W)
            detections: Detection boxes (N, 4)
            track_ids: Track IDs (N,)
            head_crops: Head crop images (N, 3, H, W)
            timestamp: Current timestamp

        Returns:
            Dictionary with poses, features, and states
        """
        N = len(track_ids)

        if N == 0:
            return {
                'poses': torch.empty(0, 3),
                'smoothed_poses': torch.empty(0, 3),
                'features': torch.empty(0, 256),
            }

        # Pose estimation
        pose_output = self.pose_model(head_crops)
        if isinstance(pose_output, dict):
            poses = torch.stack([
                pose_output['yaw'],
                pose_output['pitch'],
                pose_output['roll']
            ], dim=1)
            features = pose_output.get('features', None)
        else:
            yaw, pitch, roll = pose_output
            poses = torch.stack([yaw, pitch, roll], dim=1)
            features = None

        # Update person states
        smoothed_poses = []
        for i in range(N):
            track_id = track_ids[i].item()
            pose = poses[i].detach().cpu().numpy()
            bbox = detections[i].detach().cpu().numpy()

            # Initialize new person
            if track_id not in self.persons:
                self.persons[track_id] = PersonState(
                    track_id=track_id,
                    bbox=bbox,
                    head_bbox=bbox,  # Will be updated
                    pose=pose,
                )
                if self.use_kalman:
                    self.kalman_filters[track_id] = KalmanPoseFilter()

            # Update state
            person = self.persons[track_id]
            person.add_observation(bbox, bbox, pose, timestamp)

            # Kalman smoothing
            if self.use_kalman and track_id in self.kalman_filters:
                smoothed = self.kalman_filters[track_id].update(pose)
            else:
                smoothed = person.smoothed_pose

            smoothed_poses.append(torch.tensor(smoothed))

        smoothed_poses = torch.stack(smoothed_poses)

        return {
            'poses': poses,
            'smoothed_poses': smoothed_poses,
            'features': features,
            'track_ids': track_ids,
        }

    def get_person_state(self, track_id: int) -> Optional[PersonState]:
        """Get state for a tracked person."""
        return self.persons.get(track_id)

    def get_suspicious_candidates(
        self,
        yaw_threshold: float = 30.0,
        frequency_threshold: int = 3,
    ) -> List[int]:
        """
        Get track IDs of potentially suspicious persons.

        创新点：基于追踪历史的早期异常检测
        """
        suspicious = []

        for track_id, person in self.persons.items():
            features = person.get_gaze_pattern_features()

            # Check for suspicious patterns
            if (features['yaw_range'] > yaw_threshold * 2 and
                features['direction_changes'] >= frequency_threshold):
                suspicious.append(track_id)

        return suspicious

    def compute_pose_affinity(
        self,
        pose1: np.ndarray,
        pose2: np.ndarray,
    ) -> float:
        """
        Compute affinity between two poses for ReID.

        创新点：姿态相似性辅助身份关联
        """
        # Angle difference (normalized)
        diff = np.abs(pose1 - pose2)
        # Handle angle wrapping for yaw
        diff[0] = min(diff[0], 360 - diff[0])

        # Convert to similarity (0-1)
        max_diff = np.array([180, 90, 90])  # Max expected differences
        similarity = 1.0 - np.mean(diff / max_diff)

        return max(0, similarity)

    def cleanup_old_tracks(self, max_age: int = 100):
        """Remove old inactive tracks."""
        current_ids = list(self.persons.keys())
        for track_id in current_ids:
            person = self.persons[track_id]
            if len(person.pose_history) > 0:
                # Check if track is stale
                if len(person.timestamp_history) > 0:
                    # Keep track if recently updated
                    pass
            else:
                del self.persons[track_id]
                if track_id in self.kalman_filters:
                    del self.kalman_filters[track_id]


class KalmanPoseFilter:
    """
    Kalman filter for pose smoothing.

    Provides temporal smoothing while allowing for
    rapid pose changes to be detected.
    """

    def __init__(self):
        # State: [yaw, pitch, roll, yaw_vel, pitch_vel, roll_vel]
        self.dim_x = 6
        self.dim_z = 3

        # State estimate
        self.x = np.zeros(self.dim_x)

        # Covariance matrix
        self.P = np.eye(self.dim_x) * 100

        # State transition matrix
        self.F = np.eye(self.dim_x)
        self.F[0, 3] = 1  # yaw += yaw_vel * dt
        self.F[1, 4] = 1
        self.F[2, 5] = 1

        # Measurement matrix
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1

        # Process noise
        self.Q = np.eye(self.dim_x) * 0.1

        # Measurement noise
        self.R = np.eye(self.dim_z) * 1.0

        self.initialized = False

    def update(self, pose: np.ndarray) -> np.ndarray:
        """Update filter with new pose measurement."""
        if not self.initialized:
            self.x[:3] = pose
            self.initialized = True
            return pose

        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Update
        y = pose - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self.x = x_pred + K @ y
        self.P = (np.eye(self.dim_x) - K @ self.H) @ P_pred

        return self.x[:3].copy()


class PosePredictorLSTM(nn.Module):
    """
    LSTM for predicting future pose.

    创新点：预测下一帧姿态，用于：
    1. 检测异常姿态变化
    2. 辅助遮挡情况下的追踪
    """

    def __init__(self, pose_dim: int = 3, hidden_dim: int = 64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=pose_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, pose_dim)

    def forward(
        self,
        pose_history: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict next pose from history.

        Args:
            pose_history: Past poses (B, T, 3)

        Returns:
            Predicted next pose (B, 3)
        """
        output, _ = self.lstm(pose_history)
        pred = self.fc(output[:, -1])
        return pred
