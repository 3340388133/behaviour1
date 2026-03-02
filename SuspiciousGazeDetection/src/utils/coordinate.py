"""
Coordinate System Transformations

Handle coordinate transformations between different camera positions
(front view vs side view) for consistent pose representation.

创新点：自适应坐标系转换，统一处理正机位和侧机位数据
"""

import numpy as np
from typing import Tuple, Optional
from enum import Enum


class CameraPosition(Enum):
    """Camera position enumeration."""
    FRONT = "front"     # 正机位
    SIDE_LEFT = "side_left"    # 左侧机位
    SIDE_RIGHT = "side_right"  # 右侧机位
    UNKNOWN = "unknown"


class CoordinateTransformer:
    """
    Coordinate System Transformer

    Handles transformation of head pose angles between different
    camera coordinate systems to a unified world coordinate system.

    创新点：
    1. 自动检测相机位置
    2. 统一的世界坐标系表示
    3. 支持多相机融合
    """

    def __init__(
        self,
        front_yaw_offset: float = 0.0,
        side_yaw_offset: float = 90.0,
    ):
        """
        Args:
            front_yaw_offset: Yaw offset for front camera
            side_yaw_offset: Yaw offset for side cameras
        """
        self.front_yaw_offset = front_yaw_offset
        self.side_yaw_offset = side_yaw_offset

        # Camera calibration parameters (can be set externally)
        self.camera_params = {
            CameraPosition.FRONT: {
                'yaw_offset': front_yaw_offset,
                'pitch_offset': 0.0,
                'roll_offset': 0.0,
            },
            CameraPosition.SIDE_LEFT: {
                'yaw_offset': side_yaw_offset,
                'pitch_offset': 0.0,
                'roll_offset': 0.0,
            },
            CameraPosition.SIDE_RIGHT: {
                'yaw_offset': -side_yaw_offset,
                'pitch_offset': 0.0,
                'roll_offset': 0.0,
            },
        }

    def camera_to_world(
        self,
        yaw: float,
        pitch: float,
        roll: float,
        camera_position: CameraPosition,
    ) -> Tuple[float, float, float]:
        """
        Transform pose from camera coordinate to world coordinate.

        Args:
            yaw: Yaw angle in camera coordinates (degrees)
            pitch: Pitch angle in camera coordinates (degrees)
            roll: Roll angle in camera coordinates (degrees)
            camera_position: Camera position enum

        Returns:
            Tuple of (yaw, pitch, roll) in world coordinates
        """
        if camera_position not in self.camera_params:
            return yaw, pitch, roll

        params = self.camera_params[camera_position]

        # Apply offsets
        world_yaw = yaw - params['yaw_offset']
        world_pitch = pitch - params['pitch_offset']
        world_roll = roll - params['roll_offset']

        # Normalize yaw to [-180, 180]
        world_yaw = self._normalize_angle(world_yaw)

        return world_yaw, world_pitch, world_roll

    def world_to_camera(
        self,
        yaw: float,
        pitch: float,
        roll: float,
        camera_position: CameraPosition,
    ) -> Tuple[float, float, float]:
        """
        Transform pose from world coordinate to camera coordinate.

        Args:
            yaw: Yaw angle in world coordinates (degrees)
            pitch: Pitch angle in world coordinates (degrees)
            roll: Roll angle in world coordinates (degrees)
            camera_position: Target camera position

        Returns:
            Tuple of (yaw, pitch, roll) in camera coordinates
        """
        if camera_position not in self.camera_params:
            return yaw, pitch, roll

        params = self.camera_params[camera_position]

        # Apply inverse offsets
        cam_yaw = yaw + params['yaw_offset']
        cam_pitch = pitch + params['pitch_offset']
        cam_roll = roll + params['roll_offset']

        # Normalize yaw
        cam_yaw = self._normalize_angle(cam_yaw)

        return cam_yaw, cam_pitch, cam_roll

    def detect_camera_position(
        self,
        typical_yaw_range: Tuple[float, float],
    ) -> CameraPosition:
        """
        Detect camera position based on typical yaw angle range.

        创新点：根据数据统计自动推断相机位置

        Args:
            typical_yaw_range: (min_yaw, max_yaw) observed in data

        Returns:
            Detected camera position
        """
        min_yaw, max_yaw = typical_yaw_range
        center_yaw = (min_yaw + max_yaw) / 2

        # Front camera: people look roughly at camera (yaw ~0)
        if -45 < center_yaw < 45:
            return CameraPosition.FRONT

        # Side left: people look to their right relative to camera
        if 45 <= center_yaw < 135:
            return CameraPosition.SIDE_LEFT

        # Side right: people look to their left relative to camera
        if -135 < center_yaw <= -45:
            return CameraPosition.SIDE_RIGHT

        return CameraPosition.UNKNOWN

    def batch_transform(
        self,
        poses: np.ndarray,
        camera_position: CameraPosition,
        to_world: bool = True,
    ) -> np.ndarray:
        """
        Transform batch of poses.

        Args:
            poses: Poses array (N, 3) - yaw, pitch, roll
            camera_position: Camera position
            to_world: If True, camera->world; else world->camera

        Returns:
            Transformed poses (N, 3)
        """
        result = np.zeros_like(poses)

        for i in range(len(poses)):
            yaw, pitch, roll = poses[i]
            if to_world:
                yaw, pitch, roll = self.camera_to_world(
                    yaw, pitch, roll, camera_position
                )
            else:
                yaw, pitch, roll = self.world_to_camera(
                    yaw, pitch, roll, camera_position
                )
            result[i] = [yaw, pitch, roll]

        return result

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-180, 180] range."""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def compute_relative_pose(
        self,
        pose1: Tuple[float, float, float],
        pose2: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """
        Compute relative pose (difference) between two poses.

        Args:
            pose1: First pose (yaw, pitch, roll)
            pose2: Second pose (yaw, pitch, roll)

        Returns:
            Relative pose (delta_yaw, delta_pitch, delta_roll)
        """
        delta_yaw = self._normalize_angle(pose2[0] - pose1[0])
        delta_pitch = pose2[1] - pose1[1]
        delta_roll = pose2[2] - pose1[2]

        return delta_yaw, delta_pitch, delta_roll


class MultiCameraFusion:
    """
    Multi-camera pose fusion.

    创新点：融合多个相机视角的姿态估计，提高精度
    """

    def __init__(self, transformer: CoordinateTransformer):
        self.transformer = transformer

    def fuse_poses(
        self,
        poses: dict,  # {camera_position: (yaw, pitch, roll, confidence)}
    ) -> Tuple[float, float, float]:
        """
        Fuse pose estimates from multiple cameras.

        Args:
            poses: Dictionary mapping camera position to pose and confidence

        Returns:
            Fused pose in world coordinates
        """
        if not poses:
            return 0.0, 0.0, 0.0

        # Transform all to world coordinates
        world_poses = []
        confidences = []

        for camera_pos, (yaw, pitch, roll, conf) in poses.items():
            world_pose = self.transformer.camera_to_world(
                yaw, pitch, roll, camera_pos
            )
            world_poses.append(world_pose)
            confidences.append(conf)

        world_poses = np.array(world_poses)
        confidences = np.array(confidences)

        # Weighted average
        weights = confidences / confidences.sum()

        # Special handling for yaw (circular mean)
        yaw_rad = np.deg2rad(world_poses[:, 0])
        mean_sin = np.average(np.sin(yaw_rad), weights=weights)
        mean_cos = np.average(np.cos(yaw_rad), weights=weights)
        fused_yaw = np.rad2deg(np.arctan2(mean_sin, mean_cos))

        # Simple weighted average for pitch and roll
        fused_pitch = np.average(world_poses[:, 1], weights=weights)
        fused_roll = np.average(world_poses[:, 2], weights=weights)

        return fused_yaw, fused_pitch, fused_roll
