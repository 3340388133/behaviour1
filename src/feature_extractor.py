"""时序特征提取模块"""
import numpy as np
from scipy import signal
from scipy.fft import fft
from dataclasses import dataclass


@dataclass
class TemporalFeatures:
    # 统计特征
    yaw_mean: float
    yaw_std: float
    yaw_max: float
    yaw_min: float
    pitch_mean: float
    pitch_std: float

    # 动力学特征
    angular_velocity_mean: float
    angular_velocity_max: float
    angular_acceleration_mean: float

    # 频域特征
    dominant_freq: float
    dominant_energy: float

    # 切换特征
    switch_count: int
    switch_rate: float

    # 热点特征
    side_gaze_ratio: float  # |yaw| > 30° 的帧占比


class FeatureExtractor:
    """多维时序特征提取器"""

    def __init__(self, window_size: float = 2.0, fps: float = 2.0,
                 side_threshold: float = 30.0):
        """
        Args:
            window_size: 滑动窗口大小（秒）
            fps: 帧率
            side_threshold: 侧向阈值（度）
        """
        self.window_size = window_size
        self.fps = fps
        self.side_threshold = side_threshold
        self.window_frames = int(window_size * fps)

    def extract(self, pose_history: list) -> TemporalFeatures:
        """从姿态历史中提取特征

        Args:
            pose_history: PoseResult列表

        Returns:
            TemporalFeatures
        """
        if len(pose_history) < 2:
            return self._empty_features()

        yaws = np.array([p.yaw for p in pose_history])
        pitches = np.array([p.pitch for p in pose_history])

        # 统计特征
        yaw_mean = np.mean(yaws)
        yaw_std = np.std(yaws)
        yaw_max = np.max(np.abs(yaws))
        yaw_min = np.min(np.abs(yaws))
        pitch_mean = np.mean(pitches)
        pitch_std = np.std(pitches)

        # 动力学特征
        dt = 1.0 / self.fps
        angular_velocity = np.abs(np.diff(yaws)) / dt
        angular_velocity_mean = np.mean(angular_velocity) if len(angular_velocity) > 0 else 0
        angular_velocity_max = np.max(angular_velocity) if len(angular_velocity) > 0 else 0

        angular_acceleration = np.abs(np.diff(angular_velocity)) / dt if len(angular_velocity) > 1 else np.array([0])
        angular_acceleration_mean = np.mean(angular_acceleration)

        # 频域特征
        dominant_freq, dominant_energy = self._compute_fft_features(yaws)

        # 切换特征
        switch_count = self._count_switches(yaws)
        switch_rate = switch_count / (len(yaws) / self.fps) if len(yaws) > 0 else 0

        # 热点特征
        side_gaze_ratio = np.mean(np.abs(yaws) > self.side_threshold)

        return TemporalFeatures(
            yaw_mean=yaw_mean, yaw_std=yaw_std, yaw_max=yaw_max, yaw_min=yaw_min,
            pitch_mean=pitch_mean, pitch_std=pitch_std,
            angular_velocity_mean=angular_velocity_mean,
            angular_velocity_max=angular_velocity_max,
            angular_acceleration_mean=angular_acceleration_mean,
            dominant_freq=dominant_freq, dominant_energy=dominant_energy,
            switch_count=switch_count, switch_rate=switch_rate,
            side_gaze_ratio=side_gaze_ratio
        )

    def _compute_fft_features(self, yaws: np.ndarray) -> tuple:
        """计算FFT主频和能量"""
        if len(yaws) < 4:
            return 0.0, 0.0

        # 去均值
        yaws_centered = yaws - np.mean(yaws)

        # FFT
        n = len(yaws_centered)
        fft_vals = np.abs(fft(yaws_centered))[:n // 2]
        freqs = np.fft.fftfreq(n, 1 / self.fps)[:n // 2]

        if len(fft_vals) == 0:
            return 0.0, 0.0

        # 主频
        dominant_idx = np.argmax(fft_vals)
        dominant_freq = freqs[dominant_idx] if dominant_idx < len(freqs) else 0.0
        dominant_energy = fft_vals[dominant_idx] if dominant_idx < len(fft_vals) else 0.0

        return float(dominant_freq), float(dominant_energy)

    def _count_switches(self, yaws: np.ndarray) -> int:
        """计算左右切换次数"""
        if len(yaws) < 2:
            return 0

        # 定义左右区间
        signs = np.sign(yaws)
        switches = np.sum(np.abs(np.diff(signs)) > 0)

        return int(switches)

    def _empty_features(self) -> TemporalFeatures:
        return TemporalFeatures(
            yaw_mean=0, yaw_std=0, yaw_max=0, yaw_min=0,
            pitch_mean=0, pitch_std=0,
            angular_velocity_mean=0, angular_velocity_max=0,
            angular_acceleration_mean=0,
            dominant_freq=0, dominant_energy=0,
            switch_count=0, switch_rate=0,
            side_gaze_ratio=0
        )

    def to_vector(self, features: TemporalFeatures) -> np.ndarray:
        """将特征转换为向量"""
        return np.array([
            features.yaw_mean, features.yaw_std, features.yaw_max,
            features.pitch_mean, features.pitch_std,
            features.angular_velocity_mean, features.angular_velocity_max,
            features.angular_acceleration_mean,
            features.dominant_freq, features.dominant_energy,
            features.switch_count, features.switch_rate,
            features.side_gaze_ratio
        ])
