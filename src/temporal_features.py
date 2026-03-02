"""
时序特征提取模块
滑动窗口: 2.0s, 步长 0.5s
输入: track_id 的 yaw/pitch/roll 时间序列
输出: yaw/pitch/roll 的统计特征
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TemporalFeature:
    """时序特征结果"""
    window_start: float      # 窗口起始时间 (秒)
    window_end: float        # 窗口结束时间 (秒)
    track_id: int
    # yaw 特征
    yaw_mean: float          # yaw 均值
    yaw_std: float           # yaw 标准差
    yaw_range: float         # yaw 范围 (max - min)
    yaw_speed_mean: float    # yaw 变化速度均值 (度/秒)
    yaw_switch_count: int    # yaw 方向切换次数
    # pitch 特征
    pitch_mean: float        # pitch 均值
    pitch_std: float         # pitch 标准差
    pitch_range: float       # pitch 范围
    # roll 特征
    roll_mean: float         # roll 均值
    roll_std: float          # roll 标准差
    roll_range: float        # roll 范围
    sample_count: int        # 窗口内样本数

    def to_dict(self) -> dict:
        return {
            'window_start': round(self.window_start, 3),
            'window_end': round(self.window_end, 3),
            'track_id': self.track_id,
            'yaw_mean': round(self.yaw_mean, 2),
            'yaw_std': round(self.yaw_std, 2),
            'yaw_range': round(self.yaw_range, 2),
            'yaw_speed_mean': round(self.yaw_speed_mean, 2),
            'yaw_switch_count': self.yaw_switch_count,
            'pitch_mean': round(self.pitch_mean, 2),
            'pitch_std': round(self.pitch_std, 2),
            'pitch_range': round(self.pitch_range, 2),
            'roll_mean': round(self.roll_mean, 2),
            'roll_std': round(self.roll_std, 2),
            'roll_range': round(self.roll_range, 2),
            'sample_count': self.sample_count
        }


def normalize_yaw(yaw: float) -> float:
    """将 yaw 归一化到 [-180, 180]"""
    while yaw > 180:
        yaw -= 360
    while yaw < -180:
        yaw += 360
    return yaw


def yaw_diff(yaw1: float, yaw2: float) -> float:
    """计算两个 yaw 角度的差值，处理 ±180° 跳变

    例如: yaw_diff(170, -170) = -20 (而不是 340)
    """
    diff = yaw2 - yaw1
    # 处理跨越 ±180° 的情况
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    return diff


def is_yaw_jump(yaw1: float, yaw2: float, dt: float,
                max_speed: float = 180.0) -> bool:
    """检测 yaw 是否发生异常跳变

    Args:
        yaw1, yaw2: 连续两帧的 yaw 值
        dt: 时间间隔 (秒)
        max_speed: 最大合理转头速度 (度/秒)，默认 180°/s

    Returns:
        True 表示发生跳变，应该忽略
    """
    if dt <= 0:
        return True
    diff = abs(yaw_diff(yaw1, yaw2))
    speed = diff / dt
    return speed > max_speed


class TemporalFeatureExtractor:
    """时序特征提取器"""

    def __init__(
        self,
        window_size: float = 2.0,    # 窗口大小 (秒)
        step_size: float = 0.5,      # 步长 (秒)
        min_samples: int = 5,        # 窗口内最小样本数
        max_yaw_speed: float = 180.0,  # 最大合理转头速度
        switch_threshold: float = 15.0  # 方向切换阈值 (度)
    ):
        self.window_size = window_size
        self.step_size = step_size
        self.min_samples = min_samples
        self.max_yaw_speed = max_yaw_speed
        self.switch_threshold = switch_threshold

    def extract_from_track(
        self,
        times: np.ndarray,
        yaws: np.ndarray,
        track_id: int,
        pitches: np.ndarray = None,
        rolls: np.ndarray = None
    ) -> List[TemporalFeature]:
        """从单个 track 提取时序特征

        Args:
            times: 时间戳数组 (秒)
            yaws: yaw 角度数组 (度)
            track_id: 轨迹 ID
            pitches: pitch 角度数组 (度), 可选
            rolls: roll 角度数组 (度), 可选

        Returns:
            TemporalFeature 列表
        """
        if len(times) < self.min_samples:
            return []

        # 按时间排序
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        yaws = yaws[sort_idx]
        pitches = pitches[sort_idx] if pitches is not None else np.zeros_like(yaws)
        rolls = rolls[sort_idx] if rolls is not None else np.zeros_like(yaws)

        # 归一化 yaw
        yaws = np.array([normalize_yaw(y) for y in yaws])

        # 滑动窗口
        results = []
        t_start = times[0]
        t_end = times[-1]

        window_start = t_start
        while window_start + self.window_size <= t_end + self.step_size:
            window_end = window_start + self.window_size

            # 获取窗口内的数据
            mask = (times >= window_start) & (times < window_end)
            w_times = times[mask]
            w_yaws = yaws[mask]
            w_pitches = pitches[mask]
            w_rolls = rolls[mask]

            if len(w_times) >= self.min_samples:
                feature = self._compute_features(
                    w_times, w_yaws, w_pitches, w_rolls,
                    track_id, window_start, window_end
                )
                if feature is not None:
                    results.append(feature)

            window_start += self.step_size

        return results

    def _compute_features(
        self,
        times: np.ndarray,
        yaws: np.ndarray,
        pitches: np.ndarray,
        rolls: np.ndarray,
        track_id: int,
        window_start: float,
        window_end: float
    ) -> Optional[TemporalFeature]:
        """计算单个窗口的特征"""
        n = len(times)

        # 1. Yaw 基础统计
        yaw_mean = np.mean(yaws)
        yaw_std = np.std(yaws)
        yaw_range = np.max(yaws) - np.min(yaws)

        # 处理跨越 ±180° 的 range 计算
        if yaw_range > 180:
            yaws_shifted = np.where(yaws < 0, yaws + 360, yaws)
            yaw_range = np.max(yaws_shifted) - np.min(yaws_shifted)

        # 2. Pitch 统计
        pitch_mean = np.mean(pitches)
        pitch_std = np.std(pitches)
        pitch_range = np.max(pitches) - np.min(pitches)

        # 3. Roll 统计
        roll_mean = np.mean(rolls)
        roll_std = np.std(rolls)
        roll_range = np.max(rolls) - np.min(rolls)

        # 4. 计算 yaw 变化速度 (过滤跳变)
        speeds = []
        for i in range(1, n):
            dt = times[i] - times[i-1]
            if dt > 0 and not is_yaw_jump(yaws[i-1], yaws[i], dt, self.max_yaw_speed):
                diff = abs(yaw_diff(yaws[i-1], yaws[i]))
                speeds.append(diff / dt)

        yaw_speed_mean = np.mean(speeds) if speeds else 0.0

        # 5. 计算方向切换次数
        switch_count = self._count_switches(times, yaws)

        return TemporalFeature(
            window_start=window_start,
            window_end=window_end,
            track_id=track_id,
            yaw_mean=yaw_mean,
            yaw_std=yaw_std,
            yaw_range=yaw_range,
            yaw_speed_mean=yaw_speed_mean,
            yaw_switch_count=switch_count,
            pitch_mean=pitch_mean,
            pitch_std=pitch_std,
            pitch_range=pitch_range,
            roll_mean=roll_mean,
            roll_std=roll_std,
            roll_range=roll_range,
            sample_count=n
        )

    def _count_switches(self, times: np.ndarray, yaws: np.ndarray) -> int:
        """计算 yaw 方向切换次数

        方向切换定义: yaw 变化方向从正变负或从负变正，
        且累积变化量超过阈值
        """
        if len(yaws) < 3:
            return 0

        # 计算有效的 yaw 差分 (过滤跳变)
        diffs = []
        for i in range(1, len(yaws)):
            dt = times[i] - times[i-1]
            if dt > 0 and not is_yaw_jump(yaws[i-1], yaws[i], dt, self.max_yaw_speed):
                diffs.append(yaw_diff(yaws[i-1], yaws[i]))
            else:
                diffs.append(0)  # 跳变点视为无变化

        # 累积变化量并检测切换
        switch_count = 0
        cumsum = 0
        last_sign = 0

        for diff in diffs:
            cumsum += diff

            # 当累积变化超过阈值时，记录方向
            if abs(cumsum) >= self.switch_threshold:
                current_sign = 1 if cumsum > 0 else -1

                # 方向切换
                if last_sign != 0 and current_sign != last_sign:
                    switch_count += 1

                last_sign = current_sign
                cumsum = 0  # 重置累积

        return switch_count

    def extract_from_dataframe(
        self,
        df: pd.DataFrame,
        time_col: str = 'time_sec',
        yaw_col: str = 'yaw',
        pitch_col: str = 'pitch',
        roll_col: str = 'roll',
        track_col: str = 'track_id'
    ) -> pd.DataFrame:
        """从 DataFrame 提取所有 track 的时序特征

        Args:
            df: 包含 time_sec, yaw, pitch, roll, track_id 的 DataFrame
            time_col, yaw_col, pitch_col, roll_col, track_col: 列名

        Returns:
            时序特征 DataFrame
        """
        all_features = []

        # 检查 pitch/roll 列是否存在
        has_pitch = pitch_col in df.columns
        has_roll = roll_col in df.columns

        for track_id in df[track_col].unique():
            track_df = df[df[track_col] == track_id].sort_values(time_col)
            times = track_df[time_col].values
            yaws = track_df[yaw_col].values
            pitches = track_df[pitch_col].values if has_pitch else None
            rolls = track_df[roll_col].values if has_roll else None

            features = self.extract_from_track(times, yaws, track_id, pitches, rolls)
            all_features.extend([f.to_dict() for f in features])

        return pd.DataFrame(all_features)
