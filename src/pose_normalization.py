"""
姿态信号去个体化模块

核心问题：直接使用 yaw/pitch/roll 的风险
1. 模型学习"角度状态"而非"行为模式"
   - 例：正面坐的人 yaw≈0，侧面坐的人 yaw≈45°
   - 两人都没有"张望"，但原始特征差异很大

2. 个体差异
   - 不同人的习惯姿态不同（baseline不同）
   - 身高、坐姿影响相对于相机的角度

3. 相机偏置
   - 相机安装位置/角度影响绝对值
   - 同一行为在不同相机下的姿态值不同

解决方案：
1. 差分特征：关注"变化"而非"状态"
2. Baseline归一化：减去个人平均姿态
3. 相对特征：相对于窗口起始的变化
4. 统计特征：std/range 天然去个体化
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Optional
from enum import Enum


# ============================================================================
# 姿态归一化方案
# ============================================================================
class NormalizationMethod(Enum):
    """归一化方法"""
    NONE = "none"                      # 不归一化（原始值）
    BASELINE_SUBTRACT = "baseline"     # 减去 track baseline
    WINDOW_RELATIVE = "window_rel"     # 相对于窗口起始
    Z_SCORE = "z_score"                # Z-score 标准化


@dataclass
class PoseNormalizationConfig:
    """姿态归一化配置"""
    method: NormalizationMethod = NormalizationMethod.BASELINE_SUBTRACT

    # Baseline 计算参数
    baseline_window_sec: float = 3.0   # 用于计算 baseline 的初始窗口
    baseline_percentile: float = 50    # 使用中位数作为 baseline

    # Z-score 参数
    global_mean_yaw: float = 0.0       # 全局均值（可从训练集统计）
    global_std_yaw: float = 30.0       # 全局标准差
    global_mean_pitch: float = 0.0
    global_std_pitch: float = 15.0
    global_mean_roll: float = 0.0
    global_std_roll: float = 10.0


# ============================================================================
# 推荐使用的姿态时序特征
# ============================================================================
"""
特征分类：

【第一类：差分特征（推荐）】- 关注"变化"而非"状态"
- d_yaw, d_pitch, d_roll: 帧间差分（一阶导数）
- dd_yaw: 二阶差分（加速度）
- yaw_speed, pitch_speed: 变化速度（度/秒）

【第二类：统计特征（推荐）】- 天然去个体化
- yaw_std, pitch_std, roll_std: 标准差（变化幅度）
- yaw_range: 最大值-最小值（转头范围）
- yaw_switch_count: 方向切换次数

【第三类：归一化绝对特征（谨慎使用）】
- yaw_norm = yaw - baseline_yaw: 相对于个人基线
- yaw_rel = yaw - window_start_yaw: 相对于窗口起始

【第四类：原始特征（不推荐单独使用）】
- yaw_mean, pitch_mean: 绝对角度均值
- 仅作为辅助特征，不应作为主要判据

推荐特征组合（按重要性排序）：
1. yaw_speed_mean    - 转头速度（区分快速回头）
2. yaw_switch_count  - 方向切换（区分频繁张望）
3. yaw_std           - 变化幅度
4. yaw_range         - 转头范围
5. yaw_rel_max       - 相对最大偏转
6. pitch_std         - 俯仰变化
7. d_yaw_std         - 速度变化性
"""

RECOMMENDED_FEATURES = {
    # 主要特征（行为判定核心）
    "primary": [
        "yaw_speed_mean",     # 平均转头速度
        "yaw_speed_max",      # 最大转头速度
        "yaw_switch_count",   # 方向切换次数
        "yaw_std",            # yaw 标准差
        "yaw_range",          # yaw 范围
    ],

    # 次要特征（辅助判定）
    "secondary": [
        "yaw_rel_mean",       # 相对于 baseline 的平均偏转
        "yaw_rel_max",        # 相对于 baseline 的最大偏转
        "pitch_std",          # pitch 变化幅度
        "pitch_range",        # pitch 范围
        "d_yaw_std",          # 速度变化性（加速度指标）
    ],

    # 补充特征（特定场景）
    "supplementary": [
        "prolonged_side_ratio",  # 持续侧向时间比例
        "quick_turn_count",      # 快速回头次数
        "roll_std",              # roll 变化（歪头）
    ],

    # 不推荐单独使用的特征
    "not_recommended": [
        "yaw_mean",           # 绝对均值，受个体/相机影响
        "pitch_mean",
        "roll_mean",
    ]
}


@dataclass
class NormalizedPose:
    """归一化后的姿态"""
    # 原始值
    yaw: float
    pitch: float
    roll: float

    # 归一化值
    yaw_norm: float          # baseline 归一化
    pitch_norm: float
    roll_norm: float

    # 差分值（与上一帧的变化）
    d_yaw: float = 0.0
    d_pitch: float = 0.0
    d_roll: float = 0.0

    # 时间信息
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "yaw": round(self.yaw, 2),
            "pitch": round(self.pitch, 2),
            "roll": round(self.roll, 2),
            "yaw_norm": round(self.yaw_norm, 2),
            "pitch_norm": round(self.pitch_norm, 2),
            "roll_norm": round(self.roll_norm, 2),
            "d_yaw": round(self.d_yaw, 2),
            "d_pitch": round(self.d_pitch, 2),
            "d_roll": round(self.d_roll, 2),
            "timestamp": round(self.timestamp, 6)
        }


@dataclass
class TrackPoseStats:
    """Track 级别的姿态统计（用于计算 baseline）"""
    track_id: int

    # Baseline（个人习惯姿态）
    baseline_yaw: float
    baseline_pitch: float
    baseline_roll: float

    # 全局统计
    mean_yaw: float
    mean_pitch: float
    mean_roll: float
    std_yaw: float
    std_pitch: float
    std_roll: float

    # 样本信息
    sample_count: int
    duration_sec: float

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "baseline_yaw": round(self.baseline_yaw, 2),
            "baseline_pitch": round(self.baseline_pitch, 2),
            "baseline_roll": round(self.baseline_roll, 2),
            "mean_yaw": round(self.mean_yaw, 2),
            "mean_pitch": round(self.mean_pitch, 2),
            "mean_roll": round(self.mean_roll, 2),
            "std_yaw": round(self.std_yaw, 2),
            "std_pitch": round(self.std_pitch, 2),
            "std_roll": round(self.std_roll, 2),
            "sample_count": self.sample_count,
            "duration_sec": round(self.duration_sec, 3)
        }


# ============================================================================
# 姿态归一化器
# ============================================================================
class PoseNormalizer:
    """姿态归一化器"""

    def __init__(self, config: PoseNormalizationConfig = None, fps: float = 10.0):
        self.config = config or PoseNormalizationConfig()
        self.fps = fps

    def compute_track_baseline(
        self,
        timestamps: np.ndarray,
        yaws: np.ndarray,
        pitches: np.ndarray,
        rolls: np.ndarray
    ) -> TrackPoseStats:
        """计算 track 的 baseline 姿态

        Args:
            timestamps: 时间戳数组
            yaws, pitches, rolls: 姿态角度数组

        Returns:
            TrackPoseStats
        """
        # 处理 yaw 的 ±180° 跳变
        yaws = self._unwrap_yaw(yaws)

        # 使用初始窗口计算 baseline
        baseline_samples = int(self.config.baseline_window_sec * self.fps)
        baseline_samples = min(baseline_samples, len(yaws))

        if baseline_samples < 3:
            # 样本太少，使用全局统计
            baseline_yaw = np.median(yaws)
            baseline_pitch = np.median(pitches)
            baseline_roll = np.median(rolls)
        else:
            # 使用初始窗口的中位数
            baseline_yaw = np.percentile(yaws[:baseline_samples], self.config.baseline_percentile)
            baseline_pitch = np.percentile(pitches[:baseline_samples], self.config.baseline_percentile)
            baseline_roll = np.percentile(rolls[:baseline_samples], self.config.baseline_percentile)

        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0

        return TrackPoseStats(
            track_id=-1,  # 需要外部设置
            baseline_yaw=baseline_yaw,
            baseline_pitch=baseline_pitch,
            baseline_roll=baseline_roll,
            mean_yaw=np.mean(yaws),
            mean_pitch=np.mean(pitches),
            mean_roll=np.mean(rolls),
            std_yaw=np.std(yaws),
            std_pitch=np.std(pitches),
            std_roll=np.std(rolls),
            sample_count=len(yaws),
            duration_sec=duration
        )

    def normalize_sequence(
        self,
        timestamps: np.ndarray,
        yaws: np.ndarray,
        pitches: np.ndarray,
        rolls: np.ndarray,
        baseline: TrackPoseStats = None
    ) -> List[NormalizedPose]:
        """归一化姿态序列

        Args:
            timestamps: 时间戳数组
            yaws, pitches, rolls: 原始姿态角度
            baseline: 预计算的 baseline，None 则自动计算

        Returns:
            NormalizedPose 列表
        """
        if len(timestamps) == 0:
            return []

        # 处理 yaw 跳变
        yaws = self._unwrap_yaw(yaws)

        # 计算 baseline
        if baseline is None:
            baseline = self.compute_track_baseline(timestamps, yaws, pitches, rolls)

        # 归一化
        results = []
        prev_yaw, prev_pitch, prev_roll = yaws[0], pitches[0], rolls[0]
        prev_time = timestamps[0]

        for i in range(len(timestamps)):
            t = timestamps[i]
            yaw, pitch, roll = yaws[i], pitches[i], rolls[i]

            # 计算归一化值
            if self.config.method == NormalizationMethod.BASELINE_SUBTRACT:
                yaw_norm = yaw - baseline.baseline_yaw
                pitch_norm = pitch - baseline.baseline_pitch
                roll_norm = roll - baseline.baseline_roll
            elif self.config.method == NormalizationMethod.Z_SCORE:
                yaw_norm = (yaw - self.config.global_mean_yaw) / self.config.global_std_yaw
                pitch_norm = (pitch - self.config.global_mean_pitch) / self.config.global_std_pitch
                roll_norm = (roll - self.config.global_mean_roll) / self.config.global_std_roll
            elif self.config.method == NormalizationMethod.WINDOW_RELATIVE:
                yaw_norm = yaw - yaws[0]
                pitch_norm = pitch - pitches[0]
                roll_norm = roll - rolls[0]
            else:  # NONE
                yaw_norm = yaw
                pitch_norm = pitch
                roll_norm = roll

            # 计算差分
            dt = t - prev_time if t > prev_time else 1.0 / self.fps
            d_yaw = self._yaw_diff(prev_yaw, yaw) / dt
            d_pitch = (pitch - prev_pitch) / dt
            d_roll = (roll - prev_roll) / dt

            results.append(NormalizedPose(
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                yaw_norm=yaw_norm,
                pitch_norm=pitch_norm,
                roll_norm=roll_norm,
                d_yaw=d_yaw,
                d_pitch=d_pitch,
                d_roll=d_roll,
                timestamp=t
            ))

            prev_yaw, prev_pitch, prev_roll = yaw, pitch, roll
            prev_time = t

        return results

    def _unwrap_yaw(self, yaws: np.ndarray) -> np.ndarray:
        """处理 yaw 的 ±180° 跳变，使序列连续"""
        yaws = np.array(yaws, dtype=float)
        if len(yaws) < 2:
            return yaws

        unwrapped = [yaws[0]]
        for i in range(1, len(yaws)):
            diff = yaws[i] - unwrapped[-1]
            if diff > 180:
                unwrapped.append(yaws[i] - 360)
            elif diff < -180:
                unwrapped.append(yaws[i] + 360)
            else:
                unwrapped.append(yaws[i])

        return np.array(unwrapped)

    def _yaw_diff(self, yaw1: float, yaw2: float) -> float:
        """计算 yaw 差分，处理跳变"""
        diff = yaw2 - yaw1
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        return diff


# ============================================================================
# 去个体化时序特征提取器
# ============================================================================
@dataclass
class BehaviorFeature:
    """去个体化的行为特征"""
    window_start: float
    window_end: float
    track_id: int

    # 主要特征（差分/统计）
    yaw_speed_mean: float      # 平均转头速度
    yaw_speed_max: float       # 最大转头速度
    yaw_speed_std: float       # 速度变化性
    yaw_switch_count: int      # 方向切换次数
    yaw_std: float             # yaw 标准差
    yaw_range: float           # yaw 范围

    # 相对特征
    yaw_rel_mean: float        # 相对 baseline 的平均偏转
    yaw_rel_max: float         # 相对 baseline 的最大偏转
    yaw_rel_min: float         # 相对 baseline 的最小偏转

    # Pitch/Roll 特征
    pitch_std: float
    pitch_range: float
    roll_std: float

    # 行为指标
    prolonged_side_ratio: float  # 持续侧向时间比例（|yaw|>30°）
    quick_turn_count: int        # 快速回头次数（速度>60°/s）

    # 样本信息
    sample_count: int

    def to_dict(self) -> dict:
        return {
            "window_start": round(self.window_start, 3),
            "window_end": round(self.window_end, 3),
            "track_id": self.track_id,
            "yaw_speed_mean": round(self.yaw_speed_mean, 2),
            "yaw_speed_max": round(self.yaw_speed_max, 2),
            "yaw_speed_std": round(self.yaw_speed_std, 2),
            "yaw_switch_count": self.yaw_switch_count,
            "yaw_std": round(self.yaw_std, 2),
            "yaw_range": round(self.yaw_range, 2),
            "yaw_rel_mean": round(self.yaw_rel_mean, 2),
            "yaw_rel_max": round(self.yaw_rel_max, 2),
            "yaw_rel_min": round(self.yaw_rel_min, 2),
            "pitch_std": round(self.pitch_std, 2),
            "pitch_range": round(self.pitch_range, 2),
            "roll_std": round(self.roll_std, 2),
            "prolonged_side_ratio": round(self.prolonged_side_ratio, 3),
            "quick_turn_count": self.quick_turn_count,
            "sample_count": self.sample_count
        }

    def to_vector(self) -> np.ndarray:
        """转为特征向量（用于模型输入）"""
        return np.array([
            self.yaw_speed_mean,
            self.yaw_speed_max,
            self.yaw_speed_std,
            self.yaw_switch_count,
            self.yaw_std,
            self.yaw_range,
            self.yaw_rel_mean,
            self.yaw_rel_max,
            self.pitch_std,
            self.pitch_range,
            self.roll_std,
            self.prolonged_side_ratio,
            self.quick_turn_count
        ])


class BehaviorFeatureExtractor:
    """去个体化行为特征提取器"""

    def __init__(
        self,
        fps: float = 10.0,
        window_size: float = 3.0,      # 窗口大小（秒）
        step_size: float = 0.5,        # 步长（秒）
        min_samples: int = 10,         # 最小样本数
        switch_threshold: float = 15.0,  # 方向切换阈值（度）
        quick_turn_speed: float = 60.0,  # 快速回头速度阈值（度/秒）
        side_gaze_threshold: float = 30.0  # 侧向阈值（度）
    ):
        self.fps = fps
        self.window_size = window_size
        self.step_size = step_size
        self.min_samples = min_samples
        self.switch_threshold = switch_threshold
        self.quick_turn_speed = quick_turn_speed
        self.side_gaze_threshold = side_gaze_threshold

        self.normalizer = PoseNormalizer(fps=fps)

    def extract_from_track(
        self,
        timestamps: np.ndarray,
        yaws: np.ndarray,
        pitches: np.ndarray,
        rolls: np.ndarray,
        track_id: int
    ) -> Tuple[List[BehaviorFeature], TrackPoseStats]:
        """从 track 提取去个体化特征

        Returns:
            (features, baseline): 特征列表和 baseline 统计
        """
        if len(timestamps) < self.min_samples:
            return [], None

        # 排序
        sort_idx = np.argsort(timestamps)
        timestamps = timestamps[sort_idx]
        yaws = yaws[sort_idx]
        pitches = pitches[sort_idx]
        rolls = rolls[sort_idx]

        # 计算 baseline
        baseline = self.normalizer.compute_track_baseline(
            timestamps, yaws, pitches, rolls
        )
        baseline.track_id = track_id

        # 归一化序列
        normalized = self.normalizer.normalize_sequence(
            timestamps, yaws, pitches, rolls, baseline
        )

        # 滑动窗口提取特征
        features = []
        t_start = timestamps[0]
        t_end = timestamps[-1]

        window_start = t_start
        while window_start + self.window_size <= t_end + self.step_size:
            window_end = window_start + self.window_size

            # 获取窗口内的数据
            mask = (timestamps >= window_start) & (timestamps < window_end)
            w_indices = np.where(mask)[0]

            if len(w_indices) >= self.min_samples:
                w_normalized = [normalized[i] for i in w_indices]
                feature = self._compute_window_features(
                    w_normalized, track_id, window_start, window_end, baseline
                )
                if feature is not None:
                    features.append(feature)

            window_start += self.step_size

        return features, baseline

    def _compute_window_features(
        self,
        poses: List[NormalizedPose],
        track_id: int,
        window_start: float,
        window_end: float,
        baseline: TrackPoseStats
    ) -> BehaviorFeature:
        """计算单个窗口的特征"""
        n = len(poses)

        # 提取数组
        yaws_norm = np.array([p.yaw_norm for p in poses])
        pitches = np.array([p.pitch for p in poses])
        rolls = np.array([p.roll for p in poses])
        d_yaws = np.array([p.d_yaw for p in poses])

        # 速度特征（使用绝对值）
        speeds = np.abs(d_yaws)
        yaw_speed_mean = np.mean(speeds)
        yaw_speed_max = np.max(speeds)
        yaw_speed_std = np.std(speeds)

        # 方向切换次数
        switch_count = self._count_switches(d_yaws)

        # 统计特征
        yaw_std = np.std(yaws_norm)
        yaw_range = np.max(yaws_norm) - np.min(yaws_norm)

        # 相对特征
        yaw_rel_mean = np.mean(yaws_norm)
        yaw_rel_max = np.max(yaws_norm)
        yaw_rel_min = np.min(yaws_norm)

        # Pitch/Roll
        pitch_std = np.std(pitches)
        pitch_range = np.max(pitches) - np.min(pitches)
        roll_std = np.std(rolls)

        # 持续侧向比例
        side_mask = np.abs(yaws_norm) > self.side_gaze_threshold
        prolonged_side_ratio = np.sum(side_mask) / n

        # 快速回头次数
        quick_turn_count = np.sum(speeds > self.quick_turn_speed)

        return BehaviorFeature(
            window_start=window_start,
            window_end=window_end,
            track_id=track_id,
            yaw_speed_mean=yaw_speed_mean,
            yaw_speed_max=yaw_speed_max,
            yaw_speed_std=yaw_speed_std,
            yaw_switch_count=switch_count,
            yaw_std=yaw_std,
            yaw_range=yaw_range,
            yaw_rel_mean=yaw_rel_mean,
            yaw_rel_max=yaw_rel_max,
            yaw_rel_min=yaw_rel_min,
            pitch_std=pitch_std,
            pitch_range=pitch_range,
            roll_std=roll_std,
            prolonged_side_ratio=prolonged_side_ratio,
            quick_turn_count=quick_turn_count,
            sample_count=n
        )

    def _count_switches(self, d_yaws: np.ndarray) -> int:
        """计算方向切换次数"""
        if len(d_yaws) < 3:
            return 0

        # 累积变化
        cumsum = 0
        switches = 0
        last_sign = 0

        for d in d_yaws:
            cumsum += d

            if abs(cumsum) >= self.switch_threshold:
                current_sign = 1 if cumsum > 0 else -1
                if last_sign != 0 and current_sign != last_sign:
                    switches += 1
                last_sign = current_sign
                cumsum = 0

        return switches


# ============================================================================
# 特征向量标准化（用于模型训练）
# ============================================================================
FEATURE_NORMALIZATION = {
    # 特征名: (期望均值, 期望标准差) - 用于标准化
    "yaw_speed_mean": (20.0, 15.0),
    "yaw_speed_max": (50.0, 30.0),
    "yaw_speed_std": (15.0, 10.0),
    "yaw_switch_count": (2.0, 2.0),
    "yaw_std": (15.0, 10.0),
    "yaw_range": (40.0, 25.0),
    "yaw_rel_mean": (0.0, 20.0),
    "yaw_rel_max": (20.0, 15.0),
    "pitch_std": (5.0, 3.0),
    "pitch_range": (15.0, 10.0),
    "roll_std": (3.0, 2.0),
    "prolonged_side_ratio": (0.3, 0.25),
    "quick_turn_count": (1.0, 2.0),
}


def normalize_feature_vector(features: np.ndarray) -> np.ndarray:
    """标准化特征向量"""
    feature_names = list(FEATURE_NORMALIZATION.keys())
    normalized = np.zeros_like(features)

    for i, name in enumerate(feature_names):
        if i < len(features):
            mean, std = FEATURE_NORMALIZATION[name]
            normalized[i] = (features[i] - mean) / std

    return normalized
