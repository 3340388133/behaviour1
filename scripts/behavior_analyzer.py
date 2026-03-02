"""
可疑张望行为检测 - 时序特征分析
基于滑动窗口的多维特征提取与规则引擎
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class WindowFeatures:
    """滑动窗口特征"""
    start_time: float
    end_time: float
    yaw_mean: float
    yaw_std: float
    yaw_range: float
    yaw_speed_mean: float
    yaw_switch_count: int
    suspicious_score: float
    triggered_rules: List[str]


class SuspiciousBehaviorAnalyzer:
    """可疑行为分析器"""

    def __init__(
        self,
        window_size: float = 2.0,
        step_size: float = 0.5,
        fps: float = 30.0
    ):
        """
        Args:
            window_size: 滑动窗口大小(秒)
            step_size: 滑动步长(秒)
            fps: 视频帧率
        """
        self.window_size = window_size
        self.step_size = step_size
        self.fps = fps

        # 规则阈值 (可调参数)
        self.thresholds = {
            'side_gaze_angle': 30,      # 侧向阈值(度)
            'side_gaze_ratio': 0.8,     # 持续侧向比例
            'switch_count': 3,          # 切换次数阈值
            'speed_threshold': 20,      # 角速度阈值(度/秒)
            'range_threshold': 60,      # yaw范围阈值(度)
            'std_threshold': 15,        # 标准差阈值(度)
        }

        # 规则权重
        self.weights = {
            'sustained_side': 0.35,
            'frequent_scan': 0.30,
            'high_variability': 0.20,
            'large_range': 0.15,
        }

    def load_csv(self, csv_path: str) -> pd.DataFrame:
        """加载CSV数据"""
        df = pd.read_csv(csv_path)
        # 标准化列名
        df.columns = df.columns.str.strip().str.lower()
        # 确保必要列存在
        required = ['frame_id', 'time', 'yaw', 'pitch', 'roll', 'conf']
        for col in required:
            if col not in df.columns:
                # 尝试匹配
                for c in df.columns:
                    if col in c.lower():
                        df = df.rename(columns={c: col})
                        break
        return df

    def extract_window_features(self, window_df: pd.DataFrame) -> dict:
        """提取单个窗口的特征"""
        yaws = window_df['yaw'].values
        times = window_df['time'].values

        # 1. 统计特征
        yaw_mean = np.mean(yaws)
        yaw_std = np.std(yaws)
        yaw_range = np.max(yaws) - np.min(yaws)

        # 2. 动力学特征: 角速度
        if len(yaws) > 1:
            dt = np.diff(times)
            dt[dt == 0] = 1e-6  # 避免除零
            speeds = np.abs(np.diff(yaws)) / dt
            yaw_speed_mean = np.mean(speeds)
        else:
            yaw_speed_mean = 0

        # 3. 切换特征: 左右切换次数
        yaw_switch_count = self._count_switches(yaws)

        return {
            'yaw_mean': yaw_mean,
            'yaw_std': yaw_std,
            'yaw_range': yaw_range,
            'yaw_speed_mean': yaw_speed_mean,
            'yaw_switch_count': yaw_switch_count,
            'yaws': yaws  # 保留原始数据用于规则判断
        }

    def _count_switches(self, yaws: np.ndarray) -> int:
        """计算左右切换次数"""
        if len(yaws) < 2:
            return 0
        signs = np.sign(yaws)
        switches = np.sum(np.abs(np.diff(signs)) > 0)
        return int(switches)

    def apply_rules(self, features: dict) -> Tuple[float, List[str]]:
        """
        应用规则引擎计算可疑分数
        Returns:
            (score, triggered_rules)
        """
        score = 0.0
        triggered = []
        yaws = features['yaws']
        th = self.thresholds

        # 规则1: 持续侧向注视
        side_ratio = np.mean(np.abs(yaws) > th['side_gaze_angle'])
        if side_ratio >= th['side_gaze_ratio']:
            score += self.weights['sustained_side']
            triggered.append(f"sustained_side({side_ratio:.0%})")

        # 规则2: 频繁扫视
        if (features['yaw_switch_count'] >= th['switch_count'] and
            features['yaw_speed_mean'] > th['speed_threshold']):
            score += self.weights['frequent_scan']
            triggered.append(f"frequent_scan(n={features['yaw_switch_count']})")

        # 规则3: 高变异性
        if features['yaw_std'] > th['std_threshold']:
            score += self.weights['high_variability']
            triggered.append(f"high_var(std={features['yaw_std']:.1f})")

        # 规则4: 大范围转头
        if features['yaw_range'] > th['range_threshold']:
            score += self.weights['large_range']
            triggered.append(f"large_range({features['yaw_range']:.1f}°)")

        return min(1.0, score), triggered

    def analyze(self, df: pd.DataFrame) -> List[WindowFeatures]:
        """滑动窗口分析"""
        results = []
        times = df['time'].values
        start_time = times[0]
        end_time = times[-1]

        window_start = start_time
        while window_start + self.window_size <= end_time:
            window_end = window_start + self.window_size

            # 提取窗口数据
            mask = (df['time'] >= window_start) & (df['time'] < window_end)
            window_df = df[mask]

            if len(window_df) < 2:
                window_start += self.step_size
                continue

            # 提取特征
            features = self.extract_window_features(window_df)

            # 应用规则
            score, triggered = self.apply_rules(features)

            results.append(WindowFeatures(
                start_time=round(window_start, 3),
                end_time=round(window_end, 3),
                yaw_mean=round(features['yaw_mean'], 2),
                yaw_std=round(features['yaw_std'], 2),
                yaw_range=round(features['yaw_range'], 2),
                yaw_speed_mean=round(features['yaw_speed_mean'], 2),
                yaw_switch_count=features['yaw_switch_count'],
                suspicious_score=round(score, 3),
                triggered_rules=triggered
            ))

            window_start += self.step_size

        return results

    def get_suspicious_segments(
        self,
        results: List[WindowFeatures],
        score_threshold: float = 0.3
    ) -> List[dict]:
        """合并连续的可疑时间段"""
        segments = []
        current = None

        for w in results:
            if w.suspicious_score >= score_threshold:
                if current is None:
                    current = {
                        'start': w.start_time,
                        'end': w.end_time,
                        'max_score': w.suspicious_score,
                        'rules': set(w.triggered_rules)
                    }
                else:
                    current['end'] = w.end_time
                    current['max_score'] = max(current['max_score'], w.suspicious_score)
                    current['rules'].update(w.triggered_rules)
            else:
                if current is not None:
                    current['rules'] = list(current['rules'])
                    segments.append(current)
                    current = None

        if current is not None:
            current['rules'] = list(current['rules'])
            segments.append(current)

        return segments


def analyze_csv(
    csv_path: str,
    output_path: str = None,
    window_size: float = 2.0,
    step_size: float = 0.5,
    score_threshold: float = 0.3
) -> dict:
    """
    分析CSV文件并输出结果

    Args:
        csv_path: 输入CSV路径
        output_path: 输出JSON路径
        window_size: 窗口大小(秒)
        step_size: 步长(秒)
        score_threshold: 可疑阈值

    Returns:
        分析结果字典
    """
    analyzer = SuspiciousBehaviorAnalyzer(
        window_size=window_size,
        step_size=step_size
    )

    # 加载数据
    df = analyzer.load_csv(csv_path)
    print(f"加载 {len(df)} 帧数据")

    # 分析
    results = analyzer.analyze(df)
    print(f"生成 {len(results)} 个窗口")

    # 获取可疑时间段
    segments = analyzer.get_suspicious_segments(results, score_threshold)

    # 构建输出
    output = {
        'config': {
            'window_size': window_size,
            'step_size': step_size,
            'score_threshold': score_threshold,
            'thresholds': analyzer.thresholds
        },
        'summary': {
            'total_windows': len(results),
            'suspicious_windows': sum(1 for r in results if r.suspicious_score >= score_threshold),
            'suspicious_segments': len(segments)
        },
        'windows': [vars(r) for r in results],
        'suspicious_segments': segments
    }

    # 保存
    if output_path:
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {output_path}")

    # 打印摘要
    print(f"\n=== 分析结果 ===")
    print(f"可疑窗口: {output['summary']['suspicious_windows']}/{len(results)}")
    print(f"可疑时间段: {len(segments)} 个")
    for i, seg in enumerate(segments):
        print(f"  [{i+1}] {seg['start']:.1f}s - {seg['end']:.1f}s "
              f"(score={seg['max_score']:.2f})")

    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='可疑张望行为分析')
    parser.add_argument('--csv', '-c', required=True, help='输入CSV文件')
    parser.add_argument('--output', '-o', default='analysis.json', help='输出JSON')
    parser.add_argument('--window', '-w', type=float, default=2.0, help='窗口大小(秒)')
    parser.add_argument('--step', '-s', type=float, default=0.5, help='步长(秒)')
    parser.add_argument('--threshold', '-t', type=float, default=0.3, help='可疑阈值')
    args = parser.parse_args()

    analyze_csv(args.csv, args.output, args.window, args.step, args.threshold)
