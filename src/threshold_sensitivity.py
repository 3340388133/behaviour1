"""
规则阈值敏感性分析
对阈值进行 ±20% 扰动，统计规则触发次数变化
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

from temporal_features import TemporalFeatureExtractor


@dataclass
class ThresholdConfig:
    """阈值配置"""
    side_yaw: float = 30.0        # 持续侧向: |yaw| > 30°
    side_ratio: float = 0.8       # 持续侧向: 比例 > 0.8
    scan_switch: int = 3          # 频繁扫视: 切换 >= 3
    scan_speed: float = 20.0      # 频繁扫视: 速度 > 20°/s
    var_std: float = 15.0         # 高变异性: yaw_std > 15°
    range_deg: float = 60.0       # 大范围转头: yaw_range > 60°


def evaluate_with_thresholds(features: Dict, config: ThresholdConfig) -> Dict[str, bool]:
    """使用指定阈值评估规则触发"""
    yaws = features.get('yaws', np.array([]))

    # 规则1: 持续侧向
    if len(yaws) > 0:
        side_ratio = np.mean(np.abs(yaws) > config.side_yaw)
    else:
        side_ratio = 1.0 if abs(features.get('yaw_mean', 0)) > config.side_yaw else 0.0
    sustained_side = side_ratio > config.side_ratio

    # 规则2: 频繁扫视
    switch_count = features.get('yaw_switch_count', 0)
    speed = features.get('yaw_speed_mean', 0)
    frequent_scan = switch_count >= config.scan_switch and speed > config.scan_speed

    # 规则3: 高变异性
    yaw_std = features.get('yaw_std', 0)
    high_var = yaw_std > config.var_std

    # 规则4: 大范围转头
    yaw_range = features.get('yaw_range', 0)
    wide_range = yaw_range > config.range_deg

    return {
        'sustained_side_gaze': sustained_side,
        'frequent_scanning': frequent_scan,
        'high_variability': high_var,
        'wide_range_turn': wide_range
    }


def load_and_extract_features(csv_path: str) -> List[Dict]:
    """加载CSV并提取所有track的时序特征"""
    df = pd.read_csv(csv_path)
    df['track_id'] = df.groupby('frame_id').cumcount()

    extractor = TemporalFeatureExtractor()
    all_features = []

    for track_id in df['track_id'].unique():
        track_df = df[df['track_id'] == track_id].sort_values('time_sec')
        times = track_df['time_sec'].values
        yaws = track_df['yaw'].values

        features = extractor.extract_from_track(times, yaws, track_id)
        for feat in features:
            feat_dict = feat.to_dict()
            mask = (times >= feat.window_start) & (times < feat.window_end)
            feat_dict['yaws'] = yaws[mask]
            all_features.append(feat_dict)

    return all_features


def sensitivity_analysis(
    csv_paths: List[str],
    param_name: str,
    base_value: float,
    perturbations: List[float] = [-0.2, 0, 0.2]
) -> pd.DataFrame:
    """
    阈值敏感性分析

    Args:
        csv_paths: pose CSV 文件路径列表
        param_name: 要分析的参数名 (side_yaw, side_ratio, scan_switch, scan_speed, var_std, range_deg)
        base_value: 基准值
        perturbations: 扰动比例列表，如 [-0.2, 0, 0.2] 表示 -20%, 0%, +20%

    Returns:
        对比表 DataFrame
    """
    # 收集所有特征
    print(f"加载数据...")
    all_features = []
    for csv_path in csv_paths:
        features = load_and_extract_features(csv_path)
        all_features.extend(features)

    print(f"总窗口数: {len(all_features)}")

    # 对每个扰动值进行评估
    results = []
    for pert in perturbations:
        value = base_value * (1 + pert)
        if param_name == 'scan_switch':
            value = max(1, int(round(value)))

        # 创建配置
        config = ThresholdConfig()
        setattr(config, param_name, value)

        # 统计触发次数
        counts = {
            'sustained_side_gaze': 0,
            'frequent_scanning': 0,
            'high_variability': 0,
            'wide_range_turn': 0
        }

        for feat in all_features:
            triggered = evaluate_with_thresholds(feat, config)
            for rule, is_triggered in triggered.items():
                if is_triggered:
                    counts[rule] += 1

        # 计算触发率
        total = len(all_features)
        results.append({
            'perturbation': f'{pert:+.0%}',
            f'{param_name}': round(value, 2),
            '持续侧向': f"{counts['sustained_side_gaze']} ({counts['sustained_side_gaze']/total*100:.1f}%)",
            '频繁扫视': f"{counts['frequent_scanning']} ({counts['frequent_scanning']/total*100:.1f}%)",
            '高变异性': f"{counts['high_variability']} ({counts['high_variability']/total*100:.1f}%)",
            '大范围转头': f"{counts['wide_range_turn']} ({counts['wide_range_turn']/total*100:.1f}%)",
        })

    return pd.DataFrame(results)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='规则阈值敏感性分析')
    parser.add_argument('--data-dir', default='../data/pose_results', help='pose CSV 目录')
    parser.add_argument('--param', default='var_std',
                       choices=['side_yaw', 'side_ratio', 'scan_switch', 'scan_speed', 'var_std', 'range_deg'],
                       help='要分析的参数')
    parser.add_argument('--base', type=float, help='基准值（默认使用规则引擎默认值）')
    args = parser.parse_args()

    # 默认基准值
    defaults = {
        'side_yaw': 30.0,
        'side_ratio': 0.8,
        'scan_switch': 3,
        'scan_speed': 20.0,
        'var_std': 15.0,
        'range_deg': 60.0
    }

    base_value = args.base if args.base else defaults[args.param]

    # 获取所有 CSV 文件
    data_dir = Path(args.data_dir)
    csv_paths = sorted(data_dir.glob('*.csv'))
    print(f"找到 {len(csv_paths)} 个 pose CSV 文件")

    # 运行敏感性分析
    print(f"\n分析参数: {args.param}, 基准值: {base_value}")
    print(f"扰动: -20%, 0%, +20%\n")

    df = sensitivity_analysis(
        [str(p) for p in csv_paths],
        args.param,
        base_value,
        [-0.2, 0, 0.2]
    )

    print("\n" + "="*70)
    print(f"阈值敏感性分析结果 ({args.param})")
    print("="*70)
    print(df.to_string(index=False))

    return df


if __name__ == '__main__':
    main()
