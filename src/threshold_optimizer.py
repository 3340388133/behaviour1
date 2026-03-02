"""
规则阈值优化
目标：找到使规则触发率稳定且合理的最优阈值
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
import itertools

from temporal_features import TemporalFeatureExtractor


@dataclass
class ThresholdConfig:
    """阈值配置"""
    side_yaw: float = 30.0
    side_ratio: float = 0.8
    scan_switch: int = 3
    scan_speed: float = 20.0
    var_std: float = 15.0
    range_deg: float = 60.0


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


def load_all_features(data_dir: str) -> List[Dict]:
    """加载所有CSV并提取时序特征"""
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob('*.csv'))

    extractor = TemporalFeatureExtractor()
    all_features = []

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        df['track_id'] = df.groupby('frame_id').cumcount()

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


def compute_stability_score(rates: List[float]) -> float:
    """计算稳定性分数（标准差越小越稳定）"""
    if len(rates) < 2:
        return 0.0
    return 1.0 / (1.0 + np.std(rates))


def grid_search_optimal(all_features: List[Dict]) -> pd.DataFrame:
    """网格搜索最优阈值"""
    # 搜索空间
    search_space = {
        'side_yaw': [25, 30, 35, 40],
        'side_ratio': [0.6, 0.7, 0.8],
        'scan_switch': [2, 3, 4],
        'scan_speed': [10, 15, 20, 25],
        'var_std': [12, 15, 18, 20],
        'range_deg': [50, 60, 70, 80]
    }

    results = []
    total = len(all_features)

    # 单参数优化（固定其他参数）
    for param_name, values in search_space.items():
        for value in values:
            config = ThresholdConfig()
            setattr(config, param_name, value)

            counts = {rule: 0 for rule in ['sustained_side_gaze', 'frequent_scanning', 'high_variability', 'wide_range_turn']}
            for feat in all_features:
                triggered = evaluate_with_thresholds(feat, config)
                for rule, is_triggered in triggered.items():
                    if is_triggered:
                        counts[rule] += 1

            rates = {rule: count / total * 100 for rule, count in counts.items()}
            results.append({
                'param': param_name,
                'value': value,
                **{f'{rule}_rate': f"{rates[rule]:.1f}%" for rule in counts.keys()}
            })

    return pd.DataFrame(results)


def find_optimal_thresholds(all_features: List[Dict]) -> Dict:
    """找到最优阈值组合"""
    total = len(all_features)

    # 分析特征分布
    yaw_stds = [f['yaw_std'] for f in all_features]
    yaw_ranges = [f['yaw_range'] for f in all_features]
    switch_counts = [f['yaw_switch_count'] for f in all_features]
    speeds = [f['yaw_speed_mean'] for f in all_features]

    print("\n特征分布统计:")
    print(f"  yaw_std:    mean={np.mean(yaw_stds):.1f}, median={np.median(yaw_stds):.1f}, p75={np.percentile(yaw_stds, 75):.1f}")
    print(f"  yaw_range:  mean={np.mean(yaw_ranges):.1f}, median={np.median(yaw_ranges):.1f}, p75={np.percentile(yaw_ranges, 75):.1f}")
    print(f"  switch_cnt: mean={np.mean(switch_counts):.1f}, median={np.median(switch_counts):.1f}, max={max(switch_counts)}")
    print(f"  speed:      mean={np.mean(speeds):.1f}, median={np.median(speeds):.1f}, p75={np.percentile(speeds, 75):.1f}")

    # 基于分布推荐阈值（使用75分位数作为参考）
    recommended = {
        'var_std': round(np.percentile(yaw_stds, 75), 0),
        'range_deg': round(np.percentile(yaw_ranges, 75), 0),
        'scan_switch': max(1, int(np.percentile(switch_counts, 90))),
        'scan_speed': round(np.percentile(speeds, 75), 0)
    }

    print(f"\n基于分布推荐的阈值:")
    for k, v in recommended.items():
        print(f"  {k}: {v}")

    return recommended


def main():
    import argparse

    parser = argparse.ArgumentParser(description='规则阈值优化')
    parser.add_argument('--data-dir', default='../data/pose_results', help='pose CSV 目录')
    args = parser.parse_args()

    print("加载数据...")
    all_features = load_all_features(args.data_dir)
    print(f"总窗口数: {len(all_features)}")

    # 分析特征分布并推荐阈值
    recommended = find_optimal_thresholds(all_features)

    # 网格搜索
    print("\n网格搜索各参数...")
    df = grid_search_optimal(all_features)

    # 按参数分组显示
    for param in df['param'].unique():
        print(f"\n{'='*60}")
        print(f"参数: {param}")
        print('='*60)
        param_df = df[df['param'] == param]
        print(param_df.to_string(index=False))

    # 输出最优配置
    print("\n" + "="*60)
    print("推荐最优阈值配置")
    print("="*60)

    optimal = ThresholdConfig(
        side_yaw=35,           # 提高以降低敏感性
        side_ratio=0.7,        # 略微降低
        scan_switch=2,         # 降低以提高触发率
        scan_speed=15,         # 降低以提高触发率
        var_std=recommended.get('var_std', 15),
        range_deg=recommended.get('range_deg', 60)
    )

    print(f"  side_yaw: {optimal.side_yaw}° (原30°)")
    print(f"  side_ratio: {optimal.side_ratio} (原0.8)")
    print(f"  scan_switch: {optimal.scan_switch} (原3)")
    print(f"  scan_speed: {optimal.scan_speed}°/s (原20°/s)")
    print(f"  var_std: {optimal.var_std}° (原15°)")
    print(f"  range_deg: {optimal.range_deg}° (原60°)")

    # 验证最优配置
    print("\n验证最优配置触发率:")
    counts = {rule: 0 for rule in ['sustained_side_gaze', 'frequent_scanning', 'high_variability', 'wide_range_turn']}
    for feat in all_features:
        triggered = evaluate_with_thresholds(feat, optimal)
        for rule, is_triggered in triggered.items():
            if is_triggered:
                counts[rule] += 1

    total = len(all_features)
    for rule, count in counts.items():
        print(f"  {rule}: {count} ({count/total*100:.1f}%)")


if __name__ == '__main__':
    main()
