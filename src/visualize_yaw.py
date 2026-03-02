"""
Yaw 时序可视化 + 规则触发标注
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple

from temporal_features import TemporalFeatureExtractor
from rule_engine import RuleEngine


def load_pose_data(csv_path: str, track_id: int = 0) -> pd.DataFrame:
    """加载姿态数据，按 track_id 筛选"""
    df = pd.read_csv(csv_path)

    # 添加 track_id（按帧内顺序）
    df['track_id'] = df.groupby('frame_id').cumcount()

    # 筛选指定 track
    df = df[df['track_id'] == track_id].copy()
    df = df.sort_values('time_sec').reset_index(drop=True)

    return df


def extract_features_and_evaluate(df: pd.DataFrame, track_id: int = 0) -> List[dict]:
    """提取时序特征并评估规则

    Returns:
        包含窗口信息和规则评估结果的列表
    """
    extractor = TemporalFeatureExtractor()
    engine = RuleEngine()

    times = df['time_sec'].values
    yaws = df['yaw'].values

    # 提取时序特征
    features = extractor.extract_from_track(times, yaws, track_id)

    # 评估每个窗口
    results = []
    for feat in features:
        feat_dict = feat.to_dict()
        feat_dict['yaws'] = yaws[
            (times >= feat.window_start) & (times < feat.window_end)
        ]

        eval_result = engine.evaluate(feat_dict)
        results.append({
            'window_start': feat.window_start,
            'window_end': feat.window_end,
            'is_suspicious': eval_result.is_suspicious,
            'weighted_score': eval_result.weighted_score,
            'triggered_rules': [r.rule_name for r in eval_result.rules if r.triggered],
            'rules': eval_result.to_dict()['rules']
        })

    return results


# 规则颜色映射
RULE_COLORS = {
    'sustained_side_gaze': '#FF6B6B',   # 红色
    'frequent_scanning': '#4ECDC4',      # 青色
    'high_variability': '#FFE66D',       # 黄色
    'wide_range_turn': '#95E1D3'         # 绿色
}

RULE_NAMES_CN = {
    'sustained_side_gaze': '持续侧向',
    'frequent_scanning': '频繁扫视',
    'high_variability': '高变异性',
    'wide_range_turn': '大范围转头'
}


def plot_yaw_with_rules(
    df: pd.DataFrame,
    eval_results: List[dict],
    title: str = 'Yaw 时序变化',
    save_path: str = None
):
    """绑制 yaw 时序图并标注规则触发区间

    Args:
        df: 姿态数据 DataFrame
        eval_results: 规则评估结果列表
        title: 图表标题
        save_path: 保存路径（可选）
    """
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])
    fig.suptitle(title, fontsize=14, fontweight='bold')

    times = df['time_sec'].values
    yaws = df['yaw'].values

    # 上图：yaw 曲线
    ax1.plot(times, yaws, 'b-', linewidth=0.8, alpha=0.8, label='Yaw')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=30, color='red', linestyle=':', alpha=0.5, label='±30° 阈值')
    ax1.axhline(y=-30, color='red', linestyle=':', alpha=0.5)

    ax1.set_ylabel('Yaw (度)', fontsize=11)
    ax1.set_ylim(-180, 180)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # 标注规则触发区间（阴影）
    for res in eval_results:
        if res['triggered_rules']:
            for rule in res['triggered_rules']:
                color = RULE_COLORS.get(rule, 'gray')
                ax1.axvspan(
                    res['window_start'], res['window_end'],
                    alpha=0.2, color=color
                )

    # 下图：可疑分数时间线
    window_times = [(r['window_start'] + r['window_end']) / 2 for r in eval_results]
    scores = [r['weighted_score'] for r in eval_results]

    ax2.fill_between(window_times, scores, alpha=0.5, color='orange')
    ax2.plot(window_times, scores, 'o-', color='darkorange', markersize=3)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='可疑阈值')

    ax2.set_xlabel('时间 (秒)', fontsize=11)
    ax2.set_ylabel('可疑分数', fontsize=11)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    # 添加规则图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, alpha=0.3, label=RULE_NAMES_CN[rule])
        for rule, color in RULE_COLORS.items()
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")

    plt.show()
    return fig


def print_summary(eval_results: List[dict]):
    """打印规则触发统计"""
    total = len(eval_results)
    suspicious = sum(1 for r in eval_results if r['is_suspicious'])

    print(f"\n{'='*50}")
    print(f"时序窗口统计")
    print(f"{'='*50}")
    print(f"总窗口数: {total}")
    print(f"可疑窗口数: {suspicious} ({suspicious/total*100:.1f}%)")

    # 各规则触发统计
    rule_counts = {rule: 0 for rule in RULE_COLORS.keys()}
    for res in eval_results:
        for rule in res['triggered_rules']:
            rule_counts[rule] += 1

    print(f"\n规则触发统计:")
    for rule, count in rule_counts.items():
        cn_name = RULE_NAMES_CN[rule]
        print(f"  {cn_name}: {count} 次 ({count/total*100:.1f}%)")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Yaw 时序可视化')
    parser.add_argument('csv_path', help='姿态 CSV 文件路径')
    parser.add_argument('--track-id', type=int, default=0, help='轨迹 ID')
    parser.add_argument('--save', help='保存图片路径')
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    video_name = csv_path.stem

    print(f"加载数据: {csv_path}")
    df = load_pose_data(str(csv_path), args.track_id)
    print(f"数据点数: {len(df)}")

    print("提取时序特征并评估规则...")
    eval_results = extract_features_and_evaluate(df, args.track_id)

    print_summary(eval_results)

    title = f'{video_name} - Track {args.track_id} Yaw 时序分析'
    plot_yaw_with_rules(df, eval_results, title=title, save_path=args.save)


if __name__ == '__main__':
    main()
