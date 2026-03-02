#!/usr/bin/env python3
"""
生成硕士毕业论文可视化图表
基于头部姿态估计的口岸人员可疑张望行为识别系统
"""

import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import defaultdict
import os

# ============ 中文字体与论文级样式 ============
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.figsize'] = (10, 6)

OUT_DIR = 'thesis_figures'
STATS_DIR = 'data/batch_inference_output'

# 行为标签映射
BEHAVIOR_LABELS = {
    '0': 'Normal\n(正常)',
    '1': 'Glancing\n(频繁张望)',
    '2': 'QuickTurn\n(快速回头)',
    '3': 'Prolonged\n(长时间观察)',
    '4': 'LookDown\n(持续低头)',
    '5': 'LookUp\n(持续抬头)',
}

BEHAVIOR_LABELS_SHORT = {
    '0': '正常行为',
    '1': '频繁张望',
    '2': '快速回头',
    '3': '长时间观察',
    '4': '持续低头',
    '5': '持续抬头',
}

BEHAVIOR_COLORS = {
    '0': '#4CAF50',  # 绿
    '1': '#FF9800',  # 橙
    '2': '#F44336',  # 红
    '3': '#9C27B0',  # 紫
    '4': '#2196F3',  # 蓝
    '5': '#00BCD4',  # 青
}

# ============ 加载数据 ============
def load_all_stats():
    stats = []
    for f in sorted(glob.glob(f'{STATS_DIR}/*_inference_stats.json')):
        with open(f) as fp:
            d = json.load(fp)
        stats.append(d)
    return stats

all_stats = load_all_stats()

# 过滤掉太短的视频 (1.14zz-2 只有48帧)
valid_stats = [s for s in all_stats if s['total_frames_processed'] > 100]
all_video_names = [s['video_name'] for s in valid_stats]

print(f"加载 {len(all_stats)} 个视频统计，有效 {len(valid_stats)} 个")

# ============ 图1: 各视频行为分布堆叠柱状图 ============
def fig1_behavior_distribution_stacked():
    fig, ax = plt.subplots(figsize=(12, 6))

    videos = [s['video_name'] for s in valid_stats]
    behavior_keys = ['0', '1', '2', '3', '4']

    bottoms = np.zeros(len(videos))

    for bk in behavior_keys:
        vals = []
        for s in valid_stats:
            bpc = s.get('behavior_person_counts', {})
            vals.append(bpc.get(bk, 0))
        vals = np.array(vals)
        bars = ax.bar(videos, vals, bottom=bottoms,
                      color=BEHAVIOR_COLORS[bk],
                      label=BEHAVIOR_LABELS_SHORT[bk],
                      edgecolor='white', linewidth=0.5)
        # 在每段上标注数字（大于0时）
        for i, v in enumerate(vals):
            if v > 10:
                ax.text(i, bottoms[i] + v/2, str(v), ha='center', va='center',
                       fontsize=8, fontweight='bold', color='white')
        bottoms += vals

    # 顶部标注总人数
    for i, total in enumerate(bottoms):
        ax.text(i, total + 5, f'n={int(total)}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('视频场景')
    ax.set_ylabel('检测人数')
    ax.set_title('图1  各视频场景行为类别分布')
    ax.legend(loc='upper right', ncol=2)
    ax.set_xticklabels(videos, rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig1_behavior_distribution_stacked.png', bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/fig1_behavior_distribution_stacked.pdf', bbox_inches='tight')
    plt.close()
    print("  图1 完成: 各视频行为分布堆叠柱状图")

# ============ 图2: 总体行为分布饼图 ============
def fig2_overall_behavior_pie():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 汇总所有行为
    total_bpc = defaultdict(int)
    for s in valid_stats:
        for k, v in s.get('behavior_person_counts', {}).items():
            total_bpc[k] += v

    behavior_keys = sorted(total_bpc.keys())
    labels = [BEHAVIOR_LABELS_SHORT.get(k, f'类别{k}') for k in behavior_keys]
    values = [total_bpc[k] for k in behavior_keys]
    colors = [BEHAVIOR_COLORS.get(k, '#999999') for k in behavior_keys]

    # 饼图
    wedges, texts, autotexts = ax1.pie(values, labels=labels, colors=colors,
                                        autopct='%1.1f%%', startangle=90,
                                        pctdistance=0.8, textprops={'fontsize': 10})
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight('bold')
    ax1.set_title('(a) 总体行为类别占比')

    # 柱状图
    bars = ax2.bar(labels, values, color=colors, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_ylabel('检测人数')
    ax2.set_title('(b) 总体行为类别数量')
    ax2.set_xticklabels(labels, rotation=15, ha='right')

    fig.suptitle('图2  系统行为识别总体统计（共{}人次）'.format(sum(values)), fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig2_overall_behavior_pie.png', bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/fig2_overall_behavior_pie.pdf', bbox_inches='tight')
    plt.close()
    print("  图2 完成: 总体行为分布饼图+柱状图")

# ============ 图3: 人脸检测率对比 ============
def fig3_face_detection_rate():
    fig, ax = plt.subplots(figsize=(10, 5))

    videos = [s['video_name'] for s in valid_stats]
    rates = [s['face_detection_rate'] * 100 for s in valid_stats]

    colors_bar = ['#2196F3' if r > 70 else '#FF9800' if r > 50 else '#F44336' for r in rates]
    bars = ax.bar(videos, rates, color=colors_bar, edgecolor='white', linewidth=0.8)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{rate:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.axhline(y=np.mean(rates), color='red', linestyle='--', linewidth=1, alpha=0.7,
               label=f'均值: {np.mean(rates):.1f}%')
    ax.set_xlabel('视频场景')
    ax.set_ylabel('人脸检测率 (%)')
    ax.set_title('图3  各场景SSD人脸检测成功率')
    ax.set_ylim(0, 105)
    ax.legend()
    ax.set_xticklabels(videos, rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig3_face_detection_rate.png', bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/fig3_face_detection_rate.pdf', bbox_inches='tight')
    plt.close()
    print("  图3 完成: 人脸检测率对比")

# ============ 图4: 数据规模统计（帧数+人物轨迹数） ============
def fig4_data_scale():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    videos = [s['video_name'] for s in valid_stats]
    frames = [s['total_frames_processed'] for s in valid_stats]
    tracks = [len(s.get('track_behaviors', {})) for s in valid_stats]

    # 帧数
    bars1 = ax1.bar(videos, frames, color='#3F51B5', edgecolor='white')
    for bar, f in zip(bars1, frames):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{f:,}', ha='center', va='bottom', fontsize=8, rotation=45)
    ax1.set_ylabel('帧数')
    ax1.set_title('(a) 各视频处理帧数')
    ax1.set_xticklabels(videos, rotation=30, ha='right')

    # 轨迹数
    bars2 = ax2.bar(videos, tracks, color='#E91E63', edgecolor='white')
    for bar, t in zip(bars2, tracks):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(t), ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_ylabel('人物轨迹数')
    ax2.set_title('(b) 各视频检测人物轨迹数')
    ax2.set_xticklabels(videos, rotation=30, ha='right')

    total_frames = sum(frames)
    total_tracks = sum(tracks)
    fig.suptitle(f'图4  实验数据规模统计（总计 {total_frames:,} 帧，{total_tracks} 条轨迹）',
                fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig4_data_scale.png', bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/fig4_data_scale.pdf', bbox_inches='tight')
    plt.close()
    print("  图4 完成: 数据规模统计")

# ============ 图5: 可疑行为检出率热力图 ============
def fig5_suspicious_heatmap():
    fig, ax = plt.subplots(figsize=(10, 7))

    videos = [s['video_name'] for s in valid_stats]
    behavior_keys = ['0', '1', '2', '3', '4']
    behavior_labels = [BEHAVIOR_LABELS_SHORT[k] for k in behavior_keys]

    # 构建比例矩阵 (每个视频的行为比例)
    matrix = []
    for s in valid_stats:
        bpc = s.get('behavior_person_counts', {})
        total = sum(bpc.get(k, 0) for k in behavior_keys)
        if total == 0:
            total = 1
        row = [bpc.get(k, 0) / total * 100 for k in behavior_keys]
        matrix.append(row)

    matrix = np.array(matrix)

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(behavior_labels)))
    ax.set_xticklabels(behavior_labels, rotation=30, ha='right')
    ax.set_yticks(range(len(videos)))
    ax.set_yticklabels(videos)

    # 在每个格子里标数值
    for i in range(len(videos)):
        for j in range(len(behavior_keys)):
            val = matrix[i, j]
            color = 'white' if val > 40 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                   fontsize=9, color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('比例 (%)')
    ax.set_title('图5  各场景行为类别分布热力图')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig5_suspicious_heatmap.png', bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/fig5_suspicious_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print("  图5 完成: 行为分布热力图")

# ============ 图6: 系统流程图数据统计 ============
def fig6_pipeline_stats():
    fig, ax = plt.subplots(figsize=(12, 5))

    # 流水线各阶段统计
    total_frames = sum(s['total_frames_processed'] for s in valid_stats)
    total_face_det = sum(s['face_detection_count'] for s in valid_stats)
    total_fallback = sum(s['fallback_count'] for s in valid_stats)
    total_tracks = sum(len(s.get('track_behaviors', {})) for s in valid_stats)
    total_suspicious = sum(
        sum(v for k, v in s.get('behavior_person_counts', {}).items() if k != '0')
        for s in valid_stats
    )
    total_persons = sum(
        sum(v for k, v in s.get('behavior_person_counts', {}).items())
        for s in valid_stats
    )

    stages = ['输入帧数', 'SSD人脸检测', 'WHENet姿态估计\n(含fallback)',
              '人物轨迹', '行为识别人数', '可疑行为人数']
    values = [total_frames, total_face_det, total_face_det + total_fallback,
              total_tracks, total_persons, total_suspicious]

    colors = ['#90CAF9', '#64B5F6', '#42A5F5', '#2196F3', '#1E88E5', '#E53935']

    bars = ax.barh(stages, values, color=colors, edgecolor='white', height=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values)*0.01, bar.get_y() + bar.get_height()/2,
               f'{val:,}', ha='left', va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('数量')
    ax.set_title('图6  系统处理流水线各阶段数据量统计')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig6_pipeline_stats.png', bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/fig6_pipeline_stats.pdf', bbox_inches='tight')
    plt.close()
    print("  图6 完成: 流水线统计")

# ============ 图7: 正常 vs 可疑比例（突出创新价值） ============
def fig7_normal_vs_suspicious():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    videos = [s['video_name'] for s in valid_stats]

    normal_counts = []
    suspicious_counts = []
    for s in valid_stats:
        bpc = s.get('behavior_person_counts', {})
        normal = bpc.get('0', 0)
        suspicious = sum(v for k, v in bpc.items() if k != '0')
        normal_counts.append(normal)
        suspicious_counts.append(suspicious)

    # 分组柱状图
    x = np.arange(len(videos))
    width = 0.35
    bars1 = ax1.bar(x - width/2, normal_counts, width, label='正常行为', color='#4CAF50', edgecolor='white')
    bars2 = ax1.bar(x + width/2, suspicious_counts, width, label='可疑行为', color='#F44336', edgecolor='white')

    ax1.set_xlabel('视频场景')
    ax1.set_ylabel('检测人数')
    ax1.set_title('(a) 正常与可疑行为人数对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(videos, rotation=30, ha='right')
    ax1.legend()

    # 可疑率
    suspicious_rates = [s/(s+n)*100 if (s+n)>0 else 0 for n, s in zip(normal_counts, suspicious_counts)]
    bars3 = ax2.bar(videos, suspicious_rates, color=['#F44336' if r > 80 else '#FF9800' for r in suspicious_rates],
                    edgecolor='white')
    for bar, rate in zip(bars3, suspicious_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.axhline(y=np.mean(suspicious_rates), color='blue', linestyle='--', linewidth=1, alpha=0.7,
               label=f'均值: {np.mean(suspicious_rates):.1f}%')
    ax2.set_xlabel('视频场景')
    ax2.set_ylabel('可疑行为占比 (%)')
    ax2.set_title('(b) 各场景可疑行为检出率')
    ax2.set_ylim(0, 110)
    ax2.legend()
    ax2.set_xticklabels(videos, rotation=30, ha='right')

    fig.suptitle('图7  系统可疑行为识别能力分析', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig7_normal_vs_suspicious.png', bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/fig7_normal_vs_suspicious.pdf', bbox_inches='tight')
    plt.close()
    print("  图7 完成: 正常vs可疑行为分析")

# ============ 图8: 检测方式对比（SSD vs Fallback） ============
def fig8_detection_method():
    fig, ax = plt.subplots(figsize=(10, 5))

    videos = [s['video_name'] for s in valid_stats]
    face_counts = [s['face_detection_count'] for s in valid_stats]
    fallback_counts = [s['fallback_count'] for s in valid_stats]

    x = np.arange(len(videos))
    width = 0.35

    ax.bar(x - width/2, face_counts, width, label='SSD 人脸检测', color='#2196F3', edgecolor='white')
    ax.bar(x + width/2, fallback_counts, width, label='跟踪框 Fallback', color='#FF9800', edgecolor='white')

    ax.set_xlabel('视频场景')
    ax.set_ylabel('检测次数')
    ax.set_title('图8  SSD人脸检测与Fallback机制使用对比')
    ax.set_xticks(x)
    ax.set_xticklabels(videos, rotation=30, ha='right')
    ax.legend()
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig8_detection_method.png', bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/fig8_detection_method.pdf', bbox_inches='tight')
    plt.close()
    print("  图8 完成: 检测方式对比")

# ============ 图9: 综合性能雷达图 ============
def fig9_radar_chart():
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # 选几个有代表性的视频
    selected_names = ['MVI_4537', 'MVI_4538', 'MVI_4540', '1.14rg-1', '1.14zz-1']
    selected = [s for s in valid_stats if s['video_name'] in selected_names]

    # 维度
    categories = ['人脸检测率', '轨迹密度', '可疑行为占比', '行为多样性', '帧数规模']
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    radar_colors = ['#2196F3', '#4CAF50', '#F44336', '#9C27B0', '#FF9800']

    for idx, s in enumerate(selected):
        bpc = s.get('behavior_person_counts', {})
        total_persons = sum(bpc.values())
        suspicious = sum(v for k, v in bpc.items() if k != '0')
        n_categories = len([k for k, v in bpc.items() if v > 0])
        n_tracks = len(s.get('track_behaviors', {}))

        values = [
            s['face_detection_rate'] * 100,  # 人脸检测率
            min(n_tracks / 7, 100),  # 轨迹密度 (归一化)
            (suspicious / total_persons * 100) if total_persons > 0 else 0,  # 可疑比例
            n_categories / 5 * 100,  # 行为多样性
            min(s['total_frames_processed'] / 350, 100),  # 帧数规模 (归一化)
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=s['video_name'],
                color=radar_colors[idx % len(radar_colors)])
        ax.fill(angles, values, alpha=0.1, color=radar_colors[idx % len(radar_colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title('图9  多场景综合性能雷达图', y=1.08, fontsize=13)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig9_radar_chart.png', bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/fig9_radar_chart.pdf', bbox_inches='tight')
    plt.close()
    print("  图9 完成: 综合性能雷达图")

# ============ 表1: 实验结果汇总表 (LaTeX) ============
def table1_latex_summary():
    header = r"""
\begin{table}[htbp]
\centering
\caption{系统在多场景数据集上的实验结果}
\label{tab:experiment_results}
\begin{tabular}{lrrrrrrr}
\toprule
视频场景 & 帧数 & 轨迹数 & 检测率(\%) & 正常 & 可疑 & 可疑率(\%) \\
\midrule
"""
    rows = []
    total_frames = 0
    total_tracks = 0
    total_normal = 0
    total_suspicious = 0

    for s in valid_stats:
        bpc = s.get('behavior_person_counts', {})
        normal = bpc.get('0', 0)
        suspicious = sum(v for k, v in bpc.items() if k != '0')
        total = normal + suspicious
        rate = suspicious / total * 100 if total > 0 else 0
        n_tracks = len(s.get('track_behaviors', {}))

        total_frames += s['total_frames_processed']
        total_tracks += n_tracks
        total_normal += normal
        total_suspicious += suspicious

        rows.append(f"{s['video_name']} & {s['total_frames_processed']:,} & {n_tracks} & "
                    f"{s['face_detection_rate']*100:.1f} & {normal} & {suspicious} & {rate:.1f} \\\\")

    total_rate = total_suspicious / (total_normal + total_suspicious) * 100
    avg_det = np.mean([s['face_detection_rate']*100 for s in valid_stats])

    footer = f"""\\midrule
\\textbf{{合计}} & \\textbf{{{total_frames:,}}} & \\textbf{{{total_tracks}}} & \\textbf{{{avg_det:.1f}}} & \\textbf{{{total_normal}}} & \\textbf{{{total_suspicious}}} & \\textbf{{{total_rate:.1f}}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    content = header + '\n'.join(rows) + '\n' + footer
    with open(f'{OUT_DIR}/table1_experiment_results.tex', 'w') as f:
        f.write(content)
    print("  表1 完成: LaTeX 实验结果汇总表")

# ============ 表2: 行为类别详细统计表 ============
def table2_behavior_detail():
    header = r"""
\begin{table}[htbp]
\centering
\caption{各行为类别检测详细统计}
\label{tab:behavior_detail}
\begin{tabular}{lrrrrrr}
\toprule
视频场景 & 正常 & 频繁张望 & 快速回头 & 长时间观察 & 持续低头 & 合计 \\
\midrule
"""
    rows = []
    totals = defaultdict(int)

    for s in valid_stats:
        bpc = s.get('behavior_person_counts', {})
        row_vals = [bpc.get(str(i), 0) for i in range(5)]
        total = sum(row_vals)
        for i, v in enumerate(row_vals):
            totals[str(i)] += v
        rows.append(f"{s['video_name']} & {' & '.join(map(str, row_vals))} & {total} \\\\")

    grand_total = sum(totals.values())
    total_row = ' & '.join([f"\\textbf{{{totals[str(i)]}}}" for i in range(5)])

    footer = f"""\\midrule
\\textbf{{合计}} & {total_row} & \\textbf{{{grand_total}}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    content = header + '\n'.join(rows) + '\n' + footer
    with open(f'{OUT_DIR}/table2_behavior_detail.tex', 'w') as f:
        f.write(content)
    print("  表2 完成: LaTeX 行为类别详细统计表")

# ============ 主函数 ============
if __name__ == '__main__':
    print("=" * 60)
    print("  生成硕士毕业论文可视化图表")
    print("=" * 60)

    fig1_behavior_distribution_stacked()
    fig2_overall_behavior_pie()
    fig3_face_detection_rate()
    fig4_data_scale()
    fig5_suspicious_heatmap()
    fig6_pipeline_stats()
    fig7_normal_vs_suspicious()
    fig8_detection_method()
    fig9_radar_chart()
    table1_latex_summary()
    table2_behavior_detail()

    print()
    print("=" * 60)
    print(f"  全部完成！图表保存在: {OUT_DIR}/")
    print("  PNG 格式用于预览，PDF 格式用于论文插入")
    print("=" * 60)

    # 汇总信息
    total_frames = sum(s['total_frames_processed'] for s in valid_stats)
    total_tracks = sum(len(s.get('track_behaviors', {})) for s in valid_stats)
    total_persons = sum(sum(s.get('behavior_person_counts', {}).values()) for s in valid_stats)

    print(f"\n论文可引用数据:")
    print(f"  - 实验视频数: {len(valid_stats)} 个场景")
    print(f"  - 总处理帧数: {total_frames:,}")
    print(f"  - 总人物轨迹: {total_tracks}")
    print(f"  - 总识别人次: {total_persons}")
    print(f"  - 平均人脸检测率: {np.mean([s['face_detection_rate'] for s in valid_stats])*100:.1f}%")
