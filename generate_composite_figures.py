#!/usr/bin/env python3
"""生成论文用合并子图"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import numpy as np
import json, glob
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

OUT = 'thesis_figures'
STATS_DIR = 'data/batch_inference_output'

BEHAVIOR_LABELS_SHORT = {
    '0': '正常行为', '1': '频繁张望', '2': '快速回头',
    '3': '长时间观察', '4': '持续低头', '5': '持续抬头',
}
BEHAVIOR_COLORS = {
    '0': '#4CAF50', '1': '#FF9800', '2': '#F44336',
    '3': '#9C27B0', '4': '#2196F3', '5': '#00BCD4',
}

def load_stats():
    stats = []
    for f in sorted(glob.glob(f'{STATS_DIR}/*_inference_stats.json')):
        with open(f) as fp:
            stats.append(json.load(fp))
    return [s for s in stats if s['total_frames_processed'] > 100]

valid_stats = load_stats()

# ========== 合并图1: 系统检测效果展示 (2x2 关键帧) ==========
def composite_keyframes():
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    keyframes = [
        ('kf_MVI4538.jpg', '(a) 正面场景多人检测 (MVI_4538)'),
        ('kf_MVI4539.jpg', '(b) 多类别行为识别 (MVI_4539)'),
        ('kf_rg1.jpg', '(c) 侧面场景检测 (1.14rg-1)'),
        ('kf_zz4.jpg', '(d) 密集人流场景 (1.14zz-4)'),
    ]

    for ax, (fname, title) in zip(axes.flat, keyframes):
        img = mpimg.imread(f'{OUT}/{fname}')
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
        ax.axis('off')

    fig.suptitle('图1  系统可疑行为识别效果展示', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT}/composite_fig1_keyframes.png', bbox_inches='tight')
    plt.savefig(f'{OUT}/composite_fig1_keyframes.pdf', bbox_inches='tight')
    plt.close()
    print("  合并图1: 系统检测效果展示 (2x2)")

# ========== 合并图2: 总体行为统计 (饼图+柱状图+检出率) ==========
def composite_behavior_overview():
    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1])

    total_bpc = defaultdict(int)
    for s in valid_stats:
        for k, v in s.get('behavior_person_counts', {}).items():
            total_bpc[k] += v

    bkeys = sorted(total_bpc.keys())
    labels = [BEHAVIOR_LABELS_SHORT.get(k, k) for k in bkeys]
    values = [total_bpc[k] for k in bkeys]
    colors = [BEHAVIOR_COLORS.get(k, '#999') for k in bkeys]

    # (a) 饼图
    ax1 = fig.add_subplot(gs[0])
    wedges, texts, autotexts = ax1.pie(values, labels=labels, colors=colors,
                                        autopct='%1.1f%%', startangle=90,
                                        pctdistance=0.78, textprops={'fontsize': 9})
    for t in autotexts:
        t.set_fontsize(8)
        t.set_fontweight('bold')
    ax1.set_title('(a) 行为类别占比', fontsize=11, fontweight='bold')

    # (b) 柱状图
    ax2 = fig.add_subplot(gs[1])
    bars = ax2.bar(labels, values, color=colors, edgecolor='white')
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_ylabel('检测人数')
    ax2.set_title('(b) 行为类别数量', fontsize=11, fontweight='bold')
    ax2.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)

    # (c) 各场景可疑率
    ax3 = fig.add_subplot(gs[2])
    videos = [s['video_name'] for s in valid_stats]
    suspicious_rates = []
    for s in valid_stats:
        bpc = s.get('behavior_person_counts', {})
        total = sum(bpc.values())
        susp = sum(v for k, v in bpc.items() if k != '0')
        suspicious_rates.append(susp / total * 100 if total > 0 else 0)

    bar_colors = ['#F44336' if r > 90 else '#FF9800' if r > 80 else '#FFC107' for r in suspicious_rates]
    bars3 = ax3.bar(videos, suspicious_rates, color=bar_colors, edgecolor='white')
    for bar, rate in zip(bars3, suspicious_rates):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax3.axhline(y=np.mean(suspicious_rates), color='blue', linestyle='--', linewidth=1,
               label=f'均值: {np.mean(suspicious_rates):.1f}%', alpha=0.7)
    ax3.set_ylabel('可疑行为占比 (%)')
    ax3.set_title('(c) 各场景可疑率', fontsize=11, fontweight='bold')
    ax3.set_ylim(0, 110)
    ax3.legend(fontsize=9)
    ax3.set_xticklabels(videos, rotation=30, ha='right', fontsize=8)

    fig.suptitle('图2  系统行为识别总体统计（共{}人次）'.format(sum(values)),
                fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(f'{OUT}/composite_fig2_overview.png', bbox_inches='tight')
    plt.savefig(f'{OUT}/composite_fig2_overview.pdf', bbox_inches='tight')
    plt.close()
    print("  合并图2: 总体行为统计 (饼+柱+检出率)")

# ========== 合并图3: 各场景行为分布 (堆叠柱状图+热力图) ==========
def composite_distribution():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                    gridspec_kw={'width_ratios': [1, 1.1]})

    videos = [s['video_name'] for s in valid_stats]
    bkeys = ['0', '1', '2', '3', '4']

    # (a) 堆叠柱状图
    bottoms = np.zeros(len(videos))
    for bk in bkeys:
        vals = np.array([s.get('behavior_person_counts', {}).get(bk, 0) for s in valid_stats])
        ax1.bar(videos, vals, bottom=bottoms, color=BEHAVIOR_COLORS[bk],
                label=BEHAVIOR_LABELS_SHORT[bk], edgecolor='white', linewidth=0.5)
        for i, v in enumerate(vals):
            if v > 15:
                ax1.text(i, bottoms[i] + v/2, str(v), ha='center', va='center',
                        fontsize=7, fontweight='bold', color='white')
        bottoms += vals

    for i, total in enumerate(bottoms):
        ax1.text(i, total + 5, f'n={int(total)}', ha='center', va='bottom', fontsize=8)

    ax1.set_xlabel('视频场景')
    ax1.set_ylabel('检测人数')
    ax1.set_title('(a) 各场景行为类别分布', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.set_xticklabels(videos, rotation=30, ha='right', fontsize=9)

    # (b) 热力图
    matrix = []
    for s in valid_stats:
        bpc = s.get('behavior_person_counts', {})
        total = sum(bpc.get(k, 0) for k in bkeys)
        if total == 0: total = 1
        row = [bpc.get(k, 0) / total * 100 for k in bkeys]
        matrix.append(row)
    matrix = np.array(matrix)

    im = ax2.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(range(len(bkeys)))
    ax2.set_xticklabels([BEHAVIOR_LABELS_SHORT[k] for k in bkeys], rotation=30, ha='right', fontsize=9)
    ax2.set_yticks(range(len(videos)))
    ax2.set_yticklabels(videos, fontsize=9)

    for i in range(len(videos)):
        for j in range(len(bkeys)):
            val = matrix[i, j]
            color = 'white' if val > 40 else 'black'
            ax2.text(j, i, f'{val:.1f}%', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('比例 (%)')
    ax2.set_title('(b) 行为类别分布热力图', fontsize=11, fontweight='bold')

    fig.suptitle('图3  各场景行为识别详细分析', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT}/composite_fig3_distribution.png', bbox_inches='tight')
    plt.savefig(f'{OUT}/composite_fig3_distribution.pdf', bbox_inches='tight')
    plt.close()
    print("  合并图3: 行为分布 (堆叠+热力图)")

# ========== 合并图4: 检测性能分析 (检测率+Fallback+正常vs可疑) ==========
def composite_detection():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax1, ax2, ax3 = axes

    videos = [s['video_name'] for s in valid_stats]

    # (a) 人脸检测率
    rates = [s['face_detection_rate'] * 100 for s in valid_stats]
    bar_colors = ['#2196F3' if r > 70 else '#FF9800' if r > 50 else '#F44336' for r in rates]
    bars1 = ax1.bar(videos, rates, color=bar_colors, edgecolor='white')
    for bar, rate in zip(bars1, rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax1.axhline(y=np.mean(rates), color='red', linestyle='--', linewidth=1, alpha=0.7,
               label=f'均值: {np.mean(rates):.1f}%')
    ax1.set_ylabel('人脸检测率 (%)')
    ax1.set_title('(a) SSD人脸检测成功率', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=8)
    ax1.set_xticklabels(videos, rotation=35, ha='right', fontsize=8)

    # (b) SSD vs Fallback
    face_counts = [s['face_detection_count'] for s in valid_stats]
    fallback_counts = [s['fallback_count'] for s in valid_stats]
    x = np.arange(len(videos))
    width = 0.35
    ax2.bar(x - width/2, face_counts, width, label='SSD人脸检测', color='#2196F3', edgecolor='white')
    ax2.bar(x + width/2, fallback_counts, width, label='Fallback', color='#FF9800', edgecolor='white')
    ax2.set_ylabel('检测次数')
    ax2.set_title('(b) 检测方式对比', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(videos, rotation=35, ha='right', fontsize=8)
    ax2.legend(fontsize=8)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # (c) 正常 vs 可疑
    normal_counts = []
    suspicious_counts = []
    for s in valid_stats:
        bpc = s.get('behavior_person_counts', {})
        normal_counts.append(bpc.get('0', 0))
        suspicious_counts.append(sum(v for k, v in bpc.items() if k != '0'))

    ax3.bar(x - width/2, normal_counts, width, label='正常行为', color='#4CAF50', edgecolor='white')
    ax3.bar(x + width/2, suspicious_counts, width, label='可疑行为', color='#F44336', edgecolor='white')
    ax3.set_ylabel('检测人数')
    ax3.set_title('(c) 正常与可疑行为人数', fontsize=11, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(videos, rotation=35, ha='right', fontsize=8)
    ax3.legend(fontsize=8)

    fig.suptitle('图4  系统检测性能分析', fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(f'{OUT}/composite_fig4_detection.png', bbox_inches='tight')
    plt.savefig(f'{OUT}/composite_fig4_detection.pdf', bbox_inches='tight')
    plt.close()
    print("  合并图4: 检测性能 (检测率+Fallback+正常vs可疑)")

# ========== 合并图5: 数据规模与流水线 (帧数+轨迹+流水线) ==========
def composite_scale():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax1, ax2, ax3 = axes

    videos = [s['video_name'] for s in valid_stats]
    frames = [s['total_frames_processed'] for s in valid_stats]
    tracks = [len(s.get('track_behaviors', {})) for s in valid_stats]

    # (a) 帧数
    bars1 = ax1.bar(videos, frames, color='#3F51B5', edgecolor='white')
    for bar, f in zip(bars1, frames):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                f'{f:,}', ha='center', va='bottom', fontsize=7, rotation=45)
    ax1.set_ylabel('帧数')
    ax1.set_title('(a) 各视频处理帧数', fontsize=11, fontweight='bold')
    ax1.set_xticklabels(videos, rotation=35, ha='right', fontsize=8)

    # (b) 轨迹数
    bars2 = ax2.bar(videos, tracks, color='#E91E63', edgecolor='white')
    for bar, t in zip(bars2, tracks):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(t), ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax2.set_ylabel('人物轨迹数')
    ax2.set_title('(b) 各视频检测轨迹数', fontsize=11, fontweight='bold')
    ax2.set_xticklabels(videos, rotation=35, ha='right', fontsize=8)

    # (c) 流水线统计
    total_frames = sum(frames)
    total_face = sum(s['face_detection_count'] for s in valid_stats)
    total_fallback = sum(s['fallback_count'] for s in valid_stats)
    total_tracks = sum(tracks)
    total_persons = sum(sum(s.get('behavior_person_counts', {}).values()) for s in valid_stats)
    total_suspicious = sum(
        sum(v for k, v in s.get('behavior_person_counts', {}).items() if k != '0')
        for s in valid_stats
    )

    stages = ['输入帧', 'SSD检测', 'WHENet估计', '人物轨迹', '行为识别', '可疑人数']
    vals = [total_frames, total_face, total_face + total_fallback,
            total_tracks, total_persons, total_suspicious]
    colors_h = ['#90CAF9', '#64B5F6', '#42A5F5', '#2196F3', '#1E88E5', '#E53935']

    bars3 = ax3.barh(stages, vals, color=colors_h, edgecolor='white', height=0.6)
    for bar, val in zip(bars3, vals):
        ax3.text(bar.get_width() + max(vals)*0.02, bar.get_y() + bar.get_height()/2,
                f'{val:,}', ha='left', va='center', fontsize=9, fontweight='bold')
    ax3.set_xlabel('数量')
    ax3.set_title('(c) 处理流水线数据量', fontsize=11, fontweight='bold')
    ax3.invert_yaxis()

    fig.suptitle('图5  实验数据规模统计（总计{:,}帧，{:,}条轨迹）'.format(
        total_frames, total_tracks), fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(f'{OUT}/composite_fig5_scale.png', bbox_inches='tight')
    plt.savefig(f'{OUT}/composite_fig5_scale.pdf', bbox_inches='tight')
    plt.close()
    print("  合并图5: 数据规模 (帧数+轨迹+流水线)")

# ========== 合并图6: 综合性能 (雷达图 + 实验结果总结表) ==========
def composite_summary():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                    subplot_kw={'polar': False})
    # 左侧用极坐标替换
    fig.delaxes(ax1)
    ax1 = fig.add_subplot(121, polar=True)

    selected_names = ['MVI_4537', 'MVI_4538', 'MVI_4540', '1.14rg-1', '1.14zz-1']
    selected = [s for s in valid_stats if s['video_name'] in selected_names]

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
            s['face_detection_rate'] * 100,
            min(n_tracks / 7, 100),
            (suspicious / total_persons * 100) if total_persons > 0 else 0,
            n_categories / 5 * 100,
            min(s['total_frames_processed'] / 350, 100),
        ]
        values += values[:1]
        ax1.plot(angles, values, 'o-', linewidth=2, label=s['video_name'],
                color=radar_colors[idx % len(radar_colors)], markersize=4)
        ax1.fill(angles, values, alpha=0.08, color=radar_colors[idx % len(radar_colors)])

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, fontsize=9)
    ax1.set_ylim(0, 100)
    ax1.set_title('(a) 多场景综合性能雷达图', fontsize=11, fontweight='bold', y=1.08)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)

    # 右侧：结果总结表格
    ax2.axis('off')
    table_data = []
    for s in valid_stats:
        bpc = s.get('behavior_person_counts', {})
        normal = bpc.get('0', 0)
        susp = sum(v for k, v in bpc.items() if k != '0')
        total = normal + susp
        rate = susp / total * 100 if total > 0 else 0
        table_data.append([
            s['video_name'],
            f"{s['total_frames_processed']:,}",
            str(len(s.get('track_behaviors', {}))),
            f"{s['face_detection_rate']*100:.1f}%",
            f"{rate:.1f}%"
        ])

    # 合计行
    total_frames = sum(s['total_frames_processed'] for s in valid_stats)
    total_tracks = sum(len(s.get('track_behaviors', {})) for s in valid_stats)
    avg_det = np.mean([s['face_detection_rate']*100 for s in valid_stats])
    total_bpc_all = defaultdict(int)
    for s in valid_stats:
        for k, v in s.get('behavior_person_counts', {}).items():
            total_bpc_all[k] += v
    total_susp = sum(v for k, v in total_bpc_all.items() if k != '0')
    total_all = sum(total_bpc_all.values())
    table_data.append(['合计', f'{total_frames:,}', str(total_tracks),
                       f'{avg_det:.1f}%', f'{total_susp/total_all*100:.1f}%'])

    col_labels = ['场景', '帧数', '轨迹', '检测率', '可疑率']
    table = ax2.table(cellText=table_data, colLabels=col_labels,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # 表头颜色
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#3F51B5')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # 合计行颜色
    last_row = len(table_data)
    for j in range(len(col_labels)):
        table[last_row, j].set_facecolor('#E8EAF6')
        table[last_row, j].set_text_props(fontweight='bold')

    ax2.set_title('(b) 实验结果汇总表', fontsize=11, fontweight='bold', pad=20)

    fig.suptitle('图6  系统综合性能评估', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT}/composite_fig6_summary.png', bbox_inches='tight')
    plt.savefig(f'{OUT}/composite_fig6_summary.pdf', bbox_inches='tight')
    plt.close()
    print("  合并图6: 综合性能 (雷达图+结果表)")


if __name__ == '__main__':
    print("=" * 50)
    print("  生成论文合并子图")
    print("=" * 50)
    composite_keyframes()
    composite_behavior_overview()
    composite_distribution()
    composite_detection()
    composite_scale()
    composite_summary()
    print("=" * 50)
    print("  完成！共6张合并图")
    print("=" * 50)
