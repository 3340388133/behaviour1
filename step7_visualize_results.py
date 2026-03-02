#!/usr/bin/env python3
"""
Step 7 可视化: 生成论文级别的推理结果图表
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

# 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path('data/inference_output/paper_figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Normal', 'Glancing', 'QuickTurn', 'Prolonged', 'LookDown', 'LookUp']
CLASS_NAMES_CN = ['正常行为', '频繁张望', '快速回头', '长时间观察', '持续低头', '持续抬头']
CLASS_COLORS = ['#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#2196F3', '#FFEB3B']


def load_stats():
    with open('data/inference_output/MVI_4537_inference_stats.json') as f:
        return json.load(f)


def fig1_behavior_distribution(stats):
    """图1: 行为类别分布饼图+条形图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    counts = stats['behavior_person_counts']
    values = [counts.get(str(i), 0) for i in range(6)]
    total = sum(values)

    # 饼图
    nonzero_idx = [i for i, v in enumerate(values) if v > 0]
    pie_vals = [values[i] for i in nonzero_idx]
    pie_labels = [f"{CLASS_NAMES[i]}\n({values[i]})" for i in nonzero_idx]
    pie_colors = [CLASS_COLORS[i] for i in nonzero_idx]

    wedges, texts, autotexts = ax1.pie(
        pie_vals, labels=pie_labels, colors=pie_colors,
        autopct='%1.1f%%', startangle=90, pctdistance=0.75,
        textprops={'fontsize': 10}
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight('bold')
    ax1.set_title('Behavior Distribution (by person)', fontsize=13, fontweight='bold', pad=15)

    # 条形图
    bars = ax2.bar(range(6), values, color=CLASS_COLORS, edgecolor='black', linewidth=0.5)
    ax2.set_xticks(range(6))
    ax2.set_xticklabels(CLASS_NAMES, rotation=25, ha='right', fontsize=10)
    ax2.set_ylabel('Number of Persons', fontsize=11)
    ax2.set_title('Behavior Category Counts', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, values):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig1_behavior_distribution.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'fig1_behavior_distribution.pdf', bbox_inches='tight')
    plt.close()
    print(f"  [OK] fig1_behavior_distribution")


def fig2_detection_pipeline(stats):
    """图2: 检测管线性能指标"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # 人脸检测率
    face_rate = stats['face_detection_rate']
    fallback_rate = stats['fallback_count'] / (stats['face_detection_count'] + stats['fallback_count'])
    ax = axes[0]
    bars = ax.bar(['Face\nDetected', 'Fallback\n(Estimated)'],
                  [face_rate * 100, fallback_rate * 100],
                  color=['#4CAF50', '#FF9800'], edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title('Head Detection Method', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                f'{h:.1f}%', ha='center', fontsize=11, fontweight='bold')

    # 正常vs可疑
    counts = stats['behavior_person_counts']
    normal = counts.get('0', 0)
    suspicious = sum(v for k, v in counts.items() if k != '0')
    ax = axes[1]
    wedges, texts, autotexts = ax.pie(
        [normal, suspicious],
        labels=['Normal', 'Suspicious'],
        colors=['#4CAF50', '#F44336'],
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 11}
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight('bold')
    ax.set_title(f'Normal vs Suspicious\n(N={normal+suspicious})', fontsize=12, fontweight='bold')

    # 可疑行为细分
    ax = axes[2]
    susp_names = CLASS_NAMES[1:]
    susp_counts = [counts.get(str(i), 0) for i in range(1, 6)]
    susp_colors = CLASS_COLORS[1:]
    nonzero = [(n, c, col) for n, c, col in zip(susp_names, susp_counts, susp_colors) if c > 0]
    if nonzero:
        names, cnts, cols = zip(*nonzero)
        bars = ax.barh(range(len(names)), cnts, color=cols, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel('Number of Persons', fontsize=11)
        ax.set_title('Suspicious Behavior Breakdown', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        for bar, cnt in zip(bars, cnts):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    str(cnt), va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig2_detection_pipeline.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'fig2_detection_pipeline.pdf', bbox_inches='tight')
    plt.close()
    print(f"  [OK] fig2_detection_pipeline")


def fig3_temporal_behavior_timeline(stats):
    """图3: 可疑人员时间线"""
    track_behaviors = stats['track_behaviors']

    # 只展示可疑行为的 track
    suspicious_tracks = {k: v for k, v in track_behaviors.items() if v > 0}
    if not suspicious_tracks:
        print("  [SKIP] fig3: no suspicious tracks")
        return

    # 加载跟踪数据获取时间范围
    with open('data/tracked_output/MVI_4537/tracking_result.json') as f:
        tracking = json.load(f)

    track_info = {}
    for t in tracking['tracks']:
        tid = f"track_{t['track_id']:04d}"
        track_info[tid] = {
            'first': t['first_frame'],
            'last': t['last_frame'],
            'duration': t['duration_sec']
        }

    fig, ax = plt.subplots(figsize=(14, 6))

    y_pos = 0
    y_labels = []
    start_frame = 500

    for tid, beh in sorted(suspicious_tracks.items(), key=lambda x: x[1]):
        if tid not in track_info:
            continue
        info = track_info[tid]
        if info['first'] < start_frame - 50 or info['last'] < start_frame:
            continue

        color = CLASS_COLORS[beh]
        x_start = max(info['first'] - start_frame, 0) / 30.0  # convert to seconds
        x_end = (info['last'] - start_frame) / 30.0
        if x_end < 0:
            continue

        ax.barh(y_pos, x_end - x_start, left=x_start, height=0.7,
                color=color, edgecolor='black', linewidth=0.3, alpha=0.85)
        label = f"{tid.replace('track_', '#')}"
        y_labels.append(label)
        y_pos += 1

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_title('Suspicious Person Timeline (Head-Pose Based Detection)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Legend
    handles = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
               for i in range(1, 6)]
    ax.legend(handles=handles, loc='upper right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig3_temporal_timeline.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'fig3_temporal_timeline.pdf', bbox_inches='tight')
    plt.close()
    print(f"  [OK] fig3_temporal_timeline ({y_pos} tracks)")


def fig4_system_architecture():
    """图4: 系统架构流程图"""
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis('off')

    boxes = [
        (0.5, 2, 2.2, 1.6, 'Video Frame\n+ Tracking', '#E3F2FD'),
        (3.2, 2, 2.2, 1.6, 'Person ROI\nExtraction', '#E8F5E9'),
        (5.9, 2, 2.4, 1.6, 'SSD Face\nDetection', '#FFF3E0'),
        (8.8, 2, 2.4, 1.6, 'Face→Head\nBBox Expand', '#F3E5F5'),
        (11.7, 2, 2.4, 1.6, 'WHENet\nPose Estimation', '#E0F7FA'),
        (11.7, 0, 2.4, 1.4, 'SBRN Model\n+ Rule Engine', '#FFEBEE'),
    ]

    for x, y, w, h, text, color in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='#333', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=10, fontweight='bold')

    # Arrows between stages
    arrow_props = dict(arrowstyle='->', color='#555', lw=2)
    for i in range(4):
        x1 = boxes[i][0] + boxes[i][2]
        x2 = boxes[i+1][0]
        y_mid = boxes[i][1] + boxes[i][3]/2
        ax.annotate('', xy=(x2, y_mid), xytext=(x1, y_mid), arrowprops=arrow_props)

    # Arrow from WHENet down to SBRN
    ax.annotate('', xy=(12.9, 1.4), xytext=(12.9, 2.0), arrowprops=arrow_props)

    # Output label
    ax.text(12.9, 4.2, 'Head-Pose Based Behavior Recognition Pipeline',
            ha='center', va='center', fontsize=14, fontweight='bold',
            style='italic')

    fig.savefig(OUTPUT_DIR / 'fig4_system_architecture.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'fig4_system_architecture.pdf', bbox_inches='tight')
    plt.close()
    print(f"  [OK] fig4_system_architecture")


def fig5_comparison_before_after():
    """图5: 改进前后对比（使用关键帧拼接）"""
    import cv2

    keyframe_dir = Path('data/inference_output/MVI_4537_keyframes')

    # 选取代表性帧
    frames_to_show = ['frame_000800.jpg', 'frame_001550.jpg', 'frame_002000.jpg', 'frame_003050.jpg']
    available = [f for f in frames_to_show if (keyframe_dir / f).exists()]

    if len(available) < 2:
        print("  [SKIP] fig5: not enough keyframes")
        return

    images = []
    for fname in available[:4]:
        img = cv2.imread(str(keyframe_dir / fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize to consistent height
        h, w = img.shape[:2]
        new_h = 400
        new_w = int(w * new_h / h)
        img = cv2.resize(img, (new_w, new_h))
        images.append((fname, img))

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (fname, img) in zip(axes, images):
        ax.imshow(img)
        frame_num = fname.replace('frame_', '').replace('.jpg', '')
        ax.set_title(f'Frame {frame_num}', fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle('Head-Pose Based Behavior Recognition: Sample Results',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig5_sample_results.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'fig5_sample_results.pdf', bbox_inches='tight')
    plt.close()
    print(f"  [OK] fig5_sample_results ({n} frames)")


def fig6_head_expansion_illustration():
    """图6: 人脸框→头部框扩展示意图"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # 模拟人脸框和头部框
    face_x, face_y = 3, 2.5
    face_w, face_h = 2, 2.5

    # 人脸框 (蓝色)
    face_rect = mpatches.Rectangle((face_x, face_y), face_w, face_h,
                                    linewidth=2, edgecolor='#2196F3',
                                    facecolor='#2196F3', alpha=0.15)
    ax.add_patch(face_rect)
    ax.plot([face_x, face_x + face_w, face_x + face_w, face_x, face_x],
            [face_y, face_y, face_y + face_h, face_y + face_h, face_y],
            'b-', linewidth=2, label='Face BBox')

    # 头部框 (红色) - expand_top=0.6, expand_bottom=0.15, expand_side=0.35
    expand_top = face_h * 0.6
    expand_bottom = face_h * 0.15
    expand_side = face_w * 0.35
    head_x = face_x - expand_side
    head_y = face_y - expand_bottom
    head_w = face_w + 2 * expand_side
    head_h = face_h + expand_top + expand_bottom

    head_rect = mpatches.Rectangle((head_x, head_y), head_w, head_h,
                                    linewidth=2.5, edgecolor='#F44336',
                                    facecolor='#F44336', alpha=0.08)
    ax.add_patch(head_rect)
    ax.plot([head_x, head_x + head_w, head_x + head_w, head_x, head_x],
            [head_y, head_y, head_y + head_h, head_y + head_h, head_y],
            'r-', linewidth=2.5, label='Head BBox (expanded)')

    # 标注扩展量
    ax.annotate('', xy=(face_x + face_w/2, face_y + face_h + expand_top),
                xytext=(face_x + face_w/2, face_y + face_h),
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
    ax.text(face_x + face_w/2 + 0.1, face_y + face_h + expand_top/2,
            f'top: {0.6:.0%}', fontsize=10, color='red')

    ax.annotate('', xy=(face_x + face_w + expand_side, face_y + face_h/2),
                xytext=(face_x + face_w, face_y + face_h/2),
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
    ax.text(face_x + face_w + 0.1, face_y + face_h/2 + 0.2,
            f'side: {0.35:.0%}', fontsize=10, color='red')

    ax.set_xlim(1, 7)
    ax.set_ylim(1, 7)
    ax.set_aspect('equal')
    ax.legend(fontsize=11, loc='upper left')
    ax.set_title('Face BBox → Head BBox Expansion Strategy', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.2)
    ax.set_xlabel('x (pixels, normalized)', fontsize=11)
    ax.set_ylabel('y (pixels, normalized)', fontsize=11)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig6_head_expansion.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'fig6_head_expansion.pdf', bbox_inches='tight')
    plt.close()
    print(f"  [OK] fig6_head_expansion")


if __name__ == '__main__':
    print("=" * 60)
    print("  生成论文可视化图表")
    print("=" * 60)

    stats = load_stats()
    print(f"\n数据: {stats['video_name']}, {stats['total_frames_processed']} frames")
    print(f"检测率: {stats['face_detection_rate']:.1%}")
    print(f"总人数: {sum(stats['behavior_person_counts'].values())}")
    print()

    fig1_behavior_distribution(stats)
    fig2_detection_pipeline(stats)
    fig3_temporal_behavior_timeline(stats)
    fig4_system_architecture()
    fig5_comparison_before_after()
    fig6_head_expansion_illustration()

    print(f"\n所有图表已保存到: {OUTPUT_DIR}/")
    print("生成文件:")
    for f in sorted(OUTPUT_DIR.glob('*')):
        print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")
