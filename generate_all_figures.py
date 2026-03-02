#!/usr/bin/env python3
"""
批量生成论文级可视化图表
从已有实验数据中提取并绘制所有关键分析图
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['savefig.bbox'] = 'tight'

OUTPUT_DIR = Path('data/paper_figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEHAVIOR_NAMES = {0: 'Normal', 1: 'Glancing', 2: 'QuickTurn', 3: 'Prolonged', 4: 'LookDown', 5: 'LookUp'}
BEHAVIOR_COLORS = {0: '#2ecc71', 1: '#e74c3c', 2: '#f39c12', 3: '#9b59b6', 4: '#3498db', 5: '#1abc9c'}
BEHAVIOR_CN = {0: '正常', 1: '频繁张望', 2: '快速回头', 3: '持续侧视', 4: '低头', 5: '抬头'}


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


# ============================================================
# Fig 1: 模型性能对比柱状图 (4种方法)
# ============================================================
def fig1_model_comparison():
    data = load_json('checkpoints/ablation_results.json')
    methods = ['Rule\nBaseline', 'LSTM', 'Transformer', 'Transformer\n+UW (Ours)']
    keys = ['rule', 'lstm', 'transformer', 'transformer_uw']

    metrics = {
        'Accuracy': [data[k]['test_accuracy'] for k in keys],
        'Precision': [data[k]['test_precision'] for k in keys],
        'Recall': [data[k]['test_recall'] for k in keys],
        'F1 Score': [data[k]['test_f1'] for k in keys],
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(methods))
    width = 0.18
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    for i, (metric, values) in enumerate(metrics.items()):
        bars = ax.bar(x + i * width, values, width, label=metric, color=colors[i], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison on Binary Classification', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylim(0.5, 1.05)
    ax.legend(loc='lower right', fontsize=9, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(OUTPUT_DIR / 'fig01_model_comparison.png')
    plt.savefig(OUTPUT_DIR / 'fig01_model_comparison.pdf')
    plt.close()
    print('  [1/15] fig01_model_comparison')


# ============================================================
# Fig 2: 训练曲线 (Loss + Accuracy + F1)
# ============================================================
def fig2_training_curves():
    data = load_json('checkpoints/ablation_sbrn.json')
    # 取前两个模型的 history
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    model_names = []
    model_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    for idx, model in enumerate(data[:5]):
        name = model['name'].replace('A0: ', '').replace('A1: ', '').replace('A2: ', '').replace('A3: ', '').replace('A4: ', '')
        model_names.append(name)
        history = model.get('history', [])
        if not history:
            continue
        epochs = [h['epoch'] for h in history]
        train_loss = [h['train_loss'] for h in history]
        val_loss = [h['val_loss'] for h in history]
        train_acc = [h['train_acc'] for h in history]
        val_acc = [h['val_acc'] for h in history]
        val_f1 = [h['val_f1_macro'] for h in history]

        c = model_colors[idx % len(model_colors)]
        axes[0].plot(epochs, train_loss, '-', color=c, alpha=0.4, linewidth=1)
        axes[0].plot(epochs, val_loss, '-', color=c, linewidth=2, label=name)
        axes[1].plot(epochs, train_acc, '-', color=c, alpha=0.4, linewidth=1)
        axes[1].plot(epochs, val_acc, '-', color=c, linewidth=2, label=name)
        axes[2].plot(epochs, val_f1, '-', color=c, linewidth=2, label=name, marker='o', markersize=2)

    axes[0].set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=7, loc='upper right')
    axes[0].grid(alpha=0.3)

    axes[1].set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend(fontsize=7, loc='lower right')
    axes[1].grid(alpha=0.3)

    axes[2].set_title('Validation Macro F1 Score', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 (Macro)')
    axes[2].legend(fontsize=7, loc='lower right')
    axes[2].grid(alpha=0.3)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('6-Class Behavior Recognition Training Dynamics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig02_training_curves.png')
    plt.savefig(OUTPUT_DIR / 'fig02_training_curves.pdf')
    plt.close()
    print('  [2/15] fig02_training_curves')


# ============================================================
# Fig 3: Per-class F1 雷达图
# ============================================================
def fig3_perclass_radar():
    data = load_json('checkpoints/ablation_sbrn.json')

    classes = list(BEHAVIOR_NAMES.values())
    N = len(classes)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    for idx, model in enumerate(data[:5]):
        name = model['name'].split(': ')[1] if ': ' in model['name'] else model['name']
        f1s = model['test_metrics']['per_class_f1']
        while len(f1s) < N:
            f1s.append(0)
        values = f1s[:N] + f1s[:1]
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[idx], label=name, markersize=4)
        ax.fill(angles, values, alpha=0.08, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(classes, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_title('Per-Class F1 Score Comparison\n(6-Class Behavior Recognition)',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='lower right', bbox_to_anchor=(1.3, 0), fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.savefig(OUTPUT_DIR / 'fig03_perclass_radar.png')
    plt.savefig(OUTPUT_DIR / 'fig03_perclass_radar.pdf')
    plt.close()
    print('  [3/15] fig03_perclass_radar')


# ============================================================
# Fig 4: 融合策略对比
# ============================================================
def fig4_fusion_comparison():
    data = load_json('experiments/comparison_results.json')
    results = data['results']

    methods = [r['Method'] for r in results]
    metrics_list = ['Precision', 'Recall', 'F1', 'AUC', 'Accuracy']

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(methods))
    width = 0.15
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

    for i, metric in enumerate(metrics_list):
        values = [float(r[metric]) for r in results]
        bars = ax.bar(x + i * width, values, width, label=metric, color=colors[i], edgecolor='white')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=6, rotation=45)

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Rule-Model Fusion Strategy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylim(0.8, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(OUTPUT_DIR / 'fig04_fusion_comparison.png')
    plt.savefig(OUTPUT_DIR / 'fig04_fusion_comparison.pdf')
    plt.close()
    print('  [4/15] fig04_fusion_comparison')


# ============================================================
# Fig 5: 混淆矩阵网格
# ============================================================
def fig5_confusion_matrices():
    data = load_json('experiments/comparison_results.json')
    cms = data['confusion_matrices']

    fig, axes = plt.subplots(1, len(cms), figsize=(4 * len(cms), 4))
    labels = ['Normal', 'Suspicious']

    for idx, (method, cm) in enumerate(cms.items()):
        ax = axes[idx] if len(cms) > 1 else axes
        cm_arr = np.array(cm)
        im = ax.imshow(cm_arr, cmap='Blues', aspect='auto')

        for i in range(2):
            for j in range(2):
                color = 'white' if cm_arr[i, j] > cm_arr.max() * 0.5 else 'black'
                ax.text(j, i, str(cm_arr[i, j]), ha='center', va='center',
                       fontsize=14, fontweight='bold', color=color)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Predicted', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Actual', fontsize=10)
        short_name = method.replace('Fusion ', 'F')
        ax.set_title(short_name, fontsize=11, fontweight='bold')

    plt.suptitle('Confusion Matrices for Different Fusion Strategies', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig05_confusion_matrices.png')
    plt.savefig(OUTPUT_DIR / 'fig05_confusion_matrices.pdf')
    plt.close()
    print('  [5/15] fig05_confusion_matrices')


# ============================================================
# Fig 6: 跨视频行为分布 (堆叠柱状图)
# ============================================================
def fig6_cross_video_distribution():
    stats_dir = Path('data/batch_inference_output')
    stats_files = sorted(stats_dir.glob('*_inference_stats.json'))

    video_names = []
    distributions = []

    for sf in stats_files:
        s = load_json(sf)
        vname = s['video_name']
        video_names.append(vname)
        counts = s['behavior_person_counts']
        total = sum(int(v) for v in counts.values())
        dist = {}
        for k, v in counts.items():
            dist[int(k)] = int(v) / total if total > 0 else 0
        distributions.append(dist)

    if not video_names:
        print('  [6/15] SKIP: no stats files')
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # 上: 绝对数量堆叠
    x = np.arange(len(video_names))
    bottom = np.zeros(len(video_names))

    for cls_id in range(6):
        values = []
        for sf in stats_files:
            s = load_json(sf)
            counts = s['behavior_person_counts']
            values.append(int(counts.get(str(cls_id), 0)))
        values = np.array(values)
        ax1.bar(x, values, bottom=bottom, label=f'{BEHAVIOR_NAMES.get(cls_id, f"Class{cls_id}")}',
                color=BEHAVIOR_COLORS.get(cls_id, '#999'), edgecolor='white', linewidth=0.5)
        bottom += values

    ax1.set_xticks(x)
    ax1.set_xticklabels(video_names, rotation=30, ha='right', fontsize=9)
    ax1.set_ylabel('Person Count', fontsize=11, fontweight='bold')
    ax1.set_title('Behavior Distribution Across Videos (Absolute Count)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8, ncol=3)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 下: 百分比堆叠
    bottom_pct = np.zeros(len(video_names))
    for cls_id in range(6):
        values_pct = []
        for dist in distributions:
            values_pct.append(dist.get(cls_id, 0) * 100)
        values_pct = np.array(values_pct)
        ax2.bar(x, values_pct, bottom=bottom_pct, label=f'{BEHAVIOR_NAMES.get(cls_id, f"Class{cls_id}")}',
                color=BEHAVIOR_COLORS.get(cls_id, '#999'), edgecolor='white', linewidth=0.5)
        # 标注百分比
        for i, v in enumerate(values_pct):
            if v > 5:
                ax2.text(i, bottom_pct[i] + v/2, f'{v:.0f}%', ha='center', va='center', fontsize=7, fontweight='bold')
        bottom_pct += values_pct

    ax2.set_xticks(x)
    ax2.set_xticklabels(video_names, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Behavior Distribution Across Videos (Percentage)', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.legend(loc='upper right', fontsize=8, ncol=3)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig06_cross_video_distribution.png')
    plt.savefig(OUTPUT_DIR / 'fig06_cross_video_distribution.pdf')
    plt.close()
    print('  [6/15] fig06_cross_video_distribution')


# ============================================================
# Fig 7: 人脸检测率 & 回退率对比
# ============================================================
def fig7_detection_rates():
    stats_dir = Path('data/batch_inference_output')
    stats_files = sorted(stats_dir.glob('*_inference_stats.json'))

    video_names = []
    det_rates = []
    fallback_rates = []
    total_persons = []
    total_frames = []

    for sf in stats_files:
        s = load_json(sf)
        video_names.append(s['video_name'])
        det_rates.append(s['face_detection_rate'] * 100)
        fb_rate = s['fallback_count'] / s['face_detection_count'] * 100 if s['face_detection_count'] > 0 else 0
        fallback_rates.append(fb_rate)
        total_persons.append(len(s['track_behaviors']))
        total_frames.append(s['total_frames_processed'])

    if not video_names:
        print('  [7/15] SKIP: no stats')
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(len(video_names))

    # 左上: 人脸检测率
    bars = axes[0, 0].bar(x, det_rates, color='#3498db', edgecolor='white')
    for bar, val in zip(bars, det_rates):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', fontsize=8, fontweight='bold')
    axes[0, 0].set_title('Face Detection Rate per Video', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Detection Rate (%)')
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].axhline(y=np.mean(det_rates), color='red', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(det_rates):.1f}%')
    axes[0, 0].legend()

    # 右上: Fallback 率
    bars = axes[0, 1].bar(x, fallback_rates, color='#e74c3c', edgecolor='white')
    for bar, val in zip(bars, fallback_rates):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                        f'{val:.1f}%', ha='center', fontsize=8, fontweight='bold')
    axes[0, 1].set_title('Pose Estimation Fallback Rate', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Fallback Rate (%)')

    # 左下: 跟踪人数
    bars = axes[1, 0].bar(x, total_persons, color='#2ecc71', edgecolor='white')
    for bar, val in zip(bars, total_persons):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        str(val), ha='center', fontsize=8, fontweight='bold')
    axes[1, 0].set_title('Total Tracked Persons per Video', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Person Count')

    # 右下: 总帧数
    bars = axes[1, 1].bar(x, [f/1000 for f in total_frames], color='#f39c12', edgecolor='white')
    for bar, val in zip(bars, total_frames):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height()/1000 + 0.3,
                        f'{val/1000:.1f}k', ha='center', fontsize=8, fontweight='bold')
    axes[1, 1].set_title('Total Frames Processed per Video', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Frames (×1000)')

    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(video_names, rotation=30, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Video Processing Statistics Overview', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig07_detection_rates.png')
    plt.savefig(OUTPUT_DIR / 'fig07_detection_rates.pdf')
    plt.close()
    print('  [7/15] fig07_detection_rates')


# ============================================================
# Fig 8: 可疑人员占比饼图矩阵
# ============================================================
def fig8_suspicious_ratio():
    stats_dir = Path('data/batch_inference_output')
    stats_files = sorted(stats_dir.glob('*_inference_stats.json'))

    if not list(stats_files):
        print('  [8/15] SKIP')
        return
    stats_files = sorted(Path('data/batch_inference_output').glob('*_inference_stats.json'))

    n = len(stats_files)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, sf in enumerate(stats_files):
        s = load_json(sf)
        counts = s['behavior_person_counts']
        normal = int(counts.get('0', 0))
        suspicious = sum(int(v) for k, v in counts.items() if k != '0')
        total = normal + suspicious

        sizes = [normal, suspicious]
        labels = [f'Normal\n{normal}', f'Suspicious\n{suspicious}']
        colors_pie = ['#2ecc71', '#e74c3c']
        explode = (0, 0.05)

        axes[idx].pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                      autopct='%1.1f%%', shadow=False, startangle=90, textprops={'fontsize': 9})
        axes[idx].set_title(f'{s["video_name"]}\n(n={total})', fontsize=10, fontweight='bold')

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Normal vs. Suspicious Person Ratio per Video', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig08_suspicious_ratio.png')
    plt.savefig(OUTPUT_DIR / 'fig08_suspicious_ratio.pdf')
    plt.close()
    print('  [8/15] fig08_suspicious_ratio')


# ============================================================
# Fig 9: Per-class F1 热力图 (方法 × 类别)
# ============================================================
def fig9_f1_heatmap():
    data = load_json('checkpoints/ablation_sbrn.json')

    model_names = []
    f1_matrix = []
    for model in data[:5]:
        name = model['name'].split(': ')[1] if ': ' in model['name'] else model['name']
        model_names.append(name)
        f1s = model['test_metrics']['per_class_f1']
        while len(f1s) < 6:
            f1s.append(0)
        f1_matrix.append(f1s[:6])

    f1_arr = np.array(f1_matrix)
    class_names = [BEHAVIOR_NAMES[i] for i in range(6)]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(f1_arr, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    for i in range(len(model_names)):
        for j in range(6):
            val = f1_arr[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=10,
                   fontweight='bold', color=color)

    ax.set_xticks(range(6))
    ax.set_xticklabels(class_names, fontsize=10, fontweight='bold')
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=9)
    ax.set_title('Per-Class F1 Score Heatmap Across Methods', fontsize=13, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('F1 Score', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig09_f1_heatmap.png')
    plt.savefig(OUTPUT_DIR / 'fig09_f1_heatmap.pdf')
    plt.close()
    print('  [9/15] fig09_f1_heatmap')


# ============================================================
# Fig 10: 消融实验汇总表 (参数量 vs F1)
# ============================================================
def fig10_ablation_scatter():
    data = load_json('checkpoints/ablation_sbrn.json')

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    for idx, model in enumerate(data[:5]):
        name = model['name'].split(': ')[1] if ': ' in model['name'] else model['name']
        n_params = model['n_params']
        f1 = model['test_metrics']['f1_macro']
        acc = model['test_metrics']['accuracy']

        ax.scatter(n_params / 1000, f1, s=acc * 300, color=colors[idx], alpha=0.8,
                  edgecolors='black', linewidth=1, zorder=5)
        ax.annotate(name, (n_params / 1000, f1), textcoords="offset points",
                   xytext=(10, 10), fontsize=8, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    ax.set_xlabel('Parameters (×1000)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test F1 (Macro)', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study: Model Complexity vs. Performance\n(bubble size ∝ accuracy)',
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(OUTPUT_DIR / 'fig10_ablation_scatter.png')
    plt.savefig(OUTPUT_DIR / 'fig10_ablation_scatter.pdf')
    plt.close()
    print('  [10/15] fig10_ablation_scatter')


# ============================================================
# Fig 11: Val F1 收敛过程对比
# ============================================================
def fig11_convergence():
    data = load_json('checkpoints/ablation_sbrn.json')

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    for idx, model in enumerate(data[:5]):
        name = model['name'].split(': ')[1] if ': ' in model['name'] else model['name']
        history = model.get('history', [])
        if not history:
            continue
        epochs = [h['epoch'] for h in history]
        val_f1 = [h['val_f1_macro'] for h in history]

        ax.plot(epochs, val_f1, '-o', color=colors[idx], linewidth=2, markersize=3, label=name)

        # 标记最佳点
        best_idx = np.argmax(val_f1)
        ax.scatter(epochs[best_idx], val_f1[best_idx], s=100, color=colors[idx],
                  edgecolors='black', linewidth=2, zorder=10, marker='*')
        ax.annotate(f'{val_f1[best_idx]:.3f}', (epochs[best_idx], val_f1[best_idx]),
                   textcoords="offset points", xytext=(5, 8), fontsize=7, fontweight='bold')

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation F1 (Macro)', fontsize=12, fontweight='bold')
    ax.set_title('Convergence Analysis: Validation F1 Over Training Epochs', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(OUTPUT_DIR / 'fig11_convergence.png')
    plt.savefig(OUTPUT_DIR / 'fig11_convergence.pdf')
    plt.close()
    print('  [11/15] fig11_convergence')


# ============================================================
# Fig 12: 各视频可疑行为类型分布 (分组柱状图)
# ============================================================
def fig12_behavior_breakdown():
    stats_dir = Path('data/batch_inference_output')
    stats_files = sorted(stats_dir.glob('*_inference_stats.json'))

    if not list(stats_files):
        print('  [12/15] SKIP')
        return
    stats_files = sorted(Path('data/batch_inference_output').glob('*_inference_stats.json'))

    video_names = []
    behavior_data = {i: [] for i in range(6)}

    for sf in stats_files:
        s = load_json(sf)
        video_names.append(s['video_name'])
        counts = s['behavior_person_counts']
        for i in range(6):
            behavior_data[i].append(int(counts.get(str(i), 0)))

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(video_names))
    width = 0.13

    for cls_id in range(6):
        offset = (cls_id - 2.5) * width
        bars = ax.bar(x + offset, behavior_data[cls_id], width,
                      label=BEHAVIOR_NAMES[cls_id], color=BEHAVIOR_COLORS[cls_id],
                      edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(video_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Person Count', fontsize=11, fontweight='bold')
    ax.set_title('Detailed Behavior Type Breakdown per Video', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, ncol=3, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig12_behavior_breakdown.png')
    plt.savefig(OUTPUT_DIR / 'fig12_behavior_breakdown.pdf')
    plt.close()
    print('  [12/15] fig12_behavior_breakdown')


# ============================================================
# Fig 13: 系统整体性能总结表格图
# ============================================================
def fig13_summary_table():
    ablation = load_json('checkpoints/ablation_results.json')
    sbrn = load_json('checkpoints/ablation_sbrn.json')

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # 二分类结果
    rows1 = []
    for key, label in [('rule', 'Rule Baseline'), ('lstm', 'LSTM'),
                        ('transformer', 'Transformer'), ('transformer_uw', 'Transformer+UW')]:
        d = ablation[key]
        rows1.append([label, f'{d["test_accuracy"]:.4f}', f'{d["test_precision"]:.4f}',
                      f'{d["test_recall"]:.4f}', f'{d["test_f1"]:.4f}'])

    table1 = ax.table(cellText=rows1,
                       colLabels=['Method', 'Accuracy', 'Precision', 'Recall', 'F1'],
                       loc='upper center', cellLoc='center',
                       bbox=[0.05, 0.55, 0.9, 0.35])
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    for key, cell in table1.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor('#3498db')
            cell.set_text_props(color='white', fontweight='bold')
        elif key[0] == len(rows1):
            cell.set_facecolor('#e8f8f5')

    # 6分类结果
    rows2 = []
    for model in sbrn[:5]:
        name = model['name'].split(': ')[1] if ': ' in model['name'] else model['name']
        m = model['test_metrics']
        rows2.append([name, f'{m["accuracy"]:.4f}', f'{m["f1_macro"]:.4f}',
                      f'{m["f1_weighted"]:.4f}', f'{m["precision"]:.4f}', f'{m["recall"]:.4f}'])

    table2 = ax.table(cellText=rows2,
                       colLabels=['Method', 'Accuracy', 'F1(Macro)', 'F1(Weighted)', 'Precision', 'Recall'],
                       loc='lower center', cellLoc='center',
                       bbox=[0.02, 0.0, 0.96, 0.42])
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    for key, cell in table2.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor('#e74c3c')
            cell.set_text_props(color='white', fontweight='bold')

    ax.text(0.5, 0.95, 'Binary Classification Results', transform=ax.transAxes,
            fontsize=13, fontweight='bold', ha='center')
    ax.text(0.5, 0.47, '6-Class Behavior Recognition Results', transform=ax.transAxes,
            fontsize=13, fontweight='bold', ha='center')

    plt.suptitle('Comprehensive Experimental Results Summary', fontsize=15, fontweight='bold', y=0.98)
    plt.savefig(OUTPUT_DIR / 'fig13_summary_table.png')
    plt.savefig(OUTPUT_DIR / 'fig13_summary_table.pdf')
    plt.close()
    print('  [13/15] fig13_summary_table')


# ============================================================
# Fig 14: 侧机位 vs 正机位对比分析
# ============================================================
def fig14_camera_angle_comparison():
    stats_dir = Path('data/batch_inference_output')
    stats_files = sorted(stats_dir.glob('*_inference_stats.json'))

    side_videos = []   # 侧机位: MVI_*
    front_videos = []  # 正机位: 1.14*

    for sf in stats_files:
        s = load_json(sf)
        name = s['video_name']
        if name.startswith('MVI'):
            side_videos.append(s)
        elif name.startswith('1.14'):
            front_videos.append(s)

    if not side_videos and not front_videos:
        print('  [14/15] SKIP')
        return

    def aggregate(videos):
        total_persons = 0
        behavior_counts = Counter()
        det_rates = []
        for v in videos:
            total_persons += len(v['track_behaviors'])
            det_rates.append(v['face_detection_rate'])
            for k, val in v['behavior_person_counts'].items():
                behavior_counts[int(k)] += int(val)
        return total_persons, behavior_counts, np.mean(det_rates) if det_rates else 0

    side_total, side_beh, side_det = aggregate(side_videos)
    front_total, front_beh, front_det = aggregate(front_videos)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 左: 行为分布对比
    classes = list(range(6))
    side_vals = [side_beh.get(c, 0) for c in classes]
    front_vals = [front_beh.get(c, 0) for c in classes]

    x = np.arange(6)
    width = 0.35
    axes[0].bar(x - width/2, side_vals, width, label=f'Side View (n={side_total})', color='#3498db')
    axes[0].bar(x + width/2, front_vals, width, label=f'Front View (n={front_total})', color='#e74c3c')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([BEHAVIOR_NAMES[i] for i in classes], rotation=30, ha='right', fontsize=8)
    axes[0].set_ylabel('Person Count')
    axes[0].set_title('Behavior Distribution\nby Camera Angle', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].grid(axis='y', alpha=0.3)

    # 中: 占比对比
    side_pct = [v/side_total*100 if side_total > 0 else 0 for v in side_vals]
    front_pct = [v/front_total*100 if front_total > 0 else 0 for v in front_vals]
    axes[1].bar(x - width/2, side_pct, width, label='Side View', color='#3498db')
    axes[1].bar(x + width/2, front_pct, width, label='Front View', color='#e74c3c')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([BEHAVIOR_NAMES[i] for i in classes], rotation=30, ha='right', fontsize=8)
    axes[1].set_ylabel('Percentage (%)')
    axes[1].set_title('Behavior Percentage\nby Camera Angle', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=8)
    axes[1].grid(axis='y', alpha=0.3)

    # 右: 检测率对比
    categories = ['Face Det. Rate', 'Suspicious Rate', 'Avg Persons/Video']
    side_susp = (side_total - side_beh.get(0, 0)) / side_total * 100 if side_total > 0 else 0
    front_susp = (front_total - front_beh.get(0, 0)) / front_total * 100 if front_total > 0 else 0
    side_avg = side_total / len(side_videos) if side_videos else 0
    front_avg = front_total / len(front_videos) if front_videos else 0

    side_metrics = [side_det * 100, side_susp, side_avg / 10]
    front_metrics = [front_det * 100, front_susp, front_avg / 10]

    x2 = np.arange(3)
    axes[2].bar(x2 - width/2, side_metrics, width, label='Side View', color='#3498db')
    axes[2].bar(x2 + width/2, front_metrics, width, label='Front View', color='#e74c3c')
    axes[2].set_xticks(x2)
    axes[2].set_xticklabels(categories, fontsize=9)
    axes[2].set_title('System Metrics\nby Camera Angle', fontsize=11, fontweight='bold')
    axes[2].legend(fontsize=8)
    axes[2].grid(axis='y', alpha=0.3)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Side View vs. Front View Camera Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig14_camera_comparison.png')
    plt.savefig(OUTPUT_DIR / 'fig14_camera_comparison.pdf')
    plt.close()
    print('  [14/15] fig14_camera_comparison')


# ============================================================
# Fig 15: 大型综合 Dashboard
# ============================================================
def fig15_dashboard():
    ablation = load_json('checkpoints/ablation_results.json')
    stats_dir = Path('data/batch_inference_output')
    stats_files = sorted(stats_dir.glob('*_inference_stats.json'))

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)

    # (0,0-1): 模型对比条形图
    ax1 = fig.add_subplot(gs[0, 0:2])
    methods = ['Rule', 'LSTM', 'Transformer', 'Trans+UW']
    keys = ['rule', 'lstm', 'transformer', 'transformer_uw']
    f1s = [ablation[k]['test_f1'] for k in keys]
    colors_bar = ['#95a5a6', '#3498db', '#f39c12', '#e74c3c']
    bars = ax1.barh(methods, f1s, color=colors_bar, edgecolor='white', height=0.6)
    for bar, val in zip(bars, f1s):
        ax1.text(val + 0.005, bar.get_y() + bar.get_height()/2., f'{val:.3f}',
                va='center', fontsize=9, fontweight='bold')
    ax1.set_xlim(0.6, 1.0)
    ax1.set_title('Binary F1 Comparison', fontsize=11, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # (0,2-3): 总计统计
    ax2 = fig.add_subplot(gs[0, 2:4])
    total_frames = sum(load_json(sf)['total_frames_processed'] for sf in stats_files)
    total_persons = sum(len(load_json(sf)['track_behaviors']) for sf in stats_files)
    total_detections = sum(load_json(sf)['face_detection_count'] for sf in stats_files)
    n_videos = len(list(stats_files))

    kpis = [('Videos', n_videos), ('Frames', f'{total_frames/1000:.0f}k'),
            ('Persons', total_persons), ('Detections', f'{total_detections/1000:.0f}k')]
    for i, (label, val) in enumerate(kpis):
        ax2.text(0.15 + i*0.22, 0.7, str(val), fontsize=22, fontweight='bold', ha='center',
                color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'][i], transform=ax2.transAxes)
        ax2.text(0.15 + i*0.22, 0.35, label, fontsize=11, ha='center', transform=ax2.transAxes)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('System-Wide Statistics', fontsize=11, fontweight='bold')

    # (1,0-1): 行为分布饼图 (全局)
    ax3 = fig.add_subplot(gs[1, 0:2])
    global_counts = Counter()
    for sf in stats_files:
        s = load_json(sf)
        for k, v in s['behavior_person_counts'].items():
            global_counts[int(k)] += int(v)

    sizes = [global_counts.get(i, 0) for i in range(6)]
    labels_pie = [f'{BEHAVIOR_NAMES[i]}\n({sizes[i]})' for i in range(6)]
    colors_pie = [BEHAVIOR_COLORS[i] for i in range(6)]
    non_zero = [(s, l, c) for s, l, c in zip(sizes, labels_pie, colors_pie) if s > 0]
    if non_zero:
        sizes_nz, labels_nz, colors_nz = zip(*non_zero)
        ax3.pie(sizes_nz, labels=labels_nz, colors=colors_nz, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 8})
    ax3.set_title('Global Behavior Distribution', fontsize=11, fontweight='bold')

    # (1,2-3): 跨视频可疑率
    ax4 = fig.add_subplot(gs[1, 2:4])
    video_names = []
    susp_rates = []
    for sf in sorted(stats_files):
        s = load_json(sf)
        video_names.append(s['video_name'])
        total = sum(int(v) for v in s['behavior_person_counts'].values())
        normal = int(s['behavior_person_counts'].get('0', 0))
        susp_rates.append((total - normal) / total * 100 if total > 0 else 0)

    colors_susp = ['#e74c3c' if r > 70 else '#f39c12' if r > 50 else '#2ecc71' for r in susp_rates]
    bars = ax4.bar(range(len(video_names)), susp_rates, color=colors_susp, edgecolor='white')
    ax4.set_xticks(range(len(video_names)))
    ax4.set_xticklabels(video_names, rotation=45, ha='right', fontsize=7)
    ax4.set_ylabel('Suspicious Rate (%)')
    ax4.axhline(y=np.mean(susp_rates), color='red', linestyle='--', alpha=0.5)
    ax4.set_title('Suspicious Rate per Video', fontsize=11, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    # (2,0-3): 时间轴 - 行为事件密度
    ax5 = fig.add_subplot(gs[2, :])
    video_labels = []
    event_counts = []
    person_counts = []
    for sf in sorted(stats_files):
        s = load_json(sf)
        video_labels.append(s['video_name'])
        event_counts.append(s['behavior_events_count'])
        person_counts.append(len(s['track_behaviors']))

    x_pos = np.arange(len(video_labels))
    ax5_twin = ax5.twinx()
    bars1 = ax5.bar(x_pos - 0.2, [e/1000 for e in event_counts], 0.4,
                     color='#3498db', alpha=0.7, label='Events (×1000)')
    bars2 = ax5_twin.bar(x_pos + 0.2, person_counts, 0.4,
                          color='#e74c3c', alpha=0.7, label='Persons')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(video_labels, rotation=30, ha='right', fontsize=8)
    ax5.set_ylabel('Behavior Events (×1000)', color='#3498db', fontsize=10)
    ax5_twin.set_ylabel('Tracked Persons', color='#e74c3c', fontsize=10)
    ax5.set_title('Behavior Events & Tracked Persons per Video', fontsize=11, fontweight='bold')

    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    ax5.grid(axis='y', alpha=0.3)

    plt.suptitle('Suspicious Gaze Detection System — Comprehensive Dashboard',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(OUTPUT_DIR / 'fig15_dashboard.png')
    plt.savefig(OUTPUT_DIR / 'fig15_dashboard.pdf')
    plt.close()
    print('  [15/15] fig15_dashboard')


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print('=' * 60)
    print('  Generating Paper Figures')
    print('=' * 60)

    fig1_model_comparison()
    fig2_training_curves()
    fig3_perclass_radar()
    fig4_fusion_comparison()
    fig5_confusion_matrices()
    fig6_cross_video_distribution()
    fig7_detection_rates()
    fig8_suspicious_ratio()
    fig9_f1_heatmap()
    fig10_ablation_scatter()
    fig11_convergence()
    fig12_behavior_breakdown()
    fig13_summary_table()
    fig14_camera_angle_comparison()
    fig15_dashboard()

    print('\n' + '=' * 60)
    total = len(list(OUTPUT_DIR.glob('*.png')))
    print(f'  Done! {total} PNG + PDF figures saved to {OUTPUT_DIR}')
    print('=' * 60)
