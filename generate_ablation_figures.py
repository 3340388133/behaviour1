#!/usr/bin/env python3
"""
生成消融实验与基线对比的论文级可视化图表

依赖 step8_ablation_baseline.py 的输出:
  data/ablation_output/ablation_results.json

生成图表:
  Fig1: 消融实验对比 (一致率/可疑率 + 行为分布 + 模块贡献度)
  Fig2: 基线方法对比 (行为分布 + 逐视频可疑率 + 测试集指标)
  Fig3: 逐场景分析 (正面 vs 侧面 heatmap)
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['savefig.bbox'] = 'tight'

OUTPUT_DIR = Path('thesis_figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEHAVIOR_NAMES = {
    0: 'Normal', 1: 'Glancing', 2: 'QuickTurn',
    3: 'Prolonged', 4: 'LookDown', 5: 'LookUp'
}
BEHAVIOR_COLORS = {
    0: '#2ecc71', 1: '#e74c3c', 2: '#f39c12',
    3: '#9b59b6', 4: '#3498db', 5: '#1abc9c'
}
BEHAVIOR_CN = {
    0: '正常', 1: '频繁张望', 2: '快速回头',
    3: '持续侧视', 4: '低头', 5: '抬头'
}

ABLATION_NAMES = {
    'Full': '完整系统',
    'A1_no_gate': '去掉门控',
    'A2_no_transformer': '去掉Transformer',
    'A3_no_rules': '去掉规则',
    'A4_no_smooth': '去掉平滑',
}

BASELINE_NAMES = {
    'B1_threshold': '纯阈值法',
    'B2_rules_only': '纯规则法',
    'B3_lstm': 'LSTM替代',
}


def load_ablation_data():
    """加载消融实验结果"""
    path = Path('data/ablation_output/ablation_results.json')
    if not path.exists():
        print(f"[ERROR] 未找到消融结果: {path}")
        print("       请先运行: python step8_ablation_baseline.py")
        return None
    with open(path, 'r') as f:
        return json.load(f)


def load_test_metrics():
    """加载测试集指标"""
    path = Path('checkpoints/ablation_results.json')
    if not path.exists():
        return {}
    with open(path, 'r') as f:
        return json.load(f)


# ============================================================
# Fig 1: 消融实验对比大图
# ============================================================

def fig1_ablation_comparison(data):
    """
    (a) 一致率 & 可疑率 grouped bar chart
    (b) 行为分布 stacked bar chart
    (c) 模块贡献度 (一致率下降量)
    """
    configs = data['configs']
    ablation_keys = ['Full', 'A1_no_gate', 'A2_no_transformer',
                     'A3_no_rules', 'A4_no_smooth']

    # 只取存在的配置
    ablation_keys = [k for k in ablation_keys if k in configs]
    if not ablation_keys:
        print("[SKIP] fig1: 没有消融实验数据")
        return

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3,
                           height_ratios=[1, 1])

    # ---- (a) 一致率 & 可疑率 ----
    ax_a = fig.add_subplot(gs[0, 0])
    labels = [ABLATION_NAMES.get(k, k) for k in ablation_keys]
    agreements = [configs[k]['metrics'].get('agreement', 0) * 100
                  for k in ablation_keys]
    suspicious = [configs[k]['metrics'].get('suspicious_rate', 0) * 100
                  for k in ablation_keys]

    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax_a.bar(x - width/2, agreements, width, label='一致率 (%)',
                     color='#3498db', edgecolor='white', linewidth=0.5)
    bars2 = ax_a.bar(x + width/2, suspicious, width, label='可疑率 (%)',
                     color='#e74c3c', edgecolor='white', linewidth=0.5)

    for bar in bars1:
        ax_a.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                  f'{bar.get_height():.1f}', ha='center', va='bottom',
                  fontsize=8, fontweight='bold')
    for bar in bars2:
        ax_a.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                  f'{bar.get_height():.1f}', ha='center', va='bottom',
                  fontsize=8, fontweight='bold')

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, fontsize=9)
    ax_a.set_ylabel('百分比 (%)', fontsize=10)
    ax_a.set_title('(a) 消融实验：一致率与可疑率', fontsize=12, fontweight='bold')
    ax_a.legend(fontsize=9)
    ax_a.set_ylim(0, 105)
    ax_a.grid(axis='y', alpha=0.3)

    # ---- (b) 行为分布 stacked bar ----
    ax_b = fig.add_subplot(gs[0, 1])
    bottom = np.zeros(len(ablation_keys))

    for cls_id in range(6):
        values = []
        for k in ablation_keys:
            dist = configs[k]['metrics'].get('behavior_distribution', {})
            values.append(dist.get(cls_id, dist.get(str(cls_id), 0)))
        values = np.array(values, dtype=float)
        ax_b.bar(x, values, 0.6, bottom=bottom,
                 label=f'{BEHAVIOR_NAMES[cls_id]} ({BEHAVIOR_CN[cls_id]})',
                 color=BEHAVIOR_COLORS[cls_id], edgecolor='white',
                 linewidth=0.5)
        bottom += values

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, fontsize=9)
    ax_b.set_ylabel('轨迹数', fontsize=10)
    ax_b.set_title('(b) 消融实验：行为分布', fontsize=12, fontweight='bold')
    ax_b.legend(fontsize=7, loc='upper right', ncol=2)
    ax_b.grid(axis='y', alpha=0.3)

    # ---- (c) 模块贡献度 ----
    ax_c = fig.add_subplot(gs[1, 0])
    full_agr = configs.get('Full', {}).get('metrics', {}).get('agreement', 1.0)
    module_keys = ['A1_no_gate', 'A2_no_transformer', 'A3_no_rules', 'A4_no_smooth']
    module_labels = ['姿态门控\n(Pose Gate)', 'Transformer\n模型',
                     '规则检测\n(Rules)', '时序平滑\n(Smoothing)']

    drops = []
    module_labels_valid = []
    for mk, ml in zip(module_keys, module_labels):
        if mk in configs:
            ablation_agr = configs[mk]['metrics'].get('agreement', 0)
            drops.append((full_agr - ablation_agr) * 100)
            module_labels_valid.append(ml)

    if drops:
        colors = ['#e74c3c' if d > 0 else '#2ecc71' for d in drops]
        bars = ax_c.barh(range(len(drops)), drops, color=colors,
                         edgecolor='white', linewidth=0.5, height=0.6)
        ax_c.set_yticks(range(len(drops)))
        ax_c.set_yticklabels(module_labels_valid, fontsize=9)
        ax_c.set_xlabel('一致率下降 (百分点)', fontsize=10)
        ax_c.axvline(x=0, color='black', linewidth=0.5)
        for i, (bar, drop) in enumerate(zip(bars, drops)):
            ax_c.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2.,
                      f'{drop:+.1f}%', ha='left', va='center',
                      fontsize=9, fontweight='bold')
    ax_c.set_title('(c) 各模块贡献度（去掉后一致率下降量）',
                   fontsize=12, fontweight='bold')
    ax_c.grid(axis='x', alpha=0.3)

    # ---- (d) Shannon 熵对比 ----
    ax_d = fig.add_subplot(gs[1, 1])
    entropies = [configs[k]['metrics'].get('shannon_entropy', 0)
                 for k in ablation_keys]
    full_entropy = configs.get('Full', {}).get('metrics', {}).get('full_shannon_entropy', 0)

    bars_d = ax_d.bar(x, entropies, 0.6, color='#8e44ad',
                      edgecolor='white', linewidth=0.5, alpha=0.8)
    ax_d.axhline(y=full_entropy, color='#e74c3c', linestyle='--',
                 linewidth=1.5, label=f'全系统参照 ({full_entropy:.3f})')

    for bar, ent in zip(bars_d, entropies):
        ax_d.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                  f'{ent:.3f}', ha='center', va='bottom',
                  fontsize=8, fontweight='bold')

    ax_d.set_xticks(x)
    ax_d.set_xticklabels(labels, fontsize=9)
    ax_d.set_ylabel('Shannon 熵', fontsize=10)
    ax_d.set_title('(d) 行为多样性 (Shannon Entropy)', fontsize=12, fontweight='bold')
    ax_d.legend(fontsize=9)
    ax_d.grid(axis='y', alpha=0.3)

    fig.suptitle('图1: 消融实验对比分析', fontsize=14, fontweight='bold', y=1.01)

    for fmt in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f'ablation_fig1_comparison.{fmt}',
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] Fig1: 消融实验对比 -> {OUTPUT_DIR}/ablation_fig1_comparison.png")


# ============================================================
# Fig 2: 基线方法对比大图
# ============================================================

def fig2_baseline_comparison(data):
    """
    (a) Full vs B1/B2/B3 行为分布对比
    (b) 各方法 per-video 可疑率折线图
    (c) 测试集指标对比 (F1/Precision/Recall)
    """
    configs = data['configs']
    compare_keys = ['Full', 'B1_threshold', 'B2_rules_only', 'B3_lstm']
    compare_keys = [k for k in compare_keys if k in configs]

    if len(compare_keys) < 2:
        print("[SKIP] fig2: 基线数据不足")
        return

    fig = plt.figure(figsize=(18, 7))
    gs = gridspec.GridSpec(1, 3, wspace=0.3, width_ratios=[1.2, 1.2, 1])

    labels_map = {**{'Full': '完整系统'}, **BASELINE_NAMES}
    labels = [labels_map.get(k, k) for k in compare_keys]

    # ---- (a) 行为分布对比 ----
    ax_a = fig.add_subplot(gs[0, 0])
    x = np.arange(len(compare_keys))
    bottom = np.zeros(len(compare_keys))

    for cls_id in range(6):
        values = []
        for k in compare_keys:
            dist = configs[k]['metrics'].get('behavior_distribution', {})
            values.append(dist.get(cls_id, dist.get(str(cls_id), 0)))
        values = np.array(values, dtype=float)
        ax_a.bar(x, values, 0.6, bottom=bottom,
                 label=f'{BEHAVIOR_NAMES[cls_id]}',
                 color=BEHAVIOR_COLORS[cls_id], edgecolor='white',
                 linewidth=0.5)
        bottom += values

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, fontsize=9)
    ax_a.set_ylabel('轨迹数', fontsize=10)
    ax_a.set_title('(a) 行为分布对比', fontsize=12, fontweight='bold')
    ax_a.legend(fontsize=7, loc='upper right', ncol=2)
    ax_a.grid(axis='y', alpha=0.3)

    # ---- (b) 逐视频可疑率 ----
    ax_b = fig.add_subplot(gs[0, 1])
    video_names = data.get('videos_analyzed', [])
    # Short names for display
    short_names = [v.replace('MVI_', 'M').replace('1.14', '') for v in video_names]

    markers = ['o', 's', '^', 'D']
    line_colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']

    for ci, config_key in enumerate(compare_keys):
        per_video = configs[config_key]['metrics'].get('per_video', {})
        sus_rates = []
        valid_videos = []
        for vi, vn in enumerate(video_names):
            if vn in per_video:
                sus_rates.append(per_video[vn].get('suspicious_rate', 0) * 100)
                valid_videos.append(vi)
        if sus_rates:
            ax_b.plot(valid_videos, sus_rates,
                      marker=markers[ci % len(markers)],
                      color=line_colors[ci % len(line_colors)],
                      label=labels[ci], linewidth=1.5, markersize=5)

    ax_b.set_xticks(range(len(video_names)))
    ax_b.set_xticklabels(short_names, fontsize=8, rotation=30)
    ax_b.set_ylabel('可疑率 (%)', fontsize=10)
    ax_b.set_title('(b) 逐视频可疑率', fontsize=12, fontweight='bold')
    ax_b.legend(fontsize=8)
    ax_b.grid(alpha=0.3)

    # ---- (c) 测试集指标 ----
    ax_c = fig.add_subplot(gs[0, 2])
    test_metrics = load_test_metrics()

    if test_metrics:
        model_keys = ['rule', 'lstm', 'transformer', 'transformer_uw']
        model_labels = ['规则基线', 'LSTM', 'Transformer', 'Trans+UW\n(本文)']
        metric_names = ['F1', 'Precision', 'Recall']
        metric_keys = ['test_f1', 'test_precision', 'test_recall']
        bar_colors = ['#e74c3c', '#3498db', '#2ecc71']

        x_test = np.arange(len(model_keys))
        width = 0.22

        for mi, (mn, mk) in enumerate(zip(metric_names, metric_keys)):
            values = [test_metrics.get(k, {}).get(mk, 0) for k in model_keys]
            bars = ax_c.bar(x_test + mi * width, values, width,
                            label=mn, color=bar_colors[mi],
                            edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars, values):
                if val > 0:
                    ax_c.text(bar.get_x() + bar.get_width()/2.,
                              bar.get_height() + 0.005,
                              f'{val:.3f}', ha='center', va='bottom',
                              fontsize=6, fontweight='bold')

        ax_c.set_xticks(x_test + width)
        ax_c.set_xticklabels(model_labels, fontsize=8)
        ax_c.set_ylabel('分数', fontsize=10)
        ax_c.set_title('(c) 测试集指标', fontsize=12, fontweight='bold')
        ax_c.legend(fontsize=8)
        ax_c.set_ylim(0, 1.1)
        ax_c.grid(axis='y', alpha=0.3)
    else:
        ax_c.text(0.5, 0.5, '测试集指标\n数据不可用',
                  ha='center', va='center', fontsize=12, transform=ax_c.transAxes)

    fig.suptitle('图2: 基线方法对比分析', fontsize=14, fontweight='bold', y=1.02)

    for fmt in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f'ablation_fig2_baseline.{fmt}',
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] Fig2: 基线方法对比 -> {OUTPUT_DIR}/ablation_fig2_baseline.png")


# ============================================================
# Fig 3: 逐场景分析 heatmap
# ============================================================

def fig3_scenario_analysis(data):
    """
    正面场景(MVI) vs 侧面场景(1.14) 在不同配置下的表现差异 heatmap
    """
    configs = data['configs']
    all_config_keys = ['Full', 'A1_no_gate', 'A2_no_transformer',
                       'A3_no_rules', 'A4_no_smooth',
                       'B1_threshold', 'B2_rules_only', 'B3_lstm']
    config_keys = [k for k in all_config_keys if k in configs]

    if not config_keys:
        print("[SKIP] fig3: 无配置数据")
        return

    all_labels = {**ABLATION_NAMES, **BASELINE_NAMES}
    config_labels = [all_labels.get(k, k) for k in config_keys]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # ---- (a) 一致率 heatmap: 正面 vs 侧面 ----
    ax_a = axes[0]
    scene_labels = ['正面场景', '侧面场景']
    matrix_agr = np.zeros((len(config_keys), 2))

    for ci, ck in enumerate(config_keys):
        m = configs[ck]['metrics']
        frontal = m.get('frontal', {}).get('agreement', 0)
        lateral = m.get('lateral', {}).get('agreement', 0)
        matrix_agr[ci, 0] = frontal * 100
        matrix_agr[ci, 1] = lateral * 100

    im_a = ax_a.imshow(matrix_agr, cmap='RdYlGn', aspect='auto',
                        vmin=0, vmax=100)
    ax_a.set_xticks(range(2))
    ax_a.set_xticklabels(scene_labels, fontsize=10)
    ax_a.set_yticks(range(len(config_keys)))
    ax_a.set_yticklabels(config_labels, fontsize=9)

    for i in range(len(config_keys)):
        for j in range(2):
            val = matrix_agr[i, j]
            text_color = 'white' if val < 40 or val > 80 else 'black'
            ax_a.text(j, i, f'{val:.1f}%', ha='center', va='center',
                      fontsize=9, fontweight='bold', color=text_color)

    ax_a.set_title('(a) 一致率 (%)', fontsize=12, fontweight='bold')
    plt.colorbar(im_a, ax=ax_a, shrink=0.8, label='%')

    # ---- (b) 可疑率 heatmap ----
    ax_b = axes[1]
    matrix_sus = np.zeros((len(config_keys), 2))

    for ci, ck in enumerate(config_keys):
        m = configs[ck]['metrics']
        frontal = m.get('frontal', {}).get('suspicious_rate', 0)
        lateral = m.get('lateral', {}).get('suspicious_rate', 0)
        matrix_sus[ci, 0] = frontal * 100
        matrix_sus[ci, 1] = lateral * 100

    im_b = ax_b.imshow(matrix_sus, cmap='YlOrRd', aspect='auto',
                        vmin=0, vmax=100)
    ax_b.set_xticks(range(2))
    ax_b.set_xticklabels(scene_labels, fontsize=10)
    ax_b.set_yticks(range(len(config_keys)))
    ax_b.set_yticklabels(config_labels, fontsize=9)

    for i in range(len(config_keys)):
        for j in range(2):
            val = matrix_sus[i, j]
            text_color = 'white' if val > 70 else 'black'
            ax_b.text(j, i, f'{val:.1f}%', ha='center', va='center',
                      fontsize=9, fontweight='bold', color=text_color)

    ax_b.set_title('(b) 可疑率 (%)', fontsize=12, fontweight='bold')
    plt.colorbar(im_b, ax=ax_b, shrink=0.8, label='%')

    # ---- (c) 逐视频一致率 heatmap ----
    ax_c = axes[2]
    video_names = data.get('videos_analyzed', [])
    short_names = [v.replace('MVI_', 'M').replace('1.14', '') for v in video_names]

    matrix_video = np.zeros((len(config_keys), len(video_names)))
    for ci, ck in enumerate(config_keys):
        per_video = configs[ck]['metrics'].get('per_video', {})
        for vi, vn in enumerate(video_names):
            if vn in per_video:
                matrix_video[ci, vi] = per_video[vn].get('agreement', 0) * 100

    im_c = ax_c.imshow(matrix_video, cmap='RdYlGn', aspect='auto',
                        vmin=0, vmax=100)
    ax_c.set_xticks(range(len(video_names)))
    ax_c.set_xticklabels(short_names, fontsize=8, rotation=45)
    ax_c.set_yticks(range(len(config_keys)))
    ax_c.set_yticklabels(config_labels, fontsize=9)

    for i in range(len(config_keys)):
        for j in range(len(video_names)):
            val = matrix_video[i, j]
            if val > 0:
                text_color = 'white' if val < 40 or val > 80 else 'black'
                ax_c.text(j, i, f'{val:.0f}', ha='center', va='center',
                          fontsize=7, fontweight='bold', color=text_color)

    ax_c.set_title('(c) 逐视频一致率 (%)', fontsize=12, fontweight='bold')
    plt.colorbar(im_c, ax=ax_c, shrink=0.8, label='%')

    fig.suptitle('图3: 正面场景 vs 侧面场景分析', fontsize=14,
                 fontweight='bold', y=1.02)

    for fmt in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f'ablation_fig3_scenario.{fmt}',
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] Fig3: 逐场景分析 -> {OUTPUT_DIR}/ablation_fig3_scenario.png")


# ============================================================
# Fig 4: 综合对比雷达图
# ============================================================

def fig4_radar_comparison(data):
    """
    Full vs A1-A4 消融实验的雷达图对比
    """
    configs = data['configs']
    config_keys = ['Full', 'A1_no_gate', 'A2_no_transformer',
                   'A3_no_rules', 'A4_no_smooth']
    config_keys = [k for k in config_keys if k in configs]

    if len(config_keys) < 2:
        print("[SKIP] fig4: 配置数据不足")
        return

    all_labels = {**ABLATION_NAMES, **BASELINE_NAMES}
    config_labels = [all_labels.get(k, k) for k in config_keys]

    # 指标维度
    dimensions = ['一致率', '可疑率', '熵', '正面一致', '侧面一致']
    num_dims = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, num_dims, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#2ecc71',
              '#1abc9c', '#e67e22', '#34495e']

    for ci, ck in enumerate(config_keys):
        m = configs[ck]['metrics']
        values = [
            m.get('agreement', 0) * 100,
            m.get('suspicious_rate', 0) * 100,
            m.get('shannon_entropy', 0) / 2.6 * 100,  # normalize to 0-100
            m.get('frontal', {}).get('agreement', 0) * 100,
            m.get('lateral', {}).get('agreement', 0) * 100,
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, markersize=4,
                color=colors[ci % len(colors)], label=config_labels[ci])
        ax.fill(angles, values, alpha=0.1, color=colors[ci % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title('消融实验多维度对比', fontsize=14, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
    ax.grid(True, alpha=0.3)

    for fmt in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f'ablation_fig4_radar.{fmt}',
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] Fig4: 雷达图 -> {OUTPUT_DIR}/ablation_fig4_radar.png")


# ============================================================
# Fig 5: 混淆矩阵 heatmap
# ============================================================

def fig5_confusion_matrices(data):
    """
    各配置的混淆矩阵 (行=全系统在线标签, 列=当前配置标签)
    """
    configs = data['configs']
    show_keys = ['Full', 'A1_no_gate', 'A2_no_transformer',
                 'A3_no_rules', 'B1_threshold', 'B2_rules_only']
    show_keys = [k for k in show_keys if k in configs
                 and 'confusion_matrix' in configs[k].get('metrics', {})]

    if not show_keys:
        print("[SKIP] fig5: 无混淆矩阵数据")
        return

    all_labels = {**ABLATION_NAMES, **BASELINE_NAMES}
    n = len(show_keys)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    class_names = ['Normal', 'Glanc', 'QTurn', 'Prolong', 'LDown', 'LUp']

    for ci, ck in enumerate(show_keys):
        ax = axes[ci]
        cm = np.array(configs[ck]['metrics']['confusion_matrix'], dtype=float)

        # 归一化（按行，即按真实标签）
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm / row_sums

        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')

        ax.set_xticks(range(6))
        ax.set_xticklabels(class_names, fontsize=8, rotation=30)
        ax.set_yticks(range(6))
        ax.set_yticklabels(class_names, fontsize=8)

        # 标注数字（原始计数 + 百分比）
        for i in range(6):
            for j in range(6):
                count = int(cm[i, j])
                pct = cm_norm[i, j]
                if count > 0:
                    text_color = 'white' if pct > 0.5 else 'black'
                    ax.text(j, i, f'{count}\n({pct:.0%})',
                            ha='center', va='center', fontsize=7,
                            fontweight='bold', color=text_color)

        label = all_labels.get(ck, ck)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('预测标签', fontsize=9)
        ax.set_ylabel('全系统标签', fontsize=9)

    # 隐藏多余子图
    for ci in range(len(show_keys), len(axes)):
        axes[ci].set_visible(False)

    fig.suptitle('图5: 各配置混淆矩阵 (行=全系统在线标签, 列=离线复现标签)',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    for fmt in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f'ablation_fig5_confusion.{fmt}',
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] Fig5: 混淆矩阵 -> {OUTPUT_DIR}/ablation_fig5_confusion.png")


# ============================================================
# Fig 6: Per-class F1 + Bootstrap CI + 显著性
# ============================================================

def fig6_perclass_f1_and_significance(data):
    """
    (a) Per-class F1 grouped bar chart with error bars
    (b) Bootstrap CI for agreement & suspicious rate
    (c) 显著性检验结果
    """
    configs = data['configs']
    all_config_keys = ['Full', 'A1_no_gate', 'A2_no_transformer',
                       'A3_no_rules', 'A4_no_smooth',
                       'B1_threshold', 'B2_rules_only', 'B3_lstm']
    config_keys = [k for k in all_config_keys if k in configs]

    if not config_keys:
        print("[SKIP] fig6: 无配置数据")
        return

    all_labels_map = {**ABLATION_NAMES, **BASELINE_NAMES}
    labels = [all_labels_map.get(k, k) for k in config_keys]

    fig = plt.figure(figsize=(20, 7))
    gs = gridspec.GridSpec(1, 3, wspace=0.3, width_ratios=[1.5, 1, 1])

    # ---- (a) Per-class F1 ----
    ax_a = fig.add_subplot(gs[0, 0])
    class_names = ['Normal', 'Glancing', 'QuickTurn', 'Prolonged', 'LookDown', 'LookUp']
    x = np.arange(len(config_keys))
    n_classes = 6
    width = 0.12

    for cls_id in range(n_classes):
        f1_values = []
        for ck in config_keys:
            pcm = configs[ck]['metrics'].get('per_class_metrics', {})
            f1_values.append(pcm.get(str(cls_id), {}).get('f1', 0))

        offset = (cls_id - n_classes / 2 + 0.5) * width
        bars = ax_a.bar(x + offset, f1_values, width,
                        label=class_names[cls_id],
                        color=BEHAVIOR_COLORS[cls_id],
                        edgecolor='white', linewidth=0.3)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, fontsize=8, rotation=20)
    ax_a.set_ylabel('F1 Score', fontsize=10)
    ax_a.set_title('(a) Per-class F1 (vs 全系统)', fontsize=12, fontweight='bold')
    ax_a.legend(fontsize=7, ncol=3, loc='upper right')
    ax_a.set_ylim(0, 1.05)
    ax_a.grid(axis='y', alpha=0.3)

    # ---- (b) Bootstrap CI ----
    ax_b = fig.add_subplot(gs[0, 1])
    means = []
    ci_lows = []
    ci_highs = []
    for ck in config_keys:
        bs = configs[ck]['metrics'].get('bootstrap_ci', {}).get('agreement', {})
        m = bs.get('mean', 0)
        lo = bs.get('ci_low', 0)
        hi = bs.get('ci_high', 0)
        means.append(m * 100)
        ci_lows.append((m - lo) * 100)
        ci_highs.append((hi - m) * 100)

    bars_b = ax_b.barh(x, means, xerr=[ci_lows, ci_highs],
                        color='#3498db', edgecolor='white', linewidth=0.5,
                        height=0.6, capsize=3, error_kw={'linewidth': 1})
    ax_b.set_yticks(x)
    ax_b.set_yticklabels(labels, fontsize=8)
    ax_b.set_xlabel('一致率 (%)', fontsize=10)
    ax_b.set_title('(b) Bootstrap 95% CI', fontsize=12, fontweight='bold')
    ax_b.grid(axis='x', alpha=0.3)

    for i, (m, lo, hi) in enumerate(zip(means, ci_lows, ci_highs)):
        ax_b.text(m + hi + 0.5, i, f'{m:.1f}%', va='center', fontsize=8)

    # ---- (c) 显著性检验 ----
    ax_c = fig.add_subplot(gs[0, 2])
    sig_keys = [k for k in config_keys if k != 'Full']
    sig_labels = [all_labels_map.get(k, k) for k in sig_keys]
    p_values = []
    mean_diffs = []
    std_diffs = []

    for ck in sig_keys:
        sig = configs[ck]['metrics'].get('significance_vs_full', {})
        p_values.append(sig.get('p_value', 1.0))
        mean_diffs.append(sig.get('mean_diff', 0) * 100)
        std_diffs.append(sig.get('std_diff', 0) * 100)

    y_pos = np.arange(len(sig_keys))
    colors_sig = ['#e74c3c' if p < 0.05 else '#95a5a6' for p in p_values]

    bars_c = ax_c.barh(y_pos, mean_diffs, xerr=std_diffs, height=0.6,
                        color=colors_sig, edgecolor='white', linewidth=0.5,
                        capsize=3, error_kw={'linewidth': 1})
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(sig_labels, fontsize=8)
    ax_c.set_xlabel('一致率差异 Δ (百分点)', fontsize=10)
    ax_c.axvline(x=0, color='black', linewidth=0.5)
    ax_c.set_title('(c) Wilcoxon 显著性检验\n(红色=p<0.05)', fontsize=12, fontweight='bold')
    ax_c.grid(axis='x', alpha=0.3)

    for i, (md, p) in enumerate(zip(mean_diffs, p_values)):
        star = " *" if p < 0.05 else ""
        ax_c.text(max(md, 0) + std_diffs[i] + 0.5, i,
                  f'p={p:.3f}{star}', va='center', fontsize=7)

    fig.suptitle('图6: Per-class F1、置信区间与显著性检验',
                 fontsize=14, fontweight='bold', y=1.02)

    for fmt in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f'ablation_fig6_significance.{fmt}',
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] Fig6: 显著性检验 -> {OUTPUT_DIR}/ablation_fig6_significance.png")


# ============================================================
# Fig 7: 行为标签迁移热力图 (Label Transition Analysis)
# ============================================================

def fig7_label_transition(data):
    """
    Full → 各配置的行为标签迁移矩阵
    显示每种行为在不同配置下的重分配情况
    """
    configs = data['configs']
    show_keys = ['A1_no_gate', 'A2_no_transformer', 'A3_no_rules',
                 'A4_no_smooth', 'B1_threshold', 'B2_rules_only']
    show_keys = [k for k in show_keys if k in configs
                 and 'confusion_matrix' in configs[k].get('metrics', {})]

    if len(show_keys) < 2:
        print("[SKIP] fig7: 不足2个配置有混淆矩阵")
        return

    all_labels = {**ABLATION_NAMES, **BASELINE_NAMES}
    class_names = ['Normal', 'Glancing', 'QuickTurn', 'Prolonged', 'LookDown', 'LookUp']

    # 构建迁移概要：对每个 Full 中的行为类, 在各配置中被分为哪一类
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    axes = axes.flatten()

    for cls_id in range(6):
        ax = axes[cls_id]
        # 对每个配置, 从混淆矩阵的第 cls_id 行获取分配情况
        config_names = []
        distributions = []

        for ck in show_keys:
            cm = np.array(configs[ck]['metrics']['confusion_matrix'], dtype=float)
            row = cm[cls_id]
            total = row.sum()
            if total > 0:
                row_pct = row / total * 100
            else:
                row_pct = np.zeros(6)
            distributions.append(row_pct)
            config_names.append(all_labels.get(ck, ck))

        distributions = np.array(distributions)
        x = np.arange(len(config_names))

        bottom = np.zeros(len(config_names))
        for target_cls in range(6):
            ax.bar(x, distributions[:, target_cls], 0.7, bottom=bottom,
                   label=class_names[target_cls] if cls_id == 0 else '',
                   color=BEHAVIOR_COLORS[target_cls], edgecolor='white', linewidth=0.3)
            # 标注 >10% 的值
            for i, val in enumerate(distributions[:, target_cls]):
                if val > 10:
                    ax.text(i, bottom[i] + val/2, f'{val:.0f}%',
                            ha='center', va='center', fontsize=6,
                            fontweight='bold', color='white' if val > 30 else 'black')
            bottom += distributions[:, target_cls]

        ax.set_xticks(x)
        ax.set_xticklabels(config_names, fontsize=7, rotation=25)
        ax.set_ylabel('百分比 (%)', fontsize=8)
        ax.set_title(f'{class_names[cls_id]} ({BEHAVIOR_CN[cls_id]})',
                     fontsize=11, fontweight='bold')
        ax.set_ylim(0, 105)

    # 统一图例
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, class_names, loc='lower center', ncol=6,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('图7: 行为标签迁移分析\n(全系统标签在各配置中的重分配情况)',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    for fmt in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f'ablation_fig7_transition.{fmt}',
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] Fig7: 标签迁移分析 -> {OUTPUT_DIR}/ablation_fig7_transition.png")


# ============================================================
# Fig 8: 逐视频详细性能网格 (Per-Video Performance Grid)
# ============================================================

def fig8_pervideo_grid(data):
    """
    每个视频×每个配置的详细指标面板:
    (a) 一致率热力图 (b) 可疑率热力图 (c) 主要行为类型
    """
    configs = data['configs']
    video_names = data.get('videos_analyzed', [])
    all_config_keys = ['Full', 'A1_no_gate', 'A2_no_transformer',
                       'A3_no_rules', 'A4_no_smooth',
                       'B1_threshold', 'B2_rules_only', 'B3_lstm']
    config_keys = [k for k in all_config_keys if k in configs]

    if not config_keys or not video_names:
        print("[SKIP] fig8: 数据不足")
        return

    all_labels = {**{'Full': '完整系统'}, **ABLATION_NAMES, **BASELINE_NAMES}
    config_labels = [all_labels.get(k, k) for k in config_keys]
    short_videos = [v.replace('MVI_', 'M').replace('1.14', '') for v in video_names]

    fig = plt.figure(figsize=(22, 10))
    gs = gridspec.GridSpec(1, 3, wspace=0.25, width_ratios=[1, 1, 1.3])

    # ---- (a) 一致率热力图 ----
    ax_a = fig.add_subplot(gs[0, 0])
    matrix = np.zeros((len(config_keys), len(video_names)))
    for ci, ck in enumerate(config_keys):
        pv = configs[ck]['metrics'].get('per_video', {})
        for vi, vn in enumerate(video_names):
            if vn in pv:
                matrix[ci, vi] = pv[vn].get('agreement', 0) * 100

    im_a = ax_a.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax_a.set_xticks(range(len(video_names)))
    ax_a.set_xticklabels(short_videos, fontsize=7, rotation=45)
    ax_a.set_yticks(range(len(config_keys)))
    ax_a.set_yticklabels(config_labels, fontsize=8)
    for i in range(len(config_keys)):
        for j in range(len(video_names)):
            val = matrix[i, j]
            tc = 'white' if val < 30 or val > 75 else 'black'
            ax_a.text(j, i, f'{val:.0f}', ha='center', va='center',
                      fontsize=6, fontweight='bold', color=tc)
    ax_a.set_title('(a) 一致率 (%)', fontsize=12, fontweight='bold')
    plt.colorbar(im_a, ax=ax_a, shrink=0.6, label='%')

    # ---- (b) 可疑率热力图 ----
    ax_b = fig.add_subplot(gs[0, 1])
    matrix_sus = np.zeros((len(config_keys), len(video_names)))
    for ci, ck in enumerate(config_keys):
        pv = configs[ck]['metrics'].get('per_video', {})
        for vi, vn in enumerate(video_names):
            if vn in pv:
                matrix_sus[ci, vi] = pv[vn].get('suspicious_rate', 0) * 100

    im_b = ax_b.imshow(matrix_sus, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    ax_b.set_xticks(range(len(video_names)))
    ax_b.set_xticklabels(short_videos, fontsize=7, rotation=45)
    ax_b.set_yticks(range(len(config_keys)))
    ax_b.set_yticklabels(config_labels, fontsize=8)
    for i in range(len(config_keys)):
        for j in range(len(video_names)):
            val = matrix_sus[i, j]
            tc = 'white' if val > 70 else 'black'
            ax_b.text(j, i, f'{val:.0f}', ha='center', va='center',
                      fontsize=6, fontweight='bold', color=tc)
    ax_b.set_title('(b) 可疑率 (%)', fontsize=12, fontweight='bold')
    plt.colorbar(im_b, ax=ax_b, shrink=0.6, label='%')

    # ---- (c) 行为分布堆叠条形图 (per video) ----
    ax_c = fig.add_subplot(gs[0, 2])
    class_names = ['Normal', 'Glancing', 'QuickTurn', 'Prolonged', 'LookDown', 'LookUp']

    # 仅显示 Full 配置的每个视频的行为分布
    full_pv = configs.get('Full', {}).get('metrics', {}).get('per_video', {})
    x = np.arange(len(video_names))
    bottom = np.zeros(len(video_names))

    for cls_id in range(6):
        values = []
        for vn in video_names:
            dist = full_pv.get(vn, {}).get('behavior_distribution', {})
            values.append(dist.get(cls_id, dist.get(str(cls_id), 0)))
        values = np.array(values, dtype=float)
        ax_c.bar(x, values, 0.7, bottom=bottom, label=class_names[cls_id],
                 color=BEHAVIOR_COLORS[cls_id], edgecolor='white', linewidth=0.3)
        bottom += values

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(short_videos, fontsize=7, rotation=45)
    ax_c.set_ylabel('轨迹数', fontsize=9)
    ax_c.set_title('(c) 完整系统各视频行为分布', fontsize=12, fontweight='bold')
    ax_c.legend(fontsize=7, ncol=2, loc='upper right')
    ax_c.grid(axis='y', alpha=0.3)

    fig.suptitle('图8: 逐视频详细性能分析',
                 fontsize=14, fontweight='bold', y=1.01)

    for fmt in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f'ablation_fig8_pervideo.{fmt}',
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] Fig8: 逐视频详细分析 -> {OUTPUT_DIR}/ablation_fig8_pervideo.png")


# ============================================================
# Fig 9: 参数敏感性曲线 (Parameter Sensitivity)
# ============================================================

def fig9_sensitivity_curves():
    """
    (a) 平滑窗口 vs 一致率/可疑率/熵
    (b) 门控阈值 vs 一致率/可疑率
    (c) 投票阈值 vs 一致率/可疑率
    """
    path = Path('data/ablation_output/sensitivity_results.json')
    if not path.exists():
        print("[SKIP] fig9: 未找到 sensitivity_results.json")
        return

    with open(path, 'r') as f:
        sens_data = json.load(f)
    results = sens_data.get('results', {})

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # ---- (a) 平滑窗口 ----
    ax = axes[0]
    sw_data = results.get('smooth_window_sweep', [])
    if sw_data:
        xs = [d['smooth_window'] for d in sw_data]
        agrs = [d['agreement'] * 100 for d in sw_data]
        sus = [d['suspicious_rate'] * 100 for d in sw_data]
        ents = [d['entropy'] for d in sw_data]

        ax.plot(xs, agrs, 'o-', color='#3498db', linewidth=2,
                markersize=6, label='一致率 (%)', zorder=3)
        ax.plot(xs, sus, 's-', color='#e74c3c', linewidth=2,
                markersize=6, label='可疑率 (%)', zorder=3)

        ax2 = ax.twinx()
        ax2.plot(xs, ents, '^--', color='#f39c12', linewidth=1.5,
                 markersize=5, label='Shannon 熵', alpha=0.8)
        ax2.set_ylabel('Shannon 熵', fontsize=10, color='#f39c12')
        ax2.tick_params(axis='y', labelcolor='#f39c12')
        ax2.legend(loc='upper right', fontsize=8)

        # 标注最优点
        best_idx = max(range(len(agrs)), key=lambda i: agrs[i])
        ax.annotate(f'最优 w={xs[best_idx]}',
                    xy=(xs[best_idx], agrs[best_idx]),
                    xytext=(xs[best_idx]+2, agrs[best_idx]+2),
                    fontsize=8, fontweight='bold', color='#3498db',
                    arrowprops=dict(arrowstyle='->', color='#3498db'))

        ax.axvline(x=8, color='gray', linestyle=':', alpha=0.5, label='默认 w=8')

    ax.set_xlabel('平滑窗口大小 (w)', fontsize=10)
    ax.set_ylabel('百分比 (%)', fontsize=10)
    ax.set_title('(a) 时序平滑窗口敏感性', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)

    # ---- (b) 门控阈值 ----
    ax = axes[1]
    yt_data = results.get('yaw_threshold_sweep', [])
    if yt_data:
        xs = [d['yaw_threshold'] for d in yt_data]
        agrs = [d['agreement'] * 100 for d in yt_data]
        sus = [d['suspicious_rate'] * 100 for d in yt_data]

        ax.plot(xs, agrs, 'o-', color='#3498db', linewidth=2,
                markersize=6, label='一致率 (%)')
        ax.plot(xs, sus, 's-', color='#e74c3c', linewidth=2,
                markersize=6, label='可疑率 (%)')

        best_idx = max(range(len(agrs)), key=lambda i: agrs[i])
        ax.annotate(f'最优 th={xs[best_idx]}°',
                    xy=(xs[best_idx], agrs[best_idx]),
                    xytext=(xs[best_idx]+3, agrs[best_idx]+2),
                    fontsize=8, fontweight='bold', color='#3498db',
                    arrowprops=dict(arrowstyle='->', color='#3498db'))

        ax.axvline(x=40, color='gray', linestyle=':', alpha=0.5, label='默认 th=40°')

        # 标注可疑率的跳变
        ax.fill_between(xs, 0, 100, where=[x <= 30 for x in xs],
                        alpha=0.05, color='red', label='高敏感区')

    ax.set_xlabel('Yaw 门控阈值 (°)', fontsize=10)
    ax.set_ylabel('百分比 (%)', fontsize=10)
    ax.set_title('(b) 姿态门控阈值敏感性', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ---- (c) 投票阈值 ----
    ax = axes[2]
    vt_data = results.get('vote_threshold_sweep', [])
    if vt_data:
        xs = [d['vote_threshold'] for d in vt_data]
        agrs = [d['agreement'] * 100 for d in vt_data]
        sus = [d['suspicious_rate'] * 100 for d in vt_data]

        ax.plot(xs, agrs, 'o-', color='#3498db', linewidth=2,
                markersize=6, label='一致率 (%)')
        ax.plot(xs, sus, 's-', color='#e74c3c', linewidth=2,
                markersize=6, label='可疑率 (%)')

        best_idx = max(range(len(agrs)), key=lambda i: agrs[i])
        ax.annotate(f'最优 vt={xs[best_idx]:.2f}',
                    xy=(xs[best_idx], agrs[best_idx]),
                    xytext=(xs[best_idx]+0.03, agrs[best_idx]+2),
                    fontsize=8, fontweight='bold', color='#3498db',
                    arrowprops=dict(arrowstyle='->', color='#3498db'))

        ax.axvline(x=0.15, color='gray', linestyle=':', alpha=0.5, label='默认 vt=0.15')

    ax.set_xlabel('投票阈值', fontsize=10)
    ax.set_ylabel('百分比 (%)', fontsize=10)
    ax.set_title('(c) 投票阈值敏感性', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle('图9: 参数敏感性分析', fontsize=14, fontweight='bold', y=1.02)

    for fmt in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f'ablation_fig9_sensitivity.{fmt}',
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] Fig9: 参数敏感性 -> {OUTPUT_DIR}/ablation_fig9_sensitivity.png")


# ============================================================
# Fig 10: 综合性能总结仪表盘 (Summary Dashboard)
# ============================================================

def fig10_summary_dashboard(data):
    """
    综合单页仪表盘:
    (a) 所有配置一致率排序条形图
    (b) Macro-F1 排序条形图
    (c) 行为多样性 (entropy) 排序
    (d) 核心发现文字总结
    """
    configs = data['configs']
    all_config_keys = ['Full', 'A1_no_gate', 'A2_no_transformer',
                       'A3_no_rules', 'A4_no_smooth',
                       'B1_threshold', 'B2_rules_only', 'B3_lstm']
    config_keys = [k for k in all_config_keys if k in configs]

    if not config_keys:
        print("[SKIP] fig10: 无配置数据")
        return

    all_labels = {**{'Full': '完整系统'}, **ABLATION_NAMES, **BASELINE_NAMES}

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # 准备数据
    names = [all_labels.get(k, k) for k in config_keys]
    agreements = [configs[k]['metrics'].get('agreement', 0) * 100 for k in config_keys]
    suspicious = [configs[k]['metrics'].get('suspicious_rate', 0) * 100 for k in config_keys]
    entropies = [configs[k]['metrics'].get('shannon_entropy', 0) for k in config_keys]
    macro_f1s = [configs[k]['metrics'].get('per_class_metrics', {}).get(
        'macro', {}).get('f1', 0) for k in config_keys]

    # ---- (a) 一致率排序 ----
    ax_a = fig.add_subplot(gs[0, 0])
    sorted_idx = np.argsort(agreements)[::-1]
    sorted_names = [names[i] for i in sorted_idx]
    sorted_agrs = [agreements[i] for i in sorted_idx]
    sorted_colors = ['#3498db' if config_keys[i] == 'Full' else
                     '#e74c3c' if config_keys[i].startswith('A') else
                     '#f39c12' for i in sorted_idx]

    bars_a = ax_a.barh(range(len(sorted_names)), sorted_agrs,
                        color=sorted_colors, edgecolor='white', linewidth=0.5)
    ax_a.set_yticks(range(len(sorted_names)))
    ax_a.set_yticklabels(sorted_names, fontsize=9)
    ax_a.invert_yaxis()
    ax_a.set_xlabel('一致率 (%)', fontsize=10)
    ax_a.set_title('(a) 一致率排序 (vs 全系统在线推理)',
                   fontsize=12, fontweight='bold')
    for i, v in enumerate(sorted_agrs):
        ax_a.text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=9, fontweight='bold')
    ax_a.grid(axis='x', alpha=0.3)

    # ---- (b) Macro-F1 排序 ----
    ax_b = fig.add_subplot(gs[0, 1])
    sorted_idx_f1 = np.argsort(macro_f1s)[::-1]
    sorted_names_f1 = [names[i] for i in sorted_idx_f1]
    sorted_f1s = [macro_f1s[i] for i in sorted_idx_f1]
    sorted_colors_f1 = ['#3498db' if config_keys[i] == 'Full' else
                         '#e74c3c' if config_keys[i].startswith('A') else
                         '#f39c12' for i in sorted_idx_f1]

    bars_b = ax_b.barh(range(len(sorted_names_f1)), sorted_f1s,
                        color=sorted_colors_f1, edgecolor='white', linewidth=0.5)
    ax_b.set_yticks(range(len(sorted_names_f1)))
    ax_b.set_yticklabels(sorted_names_f1, fontsize=9)
    ax_b.invert_yaxis()
    ax_b.set_xlabel('Macro-F1', fontsize=10)
    ax_b.set_title('(b) Macro-F1 排序 (vs 全系统)',
                   fontsize=12, fontweight='bold')
    for i, v in enumerate(sorted_f1s):
        ax_b.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9, fontweight='bold')
    ax_b.grid(axis='x', alpha=0.3)

    # ---- (c) 多维度雷达对比 (Full vs 最差 vs 最好基线) ----
    ax_c = fig.add_subplot(gs[1, 0], polar=True)
    dimensions = ['一致率', '可疑率', '行为多样性', '正面一致', '侧面一致']
    num_dims = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, num_dims, endpoint=False).tolist()
    angles += angles[:1]

    highlight_keys = ['Full']
    # Add worst ablation (A1) and best baseline (B3)
    if 'A1_no_gate' in configs:
        highlight_keys.append('A1_no_gate')
    if 'B3_lstm' in configs:
        highlight_keys.append('B3_lstm')
    if 'B1_threshold' in configs:
        highlight_keys.append('B1_threshold')

    radar_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    for ci, ck in enumerate(highlight_keys):
        m = configs[ck]['metrics']
        values = [
            m.get('agreement', 0) * 100,
            m.get('suspicious_rate', 0) * 100,
            m.get('shannon_entropy', 0) / 2.6 * 100,
            m.get('frontal', {}).get('agreement', 0) * 100,
            m.get('lateral', {}).get('agreement', 0) * 100,
        ]
        values += values[:1]
        label = all_labels.get(ck, ck)
        ax_c.plot(angles, values, 'o-', linewidth=2, markersize=4,
                  color=radar_colors[ci], label=label)
        ax_c.fill(angles, values, alpha=0.08, color=radar_colors[ci])

    ax_c.set_xticks(angles[:-1])
    ax_c.set_xticklabels(dimensions, fontsize=9)
    ax_c.set_ylim(0, 105)
    ax_c.set_title('(c) 关键配置多维度对比', fontsize=12, fontweight='bold', pad=20)
    ax_c.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

    # ---- (d) 核心发现总结 ----
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis('off')

    full_agr = configs.get('Full', {}).get('metrics', {}).get('agreement', 0) * 100
    full_sus = configs.get('Full', {}).get('metrics', {}).get('suspicious_rate', 0) * 100
    full_ent = configs.get('Full', {}).get('metrics', {}).get('shannon_entropy', 0)

    findings = [
        "核心实验发现",
        "═" * 35,
        "",
        f"1. 完整系统一致率: {full_agr:.1f}%",
        f"   可疑行为检出率: {full_sus:.1f}%",
        f"   行为多样性 (Shannon): {full_ent:.3f}",
        "",
        "2. 模块重要性排序:",
    ]

    # 计算各模块贡献
    module_info = [
        ('A1_no_gate', '姿态门控 (Pose Gate)'),
        ('A3_no_rules', '规则检测 (Rules)'),
        ('A2_no_transformer', 'Transformer 模型'),
        ('A4_no_smooth', '时序平滑 (Smoothing)'),
    ]
    drops = []
    for mk, mn in module_info:
        if mk in configs:
            a_agr = configs[mk]['metrics'].get('agreement', 0) * 100
            drop = full_agr - a_agr
            drops.append((mn, drop))

    drops.sort(key=lambda x: x[1], reverse=True)
    for rank, (mn, drop) in enumerate(drops, 1):
        marker = "***" if drop > 10 else "**" if drop > 2 else "*"
        findings.append(f"   #{rank}: {mn}  Δ={drop:+.1f}% {marker}")

    findings.extend([
        "",
        "3. 基线对比:",
    ])
    for bk in ['B1_threshold', 'B2_rules_only', 'B3_lstm']:
        if bk in configs:
            b_agr = configs[bk]['metrics'].get('agreement', 0) * 100
            bn = all_labels.get(bk, bk)
            findings.append(f"   {bn}: {b_agr:.1f}%")

    findings.extend([
        "",
        "4. 统计检验:",
    ])
    for ck in config_keys:
        if ck == 'Full':
            continue
        sig = configs[ck]['metrics'].get('significance_vs_full', {})
        if sig:
            p = sig.get('p_value', 1.0)
            cn = all_labels.get(ck, ck)
            star = " (p<0.05) *" if p < 0.05 else f" (p={p:.3f})"
            findings.append(f"   Full vs {cn}{star}")

    text = '\n'.join(findings)
    ax_d.text(0.05, 0.95, text, transform=ax_d.transAxes,
              fontsize=9, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa',
                       edgecolor='#dee2e6', linewidth=1))

    fig.suptitle('图10: 实验结果综合总结仪表盘',
                 fontsize=14, fontweight='bold', y=1.01)

    for fmt in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f'ablation_fig10_dashboard.{fmt}',
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] Fig10: 总结仪表盘 -> {OUTPUT_DIR}/ablation_fig10_dashboard.png")


# ============================================================
# Fig 11: 误差分析 (Error Analysis)
# ============================================================

def fig11_error_analysis(data):
    """
    分析配置间分歧最大的案例:
    (a) 各配置间 pairwise 一致率矩阵
    (b) 正面 vs 侧面的 per-class 准确率对比
    (c) 轨迹数 vs 一致率关系 (气泡图)
    """
    configs = data['configs']
    all_config_keys = ['Full', 'A1_no_gate', 'A2_no_transformer',
                       'A3_no_rules', 'A4_no_smooth',
                       'B1_threshold', 'B2_rules_only', 'B3_lstm']
    config_keys = [k for k in all_config_keys if k in configs]

    if len(config_keys) < 3:
        print("[SKIP] fig11: 配置数据不足")
        return

    all_labels = {**{'Full': '完整系统'}, **ABLATION_NAMES, **BASELINE_NAMES}
    config_labels = [all_labels.get(k, k) for k in config_keys]

    fig = plt.figure(figsize=(20, 7))
    gs = gridspec.GridSpec(1, 3, wspace=0.3, width_ratios=[1.2, 1, 1])

    # ---- (a) Pairwise 一致率矩阵 ----
    ax_a = fig.add_subplot(gs[0, 0])
    n = len(config_keys)
    pairwise = np.zeros((n, n))

    # 获取每个配置的行为分布 (所有轨迹)
    config_distributions = {}
    for ck in config_keys:
        dist = configs[ck]['metrics'].get('behavior_distribution', {})
        config_distributions[ck] = dist

    # 获取混淆矩阵来计算 pairwise agreement
    for i, ck_i in enumerate(config_keys):
        for j, ck_j in enumerate(config_keys):
            if i == j:
                pairwise[i, j] = 100.0
            elif j == 0:  # vs Full
                pairwise[i, j] = configs[ck_i]['metrics'].get('agreement', 0) * 100
            elif i == 0:  # Full vs others
                pairwise[i, j] = configs[ck_j]['metrics'].get('agreement', 0) * 100
            else:
                # 使用行为分布的余弦相似度作为近似
                d_i = configs[ck_i]['metrics'].get('behavior_distribution', {})
                d_j = configs[ck_j]['metrics'].get('behavior_distribution', {})
                vec_i = np.array([d_i.get(c, d_i.get(str(c), 0)) for c in range(6)], dtype=float)
                vec_j = np.array([d_j.get(c, d_j.get(str(c), 0)) for c in range(6)], dtype=float)
                norm_i = np.linalg.norm(vec_i)
                norm_j = np.linalg.norm(vec_j)
                if norm_i > 0 and norm_j > 0:
                    cosine = np.dot(vec_i, vec_j) / (norm_i * norm_j)
                    pairwise[i, j] = cosine * 100
                else:
                    pairwise[i, j] = 0

    im = ax_a.imshow(pairwise, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    ax_a.set_xticks(range(n))
    ax_a.set_xticklabels(config_labels, fontsize=7, rotation=35)
    ax_a.set_yticks(range(n))
    ax_a.set_yticklabels(config_labels, fontsize=7)
    for i in range(n):
        for j in range(n):
            tc = 'white' if pairwise[i,j] < 40 or pairwise[i,j] > 85 else 'black'
            ax_a.text(j, i, f'{pairwise[i,j]:.0f}', ha='center', va='center',
                      fontsize=7, fontweight='bold', color=tc)
    ax_a.set_title('(a) 配置间相似度矩阵', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax_a, shrink=0.7, label='相似度 (%)')

    # ---- (b) 正面 vs 侧面 per-class 准确率 ----
    ax_b = fig.add_subplot(gs[0, 1])
    class_names = ['Normal', 'Glancing', 'QTurn', 'Prolong', 'LDown', 'LUp']

    # Full 配置的正面 vs 侧面
    full_frontal = configs.get('Full', {}).get('metrics', {}).get('frontal', {})
    full_lateral = configs.get('Full', {}).get('metrics', {}).get('lateral', {})
    frontal_pcm = full_frontal.get('per_class_metrics', {})
    lateral_pcm = full_lateral.get('per_class_metrics', {})

    x = np.arange(6)
    width = 0.35
    f1_frontal = [frontal_pcm.get(str(i), {}).get('f1', 0) for i in range(6)]
    f1_lateral = [lateral_pcm.get(str(i), {}).get('f1', 0) for i in range(6)]

    ax_b.bar(x - width/2, f1_frontal, width, label='正面场景',
             color='#3498db', edgecolor='white', linewidth=0.5)
    ax_b.bar(x + width/2, f1_lateral, width, label='侧面场景',
             color='#e74c3c', edgecolor='white', linewidth=0.5)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(class_names, fontsize=8, rotation=20)
    ax_b.set_ylabel('F1 Score', fontsize=10)
    ax_b.set_title('(b) 完整系统: 正面vs侧面 Per-class F1',
                   fontsize=11, fontweight='bold')
    ax_b.legend(fontsize=9)
    ax_b.set_ylim(0, 1.05)
    ax_b.grid(axis='y', alpha=0.3)

    # ---- (c) 气泡图: 轨迹数 vs 一致率 ----
    ax_c = fig.add_subplot(gs[0, 2])
    video_names = data.get('videos_analyzed', [])
    short_videos = [v.replace('MVI_', 'M').replace('1.14', '') for v in video_names]

    # Full config per-video data
    full_pv = configs.get('Full', {}).get('metrics', {}).get('per_video', {})
    for vi, vn in enumerate(video_names):
        if vn not in full_pv:
            continue
        pv = full_pv[vn]
        n_tracks = pv.get('total_tracks', 0)
        agreement = pv.get('agreement', 0) * 100
        sus_rate = pv.get('suspicious_rate', 0) * 100

        is_frontal = vn in FRONTAL_VIDEOS
        color = '#3498db' if is_frontal else '#e74c3c'
        marker = 'o' if is_frontal else 's'

        ax_c.scatter(n_tracks, agreement, s=sus_rate * 3,
                     c=color, marker=marker, alpha=0.7, edgecolors='white',
                     linewidth=0.5, zorder=3)
        ax_c.annotate(short_videos[vi], (n_tracks, agreement),
                      fontsize=7, ha='center', va='bottom',
                      xytext=(0, 5), textcoords='offset points')

    ax_c.scatter([], [], c='#3498db', marker='o', s=50, label='正面场景')
    ax_c.scatter([], [], c='#e74c3c', marker='s', s=50, label='侧面场景')
    ax_c.set_xlabel('轨迹数', fontsize=10)
    ax_c.set_ylabel('一致率 (%)', fontsize=10)
    ax_c.set_title('(c) 轨迹数 vs 一致率\n(气泡大小=可疑率)',
                   fontsize=11, fontweight='bold')
    ax_c.legend(fontsize=8)
    ax_c.grid(alpha=0.3)

    fig.suptitle('图11: 误差分析与配置间对比',
                 fontsize=14, fontweight='bold', y=1.02)

    for fmt in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f'ablation_fig11_error.{fmt}',
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] Fig11: 误差分析 -> {OUTPUT_DIR}/ablation_fig11_error.png")


FRONTAL_VIDEOS = ['MVI_4537', 'MVI_4538', 'MVI_4539', 'MVI_4540']


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("  生成消融实验 & 基线对比可视化图表")
    print("=" * 60)
    print()

    data = load_ablation_data()
    if data is None:
        return

    num_configs = len(data.get('configs', {}))
    num_videos = data.get('num_videos', 0)
    print(f"数据概览: {num_configs} 个配置, {num_videos} 个视频")
    print()

    # 消融实验图表
    print("--- 消融实验与基线对比 ---")
    fig1_ablation_comparison(data)
    fig2_baseline_comparison(data)
    fig3_scenario_analysis(data)
    fig4_radar_comparison(data)

    # 细化分析图表
    print("\n--- 细化分析 (混淆矩阵/F1/显著性) ---")
    fig5_confusion_matrices(data)
    fig6_perclass_f1_and_significance(data)

    # 深度分析图表
    print("\n--- 深度分析 ---")
    fig7_label_transition(data)
    fig8_pervideo_grid(data)
    fig9_sensitivity_curves()  # 需要 sensitivity_results.json
    fig10_summary_dashboard(data)
    fig11_error_analysis(data)

    print()
    total_figs = len(list(OUTPUT_DIR.glob('ablation_fig*.png')))
    print(f"共生成 {total_figs} 张图表，保存至 {OUTPUT_DIR}/")
    print("完成！")


if __name__ == '__main__':
    main()
