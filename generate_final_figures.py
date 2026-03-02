#!/usr/bin/env python3
"""
生成论文终稿图表：SBRN消融、二分类对比、参数敏感性曲线
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# 全局样式
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'blue': '#2563EB',
    'red': '#DC2626',
    'green': '#059669',
    'purple': '#7C3AED',
    'orange': '#EA580C',
    'gray': '#6B7280',
    'teal': '#0D9488',
    'pink': '#DB2777',
}

output_dir = Path('/root/autodl-tmp/behaviour/thesis_figures')


# ==================== 图1: SBRN 6分类消融实验对比 ====================
def fig_sbrn_ablation():
    data = {
        'A0: Baseline\n(Transformer+CE)': {'acc': 77.4, 'f1m': 0.636, 'f1w': 0.776, 'params': 114},
        'A2: +PAPE': {'acc': 77.4, 'f1m': 0.658, 'f1w': 0.774, 'params': 455},
        'A3: +PAPE\n+BPCL': {'acc': 78.6, 'f1m': 0.585, 'f1w': 0.785, 'params': 491},
        'A4: Full\nSBRN+Aug': {'acc': 79.2, 'f1m': 0.717, 'f1w': 0.792, 'params': 491},
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    names = list(data.keys())
    colors = [COLORS['gray'], COLORS['blue'], COLORS['purple'], COLORS['green']]

    # (a) Accuracy
    ax = axes[0]
    accs = [d['acc'] for d in data.values()]
    bars = ax.bar(range(len(names)), accs, color=colors, width=0.6, edgecolor='white', linewidth=1.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('(a) Test Accuracy')
    ax.set_ylim(50, 82)
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # (b) F1-Macro
    ax = axes[1]
    f1s = [d['f1m'] for d in data.values()]
    bars = ax.bar(range(len(names)), f1s, color=colors, width=0.6, edgecolor='white', linewidth=1.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('F1-Macro')
    ax.set_title('(b) F1-Macro Score')
    ax.set_ylim(0.3, 0.80)
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # (c) 参数量 vs F1
    ax = axes[2]
    params = [d['params'] for d in data.values()]
    f1s_c = [d['f1m'] for d in data.values()]
    for i, (p, f, n) in enumerate(zip(params, f1s_c, names)):
        ax.scatter(p, f, c=colors[i], s=120, zorder=5, edgecolors='white', linewidth=1.5)
        ax.annotate(n.replace('\n', ' '), (p, f), textcoords="offset points",
                   xytext=(8, -5 if i != 2 else -15), fontsize=7)
    ax.set_xlabel('Parameters (K)')
    ax.set_ylabel('F1-Macro')
    ax.set_title('(c) Params vs F1-Macro')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(output_dir / f'sbrn_fig1_ablation.{ext}')
    plt.close()
    print('  [OK] sbrn_fig1_ablation')


# ==================== 图2: Per-class F1 热力图 ====================
def fig_perclass_heatmap():
    class_names = ['Normal', 'Glancing', 'QuickTurn', 'Prolonged', 'LookDown', 'LookUp']
    configs = ['A0: Baseline', 'A2: +PAPE', 'A3: +PAPE+BPCL', 'A4: Full SBRN']

    f1_matrix = np.array([
        [0.703, 0.822, 0.200, 0.000, 0.817, 0.636],  # A0
        [0.688, 0.823, 0.519, 0.000, 0.814, 0.444],  # A2
        [0.717, 0.840, 0.375, 0.000, 0.809, 0.182],  # A3
        [0.706, 0.840, 0.480, 0.000, 0.822, 0.737],  # A4
    ])

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(f1_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=10)

    for i in range(len(configs)):
        for j in range(len(class_names)):
            val = f1_matrix[i, j]
            color = 'white' if val < 0.4 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                   fontsize=10, fontweight='bold', color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('F1 Score', fontsize=11)
    ax.set_title('Per-Class F1 Score Across Ablation Configurations', fontsize=13, pad=10)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(output_dir / f'sbrn_fig2_perclass_f1.{ext}')
    plt.close()
    print('  [OK] sbrn_fig2_perclass_f1')


# ==================== 图3: 二分类模型对比 ====================
def fig_binary_comparison():
    models = ['Rule\nBaseline', 'LSTM', 'Transformer', 'Transformer\n+UW (Ours)']
    precision = [1.000, 0.971, 0.680, 0.862]
    recall =    [0.595, 0.790, 0.969, 0.903]
    f1 =        [0.746, 0.871, 0.799, 0.882]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) Grouped bar chart
    ax = axes[0]
    x = np.arange(len(models))
    w = 0.22
    bars1 = ax.bar(x - w, precision, w, label='Precision', color=COLORS['blue'], edgecolor='white')
    bars2 = ax.bar(x, recall, w, label='Recall', color=COLORS['orange'], edgecolor='white')
    bars3 = ax.bar(x + w, f1, w, label='F1 Score', color=COLORS['green'], edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel('Score')
    ax.set_ylim(0.5, 1.05)
    ax.legend(loc='lower right')
    ax.set_title('(a) Binary Classification: Model Comparison')

    for bars in [bars3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # (b) Precision-Recall trade-off scatter
    ax = axes[1]
    colors_list = [COLORS['gray'], COLORS['blue'], COLORS['orange'], COLORS['green']]
    markers = ['s', '^', 'D', '*']
    for i, (p, r, m, c, mk) in enumerate(zip(precision, recall, models, colors_list, markers)):
        ax.scatter(r, p, c=c, s=150 if i < 3 else 250, marker=mk, zorder=5,
                  edgecolors='black', linewidth=0.5)
        label = m.replace('\n', ' ')
        offset = (10, -5) if i != 2 else (10, 8)
        ax.annotate(f'{label}\nF1={f1[i]:.3f}', (r, p),
                   textcoords="offset points", xytext=offset, fontsize=8,
                   arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('(b) Precision-Recall Trade-off')
    ax.set_xlim(0.5, 1.05)
    ax.set_ylim(0.6, 1.05)

    # F1 iso-curves
    for f1_val in [0.7, 0.8, 0.9]:
        r_range = np.linspace(0.5, 1.0, 100)
        p_range = f1_val * r_range / (2 * r_range - f1_val)
        mask = (p_range > 0.5) & (p_range < 1.1)
        ax.plot(r_range[mask], p_range[mask], '--', color='lightgray', linewidth=0.8, alpha=0.7)
        idx = np.argmin(np.abs(p_range - 0.95))
        if mask[idx]:
            ax.text(r_range[idx], p_range[idx], f'F1={f1_val}', fontsize=7, color='gray')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(output_dir / f'sbrn_fig3_binary_comparison.{ext}')
    plt.close()
    print('  [OK] sbrn_fig3_binary_comparison')


# ==================== 图4: 参数敏感性分析三合一 ====================
def fig_sensitivity():
    with open('/root/autodl-tmp/behaviour/data/ablation_output/sensitivity_results.json') as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) Smooth window
    ax = axes[0]
    sw_data = data['results']['smooth_window_sweep']
    ws = [d['smooth_window'] for d in sw_data]
    agree = [d['agreement'] * 100 for d in sw_data]
    susp = [d['suspicious_rate'] * 100 for d in sw_data]

    ax.plot(ws, agree, 'o-', color=COLORS['blue'], linewidth=2, markersize=6, label='Agreement')
    ax.plot(ws, susp, 's--', color=COLORS['orange'], linewidth=2, markersize=5, label='Suspicious Rate')
    ax.axvline(x=8, color=COLORS['red'], linestyle=':', alpha=0.6, label='Default (w=8)')
    ax.set_xlabel('Smooth Window Size')
    ax.set_ylabel('Rate (%)')
    ax.set_title('(a) Smooth Window Sensitivity')
    ax.legend(fontsize=8)
    ax.set_ylim(48, 58)

    # (b) Yaw threshold
    ax = axes[1]
    yt_data = data['results']['yaw_threshold_sweep']
    yts = [d['yaw_threshold'] for d in yt_data]
    agree = [d['agreement'] * 100 for d in yt_data]
    susp = [d['suspicious_rate'] * 100 for d in yt_data]

    ax.plot(yts, agree, 'o-', color=COLORS['blue'], linewidth=2, markersize=6, label='Agreement')
    ax.plot(yts, susp, 's--', color=COLORS['orange'], linewidth=2, markersize=5, label='Suspicious Rate')
    ax.axvline(x=40, color=COLORS['red'], linestyle=':', alpha=0.6, label='Default (40°)')
    ax.set_xlabel('Yaw Threshold (°)')
    ax.set_ylabel('Rate (%)')
    ax.set_title('(b) Yaw Threshold Sensitivity')
    ax.legend(fontsize=8)
    ax.set_ylim(46, 58)

    # (c) Vote threshold
    ax = axes[2]
    vt_data = data['results']['vote_threshold_sweep']
    vts = [d['vote_threshold'] for d in vt_data]
    agree = [d['agreement'] * 100 for d in vt_data]
    susp = [d['suspicious_rate'] * 100 for d in vt_data]

    ax.plot(vts, agree, 'o-', color=COLORS['blue'], linewidth=2, markersize=6, label='Agreement')
    ax.plot(vts, susp, 's--', color=COLORS['orange'], linewidth=2, markersize=5, label='Suspicious Rate')
    ax.axvline(x=0.15, color=COLORS['red'], linestyle=':', alpha=0.6, label='Default (0.15)')
    ax.set_xlabel('Vote Threshold')
    ax.set_ylabel('Rate (%)')
    ax.set_title('(c) Vote Threshold Sensitivity')
    ax.legend(fontsize=8)
    ax.set_ylim(48, 58)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(output_dir / f'sensitivity_fig1_sweep.{ext}')
    plt.close()
    print('  [OK] sensitivity_fig1_sweep')


# ==================== 图5: 消融实验模块贡献度 ====================
def fig_ablation_contribution():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) 各模块去除后一致率下降
    ax = axes[0]
    modules = ['Pose Gate', 'Rules', 'Transformer', 'Smoothing']
    deltas = [31.0, 2.4, 0.5, -0.2]
    colors_bar = [COLORS['red'], COLORS['orange'], COLORS['blue'], COLORS['gray']]

    bars = ax.barh(range(len(modules)), deltas, color=colors_bar, height=0.5,
                   edgecolor='white', linewidth=1.5)
    ax.set_yticks(range(len(modules)))
    ax.set_yticklabels(modules, fontsize=11)
    ax.set_xlabel('Agreement Drop (pp)')
    ax.set_title('(a) Module Contribution (Agreement Drop When Removed)')
    ax.axvline(x=0, color='black', linewidth=0.8)

    for bar, v in zip(bars, deltas):
        x_pos = bar.get_width() + 0.5 if v > 0 else bar.get_width() - 2.5
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{v:+.1f}pp', va='center', fontsize=10, fontweight='bold')

    # (b) 基线 Shannon 熵对比
    ax = axes[1]
    methods = ['Threshold\nOnly', 'Rule\nOnly', 'Full\nSystem']
    entropies = [0.855, 0.857, 1.545]
    colors_e = [COLORS['gray'], COLORS['orange'], COLORS['green']]

    bars = ax.bar(range(len(methods)), entropies, color=colors_e, width=0.5,
                  edgecolor='white', linewidth=1.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel('Shannon Entropy')
    ax.set_title('(b) Behavior Diversity (Shannon Entropy)')
    ax.set_ylim(0, 2.0)

    for bar, v in zip(bars, entropies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{v:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(output_dir / f'ablation_fig5_contribution.{ext}')
    plt.close()
    print('  [OK] ablation_fig5_contribution')


# ==================== 图6: PAPE 对 QuickTurn 的提升效果 ====================
def fig_pape_quickturn():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # (a) QuickTurn F1 across configs
    ax = axes[0]
    configs = ['A0\nBaseline', 'A2\n+PAPE', 'A3\n+PAPE+BPCL', 'A4\nFull SBRN']
    qt_f1 = [0.200, 0.519, 0.375, 0.480]
    colors_qt = [COLORS['gray'], COLORS['blue'], COLORS['purple'], COLORS['green']]

    bars = ax.bar(range(len(configs)), qt_f1, color=colors_qt, width=0.55,
                  edgecolor='white', linewidth=1.5)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, fontsize=9)
    ax.set_ylabel('F1 Score')
    ax.set_title('(a) QuickTurn F1 Score')
    ax.set_ylim(0, 0.65)

    for bar, v in zip(bars, qt_f1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 标注提升
    ax.annotate('+160%', xy=(1, 0.519), xytext=(1.5, 0.58),
               fontsize=11, fontweight='bold', color=COLORS['red'],
               arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1.5))

    # (b) F1-Macro improvement breakdown
    ax = axes[1]
    stages = ['Baseline\n(CE)', '+PAPE', '+PAPE\n+BPCL', '+Aug\n(Full)']
    f1m = [0.636, 0.658, 0.585, 0.717]
    colors_f = [COLORS['gray'], COLORS['blue'], COLORS['purple'], COLORS['green']]

    ax.plot(range(len(stages)), f1m, 'o-', color=COLORS['blue'], linewidth=2, markersize=8, zorder=5)
    for i, (s, v) in enumerate(zip(stages, f1m)):
        ax.scatter(i, v, c=colors_f[i], s=100, zorder=6, edgecolors='black', linewidth=0.5)
        offset_y = 0.015 if i != 2 else -0.025
        ax.text(i, v + offset_y, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel('F1-Macro')
    ax.set_title('(b) F1-Macro Across Training Stages')
    ax.set_ylim(0.5, 0.78)

    # 标注BPCL下降
    ax.annotate('BPCL alone\nhurts F1', xy=(2, 0.585), xytext=(2.5, 0.54),
               fontsize=8, color=COLORS['red'], fontstyle='italic',
               arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1))

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(output_dir / f'sbrn_fig4_pape_effect.{ext}')
    plt.close()
    print('  [OK] sbrn_fig4_pape_effect')


# ==================== 图7: 三级决策框架消融对比 ====================
def fig_cascade_ablation():
    fig, ax = plt.subplots(figsize=(8, 5))

    configs = ['Full System', 'w/o Pose Gate', 'w/o Rules', 'w/o Transformer', 'w/o Smoothing']
    agreement = [50.3, 19.3, 47.9, 49.8, 50.5]
    entropy = [1.545, 1.068, 1.418, 1.466, 1.500]
    colors_c = [COLORS['green'], COLORS['red'], COLORS['orange'], COLORS['blue'], COLORS['gray']]

    scatter = ax.scatter(agreement, entropy, c=colors_c, s=200, zorder=5,
                        edgecolors='black', linewidth=0.8)

    for i, (a, e, n) in enumerate(zip(agreement, entropy, configs)):
        offset = (10, 5) if i != 1 else (10, -12)
        ax.annotate(n, (a, e), textcoords="offset points", xytext=offset,
                   fontsize=9, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    ax.set_xlabel('Agreement (%)', fontsize=12)
    ax.set_ylabel('Shannon Entropy', fontsize=12)
    ax.set_title('Ablation: Agreement vs Behavior Diversity', fontsize=13)

    # 标注理想区域
    ax.axhspan(1.4, 1.6, alpha=0.1, color=COLORS['green'])
    ax.axvspan(48, 52, alpha=0.1, color=COLORS['green'])
    ax.text(49, 1.57, 'Ideal\nRegion', fontsize=8, color=COLORS['green'],
            ha='center', fontstyle='italic')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(output_dir / f'ablation_fig6_cascade.{ext}')
    plt.close()
    print('  [OK] ablation_fig6_cascade')


if __name__ == '__main__':
    print("Generating final thesis figures...")
    fig_sbrn_ablation()
    fig_perclass_heatmap()
    fig_binary_comparison()
    fig_sensitivity()
    fig_ablation_contribution()
    fig_pape_quickturn()
    fig_cascade_ablation()
    print("\nAll figures generated successfully!")
