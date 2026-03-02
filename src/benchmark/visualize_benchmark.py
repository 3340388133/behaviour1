"""
姿态估计性能对比可视化
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict


def plot_mae_comparison(results_csv: str, output_path: str = None):
    """绘制 MAE 对比柱状图"""
    df = pd.read_csv(results_csv)

    methods = df['Method'].tolist()
    yaw_mae = df['Yaw MAE'].tolist()
    pitch_mae = df['Pitch MAE'].tolist()
    roll_mae = df['Roll MAE'].tolist()

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, yaw_mae, width, label='Yaw MAE', color='#FF6B6B')
    bars2 = ax.bar(x, pitch_mae, width, label='Pitch MAE', color='#4ECDC4')
    bars3 = ax.bar(x + width, roll_mae, width, label='Roll MAE', color='#45B7D1')

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('MAE (degrees)', fontsize=12)
    ax.set_title('Head Pose Estimation - MAE Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved to {output_path}")
    plt.show()


def plot_accuracy_comparison(results_csv: str, output_path: str = None):
    """绘制 5° 准确率对比柱状图"""
    df = pd.read_csv(results_csv)

    methods = df['Method'].tolist()
    yaw_acc = df['Yaw Acc@5°'].tolist()
    pitch_acc = df['Pitch Acc@5°'].tolist()
    roll_acc = df['Roll Acc@5°'].tolist()

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, yaw_acc, width, label='Yaw Acc@5°', color='#FF6B6B')
    bars2 = ax.bar(x, pitch_acc, width, label='Pitch Acc@5°', color='#4ECDC4')
    bars3 = ax.bar(x + width, roll_acc, width, label='Roll Acc@5°', color='#45B7D1')

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Head Pose Estimation - Accuracy@5° Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()


def plot_radar_chart(results_csv: str, output_path: str = None):
    """绘制雷达图对比"""
    df = pd.read_csv(results_csv)

    categories = ['Yaw MAE', 'Pitch MAE', 'Roll MAE', 'Yaw Acc', 'Pitch Acc', 'Roll Acc']
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    for idx, row in df.iterrows():
        # 归一化: MAE 越小越好 (取反), Acc 越大越好
        max_mae = max(df['Yaw MAE'].max(), df['Pitch MAE'].max(), df['Roll MAE'].max())
        values = [
            1 - row['Yaw MAE'] / max_mae,
            1 - row['Pitch MAE'] / max_mae,
            1 - row['Roll MAE'] / max_mae,
            row['Yaw Acc@5°'] / 100,
            row['Pitch Acc@5°'] / 100,
            row['Roll Acc@5°'] / 100
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=row['Method'],
                color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Head Pose Estimation - Overall Comparison', fontsize=14)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_inference_time(results_csv: str, output_path: str = None):
    """绘制推理时间对比"""
    df = pd.read_csv(results_csv)

    methods = df['Method'].tolist()
    times = df['Inference Time (ms)'].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(methods, times, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(methods)])

    ax.set_xlabel('Inference Time (ms)', fontsize=12)
    ax.set_title('Head Pose Estimation - Inference Time', fontsize=14)
    ax.grid(axis='x', alpha=0.3)

    for bar, time in zip(bars, times):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{time:.1f}ms', va='center', fontsize=10)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()


def generate_latex_table(results_csv: str, output_path: str = None) -> str:
    """生成 LaTeX 表格"""
    df = pd.read_csv(results_csv)

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Head Pose Estimation Methods Comparison}
\label{tab:pose_benchmark}
\begin{tabular}{l|ccc|ccc|c|c}
\hline
\textbf{Method} & \multicolumn{3}{c|}{\textbf{MAE (°)}} & \multicolumn{3}{c|}{\textbf{Acc@5° (\%)}} & \textbf{Avg MAE} & \textbf{Time} \\
 & Yaw & Pitch & Roll & Yaw & Pitch & Roll & (°) & (ms) \\
\hline
"""

    for _, row in df.iterrows():
        latex += f"{row['Method']} & {row['Yaw MAE']:.2f} & {row['Pitch MAE']:.2f} & {row['Roll MAE']:.2f} & "
        latex += f"{row['Yaw Acc@5°']:.1f} & {row['Pitch Acc@5°']:.1f} & {row['Roll Acc@5°']:.1f} & "
        latex += f"{row['Avg MAE']:.2f} & {row['Inference Time (ms)']:.1f} \\\\\n"

    latex += r"""\hline
\end{tabular}
\end{table}
"""

    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"LaTeX table saved to {output_path}")

    return latex


def generate_markdown_report(results_csv: str, output_path: str = None) -> str:
    """生成 Markdown 报告"""
    df = pd.read_csv(results_csv)

    report = """# 头部姿态估计方法性能对比报告

## 评估指标

| 方法 | Yaw MAE | Pitch MAE | Roll MAE | Yaw@5° | Pitch@5° | Roll@5° | 平均MAE | 平均Acc@5° | 推理时间 |
|------|---------|-----------|----------|--------|----------|---------|---------|------------|----------|
"""

    for _, row in df.iterrows():
        report += f"| {row['Method']} | {row['Yaw MAE']:.2f}° | {row['Pitch MAE']:.2f}° | {row['Roll MAE']:.2f}° | "
        report += f"{row['Yaw Acc@5°']:.1f}% | {row['Pitch Acc@5°']:.1f}% | {row['Roll Acc@5°']:.1f}% | "
        report += f"{row['Avg MAE']:.2f}° | {row['Avg Acc@5°']:.1f}% | {row['Inference Time (ms)']:.1f}ms |\n"

    # 找出最佳方法
    best_mae_idx = df['Avg MAE'].idxmin()
    best_acc_idx = df['Avg Acc@5°'].idxmax()
    fastest_idx = df['Inference Time (ms)'].idxmin()

    report += f"""
## 结论

- **最低平均MAE**: {df.loc[best_mae_idx, 'Method']} ({df.loc[best_mae_idx, 'Avg MAE']:.2f}°)
- **最高平均准确率**: {df.loc[best_acc_idx, 'Method']} ({df.loc[best_acc_idx, 'Avg Acc@5°']:.1f}%)
- **最快推理速度**: {df.loc[fastest_idx, 'Method']} ({df.loc[fastest_idx, 'Inference Time (ms)']:.1f}ms)

## 评估说明

- MAE (Mean Absolute Error): 平均绝对误差，越低越好
- Acc@5°: 预测误差在5°以内的样本比例，越高越好
- 推理时间: 单张图像的平均推理时间
"""

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to {output_path}")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("results_csv", help="Benchmark results CSV file")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    plot_mae_comparison(args.results_csv, str(output_dir / f"mae_comparison.{args.format}"))
    plot_accuracy_comparison(args.results_csv, str(output_dir / f"accuracy_comparison.{args.format}"))
    plot_radar_chart(args.results_csv, str(output_dir / f"radar_chart.{args.format}"))
    plot_inference_time(args.results_csv, str(output_dir / f"inference_time.{args.format}"))
    generate_latex_table(args.results_csv, str(output_dir / "benchmark_table.tex"))
    generate_markdown_report(args.results_csv, str(output_dir / "benchmark_report.md"))
