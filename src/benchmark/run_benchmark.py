#!/usr/bin/env python
"""
运行头部姿态估计方法性能对比评估

使用方法:
1. 准备 ground truth 标注文件 (CSV格式: face_path,yaw,pitch,roll)
2. 运行评估: python run_benchmark.py --gt-file gt.csv --methods whenet 6drepnet fsanet

如果没有 ground truth，先创建模板:
    python run_benchmark.py --create-template --output gt_template.csv
"""
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.pose_benchmark import (
    PoseBenchmark,
    WHENetEstimator,
    FSANetEstimator,
    SixDRepNetEstimator,
    HopeNetEstimator,
    create_gt_template
)
from benchmark.visualize_benchmark import (
    plot_mae_comparison,
    plot_accuracy_comparison,
    plot_radar_chart,
    plot_inference_time,
    generate_markdown_report
)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Head Pose Estimation Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 创建 ground truth 模板
  python run_benchmark.py --create-template --data-dir ../data --output gt_template.csv

  # 运行评估 (需要先准备好 ground truth)
  python run_benchmark.py --gt-file gt.csv --methods whenet 6drepnet

  # 生成可视化报告
  python run_benchmark.py --gt-file gt.csv --methods whenet 6drepnet --visualize
        """
    )

    parser.add_argument("--data-dir", type=str, default="../data",
                        help="数据目录 (默认: ../data)")
    parser.add_argument("--gt-file", type=str,
                        help="Ground truth CSV 文件路径")
    parser.add_argument("--output", type=str, default="benchmark_results.csv",
                        help="输出结果文件 (默认: benchmark_results.csv)")
    parser.add_argument("--output-dir", type=str, default="benchmark_output",
                        help="输出目录 (默认: benchmark_output)")
    parser.add_argument("--create-template", action="store_true",
                        help="创建 ground truth 标注模板")
    parser.add_argument("--methods", nargs="+",
                        default=["whenet"],
                        choices=["whenet", "fsanet", "6drepnet", "hopenet"],
                        help="要评估的方法 (默认: whenet)")
    parser.add_argument("--visualize", action="store_true",
                        help="生成可视化图表")
    parser.add_argument("--whenet-model", type=str,
                        help="WHENet 模型路径")
    parser.add_argument("--fsanet-model", type=str,
                        help="FSA-Net 模型路径")
    parser.add_argument("--6drepnet-model", type=str,
                        help="6DRepNet 模型路径")
    parser.add_argument("--hopenet-model", type=str,
                        help="HopeNet 模型路径")

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 创建模板模式
    if args.create_template:
        output_file = args.output if args.output != "benchmark_results.csv" else "gt_template.csv"
        create_gt_template(args.data_dir, output_file)
        print(f"\n请在 {output_file} 中填写 yaw, pitch, roll 的真实值")
        return

    # 检查 ground truth
    if not args.gt_file:
        print("错误: 需要提供 --gt-file 参数")
        print("如果没有 ground truth，请先运行: python run_benchmark.py --create-template")
        return

    if not Path(args.gt_file).exists():
        print(f"错误: Ground truth 文件不存在: {args.gt_file}")
        return

    # 创建评估器
    benchmark = PoseBenchmark(args.data_dir, args.gt_file)

    # 添加估计器
    print("加载模型...")
    if "whenet" in args.methods:
        try:
            benchmark.add_estimator(WHENetEstimator(args.whenet_model))
            print("  - WHENet 已加载")
        except Exception as e:
            print(f"  - WHENet 加载失败: {e}")

    if "fsanet" in args.methods:
        try:
            estimator = FSANetEstimator(args.fsanet_model)
            if hasattr(estimator, 'use_onnx') and estimator.use_onnx:
                benchmark.add_estimator(estimator)
                print("  - FSA-Net 已加载")
            else:
                print("  - FSA-Net 模型未找到，跳过")
        except Exception as e:
            print(f"  - FSA-Net 加载失败: {e}")

    if "6drepnet" in args.methods:
        try:
            estimator = SixDRepNetEstimator(getattr(args, '6drepnet_model', None))
            if estimator.model is not None:
                benchmark.add_estimator(estimator)
                print("  - 6DRepNet 已加载")
            else:
                print("  - 6DRepNet 模型未找到，跳过")
        except Exception as e:
            print(f"  - 6DRepNet 加载失败: {e}")

    if "hopenet" in args.methods:
        try:
            estimator = HopeNetEstimator(args.hopenet_model)
            if estimator.model is not None:
                benchmark.add_estimator(estimator)
                print("  - HopeNet 已加载")
            else:
                print("  - HopeNet 模型未找到，跳过")
        except Exception as e:
            print(f"  - HopeNet 加载失败: {e}")

    if not benchmark.estimators:
        print("错误: 没有可用的估计器")
        return

    # 运行评估
    print("\n开始评估...")
    results = benchmark.evaluate()

    # 打印结果
    benchmark.print_results(results)

    # 保存结果
    output_csv = output_dir / args.output
    benchmark.save_results(results, str(output_csv))

    # 生成可视化
    if args.visualize:
        print("\n生成可视化图表...")
        try:
            plot_mae_comparison(str(output_csv), str(output_dir / "mae_comparison.png"))
            plot_accuracy_comparison(str(output_csv), str(output_dir / "accuracy_comparison.png"))
            plot_radar_chart(str(output_csv), str(output_dir / "radar_chart.png"))
            plot_inference_time(str(output_csv), str(output_dir / "inference_time.png"))
            generate_markdown_report(str(output_csv), str(output_dir / "benchmark_report.md"))
            print(f"可视化结果已保存到 {output_dir}/")
        except Exception as e:
            print(f"可视化生成失败: {e}")


if __name__ == "__main__":
    main()
