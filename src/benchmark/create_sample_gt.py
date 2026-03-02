#!/usr/bin/env python
"""
创建抽样标注集
从大量人脸图像中抽取代表性样本用于标注
"""
import pandas as pd
import numpy as np
from pathlib import Path
import shutil


def create_sampled_gt(
    data_dir: str,
    output_csv: str,
    sample_size: int = 500,
    copy_images: bool = True,
    output_image_dir: str = None
):
    """
    创建抽样标注集

    Args:
        data_dir: 数据目录
        output_csv: 输出 CSV 文件
        sample_size: 抽样数量
        copy_images: 是否复制图像到单独目录
        output_image_dir: 图像输出目录
    """
    data_dir = Path(data_dir)

    # 收集所有人脸图像
    face_files = list(data_dir.glob("faces/**/*.jpg"))
    print(f"找到 {len(face_files)} 张人脸图像")

    if len(face_files) == 0:
        print("错误: 未找到人脸图像")
        return

    # 随机抽样
    np.random.seed(42)
    sample_size = min(sample_size, len(face_files))
    sampled_files = np.random.choice(face_files, sample_size, replace=False)

    print(f"抽样 {sample_size} 张图像")

    # 创建标注模板
    data = []
    for i, f in enumerate(sampled_files):
        rel_path = f.relative_to(data_dir)
        data.append({
            'id': i + 1,
            'face_path': str(rel_path),
            'yaw': 0.0,
            'pitch': 0.0,
            'roll': 0.0
        })

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"标注模板已保存到: {output_csv}")

    # 复制图像
    if copy_images:
        if output_image_dir is None:
            output_image_dir = Path(output_csv).parent / "sample_faces"
        output_image_dir = Path(output_image_dir)
        output_image_dir.mkdir(parents=True, exist_ok=True)

        print(f"复制图像到: {output_image_dir}")
        for i, f in enumerate(sampled_files):
            dst = output_image_dir / f"{i+1:04d}.jpg"
            shutil.copy(f, dst)

        print(f"已复制 {len(sampled_files)} 张图像")

    print(f"\n下一步: 请在 {output_csv} 中填写每张图像的 yaw, pitch, roll 真实值")
    print("标注说明:")
    print("  - yaw:   左右转头角度, 范围 [-90, 90], 正值=向右转")
    print("  - pitch: 抬头低头角度, 范围 [-90, 90], 正值=抬头")
    print("  - roll:  歪头角度, 范围 [-90, 90], 正值=向右歪")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="创建抽样标注集")
    parser.add_argument("--data-dir", default="../data", help="数据目录")
    parser.add_argument("--output", default="sample_gt.csv", help="输出CSV")
    parser.add_argument("--sample-size", type=int, default=500, help="抽样数量")
    parser.add_argument("--copy-images", action="store_true", help="复制图像")

    args = parser.parse_args()

    create_sampled_gt(
        args.data_dir,
        args.output,
        args.sample_size,
        args.copy_images
    )
