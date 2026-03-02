#!/usr/bin/env python3
"""
DeepLabCut 头部行为标注项目设置脚本
自动创建项目、选择标注帧、配置关键点
"""

import os
import shutil
import random
import yaml
from pathlib import Path

# ============ 配置 ============
PROJECT_NAME = "head_behavior"
EXPERIMENTER = "labeler"
WORKING_DIR = "/root/behaviour/dataset_root"
FRAMES_DIR = "/root/behaviour/dataset_root/frames"

# 每个视频采样的帧数（总计约200帧用于标注）
FRAMES_PER_VIDEO = 25

# 头部关键点定义
BODYPARTS = [
    "head_top",      # 头顶
    "forehead",      # 额头中心
    "nose",          # 鼻尖
    "left_ear",      # 左耳
    "right_ear",     # 右耳
    "chin",          # 下巴
]

# 骨架连接（用于可视化）
SKELETON = [
    ["head_top", "forehead"],
    ["forehead", "nose"],
    ["nose", "chin"],
    ["left_ear", "forehead"],
    ["right_ear", "forehead"],
]

# ============ 创建项目目录结构 ============
def create_project():
    project_path = Path(WORKING_DIR) / f"{PROJECT_NAME}-{EXPERIMENTER}"

    # 清理旧项目
    if project_path.exists():
        print(f"删除旧项目: {project_path}")
        shutil.rmtree(project_path)

    # 创建目录结构
    dirs = [
        project_path,
        project_path / "labeled-data",
        project_path / "training-datasets",
        project_path / "dlc-models",
        project_path / "videos",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"创建目录: {d}")

    return project_path

# ============ 选择标注帧 ============
def select_frames_for_labeling(project_path):
    frames_dir = Path(FRAMES_DIR)
    labeled_data_dir = project_path / "labeled-data"

    video_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])

    selected_frames = []

    for video_dir in video_dirs:
        frames = sorted(list(video_dir.glob("*.jpg")))
        if not frames:
            continue

        # 均匀采样
        n_frames = len(frames)
        step = max(1, n_frames // FRAMES_PER_VIDEO)
        sampled = frames[::step][:FRAMES_PER_VIDEO]

        # 创建该视频的标注目录
        video_label_dir = labeled_data_dir / video_dir.name
        video_label_dir.mkdir(exist_ok=True)

        # 复制帧到标注目录
        for frame_path in sampled:
            dst = video_label_dir / frame_path.name
            shutil.copy2(frame_path, dst)
            selected_frames.append(str(dst))

        print(f"  {video_dir.name}: 选取 {len(sampled)}/{n_frames} 帧")

    print(f"\n总计选取 {len(selected_frames)} 帧用于标注")
    return selected_frames

# ============ 生成配置文件 ============
def create_config(project_path):
    config = {
        "Task": PROJECT_NAME,
        "scorer": EXPERIMENTER,
        "date": "2025-01-22",
        "multianimalproject": True,  # 多人场景

        # 关键点配置
        "individuals": ["ind1", "ind2", "ind3", "ind4", "ind5"],  # 最多5人同时出现
        "uniquebodyparts": [],
        "multianimalbodyparts": BODYPARTS,
        "bodyparts": BODYPARTS,
        "skeleton": SKELETON,

        # 训练配置
        "TrainingFraction": [0.8],
        "iteration": 0,
        "default_net_type": "dlcrnet_ms5",
        "default_augmenter": "multi-animal-imgaug",
        "snapshotindex": -1,
        "batch_size": 8,

        # 路径
        "project_path": str(project_path),
        "video_sets": {},

        # 裁剪配置（4K视频较大，可选裁剪）
        "cropping": False,

        # 关键点颜色（用于可视化）
        "colormap": "rainbow",
        "dotsize": 12,
        "alphavalue": 0.7,
        "pcutoff": 0.6,

        # 标注配置
        "move2corner": True,
        "corner2move2": [50, 50],
        "skeleton_color": "black",
    }

    # 添加视频集（使用帧目录作为视频源）
    frames_dir = Path(FRAMES_DIR)
    for video_dir in frames_dir.iterdir():
        if video_dir.is_dir():
            # 使用第一帧获取分辨率
            first_frame = next(video_dir.glob("*.jpg"), None)
            if first_frame:
                import cv2
                img = cv2.imread(str(first_frame))
                if img is not None:
                    h, w = img.shape[:2]
                    config["video_sets"][str(video_dir)] = {"crop": f"0, {w}, 0, {h}"}

    config_path = project_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"\n配置文件已创建: {config_path}")
    return config_path

# ============ 创建CollectedData CSV模板 ============
def create_annotation_template(project_path):
    """为每个视频目录创建空的标注CSV模板"""
    import pandas as pd

    labeled_data_dir = project_path / "labeled-data"
    scorer = EXPERIMENTER
    individuals = ["ind1", "ind2", "ind3", "ind4", "ind5"]

    for video_dir in labeled_data_dir.iterdir():
        if not video_dir.is_dir():
            continue

        frames = sorted(list(video_dir.glob("*.jpg")))
        if not frames:
            continue

        # 创建多级列索引
        columns = pd.MultiIndex.from_product(
            [[scorer], individuals, BODYPARTS, ["x", "y"]],
            names=["scorer", "individuals", "bodyparts", "coords"]
        )

        # 创建空DataFrame
        index = [f"labeled-data/{video_dir.name}/{f.name}" for f in frames]
        df = pd.DataFrame(index=index, columns=columns)

        # 保存
        csv_path = video_dir / f"CollectedData_{scorer}.csv"
        df.to_csv(csv_path)

        h5_path = video_dir / f"CollectedData_{scorer}.h5"
        df.to_hdf(h5_path, key="df_with_missing", mode="w")

    print(f"标注模板已创建")

# ============ 主函数 ============
def main():
    print("=" * 50)
    print("DeepLabCut 头部行为标注项目设置")
    print("=" * 50)

    # 1. 创建项目
    print("\n[1/4] 创建项目目录...")
    project_path = create_project()

    # 2. 选择标注帧
    print("\n[2/4] 选择标注帧...")
    selected_frames = select_frames_for_labeling(project_path)

    # 3. 创建配置
    print("\n[3/4] 创建配置文件...")
    config_path = create_config(project_path)

    # 4. 创建标注模板
    print("\n[4/4] 创建标注模板...")
    create_annotation_template(project_path)

    print("\n" + "=" * 50)
    print("项目设置完成！")
    print("=" * 50)
    print(f"\n项目路径: {project_path}")
    print(f"配置文件: {config_path}")
    print(f"待标注帧数: {len(selected_frames)}")
    print(f"\n头部关键点 ({len(BODYPARTS)} 个):")
    for i, bp in enumerate(BODYPARTS, 1):
        print(f"  {i}. {bp}")

    return str(config_path)

if __name__ == "__main__":
    config_path = main()
    print(f"\n\n下一步: 运行以下命令启动标注界面:")
    print(f"  python3 /root/behaviour/scripts/launch_labeling_gui.py")
