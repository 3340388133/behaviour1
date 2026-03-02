#!/usr/bin/env python3
"""
Step 1: 数据集目录结构构建
对应论文 Section III-A "Data Collection"

功能：
1. 扫描原始视频目录
2. 按照论文规范统一命名
3. 创建标准化 dataset_root 目录结构
4. 生成完整元数据文件
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import cv2
from tqdm import tqdm


# ============== 配置参数 ==============
RAW_VIDEO_DIR = "data/raw_videos"
VIDEO_OVERVIEW_JSON = "data/video_dataset_overview.json"
DATASET_ROOT = "dataset_root"

# 命名映射规则
VIEW_MAP = {
    "front": "F",
    "side": "S",
    "unknown": "U"
}

LIGHTING_MAP = {
    "indoor": "IN",
    "outdoor": "OUT",
    "unknown": "UNK"
}

SCENE_MAP = {
    "manual_queue": "MQ",
    "gate": "GT"
}


def load_video_overview(json_path: str) -> List[Dict]:
    """加载视频概览数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_standard_name(video_info: Dict, index: int) -> str:
    """
    生成标准化视频名称
    格式: {scene}_{view}_{lighting}_{index:03d}.mp4
    例如: MQ_S_IN_001.mp4
    """
    scene = SCENE_MAP.get(video_info['scene'], 'UNK')
    view = VIEW_MAP.get(video_info['camera_view'], 'U')
    lighting = LIGHTING_MAP.get(video_info['lighting'], 'UNK')

    return f"{scene}_{view}_{lighting}_{index:03d}.mp4"


def compute_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """计算文件 MD5 哈希值"""
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def create_dataset_structure(dataset_root: str) -> Dict[str, Path]:
    """
    创建标准化数据集目录结构

    dataset_root/
    ├── videos/              # 标准化命名的视频
    │   ├── raw/             # 原始视频软链接
    │   └── processed/       # 处理后的视频（预留）
    ├── frames/              # 抽帧图像（Step 2）
    ├── annotations/         # 标注文件
    │   ├── detection/       # 检测标注
    │   ├── tracking/        # 跟踪标注
    │   └── behavior/        # 行为标注
    ├── features/            # 提取的特征
    │   ├── pose/            # 姿态特征
    │   └── temporal/        # 时序特征
    ├── metadata/            # 元数据文件
    └── splits/              # 数据集划分
    """
    root = Path(dataset_root)

    dirs = {
        'root': root,
        'videos_raw': root / 'videos' / 'raw',
        'videos_processed': root / 'videos' / 'processed',
        'frames': root / 'frames',
        'annotations_detection': root / 'annotations' / 'detection',
        'annotations_tracking': root / 'annotations' / 'tracking',
        'annotations_behavior': root / 'annotations' / 'behavior',
        'features_pose': root / 'features' / 'pose',
        'features_temporal': root / 'features' / 'temporal',
        'metadata': root / 'metadata',
        'splits': root / 'splits'
    }

    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"创建目录: {path}")

    return dirs


def process_videos(
    video_overview: List[Dict],
    raw_video_dir: str,
    dirs: Dict[str, Path]
) -> List[Dict]:
    """
    处理视频：过滤、重命名、创建软链接
    返回处理后的元数据列表
    """
    processed_metadata = []
    raw_dir = Path(raw_video_dir)

    # 按场景和视角分组计数
    counters = {}

    # 只处理可用的视频
    usable_videos = [v for v in video_overview if v['usable_for_dataset']]

    print(f"\n处理 {len(usable_videos)} 个可用视频...")

    for video_info in tqdm(usable_videos, desc="处理视频"):
        original_name = video_info['video_id']
        original_path = raw_dir / original_name

        if not original_path.exists():
            print(f"警告: 视频不存在 {original_path}")
            continue

        # 生成分组键
        group_key = f"{video_info['scene']}_{video_info['camera_view']}_{video_info['lighting']}"
        counters[group_key] = counters.get(group_key, 0) + 1

        # 生成标准化名称
        standard_name = generate_standard_name(video_info, counters[group_key])

        # 创建软链接到 videos/raw
        link_path = dirs['videos_raw'] / standard_name

        # 如果链接已存在，先删除
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()

        # 创建相对路径的软链接
        rel_path = os.path.relpath(original_path.absolute(), link_path.parent.absolute())
        link_path.symlink_to(rel_path)

        # 计算文件哈希（用于数据完整性校验）
        file_hash = compute_file_hash(str(original_path))

        # 构建完整元数据
        metadata = {
            "video_id": standard_name.replace('.mp4', ''),
            "original_name": original_name,
            "standard_name": standard_name,
            "relative_path": f"videos/raw/{standard_name}",

            # 场景属性
            "scene_type": video_info['scene'],
            "camera_view": video_info['camera_view'],
            "lighting_condition": video_info['lighting'],
            "crowd_density": video_info['density'],

            # 技术参数
            "duration_sec": video_info['duration_sec'],
            "fps": video_info['fps'],
            "resolution": video_info['resolution'],
            "width": int(video_info['resolution'].split('x')[0]),
            "height": int(video_info['resolution'].split('x')[1]),
            "total_frames": video_info['total_frames'],
            "codec": video_info['codec'],

            # 数据集构建信息
            "usable_for_dataset": video_info['usable_for_dataset'],
            "recommended_tasks": video_info['recommended_tasks'],
            "risk_flags": video_info['risk_flags'],

            # 数据完整性
            "file_hash_md5": file_hash,
            "file_size_bytes": original_path.stat().st_size,

            # 时间戳
            "processed_at": datetime.now().isoformat()
        }

        processed_metadata.append(metadata)

    return processed_metadata


def generate_dataset_statistics(metadata: List[Dict]) -> Dict:
    """生成数据集统计信息"""
    stats = {
        "total_videos": len(metadata),
        "total_frames": sum(m['total_frames'] for m in metadata),
        "total_duration_sec": sum(m['duration_sec'] for m in metadata),
        "total_duration_hours": round(sum(m['duration_sec'] for m in metadata) / 3600, 2),

        "by_scene": {},
        "by_view": {},
        "by_lighting": {},
        "by_density": {},
        "by_resolution": {},

        "fps_range": {
            "min": min(m['fps'] for m in metadata),
            "max": max(m['fps'] for m in metadata)
        },
        "duration_range": {
            "min_sec": min(m['duration_sec'] for m in metadata),
            "max_sec": max(m['duration_sec'] for m in metadata)
        }
    }

    # 按各维度统计
    for m in metadata:
        # 场景
        scene = m['scene_type']
        if scene not in stats['by_scene']:
            stats['by_scene'][scene] = {"count": 0, "frames": 0, "duration_sec": 0}
        stats['by_scene'][scene]['count'] += 1
        stats['by_scene'][scene]['frames'] += m['total_frames']
        stats['by_scene'][scene]['duration_sec'] += m['duration_sec']

        # 视角
        view = m['camera_view']
        if view not in stats['by_view']:
            stats['by_view'][view] = {"count": 0, "frames": 0, "duration_sec": 0}
        stats['by_view'][view]['count'] += 1
        stats['by_view'][view]['frames'] += m['total_frames']
        stats['by_view'][view]['duration_sec'] += m['duration_sec']

        # 光照
        lighting = m['lighting_condition']
        if lighting not in stats['by_lighting']:
            stats['by_lighting'][lighting] = {"count": 0, "frames": 0, "duration_sec": 0}
        stats['by_lighting'][lighting]['count'] += 1
        stats['by_lighting'][lighting]['frames'] += m['total_frames']
        stats['by_lighting'][lighting]['duration_sec'] += m['duration_sec']

        # 密度
        density = m['crowd_density']
        if density not in stats['by_density']:
            stats['by_density'][density] = {"count": 0, "frames": 0, "duration_sec": 0}
        stats['by_density'][density]['count'] += 1
        stats['by_density'][density]['frames'] += m['total_frames']
        stats['by_density'][density]['duration_sec'] += m['duration_sec']

        # 分辨率
        res = m['resolution']
        if res not in stats['by_resolution']:
            stats['by_resolution'][res] = {"count": 0, "frames": 0}
        stats['by_resolution'][res]['count'] += 1
        stats['by_resolution'][res]['frames'] += m['total_frames']

    return stats


def save_metadata_files(
    metadata: List[Dict],
    stats: Dict,
    dirs: Dict[str, Path]
):
    """保存元数据文件"""
    metadata_dir = dirs['metadata']

    # 1. 保存完整视频元数据 (JSON)
    videos_json = metadata_dir / "videos.json"
    with open(videos_json, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"保存: {videos_json}")

    # 2. 保存数据集统计信息
    stats_json = metadata_dir / "dataset_statistics.json"
    with open(stats_json, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"保存: {stats_json}")

    # 3. 保存视频列表 (简化版，用于快速索引)
    video_list = metadata_dir / "video_list.txt"
    with open(video_list, 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write(f"{m['standard_name']}\n")
    print(f"保存: {video_list}")

    # 4. 保存 CSV 格式元数据
    csv_path = metadata_dir / "videos.csv"
    headers = [
        "video_id", "original_name", "standard_name", "scene_type",
        "camera_view", "lighting_condition", "crowd_density",
        "duration_sec", "fps", "resolution", "total_frames", "codec"
    ]
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(','.join(headers) + '\n')
        for m in metadata:
            row = [str(m.get(h, '')) for h in headers]
            f.write(','.join(row) + '\n')
    print(f"保存: {csv_path}")

    # 5. 保存数据集配置文件
    config = {
        "dataset_name": "QueueBehavior",
        "version": "1.0.0",
        "created_at": datetime.now().isoformat(),
        "description": "Queue behavior recognition dataset for security checkpoint scenarios",

        "data_collection": {
            "source": "Security checkpoint surveillance cameras",
            "scenarios": ["manual_queue", "gate"],
            "camera_views": ["front", "side"],
            "lighting_conditions": ["indoor", "outdoor"]
        },

        "directory_structure": {
            "videos/raw": "Original videos with standardized names",
            "videos/processed": "Processed videos (resized, normalized)",
            "frames": "Extracted frames",
            "annotations/detection": "Object detection annotations",
            "annotations/tracking": "Multi-object tracking annotations",
            "annotations/behavior": "Behavior recognition annotations",
            "features/pose": "Pose estimation features",
            "features/temporal": "Temporal features",
            "metadata": "Dataset metadata files",
            "splits": "Train/val/test splits"
        },

        "naming_convention": {
            "format": "{scene}_{view}_{lighting}_{index:03d}.mp4",
            "scene_codes": SCENE_MAP,
            "view_codes": VIEW_MAP,
            "lighting_codes": LIGHTING_MAP
        },

        "statistics": stats
    }

    config_json = dirs['root'] / "dataset_config.json"
    with open(config_json, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"保存: {config_json}")


def print_directory_tree(path: Path, prefix: str = "", max_depth: int = 3, current_depth: int = 0):
    """打印目录树"""
    if current_depth >= max_depth:
        return

    items = sorted(path.iterdir())
    dirs = [i for i in items if i.is_dir()]
    files = [i for i in items if i.is_file()]

    # 先打印文件
    for f in files[:5]:  # 最多显示5个文件
        print(f"{prefix}├── {f.name}")
    if len(files) > 5:
        print(f"{prefix}├── ... ({len(files) - 5} more files)")

    # 再打印目录
    for i, d in enumerate(dirs):
        is_last = (i == len(dirs) - 1) and len(files) == 0
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{d.name}/")

        new_prefix = prefix + ("    " if is_last else "│   ")
        print_directory_tree(d, new_prefix, max_depth, current_depth + 1)


def main():
    print("=" * 60)
    print("Step 1: 数据集目录结构构建")
    print("对应论文 Section III-A: Data Collection")
    print("=" * 60)

    # 1. 加载视频概览数据
    print("\n[1/4] 加载视频概览数据...")
    video_overview = load_video_overview(VIDEO_OVERVIEW_JSON)
    print(f"加载 {len(video_overview)} 个视频信息")

    # 2. 创建目录结构
    print("\n[2/4] 创建数据集目录结构...")
    dirs = create_dataset_structure(DATASET_ROOT)

    # 3. 处理视频
    print("\n[3/4] 处理视频文件...")
    metadata = process_videos(video_overview, RAW_VIDEO_DIR, dirs)

    # 4. 生成统计信息和保存元数据
    print("\n[4/4] 生成元数据文件...")
    stats = generate_dataset_statistics(metadata)
    save_metadata_files(metadata, stats, dirs)

    # 打印结果摘要
    print("\n" + "=" * 60)
    print("数据集构建完成!")
    print("=" * 60)

    print(f"\n📊 数据集统计:")
    print(f"  - 总视频数: {stats['total_videos']}")
    print(f"  - 总帧数: {stats['total_frames']:,}")
    print(f"  - 总时长: {stats['total_duration_hours']} 小时")

    print(f"\n📁 目录结构:")
    print_directory_tree(dirs['root'])

    print(f"\n📄 生成的元数据文件:")
    for f in (dirs['metadata']).iterdir():
        print(f"  - {f.name}")

    print(f"\n✅ 数据集根目录: {DATASET_ROOT}/")


if __name__ == "__main__":
    main()
