#!/usr/bin/env python3
"""
Step 2: 帧提取与采样策略
对应论文 Section III-B "Frame Extraction and Sampling Strategy"

功能：
1. 按固定时间间隔抽帧（默认每秒1帧）
2. 支持多种采样策略：uniform（均匀）、keyframe（关键帧）、dense（密集）
3. 生成帧级别元数据
4. 保持时序信息用于行为识别
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse


# ============== 配置参数 ==============
DATASET_ROOT = "dataset_root"
METADATA_DIR = Path(DATASET_ROOT) / "metadata"
FRAMES_DIR = Path(DATASET_ROOT) / "frames"
VIDEOS_DIR = Path(DATASET_ROOT) / "videos" / "raw"

# 采样策略配置
SAMPLING_CONFIGS = {
    "uniform_1fps": {
        "description": "均匀采样，每秒1帧",
        "method": "uniform",
        "target_fps": 1.0
    },
    "uniform_2fps": {
        "description": "均匀采样，每秒2帧",
        "method": "uniform",
        "target_fps": 2.0
    },
    "uniform_5fps": {
        "description": "均匀采样，每秒5帧",
        "method": "uniform",
        "target_fps": 5.0
    },
    "dense_10fps": {
        "description": "密集采样，每秒10帧，用于行为识别",
        "method": "uniform",
        "target_fps": 10.0
    }
}

# 默认采样配置
DEFAULT_SAMPLING = "uniform_2fps"


class FrameExtractor:
    """帧提取器"""

    def __init__(
        self,
        videos_dir: Path,
        frames_dir: Path,
        sampling_config: str = DEFAULT_SAMPLING
    ):
        self.videos_dir = videos_dir
        self.frames_dir = frames_dir
        self.config = SAMPLING_CONFIGS[sampling_config]
        self.config_name = sampling_config

    def extract_frames_from_video(
        self,
        video_path: Path,
        video_metadata: Dict
    ) -> Dict[str, Any]:
        """
        从单个视频提取帧

        返回帧提取结果的元数据
        """
        video_id = video_metadata['video_id']
        original_fps = video_metadata['fps']
        total_frames = video_metadata['total_frames']
        duration = video_metadata['duration_sec']

        # 创建视频对应的帧目录
        video_frames_dir = self.frames_dir / video_id
        video_frames_dir.mkdir(parents=True, exist_ok=True)

        # 计算采样间隔
        target_fps = self.config['target_fps']
        frame_interval = int(original_fps / target_fps)
        if frame_interval < 1:
            frame_interval = 1

        # 打开视频
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"error": f"Cannot open video: {video_path}"}

        extracted_frames = []
        frame_idx = 0
        saved_count = 0

        # 计算预期帧数
        expected_frames = int(duration * target_fps)

        with tqdm(total=expected_frames, desc=f"  {video_id}", leave=False) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 按间隔采样
                if frame_idx % frame_interval == 0:
                    # 生成帧文件名: {video_id}_frame_{frame_number:06d}.jpg
                    frame_filename = f"{video_id}_frame_{saved_count:06d}.jpg"
                    frame_path = video_frames_dir / frame_filename

                    # 保存帧
                    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    # 记录帧元数据
                    timestamp_sec = frame_idx / original_fps
                    frame_info = {
                        "frame_id": f"{video_id}_frame_{saved_count:06d}",
                        "frame_filename": frame_filename,
                        "video_id": video_id,
                        "original_frame_idx": frame_idx,
                        "extracted_frame_idx": saved_count,
                        "timestamp_sec": round(timestamp_sec, 3),
                        "relative_path": f"frames/{video_id}/{frame_filename}"
                    }
                    extracted_frames.append(frame_info)
                    saved_count += 1
                    pbar.update(1)

                frame_idx += 1

        cap.release()

        # 返回提取结果
        result = {
            "video_id": video_id,
            "original_name": video_metadata['original_name'],
            "sampling_config": self.config_name,
            "sampling_method": self.config['method'],
            "target_fps": target_fps,
            "original_fps": original_fps,
            "frame_interval": frame_interval,
            "total_original_frames": total_frames,
            "extracted_frame_count": saved_count,
            "frames_dir": f"frames/{video_id}",
            "frames": extracted_frames
        }

        return result

    def extract_all_videos(self, video_metadata_list: List[Dict]) -> List[Dict]:
        """提取所有视频的帧"""
        all_results = []

        print(f"\n采样配置: {self.config_name}")
        print(f"  - 方法: {self.config['method']}")
        print(f"  - 目标FPS: {self.config['target_fps']}")
        print(f"  - 描述: {self.config['description']}")
        print()

        for video_meta in tqdm(video_metadata_list, desc="提取视频帧"):
            video_path = self.videos_dir / video_meta['standard_name']

            if not video_path.exists():
                print(f"警告: 视频不存在 {video_path}")
                continue

            result = self.extract_frames_from_video(video_path, video_meta)
            all_results.append(result)

            # 打印进度
            if 'error' not in result:
                print(f"  ✓ {video_meta['video_id']}: {result['extracted_frame_count']} 帧")

        return all_results


def generate_frame_statistics(extraction_results: List[Dict]) -> Dict:
    """生成帧提取统计信息"""
    total_frames = sum(r.get('extracted_frame_count', 0) for r in extraction_results)
    total_videos = len(extraction_results)

    stats = {
        "total_videos_processed": total_videos,
        "total_frames_extracted": total_frames,
        "average_frames_per_video": round(total_frames / total_videos, 2) if total_videos > 0 else 0,

        "by_video": [
            {
                "video_id": r['video_id'],
                "extracted_frames": r.get('extracted_frame_count', 0),
                "original_fps": r.get('original_fps', 0),
                "target_fps": r.get('target_fps', 0)
            }
            for r in extraction_results
        ],

        "frame_count_distribution": {
            "min": min(r.get('extracted_frame_count', 0) for r in extraction_results),
            "max": max(r.get('extracted_frame_count', 0) for r in extraction_results),
        }
    }

    return stats


def save_frame_metadata(
    extraction_results: List[Dict],
    stats: Dict,
    metadata_dir: Path,
    config_name: str
):
    """保存帧提取元数据"""

    # 1. 保存完整帧元数据
    frames_json = metadata_dir / "frames.json"
    with open(frames_json, 'w', encoding='utf-8') as f:
        json.dump(extraction_results, f, ensure_ascii=False, indent=2)
    print(f"保存: {frames_json}")

    # 2. 保存帧统计信息
    frame_stats_json = metadata_dir / "frame_statistics.json"
    stats['sampling_config'] = config_name
    stats['extraction_timestamp'] = datetime.now().isoformat()
    with open(frame_stats_json, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"保存: {frame_stats_json}")

    # 3. 生成帧列表文件（用于训练）
    frame_list_txt = metadata_dir / "frame_list.txt"
    with open(frame_list_txt, 'w', encoding='utf-8') as f:
        for result in extraction_results:
            for frame in result.get('frames', []):
                f.write(f"{frame['relative_path']}\n")
    print(f"保存: {frame_list_txt}")

    # 4. 生成帧-视频映射文件
    frame_video_map = metadata_dir / "frame_video_mapping.json"
    mapping = {}
    for result in extraction_results:
        video_id = result['video_id']
        for frame in result.get('frames', []):
            mapping[frame['frame_id']] = {
                "video_id": video_id,
                "timestamp_sec": frame['timestamp_sec'],
                "original_frame_idx": frame['original_frame_idx']
            }
    with open(frame_video_map, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"保存: {frame_video_map}")

    # 5. 生成时序帧序列文件（用于行为识别）
    temporal_sequences = metadata_dir / "temporal_sequences.json"
    sequences = []
    for result in extraction_results:
        video_id = result['video_id']
        frames = result.get('frames', [])

        # 按时间顺序组织帧
        sequence = {
            "video_id": video_id,
            "frame_count": len(frames),
            "fps": result.get('target_fps', 1.0),
            "duration_sec": frames[-1]['timestamp_sec'] if frames else 0,
            "frame_sequence": [f['frame_id'] for f in frames]
        }
        sequences.append(sequence)

    with open(temporal_sequences, 'w', encoding='utf-8') as f:
        json.dump(sequences, f, ensure_ascii=False, indent=2)
    print(f"保存: {temporal_sequences}")


def print_extraction_summary(stats: Dict, frames_dir: Path):
    """打印提取摘要"""
    print("\n" + "=" * 60)
    print("帧提取完成!")
    print("=" * 60)

    print(f"\n📊 提取统计:")
    print(f"  - 处理视频数: {stats['total_videos_processed']}")
    print(f"  - 提取总帧数: {stats['total_frames_extracted']:,}")
    print(f"  - 平均每视频帧数: {stats['average_frames_per_video']}")
    print(f"  - 帧数范围: {stats['frame_count_distribution']['min']} - {stats['frame_count_distribution']['max']}")

    print(f"\n📁 帧目录结构:")
    print(f"  {frames_dir}/")

    # 列出前几个视频目录
    video_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])[:5]
    for vd in video_dirs:
        frame_count = len(list(vd.glob("*.jpg")))
        print(f"    ├── {vd.name}/ ({frame_count} frames)")
    if len(list(frames_dir.iterdir())) > 5:
        print(f"    └── ... ({len(list(frames_dir.iterdir())) - 5} more)")


def main():
    parser = argparse.ArgumentParser(description="Step 2: 帧提取")
    parser.add_argument(
        "--sampling",
        type=str,
        default=DEFAULT_SAMPLING,
        choices=list(SAMPLING_CONFIGS.keys()),
        help=f"采样配置 (默认: {DEFAULT_SAMPLING})"
    )
    parser.add_argument(
        "--videos",
        type=str,
        nargs="+",
        default=None,
        help="指定要处理的视频ID列表（默认处理所有）"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Step 2: 帧提取与采样策略")
    print("对应论文 Section III-B: Frame Extraction and Sampling Strategy")
    print("=" * 60)

    # 1. 加载视频元数据
    print("\n[1/3] 加载视频元数据...")
    videos_json = METADATA_DIR / "videos.json"
    with open(videos_json, 'r', encoding='utf-8') as f:
        video_metadata_list = json.load(f)

    # 过滤指定视频
    if args.videos:
        video_metadata_list = [
            v for v in video_metadata_list
            if v['video_id'] in args.videos
        ]
        print(f"已选择 {len(video_metadata_list)} 个视频")
    else:
        print(f"加载 {len(video_metadata_list)} 个视频")

    # 2. 提取帧
    print("\n[2/3] 提取视频帧...")
    extractor = FrameExtractor(
        videos_dir=VIDEOS_DIR,
        frames_dir=FRAMES_DIR,
        sampling_config=args.sampling
    )
    extraction_results = extractor.extract_all_videos(video_metadata_list)

    # 3. 生成统计和保存元数据
    print("\n[3/3] 保存帧元数据...")
    stats = generate_frame_statistics(extraction_results)
    save_frame_metadata(extraction_results, stats, METADATA_DIR, args.sampling)

    # 打印摘要
    print_extraction_summary(stats, FRAMES_DIR)

    print(f"\n✅ 帧数据目录: {FRAMES_DIR}/")


if __name__ == "__main__":
    main()
