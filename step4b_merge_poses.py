#!/usr/bin/env python3
"""
Step 4b: 整合所有姿态数据为统一格式

输出格式适配 TAHPNet 训练
"""

import json
from pathlib import Path
from typing import Dict
from datetime import datetime

DATA_ROOT = Path("data")
POSE_OUTPUT_DIR = DATA_ROOT / "pose_output"
OUTPUT_PATH = POSE_OUTPUT_DIR / "all_poses.json"


def merge_pose_files():
    """整合所有姿态 JSON 文件"""

    all_data = {}

    # 查找所有姿态文件
    pose_files = list(POSE_OUTPUT_DIR.glob("*_poses.json"))

    print(f"找到 {len(pose_files)} 个姿态文件")

    for pose_file in sorted(pose_files):
        video_name = pose_file.stem.replace("_poses", "")
        print(f"  处理: {video_name}")

        with open(pose_file, 'r') as f:
            data = json.load(f)

        # 提取轨迹数据
        tracks = data.get('tracks', {})

        # 统计
        total_poses = sum(len(t.get('poses', [])) for t in tracks.values())
        print(f"    轨迹数: {len(tracks)}, 姿态帧数: {total_poses}")

        all_data[video_name] = {
            'tracks': tracks,
            'video_info': data.get('video_name', video_name),
            'total_tracks': len(tracks),
            'total_poses': total_poses,
        }

    # 汇总统计
    summary = {
        'total_videos': len(all_data),
        'total_tracks': sum(v['total_tracks'] for v in all_data.values()),
        'total_poses': sum(v['total_poses'] for v in all_data.values()),
        'videos': list(all_data.keys()),
        'merged_at': datetime.now().isoformat(),
    }

    output = {
        'summary': summary,
        **all_data,
    }

    # 保存
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n整合完成!")
    print(f"  总视频数: {summary['total_videos']}")
    print(f"  总轨迹数: {summary['total_tracks']}")
    print(f"  总姿态帧: {summary['total_poses']}")
    print(f"  保存到: {OUTPUT_PATH}")

    return output


if __name__ == "__main__":
    merge_pose_files()
