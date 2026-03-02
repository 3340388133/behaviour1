#!/usr/bin/env python3
"""
从已有检测和姿态数据生成可视化视频

功能：使用已保存的detection和pose数据，在视频帧上绘制:
- 人脸边界框
- 3D坐标轴
- yaw/pitch/roll角度
- Track ID
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import json
import numpy as np
from tqdm import tqdm
from draw_utils import draw_detection_full


def visualize_video(video_id: str, output_path: str = None):
    """使用已有数据可视化视频

    Args:
        video_id: 视频ID（目录名）
        output_path: 输出视频路径
    """
    data_dir = Path(__file__).parent.parent / "data"
    frames_dir = data_dir / "frames" / video_id
    detection_dir = data_dir / "detection" / video_id
    pose_dir = data_dir / "pose" / video_id

    # 加载检测数据
    with open(detection_dir / "detections.json", 'r', encoding='utf-8') as f:
        det_data = json.load(f)

    # 加载姿态数据
    with open(pose_dir / "pose.json", 'r', encoding='utf-8') as f:
        pose_data = json.load(f)

    # 构建 frame_idx -> pose 映射
    # pose_data 按 track 组织，需要重新组织为 frame -> detections
    frame_poses = {}
    for track in pose_data.get('tracks', []):
        track_id = track['track_id']
        for pose in track.get('poses', []):
            frame_idx = pose['frame_idx']
            if frame_idx not in frame_poses:
                frame_poses[frame_idx] = {}
            frame_poses[frame_idx][track_id] = {
                'yaw': pose['yaw'],
                'pitch': pose['pitch'],
                'roll': pose['roll']
            }

    # 获取帧信息
    fps = det_data.get('fps', 10.0)
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))

    if not frame_files:
        print(f"No frames found in {frames_dir}")
        return

    # 读取第一帧获取尺寸
    sample_frame = cv2.imread(str(frame_files[0]))
    height, width = sample_frame.shape[:2]

    # 输出视频
    if output_path is None:
        output_path = str(data_dir / f"{video_id}_vis.avi")

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"可视化 {video_id}")
    print(f"  帧数: {len(frame_files)}")
    print(f"  帧率: {fps} fps")
    print(f"  输出: {output_path}")

    # 按帧处理
    det_by_frame = {f['frame_idx']: f['detections'] for f in det_data['frames']}

    for frame_file in tqdm(frame_files, desc="生成可视化视频"):
        frame_idx = int(frame_file.stem.split("_")[1])
        frame = cv2.imread(str(frame_file))

        if frame is None:
            continue

        # 获取该帧的检测
        detections = det_by_frame.get(frame_idx, [])

        for det_idx, det in enumerate(detections):
            bbox = det['bbox']
            x1, y1, x2, y2 = [int(v) for v in bbox]

            # 获取姿态（如果有）
            pose_info = frame_poses.get(frame_idx, {}).get(det_idx, None)

            if pose_info:
                yaw = pose_info['yaw']
                pitch = pose_info['pitch']
                roll = pose_info['roll']
            else:
                # 没有姿态数据，使用默认值
                yaw, pitch, roll = 0, 0, 0

            # 绘制完整可视化
            draw_detection_full(
                frame,
                bbox=(x1, y1, x2, y2),
                track_id=det_idx,
                yaw=yaw,
                pitch=pitch,
                roll=roll
            )

        writer.write(frame)

    writer.release()
    print(f"\n可视化视频已保存: {output_path}")
    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description='从已有数据生成可视化视频')
    parser.add_argument('video_id', help='视频ID（data/frames下的目录名）')
    parser.add_argument('--output', '-o', help='输出视频路径')
    args = parser.parse_args()

    visualize_video(args.video_id, args.output)


if __name__ == '__main__':
    main()
