#!/usr/bin/env python3
"""
完整可视化 Pipeline - 生成带有3D坐标轴、角度信息的视频
功能：人脸框 + 3D坐标轴 + pitch/yaw/roll角度显示
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
from tqdm import tqdm

from face_detector import RetinaFaceDetector
from head_pose import HeadPoseEstimator
from draw_utils import draw_detection_full


class VisualizationPipeline:
    """可视化 Pipeline"""

    def __init__(self, process_fps: float = 10.0, face_conf_threshold: float = 0.5):
        self.process_fps = process_fps

        print("初始化模块...")
        self.detector = RetinaFaceDetector(conf_threshold=face_conf_threshold)
        self.pose_estimator = HeadPoseEstimator()
        print("初始化完成")

    def process_video(self, video_path: str, output_video: str):
        """处理视频并生成可视化输出"""
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_interval = max(1, int(video_fps / self.process_fps))
        actual_fps = video_fps / frame_interval

        print(f"视频: {Path(video_path).name}")
        print(f"  分辨率: {width}x{height}")
        print(f"  原始帧率: {video_fps:.1f} fps")
        print(f"  处理帧率: {actual_fps:.1f} fps")
        print(f"  总帧数: {total_frames}")

        # 输出视频 - 使用 MJPG 编码 AVI 格式确保兼容性
        output_avi = output_video.replace('.mp4', '.avi')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(output_avi, fourcc, actual_fps, (width, height))
        print(f"  输出格式: AVI (MJPG)")

        frame_idx = 0
        pbar = tqdm(total=total_frames, desc="处理中")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                vis_frame = self._process_frame(frame, width, height)
                writer.write(vis_frame)

            frame_idx += 1
            pbar.update(1)

        cap.release()
        writer.release()
        pbar.close()

        print(f"\n输出视频已保存: {output_avi}")
        return output_avi

    def _process_frame(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """处理单帧：检测 -> 姿态估计 -> 可视化"""
        vis = frame.copy()

        # 1. 人脸检测
        detections = self.detector.detect(frame)

        # 2. 对每个检测结果进行姿态估计和可视化
        for idx, det in enumerate(detections):
            # 裁剪有效区域（处理边界情况）
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # 裁剪人脸
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            # 3. 姿态估计
            pose = self.pose_estimator.estimate(face_img)

            # 4. 绘制完整可视化（人脸框 + 3D坐标轴 + 角度）
            draw_detection_full(
                vis,
                bbox=(x1, y1, x2, y2),
                track_id=idx,
                yaw=pose.yaw,
                pitch=pose.pitch,
                roll=pose.roll
            )

        return vis


def main():
    import argparse

    parser = argparse.ArgumentParser(description='可视化 Pipeline')
    parser.add_argument('input', help='输入视频路径')
    parser.add_argument('--output', '-o', help='输出视频路径')
    parser.add_argument('--fps', type=float, default=10.0, help='处理帧率')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = args.output or str(input_path.parent / f"{input_path.stem}_vis.avi")

    pipeline = VisualizationPipeline(process_fps=args.fps)
    pipeline.process_video(str(input_path), output_path)


if __name__ == '__main__':
    main()
