#!/usr/bin/env python3
"""
Step 3: 人物检测与跟踪（带 Re-ID）
使用 YOLOv8 内置跟踪器（BoT-SORT / ByteTrack）

BoT-SORT 特点：
- 融合了 Re-ID 外观特征
- Camera Motion Compensation
- 支持遮挡后重识别
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict
import argparse

# ============== 配置参数 ==============
DATASET_ROOT = Path("dataset_root")
DATA_ROOT = Path("data")
RAW_VIDEOS_DIR = DATA_ROOT / "raw_videos"
TRACKED_OUTPUT_DIR = DATA_ROOT / "tracked_output"

# 跟踪配置
TRACKING_CONFIG = {
    "detector": "yolov8m.pt",
    "tracker": "botsort.yaml",      # botsort.yaml 带 Re-ID，bytetrack.yaml 不带
    "conf_thresh": 0.5,
    "iou_thresh": 0.5,
    "device": "0",                   # "0" for cuda:0, "cpu" for cpu
}


class PersonTracker:
    """基于 YOLOv8 + BoT-SORT 的人物跟踪器"""

    def __init__(self, config: dict = None):
        self.config = config or TRACKING_CONFIG
        self.model = None
        self._init_model()

    def _init_model(self):
        """初始化 YOLO 模型"""
        try:
            from ultralytics import YOLO
        except ImportError:
            print("请先安装: pip install ultralytics")
            raise

        detector = self.config["detector"]
        print(f"加载检测器: {detector}")
        self.model = YOLO(detector)

    def process_video(
        self,
        video_path: Path,
        output_dir: Path,
        save_video: bool = True,
        save_crops: bool = True,
        crop_size: Tuple[int, int] = (128, 256),
        show_progress: bool = True,
    ) -> Dict:
        """
        处理单个视频
        """
        video_name = video_path.stem
        video_output_dir = output_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)

        # 获取视频信息
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        print(f"  视频信息: {width}x{height}, {fps:.1f}fps, {total_frames} 帧")

        # 准备裁剪目录
        if save_crops:
            crops_dir = video_output_dir / "crops"
            crops_dir.mkdir(exist_ok=True)

        # 跟踪结果存储
        track_data = defaultdict(lambda: {
            "frames": [],
            "bboxes": [],
            "confidences": [],
            "timestamps": [],
        })

        # 使用 YOLO 内置跟踪
        results = self.model.track(
            source=str(video_path),
            tracker=self.config["tracker"],
            conf=self.config["conf_thresh"],
            iou=self.config["iou_thresh"],
            classes=[0],  # person only
            device=self.config["device"],
            stream=True,
            persist=True,
            verbose=False,
        )

        # 视频写入器
        video_writer = None
        if save_video:
            out_video_path = video_output_dir / f"{video_name}_tracked.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(out_video_path), fourcc, fps, (width, height)
            )

        frame_idx = 0
        for result in results:
            frame = result.orig_img
            timestamp = frame_idx / fps

            # 处理跟踪结果
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()

                for box, track_id, conf in zip(boxes, track_ids, confs):
                    x1, y1, x2, y2 = map(int, box)
                    track_id = int(track_id)

                    # 记录数据
                    track_data[track_id]["frames"].append(frame_idx)
                    track_data[track_id]["bboxes"].append([x1, y1, x2, y2])
                    track_data[track_id]["confidences"].append(float(conf))
                    track_data[track_id]["timestamps"].append(round(timestamp, 3))

                    # 保存裁剪
                    if save_crops:
                        self._save_crop(
                            frame, x1, y1, x2, y2, track_id,
                            frame_idx, crops_dir, crop_size
                        )

                    # 绘制
                    if save_video:
                        color = self._get_color(track_id)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"ID:{track_id}"
                        (tw, th), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                        )
                        cv2.rectangle(
                            frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1
                        )
                        cv2.putText(
                            frame, label, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                        )

            if save_video:
                video_writer.write(frame)

            frame_idx += 1
            if show_progress and frame_idx % 100 == 0:
                print(f"    处理: {frame_idx}/{total_frames} 帧", end='\r')

        if video_writer:
            video_writer.release()

        print(f"    处理: {frame_idx}/{total_frames} 帧 - 完成")

        # 生成元数据
        metadata = self._generate_metadata(
            video_name, video_path, track_data,
            fps, width, height, total_frames, video_output_dir
        )

        # 保存元数据
        meta_path = video_output_dir / "tracking_result.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return metadata

    def _save_crop(
        self, frame, x1, y1, x2, y2, track_id,
        frame_idx, crops_dir, crop_size
    ):
        """保存裁剪图像"""
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        track_dir = crops_dir / f"track_{track_id:04d}"
        track_dir.mkdir(exist_ok=True)

        crop_resized = cv2.resize(crop, crop_size)
        crop_path = track_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(crop_path), crop_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

    def _get_color(self, track_id: int) -> Tuple[int, int, int]:
        """为每个 track_id 生成固定颜色"""
        np.random.seed(track_id * 7)
        return tuple(np.random.randint(50, 255, 3).tolist())

    def _generate_metadata(
        self, video_name, video_path, track_data,
        fps, width, height, total_frames, output_dir
    ) -> Dict:
        """生成元数据"""
        tracks_info = []
        for track_id, data in track_data.items():
            if len(data["frames"]) == 0:
                continue
            track_info = {
                "track_id": int(track_id),
                "frame_count": len(data["frames"]),
                "first_frame": data["frames"][0],
                "last_frame": data["frames"][-1],
                "duration_sec": round(
                    (data["frames"][-1] - data["frames"][0]) / fps, 2
                ),
                "avg_confidence": round(
                    sum(data["confidences"]) / len(data["confidences"]), 3
                ),
                "frames": data["frames"],
                "bboxes": data["bboxes"],
                "timestamps": data["timestamps"],
            }
            tracks_info.append(track_info)

        tracks_info.sort(key=lambda x: x["track_id"])

        metadata = {
            "video_name": video_name,
            "video_path": str(video_path),
            "output_dir": str(output_dir),
            "video_info": {
                "fps": fps,
                "width": width,
                "height": height,
                "total_frames": total_frames,
                "duration_sec": round(total_frames / fps, 2),
            },
            "tracking_config": self.config,
            "statistics": {
                "total_tracks": len(tracks_info),
                "total_detections": sum(t["frame_count"] for t in tracks_info),
                "avg_track_length": round(
                    sum(t["frame_count"] for t in tracks_info) / len(tracks_info), 1
                ) if tracks_info else 0,
            },
            "tracks": tracks_info,
            "processed_at": datetime.now().isoformat(),
        }

        return metadata


def main():
    parser = argparse.ArgumentParser(description="Step 3: 人物检测与跟踪")
    parser.add_argument("--input", "-i", type=str, default=str(RAW_VIDEOS_DIR))
    parser.add_argument("--output", "-o", type=str, default=str(TRACKED_OUTPUT_DIR))
    parser.add_argument("--detector", type=str, default="yolov8m.pt")
    parser.add_argument(
        "--tracker", type=str, default="botsort.yaml",
        help="Tracker config: botsort.yaml, bytetrack.yaml, or custom path"
    )
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-crops", action="store_true")
    parser.add_argument("--video", type=str, default=None, help="处理单个视频")
    parser.add_argument("--conf", type=float, default=0.5)

    args = parser.parse_args()

    print("=" * 60)
    print("Step 3: 人物检测与跟踪（YOLOv8 + BoT-SORT）")
    print("=" * 60)

    config = {
        "detector": args.detector,
        "tracker": args.tracker,
        "conf_thresh": args.conf,
        "iou_thresh": 0.5,
        "device": args.device,
    }

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = PersonTracker(config)

    if args.video:
        # 单个视频
        video_path = Path(args.video)
        if not video_path.exists():
            video_path = input_dir / args.video
        if not video_path.exists():
            print(f"视频不存在: {video_path}")
            return

        print(f"\n处理: {video_path.name}")
        result = tracker.process_video(
            video_path, output_dir,
            save_video=not args.no_video,
            save_crops=not args.no_crops,
        )
        print(f"\n完成! 检测到 {result['statistics']['total_tracks']} 个人物")
        print(f"输出目录: {result['output_dir']}")
    else:
        # 所有视频（递归查找）
        video_extensions = ['.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV']
        videos = []
        for ext in video_extensions:
            videos.extend(input_dir.rglob(f'*{ext}'))

        if not videos:
            print(f"未找到视频: {input_dir}")
            return

        print(f"\n找到 {len(videos)} 个视频")

        all_results = []
        for video_path in videos:
            print(f"\n处理: {video_path.name}")
            result = tracker.process_video(
                video_path, output_dir,
                save_video=not args.no_video,
                save_crops=not args.no_crops,
            )
            all_results.append(result)
            print(f"  检测到 {result['statistics']['total_tracks']} 个人物")

        # 保存汇总
        summary = {
            "total_videos": len(all_results),
            "total_tracks": sum(r["statistics"]["total_tracks"] for r in all_results),
            "config": config,
            "videos": [
                {"name": r["video_name"], "tracks": r["statistics"]["total_tracks"]}
                for r in all_results
            ],
        }
        summary_path = output_dir / "tracking_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n全部完成! 共 {summary['total_tracks']} 个人物轨迹")


if __name__ == "__main__":
    main()
