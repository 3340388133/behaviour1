#!/usr/bin/env python3
"""
Step 3: 人物检测与跟踪（StrongSORT / DeepOCSORT）
使用 boxmot 库，Re-ID 能力更强，支持遮挡后重识别

StrongSORT 特点：
- OSNet Re-ID 模型（专门训练的行人重识别）
- ECC 相机运动补偿
- NSA Kalman 滤波
- 遮挡后能重新识别同一人
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
import torch

# ============== 配置参数 ==============
DATA_ROOT = Path("data")
RAW_VIDEOS_DIR = DATA_ROOT / "raw_videos"
TRACKED_OUTPUT_DIR = DATA_ROOT / "tracked_output"


class StrongSORTTracker:
    """基于 YOLOv8 + StrongSORT/DeepOCSORT 的人物跟踪器"""

    TRACKER_MAP = {
        "strongsort": "strongsort",
        "deepocsort": "deepocsort",
        "ocsort": "ocsort",
        "botsort": "botsort",
        "bytetrack": "bytetrack",
    }

    def __init__(
        self,
        detector: str = "yolov8m.pt",
        tracker_type: str = "strongsort",
        reid_model: str = "osnet_x0_25_msmt17.pt",
        device: str = "cuda:0",
        conf_thresh: float = 0.5,
    ):
        self.detector_name = detector
        self.tracker_type = tracker_type
        self.reid_model = reid_model
        self.device = device
        self.conf_thresh = conf_thresh

        self.model = None
        self.tracker = None
        self._init_models()

    def _init_models(self):
        """初始化检测器和跟踪器"""
        from ultralytics import YOLO

        print(f"加载检测器: {self.detector_name}")
        self.model = YOLO(self.detector_name)

        # 初始化 boxmot 跟踪器
        print(f"加载跟踪器: {self.tracker_type} (Re-ID: {self.reid_model})")
        self._init_tracker()

    def _init_tracker(self):
        """初始化 boxmot 跟踪器"""
        from boxmot import StrongSORT, DeepOCSORT, OCSORT, BoTSORT, BYTETracker

        tracker_cls = {
            "strongsort": StrongSORT,
            "deepocsort": DeepOCSORT,
            "ocsort": OCSORT,
            "botsort": BoTSORT,
            "bytetrack": BYTETracker,
        }

        cls = tracker_cls.get(self.tracker_type)
        if cls is None:
            raise ValueError(f"未知跟踪器: {self.tracker_type}")

        # StrongSORT 和 DeepOCSORT 需要 Re-ID 模型
        if self.tracker_type in ["strongsort", "deepocsort", "botsort"]:
            self.tracker = cls(
                model_weights=Path(self.reid_model),
                device=self.device,
                fp16=False,
            )
        else:
            self.tracker = cls()

    def process_video(
        self,
        video_path: Path,
        output_dir: Path,
        save_video: bool = True,
        save_crops: bool = True,
        crop_size: Tuple[int, int] = (128, 256),
    ) -> Dict:
        """处理单个视频"""
        video_name = video_path.stem
        video_output_dir = output_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)

        # 获取视频信息
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

        # 视频写入器
        video_writer = None
        if save_video:
            out_video_path = video_output_dir / f"{video_name}_tracked.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(out_video_path), fourcc, fps, (width, height)
            )

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps

            # YOLO 检测
            results = self.model(
                frame,
                conf=self.conf_thresh,
                classes=[0],  # person only
                device=self.device,
                verbose=False,
            )

            # 获取检测结果
            dets = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                if len(boxes) > 0:
                    # [x1, y1, x2, y2, conf, cls]
                    xyxy = boxes.xyxy.cpu().numpy()
                    conf = boxes.conf.cpu().numpy()
                    cls = boxes.cls.cpu().numpy()
                    dets = np.hstack([xyxy, conf.reshape(-1, 1), cls.reshape(-1, 1)])

            # 跟踪更新
            if len(dets) > 0:
                tracks = self.tracker.update(dets, frame)
            else:
                tracks = self.tracker.update(np.empty((0, 6)), frame)

            # 处理跟踪结果
            # tracks: [x1, y1, x2, y2, track_id, conf, cls, ...]
            if len(tracks) > 0:
                for track in tracks:
                    x1, y1, x2, y2 = map(int, track[:4])
                    track_id = int(track[4])
                    conf = float(track[5]) if len(track) > 5 else 1.0

                    # 记录数据
                    track_data[track_id]["frames"].append(frame_idx)
                    track_data[track_id]["bboxes"].append([x1, y1, x2, y2])
                    track_data[track_id]["confidences"].append(conf)
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
            if frame_idx % 100 == 0:
                print(f"    处理: {frame_idx}/{total_frames} 帧", end='\r')

        cap.release()
        if video_writer:
            video_writer.release()

        print(f"    处理: {frame_idx}/{total_frames} 帧 - 完成")

        # 重置跟踪器状态（为下一个视频准备）
        self._init_tracker()

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

    def _save_crop(self, frame, x1, y1, x2, y2, track_id, frame_idx, crops_dir, crop_size):
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

    def _generate_metadata(self, video_name, video_path, track_data,
                           fps, width, height, total_frames, output_dir) -> Dict:
        """生成元数据"""
        tracks_info = []
        for track_id, data in track_data.items():
            if len(data["frames"]) == 0:
                continue

            # 计算轨迹连续性（检测帧数占比）
            first_frame = data["frames"][0]
            last_frame = data["frames"][-1]
            span = last_frame - first_frame + 1
            continuity = len(data["frames"]) / span if span > 0 else 1.0

            track_info = {
                "track_id": int(track_id),
                "frame_count": len(data["frames"]),
                "first_frame": first_frame,
                "last_frame": last_frame,
                "duration_sec": round((last_frame - first_frame) / fps, 2),
                "continuity": round(continuity, 3),  # 连续性指标
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
            "tracking_config": {
                "detector": self.detector_name,
                "tracker": self.tracker_type,
                "reid_model": self.reid_model,
                "conf_thresh": self.conf_thresh,
                "device": self.device,
            },
            "statistics": {
                "total_tracks": len(tracks_info),
                "total_detections": sum(t["frame_count"] for t in tracks_info),
                "avg_track_length": round(
                    sum(t["frame_count"] for t in tracks_info) / len(tracks_info), 1
                ) if tracks_info else 0,
                "avg_continuity": round(
                    sum(t["continuity"] for t in tracks_info) / len(tracks_info), 3
                ) if tracks_info else 0,
            },
            "tracks": tracks_info,
            "processed_at": datetime.now().isoformat(),
        }

        return metadata


def main():
    parser = argparse.ArgumentParser(description="Step 3: 人物跟踪（StrongSORT/DeepOCSORT）")
    parser.add_argument("--input", "-i", type=str, default=str(RAW_VIDEOS_DIR))
    parser.add_argument("--output", "-o", type=str, default=str(TRACKED_OUTPUT_DIR))
    parser.add_argument("--detector", type=str, default="yolov8m.pt")
    parser.add_argument(
        "--tracker", type=str, default="strongsort",
        choices=["strongsort", "deepocsort", "ocsort", "botsort", "bytetrack"],
        help="跟踪器类型 (推荐: strongsort 或 deepocsort)"
    )
    parser.add_argument("--reid", type=str, default="osnet_x0_25_msmt17.pt",
                        help="Re-ID 模型")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-crops", action="store_true")
    parser.add_argument("--video", type=str, default=None, help="处理单个视频")

    args = parser.parse_args()

    print("=" * 60)
    print(f"Step 3: 人物跟踪（YOLOv8 + {args.tracker.upper()}）")
    print("=" * 60)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = StrongSORTTracker(
        detector=args.detector,
        tracker_type=args.tracker,
        reid_model=args.reid,
        device=args.device,
        conf_thresh=args.conf,
    )

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
        print(f"\n完成! 检测到 {result['statistics']['total_tracks']} 个人物轨迹")
        print(f"平均连续性: {result['statistics']['avg_continuity']:.1%}")
        print(f"输出目录: {result['output_dir']}")
    else:
        # 递归查找所有视频
        video_extensions = ['.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV']
        videos = []
        for ext in video_extensions:
            videos.extend(input_dir.rglob(f'*{ext}'))

        if not videos:
            print(f"未找到视频: {input_dir}")
            return

        print(f"\n找到 {len(videos)} 个视频")

        all_results = []
        for video_path in sorted(videos):
            print(f"\n处理: {video_path.name}")
            result = tracker.process_video(
                video_path, output_dir,
                save_video=not args.no_video,
                save_crops=not args.no_crops,
            )
            all_results.append(result)
            print(f"  轨迹数: {result['statistics']['total_tracks']}, "
                  f"连续性: {result['statistics']['avg_continuity']:.1%}")

        # 保存汇总
        summary = {
            "total_videos": len(all_results),
            "total_tracks": sum(r["statistics"]["total_tracks"] for r in all_results),
            "avg_continuity": round(
                sum(r["statistics"]["avg_continuity"] for r in all_results) / len(all_results), 3
            ) if all_results else 0,
            "config": {
                "detector": args.detector,
                "tracker": args.tracker,
                "reid_model": args.reid,
            },
            "videos": [
                {
                    "name": r["video_name"],
                    "tracks": r["statistics"]["total_tracks"],
                    "continuity": r["statistics"]["avg_continuity"],
                }
                for r in all_results
            ],
        }
        summary_path = output_dir / "tracking_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*60}")
        print(f"全部完成!")
        print(f"  总轨迹数: {summary['total_tracks']}")
        print(f"  平均连续性: {summary['avg_continuity']:.1%}")
        print(f"  输出目录: {output_dir}")


if __name__ == "__main__":
    main()
