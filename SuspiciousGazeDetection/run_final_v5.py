#!/usr/bin/env python3
"""
可疑张望行为检测系统 V5
- 滑动窗口分析（最近3秒）
- 基于yaw范围和方向变化
- 显示头部框和角度
"""

import os
import cv2
import json
import numpy as np
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class Config:
    DATA_ROOT = "/root/autodl-tmp/behaviour/data"
    OUTPUT_DIR = "/root/autodl-tmp/behaviour/SuspiciousGazeDetection/output"

    VIDEOS = {
        "front": {
            "videos": ["1.14zz-1", "1.14zz-3", "1.14zz-4"],
            "yaw_offset": 0.0,
        },
        "side": {
            "videos": ["MVI_4537", "MVI_4538"],
            "yaw_offset": 90.0,
        }
    }

    FPS = 30

    # V5: 滑动窗口3秒内的判定
    YAW_RANGE_THRESHOLD = 100.0  # yaw范围>100度
    DIRECTION_CHANGE_THRESHOLD = 6  # 方向变化>6次
    WINDOW_SIZE = 90  # 3秒窗口
    MIN_TRACK_FRAMES = 30
    MAX_FRAMES_PER_VIDEO = 5000


def normalize_angle(angle):
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


class HeadPoseEstimator:
    def __init__(self):
        self.model = None

    def load_model(self):
        if self.model is not None:
            return True
        try:
            from sixdrepnet import SixDRepNet
            self.model = SixDRepNet(gpu_id=0)
            print("  6DRepNet loaded!")
            return True
        except Exception as e:
            print(f"  ! 6DRepNet failed: {e}")
            return False

    def estimate(self, head_img):
        if self.model is None or head_img is None:
            return None
        try:
            if head_img.shape[0] < 30 or head_img.shape[1] < 30:
                return None
            pitch, yaw, roll = self.model.predict(head_img)
            return float(yaw[0])
        except:
            return None


class GazeAnalyzerV5:
    def __init__(self, cfg):
        self.cfg = cfg

    def analyze_track(self, yaw_history):
        if len(yaw_history) < self.cfg.MIN_TRACK_FRAMES:
            return {'suspicious': False, 'yaw_range': 0, 'direction_changes': 0}

        # 滑动窗口
        recent = yaw_history[-self.cfg.WINDOW_SIZE:] if len(yaw_history) > self.cfg.WINDOW_SIZE else yaw_history
        yaws = np.array(recent)

        # yaw范围
        yaw_range = np.max(yaws) - np.min(yaws)

        # 方向变化
        median_yaw = np.median(yaws)
        direction_changes = 0
        prev_side = None
        for yaw in yaws:
            current_side = 'left' if yaw > median_yaw else 'right'
            if prev_side is not None and current_side != prev_side:
                if abs(yaw - median_yaw) > 10:
                    direction_changes += 1
            prev_side = current_side

        # 需要同时满足两个条件
        is_suspicious = (yaw_range >= self.cfg.YAW_RANGE_THRESHOLD and
                        direction_changes >= self.cfg.DIRECTION_CHANGE_THRESHOLD)

        return {
            'suspicious': is_suspicious,
            'yaw_range': round(yaw_range, 1),
            'direction_changes': direction_changes
        }


class SuspiciousGazeDetectorV5:
    def __init__(self):
        self.cfg = Config()
        self.analyzer = GazeAnalyzerV5(self.cfg)
        self.head_pose = HeadPoseEstimator()

        print("=" * 60)
        print("  可疑张望行为检测系统 V5 (滑动窗口)")
        print("=" * 60)
        print(f"判定标准（3秒窗口内）:")
        print(f"  - Yaw范围 >= {self.cfg.YAW_RANGE_THRESHOLD}° AND")
        print(f"  - 方向变化 >= {self.cfg.DIRECTION_CHANGE_THRESHOLD}次")
        print(f"输出目录: {self.cfg.OUTPUT_DIR}")

        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.cfg.OUTPUT_DIR, "samples_v5"), exist_ok=True)

    def extract_head_region(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = y2 - y1, x2 - x1
        head_h, head_w = int(h * 0.28), int(w * 0.7)
        cx = (x1 + x2) // 2
        hx1, hx2 = max(0, cx - head_w // 2), min(frame.shape[1], cx + head_w // 2)
        hy1, hy2 = max(0, y1), min(frame.shape[0], y1 + head_h)
        if hy2 <= hy1 or hx2 <= hx1:
            return None, None
        return frame[hy1:hy2, hx1:hx2], (hx1, hy1, hx2, hy2)

    def process_video(self, video_name, camera_type, yaw_offset):
        from ultralytics import YOLO

        print(f"\n{'='*50}")
        print(f"处理: {video_name}")
        print(f"{'='*50}")

        if camera_type == "front":
            video_path = os.path.join(self.cfg.DATA_ROOT, "raw_videos/正机位", f"{video_name}.mp4")
        else:
            video_path = os.path.join(self.cfg.DATA_ROOT, "raw_videos/侧机位", f"{video_name}.MP4")

        if not os.path.exists(video_path):
            print(f"  ! 视频不存在")
            return None

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), self.cfg.MAX_FRAMES_PER_VIDEO)

        print(f"  分辨率: {width}x{height}, 处理帧数: {max_frames}")

        model = YOLO("/root/autodl-tmp/behaviour/yolov8m.pt")
        self.head_pose.load_model()

        output_path = os.path.join(self.cfg.OUTPUT_DIR, f"{video_name}_v5.mp4")
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        track_yaws = defaultdict(list)
        track_suspicious = {}
        track_analysis = {}
        track_last_yaw = {}
        track_warning_count = defaultdict(int)  # 记录每个ID的警告次数

        frame_idx = 0
        sample_frames = [100, 500, 1000, 2000, 3000, 4000]

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, persist=True, verbose=False, classes=[0], conf=0.5)
            annotated = frame.copy()
            green_count, red_count = 0, 0

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, tid in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)

                    head_img, head_bbox = self.extract_head_region(frame, box)
                    yaw = None

                    if head_img is not None and self.head_pose.model is not None:
                        yaw = self.head_pose.estimate(head_img)
                        if yaw is not None:
                            yaw = normalize_angle(yaw - yaw_offset)
                            track_yaws[tid].append(yaw)
                            track_last_yaw[tid] = yaw

                    # 每30帧分析
                    if frame_idx % 30 == 0 and len(track_yaws[tid]) >= self.cfg.MIN_TRACK_FRAMES:
                        analysis = self.analyzer.analyze_track(track_yaws[tid])
                        track_suspicious[tid] = analysis['suspicious']
                        track_analysis[tid] = analysis
                        # 统计警告次数
                        if analysis['suspicious']:
                            track_warning_count[tid] += 1

                    is_suspicious = track_suspicious.get(tid, False)
                    body_color = (0, 0, 255) if is_suspicious else (0, 255, 0)
                    if is_suspicious:
                        red_count += 1
                    else:
                        green_count += 1

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), body_color, 2)

                    if head_bbox:
                        hx1, hy1, hx2, hy2 = head_bbox
                        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), (0, 255, 255), 2)

                    display_yaw = yaw if yaw is not None else track_last_yaw.get(tid)
                    label = f"ID:{tid}" + (f" Y:{display_yaw:.0f}" if display_yaw is not None else "")
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (x1, y1-th-8), (x1+tw+4, y1), body_color, -1)
                    cv2.putText(annotated, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                    if is_suspicious:
                        a = track_analysis.get(tid, {})
                        cv2.putText(annotated, f"SUSPICIOUS! R:{a.get('yaw_range',0):.0f} C:{a.get('direction_changes',0)}",
                                   (x1, y2+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            # 状态栏
            cv2.rectangle(annotated, (0, 0), (width, 45), (40, 40, 40), -1)
            cv2.putText(annotated, f"Frame {frame_idx} | Green:{green_count} Red:{red_count}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 图例
            cv2.rectangle(annotated, (width-280, 8), (width-260, 28), (0, 255, 0), -1)
            cv2.putText(annotated, "Normal", (width-255, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.rectangle(annotated, (width-180, 8), (width-160, 28), (0, 0, 255), -1)
            cv2.putText(annotated, "Susp", (width-155, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.rectangle(annotated, (width-100, 8), (width-80, 28), (0, 255, 255), -1)
            cv2.putText(annotated, "Head", (width-75, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            if red_count > 0:
                cv2.rectangle(annotated, (width//2-50, 5), (width//2+50, 40), (0, 0, 200), -1)
                cv2.putText(annotated, "ALERT!", (width//2-38, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 左上角警告统计面板
            if track_warning_count:
                sorted_warnings = sorted(track_warning_count.items(), key=lambda x: x[1], reverse=True)[:8]  # 最多显示8个
                panel_h = 30 + len(sorted_warnings) * 22
                cv2.rectangle(annotated, (0, 50), (180, 50 + panel_h), (0, 0, 0), -1)
                cv2.rectangle(annotated, (0, 50), (180, 50 + panel_h), (100, 100, 100), 1)
                cv2.putText(annotated, "Warning Stats", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                for i, (tid, count) in enumerate(sorted_warnings):
                    y_pos = 95 + i * 22
                    color = (0, 0, 255) if track_suspicious.get(tid, False) else (200, 200, 200)
                    cv2.putText(annotated, f"ID {tid}: {count} warns", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                # 总计
                total_warns = sum(track_warning_count.values())
                cv2.putText(annotated, f"Total: {total_warns}", (10, 50 + panel_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            writer.write(annotated)

            if frame_idx in sample_frames:
                sample_path = os.path.join(self.cfg.OUTPUT_DIR, "samples_v5", f"{video_name}_frame_{frame_idx:05d}.jpg")
                cv2.imwrite(sample_path, annotated)
                print(f"  Frame {frame_idx}: G:{green_count} R:{red_count}")

            frame_idx += 1
            if frame_idx % 500 == 0:
                n_susp = sum(1 for v in track_suspicious.values() if v)
                print(f"  进度: {frame_idx}/{max_frames} | 可疑: {n_susp}/{len(track_suspicious)}")

        cap.release()
        writer.release()

        n_susp = sum(1 for v in track_suspicious.values() if v)
        n_total = len(track_suspicious)
        print(f"\n  完成: {n_susp}/{n_total} ({100*n_susp/max(n_total,1):.1f}%)")
        print(f"  输出: {output_path}")

        # 输出每个ID的警告统计
        if track_warning_count:
            print(f"\n  警告统计 (按ID):")
            sorted_warnings = sorted(track_warning_count.items(), key=lambda x: x[1], reverse=True)
            for tid, count in sorted_warnings:
                print(f"    ID {tid}: {count} 次警告")
            print(f"  总警告次数: {sum(track_warning_count.values())}")

        return {
            'video': video_name,
            'total': n_total,
            'suspicious': n_susp,
            'warning_counts': dict(track_warning_count)  # 每个ID的警告次数
        }

    def run(self):
        print("\n开始处理...")
        results = []
        for camera_type, cfg in self.cfg.VIDEOS.items():
            for video in cfg['videos']:
                r = self.process_video(video, camera_type, cfg['yaw_offset'])
                if r:
                    results.append(r)

        print("\n" + "=" * 60)
        print("处理完成")
        print("=" * 60)
        total = sum(r['total'] for r in results)
        susp = sum(r['suspicious'] for r in results)
        print(f"总计: {susp}/{total} ({100*susp/max(total,1):.1f}%)")

        # 汇总所有视频的警告统计
        print("\n" + "-" * 40)
        print("所有视频警告汇总:")
        print("-" * 40)
        total_warnings = 0
        for r in results:
            video_name = r['video']
            warnings = r.get('warning_counts', {})
            if warnings:
                print(f"\n{video_name}:")
                for tid, count in sorted(warnings.items(), key=lambda x: x[1], reverse=True):
                    print(f"  ID {tid}: {count} 次")
                total_warnings += sum(warnings.values())
        print(f"\n所有视频总警告次数: {total_warnings}")

        return results


if __name__ == "__main__":
    SuspiciousGazeDetectorV5().run()
    print("\n完成!")
