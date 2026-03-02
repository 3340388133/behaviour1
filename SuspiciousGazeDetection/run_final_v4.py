#!/usr/bin/env python3
"""
可疑张望行为检测系统 V4
- 大幅度转头2次即可疑
- 小幅度快速转头也可疑
- 显示头部框和角度变化
"""

import os
import sys
import cv2
import json
import numpy as np
from pathlib import Path
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

    # ===== V4: 调整后的判定标准 =====
    # 大幅度转头: >60度，2次即可疑
    BIG_TURN_THRESHOLD = 60.0
    BIG_TURN_COUNT = 2

    # 小幅度快转头: 20-60度但速度很快
    SMALL_TURN_THRESHOLD = 20.0
    ANGULAR_VELOCITY_THRESHOLD = 150.0  # >150度/秒才算快速转头
    FAST_TURN_COUNT = 8   # 8次快速小转头才算可疑

    MIN_TRACK_FRAMES = 15
    MAX_FRAMES_PER_VIDEO = 5000
    SAVE_VIDEO = True


def normalize_angle(angle):
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def angle_diff(a1, a2):
    diff = a2 - a1
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff


class HeadPoseEstimator:
    """6DRepNet头部姿态估计 - 使用安装的sixdrepnet库"""
    def __init__(self):
        self.model = None

    def load_model(self):
        if self.model is not None:
            return True

        try:
            from sixdrepnet import SixDRepNet
            self.model = SixDRepNet(gpu_id=0)
            print("  6DRepNet loaded successfully!")
            return True
        except Exception as e:
            print(f"  ! 加载6DRepNet失败: {e}")
            self.model = None
            return False

    def estimate(self, head_img):
        """估计头部姿态，返回yaw角度"""
        if self.model is None or head_img is None:
            return None

        try:
            if head_img.shape[0] < 30 or head_img.shape[1] < 30:
                return None

            # sixdrepnet expects BGR image
            pitch, yaw, roll = self.model.predict(head_img)
            return float(yaw[0])
        except Exception as e:
            return None


class GazeAnalyzerV4:
    """V4: 支持大幅度和小幅度快速转头检测"""
    def __init__(self, cfg):
        self.cfg = cfg

    def analyze_track(self, yaw_history, fps=30):
        """
        分析轨迹的yaw历史
        yaw_history: [(frame_idx, yaw_angle), ...]
        """
        if len(yaw_history) < self.cfg.MIN_TRACK_FRAMES:
            return {'suspicious': False, 'reason': 'too_short', 'big_turns': 0, 'fast_turns': 0}

        # 按帧排序
        yaw_history = sorted(yaw_history, key=lambda x: x[0])
        frames = [h[0] for h in yaw_history]
        yaws = [h[1] for h in yaw_history]

        # 计算角度变化和角速度
        big_turns = 0
        fast_turns = 0
        turn_events = []

        for i in range(1, len(yaws)):
            diff = abs(angle_diff(yaws[i-1], yaws[i]))
            dt = (frames[i] - frames[i-1]) / fps  # 秒

            if dt > 0:
                angular_velocity = diff / dt  # 度/秒

                # 大幅度转头
                if diff >= self.cfg.BIG_TURN_THRESHOLD:
                    big_turns += 1
                    turn_events.append({'type': 'big', 'angle': diff, 'velocity': angular_velocity})

                # 小幅度但快速转头
                elif diff >= self.cfg.SMALL_TURN_THRESHOLD and angular_velocity >= self.cfg.ANGULAR_VELOCITY_THRESHOLD:
                    fast_turns += 1
                    turn_events.append({'type': 'fast', 'angle': diff, 'velocity': angular_velocity})

        # 判定可疑
        is_suspicious = False
        reasons = []

        if big_turns >= self.cfg.BIG_TURN_COUNT:
            is_suspicious = True
            reasons.append(f'big_turns:{big_turns}')

        if fast_turns >= self.cfg.FAST_TURN_COUNT:
            is_suspicious = True
            reasons.append(f'fast_turns:{fast_turns}')

        # 组合判定：大转头+快转头
        if big_turns >= 1 and fast_turns >= 2:
            is_suspicious = True
            reasons.append('combo')

        return {
            'suspicious': is_suspicious,
            'big_turns': big_turns,
            'fast_turns': fast_turns,
            'total_turns': big_turns + fast_turns,
            'reasons': reasons,
            'turn_events': turn_events[:10]
        }


class SuspiciousGazeDetectorV4:
    def __init__(self):
        self.cfg = Config()
        self.analyzer = GazeAnalyzerV4(self.cfg)
        self.head_pose = HeadPoseEstimator()

        print("=" * 60)
        print("  可疑张望行为检测系统 V4")
        print("=" * 60)
        print(f"判定标准:")
        print(f"  - 大幅度转头: >{self.cfg.BIG_TURN_THRESHOLD}°, {self.cfg.BIG_TURN_COUNT}次即可疑")
        print(f"  - 小幅度快转头: >{self.cfg.SMALL_TURN_THRESHOLD}°, 速度>{self.cfg.ANGULAR_VELOCITY_THRESHOLD}°/s")
        print(f"  - 快转头次数: {self.cfg.FAST_TURN_COUNT}次即可疑")
        print(f"输出目录: {self.cfg.OUTPUT_DIR}")

        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.cfg.OUTPUT_DIR, "samples_v4"), exist_ok=True)

    def extract_head_region(self, frame, bbox):
        """从人体框中提取头部区域"""
        x1, y1, x2, y2 = map(int, bbox)
        h = y2 - y1
        w = x2 - x1

        # 头部约占上1/4，宽度取中间70%
        head_h = int(h * 0.28)
        head_w = int(w * 0.7)
        cx = (x1 + x2) // 2

        hx1 = max(0, cx - head_w // 2)
        hx2 = min(frame.shape[1], cx + head_w // 2)
        hy1 = max(0, y1)
        hy2 = min(frame.shape[0], y1 + head_h)

        if hy2 <= hy1 or hx2 <= hx1:
            return None, None

        head_img = frame[hy1:hy2, hx1:hx2]
        head_bbox = (hx1, hy1, hx2, hy2)
        return head_img, head_bbox

    def process_video(self, video_name, camera_type, yaw_offset):
        """处理单个视频，实时检测"""
        from ultralytics import YOLO

        print(f"\n{'='*50}")
        print(f"处理: {video_name} ({'正机位' if camera_type == 'front' else '侧机位'})")
        print(f"{'='*50}")

        # 加载视频
        if camera_type == "front":
            video_path = os.path.join(self.cfg.DATA_ROOT, "raw_videos/正机位", f"{video_name}.mp4")
        else:
            video_path = os.path.join(self.cfg.DATA_ROOT, "raw_videos/侧机位", f"{video_name}.MP4")

        if not os.path.exists(video_path):
            print(f"  ! 视频不存在: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_frames = min(total_frames, self.cfg.MAX_FRAMES_PER_VIDEO)

        print(f"  分辨率: {width}x{height}, FPS: {fps}, 处理帧数: {max_frames}")

        # 加载模型
        model = YOLO("/root/autodl-tmp/behaviour/yolov8m.pt")
        if not self.head_pose.load_model():
            print("  ! 头部姿态模型加载失败，将只显示检测框")

        # 输出视频
        output_path = os.path.join(self.cfg.OUTPUT_DIR, f"{video_name}_v4.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 跟踪数据
        track_yaw_history = defaultdict(list)  # track_id -> [(frame, yaw), ...]
        track_suspicious = {}  # track_id -> bool
        track_analysis = {}  # track_id -> analysis result
        track_last_yaw = {}  # track_id -> last yaw value

        frame_idx = 0
        sample_frames = [100, 500, 1000, 2000, 3000, 4000]

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLOv8跟踪
            results = model.track(frame, persist=True, verbose=False, classes=[0], conf=0.5)

            annotated = frame.copy()
            green_count = 0
            red_count = 0

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, tid in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)

                    # 提取头部并估计姿态
                    head_img, head_bbox = self.extract_head_region(frame, box)
                    yaw = None

                    if head_img is not None and self.head_pose.model is not None:
                        yaw = self.head_pose.estimate(head_img)
                        if yaw is not None:
                            yaw = normalize_angle(yaw - yaw_offset)
                            track_yaw_history[tid].append((frame_idx, yaw))
                            track_last_yaw[tid] = yaw

                    # 每隔30帧重新分析
                    if frame_idx % 30 == 0 and len(track_yaw_history[tid]) >= self.cfg.MIN_TRACK_FRAMES:
                        analysis = self.analyzer.analyze_track(track_yaw_history[tid], fps)
                        track_suspicious[tid] = analysis['suspicious']
                        track_analysis[tid] = analysis

                    is_suspicious = track_suspicious.get(tid, False)

                    # 绘制人体框
                    if is_suspicious:
                        body_color = (0, 0, 255)  # 红色
                        red_count += 1
                    else:
                        body_color = (0, 255, 0)  # 绿色
                        green_count += 1

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), body_color, 2)

                    # 绘制头部框（黄色）
                    if head_bbox is not None:
                        hx1, hy1, hx2, hy2 = head_bbox
                        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), (0, 255, 255), 2)

                    # 显示ID和角度
                    display_yaw = yaw if yaw is not None else track_last_yaw.get(tid)
                    if display_yaw is not None:
                        label = f"ID:{tid} Y:{display_yaw:.0f}"
                    else:
                        label = f"ID:{tid}"

                    # 标签背景
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (x1, y1-th-8), (x1+tw+4, y1), body_color, -1)
                    cv2.putText(annotated, label, (x1+2, y1-4),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # 可疑标记
                    if is_suspicious:
                        analysis = track_analysis.get(tid, {})
                        big = analysis.get('big_turns', 0)
                        fast = analysis.get('fast_turns', 0)
                        susp_label = f"SUSPICIOUS! B:{big} F:{fast}"
                        cv2.putText(annotated, susp_label, (x1, y2 + 18),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 状态栏
            cv2.rectangle(annotated, (0, 0), (width, 45), (40, 40, 40), -1)
            status = f"Frame {frame_idx} | Green(Normal):{green_count} Red(Suspicious):{red_count}"
            cv2.putText(annotated, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 图例
            cv2.rectangle(annotated, (width-280, 8), (width-260, 28), (0, 255, 0), -1)
            cv2.putText(annotated, "Normal", (width-255, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.rectangle(annotated, (width-180, 8), (width-160, 28), (0, 0, 255), -1)
            cv2.putText(annotated, "Susp", (width-155, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.rectangle(annotated, (width-100, 8), (width-80, 28), (0, 255, 255), -1)
            cv2.putText(annotated, "Head", (width-75, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # ALERT
            if red_count > 0:
                cv2.rectangle(annotated, (width//2-50, 5), (width//2+50, 40), (0, 0, 200), -1)
                cv2.putText(annotated, "ALERT!", (width//2-38, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            writer.write(annotated)

            # 保存示例帧
            if frame_idx in sample_frames:
                sample_path = os.path.join(self.cfg.OUTPUT_DIR, "samples_v4", f"{video_name}_frame_{frame_idx:05d}.jpg")
                cv2.imwrite(sample_path, annotated)
                print(f"  保存示例: frame_{frame_idx} (G:{green_count} R:{red_count})")

            frame_idx += 1

            if frame_idx % 500 == 0:
                n_suspicious = sum(1 for v in track_suspicious.values() if v)
                n_total = len(track_suspicious)
                print(f"  进度: {frame_idx}/{max_frames} | 可疑: {n_suspicious}/{n_total}")

        cap.release()
        writer.release()

        # 统计
        n_suspicious = sum(1 for v in track_suspicious.values() if v)
        n_total = len(track_suspicious)

        print(f"\n  完成: {video_name}")
        print(f"  总轨迹: {n_total}, 可疑: {n_suspicious} ({100*n_suspicious/max(n_total,1):.1f}%)")
        print(f"  输出: {output_path}")

        return {
            'video_name': video_name,
            'camera_type': camera_type,
            'total_tracks': n_total,
            'suspicious_tracks': n_suspicious,
            'suspicious_ratio': n_suspicious / max(n_total, 1),
            'track_analysis': {str(k): v for k, v in track_analysis.items()}
        }

    def run(self):
        print("\n" + "=" * 60)
        print("开始处理所有视频...")
        print("=" * 60)

        all_results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'big_turn_threshold': self.cfg.BIG_TURN_THRESHOLD,
                'big_turn_count': self.cfg.BIG_TURN_COUNT,
                'small_turn_threshold': self.cfg.SMALL_TURN_THRESHOLD,
                'angular_velocity_threshold': self.cfg.ANGULAR_VELOCITY_THRESHOLD,
                'fast_turn_count': self.cfg.FAST_TURN_COUNT
            },
            'videos': []
        }

        for camera_type, camera_cfg in self.cfg.VIDEOS.items():
            yaw_offset = camera_cfg['yaw_offset']

            for video_name in camera_cfg['videos']:
                result = self.process_video(video_name, camera_type, yaw_offset)
                if result:
                    all_results['videos'].append(result)

        # 保存结果
        output_file = os.path.join(self.cfg.OUTPUT_DIR, "results_v4.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            save_data = {
                'timestamp': all_results['timestamp'],
                'config': all_results['config'],
                'summary': [{
                    'video': v['video_name'],
                    'total': v['total_tracks'],
                    'suspicious': v['suspicious_tracks'],
                    'ratio': f"{v['suspicious_ratio']*100:.1f}%"
                } for v in all_results['videos']]
            }
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"\n结果已保存: {output_file}")

        # 汇总
        print("\n" + "=" * 60)
        print("处理完成 - 汇总")
        print("=" * 60)

        total_tracks = sum(v['total_tracks'] for v in all_results['videos'])
        total_suspicious = sum(v['suspicious_tracks'] for v in all_results['videos'])

        print(f"总轨迹: {total_tracks}")
        print(f"可疑轨迹: {total_suspicious} ({100*total_suspicious/max(total_tracks,1):.1f}%)")

        for v in all_results['videos']:
            print(f"  {v['video_name']}: {v['suspicious_tracks']}/{v['total_tracks']} ({v['suspicious_ratio']*100:.1f}%)")

        return all_results


if __name__ == "__main__":
    detector = SuspiciousGazeDetectorV4()
    detector.run()
    print("\n完成!")
