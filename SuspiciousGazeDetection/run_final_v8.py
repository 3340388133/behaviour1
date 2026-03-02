#!/usr/bin/env python3
"""
可疑行为检测系统 V8 - 完整姿态检测 (Yaw + Pitch + Roll)

行为分类：
0 - normal:          正常行为，视线稳定
1 - glancing:        频繁张望，3秒内左右转头≥3次，yaw变化>30°
2 - quick_turn:      快速回头，0.5秒内yaw变化>60°
3 - prolonged_watch: 长时间观察，持续>3秒注视非正前方(yaw>30°)
4 - looking_down:    持续低头，pitch<-20° 持续>5秒
5 - looking_up:      持续抬头，pitch>20° 持续>3秒
"""

import os
import cv2
import numpy as np
from collections import defaultdict, deque
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

    # ========== 行为判定阈值 (按表格定义) ==========

    # 1. glancing: 频繁张望
    GLANCING_WINDOW = 90           # 3秒窗口 (30fps * 3)
    GLANCING_MIN_TURNS = 3         # 左右转头≥3次
    GLANCING_YAW_CHANGE = 30.0     # yaw变化>30°

    # 2. quick_turn: 快速回头
    QUICK_TURN_WINDOW = 15         # 0.5秒窗口 (30fps * 0.5)
    QUICK_TURN_YAW_CHANGE = 60.0   # yaw变化>60°

    # 3. prolonged_watch: 长时间观察
    PROLONGED_WATCH_DURATION = 90  # 持续>3秒 (30fps * 3)
    PROLONGED_WATCH_YAW = 30.0     # yaw>30° (非正前方)

    # 4. looking_down: 持续低头
    LOOKING_DOWN_DURATION = 150    # 持续>5秒 (30fps * 5)
    LOOKING_DOWN_PITCH = -20.0     # pitch<-20°

    # 5. looking_up: 持续抬头
    LOOKING_UP_DURATION = 90       # 持续>3秒 (30fps * 3)
    LOOKING_UP_PITCH = 20.0        # pitch>20°

    MIN_TRACK_FRAMES = 15
    MAX_FRAMES_PER_VIDEO = 5000


def normalize_angle(angle):
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


class HeadPoseEstimator:
    """头部姿态估计器 - 返回 Yaw, Pitch, Roll"""

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
        """返回 (pitch, yaw, roll) 三个角度"""
        if self.model is None or head_img is None:
            return None, None, None
        try:
            if head_img.shape[0] < 30 or head_img.shape[1] < 30:
                return None, None, None
            pitch, yaw, roll = self.model.predict(head_img)
            return float(pitch[0]), float(yaw[0]), float(roll[0])
        except:
            return None, None, None


class BehaviorAnalyzer:
    """行为分析器 V8 - 基于 Yaw, Pitch, Roll"""

    # 行为类型定义
    NORMAL = 0
    GLANCING = 1
    QUICK_TURN = 2
    PROLONGED_WATCH = 3
    LOOKING_DOWN = 4
    LOOKING_UP = 5

    BEHAVIOR_NAMES = {
        0: 'normal',
        1: 'glancing',
        2: 'quick_turn',
        3: 'prolonged_watch',
        4: 'looking_down',
        5: 'looking_up'
    }

    BEHAVIOR_CN = {
        0: '正常',
        1: '频繁张望',
        2: '快速回头',
        3: '长时间观察',
        4: '持续低头',
        5: '持续抬头'
    }

    def __init__(self, cfg):
        self.cfg = cfg

    def detect_glancing(self, yaw_history):
        """1. 频繁张望: 3秒内左右转头≥3次，yaw变化>30°"""
        if len(yaw_history) < self.cfg.GLANCING_WINDOW:
            return False, {}

        recent = list(yaw_history)[-self.cfg.GLANCING_WINDOW:]

        # 计算方向变化次数
        turn_count = 0
        prev_direction = None

        for i in range(1, len(recent)):
            diff = recent[i] - recent[i-1]
            if abs(diff) > 5:  # 忽略小抖动
                curr_direction = 'right' if diff > 0 else 'left'
                if prev_direction and curr_direction != prev_direction:
                    # 检查这次转向的幅度
                    # 找到这个方向上的最大变化
                    turn_count += 1
                prev_direction = curr_direction

        # yaw总变化范围
        yaw_range = max(recent) - min(recent)

        is_glancing = (turn_count >= self.cfg.GLANCING_MIN_TURNS and
                       yaw_range > self.cfg.GLANCING_YAW_CHANGE)

        return is_glancing, {
            'turn_count': turn_count,
            'yaw_range': round(yaw_range, 1)
        }

    def detect_quick_turn(self, yaw_history):
        """2. 快速回头: 0.5秒内yaw变化>60°"""
        if len(yaw_history) < self.cfg.QUICK_TURN_WINDOW:
            return False, {}

        recent = list(yaw_history)[-self.cfg.QUICK_TURN_WINDOW:]

        # 在0.5秒窗口内找最大变化
        max_change = 0
        for i in range(len(recent)):
            for j in range(i+1, len(recent)):
                change = abs(recent[j] - recent[i])
                max_change = max(max_change, change)

        is_quick_turn = max_change > self.cfg.QUICK_TURN_YAW_CHANGE

        return is_quick_turn, {
            'max_change': round(max_change, 1)
        }

    def detect_prolonged_watch(self, yaw_history):
        """3. 长时间观察: 持续>3秒注视非正前方(yaw>30°)"""
        if len(yaw_history) < self.cfg.PROLONGED_WATCH_DURATION:
            return False, {}

        recent = list(yaw_history)[-self.cfg.PROLONGED_WATCH_DURATION:]

        # 检查是否持续偏离正前方
        off_center_count = sum(1 for y in recent if abs(y) > self.cfg.PROLONGED_WATCH_YAW)
        ratio = off_center_count / len(recent)

        # 80%以上时间偏离正前方
        is_prolonged = ratio > 0.8

        return is_prolonged, {
            'off_center_ratio': round(ratio * 100, 1),
            'mean_yaw': round(np.mean(recent), 1)
        }

    def detect_looking_down(self, pitch_history):
        """4. 持续低头: pitch<-20° 持续>5秒"""
        if len(pitch_history) < self.cfg.LOOKING_DOWN_DURATION:
            return False, {}

        recent = list(pitch_history)[-self.cfg.LOOKING_DOWN_DURATION:]

        # 检查是否持续低头
        down_count = sum(1 for p in recent if p < self.cfg.LOOKING_DOWN_PITCH)
        ratio = down_count / len(recent)

        is_looking_down = ratio > 0.8

        return is_looking_down, {
            'down_ratio': round(ratio * 100, 1),
            'mean_pitch': round(np.mean(recent), 1)
        }

    def detect_looking_up(self, pitch_history):
        """5. 持续抬头: pitch>20° 持续>3秒"""
        if len(pitch_history) < self.cfg.LOOKING_UP_DURATION:
            return False, {}

        recent = list(pitch_history)[-self.cfg.LOOKING_UP_DURATION:]

        # 检查是否持续抬头
        up_count = sum(1 for p in recent if p > self.cfg.LOOKING_UP_PITCH)
        ratio = up_count / len(recent)

        is_looking_up = ratio > 0.8

        return is_looking_up, {
            'up_ratio': round(ratio * 100, 1),
            'mean_pitch': round(np.mean(recent), 1)
        }

    def analyze(self, yaw_history, pitch_history, roll_history):
        """综合分析，返回检测到的行为列表"""
        results = {
            'behaviors': [],
            'details': {},
            'is_suspicious': False,
            'primary_behavior': self.NORMAL
        }

        if len(yaw_history) < self.cfg.MIN_TRACK_FRAMES:
            return results

        detected_behaviors = []

        # 检测各种行为
        glancing, glancing_info = self.detect_glancing(yaw_history)
        if glancing:
            detected_behaviors.append(self.GLANCING)
            results['details']['glancing'] = glancing_info

        quick_turn, quick_info = self.detect_quick_turn(yaw_history)
        if quick_turn:
            detected_behaviors.append(self.QUICK_TURN)
            results['details']['quick_turn'] = quick_info

        prolonged, prolonged_info = self.detect_prolonged_watch(yaw_history)
        if prolonged:
            detected_behaviors.append(self.PROLONGED_WATCH)
            results['details']['prolonged_watch'] = prolonged_info

        looking_down, down_info = self.detect_looking_down(pitch_history)
        if looking_down:
            detected_behaviors.append(self.LOOKING_DOWN)
            results['details']['looking_down'] = down_info

        looking_up, up_info = self.detect_looking_up(pitch_history)
        if looking_up:
            detected_behaviors.append(self.LOOKING_UP)
            results['details']['looking_up'] = up_info

        results['behaviors'] = detected_behaviors
        results['is_suspicious'] = len(detected_behaviors) > 0

        # 主要行为（优先级：quick_turn > glancing > prolonged > down > up）
        if detected_behaviors:
            priority = [self.QUICK_TURN, self.GLANCING, self.PROLONGED_WATCH,
                       self.LOOKING_DOWN, self.LOOKING_UP]
            for b in priority:
                if b in detected_behaviors:
                    results['primary_behavior'] = b
                    break

        return results


class SuspiciousDetectorV8:
    def __init__(self):
        self.cfg = Config()
        self.analyzer = BehaviorAnalyzer(self.cfg)
        self.head_pose = HeadPoseEstimator()

        print("=" * 60)
        print("  可疑行为检测系统 V8 (Yaw + Pitch + Roll)")
        print("=" * 60)
        print("行为判定标准:")
        print(f"  1. glancing (频繁张望): 3秒内转头≥{self.cfg.GLANCING_MIN_TURNS}次, yaw变化>{self.cfg.GLANCING_YAW_CHANGE}°")
        print(f"  2. quick_turn (快速回头): 0.5秒内yaw变化>{self.cfg.QUICK_TURN_YAW_CHANGE}°")
        print(f"  3. prolonged_watch (长时间观察): 持续>3秒, |yaw|>{self.cfg.PROLONGED_WATCH_YAW}°")
        print(f"  4. looking_down (持续低头): pitch<{self.cfg.LOOKING_DOWN_PITCH}°, 持续>5秒")
        print(f"  5. looking_up (持续抬头): pitch>{self.cfg.LOOKING_UP_PITCH}°, 持续>3秒")
        print(f"输出目录: {self.cfg.OUTPUT_DIR}")

        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.cfg.OUTPUT_DIR, "samples_v8"), exist_ok=True)

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
            print(f"  ! 视频不存在: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), self.cfg.MAX_FRAMES_PER_VIDEO)

        print(f"  分辨率: {width}x{height}, 处理帧数: {max_frames}")

        model = YOLO("/root/autodl-tmp/behaviour/yolov8m.pt")
        self.head_pose.load_model()

        output_path = os.path.join(self.cfg.OUTPUT_DIR, f"{video_name}_v8.mp4")
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        # 跟踪数据 - 存储三个角度
        track_yaws = defaultdict(lambda: deque(maxlen=300))
        track_pitches = defaultdict(lambda: deque(maxlen=300))
        track_rolls = defaultdict(lambda: deque(maxlen=300))
        track_analysis = {}
        track_suspicious = {}
        track_last_pose = {}  # 存储最后的姿态
        track_warning_count = defaultdict(int)
        track_behavior_count = defaultdict(lambda: defaultdict(int))

        frame_idx = 0
        sample_frames = [100, 500, 1000, 2000, 3000, 4000]

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, persist=True, verbose=False, classes=[0], conf=0.5)
            annotated = frame.copy()
            normal_count, suspicious_count = 0, 0

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, tid in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)

                    # 头部姿态估计
                    head_img, head_bbox = self.extract_head_region(frame, box)
                    pitch, yaw, roll = None, None, None

                    if head_img is not None and self.head_pose.model is not None:
                        pitch, yaw, roll = self.head_pose.estimate(head_img)
                        if yaw is not None:
                            yaw = normalize_angle(yaw - yaw_offset)
                            track_yaws[tid].append(yaw)
                            track_pitches[tid].append(pitch)
                            track_rolls[tid].append(roll)
                            track_last_pose[tid] = (pitch, yaw, roll)

                    # 每10帧分析一次
                    if frame_idx % 10 == 0 and len(track_yaws[tid]) >= self.cfg.MIN_TRACK_FRAMES:
                        analysis = self.analyzer.analyze(
                            list(track_yaws[tid]),
                            list(track_pitches[tid]),
                            list(track_rolls[tid])
                        )
                        track_analysis[tid] = analysis
                        track_suspicious[tid] = analysis['is_suspicious']

                        if analysis['is_suspicious']:
                            track_warning_count[tid] += 1
                            for b in analysis['behaviors']:
                                track_behavior_count[tid][b] += 1

                    is_suspicious = track_suspicious.get(tid, False)
                    analysis = track_analysis.get(tid, {})

                    # 颜色
                    if is_suspicious:
                        body_color = (0, 0, 255)  # 红色
                        suspicious_count += 1
                    else:
                        body_color = (0, 255, 0)  # 绿色
                        normal_count += 1

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), body_color, 2)

                    if head_bbox:
                        hx1, hy1, hx2, hy2 = head_bbox
                        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), (0, 255, 255), 2)

                    # 标签显示
                    last_pose = track_last_pose.get(tid, (None, None, None))
                    p, y, r = last_pose
                    label = f"ID:{tid}"
                    if y is not None:
                        label += f" Y:{y:.0f}"
                    if p is not None:
                        label += f" P:{p:.0f}"

                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (x1, y1-th-8), (x1+tw+4, y1), body_color, -1)
                    cv2.putText(annotated, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                    # 显示检测到的行为
                    if is_suspicious and analysis.get('behaviors'):
                        behavior_names = [self.analyzer.BEHAVIOR_NAMES[b] for b in analysis['behaviors']]
                        behavior_str = '+'.join(behavior_names)
                        cv2.putText(annotated, f"[{behavior_str}]",
                                   (x1, y2+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)

            # 状态栏
            cv2.rectangle(annotated, (0, 0), (width, 45), (40, 40, 40), -1)
            cv2.putText(annotated, f"V8 Frame {frame_idx} | Normal:{normal_count} Suspicious:{suspicious_count}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 图例
            legend_x = width - 400
            cv2.rectangle(annotated, (legend_x, 8), (legend_x+20, 28), (0, 255, 0), -1)
            cv2.putText(annotated, "Normal", (legend_x+25, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
            cv2.rectangle(annotated, (legend_x+90, 8), (legend_x+110, 28), (0, 0, 255), -1)
            cv2.putText(annotated, "Suspicious", (legend_x+115, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
            cv2.putText(annotated, "Y=Yaw P=Pitch", (legend_x+210, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

            if suspicious_count > 0:
                cv2.rectangle(annotated, (width//2-50, 5), (width//2+50, 40), (0, 0, 200), -1)
                cv2.putText(annotated, "ALERT!", (width//2-38, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 左上角警告统计面板
            if track_warning_count:
                sorted_warnings = sorted(track_warning_count.items(), key=lambda x: x[1], reverse=True)[:6]
                panel_h = 55 + len(sorted_warnings) * 28
                cv2.rectangle(annotated, (0, 50), (220, 50 + panel_h), (0, 0, 0), -1)
                cv2.rectangle(annotated, (0, 50), (220, 50 + panel_h), (100, 100, 100), 1)
                cv2.putText(annotated, "Warning Stats", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                for i, (tid, count) in enumerate(sorted_warnings):
                    y_pos = 98 + i * 28
                    color = (0, 0, 255) if track_suspicious.get(tid, False) else (200, 200, 200)
                    behaviors = track_behavior_count.get(tid, {})
                    behavior_str = ','.join([f"{self.analyzer.BEHAVIOR_NAMES[k][:4]}:{v}"
                                            for k, v in sorted(behaviors.items())[:3]])
                    cv2.putText(annotated, f"ID{tid}: {count}w", (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                    cv2.putText(annotated, behavior_str, (80, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150,150,150), 1)

                total_warns = sum(track_warning_count.values())
                cv2.putText(annotated, f"Total: {total_warns}", (10, 50 + panel_h - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            writer.write(annotated)

            if frame_idx in sample_frames:
                sample_path = os.path.join(self.cfg.OUTPUT_DIR, "samples_v8", f"{video_name}_frame_{frame_idx:05d}.jpg")
                cv2.imwrite(sample_path, annotated)
                print(f"  Frame {frame_idx}: Normal:{normal_count} Susp:{suspicious_count}")

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

        # 输出详细统计
        if track_warning_count:
            print(f"\n  警告统计 (按ID):")
            sorted_warnings = sorted(track_warning_count.items(), key=lambda x: x[1], reverse=True)
            for tid, count in sorted_warnings:
                behaviors = track_behavior_count.get(tid, {})
                behavior_str = ', '.join([f"{self.analyzer.BEHAVIOR_NAMES[k]}:{v}"
                                         for k, v in sorted(behaviors.items())])
                print(f"    ID {tid}: {count} 次警告 [{behavior_str}]")
            print(f"  总警告次数: {sum(track_warning_count.values())}")

        # 行为统计
        print(f"\n  行为类型统计:")
        all_behaviors = defaultdict(int)
        for tid, behaviors in track_behavior_count.items():
            for b, c in behaviors.items():
                all_behaviors[b] += c
        for b in sorted(all_behaviors.keys()):
            print(f"    {self.analyzer.BEHAVIOR_NAMES[b]}: {all_behaviors[b]} 次")

        return {
            'video': video_name,
            'total': n_total,
            'suspicious': n_susp,
            'warning_counts': dict(track_warning_count),
            'behavior_counts': {tid: dict(b) for tid, b in track_behavior_count.items()}
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

        return results


if __name__ == "__main__":
    SuspiciousDetectorV8().run()
    print("\n完成!")
