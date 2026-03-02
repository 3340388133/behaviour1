#!/usr/bin/env python3
"""
可疑行为检测系统 V9 - 严格阈值版 (目标: 10-20%可疑率)

行为分类：
0 - normal:          正常行为
1 - glancing:        频繁张望，3秒内左右转头≥4次，yaw变化>50°
2 - quick_turn:      快速回头，0.5秒内yaw变化>70°
3 - prolonged_watch: 长时间观察，持续>4秒注视非正前方(yaw>45°)
4 - looking_down:    持续低头，pitch<-25° 持续>5秒
5 - looking_up:      持续抬头，pitch>25° 持续>3秒
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

    # ========== V9: 严格阈值 ==========

    # 1. glancing: 频繁张望 (更严格)
    GLANCING_WINDOW = 90           # 3秒窗口
    GLANCING_MIN_TURNS = 4         # 左右转头≥4次 (原3)
    GLANCING_YAW_CHANGE = 50.0     # yaw变化>50° (原30)

    # 2. quick_turn: 快速回头 (更严格)
    QUICK_TURN_WINDOW = 15         # 0.5秒窗口
    QUICK_TURN_YAW_CHANGE = 70.0   # yaw变化>70° (原60)

    # 3. prolonged_watch: 长时间观察 (更严格)
    PROLONGED_WATCH_DURATION = 120 # 持续>4秒 (原3秒)
    PROLONGED_WATCH_YAW = 45.0     # yaw>45° (原30)

    # 4. looking_down: 持续低头 (更严格)
    LOOKING_DOWN_DURATION = 150    # 持续>5秒
    LOOKING_DOWN_PITCH = -25.0     # pitch<-25° (原-20)

    # 5. looking_up: 持续抬头 (更严格)
    LOOKING_UP_DURATION = 90       # 持续>3秒
    LOOKING_UP_PITCH = 25.0        # pitch>25° (原20)

    MIN_TRACK_FRAMES = 30          # 最少跟踪1秒才分析
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
            return None, None, None
        try:
            if head_img.shape[0] < 30 or head_img.shape[1] < 30:
                return None, None, None
            pitch, yaw, roll = self.model.predict(head_img)
            return float(pitch[0]), float(yaw[0]), float(roll[0])
        except:
            return None, None, None


class BehaviorAnalyzer:
    """行为分析器 V9 - 严格版"""

    NORMAL = 0
    GLANCING = 1
    QUICK_TURN = 2
    PROLONGED_WATCH = 3
    LOOKING_DOWN = 4
    LOOKING_UP = 5

    BEHAVIOR_NAMES = {
        0: 'normal', 1: 'glancing', 2: 'quick_turn',
        3: 'prolonged', 4: 'look_down', 5: 'look_up'
    }

    BEHAVIOR_COLORS = {
        0: (0, 255, 0),      # 绿色 - 正常
        1: (0, 0, 255),      # 红色 - 频繁张望
        2: (0, 165, 255),    # 橙色 - 快速回头
        3: (255, 0, 255),    # 紫色 - 长时间观察
        4: (255, 255, 0),    # 青色 - 低头
        5: (0, 255, 255),    # 黄色 - 抬头
    }

    def __init__(self, cfg):
        self.cfg = cfg

    def detect_glancing(self, yaw_history):
        """频繁张望: 3秒内左右转头≥4次，yaw范围>50°"""
        if len(yaw_history) < self.cfg.GLANCING_WINDOW:
            return False, {}

        recent = list(yaw_history)[-self.cfg.GLANCING_WINDOW:]

        # 计算转头次数 - 使用过零点检测
        mean_yaw = np.mean(recent)
        turn_count = 0
        prev_side = None

        for i, yaw in enumerate(recent):
            # 需要明显偏离中心才算
            if abs(yaw - mean_yaw) > 15:
                curr_side = 'left' if yaw > mean_yaw else 'right'
                if prev_side and curr_side != prev_side:
                    turn_count += 1
                prev_side = curr_side

        yaw_range = max(recent) - min(recent)

        is_glancing = (turn_count >= self.cfg.GLANCING_MIN_TURNS and
                       yaw_range > self.cfg.GLANCING_YAW_CHANGE)

        return is_glancing, {'turns': turn_count, 'range': round(yaw_range, 1)}

    def detect_quick_turn(self, yaw_history):
        """快速回头: 0.5秒内yaw变化>70°"""
        if len(yaw_history) < self.cfg.QUICK_TURN_WINDOW:
            return False, {}

        recent = list(yaw_history)[-self.cfg.QUICK_TURN_WINDOW:]
        max_change = max(recent) - min(recent)

        is_quick = max_change > self.cfg.QUICK_TURN_YAW_CHANGE

        return is_quick, {'change': round(max_change, 1)}

    def detect_prolonged_watch(self, yaw_history):
        """长时间观察: 持续>4秒注视非正前方(|yaw|>45°)"""
        if len(yaw_history) < self.cfg.PROLONGED_WATCH_DURATION:
            return False, {}

        recent = list(yaw_history)[-self.cfg.PROLONGED_WATCH_DURATION:]

        # 检查是否持续偏离正前方
        off_center = [y for y in recent if abs(y) > self.cfg.PROLONGED_WATCH_YAW]
        ratio = len(off_center) / len(recent)

        is_prolonged = ratio > 0.85  # 85%以上时间偏离

        return is_prolonged, {
            'ratio': round(ratio * 100, 1),
            'mean': round(np.mean(recent), 1)
        }

    def detect_looking_down(self, pitch_history):
        """持续低头: pitch<-25° 持续>5秒"""
        if len(pitch_history) < self.cfg.LOOKING_DOWN_DURATION:
            return False, {}

        recent = list(pitch_history)[-self.cfg.LOOKING_DOWN_DURATION:]
        down_count = sum(1 for p in recent if p < self.cfg.LOOKING_DOWN_PITCH)
        ratio = down_count / len(recent)

        is_down = ratio > 0.85

        return is_down, {'ratio': round(ratio * 100, 1)}

    def detect_looking_up(self, pitch_history):
        """持续抬头: pitch>25° 持续>3秒"""
        if len(pitch_history) < self.cfg.LOOKING_UP_DURATION:
            return False, {}

        recent = list(pitch_history)[-self.cfg.LOOKING_UP_DURATION:]
        up_count = sum(1 for p in recent if p > self.cfg.LOOKING_UP_PITCH)
        ratio = up_count / len(recent)

        is_up = ratio > 0.85

        return is_up, {'ratio': round(ratio * 100, 1)}

    def analyze(self, yaw_history, pitch_history, roll_history):
        results = {
            'behaviors': [],
            'details': {},
            'is_suspicious': False,
            'primary_behavior': self.NORMAL,
            'color': self.BEHAVIOR_COLORS[self.NORMAL]
        }

        if len(yaw_history) < self.cfg.MIN_TRACK_FRAMES:
            return results

        detected = []

        # 按优先级检测
        glancing, g_info = self.detect_glancing(yaw_history)
        if glancing:
            detected.append(self.GLANCING)
            results['details']['glancing'] = g_info

        quick, q_info = self.detect_quick_turn(yaw_history)
        if quick:
            detected.append(self.QUICK_TURN)
            results['details']['quick_turn'] = q_info

        prolonged, p_info = self.detect_prolonged_watch(yaw_history)
        if prolonged:
            detected.append(self.PROLONGED_WATCH)
            results['details']['prolonged'] = p_info

        down, d_info = self.detect_looking_down(pitch_history)
        if down:
            detected.append(self.LOOKING_DOWN)
            results['details']['look_down'] = d_info

        up, u_info = self.detect_looking_up(pitch_history)
        if up:
            detected.append(self.LOOKING_UP)
            results['details']['look_up'] = u_info

        results['behaviors'] = detected
        results['is_suspicious'] = len(detected) > 0

        if detected:
            # 优先级: quick_turn > glancing > prolonged > down > up
            priority = [self.QUICK_TURN, self.GLANCING, self.PROLONGED_WATCH,
                       self.LOOKING_DOWN, self.LOOKING_UP]
            for b in priority:
                if b in detected:
                    results['primary_behavior'] = b
                    results['color'] = self.BEHAVIOR_COLORS[b]
                    break

        return results


class SuspiciousDetectorV9:
    def __init__(self):
        self.cfg = Config()
        self.analyzer = BehaviorAnalyzer(self.cfg)
        self.head_pose = HeadPoseEstimator()

        print("=" * 60)
        print("  可疑行为检测系统 V9 (严格阈值, 目标10-20%)")
        print("=" * 60)
        print("行为判定标准:")
        print(f"  1. glancing: 3秒内转头≥{self.cfg.GLANCING_MIN_TURNS}次, yaw范围>{self.cfg.GLANCING_YAW_CHANGE}°")
        print(f"  2. quick_turn: 0.5秒内yaw变化>{self.cfg.QUICK_TURN_YAW_CHANGE}°")
        print(f"  3. prolonged: 持续>{self.cfg.PROLONGED_WATCH_DURATION/30:.0f}秒, |yaw|>{self.cfg.PROLONGED_WATCH_YAW}°")
        print(f"  4. look_down: pitch<{self.cfg.LOOKING_DOWN_PITCH}°, 持续>5秒")
        print(f"  5. look_up: pitch>{self.cfg.LOOKING_UP_PITCH}°, 持续>3秒")

        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.cfg.OUTPUT_DIR, "samples_v9"), exist_ok=True)

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

        print(f"  分辨率: {width}x{height}, 帧数: {max_frames}")

        model = YOLO("/root/autodl-tmp/behaviour/yolov8m.pt")
        self.head_pose.load_model()

        output_path = os.path.join(self.cfg.OUTPUT_DIR, f"{video_name}_v9.mp4")
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        track_yaws = defaultdict(lambda: deque(maxlen=300))
        track_pitches = defaultdict(lambda: deque(maxlen=300))
        track_rolls = defaultdict(lambda: deque(maxlen=300))
        track_analysis = {}
        track_suspicious = {}
        track_last_pose = {}
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

                    # 每15帧分析
                    if frame_idx % 15 == 0 and len(track_yaws[tid]) >= self.cfg.MIN_TRACK_FRAMES:
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
                    color = analysis.get('color', (0, 255, 0))

                    if is_suspicious:
                        suspicious_count += 1
                    else:
                        normal_count += 1

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                    if head_bbox:
                        hx1, hy1, hx2, hy2 = head_bbox
                        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), (0, 255, 255), 2)

                    # 标签
                    last_pose = track_last_pose.get(tid, (None, None, None))
                    p, y, r = last_pose
                    label = f"ID:{tid}"
                    if y is not None:
                        label += f" Y:{y:.0f}"
                    if p is not None:
                        label += f" P:{p:.0f}"

                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
                    cv2.putText(annotated, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                    # 行为标签
                    if is_suspicious and analysis.get('behaviors'):
                        names = [self.analyzer.BEHAVIOR_NAMES[b] for b in analysis['behaviors']]
                        cv2.putText(annotated, f"[{'+'.join(names)}]",
                                   (x1, y2+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            # 状态栏
            cv2.rectangle(annotated, (0, 0), (width, 45), (40, 40, 40), -1)
            total_now = normal_count + suspicious_count
            ratio = suspicious_count / max(total_now, 1) * 100
            cv2.putText(annotated, f"V9 Frame {frame_idx} | Normal:{normal_count} Susp:{suspicious_count} ({ratio:.0f}%)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            if suspicious_count > 0:
                cv2.rectangle(annotated, (width//2-50, 5), (width//2+50, 40), (0, 0, 200), -1)
                cv2.putText(annotated, "ALERT!", (width//2-38, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 左上角统计
            if track_warning_count:
                sorted_w = sorted(track_warning_count.items(), key=lambda x: x[1], reverse=True)[:5]
                panel_h = 50 + len(sorted_w) * 25
                cv2.rectangle(annotated, (0, 50), (200, 50 + panel_h), (0, 0, 0), -1)
                cv2.putText(annotated, "Top Warnings", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                for i, (tid, count) in enumerate(sorted_w):
                    y_pos = 95 + i * 25
                    c = (0, 0, 255) if track_suspicious.get(tid, False) else (180, 180, 180)
                    behaviors = track_behavior_count.get(tid, {})
                    b_str = ','.join([f"{self.analyzer.BEHAVIOR_NAMES[k][:4]}:{v}" for k, v in behaviors.items()][:2])
                    cv2.putText(annotated, f"ID{tid}:{count}w {b_str}", (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)

            writer.write(annotated)

            if frame_idx in sample_frames:
                cv2.imwrite(os.path.join(self.cfg.OUTPUT_DIR, "samples_v9", f"{video_name}_{frame_idx:05d}.jpg"), annotated)
                print(f"  Frame {frame_idx}: N:{normal_count} S:{suspicious_count}")

            frame_idx += 1
            if frame_idx % 500 == 0:
                n_susp = sum(1 for v in track_suspicious.values() if v)
                n_total = len(track_suspicious)
                print(f"  进度: {frame_idx}/{max_frames} | 可疑: {n_susp}/{n_total} ({100*n_susp/max(n_total,1):.1f}%)")

        cap.release()
        writer.release()

        n_susp = sum(1 for v in track_suspicious.values() if v)
        n_total = len(track_suspicious)
        print(f"\n  完成: {n_susp}/{n_total} ({100*n_susp/max(n_total,1):.1f}%)")
        print(f"  输出: {output_path}")

        if track_warning_count:
            print(f"\n  警告统计:")
            for tid, count in sorted(track_warning_count.items(), key=lambda x: x[1], reverse=True)[:10]:
                behaviors = track_behavior_count.get(tid, {})
                b_str = ', '.join([f"{self.analyzer.BEHAVIOR_NAMES[k]}:{v}" for k, v in behaviors.items()])
                print(f"    ID {tid}: {count}次 [{b_str}]")

        return {'video': video_name, 'total': n_total, 'suspicious': n_susp}

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
    SuspiciousDetectorV9().run()
