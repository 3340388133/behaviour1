#!/usr/bin/env python3
"""
可疑行为检测系统 V6 - 多维度分析
检测模式：
1. 频繁张望 (Scanning) - 头部频繁左右转动
2. 固定凝视 (Fixation) - 长时间盯着非正前方
3. 快速扫视 (Quick Glance) - 短时间内大幅度转动
4. 徘徊检测 (Loitering) - 在某区域停留过久
"""

import os
import cv2
import numpy as np
from datetime import datetime
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
    WINDOW_SIZE = 90  # 3秒窗口

    # 模式1: 频繁张望 (Scanning)
    SCANNING_YAW_RANGE = 80.0        # yaw范围>80度
    SCANNING_DIR_CHANGES = 4         # 方向变化>4次

    # 模式2: 固定凝视 (Fixation)
    FIXATION_YAW_THRESHOLD = 35.0    # 偏离正前方>35度
    FIXATION_DURATION = 60           # 持续2秒(60帧)
    FIXATION_STABILITY = 15.0        # yaw变化<15度算稳定

    # 模式3: 快速扫视 (Quick Glance)
    QUICK_GLANCE_SPEED = 25.0        # 每帧yaw变化>25度
    QUICK_GLANCE_COUNT = 2           # 出现>2次快速扫视

    # 模式4: 徘徊 (Loitering)
    LOITER_DURATION = 150            # 停留>5秒(150帧)
    LOITER_MOVE_THRESHOLD = 100      # 移动距离<100像素

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


class MultiPatternAnalyzer:
    """多模式可疑行为分析器"""

    def __init__(self, cfg):
        self.cfg = cfg

    def detect_scanning(self, yaw_history):
        """模式1: 频繁张望检测"""
        if len(yaw_history) < self.cfg.WINDOW_SIZE:
            return False, {}

        recent = list(yaw_history)[-self.cfg.WINDOW_SIZE:]
        yaws = np.array(recent)
        yaw_range = np.max(yaws) - np.min(yaws)

        # 方向变化计数
        median_yaw = np.median(yaws)
        dir_changes = 0
        prev_side = None
        for yaw in yaws:
            curr_side = 'left' if yaw > median_yaw + 10 else ('right' if yaw < median_yaw - 10 else 'center')
            if prev_side and curr_side != 'center' and prev_side != curr_side:
                dir_changes += 1
            if curr_side != 'center':
                prev_side = curr_side

        is_scanning = (yaw_range >= self.cfg.SCANNING_YAW_RANGE and
                       dir_changes >= self.cfg.SCANNING_DIR_CHANGES)

        return is_scanning, {
            'yaw_range': round(yaw_range, 1),
            'dir_changes': dir_changes
        }

    def detect_fixation(self, yaw_history):
        """模式2: 固定凝视检测 - 长时间盯着非正前方"""
        if len(yaw_history) < self.cfg.FIXATION_DURATION:
            return False, {}

        recent = list(yaw_history)[-self.cfg.FIXATION_DURATION:]
        yaws = np.array(recent)

        # 计算平均偏离角度和稳定性
        mean_yaw = np.mean(yaws)
        yaw_std = np.std(yaws)

        # 条件: 平均偏离正前方 + 角度稳定
        is_fixation = (abs(mean_yaw) >= self.cfg.FIXATION_YAW_THRESHOLD and
                       yaw_std <= self.cfg.FIXATION_STABILITY)

        return is_fixation, {
            'mean_yaw': round(mean_yaw, 1),
            'stability': round(yaw_std, 1)
        }

    def detect_quick_glance(self, yaw_history):
        """模式3: 快速扫视检测 - 短时间内大幅度转动"""
        if len(yaw_history) < 10:
            return False, {}

        recent = list(yaw_history)[-30:]  # 最近1秒
        quick_count = 0

        for i in range(1, len(recent)):
            speed = abs(recent[i] - recent[i-1])
            if speed >= self.cfg.QUICK_GLANCE_SPEED:
                quick_count += 1

        is_quick = quick_count >= self.cfg.QUICK_GLANCE_COUNT

        return is_quick, {
            'quick_count': quick_count
        }

    def detect_loitering(self, position_history):
        """模式4: 徘徊检测 - 在某区域停留过久"""
        if len(position_history) < self.cfg.LOITER_DURATION:
            return False, {}

        recent = list(position_history)[-self.cfg.LOITER_DURATION:]
        positions = np.array(recent)

        # 计算移动范围
        x_range = np.max(positions[:, 0]) - np.min(positions[:, 0])
        y_range = np.max(positions[:, 1]) - np.min(positions[:, 1])
        move_range = np.sqrt(x_range**2 + y_range**2)

        is_loitering = move_range < self.cfg.LOITER_MOVE_THRESHOLD

        return is_loitering, {
            'move_range': round(move_range, 1)
        }

    def analyze(self, yaw_history, position_history):
        """综合分析所有模式"""
        results = {
            'scanning': {'detected': False, 'details': {}},
            'fixation': {'detected': False, 'details': {}},
            'quick_glance': {'detected': False, 'details': {}},
            'loitering': {'detected': False, 'details': {}},
            'suspicious': False,
            'score': 0,
            'patterns': []
        }

        if len(yaw_history) < self.cfg.MIN_TRACK_FRAMES:
            return results

        # 检测各模式
        scan_det, scan_info = self.detect_scanning(yaw_history)
        fix_det, fix_info = self.detect_fixation(yaw_history)
        quick_det, quick_info = self.detect_quick_glance(yaw_history)
        loit_det, loit_info = self.detect_loitering(position_history)

        results['scanning'] = {'detected': scan_det, 'details': scan_info}
        results['fixation'] = {'detected': fix_det, 'details': fix_info}
        results['quick_glance'] = {'detected': quick_det, 'details': quick_info}
        results['loitering'] = {'detected': loit_det, 'details': loit_info}

        # 计算可疑评分
        score = 0
        patterns = []

        if scan_det:
            score += 40
            patterns.append('SCAN')
        if fix_det:
            score += 30
            patterns.append('FIX')
        if quick_det:
            score += 20
            patterns.append('QUICK')
        if loit_det:
            score += 25
            patterns.append('LOIT')

        results['score'] = score
        results['patterns'] = patterns
        results['suspicious'] = score >= 40  # 评分>=40判定为可疑

        return results


class SuspiciousDetectorV6:
    def __init__(self):
        self.cfg = Config()
        self.analyzer = MultiPatternAnalyzer(self.cfg)
        self.head_pose = HeadPoseEstimator()

        print("=" * 60)
        print("  可疑行为检测系统 V6 (多维度分析)")
        print("=" * 60)
        print("检测模式:")
        print(f"  1. 频繁张望: yaw范围>{self.cfg.SCANNING_YAW_RANGE}° + 变向>{self.cfg.SCANNING_DIR_CHANGES}次")
        print(f"  2. 固定凝视: 偏离>{self.cfg.FIXATION_YAW_THRESHOLD}° + 持续>{self.cfg.FIXATION_DURATION/30:.1f}秒")
        print(f"  3. 快速扫视: 转速>{self.cfg.QUICK_GLANCE_SPEED}°/帧 + 次数>{self.cfg.QUICK_GLANCE_COUNT}")
        print(f"  4. 徘徊检测: 停留>{self.cfg.LOITER_DURATION/30:.1f}秒 + 移动<{self.cfg.LOITER_MOVE_THRESHOLD}px")
        print(f"  可疑阈值: 评分>=40")
        print(f"输出目录: {self.cfg.OUTPUT_DIR}")

        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.cfg.OUTPUT_DIR, "samples_v6"), exist_ok=True)

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

        output_path = os.path.join(self.cfg.OUTPUT_DIR, f"{video_name}_v6.mp4")
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        # 跟踪数据
        track_yaws = defaultdict(lambda: deque(maxlen=300))
        track_positions = defaultdict(lambda: deque(maxlen=300))
        track_analysis = {}
        track_suspicious = {}
        track_last_yaw = {}
        track_warning_count = defaultdict(int)
        track_pattern_count = defaultdict(lambda: defaultdict(int))

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
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    # 记录位置
                    track_positions[tid].append((cx, cy))

                    # 头部姿态估计
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
                        analysis = self.analyzer.analyze(
                            list(track_yaws[tid]),
                            list(track_positions[tid])
                        )
                        track_analysis[tid] = analysis
                        track_suspicious[tid] = analysis['suspicious']

                        if analysis['suspicious']:
                            track_warning_count[tid] += 1
                            for p in analysis['patterns']:
                                track_pattern_count[tid][p] += 1

                    is_suspicious = track_suspicious.get(tid, False)
                    analysis = track_analysis.get(tid, {})

                    # 绘制
                    if is_suspicious:
                        body_color = (0, 0, 255)
                        red_count += 1
                    else:
                        body_color = (0, 255, 0)
                        green_count += 1

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), body_color, 2)

                    if head_bbox:
                        hx1, hy1, hx2, hy2 = head_bbox
                        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), (0, 255, 255), 2)

                    # 标签
                    display_yaw = yaw if yaw is not None else track_last_yaw.get(tid)
                    score = analysis.get('score', 0)
                    label = f"ID:{tid}"
                    if display_yaw is not None:
                        label += f" Y:{display_yaw:.0f}"
                    label += f" S:{score}"

                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (x1, y1-th-8), (x1+tw+4, y1), body_color, -1)
                    cv2.putText(annotated, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                    # 显示检测到的模式
                    if is_suspicious and analysis.get('patterns'):
                        patterns_str = '+'.join(analysis['patterns'])
                        cv2.putText(annotated, f"[{patterns_str}]",
                                   (x1, y2+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            # 状态栏
            cv2.rectangle(annotated, (0, 0), (width, 45), (40, 40, 40), -1)
            cv2.putText(annotated, f"V6 Frame {frame_idx} | Normal:{green_count} Suspicious:{red_count}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 图例
            cv2.rectangle(annotated, (width-350, 8), (width-330, 28), (0, 255, 0), -1)
            cv2.putText(annotated, "Normal", (width-325, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.rectangle(annotated, (width-250, 8), (width-230, 28), (0, 0, 255), -1)
            cv2.putText(annotated, "Susp", (width-225, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.rectangle(annotated, (width-170, 8), (width-150, 28), (0, 255, 255), -1)
            cv2.putText(annotated, "Head", (width-145, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(annotated, "S=Score", (width-100, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

            if red_count > 0:
                cv2.rectangle(annotated, (width//2-50, 5), (width//2+50, 40), (0, 0, 200), -1)
                cv2.putText(annotated, "ALERT!", (width//2-38, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 左上角警告统计面板
            if track_warning_count:
                sorted_warnings = sorted(track_warning_count.items(), key=lambda x: x[1], reverse=True)[:6]
                panel_h = 50 + len(sorted_warnings) * 25
                cv2.rectangle(annotated, (0, 50), (200, 50 + panel_h), (0, 0, 0), -1)
                cv2.rectangle(annotated, (0, 50), (200, 50 + panel_h), (100, 100, 100), 1)
                cv2.putText(annotated, "Warning Stats", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                for i, (tid, count) in enumerate(sorted_warnings):
                    y_pos = 95 + i * 25
                    color = (0, 0, 255) if track_suspicious.get(tid, False) else (200, 200, 200)
                    patterns = track_pattern_count.get(tid, {})
                    pattern_str = ','.join([f"{k}:{v}" for k,v in patterns.items()][:3])
                    cv2.putText(annotated, f"ID{tid}: {count}w", (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                    cv2.putText(annotated, pattern_str, (85, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150,150,150), 1)

                total_warns = sum(track_warning_count.values())
                cv2.putText(annotated, f"Total: {total_warns}", (10, 50 + panel_h - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            writer.write(annotated)

            if frame_idx in sample_frames:
                sample_path = os.path.join(self.cfg.OUTPUT_DIR, "samples_v6", f"{video_name}_frame_{frame_idx:05d}.jpg")
                cv2.imwrite(sample_path, annotated)
                print(f"  Frame {frame_idx}: N:{green_count} S:{red_count}")

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

        # 输出详细警告统计
        if track_warning_count:
            print(f"\n  警告统计 (按ID):")
            sorted_warnings = sorted(track_warning_count.items(), key=lambda x: x[1], reverse=True)
            for tid, count in sorted_warnings:
                patterns = track_pattern_count.get(tid, {})
                pattern_str = ', '.join([f"{k}:{v}" for k,v in patterns.items()])
                print(f"    ID {tid}: {count} 次警告 [{pattern_str}]")
            print(f"  总警告次数: {sum(track_warning_count.values())}")

        return {
            'video': video_name,
            'total': n_total,
            'suspicious': n_susp,
            'warning_counts': dict(track_warning_count),
            'pattern_counts': {tid: dict(p) for tid, p in track_pattern_count.items()}
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

        # 汇总所有视频的模式统计
        print("\n" + "-" * 40)
        print("模式统计汇总:")
        print("-" * 40)
        all_patterns = defaultdict(int)
        for r in results:
            for tid, patterns in r.get('pattern_counts', {}).items():
                for p, c in patterns.items():
                    all_patterns[p] += c
        for p, c in sorted(all_patterns.items(), key=lambda x: x[1], reverse=True):
            print(f"  {p}: {c} 次")

        return results


if __name__ == "__main__":
    SuspiciousDetectorV6().run()
    print("\n完成!")
