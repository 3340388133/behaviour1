#!/usr/bin/env python3
"""
可疑张望行为检测系统 V3 - 更严格的判定标准
只有真正频繁左右张望的人才会被标记为可疑
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
    POSE_DATA_DIR = "/root/autodl-tmp/behaviour/data/pose_output"
    TRACKED_DATA_DIR = "/root/autodl-tmp/behaviour/data/tracked_output"
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

    # ===== 更严格的判定参数 =====
    SAMPLE_RATE = 3
    FPS = 30

    # 转头检测 - 更严格
    TURN_ANGLE_THRESHOLD = 35.0      # 提高：单次转头至少35度才算

    # 可疑行为判定 - 大幅提高阈值
    MIN_TURNS_FOR_SUSPICIOUS = 8     # 提高：至少8次转头
    SUSPICIOUS_TURN_FREQUENCY = 1.5  # 提高：每秒至少1.5次转头才算可疑
    MIN_DIRECTION_CHANGES = 6        # 提高：至少6次方向变化

    # 可疑分数阈值
    SUSPICIOUS_SCORE_THRESHOLD = 0.6  # 提高：分数要达到0.6才算可疑

    MIN_TRACK_LENGTH = 15            # 提高：至少15帧才分析
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


def smooth_angles(angles):
    if len(angles) < 2:
        return angles
    smoothed = [angles[0]]
    for i in range(1, len(angles)):
        diff = angle_diff(smoothed[-1], angles[i])
        smoothed.append(smoothed[-1] + diff)
    return smoothed


class GazeAnalyzer:
    def __init__(self, cfg):
        self.cfg = cfg

    def detect_significant_turns(self, yaws):
        """检测显著的转头事件（大于阈值的转头）"""
        if len(yaws) < 3:
            return []

        turns = []
        yaws = np.array(yaws)

        # 计算相邻帧的角度变化
        diffs = []
        for i in range(1, len(yaws)):
            diff = angle_diff(yaws[i-1], yaws[i])
            diffs.append(diff)
        diffs = np.array(diffs)

        # 检测连续同方向的转头
        i = 0
        while i < len(diffs):
            if abs(diffs[i]) < 5:
                i += 1
                continue

            direction = np.sign(diffs[i])
            cumulative = diffs[i]
            start_idx = i

            j = i + 1
            while j < len(diffs) and np.sign(diffs[j]) == direction:
                cumulative += diffs[j]
                j += 1

            # 只有大于阈值的转头才计入
            if abs(cumulative) >= self.cfg.TURN_ANGLE_THRESHOLD:
                turns.append({
                    'start': start_idx,
                    'end': j,
                    'angle': cumulative,
                    'direction': 'left' if direction > 0 else 'right'
                })

            i = j

        return turns

    def count_direction_changes(self, yaws):
        """计算显著的方向变化次数"""
        if len(yaws) < 5:
            return 0

        smoothed = smooth_angles(yaws)
        changes = np.diff(smoothed)

        # 只计算显著的方向变化（变化量>10度）
        significant_changes = 0
        prev_direction = 0

        for i in range(len(changes)):
            if abs(changes[i]) > 10:  # 忽略小抖动
                current_direction = np.sign(changes[i])
                if prev_direction != 0 and current_direction != prev_direction:
                    significant_changes += 1
                prev_direction = current_direction

        return significant_changes

    def analyze_track(self, poses, yaw_offset=0.0):
        if len(poses) < self.cfg.MIN_TRACK_LENGTH:
            return {
                'suspicious': False,
                'score': 0.0,
                'reason': 'too_short'
            }

        yaws = [normalize_angle(p['yaw'] - yaw_offset) for p in poses]
        smoothed_yaws = smooth_angles(yaws)

        # 检测显著转头
        turns = self.detect_significant_turns(smoothed_yaws)

        # 计算方向变化
        direction_changes = self.count_direction_changes(smoothed_yaws)

        # 计算时间
        duration_sec = len(poses) * self.cfg.SAMPLE_RATE / self.cfg.FPS

        # 计算转头频率
        turn_frequency = len(turns) / max(duration_sec, 1.0)

        # 计算yaw标准差（使用smoothed版本）
        yaw_std = np.std(smoothed_yaws)

        # 严格评分
        score = 0.0
        reasons = []

        # 1. 转头频率评分（权重最高）
        if turn_frequency >= self.cfg.SUSPICIOUS_TURN_FREQUENCY:
            score += 0.35
            reasons.append(f'high_freq:{turn_frequency:.1f}/s')
        elif turn_frequency >= 1.0:
            score += 0.15

        # 2. 转头次数评分
        if len(turns) >= self.cfg.MIN_TURNS_FOR_SUSPICIOUS:
            score += 0.25
            reasons.append(f'many_turns:{len(turns)}')
        elif len(turns) >= 5:
            score += 0.1

        # 3. 方向变化评分
        if direction_changes >= self.cfg.MIN_DIRECTION_CHANGES:
            score += 0.25
            reasons.append(f'dir_changes:{direction_changes}')
        elif direction_changes >= 4:
            score += 0.1

        # 4. 活动程度评分（yaw标准差）
        if yaw_std > 50:
            score += 0.15
            reasons.append(f'active:{yaw_std:.0f}')

        is_suspicious = score >= self.cfg.SUSPICIOUS_SCORE_THRESHOLD

        return {
            'suspicious': is_suspicious,
            'score': round(score, 2),
            'num_turns': len(turns),
            'turn_frequency': round(turn_frequency, 2),
            'direction_changes': direction_changes,
            'yaw_std': round(yaw_std, 1),
            'num_poses': len(poses),
            'duration_sec': round(duration_sec, 1),
            'reasons': reasons
        }


class SuspiciousGazeDetector:
    def __init__(self):
        self.cfg = Config()
        self.analyzer = GazeAnalyzer(self.cfg)

        print("=" * 60)
        print("  可疑张望行为检测系统 V3 (严格版)")
        print("=" * 60)
        print(f"判定标准:")
        print(f"  - 转头角度阈值: {self.cfg.TURN_ANGLE_THRESHOLD}°")
        print(f"  - 最小转头次数: {self.cfg.MIN_TURNS_FOR_SUSPICIOUS}")
        print(f"  - 转头频率阈值: {self.cfg.SUSPICIOUS_TURN_FREQUENCY}/s")
        print(f"  - 可疑分数阈值: {self.cfg.SUSPICIOUS_SCORE_THRESHOLD}")
        print(f"输出目录: {self.cfg.OUTPUT_DIR}")

        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

    def load_pose_data(self, video_name):
        pose_file = os.path.join(self.cfg.POSE_DATA_DIR, f"{video_name}_poses.json")
        if not os.path.exists(pose_file):
            return None
        with open(pose_file, 'r') as f:
            return json.load(f)

    def load_tracking_data(self, video_name):
        track_file = os.path.join(self.cfg.TRACKED_DATA_DIR, video_name, "tracking_result.json")
        if not os.path.exists(track_file):
            return None
        with open(track_file, 'r') as f:
            return json.load(f)

    def analyze_video(self, video_name, camera_type, yaw_offset):
        print(f"\n{'='*50}")
        print(f"分析: {video_name} ({'正机位' if camera_type == 'front' else '侧机位'})")
        print(f"{'='*50}")

        pose_data = self.load_pose_data(video_name)
        if pose_data is None:
            print(f"  ! 姿态数据不存在")
            return None

        tracks = pose_data.get('tracks', {})
        print(f"  总轨迹数: {len(tracks)}")

        results = {
            'video_name': video_name,
            'camera_type': camera_type,
            'yaw_offset': yaw_offset,
            'total_tracks': len(tracks),
            'tracks': {},
            'suspicious_tracks': [],
            'suspicious_track_ids': set()
        }

        for track_id, track_data in tracks.items():
            poses = track_data.get('poses', [])
            analysis = self.analyzer.analyze_track(poses, yaw_offset)
            results['tracks'][track_id] = analysis

            if analysis['suspicious']:
                results['suspicious_tracks'].append(track_id)
                try:
                    num_id = int(track_id.replace('track_', ''))
                    results['suspicious_track_ids'].add(num_id)
                except:
                    pass

        suspicious_ratio = len(results['suspicious_tracks']) / len(tracks) * 100 if tracks else 0
        print(f"  可疑轨迹: {len(results['suspicious_tracks'])} / {len(tracks)} ({suspicious_ratio:.1f}%)")

        if results['suspicious_tracks']:
            print(f"\n  可疑轨迹详情:")
            for track_id in results['suspicious_tracks'][:10]:
                t = results['tracks'][track_id]
                print(f"    {track_id}: score={t['score']}, turns={t['num_turns']}, "
                      f"freq={t['turn_frequency']}/s, reasons={t['reasons']}")

        return results

    def generate_video(self, video_name, camera_type, analysis_results):
        from ultralytics import YOLO

        if camera_type == "front":
            video_path = os.path.join(self.cfg.DATA_ROOT, "raw_videos/正机位", f"{video_name}.mp4")
        else:
            video_path = os.path.join(self.cfg.DATA_ROOT, "raw_videos/侧机位", f"{video_name}.MP4")

        if not os.path.exists(video_path):
            print(f"  ! 找不到视频: {video_path}")
            return False

        print(f"\n  生成标注视频...")

        model = YOLO("/root/autodl-tmp/behaviour/yolov8m.pt")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_frames = min(total_frames, self.cfg.MAX_FRAMES_PER_VIDEO)

        print(f"  分辨率: {width}x{height}, 处理帧数: {max_frames}")

        suspicious_ids = analysis_results.get('suspicious_track_ids', set())

        output_path = os.path.join(self.cfg.OUTPUT_DIR, f"{video_name}_suspicious.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # frame->track映射
        tracking_data = self.load_tracking_data(video_name)
        frame_to_tracks = defaultdict(list)
        if tracking_data:
            for track in tracking_data.get('tracks', []):
                track_id = track.get('track_id')
                for frame_num in track.get('frames', []):
                    frame_to_tracks[frame_num].append(track_id)

        # yaw数据
        pose_data = self.load_pose_data(video_name)
        track_yaw_by_frame = {}
        if pose_data:
            for track_id, track_info in pose_data.get('tracks', {}).items():
                try:
                    num_id = int(track_id.replace('track_', ''))
                except:
                    continue
                for p in track_info.get('poses', []):
                    track_yaw_by_frame[(num_id, p['frame'])] = p['yaw']

        frame_idx = 0
        suspicious_frame_count = 0

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, persist=True, verbose=False, classes=[0], conf=0.5)

            frame_has_suspicious = False
            current_frame_tracks = frame_to_tracks.get(frame_idx, [])

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, tid in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)

                    # 检查可疑
                    is_suspicious = tid in suspicious_ids
                    for orig_id in current_frame_tracks:
                        if orig_id in suspicious_ids:
                            is_suspicious = True
                            break

                    # 颜色：可疑红色，正常绿色
                    if is_suspicious:
                        color = (0, 0, 255)
                        frame_has_suspicious = True
                    else:
                        color = (0, 255, 0)  # 纯绿色

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # 获取yaw
                    yaw_str = ""
                    for check_id in [tid] + current_frame_tracks:
                        for offset in range(-3, 4):
                            if (check_id, frame_idx + offset) in track_yaw_by_frame:
                                yaw_str = f" Y:{track_yaw_by_frame[(check_id, frame_idx + offset)]:.0f}"
                                break
                        if yaw_str:
                            break

                    label = f"ID:{tid}{yaw_str}"
                    cv2.putText(frame, label, (x1, y1 - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if is_suspicious:
                        cv2.putText(frame, "SUSPICIOUS!", (x1, y2 + 18),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 255), 3)

            if frame_has_suspicious:
                suspicious_frame_count += 1

            # 状态栏
            cv2.rectangle(frame, (0, 0), (width, 40), (40, 40, 40), -1)
            total_tracks = analysis_results['total_tracks']
            n_suspicious = len(suspicious_ids)
            status = f"Frame: {frame_idx}/{max_frames} | Suspicious: {n_suspicious}/{total_tracks} ({100*n_suspicious/total_tracks:.0f}%)"
            cv2.putText(frame, status, (10, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 图例
            cv2.rectangle(frame, (width-200, 8), (width-180, 28), (0, 255, 0), -1)
            cv2.putText(frame, "Normal", (width-175, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.rectangle(frame, (width-100, 8), (width-80, 28), (0, 0, 255), -1)
            cv2.putText(frame, "Suspicious", (width-75, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

            if n_suspicious > 0 and frame_has_suspicious:
                cv2.rectangle(frame, (width//2-60, 5), (width//2+60, 35), (0, 0, 200), -1)
                cv2.putText(frame, "ALERT!", (width//2-40, 27),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            writer.write(frame)
            frame_idx += 1

            if frame_idx % 500 == 0:
                print(f"  进度: {frame_idx}/{max_frames} ({100*frame_idx/max_frames:.0f}%)")

        cap.release()
        writer.release()

        print(f"  输出: {output_path}")
        return True

    def run(self):
        print("\n" + "=" * 60)
        print("开始处理...")
        print("=" * 60)

        all_results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'turn_angle_threshold': self.cfg.TURN_ANGLE_THRESHOLD,
                'min_turns': self.cfg.MIN_TURNS_FOR_SUSPICIOUS,
                'suspicious_frequency': self.cfg.SUSPICIOUS_TURN_FREQUENCY,
                'score_threshold': self.cfg.SUSPICIOUS_SCORE_THRESHOLD
            },
            'videos': []
        }

        for camera_type, camera_cfg in self.cfg.VIDEOS.items():
            yaw_offset = camera_cfg['yaw_offset']

            for video_name in camera_cfg['videos']:
                analysis = self.analyze_video(video_name, camera_type, yaw_offset)

                if analysis:
                    analysis_save = {k: v for k, v in analysis.items() if k != 'suspicious_track_ids'}
                    all_results['videos'].append(analysis_save)

                    if self.cfg.SAVE_VIDEO:
                        self.generate_video(video_name, camera_type, analysis)

        self._save_summary(all_results)
        return all_results

    def _save_summary(self, all_results):
        print("\n" + "=" * 60)
        print("处理完成 - 汇总")
        print("=" * 60)

        total_tracks = sum(v['total_tracks'] for v in all_results['videos'])
        total_suspicious = sum(len(v['suspicious_tracks']) for v in all_results['videos'])

        print(f"\n总轨迹: {total_tracks}")
        print(f"可疑轨迹: {total_suspicious} ({100*total_suspicious/total_tracks:.1f}%)")

        for camera_type in ['front', 'side']:
            videos = [v for v in all_results['videos'] if v['camera_type'] == camera_type]
            if videos:
                label = "正机位" if camera_type == "front" else "侧机位"
                tracks = sum(v['total_tracks'] for v in videos)
                suspicious = sum(len(v['suspicious_tracks']) for v in videos)
                print(f"\n{label}: {suspicious}/{tracks} ({100*suspicious/tracks:.1f}%)")

        output_file = os.path.join(self.cfg.OUTPUT_DIR, "final_results_v3.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n结果: {output_file}")


if __name__ == "__main__":
    detector = SuspiciousGazeDetector()
    detector.run()
    print("\n完成!")
