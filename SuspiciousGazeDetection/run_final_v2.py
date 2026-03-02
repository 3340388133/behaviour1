#!/usr/bin/env python3
"""
可疑张望行为检测系统 - 最终版本 V2
改进的角度处理和可疑行为判定

核心改进:
1. 处理角度跳变问题（-180/180边界）
2. 基于短时间窗口内的频繁转头来判定
3. 更精确的可疑行为特征提取
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

# ==================== 配置 ====================
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

    # ===== 可疑行为判定参数（重新设计）=====
    # 基于短时间窗口内的转头行为
    WINDOW_SECONDS = 3.0             # 分析窗口（秒）
    SAMPLE_RATE = 3                  # 姿态采样率
    FPS = 30                         # 视频帧率

    # 转头检测参数
    TURN_ANGLE_THRESHOLD = 25.0      # 单次转头角度阈值
    TURN_VELOCITY_THRESHOLD = 15.0   # 转头速度阈值（度/帧）

    # 可疑行为判定
    MIN_TURNS_FOR_SUSPICIOUS = 3     # 窗口内最小转头次数
    SUSPICIOUS_TURN_FREQUENCY = 0.5  # 可疑转头频率（次/秒）

    MIN_TRACK_LENGTH = 10            # 最小轨迹长度
    MAX_FRAMES_PER_VIDEO = 5000
    SAVE_VIDEO = True


def normalize_angle(angle):
    """标准化角度到-180到180"""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def angle_diff(a1, a2):
    """计算两个角度之间的最短差值"""
    diff = a2 - a1
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff


def smooth_angles(angles, window=3):
    """平滑角度序列，处理跳变"""
    if len(angles) < window:
        return angles

    smoothed = []
    for i, a in enumerate(angles):
        if i == 0:
            smoothed.append(a)
        else:
            # 检查与前一个值的差异
            diff = angle_diff(smoothed[-1], a)
            smoothed.append(smoothed[-1] + diff)

    return smoothed


# ==================== 行为分析器 ====================
class GazeAnalyzer:
    def __init__(self, cfg):
        self.cfg = cfg
        # 计算窗口大小（帧数）
        self.window_frames = int(cfg.WINDOW_SECONDS * cfg.FPS / cfg.SAMPLE_RATE)

    def detect_turns(self, yaws):
        """检测转头事件"""
        if len(yaws) < 3:
            return []

        turns = []
        yaws = np.array(yaws)

        # 计算角度变化
        diffs = []
        for i in range(1, len(yaws)):
            diff = angle_diff(yaws[i-1], yaws[i])
            diffs.append(diff)
        diffs = np.array(diffs)

        # 检测转头事件（连续同方向的角度变化累积超过阈值）
        i = 0
        while i < len(diffs):
            if abs(diffs[i]) < 3:  # 忽略小变化
                i += 1
                continue

            # 开始一个转头事件
            direction = np.sign(diffs[i])
            cumulative = diffs[i]
            start_idx = i

            # 累积同方向的变化
            j = i + 1
            while j < len(diffs) and np.sign(diffs[j]) == direction:
                cumulative += diffs[j]
                j += 1

            # 检查是否是有效转头
            if abs(cumulative) >= self.cfg.TURN_ANGLE_THRESHOLD:
                turns.append({
                    'start_frame': start_idx,
                    'end_frame': j,
                    'angle': cumulative,
                    'direction': 'left' if direction > 0 else 'right'
                })

            i = j

        return turns

    def detect_direction_changes(self, yaws):
        """检测频繁的方向变化（左右张望）"""
        if len(yaws) < 5:
            return 0, []

        # 平滑角度
        smoothed = smooth_angles(yaws)

        # 计算变化
        changes = []
        for i in range(1, len(smoothed)):
            diff = smoothed[i] - smoothed[i-1]
            changes.append(diff)

        # 检测方向变化点
        direction_changes = []
        for i in range(1, len(changes)):
            if changes[i-1] * changes[i] < 0:  # 符号变化
                if abs(changes[i-1]) > 5 and abs(changes[i]) > 5:  # 忽略小幅抖动
                    direction_changes.append(i)

        return len(direction_changes), direction_changes

    def analyze_track(self, poses, yaw_offset=0.0):
        """分析单个track"""
        if len(poses) < self.cfg.MIN_TRACK_LENGTH:
            return {
                'suspicious': False,
                'score': 0.0,
                'reason': 'insufficient_data'
            }

        # 提取yaw角度并标准化
        yaws = [normalize_angle(p['yaw'] - yaw_offset) for p in poses]

        # 平滑处理
        smoothed_yaws = smooth_angles(yaws)

        # 检测转头事件
        turns = self.detect_turns(smoothed_yaws)

        # 检测方向变化
        num_direction_changes, change_points = self.detect_direction_changes(smoothed_yaws)

        # 计算统计特征
        yaw_array = np.array(smoothed_yaws)
        yaw_std = np.std(yaw_array)
        yaw_range = np.max(yaw_array) - np.min(yaw_array)

        # 计算转头频率
        duration_sec = len(poses) * self.cfg.SAMPLE_RATE / self.cfg.FPS
        turn_frequency = len(turns) / max(duration_sec, 1.0)
        direction_change_frequency = num_direction_changes / max(duration_sec, 1.0)

        # 计算可疑分数
        score = 0.0
        reasons = []

        # 基于转头频率评分
        if turn_frequency >= self.cfg.SUSPICIOUS_TURN_FREQUENCY:
            score += 0.3
            reasons.append(f'turn_freq:{turn_frequency:.2f}/s')

        # 基于方向变化频率评分
        if direction_change_frequency >= 0.3:
            score += 0.25
            reasons.append(f'dir_change:{direction_change_frequency:.2f}/s')

        # 基于总转头次数
        if len(turns) >= self.cfg.MIN_TURNS_FOR_SUSPICIOUS:
            score += 0.2
            reasons.append(f'turns:{len(turns)}')

        # 基于标准差（头部活动程度）
        if yaw_std > 30:
            score += 0.15
            reasons.append(f'std:{yaw_std:.0f}')

        # 基于yaw范围（但要合理，不是跳变造成的）
        if 40 < yaw_range < 150:  # 合理的张望范围
            score += 0.1
            reasons.append(f'range:{yaw_range:.0f}')

        is_suspicious = score >= 0.4

        return {
            'suspicious': is_suspicious,
            'score': min(1.0, score),
            'num_turns': len(turns),
            'turn_frequency': float(turn_frequency),
            'direction_changes': num_direction_changes,
            'direction_change_frequency': float(direction_change_frequency),
            'yaw_std': float(yaw_std),
            'yaw_range': float(yaw_range),
            'num_poses': len(poses),
            'duration_sec': float(duration_sec),
            'reasons': reasons
        }


# ==================== 主系统 ====================
class SuspiciousGazeDetector:
    def __init__(self):
        self.cfg = Config()
        self.analyzer = GazeAnalyzer(self.cfg)

        print("=" * 60)
        print("  可疑张望行为检测系统 V2")
        print("=" * 60)
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

        print(f"  可疑轨迹: {len(results['suspicious_tracks'])} / {len(tracks)}")

        if results['suspicious_tracks']:
            print(f"\n  可疑轨迹详情 (前10个):")
            for track_id in results['suspicious_tracks'][:10]:
                t = results['tracks'][track_id]
                print(f"    {track_id}: score={t['score']:.2f}, turns={t['num_turns']}, "
                      f"turn_freq={t['turn_frequency']:.2f}/s, dir_changes={t['direction_changes']}")

        return results

    def generate_video_with_ultralytics(self, video_name, camera_type, analysis_results):
        """使用ultralytics的track功能生成标注视频"""
        from ultralytics import YOLO

        if camera_type == "front":
            video_path = os.path.join(self.cfg.DATA_ROOT, "raw_videos/正机位", f"{video_name}.mp4")
        else:
            video_path = os.path.join(self.cfg.DATA_ROOT, "raw_videos/侧机位", f"{video_name}.MP4")

        if not os.path.exists(video_path):
            print(f"  ! 找不到原始视频: {video_path}")
            return False

        print(f"\n  生成标注视频...")

        model = YOLO("/root/autodl-tmp/behaviour/yolov8m.pt")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_frames = min(total_frames, self.cfg.MAX_FRAMES_PER_VIDEO)

        print(f"  分辨率: {width}x{height}, 帧率: {fps:.1f}, 处理帧数: {max_frames}")

        suspicious_ids = analysis_results.get('suspicious_track_ids', set())

        output_path = os.path.join(self.cfg.OUTPUT_DIR, f"{video_name}_suspicious.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 加载追踪数据获取frame->track映射
        tracking_data = self.load_tracking_data(video_name)
        frame_to_tracks = defaultdict(list)
        if tracking_data:
            for track in tracking_data.get('tracks', []):
                track_id = track.get('track_id')
                for frame_num in track.get('frames', []):
                    frame_to_tracks[frame_num].append(track_id)

        # 获取姿态数据用于显示yaw
        pose_data = self.load_pose_data(video_name)
        track_yaw_by_frame = {}
        if pose_data:
            for track_id, track_info in pose_data.get('tracks', {}).items():
                try:
                    num_id = int(track_id.replace('track_', ''))
                except:
                    continue
                for p in track_info.get('poses', []):
                    key = (num_id, p['frame'])
                    track_yaw_by_frame[key] = p['yaw']

        frame_idx = 0
        suspicious_frame_count = 0

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # 每帧检测和追踪
            results = model.track(frame, persist=True, verbose=False, classes=[0], conf=0.5)

            frame_has_suspicious = False
            current_frame_tracks = frame_to_tracks.get(frame_idx, [])

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, tid in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)

                    # 检查是否可疑（匹配原始track_id）
                    is_suspicious = False
                    for orig_id in current_frame_tracks:
                        if orig_id in suspicious_ids:
                            is_suspicious = True
                            break

                    # 也检查当前tid是否在可疑列表
                    if tid in suspicious_ids:
                        is_suspicious = True

                    color = (0, 0, 255) if is_suspicious else (0, 200, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # 获取yaw
                    yaw_display = ""
                    for check_frame in range(max(0, frame_idx-5), frame_idx+5):
                        for check_id in [tid] + current_frame_tracks:
                            if (check_id, check_frame) in track_yaw_by_frame:
                                yaw_display = f" Y:{track_yaw_by_frame[(check_id, check_frame)]:.0f}"
                                break
                        if yaw_display:
                            break

                    label = f"ID:{tid}{yaw_display}"
                    cv2.putText(frame, label, (x1, y1 - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if is_suspicious:
                        cv2.putText(frame, "SUSPICIOUS!", (x1, y2 + 18),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 255), 3)
                        frame_has_suspicious = True

            if frame_has_suspicious:
                suspicious_frame_count += 1

            # 状态栏
            cv2.rectangle(frame, (0, 0), (width, 40), (40, 40, 40), -1)
            status = f"Frame: {frame_idx}/{max_frames} | Suspicious: {len(suspicious_ids)} tracks | Suspicious Frames: {suspicious_frame_count}"
            cv2.putText(frame, status, (10, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if suspicious_ids:
                cv2.rectangle(frame, (width - 150, 5), (width - 5, 35), (0, 0, 200), -1)
                cv2.putText(frame, "ALERT!", (width - 100, 27),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            writer.write(frame)
            frame_idx += 1

            if frame_idx % 500 == 0:
                print(f"  进度: {frame_idx}/{max_frames} ({100*frame_idx/max_frames:.1f}%)")

        cap.release()
        writer.release()

        print(f"  输出视频: {output_path}")
        return True

    def run(self):
        print("\n" + "=" * 60)
        print("开始处理所有视频...")
        print("=" * 60)

        all_results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'turn_angle_threshold': self.cfg.TURN_ANGLE_THRESHOLD,
                'suspicious_turn_frequency': self.cfg.SUSPICIOUS_TURN_FREQUENCY,
                'min_turns_for_suspicious': self.cfg.MIN_TURNS_FOR_SUSPICIOUS
            },
            'videos': []
        }

        for camera_type, camera_cfg in self.cfg.VIDEOS.items():
            yaw_offset = camera_cfg['yaw_offset']

            for video_name in camera_cfg['videos']:
                analysis = self.analyze_video(video_name, camera_type, yaw_offset)

                if analysis:
                    analysis_for_save = {k: v for k, v in analysis.items() if k != 'suspicious_track_ids'}
                    all_results['videos'].append(analysis_for_save)

                    if self.cfg.SAVE_VIDEO:
                        self.generate_video_with_ultralytics(
                            video_name, camera_type, analysis
                        )

        self._save_summary(all_results)
        return all_results

    def _save_summary(self, all_results):
        print("\n" + "=" * 60)
        print("处理完成 - 汇总报告")
        print("=" * 60)

        total_tracks = sum(v['total_tracks'] for v in all_results['videos'])
        total_suspicious = sum(len(v['suspicious_tracks']) for v in all_results['videos'])

        print(f"\n总视频数: {len(all_results['videos'])}")
        print(f"总轨迹数: {total_tracks}")
        print(f"可疑轨迹: {total_suspicious}")

        if total_tracks > 0:
            print(f"可疑比例: {100*total_suspicious/total_tracks:.1f}%")

        for camera_type in ['front', 'side']:
            videos = [v for v in all_results['videos'] if v['camera_type'] == camera_type]
            if videos:
                label = "正机位" if camera_type == "front" else "侧机位"
                tracks = sum(v['total_tracks'] for v in videos)
                suspicious = sum(len(v['suspicious_tracks']) for v in videos)
                print(f"\n{label}:")
                print(f"  视频数: {len(videos)}")
                print(f"  轨迹数: {tracks}")
                print(f"  可疑轨迹: {suspicious}")

        output_file = os.path.join(self.cfg.OUTPUT_DIR, "final_results_v2.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n结果已保存: {output_file}")


if __name__ == "__main__":
    detector = SuspiciousGazeDetector()
    detector.run()
    print("\n" + "=" * 60)
    print("全部处理完成！")
    print("=" * 60)
