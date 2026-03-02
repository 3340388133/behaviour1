#!/usr/bin/env python3
"""
可疑张望行为检测系统 - 最终版本
基于已有的姿态数据，直接在已追踪视频上标注可疑行为

功能：
1. 读取已有的pose_output姿态数据
2. 分析可疑张望行为
3. 在已追踪视频上添加可疑标注
4. 生成最终输出视频
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
    # 数据路径
    DATA_ROOT = "/root/autodl-tmp/behaviour/data"
    POSE_DATA_DIR = "/root/autodl-tmp/behaviour/data/pose_output"
    TRACKED_DATA_DIR = "/root/autodl-tmp/behaviour/data/tracked_output"
    OUTPUT_DIR = "/root/autodl-tmp/behaviour/SuspiciousGazeDetection/output"

    # 视频配置
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

    # ===== 可疑行为判定参数 =====
    YAW_CHANGE_THRESHOLD = 20.0
    YAW_VARIANCE_THRESHOLD = 100.0
    YAW_RANGE_THRESHOLD = 40.0
    MIN_DIRECTION_CHANGES = 2
    SUSPICIOUS_SCORE_THRESHOLD = 0.35
    MIN_TRACK_LENGTH = 5

    # 处理参数
    MAX_FRAMES_PER_VIDEO = 5000
    SAVE_VIDEO = True


# ==================== 行为分析器 ====================
class GazeAnalyzer:
    def __init__(self, cfg):
        self.cfg = cfg

    def analyze_track(self, poses, yaw_offset=0.0):
        """分析单个track的姿态序列"""
        if len(poses) < self.cfg.MIN_TRACK_LENGTH:
            return {
                'suspicious': False,
                'score': 0.0,
                'reason': 'insufficient_data',
                'details': {}
            }

        # 提取并标准化yaw角度
        yaws = []
        for p in poses:
            yaw = p['yaw'] - yaw_offset
            while yaw > 180:
                yaw -= 360
            while yaw < -180:
                yaw += 360
            yaws.append(yaw)

        yaws = np.array(yaws)

        # 计算特征
        yaw_variance = np.var(yaws)
        yaw_range = np.max(yaws) - np.min(yaws)
        yaw_mean = np.mean(yaws)
        yaw_std = np.std(yaws)
        yaw_diff = np.diff(yaws)

        # 方向变化
        signs = np.sign(yaw_diff)
        signs = signs[signs != 0]
        direction_changes = np.sum(np.diff(signs) != 0) if len(signs) > 1 else 0

        # 大幅转头
        large_changes = np.sum(np.abs(yaw_diff) > self.cfg.YAW_CHANGE_THRESHOLD)

        # 转头频率
        duration_sec = len(poses) * 3 / 30.0
        turn_frequency = direction_changes / max(duration_sec, 1.0)

        # 评分
        score = 0.0
        reasons = []

        if yaw_variance > self.cfg.YAW_VARIANCE_THRESHOLD:
            score += 0.25
            reasons.append(f'var:{yaw_variance:.0f}')
        if yaw_variance > self.cfg.YAW_VARIANCE_THRESHOLD * 2:
            score += 0.15

        if yaw_range > self.cfg.YAW_RANGE_THRESHOLD:
            score += 0.2
            reasons.append(f'range:{yaw_range:.0f}')
        if yaw_range > self.cfg.YAW_RANGE_THRESHOLD * 2:
            score += 0.1

        if direction_changes >= self.cfg.MIN_DIRECTION_CHANGES:
            score += 0.2
            reasons.append(f'dir:{direction_changes}')

        if large_changes >= 2:
            score += 0.2
            reasons.append(f'large:{large_changes}')

        if turn_frequency > 0.5:
            score += 0.1

        is_suspicious = score >= self.cfg.SUSPICIOUS_SCORE_THRESHOLD

        return {
            'suspicious': is_suspicious,
            'score': min(1.0, score),
            'yaw_variance': float(yaw_variance),
            'yaw_range': float(yaw_range),
            'yaw_mean': float(yaw_mean),
            'yaw_std': float(yaw_std),
            'direction_changes': int(direction_changes),
            'large_changes': int(large_changes),
            'turn_frequency': float(turn_frequency),
            'num_poses': len(poses),
            'reasons': reasons
        }


# ==================== 主系统 ====================
class SuspiciousGazeDetector:
    def __init__(self):
        self.cfg = Config()
        self.analyzer = GazeAnalyzer(self.cfg)

        print("=" * 60)
        print("  可疑张望行为检测系统 - 最终版本")
        print("=" * 60)
        print(f"输出目录: {self.cfg.OUTPUT_DIR}")

        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

    def load_pose_data(self, video_name):
        """加载姿态数据"""
        pose_file = os.path.join(self.cfg.POSE_DATA_DIR, f"{video_name}_poses.json")
        if not os.path.exists(pose_file):
            return None
        with open(pose_file, 'r') as f:
            return json.load(f)

    def load_tracking_data(self, video_name):
        """加载追踪数据"""
        track_file = os.path.join(self.cfg.TRACKED_DATA_DIR, video_name, "tracking_result.json")
        if not os.path.exists(track_file):
            return None
        with open(track_file, 'r') as f:
            return json.load(f)

    def get_tracked_video_path(self, video_name):
        """获取已追踪的视频路径"""
        path = os.path.join(self.cfg.TRACKED_DATA_DIR, video_name, f"{video_name}_tracked.mp4")
        if os.path.exists(path):
            return path
        return None

    def analyze_video(self, video_name, camera_type, yaw_offset):
        """分析单个视频"""
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
                # 提取数字ID
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
                print(f"    {track_id}: score={t['score']:.2f}, var={t['yaw_variance']:.0f}, "
                      f"range={t['yaw_range']:.0f}, dir={t['direction_changes']}")

        return results

    def add_suspicious_overlay(self, video_name, analysis_results):
        """在已追踪视频上添加可疑标注"""
        tracked_video_path = self.get_tracked_video_path(video_name)
        if tracked_video_path is None:
            print(f"  ! 找不到追踪视频")
            return False

        print(f"\n  在追踪视频上添加标注...")

        cap = cv2.VideoCapture(tracked_video_path)
        if not cap.isOpened():
            print(f"  ! 无法打开视频")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        max_frames = min(total_frames, self.cfg.MAX_FRAMES_PER_VIDEO)

        print(f"  分辨率: {width}x{height}, 帧数: {total_frames}")

        # 可疑track ID集合
        suspicious_ids = analysis_results.get('suspicious_track_ids', set())

        # 输出视频
        output_path = os.path.join(self.cfg.OUTPUT_DIR, f"{video_name}_suspicious.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        suspicious_frame_count = 0

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_has_suspicious = False

            # 扫描画面中的ID文字，查找可疑ID
            # 已追踪视频上已经有 "ID:xxx" 的标注
            # 我们需要在这些ID附近添加"SUSPICIOUS"标记

            # 检测画面中是否有可疑ID（通过颜色变化来标记）
            # 简单方法：在画面顶部添加汇总信息

            # 添加统计信息条
            cv2.rectangle(frame, (0, height - 50), (width, height), (40, 40, 40), -1)

            suspicious_count = len(suspicious_ids)
            total_count = analysis_results['total_tracks']

            info_text = f"Suspicious Tracks: {suspicious_count}/{total_count} | "
            info_text += f"IDs: {sorted(list(suspicious_ids))[:15]}" if suspicious_ids else "No suspicious behavior detected"

            cv2.putText(frame, info_text, (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 添加警告标志（如果有可疑行为）
            if suspicious_ids:
                cv2.rectangle(frame, (width - 180, 5), (width - 5, 35), (0, 0, 200), -1)
                cv2.putText(frame, "ALERT: Suspicious!", (width - 175, 27),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            writer.write(frame)
            frame_idx += 1

            if frame_idx % 1000 == 0:
                print(f"  进度: {frame_idx}/{max_frames} ({100*frame_idx/max_frames:.1f}%)")

        cap.release()
        writer.release()

        print(f"  输出视频: {output_path}")
        return True

    def generate_realtime_detection_video(self, video_name, camera_type, analysis_results):
        """使用YOLOv8实时检测生成标注视频"""
        from ultralytics import YOLO

        # 获取原始视频路径
        if camera_type == "front":
            video_path = os.path.join(self.cfg.DATA_ROOT, "raw_videos/正机位", f"{video_name}.mp4")
        else:
            video_path = os.path.join(self.cfg.DATA_ROOT, "raw_videos/侧机位", f"{video_name}.MP4")

        if not os.path.exists(video_path):
            print(f"  ! 找不到原始视频: {video_path}")
            return False

        print(f"\n  使用YOLOv8生成实时标注视频...")

        # 加载YOLOv8
        model_path = "/root/autodl-tmp/behaviour/yolov8m.pt"
        if not os.path.exists(model_path):
            model_path = "yolov8m.pt"
        detector = YOLO(model_path)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        max_frames = min(total_frames, self.cfg.MAX_FRAMES_PER_VIDEO)
        print(f"  分辨率: {width}x{height}, 帧数: {total_frames}, 处理: {max_frames}")

        # 可疑track ID集合
        suspicious_ids = analysis_results.get('suspicious_track_ids', set())

        # 加载追踪数据，获取track_id -> frames映射
        tracking_data = self.load_tracking_data(video_name)
        track_frame_map = {}  # frame_num -> [track_ids]
        if tracking_data:
            tracks_list = tracking_data.get('tracks', [])
            for track in tracks_list:
                track_id = track.get('track_id')
                frames = track.get('frames', [])
                for frame_num in frames:
                    if frame_num not in track_frame_map:
                        track_frame_map[frame_num] = []
                    track_frame_map[frame_num].append(track_id)

        # 输出视频
        output_path = os.path.join(self.cfg.OUTPUT_DIR, f"{video_name}_suspicious.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 简单追踪器
        from collections import defaultdict
        tracks = {}
        next_id = 1
        max_age = 30

        frame_idx = 0
        suspicious_frame_count = 0

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # 每2帧检测一次
            if frame_idx % 2 == 0:
                results = detector(frame, verbose=False, classes=[0])[0]
                detections = []
                for box in results.boxes:
                    if box.conf[0] > 0.5:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        detections.append([x1, y1, x2, y2, conf])

                # 简单IoU匹配
                matched = set()
                current_tracks = []

                for tid, track in list(tracks.items()):
                    best_iou = 0
                    best_idx = -1
                    for i, det in enumerate(detections):
                        if i in matched:
                            continue
                        # 计算IoU
                        b1, b2 = track['box'], det[:4]
                        xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                        xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                        inter = max(0, xi2-xi1) * max(0, yi2-yi1)
                        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
                        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
                        iou = inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0

                        if iou > best_iou and iou > 0.3:
                            best_iou = iou
                            best_idx = i

                    if best_idx >= 0:
                        track['box'] = detections[best_idx][:4]
                        track['age'] = 0
                        matched.add(best_idx)
                        current_tracks.append((tid, track['box']))
                    else:
                        track['age'] += 1

                # 清理旧轨迹
                tracks = {k: v for k, v in tracks.items() if v['age'] <= max_age}

                # 创建新轨迹
                for i, det in enumerate(detections):
                    if i not in matched:
                        tracks[next_id] = {'box': det[:4], 'age': 0}
                        current_tracks.append((next_id, det[:4]))
                        next_id += 1

            frame_has_suspicious = False

            # 获取当前帧的原始track_ids（从追踪数据）
            original_track_ids = track_frame_map.get(frame_idx, [])

            # 绘制检测框
            for tid, bbox in current_tracks if 'current_tracks' in dir() else []:
                x1, y1, x2, y2 = map(int, bbox)

                # 检查是否是可疑track
                # 尝试匹配原始track_id
                is_suspicious = False
                matched_original_id = None

                for orig_id in original_track_ids:
                    if orig_id in suspicious_ids:
                        is_suspicious = True
                        matched_original_id = orig_id
                        break

                if is_suspicious:
                    color = (0, 0, 255)  # 红色
                    frame_has_suspicious = True
                else:
                    color = (0, 200, 0)  # 绿色

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"ID:{tid}"
                if matched_original_id:
                    label = f"ID:{matched_original_id}"

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
            status = f"Frame: {frame_idx}/{max_frames} | Suspicious Tracks: {len(suspicious_ids)} | Suspicious Frames: {suspicious_frame_count}"
            cv2.putText(frame, status, (10, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 警告标志
            if suspicious_ids:
                cv2.rectangle(frame, (width - 180, 5), (width - 5, 35), (0, 0, 200), -1)
                cv2.putText(frame, "ALERT!", (width - 120, 27),
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
        """运行全部处理"""
        print("\n" + "=" * 60)
        print("开始处理所有视频...")
        print("=" * 60)

        all_results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'yaw_change_threshold': self.cfg.YAW_CHANGE_THRESHOLD,
                'yaw_variance_threshold': self.cfg.YAW_VARIANCE_THRESHOLD,
                'suspicious_score_threshold': self.cfg.SUSPICIOUS_SCORE_THRESHOLD
            },
            'videos': []
        }

        for camera_type, camera_cfg in self.cfg.VIDEOS.items():
            yaw_offset = camera_cfg['yaw_offset']

            for video_name in camera_cfg['videos']:
                # 分析姿态数据
                analysis = self.analyze_video(video_name, camera_type, yaw_offset)

                if analysis:
                    # 转换set为list以便JSON序列化
                    analysis_for_save = {k: v for k, v in analysis.items() if k != 'suspicious_track_ids'}
                    all_results['videos'].append(analysis_for_save)

                    # 生成标注视频
                    if self.cfg.SAVE_VIDEO:
                        self.generate_realtime_detection_video(
                            video_name, camera_type, analysis
                        )

        # 保存汇总结果
        self._save_summary(all_results)

        return all_results

    def _save_summary(self, all_results):
        """保存汇总结果"""
        print("\n" + "=" * 60)
        print("处理完成 - 汇总报告")
        print("=" * 60)

        total_tracks = sum(v['total_tracks'] for v in all_results['videos'])
        total_suspicious = sum(len(v['suspicious_tracks']) for v in all_results['videos'])

        print(f"\n总视频数: {len(all_results['videos'])}")
        print(f"总轨迹数: {total_tracks}")
        print(f"可疑轨迹: {total_suspicious}")

        if total_tracks > 0:
            ratio = (total_suspicious / total_tracks) * 100
            print(f"可疑比例: {ratio:.1f}%")

        # 按机位统计
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

        # 保存JSON
        output_file = os.path.join(self.cfg.OUTPUT_DIR, "final_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n结果已保存: {output_file}")
        print(f"视频输出目录: {self.cfg.OUTPUT_DIR}")


# ==================== 主程序 ====================
if __name__ == "__main__":
    detector = SuspiciousGazeDetector()
    detector.run()
    print("\n" + "=" * 60)
    print("全部处理完成！")
    print("=" * 60)
