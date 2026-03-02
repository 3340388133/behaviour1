#!/usr/bin/env python3
"""
可疑张望行为检测系统 - 全自动版本
功能：检测视频中频繁左右张望的可疑人员，输出标注视频
"""

import os
import sys
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 添加6DRepNet路径
sys.path.insert(0, "/root/autodl-tmp/behaviour/6DRepNet-master")

# ==================== 配置 ====================
class Config:
    # 数据路径
    DATA_ROOT = "/root/autodl-tmp/behaviour/data"
    OUTPUT_DIR = "/root/autodl-tmp/behaviour/SuspiciousGazeDetection/output"

    # 视频配置
    VIDEOS = {
        "front": {
            "dir": "raw_videos/正机位",
            "files": ["1.14zz-1.mp4", "1.14zz-2.mp4", "1.14zz-3.mp4", "1.14zz-4.mp4"],
            "yaw_offset": 0.0,  # 正机位不需要偏移
        },
        "side": {
            "dir": "raw_videos/侧机位",
            "files": ["MVI_4537.MP4", "MVI_4538.MP4"],
            "yaw_offset": 90.0,  # 侧机位需要90度偏移
        }
    }

    # YOLOv8检测参数
    DETECTOR_MODEL = "yolov8m.pt"
    CONF_THRESHOLD = 0.5

    # 可疑行为判定参数 - 核心阈值
    YAW_CHANGE_THRESHOLD = 25.0      # 单次大幅转头阈值(度)
    YAW_VARIANCE_THRESHOLD = 150.0   # yaw方差阈值
    YAW_RANGE_THRESHOLD = 50.0       # yaw范围阈值
    MIN_DIRECTION_CHANGES = 3        # 最小方向变化次数
    SUSPICIOUS_SCORE_THRESHOLD = 0.4 # 可疑分数阈值
    ANALYSIS_WINDOW = 30             # 分析窗口(帧数)

    # 追踪参数
    TRACKER_MAX_AGE = 30
    TRACKER_IOU_THRESH = 0.3

    # 处理参数
    PROCESS_EVERY_N_FRAMES = 2       # 每N帧处理一次，加速
    MAX_FRAMES_PER_VIDEO = 3000      # 每个视频最大处理帧数
    SAVE_VIDEO = True

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== 简单追踪器 ====================
class SimpleTracker:
    def __init__(self, max_age=30, iou_thresh=0.3):
        self.max_age = max_age
        self.iou_thresh = iou_thresh
        self.tracks = {}
        self.next_id = 1

    def update(self, detections):
        if len(detections) == 0:
            for t in self.tracks.values():
                t['age'] += 1
            self._cleanup()
            return []

        matched_det_indices = set()
        results = []

        # 匹配现有轨迹
        for tid, track in list(self.tracks.items()):
            best_iou = 0
            best_idx = -1
            for i, det in enumerate(detections):
                if i in matched_det_indices:
                    continue
                iou = self._compute_iou(track['box'], det[:4])
                if iou > best_iou and iou > self.iou_thresh:
                    best_iou = iou
                    best_idx = i

            if best_idx >= 0:
                track['box'] = detections[best_idx][:4]
                track['conf'] = detections[best_idx][4]
                track['age'] = 0
                matched_det_indices.add(best_idx)
                results.append((tid, track['box'], track['conf']))
            else:
                track['age'] += 1

        self._cleanup()

        # 为未匹配的检测创建新轨迹
        for i, det in enumerate(detections):
            if i not in matched_det_indices:
                tid = self.next_id
                self.tracks[tid] = {'box': det[:4], 'conf': det[4], 'age': 0}
                results.append((tid, det[:4], det[4]))
                self.next_id += 1

        return results

    def _cleanup(self):
        self.tracks = {k: v for k, v in self.tracks.items() if v['age'] <= self.max_age}

    def _compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    def reset(self):
        self.tracks = {}
        self.next_id = 1


# ==================== 行为分析器 ====================
class GazeAnalyzer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.histories = defaultdict(list)

    def update(self, track_id, yaw, pitch, roll):
        """更新某个track的姿态历史"""
        self.histories[track_id].append({
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll
        })
        # 限制历史长度
        max_len = self.cfg.ANALYSIS_WINDOW * 3
        if len(self.histories[track_id]) > max_len:
            self.histories[track_id] = self.histories[track_id][-max_len:]

    def analyze(self, track_id):
        """分析某个track是否可疑"""
        history = self.histories.get(track_id, [])

        if len(history) < 5:
            return {
                'suspicious': False,
                'score': 0.0,
                'reason': 'insufficient_data'
            }

        # 取最近的窗口数据
        recent = history[-self.cfg.ANALYSIS_WINDOW:]
        yaws = np.array([p['yaw'] for p in recent])

        # 计算特征
        yaw_variance = np.var(yaws)
        yaw_range = np.max(yaws) - np.min(yaws)
        yaw_diff = np.diff(yaws)

        # 方向变化次数（符号变化）
        direction_changes = np.sum(np.diff(np.sign(yaw_diff)) != 0)

        # 大幅度转头次数
        large_changes = np.sum(np.abs(yaw_diff) > self.cfg.YAW_CHANGE_THRESHOLD)

        # 计算可疑分数
        score = 0.0
        reasons = []

        if yaw_variance > self.cfg.YAW_VARIANCE_THRESHOLD:
            score += 0.25
            reasons.append(f'high_variance:{yaw_variance:.1f}')
        if yaw_variance > self.cfg.YAW_VARIANCE_THRESHOLD * 2:
            score += 0.15

        if yaw_range > self.cfg.YAW_RANGE_THRESHOLD:
            score += 0.2
            reasons.append(f'wide_range:{yaw_range:.1f}')
        if yaw_range > self.cfg.YAW_RANGE_THRESHOLD * 2:
            score += 0.1

        if direction_changes >= self.cfg.MIN_DIRECTION_CHANGES:
            score += 0.2
            reasons.append(f'dir_changes:{direction_changes}')

        if large_changes >= 2:
            score += 0.2
            reasons.append(f'large_turns:{large_changes}')

        is_suspicious = score >= self.cfg.SUSPICIOUS_SCORE_THRESHOLD

        return {
            'suspicious': is_suspicious,
            'score': min(1.0, score),
            'yaw_variance': float(yaw_variance),
            'yaw_range': float(yaw_range),
            'direction_changes': int(direction_changes),
            'large_changes': int(large_changes),
            'reasons': reasons
        }

    def reset(self):
        self.histories.clear()


# ==================== 主系统 ====================
class SuspiciousGazeDetector:
    def __init__(self):
        self.cfg = Config()

        print("=" * 60)
        print("  可疑张望行为检测系统 v2.0")
        print("=" * 60)
        print(f"设备: {self.cfg.DEVICE}")
        print(f"输出目录: {self.cfg.OUTPUT_DIR}")

        # 初始化模型
        self._init_detector()
        self._init_pose_estimator()

        # 初始化追踪器和分析器
        self.tracker = SimpleTracker(
            max_age=self.cfg.TRACKER_MAX_AGE,
            iou_thresh=self.cfg.TRACKER_IOU_THRESH
        )
        self.analyzer = GazeAnalyzer(self.cfg)

        # 创建输出目录
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

    def _init_detector(self):
        """初始化YOLOv8检测器"""
        print("\n[1/2] 加载人体检测器...")
        try:
            from ultralytics import YOLO
            model_path = f"/root/autodl-tmp/behaviour/{self.cfg.DETECTOR_MODEL}"
            if not os.path.exists(model_path):
                model_path = self.cfg.DETECTOR_MODEL
            self.detector = YOLO(model_path)
            print(f"  ✓ YOLOv8 加载成功")
        except Exception as e:
            print(f"  ✗ 检测器加载失败: {e}")
            sys.exit(1)

    def _init_pose_estimator(self):
        """初始化头部姿态估计器"""
        print("[2/2] 加载头部姿态估计器...")
        try:
            from sixdrepnet import SixDRepNet
            # 使用缓存的模型权重
            weight_path = os.path.expanduser(
                "~/.cache/torch/hub/checkpoints/6DRepNet_300W_LP_AFLW2000.pth"
            )
            if os.path.exists(weight_path):
                self.pose_model = SixDRepNet(gpu_id=0, dict_path=weight_path)
            else:
                self.pose_model = SixDRepNet(gpu_id=0)
            print(f"  ✓ 6DRepNet 加载成功")
        except Exception as e:
            print(f"  ✗ 姿态估计器加载失败: {e}")
            print(f"  ! 将使用备用方法")
            self.pose_model = None

    def _detect_persons(self, frame):
        """检测帧中的人物"""
        results = self.detector(frame, verbose=False, classes=[0])[0]
        detections = []
        for box in results.boxes:
            if box.conf[0] > self.cfg.CONF_THRESHOLD:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                detections.append([x1, y1, x2, y2, conf])
        return detections

    def _extract_head_region(self, frame, bbox):
        """从人体bbox中提取头部区域"""
        x1, y1, x2, y2 = map(int, bbox)
        body_h = y2 - y1
        body_w = x2 - x1

        # 头部区域：顶部30%，宽度稍窄
        head_h = int(body_h * 0.25)
        head_w = int(body_w * 0.6)

        cx = (x1 + x2) // 2
        head_x1 = max(0, cx - head_w // 2)
        head_x2 = min(frame.shape[1], cx + head_w // 2)
        head_y1 = max(0, y1)
        head_y2 = min(frame.shape[0], y1 + head_h)

        head_img = frame[head_y1:head_y2, head_x1:head_x2]
        head_bbox = (head_x1, head_y1, head_x2, head_y2)

        return head_img, head_bbox

    def _estimate_pose(self, head_img):
        """估计头部姿态"""
        if head_img.size == 0 or head_img.shape[0] < 10 or head_img.shape[1] < 10:
            return 0.0, 0.0, 0.0

        if self.pose_model is not None:
            try:
                # 6DRepNet需要224x224的输入
                head_resized = cv2.resize(head_img, (224, 224))
                pitch, yaw, roll = self.pose_model.predict(head_resized)
                return float(pitch[0]), float(yaw[0]), float(roll[0])
            except Exception as e:
                pass

        # 备用方法：返回0
        return 0.0, 0.0, 0.0

    def _normalize_yaw(self, yaw, offset):
        """标准化yaw角度，考虑机位偏移"""
        yaw = yaw - offset
        while yaw > 180:
            yaw -= 360
        while yaw < -180:
            yaw += 360
        return yaw

    def _draw_visualization(self, frame, track_id, bbox, yaw, pitch, roll, analysis, head_bbox):
        """绘制可视化"""
        x1, y1, x2, y2 = map(int, bbox)
        hx1, hy1, hx2, hy2 = head_bbox

        is_suspicious = analysis['suspicious']
        score = analysis['score']

        # 颜色：可疑=红色，正常=绿色
        color = (0, 0, 255) if is_suspicious else (0, 200, 0)

        # 绘制人体框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 绘制头部框
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 255, 0), 1)

        # 绘制ID和yaw角度
        label = f"ID:{track_id} Y:{yaw:.0f}"
        cv2.putText(frame, label, (x1, y1 - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 如果可疑，显示警告
        if is_suspicious:
            warning = f"SUSPICIOUS! ({score:.2f})"
            cv2.putText(frame, warning, (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 绘制警告边框
            cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (0, 0, 255), 3)

        # 绘制姿态方向指示器
        head_cx = (hx1 + hx2) // 2
        head_cy = (hy1 + hy2) // 2

        # 用箭头表示yaw方向
        arrow_len = 30
        arrow_x = int(head_cx + arrow_len * np.sin(np.radians(yaw)))
        arrow_y = int(head_cy - arrow_len * np.cos(np.radians(pitch)))
        cv2.arrowedLine(frame, (head_cx, head_cy), (arrow_x, arrow_y),
                       (0, 255, 255), 2, tipLength=0.3)

        return frame

    def process_video(self, video_path, camera_type, yaw_offset):
        """处理单个视频"""
        video_name = Path(video_path).stem
        camera_label = "正机位" if camera_type == "front" else "侧机位"

        print(f"\n{'='*50}")
        print(f"处理: {video_name} ({camera_label})")
        print(f"{'='*50}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  ✗ 无法打开视频: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 限制处理帧数
        max_frames = min(total_frames, self.cfg.MAX_FRAMES_PER_VIDEO)

        print(f"  分辨率: {width}x{height}")
        print(f"  帧率: {fps:.1f} fps")
        print(f"  总帧数: {total_frames}, 处理: {max_frames}")

        # 重置追踪器和分析器
        self.tracker.reset()
        self.analyzer.reset()

        # 输出视频写入器
        writer = None
        if self.cfg.SAVE_VIDEO:
            output_path = os.path.join(self.cfg.OUTPUT_DIR, f"{video_name}_suspicious.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 统计
        results = {
            'video_name': video_name,
            'camera_type': camera_type,
            'yaw_offset': yaw_offset,
            'fps': fps,
            'total_frames': total_frames,
            'processed_frames': 0,
            'tracks': {},
            'suspicious_tracks': [],
            'suspicious_frame_count': 0
        }

        frame_idx = 0
        processed_count = 0

        print(f"  开始处理...")

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # 跳帧处理
            if frame_idx % self.cfg.PROCESS_EVERY_N_FRAMES != 0:
                if writer:
                    writer.write(frame)
                frame_idx += 1
                continue

            # 检测人物
            detections = self._detect_persons(frame)

            # 追踪
            tracks = self.tracker.update(detections)

            frame_has_suspicious = False

            for track_id, bbox, conf in tracks:
                # 提取头部
                head_img, head_bbox = self._extract_head_region(frame, bbox)

                # 估计姿态
                pitch, yaw, roll = self._estimate_pose(head_img)

                # 标准化yaw
                yaw_normalized = self._normalize_yaw(yaw, yaw_offset)

                # 更新分析器
                self.analyzer.update(track_id, yaw_normalized, pitch, roll)

                # 分析行为
                analysis = self.analyzer.analyze(track_id)

                # 记录track数据
                if track_id not in results['tracks']:
                    results['tracks'][track_id] = {
                        'poses': [],
                        'suspicious': False,
                        'max_score': 0.0
                    }

                results['tracks'][track_id]['poses'].append({
                    'frame': frame_idx,
                    'yaw': yaw_normalized,
                    'pitch': pitch,
                    'roll': roll
                })

                if analysis['score'] > results['tracks'][track_id]['max_score']:
                    results['tracks'][track_id]['max_score'] = analysis['score']

                if analysis['suspicious']:
                    results['tracks'][track_id]['suspicious'] = True
                    frame_has_suspicious = True

                # 绘制可视化
                frame = self._draw_visualization(
                    frame, track_id, bbox, yaw_normalized, pitch, roll,
                    analysis, head_bbox
                )

            if frame_has_suspicious:
                results['suspicious_frame_count'] += 1

            # 绘制状态栏
            self._draw_status_bar(frame, frame_idx, len(tracks),
                                 results['suspicious_frame_count'], camera_label)

            # 写入输出视频
            if writer:
                writer.write(frame)

            processed_count += 1
            frame_idx += 1

            # 进度显示
            if processed_count % 100 == 0:
                progress = (frame_idx / max_frames) * 100
                print(f"  进度: {progress:.1f}% ({frame_idx}/{max_frames})")

        cap.release()
        if writer:
            writer.release()

        results['processed_frames'] = processed_count

        # 统计可疑轨迹
        suspicious_tracks = [
            tid for tid, data in results['tracks'].items()
            if data['suspicious']
        ]
        results['suspicious_tracks'] = suspicious_tracks

        # 简化输出
        results['tracks'] = {
            tid: {
                'suspicious': data['suspicious'],
                'max_score': data['max_score'],
                'num_poses': len(data['poses'])
            }
            for tid, data in results['tracks'].items()
        }

        print(f"\n  处理完成:")
        print(f"    总轨迹: {len(results['tracks'])}")
        print(f"    可疑轨迹: {len(suspicious_tracks)}")
        print(f"    可疑帧: {results['suspicious_frame_count']}")

        if self.cfg.SAVE_VIDEO:
            print(f"    输出视频: {output_path}")

        return results

    def _draw_status_bar(self, frame, frame_idx, num_tracks, suspicious_count, camera_label):
        """绘制状态栏"""
        h, w = frame.shape[:2]

        # 顶部状态栏背景
        cv2.rectangle(frame, (0, 0), (w, 35), (40, 40, 40), -1)

        # 状态文字
        status = f"Frame: {frame_idx} | Tracks: {num_tracks} | Suspicious: {suspicious_count} | {camera_label}"
        cv2.putText(frame, status, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 图例
        cv2.rectangle(frame, (w-200, 5), (w-180, 25), (0, 200, 0), -1)
        cv2.putText(frame, "Normal", (w-175, 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.rectangle(frame, (w-100, 5), (w-80, 25), (0, 0, 255), -1)
        cv2.putText(frame, "Suspicious", (w-75, 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def run(self):
        """运行全部处理"""
        print("\n" + "=" * 60)
        print("开始处理所有视频...")
        print("=" * 60)

        all_results = {
            'timestamp': datetime.now().isoformat(),
            'device': self.cfg.DEVICE,
            'config': {
                'yaw_change_threshold': self.cfg.YAW_CHANGE_THRESHOLD,
                'yaw_variance_threshold': self.cfg.YAW_VARIANCE_THRESHOLD,
                'suspicious_score_threshold': self.cfg.SUSPICIOUS_SCORE_THRESHOLD
            },
            'videos': []
        }

        for camera_type, camera_cfg in self.cfg.VIDEOS.items():
            video_dir = os.path.join(self.cfg.DATA_ROOT, camera_cfg['dir'])
            yaw_offset = camera_cfg['yaw_offset']

            for video_file in camera_cfg['files']:
                video_path = os.path.join(video_dir, video_file)

                if os.path.exists(video_path):
                    result = self.process_video(video_path, camera_type, yaw_offset)
                    if result:
                        all_results['videos'].append(result)
                else:
                    print(f"\n跳过 (文件不存在): {video_path}")

        # 保存汇总结果
        self._save_summary(all_results)

        return all_results

    def _save_summary(self, all_results):
        """保存汇总结果"""
        print("\n" + "=" * 60)
        print("处理完成 - 汇总报告")
        print("=" * 60)

        # 统计
        total_tracks = sum(len(v['tracks']) for v in all_results['videos'])
        total_suspicious = sum(len(v['suspicious_tracks']) for v in all_results['videos'])

        print(f"\n总视频数: {len(all_results['videos'])}")
        print(f"总轨迹数: {total_tracks}")
        print(f"可疑轨迹: {total_suspicious}")

        if total_tracks > 0:
            ratio = (total_suspicious / total_tracks) * 100
            print(f"可疑比例: {ratio:.1f}%")

        # 按机位分类统计
        for camera_type in ['front', 'side']:
            videos = [v for v in all_results['videos'] if v['camera_type'] == camera_type]
            if videos:
                label = "正机位" if camera_type == "front" else "侧机位"
                tracks = sum(len(v['tracks']) for v in videos)
                suspicious = sum(len(v['suspicious_tracks']) for v in videos)
                print(f"\n{label}:")
                print(f"  视频数: {len(videos)}")
                print(f"  轨迹数: {tracks}")
                print(f"  可疑轨迹: {suspicious}")

        # 保存JSON
        output_file = os.path.join(self.cfg.OUTPUT_DIR, "detection_results.json")
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
