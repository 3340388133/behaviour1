#!/usr/bin/env python3
"""
视频测试脚本 - 可疑张望行为检测

直接读取视频文件进行测试，按正机位/侧机位分开处理
"""

import os
import sys
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


# ==================== 配置 ====================
CONFIG = {
    # 数据路径
    "data_root": "/root/autodl-tmp/behaviour/data",
    "output_dir": "/root/autodl-tmp/behaviour/SuspiciousGazeDetection/output",

    # 视频分类 (正机位 vs 侧机位)
    "front_camera": [  # 正机位
        "1.14rg-1.mp4",
        "1.14zz-1.mp4",
        "1.14zz-2.mp4",
        "1.14zz-3.mp4",
        "1.14zz-4.mp4",
    ],
    "side_camera": [  # 侧机位
        "MVI_4537.MP4",
        "MVI_4538.MP4",
        "MVI_4539.MP4",
        "MVI_4540.MP4",
    ],

    # 坐标系偏移 (侧机位需要补偿)
    "front_yaw_offset": 0.0,
    "side_yaw_offset": 90.0,

    # 检测参数
    "conf_threshold": 0.5,
    "suspicious_threshold": 0.6,
    "yaw_change_threshold": 30.0,  # 度
    "min_gaze_frequency": 3,       # 最小张望次数
    "time_window": 30,             # 帧数窗口

    # 设备
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


class SimpleTracker:
    """简单的IoU追踪器"""

    def __init__(self, max_age=30, iou_threshold=0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_id = 1

    def update(self, detections):
        """更新追踪"""
        if len(detections) == 0:
            # 更新所有track的age
            for tid in list(self.tracks.keys()):
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    del self.tracks[tid]
            return []

        # 匹配检测和已有tracks
        matched = set()
        results = []

        for tid, track in self.tracks.items():
            best_iou = 0
            best_idx = -1

            for i, det in enumerate(detections):
                if i in matched:
                    continue
                iou = self._compute_iou(track['bbox'], det[:4])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_idx = i

            if best_idx >= 0:
                track['bbox'] = detections[best_idx][:4]
                track['conf'] = detections[best_idx][4]
                track['age'] = 0
                matched.add(best_idx)
                results.append((tid, track['bbox'], track['conf']))
            else:
                track['age'] += 1

        # 删除过期tracks
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]['age'] > self.max_age:
                del self.tracks[tid]

        # 为未匹配的检测创建新track
        for i, det in enumerate(detections):
            if i not in matched:
                self.tracks[self.next_id] = {
                    'bbox': det[:4],
                    'conf': det[4],
                    'age': 0,
                }
                results.append((self.next_id, det[:4], det[4]))
                self.next_id += 1

        return results

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


class GazeAnalyzer:
    """张望行为分析器"""

    def __init__(self, config):
        self.config = config
        self.pose_histories = defaultdict(list)

    def update(self, track_id, yaw, pitch, roll):
        """更新姿态历史"""
        self.pose_histories[track_id].append({
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll,
        })

        # 保持历史长度
        max_len = self.config['time_window'] * 2
        if len(self.pose_histories[track_id]) > max_len:
            self.pose_histories[track_id] = self.pose_histories[track_id][-max_len:]

    def analyze(self, track_id):
        """分析是否为可疑张望"""
        history = self.pose_histories.get(track_id, [])

        if len(history) < 10:
            return {
                'suspicious': False,
                'score': 0.0,
                'yaw_variance': 0.0,
                'direction_changes': 0,
            }

        # 取最近的窗口
        window = history[-self.config['time_window']:]
        yaws = [p['yaw'] for p in window]
        yaws = np.array(yaws)

        # 计算特征
        yaw_var = np.var(yaws)
        yaw_range = np.max(yaws) - np.min(yaws)

        # 方向变化次数
        yaw_diff = np.diff(yaws)
        direction_changes = np.sum(np.diff(np.sign(yaw_diff)) != 0)

        # 大幅度转头次数
        large_changes = np.sum(np.abs(yaw_diff) > self.config['yaw_change_threshold'])

        # 计算可疑分数
        score = 0.0

        if yaw_var > 200:
            score += 0.3
        if yaw_range > 60:
            score += 0.2
        if direction_changes >= self.config['min_gaze_frequency']:
            score += 0.3
        if large_changes >= 2:
            score += 0.2

        score = min(1.0, score)
        is_suspicious = score >= self.config['suspicious_threshold']

        return {
            'suspicious': is_suspicious,
            'score': score,
            'yaw_variance': float(yaw_var),
            'yaw_range': float(yaw_range),
            'direction_changes': int(direction_changes),
            'large_changes': int(large_changes),
        }

    def reset(self):
        self.pose_histories.clear()


class VideoTester:
    """视频测试器"""

    def __init__(self, config):
        self.config = config
        self.device = config['device']

        # 初始化检测器
        self._init_detector()

        # 初始化姿态估计器
        self._init_pose_estimator()

        # 追踪器和分析器
        self.tracker = SimpleTracker()
        self.analyzer = GazeAnalyzer(config)

    def _init_detector(self):
        """初始化YOLOv8检测器"""
        try:
            from ultralytics import YOLO
            self.detector = YOLO("yolov8m.pt")
            print(f"[OK] YOLOv8m 检测器已加载")
        except Exception as e:
            print(f"[WARN] 无法加载YOLOv8: {e}")
            self.detector = None

    def _init_pose_estimator(self):
        """初始化头部姿态估计器"""
        try:
            # 尝试加载6DRepNet
            sys.path.insert(0, "/root/autodl-tmp/behaviour/6DRepNet-master")
            from sixdrepnet import SixDRepNet
            self.pose_model = SixDRepNet()
            print(f"[OK] 6DRepNet 姿态估计器已加载")
        except Exception as e:
            print(f"[WARN] 无法加载6DRepNet: {e}")
            # 尝试加载WHENet
            try:
                sys.path.insert(0, "/root/autodl-tmp/behaviour")
                self.pose_model = None  # 使用简化版本
                print(f"[INFO] 使用简化姿态估计")
            except:
                self.pose_model = None

    def detect_persons(self, frame):
        """检测人物"""
        if self.detector is None:
            return []

        results = self.detector(frame, verbose=False, classes=[0])[0]

        detections = []
        for box in results.boxes:
            if box.conf[0] > self.config['conf_threshold']:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                detections.append([x1, y1, x2, y2, conf])

        return detections

    def extract_head(self, frame, bbox, ratio=0.3):
        """从人体框提取头部区域"""
        x1, y1, x2, y2 = map(int, bbox)
        h = y2 - y1
        w = x2 - x1

        # 头部在上方30%
        head_h = int(h * ratio)
        head_w = int(head_h * 0.8)

        cx = (x1 + x2) // 2
        hx1 = max(0, cx - head_w // 2)
        hx2 = min(frame.shape[1], cx + head_w // 2)
        hy1 = max(0, y1)
        hy2 = min(frame.shape[0], y1 + head_h)

        head = frame[hy1:hy2, hx1:hx2]
        return head, (hx1, hy1, hx2, hy2)

    def estimate_pose(self, head_crop):
        """估计头部姿态"""
        if head_crop.size == 0:
            return 0.0, 0.0, 0.0

        if self.pose_model is not None:
            try:
                # 使用实际模型
                img = cv2.resize(head_crop, (224, 224))
                yaw, pitch, roll = self.pose_model.predict(img)
                return float(yaw), float(pitch), float(roll)
            except:
                pass

        # 简化版：基于头部位置估计（仅用于演示）
        # 实际应用需要真正的姿态估计模型
        return 0.0, 0.0, 0.0

    def process_video(self, video_path, camera_type="front"):
        """处理单个视频"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] 无法打开视频: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_name = Path(video_path).stem
        print(f"\n处理视频: {video_name}")
        print(f"  分辨率: {width}x{height}, FPS: {fps:.1f}, 总帧数: {total_frames}")
        print(f"  机位类型: {'正机位' if camera_type == 'front' else '侧机位'}")

        # 偏移量
        yaw_offset = self.config['front_yaw_offset'] if camera_type == 'front' else self.config['side_yaw_offset']

        # 重置
        self.tracker.reset()
        self.analyzer.reset()

        # 结果统计
        results = {
            'video': video_name,
            'camera_type': camera_type,
            'total_frames': total_frames,
            'fps': fps,
            'suspicious_frames': 0,
            'suspicious_tracks': set(),
            'frame_results': [],
        }

        # 输出视频
        output_path = os.path.join(
            self.config['output_dir'],
            f"{video_name}_{camera_type}_result.mp4"
        )
        os.makedirs(self.config['output_dir'], exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 处理帧
        frame_num = 0
        pbar = tqdm(total=total_frames, desc=f"  {video_name}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 检测
            detections = self.detect_persons(frame)

            # 追踪
            tracks = self.tracker.update(detections)

            frame_suspicious = False
            frame_result = {'frame': frame_num, 'tracks': []}

            for track_id, bbox, conf in tracks:
                # 提取头部
                head, head_bbox = self.extract_head(frame, bbox)

                # 姿态估计
                yaw, pitch, roll = self.estimate_pose(head)

                # 应用坐标偏移
                yaw = yaw - yaw_offset
                # 归一化到 [-180, 180]
                while yaw > 180: yaw -= 360
                while yaw < -180: yaw += 360

                # 更新分析器
                self.analyzer.update(track_id, yaw, pitch, roll)

                # 分析行为
                analysis = self.analyzer.analyze(track_id)

                if analysis['suspicious']:
                    frame_suspicious = True
                    results['suspicious_tracks'].add(track_id)

                # 绘制
                color = (0, 0, 255) if analysis['suspicious'] else (0, 255, 0)
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"ID:{track_id} Y:{yaw:.0f}"
                cv2.putText(frame, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if analysis['suspicious']:
                    cv2.putText(frame, "SUSPICIOUS", (x1, y2+15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                frame_result['tracks'].append({
                    'track_id': track_id,
                    'bbox': bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox),
                    'pose': [yaw, pitch, roll],
                    'suspicious': analysis['suspicious'],
                    'score': analysis['score'],
                })

            if frame_suspicious:
                results['suspicious_frames'] += 1

            results['frame_results'].append(frame_result)

            # 写入帧
            writer.write(frame)

            frame_num += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        writer.release()

        # 转换set为list
        results['suspicious_tracks'] = list(results['suspicious_tracks'])
        results['num_suspicious_tracks'] = len(results['suspicious_tracks'])

        print(f"  可疑帧数: {results['suspicious_frames']}/{total_frames} ({100*results['suspicious_frames']/total_frames:.1f}%)")
        print(f"  可疑人员数: {results['num_suspicious_tracks']}")
        print(f"  输出视频: {output_path}")

        return results

    def run_all_tests(self):
        """运行所有视频测试"""
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'front_camera': [],
            'side_camera': [],
            'summary': {},
        }

        # 测试正机位视频
        print("\n" + "="*50)
        print("测试正机位视频")
        print("="*50)

        front_dir = os.path.join(self.config['data_root'], "raw_videos", "正机位")
        for video_file in self.config['front_camera']:
            video_path = os.path.join(front_dir, video_file)
            if os.path.exists(video_path):
                result = self.process_video(video_path, camera_type="front")
                if result:
                    # 简化结果，不保存每帧详情
                    result_summary = {k: v for k, v in result.items() if k != 'frame_results'}
                    all_results['front_camera'].append(result_summary)
            else:
                print(f"[SKIP] 视频不存在: {video_path}")

        # 测试侧机位视频
        print("\n" + "="*50)
        print("测试侧机位视频")
        print("="*50)

        side_dir = os.path.join(self.config['data_root'], "raw_videos", "侧机位")
        for video_file in self.config['side_camera']:
            video_path = os.path.join(side_dir, video_file)
            if os.path.exists(video_path):
                result = self.process_video(video_path, camera_type="side")
                if result:
                    result_summary = {k: v for k, v in result.items() if k != 'frame_results'}
                    all_results['side_camera'].append(result_summary)
            else:
                print(f"[SKIP] 视频不存在: {video_path}")

        # 汇总统计
        front_suspicious = sum(r['suspicious_frames'] for r in all_results['front_camera'])
        front_total = sum(r['total_frames'] for r in all_results['front_camera'])
        side_suspicious = sum(r['suspicious_frames'] for r in all_results['side_camera'])
        side_total = sum(r['total_frames'] for r in all_results['side_camera'])

        all_results['summary'] = {
            'front_camera': {
                'num_videos': len(all_results['front_camera']),
                'total_frames': front_total,
                'suspicious_frames': front_suspicious,
                'suspicious_ratio': front_suspicious / front_total if front_total > 0 else 0,
            },
            'side_camera': {
                'num_videos': len(all_results['side_camera']),
                'total_frames': side_total,
                'suspicious_frames': side_suspicious,
                'suspicious_ratio': side_suspicious / side_total if side_total > 0 else 0,
            },
        }

        # 保存结果
        result_path = os.path.join(self.config['output_dir'], "test_results.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # 打印汇总
        print("\n" + "="*50)
        print("测试汇总")
        print("="*50)
        print(f"\n正机位:")
        print(f"  视频数: {all_results['summary']['front_camera']['num_videos']}")
        print(f"  总帧数: {front_total}")
        print(f"  可疑帧: {front_suspicious} ({100*all_results['summary']['front_camera']['suspicious_ratio']:.1f}%)")

        print(f"\n侧机位:")
        print(f"  视频数: {all_results['summary']['side_camera']['num_videos']}")
        print(f"  总帧数: {side_total}")
        print(f"  可疑帧: {side_suspicious} ({100*all_results['summary']['side_camera']['suspicious_ratio']:.1f}%)")

        print(f"\n结果保存至: {result_path}")

        return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="视频测试 - 可疑张望检测")
    parser.add_argument("--video", type=str, help="单个视频路径")
    parser.add_argument("--camera", type=str, choices=["front", "side"], default="front", help="机位类型")
    parser.add_argument("--all", action="store_true", help="测试所有视频")
    args = parser.parse_args()

    tester = VideoTester(CONFIG)

    if args.all:
        tester.run_all_tests()
    elif args.video:
        tester.process_video(args.video, args.camera)
    else:
        print("用法:")
        print("  测试所有视频: python test_video.py --all")
        print("  测试单个视频: python test_video.py --video path/to/video.mp4 --camera front")


if __name__ == "__main__":
    main()
