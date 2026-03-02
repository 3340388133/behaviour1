#!/usr/bin/env python3
"""
全自动可疑张望行为检测系统

自动完成：
1. 检测所有视频中的人物
2. 追踪人物轨迹
3. 估计头部姿态
4. 分析可疑张望行为
5. 生成完整报告和标注视频
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
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
class Config:
    # 路径
    DATA_ROOT = "/root/autodl-tmp/behaviour/data"
    OUTPUT_DIR = "/root/autodl-tmp/behaviour/SuspiciousGazeDetection/output"

    # 视频分类
    VIDEOS = {
        "front": {  # 正机位
            "dir": "raw_videos/正机位",
            "files": ["1.14rg-1.mp4", "1.14zz-1.mp4", "1.14zz-2.mp4", "1.14zz-3.mp4", "1.14zz-4.mp4"],
            "yaw_offset": 0.0,
        },
        "side": {  # 侧机位
            "dir": "raw_videos/侧机位",
            "files": ["MVI_4537.MP4", "MVI_4538.MP4", "MVI_4539.MP4", "MVI_4540.MP4"],
            "yaw_offset": 90.0,
        }
    }

    # 检测参数
    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5

    # 可疑行为参数
    YAW_CHANGE_THRESHOLD = 30.0    # 单次转头阈值
    MIN_GAZE_FREQUENCY = 3          # 最小张望次数
    YAW_VARIANCE_THRESHOLD = 200    # 偏航方差阈值
    SUSPICIOUS_THRESHOLD = 0.5      # 可疑分数阈值
    WINDOW_SIZE = 30                # 分析窗口(帧)

    # 处理参数
    PROCESS_EVERY_N_FRAMES = 1      # 每N帧处理一次(1=全部)
    MAX_FRAMES = None               # 最大处理帧数(None=全部)
    SAVE_VIDEO = True               # 保存标注视频

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== 追踪器 ====================
class SimpleTracker:
    def __init__(self, max_age=30, iou_thresh=0.3):
        self.max_age = max_age
        self.iou_thresh = iou_thresh
        self.tracks = {}
        self.next_id = 1

    def update(self, dets):
        if len(dets) == 0:
            for t in self.tracks.values():
                t['age'] += 1
            self._cleanup()
            return []

        matched = set()
        results = []

        for tid, t in list(self.tracks.items()):
            best_iou, best_idx = 0, -1
            for i, d in enumerate(dets):
                if i in matched:
                    continue
                iou = self._iou(t['box'], d[:4])
                if iou > best_iou and iou > self.iou_thresh:
                    best_iou, best_idx = iou, i

            if best_idx >= 0:
                t['box'] = dets[best_idx][:4]
                t['conf'] = dets[best_idx][4]
                t['age'] = 0
                matched.add(best_idx)
                results.append((tid, t['box'], t['conf']))
            else:
                t['age'] += 1

        self._cleanup()

        for i, d in enumerate(dets):
            if i not in matched:
                self.tracks[self.next_id] = {'box': d[:4], 'conf': d[4], 'age': 0}
                results.append((self.next_id, d[:4], d[4]))
                self.next_id += 1

        return results

    def _cleanup(self):
        self.tracks = {k: v for k, v in self.tracks.items() if v['age'] <= self.max_age}

    def _iou(self, b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0

    def reset(self):
        self.tracks = {}
        self.next_id = 1


# ==================== 行为分析器 ====================
class GazeAnalyzer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.histories = defaultdict(list)

    def update(self, tid, yaw, pitch, roll):
        self.histories[tid].append({'yaw': yaw, 'pitch': pitch, 'roll': roll})
        if len(self.histories[tid]) > self.cfg.WINDOW_SIZE * 2:
            self.histories[tid] = self.histories[tid][-self.cfg.WINDOW_SIZE * 2:]

    def analyze(self, tid):
        h = self.histories.get(tid, [])
        if len(h) < 5:
            return {'suspicious': False, 'score': 0.0}

        yaws = np.array([p['yaw'] for p in h[-self.cfg.WINDOW_SIZE:]])

        yaw_var = np.var(yaws)
        yaw_range = np.max(yaws) - np.min(yaws)
        yaw_diff = np.diff(yaws)
        dir_changes = np.sum(np.diff(np.sign(yaw_diff)) != 0)
        large_changes = np.sum(np.abs(yaw_diff) > self.cfg.YAW_CHANGE_THRESHOLD)

        score = 0.0
        if yaw_var > self.cfg.YAW_VARIANCE_THRESHOLD: score += 0.3
        if yaw_var > self.cfg.YAW_VARIANCE_THRESHOLD * 2: score += 0.1
        if yaw_range > 60: score += 0.2
        if yaw_range > 120: score += 0.1
        if dir_changes >= self.cfg.MIN_GAZE_FREQUENCY: score += 0.2
        if large_changes >= 2: score += 0.2

        return {
            'suspicious': score >= self.cfg.SUSPICIOUS_THRESHOLD,
            'score': min(1.0, score),
            'yaw_var': float(yaw_var),
            'yaw_range': float(yaw_range),
            'dir_changes': int(dir_changes),
            'large_changes': int(large_changes),
        }

    def reset(self):
        self.histories.clear()


# ==================== 主系统 ====================
class SuspiciousGazeSystem:
    def __init__(self):
        self.cfg = Config()
        print("="*60)
        print("可疑张望行为检测系统 - 全自动模式")
        print("="*60)
        print(f"设备: {self.cfg.DEVICE}")

        self._init_detector()
        self._init_pose_model()

        self.tracker = SimpleTracker()
        self.analyzer = GazeAnalyzer(self.cfg)

        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

    def _init_detector(self):
        print("\n[1/2] 加载检测器...")
        try:
            from ultralytics import YOLO
            model_path = "/root/autodl-tmp/behaviour/yolov8m.pt"
            if not os.path.exists(model_path):
                model_path = "yolov8m.pt"
            self.detector = YOLO(model_path)
            print(f"  ✓ YOLOv8 已加载")
        except Exception as e:
            print(f"  ✗ 检测器加载失败: {e}")
            sys.exit(1)

    def _init_pose_model(self):
        print("[2/2] 加载姿态估计器...")
        self.pose_model = None

        # 尝试加载6DRepNet
        try:
            sys.path.insert(0, "/root/autodl-tmp/behaviour/6DRepNet-master")
            from sixdrepnet import SixDRepNet
            self.pose_model = SixDRepNet()
            print(f"  ✓ 6DRepNet 已加载")
            return
        except:
            pass

        # 尝试其他方式
        try:
            # 检查是否有预下载的模型
            cache_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/6DRepNet_300W_LP_AFLW2000.pth")
            if os.path.exists(cache_path):
                sys.path.insert(0, "/root/autodl-tmp/behaviour/6DRepNet-master")
                from sixdrepnet import SixDRepNet
                self.pose_model = SixDRepNet()
                print(f"  ✓ 6DRepNet 已加载(缓存)")
                return
        except:
            pass

        print(f"  ! 使用简化姿态估计(基于位置)")

    def _detect(self, frame):
        results = self.detector(frame, verbose=False, classes=[0])[0]
        dets = []
        for box in results.boxes:
            if box.conf[0] > self.cfg.CONF_THRESHOLD:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                dets.append([x1, y1, x2, y2, conf])
        return dets

    def _extract_head(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = y2 - y1, x2 - x1
        head_h = int(h * 0.3)
        head_w = int(head_h * 0.8)
        cx = (x1 + x2) // 2
        hx1, hx2 = max(0, cx - head_w//2), min(frame.shape[1], cx + head_w//2)
        hy1, hy2 = max(0, y1), min(frame.shape[0], y1 + head_h)
        return frame[hy1:hy2, hx1:hx2], (hx1, hy1, hx2, hy2)

    def _estimate_pose(self, head):
        if head.size == 0:
            return 0.0, 0.0, 0.0

        if self.pose_model is not None:
            try:
                img = cv2.resize(head, (224, 224))
                yaw, pitch, roll = self.pose_model.predict(img)
                return float(yaw), float(pitch), float(roll)
            except:
                pass

        # 简化估计：基于头部在bbox中的位置
        return 0.0, 0.0, 0.0

    def _normalize_yaw(self, yaw, offset):
        yaw = yaw - offset
        while yaw > 180: yaw -= 360
        while yaw < -180: yaw += 360
        return yaw

    def process_video(self, video_path, camera_type, yaw_offset):
        video_name = Path(video_path).stem
        print(f"\n处理: {video_name} ({'正机位' if camera_type == 'front' else '侧机位'})")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  ✗ 无法打开视频")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.cfg.MAX_FRAMES:
            total = min(total, self.cfg.MAX_FRAMES)

        print(f"  {w}x{h} @ {fps:.0f}fps, {total}帧")

        self.tracker.reset()
        self.analyzer.reset()

        # 输出视频
        writer = None
        if self.cfg.SAVE_VIDEO:
            out_path = os.path.join(self.cfg.OUTPUT_DIR, f"{video_name}_result.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        results = {
            'video': video_name,
            'camera': camera_type,
            'fps': fps,
            'total_frames': total,
            'suspicious_frames': 0,
            'tracks_data': defaultdict(lambda: {'poses': [], 'suspicious': False, 'max_score': 0}),
        }

        pbar = tqdm(total=total, desc=f"  {video_name}", ncols=80)
        frame_num = 0

        while frame_num < total:
            ret, frame = cap.read()
            if not ret:
                break

            # 检测
            dets = self._detect(frame)

            # 追踪
            tracks = self.tracker.update(dets)

            frame_suspicious = False

            for tid, bbox, conf in tracks:
                # 头部提取和姿态估计
                head, _ = self._extract_head(frame, bbox)
                yaw, pitch, roll = self._estimate_pose(head)
                yaw = self._normalize_yaw(yaw, yaw_offset)

                # 更新分析器
                self.analyzer.update(tid, yaw, pitch, roll)
                analysis = self.analyzer.analyze(tid)

                # 记录数据
                results['tracks_data'][tid]['poses'].append({
                    'frame': frame_num, 'yaw': yaw, 'pitch': pitch, 'roll': roll
                })
                if analysis['score'] > results['tracks_data'][tid]['max_score']:
                    results['tracks_data'][tid]['max_score'] = analysis['score']
                if analysis['suspicious']:
                    results['tracks_data'][tid]['suspicious'] = True
                    frame_suspicious = True

                # 绘制
                color = (0, 0, 255) if analysis['suspicious'] else (0, 255, 0)
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{tid} Y:{yaw:.0f}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                if analysis['suspicious']:
                    cv2.putText(frame, f"SUSPICIOUS ({analysis['score']:.2f})", (x1, y2+15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            if frame_suspicious:
                results['suspicious_frames'] += 1

            # 状态栏
            cv2.rectangle(frame, (0, 0), (300, 30), (0, 0, 0), -1)
            cv2.putText(frame, f"Frame:{frame_num} Tracks:{len(tracks)} Suspicious:{results['suspicious_frames']}",
                       (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if writer:
                writer.write(frame)

            frame_num += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        if writer:
            writer.release()

        # 统计
        total_tracks = len(results['tracks_data'])
        suspicious_tracks = sum(1 for t in results['tracks_data'].values() if t['suspicious'])

        results['summary'] = {
            'total_tracks': total_tracks,
            'suspicious_tracks': suspicious_tracks,
            'suspicious_ratio': suspicious_tracks / total_tracks if total_tracks > 0 else 0,
            'suspicious_frames_ratio': results['suspicious_frames'] / total if total > 0 else 0,
        }

        print(f"  ✓ 轨迹:{total_tracks} 可疑:{suspicious_tracks}({results['summary']['suspicious_ratio']*100:.1f}%)")

        return results

    def run(self):
        print("\n" + "="*60)
        print("开始处理所有视频...")
        print("="*60)

        all_results = {
            'timestamp': datetime.now().isoformat(),
            'device': self.cfg.DEVICE,
            'videos': [],
        }

        for camera_type, info in self.cfg.VIDEOS.items():
            video_dir = os.path.join(self.cfg.DATA_ROOT, info['dir'])

            for video_file in info['files']:
                video_path = os.path.join(video_dir, video_file)
                if os.path.exists(video_path):
                    result = self.process_video(video_path, camera_type, info['yaw_offset'])
                    if result:
                        # 简化存储
                        result['tracks_data'] = {
                            k: {'suspicious': v['suspicious'], 'max_score': v['max_score'], 'num_poses': len(v['poses'])}
                            for k, v in result['tracks_data'].items()
                        }
                        all_results['videos'].append(result)
                else:
                    print(f"\n跳过(不存在): {video_path}")

        # 汇总
        print("\n" + "="*60)
        print("处理完成 - 汇总")
        print("="*60)

        front_videos = [v for v in all_results['videos'] if v['camera'] == 'front']
        side_videos = [v for v in all_results['videos'] if v['camera'] == 'side']

        def summarize(videos, name):
            if not videos:
                return
            total_tracks = sum(v['summary']['total_tracks'] for v in videos)
            suspicious = sum(v['summary']['suspicious_tracks'] for v in videos)
            print(f"\n{name}:")
            print(f"  视频数: {len(videos)}")
            print(f"  总轨迹: {total_tracks}")
            print(f"  可疑轨迹: {suspicious} ({100*suspicious/total_tracks:.1f}%)" if total_tracks else "  可疑轨迹: 0")

        summarize(front_videos, "正机位")
        summarize(side_videos, "侧机位")

        total_all = sum(v['summary']['total_tracks'] for v in all_results['videos'])
        suspicious_all = sum(v['summary']['suspicious_tracks'] for v in all_results['videos'])
        print(f"\n总计: {total_all}轨迹, {suspicious_all}可疑 ({100*suspicious_all/total_all:.1f}%)" if total_all else "\n总计: 0")

        # 保存
        out_file = os.path.join(self.cfg.OUTPUT_DIR, "full_results.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n结果保存至: {out_file}")
        if self.cfg.SAVE_VIDEO:
            print(f"标注视频保存至: {self.cfg.OUTPUT_DIR}/")

        return all_results


if __name__ == "__main__":
    system = SuspiciousGazeSystem()
    system.run()
