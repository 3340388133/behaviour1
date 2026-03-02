#!/usr/bin/env python3
"""
视频推理脚本 - 使用 Transformer 模型识别6类行为，只画头部框

行为类别：
- 0: normal          正常行为      绿色
- 1: glancing        频繁张望      红色
- 2: quick_turn      快速回头      橙色
- 3: prolonged_watch  长时间观察   紫色
- 4: looking_down    持续低头      蓝色
- 5: looking_up      持续抬头      黄色

使用方法:
    python experiments/scripts/inference_video.py --video MVI_4540
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'recognition'))

import json
import argparse
import cv2
import torch
import numpy as np
from collections import deque
from tqdm import tqdm

from temporal_transformer import create_model

# 6类行为定义
BEHAVIOR_CLASSES = {
    0: ('Normal', (0, 255, 0)),           # 绿色
    1: ('Glancing', (0, 0, 255)),         # 红色
    2: ('QuickTurn', (0, 128, 255)),      # 橙色
    3: ('Prolonged', (255, 0, 128)),      # 紫色
    4: ('LookDown', (255, 128, 0)),       # 蓝色
    5: ('LookUp', (0, 255, 255)),         # 黄色
}


class BehaviorRecognizer:
    """
    全规则行为识别器（baseline-relative，适配正面/侧面机位）

    所有 6 类行为均基于 yaw/pitch/roll 相对于每个 track 自身基线的偏移判定。
    WHENet 输出的角度是相机坐标系，正面机位 yaw≈0 表示正对相机，
    侧面机位 yaw≈-110 表示正对相机。用 baseline 抵消机位差异。

    类别:
      0 normal          视线稳定
      1 glancing         3s 内 yaw 方向翻转≥3次，摆幅>30°
      2 quick_turn       0.5s 内 |Δyaw|>60°
      3 prolonged_watch  |yaw - baseline|>30° 持续>3s
      4 looking_down     pitch - baseline < -20° 持续>5s
      5 looking_up       pitch - baseline > 20° 持续>3s
    """

    # pose 采样间隔（原始视频每3帧取一次pose）
    POSE_SUBSAMPLE = 3

    def __init__(self, model_path: str = None, device: str = 'cuda',
                 smooth_window: int = 15, fps: float = 30.0):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.smooth_window = smooth_window
        self.fps = fps
        # 每个 pose 样本对应的实际秒数
        self.dt = self.POSE_SUBSAMPLE / fps

        # 模型暂不使用（全规则方案），保留接口备用
        self.model = None
        if model_path and Path(model_path).exists():
            try:
                self.model = create_model(
                    model_type='transformer',
                    pose_input_dim=3, pose_d_model=64, pose_nhead=4,
                    pose_num_layers=2, use_multimodal=False,
                    hidden_dim=128, num_classes=6, dropout=0.1,
                    uncertainty_weighting=True,
                )
                ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
                if 'model_state_dict' in ckpt:
                    self.model.load_state_dict(ckpt['model_state_dict'])
                else:
                    self.model.load_state_dict(ckpt)
                self.model = self.model.to(self.device)
                self.model.eval()
                print(f"  模型加载成功: {model_path}")
            except Exception as e:
                print(f"  模型加载跳过（当前使用纯规则方案）: {e}")
                self.model = None

        self.seq_len = 90

        # 每个 track 的状态
        self.pose_buffers = {}      # track_id -> deque of [yaw, pitch, roll]
        self.baselines = {}         # track_id -> [yaw0, pitch0, roll0]  (EMA)
        self.pred_history = {}      # track_id -> deque of (pred, conf)
        self.bbox_history = {}

    # ----------------------------------------------------------
    # baseline 估计
    # ----------------------------------------------------------
    def _update_baseline(self, track_id: str, yaw: float, pitch: float, roll: float):
        """用 EMA 维护每个 track 的 baseline 朝向（缓慢更新）。
        前10帧用较快的 alpha 建立初始 baseline，之后切换到慢速更新。
        """
        buf_len = len(self.pose_buffers.get(track_id, []))
        if track_id not in self.baselines:
            self.baselines[track_id] = [yaw, pitch, roll]
        else:
            # 前10帧快速收敛（alpha=0.2），之后慢速（alpha=0.02）
            alpha = 0.2 if buf_len <= 10 else 0.02
            bl = self.baselines[track_id]
            bl[0] += alpha * (yaw - bl[0])
            bl[1] += alpha * (pitch - bl[1])
            bl[2] += alpha * (roll - bl[2])

    # ----------------------------------------------------------
    # 中值滤波去噪
    # ----------------------------------------------------------
    @staticmethod
    def _median_filter(vals, k=3):
        """对序列做窗口为k的中值滤波，减少WHENet单帧噪声"""
        if len(vals) <= k:
            return vals
        out = list(vals[:k//2])
        for i in range(k//2, len(vals) - k//2):
            window = sorted(vals[i - k//2 : i + k//2 + 1])
            out.append(window[len(window)//2])
        out.extend(vals[len(vals) - k//2:])
        return out

    # ----------------------------------------------------------
    # 规则检测（全部 6 类）
    # ----------------------------------------------------------
    def _rule_check(self, track_id: str) -> tuple:
        """
        对 pose buffer 按规则判定 6 类行为（带中值滤波去噪）。

        Returns: (class_id, confidence)  class_id=None 表示 normal
        """
        buf = list(self.pose_buffers[track_id])
        bl = self.baselines.get(track_id, [0, 0, 0])
        n = len(buf)
        # WHENet 前20帧噪声很大，需要足够数据才开始判定
        if n < 20:
            return None, 0.0

        raw_yaws   = [p[0] for p in buf]
        raw_pitchs = [p[1] for p in buf]

        # 中值滤波去噪（WHENet单帧跳变严重）
        yaws   = self._median_filter(raw_yaws, k=5)
        pitchs = self._median_filter(raw_pitchs, k=5)

        # 相对于 baseline 的偏移
        d_yaws   = [y - bl[0] for y in yaws]
        d_pitchs = [p - bl[1] for p in pitchs]

        # -- 时间窗口（以 pose 帧数计） --
        w05s = max(2, round(0.5 / self.dt))   # 0.5 秒 = 5 samples
        w1s  = max(3, round(1.0 / self.dt))   # 1 秒 = 10 samples
        w3s  = max(5, round(3.0 / self.dt))   # 3 秒 = 30 samples
        w5s  = max(8, round(5.0 / self.dt))   # 5 秒 = 50 samples

        # ====== 2 quick_turn: 0.5s 内 |Δyaw| > 80° ======
        # 严格条件：
        #   1) 转头前2秒yaw稳定（std < 10°）
        #   2) 转头后1秒yaw也要稳定（排除纯噪声跳变）
        #   3) 前后yaw均值差 > 50°（确实转向了新方向）
        w2s = max(5, round(2.0 / self.dt))
        if n >= w2s + w05s + w1s:
            for i in range(max(0, n - w3s), n - w05s - w1s):
                delta = abs(yaws[i + w05s] - yaws[i])
                if delta > 80:
                    # 转头前2秒稳定
                    pre_start = max(0, i - w2s)
                    pre_seg = yaws[pre_start:i+1]
                    if len(pre_seg) >= 5 and np.std(pre_seg) < 10:
                        # 转头后1秒也稳定
                        post_seg = yaws[i + w05s:i + w05s + w1s]
                        if len(post_seg) >= 3 and np.std(post_seg) < 15:
                            # 前后均值差确实大
                            pre_mean = np.mean(pre_seg)
                            post_mean = np.mean(post_seg)
                            if abs(post_mean - pre_mean) > 50:
                                return 2, 0.92

        # ====== 1 glancing: 3s 内 yaw 方向翻转 ≥ 3 次，摆幅 > 30° ======
        if n >= w3s:
            seg = d_yaws[-w3s:]
            reversals = 0
            last_dir = 0
            for i in range(1, len(seg)):
                diff = seg[i] - seg[i - 1]
                cur_dir = 1 if diff > 0 else (-1 if diff < 0 else 0)
                if cur_dir != 0 and cur_dir != last_dir and last_dir != 0:
                    if abs(seg[i] - seg[i - 1]) > 5:  # 防微小抖动
                        reversals += 1
                if cur_dir != 0:
                    last_dir = cur_dir
            swing = max(seg) - min(seg)
            if reversals >= 3 and swing > 30:
                return 1, min(0.95, 0.6 + reversals * 0.05)

        # ====== 5 looking_up: pitch - baseline > 20° 持续 > 3s ======
        if n >= w3s:
            recent = d_pitchs[-w3s:]
            up_ratio = sum(1 for p in recent if p > 20) / len(recent)
            if up_ratio >= 0.7:
                return 5, min(0.95, 0.6 + up_ratio * 0.3)

        # ====== 4 looking_down: pitch - baseline < -20° 持续 > 5s ======
        if n >= w5s:
            recent = d_pitchs[-w5s:]
            dn_ratio = sum(1 for p in recent if p < -20) / len(recent)
            if dn_ratio >= 0.7:
                return 4, min(0.95, 0.6 + dn_ratio * 0.3)

        # ====== 3 prolonged_watch: |yaw - baseline| > 30° 持续 > 3s ======
        if n >= w3s:
            recent = d_yaws[-w3s:]
            off_ratio = sum(1 for y in recent if abs(y) > 30) / len(recent)
            if off_ratio >= 0.7:
                return 3, min(0.95, 0.6 + off_ratio * 0.3)

        # ====== 0 normal ======
        return 0, 0.80

    # ----------------------------------------------------------
    # bbox 平滑
    # ----------------------------------------------------------
    def smooth_bbox(self, track_id: str, bbox: list, alpha: float = 0.3) -> list:
        if track_id not in self.bbox_history:
            self.bbox_history[track_id] = bbox
            return bbox
        old = self.bbox_history[track_id]
        smoothed = [alpha * bbox[i] + (1 - alpha) * old[i] for i in range(4)]
        self.bbox_history[track_id] = smoothed
        return smoothed

    # ----------------------------------------------------------
    # 主更新接口
    # ----------------------------------------------------------
    def update(self, track_id: str, yaw: float, pitch: float, roll: float):
        """喂入一帧真实 pose，返回 (class_id, confidence)"""
        if track_id not in self.pose_buffers:
            self.pose_buffers[track_id] = deque(maxlen=self.seq_len)
            self.pred_history[track_id] = deque(maxlen=self.smooth_window)

        self.pose_buffers[track_id].append([yaw, pitch, roll])
        self._update_baseline(track_id, yaw, pitch, roll)

        buf_len = len(self.pose_buffers[track_id])
        if buf_len < 20:
            return None, 0.0

        pred, conf = self._rule_check(track_id)
        self.pred_history[track_id].append((pred, conf))
        return self._get_smoothed_pred(track_id)

    def get_last_pred(self, track_id: str):
        if track_id in self.pred_history and self.pred_history[track_id]:
            return self._get_smoothed_pred(track_id)
        return None, 0.0

    def _get_smoothed_pred(self, track_id: str):
        """多数投票 + 置信度加权"""
        history = self.pred_history[track_id]
        if not history:
            return None, 0.0
        votes = {}
        for pred, conf in history:
            votes[pred] = votes.get(pred, 0) + conf
        best_pred = max(votes, key=votes.get)
        avg_conf = votes[best_pred] / len(history)
        return best_pred, avg_conf


def load_data(video_name: str):
    """加载跟踪数据和姿态数据"""
    tracking_path = project_root / 'data' / 'tracked_output' / video_name / 'tracking_result.json'
    with open(tracking_path, 'r') as f:
        tracking_data = json.load(f)

    pose_path = project_root / 'data' / 'pose_output' / f'{video_name}_poses.json'
    with open(pose_path, 'r') as f:
        pose_data = json.load(f)

    return tracking_data, pose_data


def build_frame_data(tracking_data, pose_data):
    """构建每帧的数据: frame_id -> [{track_id, bbox, yaw, pitch, roll, has_pose}, ...]"""
    frame_dict = {}
    pose_tracks = pose_data.get('tracks', {})

    for track in tracking_data['tracks']:
        track_id = track['track_id']
        frames = track['frames']
        bboxes = track['bboxes']

        pose_key = f'track_{track_id}' if isinstance(track_id, int) else track_id
        track_poses = pose_tracks.get(pose_key, {}).get('poses', [])
        pose_by_frame = {p['frame']: p for p in track_poses}

        for i, frame_id in enumerate(frames):
            if frame_id not in frame_dict:
                frame_dict[frame_id] = []

            bbox = bboxes[i] if i < len(bboxes) else [0, 0, 100, 100]
            pose = pose_by_frame.get(frame_id, None)

            frame_dict[frame_id].append({
                'track_id': pose_key,
                'bbox': bbox,
                'yaw': pose['yaw'] if pose else 0,
                'pitch': pose['pitch'] if pose else 0,
                'roll': pose['roll'] if pose else 0,
                'has_pose': pose is not None,
            })

    return frame_dict


def run_inference(video_name: str, model_path: str, output_path: str):
    """在视频上运行推理"""

    # 加载数据
    print(f"加载数据: {video_name}")
    tracking_data, pose_data = load_data(video_name)

    # 构建帧数据
    frame_dict = build_frame_data(tracking_data, pose_data)
    print(f"有效帧数: {len(frame_dict)}")

    # 使用原始视频（不用已标注的 tracked 视频）
    raw_video_path = tracking_data.get('video_path', '')
    video_path = project_root / raw_video_path
    if not video_path.exists():
        # fallback: tracked video
        video_path = project_root / 'data' / 'tracked_output' / video_name / f'{video_name}_tracked.mp4'
    if not video_path.exists():
        print(f"错误: 找不到视频文件: {video_path}")
        return

    # 打开视频
    print(f"打开视频: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频信息: {width}x{height}, {fps:.1f}fps, {total_frames}帧")

    # 创建输出视频
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 初始化识别器
    print(f"加载模型: {model_path}")
    recognizer = BehaviorRecognizer(model_path)

    # 统计: 每个 track 的最严重行为
    track_behaviors = {}

    # 处理每一帧
    print("开始推理...")
    frame_id = 0
    pbar = tqdm(total=total_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_data = frame_dict.get(frame_id, [])

        for item in frame_data:
            track_id = item['track_id']
            bbox = item['bbox']

            # 平滑边界框
            smoothed_bbox = recognizer.smooth_bbox(track_id, bbox)

            # 从全身框推算头部框（真实头部比例）
            bx1, by1, bx2, by2 = smoothed_bbox
            body_h = by2 - by1
            head_h = body_h * 0.14
            head_w = head_h * 0.85
            head_cx = (bx1 + bx2) / 2
            x1 = int(head_cx - head_w / 2)
            y1 = int(by1)
            x2 = int(head_cx + head_w / 2)
            y2 = int(by1 + head_h)

            # 只在有真实pose数据时才更新模型，避免(0,0,0)填充造成假的头部摆动
            if item['has_pose']:
                pred, conf = recognizer.update(track_id, item['yaw'], item['pitch'], item['roll'])
            else:
                pred, conf = recognizer.get_last_pred(track_id)

            # 短ID
            short_id = track_id.split('_')[-1] if '_' in track_id else track_id[-4:]

            if pred is None:
                color = (128, 128, 128)
                label = f"#{short_id}"
            else:
                class_name, color = BEHAVIOR_CLASSES[pred]
                label = f"#{short_id} {class_name} {conf*100:.0f}%"

                # 更新 track 最严重行为
                if pred > 0:
                    if track_id not in track_behaviors:
                        track_behaviors[track_id] = pred
                    else:
                        priority = {2: 5, 1: 4, 3: 3, 4: 2, 5: 1, 0: 0}
                        if priority.get(pred, 0) > priority.get(track_behaviors[track_id], 0):
                            track_behaviors[track_id] = pred

            # 绘制头部框（唯一的框）
            thickness = 3 if pred and pred > 0 else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # 标签
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+8, y1), color, -1)
            cv2.putText(frame, label, (x1+4, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 统计面板
        behavior_person_counts = {i: 0 for i in range(6)}
        for b in track_behaviors.values():
            behavior_person_counts[b] += 1

        panel_height = 110
        cv2.rectangle(frame, (0, 0), (280, panel_height), (0, 0, 0), -1)
        cv2.putText(frame, f"Frame {frame_id} | Suspects: {len(track_behaviors)}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        y_offset = 45
        for i in [1, 2, 3, 4, 5]:
            class_name, color = BEHAVIOR_CLASSES[i]
            count = behavior_person_counts[i]
            if count > 0:
                cv2.putText(frame, f"{class_name}: {count}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 18

        out.write(frame)
        frame_id += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    print(f"\n推理完成!")
    print(f"  可疑人员: {len(track_behaviors)}")
    for i in range(1, 6):
        class_name, _ = BEHAVIOR_CLASSES[i]
        count = behavior_person_counts.get(i, 0)
        if count > 0:
            print(f"    [{i}] {class_name:12s}: {count} 人")
    print(f"  输出视频: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='视频行为识别推理（6类）')
    parser.add_argument('--video', type=str, default='MVI_4540', help='视频名称')
    parser.add_argument('--model', type=str, default='checkpoints/transformer_best.pt', help='模型路径')
    parser.add_argument('--output', type=str, help='输出视频路径')
    args = parser.parse_args()

    if not args.output:
        args.output = f'data/inference_output/{args.video}_behavior.mp4'

    run_inference(args.video, args.model, args.output)


if __name__ == '__main__':
    main()
