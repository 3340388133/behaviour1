#!/usr/bin/env python3
"""
视频推理脚本 - 使用训练好的 Transformer 模型识别6类可疑行为

行为类别：
- 0: normal        正常行为      绿色
- 1: glancing      频繁张望      红色
- 2: quick_turn    快速回头      橙色
- 3: prolonged_watch 长时间观察  紫色
- 4: looking_down  持续低头      蓝色
- 5: looking_up    持续抬头      黄色
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'recognition'))

import json
import argparse
import cv2
import torch
import numpy as np
from collections import deque
from tqdm import tqdm

from temporal_transformer import create_model

# 6类行为定义 (英文标签避免中文乱码)
BEHAVIOR_CLASSES = {
    0: ('Normal', (0, 255, 0)),           # 绿色
    1: ('Glancing', (0, 0, 255)),         # 红色
    2: ('QuickTurn', (0, 128, 255)),      # 橙色
    3: ('Prolonged', (255, 0, 128)),      # 紫色
    4: ('LookDown', (255, 128, 0)),       # 蓝色
    5: ('LookUp', (0, 255, 255)),         # 黄色
}


class RuleDetector:
    """规则检测器 - 处理有精确阈值的行为: quick_turn(2), prolonged_watch(3), looking_up(5)"""

    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def check(self, pose_buffer: list) -> tuple:
        if len(pose_buffer) < 10:
            return None, 0.0
        yaws = [p[0] for p in pose_buffer]
        pitchs = [p[1] for p in pose_buffer]

        # quick_turn: yaw变化>60° within 0.5s
        w05 = max(1, int(self.fps * 0.5 / 3))
        if len(yaws) >= w05:
            for i in range(len(yaws) - w05):
                if abs(yaws[i + w05] - yaws[i]) > 60:
                    return 2, 0.90

        # looking_up: pitch>20° 持续>3s
        w3s = max(1, int(self.fps * 3.0 / 3))
        if len(pitchs) >= w3s:
            up = sum(1 for p in pitchs[-w3s:] if p > 20)
            if up >= w3s * 0.7:
                return 5, 0.85

        # prolonged_watch: |yaw|>30° 持续>3s
        if len(yaws) >= w3s:
            off = sum(1 for y in yaws[-w3s:] if abs(y) > 30)
            if off >= w3s * 0.7:
                return 3, 0.85

        return None, 0.0


class BehaviorRecognizer:
    """混合识别器: 模型(normal/glancing/looking_down) + 规则(quick_turn/prolonged_watch/looking_up)"""

    def __init__(self, model_path: str, device: str = 'cuda',
                 smooth_window: int = 15, fps: float = 30.0):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.smooth_window = smooth_window

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

        self.rule_detector = RuleDetector(fps=fps)
        self.pose_buffers = {}
        self.seq_len = 90
        self.pred_history = {}
        self.bbox_history = {}

    def smooth_bbox(self, track_id: str, bbox: list, alpha: float = 0.3) -> list:
        if track_id not in self.bbox_history:
            self.bbox_history[track_id] = bbox
            return bbox
        old = self.bbox_history[track_id]
        smoothed = [alpha * bbox[i] + (1 - alpha) * old[i] for i in range(4)]
        self.bbox_history[track_id] = smoothed
        return smoothed

    def update(self, track_id: str, yaw: float, pitch: float, roll: float, has_pose: bool = True):
        if track_id not in self.pose_buffers:
            self.pose_buffers[track_id] = deque(maxlen=self.seq_len)
            self.pred_history[track_id] = deque(maxlen=self.smooth_window)

        if not has_pose:
            if self.pred_history[track_id]:
                return self._get_smoothed_pred(track_id)
            return None, 0.0

        self.pose_buffers[track_id].append([yaw, pitch, roll])
        buf_len = len(self.pose_buffers[track_id])
        if buf_len < 15:
            return None, 0.0

        # 1) 规则检测
        rule_pred, rule_conf = self.rule_detector.check(list(self.pose_buffers[track_id]))

        # 2) 模型推理
        pose_list = list(self.pose_buffers[track_id])
        if buf_len < self.seq_len:
            pad = [pose_list[0]] * (self.seq_len - buf_len)
            pose_list = pad + pose_list
        pose_seq = np.array(pose_list, dtype=np.float32)
        pose_tensor = torch.from_numpy(pose_seq).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.model(pose_tensor)
            probs = torch.softmax(logits, dim=1)
            model_pred = logits.argmax(dim=1).item()
            model_conf = probs[0, model_pred].item()

        # 3) 规则优先
        if rule_pred is not None and rule_conf > 0.5:
            pred, conf = rule_pred, rule_conf
        else:
            pred, conf = model_pred, model_conf

        self.pred_history[track_id].append((pred, conf))
        return self._get_smoothed_pred(track_id)

    def _get_smoothed_pred(self, track_id: str):
        history = self.pred_history[track_id]
        if not history:
            return None, 0.0
        votes = {}
        for pred, conf in history:
            votes[pred] = votes.get(pred, 0) + conf
        best_pred = max(votes, key=votes.get)
        avg_conf = votes[best_pred] / len(history)
        return best_pred, avg_conf


def load_data(video_name: str, project_root: Path):
    """加载跟踪和姿态数据"""
    tracking_path = project_root / 'data' / 'tracked_output' / video_name / 'tracking_result.json'
    pose_path = project_root / 'data' / 'pose_output' / f'{video_name}_poses.json'

    with open(tracking_path, 'r') as f:
        tracking_data = json.load(f)
    with open(pose_path, 'r') as f:
        pose_data = json.load(f)

    return tracking_data, pose_data


def build_frame_data(tracking_data, pose_data):
    """构建每帧数据"""
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
            pose = pose_by_frame.get(frame_id, {})

            frame_dict[frame_id].append({
                'track_id': pose_key,
                'bbox': bbox,
                'yaw': pose.get('yaw', 0),
                'pitch': pose.get('pitch', 0),
                'roll': pose.get('roll', 0),
                'has_pose': len(pose) > 0,  # 标记是否有真实姿态数据
            })

    return frame_dict


def run_inference(video_name: str, model_path: str, output_path: str):
    """运行推理"""
    project_root = Path(__file__).parent

    print(f"Loading data: {video_name}")
    tracking_data, pose_data = load_data(video_name, project_root)
    frame_dict = build_frame_data(tracking_data, pose_data)
    print(f"Valid frames: {len(frame_dict)}")

    # Find raw video (use original video, not tracked video which has annotations burned in)
    raw_video_path = tracking_data.get('video_path', '')
    video_path = project_root / raw_video_path
    if not video_path.exists():
        # fallback: try tracked video
        video_path = project_root / 'data' / 'tracked_output' / video_name / f'{video_name}_tracked.mp4'
    if not video_path.exists():
        print(f"Error: Video not found {video_path}")
        return

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height}, {fps:.1f}fps, {total_frames} frames")

    # Output video
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize model
    print(f"Loading model: {model_path}")
    recognizer = BehaviorRecognizer(model_path)

    # Stats for each class
    stats = {i: 0 for i in range(6)}
    stats['waiting'] = 0
    # Track behaviors (most severe behavior per track)
    track_behaviors = {}  # track_id -> max_behavior_class

    print("Starting inference...")
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
            yaw, pitch, roll = item['yaw'], item['pitch'], item['roll']
            has_pose = item.get('has_pose', True)

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

            pred, conf = recognizer.update(track_id, yaw, pitch, roll, has_pose)

            # 提取简短ID用于显示
            short_id = track_id.split('_')[-1] if '_' in track_id else track_id[-4:]

            if pred is None:
                color = (128, 128, 128)
                label = f"#{short_id}"
                stats['waiting'] += 1
            else:
                # 获取类别信息
                class_name, color = BEHAVIOR_CLASSES[pred]
                label = f"#{short_id} {class_name}"
                stats[pred] += 1

                # 更新track的最严重行为
                if pred > 0:
                    if track_id not in track_behaviors:
                        track_behaviors[track_id] = pred
                    else:
                        priority = {2: 5, 1: 4, 3: 3, 4: 2, 5: 1, 0: 0}
                        if priority.get(pred, 0) > priority.get(track_behaviors[track_id], 0):
                            track_behaviors[track_id] = pred

            # 绘制边界框（加粗非正常行为）
            thickness = 3 if pred and pred > 0 else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # 标签背景和文字
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+8, y1), color, -1)
            cv2.putText(frame, label, (x1+4, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 统计信息面板 - 显示累积统计（按人数，更稳定）
        panel_height = 100
        cv2.rectangle(frame, (0, 0), (280, panel_height), (0, 0, 0), -1)

        # 计算各类别的人数（不是帧数）
        behavior_person_counts = {i: 0 for i in range(6)}
        for tid, behavior in track_behaviors.items():
            behavior_person_counts[behavior] += 1

        # 标题
        cv2.putText(frame, f"Suspicious Persons: {len(track_behaviors)}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 各类别人数统计（只显示非正常行为）
        y_offset = 45
        for i in [1, 2, 3, 4, 5]:  # 跳过Normal
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

    print(f"\nInference Done!")
    print(f"  Suspicious persons: {len(track_behaviors)}")
    print(f"\n  Frame count by class:")
    for i in range(6):
        class_name, _ = BEHAVIOR_CLASSES[i]
        print(f"    [{i}] {class_name:12s}: {stats[i]:6d}")
    print(f"    Waiting: {stats['waiting']}")

    # 按行为类型统计人数
    print(f"\n  Person count by class:")
    behavior_counts = {i: 0 for i in range(6)}
    for track_id, behavior in track_behaviors.items():
        behavior_counts[behavior] += 1
    for i in range(1, 6):  # 跳过normal
        class_name, _ = BEHAVIOR_CLASSES[i]
        if behavior_counts[i] > 0:
            print(f"    [{i}] {class_name:12s}: {behavior_counts[i]} persons")

    print(f"\n  Output video: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='可疑行为识别推理')
    parser.add_argument('--video', type=str, default='MVI_4537', help='视频名称')
    parser.add_argument('--model', type=str, default='checkpoints/transformer_best.pt')
    parser.add_argument('--output', type=str, help='输出路径')
    args = parser.parse_args()

    if not args.output:
        args.output = f'data/inference_output/{args.video}_suspicious.mp4'

    run_inference(args.video, args.model, args.output)


if __name__ == '__main__':
    main()
