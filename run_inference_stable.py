#!/usr/bin/env python3
"""
稳定版视频推理脚本
- 更强的时序平滑（30帧窗口）
- 更强的边界框平滑
- 累积统计（只增不减）
- 固定颜色和ID显示
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

# 行为类别（英文）
BEHAVIOR_CLASSES = {
    0: ('Normal', (0, 200, 0)),
    1: ('Glancing', (0, 0, 255)),
    2: ('QuickTurn', (0, 128, 255)),
    3: ('Prolonged', (255, 0, 128)),
    4: ('LookDown', (255, 128, 0)),
    5: ('LookUp', (0, 255, 255)),
}


class StableBehaviorRecognizer:
    """稳定版行为识别器"""

    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.model = create_model(
            model_type='transformer',
            pose_input_dim=3,
            pose_d_model=64,
            pose_nhead=4,
            pose_num_layers=2,
            use_multimodal=False,
            hidden_dim=128,
            num_classes=6,
            dropout=0.1,
            uncertainty_weighting=True,
        )

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'])
        else:
            self.model.load_state_dict(ckpt)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.pose_buffers = {}
        self.seq_len = 90

        # 更大的平滑窗口
        self.smooth_window = 30
        self.pred_history = {}

        # 边界框平滑（更强）
        self.bbox_history = {}
        self.bbox_alpha = 0.15  # 更低 = 更平滑

        # 每个track的确定行为（一旦确定不再改变）
        self.confirmed_behaviors = {}

    def smooth_bbox(self, track_id: str, bbox: list) -> list:
        """强平滑边界框"""
        if track_id not in self.bbox_history:
            self.bbox_history[track_id] = list(bbox)
            return bbox

        old = self.bbox_history[track_id]
        smoothed = [
            self.bbox_alpha * bbox[i] + (1 - self.bbox_alpha) * old[i]
            for i in range(4)
        ]
        self.bbox_history[track_id] = smoothed
        return smoothed

    def get_last_pred(self, track_id: str):
        """返回该track的最近稳定预测（无新pose时复用）"""
        if track_id in self.pred_history and len(self.pred_history[track_id]) >= 5:
            return self._get_stable_pred(track_id)
        return None, 0.0

    def update(self, track_id: str, yaw: float, pitch: float, roll: float):
        """更新并返回稳定预测"""
        if track_id not in self.pose_buffers:
            self.pose_buffers[track_id] = deque(maxlen=self.seq_len)
            self.pred_history[track_id] = deque(maxlen=self.smooth_window)

        pose = [yaw, pitch, roll]
        self.pose_buffers[track_id].append(pose)

        # 至少15帧才开始预测，不足seq_len时用首帧填充
        buf_len = len(self.pose_buffers[track_id])
        if buf_len < 15:
            return None, 0.0

        pose_list = list(self.pose_buffers[track_id])
        if buf_len < self.seq_len:
            pad = [pose_list[0]] * (self.seq_len - buf_len)
            pose_list = pad + pose_list

        pose_seq = np.array(pose_list, dtype=np.float32)
        pose_tensor = torch.from_numpy(pose_seq).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _ = self.model(pose_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            conf = probs[0, pred].item()

        self.pred_history[track_id].append((pred, conf))

        # 多数投票
        return self._get_stable_pred(track_id)

    def _get_stable_pred(self, track_id: str):
        """稳定预测（需要多次确认才改变）"""
        history = self.pred_history[track_id]
        if len(history) < 5:
            return None, 0.0

        # 统计各类别票数
        votes = {}
        for pred, conf in history:
            if pred not in votes:
                votes[pred] = []
            votes[pred].append(conf)

        # 选择出现最多的（至少要有1/3的票）
        min_votes = len(history) // 3
        best_pred = None
        best_count = 0

        for pred, confs in votes.items():
            if len(confs) > best_count and len(confs) >= min_votes:
                best_pred = pred
                best_count = len(confs)

        if best_pred is None:
            return 0, 0.5  # 默认Normal

        avg_conf = sum(votes[best_pred]) / len(votes[best_pred])
        return best_pred, avg_conf

    def confirm_behavior(self, track_id: str, behavior: int):
        """确认某个track的可疑行为（不可逆）"""
        if behavior > 0:  # 非Normal
            if track_id not in self.confirmed_behaviors:
                self.confirmed_behaviors[track_id] = behavior
            else:
                # 保留更严重的行为
                priority = {2: 5, 1: 4, 3: 3, 4: 2, 5: 1, 0: 0}
                if priority.get(behavior, 0) > priority.get(self.confirmed_behaviors[track_id], 0):
                    self.confirmed_behaviors[track_id] = behavior


def load_data(video_name: str, project_root: Path, use_stable: bool = True):
    """加载数据"""
    if use_stable:
        tracking_path = project_root / 'data' / 'tracked_output_stable' / video_name / 'tracking_result.json'
        pose_path = project_root / 'data' / 'pose_output_stable' / f'{video_name}_poses.json'
    else:
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
    """运行推理"""
    project_root = Path(__file__).parent

    print(f"Loading: {video_name}")
    tracking_data, pose_data = load_data(video_name, project_root, use_stable=True)
    frame_dict = build_frame_data(tracking_data, pose_data)

    # Use raw video (not tracked video which has annotations burned in)
    raw_video_path = tracking_data.get('video_path', '')
    video_path = project_root / raw_video_path
    if not video_path.exists():
        # fallback: try tracked video
        video_path = project_root / 'data' / 'tracked_output_stable' / video_name / f'{video_name}_tracked.mp4'
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height}, {fps:.1f}fps, {total_frames} frames")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Model: {model_path}")
    recognizer = StableBehaviorRecognizer(model_path)

    print("Inferencing...")
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

            # 只在有真实pose数据时才更新模型
            if item['has_pose']:
                pred, conf = recognizer.update(track_id, item['yaw'], item['pitch'], item['roll'])
            else:
                pred, conf = recognizer.get_last_pred(track_id)

            # 短ID
            short_id = track_id.split('_')[-1] if '_' in track_id else track_id[-4:]

            if pred is None:
                color = (100, 100, 100)
                label = f"#{short_id}"
            else:
                # 确认行为
                recognizer.confirm_behavior(track_id, pred)

                # 使用确认的行为（更稳定）
                final_behavior = recognizer.confirmed_behaviors.get(track_id, pred)
                class_name, color = BEHAVIOR_CLASSES[final_behavior]

                if final_behavior == 0:
                    label = f"#{short_id}"
                    color = (0, 180, 0)
                else:
                    label = f"#{short_id} {class_name}"

            # 绘制
            thickness = 3 if pred and pred > 0 else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+8, y1), color, -1)
            cv2.putText(frame, label, (x1+4, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 统计面板（累积，只显示确认的可疑人员）
        confirmed = recognizer.confirmed_behaviors
        panel_h = 90
        cv2.rectangle(frame, (0, 0), (250, panel_h), (0, 0, 0), -1)

        cv2.putText(frame, f"Suspects: {len(confirmed)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 按类别统计
        counts = {}
        for b in confirmed.values():
            counts[b] = counts.get(b, 0) + 1

        y = 50
        for cls_id in [1, 2, 3, 4, 5]:
            if cls_id in counts:
                name, color = BEHAVIOR_CLASSES[cls_id]
                cv2.putText(frame, f"{name}: {counts[cls_id]}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y += 18

        out.write(frame)
        frame_id += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    # 最终统计
    print(f"\n=== Results ===")
    print(f"Suspicious persons: {len(recognizer.confirmed_behaviors)}")

    counts = {}
    for b in recognizer.confirmed_behaviors.values():
        counts[b] = counts.get(b, 0) + 1

    for cls_id in sorted(counts.keys()):
        name, _ = BEHAVIOR_CLASSES[cls_id]
        print(f"  {name}: {counts[cls_id]}")

    print(f"\nOutput: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='MVI_4537')
    parser.add_argument('--model', default='checkpoints/transformer_balanced.pt')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    if not args.output:
        args.output = f'data/inference_output/{args.video}_stable.mp4'

    run_inference(args.video, args.model, args.output)


if __name__ == '__main__':
    main()
