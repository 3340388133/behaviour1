"""
视频Transformer推理脚本

支持两种模式：
  1. 在线推理：实时处理视频流
  2. 离线推理：批量处理视频文件

用法:
  # 单个视频推理
  python inference_video_transformer.py --video input.mp4 --model best_model.pth

  # 批量推理
  python inference_video_transformer.py --video_dir videos/ --model best_model.pth
"""

import torch
import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict
import torch.nn.functional as F
from video_transformer_pipeline import VideoTransformerClassifier


class VideoTransformerInference:
    """视频Transformer推理器"""

    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        num_frames: int = 16,
        img_size: int = 224,
        conf_threshold: float = 0.5
    ):
        self.device = device
        self.num_frames = num_frames
        self.img_size = img_size
        self.conf_threshold = conf_threshold

        # 标签映射
        self.label_names = {
            0: 'normal',
            1: 'looking_around',
            2: 'unknown'
        }

        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = self._load_model(model_path)
        self.model.eval()

        print(f"推理器初始化完成")
        print(f"  设备: {device}")
        print(f"  帧数: {num_frames}")
        print(f"  图像大小: {img_size}")

    def _load_model(self, model_path: str):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)

        # 创建模型（使用默认参数，实际应该从config读取）
        model = VideoTransformerClassifier(
            num_classes=3,
            img_size=self.img_size,
            num_frames=self.num_frames
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)

        return model

    def preprocess_video(self, video_path: str) -> torch.Tensor:
        """
        预处理视频

        Returns:
            clip: [T, C, H, W] tensor
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 均匀采样帧
        if total_frames < self.num_frames:
            # 重复采样
            indices = np.random.choice(total_frames, self.num_frames, replace=True)
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        # 读取帧
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # 黑帧
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

        cap.release()

        frames = np.array(frames)  # [T, H, W, C]

        # 转换为tensor
        clip = torch.from_numpy(frames).float()
        clip = clip.permute(0, 3, 1, 2)  # [T, C, H, W]
        clip = clip / 255.0

        # 归一化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        clip = (clip - mean) / std

        # Resize
        clip = F.interpolate(
            clip,
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        )

        return clip

    def predict(self, video_path: str) -> Dict:
        """
        预测单个视频

        Returns:
            result: {
                'label': 'looking_around',
                'confidence': 0.85,
                'probabilities': [0.05, 0.85, 0.10]
            }
        """
        # 预处理
        clip = self.preprocess_video(video_path)
        clip = clip.unsqueeze(0).to(self.device)  # [1, T, C, H, W]

        # 推理
        with torch.no_grad():
            logits = self.model(clip)
            probs = F.softmax(logits, dim=1)

        # 解析结果
        probs = probs.cpu().numpy()[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        result = {
            'label': self.label_names[pred_class],
            'confidence': confidence,
            'probabilities': {
                self.label_names[i]: float(probs[i])
                for i in range(len(probs))
            }
        }

        return result

    def predict_batch(self, video_paths: List[str]) -> List[Dict]:
        """批量预测"""
        results = []
        for video_path in video_paths:
            try:
                result = self.predict(video_path)
                result['video_path'] = video_path
                results.append(result)
                print(f"✓ {Path(video_path).name}: {result['label']} ({result['confidence']:.2f})")
            except Exception as e:
                print(f"✗ {Path(video_path).name}: 错误 - {e}")
                results.append({
                    'video_path': video_path,
                    'error': str(e)
                })

        return results

    def predict_with_temporal_windows(
        self,
        video_path: str,
        window_size: int = 16,
        stride: int = 8
    ) -> List[Dict]:
        """
        滑动窗口预测（用于长视频）

        Returns:
            List of predictions for each window
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        results = []
        start_idx = 0

        while start_idx + window_size <= total_frames:
            # 提取窗口
            frames = []
            for i in range(start_idx, start_idx + window_size):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)

            if len(frames) == window_size:
                # 预处理
                frames = np.array(frames)
                clip = torch.from_numpy(frames).float()
                clip = clip.permute(0, 3, 1, 2)
                clip = clip / 255.0

                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                clip = (clip - mean) / std

                clip = F.interpolate(
                    clip,
                    size=(self.img_size, self.img_size),
                    mode='bilinear',
                    align_corners=False
                )

                clip = clip.unsqueeze(0).to(self.device)

                # 推理
                with torch.no_grad():
                    logits = self.model(clip)
                    probs = F.softmax(logits, dim=1)

                probs = probs.cpu().numpy()[0]
                pred_class = int(np.argmax(probs))
                confidence = float(probs[pred_class])

                # 记录结果
                results.append({
                    'start_frame': start_idx,
                    'end_frame': start_idx + window_size,
                    'start_time': start_idx / fps,
                    'end_time': (start_idx + window_size) / fps,
                    'label': self.label_names[pred_class],
                    'confidence': confidence,
                    'probabilities': {
                        self.label_names[i]: float(probs[i])
                        for i in range(len(probs))
                    }
                })

            start_idx += stride

        cap.release()
        return results


def main(args):
    # 创建推理器
    inferencer = VideoTransformerInference(
        model_path=args.model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_frames=args.num_frames,
        img_size=args.img_size,
        conf_threshold=args.conf_threshold
    )

    # 单个视频
    if args.video:
        print(f"\n推理视频: {args.video}")

        if args.sliding_window:
            # 滑动窗口模式
            results = inferencer.predict_with_temporal_windows(
                args.video,
                window_size=args.num_frames,
                stride=args.window_stride
            )

            print(f"\n检测到 {len(results)} 个时间窗口:")
            for i, res in enumerate(results):
                print(f"  窗口 {i+1}: "
                      f"{res['start_time']:.1f}s-{res['end_time']:.1f}s | "
                      f"{res['label']} ({res['confidence']:.2f})")

            # 保存结果
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\n结果已保存到: {args.output}")

        else:
            # 单次预测
            result = inferencer.predict(args.video)

            print(f"\n预测结果:")
            print(f"  标签: {result['label']}")
            print(f"  置信度: {result['confidence']:.4f}")
            print(f"  概率分布:")
            for label, prob in result['probabilities'].items():
                print(f"    {label}: {prob:.4f}")

    # 批量处理
    elif args.video_dir:
        video_dir = Path(args.video_dir)
        video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi'))

        print(f"\n找到 {len(video_files)} 个视频文件")

        results = inferencer.predict_batch([str(f) for f in video_files])

        # 保存结果
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n结果已保存到: {args.output}")

    else:
        print("错误: 请指定 --video 或 --video_dir")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='视频Transformer推理')

    # 输入
    parser.add_argument('--video', type=str, help='单个视频文件')
    parser.add_argument('--video_dir', type=str, help='视频目录（批量处理）')
    parser.add_argument('--model', type=str, required=True,
                        help='模型文件路径')

    # 推理参数
    parser.add_argument('--num_frames', type=int, default=16,
                        help='采样帧数')
    parser.add_argument('--img_size', type=int, default=224,
                        help='图像大小')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                        help='置信度阈值')

    # 滑动窗口
    parser.add_argument('--sliding_window', action='store_true',
                        help='使用滑动窗口（用于长视频）')
    parser.add_argument('--window_stride', type=int, default=8,
                        help='窗口滑动步长')

    # 输出
    parser.add_argument('--output', type=str,
                        help='结果保存路径（JSON格式）')

    args = parser.parse_args()
    main(args)
