"""
动作识别推理脚本
用于对单个视频或视频目录进行动作识别
"""

import argparse
import cv2
import torch
import numpy as np
from pathlib import Path

from action_model import R3DNet


class ActionRecognizer:
    """动作识别器"""

    def __init__(self, model_path: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_frames = 16
        self.frame_size = (224, 224)

        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classes = checkpoint['classes']

        self.model = R3DNet(num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"模型加载完成，类别: {self.classes}")

    def _load_video(self, video_path: str) -> np.ndarray:
        """加载视频并均匀采样帧"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            raise ValueError(f"无法读取视频: {video_path}")

        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.frame_size)
                frames.append(frame)
            else:
                frames.append(np.zeros((*self.frame_size, 3), dtype=np.uint8))

        cap.release()
        return np.array(frames)

    @torch.no_grad()
    def predict(self, video_path: str) -> dict:
        """对单个视频进行预测"""
        frames = self._load_video(video_path)

        # 转换为 tensor
        frames = torch.from_numpy(frames).float()
        frames = frames.permute(3, 0, 1, 2) / 255.0  # (C, T, H, W)
        frames = frames.unsqueeze(0).to(self.device)  # (1, C, T, H, W)

        outputs = self.model(frames)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()

        return {
            'class': self.classes[pred_idx],
            'confidence': probs[pred_idx].item(),
            'all_probs': {c: p.item() for c, p in zip(self.classes, probs)}
        }

    def predict_directory(self, video_dir: str) -> list:
        """对目录中的所有视频进行预测"""
        video_dir = Path(video_dir)
        extensions = {'.mp4', '.avi', '.mov', '.mkv'}
        results = []

        for video_file in video_dir.iterdir():
            if video_file.suffix.lower() in extensions:
                try:
                    result = self.predict(str(video_file))
                    result['file'] = video_file.name
                    results.append(result)
                except Exception as e:
                    print(f"处理 {video_file.name} 失败: {e}")

        return results


def main():
    parser = argparse.ArgumentParser(description='动作识别推理')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--video', type=str, help='单个视频路径')
    parser.add_argument('--video_dir', type=str, help='视频目录路径')

    args = parser.parse_args()

    recognizer = ActionRecognizer(args.model)

    if args.video:
        result = recognizer.predict(args.video)
        print(f"\n预测结果:")
        print(f"  动作: {result['class']}")
        print(f"  置信度: {result['confidence']:.2%}")

    elif args.video_dir:
        results = recognizer.predict_directory(args.video_dir)
        print(f"\n共处理 {len(results)} 个视频:")
        for r in results:
            print(f"  {r['file']}: {r['class']} ({r['confidence']:.2%})")


if __name__ == '__main__':
    main()
