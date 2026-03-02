#!/usr/bin/env python3
"""
Step 4: 头部姿态估计
对跟踪裁剪的人物图像进行头部姿态估计（Yaw, Pitch, Roll）

支持模型：
- WHENet（推荐，ONNX推理，支持大角度）
- 6DRepNet（需要网络下载权重）
"""

import os
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import argparse
from tqdm import tqdm

# ============== 配置 ==============
DATA_ROOT = Path("data")
TRACKED_OUTPUT_DIR = DATA_ROOT / "tracked_output"
POSE_OUTPUT_DIR = DATA_ROOT / "pose_output"


class WHENetPoseEstimator:
    """WHENet 头部姿态估计器 (ONNX)"""

    def __init__(self, model_path: str = None, device: str = "cuda:0"):
        self.device = device
        if model_path is None:
            model_path = str(Path(__file__).parent / "models" / "whenet_1x3x224x224_prepost.onnx")
        self.model_path = model_path
        self.session = None
        self._init_model()

    def _init_model(self):
        """初始化 ONNX 模型"""
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"加载 WHENet 模型成功: {self.model_path}")

    def estimate(self, image: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        估计头部姿态

        Args:
            image: BGR 图像（人脸或头部区域）

        Returns:
            (yaw, pitch, roll) 角度，单位：度
        """
        if self.session is None:
            return None

        try:
            # 预处理: resize 到 224x224
            resized = cv2.resize(image, (224, 224))
            input_data = resized.astype(np.float32)
            input_data = np.transpose(input_data, (2, 0, 1))  # HWC -> CHW
            input_data = np.expand_dims(input_data, 0)  # [1, 3, 224, 224]

            # 推理
            outputs = self.session.run(None, {self.input_name: input_data})
            # WHENet 输出: [yaw, roll, pitch]
            yaw = float(outputs[0][0][0])
            roll = float(outputs[0][0][1])
            pitch = float(outputs[0][0][2])

            return yaw, pitch, roll
        except Exception as e:
            return None


class SixDRepNetPoseEstimator:
    """6DRepNet 头部姿态估计器"""

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.model = None
        self._init_model()

    def _init_model(self):
        """初始化模型"""
        try:
            from sixdrepnet import SixDRepNet
            self.model = SixDRepNet()
            print("加载 6DRepNet 模型成功")
        except ImportError:
            print("6DRepNet 未安装，回退到 WHENet")
            self.model = None

    def estimate(self, image: np.ndarray) -> Optional[Tuple[float, float, float]]:
        if self.model is None:
            return None
        try:
            pitch, yaw, roll = self.model.predict(image)
            return float(yaw), float(pitch), float(roll)
        except Exception:
            return None


class HeadPoseProcessor:
    """头部姿态处理器"""

    def __init__(self, estimator_type: str = "whenet", device: str = "cuda:0",
                 model_path: str = None):
        self.device = device

        if estimator_type == "whenet":
            self.estimator = WHENetPoseEstimator(model_path, device)
        elif estimator_type == "6drepnet":
            self.estimator = SixDRepNetPoseEstimator(device)
        else:
            raise ValueError(f"不支持的估计器: {estimator_type}，请使用 whenet 或 6drepnet")

    @staticmethod
    def _crop_head_region(image: np.ndarray, head_ratio: float = 0.35) -> np.ndarray:
        """
        从全身裁剪图中截取头部区域

        Args:
            image: 全身裁剪图 (H, W, 3)
            head_ratio: 头部占全身的比例（取图像顶部这么多）

        Returns:
            头部区域图像
        """
        h, w = image.shape[:2]
        head_h = int(h * head_ratio)
        # 确保最小尺寸
        head_h = max(head_h, 50)
        return image[:head_h, :]

    def process_track(
        self,
        track_dir: Path,
        sample_rate: int = 1,
    ) -> Dict:
        """
        处理单个轨迹的所有裁剪图像

        Args:
            track_dir: 轨迹目录（包含裁剪图像）
            sample_rate: 采样率（每 N 帧取 1 帧）

        Returns:
            姿态数据字典
        """
        # 获取所有图像
        images = sorted(track_dir.glob("*.jpg"))

        if not images:
            return {"poses": [], "frames": []}

        poses = []
        frames = []
        valid_count = 0

        for i, img_path in enumerate(images):
            if i % sample_rate != 0:
                continue

            # 提取帧号
            frame_num = int(img_path.stem.split("_")[-1])

            # 读取图像
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # 截取头部区域再送入模型
            head_img = self._crop_head_region(img)

            # 估计姿态
            result = self.estimator.estimate(head_img)

            if result is not None:
                yaw, pitch, roll = result
                poses.append({
                    "frame": frame_num,
                    "yaw": round(yaw, 2),
                    "pitch": round(pitch, 2),
                    "roll": round(roll, 2),
                })
                frames.append(frame_num)
                valid_count += 1

        return {
            "total_images": len(images),
            "processed": len(images) // max(sample_rate, 1),
            "valid": valid_count,
            "poses": poses,
            "frames": frames,
        }

    def process_video_tracks(
        self,
        video_output_dir: Path,
        pose_output_dir: Path,
        sample_rate: int = 1,
    ) -> Dict:
        """处理一个视频的所有轨迹"""
        crops_dir = video_output_dir / "crops"
        if not crops_dir.exists():
            print(f"  未找到裁剪目录: {crops_dir}")
            return {}

        # 获取所有轨迹目录
        track_dirs = sorted([d for d in crops_dir.iterdir() if d.is_dir()])

        if not track_dirs:
            print(f"  未找到轨迹")
            return {}

        print(f"  找到 {len(track_dirs)} 个轨迹")

        all_poses = {}
        for track_dir in tqdm(track_dirs, desc="  处理轨迹"):
            track_id = track_dir.name  # e.g., "track_0001"
            pose_data = self.process_track(track_dir, sample_rate)
            all_poses[track_id] = pose_data

        # 保存结果
        pose_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = pose_output_dir / f"{video_output_dir.name}_poses.json"

        result = {
            "video_name": video_output_dir.name,
            "total_tracks": len(track_dirs),
            "sample_rate": sample_rate,
            "tracks": all_poses,
            "processed_at": datetime.now().isoformat(),
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result


def main():
    parser = argparse.ArgumentParser(description="Step 4: 头部姿态估计")
    parser.add_argument("--input", "-i", type=str, default=str(TRACKED_OUTPUT_DIR))
    parser.add_argument("--output", "-o", type=str, default=str(POSE_OUTPUT_DIR))
    parser.add_argument("--estimator", type=str, default="whenet",
                        choices=["whenet", "6drepnet"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sample-rate", type=int, default=3,
                        help="采样率，每 N 帧取 1 帧")
    parser.add_argument("--video", type=str, default=None,
                        help="处理单个视频")

    args = parser.parse_args()

    print("=" * 60)
    print("Step 4: 头部姿态估计")
    print("=" * 60)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = HeadPoseProcessor(args.estimator, args.device)

    if args.video:
        # 单个视频
        video_dir = input_dir / args.video
        if not video_dir.exists():
            print(f"目录不存在: {video_dir}")
            return

        print(f"\n处理: {args.video}")
        result = processor.process_video_tracks(
            video_dir, output_dir, args.sample_rate
        )
        print(f"完成! 处理了 {result.get('total_tracks', 0)} 个轨迹")
    else:
        # 所有视频
        video_dirs = [d for d in input_dir.iterdir() if d.is_dir()]

        if not video_dirs:
            print(f"未找到视频目录: {input_dir}")
            return

        print(f"\n找到 {len(video_dirs)} 个视频目录")

        all_results = []
        for video_dir in video_dirs:
            print(f"\n处理: {video_dir.name}")
            result = processor.process_video_tracks(
                video_dir, output_dir, args.sample_rate
            )
            all_results.append(result)

        # 汇总
        total_tracks = sum(r.get('total_tracks', 0) for r in all_results)
        print(f"\n{'='*60}")
        print(f"全部完成! 共处理 {total_tracks} 个轨迹")


if __name__ == "__main__":
    main()
