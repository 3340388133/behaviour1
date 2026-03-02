"""
头部姿态估计方法性能对比评估
支持方法: WHENet, FSA-Net, 6DRepNet, SynergyNet, HopeNet
评估指标: Yaw/Pitch/Roll MAE, 5°内准确率
"""
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import time


@dataclass
class PoseEstimate:
    """姿态估计结果"""
    yaw: float
    pitch: float
    roll: float
    inference_time: float = 0.0


@dataclass
class BenchmarkResult:
    """评估结果"""
    method: str
    yaw_mae: float
    pitch_mae: float
    roll_mae: float
    yaw_acc_5: float  # 5°内准确率
    pitch_acc_5: float
    roll_acc_5: float
    avg_mae: float
    avg_acc_5: float
    avg_inference_time: float
    num_samples: int


class BasePoseEstimator(ABC):
    """姿态估计器基类"""

    @abstractmethod
    def estimate(self, image: np.ndarray) -> PoseEstimate:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class WHENetEstimator(BasePoseEstimator):
    """WHENet 姿态估计器"""

    def __init__(self, model_path: str = None):
        import onnxruntime as ort
        if model_path is None:
            model_path = str(Path(__file__).parent.parent.parent / "models" / "whenet_1x3x224x224_prepost.onnx")

        providers = ['CPUExecutionProvider']
        if ort.get_device() == 'GPU':
            providers.insert(0, 'CUDAExecutionProvider')

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def estimate(self, image: np.ndarray) -> PoseEstimate:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]

        start = time.time()
        outputs = self.session.run(None, {self.input_name: img})
        inference_time = time.time() - start

        yaw, roll, pitch = outputs[0][0]
        return PoseEstimate(float(yaw), float(pitch), float(roll), inference_time)

    def get_name(self) -> str:
        return "WHENet"


class SixDRepNetEstimator(BasePoseEstimator):
    """6DRepNet 姿态估计器"""

    def __init__(self, model_path: str = None):
        import torch
        from torchvision import transforms

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 尝试加载模型
        self.model = None
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
        else:
            try:
                from sixdrepnet import SixDRepNet
                self.model = SixDRepNet()
                self.model.to(self.device)
                self.model.eval()
            except ImportError:
                print("Warning: sixdrepnet not installed. Run: pip install sixdrepnet")

    def _load_model(self, model_path: str):
        import torch
        try:
            from sixdrepnet.model import SixDRepNet
            self.model = SixDRepNet(backbone_name='RepVGG-B1g2', backbone_file='', deploy=True, pretrained=False)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Failed to load 6DRepNet model: {e}")

    def estimate(self, image: np.ndarray) -> PoseEstimate:
        if self.model is None:
            return PoseEstimate(0, 0, 0, 0)

        import torch
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        start = time.time()
        with torch.no_grad():
            yaw, pitch, roll = self.model(img_tensor)
        inference_time = time.time() - start

        return PoseEstimate(
            float(yaw.cpu().numpy()[0]),
            float(pitch.cpu().numpy()[0]),
            float(roll.cpu().numpy()[0]),
            inference_time
        )

    def get_name(self) -> str:
        return "6DRepNet"


class FSANetEstimator(BasePoseEstimator):
    """FSA-Net 姿态估计器"""

    def __init__(self, model_path: str = None):
        self.model = None
        self.input_size = (64, 64)

        if model_path and Path(model_path).exists():
            self._load_model(model_path)
        else:
            try:
                # 尝试使用 ONNX 版本
                import onnxruntime as ort
                default_path = Path(__file__).parent.parent.parent / "models" / "fsanet.onnx"
                if default_path.exists():
                    self.session = ort.InferenceSession(str(default_path))
                    self.input_name = self.session.get_inputs()[0].name
                    self.use_onnx = True
                else:
                    print("Warning: FSA-Net model not found")
                    self.use_onnx = False
            except Exception as e:
                print(f"FSA-Net init error: {e}")
                self.use_onnx = False

    def _load_model(self, model_path: str):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.use_onnx = True

    def estimate(self, image: np.ndarray) -> PoseEstimate:
        if not hasattr(self, 'use_onnx') or not self.use_onnx:
            return PoseEstimate(0, 0, 0, 0)

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis, ...]

        start = time.time()
        outputs = self.session.run(None, {self.input_name: img})
        inference_time = time.time() - start

        yaw, pitch, roll = outputs[0][0]
        return PoseEstimate(float(yaw), float(pitch), float(roll), inference_time)

    def get_name(self) -> str:
        return "FSA-Net"


class HopeNetEstimator(BasePoseEstimator):
    """HopeNet 姿态估计器"""

    def __init__(self, model_path: str = None):
        import torch
        from torchvision import transforms

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.model = None
        self.idx_tensor = torch.arange(66, dtype=torch.float32).to(self.device)

        if model_path and Path(model_path).exists():
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        import torch
        import torchvision.models as models

        # HopeNet 使用 ResNet50 backbone
        self.model = models.resnet50(pretrained=False)
        self.model.fc = torch.nn.Linear(2048, 66 * 3)  # yaw, pitch, roll 各66个bin
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def estimate(self, image: np.ndarray) -> PoseEstimate:
        if self.model is None:
            return PoseEstimate(0, 0, 0, 0)

        import torch
        import torch.nn.functional as F

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        start = time.time()
        with torch.no_grad():
            output = self.model(img_tensor)
            output = output.view(-1, 3, 66)

            yaw = F.softmax(output[:, 0, :], dim=1)
            pitch = F.softmax(output[:, 1, :], dim=1)
            roll = F.softmax(output[:, 2, :], dim=1)

            yaw = (yaw * self.idx_tensor).sum(dim=1) * 3 - 99
            pitch = (pitch * self.idx_tensor).sum(dim=1) * 3 - 99
            roll = (roll * self.idx_tensor).sum(dim=1) * 3 - 99

        inference_time = time.time() - start

        return PoseEstimate(
            float(yaw.cpu().numpy()[0]),
            float(pitch.cpu().numpy()[0]),
            float(roll.cpu().numpy()[0]),
            inference_time
        )

    def get_name(self) -> str:
        return "HopeNet"


class PoseBenchmark:
    """姿态估计方法性能对比评估"""

    def __init__(self, data_dir: str, gt_file: str = None):
        """
        Args:
            data_dir: 数据目录
            gt_file: Ground truth CSV 文件路径
                     格式: face_path,yaw,pitch,roll
        """
        self.data_dir = Path(data_dir)
        self.gt_file = gt_file
        self.estimators: Dict[str, BasePoseEstimator] = {}
        self.ground_truth: Optional[pd.DataFrame] = None

        if gt_file and Path(gt_file).exists():
            self.ground_truth = pd.read_csv(gt_file)

    def add_estimator(self, estimator: BasePoseEstimator):
        """添加姿态估计器"""
        self.estimators[estimator.get_name()] = estimator

    def load_ground_truth(self, gt_file: str):
        """加载 ground truth 标注"""
        self.ground_truth = pd.read_csv(gt_file)

    def _compute_metrics(self, predictions: List[PoseEstimate],
                         gt_yaw: np.ndarray, gt_pitch: np.ndarray,
                         gt_roll: np.ndarray) -> Dict:
        """计算评估指标"""
        pred_yaw = np.array([p.yaw for p in predictions])
        pred_pitch = np.array([p.pitch for p in predictions])
        pred_roll = np.array([p.roll for p in predictions])
        inference_times = np.array([p.inference_time for p in predictions])

        # MAE
        yaw_mae = np.mean(np.abs(pred_yaw - gt_yaw))
        pitch_mae = np.mean(np.abs(pred_pitch - gt_pitch))
        roll_mae = np.mean(np.abs(pred_roll - gt_roll))

        # 5°内准确率
        yaw_acc_5 = np.mean(np.abs(pred_yaw - gt_yaw) <= 5) * 100
        pitch_acc_5 = np.mean(np.abs(pred_pitch - gt_pitch) <= 5) * 100
        roll_acc_5 = np.mean(np.abs(pred_roll - gt_roll) <= 5) * 100

        return {
            'yaw_mae': yaw_mae,
            'pitch_mae': pitch_mae,
            'roll_mae': roll_mae,
            'yaw_acc_5': yaw_acc_5,
            'pitch_acc_5': pitch_acc_5,
            'roll_acc_5': roll_acc_5,
            'avg_mae': (yaw_mae + pitch_mae + roll_mae) / 3,
            'avg_acc_5': (yaw_acc_5 + pitch_acc_5 + roll_acc_5) / 3,
            'avg_inference_time': np.mean(inference_times) * 1000  # ms
        }

    def evaluate(self, face_images: List[str] = None) -> List[BenchmarkResult]:
        """运行评估"""
        if self.ground_truth is None:
            raise ValueError("Ground truth not loaded. Use load_ground_truth() first.")

        if face_images is None:
            face_images = self.ground_truth['face_path'].tolist()

        gt_yaw = self.ground_truth['yaw'].values
        gt_pitch = self.ground_truth['pitch'].values
        gt_roll = self.ground_truth['roll'].values

        results = []

        for name, estimator in self.estimators.items():
            print(f"Evaluating {name}...")
            predictions = []

            for face_path in face_images:
                full_path = self.data_dir / face_path
                if not full_path.exists():
                    predictions.append(PoseEstimate(0, 0, 0, 0))
                    continue

                image = cv2.imread(str(full_path))
                if image is None:
                    predictions.append(PoseEstimate(0, 0, 0, 0))
                    continue

                pred = estimator.estimate(image)
                predictions.append(pred)

            metrics = self._compute_metrics(predictions, gt_yaw, gt_pitch, gt_roll)

            result = BenchmarkResult(
                method=name,
                num_samples=len(predictions),
                **metrics
            )
            results.append(result)

        return results

    def print_results(self, results: List[BenchmarkResult]):
        """打印评估结果表格"""
        print("\n" + "=" * 100)
        print("头部姿态估计方法性能对比")
        print("=" * 100)

        headers = ["Method", "Yaw MAE", "Pitch MAE", "Roll MAE",
                   "Yaw Acc@5°", "Pitch Acc@5°", "Roll Acc@5°",
                   "Avg MAE", "Avg Acc@5°", "Time(ms)"]

        print(f"{'Method':<12} {'Yaw MAE':>10} {'Pitch MAE':>10} {'Roll MAE':>10} "
              f"{'Yaw@5°':>10} {'Pitch@5°':>10} {'Roll@5°':>10} "
              f"{'Avg MAE':>10} {'Avg@5°':>10} {'Time(ms)':>10}")
        print("-" * 100)

        for r in results:
            print(f"{r.method:<12} {r.yaw_mae:>10.2f} {r.pitch_mae:>10.2f} {r.roll_mae:>10.2f} "
                  f"{r.yaw_acc_5:>9.1f}% {r.pitch_acc_5:>9.1f}% {r.roll_acc_5:>9.1f}% "
                  f"{r.avg_mae:>10.2f} {r.avg_acc_5:>9.1f}% {r.avg_inference_time:>10.2f}")

        print("=" * 100)

    def save_results(self, results: List[BenchmarkResult], output_path: str):
        """保存评估结果到 CSV"""
        data = []
        for r in results:
            data.append({
                'Method': r.method,
                'Yaw MAE': r.yaw_mae,
                'Pitch MAE': r.pitch_mae,
                'Roll MAE': r.roll_mae,
                'Yaw Acc@5°': r.yaw_acc_5,
                'Pitch Acc@5°': r.pitch_acc_5,
                'Roll Acc@5°': r.roll_acc_5,
                'Avg MAE': r.avg_mae,
                'Avg Acc@5°': r.avg_acc_5,
                'Inference Time (ms)': r.avg_inference_time,
                'Num Samples': r.num_samples
            })

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")


def create_gt_template(data_dir: str, output_file: str):
    """创建 ground truth 标注模板"""
    data_dir = Path(data_dir)
    face_files = sorted(data_dir.glob("**/*.jpg"))

    data = []
    for f in face_files:
        rel_path = f.relative_to(data_dir)
        data.append({
            'face_path': str(rel_path),
            'yaw': 0.0,
            'pitch': 0.0,
            'roll': 0.0
        })

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Ground truth template saved to {output_file}")
    print(f"Total {len(data)} images. Please fill in the yaw, pitch, roll values.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Head Pose Estimation Benchmark")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--gt-file", type=str, help="Ground truth CSV file")
    parser.add_argument("--output", type=str, default="benchmark_results.csv", help="Output file")
    parser.add_argument("--create-template", action="store_true", help="Create GT template")
    parser.add_argument("--methods", nargs="+", default=["whenet"],
                        choices=["whenet", "fsanet", "6drepnet", "hopenet"],
                        help="Methods to evaluate")

    args = parser.parse_args()

    if args.create_template:
        create_gt_template(args.data_dir, args.gt_file or "gt_template.csv")
    else:
        benchmark = PoseBenchmark(args.data_dir, args.gt_file)

        # 添加估计器
        if "whenet" in args.methods:
            benchmark.add_estimator(WHENetEstimator())
        if "fsanet" in args.methods:
            benchmark.add_estimator(FSANetEstimator())
        if "6drepnet" in args.methods:
            benchmark.add_estimator(SixDRepNetEstimator())
        if "hopenet" in args.methods:
            benchmark.add_estimator(HopeNetEstimator())

        results = benchmark.evaluate()
        benchmark.print_results(results)
        benchmark.save_results(results, args.output)
