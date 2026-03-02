#!/usr/bin/env python3
"""
Step 5: 使用 6DRepNet 重新标注头部姿态

对应论文需求: 高精度头部姿态估计用于行为识别

输入:
    - dataset_root/frames/{video_id}/ 下的抽帧图像
    - dataset_root/annotations/detection/{video_id}/detections.json

输出:
    - dataset_root/features/pose/{video_id}/pose_6drepnet.json
    - 包含高精度 yaw/pitch/roll 角度

依赖:
    pip install torch torchvision
    pip install sixdrepnet  # 或从源码安装
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 配置
# ============================================================================

DATASET_ROOT = Path(__file__).parent.parent / "dataset_root"
FRAMES_DIR = DATASET_ROOT / "frames"
DETECTION_DIR = DATASET_ROOT / "annotations" / "detection"
POSE_DIR = DATASET_ROOT / "features" / "pose"

# 头部姿态估计参数
FACE_EXPAND_RATIO = 1.2  # 人脸框扩展比例
MIN_FACE_SIZE = 32       # 最小人脸尺寸
POSE_INPUT_SIZE = 224    # 模型输入尺寸

# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class HeadPose:
    """头部姿态数据"""
    yaw: float      # 偏航角 (左右转头) -90° ~ +90°
    pitch: float    # 俯仰角 (上下点头) -90° ~ +90°
    roll: float     # 翻滚角 (歪头) -90° ~ +90°
    confidence: float
    method: str = "6drepnet"

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# 6DRepNet 头部姿态估计器
# ============================================================================

class SixDRepNetEstimator:
    """6DRepNet 头部姿态估计器 (SOTA)"""

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if self._check_cuda() else "cpu")
        self.model = None
        self.backend = None
        self._load_model()

    def _check_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def _load_model(self):
        """加载模型 (支持多种后端)"""

        # 方案1: 尝试使用 sixdrepnet 包
        try:
            from sixdrepnet import SixDRepNet
            self.model = SixDRepNet()
            self.backend = "sixdrepnet"
            print(f"  [头部姿态] 6DRepNet (sixdrepnet包) on {self.device}")
            return
        except ImportError:
            pass

        # 方案2: 尝试使用 SixDRepNet_Pytorch
        try:
            from SixDRepNet import SixDRepNet
            import torch
            self.model = SixDRepNet(
                backbone_name='RepVGG-B1g2',
                backbone_file='',
                deploy=True,
                pretrained=False
            )
            # 尝试加载预训练权重
            weight_path = Path(__file__).parent / "weights" / "6DRepNet_300W_LP_AFLW2000.pth"
            if weight_path.exists():
                self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.backend = "sixdrepnet_pytorch"
            print(f"  [头部姿态] 6DRepNet (PyTorch) on {self.device}")
            return
        except ImportError:
            pass

        # 方案3: 使用 MediaPipe + solvePnP (备选)
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            self.backend = "mediapipe"
            print(f"  [头部姿态] MediaPipe FaceMesh + solvePnP")
            return
        except ImportError:
            pass

        # 方案4: 几何方法回退
        self.backend = "geometric"
        print(f"  [头部姿态] 几何方法 (回退)")

    def estimate(self, face_image: np.ndarray) -> HeadPose:
        """估计头部姿态"""
        if face_image is None or face_image.size == 0:
            return HeadPose(0, 0, 0, 0, self.backend)

        if face_image.shape[0] < MIN_FACE_SIZE or face_image.shape[1] < MIN_FACE_SIZE:
            return HeadPose(0, 0, 0, 0.1, self.backend)

        if self.backend == "sixdrepnet":
            return self._estimate_sixdrepnet(face_image)
        elif self.backend == "sixdrepnet_pytorch":
            return self._estimate_sixdrepnet_pytorch(face_image)
        elif self.backend == "mediapipe":
            return self._estimate_mediapipe(face_image)
        else:
            return self._estimate_geometric(face_image)

    def _estimate_sixdrepnet(self, face_image: np.ndarray) -> HeadPose:
        """使用 sixdrepnet 包估计"""
        try:
            pitch, yaw, roll = self.model.predict(face_image)
            return HeadPose(
                yaw=float(yaw),
                pitch=float(pitch),
                roll=float(roll),
                confidence=0.9,
                method="6drepnet"
            )
        except Exception as e:
            return HeadPose(0, 0, 0, 0, "6drepnet_error")

    def _estimate_sixdrepnet_pytorch(self, face_image: np.ndarray) -> HeadPose:
        """使用 PyTorch 版本估计"""
        import torch
        import torch.nn.functional as F
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        try:
            rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            img_tensor = transform(rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(img_tensor)
                # 6DRepNet 输出 6D 旋转表示, 需要转换为欧拉角
                euler = self._rotation_matrix_to_euler(output)

            return HeadPose(
                yaw=float(euler[0]),
                pitch=float(euler[1]),
                roll=float(euler[2]),
                confidence=0.85,
                method="6drepnet_pytorch"
            )
        except Exception as e:
            return HeadPose(0, 0, 0, 0, "6drepnet_pytorch_error")

    def _estimate_mediapipe(self, face_image: np.ndarray) -> HeadPose:
        """使用 MediaPipe + solvePnP 估计"""
        h, w = face_image.shape[:2]

        # 3D 人脸模型关键点
        face_3d_model = np.array([
            [0.0, 0.0, 0.0],            # 鼻尖
            [0.0, -330.0, -65.0],       # 下巴
            [-225.0, 170.0, -135.0],    # 左眼外角
            [225.0, 170.0, -135.0],     # 右眼外角
            [-150.0, -150.0, -125.0],   # 左嘴角
            [150.0, -150.0, -125.0]     # 右嘴角
        ], dtype=np.float64)

        # MediaPipe 关键点索引
        landmark_indices = [1, 152, 33, 263, 61, 291]

        # 相机内参
        focal_length = w
        camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        try:
            rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            results = self.mp_face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                return HeadPose(0, 0, 0, 0.1, "mediapipe_no_face")

            landmarks = results.multi_face_landmarks[0]
            face_2d = np.array([
                [landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h]
                for idx in landmark_indices
            ], dtype=np.float64)

            success, rotation_vec, translation_vec = cv2.solvePnP(
                face_3d_model, face_2d, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return HeadPose(0, 0, 0, 0.1, "mediapipe_pnp_failed")

            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            pose_mat = np.hstack([rotation_mat, translation_vec])
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

            pitch, yaw, roll = euler_angles.flatten()[:3]

            return HeadPose(
                yaw=float(yaw),
                pitch=float(pitch),
                roll=float(roll),
                confidence=0.8,
                method="mediapipe"
            )
        except Exception as e:
            return HeadPose(0, 0, 0, 0, f"mediapipe_error")

    def _estimate_geometric(self, face_image: np.ndarray) -> HeadPose:
        """几何方法估计 (简化回退)"""
        h, w = face_image.shape[:2]

        # 简单的基于人脸比例估计
        aspect_ratio = w / h if h > 0 else 1.0

        # 根据宽高比粗略估计 yaw
        yaw = (aspect_ratio - 1.0) * 30  # 正常人脸约为1:1
        yaw = np.clip(yaw, -30, 30)

        return HeadPose(
            yaw=float(yaw),
            pitch=0.0,
            roll=0.0,
            confidence=0.5,
            method="geometric"
        )

    def _rotation_matrix_to_euler(self, rotation_6d):
        """将 6D 旋转表示转换为欧拉角"""
        import torch

        # 6D -> 旋转矩阵
        a1, a2 = rotation_6d[:, :3], rotation_6d[:, 3:]
        b1 = F.normalize(a1, dim=1)
        b2 = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=1)
        b3 = torch.cross(b1, b2, dim=1)
        rot_mat = torch.stack([b1, b2, b3], dim=-1)

        # 旋转矩阵 -> 欧拉角
        sy = torch.sqrt(rot_mat[:, 0, 0] ** 2 + rot_mat[:, 1, 0] ** 2)
        singular = sy < 1e-6

        x = torch.atan2(rot_mat[:, 2, 1], rot_mat[:, 2, 2])
        y = torch.atan2(-rot_mat[:, 2, 0], sy)
        z = torch.atan2(rot_mat[:, 1, 0], rot_mat[:, 0, 0])

        # 转换为角度
        euler = torch.stack([y, x, z], dim=1) * 180 / np.pi
        return euler[0].cpu().numpy()


# ============================================================================
# 人脸检测器 (用于获取人脸区域)
# ============================================================================

class FaceDetectorForPose:
    """人脸检测器 (优先使用高精度检测器)"""

    def __init__(self, device: str = None):
        self.device = device
        self.backend = None
        self._load_detector()

    def _load_detector(self):
        """加载检测器"""
        # 方案1: RetinaFace
        try:
            from retinaface import RetinaFace
            self.detector = RetinaFace
            self.backend = "retinaface"
            print(f"  [人脸检测] RetinaFace")
            return
        except ImportError:
            pass

        # 方案2: InsightFace
        try:
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(allowed_modules=['detection'])
            ctx_id = 0 if self.device == "cuda" else -1
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            self.backend = "insightface"
            print(f"  [人脸检测] InsightFace")
            return
        except ImportError:
            pass

        # 方案3: MediaPipe
        try:
            import mediapipe as mp
            self.face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
            self.backend = "mediapipe"
            print(f"  [人脸检测] MediaPipe")
            return
        except ImportError:
            pass

        # 方案4: OpenCV Haar
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.backend = "opencv"
        print(f"  [人脸检测] OpenCV Haar Cascade")

    def detect(self, image: np.ndarray) -> List[Dict]:
        """检测人脸, 返回 bbox 列表"""
        if self.backend == "retinaface":
            return self._detect_retinaface(image)
        elif self.backend == "insightface":
            return self._detect_insightface(image)
        elif self.backend == "mediapipe":
            return self._detect_mediapipe(image)
        else:
            return self._detect_opencv(image)

    def _detect_retinaface(self, image: np.ndarray) -> List[Dict]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(rgb)
        results = []
        if isinstance(faces, dict):
            for face_data in faces.values():
                bbox = face_data['facial_area']
                results.append({
                    'bbox': [bbox[0], bbox[1], bbox[2], bbox[3]],
                    'confidence': face_data['score'],
                    'landmarks': face_data.get('landmarks')
                })
        return results

    def _detect_insightface(self, image: np.ndarray) -> List[Dict]:
        faces = self.app.get(image)
        return [{
            'bbox': face.bbox.astype(int).tolist(),
            'confidence': float(face.det_score),
            'landmarks': face.kps.tolist() if hasattr(face, 'kps') else None
        } for face in faces]

    def _detect_mediapipe(self, image: np.ndarray) -> List[Dict]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)
        faces = []
        if results.detections:
            h, w = image.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                faces.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': detection.score[0],
                    'landmarks': None
                })
        return faces

    def _detect_opencv(self, image: np.ndarray) -> List[Dict]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        return [{
            'bbox': [int(x), int(y), int(x+w), int(y+h)],
            'confidence': 0.9,
            'landmarks': None
        } for (x, y, w, h) in faces]


# ============================================================================
# 主处理流程
# ============================================================================

def crop_face(image: np.ndarray, bbox: List[int], expand_ratio: float = 1.2) -> np.ndarray:
    """裁剪人脸区域 (带扩展)"""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    new_w = w * expand_ratio
    new_h = h * expand_ratio

    x1 = max(0, int(cx - new_w / 2))
    y1 = max(0, int(cy - new_h / 2))
    x2 = min(image.shape[1], int(cx + new_w / 2))
    y2 = min(image.shape[0], int(cy + new_h / 2))

    return image[y1:y2, x1:x2]


def process_video(video_id: str, pose_estimator: SixDRepNetEstimator,
                  face_detector: FaceDetectorForPose, use_existing_detections: bool = True) -> Dict:
    """处理单个视频的头部姿态标注"""

    frames_path = FRAMES_DIR / video_id
    detection_path = DETECTION_DIR / video_id / "detections.json"

    if not frames_path.exists():
        print(f"  [跳过] {video_id}: 帧目录不存在")
        return None

    # 获取帧文件列表
    frame_files = sorted(frames_path.glob("*.jpg"))
    if not frame_files:
        frame_files = sorted(frames_path.glob("*.png"))

    if not frame_files:
        print(f"  [跳过] {video_id}: 无帧文件")
        return None

    # 加载已有检测结果
    existing_detections = {}
    if use_existing_detections and detection_path.exists():
        with open(detection_path, 'r') as f:
            det_data = json.load(f)
            for frame_info in det_data.get('frames', []):
                existing_detections[frame_info['frame_idx']] = frame_info.get('detections', [])

    # 处理每一帧
    results = {
        'video_id': video_id,
        'processed_at': datetime.now().isoformat(),
        'method': pose_estimator.backend,
        'frames': []
    }

    for frame_file in tqdm(frame_files, desc=f"  {video_id}", leave=False):
        # 解析帧索引
        frame_name = frame_file.stem
        try:
            frame_idx = int(frame_name.split('_')[-1])
        except:
            frame_idx = frame_files.index(frame_file)

        # 读取图像
        image = cv2.imread(str(frame_file))
        if image is None:
            continue

        # 获取人脸检测结果
        if frame_idx in existing_detections and existing_detections[frame_idx]:
            detections = existing_detections[frame_idx]
        else:
            detections = face_detector.detect(image)

        # 对每个检测到的人脸估计头部姿态
        frame_poses = []
        for det_idx, det in enumerate(detections):
            bbox = det['bbox'] if isinstance(det, dict) else det.get('bbox', [0,0,0,0])

            # 裁剪人脸
            face_img = crop_face(image, bbox, FACE_EXPAND_RATIO)

            # 估计头部姿态
            pose = pose_estimator.estimate(face_img)

            frame_poses.append({
                'detection_idx': det_idx,
                'bbox': bbox,
                'yaw': round(pose.yaw, 2),
                'pitch': round(pose.pitch, 2),
                'roll': round(pose.roll, 2),
                'confidence': round(pose.confidence, 2),
                'method': pose.method
            })

        results['frames'].append({
            'frame_idx': frame_idx,
            'frame_file': frame_file.name,
            'poses': frame_poses
        })

    return results


def main():
    """主函数"""
    print("=" * 60)
    print("Step 5: 使用 6DRepNet 重新标注头部姿态")
    print("=" * 60)

    # 初始化模型
    print("\n[1] 初始化模型...")
    pose_estimator = SixDRepNetEstimator()
    face_detector = FaceDetectorForPose()

    # 获取所有视频ID
    video_ids = sorted([d.name for d in FRAMES_DIR.iterdir() if d.is_dir()])
    print(f"\n[2] 发现 {len(video_ids)} 个视频待处理")

    # 处理每个视频
    print("\n[3] 开始处理...")
    all_results = []

    for video_id in video_ids:
        print(f"\n处理: {video_id}")
        result = process_video(video_id, pose_estimator, face_detector)

        if result:
            # 保存结果
            output_dir = POSE_DIR / video_id
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "pose_6drepnet.json"

            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)

            all_results.append({
                'video_id': video_id,
                'total_frames': len(result['frames']),
                'total_poses': sum(len(f['poses']) for f in result['frames']),
                'method': result['method']
            })

            print(f"  -> 保存: {output_file}")
            print(f"     帧数: {len(result['frames'])}, 姿态数: {sum(len(f['poses']) for f in result['frames'])}")

    # 生成汇总报告
    report = {
        'processed_at': datetime.now().isoformat(),
        'total_videos': len(all_results),
        'method': pose_estimator.backend,
        'videos': all_results
    }

    report_file = DATASET_ROOT / "step5_head_pose_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print(f"处理完成!")
    print(f"  - 视频数: {len(all_results)}")
    print(f"  - 方法: {pose_estimator.backend}")
    print(f"  - 报告: {report_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
