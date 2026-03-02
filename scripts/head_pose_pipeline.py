"""
头部姿态估计Pipeline
输入: MP4视频
输出: JSON/CSV (yaw, pitch, roll, confidence)
"""
import cv2
import json
import csv
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms


# ============== 人脸检测 (RetinaFace) ==============
class FaceDetector:
    """RetinaFace人脸检测器"""

    def __init__(self, conf_threshold=0.5, device='cuda'):
        self.conf_threshold = conf_threshold
        self.device = device if torch.cuda.is_available() else 'cpu'
        self._load_model()

    def _load_model(self):
        try:
            from retinaface import RetinaFace
            self.detector = RetinaFace
            self.backend = 'retinaface'
        except ImportError:
            # 回退到insightface
            try:
                from insightface.app import FaceAnalysis
                self.app = FaceAnalysis(allowed_modules=['detection'])
                self.app.prepare(ctx_id=0 if self.device == 'cuda' else -1, det_size=(640, 640))
                self.backend = 'insightface'
            except ImportError:
                raise ImportError("请安装 retinaface-pytorch 或 insightface")

    def detect(self, image):
        """
        检测人脸
        Args:
            image: BGR格式图像
        Returns:
            list of dict: [{bbox: [x1,y1,x2,y2], conf: float, landmarks: array}]
        """
        if self.backend == 'retinaface':
            return self._detect_retinaface(image)
        return self._detect_insightface(image)

    def _detect_retinaface(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(rgb)

        results = []
        for face_data in faces.values():
            if face_data['score'] < self.conf_threshold:
                continue
            bbox = face_data['facial_area']  # [x1, y1, x2, y2]
            landmarks = np.array([
                face_data['landmarks']['left_eye'],
                face_data['landmarks']['right_eye'],
                face_data['landmarks']['nose'],
                face_data['landmarks']['mouth_left'],
                face_data['landmarks']['mouth_right']
            ])
            results.append({
                'bbox': bbox,
                'conf': face_data['score'],
                'landmarks': landmarks
            })
        return results

    def _detect_insightface(self, image):
        faces = self.app.get(image)
        results = []
        for face in faces:
            if face.det_score < self.conf_threshold:
                continue
            results.append({
                'bbox': face.bbox.astype(int).tolist(),
                'conf': float(face.det_score),
                'landmarks': face.kps if hasattr(face, 'kps') else None
            })
        return results


# ============== 简单IOU跟踪器 ==============
class SimpleIOUTracker:
    """单目标IOU跟踪器"""

    def __init__(self, iou_threshold=0.3):
        self.iou_threshold = iou_threshold
        self.last_bbox = None
        self.track_id = 1

    def update(self, detections):
        """
        更新跟踪
        Args:
            detections: 检测结果列表
        Returns:
            最佳匹配的检测结果 (添加track_id)
        """
        if not detections:
            return None

        if self.last_bbox is None:
            # 第一帧，选择置信度最高的
            best = max(detections, key=lambda x: x['conf'])
            best['track_id'] = self.track_id
            self.last_bbox = best['bbox']
            return best

        # 计算IOU，选择最佳匹配
        best_iou = 0
        best_det = None
        for det in detections:
            iou = self._compute_iou(self.last_bbox, det['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_det = det

        if best_det and best_iou >= self.iou_threshold:
            best_det['track_id'] = self.track_id
            self.last_bbox = best_det['bbox']
            return best_det

        # IOU太低，可能是新目标或丢失
        if detections:
            best = max(detections, key=lambda x: x['conf'])
            best['track_id'] = self.track_id
            self.last_bbox = best['bbox']
            return best

        return None

    def _compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter / (area1 + area2 - inter + 1e-6)


# ============== 头部姿态估计 (WHENet) ==============
class WHENet(torch.nn.Module):
    """WHENet网络 - 支持全角度范围"""

    def __init__(self, bins_yaw=120, bins_pitch=66, bins_roll=66):
        super().__init__()
        self.bins_yaw = bins_yaw
        self.bins_pitch = bins_pitch
        self.bins_roll = bins_roll

        # Backbone: EfficientNet-B0
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.backbone = torch.nn.Sequential(*list(backbone.children())[:-1], torch.nn.Flatten())

        # 分类头
        self.fc_yaw = torch.nn.Linear(1280, bins_yaw)
        self.fc_pitch = torch.nn.Linear(1280, bins_pitch)
        self.fc_roll = torch.nn.Linear(1280, bins_roll)

        self.idx_yaw = torch.arange(bins_yaw).float()
        self.idx_pitch = torch.arange(bins_pitch).float()
        self.idx_roll = torch.arange(bins_roll).float()

    def forward(self, x):
        features = self.backbone(x)
        return self.fc_yaw(features), self.fc_pitch(features), self.fc_roll(features)

    def predict(self, x):
        yaw_l, pitch_l, roll_l = self.forward(x)
        device = x.device

        yaw_p = torch.softmax(yaw_l, dim=1)
        pitch_p = torch.softmax(pitch_l, dim=1)
        roll_p = torch.softmax(roll_l, dim=1)

        yaw = torch.sum(yaw_p * self.idx_yaw.to(device), dim=1) * 3 - 180
        pitch = torch.sum(pitch_p * self.idx_pitch.to(device), dim=1) * 3 - 99
        roll = torch.sum(roll_p * self.idx_roll.to(device), dim=1) * 3 - 99
        conf = torch.max(yaw_p, dim=1)[0]

        return yaw, pitch, roll, conf


class HeadPoseEstimator:
    """WHENet头部姿态估计器"""

    def __init__(self, model_path=None, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """加载WHENet模型"""
        self.model = WHENet()
        self.model.to(self.device)
        if self.model_path and Path(self.model_path).exists():
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

    def estimate(self, face_image, landmarks=None, image_size=None):
        """估计头部姿态"""
        if face_image.size == 0:
            return {'yaw': 0, 'pitch': 0, 'roll': 0, 'confidence': 0}

        # 预处理
        img = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
        img = img.to(self.device)

        with torch.no_grad():
            yaw, pitch, roll, conf = self.model.predict(img)

        return {
            'yaw': float(yaw.item()),
            'pitch': float(pitch.item()),
            'roll': float(roll.item()),
            'confidence': float(conf.item())
        }


# ============== 主Pipeline ==============
def process_video(video_path, output_path, conf_threshold=0.5):
    """
    处理视频并输出姿态数据

    Args:
        video_path: 输入视频路径
        output_path: 输出文件路径 (.json 或 .csv)
        conf_threshold: 人脸检测置信度阈值

    Returns:
        list: 每帧的姿态数据
    """
    # 初始化模块
    detector = FaceDetector(conf_threshold=conf_threshold)
    tracker = SimpleIOUTracker()
    pose_estimator = HeadPoseEstimator()

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"视频信息: {width}x{height}, {fps:.1f}fps, {total_frames}帧")

    results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps

        # 人脸检测
        detections = detector.detect(frame)

        # 跟踪
        tracked = tracker.update(detections)

        if tracked:
            # 裁剪人脸
            x1, y1, x2, y2 = tracked['bbox']
            # 扩展边界
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            expand = 1.2
            x1 = max(0, int(cx - w * expand / 2))
            y1 = max(0, int(cy - h * expand / 2))
            x2 = min(width, int(cx + w * expand / 2))
            y2 = min(height, int(cy + h * expand / 2))

            face_img = frame[y1:y2, x1:x2]

            if face_img.size > 0:
                # 姿态估计
                pose = pose_estimator.estimate(
                    face_img,
                    tracked.get('landmarks'),
                    (width, height)
                )

                results.append({
                    'frame': frame_idx,
                    'timestamp': round(timestamp, 3),
                    'track_id': tracked['track_id'],
                    'bbox': tracked['bbox'],
                    'face_conf': round(tracked['conf'], 3),
                    'yaw': round(pose['yaw'], 2),
                    'pitch': round(pose['pitch'], 2),
                    'roll': round(pose['roll'], 2),
                    'pose_conf': round(pose['confidence'], 3)
                })

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"处理进度: {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.1f}%)")

    cap.release()

    # 保存结果
    output_path = Path(output_path)
    if output_path.suffix == '.json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'video': str(video_path),
                'fps': fps,
                'total_frames': total_frames,
                'resolution': [width, height],
                'frames': results
            }, f, indent=2, ensure_ascii=False)
    else:  # CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)

    print(f"结果已保存到: {output_path}")
    print(f"共处理 {len(results)} 帧有效数据")

    return results


def main():
    parser = argparse.ArgumentParser(description='头部姿态估计Pipeline')
    parser.add_argument('--video', '-v', required=True, help='输入视频路径')
    parser.add_argument('--output', '-o', default='pose_results.json',
                        help='输出文件路径 (.json或.csv)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='人脸检测置信度阈值')
    args = parser.parse_args()

    process_video(args.video, args.output, args.conf)


if __name__ == '__main__':
    main()
