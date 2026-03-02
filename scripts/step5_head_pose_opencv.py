#!/usr/bin/env python3
"""
Step 5: 使用 OpenCV + dlib 进行头部姿态估计

简化版本，使用 OpenCV 内置功能进行头部姿态估计
支持: OpenCV Face Detection + PnP 求解

输入:
    - dataset_root/frames/{video_id}/ 下的抽帧图像
    - dataset_root/annotations/detection/{video_id}/detections.json

输出:
    - dataset_root/features/pose/{video_id}/pose_opencv.json
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# ============================================================================
# 配置
# ============================================================================

DATASET_ROOT = Path(__file__).parent.parent / "dataset_root"
FRAMES_DIR = DATASET_ROOT / "frames"
DETECTION_DIR = DATASET_ROOT / "annotations" / "detection"
POSE_DIR = DATASET_ROOT / "features" / "pose"

FACE_EXPAND_RATIO = 1.2
MIN_FACE_SIZE = 40


# ============================================================================
# 头部姿态估计器 (基于 OpenCV)
# ============================================================================

class OpenCVHeadPoseEstimator:
    """
    基于 OpenCV 的头部姿态估计器
    使用 Haar Cascade 或 DNN 人脸检测 + 人脸关键点 + solvePnP
    """

    def __init__(self):
        # 加载人脸检测器
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # 尝试加载 DNN 人脸检测器 (更准确)
        self.use_dnn = False
        try:
            model_path = cv2.data.haarcascades.replace(
                'haarcascades', 'dnn'
            ) + 'opencv_face_detector_uint8.pb'
            config_path = cv2.data.haarcascades.replace(
                'haarcascades', 'dnn'
            ) + 'opencv_face_detector.pbtxt'

            if os.path.exists(model_path) and os.path.exists(config_path):
                self.face_net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
                self.use_dnn = True
        except:
            pass

        # 尝试加载人脸关键点检测器 (LBF)
        self.facemark = None
        try:
            self.facemark = cv2.face.createFacemarkLBF()
            lbf_model = cv2.data.haarcascades.replace(
                'haarcascades', ''
            ) + 'lbfmodel.yaml'
            if os.path.exists(lbf_model):
                self.facemark.loadModel(lbf_model)
        except:
            pass

        # 3D 人脸模型 (用于 solvePnP)
        # 基于平均人脸的 3D 坐标
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # 鼻尖
            (0.0, -330.0, -65.0),        # 下巴
            (-225.0, 170.0, -135.0),     # 左眼外角
            (225.0, 170.0, -135.0),      # 右眼外角
            (-150.0, -150.0, -125.0),    # 左嘴角
            (150.0, -150.0, -125.0)      # 右嘴角
        ], dtype=np.float64)

        print(f"  [头部姿态] OpenCV 几何方法")
        print(f"    - 人脸检测: {'DNN' if self.use_dnn else 'Haar Cascade'}")
        print(f"    - 关键点检测: {'LBF' if self.facemark else '几何估计'}")

    def estimate_from_bbox(self, image: np.ndarray, bbox: List[int]) -> Dict:
        """
        基于人脸边界框估计头部姿态

        Args:
            image: 原始图像
            bbox: [x1, y1, x2, y2] 人脸边界框

        Returns:
            头部姿态字典 {yaw, pitch, roll, confidence, method}
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        face_width = x2 - x1
        face_height = y2 - y1

        if face_width < MIN_FACE_SIZE or face_height < MIN_FACE_SIZE:
            return self._default_pose()

        # 扩展人脸区域
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        half_w = int(face_width * FACE_EXPAND_RATIO / 2)
        half_h = int(face_height * FACE_EXPAND_RATIO / 2)

        x1 = max(0, cx - half_w)
        y1 = max(0, cy - half_h)
        x2 = min(image.shape[1], cx + half_w)
        y2 = min(image.shape[0], cy + half_h)

        face_img = image[y1:y2, x1:x2]

        if face_img.size == 0:
            return self._default_pose()

        # 尝试使用关键点方法
        if self.facemark is not None:
            pose = self._estimate_with_landmarks(face_img, (x1, y1))
            if pose['confidence'] > 0.5:
                return pose

        # 回退到几何估计方法
        return self._estimate_geometric(face_img, bbox)

    def _estimate_with_landmarks(self, face_img: np.ndarray, offset: Tuple[int, int]) -> Dict:
        """使用人脸关键点估计姿态"""
        try:
            h, w = face_img.shape[:2]
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

            # 检测人脸
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

            if len(faces) == 0:
                return self._default_pose()

            # 检测关键点
            ok, landmarks = self.facemark.fit(gray, faces)

            if not ok or len(landmarks) == 0:
                return self._default_pose()

            # 获取 68 点关键点
            pts = landmarks[0][0]

            # 提取 6 个用于 PnP 的关键点
            # 鼻尖(30), 下巴(8), 左眼外角(36), 右眼外角(45), 左嘴角(48), 右嘴角(54)
            image_points = np.array([
                pts[30],  # 鼻尖
                pts[8],   # 下巴
                pts[36],  # 左眼外角
                pts[45],  # 右眼外角
                pts[48],  # 左嘴角
                pts[54]   # 右嘴角
            ], dtype=np.float64)

            # 相机内参
            focal_length = w
            camera_matrix = np.array([
                [focal_length, 0, w / 2],
                [0, focal_length, h / 2],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))

            # 使用 solvePnP 求解姿态
            success, rotation_vec, translation_vec = cv2.solvePnP(
                self.model_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return self._default_pose()

            # 转换为欧拉角
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            pose_mat = np.hstack([rotation_mat, translation_vec])
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

            yaw, pitch, roll = euler_angles.flatten()[:3]

            return {
                'yaw': float(np.clip(yaw, -90, 90)),
                'pitch': float(np.clip(pitch, -90, 90)),
                'roll': float(np.clip(roll, -90, 90)),
                'confidence': 0.85,
                'method': 'opencv_landmarks'
            }

        except Exception as e:
            return self._default_pose()

    def _estimate_geometric(self, face_img: np.ndarray, bbox: List[int]) -> Dict:
        """
        几何方法估计头部姿态
        基于人脸边界框的宽高比和位置推断
        """
        x1, y1, x2, y2 = bbox
        face_width = x2 - x1
        face_height = y2 - y1

        if face_height == 0:
            return self._default_pose()

        # 宽高比分析
        aspect_ratio = face_width / face_height

        # 正常正面人脸宽高比约为 0.7-0.9
        # 侧脸时宽度变小，宽高比降低

        # Yaw 估计 (基于宽高比)
        # 宽高比越小，说明越侧脸
        normal_ratio = 0.8
        if aspect_ratio < normal_ratio:
            # 可能是侧脸，但无法确定方向
            yaw_magnitude = (normal_ratio - aspect_ratio) / normal_ratio * 45
        else:
            yaw_magnitude = 0

        # 使用图像分析确定侧脸方向
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 计算左右半边的平均亮度差
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]

        left_mean = np.mean(left_half)
        right_mean = np.mean(right_half)

        # 亮度差可以暗示脸的朝向
        brightness_diff = (right_mean - left_mean) / max(left_mean, right_mean, 1)

        # 综合估计 yaw
        yaw = brightness_diff * 20 + np.sign(brightness_diff) * yaw_magnitude

        # Pitch 估计 (基于人脸在图像中的垂直位置)
        # 这里简化处理，设为 0
        pitch = 0.0

        # Roll 估计 (基于人脸倾斜)
        # 尝试检测眼睛位置来估计
        roll = self._estimate_roll(gray)

        return {
            'yaw': float(np.clip(yaw, -45, 45)),
            'pitch': float(pitch),
            'roll': float(roll),
            'confidence': 0.6,
            'method': 'opencv_geometric'
        }

    def _estimate_roll(self, gray_face: np.ndarray) -> float:
        """估计 roll 角度 (头部倾斜)"""
        try:
            # 使用边缘检测找到主要方向
            edges = cv2.Canny(gray_face, 50, 150)

            # 霍夫线检测
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)

            if lines is None:
                return 0.0

            # 计算主要线条的角度
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 != 0:
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    # 只考虑接近水平的线 (可能是眼睛或嘴巴)
                    if abs(angle) < 30:
                        angles.append(angle)

            if angles:
                # 返回中位数角度
                return float(np.median(angles))

        except:
            pass

        return 0.0

    def _default_pose(self) -> Dict:
        """返回默认姿态"""
        return {
            'yaw': 0.0,
            'pitch': 0.0,
            'roll': 0.0,
            'confidence': 0.3,
            'method': 'default'
        }


# ============================================================================
# 主处理流程
# ============================================================================

def process_video(video_id: str, estimator: OpenCVHeadPoseEstimator) -> Dict:
    """处理单个视频"""

    frames_path = FRAMES_DIR / video_id
    detection_path = DETECTION_DIR / video_id / "detections.json"

    if not frames_path.exists():
        print(f"  [跳过] {video_id}: 帧目录不存在")
        return None

    # 读取检测结果
    detections_by_frame = {}
    if detection_path.exists():
        with open(detection_path, 'r') as f:
            det_data = json.load(f)
            for frame_info in det_data.get('frames', []):
                detections_by_frame[frame_info['frame_idx']] = frame_info.get('detections', [])

    # 获取帧文件
    frame_files = sorted(frames_path.glob("*.jpg"))
    if not frame_files:
        frame_files = sorted(frames_path.glob("*.png"))

    if not frame_files:
        print(f"  [跳过] {video_id}: 无帧文件")
        return None

    # 处理每一帧
    results = {
        'video_id': video_id,
        'processed_at': datetime.now().isoformat(),
        'method': 'opencv',
        'total_frames': len(frame_files),
        'frames': []
    }

    total_poses = 0

    for frame_file in tqdm(frame_files, desc=f"  {video_id}", leave=False):
        # 解析帧索引
        try:
            frame_idx = int(frame_file.stem.split('_')[-1])
        except:
            continue

        # 读取图像
        image = cv2.imread(str(frame_file))
        if image is None:
            continue

        # 获取该帧的检测结果
        frame_detections = detections_by_frame.get(frame_idx, [])

        # 对每个检测估计头部姿态
        poses = []
        for det_idx, det in enumerate(frame_detections):
            bbox = det.get('bbox', [0, 0, 0, 0])
            pose = estimator.estimate_from_bbox(image, bbox)

            poses.append({
                'detection_idx': det_idx,
                'bbox': bbox,
                **pose
            })

        if poses:
            results['frames'].append({
                'frame_idx': frame_idx,
                'frame_file': frame_file.name,
                'num_poses': len(poses),
                'poses': poses
            })
            total_poses += len(poses)

    results['total_poses'] = total_poses
    return results


def main():
    """主函数"""
    print("=" * 60)
    print("Step 5: 使用 OpenCV 进行头部姿态估计")
    print("=" * 60)

    # 初始化估计器
    print("\n[1] 初始化头部姿态估计器...")
    estimator = OpenCVHeadPoseEstimator()

    # 获取视频列表
    video_ids = sorted([d.name for d in FRAMES_DIR.iterdir() if d.is_dir()])
    print(f"\n[2] 发现 {len(video_ids)} 个视频")

    # 处理每个视频
    print("\n[3] 开始处理...")
    all_results = []

    for video_id in video_ids:
        print(f"\n处理: {video_id}")

        result = process_video(video_id, estimator)

        if result:
            # 保存结果
            output_dir = POSE_DIR / video_id
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "pose_opencv.json"

            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)

            all_results.append({
                'video_id': video_id,
                'total_frames': result['total_frames'],
                'frames_with_poses': len(result['frames']),
                'total_poses': result['total_poses']
            })

            print(f"  -> 帧数: {result['total_frames']}, 姿态数: {result['total_poses']}")

    # 生成汇总报告
    report = {
        'processed_at': datetime.now().isoformat(),
        'method': 'opencv',
        'total_videos': len(all_results),
        'total_frames': sum(r['total_frames'] for r in all_results),
        'total_poses': sum(r['total_poses'] for r in all_results),
        'videos': all_results
    }

    report_file = DATASET_ROOT / "step5_head_pose_opencv_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("处理完成!")
    print(f"  - 视频数: {len(all_results)}")
    print(f"  - 总帧数: {report['total_frames']}")
    print(f"  - 总姿态数: {report['total_poses']}")
    print(f"  - 报告: {report_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
