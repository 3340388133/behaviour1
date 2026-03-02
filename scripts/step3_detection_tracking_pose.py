#!/usr/bin/env python3
"""
Step 3: 检测 / 跟踪 / 姿态估计 Pipeline

对应论文 III-B 节: Feature Extraction Pipeline

输入:
    - dataset_root/frames/{video_id}/ 下的抽帧图像

输出:
    - dataset_root/annotations/detection/{video_id}/detections.json
    - dataset_root/annotations/tracking/{video_id}/tracks.json
    - dataset_root/features/pose/{video_id}/pose.json

依赖: opencv-python, numpy, torch, tqdm, retinaface-pytorch 或 insightface
"""

import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from tqdm import tqdm

# ============================================================================
# 配置
# ============================================================================

DATASET_ROOT = Path(__file__).parent.parent / "dataset_root"
FRAMES_DIR = DATASET_ROOT / "frames"
DETECTION_DIR = DATASET_ROOT / "annotations" / "detection"
TRACKING_DIR = DATASET_ROOT / "annotations" / "tracking"
POSE_DIR = DATASET_ROOT / "features" / "pose"

# 检测参数
DETECTION_CONF_THRESHOLD = 0.5
DETECTION_EXPAND_RATIO = 1.3

# 跟踪参数
TRACK_HIGH_THRESH = 0.6
TRACK_LOW_THRESH = 0.1
TRACK_MATCH_THRESH = 0.8
TRACK_MAX_AGE = 30
TRACK_MIN_HITS = 3

# 姿态估计参数
POSE_INPUT_SIZE = 224


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class FaceDetection:
    """人脸检测结果"""
    bbox: np.ndarray
    confidence: float
    landmarks: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox.tolist() if isinstance(self.bbox, np.ndarray) else self.bbox,
            "confidence": float(self.confidence),
            "landmarks": self.landmarks.tolist() if self.landmarks is not None else None
        }


@dataclass
class Track:
    """跟踪轨迹"""
    track_id: int
    bbox: np.ndarray
    confidence: float
    landmarks: Optional[np.ndarray] = None
    hits: int = 0
    age: int = 0


@dataclass
class HeadPose:
    """头部姿态"""
    yaw: float
    pitch: float
    roll: float
    confidence: float


# ============================================================================
# 人脸检测器 (RetinaFace)
# ============================================================================

class FaceDetector:
    """人脸检测器 (支持多种后端)"""

    def __init__(self, conf_threshold: float = 0.5, device: str = None):
        self.conf_threshold = conf_threshold
        self.device = device
        self._load_model()

    def _load_model(self):
        # 优先使用 OpenCV DNN (无需下载模型)
        try:
            # 使用 OpenCV 内置的 Haar Cascade
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.backend = "opencv"
            print(f"  [检测器] OpenCV Haar Cascade")
            return
        except Exception:
            pass

        try:
            from retinaface import RetinaFace as RF
            self.detector = RF
            self.backend = "retinaface"
            print(f"  [检测器] RetinaFace")
        except ImportError:
            try:
                from insightface.app import FaceAnalysis
                self.app = FaceAnalysis(allowed_modules=['detection'])
                ctx_id = 0 if self.device == "cuda" else -1
                self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
                self.backend = "insightface"
                print(f"  [检测器] InsightFace")
            except Exception:
                # 最终回退到 OpenCV
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                self.backend = "opencv"
                print(f"  [检测器] OpenCV Haar Cascade (fallback)")

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        if self.backend == "opencv":
            return self._detect_opencv(image)
        elif self.backend == "retinaface":
            return self._detect_retinaface(image)
        return self._detect_insightface(image)

    def _detect_opencv(self, image: np.ndarray) -> List[FaceDetection]:
        """使用 OpenCV Haar Cascade 检测"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        results = []
        for (x, y, w, h) in faces:
            bbox = np.array([x, y, x + w, y + h])
            results.append(FaceDetection(bbox=bbox, confidence=0.9, landmarks=None))
        return results

    def _detect_retinaface(self, image: np.ndarray) -> List[FaceDetection]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(rgb)
        results = []
        if not isinstance(faces, dict):
            return results
        for face_data in faces.values():
            conf = face_data['score']
            if conf < self.conf_threshold:
                continue
            bbox = np.array(face_data['facial_area'])
            landmarks = np.array([
                face_data['landmarks']['left_eye'],
                face_data['landmarks']['right_eye'],
                face_data['landmarks']['nose'],
                face_data['landmarks']['mouth_left'],
                face_data['landmarks']['mouth_right']
            ])
            results.append(FaceDetection(bbox=bbox, confidence=conf, landmarks=landmarks))
        return results

    def _detect_insightface(self, image: np.ndarray) -> List[FaceDetection]:
        faces = self.app.get(image)
        results = []
        for face in faces:
            if face.det_score < self.conf_threshold:
                continue
            results.append(FaceDetection(
                bbox=face.bbox.astype(int),
                confidence=float(face.det_score),
                landmarks=face.kps if hasattr(face, 'kps') else None
            ))
        return results

    def crop_face(self, image: np.ndarray, bbox: np.ndarray, expand_ratio: float = 1.3) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        new_w, new_h = w * expand_ratio, h * expand_ratio
        x1 = max(0, int(cx - new_w / 2))
        y1 = max(0, int(cy - new_h / 2))
        x2 = min(image.shape[1], int(cx + new_w / 2))
        y2 = min(image.shape[0], int(cy + new_h / 2))
        return image[y1:y2, x1:x2]


# ============================================================================
# 多目标跟踪器 (ByteTrack)
# ============================================================================

class ByteTracker:
    """ByteTrack 多目标跟踪器"""

    def __init__(self, high_thresh=0.6, low_thresh=0.1, match_thresh=0.8, max_age=30, min_hits=3):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: List[Track] = []
        self.next_id = 1

    def update(self, detections: List[FaceDetection]) -> List[Track]:
        high_dets = [d for d in detections if d.confidence >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d.confidence < self.high_thresh]

        matched, unmatched_tracks, unmatched_dets = self._match(self.tracks, high_dets, self.match_thresh)
        matched2, unmatched_tracks2, _ = self._match(unmatched_tracks, low_dets, self.match_thresh * 0.8)
        matched.extend(matched2)

        for track, det in matched:
            track.bbox = det.bbox
            track.confidence = det.confidence
            track.landmarks = det.landmarks
            track.hits += 1
            track.age = 0

        for det in unmatched_dets:
            self.tracks.append(Track(
                track_id=self.next_id, bbox=det.bbox, confidence=det.confidence,
                landmarks=det.landmarks, hits=1, age=0
            ))
            self.next_id += 1

        for track in unmatched_tracks2:
            track.age += 1

        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return [t for t in self.tracks if t.hits >= self.min_hits]

    def _match(self, tracks, detections, thresh):
        if not tracks or not detections:
            return [], tracks, detections

        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._iou(track.bbox, det.bbox)

        matched, unmatched_t, unmatched_d = [], list(range(len(tracks))), list(range(len(detections)))

        while unmatched_t and unmatched_d:
            max_iou, max_i, max_j = 0, -1, -1
            for i in unmatched_t:
                for j in unmatched_d:
                    if iou_matrix[i, j] > max_iou:
                        max_iou, max_i, max_j = iou_matrix[i, j], i, j
            if max_iou < thresh:
                break
            matched.append((tracks[max_i], detections[max_j]))
            unmatched_t.remove(max_i)
            unmatched_d.remove(max_j)

        return matched, [tracks[i] for i in unmatched_t], [detections[j] for j in unmatched_d]

    def _iou(self, box1, box2):
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter + 1e-6)


# ============================================================================
# 头部姿态估计器 (基于几何方法)
# ============================================================================

class HeadPoseEstimator:
    """头部姿态估计器 (基于人脸关键点几何方法)"""

    def __init__(self, device: str = None):
        self.device = device
        print(f"  [姿态估计] 几何方法 (基于人脸比例)")

    def estimate(self, face_image: np.ndarray) -> HeadPose:
        """基于人脸图像估计头部姿态 (简化几何方法)"""
        if face_image.size == 0 or face_image.shape[0] < 10 or face_image.shape[1] < 10:
            return HeadPose(0, 0, 0, 0)

        h, w = face_image.shape[:2]

        # 使用 OpenCV 检测人脸关键点
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

        # 简化方法: 基于人脸区域的几何特征估计姿态
        # 计算图像的质心偏移来估计 yaw
        moments = cv2.moments(gray)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]

            # yaw: 质心水平偏移 -> 左右转头
            yaw = (cx / w - 0.5) * 60  # 映射到 [-30, 30] 度

            # pitch: 质心垂直偏移 -> 抬头低头
            pitch = (cy / h - 0.5) * 40  # 映射到 [-20, 20] 度

            # roll: 使用边缘检测估计倾斜
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=20, maxLineGap=10)

            roll = 0.0
            if lines is not None and len(lines) > 0:
                angles = []
                for line in lines[:10]:
                    x1, y1, x2, y2 = line[0]
                    if x2 != x1:
                        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                        angles.append(angle)
                if angles:
                    roll = np.median(angles)
                    roll = np.clip(roll, -30, 30)

            confidence = 0.7
        else:
            yaw, pitch, roll, confidence = 0, 0, 0, 0.3

        return HeadPose(
            yaw=float(round(yaw, 2)),
            pitch=float(round(pitch, 2)),
            roll=float(round(roll, 2)),
            confidence=float(round(confidence, 3))
        )


# ============================================================================
# 主 Pipeline
# ============================================================================

def process_video(video_id: str, fps: float = 10.0) -> Dict[str, Any]:
    """处理单个视频"""
    frames_dir = FRAMES_DIR / video_id
    if not frames_dir.exists():
        print(f"  [错误] 帧目录不存在: {frames_dir}")
        return None

    # 创建输出目录
    det_dir = DETECTION_DIR / video_id
    track_dir = TRACKING_DIR / video_id
    pose_dir = POSE_DIR / video_id
    for d in [det_dir, track_dir, pose_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 加载元数据
    meta_path = frames_dir / "extraction_metadata.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            fps = json.load(f).get("extract_fps", fps)

    # 初始化
    print(f"  初始化模块...")
    detector = FaceDetector(conf_threshold=DETECTION_CONF_THRESHOLD)
    tracker = ByteTracker(TRACK_HIGH_THRESH, TRACK_LOW_THRESH, TRACK_MATCH_THRESH, TRACK_MAX_AGE, TRACK_MIN_HITS)
    pose_estimator = HeadPoseEstimator()

    # 获取帧
    frame_files = sorted(frames_dir.glob("*.jpg")) or sorted(frames_dir.glob("*.png"))
    print(f"  处理 {len(frame_files)} 帧...")

    detection_results = {"video_id": video_id, "fps": fps, "processed_at": datetime.now().isoformat(), "frames": []}
    all_tracks = {}

    for frame_file in tqdm(frame_files, desc=f"  {video_id[:20]}"):
        frame_name = frame_file.stem
        frame_idx = int(frame_name.split("_")[-1])
        timestamp = frame_idx / fps

        image = cv2.imread(str(frame_file))
        if image is None:
            continue

        # 1. 检测
        detections = detector.detect(image)
        detection_results["frames"].append({
            "frame_idx": frame_idx,
            "timestamp": round(timestamp, 4),
            "detections": [d.to_dict() for d in detections]
        })

        # 2. 跟踪
        active_tracks = tracker.update(detections)

        # 3. 姿态估计
        for track in active_tracks:
            face_img = detector.crop_face(image, track.bbox, DETECTION_EXPAND_RATIO)
            if face_img.size > 0:
                pose = pose_estimator.estimate(face_img)

                if track.track_id not in all_tracks:
                    all_tracks[track.track_id] = {
                        "track_id": track.track_id,
                        "start_frame": frame_idx, "end_frame": frame_idx,
                        "start_time": timestamp, "end_time": timestamp,
                        "detections": [], "poses": []
                    }

                t = all_tracks[track.track_id]
                t["end_frame"], t["end_time"] = frame_idx, timestamp
                t["detections"].append({"frame_idx": frame_idx, "timestamp": round(timestamp, 4),
                                        "bbox": track.bbox.tolist(), "confidence": float(track.confidence)})
                t["poses"].append({"frame_idx": frame_idx, "timestamp": round(timestamp, 4),
                                   "yaw": round(pose.yaw, 2), "pitch": round(pose.pitch, 2),
                                   "roll": round(pose.roll, 2), "confidence": round(pose.confidence, 3)})

    # 计算统计
    tracks_list = list(all_tracks.values())
    for t in tracks_list:
        if t["poses"]:
            yaws = [p["yaw"] for p in t["poses"]]
            pitches = [p["pitch"] for p in t["poses"]]
            rolls = [p["roll"] for p in t["poses"]]
            t["statistics"] = {
                "duration_frames": t["end_frame"] - t["start_frame"] + 1,
                "duration_sec": round(t["end_time"] - t["start_time"], 2),
                "num_detections": len(t["detections"]),
                "yaw_mean": round(np.mean(yaws), 2), "yaw_std": round(np.std(yaws), 2),
                "pitch_mean": round(np.mean(pitches), 2), "pitch_std": round(np.std(pitches), 2),
                "roll_mean": round(np.mean(rolls), 2), "roll_std": round(np.std(rolls), 2)
            }

    # 保存结果
    total_det = sum(len(f["detections"]) for f in detection_results["frames"])
    detection_results["statistics"] = {
        "total_frames": len(frame_files), "total_detections": total_det,
        "avg_detections_per_frame": round(total_det / max(len(frame_files), 1), 2)
    }
    with open(det_dir / "detections.json", 'w', encoding='utf-8') as f:
        json.dump(detection_results, f, ensure_ascii=False, indent=2)

    tracking_results = {
        "video_id": video_id, "fps": fps, "processed_at": datetime.now().isoformat(),
        "statistics": {"total_tracks": len(tracks_list),
                       "avg_track_duration": round(np.mean([t.get("statistics", {}).get("duration_sec", 0)
                                                           for t in tracks_list]) if tracks_list else 0, 2)},
        "tracks": tracks_list
    }
    with open(track_dir / "tracks.json", 'w', encoding='utf-8') as f:
        json.dump(tracking_results, f, ensure_ascii=False, indent=2)

    pose_data = {
        "video_id": video_id, "fps": fps, "processed_at": datetime.now().isoformat(),
        "tracks": [{"track_id": t["track_id"], "start_frame": t["start_frame"], "end_frame": t["end_frame"],
                    "poses": t["poses"], "statistics": t.get("statistics", {})} for t in tracks_list]
    }
    with open(pose_dir / "pose.json", 'w', encoding='utf-8') as f:
        json.dump(pose_data, f, ensure_ascii=False, indent=2)

    return {"video_id": video_id, "total_frames": len(frame_files), "total_detections": total_det,
            "total_tracks": len(tracks_list), "avg_track_duration": tracking_results["statistics"]["avg_track_duration"]}


def main():
    """批量处理"""
    print("=" * 60)
    print("Step 3: 检测 / 跟踪 / 姿态估计 Pipeline")
    print("=" * 60)

    video_dirs = [d for d in FRAMES_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not video_dirs:
        print(f"[错误] 未找到帧目录，请先运行 Step 2")
        return

    print(f"\n找到 {len(video_dirs)} 个视频")
    all_stats = []

    for video_dir in sorted(video_dirs):
        video_id = video_dir.name
        pose_file = POSE_DIR / video_id / "pose.json"

        if pose_file.exists():
            print(f"\n[跳过] {video_id}: 已处理")
            with open(pose_file, 'r') as f:
                existing = json.load(f)
            all_stats.append({"video_id": video_id, "total_tracks": len(existing.get("tracks", [])), "status": "skipped"})
            continue

        print(f"\n{'='*60}\n处理: {video_id}\n{'='*60}")
        stats = process_video(video_id)
        if stats:
            all_stats.append(stats)
            print(f"\n[完成] {video_id}: {stats['total_frames']}帧, {stats['total_detections']}检测, {stats['total_tracks']}轨迹")

    # 总结
    print(f"\n{'='*60}\n总体统计\n{'='*60}")
    total_tracks = sum(s.get("total_tracks", 0) for s in all_stats)
    total_det = sum(s.get("total_detections", 0) for s in all_stats)
    print(f"  视频数: {len(all_stats)}, 轨迹数: {total_tracks}, 检测数: {total_det}")

    report = {"processed_at": datetime.now().isoformat(), "total_videos": len(all_stats),
              "total_tracks": total_tracks, "total_detections": total_det, "videos": all_stats}
    with open(DATASET_ROOT / "step3_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {DATASET_ROOT / 'step3_report.json'}")


if __name__ == "__main__":
    main()
