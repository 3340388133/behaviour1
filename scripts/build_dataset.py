#!/usr/bin/env python3
"""
数据集构建批量处理脚本
按照 CLAUDE.md 规范完成：抽帧 → 检测 → 跟踪 → 姿态估计 → 特征提取
"""
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import cv2
import numpy as np

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
FRAMES_DIR = DATA_DIR / "frames"
DETECTION_DIR = DATA_DIR / "detection"
TRACKING_DIR = DATA_DIR / "tracking"
POSE_DIR = DATA_DIR / "pose"
FEATURES_DIR = DATA_DIR / "features"
METADATA_DIR = DATA_DIR / "metadata"


def get_video_id(video_path: Path) -> str:
    """从视频路径获取 video_id"""
    return video_path.stem


def extract_frames(video_path: Path, output_dir: Path, fps: float = 10.0):
    """抽帧"""
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / fps))

    frame_idx = 0
    save_idx = 1
    saved_count = 0

    pbar = tqdm(total=total_frames // frame_interval,
                desc=f"抽帧 {video_path.name[:10]}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            filename = output_dir / f"frame_{save_idx:06d}.jpg"
            cv2.imwrite(str(filename), frame)
            save_idx += 1
            saved_count += 1
            pbar.update(1)

        frame_idx += 1

    cap.release()
    pbar.close()

    return {
        "video_fps": video_fps,
        "extract_fps": fps,
        "total_frames": total_frames,
        "extracted_frames": saved_count
    }


def run_detection(video_id: str, frames_dir: Path, output_dir: Path):
    """运行人脸检测"""
    from src.face_detector import RetinaFaceDetector

    output_dir.mkdir(parents=True, exist_ok=True)
    faces_dir = output_dir / "faces"
    faces_dir.mkdir(exist_ok=True)

    detector = RetinaFaceDetector(conf_threshold=0.5)

    frame_files = sorted(frames_dir.glob("frame_*.jpg"))

    results = {
        "video_id": video_id,
        "fps": 10,
        "frames": []
    }

    for frame_file in tqdm(frame_files, desc=f"检测 {video_id[:10]}"):
        frame_idx = int(frame_file.stem.split("_")[1])
        image = cv2.imread(str(frame_file))

        detections = detector.detect(image)

        frame_data = {
            "frame_idx": frame_idx,
            "timestamp": (frame_idx - 1) / 10.0,
            "detections": []
        }

        for det_idx, det in enumerate(detections):
            face_filename = f"frame_{frame_idx:06d}_face_{det_idx}.jpg"
            face_path = faces_dir / face_filename

            # 裁剪并保存人脸
            face_img = detector.crop_face(image, det, expand_ratio=1.3)
            cv2.imwrite(str(face_path), face_img)

            frame_data["detections"].append({
                "det_id": det_idx,
                "bbox": det.bbox.tolist(),
                "confidence": float(det.confidence),
                "landmarks": det.landmarks.tolist() if det.landmarks is not None else None,
                "face_path": f"faces/{face_filename}"
            })

        results["frames"].append(frame_data)

    # 保存检测结果
    with open(output_dir / "detections.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return len(frame_files), sum(len(f["detections"]) for f in results["frames"])


def run_tracking(video_id: str, detection_dir: Path, output_dir: Path):
    """运行目标跟踪"""
    from src.tracker import ByteTracker
    from src.face_detector import FaceDetection

    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载检测结果
    with open(detection_dir / "detections.json", "r", encoding="utf-8") as f:
        det_data = json.load(f)

    tracker = ByteTracker(
        high_thresh=0.6,
        low_thresh=0.1,
        match_thresh=0.8,
        max_age=30,
        min_hits=3
    )

    # 按帧处理
    all_tracks = {}  # track_id -> list of detections

    for frame_data in tqdm(det_data["frames"], desc=f"跟踪 {video_id[:10]}"):
        frame_idx = frame_data["frame_idx"]

        # 构建检测对象
        detections = []
        for det in frame_data["detections"]:
            detections.append(FaceDetection(
                bbox=np.array(det["bbox"]),
                confidence=det["confidence"],
                landmarks=np.array(det["landmarks"]) if det["landmarks"] else None
            ))

        # 更新跟踪器
        tracks = tracker.update(detections)

        # 记录轨迹
        for track in tracks:
            if track.track_id not in all_tracks:
                all_tracks[track.track_id] = {
                    "track_id": track.track_id,
                    "start_frame": frame_idx,
                    "end_frame": frame_idx,
                    "detections": []
                }

            all_tracks[track.track_id]["end_frame"] = frame_idx
            all_tracks[track.track_id]["detections"].append({
                "frame_idx": frame_idx,
                "bbox": track.bbox.tolist(),
                "confidence": float(track.confidence)
            })

    # 保存跟踪结果
    results = {
        "video_id": video_id,
        "tracks": list(all_tracks.values())
    }

    with open(output_dir / "tracks.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return len(all_tracks)


def run_pose_estimation(video_id: str, detection_dir: Path,
                        tracking_dir: Path, output_dir: Path):
    """运行姿态估计"""
    from src.head_pose import HeadPoseEstimator

    output_dir.mkdir(parents=True, exist_ok=True)

    estimator = HeadPoseEstimator()

    # 加载跟踪结果
    with open(tracking_dir / "tracks.json", "r", encoding="utf-8") as f:
        track_data = json.load(f)

    faces_dir = detection_dir / "faces"

    results = {
        "video_id": video_id,
        "tracks": []
    }

    for track in tqdm(track_data["tracks"], desc=f"姿态 {video_id[:10]}"):
        track_poses = {
            "track_id": track["track_id"],
            "poses": []
        }

        for det in track["detections"]:
            frame_idx = det["frame_idx"]

            # 查找对应的人脸图像
            face_files = list(faces_dir.glob(f"frame_{frame_idx:06d}_face_*.jpg"))

            if face_files:
                # 使用第一个匹配的人脸（简化处理）
                face_img = cv2.imread(str(face_files[0]))
                pose = estimator.estimate(face_img)

                track_poses["poses"].append({
                    "frame_idx": frame_idx,
                    "yaw": pose.yaw,
                    "pitch": pose.pitch,
                    "roll": pose.roll,
                    "confidence": pose.confidence
                })

        results["tracks"].append(track_poses)

    with open(output_dir / "pose.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return len(results["tracks"])


def extract_features(video_id: str, pose_dir: Path, output_dir: Path,
                     window_size: int = 30, stride: int = 15):
    """提取时序特征"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载姿态数据
    with open(pose_dir / "pose.json", "r", encoding="utf-8") as f:
        pose_data = json.load(f)

    for track in pose_data["tracks"]:
        track_id = track["track_id"]
        poses = track["poses"]

        if len(poses) < window_size:
            continue

        # 提取原始姿态序列
        yaw = np.array([p["yaw"] for p in poses])
        pitch = np.array([p["pitch"] for p in poses])
        roll = np.array([p["roll"] for p in poses])

        # 计算一阶差分
        d_yaw = np.gradient(yaw)
        d_pitch = np.gradient(pitch)
        d_roll = np.gradient(roll)

        # 组合特征 [yaw, pitch, roll, d_yaw, d_pitch, d_roll]
        features = np.stack([yaw, pitch, roll, d_yaw, d_pitch, d_roll], axis=1)

        # 滑动窗口
        windows = []
        window_info = []

        for i in range(0, len(features) - window_size + 1, stride):
            windows.append(features[i:i + window_size])
            window_info.append({
                "window_idx": len(window_info),
                "start_frame": poses[i]["frame_idx"],
                "end_frame": poses[i + window_size - 1]["frame_idx"]
            })

        if not windows:
            continue

        # 保存特征
        windows_array = np.array(windows)
        np.save(output_dir / f"track_{track_id}.npy", windows_array)

        # 保存索引
        index_data = {
            "track_id": track_id,
            "video_id": video_id,
            "window_size": window_size,
            "stride": stride,
            "feature_dim": 6,
            "features": ["yaw", "pitch", "roll", "d_yaw", "d_pitch", "d_roll"],
            "windows": window_info
        }

        with open(output_dir / f"track_{track_id}_index.json", "w") as f:
            json.dump(index_data, f, indent=2)


def process_video(video_path: Path, skip_existing: bool = True):
    """处理单个视频的完整流程"""
    video_id = get_video_id(video_path)
    print(f"\n{'='*50}")
    print(f"处理视频: {video_id}")
    print(f"{'='*50}")

    frames_dir = FRAMES_DIR / video_id
    detection_dir = DETECTION_DIR / video_id
    tracking_dir = TRACKING_DIR / video_id
    pose_dir = POSE_DIR / video_id
    features_dir = FEATURES_DIR / video_id

    stats = {"video_id": video_id}

    # 1. 抽帧
    if skip_existing and frames_dir.exists() and len(list(frames_dir.glob("*.jpg"))) > 0:
        print(f"[跳过] 抽帧已完成: {len(list(frames_dir.glob('*.jpg')))} 帧")
        stats["frames"] = len(list(frames_dir.glob("*.jpg")))
    else:
        frame_stats = extract_frames(video_path, frames_dir, fps=10.0)
        stats["frames"] = frame_stats["extracted_frames"]
        print(f"[完成] 抽帧: {stats['frames']} 帧")

    # 2. 检测
    if skip_existing and (detection_dir / "detections.json").exists():
        print(f"[跳过] 检测已完成")
    else:
        n_frames, n_faces = run_detection(video_id, frames_dir, detection_dir)
        stats["detections"] = n_faces
        print(f"[完成] 检测: {n_faces} 个人脸")

    # 3. 跟踪
    if skip_existing and (tracking_dir / "tracks.json").exists():
        print(f"[跳过] 跟踪已完成")
    else:
        n_tracks = run_tracking(video_id, detection_dir, tracking_dir)
        stats["tracks"] = n_tracks
        print(f"[完成] 跟踪: {n_tracks} 条轨迹")

    # 4. 姿态估计
    if skip_existing and (pose_dir / "pose.json").exists():
        print(f"[跳过] 姿态估计已完成")
    else:
        n_tracks = run_pose_estimation(video_id, detection_dir, tracking_dir, pose_dir)
        print(f"[完成] 姿态估计: {n_tracks} 条轨迹")

    # 5. 特征提取
    if skip_existing and features_dir.exists() and len(list(features_dir.glob("*.npy"))) > 0:
        print(f"[跳过] 特征提取已完成")
    else:
        extract_features(video_id, pose_dir, features_dir)
        n_features = len(list(features_dir.glob("*.npy")))
        print(f"[完成] 特征提取: {n_features} 个轨迹特征")

    return stats


def main():
    parser = argparse.ArgumentParser(description="数据集构建批量处理")
    parser.add_argument("--video", type=str, help="处理单个视频")
    parser.add_argument("--all", action="store_true", help="处理所有视频")
    parser.add_argument("--force", action="store_true", help="强制重新处理")
    args = parser.parse_args()

    if args.video:
        video_path = RAW_VIDEOS_DIR / args.video
        if not video_path.exists():
            print(f"视频不存在: {video_path}")
            return
        process_video(video_path, skip_existing=not args.force)

    elif args.all:
        videos = list(RAW_VIDEOS_DIR.glob("*.mp4")) + list(RAW_VIDEOS_DIR.glob("*.MP4"))
        print(f"找到 {len(videos)} 个视频")

        all_stats = []
        for video_path in videos:
            stats = process_video(video_path, skip_existing=not args.force)
            all_stats.append(stats)

        # 保存统计信息
        METADATA_DIR.mkdir(exist_ok=True)
        with open(METADATA_DIR / "processing_stats.json", "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "videos": all_stats
            }, f, indent=2, ensure_ascii=False)

        print(f"\n处理完成，统计信息已保存到 {METADATA_DIR / 'processing_stats.json'}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
