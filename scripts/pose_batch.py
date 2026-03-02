#!/usr/bin/env python3
"""批量姿态估计脚本"""
import json
import sys
from pathlib import Path
from tqdm import tqdm
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data"
DETECTION_DIR = DATA_DIR / "detection"
TRACKING_DIR = DATA_DIR / "tracking"
POSE_DIR = DATA_DIR / "pose"

def run_pose_estimation(video_id: str):
    """运行姿态估计"""
    from src.head_pose import HeadPoseEstimator

    detection_dir = DETECTION_DIR / video_id
    tracking_dir = TRACKING_DIR / video_id
    output_dir = POSE_DIR / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    estimator = HeadPoseEstimator()

    with open(tracking_dir / "tracks.json", "r") as f:
        track_data = json.load(f)

    faces_dir = detection_dir / "faces"
    results = {"video_id": video_id, "tracks": []}

    for track in tqdm(track_data["tracks"], desc=f"姿态 {video_id[:15]}"):
        track_poses = {"track_id": track["track_id"], "poses": []}

        for det in track["detections"]:
            frame_idx = det["frame_idx"]
            face_files = list(faces_dir.glob(f"frame_{frame_idx:06d}_face_*.jpg"))

            if face_files:
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

    with open(output_dir / "pose.json", "w") as f:
        json.dump(results, f, indent=2)

    return len(results["tracks"])

def main():
    video_ids = [d.name for d in TRACKING_DIR.iterdir()
                 if d.is_dir() and not d.name.startswith('.')]

    for video_id in video_ids:
        pose_file = POSE_DIR / video_id / "pose.json"
        if pose_file.exists():
            print(f"[跳过] {video_id}: 姿态估计已完成")
            continue

        track_file = TRACKING_DIR / video_id / "tracks.json"
        if not track_file.exists():
            print(f"[跳过] {video_id}: 跟踪未完成")
            continue

        print(f"\n处理: {video_id}")
        n_tracks = run_pose_estimation(video_id)
        print(f"[完成] {video_id}: {n_tracks} 条轨迹")

if __name__ == "__main__":
    main()
