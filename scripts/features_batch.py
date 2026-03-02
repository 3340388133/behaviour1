#!/usr/bin/env python3
"""批量特征提取脚本"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data"
POSE_DIR = DATA_DIR / "pose"
FEATURES_DIR = DATA_DIR / "features"

def extract_features(video_id: str, window_size: int = 30, stride: int = 15):
    """提取时序特征"""
    pose_dir = POSE_DIR / video_id
    output_dir = FEATURES_DIR / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(pose_dir / "pose.json", "r") as f:
        pose_data = json.load(f)

    n_saved = 0
    for track in pose_data["tracks"]:
        track_id = track["track_id"]
        poses = track["poses"]

        if len(poses) < window_size:
            continue

        yaw = np.array([p["yaw"] for p in poses])
        pitch = np.array([p["pitch"] for p in poses])
        roll = np.array([p["roll"] for p in poses])

        d_yaw = np.gradient(yaw)
        d_pitch = np.gradient(pitch)
        d_roll = np.gradient(roll)

        features = np.stack([yaw, pitch, roll, d_yaw, d_pitch, d_roll], axis=1)

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

        np.save(output_dir / f"track_{track_id}.npy", np.array(windows))

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

        n_saved += 1

    return n_saved

def main():
    video_ids = [d.name for d in POSE_DIR.iterdir()
                 if d.is_dir() and not d.name.startswith('.')]

    for video_id in video_ids:
        output_dir = FEATURES_DIR / video_id
        if output_dir.exists() and len(list(output_dir.glob("*.npy"))) > 0:
            print(f"[跳过] {video_id}: 特征已提取")
            continue

        pose_file = POSE_DIR / video_id / "pose.json"
        if not pose_file.exists():
            print(f"[跳过] {video_id}: 姿态估计未完成")
            continue

        print(f"\n处理: {video_id}")
        n_tracks = extract_features(video_id)
        print(f"[完成] {video_id}: {n_tracks} 条轨迹特征")

if __name__ == "__main__":
    main()
