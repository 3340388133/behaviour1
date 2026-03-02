#!/usr/bin/env python3
"""
生成待标注样本的预览，包含姿态统计和建议标签
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"

BEHAVIOR_LABELS = {
    0: "normal",
    1: "glancing",
    2: "quick_turn",
    3: "prolonged_watch",
    4: "looking_down",
    5: "looking_up",
    -1: "uncertain",
}


def load_pose_data(video_id):
    pose_file = DATA_DIR / "pose" / video_id / "pose.json"
    if not pose_file.exists():
        return None
    with open(pose_file, 'r') as f:
        return json.load(f)


def load_feature_index(video_id, track_id):
    index_file = DATA_DIR / "features" / video_id / f"track_{track_id}_index.json"
    if not index_file.exists():
        return None
    with open(index_file, 'r') as f:
        return json.load(f)


def get_available_videos():
    pose_dir = DATA_DIR / "pose"
    if not pose_dir.exists():
        return []
    return [d.name for d in pose_dir.iterdir() if d.is_dir()]


def get_tracks_for_video(video_id):
    features_dir = DATA_DIR / "features" / video_id
    if not features_dir.exists():
        return []
    tracks = []
    for f in features_dir.glob("track_*_index.json"):
        track_id = int(f.stem.split('_')[1])
        tracks.append(track_id)
    return sorted(tracks)


def analyze_window(poses, start_frame, end_frame):
    window_poses = [p for p in poses if start_frame <= p['frame_idx'] <= end_frame]
    if not window_poses or len(window_poses) < 5:
        return None

    yaws = [p['yaw'] for p in window_poses]
    pitches = [p['pitch'] for p in window_poses]

    stats = {
        'num_frames': len(window_poses),
        'yaw_mean': round(np.mean(yaws), 1),
        'yaw_std': round(np.std(yaws), 1),
        'yaw_min': round(np.min(yaws), 1),
        'yaw_max': round(np.max(yaws), 1),
        'pitch_mean': round(np.mean(pitches), 1),
        'pitch_std': round(np.std(pitches), 1),
    }

    # 计算 yaw 变化次数
    yaw_changes = 0
    for i in range(1, len(yaws)):
        if abs(yaws[i] - yaws[i-1]) > 15:
            yaw_changes += 1
    stats['yaw_changes'] = yaw_changes

    # 最大单帧变化
    max_change = max(abs(yaws[i] - yaws[i-1]) for i in range(1, len(yaws))) if len(yaws) > 1 else 0
    stats['max_yaw_change'] = round(max_change, 1)

    # 建议标签
    suggestion = suggest_label(stats)
    stats['suggestion'] = suggestion

    return stats


def suggest_label(stats):
    yaw_std = stats['yaw_std']
    pitch_mean = stats['pitch_mean']
    yaw_mean = abs(stats['yaw_mean'])
    yaw_changes = stats['yaw_changes']
    max_change = stats['max_yaw_change']

    if yaw_std > 25 and yaw_changes >= 3:
        return (1, "glancing")
    if max_change > 20:
        return (2, "quick_turn")
    if yaw_mean > 30 and yaw_std < 15:
        return (3, "prolonged_watch")
    if pitch_mean < -20:
        return (4, "looking_down")
    if pitch_mean > 20:
        return (5, "looking_up")
    return (0, "normal")


def load_existing_annotations():
    anno_file = DATA_DIR / "annotations" / "behavior_labels.json"
    if anno_file.exists():
        with open(anno_file, 'r') as f:
            data = json.load(f)
            return {s['sample_id'] for s in data.get('samples', [])}
    return set()


def main():
    videos = get_available_videos()
    annotated = load_existing_annotations()

    all_samples = []

    for video_id in videos:
        pose_data = load_pose_data(video_id)
        if not pose_data:
            continue

        tracks = get_tracks_for_video(video_id)
        for track_id in tracks:
            track_poses = None
            for t in pose_data['tracks']:
                if t['track_id'] == track_id:
                    track_poses = t['poses']
                    break

            if not track_poses:
                continue

            index = load_feature_index(video_id, track_id)
            if not index:
                continue

            for window in index['windows']:
                sample_id = f"{video_id}_track{track_id}_win{window['window_idx']}"

                if sample_id in annotated:
                    continue

                stats = analyze_window(track_poses, window['start_frame'], window['end_frame'])
                if not stats:
                    continue

                sample = {
                    'sample_id': sample_id,
                    'video_id': video_id,
                    'track_id': track_id,
                    'window_idx': window['window_idx'],
                    'start_frame': window['start_frame'],
                    'end_frame': window['end_frame'],
                    'time_range': f"{window['start_frame']/10:.1f}s - {window['end_frame']/10:.1f}s",
                    'stats': stats,
                    'suggested_label': stats['suggestion'][0],
                    'suggested_name': stats['suggestion'][1],
                    'your_label': None,  # 用户填写
                }
                all_samples.append(sample)

    # 按建议标签分组统计
    by_suggestion = {}
    for s in all_samples:
        key = s['suggested_name']
        by_suggestion[key] = by_suggestion.get(key, 0) + 1

    print(f"待标注样本总数: {len(all_samples)}")
    print(f"已标注样本数: {len(annotated)}")
    print("\n按建议标签分布:")
    for name, count in sorted(by_suggestion.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")

    # 保存预览文件
    output = {
        'generated_at': datetime.now().isoformat(),
        'total_samples': len(all_samples),
        'by_suggestion': by_suggestion,
        'samples': all_samples
    }

    output_file = DATA_DIR / "annotations" / "pending_annotations.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n预览文件已保存到: {output_file}")

    # 生成简化的标注表格
    print("\n" + "="*80)
    print("待标注样本预览 (前20个)")
    print("="*80)
    print(f"{'序号':<4} {'sample_id':<35} {'时间':<15} {'yaw均值':<8} {'yaw_std':<8} {'建议':<12}")
    print("-"*80)

    for i, s in enumerate(all_samples[:20]):
        print(f"{i:<4} {s['sample_id']:<35} {s['time_range']:<15} {s['stats']['yaw_mean']:<8} {s['stats']['yaw_std']:<8} {s['suggested_name']:<12}")


if __name__ == "__main__":
    main()
