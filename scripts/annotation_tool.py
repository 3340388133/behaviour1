#!/usr/bin/env python3
"""
行为标注辅助工具
用于查看姿态数据并进行人工标注
"""

import json
import numpy as np
import os
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"

BEHAVIOR_LABELS = {
    0: ("normal", "正常行为 - 视线稳定，偶尔自然转头"),
    1: ("glancing", "频繁张望 - 3秒内左右转头≥3次，yaw变化>30°"),
    2: ("quick_turn", "快速回头 - 0.5秒内yaw变化>60°"),
    3: ("prolonged_watch", "长时间观察 - 持续>3秒注视非正前方(yaw>30°)"),
    4: ("looking_down", "持续低头 - pitch<-20°持续>5秒"),
    5: ("looking_up", "持续抬头 - pitch>20°持续>3秒"),
    -1: ("uncertain", "无法判断 - 遮挡严重/质量差/边界情况"),
}


def load_pose_data(video_id):
    """加载姿态数据"""
    pose_file = DATA_DIR / "pose" / video_id / "pose.json"
    if not pose_file.exists():
        return None
    with open(pose_file, 'r') as f:
        return json.load(f)


def load_feature_index(video_id, track_id):
    """加载特征索引"""
    index_file = DATA_DIR / "features" / video_id / f"track_{track_id}_index.json"
    if not index_file.exists():
        return None
    with open(index_file, 'r') as f:
        return json.load(f)


def get_available_videos():
    """获取可标注的视频列表"""
    pose_dir = DATA_DIR / "pose"
    if not pose_dir.exists():
        return []
    return [d.name for d in pose_dir.iterdir() if d.is_dir()]


def get_tracks_for_video(video_id):
    """获取视频的所有 track"""
    features_dir = DATA_DIR / "features" / video_id
    if not features_dir.exists():
        return []
    tracks = []
    for f in features_dir.glob("track_*_index.json"):
        track_id = int(f.stem.split('_')[1])
        tracks.append(track_id)
    return sorted(tracks)


def analyze_window(poses, start_frame, end_frame):
    """分析窗口内的姿态特征"""
    window_poses = [p for p in poses if start_frame <= p['frame_idx'] <= end_frame]
    if not window_poses:
        return None

    yaws = [p['yaw'] for p in window_poses]
    pitches = [p['pitch'] for p in window_poses]
    rolls = [p['roll'] for p in window_poses]

    # 计算统计量
    stats = {
        'num_frames': len(window_poses),
        'yaw': {
            'mean': np.mean(yaws),
            'std': np.std(yaws),
            'min': np.min(yaws),
            'max': np.max(yaws),
            'range': np.max(yaws) - np.min(yaws),
        },
        'pitch': {
            'mean': np.mean(pitches),
            'std': np.std(pitches),
            'min': np.min(pitches),
            'max': np.max(pitches),
        },
        'roll': {
            'mean': np.mean(rolls),
            'std': np.std(rolls),
        }
    }

    # 计算 yaw 变化次数（跨过阈值的次数）
    yaw_changes = 0
    threshold = 15  # 度
    for i in range(1, len(yaws)):
        if abs(yaws[i] - yaws[i-1]) > threshold:
            yaw_changes += 1
    stats['yaw_changes'] = yaw_changes

    # 计算最大瞬时变化
    max_yaw_change = 0
    for i in range(1, len(yaws)):
        change = abs(yaws[i] - yaws[i-1])
        if change > max_yaw_change:
            max_yaw_change = change
    stats['max_yaw_change_per_frame'] = max_yaw_change

    # 自动建议标签
    suggestion = suggest_label(stats)
    stats['suggestion'] = suggestion

    return stats


def suggest_label(stats):
    """基于规则给出标签建议"""
    yaw_std = stats['yaw']['std']
    yaw_range = stats['yaw']['range']
    pitch_mean = stats['pitch']['mean']
    yaw_mean = abs(stats['yaw']['mean'])
    yaw_changes = stats['yaw_changes']

    # 频繁张望：yaw 标准差大，变化次数多
    if yaw_std > 25 and yaw_changes >= 3:
        return (1, "glancing", "yaw_std={:.1f}°, 变化{}次".format(yaw_std, yaw_changes))

    # 快速回头：单帧变化大
    if stats['max_yaw_change_per_frame'] > 20:  # 约 60°/s at 10fps
        return (2, "quick_turn", "最大单帧变化={:.1f}°".format(stats['max_yaw_change_per_frame']))

    # 长时间观察：yaw 均值偏离中心
    if yaw_mean > 30 and yaw_std < 15:
        return (3, "prolonged_watch", "yaw均值={:.1f}°, std={:.1f}°".format(stats['yaw']['mean'], yaw_std))

    # 持续低头
    if pitch_mean < -20:
        return (4, "looking_down", "pitch均值={:.1f}°".format(pitch_mean))

    # 持续抬头
    if pitch_mean > 20:
        return (5, "looking_up", "pitch均值={:.1f}°".format(pitch_mean))

    # 正常
    return (0, "normal", "无明显异常")


def print_window_info(video_id, track_id, window_idx, window_info, stats):
    """打印窗口信息"""
    print("\n" + "="*60)
    print(f"视频: {video_id} | Track: {track_id} | 窗口: {window_idx}")
    print(f"帧范围: {window_info['start_frame']} - {window_info['end_frame']}")
    print(f"时间范围: {window_info['start_frame']/10:.1f}s - {window_info['end_frame']/10:.1f}s")
    print("-"*60)
    print(f"有效帧数: {stats['num_frames']}")
    print(f"Yaw:   均值={stats['yaw']['mean']:6.1f}°  标准差={stats['yaw']['std']:5.1f}°  范围=[{stats['yaw']['min']:.1f}°, {stats['yaw']['max']:.1f}°]")
    print(f"Pitch: 均值={stats['pitch']['mean']:6.1f}°  标准差={stats['pitch']['std']:5.1f}°  范围=[{stats['pitch']['min']:.1f}°, {stats['pitch']['max']:.1f}°]")
    print(f"Yaw变化次数(>15°): {stats['yaw_changes']}")
    print(f"最大单帧Yaw变化: {stats['max_yaw_change_per_frame']:.1f}°")
    print("-"*60)
    suggestion = stats['suggestion']
    print(f"🤖 建议标签: [{suggestion[0]}] {suggestion[1]} ({suggestion[2]})")
    print("-"*60)
    print("标签选项:")
    for label_id, (name, desc) in BEHAVIOR_LABELS.items():
        print(f"  [{label_id:2d}] {name:15s} - {desc}")
    print("  [s] 跳过  [q] 退出并保存")


def load_existing_annotations():
    """加载已有标注"""
    anno_file = DATA_DIR / "annotations" / "behavior_labels.json"
    if anno_file.exists():
        with open(anno_file, 'r') as f:
            return json.load(f)
    return {
        "version": "1.0",
        "annotator": "manual",
        "created_at": datetime.now().isoformat(),
        "samples": []
    }


def save_annotations(annotations):
    """保存标注"""
    anno_file = DATA_DIR / "annotations" / "behavior_labels.json"
    annotations['updated_at'] = datetime.now().isoformat()
    with open(anno_file, 'w') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 标注已保存到 {anno_file}")


def get_annotated_samples(annotations):
    """获取已标注的样本ID集合"""
    return {s['sample_id'] for s in annotations['samples']}


def main():
    print("="*60)
    print("        行为识别数据集 - 人工标注工具")
    print("="*60)

    # 获取可用视频
    videos = get_available_videos()
    if not videos:
        print("❌ 没有找到可标注的视频（需要先运行姿态估计）")
        return

    print("\n可标注的视频:")
    for i, v in enumerate(videos):
        tracks = get_tracks_for_video(v)
        print(f"  [{i}] {v} ({len(tracks)} tracks)")

    # 选择视频
    try:
        choice = input("\n选择视频编号 (或 'all' 标注所有): ").strip()
        if choice.lower() == 'all':
            selected_videos = videos
        else:
            selected_videos = [videos[int(choice)]]
    except (ValueError, IndexError):
        print("无效选择")
        return

    # 加载已有标注
    annotations = load_existing_annotations()
    annotated = get_annotated_samples(annotations)
    print(f"\n已有 {len(annotated)} 个标注样本")

    # 开始标注
    total_new = 0
    for video_id in selected_videos:
        pose_data = load_pose_data(video_id)
        if not pose_data:
            continue

        tracks = get_tracks_for_video(video_id)
        for track_id in tracks:
            # 获取该 track 的姿态数据
            track_poses = None
            for t in pose_data['tracks']:
                if t['track_id'] == track_id:
                    track_poses = t['poses']
                    break

            if not track_poses:
                continue

            # 获取窗口信息
            index = load_feature_index(video_id, track_id)
            if not index:
                continue

            for window in index['windows']:
                sample_id = f"{video_id}_track{track_id}_win{window['window_idx']}"

                # 跳过已标注的
                if sample_id in annotated:
                    continue

                # 分析窗口
                stats = analyze_window(track_poses, window['start_frame'], window['end_frame'])
                if not stats:
                    continue

                # 显示信息
                print_window_info(video_id, track_id, window['window_idx'], window, stats)

                # 获取用户输入
                while True:
                    user_input = input("\n请输入标签 (数字/-1/s/q): ").strip().lower()

                    if user_input == 'q':
                        save_annotations(annotations)
                        print(f"\n本次新增 {total_new} 个标注")
                        return

                    if user_input == 's':
                        print("跳过")
                        break

                    try:
                        label = int(user_input)
                        if label not in BEHAVIOR_LABELS:
                            print("无效标签，请重新输入")
                            continue

                        # 添加标注
                        sample = {
                            "sample_id": sample_id,
                            "video_id": video_id,
                            "track_id": track_id,
                            "window_idx": window['window_idx'],
                            "start_frame": window['start_frame'],
                            "end_frame": window['end_frame'],
                            "start_time": window['start_frame'] / 10,
                            "end_time": window['end_frame'] / 10,
                            "label": label,
                            "label_name": BEHAVIOR_LABELS[label][0],
                            "auto_suggestion": stats['suggestion'][0],
                            "confidence": "high" if label == stats['suggestion'][0] else "manual",
                            "annotated_at": datetime.now().isoformat()
                        }
                        annotations['samples'].append(sample)
                        annotated.add(sample_id)
                        total_new += 1
                        print(f"✓ 已标注为 [{label}] {BEHAVIOR_LABELS[label][0]}")
                        break

                    except ValueError:
                        print("请输入有效的数字")

    save_annotations(annotations)
    print(f"\n🎉 标注完成！本次新增 {total_new} 个标注")


if __name__ == "__main__":
    main()
