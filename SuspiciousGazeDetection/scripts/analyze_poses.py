#!/usr/bin/env python3
"""
姿态数据分析脚本 - 使用已有姿态数据分析可疑张望行为

直接读取pose_output中的JSON数据，按正机位/侧机位分析
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 配置
CONFIG = {
    "pose_data_dir": "/root/autodl-tmp/behaviour/data/pose_output",
    "output_dir": "/root/autodl-tmp/behaviour/SuspiciousGazeDetection/output",

    # 视频分类
    "front_camera": ["1.14rg-1", "1.14zz-1", "1.14zz-2", "1.14zz-3", "1.14zz-4", "demo_result"],
    "side_camera": ["MVI_4537", "MVI_4538", "MVI_4539", "MVI_4540"],

    # 坐标系偏移
    "front_yaw_offset": 0.0,
    "side_yaw_offset": 90.0,

    # 可疑行为检测参数
    "yaw_change_threshold": 30.0,   # 单次转头角度阈值(度)
    "min_gaze_frequency": 3,        # 最小张望次数
    "yaw_variance_threshold": 200,  # 偏航方差阈值
    "yaw_range_threshold": 60,      # 偏航范围阈值(度)
    "suspicious_score_threshold": 0.5,  # 可疑分数阈值
}


def load_pose_data(video_name):
    """加载单个视频的姿态数据"""
    pose_file = os.path.join(CONFIG["pose_data_dir"], f"{video_name}_poses.json")
    if not os.path.exists(pose_file):
        print(f"  [SKIP] 文件不存在: {pose_file}")
        return None

    with open(pose_file) as f:
        return json.load(f)


def normalize_yaw(yaw, offset=0.0):
    """归一化偏航角到[-180, 180]"""
    yaw = yaw - offset
    while yaw > 180:
        yaw -= 360
    while yaw < -180:
        yaw += 360
    return yaw


def analyze_track(poses, yaw_offset=0.0):
    """分析单个track的姿态序列"""
    if len(poses) < 5:
        return {
            "suspicious": False,
            "score": 0.0,
            "num_poses": len(poses),
            "details": "序列太短"
        }

    # 提取偏航角并归一化
    yaws = np.array([normalize_yaw(p["yaw"], yaw_offset) for p in poses])
    pitches = np.array([p["pitch"] for p in poses])

    # 计算特征
    yaw_var = np.var(yaws)
    yaw_range = np.max(yaws) - np.min(yaws)
    yaw_diff = np.diff(yaws)

    # 方向变化次数
    direction_changes = np.sum(np.diff(np.sign(yaw_diff)) != 0)

    # 大幅度转头次数
    large_changes = np.sum(np.abs(yaw_diff) > CONFIG["yaw_change_threshold"])

    # 计算可疑分数
    score = 0.0

    if yaw_var > CONFIG["yaw_variance_threshold"]:
        score += 0.3
    if yaw_var > CONFIG["yaw_variance_threshold"] * 2:
        score += 0.1

    if yaw_range > CONFIG["yaw_range_threshold"]:
        score += 0.2
    if yaw_range > CONFIG["yaw_range_threshold"] * 2:
        score += 0.1

    if direction_changes >= CONFIG["min_gaze_frequency"]:
        score += 0.2
    if direction_changes >= CONFIG["min_gaze_frequency"] * 2:
        score += 0.1

    if large_changes >= 2:
        score += 0.2

    score = min(1.0, score)
    is_suspicious = score >= CONFIG["suspicious_score_threshold"]

    return {
        "suspicious": is_suspicious,
        "score": round(score, 3),
        "num_poses": len(poses),
        "yaw_variance": round(yaw_var, 2),
        "yaw_range": round(yaw_range, 2),
        "direction_changes": int(direction_changes),
        "large_changes": int(large_changes),
        "yaw_mean": round(np.mean(yaws), 2),
        "yaw_std": round(np.std(yaws), 2),
    }


def analyze_video(video_name, camera_type):
    """分析单个视频"""
    print(f"\n分析视频: {video_name} ({camera_type})")

    data = load_pose_data(video_name)
    if data is None:
        return None

    yaw_offset = CONFIG["front_yaw_offset"] if camera_type == "front" else CONFIG["side_yaw_offset"]

    results = {
        "video_name": video_name,
        "camera_type": camera_type,
        "yaw_offset": yaw_offset,
        "total_tracks": data["total_tracks"],
        "tracks": {},
        "suspicious_tracks": [],
        "summary": {}
    }

    suspicious_count = 0
    all_scores = []

    for track_id, track_data in data["tracks"].items():
        poses = track_data.get("poses", [])
        analysis = analyze_track(poses, yaw_offset)
        results["tracks"][track_id] = analysis
        all_scores.append(analysis["score"])

        if analysis["suspicious"]:
            suspicious_count += 1
            results["suspicious_tracks"].append({
                "track_id": track_id,
                "score": analysis["score"],
                "details": analysis
            })

    # 汇总
    results["summary"] = {
        "total_tracks": len(results["tracks"]),
        "suspicious_tracks": suspicious_count,
        "suspicious_ratio": round(suspicious_count / len(results["tracks"]), 3) if results["tracks"] else 0,
        "avg_score": round(np.mean(all_scores), 3) if all_scores else 0,
        "max_score": round(np.max(all_scores), 3) if all_scores else 0,
    }

    print(f"  总轨迹数: {results['summary']['total_tracks']}")
    print(f"  可疑轨迹: {suspicious_count} ({results['summary']['suspicious_ratio']*100:.1f}%)")
    print(f"  平均分数: {results['summary']['avg_score']}")

    return results


def main():
    print("="*60)
    print("可疑张望行为分析 - 基于已有姿态数据")
    print("="*60)

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "config": CONFIG,
        "front_camera": [],
        "side_camera": [],
    }

    # 分析正机位视频
    print("\n" + "="*40)
    print("正机位视频分析")
    print("="*40)

    for video_name in CONFIG["front_camera"]:
        result = analyze_video(video_name, "front")
        if result:
            all_results["front_camera"].append(result)

    # 分析侧机位视频
    print("\n" + "="*40)
    print("侧机位视频分析")
    print("="*40)

    for video_name in CONFIG["side_camera"]:
        result = analyze_video(video_name, "side")
        if result:
            all_results["side_camera"].append(result)

    # 总体汇总
    print("\n" + "="*60)
    print("总体汇总")
    print("="*60)

    # 正机位汇总
    front_total_tracks = sum(r["summary"]["total_tracks"] for r in all_results["front_camera"])
    front_suspicious = sum(r["summary"]["suspicious_tracks"] for r in all_results["front_camera"])

    print(f"\n正机位:")
    print(f"  视频数: {len(all_results['front_camera'])}")
    print(f"  总轨迹: {front_total_tracks}")
    print(f"  可疑轨迹: {front_suspicious} ({100*front_suspicious/front_total_tracks:.1f}%)" if front_total_tracks > 0 else "  可疑轨迹: 0")

    # 侧机位汇总
    side_total_tracks = sum(r["summary"]["total_tracks"] for r in all_results["side_camera"])
    side_suspicious = sum(r["summary"]["suspicious_tracks"] for r in all_results["side_camera"])

    print(f"\n侧机位:")
    print(f"  视频数: {len(all_results['side_camera'])}")
    print(f"  总轨迹: {side_total_tracks}")
    print(f"  可疑轨迹: {side_suspicious} ({100*side_suspicious/side_total_tracks:.1f}%)" if side_total_tracks > 0 else "  可疑轨迹: 0")

    # 总计
    total_tracks = front_total_tracks + side_total_tracks
    total_suspicious = front_suspicious + side_suspicious

    print(f"\n总计:")
    print(f"  总轨迹: {total_tracks}")
    print(f"  可疑轨迹: {total_suspicious} ({100*total_suspicious/total_tracks:.1f}%)" if total_tracks > 0 else "  可疑轨迹: 0")

    # 保存结果
    output_file = os.path.join(CONFIG["output_dir"], "pose_analysis_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存至: {output_file}")

    # 输出最可疑的轨迹
    print("\n" + "="*60)
    print("TOP 10 最可疑轨迹")
    print("="*60)

    all_suspicious = []
    for camera_type in ["front_camera", "side_camera"]:
        for video_result in all_results[camera_type]:
            for track in video_result["suspicious_tracks"]:
                all_suspicious.append({
                    "video": video_result["video_name"],
                    "camera": video_result["camera_type"],
                    "track_id": track["track_id"],
                    "score": track["score"],
                    "details": track["details"]
                })

    all_suspicious.sort(key=lambda x: x["score"], reverse=True)

    for i, item in enumerate(all_suspicious[:10], 1):
        print(f"\n{i}. {item['video']} / {item['track_id']}")
        print(f"   机位: {item['camera']}, 分数: {item['score']}")
        d = item['details']
        print(f"   偏航方差: {d['yaw_variance']}, 范围: {d['yaw_range']}°")
        print(f"   方向变化: {d['direction_changes']}次, 大幅转头: {d['large_changes']}次")


if __name__ == "__main__":
    main()
