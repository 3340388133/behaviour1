#!/usr/bin/env python3
"""
批量目标跟踪脚本 - 集成 Track 关联分析

特性：
1. 单视频顺序处理
2. 自动检测 Track 断裂点
3. 构建人级时间线
4. 导出标注工具数据
"""
import json
import sys
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tracker import ByteTracker
from face_detector import FaceDetection
from track_association import (
    TrackAssociator,
    PersonTimelineBuilder,
    extract_identity_features,
    save_tracking_with_associations,
    ASSOCIATION_THRESHOLDS
)

DATA_DIR = Path(__file__).parent.parent / "data"
DETECTION_DIR = DATA_DIR / "detection"
TRACKING_DIR = DATA_DIR / "tracking"


def run_tracking(video_id: str):
    """运行目标跟踪（单视频顺序处理）

    核心改进：
    1. 所有时间索引基于统一时间轴
    2. 自动检测 Track 断裂
    3. 构建人级时间线
    """
    detection_dir = DETECTION_DIR / video_id
    output_dir = TRACKING_DIR / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载检测结果
    with open(detection_dir / "detections.json", "r", encoding="utf-8") as f:
        det_data = json.load(f)

    fps = det_data.get("fps", 10.0)

    # 初始化跟踪器
    tracker = ByteTracker(
        high_thresh=0.6,
        low_thresh=0.1,
        match_thresh=0.8,
        max_age=30,      # 30帧（3秒）
        min_hits=3
    )

    # 跟踪所有帧
    all_tracks = {}

    for frame_data in tqdm(det_data["frames"], desc=f"跟踪 {video_id[:15]}"):
        frame_idx = frame_data["frame_idx"]
        timestamp = frame_data.get("timestamp", (frame_idx - 1) / fps)

        # 构建检测对象
        detections = []
        for det in frame_data["detections"]:
            # 跳过 reject 级别的检测
            quality = det.get("quality", {})
            if quality.get("level") == "reject":
                continue

            detections.append(FaceDetection(
                bbox=np.array(det["bbox"]),
                confidence=det["confidence"],
                landmarks=np.array(det["landmarks"]) if det["landmarks"] else None
            ))

        # 更新跟踪器
        tracks = tracker.update(detections)

        # 记录每个 track 的检测
        for track in tracks:
            if track.track_id not in all_tracks:
                all_tracks[track.track_id] = {
                    "track_id": track.track_id,
                    "start_frame": frame_idx,
                    "end_frame": frame_idx,
                    "start_time": timestamp,
                    "end_time": timestamp,
                    "detections": []
                }

            all_tracks[track.track_id]["end_frame"] = frame_idx
            all_tracks[track.track_id]["end_time"] = timestamp
            all_tracks[track.track_id]["detections"].append({
                "frame_idx": frame_idx,
                "timestamp": round(timestamp, 6),
                "bbox": track.bbox.tolist(),
                "confidence": float(track.confidence),
                "landmarks": track.landmarks.tolist() if track.landmarks is not None else None
            })

    tracks_list = list(all_tracks.values())

    # 提取身份特征
    print(f"  提取身份特征...")
    for track in tracks_list:
        if track["detections"]:
            feat = extract_identity_features(track)
            track["identity_features"] = feat.to_dict()

    # 构建人级时间线
    print(f"  分析 Track 关联...")
    timeline_builder = PersonTimelineBuilder(fps)
    timelines = timeline_builder.build_timelines(tracks_list, auto_merge_only=False)

    # 保存结果
    save_tracking_with_associations(
        str(output_dir / "tracks.json"),
        video_id,
        tracks_list,
        timelines,
        fps
    )

    # 导出标注工具数据
    annotation_data = timeline_builder.export_for_annotation(
        video_id, tracks_list, timelines
    )
    with open(output_dir / "annotation_data.json", "w", encoding="utf-8") as f:
        json.dump(annotation_data, f, ensure_ascii=False, indent=2)

    return len(tracks_list), len(timelines), timelines


def print_association_report(video_id: str, timelines: list):
    """打印关联分析报告"""
    total = len(timelines)
    auto_merged = sum(1 for t in timelines if t.auto_merged)
    needs_review = sum(1 for t in timelines if t.needs_review)

    print(f"  关联分析:")
    print(f"    识别人数: {total}")
    print(f"    自动合并: {auto_merged}")
    print(f"    需人工确认: {needs_review}")

    # 显示需要人工确认的情况
    if needs_review > 0:
        print(f"  需要确认的关联:")
        for t in timelines:
            if t.needs_review:
                tracks_str = ", ".join(map(str, t.merged_track_ids))
                print(f"    Person {t.person_id}: tracks [{tracks_str}]")
                for bp in t.breakpoints:
                    if bp.needs_review:
                        print(f"      断裂点: frame {bp.track1_end_frame} -> {bp.track2_start_frame}")
                        print(f"        模式: {bp.pattern.value}, 得分: {bp.association_score:.2f}")


def main():
    """批量跟踪主函数"""
    video_ids = [d.name for d in DETECTION_DIR.iterdir()
                 if d.is_dir() and not d.name.startswith('.')]

    if not video_ids:
        print("No detections found. Run detect_faces_batch.py first.")
        return

    all_summary = {
        "total_tracks": 0,
        "total_persons": 0,
        "auto_merged": 0,
        "needs_review": 0
    }

    for video_id in sorted(video_ids):
        track_file = TRACKING_DIR / video_id / "tracks.json"

        # 检查是否已有新格式的跟踪结果
        if track_file.exists():
            with open(track_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            if "person_timelines" in existing:
                print(f"[跳过] {video_id}: 跟踪已完成（含关联分析）")
                summary = existing.get("summary", {})
                all_summary["total_tracks"] += summary.get("total_tracks", 0)
                all_summary["total_persons"] += summary.get("total_persons", 0)
                all_summary["auto_merged"] += summary.get("auto_merged_persons", 0)
                all_summary["needs_review"] += summary.get("needs_review_persons", 0)
                continue

        det_file = DETECTION_DIR / video_id / "detections.json"
        if not det_file.exists():
            print(f"[跳过] {video_id}: 检测未完成")
            continue

        print(f"\n{'='*60}")
        print(f"处理: {video_id}")
        print(f"{'='*60}")

        n_tracks, n_persons, timelines = run_tracking(video_id)

        print(f"[完成] {video_id}: {n_tracks} tracks, {n_persons} persons")
        print_association_report(video_id, timelines)

        # 累加统计
        all_summary["total_tracks"] += n_tracks
        all_summary["total_persons"] += n_persons
        all_summary["auto_merged"] += sum(1 for t in timelines if t.auto_merged)
        all_summary["needs_review"] += sum(1 for t in timelines if t.needs_review)

    # 打印总体报告
    print(f"\n{'='*60}")
    print("总体统计")
    print(f"{'='*60}")
    print(f"  总 Tracks: {all_summary['total_tracks']}")
    print(f"  总 Persons: {all_summary['total_persons']}")
    print(f"  自动合并: {all_summary['auto_merged']}")
    print(f"  需人工确认: {all_summary['needs_review']}")


if __name__ == "__main__":
    main()
