#!/usr/bin/env python3
"""
批量人脸检测脚本 - 集成质量门控

特性：
1. 检测结果质量评估
2. 低质量检测标记（不过滤，保留完整信息）
3. 生成检测质量统计报告
"""
import json
import sys
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from face_detector import RetinaFaceDetector, FaceDetection
from face_quality import QualityLevel, QUALITY_THRESHOLDS

DATA_DIR = Path(__file__).parent.parent / "data"
FRAMES_DIR = DATA_DIR / "frames"
DETECTION_DIR = DATA_DIR / "detection"


def run_detection(video_id: str, min_quality_level: QualityLevel = None):
    """运行人脸检测（带质量评估）

    Args:
        video_id: 视频ID
        min_quality_level: 最低质量等级过滤，None 表示不过滤
    """
    frames_dir = FRAMES_DIR / video_id
    output_dir = DETECTION_DIR / video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    faces_dir = output_dir / "faces"
    faces_dir.mkdir(exist_ok=True)

    # 加载抽帧元数据（获取 fps）
    extraction_meta_path = frames_dir / "extraction_metadata.json"
    if extraction_meta_path.exists():
        with open(extraction_meta_path, 'r', encoding='utf-8') as f:
            extraction_meta = json.load(f)
        fps = extraction_meta.get("extract_fps", 10.0)
    else:
        fps = 10.0

    # 初始化检测器（启用质量评估）
    detector = RetinaFaceDetector(
        conf_threshold=0.5,
        enable_quality_assessment=True
    )

    frame_files = sorted(frames_dir.glob("frame_*.jpg"))

    results = {
        "video_id": video_id,
        "fps": fps,
        "detection_time": datetime.now().isoformat(),
        "quality_thresholds": QUALITY_THRESHOLDS,
        "frames": []
    }

    # 质量统计
    quality_stats = {
        "total_detections": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
        "reject": 0,
        "by_issue": {}
    }

    for frame_file in tqdm(frame_files, desc=f"检测 {video_id[:15]}"):
        frame_idx = int(frame_file.stem.split("_")[1])
        timestamp = (frame_idx - 1) / fps

        image = cv2.imread(str(frame_file))
        if image is None:
            continue

        # 检测人脸（带质量评估，不过滤）
        detections = detector.detect(
            image,
            assess_quality=True,
            min_quality_level=min_quality_level
        )

        frame_data = {
            "frame_idx": frame_idx,
            "timestamp": round(timestamp, 6),
            "detections": []
        }

        for det_idx, det in enumerate(detections):
            # 保存人脸图像
            face_filename = f"frame_{frame_idx:06d}_face_{det_idx}.jpg"
            face_path = faces_dir / face_filename
            face_img = detector.crop_face(image, det, expand_ratio=1.3)
            if face_img.size > 0:
                cv2.imwrite(str(face_path), face_img)

            # 构建检测数据
            det_data = {
                "det_id": det_idx,
                "bbox": det.bbox.tolist(),
                "confidence": float(det.confidence),
                "landmarks": det.landmarks.tolist() if det.landmarks is not None else None,
                "face_path": f"faces/{face_filename}"
            }

            # 添加质量信息
            if det.quality:
                det_data["quality"] = det.quality.to_dict()

                # 更新统计
                quality_stats["total_detections"] += 1
                level_name = det.quality.level.value
                quality_stats[level_name] = quality_stats.get(level_name, 0) + 1

                for issue in det.quality.issues:
                    quality_stats["by_issue"][issue] = quality_stats["by_issue"].get(issue, 0) + 1

            frame_data["detections"].append(det_data)

        results["frames"].append(frame_data)

    # 添加统计信息
    results["quality_statistics"] = quality_stats

    # 保存检测结果
    with open(output_dir / "detections.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    n_faces = quality_stats["total_detections"]
    return len(frame_files), n_faces, quality_stats


def print_quality_report(video_id: str, stats: dict):
    """打印质量报告"""
    total = stats["total_detections"]
    if total == 0:
        print(f"  No detections")
        return

    high = stats.get("high", 0)
    medium = stats.get("medium", 0)
    low = stats.get("low", 0)
    reject = stats.get("reject", 0)

    print(f"  质量分布:")
    print(f"    HIGH:   {high:4d} ({high/total*100:5.1f}%)")
    print(f"    MEDIUM: {medium:4d} ({medium/total*100:5.1f}%)")
    print(f"    LOW:    {low:4d} ({low/total*100:5.1f}%)")
    print(f"    REJECT: {reject:4d} ({reject/total*100:5.1f}%)")

    if stats.get("by_issue"):
        print(f"  常见问题:")
        for issue, count in sorted(stats["by_issue"].items(), key=lambda x: -x[1])[:5]:
            print(f"    {issue}: {count}")


def main():
    """批量检测主函数"""
    # 获取已抽帧的视频
    video_ids = [d.name for d in FRAMES_DIR.iterdir()
                 if d.is_dir() and not d.name.startswith('.')]

    if not video_ids:
        print("No frames found. Run extract_frames_batch.py first.")
        return

    all_stats = {
        "total_detections": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
        "reject": 0
    }

    for video_id in sorted(video_ids):
        det_file = DETECTION_DIR / video_id / "detections.json"
        if det_file.exists():
            print(f"[跳过] {video_id}: 检测已完成")
            # 加载已有统计
            with open(det_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            if "quality_statistics" in existing:
                stats = existing["quality_statistics"]
                for key in ["total_detections", "high", "medium", "low", "reject"]:
                    all_stats[key] = all_stats.get(key, 0) + stats.get(key, 0)
            continue

        print(f"\n处理: {video_id}")
        n_frames, n_faces, stats = run_detection(video_id)
        print(f"[完成] {video_id}: {n_frames} 帧, {n_faces} 人脸")
        print_quality_report(video_id, stats)

        # 累加统计
        for key in ["total_detections", "high", "medium", "low", "reject"]:
            all_stats[key] = all_stats.get(key, 0) + stats.get(key, 0)

    # 打印总体报告
    print(f"\n{'='*60}")
    print("总体质量报告")
    print(f"{'='*60}")
    print_quality_report("ALL", all_stats)


if __name__ == "__main__":
    main()
