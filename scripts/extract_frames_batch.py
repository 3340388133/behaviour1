#!/usr/bin/env python3
"""
批量视频抽帧脚本 - 时间尺度统一版本

特性：
1. 使用时间戳驱动抽帧（精确采样）
2. 自动保存 extraction_metadata.json
3. 生成全局 video_info.json 汇总
"""
import sys
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
from datetime import datetime
from frame_extractor import extract_frames, RECOMMENDED_CONFIG

# 配置
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
FRAMES_DIR = DATA_DIR / "frames"
METADATA_DIR = DATA_DIR / "metadata"
TARGET_FPS = RECOMMENDED_CONFIG["extract_fps"]  # 10fps


def main():
    """批量抽帧主函数"""
    video_extensions = {'.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV'}
    videos = [f for f in RAW_VIDEOS_DIR.iterdir()
              if f.suffix in video_extensions and not f.name.startswith('.')]

    if not videos:
        print(f"No videos found in {RAW_VIDEOS_DIR}")
        return

    print(f"Found {len(videos)} videos to process")
    print(f"Target FPS: {TARGET_FPS}")
    print(f"Recommended config: {RECOMMENDED_CONFIG}")
    print()

    # 存储所有视频的元信息
    all_video_info = {
        "extract_time": datetime.now().isoformat(),
        "config": RECOMMENDED_CONFIG,
        "videos": []
    }

    total_frames = 0
    for video_path in sorted(videos):
        video_id = video_path.stem
        output_dir = FRAMES_DIR / video_id

        # 检查是否已有 metadata（跳过已处理的）
        metadata_path = output_dir / "extraction_metadata.json"
        if metadata_path.exists():
            print(f"[跳过] {video_id}: 已有 extraction_metadata.json")
            # 仍然加载已有信息到汇总
            with open(metadata_path, 'r', encoding='utf-8') as f:
                existing_meta = json.load(f)
            all_video_info["videos"].append({
                "video_id": video_id,
                "video_path": existing_meta.get("video_path", str(video_path)),
                "original_fps": existing_meta.get("original_fps"),
                "original_frame_count": existing_meta.get("original_frame_count"),
                "original_duration_sec": existing_meta.get("original_duration_sec"),
                "original_width": existing_meta.get("original_width"),
                "original_height": existing_meta.get("original_height"),
                "extracted_frame_count": existing_meta.get("extracted_frame_count"),
                "frames_dir": str(output_dir)
            })
            total_frames += existing_meta.get("extracted_frame_count", 0)
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {video_path.name}")
        print(f"{'='*60}")

        try:
            metadata = extract_frames(
                str(video_path),
                str(output_dir),
                fps=TARGET_FPS,
                quality=95,
                save_metadata=True
            )

            total_frames += metadata.extracted_frame_count

            # 收集视频信息
            all_video_info["videos"].append({
                "video_id": video_id,
                "video_path": str(video_path),
                "original_fps": metadata.original_fps,
                "original_frame_count": metadata.original_frame_count,
                "original_duration_sec": metadata.original_duration_sec,
                "original_width": metadata.original_width,
                "original_height": metadata.original_height,
                "extracted_frame_count": metadata.extracted_frame_count,
                "frames_dir": str(output_dir)
            })

        except Exception as e:
            print(f"Error processing {video_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 保存全局视频信息
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    video_info_path = METADATA_DIR / "video_info.json"
    with open(video_info_path, 'w', encoding='utf-8') as f:
        json.dump(all_video_info, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total videos processed: {len(all_video_info['videos'])}")
    print(f"Total frames extracted: {total_frames}")
    print(f"Global metadata saved to: {video_info_path}")


if __name__ == "__main__":
    main()
