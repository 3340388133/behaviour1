#!/usr/bin/env python3
"""
单视频抽帧脚本 - 时间尺度统一版本

用法:
    python extract_frames.py --video <video_path> --output <output_dir> [--fps 10.0]
"""
import sys
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
from frame_extractor import extract_frames, RECOMMENDED_CONFIG


def main():
    parser = argparse.ArgumentParser(
        description="视频抽帧工具（时间戳精确版）"
    )
    parser.add_argument(
        "--video", "-v",
        required=True,
        help="输入视频路径"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="输出目录"
    )
    parser.add_argument(
        "--fps", "-f",
        type=float,
        default=RECOMMENDED_CONFIG["extract_fps"],
        help=f"抽帧帧率（默认 {RECOMMENDED_CONFIG['extract_fps']}fps）"
    )
    parser.add_argument(
        "--quality", "-q",
        type=int,
        default=95,
        help="JPEG质量（默认95）"
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    print(f"Recommended config: {RECOMMENDED_CONFIG}")
    print()

    metadata = extract_frames(
        args.video,
        args.output,
        fps=args.fps,
        quality=args.quality,
        save_metadata=True
    )

    print(f"\nExtraction complete!")
    print(f"  Metadata saved to: {Path(args.output) / 'extraction_metadata.json'}")


if __name__ == "__main__":
    main()
