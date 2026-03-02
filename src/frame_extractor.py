"""
视频抽帧模块 - 时间尺度统一版本

核心改进：
1. 使用时间戳驱动抽帧，而非帧序号
2. 保存完整的帧-时间映射 metadata
3. 支持不同行为的时间尺度需求
"""
import cv2
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime
from tqdm import tqdm


# ============================================================================
# 推荐的 FPS / 窗口时长配置表
# ============================================================================
"""
行为时间尺度分析：

| 行为类型        | 定义                        | 最小时间尺度 | 推荐fps | 窗口内帧数 |
|----------------|-----------------------------|-----------  |--------|-----------|
| quick_turn     | 0.5秒内 yaw 变化 >60°        | 0.5秒       | 10fps  | 5帧       |
| glancing       | 3秒内左右转头 ≥3次           | ~1秒/次     | 10fps  | 10帧/次   |
| prolonged_watch| 持续 >3秒 注视非正前方        | 3秒         | 10fps  | 30帧      |
| looking_down   | 持续 >5秒 低头               | 5秒         | 10fps  | 50帧      |

推荐配置：
- 抽帧 fps: 10fps（满足所有行为的最小采样需求）
- 最小窗口: 2.0秒（20帧，足够检测单次转头）
- 标准窗口: 3.0秒（30帧，覆盖频繁张望定义）
- 步长: 0.5秒（保证窗口重叠，不遗漏边界行为）

关键约束：
- quick_turn 在 0.5秒内需要 ≥3 个采样点来计算速度
  10fps 提供 5帧，满足需求
- 若降到 5fps，0.5秒只有 2-3帧，边界情况不稳定
"""

RECOMMENDED_CONFIG = {
    "extract_fps": 10.0,        # 抽帧帧率
    "window_sizes": {           # 窗口大小（秒）
        "short": 1.5,           # 快速动作检测
        "standard": 3.0,        # 标准行为窗口
        "long": 5.0             # 长时间行为
    },
    "step_size": 0.5,           # 滑动步长（秒）
    "min_samples_per_window": {
        "short": 10,            # 1.5秒 × 10fps = 15帧，最小10帧
        "standard": 20,         # 3.0秒 × 10fps = 30帧，最小20帧
        "long": 40              # 5.0秒 × 10fps = 50帧，最小40帧
    }
}


@dataclass
class FrameMetadata:
    """单帧元数据"""
    frame_idx: int          # 抽帧后的序号（从1开始）
    original_frame_idx: int # 原始视频中的帧号（从0开始）
    timestamp_sec: float    # 时间戳（秒）
    filename: str           # 保存的文件名


@dataclass
class ExtractionMetadata:
    """抽帧元数据"""
    video_path: str
    video_id: str
    original_fps: float
    original_frame_count: int
    original_duration_sec: float
    original_width: int
    original_height: int
    extract_fps: float
    extracted_frame_count: int
    extract_time: str
    output_dir: str
    frames: list  # List[FrameMetadata]

    def to_dict(self) -> dict:
        """转为可序列化的字典"""
        return {
            "video_path": self.video_path,
            "video_id": self.video_id,
            "original_fps": round(self.original_fps, 3),
            "original_frame_count": self.original_frame_count,
            "original_duration_sec": round(self.original_duration_sec, 3),
            "original_width": self.original_width,
            "original_height": self.original_height,
            "extract_fps": self.extract_fps,
            "extracted_frame_count": self.extracted_frame_count,
            "extract_time": self.extract_time,
            "output_dir": self.output_dir,
            "frames": [asdict(f) for f in self.frames]
        }


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float = 10.0,
    quality: int = 95,
    save_metadata: bool = True
) -> ExtractionMetadata:
    """从视频中按指定帧率抽取帧（时间戳精确版本）

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        fps: 目标抽帧帧率，默认10fps
        quality: JPEG质量，默认95
        save_metadata: 是否保存metadata.json

    Returns:
        ExtractionMetadata 对象，包含完整的帧-时间映射
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    # 获取视频属性
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / original_fps if original_fps > 0 else 0

    # 计算采样时间点（使用时间戳驱动，而非帧间隔）
    sample_interval = 1.0 / fps  # 采样间隔（秒）
    sample_times = []
    t = 0.0
    while t < duration:
        sample_times.append(t)
        t += sample_interval

    # 存储帧元数据
    frame_metadata_list = []

    # 进度条
    pbar = tqdm(total=len(sample_times), desc=f"Extracting frames @ {fps}fps")

    # 按时间戳采样
    frame_idx = 1  # 输出帧序号从1开始
    for target_time in sample_times:
        # 计算目标帧号（四舍五入到最近的帧）
        target_frame = round(target_time * original_fps)

        # 确保不越界
        if target_frame >= total_frames:
            break

        # 定位到目标帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()

        if not ret:
            continue

        # 保存帧
        filename = f"frame_{frame_idx:06d}.jpg"
        output_path = output_dir / filename
        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

        # 记录元数据
        actual_time = target_frame / original_fps
        frame_metadata_list.append(FrameMetadata(
            frame_idx=frame_idx,
            original_frame_idx=target_frame,
            timestamp_sec=round(actual_time, 6),
            filename=filename
        ))

        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    # 构建完整元数据
    metadata = ExtractionMetadata(
        video_path=str(video_path.absolute()),
        video_id=video_path.stem,
        original_fps=original_fps,
        original_frame_count=total_frames,
        original_duration_sec=duration,
        original_width=width,
        original_height=height,
        extract_fps=fps,
        extracted_frame_count=len(frame_metadata_list),
        extract_time=datetime.now().isoformat(),
        output_dir=str(output_dir.absolute()),
        frames=frame_metadata_list
    )

    # 保存 metadata
    if save_metadata:
        metadata_path = output_dir / "extraction_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"Metadata saved to {metadata_path}")

    print(f"Extracted {len(frame_metadata_list)} frames to {output_dir}")
    print(f"  Original: {original_fps:.2f}fps, {total_frames} frames, {duration:.2f}s")
    print(f"  Extracted: {fps}fps, {len(frame_metadata_list)} frames")

    return metadata


def get_timestamp(frame_idx: int, fps: float) -> float:
    """根据抽帧序号获取时间戳（秒）

    Args:
        frame_idx: 抽帧后的帧序号（从1开始）
        fps: 抽帧帧率

    Returns:
        时间戳（秒）
    """
    return (frame_idx - 1) / fps


def get_frame_idx(timestamp_sec: float, fps: float) -> int:
    """根据时间戳获取最近的帧序号

    Args:
        timestamp_sec: 时间戳（秒）
        fps: 抽帧帧率

    Returns:
        帧序号（从1开始）
    """
    return round(timestamp_sec * fps) + 1


def load_extraction_metadata(frames_dir: str) -> Optional[ExtractionMetadata]:
    """加载抽帧元数据

    Args:
        frames_dir: 帧目录路径

    Returns:
        ExtractionMetadata 对象，如果不存在则返回 None
    """
    metadata_path = Path(frames_dir) / "extraction_metadata.json"
    if not metadata_path.exists():
        return None

    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    frames = [
        FrameMetadata(**f) for f in data.get('frames', [])
    ]

    return ExtractionMetadata(
        video_path=data['video_path'],
        video_id=data['video_id'],
        original_fps=data['original_fps'],
        original_frame_count=data['original_frame_count'],
        original_duration_sec=data['original_duration_sec'],
        original_width=data['original_width'],
        original_height=data['original_height'],
        extract_fps=data['extract_fps'],
        extracted_frame_count=data['extracted_frame_count'],
        extract_time=data['extract_time'],
        output_dir=data['output_dir'],
        frames=frames
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="视频抽帧工具（时间戳精确版）")
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="抽帧帧率（默认10fps）")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG质量（默认95）")
    args = parser.parse_args()

    extract_frames(args.video, args.output, args.fps, args.quality)
