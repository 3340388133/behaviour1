"""
视频格式转换工具
将 HEVC (H.265) 视频转换为 H.264 格式，便于 CVAT 导入
"""
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse


def convert_video(input_path: str, output_path: str,
                  target_width: int = 1920,
                  target_fps: float = None):
    """转换视频格式

    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        target_width: 目标宽度（保持宽高比）
        target_fps: 目标帧率（None 保持原帧率）
    """
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"无法打开视频: {input_path}")
        return False

    # 获取原始视频信息
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算目标尺寸
    scale = target_width / orig_width
    new_width = target_width
    new_height = int(orig_height * scale)
    # 确保高度是偶数
    new_height = new_height - (new_height % 2)

    fps = target_fps if target_fps else orig_fps

    print(f"输入: {orig_width}x{orig_height} @ {orig_fps:.1f}fps")
    print(f"输出: {new_width}x{new_height} @ {fps:.1f}fps")
    print(f"总帧数: {total_frames}")

    # 创建输出目录
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 使用 H.264 编码器 (mp4v 或 avc1)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    if not out.isOpened():
        print(f"无法创建输出视频: {output_path}")
        cap.release()
        return False

    # 转换
    with tqdm(total=total_frames, desc="转换中") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 缩放
            if scale != 1.0:
                frame = cv2.resize(frame, (new_width, new_height))

            out.write(frame)
            pbar.update(1)

    cap.release()
    out.release()

    # 检查输出文件
    output_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"输出文件大小: {output_size:.1f} MB")

    return True


def batch_convert(input_dir: str, output_dir: str,
                  target_width: int = 1920,
                  pattern: str = "*.MP4"):
    """批量转换目录下的视频"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = list(input_dir.glob(pattern)) + list(input_dir.glob(pattern.lower()))
    video_files = list(set(video_files))  # 去重

    print(f"找到 {len(video_files)} 个视频文件")

    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] 处理: {video_file.name}")

        output_path = output_dir / f"{video_file.stem}_h264.mp4"

        if output_path.exists():
            print(f"  跳过（已存在）: {output_path.name}")
            continue

        success = convert_video(
            str(video_file),
            str(output_path),
            target_width=target_width
        )

        if success:
            print(f"  完成: {output_path.name}")
        else:
            print(f"  失败: {video_file.name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='视频格式转换工具')
    parser.add_argument('input', help='输入视频或目录')
    parser.add_argument('-o', '--output', help='输出路径或目录')
    parser.add_argument('-w', '--width', type=int, default=1920,
                        help='目标宽度 (默认: 1920)')
    parser.add_argument('--batch', action='store_true',
                        help='批量处理目录')

    args = parser.parse_args()

    if args.batch:
        output_dir = args.output or 'data/videos_h264'
        batch_convert(args.input, output_dir, target_width=args.width)
    else:
        if not args.output:
            input_path = Path(args.input)
            args.output = str(input_path.parent / f"{input_path.stem}_h264.mp4")
        convert_video(args.input, args.output, target_width=args.width)
