"""测试检测+跟踪流水线"""
import cv2
import numpy as np
import time
from pathlib import Path

from face_detector import RetinaFaceDetector
from tracker import ByteTracker, Track


def draw_tracks(frame: np.ndarray, tracks: list, show_id: bool = True) -> np.ndarray:
    """在帧上绘制跟踪结果"""
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 255), (255, 128, 0), (0, 128, 255)
    ]

    for track in tracks:
        color = colors[track.track_id % len(colors)]
        x1, y1, x2, y2 = track.bbox.astype(int)

        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 绘制ID和置信度
        if show_id:
            label = f"ID:{track.track_id} {track.confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 绘制关键点
        if track.landmarks is not None:
            for pt in track.landmarks:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, color, -1)

    return frame


def test_on_video(video_path: str, max_frames: int = 300, save_output: bool = True):
    """在视频上测试跟踪流水线"""
    print(f"\n测试视频: {video_path}")
    print("=" * 50)

    # 初始化检测器和跟踪器
    detector = RetinaFaceDetector(conf_threshold=0.5)
    tracker = ByteTracker(
        high_thresh=0.6,
        low_thresh=0.1,
        match_thresh=0.8,
        max_age=30,
        min_hits=3
    )

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频信息: {width}x{height} @ {fps:.1f}fps, 共 {total_frames} 帧")
    print(f"测试帧数: {min(max_frames, total_frames)}")

    # 输出视频
    output_path = None
    writer = None
    if save_output:
        output_path = str(Path(video_path).parent / f"tracked_{Path(video_path).stem}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 统计信息
    stats = {
        'frame_count': 0,
        'total_detections': 0,
        'total_tracks': 0,
        'unique_ids': set(),
        'detection_times': [],
        'tracking_times': [],
    }

    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # 检测
        t0 = time.time()
        detections = detector.detect(frame)
        t1 = time.time()

        # 跟踪
        tracks = tracker.update(detections)
        t2 = time.time()

        # 更新统计
        stats['frame_count'] += 1
        stats['total_detections'] += len(detections)
        stats['total_tracks'] += len(tracks)
        stats['detection_times'].append(t1 - t0)
        stats['tracking_times'].append(t2 - t1)

        for track in tracks:
            stats['unique_ids'].add(track.track_id)

        # 绘制结果
        vis_frame = draw_tracks(frame.copy(), tracks)

        # 显示帧信息
        info = f"Frame: {frame_idx} | Dets: {len(detections)} | Tracks: {len(tracks)}"
        cv2.putText(vis_frame, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if writer:
            writer.write(vis_frame)

        frame_idx += 1

        # 每100帧打印进度
        if frame_idx % 100 == 0:
            print(f"  已处理 {frame_idx} 帧...")

    cap.release()
    if writer:
        writer.release()

    # 打印统计结果
    print_stats(stats, output_path)

    return stats


def print_stats(stats: dict, output_path: str = None):
    """打印统计结果"""
    print("\n" + "=" * 50)
    print("统计结果:")
    print(f"  处理帧数: {stats['frame_count']}")
    print(f"  总检测数: {stats['total_detections']}")
    print(f"  平均每帧检测: {stats['total_detections'] / max(1, stats['frame_count']):.1f}")
    print(f"  唯一轨迹ID数: {len(stats['unique_ids'])}")

    if stats['unique_ids']:
        print(f"  轨迹ID范围: {min(stats['unique_ids'])} - {max(stats['unique_ids'])}")

    avg_det_time = np.mean(stats['detection_times']) * 1000
    avg_trk_time = np.mean(stats['tracking_times']) * 1000
    total_time = avg_det_time + avg_trk_time

    print(f"\n性能统计:")
    print(f"  平均检测耗时: {avg_det_time:.1f} ms")
    print(f"  平均跟踪耗时: {avg_trk_time:.1f} ms")
    print(f"  总处理耗时: {total_time:.1f} ms/帧")
    print(f"  理论FPS: {1000 / total_time:.1f}")

    if output_path:
        print(f"\n输出视频: {output_path}")


def main():
    """主函数"""
    # 查找测试视频
    data_dir = Path(__file__).parent.parent / "data" / "raw_videos"

    if not data_dir.exists():
        print(f"数据目录不存在: {data_dir}")
        return

    videos = list(data_dir.glob("*.mp4")) + list(data_dir.glob("*.MP4"))

    if not videos:
        print("未找到视频文件")
        return

    print(f"找到 {len(videos)} 个视频")

    # 使用第一个测试集视频进行测试
    test_videos = ["12.28.mp4", "DJI_20250628142030_0040_D.MP4"]

    for video_name in test_videos:
        video_path = data_dir / video_name
        if video_path.exists():
            test_on_video(str(video_path), max_frames=300, save_output=True)
            break
    else:
        # 如果测试集视频不存在，使用第一个视频
        print(f"使用第一个视频进行测试: {videos[0].name}")
        test_on_video(str(videos[0]), max_frames=300, save_output=True)


if __name__ == "__main__":
    main()
