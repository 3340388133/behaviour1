"""
重新处理视频，添加 track_id
- 使用 ByteTrack 跟踪器为每个人分配唯一 ID
- 过滤相似帧，减少冗余数据
"""
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, 'src')

from face_detector import RetinaFaceDetector
from tracker import ByteTracker
from pose_estimator import HeadPoseEstimator


def compute_frame_hash(frame: np.ndarray, hash_size: int = 8) -> int:
    """计算帧的感知哈希 (pHash)"""
    # 缩小尺寸
    resized = cv2.resize(frame, (hash_size + 1, hash_size))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 计算差异
    diff = gray[:, 1:] > gray[:, :-1]

    # 转换为哈希值
    return sum([2 ** i for i, v in enumerate(diff.flatten()) if v])


def hamming_distance(hash1: int, hash2: int) -> int:
    """计算两个哈希值的汉明距离"""
    return bin(hash1 ^ hash2).count('1')


def is_frame_similar(frame: np.ndarray, last_frame: np.ndarray,
                     threshold: float = 0.95) -> bool:
    """检测两帧是否相似

    使用结构相似度 (简化版)
    """
    if last_frame is None:
        return False

    # 缩小尺寸加速计算
    size = (160, 90)
    f1 = cv2.resize(frame, size)
    f2 = cv2.resize(last_frame, size)

    # 转灰度
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY).astype(float)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY).astype(float)

    # 计算相关系数
    mean1, mean2 = g1.mean(), g2.mean()
    std1, std2 = g1.std(), g2.std()

    if std1 < 1 or std2 < 1:
        return True

    corr = ((g1 - mean1) * (g2 - mean2)).mean() / (std1 * std2)

    return corr > threshold


def process_video_with_tracking(
    video_path: str,
    output_csv: str,
    sample_fps: float = 10.0,
    conf_threshold: float = 0.5,
    similarity_threshold: float = 0.95
):
    """处理视频并添加 track_id，过滤相似帧"""

    video_path = Path(video_path)
    print(f"处理视频: {video_path.name}")

    # 初始化模块
    detector = RetinaFaceDetector(conf_threshold=conf_threshold)
    tracker = ByteTracker()
    pose_estimator = HeadPoseEstimator()

    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / sample_fps))

    print(f"视频 FPS: {video_fps}, 采样间隔: {frame_interval}")

    results = []
    frame_idx = 0
    last_frame = None
    skipped_frames = 0

    pbar = tqdm(total=total_frames, desc="处理中")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # 检查是否与上一帧相似
            if is_frame_similar(frame, last_frame, similarity_threshold):
                skipped_frames += 1
                frame_idx += 1
                pbar.update(1)
                continue

            last_frame = frame.copy()
            time_sec = frame_idx / video_fps

            # 人脸检测
            detections = detector.detect(frame)

            # 跟踪更新
            tracks = tracker.update(detections)

            # 对每个跟踪目标进行姿态估计
            for track in tracks:
                # 裁剪人脸
                from face_detector import FaceDetection
                face_det = FaceDetection(
                    bbox=track.bbox,
                    confidence=track.confidence,
                    landmarks=track.landmarks
                )
                face_img = detector.crop_face(frame, face_det)

                if face_img is None or face_img.size == 0:
                    continue

                # 姿态估计
                pose = pose_estimator.estimate(face_img)

                results.append({
                    'frame_id': frame_idx,
                    'time_sec': round(time_sec, 3),
                    'track_id': track.track_id,
                    'bbox_x1': int(track.bbox[0]),
                    'bbox_y1': int(track.bbox[1]),
                    'bbox_x2': int(track.bbox[2]),
                    'bbox_y2': int(track.bbox[3]),
                    'yaw': round(pose.yaw, 2),
                    'pitch': round(pose.pitch, 2),
                    'roll': round(pose.roll, 2),
                    'pose_confidence': pose.confidence
                })

        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    # 保存结果
    df = pd.DataFrame(results)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    # 统计
    n_tracks = df['track_id'].nunique() if len(df) > 0 else 0
    print(f"完成: {len(df)} 条记录, {n_tracks} 个跟踪目标")
    print(f"跳过相似帧: {skipped_frames}")

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='重新处理视频，添加 track_id')
    parser.add_argument('input', help='视频文件或目录')
    parser.add_argument('-o', '--output', default='data/pose_tracked')
    parser.add_argument('--fps', type=float, default=10.0, help='采样帧率')
    parser.add_argument('--sim', type=float, default=0.95, help='相似度阈值')
    parser.add_argument('--batch', action='store_true', help='批量处理目录')

    args = parser.parse_args()

    if args.batch:
        input_dir = Path(args.input)
        videos = list(input_dir.glob('*.MP4')) + list(input_dir.glob('*.mp4'))
        print(f"找到 {len(videos)} 个视频")

        for i, video in enumerate(videos, 1):
            print(f"\n[{i}/{len(videos)}] ", end="")
            output_csv = f"{args.output}/{video.stem}.csv"
            if Path(output_csv).exists():
                print(f"跳过（已存在）: {video.name}")
                continue
            process_video_with_tracking(
                str(video), output_csv, args.fps,
                similarity_threshold=args.sim
            )
    else:
        output_csv = f"{args.output}/{Path(args.input).stem}.csv"
        process_video_with_tracking(
            args.input, output_csv, args.fps,
            similarity_threshold=args.sim
        )
