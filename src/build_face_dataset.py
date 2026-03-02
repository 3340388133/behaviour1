"""从视频构建人脸数据集"""
import csv
import random
from pathlib import Path

import cv2
import insightface
from insightface.app import FaceAnalysis


def extract_faces_from_video(video_path: Path, faces_dir: Path, annotations_dir: Path, detector):
    """从单个视频提取人脸"""
    video_name = video_path.stem
    output_dir = faces_dir / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    annotations = []
    face_count = 0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        time_sec = frame_id / fps if fps > 0 else 0

        # InsightFace 检测
        faces = detector.get(frame)

        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]

            # 边界检查
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                face_img = frame[y1:y2, x1:x2]
                face_img = cv2.resize(face_img, (224, 224))

                face_count += 1
                face_filename = f"face_{face_count:06d}.jpg"
                face_path = output_dir / face_filename
                cv2.imwrite(str(face_path), face_img)

                annotations.append({
                    'frame_id': frame_id,
                    'time_sec': round(time_sec, 3),
                    'face_path': str(face_path.relative_to(faces_dir.parent))
                })

        frame_id += 1

    cap.release()

    # 保存 CSV
    if annotations:
        annotations_dir.mkdir(parents=True, exist_ok=True)
        csv_path = annotations_dir / f"{video_name}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['frame_id', 'time_sec', 'face_path'])
            writer.writeheader()
            writer.writerows(annotations)

    return video_name, len(annotations)


def split_dataset(video_names: list, splits_dir: Path, train_ratio: float = 0.8, seed: int = 42):
    """按视频级别划分数据集"""
    random.seed(seed)
    random.shuffle(video_names)

    split_idx = int(len(video_names) * train_ratio)
    train_videos = video_names[:split_idx]
    test_videos = video_names[split_idx:]

    splits_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_dir / 'train.txt', 'w') as f:
        f.write('\n'.join(train_videos))

    with open(splits_dir / 'test.txt', 'w') as f:
        f.write('\n'.join(test_videos))

    return train_videos, test_videos


def main():
    data_dir = Path(__file__).parent.parent / 'data'
    videos_dir = data_dir / 'raw_videos'
    faces_dir = data_dir / 'faces'
    annotations_dir = data_dir / 'annotations'
    splits_dir = data_dir / 'splits'

    # 初始化人脸检测器
    detector = FaceAnalysis(allowed_modules=['detection'])
    detector.prepare(ctx_id=-1, det_size=(640, 640))

    # 匹配 .mp4 和 .MP4
    video_files = list(videos_dir.glob('*.mp4')) + list(videos_dir.glob('*.MP4'))
    video_files = sorted(video_files)  # 按文件名排序
    print(f"找到 {len(video_files)} 个视频")
    for i, vf in enumerate(video_files, 1):
        print(f"  [{i}] {vf.name}")

    processed_videos = []
    for idx, video_path in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}] 处理: {video_path.name}")
        video_name, face_count = extract_faces_from_video(video_path, faces_dir, annotations_dir, detector)
        processed_videos.append(video_name)
        print(f"  提取 {face_count} 张人脸")

    # 数据划分
    train_videos, test_videos = split_dataset(processed_videos, splits_dir)
    print(f"\n数据划分完成:")
    print(f"  训练集: {len(train_videos)} 个视频")
    print(f"  测试集: {len(test_videos)} 个视频")


if __name__ == '__main__':
    main()
