"""
CVAT 标注导出转换工具
将 CVAT 导出的标注转换为训练数据格式
"""
import xml.etree.ElementTree as ET
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict


def parse_cvat_xml(xml_path: str) -> List[Dict]:
    """解析 CVAT XML 格式的标注

    支持 CVAT for video 1.1 格式
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = []

    # 获取视频信息
    for task in root.findall('.//task'):
        video_name = task.find('name').text if task.find('name') is not None else 'unknown'

    # 解析 track 标注（跟踪标注）
    for track in root.findall('.//track'):
        track_id = int(track.get('id', 0))
        label = track.get('label', 'unknown')

        # 获取 track 的时间范围
        boxes = track.findall('box')
        if boxes:
            frames = [int(box.get('frame')) for box in boxes]
            start_frame = min(frames)
            end_frame = max(frames)

            # 检查是否有属性标注
            for box in boxes:
                for attr in box.findall('attribute'):
                    attr_name = attr.get('name')
                    attr_value = attr.text

            annotations.append({
                'video_name': video_name,
                'track_id': track_id,
                'label_name': label,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'frame_count': len(frames)
            })

    # 解析 tag 标注（视频片段标注）
    for image in root.findall('.//image'):
        frame_id = int(image.get('id', 0))
        video_name = image.get('name', 'unknown')

        for tag in image.findall('tag'):
            label = tag.get('label', 'unknown')
            annotations.append({
                'video_name': video_name,
                'track_id': 0,
                'label_name': label,
                'start_frame': frame_id,
                'end_frame': frame_id,
                'frame_count': 1
            })

    return annotations


def parse_cvat_coco(json_path: str) -> List[Dict]:
    """解析 CVAT COCO 格式的标注"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 构建类别映射
    categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}

    # 构建图像映射
    images = {img['id']: img for img in data.get('images', [])}

    annotations = []
    for ann in data.get('annotations', []):
        image_id = ann['image_id']
        image_info = images.get(image_id, {})

        annotations.append({
            'video_name': image_info.get('file_name', 'unknown'),
            'track_id': ann.get('track_id', 0),
            'label_name': categories.get(ann['category_id'], 'unknown'),
            'frame_id': image_info.get('frame_id', 0),
            'bbox': ann.get('bbox', [])
        })

    return annotations


def convert_to_training_format(
    cvat_annotations: List[Dict],
    pose_results_dir: str,
    fps: float = 50.0,
    output_path: str = None
) -> pd.DataFrame:
    """将 CVAT 标注转换为训练数据格式

    Args:
        cvat_annotations: CVAT 解析后的标注
        pose_results_dir: pose 结果目录
        fps: 视频帧率
        output_path: 输出路径
    """
    import numpy as np

    pose_dir = Path(pose_results_dir)
    training_data = []

    # 标签映射
    label_map = {
        'normal': 0,
        'suspicious': 1,
        '正常': 0,
        '可疑': 1
    }

    for ann in cvat_annotations:
        video_name = Path(ann['video_name']).stem
        label_name = ann['label_name'].lower()
        label = label_map.get(label_name, -1)

        if label == -1:
            print(f"警告: 未知标签 '{label_name}'")
            continue

        # 计算时间范围
        start_time = ann['start_frame'] / fps
        end_time = ann['end_frame'] / fps

        # 加载 pose 数据
        pose_file = pose_dir / f"{video_name}.csv"
        if not pose_file.exists():
            # 尝试匹配部分名称
            matches = list(pose_dir.glob(f"*{video_name}*.csv"))
            if matches:
                pose_file = matches[0]
            else:
                print(f"警告: 未找到 pose 文件 {video_name}")
                continue

        pose_df = pd.read_csv(pose_file)

        # 获取时间窗口内的数据
        mask = (pose_df['time_sec'] >= start_time) & \
               (pose_df['time_sec'] <= end_time)
        window_df = pose_df[mask]

        if len(window_df) < 3:
            print(f"警告: 数据不足 {video_name} [{start_time:.1f}s-{end_time:.1f}s]")
            continue

        # 计算特征
        yaws = window_df['yaw'].values
        pitches = window_df['pitch'].values if 'pitch' in window_df else np.zeros_like(yaws)
        rolls = window_df['roll'].values if 'roll' in window_df else np.zeros_like(yaws)

        training_data.append({
            'video_name': video_name,
            'track_id': ann.get('track_id', 0),
            'start_time': round(start_time, 2),
            'end_time': round(end_time, 2),
            'label': label,
            'yaw_mean': round(np.mean(yaws), 2),
            'yaw_std': round(np.std(yaws), 2),
            'yaw_range': round(np.max(yaws) - np.min(yaws), 2),
            'pitch_mean': round(np.mean(pitches), 2),
            'pitch_std': round(np.std(pitches), 2),
            'roll_mean': round(np.mean(rolls), 2),
            'roll_std': round(np.std(rolls), 2),
            'sample_count': len(window_df),
            'source': 'cvat'
        })

    result_df = pd.DataFrame(training_data)

    if output_path and len(result_df) > 0:
        result_df.to_csv(output_path, index=False)
        print(f"\n已保存训练数据: {output_path}")
        print(f"总样本数: {len(result_df)}")
        print(f"正常样本: {(result_df['label']==0).sum()}")
        print(f"可疑样本: {(result_df['label']==1).sum()}")

    return result_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CVAT 标注转换工具')
    parser.add_argument('input', help='CVAT 导出文件 (XML 或 JSON)')
    parser.add_argument('--pose-dir', default='data/pose_results')
    parser.add_argument('--fps', type=float, default=50.0, help='视频帧率')
    parser.add_argument('--output', default='data/cvat_training_labels.csv')

    args = parser.parse_args()

    input_path = Path(args.input)

    # 根据文件类型解析
    if input_path.suffix == '.xml':
        annotations = parse_cvat_xml(str(input_path))
    elif input_path.suffix == '.json':
        annotations = parse_cvat_coco(str(input_path))
    else:
        print(f"不支持的文件格式: {input_path.suffix}")
        exit(1)

    print(f"解析到 {len(annotations)} 条标注")

    # 转换格式
    convert_to_training_format(
        annotations,
        args.pose_dir,
        fps=args.fps,
        output_path=args.output
    )
