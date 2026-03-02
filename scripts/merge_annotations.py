#!/usr/bin/env python3
"""
合并导出的标注结果到正式的标注文件
"""

import json
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"

BEHAVIOR_LABELS = {
    0: "normal",
    1: "glancing",
    2: "quick_turn",
    3: "prolonged_watch",
    4: "looking_down",
    5: "looking_up",
    -1: "uncertain",
}


def main():
    # 加载导出的标注
    export_file = DATA_DIR / "annotations" / "annotations_export (1).json"
    with open(export_file, 'r') as f:
        export_data = json.load(f)

    # 加载待标注样本的详细信息
    pending_file = DATA_DIR / "annotations" / "pending_annotations.json"
    with open(pending_file, 'r') as f:
        pending_data = json.load(f)

    # 创建 sample_id -> 详细信息的映射
    sample_info = {s['sample_id']: s for s in pending_data['samples']}

    # 加载或创建正式标注文件
    labels_file = DATA_DIR / "annotations" / "behavior_labels.json"
    if labels_file.exists():
        with open(labels_file, 'r') as f:
            labels_data = json.load(f)
    else:
        labels_data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "samples": []
        }

    # 已有的 sample_id 集合
    existing_ids = {s['sample_id'] for s in labels_data['samples']}

    # 合并新标注
    new_count = 0
    for sample_id, label in export_data['annotations'].items():
        if sample_id in existing_ids:
            continue

        info = sample_info.get(sample_id, {})
        sample = {
            "sample_id": sample_id,
            "video_id": info.get('video_id', sample_id.rsplit('_track', 1)[0]),
            "track_id": info.get('track_id', -1),
            "window_idx": info.get('window_idx', -1),
            "start_frame": info.get('start_frame', -1),
            "end_frame": info.get('end_frame', -1),
            "start_time": info.get('start_frame', 0) / 10 if info.get('start_frame') else -1,
            "end_time": info.get('end_frame', 0) / 10 if info.get('end_frame') else -1,
            "label": label,
            "label_name": BEHAVIOR_LABELS.get(label, "unknown"),
            "confidence": "manual",
            "annotated_at": datetime.now().isoformat()
        }
        labels_data['samples'].append(sample)
        new_count += 1

    # 更新元数据
    labels_data['updated_at'] = datetime.now().isoformat()
    labels_data['total_samples'] = len(labels_data['samples'])

    # 统计各类别数量
    label_counts = {}
    for s in labels_data['samples']:
        name = s['label_name']
        label_counts[name] = label_counts.get(name, 0) + 1
    labels_data['label_distribution'] = label_counts

    # 保存
    with open(labels_file, 'w') as f:
        json.dump(labels_data, f, ensure_ascii=False, indent=2)

    print(f"合并完成！")
    print(f"  新增标注: {new_count}")
    print(f"  总标注数: {labels_data['total_samples']}")
    print(f"\n标签分布:")
    for name, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")
    print(f"\n已保存到: {labels_file}")


if __name__ == "__main__":
    main()
