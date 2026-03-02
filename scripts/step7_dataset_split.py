#!/usr/bin/env python3
"""
Step 7: 数据集划分

将标注好的数据划分为训练集、验证集和测试集

划分策略:
    1. 按视频划分 (推荐): 确保同一视频的数据不会同时出现在训练和测试集
    2. 按轨迹划分: 随机打乱所有轨迹
    3. 分层采样: 确保各类别在各集合中比例一致

输出:
    - dataset_root/splits/train.json
    - dataset_root/splits/val.json
    - dataset_root/splits/test.json
    - dataset_root/splits/split_config.json
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from collections import defaultdict

# ============================================================================
# 配置
# ============================================================================

DATASET_ROOT = Path(__file__).parent.parent / "dataset_root"
BEHAVIOR_DIR = DATASET_ROOT / "annotations" / "behavior"
POSE_DIR = DATASET_ROOT / "features" / "pose"
TEMPORAL_DIR = DATASET_ROOT / "features" / "temporal"
SPLITS_DIR = DATASET_ROOT / "splits"

# 划分比例
SPLIT_RATIOS = {
    'train': 0.7,
    'val': 0.15,
    'test': 0.15
}

# 随机种子
RANDOM_SEED = 42


# ============================================================================
# 数据集划分器
# ============================================================================

class DatasetSplitter:
    """数据集划分器"""

    def __init__(self, split_ratios: Dict = None, random_seed: int = 42):
        self.split_ratios = split_ratios or SPLIT_RATIOS
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

    def split_by_video(self, video_ids: List[str]) -> Dict[str, List[str]]:
        """
        按视频划分数据集
        确保同一视频的所有数据都在同一个集合中
        """
        n_videos = len(video_ids)
        n_train = int(n_videos * self.split_ratios['train'])
        n_val = int(n_videos * self.split_ratios['val'])

        # 随机打乱
        shuffled = video_ids.copy()
        random.shuffle(shuffled)

        return {
            'train': shuffled[:n_train],
            'val': shuffled[n_train:n_train + n_val],
            'test': shuffled[n_train + n_val:]
        }

    def split_by_track(self, all_tracks: List[Dict]) -> Dict[str, List[Dict]]:
        """
        按轨迹划分数据集
        随机打乱所有轨迹
        """
        n_tracks = len(all_tracks)
        n_train = int(n_tracks * self.split_ratios['train'])
        n_val = int(n_tracks * self.split_ratios['val'])

        # 随机打乱
        shuffled = all_tracks.copy()
        random.shuffle(shuffled)

        return {
            'train': shuffled[:n_train],
            'val': shuffled[n_train:n_train + n_val],
            'test': shuffled[n_train + n_val:]
        }

    def split_stratified(self, all_tracks: List[Dict], label_key: str = 'primary_label') -> Dict[str, List[Dict]]:
        """
        分层采样划分
        确保各类别在各集合中比例一致
        """
        # 按标签分组
        label_groups = defaultdict(list)
        for track in all_tracks:
            label = track.get(label_key, 'unknown')
            label_groups[label].append(track)

        splits = {'train': [], 'val': [], 'test': []}

        # 对每个类别分别划分
        for label, tracks in label_groups.items():
            n_tracks = len(tracks)
            n_train = max(1, int(n_tracks * self.split_ratios['train']))
            n_val = max(0, int(n_tracks * self.split_ratios['val']))

            # 确保每个集合至少有一些样本
            if n_tracks < 3:
                # 样本太少，全部放入训练集
                splits['train'].extend(tracks)
                continue

            random.shuffle(tracks)
            splits['train'].extend(tracks[:n_train])
            splits['val'].extend(tracks[n_train:n_train + n_val])
            splits['test'].extend(tracks[n_train + n_val:])

        # 打乱每个集合
        for key in splits:
            random.shuffle(splits[key])

        return splits


# ============================================================================
# 数据加载
# ============================================================================

def load_all_behaviors() -> Tuple[List[str], List[Dict]]:
    """加载所有行为标注数据"""
    video_ids = []
    all_tracks = []

    for video_dir in sorted(BEHAVIOR_DIR.iterdir()):
        if not video_dir.is_dir():
            continue

        behavior_file = video_dir / "behavior.json"
        if not behavior_file.exists():
            continue

        video_id = video_dir.name
        video_ids.append(video_id)

        with open(behavior_file, 'r') as f:
            data = json.load(f)

        for behavior in data.get('behaviors', []):
            behavior['video_id'] = video_id
            all_tracks.append(behavior)

    return video_ids, all_tracks


def get_track_features(video_id: str, track_id: int) -> Dict:
    """获取轨迹的完整特征路径"""
    return {
        'pose_file': str(POSE_DIR / video_id / "pose_opencv.json"),
        'temporal_file': str(TEMPORAL_DIR / video_id / "temporal_features.json"),
        'behavior_file': str(BEHAVIOR_DIR / video_id / "behavior.json")
    }


# ============================================================================
# 主处理流程
# ============================================================================

def create_split_files(splits: Dict[str, List], split_type: str, video_ids: List[str]):
    """创建划分文件"""

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # 统计信息
    split_stats = {}

    for split_name, items in splits.items():
        # 构建详细的划分数据
        split_data = {
            'split_name': split_name,
            'created_at': datetime.now().isoformat(),
            'split_type': split_type,
            'total_samples': len(items),
        }

        if split_type == 'by_video':
            # 按视频划分
            split_data['video_ids'] = items
            split_data['num_videos'] = len(items)

            # 收集该划分下的所有轨迹
            tracks = []
            label_counts = defaultdict(int)

            for video_id in items:
                behavior_file = BEHAVIOR_DIR / video_id / "behavior.json"
                if behavior_file.exists():
                    with open(behavior_file, 'r') as f:
                        data = json.load(f)
                    for b in data.get('behaviors', []):
                        tracks.append({
                            'video_id': video_id,
                            'track_id': b['track_id'],
                            'label': b['primary_label'],
                            'confidence': b['confidence']
                        })
                        label_counts[b['primary_label']] += 1

            split_data['tracks'] = tracks
            split_data['num_tracks'] = len(tracks)
            split_data['label_distribution'] = dict(label_counts)

        else:
            # 按轨迹划分
            split_data['tracks'] = items
            split_data['num_tracks'] = len(items)

            # 统计标签分布
            label_counts = defaultdict(int)
            for track in items:
                label_counts[track.get('primary_label', 'unknown')] += 1
            split_data['label_distribution'] = dict(label_counts)

            # 收集涉及的视频
            videos = list(set(t.get('video_id', '') for t in items))
            split_data['video_ids'] = videos
            split_data['num_videos'] = len(videos)

        # 保存划分文件
        output_file = SPLITS_DIR / f"{split_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)

        split_stats[split_name] = {
            'num_videos': split_data.get('num_videos', 0),
            'num_tracks': split_data.get('num_tracks', 0),
            'label_distribution': split_data.get('label_distribution', {})
        }

        print(f"  {split_name}: {split_data.get('num_tracks', 0)} 轨迹, {split_data.get('num_videos', 0)} 视频")

    return split_stats


def main():
    """主函数"""
    print("=" * 60)
    print("Step 7: 数据集划分")
    print("=" * 60)

    # 加载数据
    print("\n[1] 加载行为标注数据...")
    video_ids, all_tracks = load_all_behaviors()

    print(f"  - 视频数: {len(video_ids)}")
    print(f"  - 轨迹数: {len(all_tracks)}")

    if not video_ids:
        print("\n错误: 未找到行为标注数据，请先运行 step6_behavior_annotation.py")
        return

    # 统计标签分布
    print("\n[2] 标签分布:")
    label_counts = defaultdict(int)
    for track in all_tracks:
        label_counts[track.get('primary_label', 'unknown')] += 1

    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = count / len(all_tracks) * 100
        print(f"    {label}: {count} ({pct:.1f}%)")

    # 初始化划分器
    splitter = DatasetSplitter(SPLIT_RATIOS, RANDOM_SEED)

    # 执行划分
    print(f"\n[3] 执行数据集划分 (比例: train={SPLIT_RATIOS['train']}, val={SPLIT_RATIOS['val']}, test={SPLIT_RATIOS['test']})")

    # 方案1: 按视频划分 (推荐)
    print("\n  方案: 按视频划分")
    video_splits = splitter.split_by_video(video_ids)

    print(f"    训练集: {len(video_splits['train'])} 视频")
    print(f"    验证集: {len(video_splits['val'])} 视频")
    print(f"    测试集: {len(video_splits['test'])} 视频")

    # 创建划分文件
    print("\n[4] 生成划分文件...")
    split_stats = create_split_files(video_splits, 'by_video', video_ids)

    # 生成配置文件
    config = {
        'created_at': datetime.now().isoformat(),
        'split_type': 'by_video',
        'split_ratios': SPLIT_RATIOS,
        'random_seed': RANDOM_SEED,
        'total_videos': len(video_ids),
        'total_tracks': len(all_tracks),
        'label_distribution': dict(label_counts),
        'splits': split_stats
    }

    config_file = SPLITS_DIR / "split_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("划分完成!")
    print(f"\n生成文件:")
    print(f"  - {SPLITS_DIR / 'train.json'}")
    print(f"  - {SPLITS_DIR / 'val.json'}")
    print(f"  - {SPLITS_DIR / 'test.json'}")
    print(f"  - {config_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
