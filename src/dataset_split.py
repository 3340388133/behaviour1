"""
数据集划分与泄漏防护模块

核心问题：信息泄漏导致模型性能虚高

泄漏类型：
1. 同一人的不同片段分到 train/test → 模型学习"谁"而非"行为"
2. 同一视频的不同帧分到 train/test → 模型学习背景/相机特征
3. 相似场景分到 train/test → 模型学习场景而非行为
"""
import numpy as np
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Optional
from pathlib import Path
from collections import defaultdict
import random


# ============================================================================
# 数据泄漏分析
# ============================================================================
"""
按 track/frame 划分的泄漏风险分析：

┌─────────────────────────────────────────────────────────────┐
│              按帧/窗口划分的泄漏问题                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  场景：同一人在视频中持续30秒                                 │
│  划分：随机将窗口分到 train (70%) / test (30%)               │
│                                                             │
│  问题：                                                      │
│  ┌─────────────────────────────────────────┐                │
│  │  Train: [0-3s] [6-9s] [12-15s] [18-21s] │ ← 同一人        │
│  │  Test:  [3-6s] [9-12s] [15-18s]         │ ← 同一人        │
│  └─────────────────────────────────────────┘                │
│                                                             │
│  结果：                                                      │
│  - 模型学习"这个人的外观/姿态习惯"                           │
│  - 测试时识别"是那个人"而非"是那个行为"                      │
│  - 性能虚高：测试集实际上是"见过的人"                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘

泄漏如何虚高性能：

1. 身份泄漏（同一人）
   - 每个人有独特的姿态习惯（baseline yaw/pitch）
   - 模型可能学习"人A习惯yaw=10°，人B习惯yaw=-5°"
   - 真实场景中遇到新人，模型失效

2. 场景泄漏（同一视频/相机）
   - 相机位置影响检测到的姿态绝对值
   - 背景特征可能与行为相关（如：某区域经常有人张望）
   - 新场景中模型失效

3. 时间泄漏（相邻帧）
   - 相邻帧高度相似（人脸外观、姿态连续）
   - 模型可能记忆序列模式而非学习行为特征

量化泄漏影响的方法：
- 对比实验：随机划分 vs 按人划分
- 如果随机划分性能明显高于按人划分 → 存在泄漏
- 典型表现：随机划分 AUC 0.95，按人划分 AUC 0.75
"""

LEAKAGE_RISKS = {
    "frame_split": {
        "risk_level": "high",
        "description": "按帧随机划分",
        "leakage_type": ["identity", "temporal", "scene"],
        "impact": "性能虚高 20-40%"
    },
    "window_split": {
        "risk_level": "high",
        "description": "按窗口随机划分",
        "leakage_type": ["identity", "scene"],
        "impact": "性能虚高 15-30%"
    },
    "track_split": {
        "risk_level": "medium",
        "description": "按 track 随机划分",
        "leakage_type": ["identity（同一人多track）", "scene"],
        "impact": "性能虚高 10-20%"
    },
    "person_split": {
        "risk_level": "low",
        "description": "按人划分（同一人所有数据在同一集）",
        "leakage_type": ["scene（如果场景与人相关）"],
        "impact": "基本无泄漏"
    },
    "video_split": {
        "risk_level": "very_low",
        "description": "按视频划分",
        "leakage_type": [],
        "impact": "无泄漏（推荐）"
    }
}


# ============================================================================
# 推荐的划分策略
# ============================================================================
"""
推荐划分粒度：按 Person 或 Video 划分

划分层级（从严格到宽松）：
1. 按视频划分（最严格）：同一视频的所有数据在同一集
2. 按人划分（推荐）：同一人的所有数据在同一集
3. 按场景划分：同一场景类型均衡分布

推荐策略：Person + Scene 分层划分
- 先按人分组
- 在保证同人同集的前提下，平衡场景分布

划分比例：
- Train: 70%
- Val: 15%
- Test: 15%
"""

@dataclass
class SplitConfig:
    """划分配置"""
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # 划分粒度
    split_by: str = "person"  # "person" / "video" / "scene"

    # 分层要求
    stratify_by: list = field(default_factory=lambda: ["scene", "behavior"])

    # 最小样本要求
    min_samples_per_split: int = 50
    min_persons_per_split: int = 3

    # 随机种子
    random_seed: int = 42

    def validate(self):
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 0.01
        assert self.split_by in ["person", "video", "scene", "track"]


# ============================================================================
# 数据集样本信息
# ============================================================================
@dataclass
class SampleInfo:
    """样本信息"""
    sample_id: str
    video_id: str
    track_id: int
    person_id: int

    # 场景属性
    scene: str           # "indoor" / "outdoor"
    camera_angle: str    # "front" / "side"
    occlusion: bool

    # 行为标签
    label: int
    label_name: str

    # 时间信息
    start_time: float
    end_time: float

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "video_id": self.video_id,
            "track_id": self.track_id,
            "person_id": self.person_id,
            "scene": self.scene,
            "camera_angle": self.camera_angle,
            "occlusion": self.occlusion,
            "label": self.label,
            "label_name": self.label_name,
            "start_time": self.start_time,
            "end_time": self.end_time
        }


@dataclass
class SplitResult:
    """划分结果"""
    split_name: str          # "train" / "val" / "test"
    sample_ids: list
    person_ids: list
    video_ids: list

    # 统计信息
    total_samples: int
    samples_per_label: dict
    samples_per_scene: dict
    samples_per_video: dict

    def to_dict(self) -> dict:
        return {
            "split_name": self.split_name,
            "sample_ids": self.sample_ids,
            "person_ids": self.person_ids,
            "video_ids": self.video_ids,
            "total_samples": self.total_samples,
            "samples_per_label": self.samples_per_label,
            "samples_per_scene": self.samples_per_scene,
            "samples_per_video": self.samples_per_video
        }


# ============================================================================
# 数据集划分器
# ============================================================================
class DatasetSplitter:
    """数据集划分器"""

    def __init__(self, config: SplitConfig = None):
        self.config = config or SplitConfig()
        self.config.validate()

    def split(self, samples: List[SampleInfo]) -> Dict[str, SplitResult]:
        """执行划分

        Args:
            samples: 样本信息列表

        Returns:
            {"train": SplitResult, "val": SplitResult, "test": SplitResult}
        """
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        if self.config.split_by == "person":
            return self._split_by_person(samples)
        elif self.config.split_by == "video":
            return self._split_by_video(samples)
        else:
            return self._split_by_person(samples)  # 默认按人

    def _split_by_person(self, samples: List[SampleInfo]) -> Dict[str, SplitResult]:
        """按人划分（推荐）"""
        # 按人分组
        person_samples = defaultdict(list)
        for s in samples:
            person_samples[s.person_id].append(s)

        person_ids = list(person_samples.keys())

        # 收集每个人的属性（用于分层）
        person_attrs = {}
        for pid in person_ids:
            p_samples = person_samples[pid]
            # 统计主要场景
            scenes = [s.scene for s in p_samples]
            main_scene = max(set(scenes), key=scenes.count)
            # 统计主要标签
            labels = [s.label for s in p_samples]
            main_label = max(set(labels), key=labels.count)

            person_attrs[pid] = {
                "scene": main_scene,
                "label": main_label,
                "sample_count": len(p_samples)
            }

        # 分层划分
        splits = self._stratified_split(
            person_ids,
            person_attrs,
            self.config.train_ratio,
            self.config.val_ratio,
            self.config.test_ratio
        )

        # 构建结果
        results = {}
        for split_name, pids in splits.items():
            split_samples = []
            for pid in pids:
                split_samples.extend(person_samples[pid])

            results[split_name] = self._build_split_result(
                split_name, split_samples, pids
            )

        return results

    def _split_by_video(self, samples: List[SampleInfo]) -> Dict[str, SplitResult]:
        """按视频划分"""
        # 按视频分组
        video_samples = defaultdict(list)
        for s in samples:
            video_samples[s.video_id].append(s)

        video_ids = list(video_samples.keys())

        # 收集每个视频的属性
        video_attrs = {}
        for vid in video_ids:
            v_samples = video_samples[vid]
            scenes = [s.scene for s in v_samples]
            main_scene = max(set(scenes), key=scenes.count)

            video_attrs[vid] = {
                "scene": main_scene,
                "sample_count": len(v_samples)
            }

        # 分层划分
        splits = self._stratified_split(
            video_ids,
            video_attrs,
            self.config.train_ratio,
            self.config.val_ratio,
            self.config.test_ratio
        )

        # 构建结果
        results = {}
        for split_name, vids in splits.items():
            split_samples = []
            pids = set()
            for vid in vids:
                for s in video_samples[vid]:
                    split_samples.append(s)
                    pids.add(s.person_id)

            results[split_name] = self._build_split_result(
                split_name, split_samples, list(pids), vids
            )

        return results

    def _stratified_split(
        self,
        ids: list,
        attrs: dict,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float
    ) -> Dict[str, list]:
        """分层划分"""
        # 按场景分组
        scene_groups = defaultdict(list)
        for id_ in ids:
            scene = attrs[id_].get("scene", "unknown")
            scene_groups[scene].append(id_)

        train_ids, val_ids, test_ids = [], [], []

        # 对每个场景分别划分
        for scene, group_ids in scene_groups.items():
            random.shuffle(group_ids)
            n = len(group_ids)

            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_ids.extend(group_ids[:n_train])
            val_ids.extend(group_ids[n_train:n_train+n_val])
            test_ids.extend(group_ids[n_train+n_val:])

        return {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids
        }

    def _build_split_result(
        self,
        split_name: str,
        samples: List[SampleInfo],
        person_ids: list,
        video_ids: list = None
    ) -> SplitResult:
        """构建划分结果"""
        sample_ids = [s.sample_id for s in samples]

        if video_ids is None:
            video_ids = list(set(s.video_id for s in samples))

        # 统计
        label_counts = defaultdict(int)
        scene_counts = defaultdict(int)
        video_counts = defaultdict(int)

        for s in samples:
            label_counts[s.label] += 1
            scene_counts[s.scene] += 1
            video_counts[s.video_id] += 1

        return SplitResult(
            split_name=split_name,
            sample_ids=sample_ids,
            person_ids=list(set(person_ids)),
            video_ids=video_ids,
            total_samples=len(samples),
            samples_per_label=dict(label_counts),
            samples_per_scene=dict(scene_counts),
            samples_per_video=dict(video_counts)
        )

    def validate_no_leakage(self, results: Dict[str, SplitResult]) -> dict:
        """验证无泄漏"""
        issues = []

        # 检查人员泄漏
        train_persons = set(results["train"].person_ids)
        val_persons = set(results["val"].person_ids)
        test_persons = set(results["test"].person_ids)

        train_val_overlap = train_persons & val_persons
        train_test_overlap = train_persons & test_persons
        val_test_overlap = val_persons & test_persons

        if train_val_overlap:
            issues.append(f"Person leakage train-val: {train_val_overlap}")
        if train_test_overlap:
            issues.append(f"Person leakage train-test: {train_test_overlap}")
        if val_test_overlap:
            issues.append(f"Person leakage val-test: {val_test_overlap}")

        # 检查视频泄漏（如果按人划分，同一视频可能在不同集）
        # 这在按人划分时是允许的，但需要记录

        return {
            "no_person_leakage": len(issues) == 0,
            "issues": issues,
            "train_persons": len(train_persons),
            "val_persons": len(val_persons),
            "test_persons": len(test_persons)
        }


# ============================================================================
# splits.json 生成规则
# ============================================================================
"""
splits.json 生成规则：

1. 划分粒度
   - 按 Person 划分：同一人的所有 track/window 在同一集
   - 确保 train/val/test 的人员完全不重叠

2. 分层要求
   - 场景均衡：indoor/outdoor 比例在各集相似
   - 标签均衡：各行为类别比例在各集相似
   - 遮挡均衡：有遮挡/无遮挡比例在各集相似

3. 最小样本要求
   - 每个集合至少 50 个样本
   - 每个集合至少 3 个不同的人

4. 文件格式
   {
     "version": "1.0",
     "split_config": { ... },
     "train": {
       "sample_ids": [...],
       "person_ids": [...],
       "statistics": { ... }
     },
     "val": { ... },
     "test": { ... },
     "leakage_check": { ... }
   }
"""

def generate_splits(
    samples: List[SampleInfo],
    output_dir: str,
    config: SplitConfig = None
) -> dict:
    """生成数据集划分文件

    Args:
        samples: 样本信息列表
        output_dir: 输出目录
        config: 划分配置

    Returns:
        划分结果
    """
    config = config or SplitConfig()
    splitter = DatasetSplitter(config)

    # 执行划分
    results = splitter.split(samples)

    # 验证无泄漏
    leakage_check = splitter.validate_no_leakage(results)

    # 构建输出
    output = {
        "version": "1.0",
        "split_config": {
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
            "split_by": config.split_by,
            "random_seed": config.random_seed
        },
        "leakage_check": leakage_check
    }

    for split_name, result in results.items():
        output[split_name] = result.to_dict()

    # 保存
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 保存完整的 splits.json
    with open(output_path / "splits.json", 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 分别保存各集的样本列表（方便使用）
    for split_name in ["train", "val", "test"]:
        split_file = output_path / f"{split_name}.json"
        with open(split_file, 'w', encoding='utf-8') as f:
            json.dump({
                "split": split_name,
                "sample_ids": results[split_name].sample_ids,
                "person_ids": results[split_name].person_ids,
                "statistics": {
                    "total_samples": results[split_name].total_samples,
                    "by_label": results[split_name].samples_per_label,
                    "by_scene": results[split_name].samples_per_scene
                }
            }, f, ensure_ascii=False, indent=2)

    print(f"Splits saved to {output_path}")
    print(f"  Train: {results['train'].total_samples} samples, {len(results['train'].person_ids)} persons")
    print(f"  Val: {results['val'].total_samples} samples, {len(results['val'].person_ids)} persons")
    print(f"  Test: {results['test'].total_samples} samples, {len(results['test'].person_ids)} persons")
    print(f"  Leakage check: {'PASS' if leakage_check['no_person_leakage'] else 'FAIL'}")

    return output


# ============================================================================
# 视频元数据解析（用于确定场景属性）
# ============================================================================
def parse_video_attributes(video_id: str) -> dict:
    """从视频文件名解析属性

    视频命名规则：
    - 正室内.mp4 → 正面、室内、无遮挡
    - 侧室外遮.MP4 → 侧面、室外、有遮挡
    """
    video_id_lower = video_id.lower()

    # 解析视角
    if "正" in video_id:
        angle = "front"
    elif "侧" in video_id:
        angle = "side"
    else:
        angle = "unknown"

    # 解析场景
    if "室内" in video_id or "地铁" in video_id:
        scene = "indoor"
    elif "室外" in video_id:
        scene = "outdoor"
    else:
        scene = "unknown"

    # 解析遮挡
    occlusion = "遮" in video_id

    return {
        "camera_angle": angle,
        "scene": scene,
        "occlusion": occlusion
    }


# ============================================================================
# 划分质量检查
# ============================================================================
def check_split_quality(results: Dict[str, SplitResult]) -> dict:
    """检查划分质量"""
    issues = []
    warnings = []

    # 检查样本量
    for split_name, result in results.items():
        if result.total_samples < 50:
            issues.append(f"{split_name}: 样本量过少 ({result.total_samples})")
        if len(result.person_ids) < 3:
            issues.append(f"{split_name}: 人数过少 ({len(result.person_ids)})")

    # 检查标签分布均衡性
    all_labels = set()
    for result in results.values():
        all_labels.update(result.samples_per_label.keys())

    for label in all_labels:
        counts = [
            results[split].samples_per_label.get(label, 0)
            for split in ["train", "val", "test"]
        ]
        total = sum(counts)
        if total > 0:
            ratios = [c / total for c in counts]
            # 检查比例是否接近 7:1.5:1.5
            expected = [0.7, 0.15, 0.15]
            for i, (r, e) in enumerate(zip(ratios, expected)):
                if abs(r - e) > 0.15:  # 允许15%偏差
                    warnings.append(
                        f"Label {label} 分布不均: {ratios}"
                    )
                    break

    # 检查场景分布
    all_scenes = set()
    for result in results.values():
        all_scenes.update(result.samples_per_scene.keys())

    for scene in all_scenes:
        counts = [
            results[split].samples_per_scene.get(scene, 0)
            for split in ["train", "val", "test"]
        ]
        total = sum(counts)
        if total > 0:
            ratios = [c / total for c in counts]
            expected = [0.7, 0.15, 0.15]
            for i, (r, e) in enumerate(zip(ratios, expected)):
                if abs(r - e) > 0.2:  # 允许20%偏差
                    warnings.append(
                        f"Scene {scene} 分布不均: {ratios}"
                    )
                    break

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings
    }
