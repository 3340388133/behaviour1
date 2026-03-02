"""
行为标注重建器 - 基于规则引擎自动生成行为标签
用于生成时序模型训练所需的伪标签数据
"""
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json

from temporal_features import TemporalFeatureExtractor
from rule_engine import RuleEngine


@dataclass
class BehaviorLabel:
    """单个行为标注"""
    video_name: str
    track_id: int
    start_time: float
    end_time: float
    label: int  # 0: normal, 1: suspicious
    score: float
    triggered_rules: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'video_name': self.video_name,
            'track_id': self.track_id,
            'start_time': round(self.start_time, 3),
            'end_time': round(self.end_time, 3),
            'label': self.label,
            'score': round(self.score, 4),
            'triggered_rules': '|'.join(self.triggered_rules) if self.triggered_rules else ''
        }


class BehaviorLabelGenerator:
    """行为标注重建器

    基于规则引擎输出自动生成行为标签，用于：
    1. 规则评估验证
    2. 时序模型训练（伪标签）
    """

    def __init__(
        self,
        window_size: float = 2.0,
        step_size: float = 0.5,
        threshold: float = 0.3,
        min_samples: int = 5
    ):
        """
        Args:
            window_size: 滑动窗口大小（秒）
            step_size: 滑动窗口步长（秒）
            threshold: 可疑判定阈值（score >= threshold 为 suspicious）
            min_samples: 窗口内最小样本数
        """
        self.window_size = window_size
        self.step_size = step_size
        self.threshold = threshold
        self.min_samples = min_samples

        self.feature_extractor = TemporalFeatureExtractor(
            window_size=window_size,
            step_size=step_size,
            min_samples=min_samples
        )
        self.rule_engine = RuleEngine()

    def generate_from_pose_csv(
        self,
        pose_csv_path: str,
        video_name: str = None
    ) -> List[BehaviorLabel]:
        """从 pose CSV 生成行为标注

        Args:
            pose_csv_path: pose CSV 文件路径
            video_name: 视频名称（默认从文件名提取）

        Returns:
            BehaviorLabel 列表
        """
        pose_csv_path = Path(pose_csv_path)
        if video_name is None:
            video_name = pose_csv_path.stem

        # 加载 pose 数据
        df = pd.read_csv(pose_csv_path)

        # 检查必要列
        required_cols = ['frame_id', 'time_sec', 'yaw']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # 检查可选列
        has_pitch = 'pitch' in df.columns
        has_roll = 'roll' in df.columns

        # 按帧分配 track_id（同一帧内的多个人脸）
        df = self._assign_track_ids(df)

        # 对每个 track 生成标注
        labels = []
        for track_id in df['track_id'].unique():
            track_labels = self._generate_track_labels(
                df, track_id, video_name, has_pitch, has_roll
            )
            labels.extend(track_labels)

        return labels

    def _assign_track_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """为每帧内的人脸分配 track_id

        简化策略：按帧内顺序分配，假设同一位置的人脸 ID 一致
        """
        df = df.copy()
        df['track_id'] = df.groupby('frame_id').cumcount()
        return df

    def _generate_track_labels(
        self,
        df: pd.DataFrame,
        track_id: int,
        video_name: str,
        has_pitch: bool = False,
        has_roll: bool = False
    ) -> List[BehaviorLabel]:
        """为单个 track 生成行为标注"""
        track_df = df[df['track_id'] == track_id].sort_values('time_sec')

        if len(track_df) < self.min_samples:
            return []

        times = track_df['time_sec'].values
        yaws = track_df['yaw'].values
        pitches = track_df['pitch'].values if has_pitch else None
        rolls = track_df['roll'].values if has_roll else None

        # 提取时序特征
        features = self.feature_extractor.extract_from_track(
            times, yaws, track_id, pitches, rolls
        )

        labels = []
        for feat in features:
            # 构建特征字典
            feat_dict = feat.to_dict()
            mask = (times >= feat.window_start) & (times < feat.window_end)
            feat_dict['yaws'] = yaws[mask]

            # 规则评估
            eval_result = self.rule_engine.evaluate(feat_dict)
            score = eval_result.weighted_score

            # 获取触发的规则
            triggered = [
                r.rule_name for r in eval_result.rules if r.triggered
            ]

            # 生成标签
            label = BehaviorLabel(
                video_name=video_name,
                track_id=track_id,
                start_time=feat.window_start,
                end_time=feat.window_end,
                label=1 if score >= self.threshold else 0,
                score=score,
                triggered_rules=triggered
            )
            labels.append(label)

        return labels

    def generate_from_directory(
        self,
        pose_dir: str,
        output_path: str = None
    ) -> pd.DataFrame:
        """批量处理目录下所有 pose CSV

        Args:
            pose_dir: pose CSV 目录
            output_path: 输出 CSV 路径（可选）

        Returns:
            合并的标注 DataFrame
        """
        pose_dir = Path(pose_dir)
        csv_files = sorted(pose_dir.glob('*.csv'))

        if not csv_files:
            print(f"No CSV files found in {pose_dir}")
            return pd.DataFrame()

        all_labels = []
        for csv_file in csv_files:
            print(f"Processing: {csv_file.name}")
            try:
                labels = self.generate_from_pose_csv(str(csv_file))
                all_labels.extend(labels)
                print(f"  -> {len(labels)} windows generated")
            except Exception as e:
                print(f"  -> Error: {e}")

        # 转换为 DataFrame
        df = pd.DataFrame([l.to_dict() for l in all_labels])

        # 保存
        if output_path and len(df) > 0:
            df.to_csv(output_path, index=False)
            print(f"\nSaved to: {output_path}")

        return df

    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """生成标注统计报告"""
        if len(df) == 0:
            return {'error': 'Empty DataFrame'}

        stats = {
            'total_windows': len(df),
            'suspicious_windows': int(df['label'].sum()),
            'normal_windows': int((df['label'] == 0).sum()),
            'suspicious_rate': f"{df['label'].mean() * 100:.1f}%",
            'videos': df['video_name'].nunique(),
            'tracks': df.groupby('video_name')['track_id'].nunique().to_dict(),
            'score_stats': {
                'mean': round(df['score'].mean(), 4),
                'std': round(df['score'].std(), 4),
                'min': round(df['score'].min(), 4),
                'max': round(df['score'].max(), 4)
            }
        }

        # 规则触发统计
        rule_counts = {}
        for rules_str in df['triggered_rules'].dropna():
            if rules_str:
                for rule in rules_str.split('|'):
                    rule_counts[rule] = rule_counts.get(rule, 0) + 1
        stats['rule_triggers'] = rule_counts

        return stats


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description='行为标注重建器 - 基于规则引擎生成伪标签'
    )
    parser.add_argument(
        'input',
        help='输入路径（pose CSV 文件或目录）'
    )
    parser.add_argument(
        '-o', '--output',
        default='data/behavior_labels.csv',
        help='输出 CSV 路径（默认: data/behavior_labels.csv）'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=0.3,
        help='可疑判定阈值（默认: 0.3）'
    )
    parser.add_argument(
        '-w', '--window',
        type=float,
        default=2.0,
        help='滑动窗口大小（秒，默认: 2.0）'
    )
    parser.add_argument(
        '-s', '--step',
        type=float,
        default=0.5,
        help='滑动窗口步长（秒，默认: 0.5）'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='输出统计报告'
    )
    args = parser.parse_args()

    # 初始化生成器
    generator = BehaviorLabelGenerator(
        window_size=args.window,
        step_size=args.step,
        threshold=args.threshold
    )

    input_path = Path(args.input)

    # 处理输入
    if input_path.is_dir():
        df = generator.generate_from_directory(
            str(input_path),
            output_path=args.output
        )
    else:
        labels = generator.generate_from_pose_csv(str(input_path))
        df = pd.DataFrame([l.to_dict() for l in labels])
        df.to_csv(args.output, index=False)
        print(f"Saved to: {args.output}")

    # 输出统计
    if args.stats and len(df) > 0:
        stats = generator.get_statistics(df)
        print("\n" + "=" * 50)
        print("统计报告")
        print("=" * 50)
        print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
