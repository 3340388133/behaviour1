"""
告警生成模块
将规则引擎输出转换为人类可读的告警理由
"""
import json
from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class Alert:
    """单条告警"""
    rule_name: str
    rule_name_cn: str
    time_start: float
    time_end: float
    score: float
    reason: str
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'rule': self.rule_name,
            'rule_cn': self.rule_name_cn,
            'time_range': {
                'start': round(self.time_start, 2),
                'end': round(self.time_end, 2),
                'duration': round(self.time_end - self.time_start, 2)
            },
            'score': round(self.score, 3),
            'reason': self.reason,
            'details': self.details
        }


@dataclass
class AlertReport:
    """告警报告"""
    video_name: str
    track_id: int
    total_windows: int
    suspicious_windows: int
    alerts: List[Alert]

    def to_dict(self) -> dict:
        return {
            'video': self.video_name,
            'track_id': self.track_id,
            'summary': {
                'total_windows': self.total_windows,
                'suspicious_windows': self.suspicious_windows,
                'suspicious_rate': f"{self.suspicious_windows/self.total_windows*100:.1f}%" if self.total_windows > 0 else "0%"
            },
            'alerts': [a.to_dict() for a in self.alerts]
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# 规则中文名映射
RULE_NAMES_CN = {
    'sustained_side_gaze': '持续侧向观察',
    'frequent_scanning': '频繁扫视',
    'high_variability': '高变异性',
    'wide_range_turn': '大范围转头'
}

# 阈值配置（与 rule_engine.py 保持一致）
THRESHOLDS = {
    'side_yaw': 35,
    'side_ratio': 0.7,
    'scan_switch': 2,
    'scan_speed': 15,
    'var_std': 50,
    'range_deg': 120
}


def generate_reason(rule_name: str, details: Dict, time_start: float, time_end: float) -> str:
    """根据规则和详情生成人类可读的告警理由"""
    duration = time_end - time_start

    if rule_name == 'sustained_side_gaze':
        ratio = details.get('side_ratio', 0)
        return (
            f"在 {time_start:.1f}s–{time_end:.1f}s 内（{duration:.1f}s），"
            f"{ratio*100:.0f}% 时间侧向观察（|yaw| > {THRESHOLDS['side_yaw']}°），"
            f"超过 {THRESHOLDS['side_ratio']*100:.0f}% 阈值"
        )

    elif rule_name == 'frequent_scanning':
        switch = details.get('switch_count', 0)
        speed = details.get('speed', 0)
        return (
            f"在 {time_start:.1f}s–{time_end:.1f}s 内（{duration:.1f}s），"
            f"头部方向切换 {switch} 次，平均转动速度 {speed:.1f}°/s，"
            f"超过阈值（切换≥{THRESHOLDS['scan_switch']}次 且 速度>{THRESHOLDS['scan_speed']}°/s）"
        )

    elif rule_name == 'high_variability':
        yaw_std = details.get('yaw_std', 0)
        return (
            f"在 {time_start:.1f}s–{time_end:.1f}s 内（{duration:.1f}s），"
            f"yaw 角度标准差为 {yaw_std:.1f}°，"
            f"超过 {THRESHOLDS['var_std']}° 阈值，头部朝向变化剧烈"
        )

    elif rule_name == 'wide_range_turn':
        yaw_range = details.get('yaw_range', 0)
        return (
            f"在 {time_start:.1f}s–{time_end:.1f}s 内（{duration:.1f}s），"
            f"yaw 角度范围达 {yaw_range:.1f}°，"
            f"超过 {THRESHOLDS['range_deg']}° 阈值，存在大范围转头行为"
        )

    return f"在 {time_start:.1f}s–{time_end:.1f}s 内触发规则 {rule_name}"


def generate_alerts(eval_results: List[Dict], video_name: str = "", track_id: int = 0) -> AlertReport:
    """
    从规则评估结果生成告警报告

    Args:
        eval_results: 规则评估结果列表（来自 visualize_yaw.extract_features_and_evaluate）
        video_name: 视频名称
        track_id: 轨迹 ID

    Returns:
        AlertReport 告警报告
    """
    alerts = []
    suspicious_count = 0

    for res in eval_results:
        if res.get('is_suspicious'):
            suspicious_count += 1

        # 为每条触发的规则生成告警
        for rule_info in res.get('rules', []):
            if rule_info.get('triggered'):
                rule_name = rule_info['rule_name']
                alert = Alert(
                    rule_name=rule_name,
                    rule_name_cn=RULE_NAMES_CN.get(rule_name, rule_name),
                    time_start=res['window_start'],
                    time_end=res['window_end'],
                    score=rule_info['score'],
                    reason=generate_reason(
                        rule_name,
                        rule_info.get('details', {}),
                        res['window_start'],
                        res['window_end']
                    ),
                    details=rule_info.get('details', {})
                )
                alerts.append(alert)

    return AlertReport(
        video_name=video_name,
        track_id=track_id,
        total_windows=len(eval_results),
        suspicious_windows=suspicious_count,
        alerts=alerts
    )


def main():
    """测试告警生成"""
    import argparse
    from pathlib import Path
    import pandas as pd
    from temporal_features import TemporalFeatureExtractor
    from rule_engine import RuleEngine
    import numpy as np

    parser = argparse.ArgumentParser(description='生成告警报告')
    parser.add_argument('csv_path', help='姿态 CSV 文件路径')
    parser.add_argument('--track-id', type=int, default=0, help='轨迹 ID')
    parser.add_argument('--output', help='输出 JSON 文件路径')
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    video_name = csv_path.stem

    # 加载数据
    df = pd.read_csv(csv_path)
    df['track_id'] = df.groupby('frame_id').cumcount()
    df = df[df['track_id'] == args.track_id].sort_values('time_sec')

    # 提取特征并评估
    extractor = TemporalFeatureExtractor()
    engine = RuleEngine()

    times = df['time_sec'].values
    yaws = df['yaw'].values
    features = extractor.extract_from_track(times, yaws, args.track_id)

    eval_results = []
    for feat in features:
        feat_dict = feat.to_dict()
        mask = (times >= feat.window_start) & (times < feat.window_end)
        feat_dict['yaws'] = yaws[mask]

        eval_result = engine.evaluate(feat_dict)
        eval_results.append({
            'window_start': feat.window_start,
            'window_end': feat.window_end,
            'is_suspicious': eval_result.is_suspicious,
            'weighted_score': eval_result.weighted_score,
            'rules': eval_result.to_dict()['rules']
        })

    # 生成告警报告
    report = generate_alerts(eval_results, video_name, args.track_id)

    # 输出
    print(report.to_json())

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report.to_json())
        print(f"\n告警报告已保存: {args.output}")


if __name__ == '__main__':
    main()
