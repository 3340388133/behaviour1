#!/usr/bin/env python3
"""
Step 8b: 参数敏感性分析
Parameter Sensitivity Analysis

分析关键参数对系统性能的影响：
  1. 时序平滑窗口大小 (smooth_window): 1,2,4,6,8,12,16,24,32
  2. 姿态门控阈值 (yaw_threshold): 20,25,30,35,40,45,50,55,60
  3. 投票阈值 (vote_threshold): 0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50

优化策略：模型只推理一次，缓存中间结果，所有 sweep 均通过后处理完成。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'recognition'))

import json
import argparse
import math
import torch
import numpy as np
from collections import deque, defaultdict
from datetime import datetime

from temporal_transformer import create_model

# 复用 step8 的类和函数
from step8_ablation_baseline import (
    AblationRecognizer, RuleDetector, NumpyEncoder,
    compute_track_label, compute_shannon_entropy,
    BEHAVIOR_CLASSES, VIDEO_NAMES, FRONTAL_VIDEOS, LATERAL_VIDEOS,
)


def load_full_system_results(project_root: Path) -> dict:
    """加载全系统在线推理结果 (与 step8 一致)"""
    results = {}
    stats_dir = project_root / 'data' / 'batch_inference_output'
    for vname in VIDEO_NAMES:
        stats_path = stats_dir / f'{vname}_inference_stats.json'
        if not stats_path.exists():
            continue
        with open(stats_path, 'r') as f:
            data = json.load(f)
        track_labels = {}
        for tid, label in data.get('track_behaviors', {}).items():
            track_labels[tid] = int(label)
        results[vname] = track_labels
    return results


def load_pose_data(project_root: Path) -> dict:
    """加载姿态数据"""
    pose_dir = project_root / 'data' / 'pose_output'
    all_pose_data = {}
    for vname in VIDEO_NAMES:
        pose_path = pose_dir / f'{vname}_poses.json'
        if not pose_path.exists():
            continue
        with open(pose_path, 'r') as f:
            data = json.load(f)
        all_pose_data[vname] = data
    return all_pose_data


# ==================== 带中间结果缓存的识别器 ====================

class InstrumentedRecognizer(AblationRecognizer):
    """继承 AblationRecognizer，额外缓存每帧中间结果（模型/规则/原始决策）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_data = defaultdict(list)  # track_id -> [frame_dict or None]

    def update(self, track_id: str, yaw: float, pitch: float, roll: float):
        """与父类逻辑完全一致，但同时记录中间结果。"""
        if track_id not in self.pose_buffers:
            self.pose_buffers[track_id] = deque(maxlen=self.seq_len * 2)
            self.pred_history[track_id] = deque(maxlen=self.smooth_window)

        self.pose_buffers[track_id].append([yaw, pitch, roll])
        buf_len = len(self.pose_buffers[track_id])

        if buf_len < 10:
            self.frame_data[track_id].append(None)
            return None, 0.0

        # 模型推理
        model_pred, model_conf = self._model_predict(track_id)

        # 规则检测
        if self.use_rules:
            rule_pred, rule_conf = self.rule_detector.check(
                list(self.pose_buffers[track_id]))
        else:
            rule_pred, rule_conf = None, 0.0

        # 姿态门控 (默认阈值)
        pose_gate_pred, pose_gate_conf = self._pose_gate(yaw, pitch)

        # 混合决策 (与父类完全一致)
        if pose_gate_pred is not None:
            pred, conf = pose_gate_pred, pose_gate_conf
        elif model_pred is not None and model_conf > 0.3:
            if rule_pred is not None and rule_pred > 0:
                if model_pred == 0 and model_conf > 0.90:
                    pred, conf = model_pred, model_conf
                else:
                    pred, conf = rule_pred, rule_conf
            else:
                pred, conf = model_pred, model_conf
        elif rule_pred is not None:
            pred, conf = rule_pred, rule_conf
        else:
            self.frame_data[track_id].append({
                'yaw': yaw, 'pitch': pitch,
                'm_pred': model_pred, 'm_conf': model_conf,
                'r_pred': rule_pred, 'r_conf': rule_conf,
                'raw_pred': None, 'raw_conf': 0.0,
            })
            return None, 0.0

        # 记录中间结果
        self.frame_data[track_id].append({
            'yaw': yaw, 'pitch': pitch,
            'm_pred': model_pred, 'm_conf': model_conf,
            'r_pred': rule_pred, 'r_conf': rule_conf,
            'raw_pred': pred, 'raw_conf': conf,
        })

        self.pred_history[track_id].append((pred, conf))
        return self._get_smoothed_pred(track_id)

    def reset(self):
        super().reset()
        self.frame_data.clear()


# ==================== 纯后处理：不同参数的 sweep ====================

def hybrid_decision(pg_pred, pg_conf, m_pred, m_conf, r_pred, r_conf):
    """复现混合决策逻辑（与 AblationRecognizer.update 步骤3 一致）"""
    if pg_pred is not None:
        return pg_pred, pg_conf
    elif m_pred is not None and m_conf > 0.3:
        if r_pred is not None and r_pred > 0:
            if m_pred == 0 and m_conf > 0.90:
                return m_pred, m_conf
            else:
                return r_pred, r_conf
        else:
            return m_pred, m_conf
    elif r_pred is not None:
        return r_pred, r_conf
    else:
        return None, 0.0


def apply_smoothing(raw_decisions, smooth_window):
    """对原始帧级决策应用滑动窗口平滑，返回平滑后的帧级预测列表。"""
    history = deque(maxlen=smooth_window)
    smoothed = []
    for item in raw_decisions:
        if item is None:
            smoothed.append((None, 0.0))
            continue
        pred, conf = item
        if pred is None:
            smoothed.append((None, 0.0))
            continue
        history.append((pred, conf))
        # 与 _get_smoothed_pred 一致
        votes = {}
        for p, c in history:
            votes[p] = votes.get(p, 0) + c
        best = max(votes, key=votes.get)
        avg_conf = votes[best] / len(history)
        smoothed.append((best, avg_conf))
    return smoothed


def sweep_smooth_window(all_intermediate, smooth_windows, all_full, vote_threshold=0.15):
    """平滑窗口 sweep：复用缓存的 raw 决策，只改变平滑窗口。"""
    results = []
    for sw in smooth_windows:
        print(f"   w={sw:>2d}: ", end='', flush=True)
        all_labels = {}

        for track_key, frames in all_intermediate.items():
            # 提取 raw decisions
            raw_decisions = []
            for f in frames:
                if f is None:
                    raw_decisions.append(None)
                else:
                    raw_decisions.append((f['raw_pred'], f['raw_conf']))

            # 应用不同平滑窗口
            smoothed = apply_smoothing(raw_decisions, sw)
            all_labels[track_key] = compute_track_label(smoothed, threshold=vote_threshold)

        metrics = compute_metrics_simple(all_labels, all_full)
        metrics['smooth_window'] = sw
        results.append(metrics)
        print(f"Agreement={metrics['agreement']:.1%}  "
              f"Suspicious={metrics['suspicious_rate']:.1%}  "
              f"Entropy={metrics['entropy']:.3f}")
    return results


def sweep_yaw_threshold(all_intermediate, yaw_thresholds, all_full,
                        smooth_window=8, vote_threshold=0.15, pitch_threshold=28):
    """门控阈值 sweep：用缓存的 model/rule 结果，重新计算 pose_gate + hybrid。"""
    results = []
    for yt in yaw_thresholds:
        print(f"   yaw_th={yt:>2d}°: ", end='', flush=True)
        all_labels = {}

        for track_key, frames in all_intermediate.items():
            raw_decisions = []
            for f in frames:
                if f is None:
                    raw_decisions.append(None)
                    continue
                yaw, pitch = f['yaw'], f['pitch']
                # 重新计算 pose_gate
                if abs(yaw) > yt:
                    pg_pred, pg_conf = 3, 0.85
                elif pitch > pitch_threshold:
                    pg_pred, pg_conf = 5, 0.85
                elif pitch < -pitch_threshold:
                    pg_pred, pg_conf = 4, 0.85
                else:
                    pg_pred, pg_conf = None, 0.0

                pred, conf = hybrid_decision(
                    pg_pred, pg_conf,
                    f['m_pred'], f['m_conf'],
                    f['r_pred'], f['r_conf'],
                )
                raw_decisions.append((pred, conf))

            smoothed = apply_smoothing(raw_decisions, smooth_window)
            all_labels[track_key] = compute_track_label(smoothed, threshold=vote_threshold)

        metrics = compute_metrics_simple(all_labels, all_full)
        metrics['yaw_threshold'] = yt
        results.append(metrics)
        print(f"Agreement={metrics['agreement']:.1%}  "
              f"Suspicious={metrics['suspicious_rate']:.1%}  "
              f"Entropy={metrics['entropy']:.3f}")
    return results


def sweep_vote_threshold(all_intermediate, vote_thresholds, all_full, smooth_window=8):
    """投票阈值 sweep：用默认参数的平滑预测，重新投票。"""
    # 先计算默认平滑后的帧级预测
    all_smoothed = {}
    for track_key, frames in all_intermediate.items():
        raw_decisions = []
        for f in frames:
            if f is None:
                raw_decisions.append(None)
            else:
                raw_decisions.append((f['raw_pred'], f['raw_conf']))
        all_smoothed[track_key] = apply_smoothing(raw_decisions, smooth_window)

    results = []
    for vt in vote_thresholds:
        print(f"   vote_th={vt:.2f}: ", end='', flush=True)
        all_labels = {}
        for track_key, smoothed in all_smoothed.items():
            all_labels[track_key] = compute_track_label(smoothed, threshold=vt)

        metrics = compute_metrics_simple(all_labels, all_full)
        metrics['vote_threshold'] = vt
        results.append(metrics)
        print(f"Agreement={metrics['agreement']:.1%}  "
              f"Suspicious={metrics['suspicious_rate']:.1%}  "
              f"Entropy={metrics['entropy']:.3f}")
    return results


# ==================== 评估指标 ====================

def compute_metrics_simple(config_labels, full_labels):
    """简单计算一致率和可疑率"""
    common = set(config_labels.keys()) & set(full_labels.keys())
    if not common:
        return {'agreement': 0, 'suspicious_rate': 0, 'entropy': 0, 'total': 0}

    agree = sum(1 for t in common if config_labels[t] == full_labels[t])
    suspicious = sum(1 for t in config_labels.values() if t > 0)
    total = len(config_labels)

    dist = defaultdict(int)
    for label in config_labels.values():
        dist[label] += 1

    return {
        'agreement': agree / len(common) if common else 0,
        'suspicious_rate': suspicious / total if total > 0 else 0,
        'entropy': compute_shannon_entropy(dict(dist)),
        'total': total,
        'behavior_distribution': dict(dist),
    }


# ==================== 主流程 ====================

def run_sensitivity_sweep(project_root, device='cpu'):
    """运行所有参数敏感性实验（模型只推理一次）"""
    print("=" * 70)
    print("  参数敏感性分析 (Parameter Sensitivity Analysis)")
    print("=" * 70)

    # 加载数据
    print("\n[1/5] 加载全系统推理结果...")
    full_system_results = load_full_system_results(project_root)
    print(f"   已加载 {len(full_system_results)} 个视频")

    print("\n[2/5] 加载姿态数据...")
    all_pose_data = load_pose_data(project_root)
    print(f"   已加载 {len(all_pose_data)} 个视频")

    model_path = str(project_root / 'checkpoints' / 'transformer_uw_best.pt')

    # 准备全系统参照标签
    all_full = {}
    for vname in all_pose_data:
        if vname in full_system_results:
            for tid, label in full_system_results[vname].items():
                all_full[f"{vname}/{tid}"] = label

    # ========== 核心：一次性模型推理 + 中间结果缓存 ==========
    print("\n[2.5/5] 运行模型推理并缓存中间结果（仅此一次）...")
    recognizer = InstrumentedRecognizer(
        model_path=model_path, device=device,
        smooth_window=8, fps=30.0,
        use_pose_gate=True, use_model=True, use_rules=True,
    )

    total_tracks = 0
    total_frames = 0
    all_intermediate = {}  # {video/track_id: [frame_dict or None, ...]}

    for vname, pose_data in all_pose_data.items():
        recognizer.reset()
        tracks = pose_data.get('tracks', {})
        for track_id, track_info in tracks.items():
            poses = track_info.get('poses', [])
            if not poses:
                continue
            poses_sorted = sorted(poses, key=lambda p: p.get('frame', 0))
            for pose in poses_sorted:
                yaw = pose.get('yaw', 0)
                pitch = pose.get('pitch', 0)
                roll = pose.get('roll', 0)
                recognizer.update(str(track_id), yaw, pitch, roll)
                total_frames += 1

            key = f"{vname}/{track_id}"
            all_intermediate[key] = recognizer.frame_data[str(track_id)].copy()
            total_tracks += 1

        # 每个视频处理完打印进度
        print(f"   {vname}: {len(tracks)} tracks")

    print(f"   共 {total_tracks} 轨迹, {total_frames} 帧")
    print("   模型推理完成，后续 sweep 均为纯后处理。")

    results = {}

    # ========== 实验1: 平滑窗口 sweep ==========
    print("\n[3/5] 平滑窗口敏感性 (smooth_window sweep)...")
    smooth_windows = [1, 2, 4, 6, 8, 12, 16, 24, 32]
    results['smooth_window_sweep'] = sweep_smooth_window(
        all_intermediate, smooth_windows, all_full)

    # ========== 实验2: 门控阈值 sweep ==========
    print("\n[4/5] 门控阈值敏感性 (yaw_threshold sweep)...")
    yaw_thresholds = [20, 25, 30, 35, 40, 45, 50, 55, 60]
    results['yaw_threshold_sweep'] = sweep_yaw_threshold(
        all_intermediate, yaw_thresholds, all_full)

    # ========== 实验3: 投票阈值 sweep ==========
    print("\n[5/5] 投票阈值敏感性 (vote_threshold sweep)...")
    vote_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    results['vote_threshold_sweep'] = sweep_vote_threshold(
        all_intermediate, vote_thresholds, all_full)

    # 保存结果
    output_dir = project_root / 'data' / 'ablation_output'
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'experiment_type': 'parameter_sensitivity',
        'generated_at': datetime.now().isoformat(),
        'total_tracks': total_tracks,
        'total_frames': total_frames,
        'results': results,
    }

    output_path = output_dir / 'sensitivity_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f"\n结果已保存: {output_path}")

    # 打印汇总
    print("\n" + "=" * 70)
    print("  参数敏感性分析汇总")
    print("=" * 70)

    smooth_results = results['smooth_window_sweep']
    threshold_results = results['yaw_threshold_sweep']
    vote_results = results['vote_threshold_sweep']

    print("\n  [平滑窗口] 最优 w =", end=' ')
    best_sw = max(smooth_results, key=lambda x: x['agreement'])
    print(f"{best_sw['smooth_window']} (Agreement={best_sw['agreement']:.1%})")

    print("  [门控阈值] 最优 yaw_th =", end=' ')
    best_yt = max(threshold_results, key=lambda x: x['agreement'])
    print(f"{best_yt['yaw_threshold']}° (Agreement={best_yt['agreement']:.1%})")

    print("  [投票阈值] 最优 vote_th =", end=' ')
    best_vt = max(vote_results, key=lambda x: x['agreement'])
    print(f"{best_vt['vote_threshold']:.2f} (Agreement={best_vt['agreement']:.1%})")

    print("\n实验完成！")
    return output


def main():
    parser = argparse.ArgumentParser(
        description='参数敏感性分析 (Parameter Sensitivity Analysis)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='计算设备 (cuda/cpu)')
    args = parser.parse_args()

    project_root = Path(__file__).parent
    run_sensitivity_sweep(project_root, device=args.device)


if __name__ == '__main__':
    main()
