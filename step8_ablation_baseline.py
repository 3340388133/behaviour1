#!/usr/bin/env python3
"""
Step 8: 消融实验与基线对比
Ablation Study & Baseline Comparison

离线复现策略：使用已有 pose_output/*.json 中的姿态数据，
只重跑 BehaviorRecognizer 部分（不重跑 YOLO/StrongSORT/SSD/WHENet），
保证所有配置使用完全相同的输入数据，对比公平。

实验配置（共 8 个）：
  Full: 完整系统（Pose Gate + Transformer + Rules + Smoothing w=8）
  A1:   去掉姿态门控
  A2:   去掉 Transformer（仅规则 + 姿态门控）
  A3:   去掉规则检测
  A4:   去掉时序平滑（smooth_window=1）
  B1:   纯阈值法（仅 Pose Gate）
  B2:   纯规则法（仅 RuleDetector）
  B3:   LSTM 替代 Transformer

评估指标：
  - 与全系统一致率 (Agreement %)
  - 可疑行为检出率 (Suspicious %)
  - 行为分布（5类人数分布）
  - 行为多样性 (Shannon Entropy)
  - 逐视频指标
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


# ==================== 自定义 JSON 编码器 ====================

class NumpyEncoder(json.JSONEncoder):
    """处理 numpy 类型的 JSON 编码器"""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ==================== 常量 ====================

BEHAVIOR_CLASSES = {
    0: 'Normal',
    1: 'Glancing',
    2: 'QuickTurn',
    3: 'Prolonged',
    4: 'LookDown',
    5: 'LookUp',
}

# 9 个视频（排除 demo_result）
VIDEO_NAMES = [
    'MVI_4537', 'MVI_4538', 'MVI_4539', 'MVI_4540',
    '1.14rg-1', '1.14zz-1', '1.14zz-2', '1.14zz-3', '1.14zz-4',
]

FRONTAL_VIDEOS = ['MVI_4537', 'MVI_4538', 'MVI_4539', 'MVI_4540']
LATERAL_VIDEOS = ['1.14rg-1', '1.14zz-1', '1.14zz-2', '1.14zz-3', '1.14zz-4']


# ==================== 规则检测器（复用 step7 逻辑） ====================

class RuleDetector:
    """规则检测器 - 与 step7 完全一致"""

    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def check(self, pose_buffer: list) -> tuple:
        if len(pose_buffer) < 10:
            return None, 0.0

        yaws = [p[0] for p in pose_buffer]
        pitchs = [p[1] for p in pose_buffer]

        # angular_velocity_turn (class 2)
        if len(yaws) >= 5:
            recent_5 = yaws[-5:]
            yaw_delta = abs(recent_5[-1] - recent_5[0])
            if yaw_delta > 25:
                return 2, 0.88

        # quick_turn (class 2)
        w2s = max(5, int(self.fps * 2.0))
        recent_yaws_qt = yaws[-w2s:] if len(yaws) >= w2s else yaws
        if len(recent_yaws_qt) >= 15:
            extrema = []
            for i in range(1, len(recent_yaws_qt) - 1):
                if (recent_yaws_qt[i] > recent_yaws_qt[i-1] and
                    recent_yaws_qt[i] > recent_yaws_qt[i+1]):
                    extrema.append((i, recent_yaws_qt[i], 'max'))
                elif (recent_yaws_qt[i] < recent_yaws_qt[i-1] and
                      recent_yaws_qt[i] < recent_yaws_qt[i+1]):
                    extrema.append((i, recent_yaws_qt[i], 'min'))
            for j in range(1, len(extrema)):
                amp = abs(extrema[j][1] - extrema[j-1][1])
                time_gap = extrema[j][0] - extrema[j-1][0]
                if amp > 45 and time_gap < int(self.fps * 1.0):
                    return 2, 0.90

        # looking_up (class 5)
        w3s = max(5, int(self.fps * 3.0))
        if len(pitchs) >= w3s:
            up = sum(1 for p in pitchs[-w3s:] if p > 20)
            if up >= w3s * 0.7:
                return 5, 0.85

        # prolonged_watch (class 3)
        if len(yaws) >= w3s:
            off = sum(1 for y in yaws[-w3s:] if abs(y) > 30)
            if off >= w3s * 0.7:
                return 3, 0.85

        # looking_down (class 4)
        w5s = max(5, int(self.fps * 5.0))
        if len(pitchs) >= w5s:
            down = sum(1 for p in pitchs[-w5s:] if p < -20)
            if down >= w5s * 0.7:
                return 4, 0.85

        # glancing (class 1)
        if len(yaws) >= w3s:
            recent_yaws = yaws[-w3s:]
            direction_changes = 0
            prev_dir = 0
            for i in range(1, len(recent_yaws)):
                diff = recent_yaws[i] - recent_yaws[i-1]
                if abs(diff) < 3:
                    continue
                curr_dir = 1 if diff > 0 else -1
                if prev_dir != 0 and curr_dir != prev_dir:
                    direction_changes += 1
                prev_dir = curr_dir
            amplitude = max(recent_yaws) - min(recent_yaws)
            if direction_changes >= 3 and amplitude > 30:
                return 1, 0.88

        return None, 0.0


# ==================== 消融识别器 ====================

class AblationRecognizer:
    """
    可配置模块开关的行为识别器

    与 step7 BehaviorRecognizer 逻辑一致，但允许关闭各个模块：
    - use_pose_gate: 姿态门控
    - use_model: Transformer/LSTM 模型推理
    - use_rules: 规则检测器
    - smooth_window: 时序平滑窗口大小（1=无平滑）
    """

    def __init__(self, model_path: str, device: str = 'cuda',
                 smooth_window: int = 8, fps: float = 30.0,
                 use_pose_gate: bool = True,
                 use_model: bool = True,
                 use_rules: bool = True,
                 model_type_override: str = None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.smooth_window = smooth_window
        self.seq_len = 90
        self.fps = fps
        self.use_pose_gate = use_pose_gate
        self.use_model = use_model
        self.use_rules = use_rules
        self.model = None
        self.model_type = None

        if use_model and model_path and Path(model_path).exists():
            try:
                ckpt = torch.load(model_path, map_location=self.device)
                sd = ckpt.get('model_state_dict', ckpt)

                if model_type_override == 'lstm':
                    self._load_lstm(sd)
                elif any(k.startswith('pape.') for k in sd.keys()):
                    self._load_sbrn(sd)
                elif any(k.startswith('sbrn.') for k in sd.keys()):
                    new_sd = {k.replace('sbrn.', ''): v for k, v in sd.items()
                              if k.startswith('sbrn.')}
                    self._load_sbrn(new_sd)
                else:
                    if model_type_override == 'lstm':
                        self._load_lstm(sd)
                    else:
                        self._load_basic_transformer(sd)
            except Exception as e:
                print(f"   [WARNING] 模型加载失败 ({model_path}): {e}")
                self.model = None

        self.rule_detector = RuleDetector(fps=fps)
        self.pose_buffers = {}
        self.pred_history = {}

    def _load_sbrn(self, state_dict):
        from models.sbrn import SBRN, SBRNConfig
        d_model = state_dict['pose_proj.0.weight'].shape[0]
        num_classes = state_dict['classifier.4.weight'].shape[0]
        num_layers = sum(1 for k in state_dict
                         if k.startswith('transformer_layers.') and k.endswith('.q_proj.weight'))
        n_proto = state_dict['bpcl.prototypes'].shape[1] if 'bpcl.prototypes' in state_dict else 3
        nhead = state_dict['pape.relative_bias_table'].shape[1] if 'pape.relative_bias_table' in state_dict else 4
        max_seq_len = (state_dict['pape.relative_bias_table'].shape[0] + 1) // 2 if 'pape.relative_bias_table' in state_dict else 128

        config = SBRNConfig(
            pose_input_dim=3, d_model=d_model, nhead=nhead,
            num_layers=num_layers, dim_feedforward=d_model * 2,
            num_classes=num_classes, hidden_dim=d_model,
            max_seq_len=max_seq_len, use_multimodal=False,
            use_contrastive='bpcl.prototypes' in state_dict,
            num_prototypes_per_class=n_proto,
            uncertainty_weighting='log_sigma_cls' in state_dict,
        )
        self.model = SBRN(config)
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_type = 'sbrn'

    def _load_basic_transformer(self, state_dict):
        num_classes = state_dict['classifier.4.weight'].shape[0]
        self.model = create_model(
            model_type='transformer',
            pose_input_dim=3, pose_d_model=64, pose_nhead=4,
            pose_num_layers=2, use_multimodal=False,
            hidden_dim=128, num_classes=num_classes, dropout=0.1,
            uncertainty_weighting='log_sigma_cls' in state_dict,
        )
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_type = 'basic'

    def _load_lstm(self, state_dict):
        # Infer num_classes from classifier output
        # LSTMBaseline has classifier structure: Linear -> ReLU -> Dropout -> Linear
        num_classes = state_dict['classifier.3.weight'].shape[0]
        hidden_dim = state_dict['lstm.weight_ih_l0'].shape[0] // 4
        num_layers = sum(1 for k in state_dict if k.startswith('lstm.weight_ih_l') and not k.endswith('reverse'))

        self.model = create_model(
            model_type='lstm',
            input_dim=3, hidden_dim=hidden_dim,
            num_layers=num_layers, num_classes=num_classes, dropout=0.1,
        )
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_type = 'lstm'

    def _pose_gate(self, yaw: float, pitch: float):
        if not self.use_pose_gate:
            return None, 0.0
        if abs(yaw) > 40:
            return 3, 0.85
        if pitch > 28:
            return 5, 0.85
        if pitch < -28:
            return 4, 0.85
        return None, 0.0

    def _model_predict(self, track_id: str):
        if not self.use_model or self.model is None:
            return None, 0.0

        pose_list = list(self.pose_buffers[track_id])
        buf_len = len(pose_list)
        if buf_len < 15:
            return None, 0.0

        # Invoke model every 5 frames to reduce CPU cost; reuse last prediction
        if buf_len % 5 != 0:
            cache_key = f"{track_id}_model_cache"
            if hasattr(self, '_model_cache') and cache_key in self._model_cache:
                return self._model_cache[cache_key]
            return None, 0.0

        if buf_len >= self.seq_len:
            pose_seq = pose_list[-self.seq_len:]
        else:
            pad = [pose_list[0]] * (self.seq_len - buf_len)
            pose_seq = pad + pose_list

        pose_array = np.array(pose_seq, dtype=np.float32)
        pose_tensor = torch.from_numpy(pose_array).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(pose_tensor)
            if isinstance(output, dict):
                logits = output['logits']
            else:
                logits = output[0]
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            conf = probs[0, pred].item()

        # Cache result
        if not hasattr(self, '_model_cache'):
            self._model_cache = {}
        self._model_cache[f"{track_id}_model_cache"] = (pred, conf)

        return pred, conf

    def _get_smoothed_pred(self, track_id: str):
        history = self.pred_history[track_id]
        if not history:
            return None, 0.0
        votes = {}
        for pred, conf in history:
            votes[pred] = votes.get(pred, 0) + conf
        best_pred = max(votes, key=votes.get)
        avg_conf = votes[best_pred] / len(history)
        return best_pred, avg_conf

    def update(self, track_id: str, yaw: float, pitch: float, roll: float):
        """与 step7 BehaviorRecognizer.update 完全一致的决策逻辑"""
        if track_id not in self.pose_buffers:
            self.pose_buffers[track_id] = deque(maxlen=self.seq_len * 2)
            self.pred_history[track_id] = deque(maxlen=self.smooth_window)

        self.pose_buffers[track_id].append([yaw, pitch, roll])
        buf_len = len(self.pose_buffers[track_id])
        if buf_len < 10:
            return None, 0.0

        # 0) 姿态门控
        pose_gate_pred, pose_gate_conf = self._pose_gate(yaw, pitch)

        # 1) 模型推理
        model_pred, model_conf = self._model_predict(track_id)

        # 2) 规则检测
        if self.use_rules:
            rule_pred, rule_conf = self.rule_detector.check(
                list(self.pose_buffers[track_id]))
        else:
            rule_pred, rule_conf = None, 0.0

        # 3) 混合决策（与 step7 完全一致）
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
            return None, 0.0

        self.pred_history[track_id].append((pred, conf))
        return self._get_smoothed_pred(track_id)

    def reset(self):
        """重置所有状态"""
        self.pose_buffers.clear()
        self.pred_history.clear()
        if hasattr(self, '_model_cache'):
            self._model_cache.clear()


# ==================== 轨迹级累积投票 ====================

def compute_track_label(frame_preds: list, threshold: float = 0.15) -> int:
    """
    与 step7 VideoAnnotator.update_track_behavior 一致的轨迹级投票
    frame_preds: list of (pred, conf) per frame
    """
    votes = defaultdict(int)
    for pred, conf in frame_preds:
        if pred is not None:
            votes[pred] += 1

    total = sum(votes.values())
    if total < 5:
        return 0

    best_abnormal = None
    best_count = 0
    for cls, cnt in votes.items():
        if cls > 0 and cnt > best_count:
            best_abnormal = cls
            best_count = cnt

    if best_abnormal is not None and best_count / total >= threshold:
        return best_abnormal
    return 0


# ==================== 评估指标 ====================

def compute_shannon_entropy(distribution: dict) -> float:
    """计算 Shannon 熵"""
    total = sum(distribution.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in distribution.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def compute_confusion_matrix(config_results: dict, full_results: dict,
                              num_classes: int = 6) -> list:
    """
    计算混淆矩阵: matrix[true][pred]
    true = full system label, pred = config label
    """
    matrix = [[0] * num_classes for _ in range(num_classes)]
    common_tracks = set(config_results.keys()) & set(full_results.keys())
    for t in common_tracks:
        true_label = full_results[t]
        pred_label = config_results[t]
        if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
            matrix[true_label][pred_label] += 1
    return matrix


def compute_per_class_metrics(confusion_matrix: list,
                               num_classes: int = 6) -> dict:
    """
    从混淆矩阵计算每个类的 precision/recall/F1

    Returns:
        {class_id: {'precision': ..., 'recall': ..., 'f1': ..., 'support': ...}}
    """
    results = {}
    for c in range(num_classes):
        tp = confusion_matrix[c][c]
        fp = sum(confusion_matrix[r][c] for r in range(num_classes)) - tp
        fn = sum(confusion_matrix[c]) - tp
        support = sum(confusion_matrix[c])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        results[c] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'tp': tp, 'fp': fp, 'fn': fn,
        }

    # Macro/Weighted averages
    classes_with_support = [c for c in range(num_classes)
                            if results[c]['support'] > 0]
    total_support = sum(results[c]['support'] for c in classes_with_support)

    if classes_with_support:
        macro_p = np.mean([results[c]['precision'] for c in classes_with_support])
        macro_r = np.mean([results[c]['recall'] for c in classes_with_support])
        macro_f1 = np.mean([results[c]['f1'] for c in classes_with_support])
        weighted_f1 = sum(results[c]['f1'] * results[c]['support']
                          for c in classes_with_support) / total_support
    else:
        macro_p = macro_r = macro_f1 = weighted_f1 = 0.0

    results['macro'] = {'precision': macro_p, 'recall': macro_r,
                         'f1': macro_f1}
    results['weighted'] = {'f1': weighted_f1}
    return results


def bootstrap_confidence_interval(config_labels: list, full_labels: list,
                                   n_bootstrap: int = 1000,
                                   ci: float = 0.95) -> dict:
    """
    Bootstrap 置信区间

    Args:
        config_labels: 当前配置的标签列表（与 full_labels 一一对应）
        full_labels: 全系统标签列表
        n_bootstrap: 重采样次数
        ci: 置信水平

    Returns:
        {'agreement': {'mean': ..., 'std': ..., 'ci_low': ..., 'ci_high': ...},
         'suspicious_rate': {...}}
    """
    n = len(config_labels)
    if n == 0:
        return {}

    config_arr = np.array(config_labels)
    full_arr = np.array(full_labels)

    agree_samples = []
    sus_samples = []

    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        c_sample = config_arr[idx]
        f_sample = full_arr[idx]
        agree_samples.append(np.mean(c_sample == f_sample))
        sus_samples.append(np.mean(c_sample > 0))

    alpha = (1 - ci) / 2
    return {
        'agreement': {
            'mean': float(np.mean(agree_samples)),
            'std': float(np.std(agree_samples)),
            'ci_low': float(np.percentile(agree_samples, alpha * 100)),
            'ci_high': float(np.percentile(agree_samples, (1 - alpha) * 100)),
        },
        'suspicious_rate': {
            'mean': float(np.mean(sus_samples)),
            'std': float(np.std(sus_samples)),
            'ci_low': float(np.percentile(sus_samples, alpha * 100)),
            'ci_high': float(np.percentile(sus_samples, (1 - alpha) * 100)),
        },
    }


def wilcoxon_test_per_video(config_video_metrics: dict,
                             full_video_metrics: dict,
                             metric_key: str = 'agreement') -> dict:
    """
    Wilcoxon signed-rank test: 基于逐视频指标检验显著性

    比较 Full 配置和当前配置在各视频上的差异

    Returns:
        {'statistic': ..., 'p_value': ..., 'significant': bool,
         'per_video_diffs': [...], 'mean_diff': ..., 'std_diff': ...}
    """
    from scipy import stats as scipy_stats

    common_videos = set(config_video_metrics.keys()) & set(full_video_metrics.keys())
    if len(common_videos) < 3:
        return {'p_value': 1.0, 'significant': False,
                'note': 'insufficient videos for test'}

    full_values = []
    config_values = []
    for v in sorted(common_videos):
        full_values.append(full_video_metrics[v].get(metric_key, 0))
        config_values.append(config_video_metrics[v].get(metric_key, 0))

    diffs = [f - c for f, c in zip(full_values, config_values)]

    # 如果差异全为零，Wilcoxon 无法计算
    if all(d == 0 for d in diffs):
        return {'statistic': 0, 'p_value': 1.0, 'significant': False,
                'per_video_diffs': diffs, 'mean_diff': 0.0, 'std_diff': 0.0}

    try:
        stat, p_value = scipy_stats.wilcoxon(full_values, config_values)
    except Exception:
        # fallback to paired t-test if wilcoxon fails
        try:
            stat, p_value = scipy_stats.ttest_rel(full_values, config_values)
        except Exception:
            stat, p_value = 0, 1.0

    return {
        'statistic': float(stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'per_video_diffs': diffs,
        'mean_diff': float(np.mean(diffs)),
        'std_diff': float(np.std(diffs)),
    }


def compute_metrics(config_results: dict, full_results: dict) -> dict:
    """
    计算一个配置相对于全系统的各项指标

    Args:
        config_results: {track_id: label} 当前配置结果
        full_results: {track_id: label} 全系统结果（参照）
    """
    # 只比较两者共有的 track
    common_tracks = sorted(set(config_results.keys()) & set(full_results.keys()))
    if not common_tracks:
        return {}

    total = len(common_tracks)
    agree = sum(1 for t in common_tracks
                if config_results[t] == full_results[t])
    suspicious_config = sum(1 for t in common_tracks
                            if config_results[t] > 0)
    suspicious_full = sum(1 for t in common_tracks
                          if full_results[t] > 0)

    # 行为分布
    dist = defaultdict(int)
    for t in common_tracks:
        dist[config_results[t]] += 1

    full_dist = defaultdict(int)
    for t in common_tracks:
        full_dist[full_results[t]] += 1

    # 混淆矩阵 & per-class 指标
    cm = compute_confusion_matrix(config_results, full_results)
    per_class = compute_per_class_metrics(cm)

    # Bootstrap 置信区间
    config_labels = [config_results[t] for t in common_tracks]
    full_labels = [full_results[t] for t in common_tracks]
    bootstrap = bootstrap_confidence_interval(config_labels, full_labels)

    return {
        'total_tracks': total,
        'agreement': agree / total if total > 0 else 0,
        'suspicious_rate': suspicious_config / total if total > 0 else 0,
        'full_suspicious_rate': suspicious_full / total if total > 0 else 0,
        'behavior_distribution': dict(dist),
        'full_behavior_distribution': dict(full_dist),
        'shannon_entropy': compute_shannon_entropy(dist),
        'full_shannon_entropy': compute_shannon_entropy(full_dist),
        'confusion_matrix': cm,
        'per_class_metrics': {str(k): v for k, v in per_class.items()},
        'bootstrap_ci': bootstrap,
    }


# ==================== 数据加载 ====================

def load_pose_data(project_root: Path, video_name: str) -> dict:
    """加载一个视频的姿态数据"""
    pose_path = project_root / 'data' / 'pose_output' / f'{video_name}_poses.json'
    if not pose_path.exists():
        return None
    with open(pose_path, 'r') as f:
        return json.load(f)


def load_full_system_results(project_root: Path) -> dict:
    """
    加载全系统推理结果（作为参照）
    返回 {video_name: {track_id: label}}
    """
    results = {}
    stats_dir = project_root / 'data' / 'batch_inference_output'
    for video_name in VIDEO_NAMES:
        stats_path = stats_dir / f'{video_name}_inference_stats.json'
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            track_behaviors = {}
            for tid, label in stats.get('track_behaviors', {}).items():
                track_behaviors[tid] = int(label)
            results[video_name] = track_behaviors
    return results


# ==================== 实验配置 ====================

def get_experiment_configs(project_root: Path) -> list:
    """定义所有实验配置"""
    transformer_path = str(project_root / 'checkpoints' / 'transformer_uw_best.pt')
    lstm_path = str(project_root / 'checkpoints' / 'lstm_best.pt')

    configs = [
        {
            'name': 'Full',
            'description': '完整系统（参照）',
            'model_path': transformer_path,
            'use_pose_gate': True,
            'use_model': True,
            'use_rules': True,
            'smooth_window': 8,
            'model_type_override': None,
        },
        # ---- 消融实验 ----
        {
            'name': 'A1_no_gate',
            'description': '去掉姿态门控',
            'model_path': transformer_path,
            'use_pose_gate': False,
            'use_model': True,
            'use_rules': True,
            'smooth_window': 8,
            'model_type_override': None,
        },
        {
            'name': 'A2_no_transformer',
            'description': '去掉 Transformer',
            'model_path': transformer_path,
            'use_pose_gate': True,
            'use_model': False,
            'use_rules': True,
            'smooth_window': 8,
            'model_type_override': None,
        },
        {
            'name': 'A3_no_rules',
            'description': '去掉规则检测',
            'model_path': transformer_path,
            'use_pose_gate': True,
            'use_model': True,
            'use_rules': False,
            'smooth_window': 8,
            'model_type_override': None,
        },
        {
            'name': 'A4_no_smooth',
            'description': '去掉时序平滑 (w=1)',
            'model_path': transformer_path,
            'use_pose_gate': True,
            'use_model': True,
            'use_rules': True,
            'smooth_window': 1,
            'model_type_override': None,
        },
        # ---- 基线方法 ----
        {
            'name': 'B1_threshold',
            'description': '纯阈值法（仅 Pose Gate）',
            'model_path': None,
            'use_pose_gate': True,
            'use_model': False,
            'use_rules': False,
            'smooth_window': 1,
            'model_type_override': None,
        },
        {
            'name': 'B2_rules_only',
            'description': '纯规则法（仅 RuleDetector）',
            'model_path': None,
            'use_pose_gate': False,
            'use_model': False,
            'use_rules': True,
            'smooth_window': 1,
            'model_type_override': None,
        },
        {
            'name': 'B3_lstm',
            'description': 'LSTM 替代 Transformer',
            'model_path': lstm_path,
            'use_pose_gate': True,
            'use_model': True,
            'use_rules': True,
            'smooth_window': 8,
            'model_type_override': 'lstm',
        },
    ]
    return configs


# ==================== 主实验流程 ====================

def run_single_config(config: dict, pose_data: dict,
                      device: str = 'cuda',
                      recognizer: 'AblationRecognizer' = None) -> dict:
    """
    在单个视频的姿态数据上运行一个配置

    Args:
        config: 实验配置
        pose_data: 已加载的 pose JSON
        device: 计算设备
        recognizer: 预创建的识别器（避免重复加载模型）

    Returns:
        {track_id: label} 映射
    """
    if recognizer is None:
        recognizer = AblationRecognizer(
            model_path=config['model_path'],
            device=device,
            smooth_window=config['smooth_window'],
            fps=30.0,
            use_pose_gate=config['use_pose_gate'],
            use_model=config['use_model'],
            use_rules=config['use_rules'],
            model_type_override=config.get('model_type_override'),
        )

    # 重置状态（复用识别器时需要）
    recognizer.reset()

    tracks = pose_data.get('tracks', {})
    track_frame_preds = defaultdict(list)

    # 按轨迹遍历，按帧顺序 feed 到识别器
    for track_id, track_info in tracks.items():
        poses = track_info.get('poses', [])
        if not poses:
            continue

        # 按帧号排序
        poses_sorted = sorted(poses, key=lambda p: p['frame'])

        for pose in poses_sorted:
            yaw = pose.get('yaw')
            pitch = pose.get('pitch')
            roll = pose.get('roll')
            if yaw is None or pitch is None:
                continue

            pred, conf = recognizer.update(track_id, yaw, pitch,
                                           roll if roll is not None else 0.0)
            track_frame_preds[track_id].append((pred, conf))

    # 轨迹级累积投票
    track_labels = {}
    for track_id, preds in track_frame_preds.items():
        track_labels[track_id] = compute_track_label(preds)

    return track_labels


def run_all_experiments(project_root: Path, device: str = 'cuda',
                        output_dir: Path = None) -> dict:
    """运行所有实验配置"""
    if output_dir is None:
        output_dir = project_root / 'data' / 'ablation_output'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载全系统结果
    print("=" * 70)
    print("  消融实验与基线对比 (Ablation Study & Baseline Comparison)")
    print("=" * 70)
    print()

    print("[1/4] 加载全系统推理结果...")
    full_system_results = load_full_system_results(project_root)
    print(f"   已加载 {len(full_system_results)} 个视频的全系统结果")
    for vn, tb in full_system_results.items():
        print(f"   {vn}: {len(tb)} tracks")

    # 加载所有视频的姿态数据
    print(f"\n[2/4] 加载姿态数据...")
    all_pose_data = {}
    for video_name in VIDEO_NAMES:
        pose_data = load_pose_data(project_root, video_name)
        if pose_data is not None:
            n_tracks = len(pose_data.get('tracks', {}))
            total_poses = sum(len(t.get('poses', []))
                              for t in pose_data.get('tracks', {}).values())
            all_pose_data[video_name] = pose_data
            print(f"   {video_name}: {n_tracks} tracks, {total_poses} pose points")
    print(f"   共加载 {len(all_pose_data)} 个视频")

    # 实验配置
    configs = get_experiment_configs(project_root)
    print(f"\n[3/4] 运行 {len(configs)} 个实验配置...")

    all_results = {}

    for ci, config in enumerate(configs):
        config_name = config['name']
        print(f"\n   --- [{ci+1}/{len(configs)}] {config_name}: {config['description']} ---")

        # 创建一次识别器，跨视频复用
        recognizer = AblationRecognizer(
            model_path=config['model_path'],
            device=device,
            smooth_window=config['smooth_window'],
            fps=30.0,
            use_pose_gate=config['use_pose_gate'],
            use_model=config['use_model'],
            use_rules=config['use_rules'],
            model_type_override=config.get('model_type_override'),
        )

        config_track_labels = {}   # {video: {track_id: label}}
        config_video_metrics = {}  # {video: metrics}

        for vi, (video_name, pose_data) in enumerate(all_pose_data.items()):
            n_tracks = len(pose_data.get('tracks', {}))
            print(f"      [{vi+1}/{len(all_pose_data)}] {video_name} "
                  f"({n_tracks} tracks)...", end='', flush=True)
            track_labels = run_single_config(config, pose_data, device,
                                             recognizer=recognizer)
            print(f" done ({len(track_labels)} labels)")
            config_track_labels[video_name] = track_labels

            # 计算与全系统的比较指标
            if video_name in full_system_results:
                metrics = compute_metrics(track_labels,
                                          full_system_results[video_name])
                config_video_metrics[video_name] = metrics

        # 汇总全局指标
        all_track_labels = {}
        all_full_labels = {}
        for vn in config_track_labels:
            for tid, label in config_track_labels[vn].items():
                all_track_labels[f"{vn}/{tid}"] = label
            if vn in full_system_results:
                for tid, label in full_system_results[vn].items():
                    all_full_labels[f"{vn}/{tid}"] = label

        global_metrics = compute_metrics(all_track_labels, all_full_labels)
        global_metrics['per_video'] = config_video_metrics

        # 分场景指标
        frontal_labels = {}
        frontal_full = {}
        lateral_labels = {}
        lateral_full = {}
        for vn in config_track_labels:
            group = 'frontal' if vn in FRONTAL_VIDEOS else 'lateral'
            for tid, label in config_track_labels[vn].items():
                key = f"{vn}/{tid}"
                if group == 'frontal':
                    frontal_labels[key] = label
                else:
                    lateral_labels[key] = label
            if vn in full_system_results:
                for tid, label in full_system_results[vn].items():
                    key = f"{vn}/{tid}"
                    if group == 'frontal':
                        frontal_full[key] = label
                    else:
                        lateral_full[key] = label

        global_metrics['frontal'] = compute_metrics(frontal_labels, frontal_full)
        global_metrics['lateral'] = compute_metrics(lateral_labels, lateral_full)

        all_results[config_name] = {
            'config': {
                'name': config_name,
                'description': config['description'],
                'use_pose_gate': config['use_pose_gate'],
                'use_model': config['use_model'],
                'use_rules': config['use_rules'],
                'smooth_window': config['smooth_window'],
            },
            'metrics': global_metrics,
        }

        # 打印摘要
        agr = global_metrics.get('agreement', 0)
        sus = global_metrics.get('suspicious_rate', 0)
        ent = global_metrics.get('shannon_entropy', 0)
        n_tracks = global_metrics.get('total_tracks', 0)
        print(f"      Tracks: {n_tracks} | Agreement: {agr:.1%} | "
              f"Suspicious: {sus:.1%} | Entropy: {ent:.3f}")

    # Wilcoxon 显著性检验: 每个配置 vs Full
    print(f"\n   --- 统计显著性检验 (Wilcoxon signed-rank test) ---")
    full_per_video = all_results.get('Full', {}).get('metrics', {}).get('per_video', {})
    significance_results = {}

    for config_name, data in all_results.items():
        if config_name == 'Full':
            continue
        config_per_video = data['metrics'].get('per_video', {})

        # 逐视频一致率差异检验
        sig = wilcoxon_test_per_video(
            config_per_video, full_per_video, metric_key='agreement')
        significance_results[config_name] = sig
        data['metrics']['significance_vs_full'] = sig

        p_str = f"p={sig['p_value']:.4f}"
        sig_mark = " *" if sig.get('significant') else ""
        print(f"      Full vs {config_name:<20s}: "
              f"Δ_mean={sig.get('mean_diff', 0):+.4f} ± "
              f"{sig.get('std_diff', 0):.4f}  {p_str}{sig_mark}")

    # 输出结果
    print(f"\n[4/4] 保存结果...")

    # 加载已有测试集指标
    ablation_results_path = project_root / 'checkpoints' / 'ablation_results.json'
    test_metrics = {}
    if ablation_results_path.exists():
        with open(ablation_results_path, 'r') as f:
            test_metrics = json.load(f)

    output = {
        'experiment_type': 'ablation_and_baseline',
        'generated_at': datetime.now().isoformat(),
        'videos_analyzed': list(all_pose_data.keys()),
        'num_videos': len(all_pose_data),
        'configs': all_results,
        'test_set_metrics': test_metrics,
        'significance_tests': significance_results,
    }

    output_path = output_dir / 'ablation_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f"   JSON: {output_path}")

    # 打印汇总表格
    print_summary_table(all_results, test_metrics)

    return output


# ==================== 打印表格 ====================

def print_summary_table(all_results: dict, test_metrics: dict):
    """打印消融实验和基线对比的汇总表格"""
    print()
    print("=" * 90)
    print("  消融实验与基线对比汇总")
    print("=" * 90)

    # 表1: 整体指标
    print()
    print("表1: 整体指标对比")
    print("-" * 90)
    header = f"{'配置':<20s} {'一致率':>8s} {'可疑率':>8s} {'熵':>7s} {'正面一致':>8s} {'侧面一致':>8s} {'轨迹数':>7s}"
    print(header)
    print("-" * 90)

    for name, data in all_results.items():
        m = data['metrics']
        fm = m.get('frontal', {})
        lm = m.get('lateral', {})
        line = (f"{name:<20s} "
                f"{m.get('agreement', 0):>7.1%} "
                f"{m.get('suspicious_rate', 0):>7.1%} "
                f"{m.get('shannon_entropy', 0):>7.3f} "
                f"{fm.get('agreement', 0):>7.1%} "
                f"{lm.get('agreement', 0):>7.1%} "
                f"{m.get('total_tracks', 0):>7d}")
        print(line)
    print("-" * 90)

    # 表2: 行为分布
    print()
    print("表2: 行为分布（人数）")
    print("-" * 90)
    header = f"{'配置':<20s} {'Normal':>8s} {'Glancing':>8s} {'QuickTurn':>9s} {'Prolonged':>9s} {'LookDown':>9s} {'LookUp':>8s}"
    print(header)
    print("-" * 90)

    for name, data in all_results.items():
        dist = data['metrics'].get('behavior_distribution', {})
        counts = [dist.get(i, 0) for i in range(6)]
        line = (f"{name:<20s} "
                f"{counts[0]:>8d} {counts[1]:>8d} {counts[2]:>9d} "
                f"{counts[3]:>9d} {counts[4]:>9d} {counts[5]:>8d}")
        print(line)
    print("-" * 90)

    # 表3: 测试集指标
    if test_metrics:
        print()
        print("表3: 测试集指标（来自 ablation_results.json）")
        print("-" * 70)
        header = f"{'模型':<25s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}"
        print(header)
        print("-" * 70)
        for model_name, metrics in test_metrics.items():
            acc = metrics.get('test_accuracy', 0)
            prec = metrics.get('test_precision', 0)
            rec = metrics.get('test_recall', 0)
            f1 = metrics.get('test_f1', 0)
            desc = metrics.get('description', model_name)
            line = f"{desc:<25s} {acc:>9.4f} {prec:>9.4f} {rec:>9.4f} {f1:>9.4f}"
            print(line)
        print("-" * 70)

    # 表4: 模块贡献度
    print()
    print("表4: 模块贡献度（去掉后一致率下降量）")
    print("-" * 60)
    full_agreement = all_results.get('Full', {}).get('metrics', {}).get('agreement', 1.0)
    ablation_names = ['A1_no_gate', 'A2_no_transformer', 'A3_no_rules', 'A4_no_smooth']
    module_names = ['姿态门控 (Pose Gate)', 'Transformer 模型', '规则检测 (Rules)', '时序平滑 (Smoothing)']

    for aname, mname in zip(ablation_names, module_names):
        if aname in all_results:
            a_agr = all_results[aname]['metrics'].get('agreement', 0)
            drop = full_agreement - a_agr
            bar = '█' * int(drop * 200) + '░' * max(0, 20 - int(drop * 200))
            print(f"  {mname:<25s}  Δ={drop:>+6.1%}  {bar}")
    print("-" * 60)

    # 表5: Per-class F1 (macro)
    print()
    print("表5: Per-class F1 (vs 全系统)")
    print("-" * 105)
    header = (f"{'配置':<20s} {'Normal':>8s} {'Glancing':>8s} {'QuickTurn':>9s} "
              f"{'Prolonged':>9s} {'LookDown':>9s} {'LookUp':>8s} {'Macro-F1':>9s}")
    print(header)
    print("-" * 105)

    for name, data in all_results.items():
        pcm = data['metrics'].get('per_class_metrics', {})
        f1s = []
        for i in range(6):
            f1 = pcm.get(str(i), {}).get('f1', 0)
            f1s.append(f1)
        macro = pcm.get('macro', {}).get('f1', 0)
        line = (f"{name:<20s} "
                f"{f1s[0]:>7.3f} {f1s[1]:>8.3f} {f1s[2]:>9.3f} "
                f"{f1s[3]:>9.3f} {f1s[4]:>9.3f} {f1s[5]:>8.3f} {macro:>9.3f}")
        print(line)
    print("-" * 105)

    # 表6: Bootstrap 置信区间
    print()
    print("表6: Bootstrap 95% 置信区间 (n=1000)")
    print("-" * 85)
    header = f"{'配置':<20s} {'一致率 mean':>11s} {'95% CI':>18s} {'可疑率 mean':>11s} {'95% CI':>18s}"
    print(header)
    print("-" * 85)

    for name, data in all_results.items():
        bs = data['metrics'].get('bootstrap_ci', {})
        agr = bs.get('agreement', {})
        sus = bs.get('suspicious_rate', {})
        agr_ci = f"[{agr.get('ci_low', 0):.3f}, {agr.get('ci_high', 0):.3f}]"
        sus_ci = f"[{sus.get('ci_low', 0):.3f}, {sus.get('ci_high', 0):.3f}]"
        print(f"{name:<20s} {agr.get('mean', 0):>10.3f} {agr_ci:>18s} "
              f"{sus.get('mean', 0):>10.3f} {sus_ci:>18s}")
    print("-" * 85)

    # 表7: 显著性检验
    print()
    print("表7: Wilcoxon signed-rank test (Full vs 各配置, 逐视频一致率)")
    print("-" * 80)
    header = f"{'对比':<30s} {'Δ_mean':>10s} {'Δ_std':>10s} {'p-value':>10s} {'显著?':>8s}"
    print(header)
    print("-" * 80)

    for name, data in all_results.items():
        if name == 'Full':
            continue
        sig = data['metrics'].get('significance_vs_full', {})
        if sig:
            p = sig.get('p_value', 1.0)
            is_sig = "Yes *" if sig.get('significant') else "No"
            print(f"{'Full vs ' + name:<30s} "
                  f"{sig.get('mean_diff', 0):>+9.4f} "
                  f"{sig.get('std_diff', 0):>9.4f} "
                  f"{p:>10.4f} {is_sig:>8s}")
    print("-" * 80)
    print("  (* p < 0.05)")

    # 表8: Full 配置的混淆矩阵
    print()
    full_cm = all_results.get('Full', {}).get('metrics', {}).get('confusion_matrix')
    if full_cm:
        print("表8: Full 配置混淆矩阵 (行=全系统在线, 列=离线复现)")
        print("-" * 70)
        classes = ['Normal', 'Glanc', 'QTurn', 'Prolong', 'LDown', 'LUp']
        header = f"{'':>10s} " + " ".join(f"{c:>8s}" for c in classes)
        print(header)
        print("-" * 70)
        for i, row in enumerate(full_cm):
            line = f"{classes[i]:>10s} " + " ".join(f"{v:>8d}" for v in row)
            print(line)
        print("-" * 70)
    print()


# ==================== 入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description='消融实验与基线对比 (Ablation Study & Baseline Comparison)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备 (cuda/cpu)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录')
    args = parser.parse_args()

    project_root = Path(__file__).parent
    output_dir = Path(args.output) if args.output else None

    results = run_all_experiments(project_root, device=args.device,
                                  output_dir=output_dir)

    print("实验完成！结果已保存。")


if __name__ == '__main__':
    main()
