#!/usr/bin/env python3
"""
Step 7: 基于头部检测的可疑行为识别推理系统
Head-Based Suspicious Behavior Recognition Inference

核心改进：
- 使用 SSD 人脸检测器在人体 ROI 内精确定位人脸
- 扩展人脸框为头部框，确保完整包围头部区域
- WHENet 实时头部姿态估计
- Transformer + 规则混合的6类行为识别
- 专业级视频标注与统计面板

管线:
  人体跟踪(预计算) → 人脸检测(SSD) → 头部框扩展 →
  姿态估计(WHENet) → 行为分类(Transformer+规则) → 视频标注

行为类别：
  0: normal          正常行为     绿色    视线稳定
  1: glancing        频繁张望     红色    3秒内左右转头≥3次, yaw变化>30°
  2: quick_turn      快速回头     橙色    0.5秒内 yaw变化>60°
  3: prolonged_watch  长时间观察  紫色    持续>3秒注视非正前方(yaw>30°)
  4: looking_down    持续低头     蓝色    pitch<-20° 持续>5秒
  5: looking_up      持续抬头     黄色    pitch>20° 持续>3秒
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'recognition'))

import json
import argparse
import cv2
import torch
import numpy as np
import onnxruntime as ort
from collections import deque, defaultdict
from tqdm import tqdm
from datetime import datetime

from temporal_transformer import create_model

# ==================== 常量定义 ====================

BEHAVIOR_CLASSES = {
    0: ('Normal',    (0, 200, 0)),       # 绿色
    1: ('Glancing',  (0, 0, 255)),       # 红色
    2: ('QuickTurn', (0, 128, 255)),     # 橙色
    3: ('Prolonged', (180, 0, 180)),     # 紫色
    4: ('LookDown',  (255, 128, 0)),     # 蓝色
    5: ('LookUp',    (0, 230, 230)),     # 黄色
}

BEHAVIOR_NAMES_CN = {
    0: '正常行为',
    1: '频繁张望',
    2: '快速回头',
    3: '长时间观察',
    4: '持续低头',
    5: '持续抬头',
}


# ==================== 头部检测器 ====================

class FaceDetectorSSD:
    """OpenCV DNN SSD 人脸检测器"""

    def __init__(self, prototxt_path: str, model_path: str, conf_threshold: float = 0.45):
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        # DNN backend
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.conf_threshold = conf_threshold

    def detect(self, image: np.ndarray) -> list:
        """检测人脸，返回 [(x1, y1, x2, y2, conf), ...]"""
        h, w = image.shape[:2]
        if h < 20 or w < 20:
            return []

        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self.conf_threshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            # 确保有效
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2, y2, conf))
        return faces

    def detect_in_roi(self, frame: np.ndarray, roi_bbox: list, expand: float = 0.15) -> list:
        """在 ROI 区域（上半身）内检测人脸"""
        fh, fw = frame.shape[:2]
        rx1, ry1, rx2, ry2 = [int(v) for v in roi_bbox]

        # 只取上半部分（头部在身体上部）
        body_h = ry2 - ry1
        body_w = rx2 - rx1
        # 搜索区域：身体上方 60%
        search_y2 = ry1 + int(body_h * 0.6)

        # 扩展搜索区域确保不遗漏
        sx1 = max(0, rx1 - int(body_w * expand))
        sy1 = max(0, ry1 - int(body_h * 0.05))
        sx2 = min(fw, rx2 + int(body_w * expand))
        sy2 = min(fh, search_y2)

        roi = frame[sy1:sy2, sx1:sx2]
        if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
            return []

        faces = self.detect(roi)
        # 转换到原图坐标
        result = []
        for fx1, fy1, fx2, fy2, conf in faces:
            result.append((fx1 + sx1, fy1 + sy1, fx2 + sx1, fy2 + sy1, conf))
        return result


def face_to_head_bbox(face_bbox, frame_shape,
                      expand_top=0.6, expand_bottom=0.15,
                      expand_side=0.35):
    """
    将人脸框扩展为头部框，确保完整包围头部

    Args:
        face_bbox: (x1, y1, x2, y2, ...) 人脸边界框
        frame_shape: 帧的 shape
        expand_top: 向上扩展比例（覆盖额头和头顶）
        expand_bottom: 向下扩展比例（覆盖下巴）
        expand_side: 左右扩展比例（覆盖耳朵和头发）
    """
    x1, y1, x2, y2 = int(face_bbox[0]), int(face_bbox[1]), int(face_bbox[2]), int(face_bbox[3])
    fw, fh = x2 - x1, y2 - y1

    new_x1 = x1 - int(fw * expand_side)
    new_y1 = y1 - int(fh * expand_top)
    new_x2 = x2 + int(fw * expand_side)
    new_y2 = y2 + int(fh * expand_bottom)

    h, w = frame_shape[:2]
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(w, new_x2)
    new_y2 = min(h, new_y2)

    return (new_x1, new_y1, new_x2, new_y2)


def estimate_head_from_body(body_bbox, frame_shape):
    """从全身框估计头部位置（fallback 方案）"""
    bx1, by1, bx2, by2 = [int(v) for v in body_bbox]
    body_w = bx2 - bx1
    body_h = by2 - by1

    # 头部约占身体高度的 1/5，宽度约为身体宽度的 55%
    head_h = max(int(body_h * 0.22), 40)
    head_w = max(int(body_w * 0.55), 35)

    cx = (bx1 + bx2) // 2
    h, w = frame_shape[:2]
    x1 = max(0, cx - head_w // 2)
    y1 = max(0, by1 - int(head_h * 0.08))
    x2 = min(w, cx + head_w // 2)
    y2 = min(h, y1 + head_h)

    return (x1, y1, x2, y2)


# ==================== 头部姿态估计器 ====================

class WHENetEstimator:
    """WHENet ONNX 头部姿态估计器"""

    def __init__(self, model_path: str):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def estimate(self, head_image: np.ndarray):
        """
        估计头部姿态
        Returns: (yaw, pitch, roll) 或 None
        """
        if head_image is None or head_image.size == 0:
            return None
        if head_image.shape[0] < 10 or head_image.shape[1] < 10:
            return None

        try:
            resized = cv2.resize(head_image, (224, 224))
            input_data = resized.astype(np.float32)
            input_data = np.transpose(input_data, (2, 0, 1))
            input_data = np.expand_dims(input_data, 0)

            outputs = self.session.run(None, {self.input_name: input_data})
            yaw = float(outputs[0][0][0])
            roll = float(outputs[0][0][1])
            pitch = float(outputs[0][0][2])

            return yaw, pitch, roll
        except Exception:
            return None


# ==================== 行为识别器 ====================

class RuleDetector:
    """规则检测器 - 处理精确阈值行为"""

    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def check(self, pose_buffer: list) -> tuple:
        """检查规则定义的行为"""
        if len(pose_buffer) < 10:
            return None, 0.0

        yaws = [p[0] for p in pose_buffer]
        pitchs = [p[1] for p in pose_buffer]

        # angular_velocity_turn (class 2): 基于角速度的瞬时转头检测
        # 最近5帧内 yaw 累计变化 > 25°，说明正在快速转头，无需等待完整 V 形
        if len(yaws) >= 5:
            recent_5 = yaws[-5:]
            yaw_delta = abs(recent_5[-1] - recent_5[0])
            if yaw_delta > 25:
                return 2, 0.88

        # quick_turn (class 2): 最近2秒内出现快速来回转头
        # 条件：短时间内 yaw 先大幅变化再反向变化（V形或倒V形）
        w2s = max(5, int(self.fps * 2.0))
        recent_yaws_qt = yaws[-w2s:] if len(yaws) >= w2s else yaws
        if len(recent_yaws_qt) >= 15:
            # 检测 V形模式: 找局部极值，相邻极值差>45°
            extrema = []
            for i in range(1, len(recent_yaws_qt) - 1):
                if (recent_yaws_qt[i] > recent_yaws_qt[i-1] and
                    recent_yaws_qt[i] > recent_yaws_qt[i+1]):
                    extrema.append((i, recent_yaws_qt[i], 'max'))
                elif (recent_yaws_qt[i] < recent_yaws_qt[i-1] and
                      recent_yaws_qt[i] < recent_yaws_qt[i+1]):
                    extrema.append((i, recent_yaws_qt[i], 'min'))
            # 需要至少2个极值（一个来回）且幅度>45°（从60°降低）
            for j in range(1, len(extrema)):
                amp = abs(extrema[j][1] - extrema[j-1][1])
                time_gap = extrema[j][0] - extrema[j-1][0]
                if amp > 45 and time_gap < int(self.fps * 1.0):
                    return 2, 0.90

        # looking_up (class 5): pitch>20° 持续>3秒
        w3s = max(5, int(self.fps * 3.0))
        if len(pitchs) >= w3s:
            up = sum(1 for p in pitchs[-w3s:] if p > 20)
            if up >= w3s * 0.7:
                return 5, 0.85

        # prolonged_watch (class 3): |yaw|>30° 持续>3秒
        if len(yaws) >= w3s:
            off = sum(1 for y in yaws[-w3s:] if abs(y) > 30)
            if off >= w3s * 0.7:
                return 3, 0.85

        # looking_down (class 4): pitch<-20° 持续>5秒
        w5s = max(5, int(self.fps * 5.0))
        if len(pitchs) >= w5s:
            down = sum(1 for p in pitchs[-w5s:] if p < -20)
            if down >= w5s * 0.7:
                return 4, 0.85

        # glancing (class 1): 3秒内转头≥3次, yaw变化>30°
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


class BehaviorRecognizer:
    """混合行为识别器: SBRN/Transformer模型 + 规则"""

    def __init__(self, model_path: str, device: str = 'cuda',
                 smooth_window: int = 8, fps: float = 30.0):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.smooth_window = smooth_window
        self.seq_len = 90
        self.fps = fps
        self.model = None
        self.model_type = None  # 'sbrn' or 'basic'

        # 尝试加载模型
        try:
            ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
            sd = ckpt.get('model_state_dict', ckpt)

            # 检测模型类型
            if any(k.startswith('pape.') for k in sd.keys()):
                # SBRN 模型
                self._load_sbrn(sd)
            elif any(k.startswith('sbrn.') for k in sd.keys()):
                # 包装过的 SBRN
                new_sd = {k.replace('sbrn.', ''): v for k, v in sd.items()
                          if k.startswith('sbrn.')}
                self._load_sbrn(new_sd)
            else:
                # 基础 Transformer
                self._load_basic_transformer(sd)
        except Exception as e:
            print(f"   [WARNING] 模型加载失败: {e}")
            print(f"   将使用纯规则模式")
            self.model = None

        self.rule_detector = RuleDetector(fps=fps)
        self.pose_buffers = {}
        self.pred_history = {}

    def _load_sbrn(self, state_dict):
        """加载 SBRN 模型"""
        sys.path.insert(0, str(Path(__file__).parent / 'src' / 'recognition'))
        from models.sbrn import SBRN, SBRNConfig

        # 从权重推断配置
        d_model = state_dict['pose_proj.0.weight'].shape[0]
        num_classes = state_dict['classifier.4.weight'].shape[0]
        num_layers = sum(1 for k in state_dict if k.startswith('transformer_layers.') and k.endswith('.q_proj.weight'))
        n_proto = state_dict['bpcl.prototypes'].shape[1] if 'bpcl.prototypes' in state_dict else 3
        nhead = state_dict['pape.relative_bias_table'].shape[1] if 'pape.relative_bias_table' in state_dict else 4
        max_seq_len = (state_dict['pape.relative_bias_table'].shape[0] + 1) // 2 if 'pape.relative_bias_table' in state_dict else 128

        config = SBRNConfig(
            pose_input_dim=3,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=d_model * 2,
            num_classes=num_classes,
            hidden_dim=d_model,
            max_seq_len=max_seq_len,
            use_multimodal=False,
            use_contrastive='bpcl.prototypes' in state_dict,
            num_prototypes_per_class=n_proto,
            uncertainty_weighting='log_sigma_cls' in state_dict,
        )
        self.model = SBRN(config)
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_type = 'sbrn'
        print(f"   SBRN 模型加载成功 (d={d_model}, layers={num_layers}, classes={num_classes})")

    def _load_basic_transformer(self, state_dict):
        """加载基础 Transformer 模型"""
        num_classes = state_dict['classifier.4.weight'].shape[0]
        d_model = state_dict.get('pose_encoder.input_proj.weight',
                                  state_dict.get('classifier.0.weight', None))

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
        print(f"   基础 Transformer 模型加载成功 (classes={num_classes})")

    def update(self, track_id: str, yaw: float, pitch: float, roll: float):
        """更新跟踪对象的姿态并返回行为预测"""
        if track_id not in self.pose_buffers:
            self.pose_buffers[track_id] = deque(maxlen=self.seq_len * 2)
            self.pred_history[track_id] = deque(maxlen=self.smooth_window)

        self.pose_buffers[track_id].append([yaw, pitch, roll])
        buf_len = len(self.pose_buffers[track_id])

        if buf_len < 10:
            return None, 0.0

        # 0) 实时姿态门控: 当前帧极端姿态直接标记
        pose_gate_pred, pose_gate_conf = self._pose_gate(yaw, pitch)

        # 1) 模型推理
        model_pred, model_conf = self._model_predict(track_id)

        # 2) 规则检测
        rule_pred, rule_conf = self.rule_detector.check(list(self.pose_buffers[track_id]))

        # 3) 混合决策
        # 姿态门控最高优先: 当前帧极端姿态不可能是Normal
        if pose_gate_pred is not None:
            pred, conf = pose_gate_pred, pose_gate_conf
        elif model_pred is not None and model_conf > 0.3:
            if rule_pred is not None and rule_pred > 0:
                # 模型 Normal 置信 > 0.90 才能阻止规则（降低阻断门槛，让规则更容易介入）
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

    def _pose_gate(self, yaw: float, pitch: float):
        """实时姿态门控: 明显头部姿态直接判定为非Normal"""
        # 侧视 (|yaw| > 40°) → Prolonged (侧视观察)
        if abs(yaw) > 40:
            return 3, 0.85
        # 抬头 (pitch > 28°) → LookUp
        if pitch > 28:
            return 5, 0.85
        # 低头 (pitch < -28°) → LookDown
        if pitch < -28:
            return 4, 0.85
        return None, 0.0

    def _model_predict(self, track_id: str):
        """模型推理 (支持 SBRN 和基础 Transformer)"""
        if self.model is None:
            return None, 0.0

        pose_list = list(self.pose_buffers[track_id])
        buf_len = len(pose_list)

        if buf_len < 15:
            return None, 0.0

        # 取最近 seq_len 帧，不足则 padding
        if buf_len >= self.seq_len:
            pose_seq = pose_list[-self.seq_len:]
        else:
            pad = [pose_list[0]] * (self.seq_len - buf_len)
            pose_seq = pad + pose_list

        pose_array = np.array(pose_seq, dtype=np.float32)
        pose_tensor = torch.from_numpy(pose_array).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(pose_tensor)

            # SBRN 返回 dict, 基础模型返回 tuple
            if isinstance(output, dict):
                logits = output['logits']
            else:
                logits = output[0]

            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            conf = probs[0, pred].item()

        return pred, conf

    def _get_smoothed_pred(self, track_id: str):
        """时序平滑预测"""
        history = self.pred_history[track_id]
        if not history:
            return None, 0.0
        votes = {}
        for pred, conf in history:
            votes[pred] = votes.get(pred, 0) + conf
        best_pred = max(votes, key=votes.get)
        avg_conf = votes[best_pred] / len(history)
        return best_pred, avg_conf

    def get_last_pose(self, track_id: str):
        """获取最近姿态"""
        if track_id in self.pose_buffers and self.pose_buffers[track_id]:
            return self.pose_buffers[track_id][-1]
        return None


# ==================== 可视化绘制 ====================

class VideoAnnotator:
    """专业视频标注器"""

    def __init__(self, frame_width: int, frame_height: int):
        self.fw = frame_width
        self.fh = frame_height
        # 统计面板尺寸
        self.panel_w = 320
        self.panel_h = 220
        # 累积统计
        self.track_behaviors = {}
        self._track_votes = {}
        self.frame_behavior_counts = defaultdict(int)

    def draw_head_bbox(self, frame, head_bbox, track_id, pred, conf,
                       yaw=None, pitch=None):
        """绘制头部检测框和标签"""
        x1, y1, x2, y2 = [int(v) for v in head_bbox]

        if pred is not None:
            class_name, color = BEHAVIOR_CLASSES[pred]
            # 非正常行为加粗框
            thickness = 3 if pred > 0 else 2
        else:
            class_name = "Detecting"
            color = (128, 128, 128)
            thickness = 1

        # 绘制头部边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # 绘制角点装饰（增强视觉效果）
        corner_len = max(8, min(x2 - x1, y2 - y1) // 4)
        # 左上角
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness + 1)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness + 1)
        # 右上角
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness + 1)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness + 1)
        # 左下角
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness + 1)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness + 1)
        # 右下角
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness + 1)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness + 1)

        # 标签文字
        short_id = str(track_id).split('_')[-1] if '_' in str(track_id) else str(track_id)[-4:]
        if pred is not None:
            label = f"#{short_id} {class_name} {conf:.0%}"
        else:
            label = f"#{short_id} {class_name}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        font_thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        # 标签背景（在框上方）
        label_y1 = max(0, y1 - th - 10)
        label_y2 = y1
        cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 10, label_y2), color, -1)
        cv2.putText(frame, label, (x1 + 5, label_y2 - 4),
                    font, font_scale, (255, 255, 255), font_thickness)

        # 姿态角度（在框下方，仅非正常行为显示）
        if pred is not None and pred > 0 and yaw is not None and pitch is not None:
            pose_text = f"Y:{yaw:.0f} P:{pitch:.0f}"
            (pw, ph), _ = cv2.getTextSize(pose_text, font, 0.4, 1)
            cv2.rectangle(frame, (x1, y2), (x1 + pw + 6, y2 + ph + 6), (0, 0, 0), -1)
            cv2.putText(frame, pose_text, (x1 + 3, y2 + ph + 3),
                        font, 0.4, color, 1)

        return frame

    def draw_statistics_panel(self, frame, frame_id, total_frames,
                              current_persons, fps_val=0):
        """绘制左上角统计面板"""
        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (5 + self.panel_w, 5 + self.panel_h),
                      (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # 边框
        cv2.rectangle(frame, (5, 5), (5 + self.panel_w, 5 + self.panel_h),
                      (100, 100, 100), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        x_start = 15
        y = 30

        # 标题
        cv2.putText(frame, "Head Pose Behavior Analysis",
                    (x_start, y), font, 0.55, (0, 220, 220), 2)
        y += 25

        # 帧信息
        progress = frame_id / max(total_frames, 1) * 100
        cv2.putText(frame, f"Frame: {frame_id}/{total_frames} ({progress:.0f}%)",
                    (x_start, y), font, 0.45, (200, 200, 200), 1)
        y += 20

        # 当前检测人数
        cv2.putText(frame, f"Tracked Persons: {current_persons}",
                    (x_start, y), font, 0.45, (200, 200, 200), 1)
        y += 22

        # 分隔线
        cv2.line(frame, (x_start, y - 5), (5 + self.panel_w - 10, y - 5),
                 (80, 80, 80), 1)
        y += 8

        # 各类别统计（按人数）
        behavior_person_counts = defaultdict(int)
        for tid, beh in self.track_behaviors.items():
            behavior_person_counts[beh] += 1

        total_suspicious = sum(v for k, v in behavior_person_counts.items() if k > 0)
        cv2.putText(frame, f"Suspicious: {total_suspicious} persons",
                    (x_start, y), font, 0.50, (0, 180, 255), 2)
        y += 22

        for i in range(6):
            class_name, color = BEHAVIOR_CLASSES[i]
            count = behavior_person_counts.get(i, 0)
            bar_color = color if count > 0 else (60, 60, 60)
            text_color = color if count > 0 else (100, 100, 100)

            cv2.putText(frame, f"{class_name:12s}: {count}",
                        (x_start, y), font, 0.42, text_color, 1)
            # 进度条
            bar_x = x_start + 170
            bar_w = 120
            max_count = max(max(behavior_person_counts.values(), default=1), 1)
            fill_w = int(bar_w * count / max_count) if count > 0 else 0
            cv2.rectangle(frame, (bar_x, y - 10), (bar_x + bar_w, y + 2),
                          (40, 40, 40), -1)
            if fill_w > 0:
                cv2.rectangle(frame, (bar_x, y - 10), (bar_x + fill_w, y + 2),
                              bar_color, -1)
            y += 18

        return frame

    def update_track_behavior(self, track_id, pred):
        """更新跟踪目标行为 - 基于累积投票"""
        if pred is None:
            return
        if track_id not in self._track_votes:
            self._track_votes[track_id] = defaultdict(int)
        self._track_votes[track_id][pred] += 1

        # 计算最终行为: 非Normal需至少占总帧数15%才认定
        votes = self._track_votes[track_id]
        total = sum(votes.values())
        if total < 5:
            return

        # 找非Normal中得票最多的
        best_abnormal = None
        best_count = 0
        for cls, cnt in votes.items():
            if cls > 0 and cnt > best_count:
                best_abnormal = cls
                best_count = cnt

        if best_abnormal is not None and best_count / total >= 0.15:
            self.track_behaviors[track_id] = best_abnormal
        else:
            self.track_behaviors[track_id] = 0


# ==================== 头部框平滑器 ====================

class BBoxSmoother:
    """边界框时序平滑"""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self.history = {}

    def smooth(self, track_id, bbox):
        if track_id not in self.history:
            self.history[track_id] = list(bbox)
            return bbox
        old = self.history[track_id]
        smoothed = [
            self.alpha * bbox[i] + (1 - self.alpha) * old[i]
            for i in range(4)
        ]
        self.history[track_id] = smoothed
        return tuple(int(v) for v in smoothed)


# ==================== 主推理管线 ====================

def build_frame_index(tracking_data, pose_data):
    """构建帧索引: frame_id → [{track_id, bbox, yaw, pitch, roll}, ...]"""
    frame_dict = {}
    pose_tracks = pose_data.get('tracks', {})

    for track in tracking_data['tracks']:
        track_id = track['track_id']
        frames = track['frames']
        bboxes = track['bboxes']

        # 姿态数据（预计算，用于跟没有实时检测的帧做 fallback）
        pose_key = f'track_{track_id}' if isinstance(track_id, int) else str(track_id)
        track_poses = pose_tracks.get(pose_key, {}).get('poses', [])
        pose_by_frame = {p['frame']: p for p in track_poses}

        for i, frame_id in enumerate(frames):
            if frame_id not in frame_dict:
                frame_dict[frame_id] = []
            bbox = bboxes[i] if i < len(bboxes) else [0, 0, 100, 100]
            pose = pose_by_frame.get(frame_id, {})
            frame_dict[frame_id].append({
                'track_id': pose_key,
                'body_bbox': bbox,
                'precomputed_yaw': pose.get('yaw'),
                'precomputed_pitch': pose.get('pitch'),
                'precomputed_roll': pose.get('roll'),
            })
    return frame_dict


def run_inference(video_name: str, model_path: str, output_path: str,
                  start_frame: int = 0, max_frames: int = 0,
                  save_keyframes: bool = True):
    """运行头部检测推理"""
    project_root = Path(__file__).parent

    # ---- 加载数据 ----
    print(f"[1/5] 加载跟踪和姿态数据: {video_name}")
    tracking_path = project_root / 'data' / 'tracked_output' / video_name / 'tracking_result.json'
    pose_path = project_root / 'data' / 'pose_output' / f'{video_name}_poses.json'

    with open(tracking_path, 'r') as f:
        tracking_data = json.load(f)
    with open(pose_path, 'r') as f:
        pose_data = json.load(f)

    frame_dict = build_frame_index(tracking_data, pose_data)
    print(f"   有效帧: {len(frame_dict)}")

    # ---- 打开视频 ----
    raw_video_path = tracking_data.get('video_path', '')
    video_path = project_root / raw_video_path
    if not video_path.exists():
        # 尝试其他路径
        for base in ['data/raw_videos/侧机位', 'data/raw_videos/正机位']:
            for ext in ['.MP4', '.mp4', '.avi']:
                p = project_root / base / f'{video_name}{ext}'
                if p.exists():
                    video_path = p
                    break
    if not video_path.exists():
        print(f"视频不存在: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"   视频: {width}x{height} @ {fps:.1f}fps, 共 {total_frames} 帧")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    end_frame = total_frames
    if max_frames > 0:
        end_frame = min(start_frame + max_frames, total_frames)
    process_count = end_frame - start_frame
    print(f"   处理范围: 帧 {start_frame} → {end_frame} (共 {process_count} 帧)")

    # ---- 初始化模块 ----
    print(f"[2/5] 初始化检测模块...")
    face_detector = FaceDetectorSSD(
        str(project_root / 'models' / 'deploy.prototxt'),
        str(project_root / 'models' / 'res10_300x300_ssd_iter_140000.caffemodel'),
        conf_threshold=0.4,
    )
    print(f"   SSD 人脸检测器已加载")

    pose_estimator = WHENetEstimator(
        str(project_root / 'models' / 'whenet_1x3x224x224_prepost.onnx')
    )
    print(f"   WHENet 姿态估计器已加载")

    print(f"[3/5] 加载行为识别模型: {model_path}")
    recognizer = BehaviorRecognizer(model_path, fps=fps)
    print(f"   Transformer 行为识别器已加载")

    # ---- 输出视频 ----
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    annotator = VideoAnnotator(width, height)
    bbox_smoother = BBoxSmoother(alpha=0.4)

    # 关键帧保存目录
    keyframe_dir = output_dir / f'{video_name}_keyframes'
    if save_keyframes:
        keyframe_dir.mkdir(parents=True, exist_ok=True)

    # ---- 推理循环 ----
    print(f"[4/5] 开始推理...")
    frame_id = start_frame
    pbar = tqdm(total=process_count, desc="Head Detection Inference")

    # 统计信息
    stats = {
        'total_face_detections': 0,
        'total_fallback_heads': 0,
        'frame_person_counts': [],
        'behavior_events': [],
    }
    # 关键帧采样间隔
    keyframe_interval = max(1, process_count // 20)  # 保存约20张关键帧
    saved_keyframes = []

    while frame_id < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        frame_data = frame_dict.get(frame_id, [])
        current_persons = len(frame_data)
        stats['frame_person_counts'].append(current_persons)

        for item in frame_data:
            track_id = item['track_id']
            body_bbox = item['body_bbox']

            # ---- 步骤 1: 人脸检测 → 头部框 ----
            faces = face_detector.detect_in_roi(frame, body_bbox)

            head_bbox = None
            face_detected = False

            if faces:
                # 选择最大的人脸（最可能是该人物的脸）
                best_face = max(faces, key=lambda f: (f[2]-f[0]) * (f[3]-f[1]))
                head_bbox = face_to_head_bbox(best_face, frame.shape)
                face_detected = True
                stats['total_face_detections'] += 1
            else:
                # Fallback: 从身体框估计
                head_bbox = estimate_head_from_body(body_bbox, frame.shape)
                stats['total_fallback_heads'] += 1

            # 平滑边界框
            head_bbox = bbox_smoother.smooth(track_id, head_bbox)

            # ---- 步骤 2: 姿态估计 ----
            hx1, hy1, hx2, hy2 = [int(v) for v in head_bbox]
            head_crop = frame[hy1:hy2, hx1:hx2]

            yaw, pitch, roll = None, None, None
            # 无论人脸是否检测成功，都尝试在头部框上运行WHENet
            if head_crop.size > 0 and head_crop.shape[0] > 10 and head_crop.shape[1] > 10:
                pose_result = pose_estimator.estimate(head_crop)
                if pose_result is not None:
                    yaw, pitch, roll = pose_result

            # 如果实时姿态失败，用预计算值
            if yaw is None and item['precomputed_yaw'] is not None:
                yaw = item['precomputed_yaw']
                pitch = item['precomputed_pitch']
                roll = item['precomputed_roll']

            # 人脸检测丢失信号: 记录历史 & 增强yaw
            if not hasattr(annotator, '_face_seen'):
                annotator._face_seen = set()
            if face_detected:
                annotator._face_seen.add(track_id)
            elif track_id in annotator._face_seen and yaw is not None:
                # 之前检测到人脸, 现在丢失 → 头部大幅转动
                # yaw 至少为 ±55° (SSD 对 >45° 侧脸基本检测不到)
                if abs(yaw) < 55:
                    yaw = 55.0 if yaw >= 0 else -55.0

            # ---- 步骤 3: 行为识别 ----
            pred, conf = None, 0.0
            if yaw is not None:
                pred, conf = recognizer.update(track_id, yaw, pitch, roll)

            # ---- 步骤 4: 绘制 ----
            annotator.draw_head_bbox(frame, head_bbox, track_id, pred, conf,
                                     yaw=yaw, pitch=pitch)
            annotator.update_track_behavior(track_id, pred)

            # 记录事件
            if pred is not None and pred > 0:
                stats['behavior_events'].append({
                    'frame': frame_id,
                    'track_id': track_id,
                    'behavior': pred,
                    'confidence': conf,
                })

        # 绘制统计面板
        annotator.draw_statistics_panel(frame, frame_id, total_frames,
                                        current_persons)

        out.write(frame)

        # 保存关键帧
        if save_keyframes and (frame_id - start_frame) % keyframe_interval == 0:
            kf_path = str(keyframe_dir / f'frame_{frame_id:06d}.jpg')
            cv2.imwrite(kf_path, frame)
            saved_keyframes.append(kf_path)

        # 保存有可疑行为的帧
        if save_keyframes and len(frame_data) > 0:
            has_suspicious = any(
                annotator.track_behaviors.get(item['track_id'], 0) > 0
                for item in frame_data
            )
            if has_suspicious and (frame_id % (keyframe_interval // 2) == 0):
                kf_path = str(keyframe_dir / f'suspicious_{frame_id:06d}.jpg')
                cv2.imwrite(kf_path, frame)
                saved_keyframes.append(kf_path)

        frame_id += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    # ---- 输出统计 ----
    print(f"\n[5/5] 推理完成!")
    print(f"   输出视频: {output_path}")
    print(f"   人脸检测成功: {stats['total_face_detections']}")
    print(f"   头部估计 fallback: {stats['total_fallback_heads']}")
    face_rate = stats['total_face_detections'] / max(1, stats['total_face_detections'] + stats['total_fallback_heads'])
    print(f"   人脸检测率: {face_rate:.1%}")
    print(f"   保存关键帧: {len(saved_keyframes)} 张")

    # 行为统计
    behavior_counts = defaultdict(int)
    for tid, beh in annotator.track_behaviors.items():
        behavior_counts[beh] += 1

    print(f"\n   行为统计 (按人数):")
    for i in range(6):
        name, _ = BEHAVIOR_CLASSES[i]
        cn = BEHAVIOR_NAMES_CN[i]
        count = behavior_counts.get(i, 0)
        print(f"     [{i}] {name:12s} ({cn}): {count} 人")

    # 保存统计结果
    stats_path = output_dir / f'{video_name}_inference_stats.json'
    stats_output = {
        'video_name': video_name,
        'total_frames_processed': process_count,
        'face_detection_count': stats['total_face_detections'],
        'fallback_count': stats['total_fallback_heads'],
        'face_detection_rate': face_rate,
        'behavior_person_counts': dict(behavior_counts),
        'track_behaviors': dict(annotator.track_behaviors),
        'behavior_events_count': len(stats['behavior_events']),
        'processed_at': datetime.now().isoformat(),
    }
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_output, f, ensure_ascii=False, indent=2)
    print(f"   统计数据: {stats_path}")

    return stats_output, saved_keyframes, annotator


def main():
    parser = argparse.ArgumentParser(
        description='基于头部检测的可疑行为识别推理系统')
    parser.add_argument('--video', type=str, default='MVI_4537',
                        help='视频名称')
    parser.add_argument('--model', type=str,
                        default='checkpoints/transformer_uw_best.pt',
                        help='行为识别模型路径')
    parser.add_argument('--output', type=str, help='输出视频路径')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='起始帧')
    parser.add_argument('--max_frames', type=int, default=3000,
                        help='最大处理帧数 (0=全部)')
    parser.add_argument('--no_keyframes', action='store_true',
                        help='不保存关键帧')
    args = parser.parse_args()

    if not args.output:
        args.output = f'data/inference_output/{args.video}_head_behavior.mp4'

    print("=" * 65)
    print("  基于头部姿态估计的口岸人员可疑张望识别系统")
    print("  Head Pose-Based Suspicious Behavior Recognition System")
    print("=" * 65)

    run_inference(
        args.video,
        args.model,
        args.output,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        save_keyframes=not args.no_keyframes,
    )


if __name__ == '__main__':
    main()
