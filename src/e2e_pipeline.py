"""
端到端 Pipeline - 可疑张望行为检测系统
串联所有模块：人脸检测 -> 跟踪 -> 姿态估计 -> 时序特征 -> 规则引擎 -> 告警生成
"""
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import defaultdict

from face_detector import RetinaFaceDetector
from tracker import ByteTracker
from head_pose import HeadPoseEstimator, PoseResult
from temporal_features import TemporalFeatureExtractor, TemporalFeature
from rule_engine import RuleEngine, EvaluationResult
from alert_generator import generate_alerts, AlertReport


@dataclass
class TrackState:
    """单个跟踪目标的状态"""
    track_id: int
    poses: List[Dict] = field(default_factory=list)  # 姿态历史
    times: List[float] = field(default_factory=list)  # 时间戳历史
    bbox: Optional[List[int]] = None
    is_active: bool = True


@dataclass
class FrameResult:
    """单帧处理结果"""
    frame_idx: int
    time_sec: float
    tracks: List[TrackState]
    alerts: List[Dict]


class E2EPipeline:
    """端到端可疑张望行为检测 Pipeline"""

    def __init__(
        self,
        process_fps: float = 10.0,
        window_size: float = 2.0,
        step_size: float = 0.5,
        alert_threshold: float = 0.5,
        face_conf_threshold: float = 0.5,
        model_path: str = None
    ):
        """
        Args:
            process_fps: 处理帧率
            window_size: 滑动窗口大小（秒）
            step_size: 滑动窗口步长（秒）
            alert_threshold: 告警阈值
            face_conf_threshold: 人脸检测置信度阈值
            model_path: 姿态估计模型路径
        """
        self.process_fps = process_fps
        self.window_size = window_size
        self.step_size = step_size
        self.alert_threshold = alert_threshold

        # 初始化各模块
        print("初始化模块...")
        self.detector = RetinaFaceDetector(conf_threshold=face_conf_threshold)
        self.tracker = ByteTracker()
        self.pose_estimator = HeadPoseEstimator(model_path)
        self.feature_extractor = TemporalFeatureExtractor(
            window_size=window_size,
            step_size=step_size
        )
        self.rule_engine = RuleEngine()

        # 跟踪状态
        self.track_states: Dict[int, TrackState] = {}
        self.all_alerts: List[Dict] = []

    def reset(self):
        """重置状态"""
        self.tracker = ByteTracker()
        self.track_states = {}
        self.all_alerts = []

    def process_video(
        self,
        video_path: str,
        output_video: str = None,
        output_json: str = None,
        visualize: bool = False
    ) -> AlertReport:
        """处理视频文件

        Args:
            video_path: 输入视频路径
            output_video: 输出视频路径（可选）
            output_json: 输出 JSON 路径（可选）
            visualize: 是否实时可视化

        Returns:
            AlertReport 告警报告
        """
        self.reset()
        video_path = Path(video_path)
        video_name = video_path.stem

        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 计算帧间隔
        frame_interval = max(1, int(video_fps / self.process_fps))
        actual_fps = video_fps / frame_interval

        print(f"视频: {video_name}")
        print(f"  原始帧率: {video_fps:.1f} fps")
        print(f"  处理帧率: {actual_fps:.1f} fps (每 {frame_interval} 帧处理一次)")
        print(f"  总帧数: {total_frames}")

        # 输出视频
        writer = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video, fourcc, actual_fps, (width, height))

        # 处理循环
        frame_idx = 0
        process_idx = 0
        pbar = tqdm(total=total_frames, desc="处理中")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                time_sec = frame_idx / video_fps
                frame_result = self._process_frame(frame, process_idx, time_sec)

                # 可视化
                if visualize or writer:
                    vis_frame = self._visualize(frame, frame_result)
                    if writer:
                        writer.write(vis_frame)
                    if visualize:
                        cv2.imshow("Detection", vis_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                process_idx += 1

            frame_idx += 1
            pbar.update(1)

        cap.release()
        if writer:
            writer.release()
        if visualize:
            cv2.destroyAllWindows()
        pbar.close()

        # 生成告警报告
        report = self._generate_report(video_name)

        # 保存 JSON
        if output_json:
            with open(output_json, 'w', encoding='utf-8') as f:
                f.write(report.to_json())
            print(f"告警报告已保存: {output_json}")

        return report

    def _process_frame(self, frame: np.ndarray, frame_idx: int, time_sec: float) -> FrameResult:
        """处理单帧"""
        # 1. 人脸检测
        detections = self.detector.detect(frame)

        # 2. 跟踪更新
        tracks = self.tracker.update(detections)

        # 3. 对每个跟踪目标进行姿态估计
        frame_alerts = []
        for track in tracks:
            track_id = track.track_id
            bbox = track.bbox

            # 初始化或获取跟踪状态
            if track_id not in self.track_states:
                self.track_states[track_id] = TrackState(track_id=track_id)

            state = self.track_states[track_id]
            state.bbox = bbox
            state.is_active = True

            # 裁剪人脸
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            face_img = frame[y1:y2, x1:x2]

            if face_img.size == 0:
                continue

            # 姿态估计
            pose = self.pose_estimator.estimate(face_img)

            # 记录姿态历史
            state.poses.append({
                'yaw': pose.yaw,
                'pitch': pose.pitch,
                'roll': pose.roll,
                'confidence': pose.confidence
            })
            state.times.append(time_sec)

            # 4. 时序特征提取和规则评估
            if len(state.times) >= 5:  # 至少5个样本
                alerts = self._evaluate_track(state, track_id)
                frame_alerts.extend(alerts)

        # 标记不活跃的跟踪
        active_ids = {t.track_id for t in tracks}
        for track_id, state in self.track_states.items():
            if track_id not in active_ids:
                state.is_active = False

        return FrameResult(
            frame_idx=frame_idx,
            time_sec=time_sec,
            tracks=list(self.track_states.values()),
            alerts=frame_alerts
        )

    def _evaluate_track(self, state: TrackState, track_id: int) -> List[Dict]:
        """评估单个跟踪目标"""
        alerts = []

        times = np.array(state.times)
        yaws = np.array([p['yaw'] for p in state.poses])

        # 提取时序特征
        features = self.feature_extractor.extract_from_track(times, yaws, track_id)

        # 对每个窗口进行规则评估
        for feat in features:
            # 检查是否已评估过该窗口
            window_key = (track_id, feat.window_start, feat.window_end)
            if hasattr(self, '_evaluated_windows') and window_key in self._evaluated_windows:
                continue

            if not hasattr(self, '_evaluated_windows'):
                self._evaluated_windows = set()
            self._evaluated_windows.add(window_key)

            # 构建特征字典
            feat_dict = feat.to_dict()
            mask = (times >= feat.window_start) & (times < feat.window_end)
            feat_dict['yaws'] = yaws[mask]

            # 规则评估
            eval_result = self.rule_engine.evaluate(feat_dict)

            if eval_result.is_suspicious:
                alert = {
                    'track_id': track_id,
                    'window_start': feat.window_start,
                    'window_end': feat.window_end,
                    'is_suspicious': True,
                    'weighted_score': eval_result.weighted_score,
                    'rules': eval_result.to_dict()['rules']
                }
                alerts.append(alert)
                self.all_alerts.append(alert)

        return alerts

    def _generate_report(self, video_name: str) -> AlertReport:
        """生成告警报告"""
        # 按 track_id 分组
        track_alerts = defaultdict(list)
        for alert in self.all_alerts:
            track_alerts[alert['track_id']].append(alert)

        # 生成每个 track 的报告
        all_eval_results = []
        for track_id, alerts in track_alerts.items():
            for alert in alerts:
                all_eval_results.append(alert)

        return generate_alerts(all_eval_results, video_name, track_id=0)

    def _visualize(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """可视化结果"""
        vis = frame.copy()
        alert_tracks = {a['track_id'] for a in result.alerts}

        for state in result.tracks:
            if not state.is_active or state.bbox is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in state.bbox]
            is_alert = state.track_id in alert_tracks

            # 边框颜色
            color = (0, 0, 255) if is_alert else (0, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # 显示 ID 和 yaw
            if state.poses:
                yaw = state.poses[-1]['yaw']
                label = f"ID:{state.track_id} Y:{yaw:.0f}"
                cv2.putText(vis, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 告警标记
            if is_alert:
                cv2.putText(vis, "ALERT!", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 显示时间和告警数
        time_str = f"Time: {result.time_sec:.1f}s | Alerts: {len(self.all_alerts)}"
        cv2.putText(vis, time_str, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return vis

    def process_pose_csv(
        self,
        csv_path: str,
        output_json: str = None
    ) -> AlertReport:
        """处理已有的 pose CSV 文件（离线模式）

        Args:
            csv_path: pose CSV 文件路径
            output_json: 输出 JSON 路径（可选）

        Returns:
            AlertReport 告警报告
        """
        csv_path = Path(csv_path)
        video_name = csv_path.stem

        print(f"加载 pose 数据: {csv_path}")
        df = pd.read_csv(csv_path)
        df['track_id'] = df.groupby('frame_id').cumcount()

        print(f"数据点数: {len(df)}")
        print(f"轨迹数: {df['track_id'].nunique()}")

        # 对每个 track 进行评估
        all_eval_results = []

        for track_id in df['track_id'].unique():
            track_df = df[df['track_id'] == track_id].sort_values('time_sec')
            times = track_df['time_sec'].values
            yaws = track_df['yaw'].values

            # 提取时序特征
            features = self.feature_extractor.extract_from_track(times, yaws, track_id)

            # 评估每个窗口
            for feat in features:
                feat_dict = feat.to_dict()
                mask = (times >= feat.window_start) & (times < feat.window_end)
                feat_dict['yaws'] = yaws[mask]

                eval_result = self.rule_engine.evaluate(feat_dict)
                all_eval_results.append({
                    'track_id': track_id,
                    'window_start': feat.window_start,
                    'window_end': feat.window_end,
                    'is_suspicious': eval_result.is_suspicious,
                    'weighted_score': eval_result.weighted_score,
                    'rules': eval_result.to_dict()['rules']
                })

        # 生成告警报告
        report = generate_alerts(all_eval_results, video_name, track_id=0)

        # 保存 JSON
        if output_json:
            with open(output_json, 'w', encoding='utf-8') as f:
                f.write(report.to_json())
            print(f"告警报告已保存: {output_json}")

        # 打印摘要
        print(f"\n{'='*50}")
        print(f"处理完成: {video_name}")
        print(f"{'='*50}")
        print(f"总窗口数: {len(all_eval_results)}")
        print(f"可疑窗口数: {report.suspicious_windows}")
        print(f"告警数: {len(report.alerts)}")

        return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description='端到端可疑张望行为检测')
    parser.add_argument('input', help='输入视频路径或 pose CSV 路径')
    parser.add_argument('--output-video', help='输出视频路径')
    parser.add_argument('--output-json', help='输出 JSON 路径')
    parser.add_argument('--fps', type=float, default=10.0, help='处理帧率')
    parser.add_argument('--window', type=float, default=2.0, help='滑动窗口大小(秒)')
    parser.add_argument('--step', type=float, default=0.5, help='滑动窗口步长(秒)')
    parser.add_argument('--threshold', type=float, default=0.5, help='告警阈值')
    parser.add_argument('--visualize', action='store_true', help='实时可视化')
    parser.add_argument('--offline', action='store_true', help='离线模式（处理 pose CSV）')
    args = parser.parse_args()

    pipeline = E2EPipeline(
        process_fps=args.fps,
        window_size=args.window,
        step_size=args.step,
        alert_threshold=args.threshold
    )

    input_path = Path(args.input)

    if args.offline or input_path.suffix == '.csv':
        # 离线模式：处理 pose CSV
        report = pipeline.process_pose_csv(
            str(input_path),
            output_json=args.output_json
        )
    else:
        # 在线模式：处理视频
        report = pipeline.process_video(
            str(input_path),
            output_video=args.output_video,
            output_json=args.output_json,
            visualize=args.visualize
        )

    # 输出告警摘要
    print("\n=== 告警摘要 ===")
    for alert in report.alerts[:10]:  # 只显示前10条
        print(f"  [{alert.time_start:.1f}s-{alert.time_end:.1f}s] "
              f"{alert.rule_name_cn}: {alert.reason[:50]}...")

    if len(report.alerts) > 10:
        print(f"  ... 共 {len(report.alerts)} 条告警")


if __name__ == '__main__':
    main()
