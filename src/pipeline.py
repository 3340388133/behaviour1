"""主Pipeline - 可疑张望行为检测系统"""
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass

from .face_detector import RetinaFaceDetector, FaceDetection
from .tracker import ByteTracker, Track
from .pose_estimator import HeadPoseEstimator, PoseResult
from .feature_extractor import FeatureExtractor, TemporalFeatures
from .rule_engine import BehaviorClassifier, RuleResult


@dataclass
class Alert:
    track_id: int
    frame_idx: int
    score: float
    rules: List[RuleResult]
    features: TemporalFeatures


class SuspiciousBehaviorDetector:
    """可疑张望行为检测器"""

    def __init__(self, fps: float = 2.0, window_size: float = 2.0,
                 conf_threshold: float = 0.5, alert_threshold: float = 0.5):
        self.fps = fps
        self.window_size = window_size
        self.conf_threshold = conf_threshold
        self.alert_threshold = alert_threshold

        # 初始化各模块
        self.detector = RetinaFaceDetector(conf_threshold=conf_threshold)
        self.tracker = ByteTracker()
        self.pose_estimator = HeadPoseEstimator(backend="6drepnet")
        self.feature_extractor = FeatureExtractor(window_size=window_size, fps=fps)
        self.classifier = BehaviorClassifier(threshold=alert_threshold)

        self.window_frames = int(window_size * fps)
        self.alerts: List[Alert] = []

    def process_video(self, video_path: str, output_path: str = None,
                      visualize: bool = True) -> List[Alert]:
        """处理视频文件

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径（可选）
            visualize: 是否可视化

        Returns:
            检测到的告警列表
        """
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(video_fps / self.fps))

        # 输出视频
        writer = None
        if output_path:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))

        self.alerts = []
        frame_idx = 0
        process_idx = 0

        pbar = tqdm(total=total_frames // frame_interval, desc="Processing")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                alerts = self._process_frame(frame, process_idx)
                self.alerts.extend(alerts)

                if visualize or writer:
                    vis_frame = self._visualize(frame, alerts)
                    if writer:
                        writer.write(vis_frame)
                    if visualize:
                        cv2.imshow("Detection", vis_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                process_idx += 1
                pbar.update(1)

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        pbar.close()

        print(f"Detected {len(self.alerts)} alerts")
        return self.alerts

    def _process_frame(self, frame: np.ndarray, frame_idx: int) -> List[Alert]:
        """处理单帧"""
        alerts = []

        # 人脸检测
        detections = self.detector.detect(frame)

        # 跟踪更新
        tracks = self.tracker.update(detections)

        # 对每个跟踪目标进行姿态估计
        for track in tracks:
            # 裁剪人脸
            face_img = self.detector.crop_face(frame, FaceDetection(
                bbox=track.bbox, confidence=track.confidence, landmarks=track.landmarks
            ))

            if face_img.size == 0:
                continue

            # 姿态估计
            pose = self.pose_estimator.estimate(face_img)

            # 置信门控
            if pose.confidence < self.conf_threshold and track.landmarks is not None:
                pose = self.pose_estimator.estimate_from_landmarks(
                    track.landmarks, (frame.shape[1], frame.shape[0])
                )

            # 更新历史
            track.pose = {'yaw': pose.yaw, 'pitch': pose.pitch,
                          'roll': pose.roll, 'conf': pose.confidence}
            track.history.append(pose)

            # 窗口满时进行行为分析
            if len(track.history) >= self.window_frames:
                features = self.feature_extractor.extract(track.history[-self.window_frames:])
                is_suspicious, score, rules = self.classifier.classify(features)

                if is_suspicious:
                    alerts.append(Alert(
                        track_id=track.track_id,
                        frame_idx=frame_idx,
                        score=score,
                        rules=rules,
                        features=features
                    ))

        return alerts

    def _visualize(self, frame: np.ndarray, alerts: List[Alert]) -> np.ndarray:
        """可视化结果"""
        vis = frame.copy()
        alert_ids = {a.track_id for a in alerts}

        for track in self.tracker.tracks:
            x1, y1, x2, y2 = track.bbox
            is_alert = track.track_id in alert_ids

            # 边框颜色
            color = (0, 0, 255) if is_alert else (0, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # 显示ID和姿态
            if track.pose:
                label = f"ID:{track.track_id} Y:{track.pose['yaw']:.0f}"
                cv2.putText(vis, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 告警标记
            if is_alert:
                cv2.putText(vis, "ALERT!", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return vis

    def process_frames(self, frame_dir: str, output_path: str = None) -> List[Alert]:
        """处理帧目录

        Args:
            frame_dir: 帧图片目录
            output_path: 输出视频路径（可选）

        Returns:
            检测到的告警列表
        """
        frame_dir = Path(frame_dir)
        frame_files = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))

        if not frame_files:
            print(f"No frames found in {frame_dir}")
            return []

        # 读取第一帧获取尺寸
        first_frame = cv2.imread(str(frame_files[0]))
        h, w = first_frame.shape[:2]

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))

        self.alerts = []

        for idx, frame_file in enumerate(tqdm(frame_files, desc="Processing frames")):
            frame = cv2.imread(str(frame_file))
            alerts = self._process_frame(frame, idx)
            self.alerts.extend(alerts)

            if writer:
                vis_frame = self._visualize(frame, alerts)
                writer.write(vis_frame)

        if writer:
            writer.release()

        print(f"Detected {len(self.alerts)} alerts")
        return self.alerts


def main():
    import argparse
    parser = argparse.ArgumentParser(description="可疑张望行为检测")
    parser.add_argument("--input", required=True, help="输入视频或帧目录")
    parser.add_argument("--output", help="输出视频路径")
    parser.add_argument("--fps", type=float, default=2.0, help="处理帧率")
    parser.add_argument("--window", type=float, default=2.0, help="滑动窗口大小(秒)")
    parser.add_argument("--threshold", type=float, default=0.5, help="告警阈值")
    parser.add_argument("--no-vis", action="store_true", help="禁用可视化")
    args = parser.parse_args()

    detector = SuspiciousBehaviorDetector(
        fps=args.fps,
        window_size=args.window,
        alert_threshold=args.threshold
    )

    input_path = Path(args.input)
    if input_path.is_dir():
        alerts = detector.process_frames(str(input_path), args.output)
    else:
        alerts = detector.process_video(str(input_path), args.output,
                                         visualize=not args.no_vis)

    # 输出告警摘要
    print("\n=== Alert Summary ===")
    for alert in alerts:
        triggered = [r.rule_name for r in alert.rules if r.triggered]
        print(f"Frame {alert.frame_idx}, Track {alert.track_id}: "
              f"score={alert.score:.2f}, rules={triggered}")


if __name__ == "__main__":
    main()
