#!/usr/bin/env python3
"""
Demo Script for Suspicious Gaze Detection

Run inference on videos and visualize results.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.tracker.strong_sort import StrongSORTWrapper, extract_head_crop
from src.utils.visualization import (
    draw_tracking_info,
    draw_gaze_trajectory,
    create_dashboard,
    save_detection_video,
)
from src.utils.coordinate import CoordinateTransformer, CameraPosition


def parse_args():
    parser = argparse.ArgumentParser(description="Suspicious Gaze Detection Demo")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input video path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="front",
        choices=["front", "side_left", "side_right"],
        help="Camera position",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--suspicious-threshold",
        type=float,
        default=0.7,
        help="Suspicious behavior threshold",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save output video",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display results in window",
    )
    return parser.parse_args()


class SuspiciousGazeDetector:
    """
    End-to-end suspicious gaze detection pipeline.
    """

    def __init__(
        self,
        checkpoint_path: str = None,
        device: str = "cuda",
        conf_threshold: float = 0.5,
        suspicious_threshold: float = 0.7,
        camera_position: str = "front",
    ):
        self.device = device
        self.conf_threshold = conf_threshold
        self.suspicious_threshold = suspicious_threshold

        # Setup camera coordinate transformer
        self.coord_transformer = CoordinateTransformer()
        self.camera_position = CameraPosition(camera_position)

        # Initialize detector (YOLOv8)
        self._init_detector()

        # Initialize tracker
        self.tracker = StrongSORTWrapper(device=device)

        # Initialize pose model
        self._init_pose_model(checkpoint_path)

        # Track histories for gaze pattern analysis
        self.gaze_histories = {}  # track_id -> list of yaw values

    def _init_detector(self):
        """Initialize YOLOv8 detector."""
        try:
            from ultralytics import YOLO
            self.detector = YOLO("yolov8m.pt")
            print("Loaded YOLOv8m detector")
        except ImportError:
            print("Warning: ultralytics not installed. Using mock detector.")
            self.detector = None

    def _init_pose_model(self, checkpoint_path: str = None):
        """Initialize head pose estimation model."""
        try:
            from src.models.head_pose import WHENetPlus
            self.pose_model = WHENetPlus(pretrained=True)
            self.pose_model.to(self.device)
            self.pose_model.eval()

            if checkpoint_path and os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.pose_model.load_state_dict(checkpoint['pose_model'])
                print(f"Loaded pose model from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load pose model: {e}")
            self.pose_model = None

    def detect_persons(self, frame: np.ndarray) -> np.ndarray:
        """Detect persons in frame."""
        if self.detector is None:
            return np.array([])

        results = self.detector(frame, verbose=False)[0]
        boxes = results.boxes

        # Filter for person class (class 0) with confidence threshold
        detections = []
        for i, cls in enumerate(boxes.cls):
            if cls == 0 and boxes.conf[i] > self.conf_threshold:
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().numpy()
                detections.append([*bbox, conf, 0])

        return np.array(detections) if detections else np.array([]).reshape(0, 6)

    def estimate_pose(self, head_crop: np.ndarray) -> tuple:
        """Estimate head pose from crop."""
        if self.pose_model is None or head_crop.size == 0:
            return 0.0, 0.0, 0.0

        # Preprocess
        crop = cv2.resize(head_crop, (224, 224))
        crop = crop.astype(np.float32) / 255.0
        crop = np.transpose(crop, (2, 0, 1))
        crop = torch.from_numpy(crop).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.pose_model(crop, camera_type=self.camera_position.value)

        yaw = output['yaw'].item()
        pitch = output['pitch'].item()
        roll = output['roll'].item()

        return yaw, pitch, roll

    def analyze_gaze_pattern(
        self,
        track_id: int,
        yaw: float,
        window_size: int = 30,
    ) -> dict:
        """
        Analyze gaze pattern for suspicious behavior.

        Args:
            track_id: Track ID
            yaw: Current yaw angle
            window_size: Analysis window size (frames)

        Returns:
            Analysis results including suspiciousness score
        """
        # Update history
        if track_id not in self.gaze_histories:
            self.gaze_histories[track_id] = []

        history = self.gaze_histories[track_id]
        history.append(yaw)

        # Keep only recent history
        if len(history) > window_size * 2:
            history = history[-window_size * 2:]
            self.gaze_histories[track_id] = history

        if len(history) < 10:
            return {
                'suspicious': False,
                'score': 0.0,
                'yaw_variance': 0.0,
                'direction_changes': 0,
            }

        # Analyze pattern
        recent = np.array(history[-window_size:])

        # Yaw variance
        yaw_var = np.var(recent)

        # Direction changes
        yaw_diff = np.diff(recent)
        direction_changes = np.sum(np.diff(np.sign(yaw_diff)) != 0)

        # Yaw range
        yaw_range = np.max(recent) - np.min(recent)

        # Compute suspiciousness score
        score = 0.0

        # High variance indicates looking around
        if yaw_var > 100:
            score += 0.3
        if yaw_var > 300:
            score += 0.2

        # Many direction changes indicate repeated looking
        if direction_changes >= 3:
            score += 0.2
        if direction_changes >= 6:
            score += 0.2

        # Large yaw range
        if yaw_range > 60:
            score += 0.1

        score = min(1.0, score)
        is_suspicious = score > self.suspicious_threshold

        return {
            'suspicious': is_suspicious,
            'score': score,
            'yaw_variance': float(yaw_var),
            'yaw_range': float(yaw_range),
            'direction_changes': int(direction_changes),
        }

    def process_frame(
        self,
        frame: np.ndarray,
        frame_num: int,
    ) -> tuple:
        """
        Process single frame through full pipeline.

        Args:
            frame: Input frame
            frame_num: Frame number

        Returns:
            Tuple of (annotated_frame, results_dict)
        """
        results = {
            'frame_num': frame_num,
            'tracks': [],
        }

        # Detect persons
        detections = self.detect_persons(frame)

        # Update tracker
        tracks = self.tracker.update(detections, frame)

        # Process each track
        for track in tracks:
            # Extract head crop
            head_crop, head_bbox = extract_head_crop(frame, track.bbox)

            # Estimate pose
            yaw, pitch, roll = self.estimate_pose(head_crop)

            # Transform to world coordinates
            world_yaw, world_pitch, world_roll = self.coord_transformer.camera_to_world(
                yaw, pitch, roll, self.camera_position
            )

            # Analyze gaze pattern
            analysis = self.analyze_gaze_pattern(track.track_id, world_yaw)

            # Store results
            track_result = {
                'track_id': track.track_id,
                'bbox': track.bbox.tolist(),
                'head_bbox': head_bbox.tolist(),
                'pose': [world_yaw, world_pitch, world_roll],
                'suspicious': analysis['suspicious'],
                'suspicious_score': analysis['score'],
            }
            results['tracks'].append(track_result)

            # Draw on frame
            frame = draw_tracking_info(
                frame,
                track.bbox,
                track.track_id,
                track.confidence,
                pose=(world_yaw, world_pitch, world_roll),
                is_suspicious=analysis['suspicious'],
            )

            # Draw gaze trajectory for suspicious persons
            if analysis['suspicious'] and track.track_id in self.gaze_histories:
                history = self.gaze_histories[track.track_id][-30:]
                center = (int(track.bbox[2]) + 50, int(track.bbox[1]) + 50)
                frame = draw_gaze_trajectory(frame, history, center, radius=40)

        # Create dashboard
        frame_info = {
            'frame_num': frame_num,
            'fps': 0,  # Will be computed in main loop
        }
        frame = create_dashboard(frame, results['tracks'], frame_info)

        return frame, results


def process_video(
    detector: SuspiciousGazeDetector,
    video_path: str,
    output_dir: str,
    save_video: bool = False,
    show: bool = False,
):
    """Process video file."""
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")

    # Setup output
    os.makedirs(output_dir, exist_ok=True)
    output_frames = []
    all_results = []

    # Process frames
    pbar = tqdm(total=total_frames, desc="Processing")
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        annotated, results = detector.process_frame(frame, frame_num)
        all_results.append(results)

        if save_video:
            output_frames.append(annotated)

        if show:
            cv2.imshow("Suspicious Gaze Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_num += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    if show:
        cv2.destroyAllWindows()

    # Save results
    video_name = Path(video_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON results
    json_path = os.path.join(output_dir, f"{video_name}_{timestamp}_results.json")
    with open(json_path, 'w') as f:
        json.dump({
            'video': video_path,
            'fps': fps,
            'total_frames': total_frames,
            'results': all_results,
        }, f, indent=2)
    print(f"Results saved to {json_path}")

    # Save video
    if save_video and output_frames:
        video_path = os.path.join(output_dir, f"{video_name}_{timestamp}_output.mp4")
        save_detection_video(video_path, output_frames, fps)
        print(f"Video saved to {video_path}")

    # Print summary
    suspicious_frames = sum(
        1 for r in all_results
        if any(t['suspicious'] for t in r['tracks'])
    )
    print(f"\nSummary:")
    print(f"  Total frames: {total_frames}")
    print(f"  Frames with suspicious activity: {suspicious_frames}")
    print(f"  Percentage: {100 * suspicious_frames / total_frames:.1f}%")


def main():
    args = parse_args()

    # Check input
    if not os.path.exists(args.input):
        print(f"Error: Input video not found: {args.input}")
        return

    # Setup device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print(f"Using device: {device}")

    # Create detector
    detector = SuspiciousGazeDetector(
        checkpoint_path=args.checkpoint,
        device=device,
        conf_threshold=args.conf_threshold,
        suspicious_threshold=args.suspicious_threshold,
        camera_position=args.camera,
    )

    # Process video
    process_video(
        detector,
        args.input,
        args.output,
        save_video=args.save_video,
        show=args.show,
    )


if __name__ == "__main__":
    main()
