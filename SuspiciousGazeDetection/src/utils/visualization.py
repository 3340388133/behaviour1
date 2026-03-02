"""
Visualization Utilities

Functions for visualizing pose estimation, tracking, and detection results.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import math


def draw_pose_axis(
    frame: np.ndarray,
    yaw: float,
    pitch: float,
    roll: float,
    center: Tuple[int, int],
    axis_length: int = 50,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw 3D pose axis on frame.

    Args:
        frame: Input frame (H, W, 3)
        yaw: Yaw angle in degrees
        pitch: Pitch angle in degrees
        roll: Roll angle in degrees
        center: Center point (x, y)
        axis_length: Length of axis lines
        thickness: Line thickness

    Returns:
        Frame with pose axis drawn
    """
    # Convert to radians
    yaw_rad = np.deg2rad(yaw)
    pitch_rad = np.deg2rad(pitch)
    roll_rad = np.deg2rad(roll)

    # Rotation matrix
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)],
    ])
    Ry = np.array([
        [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)],
    ])
    Rz = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad), np.cos(roll_rad), 0],
        [0, 0, 1],
    ])
    R = Rz @ Ry @ Rx

    # Axis endpoints
    axes = np.array([
        [axis_length, 0, 0],   # X - red
        [0, axis_length, 0],   # Y - green
        [0, 0, axis_length],   # Z - blue
    ])

    # Rotate axes
    rotated = (R @ axes.T).T

    # Project to 2D
    cx, cy = center
    colors = [
        (0, 0, 255),   # X - red
        (0, 255, 0),   # Y - green
        (255, 0, 0),   # Z - blue
    ]

    for i, (axis, color) in enumerate(zip(rotated, colors)):
        end_x = int(cx + axis[0])
        end_y = int(cy - axis[1])  # Y-axis inverted in image
        cv2.arrowedLine(
            frame, (cx, cy), (end_x, end_y),
            color, thickness, tipLength=0.3
        )

    return frame


def draw_tracking_info(
    frame: np.ndarray,
    bbox: np.ndarray,
    track_id: int,
    confidence: float,
    pose: Optional[Tuple[float, float, float]] = None,
    is_suspicious: bool = False,
    color_normal: Tuple[int, int, int] = (0, 255, 0),
    color_suspicious: Tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """
    Draw tracking information on frame.

    Args:
        frame: Input frame
        bbox: Bounding box [x1, y1, x2, y2]
        track_id: Track ID
        confidence: Detection confidence
        pose: Optional pose (yaw, pitch, roll)
        is_suspicious: Whether person is flagged as suspicious
        color_normal: Color for normal tracks
        color_suspicious: Color for suspicious tracks

    Returns:
        Frame with tracking info drawn
    """
    x1, y1, x2, y2 = bbox.astype(int)
    color = color_suspicious if is_suspicious else color_normal

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Prepare label
    label = f"ID:{track_id}"
    if pose is not None:
        yaw, pitch, roll = pose
        label += f" Y:{yaw:.0f}"

    # Draw label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
    cv2.putText(
        frame, label, (x1, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )

    # Draw pose axis if available
    if pose is not None:
        center = ((x1 + x2) // 2, y1 + (y2 - y1) // 4)
        draw_pose_axis(frame, pose[0], pose[1], pose[2], center, axis_length=30)

    # Draw suspicious indicator
    if is_suspicious:
        cv2.putText(
            frame, "SUSPICIOUS", (x1, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_suspicious, 2
        )

    return frame


def draw_gaze_trajectory(
    frame: np.ndarray,
    yaw_history: List[float],
    center: Tuple[int, int],
    radius: int = 100,
    color: Tuple[int, int, int] = (255, 255, 0),
) -> np.ndarray:
    """
    Draw gaze direction trajectory as a polar plot overlay.

    创新点：直观显示头部转动历史，便于识别异常模式

    Args:
        frame: Input frame
        yaw_history: List of yaw angles
        center: Center point for plot
        radius: Radius of polar plot
        color: Trajectory color

    Returns:
        Frame with trajectory drawn
    """
    if len(yaw_history) < 2:
        return frame

    # Draw polar grid
    cx, cy = center
    cv2.circle(frame, (cx, cy), radius, (128, 128, 128), 1)
    cv2.circle(frame, (cx, cy), radius // 2, (128, 128, 128), 1)

    # Draw direction lines
    for angle in [0, 45, 90, 135, 180, -45, -90, -135]:
        rad = np.deg2rad(angle)
        end_x = int(cx + radius * np.sin(rad))
        end_y = int(cy - radius * np.cos(rad))
        cv2.line(frame, (cx, cy), (end_x, end_y), (100, 100, 100), 1)

    # Draw trajectory
    points = []
    for i, yaw in enumerate(yaw_history):
        # Time-based radius (more recent = larger)
        r = radius * (0.3 + 0.7 * i / len(yaw_history))
        rad = np.deg2rad(yaw)
        x = int(cx + r * np.sin(rad))
        y = int(cy - r * np.cos(rad))
        points.append((x, y))

    # Draw as polyline
    for i in range(1, len(points)):
        alpha = i / len(points)
        thickness = max(1, int(3 * alpha))
        cv2.line(frame, points[i - 1], points[i], color, thickness)

    # Mark current direction
    if points:
        cv2.circle(frame, points[-1], 5, (0, 0, 255), -1)

    return frame


def create_dashboard(
    frame: np.ndarray,
    tracks: List[Dict],
    frame_info: Dict,
    dashboard_width: int = 300,
) -> np.ndarray:
    """
    Create analysis dashboard alongside video frame.

    Args:
        frame: Video frame
        tracks: List of track information dictionaries
        frame_info: Frame metadata (fps, frame_num, etc.)
        dashboard_width: Width of dashboard panel

    Returns:
        Combined frame with dashboard
    """
    H, W = frame.shape[:2]

    # Create dashboard panel
    dashboard = np.zeros((H, dashboard_width, 3), dtype=np.uint8)
    dashboard[:] = (40, 40, 40)  # Dark gray background

    y_offset = 30

    # Title
    cv2.putText(
        dashboard, "Analysis Dashboard",
        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
    )
    y_offset += 30

    # Frame info
    cv2.putText(
        dashboard, f"Frame: {frame_info.get('frame_num', 0)}",
        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
    )
    y_offset += 20

    cv2.putText(
        dashboard, f"FPS: {frame_info.get('fps', 0):.1f}",
        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
    )
    y_offset += 30

    # Tracks info
    cv2.putText(
        dashboard, f"Tracked: {len(tracks)}",
        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
    )
    y_offset += 25

    # Individual track details
    for track in tracks[:5]:  # Show up to 5 tracks
        track_id = track.get('track_id', -1)
        pose = track.get('pose', (0, 0, 0))
        suspicious = track.get('suspicious', False)

        color = (0, 0, 255) if suspicious else (0, 255, 0)
        status = "!" if suspicious else ""

        text = f"ID {track_id}: Y={pose[0]:.0f} P={pose[1]:.0f} {status}"
        cv2.putText(
            dashboard, text,
            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
        )
        y_offset += 18

    # Suspicious count
    suspicious_count = sum(1 for t in tracks if t.get('suspicious', False))
    y_offset = H - 60

    cv2.rectangle(dashboard, (5, y_offset - 5), (dashboard_width - 5, H - 10),
                  (60, 60, 60), -1)

    status_color = (0, 0, 255) if suspicious_count > 0 else (0, 255, 0)
    cv2.putText(
        dashboard, f"Suspicious: {suspicious_count}",
        (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2
    )

    # Combine frame and dashboard
    combined = np.hstack([frame, dashboard])

    return combined


def save_detection_video(
    output_path: str,
    frames: List[np.ndarray],
    fps: float = 25.0,
):
    """
    Save detection results as video.

    Args:
        output_path: Output video path
        frames: List of processed frames
        fps: Output video FPS
    """
    if not frames:
        return

    H, W = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for frame in frames:
        writer.write(frame)

    writer.release()


def plot_pose_timeline(
    poses: np.ndarray,
    timestamps: np.ndarray,
    suspicious_intervals: Optional[List[Tuple[float, float]]] = None,
    save_path: Optional[str] = None,
):
    """
    Plot pose angles over time.

    Args:
        poses: Pose array (T, 3) - yaw, pitch, roll
        timestamps: Timestamps (T,)
        suspicious_intervals: List of (start, end) suspicious time intervals
        save_path: Optional path to save plot
    """
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        labels = ['Yaw', 'Pitch', 'Roll']
        colors = ['red', 'green', 'blue']

        for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
            ax.plot(timestamps, poses[:, i], color=color, linewidth=1)
            ax.set_ylabel(f'{label} (deg)')
            ax.grid(True, alpha=0.3)

            # Highlight suspicious intervals
            if suspicious_intervals:
                for start, end in suspicious_intervals:
                    ax.axvspan(start, end, color='red', alpha=0.2)

        axes[-1].set_xlabel('Time (s)')
        plt.suptitle('Head Pose Timeline')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        else:
            plt.show()

        plt.close()

    except ImportError:
        print("matplotlib not available for plotting")
