"""
StrongSORT Tracker Wrapper

Wrapper for StrongSORT multi-object tracking with Re-ID.
Provides interface for tracking persons across video frames.

Reference:
    StrongSORT: Make DeepSORT Great Again
    https://github.com/dyhBUPT/StrongSORT
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum


class TrackState(Enum):
    """Track state enumeration."""
    TENTATIVE = 1    # New track, not yet confirmed
    CONFIRMED = 2    # Confirmed track
    DELETED = 3      # Track marked for deletion


@dataclass
class Track:
    """Single object track."""
    track_id: int
    bbox: np.ndarray          # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    features: np.ndarray      # Re-ID features
    state: TrackState = TrackState.TENTATIVE
    age: int = 0              # Frames since creation
    hits: int = 0             # Successful detections
    time_since_update: int = 0
    history: List[np.ndarray] = field(default_factory=list)

    def update(self, bbox: np.ndarray, confidence: float, features: np.ndarray):
        """Update track with new detection."""
        self.bbox = bbox
        self.confidence = confidence
        self.features = features
        self.hits += 1
        self.time_since_update = 0
        self.history.append(bbox.copy())

        # Confirm track after enough hits
        if self.hits >= 3 and self.state == TrackState.TENTATIVE:
            self.state = TrackState.CONFIRMED

    def mark_missed(self):
        """Mark track as missed in current frame."""
        self.time_since_update += 1
        self.age += 1

    def is_deleted(self) -> bool:
        """Check if track should be deleted."""
        return self.time_since_update > 30 or self.state == TrackState.DELETED

    def get_center(self) -> Tuple[float, float]:
        """Get bounding box center."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def get_size(self) -> Tuple[float, float]:
        """Get bounding box size."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1, y2 - y1)


class StrongSORTWrapper:
    """
    Wrapper for StrongSORT tracker.

    Provides a clean interface for multi-object tracking
    with support for head crops extraction.

    Args:
        reid_model: Re-ID model name (e.g., 'osnet_x0_25')
        max_age: Maximum frames to keep track without detection
        n_init: Frames needed to confirm track
        max_iou_distance: Maximum IoU distance for association
        device: Computation device
    """

    def __init__(
        self,
        reid_model: str = "osnet_x0_25",
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        device: str = "cuda",
    ):
        self.reid_model = reid_model
        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        self.device = device

        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.frame_count = 0

        # Try to import boxmot for actual tracking
        self._tracker = None
        self._init_tracker()

    def _init_tracker(self):
        """Initialize the actual tracker."""
        try:
            from boxmot import StrongSORT
            self._tracker = StrongSORT(
                reid_weights=f"{self.reid_model}.pt",
                device=self.device,
                max_age=self.max_age,
                n_init=self.n_init,
                max_iou_dist=self.max_iou_distance,
            )
        except ImportError:
            print("Warning: boxmot not installed. Using simple IoU tracker.")
            self._tracker = None

    def update(
        self,
        detections: np.ndarray,
        frame: np.ndarray,
    ) -> List[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: Detection array [N, 6] (x1, y1, x2, y2, conf, cls)
            frame: Current video frame (H, W, 3)

        Returns:
            List of active tracks
        """
        self.frame_count += 1

        if self._tracker is not None:
            # Use actual StrongSORT
            tracks = self._tracker.update(detections, frame)
            return self._convert_tracks(tracks)
        else:
            # Fallback to simple IoU matching
            return self._simple_update(detections, frame)

    def _convert_tracks(self, tracks: np.ndarray) -> List[Track]:
        """Convert tracker output to Track objects."""
        result = []
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, cls, _ = track
            track_id = int(track_id)

            if track_id not in self.tracks:
                self.tracks[track_id] = Track(
                    track_id=track_id,
                    bbox=np.array([x1, y1, x2, y2]),
                    confidence=conf,
                    class_id=int(cls),
                    features=np.zeros(512),
                    state=TrackState.CONFIRMED,
                )
            else:
                self.tracks[track_id].update(
                    np.array([x1, y1, x2, y2]),
                    conf,
                    np.zeros(512),
                )

            result.append(self.tracks[track_id])

        return result

    def _simple_update(
        self,
        detections: np.ndarray,
        frame: np.ndarray,
    ) -> List[Track]:
        """Simple IoU-based tracking fallback."""
        if len(detections) == 0:
            for track in self.tracks.values():
                track.mark_missed()
            return self._get_active_tracks()

        # Match detections to existing tracks by IoU
        matched = set()
        for track_id, track in self.tracks.items():
            best_iou = 0
            best_det_idx = -1

            for i, det in enumerate(detections):
                if i in matched:
                    continue
                iou = self._compute_iou(track.bbox, det[:4])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_det_idx = i

            if best_det_idx >= 0:
                det = detections[best_det_idx]
                track.update(det[:4], det[4], np.zeros(512))
                matched.add(best_det_idx)
            else:
                track.mark_missed()

        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched:
                self.tracks[self.next_id] = Track(
                    track_id=self.next_id,
                    bbox=det[:4],
                    confidence=det[4],
                    class_id=int(det[5]) if len(det) > 5 else 0,
                    features=np.zeros(512),
                )
                self.next_id += 1

        return self._get_active_tracks()

    def _get_active_tracks(self) -> List[Track]:
        """Get list of active (non-deleted) tracks."""
        active = []
        deleted_ids = []

        for track_id, track in self.tracks.items():
            if track.is_deleted():
                deleted_ids.append(track_id)
            elif track.state == TrackState.CONFIRMED:
                active.append(track)

        for track_id in deleted_ids:
            del self.tracks[track_id]

        return active

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0
        if self._tracker is not None:
            self._tracker.reset()

    def get_track(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        return self.tracks.get(track_id)

    def get_track_history(self, track_id: int) -> List[np.ndarray]:
        """Get bounding box history for a track."""
        track = self.tracks.get(track_id)
        return track.history if track else []


def extract_head_crop(
    frame: np.ndarray,
    body_bbox: np.ndarray,
    head_ratio: float = 0.3,
    expand_ratio: float = 1.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract head region from body bounding box.

    Args:
        frame: Video frame (H, W, 3)
        body_bbox: Body bounding box [x1, y1, x2, y2]
        head_ratio: Ratio of body height to use as head region
        expand_ratio: Expansion ratio for head crop

    Returns:
        head_crop: Cropped head image
        head_bbox: Head bounding box [x1, y1, x2, y2]
    """
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = body_bbox.astype(int)

    body_w = x2 - x1
    body_h = y2 - y1

    # Estimate head region (top portion of body)
    head_h = int(body_h * head_ratio)
    head_w = int(head_h * 0.8)  # Head is roughly as wide as tall

    # Center head horizontally
    head_cx = (x1 + x2) // 2
    head_x1 = head_cx - head_w // 2
    head_x2 = head_cx + head_w // 2

    # Head at top of body
    head_y1 = y1
    head_y2 = y1 + head_h

    # Expand crop
    exp_w = int(head_w * (expand_ratio - 1) / 2)
    exp_h = int(head_h * (expand_ratio - 1) / 2)

    head_x1 = max(0, head_x1 - exp_w)
    head_x2 = min(W, head_x2 + exp_w)
    head_y1 = max(0, head_y1 - exp_h)
    head_y2 = min(H, head_y2 + exp_h)

    head_bbox = np.array([head_x1, head_y1, head_x2, head_y2])
    head_crop = frame[head_y1:head_y2, head_x1:head_x2]

    return head_crop, head_bbox
