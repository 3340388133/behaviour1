"""
Evaluation Metrics

Metrics for evaluating head pose estimation and behavior classification.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


def compute_pose_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute Mean Absolute Error for pose estimation.

    Args:
        pred: Predicted poses (N, 3) - yaw, pitch, roll
        target: Ground truth poses (N, 3)

    Returns:
        Dictionary with MAE for each angle and overall
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Compute absolute errors
    errors = np.abs(pred - target)

    # Handle yaw angle wrapping (-180 to 180)
    yaw_errors = errors[:, 0]
    yaw_errors = np.minimum(yaw_errors, 360 - yaw_errors)

    return {
        'yaw_mae': float(np.mean(yaw_errors)),
        'pitch_mae': float(np.mean(errors[:, 1])),
        'roll_mae': float(np.mean(errors[:, 2])),
        'mean_mae': float(np.mean([
            np.mean(yaw_errors),
            np.mean(errors[:, 1]),
            np.mean(errors[:, 2]),
        ])),
    }


def compute_geodesic_error(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Compute geodesic (angular) error between rotations.

    More accurate than MAE for rotation estimation.

    Args:
        pred: Predicted poses (N, 3) - yaw, pitch, roll in degrees
        target: Ground truth poses (N, 3)

    Returns:
        Mean geodesic error in degrees
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Convert to radians
    pred_rad = np.deg2rad(pred)
    target_rad = np.deg2rad(target)

    # Convert to rotation matrices
    R_pred = euler_to_rotation_matrix(pred_rad)
    R_target = euler_to_rotation_matrix(target_rad)

    # Compute relative rotation
    R_diff = np.matmul(R_pred, R_target.transpose(0, 2, 1))

    # Extract angle from rotation matrix
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    trace = np.clip(trace, -1 + 1e-6, 3 - 1e-6)
    angle = np.arccos((trace - 1) / 2)

    return float(np.mean(np.rad2deg(angle)))


def euler_to_rotation_matrix(angles: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles to rotation matrices.

    Args:
        angles: Euler angles (N, 3) in radians

    Returns:
        Rotation matrices (N, 3, 3)
    """
    yaw, pitch, roll = angles[:, 0], angles[:, 1], angles[:, 2]

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    R = np.zeros((len(angles), 3, 3))

    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr

    return R


def compute_classification_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    pred_probs: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics for behavior detection.

    Args:
        pred: Predicted labels (N,)
        target: Ground truth labels (N,)
        pred_probs: Prediction probabilities for AUC (N, 2)

    Returns:
        Dictionary with classification metrics
    """
    metrics = {
        'accuracy': accuracy_score(target, pred),
        'precision': precision_score(target, pred, average='binary', zero_division=0),
        'recall': recall_score(target, pred, average='binary', zero_division=0),
        'f1': f1_score(target, pred, average='binary', zero_division=0),
    }

    # Confusion matrix
    cm = confusion_matrix(target, pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_positive'] = int(tp)
        metrics['true_negative'] = int(tn)
        metrics['false_positive'] = int(fp)
        metrics['false_negative'] = int(fn)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    # AUC if probabilities available
    if pred_probs is not None and len(np.unique(target)) > 1:
        try:
            metrics['auc'] = roc_auc_score(target, pred_probs[:, 1])
        except ValueError:
            metrics['auc'] = 0.0

    return metrics


def compute_tracking_metrics(
    pred_tracks: Dict[int, list],
    gt_tracks: Dict[int, list],
) -> Dict[str, float]:
    """
    Compute multi-object tracking metrics.

    Args:
        pred_tracks: Predicted tracks {track_id: [(frame, bbox), ...]}
        gt_tracks: Ground truth tracks

    Returns:
        Dictionary with MOTA, MOTP, IDF1, etc.
    """
    # Simplified metrics (full implementation would use motmetrics)
    total_gt = sum(len(t) for t in gt_tracks.values())
    total_pred = sum(len(t) for t in pred_tracks.values())

    # ID switches, fragmentations would need frame-by-frame matching
    return {
        'num_gt_tracks': len(gt_tracks),
        'num_pred_tracks': len(pred_tracks),
        'total_gt_detections': total_gt,
        'total_pred_detections': total_pred,
    }


class MetricsTracker:
    """
    Track metrics over training/evaluation.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.pose_errors = {'yaw': [], 'pitch': [], 'roll': []}
        self.predictions = []
        self.targets = []
        self.pred_probs = []
        self.losses = []

    def update_pose(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ):
        """Update pose metrics."""
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        errors = np.abs(pred - target)
        self.pose_errors['yaw'].extend(errors[:, 0].tolist())
        self.pose_errors['pitch'].extend(errors[:, 1].tolist())
        self.pose_errors['roll'].extend(errors[:, 2].tolist())

    def update_classification(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
    ):
        """Update classification metrics."""
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        self.predictions.extend(pred.tolist())
        self.targets.extend(target.tolist())

        if probs is not None:
            if isinstance(probs, torch.Tensor):
                probs = probs.detach().cpu().numpy()
            self.pred_probs.extend(probs.tolist())

    def update_loss(self, loss: float):
        """Update loss tracking."""
        self.losses.append(loss)

    def compute(self) -> Dict[str, float]:
        """Compute all tracked metrics."""
        results = {}

        # Pose metrics
        if self.pose_errors['yaw']:
            results['yaw_mae'] = np.mean(self.pose_errors['yaw'])
            results['pitch_mae'] = np.mean(self.pose_errors['pitch'])
            results['roll_mae'] = np.mean(self.pose_errors['roll'])
            results['mean_pose_mae'] = np.mean([
                results['yaw_mae'],
                results['pitch_mae'],
                results['roll_mae'],
            ])

        # Classification metrics
        if self.predictions and self.targets:
            pred = np.array(self.predictions)
            target = np.array(self.targets)
            probs = np.array(self.pred_probs) if self.pred_probs else None
            results.update(compute_classification_metrics(pred, target, probs))

        # Loss
        if self.losses:
            results['mean_loss'] = np.mean(self.losses)

        return results
