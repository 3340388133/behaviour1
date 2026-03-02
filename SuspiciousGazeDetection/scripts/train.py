#!/usr/bin/env python3
"""
Training Script for Suspicious Gaze Detection

Main training script for the complete pipeline.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.head_pose import WHENetPlus
from src.models.temporal import TemporalFusion
from src.models.classifier import SuspiciousGazeClassifier
from src.utils.metrics import MetricsTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Train Suspicious Gaze Detection")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/trained",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict, device: str):
    """Create model from config."""
    # Head pose model
    head_pose_cfg = config.get('model', {}).get('head_pose', {})
    pose_model = WHENetPlus(
        backbone=head_pose_cfg.get('backbone', 'efficientnet_b0'),
        pretrained=head_pose_cfg.get('pretrained', True),
        attention_type=head_pose_cfg.get('attention', {}).get('type', 'self'),
    )

    # Temporal model
    temporal_cfg = config.get('model', {}).get('temporal', {})
    temporal_model = TemporalFusion(
        input_size=256,  # From pose model features
        hidden_size=temporal_cfg.get('lstm', {}).get('hidden_size', 256),
        num_layers=temporal_cfg.get('lstm', {}).get('num_layers', 2),
    )

    # Classifier
    classifier = SuspiciousGazeClassifier(
        pose_dim=6,
        track_dim=128,
        hidden_dim=256,
        num_classes=2,
    )

    # Move to device
    pose_model = pose_model.to(device)
    temporal_model = temporal_model.to(device)
    classifier = classifier.to(device)

    return {
        'pose': pose_model,
        'temporal': temporal_model,
        'classifier': classifier,
    }


def create_optimizer(models: dict, config: dict):
    """Create optimizer for all models."""
    opt_cfg = config.get('training', {}).get('optimizer', {})

    params = []
    for model in models.values():
        params.extend(model.parameters())

    if opt_cfg.get('name', 'adamw') == 'adamw':
        optimizer = torch.optim.AdamW(
            params,
            lr=opt_cfg.get('lr', 1e-4),
            weight_decay=opt_cfg.get('weight_decay', 1e-4),
        )
    else:
        optimizer = torch.optim.Adam(
            params,
            lr=opt_cfg.get('lr', 1e-4),
        )

    return optimizer


def create_scheduler(optimizer, config: dict, steps_per_epoch: int):
    """Create learning rate scheduler."""
    sched_cfg = config.get('training', {}).get('scheduler', {})
    epochs = config.get('training', {}).get('epochs', 100)

    if sched_cfg.get('name') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs * steps_per_epoch,
            eta_min=sched_cfg.get('min_lr', 1e-6),
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30 * steps_per_epoch,
            gamma=0.1,
        )

    return scheduler


def train_epoch(
    models: dict,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    epoch: int,
):
    """Train for one epoch."""
    for model in models.values():
        model.train()

    metrics = MetricsTracker()
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move data to device
        head_crops = batch['head_crops'].to(device)
        poses_gt = batch['poses'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Forward pass through pose model
        pose_output = models['pose'](head_crops)

        # Forward through temporal model
        temporal_output = models['temporal'](pose_output['features'].unsqueeze(1))

        # Classification
        # For training, we use ground truth features as track features (simplified)
        track_features = torch.zeros(len(labels), 128, device=device)
        cls_output = models['classifier'](
            pose_output['features'],
            track_features,
            temporal_output['final'],
        )

        # Compute loss
        cls_loss = criterion(cls_output['logits'], labels)

        # Pose loss (if ground truth available)
        pose_loss = torch.tensor(0.0, device=device)
        if 'yaw' in pose_output:
            pose_pred = torch.stack([
                pose_output['yaw'],
                pose_output['pitch'],
                pose_output['roll']
            ], dim=1)
            pose_loss = nn.MSELoss()(pose_pred, poses_gt)

        total_loss = cls_loss + 0.5 * pose_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Update metrics
        metrics.update_loss(total_loss.item())
        pred_labels = cls_output['logits'].argmax(dim=1).detach().cpu().numpy()
        metrics.update_classification(pred_labels, labels.cpu().numpy())

        pbar.set_postfix({
            'loss': total_loss.item(),
            'lr': scheduler.get_last_lr()[0],
        })

    return metrics.compute()


def validate(
    models: dict,
    dataloader: DataLoader,
    device: str,
):
    """Validate model."""
    for model in models.values():
        model.eval()

    metrics = MetricsTracker()
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            head_crops = batch['head_crops'].to(device)
            labels = batch['labels'].to(device)

            pose_output = models['pose'](head_crops)
            temporal_output = models['temporal'](pose_output['features'].unsqueeze(1))

            track_features = torch.zeros(len(labels), 128, device=device)
            cls_output = models['classifier'](
                pose_output['features'],
                track_features,
                temporal_output['final'],
            )

            loss = criterion(cls_output['logits'], labels)

            metrics.update_loss(loss.item())
            pred_labels = cls_output['logits'].argmax(dim=1).cpu().numpy()
            probs = cls_output['probs'].cpu().numpy()
            metrics.update_classification(pred_labels, labels.cpu().numpy(), probs)

    return metrics.compute()


def save_checkpoint(
    models: dict,
    optimizer,
    epoch: int,
    metrics: dict,
    output_dir: str,
    is_best: bool = False,
):
    """Save training checkpoint."""
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'metrics': metrics,
        'pose_model': models['pose'].state_dict(),
        'temporal_model': models['temporal'].state_dict(),
        'classifier': models['classifier'].state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    # Save latest
    torch.save(checkpoint, os.path.join(output_dir, 'latest.pth'))

    # Save best
    if is_best:
        torch.save(checkpoint, os.path.join(output_dir, 'best.pth'))

    # Save periodic
    if (epoch + 1) % 10 == 0:
        torch.save(checkpoint, os.path.join(output_dir, f'epoch_{epoch+1}.pth'))


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Setup device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print(f"Using device: {device}")

    # Create models
    print("Creating models...")
    models = create_model(config, device)

    # Create optimizer and scheduler
    optimizer = create_optimizer(models, config)

    # TODO: Create actual dataloaders
    # For now, this is a placeholder
    print("\nNote: Actual data loading not implemented yet.")
    print("Please implement dataset class in src/data/datasets/")
    print("\nProject structure created successfully!")
    print(f"Output directory: {args.output}")

    # Print model summaries
    total_params = 0
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        total_params += params
        print(f"{name}: {params:,} parameters")
    print(f"Total: {total_params:,} parameters")


if __name__ == "__main__":
    main()
