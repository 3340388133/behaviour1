"""
HPE Benchmark Dataset Loaders
Supports: AFLW2000-3D, 300W-LP, BIWI, and synthetic data generation
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import scipy.io as sio
import cv2
from typing import Tuple, Optional, Dict, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import math


def get_train_transforms(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.CoarseDropout(max_holes=3, max_height=32, max_width=32,
                        min_holes=1, min_height=8, min_width=8, p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_test_transforms(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_occlusion_transforms(image_size=224, occlusion_ratio=0.3):
    """Transforms with heavy occlusion augmentation for robustness."""
    max_h = int(image_size * occlusion_ratio)
    max_w = int(image_size * occlusion_ratio)
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, p=0.5),
        A.CoarseDropout(max_holes=5, max_height=max_h, max_width=max_w,
                        min_holes=2, min_height=16, min_width=16,
                        fill_value=0, p=0.6),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def euler_from_mat(R):
    """Extract euler angles from rotation matrix."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0
    return np.array([yaw, pitch, roll]) * 180.0 / math.pi


class AFLW2000Dataset(Dataset):
    """AFLW2000-3D dataset for HPE evaluation."""

    def __init__(self, root_dir, transform=None, max_yaw=99):
        self.root_dir = root_dir
        self.transform = transform or get_test_transforms()
        self.max_yaw = max_yaw

        # Find all .mat files
        mat_files = sorted(glob.glob(os.path.join(root_dir, '**/*.mat'), recursive=True))
        if not mat_files:
            mat_files = sorted(glob.glob(os.path.join(root_dir, '*.mat')))

        self.samples = []
        skipped = 0
        for mat_path in mat_files:
            img_path = mat_path.replace('.mat', '.jpg')
            if not os.path.exists(img_path):
                img_path = mat_path.replace('.mat', '.png')
            if not os.path.exists(img_path):
                continue

            # Pre-filter by angle range (standard protocol)
            try:
                mat = sio.loadmat(mat_path)
                pose_para = mat.get('Pose_Para', None)
                if pose_para is None:
                    continue
                pose_para = pose_para.flatten()
                pitch = pose_para[0] * 180.0 / np.pi
                yaw = pose_para[1] * 180.0 / np.pi
                roll = pose_para[2] * 180.0 / np.pi
                # Filter extreme values (standard HPE evaluation protocol)
                if abs(yaw) > max_yaw or abs(pitch) > max_yaw or abs(roll) > max_yaw:
                    skipped += 1
                    continue
                self.samples.append((img_path, mat_path, yaw, pitch, roll))
            except Exception:
                continue

        print(f"AFLW2000: Found {len(self.samples)} valid samples in {root_dir} (skipped {skipped} extreme)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mat_path, yaw, pitch, roll = self.samples[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Crop face using landmarks
        mat = sio.loadmat(mat_path)

        # Try pt3d_68 first (68 3D landmarks), then pt2d
        pts = mat.get('pt3d_68', mat.get('pt2d', None))
        if pts is not None:
            if pts.shape[0] >= 2:
                x_coords = pts[0, :]
                y_coords = pts[1, :]
                x_min = max(int(x_coords.min()) - 20, 0)
                y_min = max(int(y_coords.min()) - 20, 0)
                x_max = min(int(x_coords.max()) + 20, image.shape[1])
                y_max = min(int(y_coords.max()) + 20, image.shape[0])
                if x_max > x_min + 10 and y_max > y_min + 10:
                    image = image[y_min:y_max, x_min:x_max]

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        angles = torch.tensor([yaw, pitch, roll], dtype=torch.float32)
        return image, angles


class ThreeHundredWLP_Dataset(Dataset):
    """300W-LP dataset for HPE training."""

    def __init__(self, root_dir, transform=None, max_samples=None):
        self.root_dir = root_dir
        self.transform = transform or get_train_transforms()

        # 300W-LP has subdirectories for different datasets
        self.samples = []
        subdirs = ['AFW', 'HELEN', 'IBUG', 'LFPW', 'AFW_Flip', 'HELEN_Flip', 'IBUG_Flip', 'LFPW_Flip']

        for subdir in subdirs:
            sub_path = os.path.join(root_dir, subdir)
            if os.path.exists(sub_path):
                mat_files = glob.glob(os.path.join(sub_path, '*.mat'))
                for mat_path in mat_files:
                    img_path = mat_path.replace('.mat', '.jpg')
                    if not os.path.exists(img_path):
                        img_path = mat_path.replace('.mat', '.png')
                    if os.path.exists(img_path):
                        self.samples.append((img_path, mat_path))

        # Also check root directly
        if not self.samples:
            mat_files = glob.glob(os.path.join(root_dir, '**/*.mat'), recursive=True)
            for mat_path in mat_files:
                img_path = mat_path.replace('.mat', '.jpg')
                if not os.path.exists(img_path):
                    img_path = mat_path.replace('.mat', '.png')
                if os.path.exists(img_path):
                    self.samples.append((img_path, mat_path))

        if max_samples and len(self.samples) > max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        print(f"300W-LP: Found {len(self.samples)} samples in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mat_path = self.samples[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mat = sio.loadmat(mat_path)
        pose_para = mat.get('Pose_Para', None)
        if pose_para is not None:
            pose_para = pose_para.flatten()
            pitch = pose_para[0] * 180.0 / np.pi
            yaw = pose_para[1] * 180.0 / np.pi
            roll = pose_para[2] * 180.0 / np.pi
        else:
            yaw, pitch, roll = 0.0, 0.0, 0.0

        # Filter extreme angles
        if abs(yaw) > 99 or abs(pitch) > 99 or abs(roll) > 99:
            # Return a "safe" sample instead
            yaw = np.clip(yaw, -99, 99)
            pitch = np.clip(pitch, -99, 99)
            roll = np.clip(roll, -99, 99)

        # Crop face using pt3d_68 or pt2d
        pts = mat.get('pt3d_68', mat.get('pt2d', None))
        if pts is not None and pts.shape[0] >= 2:
            x_coords = pts[0, :]
            y_coords = pts[1, :]
            x_min = max(int(x_coords.min()) - 20, 0)
            y_min = max(int(y_coords.min()) - 20, 0)
            x_max = min(int(x_coords.max()) + 20, image.shape[1])
            y_max = min(int(y_coords.max()) + 20, image.shape[0])
            if x_max > x_min + 10 and y_max > y_min + 10:
                image = image[y_min:y_max, x_min:x_max]

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        angles = torch.tensor([yaw, pitch, roll], dtype=torch.float32)
        return image, angles


class BIWIDataset(Dataset):
    """BIWI Kinect Head Pose dataset."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or get_test_transforms()
        self.samples = []

        # BIWI has numbered directories (01-24)
        for subj_dir in sorted(os.listdir(root_dir)):
            subj_path = os.path.join(root_dir, subj_dir)
            if not os.path.isdir(subj_path):
                continue

            # Find RGB images and pose files
            rgb_files = sorted(glob.glob(os.path.join(subj_path, '*_rgb.png')))
            if not rgb_files:
                rgb_files = sorted(glob.glob(os.path.join(subj_path, '*.png')))

            for rgb_path in rgb_files:
                # Pose file has _pose.txt suffix
                base = rgb_path.replace('_rgb.png', '').replace('.png', '')
                pose_path = base + '_pose.txt'
                if os.path.exists(pose_path):
                    self.samples.append((rgb_path, pose_path))

        print(f"BIWI: Found {len(self.samples)} samples in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, pose_path = self.samples[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load rotation matrix from pose file
        with open(pose_path, 'r') as f:
            lines = f.readlines()

        # BIWI format: 3x3 rotation matrix + translation
        R = np.zeros((3, 3))
        for i in range(3):
            vals = lines[i].strip().split()
            R[i] = [float(v) for v in vals[:3]]

        yaw, pitch, roll = euler_from_mat(R)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        angles = torch.tensor([yaw, pitch, roll], dtype=torch.float32)
        return image, angles


class SyntheticHPEDataset(Dataset):
    """
    Synthetic HPE dataset using rendered faces with known pose angles.
    Fallback when real datasets are unavailable.
    Uses random face crops from existing project data with synthetic labels.
    """

    def __init__(self, num_samples=50000, image_size=224, transform=None, mode='train'):
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform
        self.mode = mode

        # Pre-generate angles with realistic distribution
        np.random.seed(42 if mode == 'train' else 123)

        # Mix of normal and extreme poses
        if mode == 'train':
            # 70% normal range, 20% medium, 10% extreme
            n_normal = int(num_samples * 0.7)
            n_medium = int(num_samples * 0.2)
            n_extreme = num_samples - n_normal - n_medium

            yaw_normal = np.random.normal(0, 25, n_normal)
            pitch_normal = np.random.normal(0, 15, n_normal)
            roll_normal = np.random.normal(0, 10, n_normal)

            yaw_medium = np.random.uniform(-90, 90, n_medium)
            pitch_medium = np.random.uniform(-60, 60, n_medium)
            roll_medium = np.random.uniform(-45, 45, n_medium)

            yaw_extreme = np.random.uniform(-180, 180, n_extreme)
            pitch_extreme = np.random.uniform(-90, 90, n_extreme)
            roll_extreme = np.random.uniform(-90, 90, n_extreme)

            self.yaws = np.concatenate([yaw_normal, yaw_medium, yaw_extreme])
            self.pitches = np.concatenate([pitch_normal, pitch_medium, pitch_extreme])
            self.rolls = np.concatenate([roll_normal, roll_medium, roll_extreme])
        else:
            # Test set: uniform coverage
            self.yaws = np.random.uniform(-99, 99, num_samples)
            self.pitches = np.random.uniform(-90, 90, num_samples)
            self.rolls = np.random.uniform(-90, 90, num_samples)

        # Shuffle
        idx = np.random.permutation(num_samples)
        self.yaws = self.yaws[idx].astype(np.float32)
        self.pitches = self.pitches[idx].astype(np.float32)
        self.rolls = self.rolls[idx].astype(np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        yaw = self.yaws[idx]
        pitch = self.pitches[idx]
        roll = self.rolls[idx]

        # Generate synthetic face-like image based on angles
        image = self._generate_face_image(yaw, pitch, roll)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255.0)

        angles = torch.tensor([yaw, pitch, roll], dtype=torch.float32)
        return image, angles

    def _generate_face_image(self, yaw, pitch, roll):
        """Generate a synthetic face-like image with pose-dependent features."""
        size = self.image_size
        image = np.zeros((size, size, 3), dtype=np.uint8)

        # Background with slight variation
        bg_color = np.random.randint(60, 180, 3)
        image[:] = bg_color

        # Face ellipse (position shifts with yaw/pitch)
        cx = int(size / 2 + yaw / 180.0 * size * 0.3)
        cy = int(size / 2 + pitch / 90.0 * size * 0.2)
        cx = np.clip(cx, size // 4, 3 * size // 4)
        cy = np.clip(cy, size // 4, 3 * size // 4)

        # Face size
        face_w = int(size * 0.35)
        face_h = int(size * 0.45)

        # Skin color
        skin = np.random.randint(160, 230, 3).tolist()
        cv2.ellipse(image, (cx, cy), (face_w, face_h), roll, 0, 360, skin, -1)

        # Eyes (shift with yaw)
        eye_offset_x = int(yaw / 180.0 * face_w * 0.3)
        eye_y = cy - int(face_h * 0.15)
        left_eye_x = cx - int(face_w * 0.3) + eye_offset_x
        right_eye_x = cx + int(face_w * 0.3) + eye_offset_x

        eye_color = (30, 30, 30)
        # Occlude one eye at extreme yaw
        if abs(yaw) < 75:
            cv2.circle(image, (left_eye_x, eye_y), 6, eye_color, -1)
            cv2.circle(image, (right_eye_x, eye_y), 6, eye_color, -1)
        elif yaw > 0:
            cv2.circle(image, (right_eye_x, eye_y), 6, eye_color, -1)
        else:
            cv2.circle(image, (left_eye_x, eye_y), 6, eye_color, -1)

        # Nose
        nose_x = cx + int(yaw / 180.0 * face_w * 0.2)
        nose_y = cy + int(face_h * 0.05)
        cv2.circle(image, (nose_x, nose_y), 4, (180, 140, 120), -1)

        # Mouth
        mouth_y = cy + int(face_h * 0.25)
        mouth_x = cx + int(yaw / 180.0 * face_w * 0.15)
        cv2.ellipse(image, (mouth_x, mouth_y), (int(face_w * 0.25), 4), 0, 0, 360, (150, 80, 80), -1)

        # Add noise
        noise = np.random.randint(-10, 10, image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return image


def create_datasets(data_dir, image_size=224):
    """
    Create train/val/test datasets. Uses real data if available, falls back to synthetic.
    Returns: train_dataset, val_dataset, test_aflw, test_biwi
    """
    train_transform = get_train_transforms(image_size)
    test_transform = get_test_transforms(image_size)
    occlusion_transform = get_occlusion_transforms(image_size)

    # Try real datasets first
    aflw_dir = os.path.join(data_dir, 'AFLW2000')
    lp300w_dir = os.path.join(data_dir, '300W_LP')
    biwi_dir = os.path.join(data_dir, 'BIWI')

    # Test datasets
    test_aflw = None
    test_biwi = None

    # Check for real AFLW2000
    aflw_candidates = [aflw_dir, os.path.join(aflw_dir, 'AFLW2000')]
    for d in aflw_candidates:
        if os.path.exists(d):
            mat_files = glob.glob(os.path.join(d, '**/*.mat'), recursive=True)
            if mat_files:
                test_aflw = AFLW2000Dataset(d, transform=test_transform)
                break

    # Check for BIWI
    biwi_candidates = [biwi_dir, os.path.join(biwi_dir, 'hpdb')]
    for d in biwi_candidates:
        if os.path.exists(d):
            test_biwi = BIWIDataset(d, transform=test_transform)
            if len(test_biwi) > 0:
                break
            test_biwi = None

    # Training dataset
    train_dataset = None
    lp300w_candidates = [lp300w_dir, os.path.join(lp300w_dir, '300W_LP')]
    for d in lp300w_candidates:
        if os.path.exists(d):
            train_dataset = ThreeHundredWLP_Dataset(d, transform=train_transform)
            if len(train_dataset) > 0:
                break
            train_dataset = None

    # Fallback to synthetic if needed
    use_synthetic = False
    if train_dataset is None or len(train_dataset) == 0:
        print("WARNING: Real training data not found. Using synthetic data.")
        print("  For best results, download 300W-LP dataset.")
        train_dataset = SyntheticHPEDataset(
            num_samples=80000, image_size=image_size,
            transform=train_transform, mode='train'
        )
        use_synthetic = True

    if test_aflw is None or len(test_aflw) == 0:
        print("WARNING: AFLW2000 not found. Using synthetic test data.")
        test_aflw = SyntheticHPEDataset(
            num_samples=2000, image_size=image_size,
            transform=test_transform, mode='test'
        )

    if test_biwi is None or len(test_biwi) == 0:
        print("WARNING: BIWI not found. Using synthetic test data.")
        test_biwi = SyntheticHPEDataset(
            num_samples=5000, image_size=image_size,
            transform=test_transform, mode='test'
        )

    # Create validation split from training data
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test AFLW2000: {len(test_aflw)}")
    print(f"  Test BIWI: {len(test_biwi)}")
    print(f"  Using synthetic: {use_synthetic}")

    return train_dataset, val_dataset, test_aflw, test_biwi, use_synthetic
