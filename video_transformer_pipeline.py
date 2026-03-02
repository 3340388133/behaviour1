"""
端到端视频Transformer行为识别Pipeline
解决跟踪不稳定问题，直接从视频到行为分类

架构：视频片段 → TimeSformer → 行为类别
优势：
  1. 不依赖跟踪质量
  2. 端到端训练，避免误差累积
  3. 时空注意力自动学习重要特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict
from einops import rearrange, repeat
from tqdm import tqdm


# ============================================================================
# 1. 数据集类：从标注生成视频片段
# ============================================================================

class BehaviorVideoDataset(Dataset):
    """
    从behavior.json和frames目录生成训练数据

    每个样本：
      - video_clip: [T, C, H, W] 视频片段
      - label: 行为类别
      - bbox: 目标区域（用于裁剪ROI）
    """

    def __init__(
        self,
        behavior_json_path: str,
        frames_dir: str,
        num_frames: int = 16,
        image_size: int = 224,
        frame_interval: int = 2,  # 每隔几帧采样一次
        mode: str = 'train'
    ):
        self.frames_dir = Path(frames_dir)
        self.num_frames = num_frames
        self.image_size = image_size
        self.frame_interval = frame_interval
        self.mode = mode

        # 加载行为标注
        with open(behavior_json_path) as f:
            data = json.load(f)

        self.video_id = data['video_id']
        self.behaviors = data['behaviors']

        # 标签映射
        self.label_map = {
            'normal': 0,
            'looking_around': 1,
            'unknown': 2
        }

        # 数据增强（训练模式）
        self.use_augmentation = (mode == 'train')

        print(f"加载 {len(self.behaviors)} 个行为样本")
        print(f"标签分布: {data['label_distribution']}")

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        behavior = self.behaviors[idx]

        # 提取信息
        start_frame = behavior['start_frame']
        end_frame = behavior['end_frame']
        label = self.label_map[behavior['primary_label']]

        # 采样帧索引
        frame_indices = self._sample_frame_indices(
            start_frame, end_frame
        )

        # 加载帧
        frames = self._load_frames(frame_indices)

        # 转换为tensor
        clip = torch.from_numpy(frames).float()
        clip = clip.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]

        # 归一化
        clip = clip / 255.0
        clip = self._normalize(clip)

        # Resize
        clip = F.interpolate(
            clip,
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        )

        # 数据增强
        if self.use_augmentation:
            clip = self._augment(clip)

        return clip, label

    def _sample_frame_indices(
        self,
        start_frame: int,
        end_frame: int
    ) -> List[int]:
        """采样帧索引"""
        total_frames = end_frame - start_frame

        # 边界检查：如果duration为0或太短
        if total_frames < 1:
            # 返回重复的start_frame
            return [start_frame] * self.num_frames

        if total_frames < self.num_frames * self.frame_interval:
            # 轨迹太短，重复采样
            available = list(range(start_frame, end_frame))
            if len(available) == 0:
                # 极端情况：返回start_frame
                return [start_frame] * self.num_frames
            indices = np.random.choice(
                available,
                self.num_frames,
                replace=True
            )
            return sorted(indices)
        else:
            # 均匀采样
            step = total_frames // self.num_frames
            indices = [
                start_frame + i * step
                for i in range(self.num_frames)
            ]
            return indices

    def _load_frames(self, frame_indices: List[int]) -> np.ndarray:
        """加载帧图像"""
        frames = []
        for idx in frame_indices:
            frame_path = self.frames_dir / f"{self.video_id}_frame_{idx:06d}.jpg"

            if not frame_path.exists():
                # 如果帧不存在，使用黑帧
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = cv2.imread(str(frame_path))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame)

        return np.array(frames)

    def _normalize(self, clip: torch.Tensor) -> torch.Tensor:
        """归一化（ImageNet统计）"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (clip - mean) / std

    def _augment(self, clip: torch.Tensor) -> torch.Tensor:
        """数据增强"""
        # 随机水平翻转
        if torch.rand(1) > 0.5:
            clip = torch.flip(clip, dims=[3])

        # 随机亮度调整
        if torch.rand(1) > 0.5:
            factor = 0.8 + torch.rand(1) * 0.4  # 0.8-1.2
            clip = clip * factor
            clip = torch.clamp(clip, 0, 1)

        return clip


# ============================================================================
# 2. ROI版本：裁剪目标区域（多人场景）
# ============================================================================

class BehaviorVideoROIDataset(BehaviorVideoDataset):
    """
    带ROI裁剪的版本，用于多人场景
    使用bbox裁剪出目标人物区域
    """

    def __init__(
        self,
        behavior_json_path: str,
        frames_dir: str,
        detections_dir: str,  # 检测结果目录
        **kwargs
    ):
        super().__init__(behavior_json_path, frames_dir, **kwargs)
        self.detections_dir = Path(detections_dir)

        # 加载检测结果
        detection_file = self.detections_dir / self.video_id / 'detections.json'
        with open(detection_file) as f:
            self.detections = json.load(f)

        # 建立帧索引
        self.frame_detections = {
            frame['frame_idx']: frame['detections']
            for frame in self.detections['frames']
        }

    def __getitem__(self, idx):
        behavior = self.behaviors[idx]

        start_frame = behavior['start_frame']
        end_frame = behavior['end_frame']
        track_id = behavior['track_id']
        label = self.label_map[behavior['primary_label']]

        # 采样帧
        frame_indices = self._sample_frame_indices(start_frame, end_frame)

        # 加载帧并裁剪ROI
        frames = []
        for frame_idx in frame_indices:
            # 获取该帧的bbox
            bbox = self._get_bbox(frame_idx, track_id)

            # 加载并裁剪
            frame_path = self.frames_dir / f"{self.video_id}_frame_{frame_idx:06d}.jpg"
            if frame_path.exists():
                frame = cv2.imread(str(frame_path))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 裁剪ROI（扩展一些边界）
                frame = self._crop_roi(frame, bbox, expand_ratio=1.2)
            else:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)

            frames.append(frame)

        frames = np.array(frames)

        # 转换为tensor
        clip = torch.from_numpy(frames).float()
        clip = clip.permute(0, 3, 1, 2)
        clip = clip / 255.0
        clip = self._normalize(clip)

        # Resize
        clip = F.interpolate(
            clip,
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        )

        if self.use_augmentation:
            clip = self._augment(clip)

        return clip, label

    def _get_bbox(self, frame_idx: int, track_id: int) -> List[int]:
        """获取该帧该track的bbox"""
        if frame_idx not in self.frame_detections:
            return [0, 0, 100, 100]  # 默认bbox

        detections = self.frame_detections[frame_idx]

        # 简单策略：使用第一个检测（实际应该匹配track_id）
        if len(detections) > 0:
            return detections[0]['bbox']
        else:
            return [0, 0, 100, 100]

    def _crop_roi(
        self,
        frame: np.ndarray,
        bbox: List[int],
        expand_ratio: float = 1.2
    ) -> np.ndarray:
        """裁剪ROI区域"""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # 扩展bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = (x2 - x1) * expand_ratio, (y2 - y1) * expand_ratio

        x1 = int(max(0, cx - bw / 2))
        y1 = int(max(0, cy - bh / 2))
        x2 = int(min(w, cx + bw / 2))
        y2 = int(min(h, cy + bh / 2))

        # 裁剪
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            roi = np.zeros((224, 224, 3), dtype=np.uint8)

        return roi


# ============================================================================
# 3. 视频Transformer模型
# ============================================================================

class PatchEmbedding3D(nn.Module):
    """3D Patch Embedding for Video"""
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_frames: int = 16,
        tubelet_size: int = 2
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        # 计算patch数量
        self.num_patches_per_frame = (img_size // patch_size) ** 2
        self.num_patches = (num_frames // tubelet_size) * self.num_patches_per_frame

        # 3D卷积进行patch embedding
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size)
        )

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape

        # 调整维度用于3D卷积
        x = x.transpose(1, 2)  # [B, C, T, H, W]

        # Patch embedding
        x = self.proj(x)  # [B, D, T', H', W']

        # 展平为序列
        x = x.flatten(2).transpose(1, 2)  # [B, T'×H'×W', D]

        return x


class DividedSpaceTimeAttention(nn.Module):
    """TimeSformer的分离时空注意力"""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_frames: int = 8,
        num_patches_per_frame: int = 196,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.num_patches_per_frame = num_patches_per_frame

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 时间注意力
        self.temporal_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.temporal_proj = nn.Linear(dim, dim)

        # 空间注意力
        self.spatial_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.spatial_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: [B, N, D] where N = 1 + T × H × W (包含CLS token)
        B, N, D = x.shape
        T = self.num_frames
        HW = self.num_patches_per_frame

        # 分离CLS token
        cls_token = x[:, 0:1, :]  # [B, 1, D]
        x = x[:, 1:, :]  # [B, T×HW, D] 移除CLS token

        # ===== 时间注意力 =====
        # 重排为 [B×HW, T, D]
        x_t = rearrange(x, 'b (t hw) d -> (b hw) t d', t=T, hw=HW)

        # QKV
        qkv = self.temporal_qkv(x_t).reshape(B*HW, T, 3, self.num_heads, D // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B×HW, heads, T, dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_t = (attn @ v).transpose(1, 2).reshape(B*HW, T, D)
        x_t = self.temporal_proj(x_t)
        x_t = self.proj_drop(x_t)

        # 残差连接
        x_t = rearrange(x_t, '(b hw) t d -> b (t hw) d', b=B, hw=HW)
        x = x + x_t

        # ===== 空间注意力 =====
        # 重排为 [B×T, HW, D]
        x_s = rearrange(x, 'b (t hw) d -> (b t) hw d', t=T, hw=HW)

        # QKV
        qkv = self.spatial_qkv(x_s).reshape(B*T, HW, 3, self.num_heads, D // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_s = (attn @ v).transpose(1, 2).reshape(B*T, HW, D)
        x_s = self.spatial_proj(x_s)
        x_s = self.proj_drop(x_s)

        # 残差连接
        x_s = rearrange(x_s, '(b t) hw d -> b (t hw) d', b=B, t=T)
        x = x + x_s

        # 重新添加CLS token
        x = torch.cat([cls_token, x], dim=1)  # [B, 1+T×HW, D]

        return x


class TransformerBlock(nn.Module):
    """标准Transformer Block"""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_frames: int,
        num_patches_per_frame: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DividedSpaceTimeAttention(
            dim=dim,
            num_heads=num_heads,
            num_frames=num_frames,
            num_patches_per_frame=num_patches_per_frame,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VideoTransformerClassifier(nn.Module):
    """
    端到端视频Transformer分类器
    基于TimeSformer架构
    """
    def __init__(
        self,
        num_classes: int = 3,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 16,
        tubelet_size: int = 2,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        # Patch Embedding
        self.patch_embed = PatchEmbedding3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
            num_frames=num_frames,
            tubelet_size=tubelet_size
        )

        num_patches = self.patch_embed.num_patches
        num_patches_per_frame = self.patch_embed.num_patches_per_frame
        num_frame_tokens = num_frames // tubelet_size

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                num_frames=num_frame_tokens,
                num_patches_per_frame=num_patches_per_frame,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])

        # 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化权重
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # [B, N, D]

        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer编码
        for block in self.blocks:
            x = block(x)

        # 分类
        x = self.norm(x)
        cls_output = x[:, 0]  # 使用CLS token
        logits = self.head(cls_output)

        return logits


# ============================================================================
# 4. 训练器
# ============================================================================

class Trainer:
    """训练管理器"""
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        lr: float = 1e-4,
        num_epochs: int = 50,
        save_dir: str = 'checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.05
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs
        )

        # 损失函数（类别不平衡处理）
        self.criterion = nn.CrossEntropyLoss()

        self.best_acc = 0.0

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for videos, labels in pbar:
            videos = videos.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            logits = self.model(videos)
            loss = self.criterion(logits, labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        return total_loss / len(self.train_loader), 100. * correct / total

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for videos, labels in tqdm(self.val_loader, desc='Validation'):
                videos = videos.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(videos)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        return total_loss / len(self.val_loader), acc

    def train(self):
        print(f"开始训练，共 {self.num_epochs} 个epoch")
        print(f"设备: {self.device}")
        print(f"训练样本: {len(self.train_loader.dataset)}")
        print(f"验证样本: {len(self.val_loader.dataset)}")

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")

            # 训练
            train_loss, train_acc = self.train_epoch()

            # 验证
            val_loss, val_acc = self.validate()

            # 学习率调度
            self.scheduler.step()

            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

            # 保存最佳模型
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint(epoch, val_acc, is_best=True)
                print(f"✓ 保存最佳模型 (Acc: {val_acc:.2f}%)")

            # 定期保存
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_acc, is_best=False)

        print(f"\n训练完成！最佳验证精度: {self.best_acc:.2f}%")

    def save_checkpoint(self, epoch, acc, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'acc': acc,
        }

        if is_best:
            path = self.save_dir / 'best_model.pth'
        else:
            path = self.save_dir / f'checkpoint_epoch_{epoch+1}.pth'

        torch.save(checkpoint, path)


if __name__ == '__main__':
    print("视频Transformer Pipeline模块加载成功")
    print("包含组件:")
    print("  - BehaviorVideoDataset: 基础数据集")
    print("  - BehaviorVideoROIDataset: ROI裁剪数据集")
    print("  - VideoTransformerClassifier: TimeSformer模型")
    print("  - Trainer: 训练管理器")
