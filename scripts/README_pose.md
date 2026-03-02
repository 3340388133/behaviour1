# 头部姿态估计 Pipeline

## 快速开始

### 1. 安装依赖

```bash
# 基础依赖
pip install torch torchvision opencv-python numpy

# 人脸检测 (二选一)
pip install retinaface-pytorch
# 或
pip install insightface onnxruntime

# 头部姿态估计
pip install sixdrepnet
```

### 2. 运行

```bash
# 输出JSON
python scripts/head_pose_pipeline.py --video your_video.mp4 --output results.json

# 输出CSV
python scripts/head_pose_pipeline.py --video your_video.mp4 --output results.csv

# 调整置信度阈值
python scripts/head_pose_pipeline.py --video your_video.mp4 --conf 0.6
```

### 3. Python调用

```python
from scripts.head_pose_pipeline import process_video

results = process_video(
    video_path="video.mp4",
    output_path="results.json",
    conf_threshold=0.5
)

for r in results:
    print(f"Frame {r['frame']}: yaw={r['yaw']:.1f}, pitch={r['pitch']:.1f}")
```

---

## 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--conf` | 0.5 | 人脸检测置信度阈值，越高越严格 |
| `iou_threshold` | 0.3 | IOU跟踪阈值，低于此值认为目标丢失 |
| `expand` | 1.2 | 人脸裁剪扩展比例，建议1.1-1.3 |

---

## 输出格式

### JSON示例
```json
{
  "video": "test.mp4",
  "fps": 30.0,
  "total_frames": 300,
  "resolution": [1920, 1080],
  "frames": [
    {
      "frame": 0,
      "timestamp": 0.0,
      "track_id": 1,
      "bbox": [100, 50, 200, 180],
      "face_conf": 0.95,
      "yaw": -15.3,
      "pitch": 5.2,
      "roll": -2.1,
      "pose_conf": 0.9
    }
  ]
}
```

### 角度含义
- **yaw**: 偏航角，左右转头 (负=左转, 正=右转)
- **pitch**: 俯仰角，抬头低头 (负=低头, 正=抬头)
- **roll**: 翻滚角，歪头 (负=左歪, 正=右歪)

---

## 常见问题

**Q: 没有GPU怎么办？**
A: 代码会自动回退到CPU，但速度较慢

**Q: sixdrepnet安装失败？**
A: 代码会自动使用PnP方法回退，精度略低但可用

**Q: 检测不到人脸？**
A: 尝试降低`--conf`参数，如`--conf 0.3`
