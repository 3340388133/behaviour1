#!/usr/bin/env python3
"""
快速测试脚本 - 测试基本功能
"""
import sys
print("开始测试...")

# 测试依赖
try:
    import cv2
    print(f"[OK] OpenCV: {cv2.__version__}")
except Exception as e:
    print(f"[FAIL] OpenCV: {e}")

try:
    import torch
    print(f"[OK] PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
except Exception as e:
    print(f"[FAIL] PyTorch: {e}")

try:
    from ultralytics import YOLO
    print(f"[OK] Ultralytics YOLO")
except Exception as e:
    print(f"[FAIL] Ultralytics: {e}")

try:
    import numpy as np
    print(f"[OK] NumPy: {np.__version__}")
except Exception as e:
    print(f"[FAIL] NumPy: {e}")

# 测试视频读取
print("\n测试视频读取...")
video_path = "/root/autodl-tmp/behaviour/data/raw_videos/正机位/1.14zz-2.mp4"
cap = cv2.VideoCapture(video_path)
if cap.isOpened():
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] 视频: {w}x{h}, {fps}fps, {frames}帧")
    cap.release()
else:
    print(f"[FAIL] 无法打开视频")

# 测试YOLO
print("\n测试YOLOv8检测...")
try:
    model = YOLO("/root/autodl-tmp/behaviour/SuspiciousGazeDetection/yolov8m.pt")

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if ret:
        results = model(frame, verbose=False, classes=[0])
        boxes = results[0].boxes
        print(f"[OK] 检测到 {len(boxes)} 个人")
    else:
        print("[FAIL] 无法读取帧")
except Exception as e:
    print(f"[FAIL] YOLO检测: {e}")

print("\n测试完成!")
