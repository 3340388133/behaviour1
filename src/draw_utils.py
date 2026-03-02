"""
可视化绘制工具 - 3D坐标轴、人脸框、角度显示
"""
import cv2
import numpy as np
from typing import Tuple, Optional


def draw_axis(
    img: np.ndarray,
    yaw: float,
    pitch: float,
    roll: float,
    tdx: float,
    tdy: float,
    size: int = 100
) -> np.ndarray:
    """在图像上绘制3D坐标轴表示头部姿态

    Args:
        img: 输入图像
        yaw: 偏航角（左右转头），度
        pitch: 俯仰角（抬头低头），度
        roll: 翻滚角（歪头），度
        tdx: 坐标轴原点 x
        tdy: 坐标轴原点 y
        size: 坐标轴长度

    Returns:
        绘制后的图像
    """
    # 角度转弧度
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    # X轴 (红色) - 指向右
    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy

    # Y轴 (绿色) - 指向下
    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy

    # Z轴 (蓝色) - 指向前（朝向观察者）
    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy

    # 绘制坐标轴
    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)  # X - 红
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)  # Y - 绿
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 3)  # Z - 蓝

    return img


def draw_face_box(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    track_id: int = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """绘制人脸框

    Args:
        img: 输入图像
        bbox: 边界框 (x1, y1, x2, y2)
        track_id: 跟踪ID（可选）
        color: 框颜色 BGR
        thickness: 线宽

    Returns:
        绘制后的图像
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    if track_id is not None:
        label = f"ID:{track_id}"
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img


def draw_pose_info(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    yaw: float,
    pitch: float,
    roll: float,
    color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """在人脸框下方显示姿态角度信息

    Args:
        img: 输入图像
        bbox: 边界框 (x1, y1, x2, y2)
        yaw: 偏航角
        pitch: 俯仰角
        roll: 翻滚角
        color: 文字颜色

    Returns:
        绘制后的图像
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # 在框下方显示角度
    text_y = y2 + 15
    text = f"Y:{yaw:.1f} P:{pitch:.1f} R:{roll:.1f}"

    # 添加背景使文字更清晰
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(img, (x1, text_y - text_h - 2), (x1 + text_w + 4, text_y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x1 + 2, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return img


def draw_detection_full(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    track_id: int,
    yaw: float,
    pitch: float,
    roll: float,
    is_alert: bool = False,
    axis_size: int = 50
) -> np.ndarray:
    """完整绘制单个检测结果：人脸框 + 3D坐标轴 + 角度信息

    Args:
        img: 输入图像
        bbox: 边界框 (x1, y1, x2, y2)
        track_id: 跟踪ID
        yaw, pitch, roll: 姿态角度
        is_alert: 是否告警状态
        axis_size: 坐标轴大小

    Returns:
        绘制后的图像
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # 颜色：告警红色，正常绿色
    color = (0, 0, 255) if is_alert else (0, 255, 0)

    # 1. 绘制人脸框
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # 2. 绘制ID
    label = f"ID:{track_id}"
    cv2.putText(img, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 3. 绘制3D坐标轴（在人脸框中心）
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    # 根据人脸大小调整坐标轴大小
    face_size = min(x2 - x1, y2 - y1)
    axis_len = max(30, int(face_size * 0.6))
    draw_axis(img, yaw, pitch, roll, cx, cy, axis_len)

    # 4. 绘制角度信息
    draw_pose_info(img, bbox, yaw, pitch, roll)

    # 5. 告警标记
    if is_alert:
        cv2.putText(img, "ALERT!", (x1, y2 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return img
