#!/usr/bin/env python
"""
头部姿态人工标注工具 (从零开始)
初始角度为 0，需要用户手动调整
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def draw_pose_axes(img, yaw, pitch, roll, center=None, size=50):
    """在图像上绘制姿态坐标轴"""
    if center is None:
        h, w = img.shape[:2]
        center = (w // 2, h // 2)

    yaw_rad, pitch_rad, roll_rad = np.radians([yaw, pitch, roll])

    Rx = np.array([[1, 0, 0], [0, np.cos(pitch_rad), -np.sin(pitch_rad)], [0, np.sin(pitch_rad), np.cos(pitch_rad)]])
    Ry = np.array([[np.cos(yaw_rad), 0, np.sin(yaw_rad)], [0, 1, 0], [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]])
    Rz = np.array([[np.cos(roll_rad), -np.sin(roll_rad), 0], [np.sin(roll_rad), np.cos(roll_rad), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx

    axes = np.array([[size, 0, 0], [0, size, 0], [0, 0, size]]).T
    axes_rot = R @ axes

    cx, cy = center
    cv2.line(img, (cx, cy), (int(cx + axes_rot[0, 0]), int(cy - axes_rot[1, 0])), (0, 0, 255), 2)  # X-红
    cv2.line(img, (cx, cy), (int(cx + axes_rot[0, 1]), int(cy - axes_rot[1, 1])), (0, 255, 0), 2)  # Y-绿
    cv2.line(img, (cx, cy), (int(cx + axes_rot[0, 2]), int(cy - axes_rot[1, 2])), (255, 0, 0), 2)  # Z-蓝
    return img


def annotate_from_zero(data_dir: str, num_samples: int = 30, output_csv: str = "data/human_gt.csv"):
    """
    从零开始标注

    操作:
    - A/D: yaw ±5°
    - W/S: pitch ±5°
    - Q/E: roll ±5°
    - Space: 保存并下一张
    - R: 重置为 0
    - ESC: 退出
    """
    data_dir = Path(data_dir)

    # 收集所有人脸图像
    face_files = sorted(list(data_dir.glob("faces/**/*.jpg")))
    np.random.seed(123)
    sampled = np.random.choice(face_files, min(num_samples, len(face_files)), replace=False)

    print(f"标注 {len(sampled)} 张图像 (初始角度为 0)")
    print("\n操作说明:")
    print("  A/D: yaw ±5° (左右转头)")
    print("  W/S: pitch ±5° (抬头低头)")
    print("  Q/E: roll ±5° (歪头)")
    print("  Space: 保存并下一张")
    print("  R: 重置为 0")
    print("  ESC: 退出保存\n")

    results = []

    for idx, img_path in enumerate(sampled):
        img_orig = cv2.imread(str(img_path))
        if img_orig is None:
            continue

        scale = max(1, 300 // max(img_orig.shape[:2]))
        img_display = cv2.resize(img_orig, None, fx=scale, fy=scale)

        # 初始角度为 0
        yaw, pitch, roll = 0, 0, 0
        step = 5

        while True:
            img_show = img_display.copy()
            draw_pose_axes(img_show, yaw, pitch, roll)

            info = f"[{idx+1}/{len(sampled)}] Yaw:{yaw:+.0f} Pitch:{pitch:+.0f} Roll:{roll:+.0f}"
            cv2.putText(img_show, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img_show, "A/D:Yaw W/S:Pitch Q/E:Roll Space:Save", (10, img_show.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

            cv2.imshow("Pose Annotation (From Zero)", img_show)
            key = cv2.waitKey(0) & 0xFF

            if key == 27:  # ESC
                break
            elif key == 32:  # Space
                rel_path = img_path.relative_to(data_dir)
                results.append({'face_path': str(rel_path), 'yaw': yaw, 'pitch': pitch, 'roll': roll})
                print(f"  [{idx+1}] yaw={yaw:+.0f}, pitch={pitch:+.0f}, roll={roll:+.0f}")
                break
            elif key == ord('r') or key == ord('R'):
                yaw, pitch, roll = 0, 0, 0
            elif key == ord('a') or key == ord('A'):
                yaw -= step
            elif key == ord('d') or key == ord('D'):
                yaw += step
            elif key == ord('w') or key == ord('W'):
                pitch += step
            elif key == ord('s') or key == ord('S'):
                pitch -= step
            elif key == ord('q') or key == ord('Q'):
                roll -= step
            elif key == ord('e') or key == ord('E'):
                roll += step

            yaw = max(-180, min(180, yaw))
            pitch = max(-90, min(90, pitch))
            roll = max(-90, min(90, roll))

        if key == 27:
            break

    cv2.destroyAllWindows()

    if results:
        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"\n已保存 {len(results)} 条标注到 {output_csv}")


if __name__ == "__main__":
    annotate_from_zero("data", num_samples=30, output_csv="data/human_gt.csv")
