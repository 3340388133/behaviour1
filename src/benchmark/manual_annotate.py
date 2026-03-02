#!/usr/bin/env python
"""
头部姿态人工标注工具
显示人脸图像，辅助用户标注 yaw, pitch, roll
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def draw_pose_axes(img, yaw, pitch, roll, center=None, size=50):
    """在图像上绘制姿态坐标轴"""
    if center is None:
        h, w = img.shape[:2]
        center = (w // 2, h // 2)

    # 转换为弧度
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)

    # 旋转矩阵
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    Ry = np.array([
        [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])
    Rz = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad), np.cos(roll_rad), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx

    # 坐标轴端点
    axes = np.array([
        [size, 0, 0],   # X - 红色 (yaw)
        [0, size, 0],   # Y - 绿色 (pitch)
        [0, 0, size]    # Z - 蓝色 (roll)
    ]).T

    axes_rot = R @ axes

    # 绘制坐标轴
    cx, cy = center
    # X轴 - 红色
    cv2.line(img, (cx, cy), (int(cx + axes_rot[0, 0]), int(cy - axes_rot[1, 0])), (0, 0, 255), 2)
    # Y轴 - 绿色
    cv2.line(img, (cx, cy), (int(cx + axes_rot[0, 1]), int(cy - axes_rot[1, 1])), (0, 255, 0), 2)
    # Z轴 - 蓝色
    cv2.line(img, (cx, cy), (int(cx + axes_rot[0, 2]), int(cy - axes_rot[1, 2])), (255, 0, 0), 2)

    return img


def annotate_samples(data_dir: str, input_csv: str, output_csv: str, num_samples: int = 50):
    """
    人工标注工具

    操作说明:
    - 方向键调整角度: ←→ 调整 yaw, ↑↓ 调整 pitch
    - Q/E: 调整 roll
    - Space: 保存当前标注，下一张
    - S: 跳过当前图像
    - R: 重置角度为 0
    - ESC: 退出并保存
    """
    data_dir = Path(data_dir)
    df = pd.read_csv(input_csv)

    # 随机抽取样本
    if len(df) > num_samples:
        df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)

    print(f"标注 {len(df)} 张图像")
    print("\n操作说明:")
    print("  ←→: 调整 yaw (左右转头)")
    print("  ↑↓: 调整 pitch (抬头低头)")
    print("  Q/E: 调整 roll (歪头)")
    print("  Space: 保存并下一张")
    print("  S: 跳过")
    print("  R: 重置为 0")
    print("  ESC: 退出保存")

    results = []
    idx = 0

    while idx < len(df):
        row = df.iloc[idx]
        img_path = data_dir / row['face_path']

        if not img_path.exists():
            print(f"图像不存在: {img_path}")
            idx += 1
            continue

        img_orig = cv2.imread(str(img_path))
        if img_orig is None:
            idx += 1
            continue

        # 放大显示
        scale = max(1, 300 // max(img_orig.shape[:2]))
        img_display = cv2.resize(img_orig, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        # 初始角度 (使用已有预测作为参考)
        yaw = row.get('yaw', 0)
        pitch = row.get('pitch', 0)
        roll = row.get('roll', 0)

        step = 5  # 每次调整 5 度

        while True:
            # 绘制
            img_show = img_display.copy()
            draw_pose_axes(img_show, yaw, pitch, roll)

            # 显示信息
            info = f"[{idx+1}/{len(df)}] Yaw:{yaw:.0f} Pitch:{pitch:.0f} Roll:{roll:.0f}"
            cv2.putText(img_show, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img_show, "Space:Save  S:Skip  R:Reset  ESC:Exit", (10, img_show.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            cv2.imshow("Pose Annotation", img_show)
            key = cv2.waitKey(0) & 0xFF

            if key == 27:  # ESC
                cv2.destroyAllWindows()
                # 保存结果
                if results:
                    result_df = pd.DataFrame(results)
                    result_df.to_csv(output_csv, index=False)
                    print(f"\n已保存 {len(results)} 条标注到 {output_csv}")
                return

            elif key == 32:  # Space - 保存并下一张
                results.append({
                    'face_path': row['face_path'],
                    'yaw': yaw,
                    'pitch': pitch,
                    'roll': roll
                })
                print(f"  保存: yaw={yaw:.0f}, pitch={pitch:.0f}, roll={roll:.0f}")
                idx += 1
                break

            elif key == ord('s') or key == ord('S'):  # 跳过
                idx += 1
                break

            elif key == ord('r') or key == ord('R'):  # 重置
                yaw, pitch, roll = 0, 0, 0

            elif key == 81 or key == 2:  # 左箭头
                yaw -= step
            elif key == 83 or key == 3:  # 右箭头
                yaw += step
            elif key == 82 or key == 0:  # 上箭头
                pitch += step
            elif key == 84 or key == 1:  # 下箭头
                pitch -= step
            elif key == ord('q') or key == ord('Q'):
                roll -= step
            elif key == ord('e') or key == ord('E'):
                roll += step

            # 限制范围
            yaw = max(-90, min(90, yaw))
            pitch = max(-90, min(90, pitch))
            roll = max(-90, min(90, roll))

    cv2.destroyAllWindows()

    # 保存结果
    if results:
        result_df = pd.DataFrame(results)
        result_df.to_csv(output_csv, index=False)
        print(f"\n已保存 {len(results)} 条标注到 {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="头部姿态人工标注工具")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--input", default="data/sample_gt_labeled.csv", help="输入CSV (含预测值作为参考)")
    parser.add_argument("--output", default="data/manual_gt.csv", help="输出CSV")
    parser.add_argument("--num-samples", type=int, default=50, help="标注样本数")

    args = parser.parse_args()

    annotate_samples(args.data_dir, args.input, args.output, args.num_samples)
