#!/usr/bin/env python3
"""
启动 DeepLabCut 标注 GUI
支持 napari-deeplabcut 或备用标注界面
"""

import os
import sys

# 设置环境变量
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # 或 'offscreen' 用于无显示环境

PROJECT_PATH = "/root/behaviour/dataset_root/head_behavior-labeler"
CONFIG_PATH = f"{PROJECT_PATH}/config.yaml"

def check_display():
    """检查是否有显示环境"""
    display = os.environ.get('DISPLAY')
    if not display:
        print("警告: 未检测到 DISPLAY 环境变量")
        print("如果是远程服务器，请使用 X11 转发:")
        print("  ssh -X user@server")
        print("或使用 VNC/远程桌面连接")
        return False
    return True

def launch_napari_deeplabcut():
    """使用 napari-deeplabcut 启动标注"""
    try:
        import napari
        from napari_deeplabcut import DeepLabCutWidget

        print("启动 napari-deeplabcut 标注界面...")
        print(f"项目路径: {PROJECT_PATH}")
        print(f"配置文件: {CONFIG_PATH}")

        viewer = napari.Viewer()
        widget = DeepLabCutWidget(viewer)
        viewer.window.add_dock_widget(widget, area='right')

        # 自动加载项目
        widget.config_path.value = CONFIG_PATH

        print("\n" + "=" * 50)
        print("标注指南:")
        print("=" * 50)
        print("1. 在右侧面板中选择视频/帧目录")
        print("2. 点击 'Load' 加载帧")
        print("3. 对每个人依次标注以下关键点:")
        print("   - head_top (头顶)")
        print("   - forehead (额头)")
        print("   - nose (鼻尖)")
        print("   - left_ear (左耳)")
        print("   - right_ear (右耳)")
        print("   - chin (下巴)")
        print("4. 完成后点击 'Save' 保存")
        print("5. 使用滑块切换到下一帧")
        print("=" * 50)

        napari.run()

    except Exception as e:
        print(f"napari-deeplabcut 启动失败: {e}")
        print("尝试使用备用方案...")
        launch_simple_labeler()

def launch_simple_labeler():
    """简易标注工具（使用 OpenCV）"""
    import cv2
    import json
    import yaml
    from pathlib import Path

    # 加载配置
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    bodyparts = config['bodyparts']
    project_path = Path(PROJECT_PATH)
    labeled_data_dir = project_path / "labeled-data"

    # 获取所有待标注的帧
    all_frames = []
    for video_dir in sorted(labeled_data_dir.iterdir()):
        if video_dir.is_dir():
            frames = sorted(list(video_dir.glob("*.jpg")))
            for f in frames:
                all_frames.append((video_dir.name, f))

    if not all_frames:
        print("未找到待标注的帧!")
        return

    print(f"\n共有 {len(all_frames)} 帧待标注")
    print("=" * 50)
    print("标注操作说明:")
    print("  - 鼠标左键: 标注当前关键点")
    print("  - 鼠标右键: 跳过当前关键点 (不可见)")
    print("  - 'n': 下一帧")
    print("  - 'p': 上一帧")
    print("  - 's': 保存标注")
    print("  - 'q': 退出")
    print("  - '+'/'-': 切换个体 (多人场景)")
    print("=" * 50)

    # 标注状态
    current_frame_idx = 0
    current_individual = 0
    current_bodypart_idx = 0
    max_individuals = 5

    # 加载或创建标注数据
    annotations_file = project_path / "annotations.json"
    if annotations_file.exists():
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
    else:
        annotations = {}

    def get_frame_key(video_name, frame_path):
        return f"{video_name}/{frame_path.name}"

    def draw_frame():
        video_name, frame_path = all_frames[current_frame_idx]
        img = cv2.imread(str(frame_path))

        # 缩放显示（4K太大）
        scale = 0.5
        display_img = cv2.resize(img, None, fx=scale, fy=scale)

        # 绘制已标注的点
        frame_key = get_frame_key(video_name, frame_path)
        if frame_key in annotations:
            for ind_idx, ind_data in enumerate(annotations[frame_key]):
                color = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                        (255, 255, 0), (0, 255, 255)][ind_idx % 5]
                for bp_idx, (x, y) in enumerate(ind_data):
                    if x is not None and y is not None:
                        px, py = int(x * scale), int(y * scale)
                        cv2.circle(display_img, (px, py), 5, color, -1)
                        cv2.putText(display_img, bodyparts[bp_idx][:3],
                                   (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.4, color, 1)

        # 显示状态信息
        status = f"帧: {current_frame_idx + 1}/{len(all_frames)} | "
        status += f"个体: {current_individual + 1}/{max_individuals} | "
        status += f"关键点: {bodyparts[current_bodypart_idx]}"
        cv2.putText(display_img, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_img, f"视频: {video_name}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return display_img, scale

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_bodypart_idx, current_individual

        video_name, frame_path = all_frames[current_frame_idx]
        frame_key = get_frame_key(video_name, frame_path)
        scale = param

        if event == cv2.EVENT_LBUTTONDOWN:
            # 记录标注点（转换回原始坐标）
            orig_x, orig_y = int(x / scale), int(y / scale)

            if frame_key not in annotations:
                annotations[frame_key] = [[None] * len(bodyparts) for _ in range(max_individuals)]

            annotations[frame_key][current_individual][current_bodypart_idx] = (orig_x, orig_y)
            print(f"标注: ind{current_individual + 1}.{bodyparts[current_bodypart_idx]} = ({orig_x}, {orig_y})")

            # 移动到下一个关键点
            current_bodypart_idx = (current_bodypart_idx + 1) % len(bodyparts)

        elif event == cv2.EVENT_RBUTTONDOWN:
            # 跳过当前关键点（标记为不可见）
            if frame_key not in annotations:
                annotations[frame_key] = [[None] * len(bodyparts) for _ in range(max_individuals)]

            annotations[frame_key][current_individual][current_bodypart_idx] = (None, None)
            print(f"跳过: ind{current_individual + 1}.{bodyparts[current_bodypart_idx]} (不可见)")

            current_bodypart_idx = (current_bodypart_idx + 1) % len(bodyparts)

    def save_annotations():
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        print(f"标注已保存到: {annotations_file}")

        # 同时转换为DeepLabCut格式
        convert_to_dlc_format()

    def convert_to_dlc_format():
        """转换为DeepLabCut的CSV/H5格式"""
        import pandas as pd

        scorer = "labeler"
        individuals = [f"ind{i+1}" for i in range(max_individuals)]

        for video_dir in labeled_data_dir.iterdir():
            if not video_dir.is_dir():
                continue

            frames = sorted(list(video_dir.glob("*.jpg")))
            if not frames:
                continue

            # 创建数据
            data = {}
            for frame_path in frames:
                frame_key = get_frame_key(video_dir.name, frame_path)
                row_idx = f"labeled-data/{video_dir.name}/{frame_path.name}"

                if frame_key in annotations:
                    for ind_idx, ind_data in enumerate(annotations[frame_key]):
                        for bp_idx, point in enumerate(ind_data):
                            if point and point[0] is not None:
                                col = (scorer, individuals[ind_idx], bodyparts[bp_idx], "x")
                                if col not in data:
                                    data[col] = {}
                                data[col][row_idx] = point[0]

                                col = (scorer, individuals[ind_idx], bodyparts[bp_idx], "y")
                                if col not in data:
                                    data[col] = {}
                                data[col][row_idx] = point[1]

            if data:
                df = pd.DataFrame(data)
                df.columns = pd.MultiIndex.from_tuples(df.columns,
                    names=["scorer", "individuals", "bodyparts", "coords"])

                csv_path = video_dir / f"CollectedData_{scorer}.csv"
                df.to_csv(csv_path)
                print(f"已保存: {csv_path}")

    # 主循环
    cv2.namedWindow("Head Behavior Labeling", cv2.WINDOW_NORMAL)

    while True:
        display_img, scale = draw_frame()
        cv2.imshow("Head Behavior Labeling", display_img)
        cv2.setMouseCallback("Head Behavior Labeling", mouse_callback, scale)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            save_annotations()
            break
        elif key == ord('n'):
            current_frame_idx = min(current_frame_idx + 1, len(all_frames) - 1)
            current_bodypart_idx = 0
        elif key == ord('p'):
            current_frame_idx = max(current_frame_idx - 1, 0)
            current_bodypart_idx = 0
        elif key == ord('s'):
            save_annotations()
        elif key == ord('+') or key == ord('='):
            current_individual = min(current_individual + 1, max_individuals - 1)
            current_bodypart_idx = 0
            print(f"切换到个体 {current_individual + 1}")
        elif key == ord('-'):
            current_individual = max(current_individual - 1, 0)
            current_bodypart_idx = 0
            print(f"切换到个体 {current_individual + 1}")

    cv2.destroyAllWindows()

def main():
    print("=" * 50)
    print("DeepLabCut 头部行为标注工具")
    print("=" * 50)

    # 检查项目是否存在
    if not os.path.exists(CONFIG_PATH):
        print(f"错误: 项目配置文件不存在: {CONFIG_PATH}")
        print("请先运行: python3 /root/behaviour/scripts/setup_deeplabcut_project.py")
        sys.exit(1)

    # 检查显示环境
    has_display = check_display()

    if has_display:
        print("\n选择标注工具:")
        print("  1. napari-deeplabcut (推荐，功能完整)")
        print("  2. OpenCV简易标注器 (轻量，兼容性好)")
        print()

        # 默认尝试napari，失败则用OpenCV
        try:
            launch_napari_deeplabcut()
        except Exception as e:
            print(f"\nnapari启动失败: {e}")
            print("自动切换到OpenCV标注器...")
            launch_simple_labeler()
    else:
        print("\n无显示环境，请通过 X11 转发或 VNC 连接后重新运行")
        sys.exit(1)

if __name__ == "__main__":
    main()
