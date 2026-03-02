"""
人工标注辅助工具
用于可疑张望行为的时间片段标注
"""
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime


class AnnotationHelper:
    """视频标注辅助工具"""

    def __init__(self, video_path: str, pose_csv_path: str = None):
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem
        self.cap = cv2.VideoCapture(str(video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps

        # 加载 pose 数据（如果有）
        self.pose_df = None
        if pose_csv_path and Path(pose_csv_path).exists():
            self.pose_df = pd.read_csv(pose_csv_path)
            print(f"已加载 pose 数据: {len(self.pose_df)} 帧")

        # 标注结果
        self.annotations = []

        print(f"视频: {self.video_name}")
        print(f"时长: {self.duration:.1f}s, FPS: {self.fps:.1f}, 总帧数: {self.total_frames}")

    def get_frame_at_time(self, time_sec: float) -> np.ndarray:
        """获取指定时间的帧"""
        frame_id = int(time_sec * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.cap.read()
        return frame if ret else None

    def get_pose_at_time(self, time_sec: float, window: float = 0.5) -> dict:
        """获取指定时间附近的 pose 统计"""
        if self.pose_df is None:
            return None

        mask = (self.pose_df['time_sec'] >= time_sec - window) & \
               (self.pose_df['time_sec'] < time_sec + window)
        window_df = self.pose_df[mask]

        if len(window_df) == 0:
            return None

        return {
            'yaw_mean': window_df['yaw'].mean(),
            'yaw_std': window_df['yaw'].std(),
            'yaw_range': window_df['yaw'].max() - window_df['yaw'].min(),
            'pitch_mean': window_df['pitch'].mean(),
            'roll_mean': window_df['roll'].mean(),
            'sample_count': len(window_df)
        }

    def add_annotation(self, start_time: float, end_time: float,
                       label: int, track_id: int = 0, note: str = ""):
        """添加标注"""
        self.annotations.append({
            'video_name': self.video_name,
            'track_id': track_id,
            'start_time': round(start_time, 2),
            'end_time': round(end_time, 2),
            'duration': round(end_time - start_time, 2),
            'label': label,
            'note': note,
            'annotated_at': datetime.now().isoformat()
        })
        print(f"已添加: [{start_time:.1f}s - {end_time:.1f}s] label={label} {note}")

    def save_annotations(self, output_path: str):
        """保存标注结果"""
        df = pd.DataFrame(self.annotations)
        df.to_csv(output_path, index=False)
        print(f"已保存 {len(self.annotations)} 条标注到: {output_path}")
        return df

    def interactive_annotate(self):
        """交互式标注（命令行版本）"""
        print("\n" + "="*60)
        print("交互式标注模式")
        print("="*60)
        print("命令:")
        print("  g <time>     - 跳转到指定时间（秒）")
        print("  p <time>     - 查看指定时间的 pose 统计")
        print("  a <start> <end> <label> [note] - 添加标注")
        print("  l            - 列出所有标注")
        print("  s            - 保存并退出")
        print("  q            - 退出（不保存）")
        print("="*60 + "\n")

        while True:
            try:
                cmd = input("> ").strip().split()
                if not cmd:
                    continue

                if cmd[0] == 'q':
                    print("退出（未保存）")
                    break

                elif cmd[0] == 's':
                    output_path = f"data/manual_annotations/{self.video_name}_gt.csv"
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    self.save_annotations(output_path)
                    break

                elif cmd[0] == 'g' and len(cmd) >= 2:
                    time_sec = float(cmd[1])
                    frame = self.get_frame_at_time(time_sec)
                    if frame is not None:
                        # 显示帧（需要 GUI 环境）
                        cv2.imshow('Frame', frame)
                        cv2.waitKey(1)
                        print(f"显示 {time_sec:.1f}s 的帧")

                elif cmd[0] == 'p' and len(cmd) >= 2:
                    time_sec = float(cmd[1])
                    pose = self.get_pose_at_time(time_sec)
                    if pose:
                        print(f"时间 {time_sec:.1f}s 附近的 pose 统计:")
                        for k, v in pose.items():
                            print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
                    else:
                        print("无 pose 数据")

                elif cmd[0] == 'a' and len(cmd) >= 4:
                    start = float(cmd[1])
                    end = float(cmd[2])
                    label = int(cmd[3])
                    note = ' '.join(cmd[4:]) if len(cmd) > 4 else ""
                    self.add_annotation(start, end, label, note=note)

                elif cmd[0] == 'l':
                    print(f"\n当前标注 ({len(self.annotations)} 条):")
                    for i, ann in enumerate(self.annotations):
                        print(f"  {i+1}. [{ann['start_time']:.1f}s - {ann['end_time']:.1f}s] "
                              f"label={ann['label']} {ann['note']}")
                    print()

                else:
                    print("未知命令，输入 h 查看帮助")

            except Exception as e:
                print(f"错误: {e}")

        self.cap.release()
        cv2.destroyAllWindows()

    def close(self):
        self.cap.release()


def generate_annotation_template(pose_results_dir: str, output_path: str):
    """生成标注模板 CSV

    基于 pose 结果，按固定时间间隔生成待标注的时间片段
    """
    pose_dir = Path(pose_results_dir)
    csv_files = sorted(pose_dir.glob('*.csv'))

    all_segments = []
    segment_duration = 3.0  # 每个片段 3 秒

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        video_name = csv_file.stem

        if len(df) == 0:
            continue

        # 获取时间范围
        t_min = df['time_sec'].min()
        t_max = df['time_sec'].max()

        # 按 track_id 分组
        for track_id in df['track_id'].unique() if 'track_id' in df.columns else [0]:
            track_df = df[df['track_id'] == track_id] if 'track_id' in df.columns else df

            # 生成时间片段
            t = t_min
            while t + segment_duration <= t_max:
                # 计算该片段的 pose 统计（辅助标注）
                mask = (track_df['time_sec'] >= t) & (track_df['time_sec'] < t + segment_duration)
                seg_df = track_df[mask]

                if len(seg_df) >= 3:  # 至少 3 个样本
                    all_segments.append({
                        'video_name': video_name,
                        'track_id': track_id,
                        'start_time': round(t, 2),
                        'end_time': round(t + segment_duration, 2),
                        'yaw_mean': round(seg_df['yaw'].mean(), 1),
                        'yaw_std': round(seg_df['yaw'].std(), 1),
                        'yaw_range': round(seg_df['yaw'].max() - seg_df['yaw'].min(), 1),
                        'sample_count': len(seg_df),
                        'label': '',  # 待标注
                        'note': ''
                    })

                t += segment_duration / 2  # 50% 重叠

    # 保存模板
    template_df = pd.DataFrame(all_segments)
    template_df.to_csv(output_path, index=False)
    print(f"已生成标注模板: {output_path}")
    print(f"共 {len(template_df)} 个待标注片段")

    # 统计
    print(f"\n各视频片段数:")
    print(template_df.groupby('video_name').size())

    return template_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='人工标注辅助工具')
    parser.add_argument('--mode', choices=['template', 'annotate'], default='template',
                        help='模式: template=生成模板, annotate=交互标注')
    parser.add_argument('--pose-dir', default='data/pose_results',
                        help='pose 结果目录')
    parser.add_argument('--output', default='data/annotation_template.csv',
                        help='输出路径')
    parser.add_argument('--video', help='视频路径（annotate 模式）')
    parser.add_argument('--pose-csv', help='pose CSV 路径（annotate 模式）')

    args = parser.parse_args()

    if args.mode == 'template':
        generate_annotation_template(args.pose_dir, args.output)

    elif args.mode == 'annotate':
        if not args.video:
            print("请指定 --video 参数")
        else:
            helper = AnnotationHelper(args.video, args.pose_csv)
            helper.interactive_annotate()
