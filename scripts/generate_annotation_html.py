#!/usr/bin/env python3
"""
生成标注用的 HTML 报告
可以在浏览器中查看帧图像、人脸照片和姿态数据
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"

BEHAVIOR_LABELS = {
    0: ("normal", "正常行为", "#4CAF50"),
    1: ("glancing", "频繁张望", "#FF9800"),
    2: ("quick_turn", "快速回头", "#f44336"),
    3: ("prolonged_watch", "长时间观察", "#9C27B0"),
    4: ("looking_down", "持续低头", "#2196F3"),
    5: ("looking_up", "持续抬头", "#00BCD4"),
    -1: ("uncertain", "无法判断", "#9E9E9E"),
}

# 缓存已加载的数据
_detection_cache = {}
_tracking_cache = {}


def load_pose_data(video_id):
    pose_file = DATA_DIR / "pose" / video_id / "pose.json"
    if not pose_file.exists():
        return None
    with open(pose_file, 'r') as f:
        return json.load(f)


def load_detection_data(video_id):
    if video_id in _detection_cache:
        return _detection_cache[video_id]
    det_file = DATA_DIR / "detection" / video_id / "detections.json"
    if not det_file.exists():
        return None
    with open(det_file, 'r') as f:
        data = json.load(f)
    _detection_cache[video_id] = data
    return data


def load_tracking_data(video_id):
    if video_id in _tracking_cache:
        return _tracking_cache[video_id]
    track_file = DATA_DIR / "tracking" / video_id / "tracks.json"
    if not track_file.exists():
        return None
    with open(track_file, 'r') as f:
        data = json.load(f)
    _tracking_cache[video_id] = data
    return data


def get_track_poses(pose_data, track_id):
    for t in pose_data['tracks']:
        if t['track_id'] == track_id:
            return t['poses']
    return None


def get_track_detections(tracking_data, track_id):
    """获取指定 track 的检测序列"""
    if not tracking_data:
        return None
    for t in tracking_data['tracks']:
        if t['track_id'] == track_id:
            return t['detections']
    return None


def bbox_iou(box1, box2):
    """计算两个 bbox 的 IOU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def get_face_paths_for_track(video_id, track_id, start_frame, end_frame):
    """获取指定 track 在时间窗口内的人脸图像路径"""
    tracking_data = load_tracking_data(video_id)
    detection_data = load_detection_data(video_id)

    if not tracking_data or not detection_data:
        return []

    # 获取 track 的检测序列
    track_dets = get_track_detections(tracking_data, track_id)
    if not track_dets:
        return []

    # 构建 frame_idx -> detection 的映射
    det_by_frame = {}
    for frame_data in detection_data.get('frames', []):
        det_by_frame[frame_data['frame_idx']] = frame_data['detections']

    # 选择窗口内的几个关键帧
    window_dets = [d for d in track_dets if start_frame <= d['frame_idx'] <= end_frame]
    if not window_dets:
        return []

    # 均匀选择最多 6 个帧
    indices = np.linspace(0, len(window_dets) - 1, min(6, len(window_dets)), dtype=int)
    selected_dets = [window_dets[i] for i in indices]

    face_paths = []
    for track_det in selected_dets:
        frame_idx = track_det['frame_idx']
        track_bbox = track_det['bbox']

        # 在 detection 数据中找到匹配的人脸
        frame_dets = det_by_frame.get(frame_idx, [])
        best_match = None
        best_iou = 0.5  # IOU 阈值

        for det in frame_dets:
            iou = bbox_iou(track_bbox, det['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_match = det

        if best_match and 'face_path' in best_match:
            face_paths.append(f"detection/{video_id}/{best_match['face_path']}")

    return face_paths


def analyze_window(poses, start_frame, end_frame):
    window_poses = [p for p in poses if start_frame <= p['frame_idx'] <= end_frame]
    if not window_poses or len(window_poses) < 5:
        return None, []

    yaws = [p['yaw'] for p in window_poses]
    pitches = [p['pitch'] for p in window_poses]
    frames = [p['frame_idx'] for p in window_poses]

    stats = {
        'num_frames': len(window_poses),
        'yaw_mean': round(np.mean(yaws), 1),
        'yaw_std': round(np.std(yaws), 1),
        'yaw_min': round(np.min(yaws), 1),
        'yaw_max': round(np.max(yaws), 1),
        'pitch_mean': round(np.mean(pitches), 1),
        'pitch_std': round(np.std(pitches), 1),
    }

    # 计算 yaw 变化次数
    yaw_changes = 0
    for i in range(1, len(yaws)):
        if abs(yaws[i] - yaws[i-1]) > 15:
            yaw_changes += 1
    stats['yaw_changes'] = yaw_changes

    # 最大单帧变化
    max_change = max(abs(yaws[i] - yaws[i-1]) for i in range(1, len(yaws))) if len(yaws) > 1 else 0
    stats['max_yaw_change'] = round(max_change, 1)

    # 姿态序列数据（用于绘图）
    pose_series = list(zip(frames, yaws, pitches))

    return stats, pose_series


def suggest_label(stats):
    yaw_std = stats['yaw_std']
    pitch_mean = stats['pitch_mean']
    yaw_mean = abs(stats['yaw_mean'])
    yaw_changes = stats['yaw_changes']
    max_change = stats['max_yaw_change']

    if yaw_std > 25 and yaw_changes >= 3:
        return 1, "glancing"
    if max_change > 20:
        return 2, "quick_turn"
    if yaw_mean > 30 and yaw_std < 15:
        return 3, "prolonged_watch"
    if pitch_mean < -20:
        return 4, "looking_down"
    if pitch_mean > 20:
        return 5, "looking_up"
    return 0, "normal"


def get_frame_paths(video_id, start_frame, end_frame):
    """获取帧图像的相对路径"""
    frames_dir = DATA_DIR / "frames" / video_id
    if not frames_dir.exists():
        return []

    # 选择 6 个关键帧
    frame_indices = np.linspace(start_frame, end_frame, 6, dtype=int)
    paths = []
    for idx in frame_indices:
        frame_path = frames_dir / f"frame_{idx:06d}.jpg"
        if frame_path.exists():
            # 返回相对于 data 目录的路径
            paths.append(f"frames/{video_id}/frame_{idx:06d}.jpg")
    return paths


def generate_html_report(samples, output_path):
    """生成 HTML 报告"""

    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>行为标注工具</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .sample { background: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .sample-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
        .sample-id { font-size: 18px; font-weight: bold; color: #333; }
        .suggestion { padding: 5px 15px; border-radius: 20px; color: white; font-weight: bold; }
        .frames { display: flex; gap: 5px; overflow-x: auto; margin: 15px 0; }
        .frames img { height: 150px; border-radius: 4px; }
        .faces { display: flex; gap: 8px; overflow-x: auto; margin: 15px 0; padding: 10px; background: #e8f5e9; border-radius: 8px; }
        .faces img { height: 100px; border-radius: 4px; border: 2px solid #4CAF50; }
        .section-label { font-size: 12px; color: #666; margin-bottom: 5px; font-weight: bold; }
        .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 15px 0; }
        .stat { background: #f0f0f0; padding: 10px; border-radius: 4px; text-align: center; }
        .stat-label { font-size: 12px; color: #666; }
        .stat-value { font-size: 18px; font-weight: bold; color: #333; }
        .pose-chart { width: 100%; height: 120px; background: #fafafa; border-radius: 4px; position: relative; }
        .label-buttons { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 15px; }
        .label-btn { padding: 8px 16px; border: 2px solid; border-radius: 20px; cursor: pointer; font-weight: bold; background: white; }
        .label-btn:hover { opacity: 0.8; }
        .label-btn.selected { color: white !important; }
        .export-btn { position: fixed; bottom: 20px; right: 20px; padding: 15px 30px; background: #4CAF50; color: white; border: none; border-radius: 8px; font-size: 16px; cursor: pointer; box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
        .export-btn:hover { background: #45a049; }
        .legend { background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
        .legend h3 { margin-top: 0; }
        .legend-item { display: inline-block; margin-right: 20px; }
        .legend-color { display: inline-block; width: 20px; height: 20px; border-radius: 4px; vertical-align: middle; margin-right: 5px; }
        .progress { background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
        .progress-bar { height: 20px; background: #e0e0e0; border-radius: 10px; overflow: hidden; }
        .progress-fill { height: 100%; background: #4CAF50; transition: width 0.3s; }
        canvas { width: 100%; height: 120px; }
    </style>
</head>
<body>
    <h1>行为识别数据集 - 人工标注</h1>

    <div class="progress">
        <h3>标注进度: <span id="progress-text">0 / """ + str(len(samples)) + """</span></h3>
        <div class="progress-bar"><div class="progress-fill" id="progress-fill" style="width: 0%"></div></div>
    </div>

    <div class="legend">
        <h3>行为类别说明</h3>
"""

    for label_id, (name, desc, color) in BEHAVIOR_LABELS.items():
        html += f'        <div class="legend-item"><span class="legend-color" style="background:{color}"></span><b>[{label_id}] {name}</b>: {desc}</div>\n'

    html += """    </div>

    <div id="samples">
"""

    for i, sample in enumerate(samples):
        suggestion_color = BEHAVIOR_LABELS[sample['suggested_label']][2]
        suggestion_name = sample['suggested_name']

        html += f"""
        <div class="sample" id="sample-{i}" data-sample-id="{sample['sample_id']}">
            <div class="sample-header">
                <span class="sample-id">#{i+1} {sample['sample_id']}</span>
                <span class="suggestion" style="background:{suggestion_color}">建议: {suggestion_name}</span>
            </div>

            <div class="section-label">🎬 视频帧</div>
            <div class="frames">
"""
        for frame_path in sample.get('frame_paths', []):
            html += f'                <img src="{frame_path}" alt="frame">\n'

        html += """            </div>

            <div class="section-label">👤 人脸照片 (该目标)</div>
            <div class="faces">
"""
        for face_path in sample.get('face_paths', []):
            html += f'                <img src="{face_path}" alt="face">\n'

        if not sample.get('face_paths'):
            html += '                <span style="color:#999; padding:20px;">无人脸数据</span>\n'

        html += f"""            </div>

            <div class="stats">
                <div class="stat"><div class="stat-label">时间范围</div><div class="stat-value">{sample['time_range']}</div></div>
                <div class="stat"><div class="stat-label">Yaw 均值</div><div class="stat-value">{sample['stats']['yaw_mean']}°</div></div>
                <div class="stat"><div class="stat-label">Yaw 标准差</div><div class="stat-value">{sample['stats']['yaw_std']}°</div></div>
                <div class="stat"><div class="stat-label">Yaw 变化次数</div><div class="stat-value">{sample['stats']['yaw_changes']}</div></div>
                <div class="stat"><div class="stat-label">最大单帧变化</div><div class="stat-value">{sample['stats']['max_yaw_change']}°</div></div>
                <div class="stat"><div class="stat-label">Pitch 均值</div><div class="stat-value">{sample['stats']['pitch_mean']}°</div></div>
                <div class="stat"><div class="stat-label">Yaw 范围</div><div class="stat-value">[{sample['stats']['yaw_min']}°, {sample['stats']['yaw_max']}°]</div></div>
                <div class="stat"><div class="stat-label">有效帧数</div><div class="stat-value">{sample['stats']['num_frames']}</div></div>
            </div>

            <canvas id="chart-{i}" data-poses='{json.dumps(sample.get("pose_series", []))}'></canvas>

            <div class="label-buttons">
"""
        for label_id, (name, desc, color) in BEHAVIOR_LABELS.items():
            html += f'                <button class="label-btn" style="border-color:{color}; color:{color}" data-label="{label_id}" onclick="selectLabel({i}, {label_id})">[{label_id}] {name}</button>\n'

        html += """            </div>
        </div>
"""

    html += """    </div>

    <button class="export-btn" onclick="exportAnnotations()">导出标注结果</button>

    <script>
        const annotations = {};
        const totalSamples = """ + str(len(samples)) + """;

        function selectLabel(sampleIdx, label) {
            const sample = document.getElementById('sample-' + sampleIdx);
            const sampleId = sample.dataset.sampleId;
            annotations[sampleId] = label;

            // 更新按钮样式
            const buttons = sample.querySelectorAll('.label-btn');
            buttons.forEach(btn => {
                btn.classList.remove('selected');
                if (parseInt(btn.dataset.label) === label) {
                    btn.classList.add('selected');
                    btn.style.background = btn.style.borderColor;
                } else {
                    btn.style.background = 'white';
                }
            });

            updateProgress();
        }

        function updateProgress() {
            const count = Object.keys(annotations).length;
            document.getElementById('progress-text').textContent = count + ' / ' + totalSamples;
            document.getElementById('progress-fill').style.width = (count / totalSamples * 100) + '%';
        }

        function exportAnnotations() {
            const result = {
                exported_at: new Date().toISOString(),
                total: Object.keys(annotations).length,
                annotations: annotations
            };
            const blob = new Blob([JSON.stringify(result, null, 2)], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'annotations_export.json';
            a.click();
        }

        // 绘制姿态曲线
        document.querySelectorAll('canvas').forEach(canvas => {
            const poses = JSON.parse(canvas.dataset.poses);
            if (poses.length === 0) return;

            const ctx = canvas.getContext('2d');
            const width = canvas.offsetWidth;
            const height = canvas.offsetHeight;
            canvas.width = width;
            canvas.height = height;

            const frames = poses.map(p => p[0]);
            const yaws = poses.map(p => p[1]);
            const pitches = poses.map(p => p[2]);

            const minFrame = Math.min(...frames);
            const maxFrame = Math.max(...frames);
            const frameRange = maxFrame - minFrame || 1;

            // 绘制背景线
            ctx.strokeStyle = '#ddd';
            ctx.beginPath();
            ctx.moveTo(0, height/2);
            ctx.lineTo(width, height/2);
            ctx.stroke();

            // 绘制 ±30° 参考线
            ctx.strokeStyle = '#ffcccc';
            ctx.setLineDash([5, 5]);
            const y30 = height/2 - (30/90) * height/2;
            const yMinus30 = height/2 + (30/90) * height/2;
            ctx.beginPath();
            ctx.moveTo(0, y30);
            ctx.lineTo(width, y30);
            ctx.moveTo(0, yMinus30);
            ctx.lineTo(width, yMinus30);
            ctx.stroke();
            ctx.setLineDash([]);

            // 绘制 Yaw 曲线
            ctx.strokeStyle = '#2196F3';
            ctx.lineWidth = 2;
            ctx.beginPath();
            poses.forEach((p, i) => {
                const x = ((p[0] - minFrame) / frameRange) * width;
                const y = height/2 - (p[1]/90) * height/2;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();

            // 绘制 Pitch 曲线
            ctx.strokeStyle = '#4CAF50';
            ctx.lineWidth = 2;
            ctx.beginPath();
            poses.forEach((p, i) => {
                const x = ((p[0] - minFrame) / frameRange) * width;
                const y = height/2 - (p[2]/90) * height/2;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();

            // 图例
            ctx.font = '12px Arial';
            ctx.fillStyle = '#2196F3';
            ctx.fillText('Yaw', 10, 15);
            ctx.fillStyle = '#4CAF50';
            ctx.fillText('Pitch', 50, 15);
        });
    </script>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)


def main():
    # 加载待标注样本
    pending_file = DATA_DIR / "annotations" / "pending_annotations.json"
    if not pending_file.exists():
        print("请先运行 generate_annotation_preview.py")
        return

    with open(pending_file, 'r') as f:
        pending = json.load(f)

    samples = pending['samples']
    print(f"共 {len(samples)} 个待标注样本")

    # 为每个样本添加帧路径、人脸路径和姿态序列
    for i, sample in enumerate(samples):
        video_id = sample['video_id']
        track_id = sample['track_id']
        start_frame = sample['start_frame']
        end_frame = sample['end_frame']

        if i % 50 == 0:
            print(f"  处理样本 {i+1}/{len(samples)}...")

        # 获取帧路径
        sample['frame_paths'] = get_frame_paths(video_id, start_frame, end_frame)

        # 获取人脸图像路径
        sample['face_paths'] = get_face_paths_for_track(video_id, track_id, start_frame, end_frame)

        # 获取姿态序列
        pose_data = load_pose_data(video_id)
        if pose_data:
            poses = get_track_poses(pose_data, track_id)
            if poses:
                _, pose_series = analyze_window(poses, start_frame, end_frame)
                sample['pose_series'] = pose_series

    # 生成 HTML 报告
    output_path = DATA_DIR / "annotations" / "annotation_tool.html"
    generate_html_report(samples, output_path)
    print(f"\nHTML 标注工具已生成: {output_path}")
    print(f"请在浏览器中打开此文件进行标注")


if __name__ == "__main__":
    main()
