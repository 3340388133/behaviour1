#!/usr/bin/env python3
"""生成优化版标注工具 - 按track分组显示，支持ID问题记录"""
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data"
POSE_DIR = DATA_DIR / "pose"
FEATURES_DIR = DATA_DIR / "features"
DETECTION_DIR = DATA_DIR / "detection"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

def get_html_header():
    return '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>行为标注工具 v2</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .controls { background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; display: flex; gap: 15px; align-items: center; flex-wrap: wrap; }
        .controls select, .controls button { padding: 8px 15px; border-radius: 4px; border: 1px solid #ddd; }
        .controls button { background: #2196F3; color: white; border: none; cursor: pointer; }
        .controls button:hover { background: #1976D2; }
        .track-group { background: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .track-header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 15px; border-bottom: 2px solid #e0e0e0; margin-bottom: 15px; }
        .track-title { font-size: 20px; font-weight: bold; color: #1976D2; }
        .track-info { color: #666; font-size: 14px; }
        .report-id-btn { padding: 8px 15px; background: #f44336; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }
        .report-id-btn:hover { background: #d32f2f; }
        .sample { background: #fafafa; margin: 15px 0; padding: 15px; border-radius: 8px; border-left: 4px solid #2196F3; }
        .sample-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
        .sample-id { font-size: 16px; font-weight: bold; color: #333; }
        .suggestion { padding: 4px 12px; border-radius: 15px; color: white; font-weight: bold; font-size: 12px; }
        .faces { display: flex; gap: 8px; overflow-x: auto; margin: 10px 0; padding: 10px; background: #e8f5e9; border-radius: 8px; }
        .faces img { height: 80px; border-radius: 4px; border: 2px solid #4CAF50; }
        .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 10px 0; font-size: 12px; }
        .stat { background: #f0f0f0; padding: 8px; border-radius: 4px; text-align: center; }
        .stat-label { color: #666; }
        .stat-value { font-weight: bold; color: #333; }
        canvas { width: 100%; height: 100px; }
        .label-buttons { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
        .label-btn { padding: 6px 12px; border: 2px solid; border-radius: 15px; cursor: pointer; font-weight: bold; background: white; font-size: 12px; }
        .label-btn:hover { opacity: 0.8; }
        .label-btn.selected { color: white !important; }
        .export-btns { position: fixed; bottom: 20px; right: 20px; display: flex; gap: 10px; }
        .export-btn { padding: 12px 25px; color: white; border: none; border-radius: 8px; font-size: 14px; cursor: pointer; box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
        .export-btn.green { background: #4CAF50; }
        .export-btn.orange { background: #FF9800; }
        .export-btn:hover { opacity: 0.9; }
        .legend { background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
        .legend h3 { margin-top: 0; }
        .legend-item { display: inline-block; margin-right: 15px; font-size: 14px; }
        .legend-color { display: inline-block; width: 16px; height: 16px; border-radius: 4px; vertical-align: middle; margin-right: 5px; }
        .progress { background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
        .progress-bar { height: 20px; background: #e0e0e0; border-radius: 10px; overflow: hidden; }
        .progress-fill { height: 100%; background: #4CAF50; transition: width 0.3s; }
        .id-issues { background: #fff3e0; padding: 15px; border-radius: 8px; margin-bottom: 20px; display: none; }
        .id-issues h3 { margin-top: 0; color: #e65100; }
        .id-issue-item { background: white; padding: 10px; margin: 5px 0; border-radius: 4px; display: flex; justify-content: space-between; align-items: center; }
        .remove-issue { background: #f44336; color: white; border: none; padding: 4px 8px; border-radius: 4px; cursor: pointer; }
        .modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1000; }
        .modal-content { background: white; margin: 10% auto; padding: 20px; border-radius: 8px; width: 500px; max-width: 90%; }
        .modal-content h3 { margin-top: 0; }
        .modal-content input, .modal-content textarea { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
        .modal-content .btn-group { display: flex; gap: 10px; justify-content: flex-end; margin-top: 15px; }
        .modal-content button { padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        .modal-content .cancel-btn { background: #9e9e9e; color: white; }
        .modal-content .submit-btn { background: #4CAF50; color: white; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <h1>行为识别数据集 - 人工标注 v2</h1>
'''

def get_html_controls():
    return '''
    <div class="controls">
        <label>选择视频: <select id="video-select" onchange="filterByVideo()"></select></label>
        <label>选择Track: <select id="track-select" onchange="filterByTrack()"><option value="all">全部Track</option></select></label>
        <button onclick="showAllTracks()">显示全部</button>
        <span id="filter-info" style="color:#666;"></span>
    </div>

    <div class="progress">
        <h3>标注进度: <span id="progress-text">0 / 0</span></h3>
        <div class="progress-bar"><div class="progress-fill" id="progress-fill" style="width: 0%"></div></div>
    </div>

    <div class="id-issues" id="id-issues-panel">
        <h3>ID问题记录 (<span id="issue-count">0</span>)</h3>
        <div id="issue-list"></div>
    </div>

    <div class="legend">
        <h3>行为类别说明</h3>
        <div class="legend-item"><span class="legend-color" style="background:#4CAF50"></span><b>[0] normal</b>: 正常行为</div>
        <div class="legend-item"><span class="legend-color" style="background:#FF9800"></span><b>[1] glancing</b>: 频繁张望</div>
        <div class="legend-item"><span class="legend-color" style="background:#f44336"></span><b>[2] quick_turn</b>: 快速回头</div>
        <div class="legend-item"><span class="legend-color" style="background:#9C27B0"></span><b>[3] prolonged_watch</b>: 长时间观察</div>
        <div class="legend-item"><span class="legend-color" style="background:#2196F3"></span><b>[4] looking_down</b>: 持续低头</div>
        <div class="legend-item"><span class="legend-color" style="background:#00BCD4"></span><b>[5] looking_up</b>: 持续抬头</div>
        <div class="legend-item"><span class="legend-color" style="background:#9E9E9E"></span><b>[-1] uncertain</b>: 无法判断</div>
    </div>

    <!-- ID问题报告弹窗 -->
    <div class="modal" id="id-modal">
        <div class="modal-content">
            <h3>报告Track ID问题</h3>
            <p id="modal-track-info"></p>
            <label>问题类型:</label>
            <select id="issue-type">
                <option value="id_switch">ID切换 - 同一人被分配了不同ID</option>
                <option value="id_merge">ID合并 - 不同人被分配了相同ID</option>
                <option value="id_lost">ID丢失 - 跟踪中断</option>
                <option value="other">其他问题</option>
            </select>
            <label>关联的其他Track ID (可选):</label>
            <input type="text" id="related-tracks" placeholder="例如: 2, 5, 8">
            <label>备注说明:</label>
            <textarea id="issue-note" rows="3" placeholder="描述具体问题..."></textarea>
            <div class="btn-group">
                <button class="cancel-btn" onclick="closeModal()">取消</button>
                <button class="submit-btn" onclick="submitIdIssue()">提交</button>
            </div>
        </div>
    </div>

    <div id="tracks-container"></div>

    <div class="export-btns">
        <button class="export-btn orange" onclick="exportIdIssues()">导出ID问题</button>
        <button class="export-btn green" onclick="exportAnnotations()">导出标注</button>
    </div>
'''

def get_html_footer():
    return '''
    <script>
        const annotations = {};
        const idIssues = [];
        let totalSamples = 0;
        let currentReportTrack = null;
        const labelColors = {
            0: '#4CAF50', 1: '#FF9800', 2: '#f44336',
            3: '#9C27B0', 4: '#2196F3', 5: '#00BCD4', '-1': '#9E9E9E'
        };

        // 初始化
        document.addEventListener('DOMContentLoaded', function() {
            totalSamples = document.querySelectorAll('.sample').length;
            updateProgress();
            initFilters();
            drawAllCharts();
        });

        function initFilters() {
            const videos = new Set();
            const tracks = new Set();
            document.querySelectorAll('.track-group').forEach(g => {
                videos.add(g.dataset.video);
                tracks.add(g.dataset.video + '_track' + g.dataset.track);
            });

            const videoSelect = document.getElementById('video-select');
            videoSelect.innerHTML = '<option value="all">全部视频</option>';
            [...videos].sort().forEach(v => {
                videoSelect.innerHTML += `<option value="${v}">${v}</option>`;
            });
        }

        function filterByVideo() {
            const video = document.getElementById('video-select').value;
            const trackSelect = document.getElementById('track-select');
            trackSelect.innerHTML = '<option value="all">全部Track</option>';

            document.querySelectorAll('.track-group').forEach(g => {
                if (video === 'all' || g.dataset.video === video) {
                    g.classList.remove('hidden');
                    if (video !== 'all') {
                        trackSelect.innerHTML += `<option value="${g.dataset.track}">Track ${g.dataset.track}</option>`;
                    }
                } else {
                    g.classList.add('hidden');
                }
            });
            updateFilterInfo();
        }

        function filterByTrack() {
            const video = document.getElementById('video-select').value;
            const track = document.getElementById('track-select').value;

            document.querySelectorAll('.track-group').forEach(g => {
                const videoMatch = video === 'all' || g.dataset.video === video;
                const trackMatch = track === 'all' || g.dataset.track === track;
                if (videoMatch && trackMatch) {
                    g.classList.remove('hidden');
                } else {
                    g.classList.add('hidden');
                }
            });
            updateFilterInfo();
        }

        function showAllTracks() {
            document.getElementById('video-select').value = 'all';
            document.getElementById('track-select').value = 'all';
            document.querySelectorAll('.track-group').forEach(g => g.classList.remove('hidden'));
            updateFilterInfo();
        }

        function updateFilterInfo() {
            const visible = document.querySelectorAll('.track-group:not(.hidden)').length;
            const total = document.querySelectorAll('.track-group').length;
            document.getElementById('filter-info').textContent = `显示 ${visible}/${total} 个Track`;
        }

        function selectLabel(sampleIdx, label) {
            const sample = document.getElementById('sample-' + sampleIdx);
            const sampleId = sample.dataset.sampleId;
            annotations[sampleId] = label;

            sample.querySelectorAll('.label-btn').forEach(btn => {
                const btnLabel = btn.dataset.label;
                if (btnLabel == label) {
                    btn.classList.add('selected');
                    btn.style.background = labelColors[btnLabel];
                } else {
                    btn.classList.remove('selected');
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

        function openIdModal(video, trackId) {
            currentReportTrack = { video, trackId };
            document.getElementById('modal-track-info').textContent = `视频: ${video}, Track ID: ${trackId}`;
            document.getElementById('id-modal').style.display = 'block';
        }

        function closeModal() {
            document.getElementById('id-modal').style.display = 'none';
            document.getElementById('issue-type').value = 'id_switch';
            document.getElementById('related-tracks').value = '';
            document.getElementById('issue-note').value = '';
        }

        function submitIdIssue() {
            const issue = {
                video_id: currentReportTrack.video,
                track_id: parseInt(currentReportTrack.trackId),
                issue_type: document.getElementById('issue-type').value,
                related_tracks: document.getElementById('related-tracks').value.split(',').map(s => s.trim()).filter(s => s),
                note: document.getElementById('issue-note').value,
                timestamp: new Date().toISOString()
            };
            idIssues.push(issue);
            updateIdIssuesPanel();
            closeModal();
            alert('ID问题已记录！');
        }

        function updateIdIssuesPanel() {
            const panel = document.getElementById('id-issues-panel');
            const list = document.getElementById('issue-list');
            const count = document.getElementById('issue-count');

            if (idIssues.length > 0) {
                panel.style.display = 'block';
                count.textContent = idIssues.length;
                list.innerHTML = idIssues.map((issue, idx) => `
                    <div class="id-issue-item">
                        <span><b>${issue.video_id}</b> Track ${issue.track_id} - ${issue.issue_type} ${issue.related_tracks.length ? '(关联: ' + issue.related_tracks.join(', ') + ')' : ''}</span>
                        <button class="remove-issue" onclick="removeIssue(${idx})">删除</button>
                    </div>
                `).join('');
            } else {
                panel.style.display = 'none';
            }
        }

        function removeIssue(idx) {
            idIssues.splice(idx, 1);
            updateIdIssuesPanel();
        }

        function exportAnnotations() {
            const result = {
                exported_at: new Date().toISOString(),
                total: Object.keys(annotations).length,
                annotations: annotations
            };
            downloadJson(result, 'annotations_export.json');
        }

        function exportIdIssues() {
            if (idIssues.length === 0) {
                alert('没有ID问题记录');
                return;
            }
            const result = {
                exported_at: new Date().toISOString(),
                total: idIssues.length,
                issues: idIssues
            };
            downloadJson(result, 'id_issues_export.json');
        }

        function downloadJson(data, filename) {
            const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
        }

        function drawAllCharts() {
            document.querySelectorAll('canvas').forEach(canvas => {
                const poses = JSON.parse(canvas.dataset.poses || '[]');
                if (poses.length === 0) return;
                drawChart(canvas, poses);
            });
        }

        function drawChart(canvas, poses) {
            const ctx = canvas.getContext('2d');
            const width = canvas.offsetWidth;
            const height = canvas.offsetHeight;
            canvas.width = width;
            canvas.height = height;

            const frames = poses.map(p => p[0]);
            const yaws = poses.map(p => p[1]);
            const minFrame = Math.min(...frames);
            const maxFrame = Math.max(...frames);
            const frameRange = maxFrame - minFrame || 1;

            // 背景线
            ctx.strokeStyle = '#ddd';
            ctx.beginPath();
            ctx.moveTo(0, height/2);
            ctx.lineTo(width, height/2);
            ctx.stroke();

            // ±30° 参考线
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

            // Yaw 曲线
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

            // Pitch 曲线
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
            ctx.font = '10px Arial';
            ctx.fillStyle = '#2196F3';
            ctx.fillText('Yaw', 5, 12);
            ctx.fillStyle = '#4CAF50';
            ctx.fillText('Pitch', 35, 12);
        }
    </script>
</body>
</html>
'''


def suggest_label(yaw_std, max_change, yaw_mean, pitch_mean):
    """根据统计特征建议标签"""
    if yaw_std > 25 and max_change > 20:
        return 1, "glancing", "#FF9800"
    if max_change > 40:
        return 2, "quick_turn", "#f44336"
    if abs(yaw_mean) > 30 and yaw_std < 15:
        return 3, "prolonged_watch", "#9C27B0"
    if pitch_mean < -20:
        return 4, "looking_down", "#2196F3"
    if pitch_mean > 20:
        return 5, "looking_up", "#00BCD4"
    return 0, "normal", "#4CAF50"


def load_pose_data(video_id):
    """加载姿态数据"""
    pose_file = POSE_DIR / video_id / "pose.json"
    if not pose_file.exists():
        return None
    with open(pose_file, "r") as f:
        return json.load(f)


def load_feature_index(video_id, track_id):
    """加载特征索引"""
    index_file = FEATURES_DIR / video_id / f"track_{track_id}_index.json"
    if not index_file.exists():
        return None
    with open(index_file, "r") as f:
        return json.load(f)


# 全局缓存
_face_cache = {}

def build_face_cache(video_id):
    """预先构建人脸图片缓存"""
    if video_id in _face_cache:
        return
    faces_dir = DETECTION_DIR / video_id / "faces"
    if not faces_dir.exists():
        _face_cache[video_id] = {}
        return

    cache = {}
    for f in faces_dir.glob("frame_*_face_*.jpg"):
        frame_idx = int(f.name.split("_")[1])
        if frame_idx not in cache:
            cache[frame_idx] = []
        cache[frame_idx].append(f.name)
    _face_cache[video_id] = cache

def get_face_images(video_id, start_frame, end_frame, max_images=5):
    """获取人脸图片路径（使用缓存）"""
    build_face_cache(video_id)
    cache = _face_cache.get(video_id, {})

    face_files = []
    step = max(1, (end_frame - start_frame) // max_images)
    for frame_idx in range(start_frame, end_frame + 1, step):
        if frame_idx in cache and cache[frame_idx]:
            face_files.append(f"../detection/{video_id}/faces/{cache[frame_idx][0]}")
        if len(face_files) >= max_images:
            break
    return face_files


def generate_sample_html(sample_idx, video_id, track_id, window):
    """生成单个样本的HTML"""
    sample_id = f"{video_id}_track{track_id}_win{window['window_idx']}"
    start_frame = window['start_frame']
    end_frame = window['end_frame']
    poses = window.get('poses', [])

    # 计算统计量
    if poses:
        yaws = [p[1] for p in poses]
        pitches = [p[2] for p in poses]
        yaw_mean = np.mean(yaws)
        yaw_std = np.std(yaws)
        pitch_mean = np.mean(pitches)
        yaw_changes = np.abs(np.diff(yaws))
        max_change = np.max(yaw_changes) if len(yaw_changes) > 0 else 0
        change_count = np.sum(yaw_changes > 10)
        yaw_min, yaw_max = np.min(yaws), np.max(yaws)
    else:
        yaw_mean = yaw_std = pitch_mean = max_change = change_count = 0
        yaw_min = yaw_max = 0

    label_id, label_name, label_color = suggest_label(yaw_std, max_change, yaw_mean, pitch_mean)
    face_images = get_face_images(video_id, start_frame, end_frame)

    start_time = (start_frame - 1) / 10.0
    end_time = (end_frame - 1) / 10.0

    poses_json = json.dumps(poses)
    faces_html = ''.join(f'<img src="{f}" alt="face">' for f in face_images)

    return f'''
        <div class="sample" id="sample-{sample_idx}" data-sample-id="{sample_id}">
            <div class="sample-header">
                <span class="sample-id">Window {window['window_idx']} (帧 {start_frame}-{end_frame})</span>
                <span class="suggestion" style="background:{label_color}">建议: {label_name}</span>
            </div>
            <div class="faces">{faces_html}</div>
            <div class="stats">
                <div class="stat"><div class="stat-label">时间</div><div class="stat-value">{start_time:.1f}s-{end_time:.1f}s</div></div>
                <div class="stat"><div class="stat-label">Yaw均值</div><div class="stat-value">{yaw_mean:.1f}°</div></div>
                <div class="stat"><div class="stat-label">Yaw标准差</div><div class="stat-value">{yaw_std:.1f}°</div></div>
                <div class="stat"><div class="stat-label">最大变化</div><div class="stat-value">{max_change:.1f}°</div></div>
            </div>
            <canvas id="chart-{sample_idx}" data-poses='{poses_json}'></canvas>
            <div class="label-buttons">
                <button class="label-btn" style="border-color:#4CAF50;color:#4CAF50" data-label="0" onclick="selectLabel({sample_idx},0)">[0] normal</button>
                <button class="label-btn" style="border-color:#FF9800;color:#FF9800" data-label="1" onclick="selectLabel({sample_idx},1)">[1] glancing</button>
                <button class="label-btn" style="border-color:#f44336;color:#f44336" data-label="2" onclick="selectLabel({sample_idx},2)">[2] quick_turn</button>
                <button class="label-btn" style="border-color:#9C27B0;color:#9C27B0" data-label="3" onclick="selectLabel({sample_idx},3)">[3] prolonged</button>
                <button class="label-btn" style="border-color:#2196F3;color:#2196F3" data-label="4" onclick="selectLabel({sample_idx},4)">[4] down</button>
                <button class="label-btn" style="border-color:#00BCD4;color:#00BCD4" data-label="5" onclick="selectLabel({sample_idx},5)">[5] up</button>
                <button class="label-btn" style="border-color:#9E9E9E;color:#9E9E9E" data-label="-1" onclick="selectLabel({sample_idx},-1)">[-1] uncertain</button>
            </div>
        </div>'''


def generate_track_group_html(video_id, track_id, windows, sample_idx_start):
    """生成一个track组的HTML"""
    n_windows = len(windows)
    start_frame = windows[0]['start_frame'] if windows else 0
    end_frame = windows[-1]['end_frame'] if windows else 0

    samples_html = ""
    for i, window in enumerate(windows):
        samples_html += generate_sample_html(sample_idx_start + i, video_id, track_id, window)

    return f'''
    <div class="track-group" data-video="{video_id}" data-track="{track_id}">
        <div class="track-header">
            <div>
                <span class="track-title">{video_id} - Track {track_id}</span>
                <span class="track-info">({n_windows} 个窗口, 帧 {start_frame}-{end_frame})</span>
            </div>
            <button class="report-id-btn" onclick="openIdModal('{video_id}', '{track_id}')">报告ID问题</button>
        </div>
        {samples_html}
    </div>
    ''', sample_idx_start + n_windows


def main():
    """主函数"""
    # 获取已完成姿态估计的视频
    video_ids = []
    for d in POSE_DIR.iterdir():
        if d.is_dir() and not d.name.startswith('.'):
            if (d / "pose.json").exists():
                video_ids.append(d.name)

    print(f"找到 {len(video_ids)} 个已完成姿态估计的视频")

    # 收集所有track数据
    all_tracks = []
    for video_id in sorted(video_ids):
        pose_data = load_pose_data(video_id)
        if not pose_data:
            continue

        for track in pose_data.get("tracks", []):
            track_id = track["track_id"]
            poses = track.get("poses", [])
            if len(poses) < 30:
                continue

            # 加载特征索引获取窗口信息
            index_data = load_feature_index(video_id, track_id)
            if not index_data:
                continue

            windows = index_data.get("windows", [])
            if not windows:
                continue

            # 为每个窗口添加姿态数据
            for win in windows:
                start_f = win["start_frame"]
                end_f = win["end_frame"]
                win_poses = []
                for p in poses:
                    if start_f <= p["frame_idx"] <= end_f:
                        win_poses.append([p["frame_idx"], p["yaw"], p["pitch"]])
                win["poses"] = win_poses

            all_tracks.append({
                "video_id": video_id,
                "track_id": track_id,
                "windows": windows
            })

    print(f"共 {len(all_tracks)} 条有效轨迹")

    # 生成HTML
    html = get_html_header()
    html += get_html_controls()

    sample_idx = 0
    for track_data in all_tracks:
        track_html, sample_idx = generate_track_group_html(
            track_data["video_id"],
            track_data["track_id"],
            track_data["windows"],
            sample_idx
        )
        html += track_html

    html += get_html_footer()

    # 保存文件
    output_file = ANNOTATIONS_DIR / "annotation_tool_v2.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"已生成: {output_file}")
    print(f"共 {sample_idx} 个样本")


if __name__ == "__main__":
    main()

