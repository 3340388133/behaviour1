#!/usr/bin/env python3
"""生成标注工具v3 - 支持人物ID分组"""
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
    """获取人脸图片路径"""
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


def get_html_header():
    return '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>行为标注工具 v3 - 人物分组</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1, h2 { color: #333; }
        .controls { background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
        .controls select { padding: 8px; margin-right: 10px; }
        .track-list { display: flex; flex-wrap: wrap; gap: 15px; }
        .track-card { width: 220px; background: white; border: 2px solid #ddd; border-radius: 8px; padding: 10px; cursor: pointer; }
        .track-card:hover { border-color: #2196F3; }
        .track-card.selected { border-color: #4CAF50; background: #e8f5e9; }
        .track-card.grouped { border-color: #9C27B0; background: #f3e5f5; }
        .track-card img { width: 100%; height: 100px; object-fit: cover; border-radius: 4px; }
        .track-card .title { font-weight: bold; margin: 8px 0 4px; }
        .track-card .info { font-size: 12px; color: #666; }
        .action-bar { background: white; padding: 15px; border-radius: 8px; margin: 20px 0; display: flex; gap: 10px; align-items: center; }
        .action-bar button { padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }
        .btn-merge { background: #4CAF50; color: white; }
        .btn-new { background: #FF9800; color: white; }
        .btn-clear { background: #9e9e9e; color: white; }
        .btn-export { background: #2196F3; color: white; }
        .person-groups { margin: 20px 0; }
        .person-group { background: #e8f5e9; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #4CAF50; }
        .person-group .header { font-weight: bold; color: #2E7D32; margin-bottom: 10px; }
        .person-group .tracks { font-size: 14px; color: #666; }
        .ungrouped { background: #fff3e0; border-left-color: #FF9800; }
        .ungrouped .header { color: #E65100; }
    </style>
</head>
<body>
    <h1>Track人物分组工具</h1>
    <p>选择属于同一个人的Track，点击"合并为同一人"</p>

    <div class="controls">
        <label>选择视频: <select id="video-select" onchange="filterVideo()"></select></label>
        <span id="stats"></span>
    </div>

    <div class="action-bar">
        <button class="btn-merge" onclick="mergeSelected()">合并为同一人</button>
        <button class="btn-clear" onclick="clearSelection()">清除选择</button>
        <button class="btn-export" onclick="exportGroups()">导出分组</button>
        <span id="selection-info" style="margin-left:20px;color:#666;"></span>
    </div>

    <div class="person-groups" id="person-groups"></div>

    <h2>未分组的Track</h2>
    <div class="track-list" id="track-list"></div>
'''


def get_html_footer(tracks_data):
    """生成HTML footer，包含JavaScript"""
    tracks_json = json.dumps(tracks_data, ensure_ascii=False)
    return f'''
    <script>
        const allTracks = {tracks_json};
        const personGroups = [];  // [{{"person_id": 0, "tracks": [...]}}]
        const selectedTracks = new Set();
        let nextPersonId = 0;

        document.addEventListener('DOMContentLoaded', function() {{
            initVideoSelect();
            renderTracks();
            loadSavedGroups();
        }});

        function initVideoSelect() {{
            const videos = [...new Set(allTracks.map(t => t.video_id))].sort();
            const select = document.getElementById('video-select');
            select.innerHTML = '<option value="all">全部视频</option>';
            videos.forEach(v => {{
                select.innerHTML += `<option value="${{v}}">${{v}}</option>`;
            }});
        }}

        function filterVideo() {{
            renderTracks();
        }}

        function getFilteredTracks() {{
            const video = document.getElementById('video-select').value;
            if (video === 'all') return allTracks;
            return allTracks.filter(t => t.video_id === video);
        }}

        function getGroupedTrackIds() {{
            const ids = new Set();
            personGroups.forEach(g => {{
                g.tracks.forEach(t => ids.add(t.video_id + '_' + t.track_id));
            }});
            return ids;
        }}

        function renderTracks() {{
            const tracks = getFilteredTracks();
            const groupedIds = getGroupedTrackIds();
            const container = document.getElementById('track-list');

            container.innerHTML = tracks
                .filter(t => !groupedIds.has(t.video_id + '_' + t.track_id))
                .map(t => createTrackCard(t))
                .join('');

            updateStats();
            renderPersonGroups();
        }}

        function createTrackCard(track) {{
            const key = track.video_id + '_' + track.track_id;
            const isSelected = selectedTracks.has(key);
            const img = track.face_images[0] || '';

            return `
                <div class="track-card ${{isSelected ? 'selected' : ''}}"
                     data-key="${{key}}" onclick="toggleSelect('${{key}}')">
                    <img src="${{img}}" alt="face" onerror="this.style.display='none'">
                    <div class="title">${{track.video_id}}</div>
                    <div class="info">Track ${{track.track_id}}</div>
                    <div class="info">帧 ${{track.start_frame}}-${{track.end_frame}}</div>
                    <div class="info">${{track.n_windows}} 个窗口</div>
                </div>
            `;
        }}

        function toggleSelect(key) {{
            if (selectedTracks.has(key)) {{
                selectedTracks.delete(key);
            }} else {{
                selectedTracks.add(key);
            }}
            renderTracks();
            updateSelectionInfo();
        }}

        function updateSelectionInfo() {{
            const info = document.getElementById('selection-info');
            info.textContent = selectedTracks.size > 0
                ? `已选择 ${{selectedTracks.size}} 个Track`
                : '';
        }}

        function clearSelection() {{
            selectedTracks.clear();
            renderTracks();
            updateSelectionInfo();
        }}

        function mergeSelected() {{
            if (selectedTracks.size < 2) {{
                alert('请至少选择2个Track进行合并');
                return;
            }}

            const tracks = [];
            selectedTracks.forEach(key => {{
                const lastIdx = key.lastIndexOf('_');
                const video_id = key.substring(0, lastIdx);
                const track_id = parseInt(key.substring(lastIdx + 1));
                const track = allTracks.find(t =>
                    t.video_id === video_id && t.track_id === track_id
                );
                if (track) tracks.push(track);
            }});

            personGroups.push({{
                person_id: nextPersonId++,
                tracks: tracks
            }});

            selectedTracks.clear();
            renderTracks();
            updateSelectionInfo();
            saveGroups();
        }}

        function removeFromGroup(personId, trackKey) {{
            const group = personGroups.find(g => g.person_id === personId);
            if (group) {{
                group.tracks = group.tracks.filter(t =>
                    (t.video_id + '_' + t.track_id) !== trackKey
                );
                if (group.tracks.length === 0) {{
                    const idx = personGroups.indexOf(group);
                    personGroups.splice(idx, 1);
                }}
            }}
            renderTracks();
            saveGroups();
        }}

        function renderPersonGroups() {{
            const container = document.getElementById('person-groups');
            if (personGroups.length === 0) {{
                container.innerHTML = '';
                return;
            }}

            container.innerHTML = personGroups.map(g => `
                <div class="person-group">
                    <div class="header">人物 ${{g.person_id + 1}} (${{g.tracks.length}} 个Track)</div>
                    <div class="tracks">
                        ${{g.tracks.map(t => `
                            <span style="display:inline-block;margin:5px;padding:5px 10px;background:#fff;border-radius:4px;">
                                ${{t.video_id}} Track ${{t.track_id}}
                                <button onclick="removeFromGroup(${{g.person_id}}, '${{t.video_id}}_${{t.track_id}}')"
                                        style="margin-left:5px;background:#f44336;color:white;border:none;border-radius:3px;cursor:pointer;">×</button>
                            </span>
                        `).join('')}}
                    </div>
                </div>
            `).join('');
        }}

        function updateStats() {{
            const total = allTracks.length;
            const grouped = getGroupedTrackIds().size;
            document.getElementById('stats').textContent =
                `共 ${{total}} 个Track，已分组 ${{grouped}}，未分组 ${{total - grouped}}`;
        }}

        function saveGroups() {{
            localStorage.setItem('personGroups', JSON.stringify(personGroups));
            localStorage.setItem('nextPersonId', nextPersonId);
        }}

        function loadSavedGroups() {{
            const saved = localStorage.getItem('personGroups');
            const savedId = localStorage.getItem('nextPersonId');
            if (saved) {{
                const loaded = JSON.parse(saved);
                personGroups.push(...loaded);
                nextPersonId = savedId ? parseInt(savedId) : loaded.length;
                renderTracks();
            }}
        }}

        function exportGroups() {{
            const result = {{
                exported_at: new Date().toISOString(),
                total_persons: personGroups.length,
                persons: personGroups.map(g => ({{
                    person_id: g.person_id,
                    tracks: g.tracks.map(t => ({{
                        video_id: t.video_id,
                        track_id: t.track_id
                    }}))
                }}))
            }};

            const blob = new Blob([JSON.stringify(result, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'person_groups.json';
            a.click();
        }}
    </script>
</body>
</html>
'''


def main():
    """主函数"""
    video_ids = []
    for d in POSE_DIR.iterdir():
        if d.is_dir() and not d.name.startswith('.'):
            if (d / "pose.json").exists():
                video_ids.append(d.name)

    print(f"找到 {len(video_ids)} 个已完成姿态估计的视频")

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

            index_data = load_feature_index(video_id, track_id)
            if not index_data:
                continue

            windows = index_data.get("windows", [])
            if not windows:
                continue

            start_frame = windows[0]["start_frame"]
            end_frame = windows[-1]["end_frame"]
            face_images = get_face_images(video_id, start_frame, end_frame, max_images=3)

            all_tracks.append({
                "video_id": video_id,
                "track_id": track_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "n_windows": len(windows),
                "face_images": face_images
            })

    print(f"共 {len(all_tracks)} 条有效轨迹")

    html = get_html_header()
    html += get_html_footer(all_tracks)

    output_file = ANNOTATIONS_DIR / "annotation_tool_v3.html"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"已生成: {output_file}")


if __name__ == "__main__":
    main()
