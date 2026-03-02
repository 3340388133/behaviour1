"""目标跟踪模块 - ByteTrack 实现"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
from filterpy.kalman import KalmanFilter


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray          # [x1, y1, x2, y2]
    confidence: float
    landmarks: np.ndarray = None
    pose: dict = field(default_factory=dict)
    history: list = field(default_factory=list)  # 历史姿态
    age: int = 0              # 总帧数
    hits: int = 0             # 匹配成功次数
    time_since_update: int = 0
    state: str = "tentative"  # tentative / confirmed / lost


class KalmanBoxTracker:
    """单目标 Kalman 跟踪器"""
    count = 0

    def __init__(self, bbox: np.ndarray):
        # 状态: [x_center, y_center, area, aspect_ratio, vx, vy, va]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # 状态转移矩阵
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])

        # 观测矩阵
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        # 噪声协方差
        self.kf.R[2:, 2:] *= 10.0  # 观测噪声
        self.kf.P[4:, 4:] *= 1000.0  # 速度不确定性
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # 初始化状态
        self.kf.x[:4] = self._bbox_to_z(bbox)

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 0
        self.time_since_update = 0
        self.age = 0

    def _bbox_to_z(self, bbox: np.ndarray) -> np.ndarray:
        """bbox [x1,y1,x2,y2] -> [cx, cy, area, ratio]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2
        cy = bbox[1] + h / 2
        area = w * h
        ratio = w / (h + 1e-6)
        return np.array([[cx], [cy], [area], [ratio]])

    def _z_to_bbox(self, z: np.ndarray) -> np.ndarray:
        """[cx, cy, area, ratio] -> bbox [x1,y1,x2,y2]"""
        cx, cy, area, ratio = z.flatten()
        w = np.sqrt(area * ratio)
        h = area / (w + 1e-6)
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

    def predict(self) -> np.ndarray:
        """预测下一帧位置"""
        # 防止面积为负
        if self.kf.x[2] + self.kf.x[6] <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self._z_to_bbox(self.kf.x[:4])

    def update(self, bbox: np.ndarray):
        """用检测结果更新状态"""
        self.time_since_update = 0
        self.hits += 1
        self.kf.update(self._bbox_to_z(bbox))

    def get_state(self) -> np.ndarray:
        """获取当前 bbox"""
        return self._z_to_bbox(self.kf.x[:4])


class ByteTracker:
    """ByteTrack 跟踪器 - 支持低置信度检测的二次匹配"""

    def __init__(
        self,
        high_thresh: float = 0.6,      # 高置信度阈值
        low_thresh: float = 0.1,       # 低置信度阈值
        match_thresh: float = 0.8,     # IoU 匹配阈值（提高以减少 ID switch）
        max_age: int = 30,             # 最大丢失帧数
        min_hits: int = 3,             # 确认轨迹的最小匹配次数
        second_match_thresh: float = 0.5  # 二次匹配阈值
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.second_match_thresh = second_match_thresh

        self.trackers: List[KalmanBoxTracker] = []
        self.tracks: List[Track] = []
        self.frame_count = 0

    def update(self, detections: list) -> List[Track]:
        """更新跟踪器（ByteTrack 核心逻辑）"""
        self.frame_count += 1

        # 1. 分离高/低置信度检测
        high_dets, low_dets = [], []
        for det in detections:
            if det.confidence >= self.high_thresh:
                high_dets.append(det)
            elif det.confidence >= self.low_thresh:
                low_dets.append(det)

        # 2. 预测所有轨迹
        for tracker in self.trackers:
            tracker.predict()

        # 3. 第一次匹配：高置信度检测 vs 所有轨迹
        matched1, unmatched_trks, unmatched_high = self._match(
            self.trackers, high_dets, self.match_thresh
        )

        # 更新匹配的轨迹
        for trk_idx, det_idx in matched1:
            self._update_tracker(trk_idx, high_dets[det_idx])

        # 4. 第二次匹配：低置信度检测 vs 未匹配轨迹
        remain_trackers = [self.trackers[i] for i in unmatched_trks]
        matched2, still_unmatched_trks, _ = self._match(
            remain_trackers, low_dets, self.second_match_thresh
        )

        # 更新二次匹配的轨迹
        for local_idx, det_idx in matched2:
            global_idx = unmatched_trks[local_idx]
            self._update_tracker(global_idx, low_dets[det_idx])
            still_unmatched_trks = [i for i in still_unmatched_trks if i != local_idx]

        # 5. 为未匹配的高置信度检测创建新轨迹
        for det_idx in unmatched_high:
            self._create_tracker(high_dets[det_idx])

        # 6. 移除过期轨迹
        self._remove_dead_trackers()

        # 7. 返回确认的轨迹
        return self._get_confirmed_tracks()

    def _match(
        self,
        trackers: List[KalmanBoxTracker],
        detections: list,
        thresh: float
    ) -> Tuple[list, list, list]:
        """IoU 匹配"""
        if len(trackers) == 0 or len(detections) == 0:
            return [], list(range(len(trackers))), list(range(len(detections)))

        # 计算 IoU 矩阵
        trk_bboxes = np.array([t.get_state() for t in trackers])
        det_bboxes = np.array([d.bbox for d in detections])
        iou_matrix = self._compute_iou_matrix(trk_bboxes, det_bboxes)

        # 匈牙利匹配
        matched, unmatched_trks, unmatched_dets = self._linear_assignment(
            iou_matrix, thresh
        )

        return matched, unmatched_trks, unmatched_dets

    def _compute_iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """计算 IoU 矩阵"""
        n, m = len(boxes1), len(boxes2)
        iou = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                iou[i, j] = self._iou(boxes1[i], boxes2[j])

        return iou

    def _iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """计算两个 bbox 的 IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter / (area1 + area2 - inter + 1e-6)

    def _linear_assignment(
        self,
        iou_matrix: np.ndarray,
        thresh: float
    ) -> Tuple[list, list, list]:
        """线性分配（匈牙利算法）"""
        if iou_matrix.size == 0:
            return [], list(range(iou_matrix.shape[0])), list(range(iou_matrix.shape[1]))

        try:
            import lap
            cost = 1 - iou_matrix
            _, x, y = lap.lapjv(cost, extend_cost=True, cost_limit=1 - thresh)

            matched = []
            for i, j in enumerate(x):
                if j >= 0:
                    matched.append((i, j))

            unmatched_trks = [i for i in range(len(x)) if x[i] < 0]
            unmatched_dets = [j for j in range(len(y)) if y[j] < 0]

        except ImportError:
            matched, unmatched_trks, unmatched_dets = self._greedy_match(iou_matrix, thresh)

        return matched, unmatched_trks, unmatched_dets

    def _greedy_match(
        self,
        iou_matrix: np.ndarray,
        thresh: float
    ) -> Tuple[list, list, list]:
        """贪婪匹配（lap 不可用时的回退）"""
        matched = []
        used_trks, used_dets = set(), set()

        # 按 IoU 降序排列
        indices = np.argsort(-iou_matrix.flatten())

        for idx in indices:
            i = idx // iou_matrix.shape[1]
            j = idx % iou_matrix.shape[1]

            if i in used_trks or j in used_dets:
                continue
            if iou_matrix[i, j] < thresh:
                break

            matched.append((i, j))
            used_trks.add(i)
            used_dets.add(j)

        unmatched_trks = [i for i in range(iou_matrix.shape[0]) if i not in used_trks]
        unmatched_dets = [j for j in range(iou_matrix.shape[1]) if j not in used_dets]

        return matched, unmatched_trks, unmatched_dets

    def _update_tracker(self, trk_idx: int, detection):
        """更新指定轨迹"""
        tracker = self.trackers[trk_idx]
        tracker.update(detection.bbox)

        # 同步 Track 对象
        track = self.tracks[trk_idx]
        track.bbox = tracker.get_state().astype(int)
        track.confidence = detection.confidence
        track.landmarks = detection.landmarks
        track.hits += 1
        track.time_since_update = 0

        if track.hits >= self.min_hits:
            track.state = "confirmed"

    def _create_tracker(self, detection):
        """创建新轨迹"""
        tracker = KalmanBoxTracker(detection.bbox)
        self.trackers.append(tracker)

        track = Track(
            track_id=tracker.id,
            bbox=detection.bbox.astype(int),
            confidence=detection.confidence,
            landmarks=detection.landmarks,
            hits=1,
            state="tentative"
        )
        self.tracks.append(track)

    def _remove_dead_trackers(self):
        """移除过期轨迹"""
        alive_indices = []

        for i, (tracker, track) in enumerate(zip(self.trackers, self.tracks)):
            if tracker.time_since_update <= self.max_age:
                alive_indices.append(i)
                track.time_since_update = tracker.time_since_update
                track.age = tracker.age
                if tracker.time_since_update > 0:
                    track.state = "lost"
            # 过期的直接丢弃

        self.trackers = [self.trackers[i] for i in alive_indices]
        self.tracks = [self.tracks[i] for i in alive_indices]

    def _get_confirmed_tracks(self) -> List[Track]:
        """返回已确认的轨迹"""
        return [t for t in self.tracks if t.state == "confirmed" or t.hits >= self.min_hits]

    def reset(self):
        """重置跟踪器"""
        self.trackers = []
        self.tracks = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0
