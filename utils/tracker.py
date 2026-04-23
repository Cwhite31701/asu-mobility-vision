"""
SORT (Simple Online and Realtime Tracking) tracker.
Lightweight Kalman filter-based tracker for multi-object tracking.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(bb_test, bb_gt):
    """Compute IOU between two bounding boxes [x1,y1,x2,y2]."""
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])

    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    intersection = w * h

    area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_gt   = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area_test + area_gt - intersection + 1e-6
    return intersection / union


class KalmanBoxTracker:
    """Tracks a single object with a constant-velocity Kalman filter."""
    count = 0

    def __init__(self, bbox, cls_id):
        from filterpy.kalman import KalmanFilter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=float)
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=float)
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.cls_id = cls_id

    def _convert_bbox_to_z(self, bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / max(h, 1e-6)
        return np.array([x, y, s, r]).reshape((4, 1))

    def _convert_x_to_bbox(self, x):
        w = np.sqrt(max(x[2] * x[3], 1e-6))
        h = x[2] / max(w, 1e-6)
        return np.array([
            x[0] - w / 2.0,
            x[1] - h / 2.0,
            x[0] + w / 2.0,
            x[1] + h / 2.0,
        ]).flatten()

    def update(self, bbox, cls_id):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.cls_id = cls_id
        self.kf.update(self._convert_bbox_to_z(bbox))

    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return self._convert_x_to_bbox(self.kf.x)


class SORTTracker:
    """Multi-object SORT tracker."""

    def __init__(self, max_age=10, min_hits=2, iou_threshold=0.25):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections: list[dict]) -> list[dict]:
        """
        Args:
            detections: list of dicts with keys: x1, y1, x2, y2, cls_id, conf

        Returns:
            list of tracked objects: x1, y1, x2, y2, track_id, cls_id
        """
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)
            trks = np.delete(trks, t, axis=0)

        if len(detections) == 0:
            matched_dets = set()
            unmatched_dets = []
        else:
            det_array = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections])

            if len(self.trackers) == 0:
                matched_dets = set()
                unmatched_dets = list(range(len(detections)))
            else:
                # Build IOU matrix
                iou_matrix = np.zeros((len(detections), len(self.trackers)))
                for di, det in enumerate(detections):
                    for ti, trk in enumerate(self.trackers):
                        iou_matrix[di, ti] = iou(
                            [det["x1"], det["y1"], det["x2"], det["y2"]],
                            trks[ti, :4].tolist()
                        )

                row_ind, col_ind = linear_sum_assignment(-iou_matrix)
                matched_dets = set()
                for r, c in zip(row_ind, col_ind):
                    if iou_matrix[r, c] >= self.iou_threshold:
                        self.trackers[c].update(
                            [detections[r]["x1"], detections[r]["y1"],
                             detections[r]["x2"], detections[r]["y2"]],
                            detections[r]["cls_id"]
                        )
                        matched_dets.add(r)
                unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            d = detections[i]
            trk = KalmanBoxTracker([d["x1"], d["y1"], d["x2"], d["y2"]], d["cls_id"])
            self.trackers.append(trk)

        # Collect results
        ret = []
        to_del = []
        for t, trk in enumerate(self.trackers):
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                state = trk.get_state()
                ret.append({
                    "x1": float(state[0]),
                    "y1": float(state[1]),
                    "x2": float(state[2]),
                    "y2": float(state[3]),
                    "track_id": trk.id,
                    "cls_id": trk.cls_id,
                })
            if trk.time_since_update > self.max_age:
                to_del.append(t)

        for t in reversed(to_del):
            self.trackers.pop(t)

        return ret
