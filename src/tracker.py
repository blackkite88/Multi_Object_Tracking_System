"""
Name: Kushwanth Vellanki / Mayank Goud Polagoni
Date: 04-23-2026
Purpose: Implements a custom multi-object tracker using Kalman filter and Hungarian matching.
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def bbox_center(bbox):
    """
    Function: bbox_center
    Computes the center point of a bounding box.

    bbox : [x1, y1, x2, y2]

    Returns:
    (cx, cy) : center coordinates
    """
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy


def iou(boxA, boxB):
    """
    Function: iou
    Computes the intersection over union between two bounding boxes.

    boxA : first bounding box [x1, y1, x2, y2]
    boxB : second bounding box [x1, y1, x2, y2]

    Returns:
    iou_value : overlap ratio between 0 and 1
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union = areaA + areaB - inter_area
    if union == 0:
        return 0.0

    return inter_area / union


class Track:
    """
    Class: Track
    Stores one tracked object and its state.
    """

    def __init__(self, track_id, bbox):
        """
        Function: __init__
        Creates a new track with Kalman filter state.

        track_id : unique integer track id
        bbox     : initial bounding box [x1, y1, x2, y2]
        """
        self.track_id = track_id
        self.bbox = bbox
        self.age = 0
        self.missed = 0
        self.history = []

        cx, cy = bbox_center(bbox)

        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32
        )
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        self.kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)

        self.predicted_center = (cx, cy)
        self.history.append((cx, cy))

    def predict(self):
        """
        Function: predict
        Predicts the next object center using the Kalman filter.

        Returns:
        predicted_center : predicted (x, y)
        """
        pred = self.kf.predict()
        px = int(pred[0][0])
        py = int(pred[1][0])
        self.predicted_center = (px, py)
        return self.predicted_center

    def update(self, bbox):
        """
        Function: update
        Updates the track using a new detection bounding box.

        bbox : detected bounding box [x1, y1, x2, y2]
        """
        self.bbox = bbox
        cx, cy = bbox_center(bbox)

        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        self.kf.correct(measurement)

        self.history.append((cx, cy))
        if len(self.history) > 30:
            self.history.pop(0)

        self.age += 1
        self.missed = 0

    def mark_missed(self):
        """
        Function: mark_missed
        Marks the track as missed for one frame.
        """
        self.missed += 1
        self.age += 1


class MultiObjectTracker:
    """
    Class: MultiObjectTracker
    Manages all active tracks and matches detections to tracks.
    """

    def __init__(self, max_missed=10, iou_threshold=0.3, distance_threshold=80, use_iou=True):
        """
        Function: __init__
        Initializes the multi-object tracker.

        max_missed         : maximum missed frames before deleting a track
        iou_threshold      : minimum IOU needed for valid match
        distance_threshold : maximum center distance for valid match
        use_iou            : True for IOU matching, False for distance matching
        """
        self.max_missed = max_missed
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.use_iou = use_iou
        self.tracks = []
        self.next_id = 0

    def _compute_cost_matrix(self, detections):
        """
        Function: _compute_cost_matrix
        Builds a cost matrix between existing tracks and detections.

        detections : list of detections [x1, y1, x2, y2, conf, class_id]

        Returns:
        cost_matrix : 2D numpy array
        """
        num_tracks = len(self.tracks)
        num_dets = len(detections)
        cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)

        for i, track in enumerate(self.tracks):
            track_center = track.predicted_center if track.predicted_center is not None else bbox_center(track.bbox)

            for j, det in enumerate(detections):
                det_bbox = det[:4]

                if self.use_iou:
                    overlap = iou(track.bbox, det_bbox)
                    cost_matrix[i, j] = 1.0 - overlap
                else:
                    det_center = bbox_center(det_bbox)
                    dist = np.linalg.norm(np.array(track_center) - np.array(det_center))
                    cost_matrix[i, j] = dist

        return cost_matrix

    def update(self, detections):
        """
        Function: update
        Updates all tracks using current detections.

        detections : list of detections [x1, y1, x2, y2, conf, class_id]

        Returns:
        tracks : current active tracks
        """
        for track in self.tracks:
            track.predict()

        if len(self.tracks) == 0:
            for det in detections:
                self._start_new_track(det[:4])
            return self.tracks

        if len(detections) == 0:
            for track in self.tracks:
                track.mark_missed()
            self._remove_lost_tracks()
            return self.tracks

        cost_matrix = self._compute_cost_matrix(detections)
        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_idx, col_idx):
            det_bbox = detections[c][:4]

            if self.use_iou:
                match_iou = iou(self.tracks[r].bbox, det_bbox)
                if match_iou < self.iou_threshold:
                    continue
            else:
                track_center = self.tracks[r].predicted_center if self.tracks[r].predicted_center is not None else bbox_center(self.tracks[r].bbox)
                det_center = bbox_center(det_bbox)
                dist = np.linalg.norm(np.array(track_center) - np.array(det_center))
                if dist > self.distance_threshold:
                    continue

            self.tracks[r].update(det_bbox)
            matched_tracks.add(r)
            matched_dets.add(c)

        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track.mark_missed()

        for j, det in enumerate(detections):
            if j not in matched_dets:
                self._start_new_track(det[:4])

        self._remove_lost_tracks()
        return self.tracks

    def _start_new_track(self, bbox):
        """
        Function: _start_new_track
        Creates a new track from a detection.

        bbox : detection bounding box [x1, y1, x2, y2]
        """
        track = Track(self.next_id, bbox)
        self.tracks.append(track)
        self.next_id += 1

    def _remove_lost_tracks(self):
        """
        Function: _remove_lost_tracks
        Removes tracks that have been missed too many times.
        """
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]