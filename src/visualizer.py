"""
Name: Kushwanth Vellanki / Mayank Goud Polagoni
Date: 04-23-2026
Purpose: Draws detections, tracks, labels, trajectories, and predicted points.
"""

import cv2


def draw_detections(frame, detections, class_names=None):
    """
    Function: draw_detections
    Draws detection bounding boxes and labels on the frame.

    frame       : image to draw on
    detections  : list of detections [x1, y1, x2, y2, confidence, class_id]
    class_names : optional dict/list mapping class ids to names

    Returns:
    frame : annotated image
    """
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det

        label = f"Class {cls_id} {conf:.2f}"
        if class_names is not None:
            if isinstance(class_names, dict) and cls_id in class_names:
                label = f"{class_names[cls_id]} {conf:.2f}"
            elif isinstance(class_names, list) and 0 <= cls_id < len(class_names):
                label = f"{class_names[cls_id]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return frame


def draw_tracks(frame, tracks):
    """
    Function: draw_tracks
    Draws tracked object boxes, IDs, trajectory points, and predicted point.

    frame  : image to draw on
    tracks : list of Track objects

    Returns:
    frame : annotated image
    """
    for track in tracks:
        x1, y1, x2, y2 = track.bbox

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"ID {track.track_id}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

        for i in range(1, len(track.history)):
            pt1 = track.history[i - 1]
            pt2 = track.history[i]
            cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

        if track.predicted_center is not None:
            px, py = track.predicted_center
            cv2.circle(frame, (int(px), int(py)), 5, (0, 0, 255), -1)
            cv2.putText(
                frame,
                "Pred",
                (int(px) + 5, int(py) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

    return frame