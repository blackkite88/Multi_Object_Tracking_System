"""
Name: Kushwanth Vellanki / Mayank Goud Polagoni
Date: 04-23-2026
Purpose: Runs real-time detection, tracking, and motion prediction.
"""

import time
import cv2

from src.detector import YOLODetector
from src.visualizer import draw_tracks
from src.tracker import MultiObjectTracker


def main():
    """
    Function: main
    Runs the complete webcam-based multi-object tracking pipeline.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detector = YOLODetector(
        model_path="yolov8n.pt",
        class_ids=[0],   # person only
        conf_thresh=0.3
    )

    tracker = MultiObjectTracker(
        max_missed=20,
        iou_threshold=0.3,
        distance_threshold=120,
        use_iou=False
    )

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections)

        frame = draw_tracks(frame, tracks)

        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Multi-Object Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()