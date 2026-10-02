"""
Name: Kushwanth Vellanki / Mayank Goud Polagoni
Date: 04-23-2026
Purpose: Loads a YOLO model and runs object detection on video frames.
"""

from ultralytics import YOLO


class YOLODetector:
    """
    Class: YOLODetector
    Loads a pretrained YOLO model and returns detections from a frame.

    model_path : path to YOLO model weights
    class_ids  : optional set/list of class ids to keep
    conf_thresh: minimum confidence threshold
    """

    def __init__(self, model_path: str = "yolov8n.pt", class_ids=None, conf_thresh: float = 0.4):
        """
        Function: __init__
        Initializes the YOLO detector.

        model_path  : path to the YOLO model weights
        class_ids   : optional class ids to keep, or None for all classes
        conf_thresh : minimum confidence for detections
        """
        self.model = YOLO(model_path)
        self.class_ids = set(class_ids) if class_ids is not None else None
        self.conf_thresh = conf_thresh

    def detect(self, frame):
        """
        Function: detect
        Runs YOLO detection on one frame and returns detections.

        frame : input BGR image

        Returns:
        detections : list of detections in the format
                     [x1, y1, x2, y2, confidence, class_id]
        """
        results = self.model(frame, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                conf = float(box.conf[0].item())
                cls_id = int(box.cls[0].item())

                if conf < self.conf_thresh:
                    continue

                if self.class_ids is not None and cls_id not in self.class_ids:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append([
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                    conf,
                    cls_id
                ])

        return detections