"""
object_detector.py
==================
YOLOv8-powered detection of people and prohibited classroom items
(smartphones, books, laptops, etc.).

* Person detections seed the DeepSORT tracker.
* Prohibited-item detections feed the behaviour analyzer and raise alerts when
  they persist for `alert_frames` consecutive frames.

Per-class confidence thresholds (config.OBJECT_DETECTION["prohibited_classes"])
are tuned for classroom environments to minimise false positives.
"""

import os

import config

try:
    from ultralytics import YOLO, YOLOWorld
    _YOLO_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    YOLO = None
    YOLOWorld = None
    _YOLO_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    cv2 = None
    _CV2_AVAILABLE = False


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class ObjectDetector:
    def __init__(self, model_path: str = None, device: str = "cpu",
                 cfg: dict = None, sensitivity: str = None,
                 world_frame_interval: int = 1):
        self.cfg = cfg or config.OBJECT_DETECTION
        self.profile = config.sensitivity_profile(sensitivity)
        self.device = device
        self.backend = "disabled"
        self.model = None
        self.world_model = None
        self.world_available = False
        self.world_frame_interval = max(1, int(world_frame_interval or 1))
        self._frame_counter = 0
        self._last_world = ([], [])
        self.hog = None
        self.last_error = None
        self.conf = self.cfg["confidence_threshold"]
        self.world_conf = self.cfg["world_confidence_threshold"]
        self.iou = self.cfg["iou_threshold"]
        self.prohibited = self.cfg["prohibited_classes"]
        self.aliases = self.cfg.get("label_aliases", {})
        self.person_class = self.cfg["person_class"]
        self.names = {}
        self.world_names = {}

        if _YOLO_AVAILABLE:
            try:
                path = model_path or self.cfg["model_path"]
                # ultralytics auto-downloads the weights file if it is not present.
                if not os.path.exists(path):
                    # Fall back to the model name so ultralytics fetches it to cache.
                    path = os.path.basename(path)
                self.model = YOLO(path)
                # ultralytics exposes id->name mapping
                self.names = self.model.names
                self.backend = "yolov8"
                self._init_world_detector()
                return
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                self.model = None

        if _CV2_AVAILABLE:
            try:
                self.hog = cv2.HOGDescriptor()
                self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                self.backend = "opencv_hog"
                return
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                self.hog = None

        raise ImportError(
            "No usable object detector is available. Install ultralytics for "
            "YOLOv8, or install opencv-python for the fallback person detector."
        )

    @property
    def engine_label(self):
        if self.backend == "yolov8":
            return "YOLOv8 + YOLO-World" if self.world_available else "YOLOv8"
        if self.backend == "opencv_hog":
            return "OpenCV HOG person fallback"
        return "unavailable"

    @property
    def supports_prohibited_objects(self):
        return self.backend == "yolov8" or self.world_available

    def _init_world_detector(self):
        if not (self.cfg.get("enable_world_detector") and YOLOWorld is not None):
            return
        try:
            path = self.cfg["world_model_path"]
            if not os.path.exists(path):
                path = os.path.basename(path)
            self.world_model = YOLOWorld(path)
            self.world_model.set_classes(self.cfg["world_classes"])
            self.world_names = self.world_model.names
            self.world_available = True
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            self.world_model = None
            self.world_available = False

    def _label(self, cls_id: int) -> str:
        if isinstance(self.names, dict):
            return self.names.get(cls_id, str(cls_id))
        try:
            return self.names[cls_id]
        except Exception:
            return str(cls_id)

    def _world_label(self, cls_id: int) -> str:
        if isinstance(self.world_names, dict):
            return self.world_names.get(cls_id, str(cls_id))
        try:
            return self.world_names[cls_id]
        except Exception:
            return str(cls_id)

    def _canonical(self, label: str) -> str:
        return self.aliases.get(label, label)

    def _threshold(self, label: str, world=False) -> float:
        base = self.prohibited.get(label, self.prohibited.get(self._canonical(label), self.conf))
        scale = self.profile["world_conf_scale"] if world else self.profile["object_conf_scale"]
        floor = 0.05 if world else 0.10
        return max(floor, float(base) * float(scale))

    def detect(self, frame):
        """
        Run detection on a BGR frame.

        Returns (persons, objects):
            persons -> [{"bbox": [x1,y1,x2,y2], "conf": float}, ...]
            objects -> [{"bbox": [x1,y1,x2,y2], "conf": float, "label": str}, ...]
        """
        if self.backend == "opencv_hog":
            return self._detect_hog(frame)
        return self._detect_yolo(frame)

    def _detect_yolo(self, frame):
        results = self.model.predict(
            frame, conf=max(0.10, self.conf * self.profile["object_conf_scale"]), iou=self.iou,
            device=self.device, verbose=False,
        )[0]

        persons, objects = [], []
        if results.boxes is None:
            return persons, objects

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self._label(cls_id)
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            if label == self.person_class:
                persons.append({"bbox": [x1, y1, x2, y2], "conf": round(conf, 3)})
            canonical = self._canonical(label)
            if label != self.person_class and canonical in self.prohibited and conf >= self._threshold(canonical):
                objects.append({
                    "bbox": [x1, y1, x2, y2],
                    "conf": round(conf, 3),
                    "label": canonical,
                    "source": "yolov8",
                })
        if self.world_available:
            try:
                w_persons, w_objects = self._detect_world_sampled(frame)
                if not persons:
                    persons = w_persons
                objects.extend(w_objects)
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                self.world_available = False
        return persons, self._dedupe(objects)

    def _detect_world_sampled(self, frame):
        self._frame_counter += 1
        should_run = (
            self._frame_counter == 1
            or self.world_frame_interval <= 1
            or self._frame_counter % self.world_frame_interval == 0
        )
        if should_run:
            self._last_world = self._detect_world(frame)
        return self._last_world

    def _detect_world(self, frame):
        results = self.world_model.predict(
            frame,
            conf=max(0.05, self.world_conf * self.profile["world_conf_scale"]),
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]

        persons, objects = [], []
        if results.boxes is None:
            return persons, objects

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self._world_label(cls_id)
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            canonical = self._canonical(label)

            if canonical == self.person_class:
                persons.append({"bbox": [x1, y1, x2, y2], "conf": round(conf, 3)})
            elif canonical in self.prohibited and conf >= self._threshold(canonical, world=True):
                objects.append({
                    "bbox": [x1, y1, x2, y2],
                    "conf": round(conf, 3),
                    "label": canonical,
                    "raw_label": label,
                    "source": "yolo_world",
                })
        return persons, objects

    def _dedupe(self, objects):
        keep = []
        thr = self.cfg.get("dedupe_iou_threshold", 0.55)
        for obj in sorted(objects, key=lambda o: o.get("conf", 0.0), reverse=True):
            duplicate = False
            for existing in keep:
                if existing["label"] == obj["label"] and _iou(existing["bbox"], obj["bbox"]) >= thr:
                    duplicate = True
                    break
            if not duplicate:
                keep.append(obj)
        return keep

    def _detect_hog(self, frame):
        """Dependency-light fallback: detect people only, never prohibited items."""
        persons = []
        if self.hog is None:
            return persons, []

        rects, weights = self.hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        for (x, y, w, h), weight in zip(rects, weights):
            conf = float(weight)
            if conf < 0.35:
                continue
            persons.append({
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "conf": round(min(0.99, max(0.35, conf / 2.0)), 3),
            })
        return persons, []
