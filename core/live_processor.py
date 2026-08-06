"""
live_processor.py
=================
Real-time monitoring worker, engineered for a smooth classroom-grade live feed.

Three decoupled stages so the displayed video never stutters:

    1. CameraStream thread   — continuously grabs the newest frame (drops stale
       ones) and keeps it at display resolution.
    2. Detection loop        — reads the latest frame, runs YOLOv8 + head-pose +
       tracking on a smaller copy (for speed), scores behaviour, and publishes
       the latest results. Runs as fast as the accelerator allows.
    3. frames() generator    — streams the freshest camera frame with the latest
       detection overlay at a fixed display FPS, independent of detection speed.

Because display is decoupled from detection, the feed stays fluid (≈30 FPS)
even while heavy inference runs, and uses the GPU (CUDA / Apple-Silicon MPS)
automatically when available.
"""

import threading
import time
from collections import deque

import cv2

import config
from detection.head_pose_detector import HeadPoseDetector
from detection.object_detector import ObjectDetector
from detection.tracker import StudentTracker
from detection.behavior_analyzer import BehaviorAnalyzer
from .annotator import annotate_frame, resize_max_width


class CameraStream:
    """Threaded, latest-frame-only capture, kept at display resolution."""

    def __init__(self, source, target_width=0):
        self.source = source
        self.target_width = target_width
        self.is_file = isinstance(source, str) and not str(source).isdigit()
        self.cap = None
        self.frame = None
        self.seq = 0
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not self.cap.isOpened():
            return False
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        return True

    def _reader(self):
        while self.running:
            ok, fr = self.cap.read()
            if not ok:
                if self.is_file:
                    self.running = False
                    break
                time.sleep(0.02)
                continue
            if self.target_width:
                fr = resize_max_width(fr, self.target_width)
            with self.lock:
                self.frame = fr
                self.seq += 1

    def read(self):
        with self.lock:
            if self.frame is None:
                return self.seq, None
            return self.seq, self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.5)
        if self.cap is not None:
            self.cap.release()


class LiveProcessor:
    def __init__(self, source=0, device=None, sensitivity=None):
        self.source = source
        self.device = config.resolve_device(device)
        self.sensitivity = sensitivity or config.DEFAULT_SENSITIVITY
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._stream = None
        self._error = None
        self._engines = {}
        self._overlay = {"result": {"students": [], "proximity_pairs": []},
                         "objects": [], "meta": {}}
        self._stats = {"running": False, "students": [], "alerts": [],
                       "total_alerts": 0, "fps": 0.0, "frame": 0,
                       "processed": 0, "engines": {}}

    # ------------------------------------------------------------------ #
    def start(self):
        if self._running:
            return
        self._running = True
        self._error = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        with self._lock:
            self._stats["running"] = False

    def is_running(self):
        return self._running

    # ------------------------------------------------------------------ #
    def _loop(self):
        stream = CameraStream(self.source, config.LIVE_MAX_WIDTH)
        if not stream.start():
            self._error = ("Unable to access the camera/source. Check the device "
                           "index/permissions and that no other app is using it.")
            self._running = False
            with self._lock:
                self._stats["running"] = False
            return
        self._stream = stream

        try:
            head_pose = HeadPoseDetector()
            detector = ObjectDetector(
                device=self.device,
                sensitivity=self.sensitivity,
                world_frame_interval=config.LIVE_WORLD_FRAME_INTERVAL,
            )
            tracker = StudentTracker(device=self.device,
                                     prefer_deepsort=config.LIVE_USE_DEEPSORT)
            analyzer = BehaviorAnalyzer(head_pose=head_pose, sensitivity=self.sensitivity)
            analyzer.set_active_signals(
                head_pose=head_pose.available,
                object=detector.supports_prohibited_objects,
            )
        except Exception as exc:  # noqa: BLE001
            self._error = f"Failed to initialise detection engines: {exc}"
            self._running = False
            stream.stop()
            return

        self._engines = {
            "object_detection": detector.engine_label,
            "prohibited_item_detection": (
                detector.engine_label if detector.supports_prohibited_objects else "unavailable"
            ),
            "head_pose": head_pose.engine_label,
            "tracking": "DeepSORT" if tracker.backend == "deepsort" else "IoU tracker",
            "device": self.device,
            "sensitivity": config.sensitivity_profile(self.sensitivity)["label"],
        }

        idx = 0
        last_seq = -1
        recent = deque(maxlen=60)
        t0 = time.time()
        fps_t, fps_n, fps = time.time(), 0, 0.0
        det_w = config.LIVE_DETECT_WIDTH

        while self._running:
            seq, frame = stream.read()
            if frame is None or seq == last_seq:
                if stream.is_file and not stream.running and seq == last_seq:
                    break
                time.sleep(0.003)
                continue
            last_seq = seq
            ts = time.time() - t0

            try:
                det = resize_max_width(frame, det_w) if det_w else frame
                scale = frame.shape[1] / float(det.shape[1])
                persons, objs = detector.detect(det)
                tracked = tracker.update(persons, det)
                faces = head_pose.process(det)
                result = analyzer.update(idx, ts, tracked, faces, objs, det.shape)
                o_result, o_objs = self._scale_overlay(result, objs, scale)
                for a in result["alerts"]:
                    recent.appendleft(a)
            except Exception:
                idx += 1
                continue

            fps_n += 1
            now = time.time()
            if now - fps_t >= 0.5:
                fps = round(fps_n / (now - fps_t), 1)
                fps_t, fps_n = now, 0

            meta = {"frame": idx, "time_str": analyzer._fmt_time(ts),
                    "total_alerts": len(analyzer.alerts), "fps": fps}
            with self._lock:
                self._overlay = {"result": o_result, "objects": o_objs, "meta": meta}
                self._stats = {
                    "running": True,
                    "students": result["students"],
                    "alerts": list(recent)[:20],
                    "total_alerts": len(analyzer.alerts),
                    "fps": fps,
                    "frame": idx,
                    "processed": idx + 1,
                    "engines": self._engines,
                }
            idx += 1

        stream.stop()
        head_pose.close()
        self._running = False
        with self._lock:
            self._stats["running"] = False

    @staticmethod
    def _scale_overlay(result, objects, scale):
        """Scale detection-resolution boxes up to display resolution."""
        if abs(scale - 1.0) < 1e-3:
            return result, objects

        def sb(b):
            return [int(b[0] * scale), int(b[1] * scale),
                    int(b[2] * scale), int(b[3] * scale)]

        students = []
        for st in result.get("students", []):
            s2 = dict(st)
            s2["bbox"] = sb(st["bbox"])
            s2["center"] = (int(st["center"][0] * scale), int(st["center"][1] * scale))
            students.append(s2)
        objs = [{**o, "bbox": sb(o["bbox"])} for o in objects]
        return {"students": students,
                "proximity_pairs": result.get("proximity_pairs", [])}, objs

    # ------------------------------------------------------------------ #
    def frames(self):
        """MJPEG generator — fixed display FPS, decoupled from detection."""
        interval = 1.0 / max(1, config.LIVE_DISPLAY_FPS)
        while self._running:
            stream = self._stream
            if stream is None:
                time.sleep(0.05)
                continue
            seq, frame = stream.read()
            if frame is None:
                time.sleep(0.03)
                continue
            with self._lock:
                ov = self._overlay
            try:
                annotate_frame(frame, ov["result"], ov["objects"], ov["meta"])
                ok, buf = cv2.imencode(".jpg", frame,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])
                if ok:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            except Exception:
                pass
            time.sleep(interval)

    def stats(self):
        with self._lock:
            s = dict(self._stats)
        if self._error:
            s["error"] = self._error
        return s
