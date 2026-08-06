"""
head_pose_detector.py
=====================
Estimates head orientation (yaw / pitch / roll) for every face in a frame and
classifies sideways / downward gaze.

Engine auto-selection (most accurate available wins):
    1. MediaPipe FaceMesh    — legacy ``mediapipe.solutions.face_mesh`` API
    2. MediaPipe FaceLandmarker — new ``mediapipe.tasks`` API (model auto-
       downloaded to models/face_landmarker.task on first run)
    3. OpenCV Haar cascades  — coarse, dependency-free fallback

All engines feed the SAME geometry: selected facial landmarks + a
Perspective-n-Point (solvePnP) solve against a generic 3D face model. This
keeps yaw/pitch/roll consistent regardless of which MediaPipe API is installed.

`process()` never raises — on any internal error it returns [], so a single bad
frame can never crash a monitoring session.

A per-face rolling baseline ("dynamic adjustment") absorbs each student's
natural resting posture so thresholds adapt to camera angle and seating.
"""

import math
import os
from collections import deque, defaultdict

import cv2
import numpy as np

import config

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except Exception:  # pragma: no cover
    mp = None
    _MP_AVAILABLE = False

# Legacy solutions API (older mediapipe)
try:
    from mediapipe import solutions as _mp_solutions
    _LEGACY_AVAILABLE = True
except Exception:
    _mp_solutions = None
    _LEGACY_AVAILABLE = False

# New tasks API (mediapipe >= ~0.10 where solutions was dropped)
try:
    from mediapipe.tasks import python as _mp_tasks
    from mediapipe.tasks.python import vision as _mp_vision
    _TASKS_AVAILABLE = True
except Exception:
    _mp_tasks = _mp_vision = None
    _TASKS_AVAILABLE = False


_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),        # 1   nose tip
        (0.0, -63.6, -12.5),    # 199 chin
        (-43.3, 32.7, -26.0),   # 33  left eye, outer corner
        (43.3, 32.7, -26.0),    # 263 right eye, outer corner
        (-28.9, -28.9, -24.1),  # 61  left mouth corner
        (28.9, -28.9, -24.1),   # 291 right mouth corner
    ],
    dtype=np.float64,
)
_LANDMARK_IDS = [1, 199, 33, 263, 61, 291]


def _rotation_matrix_to_euler(R):
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy >= 1e-6:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0.0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


class HeadPoseDetector:
    def __init__(self, cfg: dict = None):
        self.cfg = cfg or config.HEAD_POSE
        self._baselines = defaultdict(lambda: {"yaw": deque(maxlen=60),
                                               "pitch": deque(maxlen=60)})
        self.mode = "disabled"
        self._mesh = None            # legacy FaceMesh
        self._landmarker = None      # tasks FaceLandmarker
        self._face_cascade = self._profile_cascade = self._eye_cascade = None

        # 1) legacy FaceMesh
        if _LEGACY_AVAILABLE:
            try:
                self._mesh = _mp_solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=self.cfg["max_num_faces"],
                    refine_landmarks=True,
                    min_detection_confidence=self.cfg["min_detection_confidence"],
                    min_tracking_confidence=self.cfg["min_tracking_confidence"],
                )
                self.mode = "mediapipe"
            except Exception:
                self._mesh = None

        # 2) tasks FaceLandmarker
        if self.mode == "disabled" and _TASKS_AVAILABLE:
            model = self._ensure_tasks_model()
            if model:
                try:
                    opts = _mp_vision.FaceLandmarkerOptions(
                        base_options=_mp_tasks.BaseOptions(model_asset_path=model),
                        running_mode=_mp_vision.RunningMode.IMAGE,
                        num_faces=self.cfg["max_num_faces"],
                        min_face_detection_confidence=self.cfg["min_detection_confidence"],
                    )
                    self._landmarker = _mp_vision.FaceLandmarker.create_from_options(opts)
                    self.mode = "mediapipe_tasks"
                except Exception:
                    self._landmarker = None

        # 3) Haar fallback
        if self.mode == "disabled":
            try:
                base = cv2.data.haarcascades
                self._face_cascade = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
                self._profile_cascade = cv2.CascadeClassifier(base + "haarcascade_profileface.xml")
                self._eye_cascade = cv2.CascadeClassifier(base + "haarcascade_eye.xml")
                if not self._face_cascade.empty():
                    self.mode = "haar"
            except Exception:
                self.mode = "disabled"

    def _ensure_tasks_model(self):
        path = self.cfg.get("tasks_model")
        if path and os.path.exists(path):
            return path
        url = self.cfg.get("tasks_model_url")
        if not (path and url):
            return None
        try:
            import urllib.request
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(url, path)
            return path if os.path.exists(path) else None
        except Exception:
            return None

    @property
    def available(self):
        return self.mode != "disabled"

    @property
    def engine_label(self):
        return {
            "mediapipe": "MediaPipe FaceMesh",
            "mediapipe_tasks": "MediaPipe FaceLandmarker",
            "haar": "OpenCV Haar (fallback)",
            "disabled": "disabled",
        }.get(self.mode, self.mode)

    # ------------------------------------------------------------------ #
    def process(self, frame):
        """Return [{"bbox","center","yaw","pitch","roll"}]; never raises."""
        try:
            if self.mode in ("mediapipe", "mediapipe_tasks"):
                return self._process_landmarks(frame)
            if self.mode == "haar":
                return self._process_haar(frame)
        except Exception:
            return []
        return []

    def _extract_face_landmarks(self, frame):
        """Return a list (per face) of normalised (x, y) landmark tuples."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = []
        if self.mode == "mediapipe":
            rgb.flags.writeable = False
            res = self._mesh.process(rgb)
            if res.multi_face_landmarks:
                for f in res.multi_face_landmarks:
                    out.append([(lm.x, lm.y) for lm in f.landmark])
        else:  # mediapipe_tasks
            rgb = np.ascontiguousarray(rgb)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self._landmarker.detect(mp_img)
            for f in res.face_landmarks:
                out.append([(lm.x, lm.y) for lm in f])
        return out

    def _process_landmarks(self, frame):
        h, w = frame.shape[:2]
        faces_lms = self._extract_face_landmarks(frame)
        if not faces_lms:
            return []

        focal = float(w)
        cam = np.array([[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros((4, 1), dtype=np.float64)

        faces = []
        for lms in faces_lms:
            if len(lms) <= max(_LANDMARK_IDS):
                continue
            image_points = np.array([(lms[i][0] * w, lms[i][1] * h) for i in _LANDMARK_IDS],
                                    dtype=np.float64)
            xs = [p[0] * w for p in lms]
            ys = [p[1] * h for p in lms]
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            ok, rvec, _t = cv2.solvePnP(_MODEL_POINTS, image_points, cam, dist,
                                        flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            rmat, _ = cv2.Rodrigues(rvec)
            pitch, yaw, roll = _rotation_matrix_to_euler(rmat)
            faces.append({
                "bbox": [x1, y1, x2, y2],
                "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                "yaw": round(float(yaw), 2),
                "pitch": round(float(pitch), 2),
                "roll": round(float(roll), 2),
            })
        return faces

    def _process_haar(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        faces = []
        for (x, y, fw, fh) in self._face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40)):
            roi = gray[y:y + fh, x:x + fw]
            eyes = self._eye_cascade.detectMultiScale(roi, 1.1, 6,
                                                      minSize=(int(fw * 0.1), int(fh * 0.1)))
            yaw, pitch = 0.0, 0.0
            if len(eyes) >= 2:
                eyes = sorted(eyes, key=lambda e: e[0])
                mid_x = (eyes[0][0] + eyes[0][2] / 2 + eyes[-1][0] + eyes[-1][2] / 2) / 2
                yaw = float((mid_x - fw / 2) / (fw / 2) * 30.0)
                eye_y = (eyes[0][1] + eyes[-1][1]) / 2 / fh
                pitch = float((eye_y - 0.38) * 55.0)
            else:
                pitch = 22.0
            faces.append({
                "bbox": [int(x), int(y), int(x + fw), int(y + fh)],
                "center": (int(x + fw // 2), int(y + fh // 2)),
                "yaw": round(yaw, 1), "pitch": round(pitch, 1), "roll": 0.0,
            })
        for (x, y, fw, fh) in self._profile_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40)):
            cx, cy = x + fw // 2, y + fh // 2
            if any(f["bbox"][0] <= cx <= f["bbox"][2] and f["bbox"][1] <= cy <= f["bbox"][3]
                   for f in faces):
                continue
            faces.append({"bbox": [int(x), int(y), int(x + fw), int(y + fh)],
                          "center": (int(cx), int(cy)),
                          "yaw": 35.0, "pitch": 0.0, "roll": 0.0})
        return faces

    # ------------------------------------------------------------------ #
    def classify(self, student_id, yaw, pitch):
        yaw_thr = self.cfg["yaw_threshold"]
        pitch_down = self.cfg["pitch_down_threshold"]
        pitch_up = self.cfg["pitch_up_threshold"]

        yaw_adj, pitch_adj = yaw, pitch
        if self.cfg.get("dynamic_adjustment", False):
            b = self._baselines[student_id]
            b["yaw"].append(yaw)
            b["pitch"].append(pitch)
            if len(b["yaw"]) >= 10:
                yaw_adj = yaw - float(np.median(b["yaw"]))
                pitch_adj = pitch - float(np.median(b["pitch"]))
                margin = self.cfg.get("dynamic_margin", 0.0)
                yaw_thr += margin
                pitch_down += margin
                pitch_up += margin

        return {
            "sideways": abs(yaw_adj) > yaw_thr,
            "down": pitch_adj > pitch_down,
            "up": pitch_adj < -pitch_up,
            "yaw_adj": round(yaw_adj, 2),
            "pitch_adj": round(pitch_adj, 2),
        }

    def close(self):
        for obj in (self._mesh, self._landmarker):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass
        # Drop references so the underlying engine is finalised while the
        # interpreter is still alive (avoids noisy shutdown-time finalisers).
        self._mesh = None
        self._landmarker = None
        self.mode = "disabled"
