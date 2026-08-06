"""
config.py
=========
Central configuration for the AI-Powered Classroom Cheating Detection System.

Administrators can tune every detection parameter here WITHOUT touching the
detection / application code. Changes apply on the next session start (live)
or the next processing job (video upload).

Sections
--------
1. Paths & Flask settings
2. Compute / device settings
3. Object detection (YOLOv8)
4. Head pose detection (MediaPipe + solvePnP)
5. Tracking (DeepSORT)
6. Movement & zone monitoring
7. Behavioral scoring & severity levels
"""

import os

# --------------------------------------------------------------------------- #
# 1. Paths & Flask settings
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
DATA_FOLDER = os.path.join(BASE_DIR, "data")

for _d in (UPLOAD_FOLDER, OUTPUT_FOLDER, REPORT_FOLDER, MODEL_FOLDER, DATA_FOLDER):
    os.makedirs(_d, exist_ok=True)

# Flask
SECRET_KEY = os.environ.get("SECRET_KEY", "cheatguard-ai-secret-key-change-me")
# 500 MB upload ceiling (matches the documented limit)
MAX_CONTENT_LENGTH = 500 * 1024 * 1024
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

# History database (lightweight JSON index of processed jobs)
HISTORY_INDEX = os.path.join(DATA_FOLDER, "history.json")

# --------------------------------------------------------------------------- #
# 2. Compute / device settings
# --------------------------------------------------------------------------- #
# "cpu" or "cuda". Auto-detected at runtime if set to "auto".
DEVICE = os.environ.get("DEVICE", "auto")

# Process every Nth frame for heavy detectors during live mode to keep FPS high.
# 1 = every frame. Increase on slower CPUs.
LIVE_FRAME_SKIP = int(os.environ.get("LIVE_FRAME_SKIP", "1"))
VIDEO_FRAME_SKIP = int(os.environ.get("VIDEO_FRAME_SKIP", "1"))

# Target/expected FPS used for velocity normalisation when source FPS is unknown
DEFAULT_FPS = 30.0

# Frames are downscaled to these max widths before detection to keep real-time
# performance high (set to 0 to disable and process at native resolution).
LIVE_MAX_WIDTH = int(os.environ.get("LIVE_MAX_WIDTH", "960"))
# Detection runs on a smaller copy for speed; the display stays at LIVE_MAX_WIDTH.
LIVE_DETECT_WIDTH = int(os.environ.get("LIVE_DETECT_WIDTH", "640"))
VIDEO_MAX_WIDTH = int(os.environ.get("VIDEO_MAX_WIDTH", "1280"))
# JPEG quality for the live MJPEG stream (1-100).
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "75"))
# Target display frame-rate of the live stream (decoupled from detection rate).
LIVE_DISPLAY_FPS = int(os.environ.get("LIVE_DISPLAY_FPS", "30"))
# Live mode uses the fast IoU tracker by default (no neural re-ID per frame);
# video processing uses DeepSORT for maximum identity accuracy.
LIVE_USE_DEEPSORT = os.environ.get("LIVE_USE_DEEPSORT", "0") in ("1", "true", "True")
# YOLO-World is heavier than COCO YOLO. Live mode samples it every N detection
# frames and reuses the latest boxes between samples to keep the stream smooth.
LIVE_WORLD_FRAME_INTERVAL = int(os.environ.get("LIVE_WORLD_FRAME_INTERVAL", "2"))
VIDEO_WORLD_FRAME_INTERVAL = int(os.environ.get("VIDEO_WORLD_FRAME_INTERVAL", "1"))

# --------------------------------------------------------------------------- #
# 3. Object detection (YOLOv8 + optional YOLO-World)
# --------------------------------------------------------------------------- #
OBJECT_DETECTION = {
    # Weights are auto-downloaded by ultralytics on first run if not present.
    # yolov8s = good accuracy/speed balance (recommended). Use yolov8n.pt for
    # weak CPUs, or yolov8m.pt / yolov8l.pt for maximum accuracy on a GPU.
    # Swap for custom classroom-tuned weights by placing them in models/.
    "model_path": os.path.join(MODEL_FOLDER, "yolov8s.pt"),
    # YOLO-World adds open-vocabulary detection for exam-specific items that
    # COCO YOLOv8 does not know well, especially headphones / earbuds.
    "enable_world_detector": os.environ.get("ENABLE_WORLD_DETECTOR", "1")
        not in ("0", "false", "False"),
    "world_model_path": os.path.join(MODEL_FOLDER, "yolov8s-worldv2.pt"),
    # Global confidence floor for any detection to be considered.
    "confidence_threshold": 0.30,
    "world_confidence_threshold": 0.12,
    "iou_threshold": 0.45,
    # COCO class id for "person" — used to seed the tracker.
    "person_class": "person",
    # Prohibited / suspicious items mapped to a per-class confidence threshold.
    # Thresholds are tuned for classroom environments to limit false alerts.
    "prohibited_classes": {
        "cell phone": 0.25,
        "phone": 0.20,
        "smartphone": 0.20,
        "book": 0.40,
        "textbook": 0.30,
        "paper": 0.24,
        "notes": 0.20,
        "cheat sheet": 0.18,
        "calculator": 0.22,
        "laptop": 0.40,
        "remote": 0.40,   # often confused with phones; kept for coverage
        "tablet": 0.40,
        "headphones": 0.10,
        "headset": 0.10,
        "earphones": 0.09,
        "earbuds": 0.08,
        "wireless earbuds": 0.08,
        "smart watch": 0.16,
    },
    "world_classes": [
        "person",
        "cell phone",
        "phone",
        "smartphone",
        "book",
        "textbook",
        "paper",
        "notes",
        "cheat sheet",
        "calculator",
        "laptop",
        "tablet",
        "headphones",
        "headset",
        "earphones",
        "earbuds",
        "wireless earbuds",
        "smart watch",
    ],
    "label_aliases": {
        "phone": "cell phone",
        "smartphone": "cell phone",
        "textbook": "book",
        "notes": "paper",
        "cheat sheet": "paper",
        "headset": "headphones",
        "earphones": "headphones",
        "earbuds": "headphones",
        "wireless earbuds": "headphones",
    },
    "audio_device_labels": {"headphones", "headset", "earphones", "earbuds", "wireless earbuds"},
    # High-confidence malpractice items should push risk faster than a generic object.
    "risk_increments": {
        "headphones": 0.36,
        "cell phone": 0.32,
        "smart watch": 0.30,
        "calculator": 0.26,
        "paper": 0.24,
        "book": 0.22,
        "laptop": 0.20,
        "tablet": 0.28,
    },
    "alert_frames_by_class": {
        "headphones": 3,
        "cell phone": 4,
        "smart watch": 4,
        "calculator": 5,
        "paper": 5,
    },
    # A prohibited object must persist for this many frames before raising an alert.
    "alert_frames": 6,
    "dedupe_iou_threshold": 0.55,
    # Colours (BGR) used to draw object boxes.
    "box_color": (0, 140, 255),
    "person_box_color": (0, 255, 0),
}

SENSITIVITY_PROFILES = {
    "strict": {
        "label": "Strict Exam",
        "object_conf_scale": 0.78,
        "world_conf_scale": 0.72,
        "alert_frame_scale": 0.65,
        "risk_scale": 1.18,
    },
    "balanced": {
        "label": "Balanced",
        "object_conf_scale": 1.0,
        "world_conf_scale": 1.0,
        "alert_frame_scale": 1.0,
        "risk_scale": 1.0,
    },
    "conservative": {
        "label": "Conservative",
        "object_conf_scale": 1.18,
        "world_conf_scale": 1.22,
        "alert_frame_scale": 1.35,
        "risk_scale": 0.88,
    },
}
DEFAULT_SENSITIVITY = os.environ.get("DEFAULT_SENSITIVITY", "strict")

# --------------------------------------------------------------------------- #
# 4. Head pose detection (MediaPipe FaceMesh + solvePnP)
# --------------------------------------------------------------------------- #
HEAD_POSE = {
    # Yaw magnitude (degrees) above which the student is "looking sideways".
    "yaw_threshold": 22.0,
    # Pitch (degrees) below which the student is "looking down" (notes/phone).
    "pitch_down_threshold": 18.0,
    # Pitch (degrees) above which the student is "looking up".
    "pitch_up_threshold": 18.0,
    # Sustained frames before a head-pose alert fires (kills momentary glances).
    "alert_frames": 12,
    # Dynamic threshold adaptation: widens/narrows thresholds based on the
    # rolling baseline of a student's natural posture.
    "dynamic_adjustment": True,
    "dynamic_margin": 6.0,
    # FaceMesh runtime config
    "max_num_faces": 8,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    # New MediaPipe Tasks API model (used when the legacy solutions.face_mesh
    # API is unavailable). Auto-downloaded to models/ on first run if missing.
    "tasks_model": os.path.join(MODEL_FOLDER, "face_landmarker.task"),
    "tasks_model_url": ("https://storage.googleapis.com/mediapipe-models/"
                        "face_landmarker/face_landmarker/float16/1/"
                        "face_landmarker.task"),
}

# --------------------------------------------------------------------------- #
# 5. Tracking (DeepSORT)
# --------------------------------------------------------------------------- #
TRACKING = {
    "max_age": 30,          # frames to keep a lost track alive
    "n_init": 3,            # detections before a track is confirmed
    "max_iou_distance": 0.7,
    "max_cosine_distance": 0.3,
    "nn_budget": 100,
    "embedder": "mobilenet",  # appearance embedder for re-identification
    "half": False,
    # Fallback IoU tracker (used automatically if deep-sort-realtime is not
    # installed or fails to initialise) — minimum IoU to associate a detection
    # with an existing track.
    "iou_match_threshold": 0.3,
}

# --------------------------------------------------------------------------- #
# 6. Movement & zone monitoring
# --------------------------------------------------------------------------- #
MOVEMENT = {
    # Number of past centroids retained per student for velocity/accel analysis.
    "history_length": 30,
    # Velocity (pixels / frame) above which movement is "excessive".
    "velocity_threshold": 18.0,
    # Acceleration (pixels / frame^2) above which movement is "erratic".
    "acceleration_threshold": 10.0,
    # Distance (pixels) below which two students are "too close" (interaction).
    "proximity_threshold": 130.0,
    # Sustained frames before a movement/proximity alert fires.
    "alert_frames": 10,
}

# Seating zones (normalised 0-1 coords: x1, y1, x2, y2). Leaving an assigned
# zone raises a "left seat" flag. Empty list disables zone monitoring.
ZONES = {
    "enabled": False,
    "boundaries": [
        # e.g. {"id": "A", "rect": [0.0, 0.0, 0.33, 1.0]},
    ],
}

# --------------------------------------------------------------------------- #
# 7. Behavioral scoring & severity levels
# --------------------------------------------------------------------------- #
BEHAVIOR = {
    # Relative weights of each signal in the unified risk score (sum ~= 1.0).
    "weights": {
        "head_pose": 0.30,   # sideways / downward looking frequency
        "object": 0.35,      # phone / prohibited item detection
        "movement": 0.15,    # excessive / erratic movement
        "proximity": 0.20,   # closeness to other students
    },
    # Per-event contribution before weighting (each clamped to 1.0 internally).
    "increments": {
        "looking_sideways": 0.08,
        "looking_down": 0.06,
        "object_detected": 0.20,
        "excessive_movement": 0.05,
        "proximity": 0.07,
        "left_zone": 0.10,
    },
    # Multiplicative decay applied each frame so risk cools down when behaviour
    # returns to normal (closer to 1.0 = slower cool-down).
    "decay": 0.985,
    # Severity bands: name -> (lower_inclusive, upper_exclusive, BGR colour, hex)
    "severity": [
        ("NORMAL", 0.00, 0.25, (0, 200, 0),   "#22c55e"),
        ("LOW",    0.25, 0.50, (0, 215, 255), "#eab308"),
        ("MEDIUM", 0.50, 0.75, (0, 140, 255), "#f97316"),
        ("HIGH",   0.75, 1.01, (0, 0, 255),   "#ef4444"),
    ],
}


def severity_for(score: float):
    """Return (name, bgr_color, hex_color) for a given 0-1 risk score."""
    for name, lo, hi, bgr, hexc in BEHAVIOR["severity"]:
        if lo <= score < hi:
            return name, bgr, hexc
    # Fallback to the top band
    name, _, _, bgr, hexc = BEHAVIOR["severity"][-1]
    return name, bgr, hexc


def resolve_device(requested: str = None) -> str:
    """Resolve 'auto' to the best available accelerator.

    Priority: CUDA (NVIDIA) -> MPS (Apple Silicon GPU) -> CPU.
    """
    req = requested or DEVICE
    if req != "auto":
        return req
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def sensitivity_profile(name: str = None) -> dict:
    key = name or DEFAULT_SENSITIVITY
    return SENSITIVITY_PROFILES.get(key) or SENSITIVITY_PROFILES.get(
        DEFAULT_SENSITIVITY, SENSITIVITY_PROFILES["strict"])


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
