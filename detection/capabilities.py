"""
capabilities.py
===============
Lightweight runtime probe of which detection engines are available, plus the
resolved compute device. Imports are checked lazily inside the function so this
module itself pulls in no heavy dependencies and is safe to import at app start.
"""

import importlib

import config


def _has(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def capabilities():
    caps = {
        "opencv": _has("cv2"),
        "yolov8": _has("ultralytics"),
        "yolo_world": False,
        "clip": _has("clip"),
        "mediapipe": _has("mediapipe"),
        "deepsort": _has("deep_sort_realtime"),
        "torch": _has("torch"),
    }
    if caps["yolov8"]:
        try:
            from ultralytics import YOLOWorld  # noqa: F401
            caps["yolo_world"] = True
        except Exception:
            caps["yolo_world"] = False
    device = config.resolve_device()
    cuda = False
    if caps["torch"]:
        try:
            import torch
            cuda = bool(torch.cuda.is_available())
        except Exception:
            cuda = False

    # Which MediaPipe API is present (legacy solutions vs new tasks)
    head_pose_engine = "OpenCV Haar (fallback)" if caps["opencv"] else "unavailable"
    if caps["mediapipe"]:
        if _has("mediapipe.solutions"):
            head_pose_engine = "MediaPipe FaceMesh"
        elif _has("mediapipe.tasks.python.vision"):
            head_pose_engine = "MediaPipe FaceLandmarker"

    if caps["yolov8"]:
        if caps["yolo_world"] and caps["clip"] and config.OBJECT_DETECTION.get("enable_world_detector"):
            object_engine = "YOLOv8 + YOLO-World"
            prohibited_engine = "YOLOv8 + YOLO-World"
        else:
            object_engine = "YOLOv8"
            prohibited_engine = "YOLOv8"
    elif caps["opencv"]:
        object_engine = "OpenCV HOG person fallback"
        prohibited_engine = "unavailable"
    else:
        object_engine = "unavailable"
        prohibited_engine = "unavailable"

    warnings = []
    if not caps["yolov8"]:
        warnings.append(
            "YOLOv8 is not installed; prohibited-object detection is disabled."
        )
    elif config.OBJECT_DETECTION.get("enable_world_detector") and not (caps["yolo_world"] and caps["clip"]):
        warnings.append(
            "YOLO-World is not fully available; headphone/earbud detection may be weaker."
        )
    if not caps["mediapipe"]:
        warnings.append(
            "MediaPipe is not installed; head pose uses the OpenCV Haar fallback."
        )
    if not caps["deepsort"]:
        warnings.append(
            "DeepSORT is not installed; tracking uses the built-in IoU fallback."
        )

    # Engines as they will actually run, given graceful fallbacks.
    engines = {
        "object_detection": object_engine,
        "prohibited_item_detection": prohibited_engine,
        "head_pose": head_pose_engine,
        "tracking": "DeepSORT" if caps["deepsort"] else "IoU tracker (fallback)",
    }

    return {
        "device": device,
        "cuda_available": cuda,
        "libraries": caps,
        "engines": engines,
        "ready": caps["opencv"],
        "full_detection_ready": caps["opencv"] and caps["yolov8"] and caps["mediapipe"],
        "sensitivity_profiles": {
            key: val["label"] for key, val in config.SENSITIVITY_PROFILES.items()
        },
        "default_sensitivity": config.DEFAULT_SENSITIVITY,
        "malpractice_items": sorted(config.OBJECT_DETECTION["prohibited_classes"]),
        "warnings": warnings,
    }
