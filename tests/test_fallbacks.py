"""
Tests for dependency fallbacks that keep the app usable on lightweight installs.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from detection import object_detector


def test_object_detector_falls_back_without_yolo():
    old_yolo = object_detector._YOLO_AVAILABLE
    old_cls = object_detector.YOLO
    try:
        object_detector._YOLO_AVAILABLE = False
        object_detector.YOLO = None
        detector = object_detector.ObjectDetector()
        assert detector.backend == "opencv_hog"
        assert detector.supports_prohibited_objects is False
        frame = np.zeros((160, 160, 3), dtype=np.uint8)
        persons, objects = detector.detect(frame)
        assert isinstance(persons, list)
        assert objects == []
    finally:
        object_detector._YOLO_AVAILABLE = old_yolo
        object_detector.YOLO = old_cls


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
