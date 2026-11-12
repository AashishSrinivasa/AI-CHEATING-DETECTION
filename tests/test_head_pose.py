"""
Geometry tests for the head-pose solver (regression for the old ±180° pitch bug).

Synthetic faces are built in camera coordinates (x right, y down, z into the
scene) with the nose tip closest to the camera, rotated by a known physical
pose, projected with a pinhole camera and passed to the real solver.
"""

import math
import os
import sys
from collections import deque, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

import config
from detection.head_pose_detector import HeadPoseDetector, solve_head_pose

W, H = 640, 480
CAM = np.array([[W, 0, W / 2], [0, W, H / 2], [0, 0, 1]], dtype=np.float64)

# Six facial points in camera axes (mm): nose tip, chin, outer eye corners,
# mouth corners. Eyes/mouth are further from the camera than the nose tip.
FACE = np.array([
    (0.0, 0.0, 0.0),
    (0.0, 66.0, 14.0),
    (-45.0, -34.0, 28.0),
    (45.0, -34.0, 28.0),
    (-27.0, 30.0, 22.0),
    (27.0, 30.0, 22.0),
], dtype=np.float64)


def _project(down_deg=0.0, turn_deg=0.0):
    """Image points of FACE after tilting the head down and turning it sideways."""
    a, b = math.radians(down_deg), math.radians(turn_deg)
    # Positive rotation about x moves the nose tip (facing -z) DOWN in the image.
    rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    pts, _ = cv2.projectPoints(FACE, cv2.Rodrigues(ry @ rx)[0], np.array([0.0, 0.0, 650.0]),
                               CAM, np.zeros(4))
    return pts.reshape(-1, 2)


def _classifier():
    det = HeadPoseDetector.__new__(HeadPoseDetector)   # skip loading MediaPipe
    det.cfg = config.HEAD_POSE
    det._baselines = defaultdict(lambda: {"yaw": deque(maxlen=60), "pitch": deque(maxlen=60)})
    return det


def test_projection_sanity():
    # Looking down must move the nose tip down relative to the eyes.
    level, down = _project(0, 0), _project(25, 0)
    eye_y = lambda p: (p[2, 1] + p[3, 1]) / 2
    assert down[0, 1] - eye_y(down) > level[0, 1] - eye_y(level)


def test_frontal_face_has_near_zero_angles():
    pitch, yaw, roll = solve_head_pose(_project(0, 0), W, H)
    assert abs(pitch) < 3 and abs(yaw) < 3 and abs(roll) < 3, (pitch, yaw, roll)


def test_looking_down_gives_positive_pitch():
    for deg in (15, 25, 35):
        pitch, yaw, _ = solve_head_pose(_project(deg, 0), W, H)
        assert abs(pitch - deg) < 4, (deg, pitch)
        assert abs(yaw) < 4, (deg, yaw)


def test_looking_up_gives_negative_pitch():
    pitch, _, _ = solve_head_pose(_project(-20, 0), W, H)
    assert abs(pitch + 20) < 4, pitch


def test_turning_sideways_gives_yaw_magnitude():
    for deg in (20, 35, -35):
        pitch, yaw, _ = solve_head_pose(_project(0, deg), W, H)
        assert abs(abs(yaw) - abs(deg)) < 4, (deg, yaw)
        assert abs(pitch) < 4, (deg, pitch)


def test_classifier_flags_follow_the_pose():
    det = _classifier()
    p, y, _ = solve_head_pose(_project(0, 0), W, H)
    flags = det.classify(1, y, p)
    assert not flags["down"] and not flags["sideways"] and not flags["up"]
    p, y, _ = solve_head_pose(_project(30, 0), W, H)
    assert det.classify(2, y, p)["down"]
    p, y, _ = solve_head_pose(_project(0, 35), W, H)
    assert det.classify(3, y, p)["sideways"]


def test_baseline_absorbs_resting_posture_without_wraparound():
    # A student who normally sits slightly head-down must not be flagged,
    # and a real glance down from that posture must be.
    det = _classifier()
    for _ in range(20):
        p, y, _ = solve_head_pose(_project(10, 0), W, H)
        assert not det.classify(7, y, p)["down"]
    p, y, _ = solve_head_pose(_project(40, 0), W, H)
    assert det.classify(7, y, p)["down"]


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
