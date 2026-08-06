"""
Unit tests for the behavioural scoring engine and config helpers.

These tests depend only on the standard library + `config` (no opencv /
mediapipe / torch), so they run fast anywhere:

    python -m pytest tests/            # if pytest is installed
    python tests/test_behavior.py      # plain runner
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from detection.behavior_analyzer import BehaviorAnalyzer

FRAME = (720, 1280, 3)


def test_severity_bands():
    assert config.severity_for(0.10)[0] == "NORMAL"
    assert config.severity_for(0.30)[0] == "LOW"
    assert config.severity_for(0.60)[0] == "MEDIUM"
    assert config.severity_for(0.90)[0] == "HIGH"


def test_allowed_file():
    assert config.allowed_file("exam.mp4")
    assert config.allowed_file("clip.MKV")
    assert not config.allowed_file("notes.txt")


def test_flagged_student_outscores_calm_student():
    an = BehaviorAnalyzer(fps=30.0, head_pose=None)
    tracked = [
        {"id": 1, "bbox": [100, 100, 220, 360], "center": (160, 230)},
        {"id": 2, "bbox": [800, 100, 920, 360], "center": (860, 230)},
    ]
    faces = [{"bbox": [120, 120, 200, 200], "center": (160, 160),
              "yaw": 40.0, "pitch": 3.0, "roll": 0.0}]
    objects = [{"bbox": [150, 300, 210, 360], "center": (180, 330),
                "conf": 0.8, "label": "cell phone"}]
    snap = None
    for f in range(20):
        snap = an.update(f, f / 30.0, tracked, faces, objects, FRAME)
    s1 = next(s for s in snap["students"] if s["id"] == 1)
    s2 = next(s for s in snap["students"] if s["id"] == 2)
    assert s1["risk"] > s2["risk"]
    assert s2["risk"] == 0.0
    types = {a["type"] for a in an.alerts}
    assert "looking_sideways" in types
    assert "object" in types


def test_risk_decays_when_behaviour_stops():
    an = BehaviorAnalyzer(fps=30.0, head_pose=None)
    tracked = [{"id": 1, "bbox": [100, 100, 220, 360], "center": (160, 230)}]
    faces = [{"bbox": [120, 120, 200, 200], "center": (160, 160),
              "yaw": 45.0, "pitch": 0.0, "roll": 0.0}]
    for f in range(20):
        snap = an.update(f, f / 30.0, tracked, faces, [], FRAME)
    peak = snap["students"][0]["risk"]
    for f in range(20, 80):
        snap = an.update(f, f / 30.0, tracked, [], [], FRAME)
    assert snap["students"][0]["risk"] < peak


def test_report_summary_shape():
    an = BehaviorAnalyzer(fps=30.0, head_pose=None)
    tracked = [{"id": 1, "bbox": [100, 100, 220, 360], "center": (160, 230)}]
    for f in range(5):
        an.update(f, f / 30.0, tracked, [], [], FRAME)
    rep = an.summary()
    assert rep["total_students"] == 1
    assert set(rep["severity_distribution"]) == {"NORMAL", "LOW", "MEDIUM", "HIGH"}
    assert "students" in rep and "alerts" in rep


def test_signal_renormalisation_keeps_score_bounded():
    an = BehaviorAnalyzer(fps=30.0, head_pose=None)
    an.set_active_signals(head_pose=False)
    tracked = [{"id": 1, "bbox": [100, 100, 220, 360], "center": (160, 230)}]
    objects = [{"bbox": [150, 300, 210, 360], "center": (180, 330),
                "conf": 0.9, "label": "cell phone"}]
    snap = None
    for f in range(40):
        snap = an.update(f, f / 30.0, tracked, [], objects, FRAME)
    risk = snap["students"][0]["risk"]
    assert 0.0 <= risk <= 1.0
    assert risk > 0.0  # object signal alone still raises risk


def test_headphones_raise_fast_malpractice_alert():
    an = BehaviorAnalyzer(fps=30.0, head_pose=None, sensitivity="strict")
    tracked = [{"id": 1, "bbox": [100, 80, 260, 380], "center": (180, 230)}]
    objects = [{"bbox": [120, 90, 235, 170], "center": (178, 130),
                "conf": 0.72, "label": "headphones", "source": "yolo_world"}]
    snap = None
    for f in range(4):
        snap = an.update(f, f / 30.0, tracked, [], objects, FRAME)
    assert snap["students"][0]["object"] == "headphones"
    assert snap["students"][0]["risk"] >= 0.25
    assert any(a["type"] == "object" and "Audio device" in a["message"]
               for a in an.alerts)


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
