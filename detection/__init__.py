"""
Detection package for the AI-Powered Classroom Cheating Detection System.

Modules
-------
head_pose_detector : MediaPipe FaceMesh + solvePnP yaw/pitch/roll estimation.
object_detector    : YOLOv8 prohibited-item & person detection.
tracker            : DeepSORT persistent multi-student tracking.
behavior_analyzer  : Unified per-student behavioural risk scoring.

Note: submodules are imported directly (e.g.
``from detection.behavior_analyzer import BehaviorAnalyzer``) rather than
re-exported here, so that importing the lightweight ``behavior_analyzer`` does
not pull in the heavy CV dependencies (opencv, mediapipe, ultralytics) before
they are needed.
"""
