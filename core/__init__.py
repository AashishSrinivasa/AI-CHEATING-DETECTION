"""
Core orchestration package: job management plus the live / video processing
pipelines.

Only the dependency-light ``JobManager`` is re-exported here. The processors
(``VideoProcessor``, ``LiveProcessor``) and the ``annotator`` pull in heavy CV
dependencies, so import them directly from their submodules where needed:

    from core.video_processor import VideoProcessor
    from core.live_processor import LiveProcessor

This keeps ``from core.jobs import JobManager`` (used at app start-up) free of
opencv / ultralytics / mediapipe imports.
"""

from .jobs import JobManager

__all__ = ["JobManager"]
