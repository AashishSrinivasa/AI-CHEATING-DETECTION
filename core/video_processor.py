"""
video_processor.py
==================
Background worker that analyses an uploaded examination video end-to-end:

    read frames -> YOLOv8 detect -> track -> head-pose -> score
                -> annotate -> write output video -> emit JSON report

Robustness:
    * Every per-frame step is wrapped — a single undecodable/odd frame is
      skipped, never aborting the whole job.
    * Frames are downscaled to config.VIDEO_MAX_WIDTH for speed; the annotated
      output is written at that (consistent) resolution.
    * Detection engines degrade gracefully (see detection/*), and which engines
      ran is recorded in the report.

Produces:
    outputs/<job_id>.mp4   annotated video
    reports/<job_id>.json  frame-accurate incident report
"""

import json
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime

import cv2

import config
from detection.head_pose_detector import HeadPoseDetector
from detection.object_detector import ObjectDetector
from detection.tracker import StudentTracker
from detection.behavior_analyzer import BehaviorAnalyzer
from .annotator import annotate_frame, resize_max_width


class VideoProcessor:
    def __init__(self, job_id, input_path, job_manager, device=None, sensitivity=None):
        self.job_id = job_id
        self.input_path = input_path
        self.jm = job_manager
        self.device = config.resolve_device(device)
        self.sensitivity = sensitivity or config.DEFAULT_SENSITIVITY
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            self._process()
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.jm.update(self.job_id, status="failed",
                           message=f"Processing failed: {exc}")

    # ------------------------------------------------------------------ #
    def _process(self):
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise RuntimeError("Unable to open the uploaded video file.")

        fps = cap.get(cv2.CAP_PROP_FPS) or config.DEFAULT_FPS
        if not fps or fps <= 0:
            fps = config.DEFAULT_FPS
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        # Detectors (per job so state is isolated)
        head_pose = HeadPoseDetector()
        detector = ObjectDetector(
            device=self.device,
            sensitivity=self.sensitivity,
            world_frame_interval=config.VIDEO_WORLD_FRAME_INTERVAL,
        )
        tracker = StudentTracker(device=self.device)
        analyzer = BehaviorAnalyzer(fps=fps, head_pose=head_pose, sensitivity=self.sensitivity)
        analyzer.set_active_signals(
            head_pose=head_pose.available,
            object=detector.supports_prohibited_objects,
        )

        engines = {
            "object_detection": detector.engine_label,
            "prohibited_item_detection": (
                detector.engine_label if detector.supports_prohibited_objects else "unavailable"
            ),
            "head_pose": head_pose.engine_label,
            "tracking": "DeepSORT" if tracker.backend == "deepsort" else "IoU tracker (fallback)",
            "sensitivity": config.sensitivity_profile(self.sensitivity)["label"],
        }

        out_path = os.path.join(config.OUTPUT_FOLDER, f"{self.job_id}.mp4")
        writer = None
        out_w = out_h = None

        self.jm.update(self.job_id, status="processing", progress=0,
                       message="Analysing video...", device=self.device)

        start = time.time()
        frame_idx = 0
        last_result = {"students": [], "proximity_pairs": []}
        last_objects = []

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                break

            frame = resize_max_width(frame, config.VIDEO_MAX_WIDTH)
            if writer is None:
                out_h, out_w = frame.shape[:2]
                writer = self._open_writer(out_path, fps, (out_w, out_h))
            ts = frame_idx / fps

            try:
                persons, objs = detector.detect(frame)
                tracked = tracker.update(persons, frame)
                faces = head_pose.process(frame)
                last_result = analyzer.update(frame_idx, ts, tracked, faces,
                                              objs, frame.shape)
                last_objects = objs
            except Exception:
                # skip analysis for this frame but still write it out
                pass

            meta = {"frame": frame_idx, "time_str": analyzer._fmt_time(ts),
                    "total_alerts": len(analyzer.alerts), "fps": round(fps, 1)}
            try:
                annotate_frame(frame, last_result, last_objects, meta)
            except Exception:
                pass
            writer.write(frame)

            frame_idx += 1
            if frame_idx % 10 == 0:
                if total:
                    pct = min(99, int(frame_idx / total * 100))
                    eta = self._eta(start, frame_idx, total)
                    msg = f"Processing frame {frame_idx}/{total}  (ETA {eta})"
                else:
                    pct = min(99, frame_idx // 30)
                    msg = f"Processing frame {frame_idx}"
                self.jm.update(self.job_id, progress=pct, message=msg)

        cap.release()
        if writer is not None:
            writer.release()
        head_pose.close()

        if self._stop.is_set():
            self.jm.update(self.job_id, status="failed", message="Cancelled")
            return
        if frame_idx == 0:
            raise RuntimeError("No frames could be read from the video.")

        self.jm.update(self.job_id, progress=99, message="Finalising video...")
        self._maybe_transcode(out_path)

        proc_time = round(time.time() - start, 2)
        summary = analyzer.summary()
        report = {
            "job_id": self.job_id,
            "source_file": os.path.basename(self.input_path),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "device": self.device,
            "engines": engines,
            "video": {
                "width": out_w, "height": out_h, "fps": round(fps, 2),
                "total_frames": frame_idx,
                "duration_sec": round(frame_idx / fps, 2),
            },
            "processing_time_sec": proc_time,
        }
        report.update(summary)

        report_path = os.path.join(config.REPORT_FOLDER, f"{self.job_id}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        self.jm.update(
            self.job_id, status="completed", progress=100,
            message="Analysis complete",
            output_video=os.path.basename(out_path),
            report=os.path.basename(report_path),
            total_alerts=report["total_alerts"],
            total_students=report["total_students"],
            duration_sec=report["video"]["duration_sec"],
            processing_time_sec=proc_time,
            severity_distribution=report["severity_distribution"],
            engines=engines,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _open_writer(path, fps, size):
        """Open a VideoWriter, preferring H.264 (browser-playable) codecs."""
        for tag in ("avc1", "H264", "mp4v"):
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*tag), fps, size)
            if writer.isOpened():
                return writer
            writer.release()
        # Last resort — always returns a writer object.
        return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    @staticmethod
    def _eta(start, done, total):
        elapsed = time.time() - start
        if done <= 0:
            return "--:--"
        remaining = elapsed / done * (total - done)
        return f"{int(remaining // 60):02d}:{int(remaining % 60):02d}"

    @staticmethod
    def _maybe_transcode(path):
        """Re-encode mp4v -> H.264 (yuv420p, faststart) if ffmpeg is present."""
        if not shutil.which("ffmpeg"):
            return path
        tmp = path.replace(".mp4", "_h264.mp4")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 "-loglevel", "error", tmp],
                check=True, capture_output=True,
            )
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
        return path
