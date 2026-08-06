"""
jobs.py
=======
Thread-safe in-memory job registry with JSON persistence. Backs the video
upload-processing pipeline and the "Processing History" dashboard.

A job record looks like:
    {
      "job_id": "...", "filename": "...", "size": 1234,
      "status": "queued|processing|completed|failed",
      "progress": 0-100, "message": "...",
      "created_at": "...", "updated_at": "...",
      "output_video": "<id>.mp4", "report": "<id>.json",
      "total_alerts": int, "total_students": int,
      "duration_sec": float, "processing_time_sec": float,
      "severity_distribution": {...}
    }
"""

import json
import os
import threading
from datetime import datetime

import config


class JobManager:
    def __init__(self, index_path: str = None):
        self.index_path = index_path or config.HISTORY_INDEX
        self._lock = threading.Lock()
        self.jobs = {}
        self._load()

    def _load(self):
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    self.jobs = json.load(f)
            except Exception:
                self.jobs = {}

    def _persist(self):
        tmp = self.index_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.jobs, f, indent=2)
            os.replace(tmp, self.index_path)
        except Exception:
            pass

    def create(self, job_id, filename, size=None):
        with self._lock:
            self.jobs[job_id] = {
                "job_id": job_id,
                "filename": filename,
                "size": size,
                "status": "queued",
                "progress": 0,
                "message": "Queued for processing",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            self._persist()
            return dict(self.jobs[job_id])

    def update(self, job_id, **kwargs):
        with self._lock:
            if job_id not in self.jobs:
                return None
            self.jobs[job_id].update(kwargs)
            self.jobs[job_id]["updated_at"] = datetime.now().isoformat(timespec="seconds")
            self._persist()
            return dict(self.jobs[job_id])

    def get(self, job_id):
        with self._lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    def list(self):
        with self._lock:
            return sorted(
                (dict(j) for j in self.jobs.values()),
                key=lambda j: j.get("created_at", ""),
                reverse=True,
            )

    def delete(self, job_id):
        with self._lock:
            removed = self.jobs.pop(job_id, None)
            self._persist()
            return removed is not None
