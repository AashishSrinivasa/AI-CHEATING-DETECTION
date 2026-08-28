"""
app.py
======
Flask web application for the AI-Powered Classroom Cheating Detection System.

Provides two operation modes and a reporting dashboard:

    Live Camera Detection   /live      (MJPEG stream + live risk stats)
    Video Upload Processing  /upload   (background analysis + JSON report)
    Processing History       /history
    Incident Report          /report/<job_id>

Heavy CV dependencies (ultralytics, mediapipe, deep-sort-realtime, opencv) are
imported lazily inside the live/upload handlers so the web server always boots
and the history/report pages remain available even before models are installed.
"""

import os
import threading
import uuid

from flask import (Flask, Response, abort, jsonify, redirect,
                   render_template, request, send_from_directory, url_for)
from werkzeug.utils import secure_filename

import config
from core.jobs import JobManager
from detection.capabilities import capabilities

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

jobs = JobManager()

# Single shared live-camera worker (one physical camera at a time).
live_processor = None
live_lock = threading.Lock()


def _valid_sensitivity(value):
    if value in config.SENSITIVITY_PROFILES:
        return value
    if config.DEFAULT_SENSITIVITY in config.SENSITIVITY_PROFILES:
        return config.DEFAULT_SENSITIVITY
    return "strict"


# --------------------------------------------------------------------------- #
# Template globals
# --------------------------------------------------------------------------- #
@app.context_processor
def inject_globals():
    return {
        "severity_bands": [
            {"name": n, "lo": lo, "hi": hi, "hex": hexc}
            for (n, lo, hi, _bgr, hexc) in config.BEHAVIOR["severity"]
        ],
        "max_upload_mb": config.MAX_CONTENT_LENGTH // (1024 * 1024),
        "allowed_ext": sorted(config.ALLOWED_EXTENSIONS),
        "sensitivity_profiles": config.SENSITIVITY_PROFILES,
        "default_sensitivity": _valid_sensitivity(config.DEFAULT_SENSITIVITY),
        "is_vercel": config.IS_VERCEL,
    }


def _dashboard_stats(job_list):
    completed = [j for j in job_list if j.get("status") == "completed"]
    return {
        "total_jobs": len(job_list),
        "completed": len(completed),
        "processing": sum(1 for j in job_list
                          if j.get("status") in ("processing", "queued")),
        "total_alerts": sum(j.get("total_alerts", 0) or 0 for j in completed),
        "total_students": sum(j.get("total_students", 0) or 0 for j in completed),
    }


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    job_list = jobs.list()
    return render_template("index.html", jobs=job_list[:5],
                           stats=_dashboard_stats(job_list))


@app.route("/live")
def live_page():
    return render_template("live.html")


@app.route("/upload")
def upload_page():
    return render_template("upload.html")


@app.route("/history")
def history_page():
    return render_template("history.html", jobs=jobs.list())


@app.route("/report/<job_id>")
def report_page(job_id):
    job = jobs.get(job_id)
    if not job:
        abort(404)
    return render_template("report.html", job=job)


# --------------------------------------------------------------------------- #
# Live camera API
# --------------------------------------------------------------------------- #
@app.route("/api/live/start", methods=["POST"])
def api_live_start():
    global live_processor
    try:
        from core.live_processor import LiveProcessor
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=f"Missing dependencies: {exc}"), 500

    source = 0
    sensitivity = config.DEFAULT_SENSITIVITY
    if request.is_json and request.json:
        source = request.json.get("source", 0)
        sensitivity = request.json.get("sensitivity", config.DEFAULT_SENSITIVITY)
    sensitivity = _valid_sensitivity(sensitivity)
    try:
        source = int(source)
    except (TypeError, ValueError):
        pass  # allow a string path / URL source

    with live_lock:
        if live_processor is None or not live_processor.is_running():
            live_processor = LiveProcessor(source=source, sensitivity=sensitivity)
            live_processor.start()
    return jsonify(running=True, source=source, sensitivity=sensitivity)


@app.route("/api/live/stop", methods=["POST"])
def api_live_stop():
    global live_processor
    with live_lock:
        if live_processor is not None:
            live_processor.stop()
    return jsonify(running=False)


@app.route("/api/live/stats")
def api_live_stats():
    with live_lock:
        lp = live_processor
    if lp is None or not lp.is_running():
        st = {"running": False, "students": [], "alerts": [],
              "total_alerts": 0, "fps": 0.0}
        if lp is not None:
            err = lp.stats().get("error")
            if err:
                st["error"] = err
        return jsonify(st)
    return jsonify(lp.stats())


@app.route("/video_feed")
def video_feed():
    with live_lock:
        lp = live_processor
    if lp is None or not lp.is_running():
        return Response(b"", mimetype="text/plain")
    return Response(lp.frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# --------------------------------------------------------------------------- #
# Video upload & processing API
# --------------------------------------------------------------------------- #
@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "video" not in request.files:
        return jsonify(error="No file part in request."), 400
    file = request.files["video"]
    if not file or file.filename == "":
        return jsonify(error="No file selected."), 400
    if not config.allowed_file(file.filename):
        return jsonify(error="Unsupported format. Allowed: "
                             + ", ".join(sorted(config.ALLOWED_EXTENSIONS))), 400

    job_id = uuid.uuid4().hex[:12]
    ext = file.filename.rsplit(".", 1)[1].lower()
    original = secure_filename(file.filename)
    saved_path = os.path.join(config.UPLOAD_FOLDER, f"{job_id}.{ext}")
    file.save(saved_path)
    size = os.path.getsize(saved_path)

    jobs.create(job_id, original, size)
    sensitivity = request.form.get("sensitivity", config.DEFAULT_SENSITIVITY)
    sensitivity = _valid_sensitivity(sensitivity)
    jobs.update(job_id, sensitivity=sensitivity,
                sensitivity_label=config.sensitivity_profile(sensitivity)["label"])

    try:
        from core.video_processor import VideoProcessor
    except Exception as exc:  # noqa: BLE001
        jobs.update(job_id, status="failed",
                    message=f"Missing dependencies: {exc}")
        return jsonify(error=f"Missing dependencies: {exc}",
                       job_id=job_id), 500

    worker = VideoProcessor(job_id, saved_path, jobs, sensitivity=sensitivity)
    threading.Thread(target=worker.run, daemon=True).start()
    return jsonify(job_id=job_id, status="queued", filename=original)


@app.route("/api/status/<job_id>")
def api_status(job_id):
    job = jobs.get(job_id)
    if not job:
        abort(404)
    return jsonify(job)


@app.route("/api/history")
def api_history():
    return jsonify(jobs.list())


@app.route("/api/system")
def api_system():
    """Active detection engines + compute device, for the status indicator."""
    return jsonify(capabilities())


@app.route("/api/health")
def api_health():
    """Small health probe for Docker, load balancers and uptime checks."""
    caps = capabilities()
    status = "ok" if caps["ready"] else "degraded"
    return jsonify(status=status, ready=caps["ready"], system=caps)


@app.route("/api/report/<job_id>")
def api_report(job_id):
    job = jobs.get(job_id)
    if not job or not job.get("report"):
        abort(404)
    path = os.path.join(config.REPORT_FOLDER, job["report"])
    if not os.path.exists(path):
        abort(404)
    with open(path, "r", encoding="utf-8") as f:
        return Response(f.read(), mimetype="application/json")


@app.route("/api/job/<job_id>", methods=["DELETE"])
def api_delete_job(job_id):
    job = jobs.get(job_id)
    if not job:
        abort(404)
    # remove generated artefacts
    if job.get("output_video"):
        p = os.path.join(config.OUTPUT_FOLDER, job["output_video"])
        if os.path.exists(p):
            os.remove(p)
    if job.get("report"):
        p = os.path.join(config.REPORT_FOLDER, job["report"])
        if os.path.exists(p):
            os.remove(p)
    # remove any matching upload
    for ext in config.ALLOWED_EXTENSIONS:
        up = os.path.join(config.UPLOAD_FOLDER, f"{job_id}.{ext}")
        if os.path.exists(up):
            os.remove(up)
    jobs.delete(job_id)
    return jsonify(deleted=True)


# --------------------------------------------------------------------------- #
# File serving
# --------------------------------------------------------------------------- #
@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(config.OUTPUT_FOLDER, filename)


@app.route("/download/video/<job_id>")
def download_video(job_id):
    job = jobs.get(job_id)
    if not job or not job.get("output_video"):
        abort(404)
    return send_from_directory(config.OUTPUT_FOLDER, job["output_video"],
                               as_attachment=True)


@app.route("/download/report/<job_id>")
def download_report(job_id):
    job = jobs.get(job_id)
    if not job or not job.get("report"):
        abort(404)
    return send_from_directory(config.REPORT_FOLDER, job["report"],
                               as_attachment=True,
                               download_name=f"report_{job_id}.json")


# --------------------------------------------------------------------------- #
# Error handlers
# --------------------------------------------------------------------------- #
@app.errorhandler(413)
def too_large(_e):
    return jsonify(error=f"File too large. Maximum is "
                         f"{config.MAX_CONTENT_LENGTH // (1024 * 1024)} MB."), 413


@app.errorhandler(404)
def not_found(_e):
    if request.path.startswith("/api/"):
        return jsonify(error="Not found"), 404
    return render_template("404.html"), 404


if __name__ == "__main__":
    # Port is configurable; on macOS, port 5000 is often taken by the AirPlay
    # Receiver (Control Center) — set PORT or disable it in System Settings.
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "1") not in ("0", "false", "False")
    print("=" * 64)
    print(" AI-Powered Classroom Cheating Detection System")
    print(" Device:", config.resolve_device())
    print(f" Open http://127.0.0.1:{port} in your browser")
    print("=" * 64)
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
