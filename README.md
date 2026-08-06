# 🎓 AI-Powered Classroom Cheating Detection System

**Real-Time Monitoring & Behavioral Analysis**

An intelligent examination-monitoring solution that leverages computer vision, deep
learning and behavioural analytics to detect potential cheating behaviours in
classroom environments. It combines **YOLOv8 object detection**, **head-pose
estimation**, **movement tracking** and **student-interaction analysis** to provide
comprehensive real-time surveillance during examinations — all through a modern,
web-based dashboard.

> **Tech stack:** Python · Flask · OpenCV · PyTorch · YOLOv8 · MediaPipe · Deep SORT ·
> NumPy · HTML5 · CSS3 · JavaScript

---

## ✨ Core Detection Capabilities

| Capability | Engine | What it does |
|---|---|---|
| **Head-Pose Detection** | MediaPipe (FaceMesh *or* FaceLandmarker) + `solvePnP`, with OpenCV Haar fallback | Estimates yaw/pitch/roll; flags sideways looking (copying) and downward gaze (notes/phone). Dynamic per-student baselines adapt to seating/camera angle and suppress false positives from momentary movements. |
| **Object Detection** | YOLOv8 + YOLO-World | Detects prohibited items — smartphones, books, papers, calculators, smart watches, headphones and earbuds — with per-class confidence thresholds tuned for classrooms. Alerts only after sustained presence. |
| **Movement & Zone Tracking** | DeepSORT (with built-in IoU-tracker fallback) | Persistent student IDs across the session; velocity/acceleration analysis; inter-student distance for proximity/interaction flags; optional seating-zone boundaries. |
| **Behavioural Scoring** | Weighted fusion | A continuously updated **0.0–1.0 risk score** per student combining head-pose, object, movement and proximity signals, mapped to **Normal / Low / Medium / High** severity. |

---

## 🖥️ Dual Operation Modes

### 1. Live Camera Detection
Real-time webcam monitoring at **20–30 FPS on CPU** (60+ FPS with CUDA). The live
dashboard shows the annotated stream with bounding boxes, risk scores and instant
alerts. Start/stop with a single click; view per-student behavioural stats
simultaneously.

### 2. Video Upload & Processing
Drag-and-drop pre-recorded exams (**up to 500 MB**, MP4/AVI/MOV/MKV). Background
processing analyses every frame, produces an **annotated output video** and a
**frame-accurate JSON report**. An interactive timeline lets you click an incident
to jump straight to that moment in the video.

---

## 🏗️ Architecture

```
┌────────────────────────── Flask web app (app.py) ──────────────────────────┐
│  Dashboard · Live · Upload · History · Report  +  JSON / streaming APIs     │
└───────────────┬───────────────────────────────────────────┬────────────────┘
                │                                             │
        core/live_processor.py                       core/video_processor.py
                │                                             │
                └──────────────► detection/ ◄─────────────────┘
                                    │
   head_pose_detector ─┐            │
   object_detector ────┼──► behavior_analyzer ──► risk score (0–1) + alerts
   tracker (DeepSORT) ─┘            │
                                    └──► core/annotator.py (overlays)
```

### Project structure
```
.
├── app.py                      # Flask routes, APIs, streaming
├── config.py                   # ALL tunable parameters (see below)
├── requirements.txt
├── detection/
│   ├── head_pose_detector.py   # MediaPipe + solvePnP yaw/pitch
│   ├── object_detector.py      # YOLOv8 person + prohibited items
│   ├── tracker.py              # DeepSORT persistent IDs
│   └── behavior_analyzer.py    # unified risk scoring + alerts
├── core/
│   ├── annotator.py            # shared OpenCV overlays
│   ├── jobs.py                 # job registry + history persistence
│   ├── video_processor.py      # background video analysis worker
│   └── live_processor.py       # webcam worker + MJPEG stream
├── templates/                  # Jinja2 pages
├── static/                     # CSS + JS
├── uploads/  outputs/  reports/  models/  data/
```

---

## 🚀 Installation & Deployment

**Requirements:** Python 3.8+ (3.9–3.11 recommended for MediaPipe wheels).

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run (model weights download automatically on first run)
python app.py
```

Open **http://127.0.0.1:5000**.

If you start the app before installing the full ML stack, it still runs in a
degraded but usable mode with OpenCV person/head-pose fallbacks. Install the full
`requirements.txt` stack to enable YOLOv8 prohibited-item detection, MediaPipe
landmarks, YOLO-World headphone/earbud detection and DeepSORT re-identification.

> **macOS note:** port 5000 is often occupied by the AirPlay Receiver (Control
> Center). Either disable it in *System Settings → General → AirDrop &amp; Handoff*,
> or run on another port: `PORT=5055 python app.py`.
>
> **Live camera:** the first time you start live monitoring, macOS will ask for
> camera permission for your terminal/IDE — allow it. If the camera can't be
> opened, the UI shows a clear message instead of crashing.

### GPU acceleration
Set the device explicitly (auto-detected by default):
```bash
DEVICE=cuda python app.py        # requires a CUDA-enabled PyTorch build
DEVICE=mps python app.py         # Apple Silicon GPU
```

### Accuracy tuning for presentation
The live and upload screens include three profiles:

- **Strict Exam**: default. Best for demos and high-sensitivity malpractice
  detection. Headphones/earbuds, phones and smart watches alert faster.
- **Balanced**: normal classroom monitoring with fewer false positives.
- **Conservative**: slower alerting when the room has many confusing objects.

For live demos, use good lighting, keep the camera at face/upper-body height,
and keep the student within the frame. Headphones and earbuds are detected by
YOLO-World, while phones/books/papers are detected by both YOLOv8 and YOLO-World.
If live FPS drops, increase `LIVE_WORLD_FRAME_INTERVAL` to `3`; if headphone
detection needs to be more responsive, set it back to `1`.

### Production
```bash
gunicorn -w 1 --threads 8 -b 0.0.0.0:8000 app:app
```
> Use a single worker with multiple threads (`-w 1 --threads N`): the live camera
> and in-memory job registry are process-local. Scale video throughput with
> additional concurrent upload jobs rather than workers.

Optional: install **ffmpeg** so processed videos are re-encoded to H.264 for
seamless in-browser playback.

### Docker
```bash
cp .env.example .env
docker compose up --build
```

Open **http://127.0.0.1:8000**. Uploads, reports, outputs, history and model
weights are mounted as local volumes so data survives container restarts.

---

## ⚙️ Configurable Detection Parameters (`config.py`)

Everything is tunable without touching detection code:

- **Head pose** — `yaw_threshold`, `pitch_down_threshold`, `alert_frames`,
  `dynamic_adjustment`.
- **Objects** — global + per-class `confidence_threshold`s, `alert_frames`,
  prohibited class list.
- **Movement & zones** — `velocity_threshold`, `acceleration_threshold`,
  `proximity_threshold`, seating-zone boundaries.
- **Scoring** — component `weights`, per-event `increments`, `decay`, and the
  four severity bands with colours.

Changes apply on the next live session / processing job.

---

## 📊 Output & Reporting

Processed videos include colour-coded student boxes, risk bars, head-pose
readouts, prohibited-object boxes and red proximity-interaction lines, plus an
on-frame HUD.

The **JSON report** (`reports/<job_id>.json`) contains:
- Video metadata (resolution, fps, frames, duration, processing time, device)
- `total_students`, `total_alerts`
- `severity_distribution` and `alert_type_distribution`
- Per-student summaries (peak/final risk, severity, behaviour event counts)
- A chronological `alerts` timeline with **frame numbers + timestamps**

---

## 🛡️ Robustness & Compatibility

Engineered to keep running rather than crash:

- **Engine auto-selection & fallbacks.** Head pose uses MediaPipe's legacy
  *FaceMesh* API, or the newer *FaceLandmarker* Tasks API (model auto-downloaded
  to `models/`), or an OpenCV Haar-cascade fallback — whichever is available.
  Object detection uses YOLOv8 + YOLO-World when available, or an OpenCV HOG
  people-only fallback so live/upload flows do not hard-fail on lightweight installs.
  Tracking uses DeepSORT, or a built-in IoU tracker if DeepSORT can't initialise.
  The active engines are shown live in the top bar and via `GET /api/system`.
- **Per-frame isolation.** A single undecodable/odd frame is skipped, never
  aborting a live session or a video job.
- **Score renormalisation.** If a signal (e.g. head pose) is unavailable, the
  0–1 risk score renormalises over the remaining signals so it stays meaningful.
- **Low-latency live capture.** A dedicated capture thread always serves the
  freshest frame (drops stale ones), so latency never accumulates in real time.
- **Browser-playable output.** Annotated videos are written as H.264 (`avc1`)
  so they play inline in the report; falls back to `mp4v` (+ optional ffmpeg
  transcode) if H.264 is unavailable.
- **Graceful start-up.** The web server boots and the dashboard/history/report
  pages work even before the CV models are installed.

## 🔒 Privacy & Ethical Considerations

Designed for **legitimate educational monitoring with proper student notification**.
Processed videos and detection data are stored locally under access control.
Institutions should review local privacy regulations and establish clear policies
on surveillance scope, data retention and student rights. The technology is meant
to **augment, not replace**, human judgement in academic-integrity decisions.

---

## 📈 Performance

| Scenario | CPU | CUDA GPU |
|---|---|---|
| Live detection | 20–30 FPS | 60+ FPS |
| 10-min 1080p video | ~5–10 min | ~2–3 min |

Minimum recommended source: **720p @ 30 FPS**. Supports multiple concurrent upload
jobs and scales from small groups to lecture halls with appropriate camera placement.
