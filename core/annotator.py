"""
annotator.py
============
Shared OpenCV drawing routines used by both the live and video processors so
that the on-screen overlays are identical across operation modes:

    * Colour-coded student boxes (severity colour) with ID + risk score
    * A thin risk bar beneath each student
    * Orange / red boxes for prohibited objects with class + confidence
    * Red interaction lines connecting students detected in proximity
    * A semi-transparent HUD panel with concise session status
"""

import cv2

import config

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_TEXT_DARK = (20, 24, 30)
_AUDIO_LABELS = {"headphones", "headset", "earphones", "earbuds", "wireless earbuds"}
_RESERVE_PAD = 6


def resize_max_width(frame, max_w):
    """Downscale a frame so its width <= max_w (keeps aspect). No-op if smaller."""
    if not max_w or frame.shape[1] <= max_w:
        return frame
    scale = max_w / float(frame.shape[1])
    return cv2.resize(frame, (int(max_w), int(round(frame.shape[0] * scale))),
                      interpolation=cv2.INTER_AREA)


def _label_geometry(img, text, org, scale=0.5, thickness=1):
    (tw, th), bl = cv2.getTextSize(text, _FONT, scale, thickness)
    x, y = org
    y = max(y, th + bl + 4)
    y = min(y, img.shape[0] - 4)
    x = max(4, min(x, img.shape[1] - tw - 12))
    rect = (x, max(0, y - th - bl - 6), x + tw + 10, min(img.shape[0], y + 2))
    text_org = (x + 3, y - bl - 2)
    return rect, text_org


def _label(img, text, org, color, scale=0.5, thickness=1, text_color=_WHITE):
    rect, text_org = _label_geometry(img, text, org, scale, thickness)
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), color, -1)
    cv2.putText(img, text, text_org, _FONT, scale,
                text_color, thickness, cv2.LINE_AA)
    return rect


def _overlaps(a, b, pad=_RESERVE_PAD):
    return not (
        a[2] + pad <= b[0] or
        a[0] - pad >= b[2] or
        a[3] + pad <= b[1] or
        a[1] - pad >= b[3]
    )


def _label_avoiding(img, text, x, candidates, color, reserved,
                    scale=0.5, thickness=1, text_color=_WHITE):
    """Draw a filled label at the first candidate baseline that does not collide."""
    fallback = None
    for y in candidates:
        rect, _ = _label_geometry(img, text, (x, int(y)), scale, thickness)
        fallback = (rect, int(y))
        if not any(_overlaps(rect, taken) for taken in reserved):
            drawn = _label(img, text, (x, int(y)), color, scale, thickness, text_color)
            reserved.append(drawn)
            return drawn
    rect, y = fallback
    drawn = _label(img, text, (x, y), color, scale, thickness, text_color)
    reserved.append(drawn)
    return drawn


def _above_or_inside_candidates(y1, y2):
    if y1 < 130:
        return [y1 + 58, y1 + 86, y2 + 24, y2 + 50, y1 - 6]
    return [y1 - 6, y1 + 30, y2 + 24, y2 + 50]


def _object_label_candidates(y1, y2):
    if y1 < 150:
        return [y2 + 24, y2 + 50, y1 + 58, y1 - 4]
    return [y1 - 4, y2 + 24, y2 + 50, y1 + 32]


def _severity_rank(name):
    return {"NORMAL": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}.get(name, 0)


def _pct(score):
    return int(round(max(0.0, min(1.0, float(score or 0.0))) * 100))


def _banner(frame, text, color, y):
    (tw, th), bl = cv2.getTextSize(text, _FONT, 0.62, 2)
    x = 12
    x2 = min(frame.shape[1] - 12, x + tw + 24)
    y2 = y + th + bl + 16
    cv2.rectangle(frame, (x, y), (x2, y2), color, -1)
    cv2.putText(frame, text, (x + 12, y + th + 8), _FONT, 0.62,
                _WHITE, 2, cv2.LINE_AA)
    return y + th + bl + 22, (x, y, x2, y2)


def _top_hud(frame, students, meta):
    high = sum(1 for s in students if s["severity"] == "HIGH")
    med = sum(1 for s in students if s["severity"] == "MEDIUM")
    alerts = meta.get("total_alerts", 0)
    fps = meta.get("fps", "-")
    time_s = meta.get("time_str", "-")
    status = "ATTENTION" if high or alerts else ("WATCH" if med else "CLEAR")
    color = (0, 0, 220) if high or alerts else ((0, 140, 255) if med else (34, 150, 70))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 46), _BLACK, -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)
    cv2.rectangle(frame, (0, 0), (8, 46), color, -1)
    cv2.putText(frame, status, (18, 30), _FONT, 0.78, _WHITE, 2, cv2.LINE_AA)

    if frame.shape[1] < 760:
        details = f"S {len(students)}   A {alerts}   H/M {high}/{med}   {fps} FPS"
    else:
        details = (f"Students {len(students)}   Alerts {alerts}   "
                   f"High/Medium {high}/{med}   FPS {fps}   {time_s}")
    cv2.putText(frame, details, (180, 29), _FONT, 0.52, (230, 235, 240), 1, cv2.LINE_AA)


def annotate_frame(frame, result, objects, meta=None):
    """Draw all overlays onto `frame` in place and return it."""
    h, w = frame.shape[:2]
    students = result.get("students", [])
    centers = {s["id"]: tuple(map(int, s["center"])) for s in students}
    reserved = [(0, h - 30, w, h)]

    # --- interaction (proximity) lines -------------------------------------
    for pair in result.get("proximity_pairs", []):
        a, b = pair
        if a in centers and b in centers:
            cv2.line(frame, centers[a], centers[b], (0, 0, 255), 3)

    # --- student boxes and risk bars ---------------------------------------
    student_order = sorted(students, key=lambda item: _severity_rank(item["severity"]))
    for s in student_order:
        x1, y1, x2, y2 = s["bbox"]
        color = tuple(int(c) for c in s["bgr"])
        thickness = 4 if s["severity"] in ("MEDIUM", "HIGH") else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        bar_w = max(0, x2 - x1)
        fill = int(bar_w * s["risk"])
        by = min(y2 + 6, h - 40)
        cv2.rectangle(frame, (x1, by), (x2, by + 10), (35, 35, 35), -1)
        if fill > 0:
            cv2.rectangle(frame, (x1, by), (x1 + fill, by + 10), color, -1)

    # --- prohibited objects ------------------------------------------------
    for o in objects:
        x1, y1, x2, y2 = o["bbox"]
        label = o["label"]
        is_audio = label in _AUDIO_LABELS
        obj_color = (0, 0, 230) if is_audio else (0, 100, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), obj_color, 3)

    # Draw fixed UI chrome after boxes so it stays readable.
    if meta:
        _top_hud(frame, students, meta)
        reserved.append((0, 0, w, 50))

    banner_y = 56 if meta else 10
    for o in objects:
        label = o["label"]
        conf = int(round(float(o.get("conf", 0.0)) * 100))
        is_audio = label in _AUDIO_LABELS
        if is_audio:
            obj_color = (0, 0, 230)
            banner_y, rect = _banner(
                frame, f"AUDIO DEVICE DETECTED: {label.upper()} {conf}%",
                obj_color, banner_y,
            )
            reserved.append(rect)

    # Labels are placed after persistent banners and avoid each other.
    for pair in result.get("proximity_pairs", []):
        a, b = pair
        if a in centers and b in centers:
            mx = (centers[a][0] + centers[b][0]) // 2
            my = (centers[a][1] + centers[b][1]) // 2
            _label_avoiding(frame, "TOO CLOSE", mx - 40,
                            [my, my + 26, my - 26], (0, 0, 220), reserved, 0.55, 2)

    # --- students ----------------------------------------------------------
    for s in sorted(students, key=lambda item: _severity_rank(item["severity"]), reverse=True):
        x1, y1, x2, y2 = s["bbox"]
        color = tuple(int(c) for c in s["bgr"])
        label = f"STUDENT {s['id']}  {s['severity']}  RISK {_pct(s['risk'])}%"
        _label_avoiding(frame, label, x1, _above_or_inside_candidates(y1, y2),
                        color, reserved, 0.58, 2)

    # --- prohibited object labels -----------------------------------------
    for o in objects:
        x1, y1, x2, y2 = o["bbox"]
        label = o["label"]
        conf = int(round(float(o.get("conf", 0.0)) * 100))
        is_audio = label in _AUDIO_LABELS
        obj_color = (0, 0, 230) if is_audio else (0, 100, 255)
        _label_avoiding(frame, f"{label.upper()} {conf}%  MALPRACTICE",
                        x1, _object_label_candidates(y1, y2),
                        obj_color, reserved, 0.55, 2)

    # --- legend ------------------------------------------------------------
    if w < 760:
        legend = "GREEN CLEAR   YELLOW WATCH   RED MALPRACTICE"
    else:
        legend = "GREEN CLEAR   YELLOW WATCH   ORANGE MEDIUM   RED HIGH / MALPRACTICE"
    cv2.rectangle(frame, (0, h - 28), (w, h), _BLACK, -1)
    cv2.putText(frame, legend, (12, h - 9), _FONT, 0.48, (235, 235, 235), 1, cv2.LINE_AA)

    return frame
