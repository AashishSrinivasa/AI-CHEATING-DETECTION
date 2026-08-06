"""
behavior_analyzer.py
====================
Fuses signals from the head-pose, object and tracking modules into a single,
continuously-updated risk score (0.0 - 1.0) per student.

Signals & weights (see config.BEHAVIOR["weights"]):
    head_pose  : frequency of sideways / downward looking
    object     : prohibited-item detections near the student
    movement   : excessive velocity / erratic acceleration
    proximity  : closeness to other students (possible interaction)

Each component accumulates per-event increments, is clamped to [0, 1] and
decays every frame so that risk cools down when behaviour returns to normal.
Sustained behaviour (>= alert_frames) raises a timestamped alert; momentary
movements are ignored, eliminating false positives.
"""

import math
from collections import deque, defaultdict

import config


ALERT_MESSAGES = {
    "looking_sideways": "Looking sideways (possible copying from neighbour)",
    "looking_down": "Looking down (possible notes / phone on lap)",
    "object": "Prohibited object detected",
    "movement": "Excessive / erratic movement",
    "proximity": "Too close to another student (possible interaction)",
    "left_zone": "Left assigned seating zone",
}


class StudentState:
    """Per-student behavioural memory for the duration of a session."""

    def __init__(self, sid: int, history_len: int):
        self.id = sid
        self.centroids = deque(maxlen=history_len)
        self.velocities = deque(maxlen=history_len)
        self.components = {"head_pose": 0.0, "object": 0.0,
                           "movement": 0.0, "proximity": 0.0}
        self.risk = 0.0
        self.severity = "NORMAL"
        self.max_risk = 0.0

        self.counters = defaultdict(int)   # consecutive frames a behaviour holds
        self.active = defaultdict(bool)    # whether an alert is currently raised

        self.events = defaultdict(int)     # aggregate counts for the report
        self.frames_present = 0
        self.first_frame = None
        self.last_frame = None

        self.last_pose = None              # {"yaw","pitch","yaw_adj","pitch_adj"}
        self.last_object = None            # label of most recent prohibited item
        self.last_object_conf = None
        self.last_object_source = None
        self.last_velocity = 0.0


class BehaviorAnalyzer:
    def __init__(self, fps: float = None, head_pose=None,
                 behavior_cfg: dict = None, movement_cfg: dict = None,
                 sensitivity: str = None):
        self.cfg = behavior_cfg or config.BEHAVIOR
        self.move = movement_cfg or config.MOVEMENT
        self.profile = config.sensitivity_profile(sensitivity)
        self.fps = fps or config.DEFAULT_FPS
        self.head_pose = head_pose          # used for dynamic head-pose classify
        self.students = {}
        self.alerts = []                    # global chronological alert log
        self.frame_count = 0
        # Which signals are contributing. Lets the unified score renormalise so
        # it stays a meaningful 0-1 even if an engine (e.g. head pose) is off.
        self.active = {"head_pose": True, "object": True,
                       "movement": True, "proximity": True}

    def set_active_signals(self, **flags):
        """Enable/disable signals, e.g. set_active_signals(head_pose=False)."""
        self.active.update(flags)

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _center(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _point_in_box(pt, box):
        x, y = pt
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    @staticmethod
    def _dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _fmt_time(seconds):
        s = int(seconds)
        return f"{s // 60:02d}:{s % 60:02d}"

    def _get(self, sid):
        if sid not in self.students:
            self.students[sid] = StudentState(sid, self.move["history_length"])
        return self.students[sid]

    def _alert_frames(self, key, label=None):
        if key in ("looking_sideways", "looking_down"):
            base = config.HEAD_POSE["alert_frames"]
            return max(2, int(round(base * self.profile["alert_frame_scale"])))
        if key == "object":
            base = config.OBJECT_DETECTION.get("alert_frames_by_class", {}).get(
                label, config.OBJECT_DETECTION["alert_frames"])
            return max(2, int(round(base * self.profile["alert_frame_scale"])))
        return max(2, int(round(self.move["alert_frames"] * self.profile["alert_frame_scale"])))

    def _classify_pose(self, sid, yaw, pitch):
        if self.head_pose is not None:
            return self.head_pose.classify(sid, yaw, pitch)
        # Static fallback if no detector reference was supplied
        hp = config.HEAD_POSE
        return {
            "sideways": abs(yaw) > hp["yaw_threshold"],
            "down": pitch > hp["pitch_down_threshold"],
            "up": pitch < -hp["pitch_up_threshold"],
            "yaw_adj": yaw, "pitch_adj": pitch,
        }

    # ------------------------------------------------------------------ #
    # Main per-frame update
    # ------------------------------------------------------------------ #
    def update(self, frame_idx, timestamp, tracked, faces, objects, frame_shape):
        """
        Update all student states for the current frame.

        Returns a render-ready snapshot:
            {
              "students": [ {id, bbox, center, risk, severity, hex, bgr,
                             components, pose, object, velocity}, ... ],
              "proximity_pairs": [ [id_a, id_b], ... ],
              "alerts": [ <new alerts this frame> ],
            }
        """
        self.frame_count += 1
        h, w = frame_shape[:2]
        diag = math.hypot(w, h)
        inc = self.cfg["increments"]
        decay = self.cfg["decay"]
        weights = self.cfg["weights"]
        active_w = {k: v for k, v in weights.items() if self.active.get(k, True)}
        total_w = sum(active_w.values()) or 1.0
        new_alerts = []
        present = set()

        # ---- 1. associate faces -> students (face centre inside person box) --
        face_for = {}
        for face in faces:
            fc = face["center"]
            best, best_area = None, None
            for s in tracked:
                if self._point_in_box(fc, s["bbox"]):
                    x1, y1, x2, y2 = s["bbox"]
                    area = (x2 - x1) * (y2 - y1)
                    if best is None or area < best_area:
                        best, best_area = s["id"], area
            if best is not None and best not in face_for:
                face_for[best] = face

        # ---- 2. associate objects -> nearest student within range ----------
        obj_for = defaultdict(list)
        max_obj_dist = 0.28 * diag
        for obj in objects:
            oc = self._center(obj["bbox"])
            best, bestd = None, None
            for s in tracked:
                d = self._dist(oc, s["center"])
                if best is None or d < bestd:
                    best, bestd = s["id"], d
            if best is not None and bestd <= max_obj_dist:
                obj_for[best].append(obj)

        # ---- 3. proximity pairs --------------------------------------------
        close_ids = set()
        prox_pairs = []
        prox_thr = self.move["proximity_threshold"]
        for i in range(len(tracked)):
            for j in range(i + 1, len(tracked)):
                d = self._dist(tracked[i]["center"], tracked[j]["center"])
                if d < prox_thr:
                    close_ids.add(tracked[i]["id"])
                    close_ids.add(tracked[j]["id"])
                    prox_pairs.append([tracked[i]["id"], tracked[j]["id"]])

        # ---- 4. per-student processing -------------------------------------
        snapshot_students = []
        for s in tracked:
            sid = s["id"]
            present.add(sid)
            st = self._get(sid)
            st.frames_present += 1
            if st.first_frame is None:
                st.first_frame = frame_idx
            st.last_frame = frame_idx
            st.centroids.append(s["center"])

            # decay all components before adding this frame's evidence
            for k in st.components:
                st.components[k] *= decay

            # -- head pose --
            pose = face_for.get(sid)
            if pose is not None:
                flags = self._classify_pose(sid, pose["yaw"], pose["pitch"])
                st.last_pose = {
                    "yaw": pose["yaw"], "pitch": pose["pitch"],
                    "yaw_adj": flags["yaw_adj"], "pitch_adj": flags["pitch_adj"],
                }
                if flags["sideways"]:
                    st.components["head_pose"] = min(
                        1.0, st.components["head_pose"] + inc["looking_sideways"])
                    st.events["looking_sideways"] += 1
                    st.counters["looking_sideways"] += 1
                else:
                    self._reset(st, "looking_sideways")
                if flags["down"]:
                    st.components["head_pose"] = min(
                        1.0, st.components["head_pose"] + inc["looking_down"])
                    st.events["looking_down"] += 1
                    st.counters["looking_down"] += 1
                else:
                    self._reset(st, "looking_down")
            else:
                st.last_pose = None
                self._reset(st, "looking_sideways")
                self._reset(st, "looking_down")

            # -- objects --
            objs = obj_for.get(sid, [])
            if objs:
                def obj_strength(o):
                    label = o["label"]
                    risk_inc = config.OBJECT_DETECTION.get("risk_increments", {}).get(
                        label, inc["object_detected"])
                    return risk_inc * float(o.get("conf", 0.5))

                primary = max(objs, key=obj_strength)
                st.last_object = primary["label"]
                st.last_object_conf = primary.get("conf")
                st.last_object_source = primary.get("source")
                risk_inc = config.OBJECT_DETECTION.get("risk_increments", {}).get(
                    st.last_object, inc["object_detected"])
                st.components["object"] = min(
                    1.0, st.components["object"] + risk_inc * self.profile["risk_scale"])
                for o in objs:
                    st.events[f"object:{o['label']}"] += 1
                st.counters["object"] += 1
            else:
                st.last_object = None
                st.last_object_conf = None
                st.last_object_source = None
                self._reset(st, "object")

            # -- movement (velocity & acceleration) --
            v = 0.0
            if len(st.centroids) >= 2:
                v = self._dist(st.centroids[-1], st.centroids[-2])
            a = abs(v - st.velocities[-1]) if st.velocities else 0.0
            st.velocities.append(v)
            st.last_velocity = round(v, 2)
            if v > self.move["velocity_threshold"] or a > self.move["acceleration_threshold"]:
                st.components["movement"] = min(
                    1.0, st.components["movement"] + inc["excessive_movement"])
                st.events["excessive_movement"] += 1
                st.counters["movement"] += 1
            else:
                self._reset(st, "movement")

            # -- proximity --
            if sid in close_ids:
                st.components["proximity"] = min(
                    1.0, st.components["proximity"] + inc["proximity"])
                st.events["proximity"] += 1
                st.counters["proximity"] += 1
            else:
                self._reset(st, "proximity")

            # -- seating-zone monitoring (left assigned area) --
            zcfg = config.ZONES
            if zcfg.get("enabled") and zcfg.get("boundaries"):
                ncx, ncy = s["center"][0] / w, s["center"][1] / h
                inside = any(
                    z["rect"][0] <= ncx <= z["rect"][2]
                    and z["rect"][1] <= ncy <= z["rect"][3]
                    for z in zcfg["boundaries"]
                )
                if not inside:
                    st.components["movement"] = min(
                        1.0, st.components["movement"] + inc["left_zone"])
                    st.events["left_zone"] += 1
                    st.counters["left_zone"] += 1
                else:
                    self._reset(st, "left_zone")
            else:
                self._reset(st, "left_zone")

            # -- unified risk score --
            risk = sum(st.components[k] * weights[k] for k in active_w) / total_w
            risk *= self.profile["risk_scale"]
            st.risk = max(0.0, min(1.0, risk))
            name, bgr, hexc = config.severity_for(st.risk)
            st.severity = name
            st.max_risk = max(st.max_risk, st.risk)

            # -- emit alerts for any sustained behaviour --
            self._emit(st, frame_idx, timestamp, new_alerts)

            snapshot_students.append({
                "id": sid,
                "bbox": s["bbox"],
                "center": s["center"],
                "risk": round(st.risk, 3),
                "severity": name,
                "hex": hexc,
                "bgr": bgr,
                "components": {k: round(val, 3) for k, val in st.components.items()},
                "pose": st.last_pose,
                "object": st.last_object,
                "object_conf": st.last_object_conf,
                "object_source": st.last_object_source,
                "velocity": st.last_velocity,
            })

        # ---- 5. cool down students not present this frame ------------------
        for sid, st in self.students.items():
            if sid in present:
                continue
            for k in st.components:
                st.components[k] *= decay
            risk = sum(st.components[k] * weights[k] for k in active_w) / total_w
            risk *= self.profile["risk_scale"]
            st.risk = max(0.0, min(1.0, risk))
            st.severity = config.severity_for(st.risk)[0]

        return {
            "students": snapshot_students,
            "proximity_pairs": prox_pairs,
            "alerts": new_alerts,
        }

    # ------------------------------------------------------------------ #
    def _reset(self, st, key):
        st.counters[key] = 0
        st.active[key] = False

    def _emit(self, st, frame_idx, ts, new_alerts):
        for key in ("looking_sideways", "looking_down", "object",
                    "movement", "proximity", "left_zone"):
            count = st.counters.get(key, 0)
            if count >= self._alert_frames(key, st.last_object) and not st.active[key]:
                st.active[key] = True
                msg = ALERT_MESSAGES[key]
                if key == "object" and st.last_object:
                    if st.last_object in config.OBJECT_DETECTION.get("audio_device_labels", set()):
                        msg = f"Audio device detected: {st.last_object}"
                    else:
                        msg = f"Prohibited object detected: {st.last_object}"
                alert = {
                    "frame": frame_idx,
                    "timestamp": round(float(ts), 2),
                    "time_str": self._fmt_time(ts),
                    "student_id": st.id,
                    "type": key,
                    "message": msg,
                    "risk": round(st.risk, 3),
                    "severity": st.severity,
                }
                self.alerts.append(alert)
                new_alerts.append(alert)
                st.events["alerts"] += 1

    # ------------------------------------------------------------------ #
    def summary(self):
        """Aggregate everything into a JSON-serialisable report body."""
        students = []
        for sid, st in sorted(self.students.items()):
            students.append({
                "student_id": sid,
                "frames_present": st.frames_present,
                "first_frame": st.first_frame,
                "last_frame": st.last_frame,
                "max_risk": round(st.max_risk, 3),
                "final_risk": round(st.risk, 3),
                "final_severity": config.severity_for(st.max_risk)[0],
                "total_alerts": st.events.get("alerts", 0),
                "events": dict(st.events),
            })

        sev_counts = defaultdict(int)
        type_counts = defaultdict(int)
        for a in self.alerts:
            sev_counts[a["severity"]] += 1
            type_counts[a["type"]] += 1

        return {
            "total_students": len(self.students),
            "total_alerts": len(self.alerts),
            "severity_distribution": {k: sev_counts.get(k, 0)
                                      for k, *_ in config.BEHAVIOR["severity"]},
            "alert_type_distribution": dict(type_counts),
            "students": students,
            "alerts": self.alerts,
        }
