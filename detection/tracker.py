"""
tracker.py
==========
Persistent multi-student tracking.

Primary backend : DeepSORT (deep-sort-realtime) — appearance + motion matching,
                  robust through occlusions (as described in the spec).
Fallback backend: a dependency-free IoU tracker that activates automatically if
                  deep-sort-realtime is unavailable or fails at runtime, so a
                  tracking failure can never bring down a monitoring session.

Both backends expose the same `update(persons, frame)` contract and assign
stable integer student IDs for the whole session.
"""

import config

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    _DEEPSORT_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    _DEEPSORT_AVAILABLE = False


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center(b):
    return ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)


class _IoUTracker:
    """Lightweight SORT-style tracker (greedy IoU association, no Kalman)."""

    def __init__(self, max_age, n_init, iou_threshold):
        self.max_age = max_age
        self.n_init = n_init
        self.iou_threshold = iou_threshold
        self.tracks = {}     # id -> {bbox, center, tsu (time since update), hits}
        self.next_id = 1

    def update(self, persons):
        dets = [p["bbox"] for p in persons]

        # candidate (iou, track_id, det_index) pairs above threshold
        cand = []
        for tid, tr in self.tracks.items():
            for di, db in enumerate(dets):
                iou = _iou(tr["bbox"], db)
                if iou >= self.iou_threshold:
                    cand.append((iou, tid, di))
        cand.sort(reverse=True)

        matched_t, matched_d, assigned = set(), set(), {}
        for iou, tid, di in cand:
            if tid in matched_t or di in matched_d:
                continue
            matched_t.add(tid)
            matched_d.add(di)
            tr = self.tracks[tid]
            tr["bbox"] = dets[di]
            tr["center"] = _center(dets[di])
            tr["tsu"] = 0
            tr["hits"] += 1
            assigned[di] = tid

        # unmatched detections -> new tracks
        for di, db in enumerate(dets):
            if di in matched_d:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {"bbox": db, "center": _center(db), "tsu": 0, "hits": 1}
            matched_t.add(tid)

        # age & prune tracks not updated this frame
        for tid in list(self.tracks.keys()):
            if tid not in matched_t:
                self.tracks[tid]["tsu"] += 1
                if self.tracks[tid]["tsu"] > self.max_age:
                    del self.tracks[tid]

        out = []
        for tid, tr in self.tracks.items():
            if tr["tsu"] == 0 and tr["hits"] >= self.n_init:
                out.append({"id": tid, "bbox": list(tr["bbox"]), "center": tr["center"]})
        return out


class StudentTracker:
    def __init__(self, device: str = "cpu", cfg: dict = None, prefer_deepsort: bool = True):
        self.cfg = cfg or config.TRACKING
        self.backend = "iou"
        self._ds = None

        if prefer_deepsort and _DEEPSORT_AVAILABLE:
            try:
                self._ds = DeepSort(
                    max_age=self.cfg["max_age"],
                    n_init=self.cfg["n_init"],
                    max_iou_distance=self.cfg["max_iou_distance"],
                    max_cosine_distance=self.cfg["max_cosine_distance"],
                    nn_budget=self.cfg["nn_budget"],
                    embedder=self.cfg["embedder"],
                    half=self.cfg["half"],
                    bgr=True,
                )
                self.backend = "deepsort"
            except Exception:
                self._ds = None
                self.backend = "iou"

        self._iou = _IoUTracker(
            max_age=self.cfg["max_age"],
            n_init=self.cfg["n_init"],
            iou_threshold=self.cfg["iou_match_threshold"],
        )

    # ------------------------------------------------------------------ #
    def update(self, persons, frame):
        """Return confirmed tracks: [{"id", "bbox", "center"}, ...]."""
        if self._ds is not None:
            try:
                return self._update_deepsort(persons, frame)
            except Exception:
                # Permanently fall back if DeepSORT errors mid-session.
                self._ds = None
                self.backend = "iou"
        return self._iou.update(persons)

    def _update_deepsort(self, persons, frame):
        raw = []
        for p in persons:
            x1, y1, x2, y2 = p["bbox"]
            raw.append(([x1, y1, x2 - x1, y2 - y1], p.get("conf", 0.9), "person"))
        tracks = self._ds.update_tracks(raw, frame=frame)
        out = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            l, top, r, b = [int(v) for v in t.to_ltrb()]
            out.append({"id": int(t.track_id), "bbox": [l, top, r, b],
                        "center": ((l + r) // 2, (top + b) // 2)})
        return out
