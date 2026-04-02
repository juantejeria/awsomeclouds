"""
Lightweight center-distance multi-object tracker for cross-frame animal tracking.

Uses center-point distance (normalised by bbox size) instead of IoU for
matching — much more robust when animals walk side-by-side and their body
bounding boxes overlap heavily.

Zero new dependencies — uses only Python builtins + numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def _bbox_center(box: list[float]) -> tuple[float, float]:
    """Return (cx, cy) center of an [x1, y1, x2, y2] bbox."""
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _bbox_diag(box: list[float]) -> float:
    """Return diagonal length of an [x1, y1, x2, y2] bbox."""
    w = box[2] - box[0]
    h = box[3] - box[1]
    return math.sqrt(w * w + h * h)


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute Intersection-over-Union for two [x1, y1, x2, y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter == 0.0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def center_distance(box_a: list[float], box_b: list[float]) -> float:
    """
    Normalised center distance between two bboxes.

    Returns a value in [0, inf).  A value of 0 means the centers coincide;
    a value of 1.0 means the centers are separated by one diagonal-length
    of the average bbox.  Lower is better.
    """
    ca = _bbox_center(box_a)
    cb = _bbox_center(box_b)
    dist = math.sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2)
    avg_diag = (_bbox_diag(box_a) + _bbox_diag(box_b)) / 2.0
    if avg_diag <= 0:
        return float("inf")
    return dist / avg_diag


@dataclass
class Track:
    """A single tracked object across multiple frames."""
    track_id: int
    detections: list[tuple[int, dict]] = field(default_factory=list)
    # detections: list of (frame_idx, det_dict) where det_dict comes from
    # detect_best_boxes_multiscale (has keys: bbox, bbox_padded, confidence, animal, is_face, …)
    frames_since_seen: int = 0
    active: bool = True

    @property
    def last_bbox(self) -> list[float]:
        """Bounding box of the most recent detection."""
        return self.detections[-1][1]["bbox"]


class MultiObjectTracker:
    """
    Center-distance multi-object tracker with IoU fallback.

    Matching strategy per frame:
      1. Compute normalised center distance between each active track's
         last bbox and each new detection.
      2. Greedy assignment by ascending distance (closest first).
      3. A match is accepted if distance < ``max_center_dist``.
      4. Unmatched detections create new tracks.
      5. Unmatched tracks age; deactivated after ``max_age`` frames.

    Center distance is far more robust than pure IoU when animals walk
    side-by-side (overlapping body bboxes but distinct centers).
    """

    def __init__(
        self,
        max_center_dist: float = 0.6,
        max_age: int = 15,
        min_track_length: int = 5,
        # Legacy param kept for API compat — ignored in favour of center dist
        iou_threshold: float = 0.15,
    ) -> None:
        self.max_center_dist = max_center_dist
        self.max_age = max_age
        self.min_track_length = min_track_length
        self._tracks: list[Track] = []
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        frame_idx: int,
        detections: list[dict],
    ) -> list[tuple[int, dict]]:
        """
        Process one frame's detections and return matched (track_id, det) pairs.

        Parameters
        ----------
        frame_idx : int
            Index of the current frame (for bookkeeping).
        detections : list[dict]
            Detection dicts from ``detect_best_boxes_multiscale``.

        Returns
        -------
        list[tuple[int, dict]]
            Each element is ``(track_id, detection_dict)`` for every
            detection that was either matched to an existing track or
            assigned a brand-new track.
        """
        active_tracks = [t for t in self._tracks if t.active]

        if not active_tracks or not detections:
            # Nothing to match — all detections become new tracks
            results: list[tuple[int, dict]] = []
            for det in detections:
                tid = self._create_track(frame_idx, det)
                results.append((tid, det))
            # Age unmatched active tracks
            for t in active_tracks:
                t.frames_since_seen += 1
                if t.frames_since_seen > self.max_age:
                    t.active = False
            return results

        # Build distance matrix: (distance, track_idx, det_idx)
        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(active_tracks):
            for di, det in enumerate(detections):
                dist = center_distance(track.last_bbox, det["bbox"])
                if dist <= self.max_center_dist:
                    pairs.append((dist, ti, di))

        # Greedy assignment by ascending distance (closest first)
        pairs.sort(key=lambda x: x[0])
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        results = []

        for dist_val, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            track = active_tracks[ti]
            det = detections[di]
            track.detections.append((frame_idx, det))
            track.frames_since_seen = 0
            matched_tracks.add(ti)
            matched_dets.add(di)
            results.append((track.track_id, det))

        # Unmatched detections → new tracks
        for di, det in enumerate(detections):
            if di not in matched_dets:
                tid = self._create_track(frame_idx, det)
                results.append((tid, det))

        # Unmatched active tracks → age
        for ti, track in enumerate(active_tracks):
            if ti not in matched_tracks:
                track.frames_since_seen += 1
                if track.frames_since_seen > self.max_age:
                    track.active = False

        return results

    def get_tracks(self, min_length: int | None = None) -> list[Track]:
        """
        Return all tracks, optionally filtered by minimum detection count.

        Parameters
        ----------
        min_length : int | None
            If provided, only return tracks with at least this many detections.
            Defaults to ``self.min_track_length``.
        """
        if min_length is None:
            min_length = self.min_track_length
        return [t for t in self._tracks if len(t.detections) >= min_length]

    @property
    def all_tracks(self) -> list[Track]:
        """Return every track regardless of length."""
        return list(self._tracks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_track(self, frame_idx: int, det: dict) -> int:
        tid = self._next_id
        self._next_id += 1
        self._tracks.append(Track(
            track_id=tid,
            detections=[(frame_idx, det)],
            frames_since_seen=0,
            active=True,
        ))
        return tid
