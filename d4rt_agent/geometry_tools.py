"""Geometry tool layer for the D4RT agent (v1).

Wraps a pre-built demo bundle (``demo/<case>/assets/demo_data.json``) as a set of
deterministic geometric queries the VLM can call. All spatial math lives here and
returns *scalars / short structs* -- the VLM never sees raw point clouds.

Coordinate frame: ``xyzRef0`` is the reconstructed 3D position (reference frame 0),
treated as metric meters. Time is indexed by integer frame; wall-clock uses ``fps``.

Two backends are exposed via ``source``:
  * ``"pred"`` -- D4RT's predicted tracks (what the agent actually reasons over).
  * ``"gt"``   -- ground-truth tracks (used only for scoring / auto-GT generation).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TrackHit:
    track_id: int
    pixel_dist: float
    uv: tuple[float, float]


class DemoGeometry:
    """Deterministic geometry queries backed by a demo_data.json bundle."""

    def __init__(self, demo_dir: str | Path, source: str = "pred") -> None:
        self.demo_dir = Path(demo_dir)
        data_path = self.demo_dir / "assets" / "demo_data.json"
        if not data_path.exists():
            raise FileNotFoundError(f"demo_data.json not found: {data_path}")
        with open(data_path) as f:
            self._data = json.load(f)

        meta = self._data["meta"]
        self.fps = float(meta["fps"])
        self.num_frames = int(meta["numFrames"])
        self.width = int(meta["videoWidth"])
        self.height = int(meta["videoHeight"])

        # Dense point cloud carries the authoritative dynamic/static label.
        pts = self._data["points"]
        self._pts_uv = np.asarray(pts["uvPx"], dtype=np.float64)          # (T, P, 2)
        self._pts_is_dynamic = np.asarray(pts["isDynamic"], dtype=np.int64)  # (P,)
        self._pts_motion = np.asarray(pts["motionScore"], dtype=np.float64)  # (P,)

        self.set_source(source)

    # -- backend selection -------------------------------------------------
    def set_source(self, source: str) -> None:
        key = {"pred": "tracks", "gt": "tracksGt"}.get(source)
        if key is None:
            raise ValueError(f"source must be 'pred' or 'gt', got {source!r}")
        self.source = source
        tr = self._data[key]
        self._xyz = np.asarray(tr["xyzRef0"], dtype=np.float64)     # (N, T, 3)
        self._uv = np.asarray(tr["uvPx"], dtype=np.float64)         # (N, T, 2)
        self._vis = np.asarray(tr["visibility"], dtype=np.int64)    # (N, T)
        self._query_uv = np.asarray(tr["queryUvPx"], dtype=np.float64)  # (N, 2)
        self._query_t = np.asarray(tr["queryTSrc"], dtype=np.int64)     # (N,)
        self.num_tracks = self._xyz.shape[0]

    # -- helpers -----------------------------------------------------------
    def _check_t(self, t: int) -> int:
        t = int(t)
        if not (0 <= t < self.num_frames):
            raise ValueError(f"frame {t} out of range [0, {self.num_frames})")
        return t

    def _check_id(self, track_id: int) -> int:
        track_id = int(track_id)
        if not (0 <= track_id < self.num_tracks):
            raise ValueError(f"track_id {track_id} out of range [0, {self.num_tracks})")
        return track_id

    # -- grounding ---------------------------------------------------------
    def nearest_track(self, u: float, v: float, t: int) -> TrackHit:
        """Snap a (u, v) pixel at frame t to the closest *visible* track."""
        t = self._check_t(t)
        uv_t = self._uv[:, t, :]
        vis_t = self._vis[:, t] > 0
        if not vis_t.any():
            raise ValueError(f"no visible tracks at frame {t}")
        d = np.full(self.num_tracks, np.inf)
        d[vis_t] = np.linalg.norm(uv_t[vis_t] - np.array([u, v]), axis=1)
        tid = int(np.argmin(d))
        return TrackHit(track_id=tid, pixel_dist=float(d[tid]), uv=(float(uv_t[tid, 0]), float(uv_t[tid, 1])))

    def nearby_tracks(self, u: float, v: float, t: int, radius_px: float = 12.0) -> list[int]:
        """Return visible tracks whose projection lies within ``radius_px``."""
        t = self._check_t(t)
        distance = np.linalg.norm(self._uv[:, t] - np.array([u, v]), axis=1)
        keep = (self._vis[:, t] > 0) & (distance <= float(radius_px))
        return [int(i) for i in np.flatnonzero(keep)]

    def deduplicate_tracks(self, track_ids: list[int], atol: float = 1e-6) -> list[int]:
        """Collapse repeated query entries that contain the same full trajectory."""
        unique: list[int] = []
        for track_id in track_ids:
            track_id = self._check_id(track_id)
            duplicate = any(
                np.array_equal(self._vis[track_id], self._vis[other])
                and np.allclose(self._xyz[track_id], self._xyz[other], rtol=0.0, atol=atol)
                for other in unique
            )
            if not duplicate:
                unique.append(track_id)
        return unique

    def ground_track_group(
        self, u: float, v: float, t: int, radius_px: float = 12.0
    ) -> dict:
        """Ground a point to nearby tracks and explicitly collapse duplicates."""
        hit = self.nearest_track(u, v, t)
        candidates = self.nearby_tracks(u, v, t, radius_px=radius_px)
        if not candidates:
            candidates = [hit.track_id]
        unique = self.deduplicate_tracks(candidates)
        return {
            "representative_track_id": unique[0],
            "candidate_track_ids": candidates,
            "unique_track_ids": unique,
            "duplicate_tracks_removed": len(candidates) - len(unique),
            "nearest_pixel_distance": round(hit.pixel_dist, 4),
            "radius_px": float(radius_px),
        }

    @staticmethod
    def _metric_spread(values: list[float]) -> dict:
        array = np.asarray(values, dtype=np.float64)
        return {
            "median": round(float(np.median(array)), 4),
            "min": round(float(np.min(array)), 4),
            "max": round(float(np.max(array)), 4),
            "std": round(float(np.std(array)), 4),
        }

    def group_endpoint_measurement(self, track_ids: list[int], t0: int, t1: int) -> dict:
        """Median endpoint measurement across deduplicated nearby tracks."""
        unique = self.deduplicate_tracks(track_ids)
        if not unique:
            raise ValueError("track group is empty")
        values = [self.displacement(track_id, t0, t1) for track_id in unique]
        result = self.endpoint_measurement(unique[0], t0, t1)
        spread = self._metric_spread(values)
        result.update({
            "track_ids": unique,
            "endpoint_displacement_m": spread["median"],
            "metric_spread_m": spread,
        })
        return result

    def group_path_measurement(self, track_ids: list[int], t0: int, t1: int) -> dict:
        """Median visible path length across deduplicated nearby tracks."""
        unique = self.deduplicate_tracks(track_ids)
        if not unique:
            raise ValueError("track group is empty")
        values = [self.path_length(track_id, t0, t1) for track_id in unique]
        result = self.path_measurement(unique[0], t0, t1)
        spread = self._metric_spread(values)
        result.update({
            "track_ids": unique,
            "path_length_m": spread["median"],
            "metric_spread_m": spread,
        })
        return result

    # -- per-track geometry ------------------------------------------------
    def visible(self, track_id: int, t: int) -> bool:
        return bool(self._vis[self._check_id(track_id), self._check_t(t)] > 0)

    def position(self, track_id: int, t: int) -> tuple[float, float, float]:
        """3D position (meters) of a track at frame t."""
        track_id, t = self._check_id(track_id), self._check_t(t)
        xyz = self._xyz[track_id, t]
        return (float(xyz[0]), float(xyz[1]), float(xyz[2]))

    def pixel(self, track_id: int, t: int) -> tuple[float, float]:
        track_id, t = self._check_id(track_id), self._check_t(t)
        uv = self._uv[track_id, t]
        return (float(uv[0]), float(uv[1]))

    def distance(self, id_a: int, id_b: int, t: int) -> float:
        """Euclidean 3D distance (meters) between two tracks at frame t."""
        a = np.asarray(self.position(id_a, t))
        b = np.asarray(self.position(id_b, t))
        return float(np.linalg.norm(a - b))

    def displacement(self, track_id: int, t0: int, t1: int) -> float:
        """Straight-line distance (meters) between a track's position at t0 and t1."""
        a = np.asarray(self.position(track_id, t0))
        b = np.asarray(self.position(track_id, t1))
        return float(np.linalg.norm(a - b))

    def visible_segments(self, track_id: int, t0: int, t1: int) -> list[list[int]]:
        """Return inclusive, contiguous visible intervals inside ``[t0, t1]``."""
        track_id = self._check_id(track_id)
        t0, t1 = self._check_t(t0), self._check_t(t1)
        lo, hi = sorted((t0, t1))
        visible = self._vis[track_id, lo:hi + 1] > 0
        segments: list[list[int]] = []
        start: int | None = None
        for offset, is_visible in enumerate(np.append(visible, False)):
            frame = lo + offset
            if is_visible and start is None:
                start = frame
            elif not is_visible and start is not None:
                segments.append([start, frame - 1])
                start = None
        return segments

    def endpoint_measurement(self, track_id: int, t0: int, t1: int) -> dict:
        """Endpoint displacement with visibility diagnostics."""
        track_id = self._check_id(track_id)
        t0, t1 = self._check_t(t0), self._check_t(t1)
        p0 = self.position(track_id, t0)
        p1 = self.position(track_id, t1)
        return {
            "track_id": track_id,
            "t0": t0,
            "t1": t1,
            "position_t0_m": [round(value, 4) for value in p0],
            "position_t1_m": [round(value, 4) for value in p1],
            "endpoint_displacement_m": round(self.displacement(track_id, t0, t1), 4),
            "endpoint_visible": [self.visible(track_id, t0), self.visible(track_id, t1)],
        }

    def path_length(self, track_id: int, t0: int, t1: int) -> float:
        """Total path length (meters) accumulated over visible frames in [t0, t1]."""
        track_id = self._check_id(track_id)
        t0, t1 = self._check_t(t0), self._check_t(t1)
        lo, hi = (t0, t1) if t0 <= t1 else (t1, t0)
        xyz = self._xyz[track_id, lo:hi + 1]
        vis = self._vis[track_id, lo:hi + 1] > 0
        total = 0.0
        prev = None
        for p, ok in zip(xyz, vis):
            if not ok:
                prev = None
                continue
            if prev is not None:
                total += float(np.linalg.norm(p - prev))
            prev = p
        return total

    def path_measurement(self, track_id: int, t0: int, t1: int) -> dict:
        """Visible-only path length with coverage and gap diagnostics.

        No distance is added across a visibility gap. This is intentionally a
        conservative measurement rather than an implicit interpolation.
        """
        track_id = self._check_id(track_id)
        t0, t1 = self._check_t(t0), self._check_t(t1)
        lo, hi = sorted((t0, t1))
        visible_frames = int(np.count_nonzero(self._vis[track_id, lo:hi + 1] > 0))
        total_frames = hi - lo + 1
        segments = self.visible_segments(track_id, lo, hi)
        return {
            "track_id": track_id,
            "t0": t0,
            "t1": t1,
            "path_length_m": round(self.path_length(track_id, lo, hi), 4),
            "visible_frames": visible_frames,
            "total_frames": total_frames,
            "coverage": round(visible_frames / total_frames, 4),
            "visible_segments": segments,
            "measurement": "visible trajectory only",
        }

    def speed(self, track_id: int, t0: int, t1: int, mode: str = "average") -> float:
        """Speed in m/s between t0 and t1.

        mode='average' -> net displacement / elapsed time.
        mode='path'    -> path length / elapsed time (mean instantaneous speed).
        """
        t0, t1 = self._check_t(t0), self._check_t(t1)
        dt = abs(t1 - t0) / self.fps
        if dt == 0:
            return 0.0
        dist = self.displacement(track_id, t0, t1) if mode == "average" else self.path_length(track_id, t0, t1)
        return dist / dt

    def motion_label(self, track_id: int, move_thresh: float = 0.15) -> dict:
        """Classify moving/static from the track's own trajectory (net displacement).

        This is derived purely from the track geometry, so it is consistent with the
        displacement/speed the same backend reports. ``move_thresh`` is meters of net
        displacement over the visible span.
        """
        track_id = self._check_id(track_id)
        vis_frames = np.where(self._vis[track_id] > 0)[0]
        if vis_frames.size < 2:
            return {"is_moving": False, "net_displacement_m": 0.0, "visible_frames": int(vis_frames.size)}
        t0, t1 = int(vis_frames[0]), int(vis_frames[-1])
        net = self.displacement(track_id, t0, t1)
        return {"is_moving": bool(net > move_thresh), "net_displacement_m": round(net, 4),
                "visible_frames": int(vis_frames.size)}

    def is_dynamic(self, track_id: int) -> dict:
        """Auxiliary: nearest dense point's isDynamic label (D4RT's motion segmentation)."""
        track_id = self._check_id(track_id)
        # Match at the track's query frame in pixel space against the point cloud.
        t = int(self._query_t[track_id])
        t = t if 0 <= t < self.num_frames else 0
        uv = self._uv[track_id, t]
        d = np.linalg.norm(self._pts_uv[t] - uv, axis=1)
        p = int(np.argmin(d))
        return {
            "is_dynamic": bool(self._pts_is_dynamic[p]),
            "motion_score": float(self._pts_motion[p]),
            "match_pixel_dist": float(d[p]),
        }

    def trajectory_summary(self, track_id: int) -> dict:
        """Compact description of a track's motion over the whole clip."""
        track_id = self._check_id(track_id)
        vis = self._vis[track_id] > 0
        vis_frames = np.where(vis)[0]
        if vis_frames.size == 0:
            return {"track_id": track_id, "visible_frames": 0}
        t0, t1 = int(vis_frames[0]), int(vis_frames[-1])
        return {
            "track_id": track_id,
            "visible_frames": int(vis.sum()),
            "first_visible_t": t0,
            "last_visible_t": t1,
            "net_displacement_m": round(self.displacement(track_id, t0, t1), 4),
            "path_length_m": round(self.path_length(track_id, t0, t1), 4),
            "avg_speed_mps": round(self.speed(track_id, t0, t1), 4),
            **self.is_dynamic(track_id),
        }
