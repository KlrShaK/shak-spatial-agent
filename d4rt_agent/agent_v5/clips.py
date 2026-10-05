"""One clip as every v5 stage sees it: its frames on disk, its time axis, and what Qwen is shown.

Every consumer (SAM3's video predictor, the backbone, the keyframes Qwen reads) reads the same
folder ``<cache>/clips/<slug>/frames/%06d.jpg``, so a frame index means the same frame everywhere.

**Dyn-Bench.** The frames are the generator frame list (GF) rebuilt from
``frame_sampling.json`` (``scene.generator_frames``), and seconds are ``gf_index / fps`` with the
per-source rate in ``scene.FPS_GF`` -- see ``dyn_dataset_exploration/logs/DECOMPOSITION.md``
before touching either. The final answer turn shows the benchmark's own 16 sampled frames.

**DSI-Bench.** The decoded video, strided to at most ``MAX_GF_FRAMES`` frames; seconds come from
the container's frame rate. The answer turn shows 32 uniformly spaced frames, as v4 did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from d4rt_agent.agent_v5.scene import frame_rate, generator_frames, list_frames, seconds_axis

MAX_GF_FRAMES = 400          # Point4D's cap; longer DSI videos are strided
DSI_ANSWER_FRAMES = 32
KEYFRAMES = 8                # what the decomposition and tool turns show
QWEN_LONG_SIDE = 640         # frames are downsized for Qwen: 16-32 full frames would fill the context
KEYFRAME_LONG_SIDE = 448


@dataclass
class Clip:
    slug: str
    record: dict[str, Any]          # the clip.json payload (dataset, gf, fps_gf, sampled16, source_hw, portrait)
    frames_dir: Path
    _cache: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def dataset(self) -> str:
        return str(self.record["dataset"])

    @property
    def length(self) -> int:
        return int(self.record["gf"]["length"])

    @property
    def fps(self) -> float | None:
        return self.record.get("fps_gf")

    @property
    def answer_frames(self) -> list[int]:
        """The frames the benchmark's own protocol shows (16 for Dyn-Bench, 32 for DSI)."""

        return [int(i) for i in self.record["sampled16"]]

    def keyframes(self, n: int = KEYFRAMES) -> list[int]:
        """``n`` frames evenly spread over the answer frames (first and last included)."""

        frames = self.answer_frames
        if len(frames) <= n:
            return frames
        picks = np.unique(np.round(np.linspace(0, len(frames) - 1, n)).astype(int))
        return [frames[i] for i in picks]

    def seconds(self, index: int) -> float | None:
        return None if self.fps is None else float(index) / float(self.fps)

    def label(self, index: int) -> str:
        t = self.seconds(index)
        return f"Frame {index}" + (f" (t={t:.2f} s)" if t is not None else "")

    def frame(self, index: int) -> np.ndarray:
        """(H, W, 3) uint8 RGB at source resolution."""

        if index not in self._cache:
            from PIL import Image

            with Image.open(self.frames_dir / f"{int(index):06d}.jpg") as image:
                self._cache[index] = np.asarray(image.convert("RGB"))
        return self._cache[index]

    def frames(self, indices: Sequence[int]) -> np.ndarray:
        return np.stack([self.frame(i) for i in indices])

    def all_frames(self, max_frames: int = MAX_GF_FRAMES) -> tuple[np.ndarray, np.ndarray]:
        """The GF frames for a backbone, strided to at most ``max_frames``, and their GF indices."""

        step = int(np.ceil(self.length / max_frames))
        gf_index = np.arange(0, self.length, step)
        return np.stack([self._read(i) for i in gf_index]), gf_index

    def _read(self, index: int) -> np.ndarray:
        # not cached: a backbone pass reads up to 400 frames once
        from PIL import Image

        with Image.open(self.frames_dir / f"{int(index):06d}.jpg") as image:
            return np.asarray(image.convert("RGB"))

    def pil(self, index: int, long_side: int = QWEN_LONG_SIDE) -> Any:
        from PIL import Image

        image = Image.fromarray(self.frame(index))
        scale = long_side / max(image.size)
        if scale < 1:
            image = image.resize((round(image.width * scale), round(image.height * scale)))
        return image


# --------------------------------------------------------------------------- #
# Dyn-Bench
# --------------------------------------------------------------------------- #

def dynbench_record(dyn_root: Path, dataset: str, video: str) -> dict[str, Any]:
    """Everything later stages need to know about one Dyn-Bench clip, from disk alone
    (``decomposition.build_catalog.clip_record``, minus the GT-mask diagnostics)."""

    from PIL import Image

    from dyn_dataset_exploration.dyn_bench_data import video_slug

    clip = f"{dataset}_{video}"
    frames_dir = Path(dyn_root) / "frames" / dataset / clip
    sampling = json.loads(
        (Path(dyn_root) / "multi_json" / dataset / f"{clip}_frame_sampling.json").read_text())
    gf = generator_frames(list_frames(frames_dir), sampling)
    rate = frame_rate(dataset)
    with Image.open(frames_dir / gf.names[0]) as image:
        width, height = image.size
    return {
        "dataset": dataset, "video": video, "video_key": f"{dataset}/{video}",
        "clip_slug": video_slug(dataset, video), "frames_dir": str(frames_dir),
        "gf": {"names": list(gf.names), "native": list(gf.native), "mode": gf.mode, "stride": gf.stride,
               "length": gf.length, "reported_total": gf.reported_total},
        "sampled16": list(gf.sampled),
        "fps_gf": rate.fps, "fps_status": rate.status, "fps_evidence": rate.evidence,
        "t_s": seconds_axis(gf.length, rate.fps),
        "duration_s": None if rate.fps is None else round((gf.length - 1) / rate.fps, 4),
        "source_hw": [height, width], "portrait": height > width,
    }


def materialize_frames(record: Mapping[str, Any], clip_dir: Path) -> Path:
    """Write the GF as ``frames/%06d.jpg``: JPEG sources are linked, others converted."""

    from PIL import Image

    out = clip_dir / "frames"
    out.mkdir(parents=True, exist_ok=True)
    source = Path(record["frames_dir"])
    for index, name in enumerate(record["gf"]["names"]):
        target = out / f"{index:06d}.jpg"
        if target.exists() or target.is_symlink():
            continue
        if Path(name).suffix.lower() in {".jpg", ".jpeg"}:
            os.symlink(source / name, target)
        else:
            with Image.open(source / name) as image:
                image.convert("RGB").save(target, quality=95)
    return out


def open_dynbench(dyn_root: Path, dataset: str, video: str, cache: Path) -> Clip:
    from dyn_dataset_exploration.dyn_bench_data import video_slug

    slug = video_slug(dataset, video)
    clip_dir = Path(cache) / "clips" / slug
    path = clip_dir / "clip.json"
    if path.exists():
        record = json.loads(path.read_text())
    else:
        record = dynbench_record(dyn_root, dataset, video)
        clip_dir.mkdir(parents=True, exist_ok=True)
        _write_json(path, record)
    return Clip(slug, record, materialize_frames(record, clip_dir))


# --------------------------------------------------------------------------- #
# DSI-Bench
# --------------------------------------------------------------------------- #

def open_dsi(video_path: Path, relative_path: str, cache: Path) -> Clip:
    """Decode a DSI-Bench video once into the frame folder every stage reads."""

    from d4rt_agent.dsi_bench_data import video_slug

    slug = "dsi_" + video_slug(relative_path) + "_" + _short_hash(str(video_path))
    clip_dir = Path(cache) / "clips" / slug
    path = clip_dir / "clip.json"
    if path.exists():
        return Clip(slug, json.loads(path.read_text()), clip_dir / "frames")

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(bgr)
    capture.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")
    step = int(np.ceil(len(frames) / MAX_GF_FRAMES))
    kept = frames[::step]
    out = clip_dir / "frames"
    out.mkdir(parents=True, exist_ok=True)
    for index, bgr in enumerate(kept):
        cv2.imwrite(str(out / f"{index:06d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    height, width = kept[0].shape[:2]
    fps_gf = (fps / step) if np.isfinite(fps) and fps > 0 else None
    sampled = np.unique(np.round(np.linspace(0, len(kept) - 1, min(DSI_ANSWER_FRAMES, len(kept))))).astype(int)
    record = {
        "dataset": "dsi", "video": relative_path, "video_key": str(video_path), "clip_slug": slug,
        "frames_dir": str(out),
        "gf": {"length": len(kept), "stride": step, "mode": "decoded", "decoded_total": len(frames)},
        "sampled16": [int(i) for i in sampled],
        "fps_gf": fps_gf, "fps_status": "container" if fps_gf else "unverified",
        "source_hw": [height, width], "portrait": height > width,
        "duration_s": None if fps_gf is None else round((len(kept) - 1) / fps_gf, 4),
    }
    _write_json(path, record)
    return Clip(slug, record, out)


def _short_hash(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode()).hexdigest()[:6]


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)
