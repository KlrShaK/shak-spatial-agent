"""Shared configuration and small helpers for the traj3d pipeline.

The pipeline turns a text-named subject into a 3D motion trajectory by chaining
three models that know nothing about each other:

    SAM3            text phrase -> per-frame masks of one subject
    D4RT            mask points -> 3D positions through the clip
    OrientAnythingV2  masked crop -> the subject's facing direction

Every stage writes its intermediates to disk so it can be audited on its own.
That matters because the failure modes are silent: SAM3 can segment the wrong
person, D4RT can latch a query point onto the background behind a limb, and
OrientAnything can be confidently wrong about a coordinate convention.  A
pipeline that only emitted the final plot would hide all three.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# .../Open-d4rt/d4rt_agent/sam_orientany_d4rt_test/traj3d_common.py -> Open-d4rt
REPO_ROOT = Path(__file__).resolve().parents[2]

# The same root the DSI-Bench runner uses, so a traj3d run and a benchmark run
# refer to byte-identical video files.
DSI_VIDEO_ROOT = Path("/cluster/work/igp_psr/spanwar/datasets/DSI-Bench/videos/std")

THIRD_PARTY = REPO_ROOT / "third-party-tools"
SAM3_DIR = THIRD_PARTY / "sam3"
ORIENT_DIR = THIRD_PARTY / "Orient-Anything-V2"

D4RT_CONFIG = REPO_ROOT / "checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml"
D4RT_CHECKPOINT = REPO_ROOT / "checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt"

ORIENT_HF_REPO = "Viglong/OriAnyV2_ckpt"
ORIENT_HF_FILE = "demo_ckpts/rotmod_realrotaug_best.pt"

RESULTS_ROOT = REPO_ROOT / "d4rt_agent/results"

# Number of subject points tracked through the clip.  D4RT batches all points
# into one call, so this is not a cost knob; it is a robustness knob for the
# track-level outlier rejection in traj3d_geometry.
NUM_SUBJECT_POINTS = 20

# Square grid used to recover camera motion.  A plain uniform grid is correct
# here -- see recover_camera_poses() for why the points need not be static or
# even off-subject.
CAMERA_GRID_SIZE = 12


@dataclass(frozen=True)
class Target:
    """One (video, subject phrase) pair to process."""

    relative_path: str
    subject: str

    @property
    def video_path(self) -> Path:
        return DSI_VIDEO_ROOT / self.relative_path

    @property
    def slug(self) -> str:
        # Reused from the benchmark so traj3d output lines up with the
        # DSI-Bench answers that already exist for these same three clips.
        from d4rt_agent.dsi_bench_data import video_slug

        return video_slug(self.relative_path)


TARGETS: tuple[Target, ...] = (
    Target("CameraBench/M0jmSsQ5ptw.3.12.mp4", "the runner"),
    Target(
        "CameraBench/99f927656086265ccda299e2c2ec394f4bfded4daa3ea64efcb4ee6c95b8c7de.0.mp4",
        "the person in the yellow jacket",
    ),
    Target("SynFMC/Rendered_Traj_Results/dynamic/6/video_480p.mp4", "the character in pink"),
)


@dataclass
class VideoPaths:
    """Every path one video's outputs live at, created on construction."""

    root: Path
    frames: Path = field(init=False)
    masks: Path = field(init=False)
    masks_overlay: Path = field(init=False)
    d4rt: Path = field(init=False)
    orientation: Path = field(init=False)
    analysis: Path = field(init=False)
    plots: Path = field(init=False)

    def __post_init__(self) -> None:
        self.frames = self.root / "frames"
        self.masks = self.root / "masks"
        self.masks_overlay = self.root / "masks_overlay"
        self.d4rt = self.root / "d4rt"
        self.orientation = self.root / "orientation"
        self.analysis = self.root / "analysis"
        self.plots = self.root / "plots"

    def mkdirs(self) -> None:
        for path in (
            self.root,
            self.frames,
            self.masks,
            self.masks_overlay,
            self.d4rt,
            self.orientation,
            self.analysis,
            self.plots,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def segmentation_json(self) -> Path:
        return self.root / "segmentation.json"

    @property
    def points_json(self) -> Path:
        return self.root / "points.json"

    @property
    def candidates_jpg(self) -> Path:
        return self.root / "candidates.jpg"


def default_run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return RESULTS_ROOT / f"traj3d_{stamp}"


def video_paths(run_dir: Path, target: Target) -> VideoPaths:
    return VideoPaths(root=Path(run_dir) / target.slug)


def write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically.

    A traj3d job can be killed by the SLURM time limit at any moment, and the
    runner decides what to skip by looking for these files.  A half-written
    file would make a video look complete when it is not.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    os.replace(tmp, path)


def _json_default(value: Any) -> Any:
    import numpy as np

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def git_sha() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def run_metadata(run_dir: Path) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "git_sha": git_sha(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.uname().nodename,
        "num_subject_points": NUM_SUBJECT_POINTS,
        "camera_grid_size": CAMERA_GRID_SIZE,
        "d4rt_checkpoint": str(D4RT_CHECKPOINT),
        "targets": [
            {"relative_path": t.relative_path, "subject": t.subject, "slug": t.slug}
            for t in TARGETS
        ],
    }
