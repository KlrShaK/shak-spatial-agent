"""The bucket registry: what the agent may choose, and what each choice costs.

One table, read by three different consumers -- the prompt that lists the
choices, the validator that accepts one, and the pipeline that decides which
models to run.  Keeping them on one row is what stops the prompt offering a
bucket the pipeline cannot produce.

The buckets and their names come from `dsi_dataset_exploration`'s taxonomy pass,
which mapped DSI-Bench's six `cate` values onto them with 100% blind-LLM
agreement.  The `cate` mapping is recorded here for scoring only: the agent is
never shown it, because telling a model which category a question belongs to
hands it the reasoning recipe and invalidates the evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from d4rt_agent.sam_orientany_d4rt_test import traj3d_buckets

PROMPTS = Path(__file__).resolve().parent / "prompts"


@dataclass(frozen=True)
class Bucket:
    name: str
    # One line, shown to the agent when it chooses. Describes the QUANTITY and
    # its FRAME, never the wording of any answer option.
    description: str
    needs_subject: bool
    needs_orientation: bool
    builder: Callable[[traj3d_buckets.BucketInputs], dict]
    contract_file: str
    # Which DSI-Bench `cate` values this bucket is expected to serve. Scoring
    # only -- never shown to the agent.
    cates: tuple[int, ...]

    @property
    def contract(self) -> str:
        return (PROMPTS / self.contract_file).read_text(encoding="utf-8")

    @property
    def stages(self) -> tuple[str, ...]:
        """The pipeline stages this bucket actually needs.

        `analyze` always runs, and the camera grid query with it: camera pose is
        what compensates Orient-Anything for camera motion, and the grid is what
        `estimate_intrinsics` needs -- `traj3d_analyze` explains why the 20
        subject points are the wrong input for that.
        """

        stages = ["query_camera", "analyze"]
        if self.needs_subject:
            stages = ["segment", "query_subject", *stages]
        if self.needs_orientation:
            stages.insert(-1, "orient")
        return tuple(stages)


BUCKETS: tuple[Bucket, ...] = (
    Bucket(
        name="traj_subject_own_frame",
        description=(
            "How the SUBJECT moved through the scene, described in the subject's own "
            "starting position and facing -- its own forward/back, left/right, up/down, "
            "plus how much the subject turned. Independent of what the camera did."
        ),
        needs_subject=True,
        needs_orientation=True,
        builder=traj3d_buckets.traj_subject_own_frame,
        contract_file="contract_traj_subject_own_frame.md",
        cates=(0, 1),
    ),
    Bucket(
        name="traj_camera_own_frame",
        description=(
            "How the OBSERVER (the camera itself) moved and turned, relative to where it "
            "started and which way it was facing -- its own forward/back, left/right, "
            "up/down translation, together with how far it panned, tilted and rolled. "
            "Needs no subject."
        ),
        needs_subject=False,
        needs_orientation=False,
        builder=traj3d_buckets.traj_camera_own_frame,
        contract_file="contract_traj_camera_own_frame.md",
        cates=(2, 3),
    ),
    Bucket(
        name="sub_obs_distance_change",
        description=(
            "Only how far apart the subject and the observer were, at the start versus the "
            "end -- a single distance, getting larger or smaller. No directions involved."
        ),
        needs_subject=True,
        needs_orientation=False,
        builder=traj3d_buckets.sub_obs_distance_change,
        contract_file="contract_sub_obs_distance_change.md",
        cates=(4,),
    ),
    Bucket(
        name="orient_camera_subject_frame",
        description=(
            "Which SIDE OF THE SUBJECT the observer was on -- the subject's own front, "
            "back, left or right -- and how that changed over the clip. Answers shaped "
            "like 'from the subject's X to its Y'."
        ),
        needs_subject=True,
        needs_orientation=True,
        builder=traj3d_buckets.orient_camera_subject_frame,
        contract_file="contract_orient_camera_subject_frame.md",
        cates=(5,),
    ),
)

BY_NAME = {bucket.name: bucket for bucket in BUCKETS}
NAMES = tuple(bucket.name for bucket in BUCKETS)

# cate -> the bucket the taxonomy pass expects for it. Scoring only.
CATE_TO_BUCKET = {cate: bucket.name for bucket in BUCKETS for cate in bucket.cates}


def get(name: str) -> Bucket:
    if name not in BY_NAME:
        raise KeyError(f"unknown bucket {name!r}; expected one of {list(NAMES)}")
    return BY_NAME[name]


def choices_block() -> str:
    """The buckets as the agent sees them when choosing."""

    return "\n".join(
        f"{index}. `{bucket.name}` — {bucket.description}"
        for index, bucket in enumerate(BUCKETS, start=1)
    )
