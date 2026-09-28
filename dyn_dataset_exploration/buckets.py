"""Axis B: the measurement buckets a Dyn-Bench question might be asking for.

Same question ``dsi_dataset_exploration`` asked -- *what physical quantity, in
what reference frame, does this question actually measure?* -- with the answer
space widened to fit Dyn-Bench.

The first four buckets are the ones ``d4rt_agent/agent_v4/buckets.py`` already
implements, named identically so the coverage verdict in the report is a lookup
rather than a judgement call. The rest are **candidates**: quantities the
deterministic family scan in ``dyn_bench_data.FAMILIES`` found evidence for, but
whose status as a real, separable measurement is exactly what the blind LLM pass
is run to decide. A candidate that no sampled question lands in is a candidate
AgentV5 does not have to build.

Four of the candidates were not in the original design. Two came from the
deterministic family scan, which surfaced them with counts too large to ignore:
``subject_motion_profile`` (300 rows ask whether a subject's motion is
continuous, intermittent or absent) and ``subject_orient_over_time`` (35 rows ask
which way a scene object faces).

The other two came from reading the first classification pass's ``OTHER`` pile,
which was not noise -- it was one missing distinction, stated over and over. Half
of it read like *"how does the kitesurfer's position relative to the camera
change?"*, and the model's own rationales said so: "asks for the object's
displacement in the camera frame, which is not represented by any single listed
bucket". That is ``traj_subject_camera_frame``. Most of the rest read like *"the
truck's motion relative to the road's curvature"* -- a trajectory in a frame
fixed to the static scene, which is ``traj_subject_scene_frame``. Both were added
and the whole sample was reclassified under the extended list, so the counts in
the report come from one codebook rather than two.

Three of these four -- ``traj_subject_camera_frame``, ``subject_orient_over_time``
and (as the camera-side mirror) ``camera_elevation_rel_subject`` -- correspond to
buckets the DSI-Bench pass defined and then found **absent** from DSI-Bench. They
are present here. That asymmetry is the shortest statement of why AgentV4's four
buckets do not carry over.

``semantic_only`` is deliberate and load-bearing. AgentV5 needs a way to decide
not to run the pipeline at all; without this bucket the classifier would be
forced to invent a geometric reading of "What best describes the main activity
in the video?", and the report would overstate how much of Dyn-Bench is a
measurement problem.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Bucket:
    """One measurement the agent could be asked to make.

    ``v4_builder`` names the existing ``traj3d_buckets`` function that already
    produces this measurement, or ``None`` when nothing does. It is the whole
    point of the exercise: the report's "needs new work" column is this field.
    """

    name: str
    description: str
    v4_builder: str | None = None

    @property
    def covered_by_v4(self) -> bool:
        return self.v4_builder is not None


BUCKETS: tuple[Bucket, ...] = (
    # --- the four AgentV4 already measures ------------------------------- #
    Bucket(
        "traj_subject_own_frame",
        "Trajectory of a scene object, expressed in that object's own starting frame "
        "(its own forward/lateral/vertical axes)",
        "traj3d_buckets.traj_subject_own_frame",
    ),
    Bucket(
        "traj_camera_own_frame",
        "Trajectory of the camera, expressed in the camera's own starting frame "
        "(forward/lateral/vertical translation plus pan/tilt/roll)",
        "traj3d_buckets.traj_camera_own_frame",
    ),
    Bucket(
        "sub_obs_distance_change",
        "Whether the range between a scene object and the camera grew, shrank or held, "
        "as an ordinal judgement rather than a number",
        "traj3d_buckets.sub_obs_distance_change",
    ),
    Bucket(
        "orient_camera_subject_frame",
        "Which side of a scene object the camera is on (front/back/left/right sector of "
        "the object's own facing), and how that sector changes",
        "traj3d_buckets.orient_camera_subject_frame",
    ),
    # --- candidates ------------------------------------------------------ #
    Bucket(
        "traj_subject_camera_frame",
        "Trajectory of a scene object expressed in the CAMERA's frame -- nearer/farther, "
        "left/right, up/down as the camera sees it, not in the object's own axes",
    ),
    Bucket(
        "traj_subject_scene_frame",
        "Trajectory of a scene object relative to the static scene around it -- a fixed "
        "landmark, a road, a court, the room -- rather than its own axes or the camera's",
    ),
    Bucket(
        "subject_subject_relation",
        "Distance or bearing between TWO scene objects, and how it changes -- neither of "
        "them being the camera",
    ),
    Bucket(
        "metric_distance",
        "Absolute distance in real-world units (metres) between two entities at a named "
        "moment -- a number, not an ordinal judgement",
    ),
    Bucket(
        "metric_speed",
        "Absolute speed in real-world units (m/s) of a scene object over a named interval",
    ),
    Bucket(
        "instance_count",
        "How many distinct instances of a category are present in the clip",
    ),
    Bucket(
        "event_order",
        "The onset times of two named events, compared -- which happened first",
    ),
    Bucket(
        "camera_zoom_vs_dolly",
        "Whether the camera's focal length changed, separated from whether the camera "
        "translated toward or away from the scene",
    ),
    Bucket(
        "camera_motion_profile",
        "The camera's speed as a function of time -- steady vs. accelerating vs. "
        "stop-start, and high-frequency jitter (handheld shake)",
    ),
    Bucket(
        "subject_motion_profile",
        "A scene object's speed as a function of time -- whether it moves at all, and "
        "whether that motion is continuous, intermittent or a single brief burst",
    ),
    Bucket(
        "subject_orient_over_time",
        "Change in which way a scene object faces, over time, independent of where it is "
        "and where the camera is",
    ),
    Bucket(
        "subject_visibility",
        "Whether a scene object is inside the image frame, where in the frame it sits, "
        "and at which times it enters or leaves",
    ),
    Bucket(
        "apparent_size_change",
        "The object's projected size in the image plane over time -- an image-plane "
        "quantity, which range and focal length both feed into",
    ),
    Bucket(
        "camera_elevation_rel_subject",
        "The camera's elevation relative to the subject -- shot from above, at eye level, "
        "or from below",
    ),
    Bucket(
        "scene_revisit",
        "Whether the view at the end of the clip shows the same place as the view at the "
        "start",
    ),
    Bucket(
        "semantic_only",
        "No geometric measurement decides it -- the question is about what is happening, "
        "what something is, or how a scene reads",
    ),
)
OTHER_BUCKET = "OTHER"
BUCKET_BY_NAME: dict[str, Bucket] = {bucket.name: bucket for bucket in BUCKETS}
BUCKET_NAMES: tuple[str, ...] = tuple(bucket.name for bucket in BUCKETS)
BUCKET_KEYS: tuple[str, ...] = (*BUCKET_NAMES, OTHER_BUCKET)
V4_BUCKETS: tuple[str, ...] = tuple(b.name for b in BUCKETS if b.covered_by_v4)


def bucket_glossary() -> str:
    """The bucket list as the classifier sees it -- names and descriptions only.

    Deliberately omits ``v4_builder``: telling the classifier which buckets we
    already implement would bias it toward them, and the whole value of the pass
    is an unbiased count of what is actually there.
    """

    lines = [f"- {bucket.name}: {bucket.description}" for bucket in BUCKETS]
    lines.append(f"- {OTHER_BUCKET}: none of the above fit")
    return "\n".join(lines)
