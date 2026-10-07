"""Hand-authored Query Programs for the catalog questions.

Written by reading each question and looking at its frames, not by running the rule
parser, so that ``program.diff_programs(parsed, reference)`` separates *parse* errors from
everything downstream. Where a slot needs the frames (the implicit main subject), the
reference fills it the way a person watching the clip would.

Option texts are not repeated here; ``reference_program`` fills them from the catalog.
"""

from __future__ import annotations

import copy
from typing import Mapping

from dyn_dataset_exploration.decomposition.program import (
    Entity,
    Event,
    Measurement,
    OptionSpec,
    QueryProgram,
    Window,
    validate,
)


def _options(**atoms: dict | None) -> list[OptionSpec]:
    return [OptionSpec(letter, "", value, "reference") for letter, value in sorted(atoms.items())]


def _camera() -> Entity:
    return Entity("cam", "camera", role="observer")


REFERENCE: dict[str, QueryProgram] = {
    # The camera pushes toward three students in lab coats: they grow in the frame while
    # the background parallax says the camera moved, not zoomed.
    "cameraqa_davis_lab_coat_6b077a14_0": QueryProgram(
        qid="cameraqa_davis_lab_coat_6b077a14_0", kind="camera_path", route="measure",
        answer_form="camera_motion", source="reference",
        entities=[_camera()],
        measurements=[Measurement("m1", "camera_motion", ["cam"], "dominant_motion", frame="levelled")],
        options=_options(
            A={"motion": "dolly", "direction": "forward"},
            B={"motion": "tilt", "direction": "up"},
            C={"motion": "pan", "direction": "left"},
            D={"motion": "complex", "direction": None},
        ),
    ),
    # Synthetic DynamicReplica render: the camera is static.
    "cameraqa_dynamic_replica_04ff5e_4_obj_source_left_f2cef79b_2": QueryProgram(
        qid="cameraqa_dynamic_replica_04ff5e_4_obj_source_left_f2cef79b_2", kind="camera_shake",
        route="measure", answer_form="shake_level", source="reference",
        entities=[_camera()],
        measurements=[Measurement("m1", "camera_jitter", ["cam"], "level")],
        options=_options(A={"level": 0}, B={"level": 1}, C={"level": 2}, D={"level": 3}),
    ),
    # Two cars on an airfield; the camera follows the grey sedan in the foreground, which
    # is what "the main subject" refers to.
    "cameraqa_dynpose_32439b6a_737d_473b_b7fb_3aead1f6d0c4_069c8138_1": QueryProgram(
        qid="cameraqa_dynpose_32439b6a_737d_473b_b7fb_3aead1f6d0c4_069c8138_1",
        kind="cam_subject_range", route="measure", answer_form="trend", source="reference",
        entities=[_camera(), Entity("e1", "main_subject", "gray sedan", note="the main subject")],
        measurements=[Measurement("m1", "range", ["e1", "cam"], "trend")],
        options=_options(
            A={"trend": "closer"}, B={"trend": "farther"},
            C={"trend": "closer_then_farther"}, D={"trend": "constant"},
        ),
    ),
    # A trainer and a sea lion; "the person" is the trainer. Which axes "right and forward"
    # are measured in is the generator's convention, hence frame="convention".
    "qa_ytvis_1b0ad425e5_b6f634ee_0": QueryProgram(
        qid="qa_ytvis_1b0ad425e5_b6f634ee_0", kind="subject_direction", route="measure",
        answer_form="direction", source="reference",
        entities=[Entity("e1", "object", "person")],
        measurements=[Measurement("m1", "displacement", ["e1"], "direction", frame="convention")],
        options=_options(
            A={"x": "right", "z": "forward"}, B={"x": "left", "z": "backward"},
            C={"x": "right", "y": "up"}, D={"x": "left", "y": "down"},
        ),
    ),
    # A man in white by the wall and a man in a brown coat who runs out of the doorway
    # toward and past the camera.
    "qa_dynamic_replica_04ff5e_4_obj_source_left_f2cef79b_0": QueryProgram(
        qid="qa_dynamic_replica_04ff5e_4_obj_source_left_f2cef79b_0", kind="pair_relation",
        route="measure", answer_form="trend", source="reference",
        entities=[Entity("g1", "group", "person", count=2, role="pair")],
        measurements=[Measurement("m1", "range", ["g1[0]", "g1[1]"], "trend")],
        options=_options(
            A={"trend": "closer_then_farther"}, B={"trend": "farther"},
            C={"trend": "constant"}, D={"trend": "closer_then_constant"},
        ),
    ),
    # Same airfield clip; at 1.0 s (GF 12 at 12 fps) the black car is just behind the sedan.
    "qa_dynpose_32439b6a_737d_473b_b7fb_3aead1f6d0c4_069c8138_0": QueryProgram(
        qid="qa_dynpose_32439b6a_737d_473b_b7fb_3aead1f6d0c4_069c8138_0", kind="metric_distance",
        route="measure", answer_form="metric", source="reference",
        entities=[Entity("g1", "group", "car", count=2, role="pair")],
        window=Window("instant", 1.0),
        measurements=[Measurement("m1", "range", ["g1[0]", "g1[1]"], "value_at", metric=True)],
        options=_options(
            A={"value": 1.6, "unit": "m"}, B={"value": 3.0, "unit": "m"},
            C={"value": 5.0, "unit": "m"}, D={"value": 7.5, "unit": "m"},
        ),
    ),
    # Two children, a boy in blue and a girl in pink, walking on a road (portrait video).
    "qa_sav_017974_manual_818c7cc6_1": QueryProgram(
        qid="qa_sav_017974_manual_818c7cc6_1", kind="instance_count", route="measure",
        answer_form="count", source="reference",
        entities=[Entity("c1", "category", "person", role="counted")],
        measurements=[Measurement("m1", "instances", ["c1"], "count")],
        options=_options(A={"count": 1}, B={"count": 2}, C={"count": 3}, D={"count": 4}),
    ),
    # Portrait garden clip: the camera pans right across the lawn; a person in a dark
    # jacket appears at the right edge only in the last frames.
    "cameraqa_sav_026989_manual_8cbd9e78_2": QueryProgram(
        qid="cameraqa_sav_026989_manual_8cbd9e78_2", kind="event_order", route="measure",
        answer_form="order", source="reference",
        entities=[_camera(), Entity("e2", "object", "person in dark jacket", role="event_actor")],
        events=[
            Event("ev1", "cam", "pan_right", "the camera begins a noticeable rightward pan"),
            Event("ev2", "e2", "enters_frame", "the person in the dark jacket enters the scene"),
        ],
        measurements=[
            Measurement("m1", "onset", ["ev1"], "onset"),
            Measurement("m2", "onset", ["ev2"], "onset"),
            Measurement("m3", "order", ["m1", "m2"], "order"),
        ],
        options=_options(
            A={"first": "ev1"}, B={"first": "ev2"}, C={"first": "tie"},
            D={"first": "ev1", "only": "ev1"},
        ),
    ),
    # A fishing boat crossing in front of a rocky shoreline; the shore is a region, so the
    # measurement is distance to its surface, and the leaf defaults to B.
    "sceneqa_davis_boat_d573aaba_1": QueryProgram(
        qid="sceneqa_davis_boat_d573aaba_1", kind="landmark_relation", route="measure",
        answer_form="trend", source="reference",
        entities=[Entity("e1", "object", "boat"),
                  Entity("l1", "landmark_region", "shoreline", role="reference")],
        measurements=[Measurement("m1", "region_distance", ["e1", "l1"], "trend")],
        options=_options(
            A={"trend": "closer"}, B={"trend": "farther"},
            C={"trend": "constant"}, D={"trend": "closer_then_farther"},
        ),
    ),
    # Dashcam on a desert road behind a car; the last frame is black.
    "qa_sav_000040_manual_b7fd5545_0": QueryProgram(
        qid="qa_sav_000040_manual_b7fd5545_0", kind="semantic", route="vlm_only",
        answer_form="label", source="reference",
        options=_options(A=None, B=None, C=None, D=None),
    ),
}

for _program in REFERENCE.values():
    validate(_program)


def reference_program(qid: str, options: Mapping[str, str] | None = None) -> QueryProgram:
    """A copy of the reference program, with option texts filled in when given."""

    program = copy.deepcopy(REFERENCE[qid])
    if options:
        for spec in program.options:
            spec.text = options.get(spec.letter, "")
    return program
