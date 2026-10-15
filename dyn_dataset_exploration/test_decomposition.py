"""CPU tests for the decomposition package (M1: frames, time axis, programs, parsers).

What is under test is what would silently produce a wrong answer rather than raise: a
frame index that points at the wrong file, a timestamp mapped with the wrong rate, an
option parsed into the opposite claim, a program that references something undefined.
Tests marked ``data`` need the Dyn-Bench tree and pyarrow, and skip without them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from dyn_dataset_exploration.decomposition import program as prog
from dyn_dataset_exploration.decomposition.program import (
    Entity,
    Event,
    Measurement,
    QueryProgram,
    camera_motion_atoms,
    count_atoms,
    diff_programs,
    direction_atoms,
    duplicate_options,
    metric_atoms,
    order_atoms,
    rule_parse,
    sam3_phrase,
    shake_atoms,
    signature,
    timestamps,
    trend_atoms,
    validate,
    window_from_question,
)
from dyn_dataset_exploration.decomposition.reference_programs import REFERENCE
from dyn_dataset_exploration.decomposition.scene4d import (
    FPS_GF,
    generator_frames,
    list_frames,
    seconds_axis,
    seconds_to_index,
)

DYN_ROOT = Path("/cluster/work/igp_psr/spanwar/datasets/Dyn-Bench")
needs_data = pytest.mark.skipif(not (DYN_ROOT / "multi_json").is_dir(), reason="Dyn-Bench not on disk")


def _sampling(names_by_index: dict[int, str], total: int) -> dict:
    return {
        "total_frames": total,
        "frame_indices": sorted(names_by_index),
        "frame_paths": [f"/elsewhere/{names_by_index[i]}" for i in sorted(names_by_index)],
    }


# --------------------------------------------------------------------------- #
# Generator frames and seconds
# --------------------------------------------------------------------------- #

def test_generator_frames_identity():
    names = [f"frame_{i:04d}.jpg" for i in range(10)]
    gf = generator_frames(names, _sampling({i: names[i] for i in (0, 3, 6, 9)}, 10))
    assert gf.mode == "identity" and gf.length == 10 and gf.native[9] == 9


def test_generator_frames_stride_matches_dynamic_replica_layout():
    names = [f"clip-{i:04d}.png" for i in range(300)]
    indices = np.floor(np.linspace(0, 59, 16)).astype(int)
    gf = generator_frames(names, _sampling({int(i): names[5 * int(i)] for i in indices}, 60))
    assert gf.mode == "stride" and gf.stride == 5 and gf.length == 60
    assert gf.names[59] == "clip-0295.png"


def test_generator_frames_length_is_the_sampled_span_not_total_frames():
    # davis_bmx-bumps ships total_frames=69 while its samples run to index 89 of 90 files.
    names = [f"frame_{i:04d}.jpg" for i in range(90)]
    gf = generator_frames(names, _sampling({0: names[0], 45: names[45], 89: names[89]}, 69))
    assert gf.length == 90 and gf.reported_total == 69


def test_generator_frames_rejects_a_sample_it_cannot_reproduce():
    names = [f"frame_{i:04d}.jpg" for i in range(10)]
    with pytest.raises(ValueError):
        generator_frames(names, _sampling({0: names[0], 9: "frame_0042.jpg"}, 10))


def test_list_frames_sorts_by_number_not_string(tmp_path):
    for name in ("f_10.jpg", "f_9.jpg", "f_100.png", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    assert list_frames(tmp_path) == ["f_9.jpg", "f_10.jpg", "f_100.png"]


def test_seconds_axis_and_index():
    assert seconds_axis(5, None) is None
    assert seconds_axis(3, 6.0) == [0.0, round(1 / 6, 6), round(2 / 6, 6)]
    assert seconds_to_index(1.0, 12.0, 51) == 12
    assert seconds_to_index(99.0, 6.0, 30) == 29
    assert seconds_to_index(1.0, None, 30) is None


def test_every_unverified_source_has_no_rate():
    for dataset, rate in FPS_GF.items():
        assert (rate.fps is None) == (rate.status == "unverified"), dataset


# --------------------------------------------------------------------------- #
# Surface facts
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text, expected", [
    ("At 5.83 seconds, what is the approximate distance?", [5.83]),
    ("At the 5-second mark, what is the distance?", [5.0]),
    ("Between t=2.0s and t=3.0s, what is the camera's dominant translation direction?", [2.0, 3.0]),
    ("Between the start (0.0s) and the end (1.63s), how does it change?", [0.0, 1.63]),
    ("during the first 25% of the video (approx. 0-3s)?", [0.0, 3.0]),
    ("From the beginning (t≈0s) to the end (t≈11s), does it change?", [0.0, 11.0]),
    ("At approximately 10 seconds into the video, what is the distance?", [10.0]),
    ("What is the approximate average speed? A. 0.5 m/s", []),
])
def test_timestamps(text, expected):
    assert timestamps(text) == expected


def test_windows_from_questions():
    assert window_from_question("At the 1.0-second mark, what is the distance?").kind == "instant"
    w = window_from_question("From the first frame to the frame at 4.67 seconds, how does it move?")
    assert (w.kind, w.t0, w.t1) == ("interval", 0.0, 4.67)
    w = window_from_question("Is the camera stationary during the first 25% of the video (approx. 0-3s)?")
    assert (w.kind, w.phase, w.t0, w.t1) == ("phase", "first_25", 0.0, 3.0)
    assert window_from_question("From the start to the middle of the video, how does it move?").phase \
        == "start_to_middle"
    w = window_from_question("From the first frame to the frame where the gate is fully open, how?")
    assert w.kind == "event" and w.event.startswith("the gate is fully open")
    assert window_from_question("How many distinct people are visible?").kind == "whole"


def test_duplicate_options_are_grouped():
    assert duplicate_options({"A": "Moves right and forward", "B": "Moves left",
                              "C": "moves right and forward."}) == [["A", "C"]]


def test_sam3_phrase_drops_articles_and_caps_length():
    assert sam3_phrase("the person in the dark jacket") == (
        "person in dark jacket", "the person in the dark jacket")
    phrase, note = sam3_phrase("the jockey wearing a red helmet and riding a dark brown horse")
    assert len(phrase.split()) == 5 and note is not None
    assert sam3_phrase("boat") == ("boat", None)


# --------------------------------------------------------------------------- #
# Option parsers
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text, expected", [
    ("Dollies forward", {"motion": "dolly", "direction": "forward"}),
    ("Pans from right to left", {"motion": "pan", "direction": "left"}),
    ("Tilts up", {"motion": "tilt", "direction": "up"}),
    ("Complex movements including panning left, tilting down", {"motion": "complex", "direction": None}),
    ("Remains fixed with slight shaking", {"motion": "static", "direction": None, "shake": True}),
    ("No noticeable movement", {"motion": "static", "direction": None, "shake": False}),
    ("Horizontal: Panning or scanning left/right across the scene (Pan).", {"motion": "pan", "direction": None}),
    ("Vertical: Moving up or down (Tilt/Pedestal).", {"motion": "vertical", "direction": None}),
    ("Depth: Distinctly pushing forward or pulling back (Dolly).", {"motion": "dolly", "direction": None}),
    ("Arcs counterclockwise", {"motion": "orbit", "direction": "ccw"}),
    ("Moves leftward", {"motion": "truck", "direction": "left"}),
])
def test_camera_motion_atoms(text, expected):
    assert camera_motion_atoms(text) == expected


def test_shake_atoms_read_the_leading_label():
    assert [shake_atoms(t)["level"] for t in (
        "Stable: Almost no shake", "Slight (Breathing): Natural slight handheld sway",
        "Shaky: Noticeable bumping", "Severe: Violent shaking")] == [0, 1, 2, 3]


@pytest.mark.parametrize("text, trend", [
    ("The subject gradually grows larger in the frame, with increasing clarity of details.", "closer"),
    ("The subject gradually becomes smaller in the picture and blends into the background.", "farther"),
    ("The distance decreases first, and then increases", "closer_then_farther"),
    ("The distance increases first, and then decreases", "farther_then_closer"),
    ("The distance remains stable.", "constant"),
    ("They get closer, then stop", "closer_then_constant"),
    ("They get farther apart", "farther"),
    ("They stay about the same distance apart", "constant"),
    ("They move in and out of proximity repeatedly", "fluctuating"),
    ("It moves slightly closer to the shore, then drifts back out.", "closer_then_farther"),
    ("It maintains a consistent distance from the shore.", "constant"),
    ("They briefly meet then separate", "closer_then_farther"),
    ("Fluctuating: Distance is unstable, moving back and forth.", "fluctuating"),
])
def test_trend_atoms(text, trend):
    assert trend_atoms(text) == {"trend": trend}


def test_trend_atoms_mark_unverifiable_claims():
    assert trend_atoms("Impossible to determine") == {"undeterminable": True}
    assert trend_atoms("They move in circles around each other") == {"circling": True}


@pytest.mark.parametrize("text, expected", [
    ("Moves right and forward", {"x": "right", "z": "forward"}),
    ("It moves left and backward", {"x": "left", "z": "backward"}),
    ("Moves right and slightly backward", {"x": "right", "z": "backward", "slight": ["z"]}),
    ("It moves downward, and forward", {"y": "down", "z": "forward"}),
    ("It remains stationary.", {"stationary": True}),
    ("It moves straight toward the camera", {"trend": "closer"}),
    ("It moves sideways across the frame without changing distance.", {"lateral": True}),
])
def test_direction_atoms(text, expected):
    assert direction_atoms(text) == expected


def test_metric_and_count_atoms():
    assert metric_atoms("About 1.2 meters") == {"value": 1.2, "unit": "m"}
    assert metric_atoms("0.5 m/s") == {"value": 0.5, "unit": "m/s"}
    assert metric_atoms("50 cm") == {"value": 0.5, "unit": "m"}
    assert metric_atoms("Two") is None
    assert count_atoms("Three") == {"count": 3} and count_atoms("5") == {"count": 5}


def test_order_atoms_resolve_each_option_to_an_event():
    events = [Event("ev1", "cam", "pan_right", "the camera begins a noticeable rightward pan"),
              Event("ev2", "e2", "enters_frame", "the person in the dark jacket enters the scene")]
    assert order_atoms("Camera begins rightward pan first", events) == {"first": "ev1"}
    assert order_atoms("Person enters scene first", events) == {"first": "ev2"}
    assert order_atoms("Both occur in the same frame", events) == {"first": "tie"}
    assert order_atoms("Only the camera motion occurs; person never enters", events) == \
        {"first": "ev1", "only": "ev1"}
    assert order_atoms("Only one of these events occurs", events) == {"first": "one_only"}
    assert order_atoms("The person enters before the camera pans", events) == {"first": "ev2"}


# --------------------------------------------------------------------------- #
# Programs
# --------------------------------------------------------------------------- #

def test_validate_catches_undefined_and_malformed_parts():
    bad = QueryProgram(
        qid="x", kind="pair_relation", route="measure", answer_form="trend",
        entities=[Entity("g1", "group", "person", count=1),
                  Entity("e1", "object", "a very long phrase with many words")],
        measurements=[Measurement("m1", "range", ["g1[0]", "e9"], "trend")],
    )
    with pytest.raises(ValueError) as error:
        validate(bad)
    message = str(error.value)
    assert "count >= 2" in message and "longer than 5 words" in message and "'e9'" in message


def test_validate_rejects_an_unknown_kind_and_a_measured_semantic_question():
    with pytest.raises(ValueError):
        validate(QueryProgram(qid="x", kind="nope", route="measure", answer_form="label"))
    with pytest.raises(ValueError):
        validate(QueryProgram(qid="x", kind="semantic", route="vlm_only", answer_form="label",
                              measurements=[Measurement("m1", "range", ["camera"], "trend")]))


def test_reference_programs_round_trip_and_are_first_iteration():
    for qid, program in REFERENCE.items():
        again = QueryProgram.from_dict(json.loads(json.dumps(program.to_dict())))
        assert signature(again) == signature(program), qid
        assert prog.KIND_BY_NAME[program.kind].iteration == 1, qid


def test_validate_slots_rejects_what_a_model_gets_wrong():
    with pytest.raises(ValueError, match="required"):
        prog.validate_slots("subject_direction", {})
    with pytest.raises(ValueError, match="unexpected slots"):
        prog.validate_slots("instance_count", {"category": "person", "colour": "red"})
    with pytest.raises(ValueError, match="predicate"):
        prog.validate_slots("event_order", {"events": [
            {"actor": "camera", "predicate": "swoops"}, {"actor": "dog", "predicate": "enters_frame"}]})
    with pytest.raises(ValueError, match="exactly two"):
        prog.validate_slots("event_order", {"events": []})
    prog.validate_slots("cam_subject_range", {"subject": None, "implicit": True})


def test_model_slots_assemble_into_the_reference_program():
    """What Qwen is asked for is enough to rebuild the reference program exactly."""

    options = {"A": "Camera begins rightward pan first", "B": "Person enters scene first",
               "C": "Both occur in the same frame",
               "D": "Only the camera motion occurs; person never enters"}
    question = ("Which happens first: the camera begins a noticeable rightward pan, or the "
                "person in the dark jacket enters the scene?")
    qid = "cameraqa_sav_026989_manual_8cbd9e78_2"
    slots = {"events": [
        {"actor": "camera", "predicate": "pan_right", "text": "the camera begins a noticeable rightward pan"},
        {"actor": "the person in the dark jacket", "predicate": "enters_frame",
         "text": "the person in the dark jacket enters the scene"}]}
    program = prog.assemble(qid, "event_order", slots, question, options, source="qwen")
    assert diff_programs(program, REFERENCE[qid]) == []

    qid = "qa_dynamic_replica_04ff5e_4_obj_source_left_f2cef79b_0"
    options = {"A": "They get closer, then farther", "B": "They get farther apart",
               "C": "They stay about the same distance", "D": "They get closer, then stop"}
    program = prog.assemble(qid, "pair_relation", {"category": "people"},
                            "Do the two people get closer or farther apart during the video?",
                            options, source="qwen")
    assert diff_programs(program, REFERENCE[qid]) == []


def test_host_corrects_what_qwen_got_wrong_on_the_catalog():
    # Qwen omitted the event texts; option D must still resolve to the camera event.
    options = {"A": "Camera begins rightward pan first", "B": "Person enters scene first",
               "C": "Both occur in the same frame",
               "D": "Only the camera motion occurs; person never enters"}
    program = prog.assemble(
        "q", "event_order",
        {"events": [{"actor": "camera", "predicate": "pan_right"},
                    {"actor": "person in dark jacket", "predicate": "enters_frame"}]},
        "Which happens first: the camera begins a noticeable rightward pan, or the person in "
        "the dark jacket enters the scene?", options, source="qwen")
    assert {o.letter: o.atoms for o in program.options}["D"] == {"first": "ev1", "only": "ev1"}

    # Qwen named the main subject but called it explicit; the question says it is implicit.
    program = prog.assemble("q", "cam_subject_range", {"subject": "gray car", "implicit": False},
                            "How does the distance between the camera and the main subject change?",
                            {"A": "The distance remains stable."}, source="qwen")
    assert program.entities[1].kind == "main_subject"

    # Qwen counted "child" for a question about "people".
    with pytest.raises(ValueError, match="not a word from the question"):
        prog.assemble("q", "instance_count", {"category": "child"},
                      "How many distinct people are visible in the video?", {"A": "One"}, source="qwen")
    prog.assemble("q", "instance_count", {"category": "people"},
                  "How many distinct people are visible in the video?", {"A": "One"}, source="qwen")


def test_kinds_block_lists_every_kind_once():
    block = prog.kinds_block()
    assert all(block.count(f"`{kind.name}`") == 1 for kind in prog.KINDS)


# --------------------------------------------------------------------------- #
# Real data
# --------------------------------------------------------------------------- #

def _rows():
    pytest.importorskip("pyarrow")
    from dyn_dataset_exploration.dyn_bench_data import load_rows, qa_parquet
    return load_rows(qa_parquet(DYN_ROOT))


def _sampled_span() -> dict[str, int]:
    spans = {}
    for dataset in sorted(os.listdir(DYN_ROOT / "multi_json")):
        for name in os.listdir(DYN_ROOT / "multi_json" / dataset):
            if name.endswith("_frame_sampling.json"):
                payload = json.loads((DYN_ROOT / "multi_json" / dataset / name).read_text())
                video = name[len(dataset) + 1: -len("_frame_sampling.json")]
                spans[f"{dataset}/{video}"] = max(payload["frame_indices"]) + 1
    return spans


@needs_data
def test_generator_frames_rebuild_every_clip():
    failures = []
    for dataset in sorted(os.listdir(DYN_ROOT / "multi_json")):
        for name in sorted(os.listdir(DYN_ROOT / "multi_json" / dataset)):
            if not name.endswith("_frame_sampling.json"):
                continue
            clip = name[: -len("_frame_sampling.json")]
            sampling = json.loads((DYN_ROOT / "multi_json" / dataset / name).read_text())
            try:
                generator_frames(list_frames(DYN_ROOT / "frames" / dataset / clip), sampling)
            except ValueError as error:
                failures.append((clip, str(error)))
    assert not failures, failures[:5]


@needs_data
def test_no_timestamp_falls_past_its_clip_under_the_rate_table():
    spans = _sampled_span()
    late = []
    for row in _rows():
        rate = FPS_GF[row.dataset]
        if rate.fps is None:
            continue
        end = (spans[row.video_key] - 1) / rate.fps
        late += [(row.question_id, t, end) for t in timestamps(row.question) if t > end + 0.25]
    assert not late, late[:5]


@needs_data
def test_catalog_rule_parses_agree_with_the_references():
    from dyn_dataset_exploration.decomposition.build_catalog import CATALOG

    rows = {row.question_id: row for row in _rows()}
    for qids in CATALOG.values():
        for qid in qids:
            row = rows[qid]
            parsed = rule_parse(qid, row.question, row.options)
            assert parsed is not None, qid
            wrong = [line for line in diff_programs(parsed, REFERENCE[qid]) if "open slot" not in line]
            assert not wrong, (qid, wrong)


@needs_data
def test_templated_kinds_parse_every_option_and_every_answer():
    stats = {1: [0, 0, 0, 0], 2: [0, 0, 0, 0]}          # per iteration: rows, options, unread options, unread answers
    for row in _rows():
        program = rule_parse(row.question_id, row.question, row.options)
        if program is None or program.route != "measure" or program.kind == "landmark_relation":
            continue
        s = stats[prog.KIND_BY_NAME[program.kind].iteration]
        s[0] += 1
        s[1] += len(program.options)
        s[2] += sum(o.atoms is None for o in program.options)
        s[3] += next(o for o in program.options if o.letter == row.answer).atoms is None
    rows1, options1, unread1, answers1 = stats[1]
    assert rows1 > 1500 and unread1 / options1 < 0.01 and answers1 / rows1 < 0.01
    rows2, options2, unread2, answers2 = stats[2]
    assert rows2 > 1000 and answers2 / rows2 < 0.06     # zoom/motion narratives are left to Qwen


# --------------------------------------------------------------------------- #
# event sentences: the host's reading outranks the model's where they disagree
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text, actor, predicate", [
    ("the camera begins to pan slightly", "camera", "pan_any"),
    ("the camera begins a noticeable rightward pan", "camera", "pan_right"),
    ("the camera begins its dominant leftward translation", "camera", "starts_moving"),
    ("the white tiger starts walking toward the camera", "white tiger", "starts_moving"),
    ("the second camel becomes visible", "second camel", "becomes_visible"),
    ("the large window view becomes fully visible in the background", "large window view", "becomes_visible"),
    ("the car on the left begins to accelerate", "car on the left", "starts_moving"),
    ("the camera focus starts to blur", "camera focus", "other"),
    ("the child chases the ball", None, "other"),
])
def test_event_slot_reads_the_sentence(text, actor, predicate):
    slot = prog.event_slot(text)
    assert slot["predicate"] == predicate
    if actor is not None:
        assert slot["actor"] == actor


def test_normalise_event_slots_moves_a_mislabelled_actor_off_the_camera():
    fixed = prog.normalise_event_slots([
        {"actor": "camera", "predicate": "other", "text": "the white tiger starts walking toward the camera"},
        {"actor": "camera", "predicate": "pan_left", "text": "the camera pans left"},
        {"actor": "e2", "predicate": "other", "text": "the dog starts running"},
    ])
    assert (fixed[0]["actor"], fixed[0]["predicate"]) == ("white tiger", "starts_moving")
    assert (fixed[1]["actor"], fixed[1]["predicate"]) == ("camera", "pan_left")
    assert fixed[2]["predicate"] == "starts_moving"


def test_upgrade_events_keeps_the_actor_but_sharpens_other():
    program = prog.QueryProgram.from_dict({
        "qid": "q", "kind": "event_order", "route": "measure", "answer_form": "event_order",
        "entities": [], "window": {"kind": "whole"}, "measurements": [], "options": [],
        "events": [{"id": "ev1", "actor": "e1", "predicate": "other", "text": "the car on the left begins to accelerate"},
                   {"id": "ev2", "actor": "e2", "predicate": "other", "text": "the person sitting down"}]})
    prog.upgrade_events(program)
    assert [e.predicate for e in program.events] == ["starts_moving", "other"]


# --------------------------------------------------------------------------- #
# determinism: the same artifacts and constants always give the same answers
# --------------------------------------------------------------------------- #

SMALL_CATALOG = Path(__file__).resolve().parent / "outputs" / "decomposition"


@pytest.mark.skipif(not (SMALL_CATALOG / "catalog.json").exists(), reason="catalog artifacts not on disk")
def test_the_pipeline_is_deterministic_on_the_small_catalog():
    from dyn_dataset_exploration.decomposition import pipeline

    first = [s.to_dict() for s in pipeline.solve_all(pipeline.Context(SMALL_CATALOG))]
    second = [s.to_dict() for s in pipeline.solve_all(pipeline.Context(SMALL_CATALOG))]
    assert json.dumps(first, sort_keys=True, default=str) == json.dumps(second, sort_keys=True, default=str)
    assert first, "the catalog was empty"


# --------------------------------------------------------------------------- #
# S2 fallback: a simpler prompt when the segmenter finds nothing for the question's own words
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("phrase, ladder", [
    ("camel walking near fence", ["camel walking near fence", "camel"]),
    ("brown horse with rider", ["brown horse with rider", "brown horse", "horse"]),
    ("person in pink suit", ["person in pink suit", "person"]),
    ("large building", ["large building", "building"]),          # "building" is a noun, not a participle
    ("second skier", ["second skier", "skier"]),
    ("camel", ["camel"]),
    ("it", ["it"]),                                              # a pronoun names nothing to segment
])
def test_phrase_ladder(phrase, ladder):
    assert prog.phrase_ladder(phrase) == ladder


def test_context_grounding_climbs_the_ladder_and_says_so(tmp_path):
    from dyn_dataset_exploration.decomposition import pipeline
    from dyn_dataset_exploration.decomposition.scene4d import Grounding

    def make(phrase, k):
        z = np.zeros
        return Grounding(phrase, np.arange(k), z((k, 2, 4, 4), bool), z((k, 2)), z((k, 2, 4)), z((k, 2)), z((k, 2), bool))

    ctx = pipeline.Context.__new__(pipeline.Context)
    ctx.out, ctx._groundings = tmp_path, {}
    shelf = {"camel walking near fence": make("camel walking near fence", 0), "camel": make("camel", 2)}
    original = pipeline.load_grounding
    pipeline.load_grounding = lambda out, slug, phrase: shelf.get(phrase)
    try:
        g = ctx.grounding("clip", "camel walking near fence")
        assert g.count == 2 and g.used_phrase == "camel"
        assert ctx.grounding("clip", "dog").used_phrase is None if ctx.grounding("clip", "dog") else True
    finally:
        pipeline.load_grounding = original


# --------------------------------------------------------------------------- #
# rule coverage: the same question about one subject, worded several ways
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("question, kind, slots", [
    ("How does the distance between the camera and the woman in the pink dress change?", "cam_subject_range",
     {"subject": "woman in the pink dress", "implicit": False}),
    ("Over the entire video, does the camera’s distance to the white boat change significantly?", "cam_subject_range",
     {"subject": "white boat", "implicit": False}),
    ("Over the interval from t≈0s to t≈5s, does the camera get closer to the small gray cat carrying the red toy?",
     "cam_subject_range", {"subject": "small gray cat carrying the red toy", "implicit": False}),
    ("Over the course of the video, does the dog get closer to or farther from the camera?", "cam_subject_range",
     {"subject": "dog", "implicit": False}),
    ("How does the distance between the camera and the main subject change?", "cam_subject_range",
     {"subject": None, "implicit": True}),
    ("Over the course of the video, do the two flower pots get closer to each other or farther apart?", "pair_relation",
     {"category": "flower pot"}),
    ("Over the course of the video, do the two horses get closer, farther apart, or stay about the same distance?",
     "pair_relation", {"category": "horse"}),
    ("Over the course of the video, do the dog and the cat get closer to each other or farther apart?", "pair_relation",
     {"a": "dog", "b": "cat"}),
    ("What is the approximate average speed of the person on the bike over the entire video?", "metric_speed",
     {"subject": "person on the bike"}),
    ("Does the camera remain stationary or move during the entire video duration?", None, None),   # the camera's, not a subject's
    ("Does the person remain stationary or move during the video?", "subject_motion_profile", {"subject": "person"}),
])
def test_rules_read_each_wording_of_a_question(question, kind, slots):
    found = prog.rule_slots(question)
    if kind is None:
        assert found is None
    else:
        assert found == (kind, slots)


def test_context_with_a_clip_budget_forgets_the_oldest_clips(tmp_path):
    from dyn_dataset_exploration.decomposition import pipeline

    ctx = pipeline.Context.__new__(pipeline.Context)
    ctx.out, ctx.max_clips, ctx._recent = tmp_path, 2, []
    ctx._scenes = {("a", "all"): 1, ("b", "all"): 2, ("c", "all"): 3}
    ctx._groundings = {("a", "dog"): 1, ("c", "cat"): 2}
    ctx._tracks = {("a", "all", ()): 1}
    for slug in ("a", "b", "c"):
        ctx._touch(slug)
    assert ctx._recent == ["b", "c"]
    assert ("a", "all") not in ctx._scenes and ("a", "dog") not in ctx._groundings and not ctx._tracks
    assert ("b", "all") in ctx._scenes and ("c", "cat") in ctx._groundings


# --------------------------------------------------------------------------- #
# questions no template reads, and kinds the policy was never fitted on
# --------------------------------------------------------------------------- #

def test_an_unsupported_parse_becomes_a_model_only_program(tmp_path):
    from dyn_dataset_exploration.decomposition import pipeline

    (tmp_path / "programs").mkdir()
    (tmp_path / "programs" / "q.pipeline.json").write_text(json.dumps(
        {"qid": "q", "kind": "other", "unsupported": True, "reason": "kind 'other' has no template"}))
    program = pipeline.load_program(tmp_path, "q")
    assert (program.kind, program.route) == ("semantic", "vlm_only") and "no template" in program.notes[0]


def test_a_kind_without_a_fitted_policy_is_answered_by_the_models_own_probabilities():
    from dyn_dataset_exploration.decomposition import interpret

    assert "orbit_direction" not in interpret.FUSION
    scores = [interpret.OptionScore(l, "", 1.0 if l == "A" else 0.0, "match", "") for l in "ABCD"]
    letter, _ = interpret.fuse_for_kind("orbit_direction", scores, {"A": -3.0, "B": -0.2, "C": -2.0, "D": -4.0})
    assert letter == "B"                       # the measurement's support is ignored; Qwen's top letter wins
