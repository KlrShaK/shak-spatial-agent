"""Tests for S3/S4 on synthetic scenes whose answers are known by construction."""

from __future__ import annotations

import numpy as np
import pytest

from dyn_dataset_exploration.decomposition import interpret, measure
from dyn_dataset_exploration.decomposition import program as P
from dyn_dataset_exploration.decomposition.program import (
    Entity, Event, Measurement, QueryProgram, Window, assemble, validate,
)
from dyn_dataset_exploration.decomposition.scene4d import (
    lift_instances, resolve_window, WindowUnresolved,
)
from dyn_dataset_exploration.decomposition.synthetic import build_scene, rot_x, rot_y

CAM = Entity("cam", "camera", role="observer")


@pytest.fixture(autouse=True)
def uncalibrated_constants():
    """These tests pin the behaviour of the measurements at their documented defaults. Importing
    ``pipeline`` (anywhere in the process) applies ``calibrated.json``, so each test starts from the
    defaults and the calibrated state is put back afterwards."""

    from dyn_dataset_exploration.decomposition import pipeline

    saved = pipeline.snapshot()
    pipeline.reset()
    yield
    pipeline.apply_state(saved)


def _measurement(signal: str, of: list[str], reduce: str = "trend", **kw) -> Measurement:
    return Measurement("m1", signal, of, reduce, **kw)


def _prog(kind: str, form: str, entities, measurements, **kw) -> QueryProgram:
    return QueryProgram(qid="t", kind=kind, route="measure", answer_form=form, entities=entities,
                        measurements=measurements, **kw)


# --- trends -------------------------------------------------------------------------- #

def test_zigzag_ignores_wiggles_smaller_than_tolerance():
    x = np.array([1.0, 1.02, 0.99, 1.01, 1.0, 1.02])
    assert measure.zigzag(x, 0.1) == [0, 5]


def test_classify_trend_shapes():
    t = np.linspace(0, 1, 40)
    tol = 0.1
    assert measure.classify_trend(1 + 0 * t, tol)["trend"] == "constant"
    assert measure.classify_trend(1 - 0.5 * t, tol)["trend"] == "closer"
    assert measure.classify_trend(1 + 0.5 * t, tol)["trend"] == "farther"
    assert measure.classify_trend(1 + 0.6 * np.abs(t - 0.5), tol)["trend"] == "farther_then_closer" \
        or measure.classify_trend(1 + 0.6 * np.abs(t - 0.5), tol)["trend"] == "closer_then_farther"
    assert measure.classify_trend(1 + 0.6 * np.abs(t - 0.5), tol)["trend"] == "closer_then_farther"
    assert measure.classify_trend(1 - 0.6 * np.abs(t - 0.5), tol)["trend"] == "farther_then_closer"
    assert measure.classify_trend(1 + 0.4 * np.sin(6 * np.pi * t), tol)["trend"] == "fluctuating"
    assert measure.classify_trend(np.array([1.0, np.nan, 1.0]), tol)["trend"] is None


def test_classify_trend_reports_when_a_slope_levels_off():
    x = np.concatenate([np.linspace(1.0, 0.4, 20), np.full(20, 0.4)])
    got = measure.classify_trend(x, 0.1)
    assert got["trend"] == "closer" and got["plateau_from"] is not None and got["plateau_from"] < 0.6


# --- camera -------------------------------------------------------------------------- #

def _camera_value(**kw):
    scene, _ = build_scene(**kw)
    r = measure.measure_camera_motion(scene, None, _measurement("camera_motion", ["cam"], "dominant_motion"))
    assert r.available
    return r.value


def test_camera_static():
    assert _camera_value()["motion"] == "static"


def test_camera_dolly_forward_and_backward():
    v = _camera_value(camera_t=lambda t: (0, 0, 0.02 * t))
    assert (v["motion"], v["direction"]) == ("dolly", "forward")
    v = _camera_value(camera_t=lambda t: (0, 0, -0.02 * t))
    assert (v["motion"], v["direction"]) == ("dolly", "backward")


def test_camera_pan_left_and_right():
    v = _camera_value(camera_R=lambda t: rot_y(0.6 * t))
    assert (v["motion"], v["direction"]) == ("pan", "right")
    v = _camera_value(camera_R=lambda t: rot_y(-0.6 * t))
    assert (v["motion"], v["direction"]) == ("pan", "left")


def test_camera_tilt_up():
    v = _camera_value(camera_R=lambda t: rot_x(0.6 * t))
    assert (v["motion"], v["direction"]) == ("tilt", "up")


def test_camera_truck_and_orbit():
    v = _camera_value(camera_t=lambda t: (0.02 * t, 0, 0))
    assert (v["motion"], v["direction"]) == ("truck", "right")
    # moving left while turning right keeps a subject in view: a clockwise orbit
    v = _camera_value(camera_t=lambda t: (-0.02 * t, 0, 0), camera_R=lambda t: rot_y(0.6 * t))
    assert (v["motion"], v["direction"]) == ("orbit", "cw")


def test_camera_two_strong_motions_are_complex():
    v = _camera_value(camera_t=lambda t: (0, 0, 0.02 * t), camera_R=lambda t: rot_y(0.9 * t))
    assert v["motion"] == "complex"


def test_camera_window_restricts_the_measurement():
    scene, _ = build_scene(frames=60, fps=6.0, camera_t=lambda t: (0, 0, 0.03 * max(t - 30, 0)))
    m = _measurement("camera_motion", ["cam"], "dominant_motion")
    first = measure.measure_camera_motion(scene, Window("phase", phase="first_half"), m)
    second = measure.measure_camera_motion(scene, Window("phase", phase="second_half"), m)
    assert first.value["motion"] == "static" and second.value["motion"] == "dolly"


def test_jitter_levels():
    rng = np.random.default_rng(0)
    calm, _ = build_scene(frames=60, camera_R=lambda t: rot_y(0.02 * rng.normal()))
    shaky, _ = build_scene(frames=60, camera_R=lambda t: rot_y(1.5 * rng.normal()))
    m = _measurement("camera_jitter", ["cam"], "level")
    assert measure.measure_camera_jitter(calm, None, m).value["level"] == 0
    assert measure.measure_camera_jitter(shaky, None, m).value["level"] >= 2


# --- entities ------------------------------------------------------------------------ #

def _run(scene, groundings, program):
    return measure.run_program(scene, program, groundings)


def test_lift_recovers_the_entity_world_position():
    scene, g = build_scene(entities=[("dog", lambda t: (0.5, 0.2, 3.0 + 0.05 * t))],
                           rng=np.random.default_rng(1))
    track = lift_instances(scene, g["dog"], [0], entity_id="e1")
    assert track.present.all() and track.confident.all()
    assert np.allclose(track.smooth[10], (0.5, 0.2, 3.5), atol=0.05)


def test_range_to_the_camera_v_shape():
    path = lambda t: (0.0, 0.0, 4.0 + 0.1 * abs(t - 20) - 0.0)  # noqa: E731
    scene, g = build_scene(frames=41, entities=[("dog", path)])
    program = _prog("cam_subject_range", "trend", [CAM, Entity("e1", "object", "dog")],
                    [_measurement("range", ["e1", "cam"])])
    results, _ = _run(scene, g, program)
    assert results["m1"].value["trend"] == "farther_then_closer" or results["m1"].value["trend"] == "closer_then_farther"
    assert results["m1"].value["trend"] == "closer_then_farther"


def test_range_change_is_judged_as_a_ratio_so_doubling_and_halving_are_alike():
    """A subject that goes from 4 to 8 units away and one that goes from 8 to 4 change by the same
    factor; in linear units the second change would be judged smaller against its (larger) start."""

    def trend(z0, z1):
        scene, g = build_scene(frames=41, entities=[("dog", lambda t: (0.0, 0.0, z0 + (z1 - z0) * t / 40))])
        program = _prog("cam_subject_range", "trend", [CAM, Entity("e1", "object", "dog")],
                        [_measurement("range", ["e1", "cam"])])
        results, _ = _run(scene, g, program)
        return results["m1"].value

    with_tol = dict(measure.TUNING)
    measure.TUNING["range_rel_tol"] = 0.6                     # a change under a factor 1.6 is "constant"
    try:
        assert trend(4.0, 8.0)["trend"] == "farther" and trend(8.0, 4.0)["trend"] == "closer"
        assert trend(4.0, 5.5)["trend"] == "constant" and trend(5.5, 4.0)["trend"] == "constant"
        assert trend(4.0, 8.0)["ratio_end_over_start"] == pytest.approx(2.0, rel=0.1)
    finally:
        measure.TUNING.clear()
        measure.TUNING.update(with_tol)


def test_range_constant_when_the_entity_keeps_its_distance():
    scene, g = build_scene(entities=[("dog", lambda t: (0.02 * t, 0, 4.0))], rng=np.random.default_rng(2))
    program = _prog("cam_subject_range", "trend", [CAM, Entity("e1", "object", "dog")],
                    [_measurement("range", ["e1", "cam"])])
    results, _ = _run(scene, g, program)
    assert results["m1"].value["trend"] == "constant"


def test_pair_relation_of_two_entities():
    scene, g = build_scene(entities=[("cat", lambda t: (-1.0 - 0.03 * t, 0, 4.0)),
                                     ("dog", lambda t: (1.0 + 0.03 * t, 0, 4.0))])
    program = _prog("pair_relation", "trend",
                    [Entity("e1", "object", "cat"), Entity("e2", "object", "dog")],
                    [_measurement("range", ["e1", "e2"])])
    results, _ = _run(scene, g, program)
    assert results["m1"].value["trend"] == "farther"


def test_displacement_words_in_each_frame():
    scene, g = build_scene(frames=30, entities=[("dog", lambda t: (0.03 * t, 0.0, 4.0))])
    program = _prog("subject_direction", "direction", [Entity("e1", "object", "dog")],
                    [_measurement("displacement", ["e1"], "direction", frame="convention")])
    results, _ = _run(scene, g, program)
    v = results["m1"].value
    for frame in ("world0", "levelled0", "camera_end", "camera_rel"):
        words = interpret.direction_words(v, {"frame": frame, "z_toward_camera": False})["words"]
        assert words == {"x": "right", "y": None, "z": None}, frame
    assert interpret.direction_words(v, {"frame": "image", "z_toward_camera": False})["words"]["x"] == "right"


def test_camera_motion_is_cancelled_in_world_frames_but_not_camera_rel():
    # a still entity, camera trucking right: in the world it does not move; the camera sees it slide left
    scene, g = build_scene(frames=30, camera_t=lambda t: (0.03 * t, 0, 0), entities=[("dog", lambda t: (0.0, 0.0, 4.0))])
    program = _prog("subject_direction", "direction", [Entity("e1", "object", "dog")],
                    [_measurement("displacement", ["e1"], "direction", frame="convention")])
    v = _run(scene, g, program)[0]["m1"].value
    assert interpret.direction_words(v, {"frame": "world0"})["words"] == {"x": None, "y": None, "z": None}
    assert interpret.direction_words(v, {"frame": "camera_rel"})["words"]["x"] == "left"


def test_instance_count_and_its_stability():
    scene, g = build_scene(entities=[("person", lambda t: (1.0, 0, 4.0))])
    program = _prog("instance_count", "count", [Entity("c1", "category", "person")],
                    [_measurement("instances", ["c1"], "count")])
    results, tracks = _run(scene, g, program)
    assert results["m1"].value["count"] == 1 and results["m1"].value["stable"]


def test_event_order_camera_pan_before_entity_enters():
    scene, g = build_scene(
        frames=60, fps=6.0, camera_R=lambda t: rot_y(0.8 * max(t - 15, 0)),
        entities=[("man", lambda t: (0.5, 0, 5.0) if t >= 40 else None)])
    program = _prog(
        "event_order", "order", [CAM, Entity("e2", "object", "man", role="event_actor")],
        [Measurement("m1", "onset", ["ev1"], "onset"), Measurement("m2", "onset", ["ev2"], "onset"),
         Measurement("m3", "order", ["m1", "m2"], "order")],
        events=[Event("ev1", "cam", "pan_right", "the camera pans right"),
                Event("ev2", "e2", "enters_frame", "the man enters")])
    results, _ = _run(scene, g, program)
    assert 15 <= results["m1"].value["onset"] <= 20
    assert results["m2"].value["onset"] == 40
    assert results["m3"].value["first"] == "ev1"


def test_event_order_reverse():
    scene, g = build_scene(
        frames=60, fps=6.0, camera_R=lambda t: rot_y(0.8 * max(t - 40, 0)),
        entities=[("man", lambda t: (0.5, 0, 5.0) if t >= 10 else None)])
    program = _prog(
        "event_order", "order", [CAM, Entity("e2", "object", "man", role="event_actor")],
        [Measurement("m1", "onset", ["ev1"], "onset"), Measurement("m2", "onset", ["ev2"], "onset"),
         Measurement("m3", "order", ["m1", "m2"], "order")],
        events=[Event("ev1", "cam", "pan_right", "pan"), Event("ev2", "e2", "enters_frame", "enters")])
    assert _run(scene, g, program)[0]["m3"].value["first"] == "ev2"


def test_metric_value_at_an_instant():
    scene, g = build_scene(fps=6.0, metres_per_unit=2.0,
                           entities=[("a", lambda t: (-1.0, 0, 4.0)), ("b", lambda t: (1.5, 0, 4.0))])
    program = _prog("metric_distance", "metric",
                    [Entity("e1", "object", "a"), Entity("e2", "object", "b")],
                    [_measurement("range", ["e1", "e2"], "value_at", metric=True)],
                    window=Window("instant", 2.0))
    v = _run(scene, g, program)[0]["m1"].value
    assert v["metres"] == pytest.approx(5.0, rel=0.05)


def test_window_resolution():
    scene, _ = build_scene(frames=61, fps=6.0)
    assert resolve_window(scene, Window("interval", 0.0, 5.0))[:2] == (0, 30)
    assert resolve_window(scene, Window("instant", 2.0))[:2] == (12, 12)
    assert resolve_window(scene, Window("phase", phase="last_25"))[:2] == (45, 60)
    no_rate, _ = build_scene(frames=20, fps=None)
    with pytest.raises(WindowUnresolved):
        resolve_window(no_rate, Window("instant", 2.0))


def test_a_missing_grounding_is_unavailable_not_a_guess():
    scene, g = build_scene(entities=[("dog", lambda t: (0, 0, 4.0))])
    program = _prog("cam_subject_range", "trend", [CAM, Entity("e1", "object", "cat")],
                    [_measurement("range", ["e1", "cam"])])
    results, _ = _run(scene, g, program)
    assert not results["m1"].available and "ground" in results["m1"].reason


# --- interpretation ------------------------------------------------------------------ #

def test_direction_scoring_weights_each_axis_by_how_far_it_can_be_trusted():
    v = {"frames": {"camera_rel": [0.5, 0.0, 0.5]}, "scene_scale": 1.0, "noise": 0.001}
    conv = {"frame": "camera_rel", "z_toward_camera": False}
    old_conv, old_w = dict(interpret.DIRECTION_CONVENTION), {k: interpret.DECISION[k] for k in ("dir_w_x", "dir_w_y", "dir_w_z")}
    interpret.DIRECTION_CONVENTION.update(conv)
    interpret.DECISION.update(dir_w_x=1.0, dir_w_y=0.5, dir_w_z=0.3)
    try:
        score = lambda atoms: interpret.score_direction(atoms, v)[0]  # noqa: E731
        assert score({"x": "right", "z": "forward"}) == 1.0
        assert score({"x": "right"}) == pytest.approx(1 - 0.15 * 0.3)          # unmentioned motion along z costs a little
        assert score({"x": "left"}) == 0.0                                      # the reliable axis wrong: no support
        assert score({"x": "right", "z": "backward"}) == pytest.approx(1 / 1.3)   # z wrong: still mostly supported
        assert score({"x": "left", "z": "forward"}) == pytest.approx(0.3 / 1.3)
        assert score({"x": "right", "z": "backward", "slight": ["z"]}) == pytest.approx(1 / 1.3)
        assert score({"random": True}) is None
        # an option that differs from another only in depth is (nearly) a tie once z is down-weighted...
        interpret.DECISION["dir_w_z"] = 0.0
        assert score({"x": "right", "z": "forward"}) == score({"x": "right", "z": "backward"}) == 1.0
        # ...and a claim made only about an axis with no weight cannot be checked at all
        assert score({"z": "forward"}) is None
    finally:
        interpret.DIRECTION_CONVENTION.clear()
        interpret.DIRECTION_CONVENTION.update(old_conv)
        interpret.DECISION.update(old_w)


def test_metric_scoring_prefers_the_nearest_option_in_log_space():
    v = {"metres": 3.0, "scale_iqr_over_median": 0.05}
    scores = {x: interpret.score_metric({"value": x, "unit": "m"}, v)[0] for x in (1.5, 3.0, 5.0, 10.0)}
    assert max(scores, key=scores.get) == 3.0 and scores[3.0] > 0.95 and scores[10.0] < 0.01
    assert interpret.score_metric({"value": 3.0}, {"units": 1.0})[0] is None


def test_count_and_order_scoring():
    assert interpret.score_count({"count": 2}, {"count": 2})[0] == 1.0
    assert interpret.score_count({"count": 3}, {"count": 2})[0] == 0.3
    assert interpret.score_order({"first": "ev1"}, {"first": "ev1", "onsets": [1, 9]})[0] == 1.0
    assert interpret.score_order({"first": "ev2"}, {"first": "ev1"})[0] == 0.0
    assert interpret.score_order({"first": "ev1"}, {"first": "one_only:ev1"})[0] is None


def test_trend_scoring_uses_the_plateau():
    got = {"trend": "closer", "net_units": -1.0, "plateau_from": 0.4}
    assert interpret.score_trend({"trend": "closer_then_constant"}, got)[0] == 1.0
    assert interpret.score_trend({"trend": "closer"}, got)[0] == 0.9
    got["plateau_from"] = 0.95
    assert interpret.score_trend({"trend": "closer_then_constant"}, got)[0] == 0.4
    assert interpret.score_trend({"undeterminable": True}, got)[0] is None


def test_decision_leaf_a_needs_a_clear_winner_and_falls_back_otherwise():
    scene, g = build_scene(entities=[("dog", lambda t: (0.0, 0.0, 4.0 - 0.06 * t))])
    options = {"A": "The distance decreases.", "B": "The distance increases.", "C": "The distance remains stable.",
               "D": "The distance decreases first, and then increases"}
    program = assemble("t", "cam_subject_range", {"subject": "dog", "implicit": False},
                       "How does the distance between the camera and the dog change?", options, source="rules")
    results, _ = measure.run_program(scene, program, g)
    decision = interpret.decide(program, results)
    assert decision.leaf == "A" and decision.chosen == "A"
    # an untrusted kind never answers deterministically
    assert interpret.decide(program, results, trusted=set()).leaf == "B"
    # a failed measurement is a fallback, not a guess
    bad, _ = measure.run_program(scene, program, {})
    assert interpret.decide(program, bad).leaf == "B"
    # duplicate options: the earlier letter wins and is not competing with its twin
    dup = assemble("t", "cam_subject_range", {"subject": "dog", "implicit": False},
                   "How does the distance between the camera and the dog change?",
                   {**options, "B": "The distance decreases."}, source="rules")
    decision = interpret.decide(dup, results)
    assert decision.chosen == "A" and decision.leaf == "A"


def test_programs_from_the_synthetic_tests_validate():
    validate(_prog("cam_subject_range", "trend", [CAM, Entity("e1", "object", "dog")],
                   [_measurement("range", ["e1", "cam"])]))


# --- iteration-2 measurements --------------------------------------------------------- #

def _run_kind(scene, g, kind, form, entities, m):
    program = _prog(kind, form, entities, [m])
    return measure.run_program(scene, program, g)[0]["m1"]


def test_zoom_none_in_out_mixed():
    m = _measurement("focal", ["cam"], "verdict")
    def zoom(fn):
        scene, _ = build_scene(frames=40, focal=fn)
        return measure.measure_zoom(scene, None, m).value["zoom"]
    assert zoom(lambda t: 60.0) == "none"
    assert zoom(lambda t: 60.0 * (1 + 0.02 * t)) == "in"
    assert zoom(lambda t: 60.0 / (1 + 0.02 * t)) == "out"
    assert zoom(lambda t: 60.0 * (1 + 0.5 * np.sin(np.pi * t / 20))) == "mixed"


def test_rhythm_shapes():
    m = _measurement("camera_speed", ["cam"], "profile")
    def rhythm(fn, frames=60):
        scene, _ = build_scene(frames=frames, camera_t=fn)
        return measure.measure_rhythm(scene, None, m).value["rhythm"]
    assert rhythm(lambda t: (0, 0, 0)) == "fixed"
    assert rhythm(lambda t: (0, 0, 0.02 * t)) == "steady"
    assert rhythm(lambda t: (0, 0, 0.04 * max(t - 35, 0))) == "still_then_burst"
    assert rhythm(lambda t: (0, 0, 0.02 * min(t, 25))) == "moving_then_stop"
    assert rhythm(lambda t: (0, 0, 0.001 * t + 0.0015 * t * t / 6)) == "speeds_up"


def test_revisit_same_and_different():
    m = _measurement("covisibility", ["cam"], "verdict")
    same, _ = build_scene(frames=30, camera_R=lambda t: rot_y(0.05 * t))
    diff, _ = build_scene(frames=30, camera_R=lambda t: rot_y(2.0 * t))
    assert measure.measure_revisit(same, None, m).value["revisit"] == "same"
    assert measure.measure_revisit(diff, None, m).value["revisit"] == "different"


def test_apparent_size_larger_smaller_unchanged():
    m = _measurement("area", ["e1"], "shape")
    ent = [Entity("e1", "object", "dog")]
    def size(path):
        scene, g = build_scene(frames=40, entities=[("dog", path, 0.5)])
        return _run_kind(scene, g, "apparent_size", "size", ent, m).value["size"]
    assert size(lambda t: (0.0, 0.0, 8.0 - 0.12 * t)) == "larger"
    assert size(lambda t: (0.0, 0.0, 3.0 + 0.12 * t)) == "smaller"
    assert size(lambda t: (0.0, 0.0, 5.0)) == "unchanged"


def test_visibility_at_the_end():
    ent = [Entity("e1", "object", "dog")]
    m = _measurement("presence", ["e1"], "verdict")
    def verdict(path):
        scene, g = build_scene(frames=40, entities=[("dog", path, 0.5)])
        program = _prog("subject_visibility", "visibility", ent, [m], window=Window("phase", phase="end"))
        return measure.run_program(scene, program, g)[0]["m1"].value["visible"]
    assert verdict(lambda t: (0.0, 0.0, 5.0)) == "full"
    assert verdict(lambda t: (0.0, 0.0, 5.0) if t < 30 else None) == "out"


def test_elevation_high_eye_low():
    m = _measurement("elevation", ["e1", "cam"], "verdict", frame="levelled")
    def angle(cam_y):
        scene, g = build_scene(frames=30, camera_t=lambda t: (0.0, cam_y, 0.0), entities=[("dog", lambda t: (0.0, 0.0, 5.0))])
        program = _prog("cam_subject_elevation", "elevation", [CAM, Entity("e1", "object", "dog")], [m])
        return measure.run_program(scene, program, g)[0]["m1"].value["angle"]
    assert angle(0.0) == "eye"
    assert angle(-2.0) == "high"        # camera above (up is -Y)
    assert angle(2.0) == "low"


def test_motion_profile_none_continuous_once_intermittent():
    ent = [Entity("e1", "object", "dog")]
    m = _measurement("speed", ["e1"], "profile")
    def profile(path):
        scene, g = build_scene(frames=60, fps=6.0, entities=[("dog", path)])
        return _run_kind(scene, g, "subject_motion_profile", "motion_profile", ent, m).value["motion"]
    assert profile(lambda t: (0.0, 0.0, 5.0)) == "none"
    assert profile(lambda t: (0.05 * t, 0.0, 5.0)) == "continuous"
    assert profile(lambda t: (0.05 * min(max(t - 25, 0), 8), 0.0, 5.0)) == "once"
    assert profile(lambda t: (0.05 * (min(t, 15) + min(max(t - 30, 0), 15)), 0.0, 5.0)) == "intermittent"


def test_categorical_scoring_of_new_forms():
    assert interpret.SCORERS["zoom"]({"zoom": "none"}, {"zoom": "none"})[0] == 1.0
    assert interpret.SCORERS["zoom"]({"zoom": "in"}, {"zoom": "none"})[0] == 0.0
    assert interpret.SCORERS["revisit"]({"revisit": "obscured"}, {"revisit": "same"})[0] is None
    assert interpret.SCORERS["visibility"]({"visible": "unclear"}, {"visible": "full"})[0] is None
    assert interpret.SCORERS["size"]({"size": "larger"}, {"size": "unchanged"})[0] == 0.15


# --- robustness: no measurement may raise, whatever the scene looks like ---------------- #

def _all_programs():
    cam, dog, cat = Entity("cam", "camera", role="observer"), Entity("e1", "object", "dog"), Entity("e2", "object", "cat")
    group = Entity("g1", "group", "dog", count=2, role="pair")
    ev = [Event("ev1", "cam", "pan_right", "pan"), Event("ev2", "e1", "enters_frame", "enters")]
    return [
        _prog("camera_path", "camera_motion", [cam], [_measurement("camera_motion", ["cam"], "dominant_motion")]),
        _prog("camera_shake", "shake_level", [cam], [_measurement("camera_jitter", ["cam"], "level")]),
        _prog("camera_zoom", "zoom", [cam], [_measurement("focal", ["cam"], "verdict")]),
        _prog("camera_rhythm", "rhythm", [cam], [_measurement("camera_speed", ["cam"], "profile")]),
        _prog("scene_revisit", "revisit", [cam], [_measurement("covisibility", ["cam"], "verdict")]),
        _prog("cam_subject_range", "trend", [cam, dog], [_measurement("range", ["e1", "cam"])]),
        _prog("pair_relation", "trend", [dog, cat], [_measurement("range", ["e1", "e2"])]),
        _prog("pair_relation", "trend", [group], [_measurement("range", ["g1[0]", "g1[1]"])]),
        _prog("metric_distance", "metric", [dog, cat], [_measurement("range", ["e1", "e2"], "value_at", metric=True)],
              window=Window("instant", 1.0)),
        _prog("subject_direction", "direction", [dog], [_measurement("displacement", ["e1"], "direction", frame="convention")]),
        _prog("instance_count", "count", [Entity("c1", "category", "dog")], [_measurement("instances", ["c1"], "count")]),
        _prog("apparent_size", "size", [dog], [_measurement("area", ["e1"], "shape")]),
        _prog("subject_visibility", "visibility", [dog], [_measurement("presence", ["e1"], "verdict")],
              window=Window("phase", phase="end")),
        _prog("cam_subject_elevation", "elevation", [cam, dog], [_measurement("elevation", ["e1", "cam"], "verdict", frame="levelled")]),
        _prog("subject_motion_profile", "motion_profile", [dog], [_measurement("speed", ["e1"], "profile")]),
        _prog("event_order", "order", [cam, dog],
              [Measurement("m1", "onset", ["ev1"], "onset"), Measurement("m2", "onset", ["ev2"], "onset"),
               Measurement("m3", "order", ["m1", "m2"], "order")], events=ev),
    ]


@pytest.mark.parametrize("frames", [3, 6, 25])
@pytest.mark.parametrize("fps", [None, 6.0, 30.0])
@pytest.mark.parametrize("scenario", ["normal", "gaps", "absent", "border"])
def test_no_measurement_raises(frames, fps, scenario):
    rng = np.random.default_rng(frames)
    def path(t):
        if scenario == "absent":
            return None
        if scenario == "gaps" and rng.random() < 0.5:
            return None
        if scenario == "border":
            return (4.0, 0.0, 3.0 + 0.1 * t)       # projects outside the image
        return (0.3 * np.sin(t / 3), 0.0, 4.0 + 0.05 * t)
    scene, g = build_scene(frames=frames, fps=fps, camera_R=lambda t: rot_y(0.3 * t),
                           entities=[("dog", path, 0.4), ("cat", lambda t: (-0.5, 0.0, 4.5), 0.4)])
    for program in _all_programs():
        results, _ = measure.run_program(scene, program, g)
        assert set(results) == {m.id for m in program.measurements}, program.kind
        for r in results.values():
            assert r.available or r.reason
            r.to_dict()
        interpret.decide(program, results)


def test_tracks_and_windows_on_a_scene_that_ends_before_the_window():
    scene, g = build_scene(frames=10, fps=6.0)
    program = _prog("camera_path", "camera_motion", [CAM], [_measurement("camera_motion", ["cam"], "dominant_motion")],
                    window=Window("interval", 50.0, 60.0))
    r = measure.run_program(scene, program, g)[0]["m1"]
    assert not r.available or r.window[1] <= scene.length - 1


def test_a_result_that_rests_on_a_simpler_prompt_says_so_and_is_not_high_confidence():
    scene, g = build_scene(entities=[("dog", lambda t: (0.0, 0.0, 4.0 - 0.06 * t))])
    program = _prog("cam_subject_range", "trend", [CAM, Entity("e1", "object", "dog")],
                    [_measurement("range", ["e1", "cam"])])
    plain, _ = _run(scene, g, program)
    assert plain["m1"].confidence == "high" and not any("simpler prompt" in n for n in plain["m1"].notes)
    g["dog"].used_phrase = "animal"
    flagged, _ = _run(scene, g, program)
    assert flagged["m1"].confidence == "medium"
    assert any("simpler prompt 'animal'" in n for n in flagged["m1"].notes)


# --- metric speed -------------------------------------------------------------------- #

def _speed(path, *, frames=40, fps=6.0, mpu=2.0, camera_t=lambda t: (0.0, 0.0, 0.0), mode="net"):
    scene, g = build_scene(frames=frames, fps=fps, metres_per_unit=mpu, camera_t=camera_t,
                           entities=[("dog", path)], rng=np.random.default_rng(3))
    program = _prog("metric_speed", "metric", [Entity("e1", "object", "dog")],
                    [_measurement("speed_metric", ["e1"], "value", metric=True)])
    old = measure.TUNING["speed_mode"]
    measure.TUNING["speed_mode"] = mode
    try:
        return measure.run_program(scene, program, g)[0]["m1"]
    finally:
        measure.TUNING["speed_mode"] = old


def test_metric_speed_recovers_a_known_speed_in_metres_per_second():
    # 0.06 units per frame at 6 fps = 0.36 units/s; 2 m per unit -> 0.72 m/s
    r = _speed(lambda t: (0.06 * t, 0.0, 4.0))
    assert r.available and r.value["mps"] == pytest.approx(0.72, rel=0.1)
    assert r.value["net_mps"] == pytest.approx(0.72, rel=0.1) and r.value["path_mps"] == pytest.approx(0.72, rel=0.15)


def test_metric_speed_is_the_subjects_own_motion_not_the_cameras():
    # a camera dolly-ing forward while the dog stands still: the dog's world speed is zero
    r = _speed(lambda t: (0.0, 0.0, 6.0), camera_t=lambda t: (0.0, 0.0, 0.05 * t))
    assert r.available and r.value["mps"] < 0.05


def test_metric_speed_net_and_path_differ_when_the_subject_turns_back():
    def there_and_back(t):
        return (0.08 * (t if t <= 20 else 40 - t), 0.0, 4.0)
    net, path = _speed(there_and_back, mode="net").value, _speed(there_and_back, mode="path").value
    assert net["net_mps"] < 0.3 * path["path_mps"]          # net displacement ends where it started
    assert path["path_mps"] == pytest.approx(0.96, rel=0.2)  # 0.08*6*2 = 0.96 m/s along the path


def test_metric_speed_needs_a_frame_rate_and_a_metric_scale():
    no_scale = _speed(lambda t: (0.06 * t, 0.0, 4.0), mpu=None)
    assert no_scale.available and "mps" not in no_scale.value and no_scale.confidence == "low"
    scene, g = build_scene(frames=40, fps=None, entities=[("dog", lambda t: (0.06 * t, 0.0, 4.0))])
    program = _prog("metric_speed", "metric", [Entity("e1", "object", "dog")],
                    [_measurement("speed_metric", ["e1"], "value", metric=True)])
    r = measure.run_program(scene, program, g)[0]["m1"]
    assert not r.available and "frame rate" in r.reason


def test_speed_options_are_scored_in_log_space_in_metres_per_second():
    v = {"mps": 3.0, "scale_iqr_over_median": 0.05}
    scores = {x: interpret.score_metric({"value": x, "unit": "m/s"}, v)[0] for x in (1.5, 3.2, 5.8, 8.0)}
    assert max(scores, key=scores.get) == 3.2
    assert interpret.score_metric({"value": 3.0, "unit": "m/s"}, {"metres": 3.0})[0] is None   # a distance is not a speed


def test_the_average_speed_template_is_read_by_rules():
    q = "What is the approximate average speed of the dancer in the pink dress from the start to the end of the video?"
    program = assemble("t", "metric_speed", {"subject": "dancer in the pink dress"}, q,
                       {"A": "0.3 m/s", "B": "0.8 m/s", "C": "1.5 m/s", "D": "2.2 m/s"}, source="rules")
    assert program.kind == "metric_speed" and program.window.kind == "whole"
    assert [o.atoms["value"] for o in program.options] == [0.3, 0.8, 1.5, 2.2]
    assert program.measurements[0].signal == "speed_metric" and program.measurements[0].metric


# --- camera translation direction ---------------------------------------------------- #

def _camera_direction(camera_t, options, frames=40):
    scene, _ = build_scene(frames=frames, fps=6.0, camera_t=camera_t)
    program = _prog("camera_translation", "direction", [CAM],
                    [_measurement("camera_displacement", ["cam"], "direction", frame="levelled")])
    r = measure.run_program(scene, program, {})[0]["m1"]
    return r, {k: interpret.score_direction(P.direction_atoms(text), r.value)[0] for k, text in options.items()}


def test_camera_translation_direction_reads_the_cameras_own_axes():
    options = {"A": "Rightward and forward.", "B": "Leftward and backward.", "C": "Upward and leftward.",
               "D": "Rightward and backward."}
    r, scores = _camera_direction(lambda t: (0.03 * t, 0.0, 0.03 * t), options)
    assert r.available and scores["A"] == 1.0 and scores["B"] == 0.0 and scores["D"] == pytest.approx(0.5)
    _, scores = _camera_direction(lambda t: (-0.03 * t, 0.0, -0.03 * t), options)
    assert scores["B"] == 1.0 and scores["A"] == 0.0
    # up is -y (down is +y) in the camera frame
    _, scores = _camera_direction(lambda t: (-0.03 * t, -0.03 * t, 0.0), options)
    assert scores["C"] == 1.0


def test_camera_translation_forward_backward_is_not_down_weighted_like_a_subjects():
    v = {"frames": {"levelled0": [0.5, 0.0, 0.5]}, "direct": True, "noise": 0.0, "scene_scale": 1.0}
    assert interpret.score_direction({"x": "right", "z": "backward"}, v)[0] == pytest.approx(0.5)   # z counts fully
    subject = {"frames": {k: [0.5, 0.0, 0.5] for k in ("camera_rel", "world0", "levelled0", "camera_end", "image")},
               "noise": 0.0, "scene_scale": 1.0}
    assert interpret.score_direction({"x": "right", "z": "backward"}, subject)[0] > 0.7             # z barely counts


# --- soft fusion of the model and the measurement ------------------------------------- #

def test_fusion_is_the_model_alone_when_the_measurement_weight_is_zero_and_the_measurement_when_large():
    logprobs = {"A": -0.1, "B": -3.0, "C": -4.0, "D": -4.5}          # the model is confident in A
    support = {"A": 0.0, "B": 1.0, "C": 0.0, "D": 0.0}                # the measurement says B
    assert interpret.fuse_scores(support, [], logprobs, 1.0, 0.0, 0.1)[0] == "A"
    assert interpret.fuse_scores(support, [], logprobs, 1.0, 4.0, 0.1)[0] == "B"
    # a middling weight lets the measurement break a near-tie but not overturn a confident model
    close = {"A": -0.9, "B": -1.0, "C": -3.0, "D": -3.0}
    assert interpret.fuse_scores(support, [], close, 1.0, 0.5, 0.1)[0] == "B"
    assert interpret.fuse_scores(support, [], logprobs, 1.0, 0.5, 0.1)[0] == "A"


def test_fusion_gives_an_uncheckable_option_no_measurement_term_and_reports_a_margin():
    logprobs = {"A": -1.2, "B": -1.2, "C": -3.0}
    letter, margin, posterior = interpret.fuse_scores({"A": 0.0, "B": 0.9, "C": 0.0}, ["A"], logprobs, 1.0, 2.0, 0.1)
    assert letter in ("A", "B") and 0.0 <= margin <= 1.0 and sum(posterior.values()) == pytest.approx(1.0)
    # A is unverifiable, so it is not punished for the measurement's silence: B (verified, supported) vs A
    # (unchecked) must be closer than B vs a verified option the measurement rejects
    assert posterior["A"] > posterior["C"]
