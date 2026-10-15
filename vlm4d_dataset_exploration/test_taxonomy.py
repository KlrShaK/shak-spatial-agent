"""VLM4D taxonomy rules: GPU-free, no network. The last tests need the dataset on disk and skip otherwise."""

from pathlib import Path

import pytest

from vlm4d_dataset_exploration import taxonomy as T

ROOT = Path("/cluster/work/igp_psr/spanwar/datasets/VLM4D")

# The 9 rows of the dataset's preview parquet, with its own `reasoning_focus` label.
PREVIEW = [
    ("What direction is the horse in the foreground moving towards?", "Right", "translational"),
    ("How is the jet-ski rotating?", "Clockwise", "rotational"),
    ("Is the cat turning to the left or right from its own perspective?", "Left", "rotational"),  # perspective-aware rotation
    ("Which direction does the person move the cutlet from it's original position?", "Right", "translational"),
    ("Which direction does the person turn to pick up the branch?", "Right", "rotational"),
    ("How many times does the person look to the right before turning?", "Once", "counting"),
    ("What direction is the sports car moving towards?", "Right", "translational"),
    ("What direction is the battle robot moving towards?", "Not moving", "translational"),
    ("Which direction did the dog move?", "No dog there", "false_positive"),
]


@pytest.mark.parametrize("question,gold,expected", PREVIEW)
def test_paper_type_agrees_with_the_preview_labels(question, gold, expected):
    assert T.paper_type(question, gold) == expected


def test_perspective_flag():
    assert T.perspective("Is the cat turning to the left or right from its own perspective?")
    assert T.perspective("With respect to the sink, where is the fridge?")
    assert not T.perspective("What direction is the dog moving towards?")


@pytest.mark.parametrize("text", ["no dog there", "there are no people in the video", "No plane", "no such men there",
                                  "No bowl in the video.", "there is no tennis player", "no robot's foot there"])
def test_absence_options(text):
    assert T.is_absent(text) and T.option_form(text) == "absent"


@pytest.mark.parametrize("text", ["No", "no rotation", "no turn", "not moving", "no hands", "None", "north"])
def test_not_absence(text):
    assert not T.is_absent(text)


def test_false_positive_covers_absent_events_but_not_static_answers():
    assert T.is_false_positive("Nothing.") and T.is_false_positive("In no direction, he did not touch the white bottle.")
    assert not T.is_false_positive("not moving") and not T.is_false_positive("The dark bottle did not move.")


@pytest.mark.parametrize("question,options,source,kind", [
    ("Which rotational direction does the person turn to go to the fridge?", ["clockwise", "counter-clockwise", "not turning", "up"],
     "ego4d", "camera_path"),
    ("Which way does the person turn to get the pan?", ["forward", "right", "left", "backward"], "ego4d", "camera_path"),
    ("How many times does he pan his head from left to right?", ["1", "2", "3", "4"], "ego4d", "repetition_count"),
    ("Which direction is the pan handle moving?", ["moving right", "moving up", "moving left", "not moving"], "ego4d",
     "subject_direction"),                                                   # a frying pan, not a camera pan
    ("In which direction does the person move to go the sink?", ["Forward", "Left", "Backward", "Right"], "ego4d",
     "camera_translation"),
    ("Does the person move closer to the oven throughout the video?", ["Yes", "No", "Not sure", "Possibly"], "ego4d",
     "cam_subject_range"),
    ("Is the car driving forwards or backwards from its own perspective?", ["forwards", "backwards", "not moving", "right"],
     "davis", "traj_subject_own_frame"),
    ("Is the cat turning to the left or right from the camera perspective?", ["left", "right", "towards", "not moving"],
     "davis", "subject_turn"),
    ("Which direction did the insect turn?", ["no insect there", "clockwise", "not moving", "anticlockwise"], "synthetic",
     "rotation_sense"),
    ("Is the dancer spinning clockwise or counter-clockwise?", ["clockwise", "counter-clockwise", "not spinning", "x"],
     "davis", "rotation_sense"),
    ("Is the dog coming closer or moving further away?", ["coming closer", "going further away", "left", "not moving"],
     "davis", "cam_subject_range"),
    ("How many people are moving to the left?", ["0", "1", "2", "3"], "davis", "count_with_predicate"),
    ("How many times does the batter hit the ball to the left?", ["0", "1", "2", "3"], "davis", "repetition_count"),
    ("Where is the butter pack placed with respect to the bread slices?", ["above", "in front", "to the right", "to the left"],
     "ego4d", "relative_position"),
    ("Which direction does the person move the vegetable with respect to the sink?", ["forward", "left", "right", "backward"],
     "ego4d", "landmark_relation"),
    ("Which direction is the receipt moving with respect to the camera?", ["right", "left", "toward", "away"], "ego4d",
     "subject_direction"),                                                   # the camera is a frame, not a landmark
    ("Which hand does he use to drink from the cup?", ["Both", "Right", "Left", "Not enough information"], "ego4d",
     "hand_attribution"),
    ("At the end of the video, which way is the koala facing?", ["right", "left", "no koala there", "neither"], "davis",
     "camera_viewpoint"),
    ("Is the camera rotating clockwise or counter-clockwise around the girl?", ["clockwise", "counter-clockwise", "x", "y"],
     "davis", "orbit_direction"),
    ("Does the basketball hit the backboard?", ["Yes", "No", "It's unclear", "There is no basketball"], "ego4d", "semantic"),
])
def test_kind_rules(question, options, source, kind):
    assert T.classify(question, options, source)["kind"] == kind


def test_how_many_degrees_is_not_counting():
    q = "After placing down the first log the worker rotated it. In what direction did he rotate it by and by how many degrees?"
    assert T.motion_type(q) == "rotational"
    assert T.classify(q, ["Clockwise and 90 degrees."] * 4, "ego4d")["kind"] == "rotation_sense"


def test_program_adds_actor_existence_and_window_steps():
    labels = T.classify("Did the person move to the left or right to pick up the bowl at the end of the video?",
                        ["Moved to the left", "No bowl in the video.", "Not enough information", "Moved to the right"], "ego4d")
    steps = T.program(labels)
    assert steps[0].startswith("resolve_actor") and steps[1].startswith("exists") and steps[2] == "window(end)"
    assert steps[-1] == "final_answer(options)"
    assert T.program(T.classify("How is the jet-ski rotating?", ["clockwise", "counter-clockwise", "not rotating", "x"],
                                "davis"))[0].startswith("ground")


def test_kinds_match_agent_v5():
    from d4rt_agent.agent_v5.kinds import KIND_BY_NAME

    for k in T.KINDS:
        assert k.name in T.SUBTASKS
        if k.name == "other":
            continue
        assert (k.name in KIND_BY_NAME) != k.new, k.name                   # new <=> absent from AgentV5
        if not k.new:
            assert KIND_BY_NAME[k.name].family == k.family, k.name


needs_data = pytest.mark.skipif(not (ROOT / "QA_local" / "real_mc.json").exists(), reason="VLM4D not on this machine")


@needs_data
def test_every_question_is_labelled_and_the_proportions_match_the_paper():
    from d4rt_agent.agent_v5.vlm4d_report import is_fp
    from vlm4d_dataset_exploration.vlm4d_data import labelled_rows

    rows = labelled_rows(ROOT)
    assert len(rows) == 1816 and all(r["kind"] in T.KIND_BY_NAME and r["paper_type"] in T.PAPER_TYPES for r in rows)
    syn = [r for r in rows if r["split"] == "synthetic"]
    assert sum(r["paper_type"] == "false_positive" for r in syn) == sum(is_fp(r["gold"]) for r in syn) == 45
    real = [r for r in rows if r["split"] == "real"]
    for t, paper in (("translational", 55), ("rotational", 19), ("counting", 17)):       # paper Fig. 2, %
        assert abs(100 * sum(r["paper_type"] == t for r in real) / len(real) - paper) < 3, t
    assert sum(r["kind"] == "other" for r in rows) <= 0.01 * len(rows)
    assert all(r["ego_wearer"] <= (r["source"] == "ego4d") for r in rows)          # only Ego4D has a wearer
