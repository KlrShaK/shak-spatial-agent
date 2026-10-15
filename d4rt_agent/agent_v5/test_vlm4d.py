"""VLM4D loader for the AgentV5 runner: GPU-free."""

import json
from argparse import Namespace
from pathlib import Path

import pytest

from d4rt_agent.agent_v5.run import parse_args, select, vlm4d_entries

REAL_ROOT = Path("/cluster/work/igp_psr/spanwar/datasets/VLM4D")


def _tree(tmp_path, real, synthetic):
    (tmp_path / "QA_local").mkdir(parents=True)
    for name, rows in (("real_mc.json", real), ("synthetic_mc.json", synthetic)):
        (tmp_path / "QA_local" / name).write_text(json.dumps(rows))
    return Namespace(vlm4d_root=tmp_path)


def _row(tmp_path, folder, video, n, choices, answer):
    return {"id": f"validation_{n}", "video": str(tmp_path / folder / video), "question_type": "multiple-choice",
            "question": "q?", "choices": choices, "answer": answer}


def test_answer_text_becomes_a_letter_and_numbers_are_text(tmp_path):
    args = _tree(tmp_path,
                 [_row(tmp_path, "videos_real/davis", "a_b.mp4", 1, {"A": "left", "B": 2, "C": "right", "D": "up"}, 2),
                  _row(tmp_path, "videos_real/ego4d", "x.mp4", 2, {"A": "same", "B": "same", "C": "other", "D": "z"}, "same")],
                 [_row(tmp_path, "videos_synthetic", "synth_001.mp4", 1, {"A": "l", "B": "r", "C": "u", "D": "d"}, "r")])
    e = vlm4d_entries(args)
    assert [x["answer"] for x in e] == ["B", "A", "B"]
    assert e[0]["options"]["B"] == "2"                        # a numeric option is shown as text
    assert e[1]["answer_also"] == ["B"]                       # two options with the same text: either counts
    assert e[0]["answer_also"] == [] and e[2]["answer_also"] == []


def test_ids_group_by_video_and_sources(tmp_path):
    args = _tree(tmp_path,
                 [_row(tmp_path, "videos_real/davis", "a_b.mp4", 1, {"A": "x", "B": "y", "C": "z", "D": "w"}, "x"),
                  _row(tmp_path, "videos_real/davis", "a_b.mp4", 2, {"A": "x", "B": "y", "C": "z", "D": "w"}, "y")],
                 [_row(tmp_path, "videos_synthetic", "synth_001.mp4", 1, {"A": "x", "B": "y", "C": "z", "D": "w"}, "z")])
    e = vlm4d_entries(args)
    assert [x["id"] for x in e] == ["vlm4d_real_davis-a-b_1", "vlm4d_real_davis-a-b_2", "vlm4d_synthetic_synth-001_1"]
    # the report resamples by qid.rsplit("_", 1)[0]: the two questions on one video are one cluster
    assert len({x["id"].rsplit("_", 1)[0] for x in e}) == 2
    assert [x["group"] for x in e] == ["davis", "davis", "synthetic"] and [x["task"] for x in e] == ["real", "real", "synthetic"]
    assert e[0]["relative_path"] == "videos_real/davis/a_b.mp4"


def test_an_unmatched_answer_is_an_error(tmp_path):
    args = _tree(tmp_path, [_row(tmp_path, "videos_real/davis", "a.mp4", 1, {"A": "x", "B": "y", "C": "z", "D": "w"}, "nope")], [])
    with pytest.raises(ValueError):
        vlm4d_entries(args)


def test_vlm4d_is_a_bench_and_selection_keeps_a_videos_questions_together(tmp_path):
    args = parse_args(["--bench", "vlm4d", "--results-dir", str(tmp_path / "r"), "--shard", "0/2"])
    assert args.bench == "vlm4d"
    rows = [{"id": f"q{i}", "video_path": f"/v/{i // 2}.mp4", "dataset": "vlm4d_real", "video": "x", "group": "g"} for i in range(8)]
    mine = select(rows, args)
    assert {r["video_path"] for r in mine} == {"/v/0.mp4", "/v/2.mp4"} and len(mine) == 4


@pytest.mark.skipif(not (REAL_ROOT / "QA_local" / "real_mc.json").exists(), reason="VLM4D not on this machine")
def test_the_real_files():
    e = vlm4d_entries(Namespace(vlm4d_root=REAL_ROOT))
    assert len(e) == 1816 and sum(x["task"] == "real" for x in e) == 1371 and sum(x["task"] == "synthetic" for x in e) == 445
    assert len({x["id"] for x in e}) == 1816
    assert all(Path(x["video_path"]).exists() for x in e)
    assert all(len(x["options"]) == 4 and x["answer"] in x["options"] for x in e)
    assert len({x["video_path"] for x in e}) == 1000                 # 600 real + 400 synthetic videos
    assert sum(bool(x["answer_also"]) for x in e) == 1


# --------------------------------------------------------------------------- #
# agreement with the original VLM4D repo (skipped where it is not on disk)
# --------------------------------------------------------------------------- #

from d4rt_agent.agent_v5 import vlm4d_report as VR

REPO = Path("/cluster/work/igp_psr/spanwar/VLM4D")
needs_repo = pytest.mark.skipif(not (REPO / VR.PAPER).exists() or not REAL_ROOT.exists(), reason="VLM4D repo or data missing")


@needs_repo
def test_ids_and_gold_answers_match_the_papers_graded_files():
    entries = vlm4d_entries(Namespace(vlm4d_root=REAL_ROOT))
    mine = {(e["task"], int(e["id"].rsplit("_", 1)[-1])): e for e in entries}
    theirs = VR.paper_rows(REPO, "cot", "gpt-4o_32frame.json")
    assert set(mine) == set(theirs)                               # the same 1816 (split, id) keys
    for key, row in theirs.items():
        e = mine[key]
        assert str(row["gold"]).strip() == e["options"][e["answer"]], key      # my letter mapping == their ground truth
    # their exo/ego boundary (id 922) is exactly davis + youtube-vos vs ego4d
    real = [e for e in entries if e["task"] == "real"]
    assert {e["group"] for e in real if int(e["id"].rsplit("_", 1)[-1]) <= VR.EXO_MAX_ID} == {"davis", "youtube-vos"}
    assert {e["group"] for e in real if int(e["id"].rsplit("_", 1)[-1]) > VR.EXO_MAX_ID} == {"ego4d"}


@needs_repo
def test_breakdown_equals_the_repos_own_statistics_script(tmp_path):
    pd = pytest.importorskip("pandas")
    src = (REPO / "acc_final_statistics.py").read_text().split("# === Example usage ===")[0]
    ns: dict = {"os": __import__("os"), "json": json, "pd": pd}
    exec(src.replace("import pandas as pd", ""), ns)                  # their compute_accuracy, unmodified
    for prompt in ("cot", "direct-output"):
        out = tmp_path / f"{prompt}.csv"
        ns["compute_accuracy"](str(REPO / VR.PAPER / f"real_mc_{prompt}"), str(REPO / VR.PAPER / f"synthetic_mc_{prompt}"), str(out))
        official = pd.read_csv(out).set_index("Model (JSON File)")
        assert len(official) >= 20
        for fname, row in official.iterrows():
            ours = VR.breakdown(list(VR.paper_rows(REPO, prompt, fname).values()))
            assert round(ours["overall"], 2) == row["Overall Accuracy (%)"], (prompt, fname)
            assert round(ours["exo"], 2) == row["Real Exo-centric Accuracy (%)"]
            assert round(ours["ego"], 2) == row["Real Ego-centric Accuracy (%)"]
            assert round(ours["synth_fp"], 2) == row["Synthetic FP Accuracy (%)"]
            assert round(ours["synth_directional"], 2) == row["Synthetic Directional Accuracy (%)"]
