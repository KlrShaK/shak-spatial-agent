import json

from d4rt_agent.agent_v5_jev import report as R


def _answer(path, qid, letter, gt, kind="camera_path", cap_only=None):
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{qid}.json").write_text(json.dumps({
        "question_id": qid, "letter": letter, "correct": letter == gt, "kind": kind, "family": "camera",
        "status": "answered", "tools": [{"tool": "camera_motion"}], "measuring_steps": 1, "peak_tokens": 0,
        "caption_only_letter": cap_only, "reference": {"answer": gt, "group": "g", "task": "cameraqa", "dataset": "davis"}}))


def test_report_pairs_against_v5_and_the_ablation(tmp_path):
    mine, theirs = tmp_path / "mine", tmp_path / "v5"
    for i, (m, v, cap) in enumerate([("A", "B", "B"), ("A", "A", "A"), ("C", "A", "A"), ("B", "B", "C")]):
        qid = f"clipx_{i}"
        _answer(mine / "answers", qid, m, "A" if i < 3 else "B", cap_only=cap)
        _answer(theirs / "answers", qid, v, "A" if i < 3 else "B")
    text = R.report(mine, theirs, None)
    assert "**75.0%** over 4 questions" in text
    assert "AgentV5 + Qwen3-VL-8B" in text and "caption only" in text
    assert "| camera_path | 4 | 75.0% | 75.0% | +0.0% |" in text


def test_usage_cost(tmp_path):
    (tmp_path / "decisions").mkdir()
    (tmp_path / "decisions" / "usage.jsonl").write_text(
        json.dumps({"input_tokens": 1_000_000}) + "\n" + json.dumps({"input_tokens": 500_000}) + "\n")
    assert "2 API calls, 1,500,000 input tokens" in R.usage(tmp_path)
