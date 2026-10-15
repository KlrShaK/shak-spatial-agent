"""CPU regressions for sampling, sizing, reporting and failed submissions."""

import json
import os
import subprocess

import pytest

from d4rt_agent.agent_v5.research import subset_eval as E


def population():
    return [dict(id=f"{source}-{kind}-{i}", source=source, kind=kind,
                 stratum=f"{source}:{kind}", clip=f"{source}/{i}", answer=i % 4, correct=True)
            for source in ("large", "small") for kind in ("common", "rare")
            for i in range(30 if source == "large" and kind == "common" else 3)]


def test_order_is_answer_blind_reproducible_and_covers_rare_cells():
    rows = population()
    a = E.stratified_order(rows, 123)
    b = E.stratified_order([{**r, "answer": "changed", "correct": False} for r in reversed(rows)], 123)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert len({r["id"] for r in a}) == len(rows)
    assert {r["stratum"] for r in a[:4]} == {r["stratum"] for r in rows}
    assert [r["id"] for r in a] != [r["id"] for r in E.stratified_order(rows, 124)]


def test_sizing_accounts_for_elapsed_uncertainty_and_overhead():
    spec = dict(order=[dict(id=str(i), stratum="s") for i in range(20)], pilot=4, population={"s": 20})
    plan = dict(budget_seconds=1000, overhead_seconds=100, uncertainty=1.25)
    n, projected = E.sized_units(spec, 200, {str(i): 50 for i in range(4)}, plan)
    assert n == 15 and projected == 987.5
    with pytest.raises(RuntimeError, match="runtime budget"):
        E.sized_units(spec, 950, {str(i): 50 for i in range(4)}, plan)


def test_dyn_selection_retains_whole_clips_and_pilot():
    spec = dict(order=[dict(id=f"c{i}", question_ids=[f"c{i}-q{j}" for j in range(7)]) for i in range(120)])
    pilot = E.selection(spec, 8)
    final = E.selection(spec, 23)
    assert len(pilot) == 56 and len(final) == 161 and final[:56] == pilot


def test_population_weights_and_dsi_shared_video_clusters():
    rows = [dict(stratum="large", clip="one", correct=True), dict(stratum="rare", clip="two", correct=False)]
    result = E.weighted_summary(rows, {"large": 90, "rare": 10}, draws=100)
    assert result["accuracy"] == 0.5
    assert result["population_weighted_accuracy"] == 0.9
    duplicated = E.weighted_summary(rows * 4, {"large": 90, "rare": 10}, draws=100)
    assert duplicated["source_videos"] == 2 and duplicated["completed"] == 8
    assert result["weighted_source_video_bootstrap_95_ci"] == duplicated["weighted_source_video_bootstrap_95_ci"]


def test_second_submission_failure_records_and_cancels_first(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    plan_path = tmp_path / "plan.json"
    E.write_json(plan_path, dict(source=str(source), results=str(tmp_path / "results"), dependencies=str(tmp_path)))
    calls = []

    def fake_command(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "scancel":
            return ""
        if len([a for a in calls if a[0] == "sbatch"]) == 1:
            return "12345"
        raise RuntimeError("controller unreachable")

    monkeypatch.setattr(E, "command", fake_command)
    with pytest.raises(RuntimeError, match="unreachable"):
        E.submit_pair(plan_path, "dsi")
    record = json.loads((tmp_path / "results/dsi/submission.json").read_text())
    assert record["phase"] == "submission_failed" and record["jobs"] == {"a100": "12345"}
    assert ["scancel", "12345"] in calls


def test_submission_rejects_missing_job_id(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    E.write_json(plan, dict(source=str(tmp_path), results=str(tmp_path / "results"), dependencies=str(tmp_path)))
    monkeypatch.setattr(E, "command", lambda *a, **k: "")
    with pytest.raises(RuntimeError, match="job ID"):
        E.submit_pair(plan, "vlm4d")
    assert E.read(tmp_path / "results/vlm4d/submission.json")["phase"] == "submission_failed"


def test_controller_starts_dyn_only_after_a_worker_finishes(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    E.write_json(plan, dict(results=str(tmp_path), benchmarks={b: {} for b in ("dsi", "vlm4d", "dynbench")}))
    submitted, checks = [], []
    iteration = [0]

    def submit(path, bench):
        submitted.append(bench)
        E.write_json(tmp_path / bench / "submission.json", {})
        if bench == "dynbench":
            assert iteration[0] == 1

    def status(plan, bench):
        phase = "waiting_for_slot" if bench not in submitted else "evaluation"
        if iteration[0] >= 1 and bench == "vlm4d" or iteration[0] >= 2:
            phase = "complete"
        checks.append((iteration[0], bench, phase))
        return dict(phase=phase)

    monkeypatch.setattr(E, "submit_pair", submit)
    monkeypatch.setattr(E, "benchmark_status", status)
    monkeypatch.setattr(E, "save_report", lambda p: None)
    monkeypatch.setattr(E.time, "sleep", lambda seconds: iteration.__setitem__(0, iteration[0] + 1))
    E.control(plan)
    assert submitted == ["dsi", "vlm4d", "dynbench"]
    assert E.read(tmp_path / "status.json")["phase"] == "complete"


@pytest.mark.parametrize("accounting_error", (RuntimeError("database unreachable"), subprocess.TimeoutExpired("sacct", 45)))
def test_accounting_outage_uses_live_queue_and_preserves_worker_state(tmp_path, monkeypatch, accounting_error):
    directory = tmp_path / "dynbench"
    E.write_json(directory / "submission.json", dict(jobs={"a100": "123", "blackwell": "456"}, phase="submitted", attempt="a"))
    E.write_json(directory / "worker_state.json", dict(phase="evaluation", planned_questions=56))
    calls = []

    def fake_command(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "sacct":
            raise accounting_error
        assert argv[0] == "squeue"
        return "123|RUNNING|01:00|node"

    monkeypatch.setattr(E, "command", fake_command)
    monkeypatch.setattr(E, "MAIN_REPO", tmp_path)
    result = E.benchmark_status(dict(results=str(tmp_path)), "dynbench")
    assert result["phase"] == "evaluation" and result["scheduler_source"] == "squeue"
    assert result["scheduler_error"] and result["scheduler"].startswith("123|RUNNING")
    assert [a[0] for a in calls] == ["sacct", "squeue"]


def test_scheduler_outage_is_reported_without_discarding_completed_work(tmp_path, monkeypatch):
    directory = tmp_path / "dsi"
    E.write_json(directory / "submission.json", dict(jobs={"a100": "123", "blackwell": "456"}, phase="submitted", attempt="a"))
    E.write_json(directory / "worker_state.json", dict(phase="complete", planned_questions=252))

    def fail(argv, **kwargs):
        raise RuntimeError(f"{argv[0]} unavailable")

    monkeypatch.setattr(E, "command", fail)
    monkeypatch.setattr(E, "MAIN_REPO", tmp_path)
    result = E.benchmark_status(dict(results=str(tmp_path)), "dsi")
    assert result["phase"] == "complete" and result["scheduler_source"] == "unavailable"
    assert "sacct unavailable" in result["scheduler_error"] and "squeue unavailable" in result["scheduler_error"]


def extension_plan(tmp_path):
    order = [dict(id=f"q{i}", stratum="slow" if i % 2 else "fast", clip=f"c{i}") for i in range(10)]
    plan = dict(results=str(tmp_path), source_sha256="s", uncertainty=1.25, overhead_seconds=100,
                benchmarks={"vlm4d": dict(order=order, ceiling=10)})
    directory = tmp_path / "vlm4d"
    E.write_json(directory / "worker_state.json", dict(phase="complete", job_id="7"))
    E.write_json(directory / "final_selection.json", dict(question_ids={"vlm4d": ["q0", "q1"]}, units=2, source_sha256="s"))
    for qid, seconds in (("q0", 10), ("q1", 30)):
        E.write_json(directory / "answers" / f"{qid}.json", dict(wall_seconds=seconds))
    E.write_json(tmp_path / "plan.json", plan)
    return tmp_path / "plan.json", directory


def test_extension_is_a_nested_prefix_sized_by_observed_stratum_rates(tmp_path):
    plan, directory = extension_plan(tmp_path)
    # 1.25 * (load 20 + fast 10 + slow 30 + fast 10) + 100 = 187.5; a further slow unit would exceed 200.
    n, projected = E.extend(plan, "vlm4d", 200, load_seconds=20)
    final = E.read(directory / "final_selection.json")
    assert n == 5 and projected == 187.5
    assert final["question_ids"]["vlm4d"] == [f"q{i}" for i in range(5)]
    assert E.read(directory / "final_selection_2units.json")["units"] == 2
    assert E.read(directory / "worker_state_7.json")["phase"] == "complete"
    capped, _ = extension_plan(tmp_path / "capped")
    assert E.extend(capped, "vlm4d", 10_000, max_units=6, load_seconds=0)[0] == 6


def test_extension_refuses_an_unfinished_worker(tmp_path):
    plan, directory = extension_plan(tmp_path)
    E.write_json(directory / "worker_state.json", dict(phase="evaluation"))
    with pytest.raises(RuntimeError, match="completed worker"):
        E.extend(plan, "vlm4d", 1000)


def test_single_partition_submission_passes_time_limit(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    E.write_json(plan, dict(source=str(tmp_path), results=str(tmp_path / "results"), dependencies=str(tmp_path)))
    calls = []

    def fake_command(argv, **kwargs):
        calls.append(argv)
        return "55" if argv[0] == "sbatch" else "JobId=55 JobState=PENDING"

    monkeypatch.setattr(E, "command", fake_command)
    record = E.submit_pair(plan, "dynbench", ("blackwell",), "09:30:00")
    sbatch = [a for a in calls if a[0] == "sbatch"]
    assert record["jobs"] == {"blackwell": "55"} and len(sbatch) == 1
    assert "--time=09:30:00" in sbatch[0] and sbatch[0][-1].endswith("run_agent_v5_blackwell.slurm")
    assert (tmp_path / "results/dynbench/.race" / f"{record['attempt']}.jobs").read_text() == "55\n"


def test_extended_worker_progress_counts_from_its_own_start(tmp_path, monkeypatch):
    directory = tmp_path / "vlm4d"
    E.write_json(directory / "submission.json", dict(jobs={"a100": "9"}, phase="submitted", attempt="a"))
    E.write_json(directory / "worker_state.json", dict(phase="evaluation", planned_questions=4, started_at=E.utc()))
    E.write_json(directory / "answers/old.json", dict(wall_seconds=5, status="answered"))
    os.utime(directory / "answers/old.json", (0, 0))           # an answer from the earlier job, long ago
    monkeypatch.setattr(E, "command", lambda argv, **k: "9|RUNNING|00:10|node")
    monkeypatch.setattr(E, "MAIN_REPO", tmp_path)
    result = E.benchmark_status(dict(results=str(tmp_path)), "vlm4d")
    assert result["last_progress_age_seconds"] < 60


def test_unanswered_recovery_is_counted_as_completed_and_incorrect():
    rows = [dict(stratum="s", clip="a", correct=True, status="answered", answered=True),
            dict(stratum="s", clip="b", correct=False, status="unanswered", answered=False)]
    result = E.weighted_summary(rows, {"s": 2}, draws=100)
    assert result["completed"] == 2 and result["answered"] == 1 and result["errors"] == 0
    assert result["accuracy"] == 0.5 and result["status_counts"] == {"answered": 1, "unanswered": 1}
