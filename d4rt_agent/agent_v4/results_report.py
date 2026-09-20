"""Render an agent-v4 run as a readable markdown report.

Mirrors `results/dsi_bench/DSI_BENCH_RESULTS.md` so the two can be read side by
side, but the body is different because the agent is: v2 emitted a free-form tool
trace, v4 emits a bucket choice, a subject, and one host-generated measurement.
What is worth showing is therefore *which* measurement the agent asked for, what
came back, and whether the answer follows from it.

Every question carries its measurement verbatim. A reader who disagrees with an
answer can see whether the agent misread the numbers or the numbers were wrong,
which is the distinction the whole design turns on.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from d4rt_agent.agent_v4.buckets import CATE_TO_BUCKET
from d4rt_agent.dsi_bench_data import parse_choice_letter, read_manifest

# Where the clip previews already live. The v4 runs answer the same 25 questions
# as the v2 report, so its rendered gifs and mp4s are reused rather than rebuilt.
ASSET_DIR = Path("d4rt_agent/results/dsi_bench")


def letter_of(text: Any, options: Mapping[str, str]) -> str | None:
    """Parse an option letter, tolerating the wrappers models actually emit.

    The tool-free control is told to reply `<answer>B</answer>` and often replies
    `<B>`; the stored `parsed_letter` for that run is None on 24 of 25 rows, which
    reads as 0% accuracy and is really 52%. Parsing every run through one function
    here is what stops that becoming a reported number.
    """

    if not isinstance(text, str) or not text.strip():
        return None
    found = parse_choice_letter(text, list(options) or None)
    if found:
        return found
    match = re.search(r"[<\[(\"']?\s*\b([A-D])\b\s*[>\])\"']?", text.strip())
    return match.group(1) if match else None


def _load(directory: Path) -> dict[str, dict[str, Any]]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))}


def _slug(record: Mapping[str, Any]) -> str:
    from d4rt_agent.dsi_bench_data import video_slug
    return video_slug(record["relative_path"])


def _anchor(heading: str) -> str:
    """GitHub's heading slug: lowercase, punctuation dropped, spaces hyphenated.

    Worth doing properly -- the v2 report's hand-built anchors (`#1-camerabench-task-0`
    against a heading that slugifies to `#1-camerabench--task-0-objstatic-cam`) do not
    resolve, so its index does not navigate.
    """

    slug = heading.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    return re.sub(r"\s", "-", slug)


def _heading(index: int, ref: Mapping[str, Any]) -> str:
    return f"{index}. {ref.get('dataset')} — task {ref.get('cate')}: {ref.get('category_name')}"


def _truncate(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _config_table(manifest: Mapping[str, Any], meta: Mapping[str, Any],
                  records: Sequence[Mapping[str, Any]]) -> list[str]:
    sampling = manifest.get("sampling", {})
    counts = manifest.get("counts", {})
    measured = sum(1 for r in records if r.get("measurement_available"))
    return [
        "## Run configuration", "", "| | |", "| --- | --- |",
        f"| Questions | {len(records)} |",
        f"| Sampling design | {sampling.get('design')} — {sampling.get('description', '')} |",
        f"| Sampling seed | `{sampling.get('seed')}` (requested `{sampling.get('requested_seed')}`) |",
        f"| Source CSV | `{manifest.get('csv_path')}` |",
        f"| CSV sha256 | `{str(manifest.get('csv_sha256'))[:16]}…` |",
        f"| Ground-truth histogram | {counts.get('gt_histogram')} |",
        f"| Per dataset | {counts.get('per_dataset')} |",
        f"| Per task | {counts.get('per_category')} |",
        f"| Qwen model | `{meta.get('qwen_model')}` |",
        f"| D4RT checkpoint | `{meta.get('d4rt_checkpoint')}` |",
        f"| Agent | v4 — dynamic prompt composition |",
        f"| Max steps | {meta.get('max_steps')} |",
        f"| Run created | {meta.get('created_at')} |",
        f"| SLURM job | `{meta.get('slurm_job_id')}` |",
        f"| Questions with a usable measurement | {measured} / {len(records)} |",
        "",
    ]


def _headline(records: Sequence[Mapping[str, Any]],
              baseline: Mapping[str, dict[str, Any]]) -> list[str]:
    scored = [(r, letter_of(r.get("answer_text"), (r.get("reference") or {}).get("options") or {}))
              for r in records]
    correct = [r for r, got in scored if got and got == (r.get("reference") or {}).get("gt")]
    measured = [r for r in records if r.get("measurement_available")]
    correct_measured = [r for r in correct if r.get("measurement_available")]
    unmeasured = [r for r in records if not r.get("measurement_available")]
    correct_unmeasured = [r for r in correct if not r.get("measurement_available")]

    base_hits = sum(
        1 for r in records
        if (bl := baseline.get(r["question_id"]))
        and letter_of(bl.get("raw_response"), (r.get("reference") or {}).get("options") or {})
        == (r.get("reference") or {}).get("gt")
    )
    total = len(records)
    out = [
        "## Headline", "",
        "| | correct | |",
        "| --- | ---: | --- |",
        f"| **Agent v4** | **{len(correct)}/{total}** ({len(correct)/total:.0%}) | this run |",
        f"| Tool-free Qwen, same 32 frames | {base_hits}/{total} ({base_hits/total:.0%}) | control |",
        "",
        "Scored separately, because accuracy alone cannot tell a measurement from a guess:", "",
        f"- **{len(correct_measured)}/{len(measured)}** "
        f"({len(correct_measured)/len(measured):.0%} of questions where a measurement was available)",
        f"- **{len(correct_unmeasured)}/{len(unmeasured)}** "
        f"({len(correct_unmeasured)/max(len(unmeasured),1):.0%} of questions where it was not) — "
        "answered from the frames alone, with the pipeline contributing nothing",
        "",
    ]
    if len(unmeasured) and len(correct_unmeasured) / len(unmeasured) > (
            len(correct_measured) / max(len(measured), 1)):
        out += [
            f"> On this sample the unmeasured rows scored *higher*, on n={len(unmeasured)}. "
            "Do not read that as the measurement hurting: a 200-question uniform draw of the "
            "same agent gives 71% measured against 57% unmeasured. At four questions per task "
            "these splits are dominated by which particular clips happened to fail.",
            "",
        ]

    by_cate: dict[int, list[bool]] = collections.defaultdict(list)
    for record, got in scored:
        ref = record.get("reference") or {}
        by_cate[ref.get("cate")].append(got == ref.get("gt"))
    out += ["### By task", "",
            "| cate | task | measurement the taxonomy expects | n | correct |",
            "| ---: | --- | --- | ---: | --- |"]
    for cate in sorted(c for c in by_cate if c is not None):
        hits = by_cate[cate]
        name = next((r["reference"]["category_name"] for r in records
                     if (r.get("reference") or {}).get("cate") == cate), "?")
        out.append(f"| {cate} | {name} | `{CATE_TO_BUCKET.get(cate)}` | {len(hits)} | "
                   f"{sum(hits)}/{len(hits)} ({sum(hits)/len(hits):.0%}) |")

    agreed = sum(1 for r in records
                 if r.get("chosen_bucket") == (r.get("reference") or {}).get("expected_bucket"))
    out += ["", "### Did the agent choose the measurement the taxonomy expects?", "",
            f"**{agreed}/{total}** ({agreed/total:.0%}). The agent is never shown the `cate` "
            "label — it classifies from the question text alone, so this is the "
            "classification step scoring itself against the taxonomy.", ""]

    reasons = collections.Counter(
        (r.get("measurement_unavailable_reason") or "?").split(":")[0] for r in unmeasured)
    if reasons:
        out += ["### Why a measurement was unavailable", "", "| n | reason |", "| ---: | --- |"]
        out += [f"| {n} | {reason} |" for reason, n in reasons.most_common()]
        out.append("")
    return out


def _glance(records: Sequence[Mapping[str, Any]],
            baseline: Mapping[str, dict[str, Any]]) -> list[str]:
    out = ["## At a glance", "",
           "| # | Dataset | Task | GT | Bucket chosen | Measured | Agent answer | Qwen-only |",
           "| ---: | --- | --- | :---: | --- | :---: | --- | :---: |"]
    for index, record in enumerate(records, start=1):
        ref = record.get("reference") or {}
        options = ref.get("options") or {}
        got = letter_of(record.get("answer_text"), options)
        gt = ref.get("gt")
        bl = baseline.get(record["question_id"]) or {}
        bl_letter = letter_of(bl.get("raw_response"), options)
        mark = lambda letter: f"**{letter}**" if letter and letter == gt else (letter or "—")
        anchor = "#" + _anchor(_heading(index, ref))
        out.append(
            f"| [{index}]({anchor}) | {ref.get('dataset')} | {ref.get('category_name')} | "
            f"**{gt}** | `{record.get('chosen_bucket')}` | "
            f"{'yes' if record.get('measurement_available') else '**no**'} | "
            f"{mark(got)}: {_truncate(options.get(got, '—'), 34)} | {mark(bl_letter)} |"
        )
    out.append("")
    return out


def _metrics_block(record: Mapping[str, Any]) -> list[str]:
    """The measurement's own table, so an answer can be checked against it."""
    from d4rt_agent.agent_v4.orchestrator import _render_metrics

    rendered = _render_metrics((record.get("report") or {}).get("metrics") or {})
    if not rendered:
        return []
    return ["<details><summary>Measurement detail</summary>", "", "```", rendered, "```", "",
            "</details>", ""]


def _question_section(index: int, record: Mapping[str, Any],
                      baseline: Mapping[str, dict[str, Any]],
                      questions: Mapping[str, str]) -> list[str]:
    ref = record.get("reference") or {}
    options = ref.get("options") or {}
    gt = ref.get("gt")
    got = letter_of(record.get("answer_text"), options)
    slug = _slug(record)
    report = record.get("report") or {}

    out = [f"## {_heading(index, ref)}", ""]
    gif = ASSET_DIR / "gifs" / f"{slug}.gif"
    if gif.exists():
        out += [f"![clip {index}](../dsi_bench/gifs/{slug}.gif)", ""]
    video = ASSET_DIR / "videos" / f"{slug}.mp4"
    if video.exists():
        out += [f"[▶ full-quality MP4](../dsi_bench/videos/{slug}.mp4) · "
                f"source: `{record.get('relative_path')}`", ""]

    out += [f"**Question.** {questions.get(record['question_id'], '')}", ""]
    for letter in sorted(options):
        marker = " ← **ground truth**" if letter == gt else ""
        out.append(f"- **{letter}.** {options[letter]}{marker}")
    out.append("")

    out += [f"**Step 1 — measurement chosen.** `{record.get('chosen_bucket')}`"
            + ("" if record.get("chosen_bucket") == ref.get("expected_bucket")
               else f" — the taxonomy expects `{ref.get('expected_bucket')}`"), ""]
    if record.get("subject"):
        detail = f"**Step 2 — subject named.** `{record['subject']}`"
        if record.get("grounding_ambiguous"):
            detail += (f" (the segmenter was unsure; the agent picked candidate "
                       f"#{record.get('chosen_candidate_index')} from the crops)")
        if (record.get("subject_attempts") or 0) > 1:
            detail += f" — after {record['subject_attempts']} attempts"
        out += [detail, ""]

    if report.get("available"):
        out += ["**Measurement returned.**", "",
                "> " + "\n> ".join(str(report.get("text", "")).split("\n")), ""]
        out += _metrics_block(record)
    else:
        out += ["**Measurement UNAVAILABLE.** "
                f"{report.get('unavailable_reason') or record.get('measurement_unavailable_reason')}",
                "", "The agent answered from the frames alone; the pipeline contributed nothing "
                "to this row.", ""]

    verdict = "matches ground truth" if got == gt else f"**ground truth is {gt}**"
    out += ["**Agent answer.**", "", f"> {record.get('answer_text')}", "", f"*({verdict})*", ""]
    if record.get("limitations"):
        out += [f"*Agent-stated limitations:* {record['limitations']}", ""]

    bl = baseline.get(record["question_id"])
    if bl:
        bl_letter = letter_of(bl.get("raw_response"), options)
        agree = "matches ground truth" if bl_letter == gt else f"ground truth is {gt}"
        out += ["**Tool-free Qwen baseline.**", "",
                f"> **{bl_letter or '—'}** — {options.get(bl_letter, 'unparsable')} ({agree}). "
                f"Raw reply: `{_truncate(bl.get('raw_response', ''), 60)}`", ""]

    trace = record.get("trace") or []
    out += [f"<details><summary>Agent trace ({record.get('steps_used')} steps, "
            f"{record.get('wall_seconds', 0):.0f}s)</summary>", "",
            "| Step | Expected | Status | Action | Justification |",
            "| ---: | --- | --- | --- | --- |"]
    for attempt in trace:
        parsed = attempt.get("parsed_action") or {}
        args = parsed.get("arguments") or {}
        detail = args.get("bucket") or args.get("subject") or (
            f"index {args['index']}" if "index" in args else "")
        note = attempt.get("error") or args.get("justification") or ""
        out.append(f"| {attempt.get('step')} | `{attempt.get('expected_action')}` | "
                   f"{attempt.get('status')} | {parsed.get('action', '—')} {detail} | "
                   f"{_truncate(note, 90)} |")
    out += ["", "</details>", ""]

    if trace:
        out += ["<details><summary>Agent reasoning, verbatim</summary>", ""]
        for attempt in trace:
            out += [f"**Step {attempt.get('step')} ({attempt.get('expected_action')}):**", "",
                    "```", str(attempt.get("raw_response", "")).strip()[:1800], "```", ""]
        out += ["</details>", ""]
    out.append("---")
    out.append("")
    return out


PREAMBLE = """# DSI-Bench evaluation — agent v4 (dynamic prompt composition)

25 questions from DSI-Bench (`std` split), sampled evenly across the benchmark's
5 source datasets and 6 motion-reasoning tasks — the same manifest the earlier
`simple_v2` runs used, so the two are directly comparable.

Agent v4 replaces free-form tool use with a fixed, host-driven sequence. The agent
classifies the question into one of four measurement buckets, names the subject if
that bucket needs one, and is then handed **that one bucket's measurement** and the
contract for reading it. It never issues its own D4RT queries.

Each section below shows the measurement the agent asked for, what came back, and
the answer it gave — so a disagreement can be traced to either a misread number or
a wrong one.

> **Read this as a diagnostic, not a benchmark score.** The sample is balanced by
> design — each task carries 4–5 questions, while the full benchmark is 33%
> *Obj:moving cam* and 31% *Cam:dynamic scene* — so it shows where the pipeline
> helps and where it does not, rather than estimating DSI-Bench accuracy. With 25
> four-way questions, chance alone lands around 6 correct. A 200-question uniform
> draw is reported separately for the accuracy estimate.

> **Accuracy alone overstates the pipeline.** Where a measurement is unavailable the
> agent still answers, from the frames alone. Those rows can be right by ordinary
> visual reasoning and are marked **no** in the *Measured* column; the
> accuracy-where-measured figure is the one that reflects this architecture.

> **Positions carry no absolute scale.** DSI-Bench ships no ground-truth geometry, so
> the numbers are in consistent but arbitrary units. Signs, ratios and directions are
> meaningful; magnitudes are not metres. Angles are degrees.

> **Vertical is gravity, horizontal is the camera's.** For camera motion, up/down is
> measured against a world vertical recovered from the scene's dominant plane, while
> forward/back and left/right stay the camera's own heading at frame 0. The two only
> coincide when the camera is level.
"""


def render(results_dir: Path, baseline_dir: Path | None = None) -> str:
    results_dir = Path(results_dir)
    answers = _load(results_dir / "answers")
    if not answers:
        raise SystemExit(f"no answers under {results_dir}")
    manifest = read_manifest(results_dir / "manifest.json") if (
        results_dir / "manifest.json").exists() else read_manifest(
        Path("d4rt_agent/results/dsi_bench/manifest.json"))
    meta_path = results_dir / "run_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    order = {q["question_id"]: i for i, q in enumerate(manifest["questions"])}
    records = sorted(answers.values(), key=lambda r: order.get(r["question_id"], 1 << 30))
    baseline = _load(baseline_dir) if baseline_dir and Path(baseline_dir).is_dir() else {}

    lines = [PREAMBLE, ""]
    lines += _config_table(manifest, meta, records)
    lines += _headline(records, baseline)
    lines += _glance(records, baseline)
    questions = {q["question_id"]: q["question"] for q in manifest["questions"]}
    for index, record in enumerate(records, start=1):
        lines += _question_section(index, record, baseline, questions)
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, nargs="?",
                        default=Path("d4rt_agent/results/agent_v4"))
    parser.add_argument("--baseline-dir", type=Path,
                        default=Path("d4rt_agent/results/dsi_bench/baseline"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    text = render(args.results_dir, args.baseline_dir)
    out = args.out or (args.results_dir / "AGENT_V4_RESULTS.md")
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
