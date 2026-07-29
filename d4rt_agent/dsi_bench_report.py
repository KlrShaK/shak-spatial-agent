"""Assemble the DSI-Bench run into a reviewable markdown report.

Runs on CPU after the GPU passes::

    python -m d4rt_agent.dsi_bench_report

Copies each clip beside its result, renders the exact 32 frames the models saw as
both an animation and an indexed contact sheet, and lays ground truth next to the
agent's answer and the tool-free control's answer.

**Nothing here is scored.** No accuracy is computed and no letter is extracted
from the agent's prose, because the answers are meant to be judged by reading
them.  The one letter reported for the control is the tag its own prompt asks it
to emit, recorded verbatim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np

from .dsi_bench_data import CATEGORY_NAMES, parse_choice_letter, read_manifest
from .dsi_bench_show import render_trace_markdown, trace_boundary_metrics
from .orchestration_audit import count_action_objects, iter_attempts
from .simple_v2_contracts import sample_video_cpu


DEFAULT_RESULTS_DIR = Path("d4rt_agent/results/dsi_bench")
REPORT_NAME = "DSI_BENCH_RESULTS.md"

GIF_WIDTH = 320
GIF_FRAME_MS = 150
SHEET_COLUMNS = 8
SHEET_TILE_WIDTH = 240
SHEET_JPEG_QUALITY = 88

# One grounding tile is one GIF frame wide, so the box can be checked against
# the animation at the same scale.
GROUNDING_TILE_WIDTH = GIF_WIDTH
GROUNDING_COLUMNS = 4
GROUNDING_CAPTION_PX = 20


def _load_records(directory: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not directory.exists():
        return records
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        records[record["question_id"]] = record
    return records


def build_gif(frames: np.ndarray, destination: Path) -> None:
    """Animate the sampled frames so the motion is visible inline.

    Markdown renderers that strip <video> still render an animated GIF, and every
    question here is about motion, so a still frame would hide the evidence.
    """

    from PIL import Image

    destination.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    size = (GIF_WIDTH, max(1, round(height * GIF_WIDTH / width)))
    images = [Image.fromarray(frame).resize(size, Image.BILINEAR) for frame in frames]
    images[0].save(
        destination,
        save_all=True,
        append_images=images[1:],
        duration=GIF_FRAME_MS,
        loop=0,
        optimize=True,
    )


def build_contact_sheet(frames: np.ndarray, destination: Path) -> None:
    """Lay the 32 frames out in an indexed grid.

    The index labels are the point: the agent's `t_src`, `t_tgt` and `t_cam`
    arguments are frame numbers, so a reviewer needs to see which image each of
    those refers to.
    """

    import cv2

    destination.parent.mkdir(parents=True, exist_ok=True)
    tiles = []
    for index, frame in enumerate(frames):
        height, width = frame.shape[:2]
        tile_height = max(1, round(height * SHEET_TILE_WIDTH / width))
        tile = cv2.resize(frame, (SHEET_TILE_WIDTH, tile_height))
        tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        label = str(index)
        cv2.putText(tile, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(tile, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    tallest = max(tile.shape[0] for tile in tiles)
    tiles = [
        cv2.copyMakeBorder(tile, 0, tallest - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        for tile in tiles
    ]
    rows = [
        np.hstack(tiles[start : start + SHEET_COLUMNS])
        for start in range(0, len(tiles), SHEET_COLUMNS)
    ]
    cv2.imwrite(str(destination), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, SHEET_JPEG_QUALITY])


def _grounding_tile(frame: np.ndarray, query: Mapping[str, Any], call_id: str) -> np.ndarray:
    """One `query_d4rt` call: the frame it grounded on, with the box it drew."""

    import cv2

    height, width = frame.shape[:2]
    scale = GROUNDING_TILE_WIDTH / width
    tile = cv2.resize(frame, (GROUNDING_TILE_WIDTH, max(1, round(height * scale))))
    tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)

    x0, y0, x1, y1 = (round(value * scale) for value in query["bbox_pixel"])
    cv2.rectangle(tile, (x0, y0), (x1, y1), (0, 230, 0), 1, cv2.LINE_AA)

    # point_index 0 carries offset [0, 0] -- it is the centroid, and the rest are
    # the ensemble's seeded offsets around it.
    for point in query.get("query_points", []):
        u, v = (round(value * scale) for value in point["pixel_uv"])
        if point.get("point_index") == 0:
            cv2.circle(tile, (u, v), 3, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(tile, (u, v), 4, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(tile, (u, v), 2, (0, 220, 255), 1, cv2.LINE_AA)

    index = str(query["t_src"])
    for colour, thickness in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(tile, index, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, thickness, cv2.LINE_AA)

    tile = cv2.copyMakeBorder(
        tile, 0, GROUNDING_CAPTION_PX, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    caption = f"{call_id}  f{query['t_src']}  {str(query.get('label', ''))[:26]}"
    cv2.putText(
        tile,
        caption,
        (4, tile.shape[0] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return tile


def build_grounding_sheet(
    frames: np.ndarray, record: Mapping[str, Any] | None, destination: Path
) -> bool:
    """Every box the agent grounded, on the frame it grounded it on.

    The numeric trace says `bbox_2d_1000=[480, 300, 540, 620]`, which is not
    something a reviewer can check by eye.  Drawing it answers the question the
    trace cannot: was the agent actually looking at the object it named?
    """

    import cv2

    queries = [
        (call_id, value)
        for call_id, value in (record or {}).get("evidence", {}).items()
        if call_id.startswith("d4rt_") and value.get("bbox_pixel")
    ]
    if not queries:
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    tiles = [
        _grounding_tile(frames[value["t_src"]], value, call_id) for call_id, value in queries
    ]
    blank = np.zeros_like(tiles[0])
    while len(tiles) % GROUNDING_COLUMNS:
        tiles.append(blank)
    rows = [
        np.hstack(tiles[start : start + GROUNDING_COLUMNS])
        for start in range(0, len(tiles), GROUNDING_COLUMNS)
    ]
    cv2.imwrite(str(destination), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, SHEET_JPEG_QUALITY])
    return True


def _baseline_letter(record: Mapping[str, Any] | None, entry: Mapping[str, Any]) -> str | None:
    """Re-derive the control's letter from its raw reply.

    Parsing at read time rather than trusting the stored field keeps old runs
    readable after the extractor improves -- the raw text is the record of what
    the model said, and the letter is only an interpretation of it.
    """

    if not record or record.get("status") != "complete":
        return None
    return parse_choice_letter(record.get("raw_response", ""), entry["option_letters"])


def _final_text(record: Mapping[str, Any] | None) -> str:
    if not record:
        return "_(not run)_"
    if record.get("status") in {"failed", "error"}:
        return f"_({record['status']}: {str(record.get('error', ''))[:160]})_"
    answer = record.get("final_answer") or {}
    return str(answer.get("text") or answer.get("value") or "_(no answer)_")


def _first_clause(text: str, limit: int = 64) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    line = line.replace("|", "\\|")
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _trace_rows(record: Mapping[str, Any]) -> list[str]:
    rows = []
    for entry in record.get("trace", []):
        action = entry.get("parsed_action") or {}
        arguments = action.get("arguments") or {}
        name = action.get("action", "—")
        target = arguments.get("t_tgt")
        suffix = str(entry.get("discarded_suffix", ""))
        if suffix.strip():
            extra_actions = count_action_objects(suffix)
            suffix_text = f"yes ({extra_actions} action{'s' if extra_actions != 1 else ''})"
        else:
            suffix_text = "no"
        rows.append(
            "| {step} | {status} | {stage} | {name} | {label} | {t_src} | {t_tgt} | "
            "{t_cam} | {suffix} | {why} |".format(
                step=entry.get("step", ""),
                status=entry.get("status", ""),
                stage=entry.get("failure_stage") or "—",
                name=name,
                label=str(arguments.get("label", ""))[:28].replace("|", "\\|"),
                t_src=arguments.get("t_src", ""),
                t_tgt=(f"{target[0]}…{target[-1]} ({len(target)})" if isinstance(target, list) and len(target) > 3
                       else (target if target is not None else "")),
                t_cam=arguments.get("t_cam", ""),
                suffix=suffix_text,
                why=str(arguments.get("justification", entry.get("error", "")))[:90].replace("|", "\\|"),
            )
        )
    return rows


def _header(manifest: Mapping[str, Any], agent: Mapping[str, Any], baseline: Mapping[str, Any]) -> list[str]:
    sampling = manifest["sampling"]
    counts = manifest["counts"]
    sample_agent = next(iter(agent.values()), {})
    sample_baseline = next(iter(baseline.values()), {})
    lines = [
        "# DSI-Bench evaluation — D4RT agent vs. tool-free Qwen",
        "",
        "25 questions from DSI-Bench (`std` split), sampled evenly across the benchmark's",
        "5 source datasets and 6 motion-reasoning tasks. Each row shows the clip, the",
        "question, the dataset's ground-truth answer, the D4RT agent's answer, and the same",
        "Qwen model answering from the same 32 frames with no tools.",
        "",
        "**These answers are not auto-scored.** The agent replies in prose so its reasoning",
        "stays visible, and the comparison against ground truth is meant to be read.",
        "",
        "## Run configuration",
        "",
        "| | |",
        "| --- | --- |",
        f"| Questions | {len(manifest['questions'])} |",
        f"| Sampling design | {sampling['design']} — {sampling['description']} |",
        f"| Sampling seed | `{sampling['seed']}` (requested `{sampling['requested_seed']}`) |",
        f"| Source CSV | `{manifest['csv_path']}` |",
        f"| CSV sha256 | `{manifest['csv_sha256'][:16]}…` |",
        f"| Ground-truth histogram | {counts['gt_histogram']} |",
        f"| Per dataset | {counts['per_dataset']} |",
        f"| Per task | {counts['per_category']} |",
    ]
    if sample_agent:
        alignment = sample_agent.get("benchmark_alignment", {})
        lines += [
            f"| Point mode | `{sample_agent.get('point_mode', '?')}` |",
            f"| Position scale | `{alignment.get('type', '?')}` (scale {alignment.get('scale', '?')}) |",
        ]
    if sample_baseline:
        lines.append(f"| Qwen model | `{sample_baseline.get('qwen_model', '?')}` |")
    statuses: dict[str, int] = {}
    for record in agent.values():
        statuses[record["status"]] = statuses.get(record["status"], 0) + 1
    if statuses:
        lines.append(f"| Agent run status | {statuses} |")
        tracker_backed = sum(1 for record in agent.values() if record.get("d4rt_used"))
        lines.append(f"| Answers citing D4RT | {tracker_backed} / {len(agent)} |")
        boundary = {
            "turns": 0,
            "discarded_suffixes": 0,
            "discarded_suffixes_with_action": 0,
        }
        for record in agent.values():
            for _, attempt in iter_attempts(record):
                metrics = trace_boundary_metrics(attempt)
                for key in boundary:
                    boundary[key] += metrics[key]
        lines += [
            f"| Model turns audited | {boundary['turns']} |",
            (
                "| Turns with discarded suffixes | "
                f"{boundary['discarded_suffixes']} / {boundary['turns']} |"
            ),
            (
                "| Discarded suffixes containing another action | "
                f"{boundary['discarded_suffixes_with_action']} |"
            ),
        ]
    lines += [
        "",
        "> **Read this as a diagnostic, not a benchmark score.** The sample is balanced by",
        "> design — each task carries 4–5 questions, while the full benchmark is 33%",
        "> *Obj:moving cam* and 31% *Cam:dynamic scene* — so it is built to show where the",
        "> tracker helps and where it does not, not to estimate DSI-Bench accuracy. With 25",
        "> four-way questions, chance alone lands around 6 correct.",
        "",
        "> **Positions carry no absolute scale.** DSI-Bench ships no ground-truth geometry, so",
        "> the agent's numbers are in consistent but arbitrary units. Signs, ratios and",
        "> directions are meaningful; magnitudes are not metres.",
        "",
    ]
    return lines


def _index_table(
    manifest: Mapping[str, Any], agent: Mapping[str, Any], baseline: Mapping[str, Any]
) -> list[str]:
    lines = [
        "## At a glance",
        "",
        "| # | Dataset | Task | GT | Agent answer | Qwen-only | D4RT |",
        "| ---: | --- | --- | :---: | --- | :---: | :---: |",
    ]
    for index, entry in enumerate(manifest["questions"], start=1):
        question_id = entry["question_id"]
        agent_record = agent.get(question_id)
        baseline_record = baseline.get(question_id)
        gt_text = f"**{entry['gt']}**"
        agent_text = _first_clause(_final_text(agent_record))
        letter = _baseline_letter(baseline_record, entry)
        if letter:
            baseline_text = f"**{letter}**" if letter == entry["gt"] else letter
        elif baseline_record and baseline_record.get("status") == "complete":
            baseline_text = "?"
        elif baseline_record:
            baseline_text = "err"
        else:
            baseline_text = "—"
        used = agent_record.get("d4rt_used") if agent_record else None
        lines.append(
            f"| [{index}](#{index}-{entry['dataset'].lower()}-task-{entry['cate']}) "
            f"| {entry['dataset']} | {CATEGORY_NAMES[entry['cate']]} | {gt_text} "
            f"| {agent_text} | {baseline_text} | {'yes' if used else 'no' if agent_record else '—'} |"
        )
    lines.append("")
    return lines


def _question_section(
    index: int,
    entry: Mapping[str, Any],
    agent_record: Mapping[str, Any] | None,
    baseline_record: Mapping[str, Any] | None,
) -> list[str]:
    slug = entry["video_slug"]
    lines = [
        f"## {index}. {entry['dataset']} — task {entry['cate']}: {CATEGORY_NAMES[entry['cate']]}",
        "",
        f"![clip {index}](gifs/{slug}.gif)",
        "",
        f"[▶ full-quality MP4](videos/{slug}.mp4) · source: `{entry['relative_path']}`",
        "",
        f'<video controls width="480" src="videos/{slug}.mp4"></video>',
        "",
        f"**Question.** {entry['question']}",
        "",
    ]
    for letter in entry["option_letters"]:
        marker = " ← **ground truth**" if letter == entry["gt"] else ""
        lines.append(f"- **{letter}.** {entry['options'][letter]}{marker}")
    lines += ["", "**Agent answer.**", "", f"> {_final_text(agent_record).strip()}", ""]

    if agent_record:
        answer = agent_record.get("final_answer") or {}
        limitations = str(answer.get("limitations", "")).strip()
        if limitations:
            lines += [f"*Agent-stated limitations:* {limitations}", ""]
        if agent_record.get("status") == "complete_relaxed":
            lines += [
                "> ⚠️ This answer was produced on the relaxed retry: the agent could not reach",
                "> an answer while required to cite a D4RT measurement, so it is not",
                "> tracker-backed.",
                "",
            ]

    if baseline_record:
        letter = _baseline_letter(baseline_record, entry)
        raw = str(baseline_record.get("raw_response", baseline_record.get("error", ""))).strip()
        if letter:
            verdict = "matches ground truth" if letter == entry["gt"] else f"ground truth is {entry['gt']}"
            shown = f"**{letter}** — {entry['options'].get(letter, '?')} ({verdict}). Raw reply: `{raw[:80]}`"
        else:
            shown = f"_no letter could be read from the reply:_ `{raw[:200]}`"
        lines += ["**Qwen-only baseline.**", "", f"> {shown}", ""]

    if agent_record:
        attempts = list(iter_attempts(agent_record))
        for attempt_name, attempt in attempts:
            if not attempt.get("trace"):
                continue
            rows = _trace_rows(attempt)
            boundary = trace_boundary_metrics(attempt)
            label = (
                "Agent"
                if len(attempts) == 1
                else f"{attempt_name.capitalize()} attempt"
            )
            timing = (
                f", {agent_record.get('wall_seconds', 0):.0f}s"
                if len(attempts) == 1 else ""
            )
            lines += [
                f"<details><summary>{label} trace "
                f"({len(rows)} steps{timing}; "
                f"{boundary['discarded_suffixes']} discarded suffixes, "
                f"{boundary['discarded_suffixes_with_action']} with another action)"
                "</summary>",
                "",
                "| Step | Status | Stage | Action | Label | t_src | t_tgt | t_cam | "
                "Discarded suffix | Justification |",
                "| ---: | --- | --- | --- | --- | ---: | --- | ---: | --- | --- |",
                *rows,
                "",
                "</details>",
                "",
            ]
            # The table above says what the agent did; this says why it thought
            # so, based on the effective model-visible response.
            lines += [
                f"<details><summary>{label} thinking log "
                f"({len(rows)} steps)</summary>",
                "",
                *render_trace_markdown(attempt),
                "</details>",
                "",
            ]
    grounded = [
        call_id
        for call_id in (agent_record or {}).get("evidence", {})
        if call_id.startswith("d4rt_")
    ]
    if grounded:
        lines += [
            f"<details><summary>Grounded objects ({len(grounded)} query_d4rt calls)</summary>",
            "",
            f"![groundings {index}](groundings/{slug}.jpg)",
            "",
            "Each tile is one `query_d4rt` call, drawn on its `t_src` frame: green box as the "
            "agent placed it, red dot the centroid, yellow dots the four other `ensemble5` "
            "sample points. The number top-left is the sampled frame index.",
            "",
            "</details>",
            "",
        ]
    lines += [
        "<details><summary>The 32 sampled frames the models saw</summary>",
        "",
        f"![contact sheet {index}](contact_sheets/{slug}.jpg)",
        "",
        "</details>",
        "",
        "---",
        "",
    ]
    return lines


def build_report(results_dir: Path, skip_media: bool = False) -> Path:
    results_dir = Path(results_dir)
    manifest = read_manifest(results_dir / "manifest.json")
    agent = _load_records(results_dir / "answers")
    baseline = _load_records(results_dir / "baseline")

    lines = _header(manifest, agent, baseline)
    lines += _index_table(manifest, agent, baseline)
    lines += ["## Questions", ""]

    for index, entry in enumerate(manifest["questions"], start=1):
        slug = entry["video_slug"]
        if not skip_media:
            source = Path(entry["video_path"])
            destination = results_dir / "videos" / f"{slug}.mp4"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copyfile(source, destination)
            gif_path = results_dir / "gifs" / f"{slug}.gif"
            sheet_path = results_dir / "contact_sheets" / f"{slug}.jpg"
            grounding_path = results_dir / "groundings" / f"{slug}.jpg"
            if not gif_path.exists() or not sheet_path.exists() or not grounding_path.exists():
                # Re-sampled from the copy: the sampler is deterministic, so these
                # are exactly the frames both models were shown.
                sampled = sample_video_cpu(destination)
                if not gif_path.exists():
                    build_gif(sampled.frames_rgb, gif_path)
                if not sheet_path.exists():
                    build_contact_sheet(sampled.frames_rgb, sheet_path)
                if not grounding_path.exists():
                    build_grounding_sheet(
                        sampled.frames_rgb, agent.get(entry["question_id"]), grounding_path
                    )
                del sampled
            print(f"[{index:2d}/{len(manifest['questions'])}] media ready for {slug[:48]}", flush=True)
        lines += _question_section(
            index, entry, agent.get(entry["question_id"]), baseline.get(entry["question_id"])
        )

    report_path = results_dir / REPORT_NAME
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--skip-media", action="store_true", help="rebuild only the markdown, reusing existing media"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    path = build_report(args.results_dir, skip_media=args.skip_media)
    print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
