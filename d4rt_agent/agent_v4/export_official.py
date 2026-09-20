"""Export our answers into DSI-Bench's own result format, for their evaluate.py.

Their scorer (`dsibench/evaluate.py`) reads, per augmentation, a metadata CSV and
a result CSV of the SAME length and row order, and compares
`str(result.final_answer)[0]` against `str(meta.GT)`. It does no parsing of its
own -- extraction happens upstream, in their inference script.

So the strongest comparability claim available is: produce that CSV and run their
scorer unmodified. This module writes it. What it cannot make identical is the
extraction step, which is inherently method-specific -- see REPORT notes.

Rows the run has not answered are written as `E`, which is what their inference
script assigns to permanently-failed samples and which their scorer counts wrong.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Sequence

from d4rt_agent.agent_v4.results_report import letter_of

AUGS = ("std", "reverse", "hflip", "reverse_hflip")
UNANSWERED = "E"   # their sentinel for a sample that never produced a letter


def official_extract(text: str | None) -> str | None:
    """DSI-Bench's own extractor, verbatim from `qwen_inference_code.py`."""

    match = re.search(r"<answer>\s*([A-D])\s*</answer>", text or "", re.IGNORECASE)
    return match.group(1).upper() if match else None


def _answer_letter(record: dict[str, Any], strict: bool) -> tuple[str, str]:
    """Return ``(letter, raw_text)`` for one answer record, either method."""

    if "answer_text" in record:                      # agent v4
        raw = record.get("answer_text") or ""
        options = (record.get("reference") or {}).get("options") or {}
    else:                                            # tool-free control
        raw = record.get("raw_response") or ""
        options = record.get("options") or {}
    letter = official_extract(raw) if strict else letter_of(raw, options)
    return (letter or UNANSWERED), raw


def export(results_dir: Path, answers_subdir: str, out_csv: Path,
           manifest: Path, strict: bool = False) -> dict[str, int]:
    """Write one augmentation's results in their row order.

    ``entries`` is NOT already in that order: a census manifest groups
    questions by video (so the run loads each clip once), which is a
    different order than the metadata CSV. Each entry carries its own
    ``csv_row_index`` from `dsi_bench_data.py`, so rows are placed by that
    index rather than by manifest iteration order -- appending in manifest
    order silently scrambled the answer-to-ground-truth alignment for any
    manifest whose row order isn't already the CSV's (every census run).
    """

    entries = json.loads(Path(manifest).read_text())["questions"]
    answers = {p.stem: json.loads(p.read_text())
               for p in (Path(results_dir) / answers_subdir).glob("*.json")}

    # A census manifest covers every CSV row exactly once, so max index + 1
    # equals the CSV's length; a partial (e.g. 200-question) manifest does
    # not, and its export is not meant to be diffed against the official
    # scorer, which requires equal-length meta/result files anyway.
    size = max((e["csv_row_index"] for e in entries), default=0) + (1 if entries else 0)
    rows: list[dict[str, str] | None] = [None] * size
    found = 0
    for entry in entries:
        idx = entry["csv_row_index"]
        record = answers.get(entry["question_id"])
        if record is None:
            rows[idx] = {"result_text": "", "final_answer": UNANSWERED}
            continue
        found += 1
        letter, raw = _answer_letter(record, strict)
        rows[idx] = {"result_text": raw, "final_answer": letter}
    rows = [r or {"result_text": "", "final_answer": UNANSWERED} for r in rows]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["result_text", "final_answer"])
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": len(rows), "answered": found,
            "unextracted": sum(1 for r in rows if r["final_answer"] == UNANSWERED)}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("agent", "control"), default="agent")
    parser.add_argument("--out", type=Path, default=Path("d4rt_agent/results/official_export"))
    parser.add_argument("--model-name", default=None,
                        help="the {model}.csv stem their scorer looks for")
    parser.add_argument("--strict", action="store_true",
                        help="use their <answer>X</answer> extractor instead of ours")
    parser.add_argument("--census", action="store_true",
                        help="export the full census dirs instead of the 200-question draw")
    args = parser.parse_args(argv)

    name = args.model_name or ("agent_v4" if args.method == "agent" else "control_qwen3vl8b")
    R = Path("d4rt_agent/results")
    # (answers dir, subdir, manifest) per augmentation. The manifest is named
    # separately because the control's std run reuses the agent's manifest rather
    # than carrying its own.
    if args.census:
        dirs = {a: (R / "census" / a, "answers", R / "census" / a / "manifest.json")
                for a in AUGS}
    elif args.method == "agent":
        dirs = {"std": (R / "agent_v4_random200", "answers",
                        R / "agent_v4_random200" / "manifest.json"),
                **{a: (R / f"agent_v4_{a}", "answers", R / f"agent_v4_{a}" / "manifest.json")
                   for a in AUGS[1:]}}
    else:
        dirs = {"std": (R / "agent_v4_random200_control", "baseline",
                        R / "agent_v4_random200" / "manifest.json"),
                **{a: (R / f"control200_{a}", "baseline",
                       R / f"control200_{a}" / "manifest.json") for a in AUGS[1:]}}

    for aug in AUGS:
        src, sub, manifest = dirs[aug]
        stats = export(src, sub, args.out / aug / f"{name}.csv", manifest,
                       strict=args.strict)
        print(f"  {aug:14s} rows={stats['rows']:5d} answered={stats['answered']:5d} "
              f"unextracted={stats['unextracted']:5d} -> {args.out/aug/(name+'.csv')}")


if __name__ == "__main__":
    main()
