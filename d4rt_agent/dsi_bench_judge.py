"""Map the agent's free-text DSI-Bench answers to option letters with an LLM judge.

The agent answers in prose; DSI-Bench is multiple choice.  This module asks a
small OpenAI model (``gpt-5.6-luna`` by default) to read each answer and report
the single option the agent *committed to*, or ``UNMAPPABLE`` when it committed
to none.  The judge is an extractor, not a grader: it never sees the ground
truth, and correctness is decided here by comparing its letter to ``gt``.

Efficiency is the OpenAI Batch API -- one file, one job, ~50% cheaper -- rather
than a call per question.  The ``failed`` agent records have no final answer to
read, so they are labelled ``NO_ANSWER`` locally without an API call.

The pipeline is a sequence of resumable subcommands, all writing under
``<results>/judge/``::

    source ~/.bashrc && python -m d4rt_agent.dsi_bench_judge prepare
    source ~/.bashrc && python -m d4rt_agent.dsi_bench_judge submit
    source ~/.bashrc && python -m d4rt_agent.dsi_bench_judge status --wait
    source ~/.bashrc && python -m d4rt_agent.dsi_bench_judge collect
    python -m d4rt_agent.dsi_bench_judge score

``prepare`` and ``score`` need no network.  Everything that touches the API reads
``OPENAI_API_KEY`` from the environment (hence ``source ~/.bashrc``) and never
prints it.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import requests

from .dsi_bench_data import CATEGORY_NAMES, parse_choice_letter, read_manifest


DEFAULT_RESULTS_DIR = Path("d4rt_agent/results/dsi_bench_full")
SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "dsi_bench_judge_system.md"

DEFAULT_MODEL = "gpt-5.6-luna"
# Generous on purpose: the visible output is a letter plus a short phrase, but a
# too-small cap truncates any model that spends hidden tokens before answering.
DEFAULT_MAX_COMPLETION_TOKENS = 512
BATCH_ENDPOINT = "/v1/chat/completions"

# The two ways a question ends up without a usable letter.  Kept distinct so the
# report can separate "the agent never answered" from "the agent answered but
# committed to no option".  Neither counts as correct.
SENTINEL_UNMAPPABLE = "UNMAPPABLE"
SENTINEL_NO_ANSWER = "NO_ANSWER"

# Batch states past which no more work will happen.
TERMINAL_BATCH_STATES = {"completed", "failed", "expired", "cancelled"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _base_url() -> str:
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")


def _api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Run this command as:\n"
            "  source ~/.bashrc && python -m d4rt_agent.dsi_bench_judge ..."
        )
    return key


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_api_key()}"}


def _write_atomic(path: Path, payload: Any) -> None:
    """Write JSON so an interrupted command never leaves a half-parsed file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _judge_dir(results_dir: Path) -> Path:
    return Path(results_dir) / "judge"


def _load_answers(results_dir: Path) -> dict[str, dict[str, Any]]:
    directory = Path(results_dir) / "answers"
    if not directory.exists():
        raise SystemExit(f"no answers directory: {directory}")
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        records[str(record["question_id"])] = record
    if not records:
        raise SystemExit(f"no answer files in {directory}")
    return records


# --------------------------------------------------------------------------- #
# Request construction (pure, no network -- exercised by the offline tests)
# --------------------------------------------------------------------------- #

def format_options(record: Mapping[str, Any]) -> str:
    options = record["options"]
    return "\n".join(f"{letter}: {options[letter]}" for letter in record["option_letters"])


def judge_messages(record: Mapping[str, Any], system_prompt: str) -> list[dict[str, str]]:
    """Build the two-message chat for one answer.  Ground truth is never included."""

    answer_text = str(record["final_answer"]["text"])
    user = (
        f"Question:\n{record['question']}\n\n"
        f"Options:\n{format_options(record)}\n\n"
        f"Model answer:\n{answer_text}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def response_format(option_letters: Sequence[str]) -> dict[str, Any]:
    """A strict json-schema whose ``choice`` enum is this question's own letters.

    The per-question enum is the first line of defence against a hallucinated
    option: a compliant model physically cannot return a letter that was not on
    offer.  ``collect`` re-checks anyway, for models that ignore strict mode.
    """

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "committed_choice",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "choice": {"type": "string", "enum": [*option_letters, SENTINEL_UNMAPPABLE]},
                    "rationale": {"type": "string"},
                },
                "required": ["choice", "rationale"],
                "additionalProperties": False,
            },
        },
    }


def batch_line(
    record: Mapping[str, Any],
    *,
    model: str,
    system_prompt: str,
    max_completion_tokens: int,
) -> dict[str, Any]:
    """One JSONL request line for the Batch API."""

    return {
        "custom_id": str(record["question_id"]),
        "method": "POST",
        "url": BATCH_ENDPOINT,
        "body": {
            "model": model,
            "messages": judge_messages(record, system_prompt),
            "response_format": response_format(record["option_letters"]),
            "max_completion_tokens": max_completion_tokens,
        },
    }


def _select_complete(
    answers: Mapping[str, dict[str, Any]], limit: int | None, only: Sequence[str] | None
) -> list[dict[str, Any]]:
    complete = [r for r in answers.values() if r.get("status") == "complete"]
    complete.sort(key=lambda r: str(r["question_id"]))
    if only:
        wanted = set(only)
        complete = [r for r in complete if str(r["question_id"]) in wanted]
    if limit is not None:
        complete = complete[:limit]
    return complete


# --------------------------------------------------------------------------- #
# Result parsing (pure, no network)
# --------------------------------------------------------------------------- #

def extract_choice(content: str, option_letters: Sequence[str]) -> tuple[str, str]:
    """Recover ``(choice, rationale)`` from one judge reply, defensively.

    Tries the structured JSON first, then falls back to reading a bare letter or
    the ``UNMAPPABLE`` token out of free text, then to ``parse_choice_letter``.
    Anything that does not resolve to one of this question's letters becomes
    ``UNMAPPABLE`` -- the judge is never allowed to invent an option here, even
    if the model bypassed strict mode.
    """

    letters = {letter.upper() for letter in option_letters}
    rationale = ""
    choice = ""
    text = (content or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, Mapping):
            choice = str(obj.get("choice", "")).strip()
            rationale = str(obj.get("rationale", "")).strip()
    except (ValueError, TypeError):
        pass

    upper = choice.upper()
    if upper in letters:
        return upper, rationale
    if upper == SENTINEL_UNMAPPABLE:
        return SENTINEL_UNMAPPABLE, rationale

    # No clean structured choice -- try to read one out of the raw content.
    if SENTINEL_UNMAPPABLE in text.upper():
        return SENTINEL_UNMAPPABLE, rationale
    salvaged = parse_choice_letter(text, option_letters)
    if salvaged is not None:
        return salvaged, rationale
    return SENTINEL_UNMAPPABLE, rationale


def _result_content(line: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Pull the message content out of one Batch output line, or an error string."""

    if line.get("error"):
        return None, f"request_error: {json.dumps(line['error'])[:300]}"
    response = line.get("response") or {}
    if response.get("status_code") != 200:
        return None, f"http_{response.get('status_code')}: {json.dumps(response.get('body'))[:300]}"
    try:
        return response["body"]["choices"][0]["message"]["content"], None
    except (KeyError, IndexError, TypeError):
        return None, f"unparseable_body: {json.dumps(response.get('body'))[:300]}"


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #

def prepare(args: argparse.Namespace) -> int:
    answers = _load_answers(args.results_dir)
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    judge_dir = _judge_dir(args.results_dir)
    judge_dir.mkdir(parents=True, exist_ok=True)

    complete = _select_complete(answers, args.limit, args.only)
    lines = [
        batch_line(
            record,
            model=args.model,
            system_prompt=system_prompt,
            max_completion_tokens=args.max_completion_tokens,
        )
        for record in complete
    ]
    input_path = judge_dir / "batch_input.jsonl"
    tmp = input_path.with_suffix(".jsonl.tmp")
    tmp.write_text(
        "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines),
        encoding="utf-8",
    )
    os.replace(tmp, input_path)

    # Failed records have no final answer to read; label them without a call.
    auto = {
        str(r["question_id"]): SENTINEL_NO_ANSWER
        for r in answers.values()
        if r.get("status") != "complete"
    }
    _write_atomic(judge_dir / "auto_labels.json", auto)

    statuses = defaultdict(int)
    for r in answers.values():
        statuses[r.get("status")] += 1
    print(f"answers: {len(answers)}  statuses: {dict(statuses)}")
    print(f"wrote {len(lines)} judge requests -> {input_path}")
    print(f"wrote {len(auto)} auto-labels ({SENTINEL_NO_ANSWER}) -> {judge_dir / 'auto_labels.json'}")
    print(f"model={args.model}  max_completion_tokens={args.max_completion_tokens}")
    return 0


def submit(args: argparse.Namespace) -> int:
    judge_dir = _judge_dir(args.results_dir)
    input_path = judge_dir / "batch_input.jsonl"
    if not input_path.exists():
        raise SystemExit(f"no batch input: {input_path}. Run 'prepare' first.")
    meta_path = judge_dir / "batch_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") not in TERMINAL_BATCH_STATES:
            raise SystemExit(
                f"a batch is already open: id={meta.get('batch_id')} status={meta.get('status')}.\n"
                "Run 'status' to check it, or delete batch_meta.json to start over."
            )

    base = _base_url()
    # Upload the input file.
    with input_path.open("rb") as handle:
        upload = requests.post(
            f"{base}/files",
            headers=_auth_headers(),
            files={"file": (input_path.name, handle, "application/jsonl")},
            data={"purpose": "batch"},
            timeout=120,
        )
    if upload.status_code != 200:
        raise SystemExit(f"file upload failed: HTTP {upload.status_code}: {upload.text[:300]}")
    file_id = upload.json()["id"]

    # Create the batch job.
    create = requests.post(
        f"{base}/batches",
        headers={**_auth_headers(), "Content-Type": "application/json"},
        json={
            "input_file_id": file_id,
            "endpoint": BATCH_ENDPOINT,
            "completion_window": "24h",
        },
        timeout=120,
    )
    if create.status_code != 200:
        raise SystemExit(f"batch creation failed: HTTP {create.status_code}: {create.text[:300]}")
    body = create.json()
    meta = {
        "batch_id": body["id"],
        "input_file_id": file_id,
        "status": body.get("status"),
        "endpoint": BATCH_ENDPOINT,
        "model": args.model,
        "request_count": sum(1 for _ in input_path.open()),
        "submitted_at": _utc_now(),
    }
    _write_atomic(meta_path, meta)
    print(f"submitted batch {meta['batch_id']} (status {meta['status']}, {meta['request_count']} requests)")
    print("next: python -m d4rt_agent.dsi_bench_judge status --wait")
    return 0


def _fetch_batch(batch_id: str) -> dict[str, Any]:
    resp = requests.get(f"{_base_url()}/batches/{batch_id}", headers=_auth_headers(), timeout=60)
    if resp.status_code != 200:
        raise SystemExit(f"batch fetch failed: HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def status(args: argparse.Namespace) -> int:
    meta_path = _judge_dir(args.results_dir) / "batch_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"no batch metadata: {meta_path}. Run 'submit' first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    batch_id = meta["batch_id"]

    deadline = time.monotonic() + args.max_wait
    while True:
        body = _fetch_batch(batch_id)
        counts = body.get("request_counts", {})
        meta.update(
            status=body.get("status"),
            request_counts=counts,
            output_file_id=body.get("output_file_id"),
            error_file_id=body.get("error_file_id"),
            checked_at=_utc_now(),
        )
        _write_atomic(meta_path, meta)
        print(
            f"batch {batch_id}: {body.get('status'):12s} "
            f"completed={counts.get('completed')} failed={counts.get('failed')} "
            f"total={counts.get('total')}"
        )
        if body.get("status") in TERMINAL_BATCH_STATES:
            if body.get("status") == "completed":
                print("done: python -m d4rt_agent.dsi_bench_judge collect")
            return 0
        if not args.wait or time.monotonic() >= deadline:
            if args.wait:
                print(f"still running after {args.max_wait}s; re-run 'status --wait' later.")
            return 0
        time.sleep(args.poll_interval)


def _download_file(file_id: str, destination: Path) -> None:
    resp = requests.get(f"{_base_url()}/files/{file_id}/content", headers=_auth_headers(), timeout=300)
    if resp.status_code != 200:
        raise SystemExit(f"file download failed: HTTP {resp.status_code}: {resp.text[:300]}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(resp.content)


def collect(args: argparse.Namespace) -> int:
    judge_dir = _judge_dir(args.results_dir)
    meta_path = judge_dir / "batch_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"no batch metadata: {meta_path}. Run 'submit' first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != "completed":
        raise SystemExit(
            f"batch status is {meta.get('status')!r}, not 'completed'. "
            "Run 'status --wait' until it completes."
        )

    output_path = judge_dir / "batch_output.jsonl"
    if meta.get("output_file_id") and (args.force or not output_path.exists()):
        _download_file(meta["output_file_id"], output_path)
    if meta.get("error_file_id"):
        _download_file(meta["error_file_id"], judge_dir / "batch_errors.jsonl")
    if not output_path.exists():
        raise SystemExit(f"no output file downloaded: {output_path}")

    answers = _load_answers(args.results_dir)
    manifest = read_manifest(Path(args.results_dir) / "manifest.json")
    letters = {str(q["question_id"]): list(q["option_letters"]) for q in manifest["questions"]}

    judgments: dict[str, dict[str, Any]] = {}
    api_errors = 0
    for raw in output_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        line = json.loads(raw)
        qid = str(line["custom_id"])
        content, error = _result_content(line)
        if error is not None:
            api_errors += 1
            judgments[qid] = {"choice": SENTINEL_UNMAPPABLE, "source": "gpt_error", "rationale": "", "raw": error}
            continue
        choice, rationale = extract_choice(content, letters.get(qid, ["A", "B", "C", "D"]))
        judgments[qid] = {"choice": choice, "source": "gpt", "rationale": rationale, "raw": content}

    # Fold in the locally-labelled failed records.
    auto_path = judge_dir / "auto_labels.json"
    auto = json.loads(auto_path.read_text(encoding="utf-8")) if auto_path.exists() else {}
    for qid, label in auto.items():
        judgments.setdefault(qid, {"choice": label, "source": "auto_failed", "rationale": "", "raw": None})

    # Every manifest question must be covered exactly once -- unless this is an
    # explicitly partial collect (a smoke-test batch, or an early inspection).
    expected = {str(q["question_id"]) for q in manifest["questions"]}
    missing = expected - set(judgments)
    extra = set(judgments) - expected
    if extra:
        raise SystemExit(f"judgments for unknown question_ids: {sorted(extra)[:5]}")
    if missing and not args.partial:
        raise SystemExit(
            f"judgment coverage incomplete: {len(missing)} question(s) missing.\n"
            f"  first missing: {sorted(missing)[:5]}\n"
            "Pass --partial to write an incomplete judgments.json anyway (score will refuse it)."
        )

    _write_atomic(judge_dir / "judgments.json", judgments)
    by_source = defaultdict(int)
    for j in judgments.values():
        by_source[j["source"]] += 1
    print(f"wrote {len(judgments)} judgments -> {judge_dir / 'judgments.json'}")
    print(f"sources: {dict(by_source)}")
    if api_errors:
        print(f"WARNING: {api_errors} requests returned an API error (labelled {SENTINEL_UNMAPPABLE}, source=gpt_error).")
    return 0


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def _blank_cell() -> dict[str, int]:
    return {"n": 0, "mapped": 0, "unmappable": 0, "no_answer": 0, "correct": 0}


def _accumulate(cell: dict[str, int], choice: str, gt: str) -> None:
    cell["n"] += 1
    if choice == SENTINEL_NO_ANSWER:
        cell["no_answer"] += 1
    elif choice == SENTINEL_UNMAPPABLE:
        cell["unmappable"] += 1
    else:
        cell["mapped"] += 1
        if choice == gt:
            cell["correct"] += 1


def _row(name: str, cell: Mapping[str, int]) -> str:
    n, mapped = cell["n"], cell["mapped"]
    acc_all = cell["correct"] / n if n else 0.0
    acc_mapped = cell["correct"] / mapped if mapped else 0.0
    map_rate = mapped / n if n else 0.0
    return (
        f"| {name} | {n} | {mapped} | {cell['unmappable']} | {cell['no_answer']} | "
        f"{cell['correct']} | {acc_all * 100:.1f}% | {acc_mapped * 100:.1f}% | {map_rate * 100:.1f}% |"
    )


_TABLE_HEADER = (
    "| Group | n | mapped | unmappable | no_answer | correct | acc_all | acc_mapped | map_rate |\n"
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
)


def build_score_report(results_dir: Path) -> str:
    judge_dir = _judge_dir(results_dir)
    judgments = json.loads((judge_dir / "judgments.json").read_text(encoding="utf-8"))
    answers = _load_answers(results_dir)
    manifest = read_manifest(Path(results_dir) / "manifest.json")

    # Refuse to score a partial judgments file: an accuracy over a subset of the
    # census would silently misrepresent the run.
    missing = {str(q["question_id"]) for q in manifest["questions"]} - set(judgments)
    if missing:
        raise SystemExit(
            f"refusing to score: {len(missing)} question(s) have no judgment "
            f"(e.g. {sorted(missing)[:3]}). Run a full 'collect' first."
        )

    overall = _blank_cell()
    by_dataset: dict[str, dict[str, int]] = defaultdict(_blank_cell)
    by_cate: dict[int, dict[str, int]] = defaultdict(_blank_cell)
    by_cited = {True: _blank_cell(), False: _blank_cell()}

    # Judge-vs-regex agreement, on the answers a regex can already read a letter from.
    regex_total = regex_agree = 0

    for question in manifest["questions"]:
        qid = str(question["question_id"])
        gt = str(question["gt"])
        choice = str(judgments[qid]["choice"])
        record = answers[qid]

        _accumulate(overall, choice, gt)
        _accumulate(by_dataset[record["dataset"]], choice, gt)
        _accumulate(by_cate[int(record["cate"])], choice, gt)
        _accumulate(by_cited[bool(record.get("d4rt_used"))], choice, gt)

        if record.get("status") == "complete":
            regex_letter = parse_choice_letter(
                str(record["final_answer"]["text"]), question["option_letters"]
            )
            if regex_letter is not None:
                regex_total += 1
                if regex_letter == choice:
                    regex_agree += 1

    lines: list[str] = []
    lines.append("# DSI-Bench full-split run — accuracy\n")
    lines.append(
        f"Generated {_utc_now()} from `judge/judgments.json` "
        f"({overall['n']} questions). Letters were extracted from the agent's prose by "
        f"the `{SYSTEM_PROMPT_PATH.name}` judge; correctness is `choice == gt`.\n"
    )
    lines.append(
        "> `acc_all` counts `unmappable` and `no_answer` as wrong — the honest benchmark "
        "number. `acc_mapped` is accuracy among answers that committed to an option. "
        "Four-way chance ≈ 25%.\n"
    )
    agree_pct = (regex_agree / regex_total * 100) if regex_total else 0.0
    lines.append(
        f"> **Judge validation:** on the {regex_total} answers a regex can already read a "
        f"letter from, the judge agrees {regex_agree}/{regex_total} = {agree_pct:.1f}% of the "
        "time. High agreement means the judge is extracting the stated choice, not inventing one.\n"
    )

    lines.append("## Overall\n")
    lines.append(_TABLE_HEADER)
    lines.append(_row("all questions", overall))
    lines.append("")

    lines.append("## By source dataset\n")
    lines.append(_TABLE_HEADER)
    for name in sorted(by_dataset):
        lines.append(_row(name, by_dataset[name]))
    lines.append("")

    lines.append("## By reasoning task\n")
    lines.append(_TABLE_HEADER)
    for cate in sorted(by_cate):
        lines.append(_row(f"c{cate} · {CATEGORY_NAMES[cate]}", by_cate[cate]))
    lines.append("")

    lines.append("## By D4RT citation (agent's own final answer)\n")
    lines.append(_TABLE_HEADER)
    lines.append(_row("cited a D4RT call", by_cited[True]))
    lines.append(_row("did not cite D4RT", by_cited[False]))
    lines.append("")

    return "\n".join(lines) + "\n"


def score(args: argparse.Namespace) -> int:
    report = build_score_report(args.results_dir)
    out_path = Path(args.results_dir) / "DSI_BENCH_SCORED.md"
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {out_path}")
    return 0


def run(args: argparse.Namespace) -> int:
    rc = prepare(args)
    if rc:
        return rc
    rc = submit(args)
    if rc:
        return rc
    return status(args)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_prepare_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", default=os.environ.get("JUDGE_MODEL", DEFAULT_MODEL))
        p.add_argument("--max-completion-tokens", type=int, default=DEFAULT_MAX_COMPLETION_TOKENS)
        p.add_argument("--limit", type=int, default=None, help="judge only the first N complete answers (smoke test)")
        p.add_argument("--only", nargs="*", default=None, help="judge only these question_ids")

    p_prepare = sub.add_parser("prepare", help="build the batch input (no network)")
    add_prepare_opts(p_prepare)
    p_prepare.set_defaults(func=prepare)

    p_submit = sub.add_parser("submit", help="upload the input and open a batch job")
    p_submit.add_argument("--model", default=os.environ.get("JUDGE_MODEL", DEFAULT_MODEL))
    p_submit.set_defaults(func=submit)

    p_status = sub.add_parser("status", help="check (and optionally wait on) the batch")
    p_status.add_argument("--wait", action="store_true", help="poll until terminal or --max-wait")
    p_status.add_argument("--max-wait", type=int, default=900, help="seconds to poll before giving up (default 900)")
    p_status.add_argument("--poll-interval", type=int, default=30)
    p_status.set_defaults(func=status)

    p_collect = sub.add_parser("collect", help="download results into judgments.json")
    p_collect.add_argument("--force", action="store_true", help="re-download the output file")
    p_collect.add_argument("--partial", action="store_true", help="allow an incomplete batch (smoke test / early look)")
    p_collect.set_defaults(func=collect)

    p_score = sub.add_parser("score", help="write the grouped accuracy report (no network)")
    p_score.set_defaults(func=score)

    p_run = sub.add_parser("run", help="prepare + submit + status --wait")
    add_prepare_opts(p_run)
    p_run.add_argument("--wait", action="store_true", default=True)
    p_run.add_argument("--max-wait", type=int, default=900)
    p_run.add_argument("--poll-interval", type=int, default=30)
    p_run.set_defaults(func=run)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
