"""Load VLM4D, label every question with ``taxonomy``, and measure the videos.

Questions come from AgentV5's own loader (``d4rt_agent.agent_v5.run.vlm4d_entries``), so the question ids
here are the ids of AgentV5's answer files and a run can be joined to these labels directly. No network,
no model. Commands (from the repo root)::

    python -m vlm4d_dataset_exploration.vlm4d_data manifest              # logs/manifest.json, seconds
    sbatch scripts/run_cpu_task.slurm -m vlm4d_dataset_exploration.vlm4d_data videos   # logs/videos.json (reads 1000 mp4 headers)
"""

from __future__ import annotations

import argparse
from argparse import Namespace
import json
from pathlib import Path
from typing import Any, Sequence

from vlm4d_dataset_exploration import taxonomy as T

DEFAULT_ROOT = Path("/cluster/work/igp_psr/spanwar/datasets/VLM4D")
DEFAULT_REPO = Path("/cluster/work/igp_psr/spanwar/VLM4D")       # the paper's repo: graded model outputs
DEFAULT_RESULTS_DIR = Path("vlm4d_dataset_exploration/logs")
EXO_MAX_ID = 922                                                  # the benchmark's exo/ego split of the real ids


def view_of(source: str) -> str:
    return {"ego4d": "ego", "synthetic": "synthetic"}.get(source, "exo")


def load_rows(root: Path = DEFAULT_ROOT) -> list[dict[str, Any]]:
    """One row per question: AgentV5's entry plus ``split``, ``n`` (the benchmark id number), ``source``,
    ``view`` and ``gold`` (the answer text). ``answer_also`` lists a second letter when two options share
    the gold text (one question)."""

    from d4rt_agent.agent_v5.run import vlm4d_entries

    rows = []
    for e in vlm4d_entries(Namespace(vlm4d_root=Path(root))):
        rows.append({"qid": e["id"], "split": e["task"], "n": int(e["id"].rsplit("_", 1)[-1]), "source": e["group"],
                     "view": view_of(e["group"]), "video": e["relative_path"], "question": e["question"].strip(),
                     "options": e["options"], "answer": e["answer"], "answer_also": e["answer_also"],
                     "gold": e["options"][e["answer"]]})
    return rows


def label(row: dict[str, Any]) -> dict[str, Any]:
    """The row with both axes and the sub-task program added."""

    labels = T.classify(row["question"], list(row["options"].values()), row["source"])
    return {**row, "paper_type": T.paper_type(row["question"], row["gold"]),
            "motion_type": T.motion_type(row["question"]), "perspective": T.perspective(row["question"]),
            **labels, "program": T.program(labels)}


def labelled_rows(root: Path = DEFAULT_ROOT) -> list[dict[str, Any]]:
    return [label(r) for r in load_rows(root)]


def video_stats(path: Path) -> dict[str, Any]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return {"frames": frames, "fps": round(fps, 3), "width": w, "height": h,
            "seconds": round(frames / fps, 3) if fps else None}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["manifest", "videos"])
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_RESULTS_DIR)
    args = p.parse_args(argv)
    if args.command == "manifest":
        rows = labelled_rows(args.root)
        write_json(args.out / "manifest.json", {"source": str(args.root), "questions": rows})
        print(f"{len(rows)} questions -> {args.out / 'manifest.json'}")
    else:
        videos = sorted({r["video"] for r in load_rows(args.root)})
        stats = {v: {"source": v.split("/")[1] if v.startswith("videos_real/") else "synthetic",
                     **video_stats(args.root / v)} for v in videos}
        write_json(args.out / "videos.json", stats)
        print(f"{len(stats)} videos -> {args.out / 'videos.json'}")


if __name__ == "__main__":
    main()
