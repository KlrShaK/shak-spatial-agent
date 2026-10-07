"""Execute ``testing_decomposition.ipynb`` once per branch of the decision graph (CPU).

Run in ``envs/point4d`` (it has nbclient; it has no nbconvert)::

    python -m dyn_dataset_exploration.decomposition.execute_notebook            # every branch
    python -m dyn_dataset_exploration.decomposition.execute_notebook --kinds pair_relation
    python -m dyn_dataset_exploration.decomposition.execute_notebook --backend d4rt   # needs PART=d4rt_scene prep

One executed copy per selection is written to
``dyn_dataset_exploration/outputs/decomposition/notebooks/``; the notebook itself is left
un-executed so its diff stays reviewable.
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path
from typing import Sequence

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "dyn_dataset_exploration" / "testing_decomposition.ipynb"
OUT = REPO / "dyn_dataset_exploration" / "outputs" / "decomposition" / "notebooks"
BRANCHES = ("camera_vocab", "cam_subject_range", "subject_direction", "pair_relation", "metric_distance",
            "instance_count", "event_order", "narrative", "semantic")


def run(kind: str, pick: int, catalog: str = "small", extra: dict[str, str] | None = None,
        timeout: int = 600) -> Path:
    book = nbformat.read(NOTEBOOK, as_version=4)
    book = copy.deepcopy(book)
    settings = {"CATALOG": catalog, "KIND": kind, "PICK": str(pick), **(extra or {})}
    for cell in book.cells:
        if cell.cell_type != "code":
            continue
        cell.outputs, cell.execution_count = [], None
        for name, value in settings.items():
            literal = value if re.fullmatch(r"\d+", value) else f'"{value}"'
            cell.source = re.sub(rf"(?m)^{name} = [^\s#]+", f"{name} = {literal}", cell.source, count=1)
    NotebookClient(book, timeout=timeout, kernel_name="point4d",
                   resources={"metadata": {"path": str(REPO)}}).execute()
    OUT.mkdir(parents=True, exist_ok=True)
    backend = (extra or {}).get("BACKEND", "point4d")
    target = OUT / f"{catalog}_{kind}_{pick}{'' if backend == 'point4d' else '_' + backend}.ipynb"
    nbformat.write(book, target)
    return target


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kinds", nargs="+", default=list(BRANCHES))
    parser.add_argument("--catalog", default="small")
    parser.add_argument("--pick", type=int, default=0)
    parser.add_argument("--backend", default="point4d", choices=["point4d", "d4rt"], help="3D model (see BACKEND)")
    args = parser.parse_args(argv)
    failed = []
    for kind in args.kinds:
        try:
            print("executed", run(kind, args.pick, args.catalog, extra={"BACKEND": args.backend}), flush=True)
        except Exception as error:  # noqa: BLE001 -- report every branch, then exit non-zero
            failed.append(kind)
            print(f"FAILED {kind}: {type(error).__name__}: {str(error)[-800:]}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
