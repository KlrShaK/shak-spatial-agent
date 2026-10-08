"""Command line for the jev-flow jobs: AgentV5's arguments plus the few this arm adds."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from d4rt_agent.agent_v5.run import parse_args as v5_parse_args

DEFAULT_CAPTION_MODEL = "Qwen/Qwen3-VL-2B-Instruct"


def parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    mine = argparse.ArgumentParser(add_help=False)
    mine.add_argument("--captions-dir", type=Path, default=None, help="default: <results-dir>/captions")
    mine.add_argument("--caption-model", default=DEFAULT_CAPTION_MODEL)
    mine.add_argument("--decisions-cache", type=Path, default=None, help="default: <results-dir>/decisions")
    mine.add_argument("--api-mode", choices=["live", "collect", "cached"], default="live")
    mine.add_argument("--seed-cache", type=Path, default=None,
                      help="a finished V5 run's cache dir: its scenes and groundings are linked in, never written")
    mine.add_argument("--max-tool-steps", type=int, default=6)
    mine.add_argument("--window-mode", choices=["regex", "hybrid"], default="regex",
                      help="regex: AgentV5's reading of the question; hybrid: the model reads the span structure from the frame grid")
    ns, rest = mine.parse_known_args(argv)
    args = v5_parse_args(rest)
    for key, value in vars(ns).items():
        setattr(args, key, value)
    args.captions_dir = args.captions_dir or (args.results_dir / "captions")
    args.decisions_cache = args.decisions_cache or (args.results_dir / "decisions")
    return args
