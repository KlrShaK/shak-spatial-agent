"""Offline metrics, selection and comparisons; local-Qwen audits require a compute job."""
import argparse
import json
import os
from pathlib import Path
from .compare import compare_existing, gate, prediction_rows
from .metrics import attach_audits, audit_predictions, summarize
from .selection import select_quick_check, DEFAULT_COUNTS


def _prediction_rows(paths):
    """Metrics and audits may combine benchmarks; comparison remains per benchmark."""
    result = {}
    for path in paths:
        rows = prediction_rows(path)
        if set(result) & set(rows):
            raise ValueError('duplicate predictions; provide each run/shard once')
        result.update(rows)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    selection = commands.add_parser('select')
    selection.add_argument('--pools', type=Path, required=True)
    selection.add_argument('--counts', type=json.loads, default=DEFAULT_COUNTS)
    selection.add_argument('--seed', type=int, default=0)
    metrics = commands.add_parser('metrics')
    metrics.add_argument('runs', type=Path, nargs='+')
    metrics.add_argument('--audit-file', type=Path, help='Attach a matching consolidated flip-audit map')
    comparison = commands.add_parser('compare')
    comparison.add_argument('--candidate', type=Path, nargs='+', required=True)
    comparison.add_argument('--baselines', type=Path, required=True, help='JSON mapping artifact names to run directories')
    comparison.add_argument('--v5', help='Named V5 artifact for gate')
    comparison.add_argument('--frames-only')
    comparison.add_argument('--v4')
    comparison.add_argument('--selection-file', type=Path, help='Require full approved-selection coverage at the gate')
    audit = commands.add_parser('audit', help='Separate local Qwen audit; run through a GPU SLURM launcher')
    audit.add_argument('runs', type=Path, nargs='+')
    audit.add_argument('--qwen-model', default=os.environ.get('MODEL_DIR', 'Qwen/Qwen3-VL-8B-Instruct'))
    audit.add_argument('--max-new-tokens', type=int, default=512)
    audit.add_argument('--seed', type=int, default=42)
    for child in (selection, metrics, comparison, audit):
        child.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'select':
        result = select_quick_check(json.loads(args.pools.read_text()), args.counts, args.seed)
    elif args.command == 'metrics':
        rows = _prediction_rows(args.runs)
        if args.audit_file:
            rows = attach_audits(rows, json.loads(args.audit_file.read_text()))
        result = summarize(rows.values())
    elif args.command == 'audit':
        from d4rt_agent.simple_v2 import OfflineQwen
        qwen = OfflineQwen(args.qwen_model, max_new_tokens=args.max_new_tokens, seed=args.seed)
        result = audit_predictions(_prediction_rows(args.runs), qwen)
        result.update(model=args.qwen_model, seed=args.seed)
    else:
        candidate = prediction_rows(args.candidate)
        baselines = {name: prediction_rows(paths) for name, paths in json.loads(args.baselines.read_text()).items()}
        expected = json.loads(args.selection_file.read_text()) if args.selection_file else None
        result = (gate(candidate, baselines, v5=args.v5, frames_only=args.frames_only, v4=args.v4,
                       expected_ids=expected)
                  if args.v5 else compare_existing(candidate, baselines))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
