"""Run AgentV6, or list questions and write model-free answer stubs with --dry-run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import time
import traceback

from d4rt_agent.agent_v5.run import (
    DEFAULT_CATALOG, DEFAULT_QWEN_MODEL, dsi_entries, dynbench_entries,
    open_clip, select, validate_resume, vlm4d_entries, write_json,
)

from .metadata import enrich_entries, REPORTING_FIELDS

ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench', choices=['dsi', 'dynbench', 'vlm4d'], default='dsi')
    parser.add_argument('--backbone', choices=['point4d', 'd4rt'], default='point4d')
    parser.add_argument('--results-dir', type=Path, default=Path('d4rt_agent/results/agent_v6'))
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--third-party-dir', type=Path, default=Path(os.environ.get(
        'SHAK_THIRD_PARTY', '/cluster/work/igp_psr/spanwar/shak-spatial-agent/third-party-tools')))
    parser.add_argument('--qwen-model', default=os.environ.get('MODEL_DIR', DEFAULT_QWEN_MODEL))
    parser.add_argument('--catalog', type=Path, default=DEFAULT_CATALOG)
    parser.add_argument('--all', action='store_true')
    for name, folder in [('dyn', 'Dyn-Bench'), ('dsi', 'DSI-Bench'), ('vlm4d', 'VLM4D')]:
        parser.add_argument(f'--{name}-root', type=Path,
                            default=Path('/cluster/work/igp_psr/spanwar/datasets') / folder)
    parser.add_argument('--split', default='std')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--random-sample', type=int, default=None)
    parser.add_argument('--only', nargs='*')
    parser.add_argument('--selection-file', type=Path)
    parser.add_argument('--metadata-file', type=Path, help='Reporting-only quantity and level metadata')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--smoke', type=int)
    parser.add_argument('--shuffle-seed', type=int)
    parser.add_argument('--shard', default='0/1')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--fail-fast', action='store_true', help='Stop after recording an infrastructure error')
    parser.add_argument('--max-steps', type=int, default=14)
    parser.add_argument('--token-budget', type=int, default=32000)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    args = parser.parse_args(argv)
    if not 1 <= args.max_steps <= 14:
        parser.error('--max-steps must be between 1 and 14')
    try:
        index, count = map(int, args.shard.split('/'))
        if not 0 <= index < count:
            raise ValueError
    except ValueError:
        parser.error('--shard must be i/n with 0 <= i < n')
    return args


def load_entries(args):
    if args.bench != 'dsi':
        return {'dynbench': dynbench_entries, 'vlm4d': vlm4d_entries}[args.bench](args)
    if args.manifest:
        if not args.manifest.is_file():
            raise FileNotFoundError(args.manifest)
        return dsi_entries(args)
    # V5 writes its manifest while loading. Keep that temporary so dry runs leave only stubs.
    from d4rt_agent.dsi_bench_data import build_manifest

    manifest = build_manifest(dsi_root=args.dsi_root, split=args.split,
                              census=args.random_sample is None, random_size=args.random_sample)
    with tempfile.TemporaryDirectory(prefix='agent-v6-manifest-') as directory:
        clone = argparse.Namespace(**vars(args))
        clone.manifest = Path(directory) / 'manifest.json'
        clone.manifest.write_text(json.dumps(manifest))
        return dsi_entries(clone)


def answer_record(entry, args, solution, seconds=0.0):
    """Ground truth is attached only after solve; it never enters a model task."""
    record = dict(solution)
    record.update(schema_version=6, question_id=entry['id'], bench=args.bench,
                  wall_seconds=seconds,
                  reference={'answer': entry['answer'], 'answer_also': entry.get('answer_also', []),
                             'question': entry['question'], 'options': entry['options'],
                             'task': entry.get('task'), 'group': entry.get('group'),
                             'dataset': entry.get('dataset'), 'level': entry.get('level'),
                             'quantity_group': entry.get('quantity_group')},
                  clip={'relative_path': entry.get('relative_path') or
                        f"{entry.get('dataset')}/{entry.get('video')}",
                        'split': args.split if args.bench == 'dsi' else None})
    record['reference'].update({key: entry[key] for key in REPORTING_FIELDS if key in entry})
    record['correct'] = (record.get('letter') in [entry['answer'], *entry.get('answer_also', [])]
                         if record.get('answered') else False)
    return record


def run(args):
    catalog = enrich_entries(load_entries(args), args.bench, split=args.split,
                             metadata_path=args.metadata_file, selection_file=args.selection_file,
                             catalog_path=args.catalog)
    args._dataset_sha256 = hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()
    entries = select(catalog, args)
    answers = args.results_dir / 'answers'
    print(f'{len(entries)} questions ({args.bench}, {args.backbone})', flush=True)
    if args.dry_run:
        for entry in entries:
            target = answers / f"{entry['id']}.json"
            if target.exists() and json.loads(target.read_text()).get('status') != 'dry_run':
                raise ValueError('dry run must not overwrite live answers; use a separate results directory')
            write_json(target, answer_record(entry, args, dict(status='dry_run', answered=False, letter=None)))
            print(f"{entry['id']}: {entry['question']}", flush=True)
        return 0
    return run_live(entries, args)


def run_live(entries, args):
    from d4rt_agent.agent_v5.research.snapshot import snapshot

    model_path = Path(args.qwen_model).expanduser()
    if model_path.exists():
        args.qwen_model = str(model_path.resolve())
    source = ROOT / 'source_manifest.json'
    if not source.exists():
        # Freeze the imported implementation as well as V6 before any inference.
        destination = args.results_dir / 'source'
        if destination.exists():
            raise ValueError('run from the frozen source directory when resuming this run')
        snapshot(ROOT, destination)
        argv = []
        for key, value in vars(args).items():
            if key.startswith('_') or value is None or value is False:
                continue
            argv.append('--' + key.replace('_', '-'))
            if value is True:
                continue
            values = value if isinstance(value, list) else [value]
            argv.extend(str(v.resolve()) if isinstance(v, Path) else str(v) for v in values)
        extra_paths = [str(Path(p).resolve()) for p in os.environ.get('PYTHONPATH', '').split(os.pathsep)
                       if p and Path(p).resolve() != ROOT.resolve()]
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(destination.resolve()), *extra_paths]))
        return subprocess.call([sys.executable, '-m', 'd4rt_agent.agent_v6.run', *argv],
                               cwd=destination, env=env)
    else:
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    contract = {k: getattr(args, k) for k in ('bench', 'backbone', 'qwen_model', 'max_steps',
                                            'token_budget', 'max_new_tokens', 'split')}
    contract['third_party_dir'] = str(args.third_party_dir.resolve())
    contract['source_sha256'] = source_hash
    contract['dataset_sha256'] = args._dataset_sha256
    signature = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    validate_resume(args.results_dir, signature)
    write_json(args.results_dir / 'run_config.json', dict(contract, run_signature=signature))
    cache = args.cache_dir or args.results_dir / 'cache'
    os.environ['SHAK_THIRD_PARTY'] = str(args.third_party_dir.resolve())
    from d4rt_agent.agent_v5.models import Models, gpu_memory_report
    from .agent import AgentV6
    from .workspace import Workspace

    from .cache import ClipArchive
    archive = ClipArchive(cache, args.backbone)
    models = Models(qwen_model=args.qwen_model, backbone=args.backbone, cache=archive.directory,
                    max_new_tokens=args.max_new_tokens)
    from .tools.captioner import VideoQwen
    agent = AgentV6(qwen=VideoQwen(models.qwen()), max_steps=args.max_steps, token_budget=args.token_budget,
                    max_new_tokens=args.max_new_tokens)
    previous, workspace = None, None
    for entry in entries:
        target = args.results_dir / 'answers' / f"{entry['id']}.json"
        if target.exists() and not args.force:
            continue
        started = time.monotonic()
        try:
            key = entry.get('video_path') or (entry.get('dataset'), entry.get('video'))
            if key != previous:
                archive.activate(key)
                workspace = Workspace(open_clip(entry, args, archive.directory), models=models)
                previous = key
            solution = agent.solve({'id': entry['id'], 'question': entry['question'],
                                    'options': entry['options']}, workspace=workspace)
        except Exception as error:
            traceback.print_exc()
            solution = dict(status='error', answered=False, letter=None, error=f'{type(error).__name__}: {error}')
        record = answer_record(entry, args, solution, time.monotonic() - started)
        record.update(gpu_memory_report())
        write_json(target, record)
        archive.flush()
        print(f"{entry['id']}: {record.get('letter')} ({record['status']})", flush=True)
        if args.fail_fast and record['status'] == 'error':
            archive.close()
            raise RuntimeError(f"{entry['id']}: {record['error']}")
    archive.close()
    return 0


def main(argv=None):
    raise SystemExit(run(parse_args(argv)))


if __name__ == '__main__':
    main()
