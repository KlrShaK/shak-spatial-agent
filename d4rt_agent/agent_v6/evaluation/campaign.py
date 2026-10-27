"""Persistent SLURM quick-check campaign with conditional, ordered full runs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from d4rt_agent.agent_v5.run import write_json
from d4rt_agent.agent_v6.evaluation.compare import gate, expected_coverage, compare_existing
from d4rt_agent.agent_v6.evaluation.metrics import attach_audits, summarize

BENCHES = ('dsi', 'dynbench', 'vlm4d')
SPLITS = ('std', 'reverse', 'hflip', 'reverse_hflip')
PYTHON = '/cluster/work/igp_psr/spanwar/envs/d4rt_bw_third_party/bin/python'
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL'}


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def command(args, **kwargs):
    result = subprocess.run(args, text=True, capture_output=True, timeout=90, **kwargs)
    if result.returncode:
        raise RuntimeError(f'{args[0]} failed: {result.stderr[-3000:]} {result.stdout[-1000:]}')
    return result.stdout.strip()


def run_directories(plan, stage, bench):
    base = Path(plan['results']) / stage / bench
    return [base / split for split in SPLITS] if bench == 'dsi' else [base]


def predictions(plan, stage, bench):
    rows = {}
    for directory in run_directories(plan, stage, bench):
        split = directory.name if bench == 'dsi' else 'all'
        for path in sorted((directory / 'answers').glob('*.json')):
            row = read(path)
            key = f'{bench}:{split}:{row["question_id"]}'
            if key in rows:
                raise ValueError(f'duplicate result {key}')
            rows[key] = row
    return rows


def check_records(rows):
    """Infrastructure failures halt promotion; ordinary unanswered predictions still score."""
    failures = {key: row.get('error', row.get('status')) for key, row in rows.items()
                if row.get('status') in ('error', 'dry_run')}
    for key, row in rows.items():
        for report in row.get('reports', []):
            reason = report.get('reason') or ''
            if any(token in reason for token in ('ModuleNotFoundError', 'CUDA out of memory',
                                                 'unexpected keyword argument', 'No module named')):
                failures[key] = reason
    if failures:
        raise RuntimeError(f'infrastructure failures: {json.dumps(failures)[:4000]}')


def verify_inputs(plan):
    paths = dict(plan['input_hashes'])
    paths[str(Path(plan['source']) / 'source_manifest.json')] = plan['source_sha256']
    for path, expected in paths.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'frozen campaign input changed: {path}')


def worker(plan_path, stage, bench):
    plan = read(plan_path)
    verify_inputs(plan)
    if stage == 'baseline':
        from .regenerate_d7 import regenerate
        regenerate(plan)
        write_json(Path(plan['results']) / 'jobs' / 'baseline_dynbench_complete.json',
                   {'finished_at': utc(), 'job': os.environ.get('SLURM_JOB_ID')})
        return
    selection = plan['quick_selection'] if stage in ('smoke', 'quick') else plan['full_selection']
    result_stage = 'quick' if stage == 'smoke' else stage
    benches = BENCHES if stage == 'smoke' else (bench,)
    for current in benches:
        for directory in run_directories(plan, result_stage, current):
            split = directory.name if current == 'dsi' else 'std'
            args = [sys.executable, '-u', '-m', 'd4rt_agent.agent_v6.run',
                    '--fail-fast', '--bench', current, '--backbone', 'point4d', '--results-dir', str(directory),
                    '--selection-file', selection, '--qwen-model', plan['qwen_model'],
                    '--catalog', plan['catalog'], '--metadata-file', plan['metadata'][current],
                    '--third-party-dir', plan['third_party_dir'], '--split', split]
            if current == 'dsi':
                args += ['--manifest', plan['manifests'][split]]
            if stage == 'smoke' and current == 'dsi' and split == 'std' and plan.get('smoke_probe'):
                probe = plan['smoke_probe']
                if probe not in read(selection)['question_ids']['dsi']:
                    raise ValueError('recovery smoke probe must belong to the unchanged quick-check selection')
                probe_args = [*args, '--only', probe]
                print(json.dumps({'at': utc(), 'recovery_probe': probe, 'command': probe_args}), flush=True)
                subprocess.run(probe_args, check=True, cwd=plan['source'])
                check_records(predictions(plan, result_stage, current))
            if stage == 'smoke':
                args += ['--limit', '2']
            print(json.dumps({'at': utc(), 'command': args}), flush=True)
            subprocess.run(args, check=True, cwd=plan['source'])
            check_records(predictions(plan, result_stage, current))
        if stage != 'smoke':
            rows = predictions(plan, stage, current)
            # With one benchmark present the coverage helper expects its whole selection.
            reasons = expected_coverage(rows, read(selection))
            if not rows:
                reasons.append('no predictions were written')
            if reasons:
                raise RuntimeError('; '.join(reasons))
            audit = Path(plan['results']) / stage / f'{current}_audit.json'
            args = [sys.executable, '-u', '-m', 'd4rt_agent.agent_v6.evaluation', 'audit',
                    *map(str, run_directories(plan, stage, current)),
                    '--qwen-model', plan['qwen_model'], '--out', str(audit)]
            subprocess.run(args, check=True, cwd=plan['source'])
    write_json(Path(plan['results']) / 'jobs' / f'{stage}_{bench}_complete.json',
               {'finished_at': utc(), 'job': os.environ.get('SLURM_JOB_ID')})


def summarize_stage(plan, stage, *, apply_gate=False):
    output = {'stage': stage, 'updated_at': utc(), 'benchmarks': {}}
    exclusions_path = Path(plan['results']) / 'inputs/paper_reference_exclusions.json'
    if exclusions_path.is_file():
        output['paper_reference_exclusions'] = {'records': len(read(exclusions_path)), 'audit': str(exclusions_path)}
    all_rows = []
    for bench in BENCHES:
        rows = predictions(plan, stage, bench)
        check_records(rows)
        selection = read(plan[f'{stage}_selection'])
        reasons = expected_coverage(rows, selection)
        if reasons:
            raise RuntimeError('; '.join(reasons))
        audit = Path(plan['results']) / stage / f'{bench}_audit.json'
        if audit.is_file():
            rows = attach_audits(rows, read(audit))
        baselines = read(plan['baselines'][bench])
        if bench == 'dynbench' and plan.get('regenerate_d7'):
            regenerated = read(Path(plan['results']) / 'regenerated_d7/predictions.json')
            baselines['d7_regenerated'] = regenerated
            frames = baselines.setdefault('frames_only', {})
            for key, row in regenerated.items():
                if key not in frames:
                    frames[key] = {**row, 'letter': row['frames_letter'],
                                   'answered': row['frames_letter'] is not None,
                                   'correct': row['frames_letter'] == row['reference']['answer']}
        if apply_gate:
            settings = plan['gate_baselines'][bench]
            comparison = gate(rows, baselines, **settings, expected_ids=selection, seed=plan['seed'])
        else:
            comparison = compare_existing(rows, baselines, seed=plan['seed'])
        output['benchmarks'][bench] = {'metrics': summarize(rows.values()), 'comparison': comparison}
        all_rows.extend(rows.values())
    output['metrics'] = summarize(all_rows)
    if apply_gate:
        output['passed'] = all(value['comparison']['passed'] for value in output['benchmarks'].values())
    write_json(Path(plan['results']) / f'{stage}_report.json', output)
    lines = [f'# AgentV6 {stage} evaluation', '', f'Updated: {output["updated_at"]}', '']
    if apply_gate:
        lines += [f'Full-run gate: {"PASS" if output["passed"] else "FAIL"}', '']
    if output.get('paper_reference_exclusions', {}).get('records'):
        excluded = output['paper_reference_exclusions']
        lines += [f'Paper comparisons exclude {excluded["records"]} stored records with incompatible questions, '
                  f'options or reference answers. Audit: {excluded["audit"]}. Paired coverage is shown per system.', '']
    for bench, value in output['benchmarks'].items():
        metric = value['metrics']['overall']
        lines += [f'## {bench}', '', f'{metric["accuracy"]:.1%} over {metric["questions"]} records.', '']
        comparison = value['comparison']
        if apply_gate:
            lines += [f'Gate: {"PASS" if comparison["passed"] else "FAIL"}', '', *comparison['reasons'], '']
            comparison = comparison['comparisons']
        lines += ['| Existing system | n | Existing | V6 gain | Paired 95% interval |',
                  '|---|---:|---:|---:|---|']
        for name, item in comparison.items():
            if item.get('status') == 'ok':
                lo, hi = item['paired_95_ci']
                lines.append(f'| {name} | {item["questions"]} | {item["baseline_accuracy"]:.1%} | '
                             f'{item["paired_gain"]:+.1%} | {lo:+.1%} to {hi:+.1%} |')
        lines.append('')
    (Path(plan['results']) / f'{stage}_report.md').write_text('\n'.join(lines) + '\n')
    return output


def next_task(state, quick_report, regenerate_d7=False):
    completed = set(state.get('completed', []))
    for name in ('smoke:all', *(['baseline:dynbench'] if regenerate_d7 else []),
                 *(f'quick:{bench}' for bench in BENCHES)):
        if name not in completed:
            return name
    if quick_report is None:
        return 'gate'
    if not quick_report.get('passed') or any(
            not quick_report.get('benchmarks', {}).get(bench, {}).get('comparison', {}).get('passed')
            for bench in BENCHES):
        return 'gate_failed'
    for bench in BENCHES:
        name = f'full:{bench}'
        if name not in completed:
            return name
    return 'complete'


def submit(plan_path, task, platform='blackwell', attempt_id=None):
    plan = read(plan_path)
    stage, bench = task.split(':')
    env = dict(os.environ, V6_PLAN=str(Path(plan_path).resolve()), V6_STAGE=stage, V6_BENCH=bench)
    if attempt_id:
        env['V6_ATTEMPT_ID'] = attempt_id
    hours = 4 if stage == 'smoke' else 24
    args = ['sbatch', '--parsable', '--cpus-per-task=4', '--mem-per-cpu=12G', '--tmp=24G',
            f'--time={hours:02d}:00:00', f'--job-name=v6-{stage}-{bench}-{platform}',
            f'--output={plan["results"]}/jobs/{stage}_{bench}_%j.out']
    if platform == 'blackwell':
        args += [f'--partition=cuda13pr.{hours}h', '--gpus=nvidia_rtx_pro_6000:1', '--constraint=EPYC_9654']
    else:
        args += [f'--partition=gpupr.{hours}h', '--gpus=nvidia_a100_80gb_pcie:1', '--constraint=EPYC_7413|EPYC_9354']
    launcher = Path(plan['source']) / 'scripts/agent_v6_job.sh'
    if attempt_id:
        launcher = Path(os.environ.get('V6_RACE_LAUNCHER',
                                      str(Path(plan['source']) / 'scripts/agent_v6_race_job.sh')))
    args += [str(launcher)]
    job = command(args, cwd=plan['source'], env=env).split(';')[0]
    if not job.isdigit():
        raise RuntimeError(f'invalid scheduler job ID {job!r}')
    return {'job': job, 'task': task, 'submitted_at': utc(), 'platform': platform}


def submit_race(plan_path, task):
    plan = read(plan_path)
    attempt = uuid.uuid4().hex
    race = Path(plan['results']) / '.race'
    race.mkdir(exist_ok=True)
    jobs_file = race / f'{attempt}.jobs'
    contenders = []
    try:
        for platform in ('blackwell', 'a100'):
            contenders.append(submit(plan_path, task, platform, attempt))
            temporary = jobs_file.with_suffix('.tmp')
            temporary.write_text(''.join(f'{item["job"]}\n' for item in contenders))
            temporary.replace(jobs_file)
    except BaseException:
        if contenders:
            command(['scancel', *(item['job'] for item in contenders)])
        raise
    return {'task': task, 'submitted_at': utc(), 'attempt_id': attempt,
            'claim': str(race / f'{attempt}.claim/job'), 'contenders': contenders}


def active_status(active):
    if not active.get('contenders'):
        return job_status(active['job'])
    claim = Path(active['claim'])
    winner = claim.read_text().strip() if claim.is_file() else ''
    if winner:
        jobs = {item['job']: item for item in active['contenders']}
        if winner not in jobs:
            raise ValueError(f'race claim names an unknown job: {winner}')
        active.update(job=winner, platform=jobs[winner]['platform'])
        if not active.get('siblings_cancelled'):
            try:
                command(['scancel', *(job for job in jobs if job != winner)])
                active['siblings_cancelled'] = True
                active.pop('cancellation_error', None)
            except RuntimeError as error:
                active['cancellation_error'] = str(error)
        return job_status(winner)
    for contender in active['contenders']:
        contender['scheduler'] = job_status(contender['job'])
    states = [item['scheduler']['state'] for item in active['contenders']]
    if all(state in TERMINAL for state in states):
        interrupted = {'TIMEOUT', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL'}
        state = 'NODE_FAIL' if all(s in interrupted for s in states) else 'FAILED'
        return {'state': state, 'reason': 'all race contenders ended before claiming', 'contenders': states}
    return {'state': 'RACE_WAITING', 'contenders': states}


def job_status(job):
    try:
        output = command(['squeue', '--noheader', '-j', job, '-o', '%i|%T|%M|%R'])
    except RuntimeError:
        output = ''  # Completed job IDs may disappear from the live queue.
    if output:
        fields = output.splitlines()[0].split('|', 3)
        return {'state': fields[1], 'elapsed': fields[2], 'location': fields[3]}
    output = command(['sacct', '-n', '-P', '-j', job, '--format=JobIDRaw,State,Elapsed,ExitCode'])
    for line in output.splitlines():
        fields = line.split('|')
        if fields[0] == job:
            return {'state': fields[1].split()[0], 'elapsed': fields[2], 'exit': fields[3]}
    return {'state': 'ACCOUNTING_PENDING'}


def progress(plan, stage):
    result = {}
    for bench in BENCHES:
        rows = predictions(plan, stage, bench)
        result[bench] = {'records': len(rows), 'errors': sum(r.get('status') == 'error' for r in rows.values()),
                         'unanswered': sum(not r.get('answered') for r in rows.values())}
    return result


def control(plan_path):
    plan = read(plan_path)
    verify_inputs(plan)
    root = Path(plan['results'])
    (root / 'jobs').mkdir(exist_ok=True)
    state_path = root / 'status.json'
    state = read(state_path) if state_path.exists() else {'completed': [], 'history': [], 'created_at': utc()}
    state.update(controller_job=os.environ.get('SLURM_JOB_ID'), phase='running')
    report_path = root / 'quick_report.json'
    next_report = 0
    try:
        while True:
            active = state.get('active')
            if active:
                stage, bench = active['task'].split(':')
                marker = root / 'jobs' / f'{stage}_{bench}_complete.json'
                try:
                    status = active_status(active)
                except (RuntimeError, subprocess.TimeoutExpired) as error:
                    status = {'state': 'SCHEDULER_UNAVAILABLE', 'error': str(error)}
                active['scheduler'] = status
                state['progress'] = progress(plan, 'quick' if stage == 'smoke' else stage)
                terminal = status['state'] in TERMINAL
                if terminal:
                    state['history'].append(active)
                    state.pop('active')
                    if status['state'] == 'COMPLETED' and marker.is_file():
                        state['completed'].append(active['task'])
                    elif status['state'] in ('TIMEOUT', 'NODE_FAIL', 'PREEMPTED'):
                        attempts = sum(item['task'] == active['task'] for item in state['history'])
                        if attempts >= 5:
                            raise RuntimeError(f'{active["task"]}: five scheduler interruptions')
                        # Resume the identical frozen run; never change a selection/configuration.
                    else:
                        raise RuntimeError(f'{active["task"]} job {active.get("job", active.get("attempt_id"))} ended {status}; inspect its log')
            if not state.get('active'):
                quick_report = read(report_path) if report_path.exists() else None
                task = next_task(state, quick_report, plan.get('regenerate_d7', False))
                if task == 'gate':
                    summarize_stage(plan, 'quick', apply_gate=True)
                    continue
                if task in ('gate_failed', 'complete'):
                    if task == 'complete':
                        summarize_stage(plan, 'full')
                    state.update(phase=task, finished_at=utc())
                    write_json(state_path, state)
                    print(json.dumps(state), flush=True)
                    return
                # One race pair at a time (#73); only the atomic claim winner runs models.
                try:
                    existing = command(['squeue', '-u', os.environ['USER'], '--noheader', '-o', '%b|%T'])
                except (RuntimeError, subprocess.TimeoutExpired) as error:
                    state['scheduler_error'] = str(error)
                    state['updated_at'] = utc()
                    write_json(state_path, state)
                    time.sleep(30)
                    continue
                gpu = [line for line in existing.splitlines() if 'gpu:' in line]
                running = sum('RUNNING' in line or 'COMPLETING' in line for line in gpu)
                pending = sum('PENDING' in line or 'CONFIGURING' in line for line in gpu)
                if running == 0 and pending == 0:
                    state['active'] = submit_race(plan_path, task)
            state['updated_at'] = utc()
            write_json(state_path, state)
            if time.monotonic() >= next_report:
                write_json(root / 'monitor' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json'), state)
                print(json.dumps(state), flush=True)
                next_report = time.monotonic() + 1800
            time.sleep(30)
    except BaseException as error:
        state.update(phase='failed', error=f'{type(error).__name__}: {error}', updated_at=utc())
        write_json(state_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('worker', 'control'))
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--stage', choices=('smoke', 'baseline', 'quick', 'full'))
    parser.add_argument('--bench', choices=(*BENCHES, 'all'))
    args = parser.parse_args()
    (control(args.plan) if args.command == 'control' else worker(args.plan, args.stage, args.bench))


if __name__ == '__main__':
    main()
