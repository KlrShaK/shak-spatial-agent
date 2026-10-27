"""Request recurring checks in the existing Codex session, independently of GPU jobs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def health(root, state, now):
    phase = state.get('phase')
    if phase in ('complete', 'gate_failed'):
        return phase
    if phase == 'failed':
        return 'campaign_failed'
    metadata = root / 'jobs/control_login.json'
    if not metadata.is_file():
        return 'supervisor_missing'
    process = read(metadata)
    if process['host'] == socket.gethostname():
        try:
            os.kill(int(process['pid']), 0)
        except ProcessLookupError:
            return 'supervisor_missing'
    if state.get('updated_at'):
        updated = datetime.fromisoformat(state['updated_at'])
        if (now - updated).total_seconds() > 600:
            return 'stale_supervisor_heartbeat'
    return None


def enqueue(config, config_path, reason):
    message = (
        f'[Automated AgentV6 monitoring: {reason}; authorized by user] '
        f'Read monitoring instructions beside {config_path}. Follow its current plan pointer. '
        'Check Slurm, worker logs, result errors and supervisor health. Update the user with '
        'completed/expected counts, accuracy per benchmark/quantity, unanswered/unavailable rates, '
        'runtime/model/doctor calls and peak GPU memory where recorded; compare only matching completed '
        'baseline IDs and label partial results. Investigate and fix infrastructure failures, then resume '
        'the A100/Blackwell race using frozen source and the same selection/gates. Keep the supervisor '
        'on login. Update the monitor plan pointer after any new run. Reuse this watcher; do not create '
        'a duplicate or launch full runs before the complete quick-check gate passes.'
    )
    return subprocess.run([config['codex'], 'queue', '--thread', config['thread_id'],
                           '--message', message], text=True, capture_output=True, timeout=60,
                          cwd=config['repository'], check=True).stdout.strip()


def monitor(config_path):
    config_path = Path(config_path).resolve()
    state_path = config_path.with_name('watcher_state.json')
    config = read(config_path)
    interval = int(config.get('interval_seconds', 1800))
    next_report = time.monotonic() + interval
    last_alert = None
    status = {'pid': os.getpid(), 'host': socket.gethostname(), 'phase': 'watching',
              'started_at': datetime.now(timezone.utc).isoformat(), 'queued_checks': 0}
    while True:
        now = datetime.now(timezone.utc)
        try:
            config = read(config_path)
            if not config.get('enabled', True):
                status.update(phase='stopped', updated_at=now.isoformat())
                write(state_path, status)
                return
            plan_path = Path(config['plan'])
            root = Path(read(plan_path)['results'])
            campaign = read(root / 'status.json')
            problem = health(root, campaign, now)
            signature = (str(plan_path), problem, campaign.get('updated_at'))
            reason = problem if problem and signature != last_alert else None
            if reason is None and time.monotonic() >= next_report:
                reason = '30-minute progress update'
            if reason:
                receipt = enqueue(config, config_path, reason)
                status.update(last_notice_at=now.isoformat(), last_notice=reason,
                              receipt=receipt, queued_checks=status['queued_checks'] + 1)
                status.pop('error', None)
                print(json.dumps(status), flush=True)
                next_report = time.monotonic() + int(config.get('interval_seconds', 1800))
                if problem:
                    last_alert = signature
                if problem in ('complete', 'gate_failed'):
                    status.update(phase='finished', updated_at=now.isoformat())
                    write(state_path, status)
                    return
            status.update(plan=str(plan_path), campaign_phase=campaign.get('phase'),
                          health=problem or 'ok', updated_at=now.isoformat(),
                          next_check_in_seconds=max(0, round(next_report - time.monotonic())))
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            status.update(error=f'{type(error).__name__}: {error}', updated_at=now.isoformat())
            print(json.dumps(status), flush=True)
        write(state_path, status)
        time.sleep(min(60, int(config.get('poll_seconds', 30))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    monitor(parser.parse_args().config)
