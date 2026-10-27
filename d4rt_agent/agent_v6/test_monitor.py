from datetime import datetime, timedelta, timezone
import json
import socket
from types import SimpleNamespace

import pytest

from .evaluation import monitor


def supervisor(tmp_path):
    (tmp_path / 'jobs').mkdir()
    (tmp_path / 'jobs/control_login.json').write_text(json.dumps({'pid': 123, 'host': socket.gethostname()}))


@pytest.mark.parametrize('phase,expected', [('failed', 'campaign_failed'), ('complete', 'complete'),
                                            ('gate_failed', 'gate_failed')])
def test_monitor_reports_failures_and_terminal_gate_states(tmp_path, phase, expected):
    assert monitor.health(tmp_path, {'phase': phase}, datetime.now(timezone.utc)) == expected


def test_monitor_detects_dead_supervisor_even_without_failed_status(tmp_path, monkeypatch):
    supervisor(tmp_path)
    def dead(*args):
        raise ProcessLookupError()
    monkeypatch.setattr(monitor.os, 'kill', dead)
    assert monitor.health(tmp_path, {'phase': 'running'}, datetime.now(timezone.utc)) == 'supervisor_missing'


def test_monitor_detects_stale_heartbeat_but_accepts_healthy_pending_jobs(tmp_path, monkeypatch):
    supervisor(tmp_path)
    monkeypatch.setattr(monitor.os, 'kill', lambda *args: None)
    now = datetime.now(timezone.utc)
    state = {'phase': 'running', 'updated_at': now.isoformat()}
    assert monitor.health(tmp_path, state, now) is None
    state['updated_at'] = (now - timedelta(minutes=11)).isoformat()
    assert monitor.health(tmp_path, state, now) == 'stale_supervisor_heartbeat'


def test_monitor_queues_checks_to_the_existing_session_without_model_override(tmp_path, monkeypatch):
    requests = []
    def queue(args, **kwargs):
        requests.append((args, kwargs))
        return SimpleNamespace(stdout='Queued message receipt')
    monkeypatch.setattr(monitor.subprocess, 'run', queue)
    config = {'codex': '/bin/codex', 'thread_id': 'existing-session', 'repository': str(tmp_path)}
    assert monitor.enqueue(config, tmp_path / 'config.json', 'campaign_failed') == 'Queued message receipt'
    args, kwargs = requests[0]
    assert args[:5] == ['/bin/codex', 'queue', '--thread', 'existing-session', '--message']
    assert 'campaign_failed' in args[5] and 'matching completed' in args[5]
    assert '--model' not in args and kwargs['check'] is True
