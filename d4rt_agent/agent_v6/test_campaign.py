"""CPU fixtures for promotion decisions and frozen evaluation inputs."""
import hashlib
import json

import pytest

from .evaluation.campaign import BENCHES, check_records, next_task, verify_inputs


def completed_quick():
    return {'completed': ['smoke:all', *(f'quick:{bench}' for bench in BENCHES)]}


def passing_report():
    return {'passed': True, 'benchmarks': {bench: {'comparison': {'passed': True}} for bench in BENCHES}}


def test_full_runs_require_every_complete_quick_benchmark_and_every_gate():
    state = completed_quick()
    assert next_task(state, None) == 'gate'
    for bench in BENCHES:
        partial = {'completed': [task for task in state['completed'] if task != f'quick:{bench}']}
        assert next_task(partial, passing_report()) == f'quick:{bench}'
        failed = passing_report()
        failed['benchmarks'][bench]['comparison']['passed'] = False
        assert next_task(state, failed) == 'gate_failed'
    assert next_task(state, {'passed': True}) == 'gate_failed'
    assert next_task(state, {'passed': False}) == 'gate_failed'


def test_full_runs_keep_protocol_priority_order():
    state = completed_quick()
    for bench in BENCHES:
        assert next_task(state, passing_report()) == f'full:{bench}'
        state['completed'].append(f'full:{bench}')
    assert next_task(state, passing_report()) == 'complete'


def test_deleted_d7_comparisons_are_regenerated_before_quick_check_and_promotion():
    state = {'completed': ['smoke:all']}
    assert next_task(state, None, regenerate_d7=True) == 'baseline:dynbench'
    state['completed'].append('baseline:dynbench')
    assert next_task(state, None, regenerate_d7=True) == 'quick:dsi'
    assert next_task(completed_quick(), passing_report(), regenerate_d7=True) == 'baseline:dynbench'


def test_infrastructure_errors_stop_campaign_but_model_abstention_is_scored():
    check_records({'q': {'status': 'unanswered', 'answered': False}})
    with pytest.raises(RuntimeError, match='infrastructure'):
        check_records({'q': {'status': 'error', 'error': 'video failed to open'}})
    with pytest.raises(RuntimeError, match='infrastructure'):
        check_records({'q': {'reports': [{'reason': "No module named 'point4d'"}]}})


def test_frozen_input_change_is_rejected(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    manifest = source / 'source_manifest.json'
    manifest.write_text('{}')
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps({'question_ids': {'dynbench': ['q']}}))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    plan = {'source': str(source), 'source_sha256': digest(manifest),
            'input_hashes': {str(selection): digest(selection)}}
    verify_inputs(plan)
    selection.write_text('{}')
    with pytest.raises(ValueError, match='input changed'):
        verify_inputs(plan)


@pytest.mark.parametrize('probe,allowed', [('selected', True), ('outside-selection', False)])
def test_recovery_smoke_runs_failed_selected_question_first_without_changing_selection(monkeypatch, tmp_path, probe, allowed):
    from .evaluation import campaign
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps({'question_ids': {'dsi': ['selected']}}))
    plan = {'results': str(tmp_path / 'results'), 'source': str(tmp_path),
            'quick_selection': str(selection), 'smoke_probe': probe, 'qwen_model': 'local-model',
            'catalog': 'catalog', 'third_party_dir': 'third-party',
            'metadata': {b: 'metadata' for b in BENCHES},
            'manifests': {s: 'manifest' for s in campaign.SPLITS}}
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    calls = []
    monkeypatch.setattr(campaign, 'verify_inputs', lambda p: None)
    monkeypatch.setattr(campaign, 'predictions', lambda *a: {})
    monkeypatch.setattr(campaign.subprocess, 'run', lambda args, **kwargs: calls.append(args))
    if not allowed:
        with pytest.raises(ValueError, match='unchanged quick-check selection'):
            campaign.worker(path, 'smoke', 'all')
        assert not calls
        return
    campaign.worker(path, 'smoke', 'all')
    assert len(calls) == 7
    assert calls[0][-2:] == ['--only', 'selected']
    assert all('--only' not in command for command in calls[1:])
    assert all(command[-2:] == ['--limit', '2'] for command in calls[1:])
    assert json.loads(selection.read_text()) == {'question_ids': {'dsi': ['selected']}}


def test_completed_job_falls_back_to_accounting_when_queue_forgets_id(monkeypatch):
    from .evaluation import campaign
    def scheduler(args):
        if args[0] == 'squeue':
            raise RuntimeError('Invalid job id specified')
        return '123|COMPLETED|00:20:00|0:0|'
    monkeypatch.setattr(campaign, 'command', scheduler)
    assert campaign.job_status('123')['state'] == 'COMPLETED'


def test_race_submits_both_platforms_with_one_shared_attempt(monkeypatch, tmp_path):
    from .evaluation import campaign
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'results': str(tmp_path)}))
    calls = []
    def submit(path, task, platform, attempt):
        calls.append((platform, attempt))
        return {'job': str(len(calls)), 'platform': platform}
    monkeypatch.setattr(campaign, 'submit', submit)
    active = campaign.submit_race(plan, 'smoke:all')
    assert [p for p, _ in calls] == ['blackwell', 'a100']
    assert {attempt for _, attempt in calls} == {active['attempt_id']}
    assert (tmp_path / '.race' / f'{active["attempt_id"]}.jobs').read_text() == '1\n2\n'


def test_partial_race_submission_cancels_the_queued_contender(monkeypatch, tmp_path):
    from .evaluation import campaign
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'results': str(tmp_path)}))
    def submit(path, task, platform, attempt):
        if platform == 'a100':
            raise RuntimeError('submission rejected')
        return {'job': '1', 'platform': platform}
    cancellations = []
    monkeypatch.setattr(campaign, 'submit', submit)
    monkeypatch.setattr(campaign, 'command', lambda args: cancellations.append(args))
    with pytest.raises(RuntimeError, match='submission rejected'):
        campaign.submit_race(plan, 'smoke:all')
    assert cancellations == [['scancel', '1']]


def test_race_tracks_claim_winner_and_cancels_only_the_sibling(monkeypatch, tmp_path):
    from .evaluation import campaign
    claim = tmp_path / 'job'
    claim.write_text('2\n')
    active = {'claim': str(claim), 'contenders': [{'job': '1', 'platform': 'blackwell'},
                                                 {'job': '2', 'platform': 'a100'}]}
    cancellations = []
    monkeypatch.setattr(campaign, 'command', lambda args: cancellations.append(args))
    monkeypatch.setattr(campaign, 'job_status', lambda job: {'state': 'RUNNING' if job == '2' else 'CANCELLED'})
    assert campaign.active_status(active)['state'] == 'RUNNING'
    assert active['job'] == '2' and active['platform'] == 'a100'
    assert cancellations == [['scancel', '1']]
    campaign.active_status(active)
    assert len(cancellations) == 1


def test_race_waits_for_claim_and_does_not_treat_a_loser_as_completion(monkeypatch, tmp_path):
    from .evaluation import campaign
    active = {'claim': str(tmp_path / 'job'), 'contenders': [{'job': '1'}, {'job': '2'}]}
    monkeypatch.setattr(campaign, 'job_status', lambda job: {'state': 'COMPLETED' if job == '1' else 'RUNNING'})
    assert campaign.active_status(active)['state'] == 'RACE_WAITING'
    assert 'job' not in active
    monkeypatch.setattr(campaign, 'job_status', lambda job: {'state': 'NODE_FAIL'})
    assert campaign.active_status(active)['state'] == 'NODE_FAIL'
