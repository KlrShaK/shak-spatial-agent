"""CPU fixtures for promotion decisions and frozen evaluation inputs."""
import hashlib
import json

import pytest

from .evaluation.campaign import BENCHES, check_records, next_task, verify_inputs
from .evaluation.prepare_campaign import normalize_prediction, paper_reference_mismatches


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


def test_existing_baseline_reference_cannot_silently_change():
    entry = {'id': 'q', 'answer': 'B', 'options': {'A': 'left', 'B': 'right'},
             'question': 'Which way?', 'relative_path': 'std/clip.mp4', 'quantity_group': 'motion'}
    baseline = {'correct': True, 'answered': True, 'letter': 'B',
                'reference': {'answer': 'B', 'options': entry['options'], 'question': entry['question']}}
    row = normalize_prediction(baseline, entry, 'dsi', 'std')
    assert row['correct'] and row['clip']['relative_path'] == entry['relative_path']
    baseline['reference']['answer'] = 'A'
    with pytest.raises(ValueError, match='reference mismatch'):
        normalize_prediction(baseline, entry, 'dsi', 'std')


def test_paper_id_reuse_cannot_match_a_different_question_or_reference():
    entry = {'question': 'Which way is it rotating?', 'options': {'A': 'clockwise', 'B': 'still'}, 'answer': 'A'}
    row = {'question': entry['question'], 'choices': entry['options'], 'ground_truth_answer': 'clockwise'}
    assert paper_reference_mismatches(row, entry) == []
    assert paper_reference_mismatches({**row, 'question': 'Which way is it moving?'}, entry) == ['question']
    assert paper_reference_mismatches({**row, 'ground_truth_answer': 'still'}, entry) == ['ground_truth_answer']
    assert paper_reference_mismatches({**row, 'choices': {'A': 'still', 'B': 'clockwise'}}, entry) == ['choices']


def test_paper_numeric_options_match_the_benchmark_loader_text_options():
    entry = {'question': 'How many?', 'options': {'A': '1', 'B': '2'}, 'answer': 'A'}
    row = {'question': 'How many?', 'choices': {'A': 1, 'B': 2}, 'ground_truth_answer': 1}
    assert paper_reference_mismatches(row, entry) == []
    assert paper_reference_mismatches({**row, 'ground_truth_answer': 2}, entry) == ['ground_truth_answer']


@pytest.mark.parametrize('baseline', [
    {'answer_text': 'B: right', 'answered': True},
    {'parsed_letter': 'B', 'gt': 'B', 'options': {'A': 'left', 'B': 'right'}},
    {'prediction': 'B. right', 'answer': 'B', 'options': ['A. left', 'B. right'], 'accuracy': 1.0},
    {'prediction': 'B. right', 'answer': 'b', 'options': ['A. left', 'B. right'], 'accuracy': 1.0},
])
def test_existing_v4_control_and_official_predictions_keep_their_actual_scores(baseline):
    entry = {'id': 'q', 'answer': 'B', 'options': {'A': 'left', 'B': 'right'},
             'question': 'Which way?', 'relative_path': 'clip.mp4'}
    row = normalize_prediction(baseline, entry, 'dsi', 'std')
    assert row['letter'] == 'B' and row['correct'] and row['answered']


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


def test_d7_regeneration_restores_only_selected_questions_and_packs_clip_files(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from dyn_dataset_exploration.decomposition import build_catalog, program
    from .evaluation import regenerate_d7

    source = tmp_path / 'source'
    source.mkdir()
    entries = [dict(qid=qid, dataset='test', video=qid, clip_slug=qid,
                    question='What happens?', options={'A': 'left', 'B': 'right'}, answer='B')
               for qid in ('selected', 'excluded')]
    catalog = tmp_path / 'catalog.json'
    catalog.write_text(json.dumps({'questions': entries}))
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps({'question_ids': {'dynbench': ['selected']}}))
    baselines = tmp_path / 'baselines.json'
    baselines.write_text(json.dumps({'frames_only': {'dynbench:all:selected': {'letter': 'A'}}}))
    plan = dict(source=str(source), results=str(tmp_path / 'results'), dyn_root='/unused',
                catalog=str(catalog), quick_selection=str(selection), qwen_model='fixture',
                baselines={'dynbench': str(baselines)})
    requested = []
    monkeypatch.setattr(build_catalog, 'clip_record', lambda root, dataset, video: {'clip_slug': video})
    monkeypatch.setattr(build_catalog, 'materialize_frames', lambda *args: None)
    monkeypatch.setattr(program, 'rule_parse', lambda *args: SimpleNamespace(
        to_dict=lambda: {'route': 'vlm_only', 'entities': []}))

    def run(args, **kwargs):
        requested.append(args)
        if '--collect' in args:
            out = regenerate_d7.Path(args[args.index('--collect') + 1])
            assert json.loads((out / 'answers/selected.frames_only.json').read_text())['answer'] == 'A'
            (out / 'solutions.json').write_text(json.dumps({'selected': {'final': 'B', 'frames_only': 'A'}}))

    monkeypatch.setattr(regenerate_d7.subprocess, 'run', run)
    rows = regenerate_d7.regenerate(plan)
    assert set(rows) == {'dynbench:all:selected'}
    assert rows['dynbench:all:selected']['correct']
    assert rows['dynbench:all:selected']['frames_letter'] == 'A'
    assert len(list((tmp_path / 'results/regenerated_d7/cache/point4d').glob('*.tar.gz'))) == 1
    assert not any('prep_point4d' in str(args) for args in requested)
    assert not any('ground' in args for args in requested)
    requested.clear()
    assert regenerate_d7.regenerate(plan) == rows
    assert not requested


def test_d7_scene_reuse_requires_the_same_generator_clock_and_rejects_missing_scale(tmp_path):
    import numpy as np
    from .evaluation.regenerate_d7 import reuse_scene

    cached = tmp_path / 'cache/clips/clip/point4d'
    cached.mkdir(parents=True)
    clip = {'gf': {'length': 3}, 'portrait': False, 'source_hw': [48, 80]}
    arrays = dict(gf_index=[0, 1, 2], model_hw=[294, 518], geometry_stride=4, source_hw=[48, 80])
    np.savez(cached / 'scene.npz', **arrays)
    (cached / 'metric.json').write_text(json.dumps({'metres_per_unit': None}))
    out = tmp_path / 'matching'
    provenance = reuse_scene(tmp_path / 'cache', out, 'clip', clip)
    assert len(provenance) == 1 and provenance[0]['source'].endswith('scene.npz')
    assert (out / 'clips/clip/point4d/scene.npz').is_file()
    assert not (out / 'clips/clip/point4d/metric.json').exists()
    np.savez(cached / 'scene.npz', **{**arrays, 'gf_index': [0, 2]})
    assert reuse_scene(tmp_path / 'cache', tmp_path / 'mismatch', 'clip', clip) == []
    assert not (tmp_path / 'mismatch').exists()
