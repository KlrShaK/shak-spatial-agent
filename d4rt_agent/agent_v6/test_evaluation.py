from copy import deepcopy
import pytest
from .evaluation.selection import select_quick_check, DSI_SPLITS
from .evaluation.compare import compare_existing, gate
from .evaluation.metrics import summarize, flip_audit
from .test_calls import Scripted


def row(qid, correct, split='std', bench='dsi'):
    return dict(question_id=qid, correct=correct, answered=True, letter='B', because='distance falls',
                bench=bench, clip={'relative_path': f'{split}/same-source.mp4', 'split': split},
                reference={'options': {'A':'away','B':'closer'}, 'answer':'B', 'quantity_group':'distance', 'task':'c4'})


def test_selection_is_deterministic_answer_blind_and_keeps_dsi_together():
    pool = {'dsi': [dict(qid=str(i), split=s, quantity_group=str(i % 2), answer='A', correct=False)
                    for i in range(10) for s in DSI_SPLITS]}
    selected = select_quick_check(pool, {'dsi':3}, 42)
    changed = deepcopy(pool)
    changed['dsi'].reverse()
    for r in changed['dsi']:
        r.update(answer='D', correct=True)
    assert select_quick_check(changed, {'dsi':3}, 42) == selected
    assert len(selected['records']) == 12
    assert all('answer' not in r and 'correct' not in r for r in selected['records'])


def test_incomplete_dsi_pool_is_rejected():
    with pytest.raises(ValueError, match='four'):
        select_quick_check({'dsi':[dict(qid='x', split='std', quantity_group='motion')]}, {'dsi':1})


def test_paired_comparison_clusters_all_renderings_and_questions_by_source_clip():
    candidate = {f'{s}:{i}': row(str(i), True, s) for s in DSI_SPLITS for i in range(2)}
    baseline = {k: dict(r, correct=False) for k, r in candidate.items()}
    result = compare_existing(candidate, {'v5':baseline}, draws=100)['v5']
    assert result['source_clusters'] == 1 and result['questions'] == 8
    assert result['paired_95_ci'] == [1.0,1.0]


def test_gate_checks_paired_gain_quantity_loss_and_reversed_v4():
    candidate = {s: row('q', True, s) for s in DSI_SPLITS}
    baseline = {s: row('q', False, s) for s in DSI_SPLITS}
    assert gate(candidate, {'v5':baseline,'frames':baseline,'v4':baseline}, v5='v5', frames_only='frames', v4='v4', draws=20)['passed']
    assert not gate(candidate, {}, v5='missing', draws=20)['passed']
    candidate['reverse']['correct'] = False
    v4 = {s: row('q', True, s) for s in DSI_SPLITS}
    assert not gate(candidate, {'v5':baseline,'v4':v4}, v5='v5', v4='v4', draws=20)['passed']


def test_metrics_distinguish_incomplete_dsi_groups_and_missing_audit():
    rows = [row('q', s != 'reverse', s) for s in DSI_SPLITS] + [row('partial', True)]
    result = summarize(rows)
    assert result['dsi_groupwise']['accuracy_at_least_three'] == 1
    assert result['dsi_groupwise']['incomplete_groups'] == 1
    assert result['process']['reason_flip_rate'] is None


def test_flip_audit_has_no_ground_truth_and_does_not_change_answer():
    qwen = Scripted('{"reason_contradiction":false,"report_contradiction":false,"override_supported":false}')
    source = row('q', True)
    source['reference']['answer'] = 'SECRET_GOLD'
    result = flip_audit(source, qwen)
    assert result['status'] == 'ok'
    assert 'SECRET_GOLD' not in str(qwen.calls)
    assert source['letter'] == 'B'


def test_selection_preserves_reporting_metadata_without_gold():
    source = {'dynbench': [{'qid': 'q', 'quantity_group': 'zoom', 'task': 'cameraqa',
                           'level_status': 'hypothesis', 'hypothesis_level': 'Camera reasoning',
                           'level_provenance': 'hypothesis: TASK_TO_LEVEL', 'rule_kind': 'camera_zoom',
                           'answer': 'SECRET', 'correct': True}]}
    record = select_quick_check(source, {'dynbench': 1})['records'][0]
    assert record['rule_kind'] == 'camera_zoom'
    assert record['level_status'] == 'hypothesis'
    assert 'answer' not in record and 'correct' not in record


@pytest.mark.parametrize('missing', ['std', 'reverse', 'hflip', 'reverse_hflip'])
def test_gate_never_accepts_incomplete_dsi_renderings(missing):
    candidate = {s: row('q', True, s) for s in DSI_SPLITS if s != missing}
    baseline = {s: row('q', False, s) for s in candidate}
    result = gate(candidate, {'v5': baseline, 'frames': baseline, 'v4': baseline},
                  v5='v5', frames_only='frames', v4='v4', draws=10)
    assert not result['passed']
    assert any('all four renderings' in reason for reason in result['reasons'])


def test_gate_checks_full_approved_selection_even_if_paired_subset_wins():
    candidate = {s: row('q', True, s) for s in DSI_SPLITS}
    baseline = {s: row('q', False, s) for s in DSI_SPLITS}
    result = gate(candidate, {'v5': baseline, 'frames': baseline, 'v4': baseline},
                  v5='v5', frames_only='frames', v4='v4', expected_ids=['q', 'missing-q'], draws=10)
    assert not result['passed']
    assert any('approved selection is incomplete' in reason for reason in result['reasons'])


def test_gate_accepts_complete_approved_manifest_and_rejects_extra_records():
    candidate = {s: row('q', True, s) for s in DSI_SPLITS}
    baseline = {s: row('q', False, s) for s in DSI_SPLITS}
    selection = {'records': [{'bench': 'dsi', 'qid': 'q', 'split': s} for s in DSI_SPLITS]}
    arguments = dict(v5='v5', frames_only='frames', v4='v4', draws=10)
    baselines = {'v5': baseline, 'frames': baseline, 'v4': baseline}
    assert gate(candidate, baselines, expected_ids=selection, **arguments)['passed']
    selection['records'] = [{'bench': 'dsi', 'qid': 'other', 'split': s} for s in DSI_SPLITS]
    result = gate(candidate, baselines, expected_ids=selection, **arguments)
    assert not result['passed']
    assert any('outside the approved selection' in reason for reason in result['reasons'])


def test_official_dyn_breakdown_uses_known_task_and_separates_level_hypothesis():
    source = row('q', True, bench='dynbench')
    source['reference'].update(task='cameraqa', level=None,
                               hypothesis_level='Dynamic Camera-Object Reasoning', level_status='hypothesis')
    result = summarize([source])
    assert 'dynbench:task:cameraqa' in result['official']
    assert 'Dynamic Camera-Object Reasoning' in result['dyn_level_hypothesis']
    assert not any('Reasoning' in key for key in result['official'])


def test_consolidated_audit_attaches_without_modifying_predictions_or_seeing_gold():
    from .evaluation.metrics import audit_predictions, attach_audits
    source = {'dsi:std:q': row('q', True)}
    source['dsi:std:q']['reference']['answer'] = 'SECRET_GOLD'
    qwen = Scripted('{"reason_contradiction":true,"report_contradiction":false,"override_supported":false}')
    audits = audit_predictions(source, qwen)
    attached = attach_audits(source, audits)
    assert 'SECRET_GOLD' not in str(qwen.calls)
    assert 'flip_audit' not in source['dsi:std:q']
    assert summarize(attached.values())['process']['reason_flip_rate'] == 1
    assert 'dsi:std:q' in audits['audits']
    source['dsi:std:q']['because'] = 'A changed rationale'
    with pytest.raises(ValueError, match='does not match'):
        attach_audits(source, audits)


def test_audit_and_metrics_cli_use_local_mock_and_consolidated_file(tmp_path, monkeypatch):
    import json
    from .evaluation.__main__ import main
    from d4rt_agent import simple_v2
    run = tmp_path / 'run'
    (run / 'answers').mkdir(parents=True)
    (run / 'run_config.json').write_text(json.dumps({'bench': 'dsi', 'split': 'std'}))
    (run / 'answers/q.json').write_text(json.dumps(row('q', True)))
    qwen = Scripted('{"reason_contradiction":false,"report_contradiction":true,"override_supported":false}')
    calls = []
    def local_model(model, **kwargs):
        calls.append((model, kwargs))
        return qwen
    monkeypatch.setattr(simple_v2, 'OfflineQwen', local_model)
    audit_file, metric_file = tmp_path / 'audit.json', tmp_path / 'metrics.json'
    main(['audit', str(run), '--qwen-model', '/local/mock-qwen', '--out', str(audit_file)])
    assert calls[0][0] == '/local/mock-qwen'
    main(['metrics', str(run), '--audit-file', str(audit_file), '--out', str(metric_file)])
    result = json.loads(metric_file.read_text())
    assert result['process']['report_flip_rate'] == 1
    assert result['process']['flip_audited'] == 1


def test_selection_rejects_missing_quantity_metadata():
    with pytest.raises(ValueError, match='missing quantity'):
        select_quick_check({'dynbench': [{'qid': 'q', 'quantity_group': None}]}, {'dynbench': 1})


def test_approved_question_ids_cannot_be_hidden_by_incomplete_manifest_records():
    candidate = {s: row('q', True, s) for s in DSI_SPLITS}
    baseline = {s: row('q', False, s) for s in DSI_SPLITS}
    selection = {'question_ids': {'dsi': ['q', 'missing']}, 'records': [
        {'bench': 'dsi', 'qid': 'q', 'split': s} for s in DSI_SPLITS]}
    result = gate(candidate, {'v5': baseline, 'frames': baseline, 'v4': baseline},
                  v5='v5', frames_only='frames', v4='v4', expected_ids=selection, draws=10)
    assert not result['passed']
    assert any('approved selection is incomplete' in reason for reason in result['reasons'])
