"""Accuracy, process/cost summaries, and explicit offline contradiction audits."""
from collections import Counter, defaultdict
import json
import hashlib

from d4rt_agent.agent_v5.research.protocol import score


def accuracy(rows):
    values = list(rows)
    return {'questions': len(values), 'accuracy': sum(score(r) for r in values) / len(values) if values else None}


def summarize(rows):
    rows = list(rows)
    groups = {name: defaultdict(list) for name in ('benchmark', 'quantity', 'official', 'dyn_level_hypothesis')}
    for row in rows:
        ref = row.get('reference', {})
        bench = row.get('bench', 'unknown')
        groups['benchmark'][bench].append(row)
        groups['quantity'][ref.get('quantity_group') or row.get('quantity_group') or 'unclassified'].append(row)
        if bench == 'dynbench':
            # Shipped task names are observed; their paper-level mapping is only a hypothesis.
            if ref.get('level_status') == 'official' and ref.get('level'):
                official = 'level:' + ref['level']
            else:
                official = 'task:' + str(ref.get('task') or 'unclassified')
            if ref.get('hypothesis_level'):
                groups['dyn_level_hypothesis'][ref['hypothesis_level']].append(row)
        elif bench == 'dsi':
            category = ref.get('cate', ref.get('group'))
            official = f'c{category}' if category is not None else ref.get('task')
        else:
            official = ref.get('task')
        groups['official'][f'{bench}:{official or "unclassified"}'].append(row)
    reports = [report for row in rows for report in row.get('reports', [])]
    processes = [row.get('process', {}) for row in rows]
    dsi_groups = defaultdict(dict)
    for row in rows:
        if row.get('bench') == 'dsi':
            split = row.get('clip', {}).get('split')
            key = row['question_id']
            if split in dsi_groups[key]:
                raise ValueError(f'duplicate DSI rendering {key}/{split}')
            dsi_groups[key][split] = row
    complete = [list(g.values()) for g in dsi_groups.values() if
                set(g) == {'std', 'reverse', 'hflip', 'reverse_hflip'}]
    cost = {}
    for key in ('wall_seconds', 'model_calls', 'output_tokens', 'peak_context_tokens', 'peak_gpu_gb'):
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        cost[key] = {'mean': sum(values) / len(values) if values else None,
                     'max': max(values) if values else None, 'recorded': len(values)}
    planned = sum(p.get('planned_items', 0) for p in processes)
    executed = sum(p.get('executed_items', 0) for p in processes)
    audit = [row['flip_audit'] for row in rows if row.get('flip_audit', {}).get('status') == 'ok']
    return {'overall': accuracy(rows),
            'breakdown_notes': {'dynbench': 'Task labels are reported directly. Paper levels inferred from TASK_TO_LEVEL are hypotheses, not official per-question labels.'},
            **{name: {k: accuracy(v) for k, v in sorted(group.items())} for name, group in groups.items()},
            'dsi_groupwise': {'complete_groups': len(complete), 'incomplete_groups': len(dsi_groups) - len(complete),
                              'accuracy_at_least_three': sum(sum(score(r) for r in g) >= 3 for g in complete) / len(complete) if complete else None},
            'process': {**{key: sum(p.get(key, 0) for p in processes) for key in
                           ('normalizer_fixes', 'doctor_fixes', 'sent_back', 'doctor_calls')},
                        'unavailable_rate': sum(r.get('status') == 'unavailable' for r in reports) / len(reports) if reports else None,
                        'planned_items': planned, 'executed_items': executed,
                        'plan_adherence': executed / planned if planned else None,
                        'flip_audited': len(audit),
                        'reason_flip_rate': sum(a['reason_contradiction'] for a in audit) / len(audit) if audit else None,
                        'report_flip_rate': sum(a['report_contradiction'] and not a['override_supported'] for a in audit) / len(audit) if audit else None},
            'letter_distribution': dict(Counter(row.get('letter') or 'unanswered' for row in rows)), 'cost': cost}


def audit_packet(row):
    """Only evidence needed to inspect the prediction; never its score or gold."""
    options = row.get('reference', {}).get('options', {})
    packet = {'chosen_option': options.get(row.get('letter')), 'because': row.get('because', ''),
              'last_report': row.get('reports', [])[-1] if row.get('reports') else None}
    if packet['last_report']:
        packet['last_report'] = {k: packet['last_report'].get(k) for k in ('headline', 'status', 'reason')}
    return packet


def audit_fingerprint(row):
    payload = {'answered': row.get('answered'), 'letter': row.get('letter'), **audit_packet(row)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def audit_predictions(rows, qwen):
    """One consolidated map keyed by benchmark, rendering and question ID."""
    return {'schema_version': 6, 'audits': {
        key: {**flip_audit(row, qwen), 'prediction_sha256': audit_fingerprint(row)}
        for key, row in sorted(rows.items())}}


def attach_audits(rows, payload):
    audits = payload.get('audits', payload)
    unknown = set(audits) - set(rows)
    if unknown:
        raise ValueError(f'audit file has {len(unknown)} unknown prediction IDs')
    result = {}
    for key, row in rows.items():
        copy = dict(row)
        if key in audits:
            audit = audits[key]
            if audit.get('prediction_sha256') != audit_fingerprint(row):
                raise ValueError(f'audit does not match the current prediction: {key}')
            copy['flip_audit'] = audit
        result[key] = copy
    return result


def flip_audit(row, qwen):
    """Separate local-Qwen audit: never sees ground truth or modifies a prediction."""
    options = row.get('reference', {}).get('options', {})
    if not row.get('answered') or row.get('letter') not in options:
        return {'status': 'unanswered'}
    from pathlib import Path
    instruction = (Path(__file__).parents[1] / 'prompts/flip_audit.md').read_text()
    packet = audit_packet(row)
    response = qwen.generate([{'role': 'system', 'content': [{'type': 'text', 'text': instruction}]},
                              {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps(packet)}]}])
    from ..calls import parse_object
    try:
        result = parse_object(response)
        keys = ('reason_contradiction', 'report_contradiction', 'override_supported')
        if not all(isinstance(result.get(k), bool) for k in keys):
            raise ValueError('audit needs three booleans')
        return {'status': 'ok', **{k: result[k] for k in keys}, 'explanation': result.get('explanation', '')}
    except ValueError:
        return {'status': 'unavailable', 'raw': response}
