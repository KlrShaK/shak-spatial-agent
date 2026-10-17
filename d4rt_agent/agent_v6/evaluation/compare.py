"""Paired comparisons and a gate that requires substantive evidence."""
from collections import defaultdict

from .selection import DSI_SPLITS
from d4rt_agent.agent_v5.research.protocol import clip_key, paired_compare, prediction_rows, score


def compare_existing(candidate, baselines, seed=0, draws=10000):
    comparisons = {}
    for name, baseline in baselines.items():
        common = sorted(set(candidate) & set(baseline))
        if not common:
            comparisons[name] = {'status': 'no_overlap', 'questions': 0}
            continue
        result = paired_compare({k: baseline[k] for k in common}, {k: candidate[k] for k in common}, seed, draws)
        result.update(status='ok', candidate_unpaired=len(candidate) - len(common),
                      baseline_unpaired=len(baseline) - len(common))
        comparisons[name] = result
    return comparisons


def _identity(row):
    bench = row.get('bench')
    split = row.get('clip', {}).get('split') if bench == 'dsi' else 'all'
    return bench, split, str(row['question_id'])


def expected_coverage(candidate, expected_ids):
    """Expected IDs may be a selection manifest, benchmark→IDs, or one ID list."""
    if expected_ids is None:
        return []
    actual = {_identity(row) for row in candidate.values()}
    benches = {bench for bench, _, _ in actual}
    if isinstance(expected_ids, dict) and 'records' in expected_ids:
        expected = {(row['bench'], row.get('split') if row['bench'] == 'dsi' else 'all',
                     str(row.get('qid', row.get('id', row.get('question_id')))))
                    for row in expected_ids['records'] if row['bench'] in benches}
        for bench, qids in expected_ids.get('question_ids', {}).items():
            if bench in benches:
                expected.update((bench, split, str(qid)) for qid in qids
                                for split in (DSI_SPLITS if bench == 'dsi' else ('all',)))
    else:
        if isinstance(expected_ids, dict):
            ids = expected_ids.get('question_ids', expected_ids)
        elif len(benches) == 1:
            ids = {next(iter(benches)): expected_ids}
        else:
            return ['expected IDs require an explicit benchmark mapping']
        expected = {(bench, split, str(qid)) for bench in benches for qid in ids.get(bench, [])
                    for split in (DSI_SPLITS if bench == 'dsi' else ('all',))}
    reasons = []
    if expected - actual:
        reasons.append(f'approved selection is incomplete: {len(expected - actual)} missing rendering records')
    if actual - expected:
        reasons.append(f'candidate has {len(actual - expected)} records outside the approved selection')
    return reasons


def gate(candidate, baselines, *, v5, frames_only=None, v4=None, expected_ids=None, seed=0, draws=10000):
    """Require positive paired bounds, <=5-point quantity losses and DSI reversal parity.

    Call per benchmark on its actual quick-check records. Explicit baseline names
    identify the existing prediction artifacts; missing comparisons fail closed.
    """
    comparisons = compare_existing(candidate, baselines, seed, draws)
    reasons = expected_coverage(candidate, expected_ids)
    dsi = defaultdict(list)
    for row in candidate.values():
        if row.get('bench') == 'dsi':
            dsi[str(row['question_id'])].append(row.get('clip', {}).get('split'))
    for qid, splits in dsi.items():
        if len(splits) != 4 or set(splits) != DSI_SPLITS:
            reasons.append(f'DSI {qid}: all four renderings are required exactly once')
    if any(row.get('bench') in ('dsi', 'dynbench') for row in candidate.values()) and frames_only is None:
        reasons.append('frames-only Qwen comparison is required for this benchmark')
    for name in [v5, *([frames_only] if frames_only else [])]:
        result = comparisons.get(name, {})
        if result.get('status') != 'ok' or result.get('candidate_unpaired') != 0:
            reasons.append(f'{name}: incomplete paired coverage')
        elif not result['positive_interval']:
            reasons.append(f'{name}: paired lower bound is not positive')
    quantities = {}
    for key, row in candidate.items():
        quantity = row.get('quantity_group') or row.get('reference', {}).get('quantity_group')
        if not quantity or quantity == 'unclassified':
            reasons.append('quantity labels are incomplete')
            break
        quantities.setdefault(quantity, []).append(key)
    group_deltas = {}
    for quantity, keys in quantities.items():
        deltas = []
        for name, baseline in baselines.items():
            common = [key for key in keys if key in baseline]
            if common:
                delta = sum(score(candidate[k]) - score(baseline[k]) for k in common) / len(common)
                deltas.append(delta)
                if delta < -0.05 - 1e-12:
                    reasons.append(f'{quantity}: more than 5 points below {name}')
        if not deltas:
            reasons.append(f'{quantity}: no existing comparison')
        group_deltas[quantity] = min(deltas) if deltas else None
    reversed_keys = [k for k, row in candidate.items() if row.get('bench') == 'dsi' and
                     row.get('clip', {}).get('split') in ('reverse', 'reverse_hflip')]
    if any(row.get('bench') == 'dsi' for row in candidate.values()):
        if not reversed_keys or v4 not in baselines or any(k not in baselines[v4] for k in reversed_keys):
            reasons.append('DSI reversed comparison against V4 is incomplete')
        elif sum(score(candidate[k]) - score(baselines[v4][k]) for k in reversed_keys) < 0:
            reasons.append('DSI reversed accuracy is below V4')
    if not candidate:
        reasons.append('no candidate predictions')
    return {'passed': not reasons, 'reasons': sorted(set(reasons)), 'comparisons': comparisons,
            'worst_quantity_deltas': group_deltas}
