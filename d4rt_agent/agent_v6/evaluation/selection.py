"""Parameterized, deterministic selection from previously answered question pools."""
from collections import defaultdict
import random

from ..metadata import reporting_metadata

DEFAULT_COUNTS = {'dsi': 40, 'dynbench': 70, 'vlm4d': 70}
DSI_SPLITS = frozenset({'std', 'reverse', 'hflip', 'reverse_hflip'})


def select_quick_check(pools, counts=None, seed=0):
    """Read identity and quantity strata only; never answers or earlier correctness.

    DSI pools contain all four rendering records per question. The returned IDs
    apply across splits; records explicitly preserve those renderings together.
    This function is not invoked by the runner: the real draw awaits approval.
    """
    counts = DEFAULT_COUNTS if counts is None else counts
    result, records = {}, []
    for bench, count in sorted(counts.items()):
        if count < 0:
            raise ValueError('selection counts must be nonnegative')
        groups = defaultdict(list)
        for row in pools[bench]:
            groups[str(row.get('qid', row.get('id')))].append(row)
        strata = defaultdict(list)
        for qid, variants in sorted(groups.items()):
            if bench == 'dsi' and (len(variants) != 4 or {r['split'] for r in variants} != DSI_SPLITS):
                raise ValueError(f'DSI {qid} needs all four renderings')
            quantities = {r['quantity_group'] for r in variants}
            if None in quantities or '' in quantities:
                raise ValueError(f'missing quantity metadata for {qid}')
            if len(quantities) != 1:
                raise ValueError(f'inconsistent quantity group for {qid}')
            strata[next(iter(quantities))].append(qid)
        if sum(map(len, strata.values())) < count:
            raise ValueError(f'{bench} has too few eligible question IDs')
        rng = random.Random(f'{seed}:{bench}')
        keys = sorted(strata)
        for members in strata.values():
            rng.shuffle(members)
        rng.shuffle(keys)
        chosen = []
        while len(chosen) < count:
            for key in keys:
                if strata[key] and len(chosen) < count:
                    chosen.append(strata[key].pop())
        result[bench] = chosen
        for qid in chosen:
            for row in sorted(groups[qid], key=lambda r: r.get('split', '')):
                records.append({'bench': bench, 'qid': qid, 'split': row.get('split'),
                                **reporting_metadata(row)})
    return {'schema_version': 6, 'seed': seed, 'counts': dict(counts), 'question_ids': result,
            'records': records, 'selection': 'seeded round-robin quantity strata; answer-blind'}
