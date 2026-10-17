"""Reporting-only benchmark metadata; never supplied to the reasoning agent.

Dyn's coarse task labels are known. Their correspondence to the paper's levels
is an explicitly documented hypothesis, so it is stored separately from any
verified level supplied by an external manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

from d4rt_agent.agent_v6.design.analysis.quantity_census import DSI, DYN, VLM4D, REPO as SHARED_REPO
from dyn_dataset_exploration.dyn_bench_data import TASK_TO_LEVEL

REPO = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = {
    'dynbench': REPO / 'dyn_dataset_exploration/outputs/decomposition_eval/catalog.json',
    'vlm4d': REPO / 'vlm4d_dataset_exploration/logs/manifest.json',
}
SHARED_METADATA = {bench: SHARED_REPO / path.relative_to(REPO)
                   for bench, path in DEFAULT_METADATA.items()}
REPORTING_FIELDS = (
    'quantity_group', 'quantity_provenance', 'cate', 'category_name', 'task', 'group',
    'branch', 'rule_kind', 'kind', 'dataset', 'level', 'hypothesis_level',
    'level_status', 'level_provenance', 'level_source',
)


def reporting_metadata(row):
    """An allowlist prevents answers and correctness entering selected manifests."""
    return {key: row[key] for key in REPORTING_FIELDS if row.get(key) is not None}


def _rows(payload):
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return payload.get('records', payload.get('questions', []))
    raise ValueError('metadata must contain a records or questions list')


def _index(payload, bench):
    result = {}
    for row in _rows(payload):
        if row.get('bench', bench) != bench:
            continue
        qid = row.get('qid', row.get('question_id', row.get('id')))
        if qid is None:
            raise ValueError('metadata row needs qid, question_id or id')
        split = row.get('split') if bench == 'dsi' else None
        key = (str(qid), split)
        value = reporting_metadata(row)
        if key in result and result[key] != value:
            raise ValueError(f'conflicting metadata for {bench}/{qid}/{split}')
        result[key] = value
    return result


def _category(entry):
    value = entry.get('cate', entry.get('group'))
    try:
        return int(str(value).removeprefix('c'))
    except (ValueError, TypeError):
        return None


def enrich_entries(entries, bench, *, split=None, metadata_path=None,
                   selection_file=None, catalog_path=None, metadata=None, selection=None):
    """Copy entries and attach labels from metadata and an approved selection.

    Priority is selection record, metadata manifest, then loader fields. Passing
    ``metadata=[]`` avoids all default file reads, useful for synthetic tests.
    Reads happen only when called by a runner, never at import time.
    """
    if bench not in ('dsi', 'dynbench', 'vlm4d'):
        raise ValueError(f'unknown benchmark: {bench}')
    source = 'provided metadata'
    if metadata is None:
        path = metadata_path or (catalog_path if bench == 'dynbench' else None) or DEFAULT_METADATA.get(bench)
        # Frozen snapshots and worktrees omit ignored reporting artifacts. Resolve
        # their shared copy only on demand; explicit paths must never fall back.
        if metadata_path is None and path is not None and not Path(path).is_file():
            path = SHARED_METADATA.get(bench, path)
        if path is not None and (metadata_path is not None or Path(path).is_file()):
            path = Path(path)
            metadata = json.loads(path.read_text())
            source = str(path)
        else:
            metadata = []
    if selection is None and selection_file is not None:
        selection = json.loads(Path(selection_file).read_text())
    sources = [(source, _index(metadata, bench)),
               ('selection manifest' if selection_file is None else str(selection_file), _index(selection, bench))]
    result = []
    for original in entries:
        entry = dict(original)
        qid = str(entry['id'])
        quantity_source = entry.get('quantity_provenance', 'loader metadata')
        level_source = entry.get('level_provenance', entry.get('level_source'))
        for name, index in sources:
            extra = index.get((qid, split), index.get((qid, None), {}))
            if extra.get('quantity_group') is not None:
                quantity_source = extra.get('quantity_provenance', name)
            if extra.get('level') is not None:
                level_source = extra.get('level_provenance', extra.get('level_source', name))
            entry.update(extra)
        if bench == 'dsi':
            category = _category(entry)
            if category is not None:
                entry['cate'] = category
            derived = DSI.get(category)
            rule = 'DSI category'
        elif bench == 'dynbench':
            kind = entry.get('rule_kind') or entry.get('branch') or entry.get('group')
            derived = DYN.get(kind)
            rule = 'Dyn decomposition kind'
            if entry.get('level') is not None:
                entry['level_provenance'] = level_source or 'provided metadata (unverified)'
                entry.setdefault('level_status', 'provided_unverified')
            else:
                entry['level'] = None
                entry['hypothesis_level'] = TASK_TO_LEVEL.get(entry.get('task'))
                entry['level_status'] = 'hypothesis' if entry['hypothesis_level'] else 'unavailable'
                entry['level_provenance'] = ('hypothesis: dyn_dataset_exploration.dyn_bench_data.TASK_TO_LEVEL'
                                             if entry['hypothesis_level'] else 'no verified per-question level supplied')
        else:
            derived = VLM4D.get(entry.get('kind'))
            rule = 'VLM4D manifest kind'
            if entry.get('task') not in ('real', 'synthetic'):
                dataset = entry.get('dataset', '')
                if dataset in ('vlm4d_real', 'vlm4d_synthetic'):
                    entry['task'] = dataset.removeprefix('vlm4d_')
        if entry.get('quantity_group') is None:
            entry['quantity_group'] = derived
            quantity_source = ('design/analysis/quantity_census.py: ' + rule if derived is not None
                               else 'unavailable: no mapped reporting label')
        entry['quantity_provenance'] = quantity_source
        result.append(entry)
    return result
