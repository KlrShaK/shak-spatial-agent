"""Freeze answer-blind selections and existing comparisons on a SLURM CPU node."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import shutil

from d4rt_agent.agent_v5.run import write_json
from d4rt_agent.agent_v5.research.snapshot import snapshot
from d4rt_agent.agent_v6.run import parse_args, load_entries
from d4rt_agent.agent_v6.metadata import enrich_entries, reporting_metadata
from .selection import select_quick_check
from .campaign import SPLITS

MAIN = Path('/cluster/work/igp_psr/spanwar/shak-spatial-agent')
RESULTS = MAIN / 'd4rt_agent/results'
V4_RUNS = dict(zip(SPLITS, ('agent_v4_random200', 'agent_v4_reverse', 'agent_v4_hflip', 'agent_v4_reverse_hflip')))
COMP = RESULTS / 'agent_v5_subset_20261014T021547Z_retry'


def read(path):
    return json.loads(Path(path).read_text())


def rebuild_dyn_inputs(destination):
    """Restore identities from saved predictions and recover decisions from logs.

    No clips are decoded, no new draw is made, and no model is run here. Deleted
    D7 predictions are regenerated separately on the selected quick-check only.
    """
    from dyn_dataset_exploration.dyn_bench_data import load_rows, qa_parquet
    from dyn_dataset_exploration.decomposition.program import rule_parse

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    previous = RESULTS / 'agent_v5_eval2800_point4d_20261005T183754Z/answers'
    recorded = {row['question_id']: row['reference'] for row in (read(p) for p in sorted(previous.glob('*.json')))}
    if len(recorded) != 2800:
        raise ValueError('recovery requires the original 2800 recorded question identities')
    entries = []
    for row in load_rows(qa_parquet()):
        if row.question_id not in recorded:
            continue
        ref = recorded[row.question_id]
        if ref['answer'] != row.answer or ref['options'] != row.options:
            raise ValueError(f'original Dyn reference mismatch: {row.question_id}')
        try:
            program = rule_parse(row.question_id, row.question, row.options)
        except ValueError:
            program = None
        kind = program.kind if program else 'unparsed'
        if ref.get('group') != kind:
            raise ValueError(f'original Dyn rule label changed: {row.question_id}')
        entries.append(dict(qid=row.question_id, dataset=row.dataset, video=row.video,
                            clip_slug=row.video_slug, task=row.task, question=row.question,
                            options=row.options, answer=row.answer, branch=kind,
                            rule_kind=program.kind if program else None, split='eval'))
    if len(entries) != 2800:
        raise ValueError('raw benchmark no longer contains all original evaluation IDs')
    catalog = destination / 'catalog.json'
    write_json(catalog, {'questions': entries, 'recovered_from': str(previous),
                         'selection': 'exact recorded IDs; no resampling'})
    decisions, provenance = {}, []
    logs = sorted((MAIN / 'logs/decomposition').glob('prep_ev-*.out'), key=lambda p: (p.stat().st_mtime_ns, p.name))
    for path in logs:
        payload = path.read_bytes()
        used = 0
        for mode, qid, letter in re.findall(r'\b(answer|letters|evidence)\s+(\S+):\s+([A-Z]|None)\s*$',
                                          payload.decode(errors='replace'), re.M):
            if qid in recorded:
                decisions.setdefault(qid, {})[{'answer': 'frames'}.get(mode, mode)] = None if letter == 'None' else letter
                used += 1
        if used:
            provenance.append({'source': str(path), 'sha256': hashlib.sha256(payload).hexdigest(), 'decisions': used})
    rows = [{'qid': entry['qid'], 'gt': entry['answer'], **decisions.get(entry['qid'], {})} for entry in entries]
    recovered = destination / 'recovered_predictions.json'
    write_json(recovered, rows)
    write_json(destination / 'recovery_provenance.json', {'logs': provenance, 'catalog': str(catalog),
                'coverage': {column: sum(column in row for row in rows) for column in ('frames', 'letters', 'evidence')},
                'd7': 'Deleted; regenerate only selected quick-check questions using committed calibrated.eval.json.'})
    return catalog, recovered


def normalize_prediction(row, entry, bench, split):
    """Use benchmark identities for paired joins; preserve the existing decision/score."""
    from d4rt_agent.dsi_bench_data import parse_choice_letter

    ref = row.get('reference', row)
    gold = ref.get('answer', ref.get('gt'))
    if isinstance(gold, str) and gold.strip().upper() in entry['options']:
        gold = gold.strip().upper()
    if gold is not None and gold != entry['answer']:
        raise ValueError(f'reference mismatch: {bench}/{entry["id"]}: {gold} != {entry["answer"]}')
    options = ref.get('options')
    if isinstance(options, list):
        options = {text.split('.', 1)[0].strip(): text.split('.', 1)[1].strip() for text in options}
    if options is not None and options != entry['options']:
        raise ValueError(f'option mismatch: {bench}/{entry["id"]}')
    if ref.get('question') is not None and ref['question'].strip() != entry['question'].strip():
        raise ValueError(f'question mismatch: {bench}/{entry["id"]}')
    correct = row.get('correct')
    letter = row.get('letter', row.get('parsed_letter', row.get('pred')))
    if letter is None and isinstance(row.get('answer_text'), str):
        letter = parse_choice_letter(row['answer_text'], entry['options'])
    if letter is None and isinstance(row.get('prediction'), str):
        letter = parse_choice_letter(row['prediction'], entry['options'])
    answered = row.get('answered', letter is not None)
    if isinstance(row.get('accuracy'), (int, float)):
        correct = row['accuracy'] == 1
    if not isinstance(correct, bool):
        correct = bool(answered and
                       letter in [entry['answer'], *entry.get('answer_also', [])])
    return {'question_id': entry['id'], 'bench': bench, 'correct': correct,
            'answered': answered, 'letter': letter,
            'clip': {'relative_path': entry.get('relative_path') or f'{entry["dataset"]}/{entry["video"]}',
                     'split': split if bench == 'dsi' else None},
            'reference': {'answer': entry['answer'], 'options': entry['options'], **reporting_metadata(entry)}}


def paper_reference_mismatches(row, entry):
    """Paper IDs alone do not identify questions across benchmark revisions."""
    expected = {'question': entry['question'], 'choices': entry['options'],
                'ground_truth_answer': entry['options'][entry['answer']]}
    actual = {field: row.get(field) for field in expected}
    # The benchmark loader uses text options even when the raw counts are JSON numbers.
    if isinstance(actual['choices'], dict):
        actual['choices'] = {str(k): str(v).strip() for k, v in actual['choices'].items()}
    for field in ('question', 'ground_truth_answer'):
        if actual[field] is not None:
            actual[field] = str(actual[field]).strip()
        expected[field] = str(expected[field]).strip()
    return [field for field, value in expected.items() if actual[field] != value]


def prepare(out, seed):
    out = Path(out).resolve()
    if (out / 'plan.json').exists():
        raise ValueError('campaign already prepared; reuse its immutable plan')
    out.mkdir(parents=True, exist_ok=True)
    inputs = out / 'inputs'
    inputs.mkdir(exist_ok=True)
    manifests, populations = {}, {}
    provenance = []
    print('Freezing catalogs and existing prediction sets.', flush=True)

    def copied(source, name):
        source = Path(source)
        target = inputs / name
        shutil.copy2(source, target)
        provenance.append({'source': str(source), 'copied': str(target),
                           'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
        return str(target)

    original_catalog = MAIN / 'dyn_dataset_exploration/outputs/decomposition_eval/catalog.json'
    d7 = MAIN / 'dyn_dataset_exploration/outputs/decomposition_eval/eval_rows.json'
    regenerated_d7 = not d7.is_file()
    if not original_catalog.is_file() or regenerated_d7:
        recovered_catalog, recovered_predictions = rebuild_dyn_inputs(out / 'recovery')
        original_catalog = original_catalog if original_catalog.is_file() else recovered_catalog
        d7 = d7 if d7.is_file() else recovered_predictions
    catalog = copied(original_catalog, 'dyn_catalog.json')
    vlm_metadata = copied(MAIN / 'vlm4d_dataset_exploration/logs/manifest.json', 'vlm_metadata.json')
    for split, run in V4_RUNS.items():
        manifests[split] = copied(RESULTS / run / 'manifest.json', f'dsi_{split}.json')
        args = parse_args(['--bench', 'dsi', '--manifest', manifests[split], '--split', split])
        entries = enrich_entries(load_entries(args), 'dsi', split=split, metadata=[])
        if len(entries) != 200 or len({e['id'] for e in entries}) != 200:
            raise ValueError('expected V4 200-question manifest')
        populations['dsi', split] = {e['id']: e for e in entries}
    base_ids = set(populations['dsi', 'std'])
    if any(set(populations['dsi', split]) != base_ids for split in SPLITS):
        raise ValueError('DSI rendering manifests have different question IDs')
    for bench, metadata in [('dynbench', catalog), ('vlm4d', vlm_metadata)]:
        args = parse_args(['--bench', bench, '--catalog', catalog])
        entries = enrich_entries(load_entries(args), bench, metadata_path=metadata)
        expected = {'dynbench': 2800, 'vlm4d': 1816}[bench]
        if len(entries) != expected or len({e['id'] for e in entries}) != expected:
            raise ValueError(f'{bench}: expected {expected} catalog questions')
        populations[bench, 'all'] = {e['id']: e for e in entries}

    baselines = {bench: {} for bench in ('dsi', 'dynbench', 'vlm4d')}

    def add_runs(bench, name, directories):
        rows, hashes, paths = {}, hashlib.sha256(), []
        for directory, split in directories:
            directory = Path(directory)
            folder = directory / 'answers' if (directory / 'answers').is_dir() else directory / 'baseline'
            files = sorted(path for path in folder.glob('*.json') if path.stem in populations[bench, split])
            if not files:
                raise ValueError(f'no existing predictions in {directory}')
            paths.append(str(directory))
            for path in files:
                payload = path.read_bytes()
                hashes.update(path.name.encode() + b'\0' + payload)
                row = json.loads(payload)
                qid = str(row['question_id'])
                entry = populations[bench, split].get(qid)
                if entry is None:
                    continue
                key = f'{bench}:{split}:{qid}'
                if key in rows:
                    raise ValueError(f'duplicate baseline prediction {key}')
                rows[key] = normalize_prediction(row, entry, bench, split)
        baselines[bench][name] = rows
        print(f'{bench}/{name}: {len(rows)} matched records', flush=True)
        provenance.append({'baseline': name, 'bench': bench, 'sources': paths,
                           'sha256_of_sorted_filenames_and_bytes': hashes.hexdigest(), 'matched_records': len(rows)})
        return rows

    add_runs('dsi', 'v4_random200', [(RESULTS / V4_RUNS[s], s) for s in SPLITS])
    # Locate the four existing control renderings without using their accuracy.
    controls = []
    for split in SPLITS:
        choices = [RESULTS / f'control200_{split}']
        if split == 'std':
            choices += [RESULTS / 'control200', RESULTS / 'control200_std', RESULTS / 'agent_v4_random200_control']
        found = [p for p in choices if (p / 'answers').is_dir() or (p / 'baseline').is_dir()]
        if not found:
            raise ValueError(f'missing existing DSI frames-only rendering {split}')
        controls.append((found[0], split))
    add_runs('dsi', 'frames_only', controls)
    comp_dsi = add_runs('dsi', 'v5_compositional', [(COMP / 'dsi' / s, s) for s in SPLITS])
    add_runs('dsi', 'v5_legacy_point4d', [(RESULTS / 'agent_v5_dsi800_point4d_20261010T012544Z', 'std')])
    census = RESULTS / 'census/std'
    if (census / 'answers').is_dir():
        add_runs('dsi', 'v4_census_std', [(census, 'std')])
    add_runs('dynbench', 'v5_point4d', [(RESULTS / 'agent_v5_eval2800_point4d_20261005T183754Z', 'all')])
    add_runs('dynbench', 'v5_d4rt', [(RESULTS / 'agent_v5_eval2800_d4rt_20261006T230228Z', 'all')])
    add_runs('dynbench', 'qwen_official', [(RESULTS / 'qwen3vl8b_dynbench_official_20261007T175618Z', 'all')])
    add_runs('vlm4d', 'v5_legacy', [(RESULTS / 'agent_v5_vlm4d_point4d_20261010T012544Z', 'all')])
    comp_vlm = add_runs('vlm4d', 'v5_compositional', [(COMP / 'vlm4d', 'all')])

    provenance.append({'source': str(d7), 'sha256': hashlib.sha256(d7.read_bytes()).hexdigest()})
    for column, name in [('frames', 'frames_only'), ('letters', 'option_log_probs'),
                         ('evidence', 'qwen_with_evidence'), ('pipeline', 'd7')]:
        rows = {}
        for row in read(d7):
            qid = str(row.get('qid', row.get('question_id')))
            entry = populations['dynbench', 'all'].get(qid)
            if entry is None:
                continue
            if row['gt'] != entry['answer']:
                raise ValueError(f'D7 reference mismatch: {qid}')
            if column not in row:
                continue
            prediction = row[column]
            rows[f'dynbench:all:{qid}'] = normalize_prediction(
                {'letter': prediction, 'answered': prediction is not None,
                 'correct': prediction == row['gt']}, entry, 'dynbench', 'all')
        if rows:
            baselines['dynbench'][name] = rows

    from d4rt_agent.agent_v5.vlm4d_report import paper_models, DEFAULT_REPO
    paper_exclusions = []
    for prompt in ('cot', 'direct-output'):
        for model in paper_models(DEFAULT_REPO, prompt):
            source = {}
            for split in ('real', 'synthetic'):
                path = DEFAULT_REPO / 'processed_outputs_paper_results' / f'{split}_mc_{prompt}' / model
                payload = path.read_bytes()
                provenance.append({'source': str(path), 'sha256': hashlib.sha256(payload).hexdigest()})
                for item in json.loads(payload):
                    key = split, int(item['id'].rsplit('_', 1)[-1])
                    if key in source:
                        raise ValueError(f'duplicate paper ID: {prompt}/{model}/{key}')
                    source[key] = item
            rows = {}
            for qid, entry in populations['vlm4d', 'all'].items():
                key = entry['task'], int(qid.rsplit('_', 1)[-1])
                if key in source:
                    item = source[key]
                    mismatches = paper_reference_mismatches(item, entry)
                    if mismatches:
                        paper_exclusions.append({'prompt': prompt, 'model': model, 'qid': qid,
                                                 'mismatched_fields': mismatches,
                                                 'paper_reference': {field: item.get(field) for field in mismatches}})
                        continue
                    rows[f'vlm4d:all:{qid}'] = normalize_prediction(
                        {'correct': item['correct'], 'answered': True}, entry, 'vlm4d', 'all')
            baselines['vlm4d'][f'paper:{prompt}:{model}'] = rows
            print(f'vlm4d/paper:{prompt}:{model}: {len(rows)} compatible records', flush=True)
    write_json(inputs / 'paper_reference_exclusions.json', paper_exclusions)
    print(f'Excluded {len(paper_exclusions)} incompatible paper records; identities and selection are unchanged.', flush=True)

    # Availability defines the pools; no previous answer or correctness enters the selector.
    dsi_ids = {qid for qid in base_ids if all(f'dsi:{s}:{qid}' in comp_dsi for s in SPLITS)}
    vlm_ids = {key.split(':', 2)[2] for key in comp_vlm}
    if len(dsi_ids) != 63 or len(vlm_ids) != 300:
        raise ValueError(f'unexpected compositional pool sizes: DSI {len(dsi_ids)}, VLM4D {len(vlm_ids)}')
    pools = {'dsi': [{'qid': qid, 'split': s, **reporting_metadata(populations['dsi', s][qid])}
                     for qid in sorted(dsi_ids) for s in SPLITS],
             'dynbench': [{'qid': qid, **reporting_metadata(e)} for qid, e in populations['dynbench', 'all'].items()],
             'vlm4d': [{'qid': qid, **reporting_metadata(populations['vlm4d', 'all'][qid])}
                       for qid in sorted(vlm_ids)]}
    selection = select_quick_check(pools, seed=seed)
    write_json(inputs / 'quick_selection.json', selection)
    write_json(inputs / 'selection_pools.json', pools)
    full_ids = {'dsi': sorted(base_ids), 'dynbench': sorted(populations['dynbench', 'all']),
                'vlm4d': random.Random(seed).sample(sorted(populations['vlm4d', 'all']), 1000)}
    full = {'schema_version': 6, 'seed': seed, 'question_ids': full_ids,
            'selection': 'DSI existing V4 draw; Dyn 2800 split; VLM4D seeded uniform 1000; answer-blind', 'records': []}
    for bench, ids in full_ids.items():
        for qid in ids:
            for split in SPLITS if bench == 'dsi' else ('all',):
                full['records'].append({'bench': bench, 'qid': qid,
                                        'split': split if bench == 'dsi' else None,
                                        **reporting_metadata(populations[bench, split][qid])})
    write_json(inputs / 'full_selection.json', full)
    metadata = {'dynbench': catalog, 'vlm4d': vlm_metadata, 'dsi': str(inputs / 'dsi_metadata.json')}
    write_json(Path(metadata['dsi']), {'records': [r for r in full['records'] if r['bench'] == 'dsi']})
    baseline_paths = {}
    for bench, systems in baselines.items():
        baseline_paths[bench] = str(inputs / f'{bench}_baselines.json')
        write_json(Path(baseline_paths[bench]), systems)
    gate_baselines = {'dsi': {'v5': 'v5_compositional', 'frames_only': 'frames_only', 'v4': 'v4_random200'},
                      'dynbench': {'v5': 'v5_point4d', 'frames_only': 'frames_only'},
                      'vlm4d': {'v5': 'v5_legacy'}}
    for bench, settings in gate_baselines.items():
        wanted = {f'{bench}:{split}:{qid}' for qid in selection['question_ids'][bench]
                  for split in SPLITS if bench == 'dsi'} if bench == 'dsi' else {
                      f'{bench}:all:{qid}' for qid in selection['question_ids'][bench]}
        for name in settings.values():
            if name == 'frames_only' and bench == 'dynbench' and regenerated_d7:
                continue  # Missing recorded decisions are regenerated on the quick-check only.
            if wanted - set(baselines[bench][name]):
                raise ValueError(f'{bench}/{name}: incomplete selected baseline coverage')
    models = sorted((Path('/cluster/work/igp_psr/spanwar/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots')).iterdir())
    models = [p for p in models if (p / 'config.json').is_file()]
    if len(models) != 1:
        raise ValueError(f'expected one pinned local Qwen snapshot, found {models}')
    source = out / 'source'
    snapshot(MAIN, source)
    plan = {'schema_version': 6, 'created_at': datetime.now(timezone.utc).isoformat(), 'seed': seed,
            'results': str(out), 'source': str(source),
            'source_sha256': hashlib.sha256((source / 'source_manifest.json').read_bytes()).hexdigest(),
            'qwen_model': str(models[0]), 'third_party_dir': str(MAIN / 'third-party-tools'),
            'catalog': catalog, 'metadata': metadata, 'manifests': manifests,
            'quick_selection': str(inputs / 'quick_selection.json'), 'full_selection': str(inputs / 'full_selection.json'),
            'baselines': baseline_paths, 'gate_baselines': gate_baselines, 'provenance': provenance,
            'regenerate_d7': regenerated_d7,
            'd7_cache_source': str(RESULTS / 'agent_v5_eval2800_point4d_20261005T183754Z/cache'),
            'dyn_root': '/cluster/work/igp_psr/spanwar/datasets/Dyn-Bench',
            'authorization': 'User requested quick-check evaluation and full evaluation conditional on the design gate passing.',
            'gate_policy': 'Every benchmark passes #72 before any full run; no accuracy-based retries or reselection.'}
    plan['input_hashes'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs.glob('*.json'))}
    write_json(out / 'plan.json', plan)
    print(json.dumps({'plan': str(out / 'plan.json'), 'records': len(selection['records']),
                      'strata': dict(Counter(f"{r['bench']}:{r['quantity_group']}" for r in selection['records']))}, default=str), flush=True)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261018)
    args = parser.parse_args()
    prepare(args.out, args.seed)


if __name__ == '__main__':
    main()
