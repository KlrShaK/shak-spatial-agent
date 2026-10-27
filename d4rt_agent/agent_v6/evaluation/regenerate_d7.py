"""Regenerate deleted D7 comparisons on the frozen quick-check, one packed clip at a time."""
from __future__ import annotations

from collections import defaultdict
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

from d4rt_agent.agent_v5.run import write_json
from d4rt_agent.agent_v6.cache import ClipArchive


def reuse_scene(cache, out, slug, clip):
    """The V5 Point4D adapter stores the original D7 scene layout and sampling."""
    import numpy as np

    if not cache:
        return []
    base = Path(cache) / 'clips' / slug / 'point4d'
    scene = base / 'scene.npz'
    if not scene.is_file():
        return []
    expected = np.arange(0, clip['gf']['length'], int(np.ceil(clip['gf']['length'] / 400)))
    with np.load(scene) as arrays:
        hw = (518, 294) if clip['portrait'] else (294, 518)
        if (not np.array_equal(arrays['gf_index'], expected)
                or tuple(arrays['model_hw']) != hw or int(arrays['geometry_stride']) != 4
                or tuple(arrays['source_hw']) != tuple(clip['source_hw'])):
            return []
    destination = out / 'clips' / slug / 'point4d'
    destination.mkdir(parents=True, exist_ok=True)
    provenance = []
    for name in ('scene.npz', 'metric.json'):
        source = base / name
        if source.is_file():
            if name == 'metric.json':
                metric = json.loads(source.read_text())
                scale = metric.get('metres_per_unit')
                if scale is None or not np.isfinite(scale) or scale <= 0:
                    continue
            shutil.copy2(source, destination / name)
            digest = hashlib.sha256()
            with source.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
            provenance.append({'source': str(source), 'sha256': digest.hexdigest()})
    return provenance


def regenerate(plan):
    from dyn_dataset_exploration.decomposition.build_catalog import clip_record, materialize_frames
    from dyn_dataset_exploration.decomposition.program import rule_parse

    root = Path(plan['results']) / 'regenerated_d7'
    target = root / 'predictions.json'
    rows = json.loads(target.read_text()) if target.is_file() else {}
    selection = json.loads(Path(plan['quick_selection']).read_text())
    wanted = set(selection['question_ids']['dynbench'])
    catalog = json.loads(Path(plan['catalog']).read_text())['questions']
    groups = defaultdict(list)
    for entry in catalog:
        if entry['qid'] in wanted:
            groups[entry['clip_slug']].append(entry)
    recorded = json.loads(Path(plan['baselines']['dynbench']).read_text())
    env = dict(os.environ, DYN_CALIBRATED=str(Path(plan['source']) /
               'dyn_dataset_exploration/decomposition/calibrated.eval.json'))

    def stage(module, out, *arguments):
        args = [sys.executable, '-u', '-m', module, '--out', str(out), *arguments]
        print(json.dumps({'regeneration_command': args}), flush=True)
        subprocess.run(args, check=True, cwd=plan['source'], env=env)

    with ClipArchive(root / 'cache', 'point4d') as archive:
        for slug, entries in sorted(groups.items()):
            if all(f'dynbench:all:{entry["qid"]}' in rows for entry in entries):
                continue
            out = archive.activate(slug)
            write_json(out / 'catalog.json', {'questions': entries})
            first = entries[0]
            clip = clip_record(Path(plan['dyn_root']), first['dataset'], first['video'])
            # GT object diagnostics are irrelevant to this regeneration.
            clip.pop('gt_objects', None)
            clip.pop('gt_masks_dir', None)
            write_json(out / 'clips' / slug / 'clip.json', clip)
            materialize_frames(clip, out / 'clips' / slug)
            for entry in entries:
                try:
                    program = rule_parse(entry['qid'], entry['question'], entry['options'])
                except ValueError:
                    program = None
                if program:
                    write_json(out / 'programs' / f'{entry["qid"]}.rules.json', program.to_dict())
                    write_json(out / 'programs' / f'{entry["qid"]}.pipeline.json', program.to_dict())
                key = f'dynbench:all:{entry["qid"]}'
                if key in recorded.get('frames_only', {}):
                    write_json(out / 'answers' / f'{entry["qid"]}.frames_only.json',
                               {'answer': recorded['frames_only'][key]['letter'], 'source': 'recovered existing log'})
            prep = 'dyn_dataset_exploration.decomposition.prep_qwen_sam3'
            stage(prep, out, '--qwen-model', plan['qwen_model'], '--stages', 'parse', 'name')
            programs = [json.loads((out / 'programs' / f'{entry["qid"]}.pipeline.json').read_text()) for entry in entries]
            measured = [program for program in programs if program.get('route') == 'measure']
            reused = []
            if measured:
                reused = reuse_scene(plan.get('d7_cache_source'), out, slug, clip)
                if any(entity.get('phrase') and entity.get('kind') != 'camera'
                       for program in measured for entity in program.get('entities', [])):
                    stage(prep, out, '--stages', 'ground')
                point4d = 'dyn_dataset_exploration.decomposition.prep_point4d'
                if not (out / 'clips' / slug / 'point4d/scene.npz').is_file():
                    stage(point4d, out, '--stages', 'scene')
                if (any(item.get('metric') for program in measured for item in program.get('measurements', []))
                        and not (out / 'clips' / slug / 'point4d/metric.json').is_file()):
                    stage(point4d, out, '--stages', 'metric')
            stage(prep, out, '--qwen-model', plan['qwen_model'], '--stages', 'answer', 'letters')
            # The committed original calibration defines which leaf-B questions
            # need an evidence answer. No policy is fitted on these questions.
            collect = Path(__file__).with_name('regenerate_d7.py')
            subprocess.run([sys.executable, '-u', str(collect), '--collect', str(out),
                            '--qwen-model', plan['qwen_model']], check=True, cwd=plan['source'], env=env)
            solutions = json.loads((out / 'solutions.json').read_text())
            for entry in entries:
                key = f'dynbench:all:{entry["qid"]}'
                solution = solutions[entry['qid']]
                letter = solution['final']
                rows[key] = {'bench': 'dynbench', 'question_id': entry['qid'],
                             'letter': letter, 'answered': letter is not None,
                             'frames_letter': solution['frames_only'],
                             'correct': letter == entry['answer'],
                             'clip': {'relative_path': f'{entry["dataset"]}/{entry["video"]}', 'split': None},
                             'reference': {'answer': entry['answer'], 'options': entry['options']},
                             'regeneration': 'original D7 code and committed calibrated.eval.json',
                             'reused_scene_inputs': reused,
                             'solution': solution}
            write_json(target, rows)
            print(f'Regenerated D7: {len(rows)}/{len(wanted)} quick-check questions', flush=True)
    if {row['question_id'] for row in rows.values()} != wanted:
        raise ValueError('regenerated D7 coverage differs from the frozen quick-check')
    return rows


def collect(out, model):
    from dyn_dataset_exploration.decomposition import interpret, pipeline

    context = pipeline.Context(out, max_clips=1)
    if not pipeline.CALIBRATION_LOADED:
        raise ValueError('the original evaluation calibration is required')
    for qid in context.catalog:
        solution = pipeline.solve(context, qid)
        if solution.decision.leaf == 'B' and pipeline.FALLBACK_BY_KIND.get(solution.program.kind) == 'evidence':
            path = out / 'results' / f'{qid}.evidence.txt'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(interpret.evidence_text(solution.program, solution.results, solution.decision))
    if list((out / 'results').glob('*.evidence.txt')):
        from d4rt_agent.simple_v2 import OfflineQwen
        from dyn_dataset_exploration.decomposition.prep_qwen_sam3 import Asker, stage_evidence
        stage_evidence(out, list(context.catalog.values()), Asker(OfflineQwen(model, max_new_tokens=400)))
    write_json(out / 'solutions.json', {qid: pipeline.solve(context, qid).to_dict() for qid in context.catalog})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect', type=Path, required=True)
    parser.add_argument('--qwen-model', required=True)
    args = parser.parse_args()
    collect(args.collect, args.qwen_model)
