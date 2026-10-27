"""Small catalog fixtures; real dataset counts belong to a CPU SLURM check."""
import json
from pathlib import Path

import pytest

from . import run


@pytest.fixture(autouse=True)
def synthetic_metadata_only(monkeypatch):
    monkeypatch.setattr(run, 'enrich_entries', lambda entries, *args, **kwargs: entries)


@pytest.mark.parametrize('bench', ['dsi', 'dynbench', 'vlm4d'])
def test_dry_run_only_writes_stubs(tmp_path, monkeypatch, bench):
    entries = [dict(id=f'q{i}', question='Which way?', options={'A': 'left', 'B': 'right'},
                    answer='B', dataset='synthetic', video='clip', video_path='clip.mp4') for i in range(7)]
    monkeypatch.setattr(run, 'load_entries', lambda args: entries)
    monkeypatch.setattr(run, 'open_clip', lambda *args: pytest.fail('dry run opened a clip'))
    args = run.parse_args(['--bench', bench, '--dry-run', '--limit', '5', '--results-dir', str(tmp_path)])
    assert run.run(args) == 0
    files = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*') if p.is_file())
    assert len(files) == 5
    assert all(p.parent == Path('answers') for p in files)
    assert all(json.loads((tmp_path / p).read_text())['status'] == 'dry_run' for p in files)


def test_dsi_loader_does_not_persist_manifest(tmp_path, monkeypatch):
    from d4rt_agent import dsi_bench_data
    manifest = {'questions': [dict(question_id='q0', question='Where?', options={'A': 'here'}, gt='A',
                                   relative_path='video.mp4', video_path='/video.mp4')]}
    monkeypatch.setattr(dsi_bench_data, 'build_manifest', lambda **kwargs: manifest)
    args = run.parse_args(['--results-dir', str(tmp_path)])
    assert run.load_entries(args)[0]['id'] == 'q0'
    assert not list(tmp_path.iterdir())


def test_answer_reference_is_separate_and_accepts_equivalent_letters():
    args = run.parse_args([])
    entry = dict(id='q', question='Where?', options={'A': 'here', 'B': 'here'}, answer='A', answer_also=['B'])
    record = run.answer_record(entry, args, dict(status='ok', answered=True, letter='B'))
    assert record['correct']
    assert record['reference']['answer'] == 'A'


def test_dry_run_does_not_overwrite_live_answer(tmp_path, monkeypatch):
    entry = dict(id='q', question='Where?', options={'A': 'here'}, answer='A', video_path='clip.mp4')
    monkeypatch.setattr(run, 'load_entries', lambda args: [entry])
    (tmp_path / 'answers').mkdir()
    (tmp_path / 'answers/q.json').write_text('{"status":"ok"}')
    with pytest.raises(ValueError, match='overwrite'):
        run.run(run.parse_args(['--dry-run', '--results-dir', str(tmp_path)]))


def test_packed_cache_one_archive_per_clip_and_backbone(tmp_path):
    from .cache import ClipArchive
    with ClipArchive(tmp_path, 'point4d') as archive:
        archive.activate('video-a')
        (archive.directory / 'mask.txt').write_text('mask data')
        archive.activate('video-b')
        assert not (archive.directory / 'mask.txt').exists()
        archive.activate('video-a')
        assert (archive.directory / 'mask.txt').read_text() == 'mask data'
    assert len(list(tmp_path.rglob('*.tar.gz'))) == 2
    assert len([p for p in tmp_path.rglob('*') if p.is_file()]) == 2


@pytest.mark.parametrize('model_arg', ['./models/qwen', 'Qwen/Qwen3-VL-8B-Instruct'])
def test_live_reexecutes_frozen_source_before_importing_models(tmp_path, monkeypatch, model_arg):
    from d4rt_agent.agent_v5.research import snapshot
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'models/qwen').mkdir(parents=True)
    args = run.parse_args(['--results-dir', str(tmp_path / 'results'), '--qwen-model', model_arg])
    args._dataset_sha256 = 'catalog'
    calls = []
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setenv('PYTHONPATH', '/external/point4d:external/relative:.')
    def freeze(root, dest):
        dest.mkdir(parents=True)
        (dest / 'source_manifest.json').write_text('{}')
    monkeypatch.setattr(snapshot, 'snapshot', freeze)
    monkeypatch.setattr(run.subprocess, 'call', lambda command, **kwargs: calls.append((command, kwargs)) or 0)
    assert run.run_live([], args) == 0
    command, kwargs = calls[0]
    assert command[1:3] == ['-m', 'd4rt_agent.agent_v6.run']
    assert kwargs['cwd'] == args.results_dir / 'source'
    assert kwargs['env']['PYTHONPATH'].split(':') == [str((args.results_dir / 'source').resolve()),
                                                    '/external/point4d', str(tmp_path / 'external/relative')]
    model = command[command.index('--qwen-model') + 1]
    assert model == (str(tmp_path / 'models/qwen') if model_arg.startswith('./') else model_arg)


def test_fail_fast_records_first_runtime_error_before_stopping(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    from d4rt_agent.agent_v5 import models
    (tmp_path / 'source_manifest.json').write_text('{}')
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(models, 'Models', lambda **kwargs: SimpleNamespace(qwen=lambda: object()))
    monkeypatch.setattr(models, 'gpu_memory_report', lambda: {})
    monkeypatch.setenv('SHAK_THIRD_PARTY', str(tmp_path))
    opened = []
    def unavailable(entry, *args):
        opened.append(entry['id'])
        raise RuntimeError('fixture video cannot open')
    monkeypatch.setattr(run, 'open_clip', unavailable)
    args = run.parse_args(['--fail-fast', '--results-dir', str(tmp_path / 'results')])
    args._dataset_sha256 = 'fixture'
    entries = [dict(id=f'q{i}', video_path=f'clip{i}', question='Where?',
                    options={'A': 'left'}, answer='A') for i in range(2)]
    with pytest.raises(RuntimeError, match='fixture video cannot open'):
        run.run_live(entries, args)
    assert opened == ['q0']
    saved = json.loads((args.results_dir / 'answers/q0.json').read_text())
    assert saved['status'] == 'error' and not saved['answered']
    assert not (args.results_dir / 'answers/q1.json').exists()
    assert 'raise RuntimeError' in capsys.readouterr().err
