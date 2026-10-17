"""Synthetic loader/manifest shapes only; never read benchmark datasets here."""

import json

from .metadata import enrich_entries


def test_dsi_loader_category_maps_to_quantity_without_metadata():
    entries = [{'id': 'q', 'group': 4, 'task': 'distance change'}]
    result = enrich_entries(entries, 'dsi', metadata=[])
    assert result[0]['quantity_group'] == 'entity_camera'
    assert result[0]['cate'] == 4
    assert entries[0] == {'id': 'q', 'group': 4, 'task': 'distance change'}


def test_dyn_loader_group_maps_quantity_and_task_level_stays_hypothetical():
    result = enrich_entries([{'id': 'q', 'group': 'camera_path', 'task': 'cameraqa'}],
                            'dynbench', metadata=[])[0]
    assert result['quantity_group'] == 'camera_motion'
    assert result['level'] is None
    assert result['hypothesis_level'] == 'Dynamic Camera-Object Reasoning'
    assert result['level_status'] == 'hypothesis'
    assert result['level_provenance'].startswith('hypothesis:')


def test_dyn_catalog_rule_kind_overrides_umbrella_branch(tmp_path):
    path = tmp_path / 'catalog.json'
    path.write_text(json.dumps({'questions': [{'qid': 'q', 'branch': 'camera',
                                              'rule_kind': 'camera_zoom', 'task': 'cameraqa'}]}))
    result = enrich_entries([{'id': 'q', 'group': 'camera', 'task': 'cameraqa'}],
                            'dynbench', catalog_path=path)[0]
    assert result['quantity_group'] == 'zoom'
    assert result['rule_kind'] == 'camera_zoom'


def test_vlm4d_manifest_kind_joins_by_qid_and_preserves_real_split(tmp_path):
    path = tmp_path / 'metadata.json'
    path.write_text(json.dumps({'questions': [{'qid': 'vlm_q', 'kind': 'subject_turn', 'answer': 'SECRET'}]}))
    result = enrich_entries([{'id': 'vlm_q', 'task': 'real', 'group': 'ego4d'}],
                            'vlm4d', metadata_path=path)[0]
    assert result['quantity_group'] == 'rotation'
    assert result['task'] == 'real'
    assert 'answer' not in result


def test_selected_record_metadata_survives_identity_only_loader_selection():
    selection = {'question_ids': {'dsi': ['q']}, 'records': [
        {'bench': 'dsi', 'qid': 'q', 'split': 'std', 'quantity_group': 'entity_motion', 'cate': 1},
        {'bench': 'dsi', 'qid': 'q', 'split': 'reverse', 'quantity_group': 'entity_motion', 'cate': 1}]}
    result = enrich_entries([{'id': 'q', 'group': 4}], 'dsi', split='reverse',
                            metadata=[], selection=selection)[0]
    assert result['quantity_group'] == 'entity_motion'
    assert result['cate'] == 1
    assert result['quantity_provenance'] == 'selection manifest'


def test_explicit_official_level_requires_provenance_status():
    metadata = [{'qid': 'q', 'level': 'Verified level', 'level_status': 'official',
                 'level_provenance': 'released per-question annotations', 'rule_kind': 'subject_direction'}]
    result = enrich_entries([{'id': 'q', 'task': 'qa'}], 'dynbench', metadata=metadata)[0]
    assert result['level'] == 'Verified level'
    assert result['level_status'] == 'official'
    assert result['level_provenance'] == 'released per-question annotations'
    assert 'hypothesis_level' not in result


def test_supplied_unverified_level_does_not_become_official():
    result = enrich_entries([{'id': 'q', 'task': 'qa'}], 'dynbench',
                            metadata=[{'qid': 'q', 'level': 'Informal level'}])[0]
    assert result['level_status'] == 'provided_unverified'


def test_unknown_quantity_is_marked_missing_instead_of_guessing():
    result = enrich_entries([{'id': 'q', 'task': 'real'}], 'vlm4d', metadata=[])[0]
    assert result['quantity_group'] is None
    assert result['quantity_provenance'].startswith('unavailable:')


def test_default_manifest_lazily_falls_back_to_shared_artifact(tmp_path, monkeypatch):
    from . import metadata as module
    shared = tmp_path / 'shared-manifest.json'
    shared.write_text(json.dumps({'questions': [{'qid': 'q', 'kind': 'subject_turn'}]}))
    monkeypatch.setitem(module.DEFAULT_METADATA, 'vlm4d', tmp_path / 'missing-snapshot-manifest.json')
    monkeypatch.setitem(module.SHARED_METADATA, 'vlm4d', shared)
    result = enrich_entries([{'id': 'q', 'task': 'real'}], 'vlm4d')[0]
    assert result['quantity_group'] == 'rotation'


def test_existing_local_manifest_precedes_shared_fallback(tmp_path, monkeypatch):
    from . import metadata as module
    local, shared = tmp_path / 'local.json', tmp_path / 'shared.json'
    local.write_text(json.dumps({'questions': [{'qid': 'q', 'kind': 'subject_direction'}]}))
    shared.write_text(json.dumps({'questions': [{'qid': 'q', 'kind': 'subject_turn'}]}))
    monkeypatch.setitem(module.DEFAULT_METADATA, 'vlm4d', local)
    monkeypatch.setitem(module.SHARED_METADATA, 'vlm4d', shared)
    assert enrich_entries([{'id': 'q'}], 'vlm4d')[0]['quantity_group'] == 'entity_motion'


def test_explicit_metadata_path_precedes_shared_fallback_and_never_hides_missing_file(tmp_path, monkeypatch):
    import pytest
    from . import metadata as module
    explicit, shared = tmp_path / 'explicit.json', tmp_path / 'shared.json'
    explicit.write_text(json.dumps({'questions': [{'qid': 'q', 'kind': 'subject_direction'}]}))
    shared.write_text(json.dumps({'questions': [{'qid': 'q', 'kind': 'subject_turn'}]}))
    monkeypatch.setitem(module.DEFAULT_METADATA, 'vlm4d', tmp_path / 'missing-local.json')
    monkeypatch.setitem(module.SHARED_METADATA, 'vlm4d', shared)
    assert enrich_entries([{'id': 'q'}], 'vlm4d', metadata_path=explicit)[0]['quantity_group'] == 'entity_motion'
    with pytest.raises(FileNotFoundError):
        enrich_entries([{'id': 'q'}], 'vlm4d', metadata_path=tmp_path / 'missing-explicit.json')


def test_supplied_empty_metadata_bypasses_all_manifest_reads(monkeypatch):
    from pathlib import Path
    def forbidden_read(*args, **kwargs):
        raise AssertionError('metadata=[] must not read any manifest')
    monkeypatch.setattr(Path, 'read_text', forbidden_read)
    result = enrich_entries([{'id': 'q', 'task': 'real'}], 'vlm4d', metadata=[])[0]
    assert result['quantity_group'] is None
