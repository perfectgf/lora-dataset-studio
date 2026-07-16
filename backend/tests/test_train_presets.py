"""Training presets: named snapshots of the advanced settings, import/export
friendly. The core promise is SCHEMA TOLERANCE — a preset from another app
version applies with unknown keys ignored and invalid values reported, never
a hard failure.
"""


def _create_ds(client, name='Preset', trigger='pres', train_type='krea',
               kind=None):
    return client.post('/api/dataset/create',
                       json={'name': name, 'trigger_word': trigger,
                             'train_type': train_type,
                             'kind': kind}).get_json()['id']


def test_save_from_dataset_snapshot_and_list(client, app):
    ds_id = _create_ds(client)
    # give the dataset one explicit setting to snapshot
    with app.app_context():
        from app.services import lora_training as lt
        lt.update_train_settings('local', ds_id, {'rank': 32})
    r = client.post('/api/train/presets', json={'name': 'My Krea', 'dataset_id': ds_id})
    assert r.status_code == 200
    body = r.get_json()
    assert body['ok'] is True and body['created'] is True
    assert body['train_type'] == 'krea'
    assert body['dataset_kind'] == 'character'
    assert body['variants'] == ['base']
    assert body['settings'] == {'rank': 32}
    listed = client.get('/api/train/presets').get_json()['presets']
    assert any(p['name'] == 'My Krea' and p['settings'] == {'rank': 32} for p in listed)


def test_save_overwrites_by_name(client):
    r1 = client.post('/api/train/presets',
                     json={'name': 'Dup', 'train_type': 'zimage', 'settings': {'rank': 16}})
    r2 = client.post('/api/train/presets',
                     json={'name': 'Dup', 'train_type': 'zimage', 'settings': {'rank': 64}})
    assert r1.get_json()['created'] is True
    assert r2.get_json()['created'] is False
    listed = client.get('/api/train/presets').get_json()['presets']
    assert [p['settings'] for p in listed if p['name'] == 'Dup'] == [{'rank': 64}]


def test_apply_is_schema_tolerant(client, app):
    """Unknown keys (future app versions) are ignored, invalid values rejected,
    valid keys land — all in one call, never fatal."""
    ds_id = _create_ds(client)
    r = client.post(f'/api/dataset/{ds_id}/train/presets/apply', json={'settings': {
        'rank': 32,                       # valid → applied
        'dropout': 0.1,                   # valid → applied
        'rank_v2_search_space': [1, 2],   # unknown → ignored
        'save_every': 123,                # invalid value → rejected
    }})
    assert r.status_code == 200
    body = r.get_json()
    assert body['ok'] is True
    assert body['ignored'] == ['rank_v2_search_space']
    assert [x['key'] for x in body['rejected']] == ['save_every']
    with app.app_context():
        from app.services import lora_training as lt
        stored = lt.snapshot_train_settings('local', ds_id)
        assert stored == {'rank': 32, 'dropout': 0.1}


def test_apply_replaces_previous_settings(client, app):
    """A preset REPLACES the explicit settings — keys absent from the preset
    fall back to defaults instead of surviving from before."""
    ds_id = _create_ds(client)
    with app.app_context():
        from app.services import lora_training as lt
        lt.update_train_settings('local', ds_id, {'rank': 64, 'dropout': 0.3})
    client.post(f'/api/dataset/{ds_id}/train/presets/apply',
                json={'settings': {'rank': 16}})
    with app.app_context():
        from app.services import lora_training as lt
        assert lt.snapshot_train_settings('local', ds_id) == {'rank': 16}


def test_builtin_presets_listed_first_and_undeletable(client):
    listed = client.get('/api/train/presets').get_json()['presets']
    assert listed and listed[0]['id'] == 'builtin-krea-character'
    assert listed[0]['builtin'] is True
    assert listed[0]['dataset_kind'] == 'character'
    ids = {p['id'] for p in listed}
    assert 'builtin-style' not in ids
    assert {
        'builtin-style-krea-raw',
        'builtin-style-klein-base',
        'builtin-style-zimage-base',
        'builtin-style-flux1',
        'builtin-style-sdxl',
    } <= ids
    # the delete route only matches integer ids — built-ins are unreachable
    assert client.delete('/api/train/presets/builtin-krea-character').status_code == 404


def test_style_builtin_catalogue_has_researched_family_settings(client):
    listed = client.get('/api/train/presets').get_json()['presets']
    styles = {p['id']: p for p in listed if p.get('dataset_kind') == 'style'}
    assert styles['builtin-style-krea-raw']['variants'] == ['base', 'raw']
    assert styles['builtin-style-klein-base']['variants'] == ['4b', '9b']
    assert styles['builtin-style-zimage-base']['variants'] == ['base']
    expected = {
        'builtin-style-krea-raw': (32, 32, '768,1024', 'linear'),
        'builtin-style-klein-base': (32, 32, '768,1024', 'weighted'),
        'builtin-style-zimage-base': (32, 32, '768,1024', 'weighted'),
        'builtin-style-flux1': (16, 16, '768,1024', 'weighted'),
        'builtin-style-sdxl': (32, 16, '1024', None),
    }
    for preset_id, (rank, alpha, resolution, timestep) in expected.items():
        settings = styles[preset_id]['settings']
        assert (settings['rank'], settings['alpha'], settings['resolution']) == (
            rank, alpha, resolution)
        assert settings.get('timestep_type') == timestep
        assert settings['save_every'] == settings['sample_every'] == 250
        assert settings['max_step_saves'] == 10
        assert len(settings['sample_prompts']) == 8
        assert all('{trigger}' not in prompt for prompt in settings['sample_prompts'])
        # ``dropout`` is network dropout, not caption dropout. Style caption
        # dropout is applied by the service's family policy.
        assert 'dropout' not in settings
        assert 'ema' not in settings


def test_every_builtin_applies_cleanly(client, app):
    """The shipped presets must ALWAYS apply with zero ignored keys and zero
    rejected values — this is the guard that catches a choice-list drifting
    away from what the built-ins promise."""
    from app.services.lora_training import BUILTIN_TRAIN_PRESETS
    ds_id = _create_ds(client)
    for preset in BUILTIN_TRAIN_PRESETS:
        r = client.post(f'/api/dataset/{ds_id}/train/presets/apply',
                        json={'settings': preset['settings']})
        body = r.get_json()
        assert body['ok'] is True, preset['id']
        assert body['ignored'] == [], preset['id']
        assert body['rejected'] == [], preset['id']


def test_apply_by_preset_id_and_delete(client):
    ds_id = _create_ds(client)
    pid = client.post('/api/train/presets',
                      json={'name': 'ById', 'train_type': 'krea',
                            'settings': {'rank': 32}}).get_json()['id']
    r = client.post(f'/api/dataset/{ds_id}/train/presets/apply', json={'preset_id': pid})
    assert r.get_json()['ok'] is True
    assert client.delete(f'/api/train/presets/{pid}').get_json()['ok'] is True
    assert client.delete(f'/api/train/presets/{pid}').status_code == 404


def test_apply_style_builtin_by_string_id_and_legacy_alias(client, app):
    ds_id = _create_ds(client, train_type='krea', kind='style')
    url = f'/api/dataset/{ds_id}/train/presets/apply'
    r = client.post(url, json={
        'preset_id': 'builtin-style-krea-raw',
        'train_type': 'krea',
        'variant': 'base',
    })
    assert r.status_code == 200
    body = r.get_json()
    assert body['preset_id'] == 'builtin-style-krea-raw'
    assert body['ignored'] == [] and body['rejected'] == []
    assert body['train_settings']['effective_rank'] == 32
    with app.app_context():
        from app.services import lora_training as lt
        stored = lt.snapshot_train_settings('local', ds_id)
        assert stored['rank'] == stored['alpha'] == 32
        assert 'dropout' not in stored and 'ema' not in stored

    alias = client.post(url, json={
        'preset_id': 'builtin-style',
        'train_type': 'krea',
        'variant': 'raw',
    })
    assert alias.status_code == 200
    assert alias.get_json()['preset_id'] == 'builtin-style-krea-raw'


def test_builtin_scope_mismatches_never_mutate_dataset(client, app):
    cases = [
        # kind mismatch
        ('krea', None, 'base', 'builtin-style-krea-raw'),
        # family mismatch
        ('zimage', 'style', 'base', 'builtin-style-krea-raw'),
        # variant mismatch
        ('zimage', 'style', 'turbo', 'builtin-style-zimage-base'),
    ]
    for idx, (family, kind, variant, preset_id) in enumerate(cases):
        ds_id = _create_ds(client, name=f'Scope {idx}', trigger=f'scope{idx}',
                           train_type=family, kind=kind)
        with app.app_context():
            from app.services import lora_training as lt
            lt.update_train_settings('local', ds_id, {'rank': 64})
        r = client.post(f'/api/dataset/{ds_id}/train/presets/apply', json={
            'preset_id': preset_id,
            'train_type': family,
            'variant': variant,
        })
        assert r.status_code == 409
        assert r.get_json()['error_code'] == 'PRESET_SCOPE'
        with app.app_context():
            from app.services import lora_training as lt
            assert lt.snapshot_train_settings('local', ds_id) == {'rank': 64}

    # With no requested variant, the persisted/default Turbo selection remains
    # authoritative; omitting the field cannot sneak the Base recipe through.
    ds_id = _create_ds(client, name='Scope absent', trigger='scope_absent',
                       train_type='zimage', kind='style')
    with app.app_context():
        from app.services import lora_training as lt
        lt.update_train_settings('local', ds_id, {'rank': 64})
    r = client.post(f'/api/dataset/{ds_id}/train/presets/apply', json={
        'preset_id': 'builtin-style-zimage-base',
        'train_type': 'zimage',
    })
    assert r.status_code == 409
    assert r.get_json()['error_code'] == 'PRESET_SCOPE'
    with app.app_context():
        from app.services import lora_training as lt
        assert lt.snapshot_train_settings('local', ds_id) == {'rank': 64}


def test_numeric_preset_family_mismatch_is_409_without_mutation(client, app):
    ds_id = _create_ds(client, train_type='zimage', kind='style')
    pid = client.post('/api/train/presets', json={
        'name': 'Krea only', 'train_type': 'krea', 'settings': {'rank': 32},
    }).get_json()['id']
    with app.app_context():
        from app.services import lora_training as lt
        lt.update_train_settings('local', ds_id, {'rank': 64})
    r = client.post(f'/api/dataset/{ds_id}/train/presets/apply', json={
        'preset_id': pid, 'train_type': 'zimage', 'variant': 'base',
    })
    assert r.status_code == 409
    assert r.get_json()['error_code'] == 'PRESET_SCOPE'
    with app.app_context():
        from app.services import lora_training as lt
        assert lt.snapshot_train_settings('local', ds_id) == {'rank': 64}


def test_new_numeric_preset_kind_and_variant_scope_is_enforced(client, app):
    pid = client.post('/api/train/presets', json={
        'name': 'Scoped Style Base',
        'train_type': 'zimage',
        'dataset_kind': 'style',
        'variants': ['base'],
        'settings': {'rank': 32},
    }).get_json()['id']
    ds_id = _create_ds(client, train_type='zimage', kind='style')
    with app.app_context():
        from app.services import lora_training as lt
        lt.update_train_settings('local', ds_id, {'rank': 64})
    url = f'/api/dataset/{ds_id}/train/presets/apply'
    r = client.post(url, json={
        'preset_id': pid, 'train_type': 'zimage', 'variant': 'turbo',
    })
    assert r.status_code == 409
    with app.app_context():
        from app.services import lora_training as lt
        assert lt.snapshot_train_settings('local', ds_id) == {'rank': 64}

    r = client.post(url, json={
        'preset_id': pid, 'train_type': 'zimage', 'variant': 'base',
    })
    assert r.status_code == 200
    with app.app_context():
        from app.services import lora_training as lt
        assert lt.snapshot_train_settings('local', ds_id) == {'rank': 32}


def test_training_preset_scope_columns_exist(app):
    with app.app_context():
        from sqlalchemy import text
        from app.extensions import db
        cols = {row[1] for row in db.session.execute(
            text('PRAGMA table_info(training_preset)'))}
    assert {'dataset_kind', 'variants'} <= cols


def test_save_and_import_reject_invalid_family_without_zimage_fallback(client):
    ds_id = _create_ds(client, train_type='krea')
    snapshot = client.post('/api/train/presets', json={
        'name': 'Bad snapshot family', 'dataset_id': ds_id,
        'train_type': 'not-a-family', 'variant': 'base',
    })
    assert snapshot.status_code == 400
    assert snapshot.get_json()['error'] == 'invalid train_type'

    imported = client.post('/api/train/presets', json={
        'name': 'Bad import family', 'train_type': 'not-a-family',
        'settings': {'rank': 32},
    })
    assert imported.status_code == 400
    assert imported.get_json()['error'] == 'invalid train_type'
    names = {p['name'] for p in client.get('/api/train/presets').get_json()['presets']}
    assert 'Bad snapshot family' not in names
    assert 'Bad import family' not in names


def test_save_and_import_reject_invalid_variant_without_empty_scope(client):
    ds_id = _create_ds(client, train_type='krea')
    snapshot = client.post('/api/train/presets', json={
        'name': 'Bad snapshot variant', 'dataset_id': ds_id,
        'train_type': 'krea', 'variant': '9b',
    })
    assert snapshot.status_code == 400

    imported = client.post('/api/train/presets', json={
        'name': 'Bad import variant', 'train_type': 'zimage',
        'variants': ['9b'], 'settings': {'rank': 32},
    })
    assert imported.status_code == 400
    names = {p['name'] for p in client.get('/api/train/presets').get_json()['presets']}
    assert 'Bad snapshot variant' not in names
    assert 'Bad import variant' not in names
