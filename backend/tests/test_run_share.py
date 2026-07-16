"""'Share configuration' per-run download (🏋️ Runs hub): a paste-safe .txt of
everything a launch sent to ai-toolkit, for both local and cloud runs, with
graceful degradation on pre-snapshot rows and strict redaction (no home paths,
no secrets)."""
import json


def _mkds(client, name='Lola', trigger='lola', kind=None):
    payload = {'name': name, 'trigger_word': trigger}
    if kind is not None:
        payload['kind'] = kind
    return client.post('/api/dataset/create',
                       json=payload).get_json()['id']


def _add_local_rec(app, ds, **over):
    from app.extensions import db
    from app.models import TrainingRunRecord
    with app.app_context():
        settings = over.pop('settings', {
            'trigger': 'lola', 'rank': 32, 'alpha': 32, 'resolution': [768, 1024],
            'save_every': 250, 'max_step_saves': 4, 'optimizer': 'adamw8bit',
            'lr': 0.0001, 'timestep_type': 'linear'})
        rec = TrainingRunRecord(
            dataset_id=ds, family=over.pop('family', 'krea'),
            source=over.pop('source', 'local'), fingerprint='fp', version=over.pop('version', 3),
            steps=over.pop('steps', 2000), masked=over.pop('masked', True),
            settings=json.dumps(settings) if settings is not None else None,
            manifest=json.dumps([[1, 'a', 'b'], [2, 'c', 'd']]), **over)
        db.session.add(rec)
        db.session.commit()
        return rec.id


def test_share_local_run_lists_every_parameter(client, app):
    ds = _mkds(client)
    rid = _add_local_rec(app, ds)
    r = client.get(f'/api/dataset/train/runs/rec-{rid}/share')
    assert r.status_code == 200
    cd = r.headers['Content-Disposition']
    assert 'attachment' in cd and cd.endswith('.txt"')
    assert 'lds-config-lola-krea-v3-' in cd
    body = r.get_data(as_text=True)
    # header
    assert 'App version:' in body
    assert 'Source:' in body and 'local' in body
    assert 'Krea 2' in body
    assert 'Lola' in body and 'version v3' in body and '2 image(s)' in body
    # every training parameter, labeled with units
    assert 'Trigger word' in body and 'lola' in body
    assert 'LoRA rank' in body and '32' in body
    assert 'LoRA alpha' in body
    assert 'Resolution' in body and '768 + 1024 px' in body
    assert 'Save every' in body and '250 steps' in body
    assert 'Max saved checkpoints' in body
    assert 'Optimizer' in body and 'adamw8bit' in body
    assert 'Learning rate' in body and '0.0001' in body
    assert 'Timestep type' in body and 'linear' in body
    assert 'Masked training:' in body and 'yes' in body
    # outcome + footer
    assert 'Target steps:' in body and '2000' in body
    assert 'paste-safe, no local paths or keys' in body


def test_share_style_hides_internal_id_but_character_keeps_trigger(client, app):
    """Style is always-on: its generated identifier must never be advertised as
    an activation word, including for an older snapshot that still stored it."""
    style_id = _mkds(client, name='Ink style', trigger='', kind='style')
    with app.app_context():
        from app.extensions import db
        from app.models import FaceDataset
        from app.services import lora_training as lt

        style = db.session.get(FaceDataset, style_id)
        internal_id = style.trigger_word
        snapshot = lt.launch_settings_snapshot(style, 'krea')
        assert 'trigger' not in snapshot
        assert snapshot['style_mode'] == 'always_on'

    # Simulate a pre-fix record: run_share must also suppress its stale trigger.
    stale = dict(snapshot, trigger=internal_id)
    style_run = _add_local_rec(app, style_id, settings=stale)
    style_body = client.get(
        f'/api/dataset/train/runs/rec-{style_run}/share').get_data(as_text=True)
    assert 'Trigger word' not in style_body
    assert internal_id not in style_body
    assert 'Style mode:' in style_body
    assert 'always-on (no activation trigger)' in style_body

    character_id = _mkds(client, name='Lola character', trigger='lola_char')
    character_run = _add_local_rec(
        app, character_id, settings={'trigger': 'lola_char', 'rank': 32})
    character_body = client.get(
        f'/api/dataset/train/runs/rec-{character_run}/share').get_data(as_text=True)
    assert 'Trigger word' in character_body
    assert 'lola_char' in character_body
    assert 'Style mode:' not in character_body


def test_share_cloud_run_outcome_redaction_and_no_secret(client, app, monkeypatch):
    """Cloud run: settings come from the linked registry row, the outcome from
    the pod row; a home path in the error is redacted and NO secret ever leaks
    even when the app is fully configured."""
    monkeypatch.setenv('HF_TOKEN', 'hf_supersecret_value')
    monkeypatch.setenv('VAST_API_KEY', 'vast_supersecret_value')
    ds = _mkds(client)
    from app.extensions import db
    from app.models import CloudTrainingRun, TrainingRunRecord
    with app.app_context():
        crun = CloudTrainingRun(
            dataset_id=ds, status='error', run_name='r', gpu_name='RTX 4090',
            price_per_hour=0.30,
            error='crash writing C:\\Users\\secretuser\\staging\\x.safetensors',
            train_params=json.dumps({'train_type': 'krea', 'steps': 2000,
                                     'masked': True, 'version': 2}))
        db.session.add(crun)
        db.session.commit()
        db.session.add(TrainingRunRecord(
            dataset_id=ds, family='krea', source='cloud', fingerprint='fp',
            version=2, steps=2000, masked=True, variant='base', cloud_run_id=crun.id,
            settings=json.dumps({'rank': 48, 'alpha': 48})))
        db.session.commit()
        cid = crun.id
    r = client.get(f'/api/dataset/train/runs/cloud-{cid}/share')
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    # outcome from the pod row
    assert 'Status:' in body and 'error' in body
    assert 'GPU:' in body and 'RTX 4090' in body
    assert 'Error:' in body
    assert 'Cost:' in body
    # settings from the linked registry record
    assert 'LoRA rank' in body and '48' in body
    # variant 'base' renders as 'Raw'
    assert 'Raw' in body
    # redaction: the OS account name is stripped, replaced by ~
    assert 'secretuser' not in body
    assert '~' in body
    # secrets never appear, even though both are configured
    assert 'hf_supersecret_value' not in body
    assert 'vast_supersecret_value' not in body


def test_share_legacy_run_degrades_gracefully(client, app):
    """A cloud run that predates the settings snapshot has no registry row —
    the parameters section says so instead of failing (retrofit rule)."""
    ds = _mkds(client)
    from app.extensions import db
    from app.models import CloudTrainingRun
    with app.app_context():
        crun = CloudTrainingRun(dataset_id=ds, status='done', run_name='old')
        db.session.add(crun)
        db.session.commit()
        cid = crun.id
    r = client.get(f'/api/dataset/train/runs/cloud-{cid}/share')
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert 'not recorded on this run' in body
    assert 'Status:' in body and 'done' in body


def test_share_unknown_run_404(client, app):
    _mkds(client)
    for key in ('cloud-9999', 'rec-9999', 'bogus', 'cloud-abc', 'rec-'):
        assert client.get(f'/api/dataset/train/runs/{key}/share').status_code == 404


def test_all_runs_exposes_share_keys(client, app):
    """The hub payload carries a share_key on every row (local rec-<id>, cloud
    cloud-<id>) and on active runs — the frontend button keys on it."""
    ds = _mkds(client)
    from app.extensions import db
    from app.models import CloudTrainingRun, TrainingRunRecord
    from app.services import cloud_training as ct
    with app.app_context():
        db.session.add(TrainingRunRecord(
            dataset_id=ds, family='krea', source='local', fingerprint='fp1',
            version=1, steps=2000, masked=True, settings=json.dumps({'rank': 32})))
        active = CloudTrainingRun(dataset_id=ds, status='training', run_name='live',
                                  vast_label='lds-1')
        done = CloudTrainingRun(dataset_id=ds, status='done', run_name='c')
        db.session.add_all([active, done])
        db.session.commit()
        db.session.add(TrainingRunRecord(
            dataset_id=ds, family='zimage', source='cloud', fingerprint='fp2',
            version=1, steps=2000, masked=True, cloud_run_id=done.id,
            settings=json.dumps({'rank': 48})))
        db.session.commit()
        out = ct.all_runs(limit=10)
    assert all(r.get('share_key') for r in out['recent'])
    keys = [r['share_key'] for r in out['recent']]
    assert any(k.startswith('rec-') for k in keys)
    assert any(k.startswith('cloud-') for k in keys)
    # the active run is shareable via its pod-row key
    assert out['actives'] and all(a['share_key'].startswith('cloud-') for a in out['actives'])
