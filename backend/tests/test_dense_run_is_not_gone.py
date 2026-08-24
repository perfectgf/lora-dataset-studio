"""A full model that lives on Hugging Face is not a gone run.

THE INCIDENT THIS PINS
----------------------
`checkpoint_ready` was read off `checkpoint_local_path` — a LoRA-lane column the
dense lane only ever fills when the LOCAL delivery lands, a feature younger than
the dense lane itself. So every full model delivered to Hugging Face only, which
is every dense run trained before that feature, came back as `False`: the canvas
dimmed the card, badged it `gone`, and offered "Remove this run" under the words
"No checkpoints left on disk". Accepting that offer discards the lineage, the
notes, the version and the only recorded pointer to the private repository
holding a model that cost eight hours of GPU.

Three states, three answers, and the LoRA lane must not move a millimetre.
"""
import json
from app.extensions import db



def _dense_run(dataset_id, tmp_path, *, status='done', hf_repo='acme/dense-1',
               artifact_status='available', files=(), delivery='hub'):
    """A finished dense run. `files` land in its staging dir — the legacy read
    of the checkpoint store, which is what a locally delivered run looks like."""
    from app.extensions import db
    from app.models import CloudTrainingRun
    staging = tmp_path / f'stg{dataset_id}'
    staging.mkdir(parents=True, exist_ok=True)
    for name in files:
        (staging / name).write_bytes(b'W' * 16)
    params = {'training_mode': 'full_transformer', 'train_type': 'krea',
              'variant': 'base', 'steps': 3000, 'dense_delivery': delivery}
    if hf_repo:
        params['hf_repo_id'] = hf_repo
        params['hf_url'] = f'https://huggingface.co/{hf_repo}'
    if artifact_status:
        params['artifact_status'] = artifact_status
    run = CloudTrainingRun(dataset_id=dataset_id, status=status, job_name='j',
                           staging_dir=str(staging),
                           train_params=json.dumps(params))
    db.session.add(run)
    db.session.commit()
    return run


def _rec(dataset_id, crun):
    from app.extensions import db
    from app.models import TrainingRunRecord
    r = TrainingRunRecord(dataset_id=dataset_id, family='krea', source='cloud',
                          base_model='', variant='base', steps=3000, version=1,
                          fingerprint='fp', manifest='[]', cloud_run_id=crun.id)
    db.session.add(r)
    db.session.commit()
    return r


# --- the three states -------------------------------------------------------------

def test_a_hub_only_full_model_is_not_gone(app, tmp_path):
    """The exact shape of every dense run trained before the local delivery."""
    from app.services import cloud_training as ct
    with app.app_context():
        crun = _dense_run(1, tmp_path)
        assert ct.dense_artifact_state(crun) == ct.DENSE_ON_HUB
        node = ct.run_lineage(_rec(1, crun).id)['nodes'][0]
        assert node['dense_artifact'] == 'hub'
        assert node['training_mode'] == 'full_transformer'
        # None, not False: the tri-state every existing reader already treats as
        # "cannot say", so an older frontend refuses the deletion too.
        assert node['checkpoint_ready'] is None


def test_a_full_model_on_this_disk_reads_as_present(app, tmp_path):
    from app.services import cloud_training as ct
    with app.app_context():
        crun = _dense_run(2, tmp_path, delivery='both',
                          files=('Krea_x.safetensors', 'Krea_x_fp8.safetensors'))
        assert ct.dense_artifact_state(crun) == ct.DENSE_ON_DISK
        node = ct.run_lineage(_rec(2, crun).id)['nodes'][0]
        assert node['dense_artifact'] == 'local'
        assert node['checkpoint_ready'] is True
        # checkpoint_local_path is still empty — presence came from the store.
        assert crun.checkpoint_local_path is None


def test_a_full_model_verified_missing_really_is_gone(app, tmp_path):
    """The generosity has a floor: an explicitly verified `missing` is absence."""
    from app.services import cloud_training as ct
    with app.app_context():
        crun = _dense_run(3, tmp_path, artifact_status='missing')
        assert ct.dense_artifact_state(crun) == ct.DENSE_GONE
        node = ct.run_lineage(_rec(3, crun).id)['nodes'][0]
        assert node['dense_artifact'] == 'none'
        assert node['checkpoint_ready'] is False


def test_an_unverified_upload_is_never_called_gone(app, tmp_path):
    """Pending / unverifiable is not proof of a lost model. The two mistakes do
    not cost the same, so this one is deliberately generous."""
    from app.services import cloud_training as ct
    with app.app_context():
        for i, status in enumerate(('pending', 'uploading', 'verification_pending')):
            crun = _dense_run(10 + i, tmp_path, artifact_status=status)
            assert ct.dense_artifact_state(crun) == ct.DENSE_ON_HUB, status


def test_a_dense_run_with_no_repository_and_no_file_is_gone(app, tmp_path):
    from app.services import cloud_training as ct
    with app.app_context():
        crun = _dense_run(4, tmp_path, hf_repo=None, artifact_status=None)
        assert ct.dense_artifact_state(crun) == ct.DENSE_GONE


# --- the deletion guard -------------------------------------------------------------

def test_removing_a_hub_only_full_model_run_is_refused(app, tmp_path):
    from app.services import cloud_training as ct
    with app.app_context():
        rec = _rec(5, _dense_run(5, tmp_path))
        assert ct.delete_run_record(rec.id) == 'has_model'
        # and the record is still there
        from app.models import TrainingRunRecord
        assert db.session.get(TrainingRunRecord, rec.id) is not None


def test_the_route_answers_409_with_a_sentence_naming_the_repository(app, client, tmp_path):
    with app.app_context():
        rec_id = _rec(6, _dense_run(6, tmp_path)).id
    res = client.delete(f'/api/dataset/train/runs/{rec_id}')
    assert res.status_code == 409
    assert 'Hugging Face' in res.get_json()['error']


def test_the_impact_preview_names_the_blocker(app, client, tmp_path):
    with app.app_context():
        rec_id = _rec(7, _dense_run(7, tmp_path)).id
    body = client.get(f'/api/dataset/train/runs/{rec_id}/deletion-impact').get_json()
    assert body['removal_blocker'] == 'has_model'


def test_a_genuinely_empty_dense_run_stays_removable(app, tmp_path):
    """The fix must not turn every dense run into an undeletable ghost."""
    from app.services import cloud_training as ct
    with app.app_context():
        rec = _rec(8, _dense_run(8, tmp_path, hf_repo=None, artifact_status=None))
        assert ct.delete_run_record(rec.id) == 'deleted'


# --- the LoRA lane does not move ------------------------------------------------------

def test_a_lora_run_answers_exactly_as_before(app, tmp_path):
    from app.extensions import db
    from app.models import CloudTrainingRun
    from app.services import cloud_training as ct
    with app.app_context():
        staging = tmp_path / 'lora'
        staging.mkdir(parents=True, exist_ok=True)
        final = staging / 'lora_x.safetensors'
        final.write_bytes(b'F')
        run = CloudTrainingRun(dataset_id=9, status='done', job_name='j',
                               staging_dir=str(staging),
                               checkpoint_local_path=str(final),
                               train_params=json.dumps({'steps': 1000,
                                                        'train_type': 'krea'}))
        db.session.add(run)
        db.session.commit()
        node = ct.run_lineage(_rec(9, run).id)['nodes'][0]
        assert node['checkpoint_ready'] is True          # the old boolean, unchanged
        assert node['training_mode'] == 'lora'
        assert 'dense_artifact' not in node              # additive, dense-only
        # deleting is still refused for the old reason, with the old word
        assert ct.delete_run_record(node['record_id']) == 'has_saves'


def test_a_local_record_is_labelled_lora(app, tmp_path):
    from app.extensions import db
    from app.models import TrainingRunRecord
    from app.services import cloud_training as ct
    with app.app_context():
        r = TrainingRunRecord(dataset_id=11, family='krea', source='local',
                              base_model='', variant='base', steps=1000, version=1,
                              fingerprint='fp', manifest='[]')
        db.session.add(r)
        db.session.commit()
        node = ct.run_lineage(r.id)['nodes'][0]
        assert node['training_mode'] == 'lora'
        assert 'dense_artifact' not in node
