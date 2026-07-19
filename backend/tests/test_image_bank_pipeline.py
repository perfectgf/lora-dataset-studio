"""🗃️ Image bank — the "Launch all" chained triage pipeline.

The pipeline chains the EXISTING passes; here we exercise the ORCHESTRATION
(order, auto-reject params, per-step skip-with-reason, cancel, the persisted
report, and the "heavy passes only touch survivors" guarantee) with the heavy
ML passes mocked — no torch, no Ollama. Background jobs run inline under TESTING
(see bank_jobs.start), so a POST /pipeline runs the whole chain synchronously.
"""
import os

from PIL import Image


# --- factories (mirror test_image_bank_scoring) ------------------------------
def _save(path, im):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.lower().endswith(('.jpg', '.jpeg')):
        im.save(path, 'JPEG', quality=92)
    else:
        im.save(path)


def _flat(size=128, value=128):
    return Image.new('RGB', (size, size), (value, value, value))


def _photo(size=256):
    im = Image.new('L', (size, size))
    c, r2 = size / 2, (size / 3) ** 2
    im.putdata([min(255, int(150 * x / size + 50 * y / size)
                    + (80 if (x - c) ** 2 + (y - c) ** 2 < r2 else 0))
                for y in range(size) for x in range(size)])
    return im.convert('RGB')


def _mkbank(client, tmp_path, files, name='B'):
    src = tmp_path / 'src'
    for rel, im in files.items():
        _save(str(src / rel), im)
    r = client.post('/api/bank/create', json={'name': name, 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


def _fake_pass(log, name, *, mutate=None):
    """A stand-in for one of the heavy _X_job factories: records that it ran (and
    the SURVIVOR count it would see) instead of touching torch/Ollama."""
    def factory(*_a, **_k):
        def run(job):
            from app.models import BankImage
            survivors = (BankImage.query.filter(BankImage.status != 'reject').count())
            log.append((name, survivors))
            if mutate:
                mutate()
        return run
    return factory


def _report(client, bank_id):
    return client.get(f'/api/bank/{bank_id}').get_json().get('pipeline_report')


# --- order + report ----------------------------------------------------------
def test_pipeline_runs_every_step_in_canonical_order(client, tmp_path, monkeypatch):
    from app.services import image_bank_service as svc
    log = []
    # Heavy passes mocked; scan + auto_reject run for real (pure PIL / SQL).
    monkeypatch.setattr(svc, '_score_prereq', lambda: None)
    monkeypatch.setattr(svc, '_watermark_prereq', lambda: None)
    monkeypatch.setattr(svc, '_faces_prereq', lambda: None)
    monkeypatch.setattr(svc, '_caption_prereq', lambda: None)
    monkeypatch.setattr(svc, '_gpu_busy_reason', lambda: None)
    monkeypatch.setattr(svc, '_score_job', _fake_pass(log, 'score'))
    monkeypatch.setattr(svc, '_watermark_job', _fake_pass(log, 'watermark'))
    monkeypatch.setattr(svc, '_faces_job', _fake_pass(log, 'faces'))
    monkeypatch.setattr(svc, '_caption_job', _fake_pass(log, 'caption'))
    # semantic_dedup reads the ✨ Score embedding cache (mocked away here) — stand
    # it in so the step runs to 'done' like the other mocked passes.
    monkeypatch.setattr(svc, 'rebuild_semantic_dup_groups',
                        lambda *_a, **_k: (log.append(('semantic_dedup', 0)) or 0))

    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _photo(), 'b.jpg': _flat()})
    r = client.post(f'/api/bank/{bank_id}/pipeline', json={
        'steps': list(svc.PIPELINE_STEPS), 'reject_flags': [], 'resolve_dups': False})
    assert r.status_code == 202, r.get_json()

    assert [name for name, _ in log] == ['score', 'semantic_dedup', 'watermark',
                                         'faces', 'caption']
    report = _report(client, bank_id)
    assert report is not None
    assert [s['step'] for s in report['steps']] == list(svc.PIPELINE_STEPS)
    assert all(s['status'] == 'done' for s in report['steps']), report['steps']
    assert report['cancelled'] is False
    assert report['counts']['total'] == 2


# --- auto-reject params ------------------------------------------------------
def test_pipeline_auto_reject_honors_flags_and_dedup(client, tmp_path, monkeypatch):
    from app.services import image_bank_service as svc
    # Two identical flat images (a duplicate pair) + one photo. 'uniform' flags
    # the flats; dedup would also target the pair.
    bank_id, _src = _mkbank(client, tmp_path, {
        'flat1.jpg': _flat(value=128), 'flat2.jpg': _flat(value=128),
        'photo.jpg': _photo()})
    r = client.post(f'/api/bank/{bank_id}/pipeline', json={
        'steps': ['scan', 'auto_reject'],
        'reject_flags': ['uniform'], 'resolve_dups': True})
    assert r.status_code == 202, r.get_json()

    report = _report(client, bank_id)
    steps = {s['step']: s for s in report['steps']}
    assert steps['scan']['status'] == 'done'
    ar = steps['auto_reject']
    assert ar['status'] == 'done'
    # The flats are uniform → rejected; nothing is deleted from disk.
    assert ar['counts']['rejected'] >= 2
    assert (_src / 'flat1.jpg').is_file()
    with client.application.app_context():
        from app.models import BankImage
        rej = BankImage.query.filter_by(bank_id=bank_id, status='reject').count()
        assert rej >= 2


def test_pipeline_auto_reject_skips_flags_when_not_requested(client, tmp_path):
    bank_id, _src = _mkbank(client, tmp_path, {'flat.jpg': _flat()})
    client.post(f'/api/bank/{bank_id}/pipeline', json={
        'steps': ['scan', 'auto_reject'], 'reject_flags': [], 'resolve_dups': False})
    report = _report(client, bank_id)
    ar = next(s for s in report['steps'] if s['step'] == 'auto_reject')
    assert ar['counts']['rejected'] == 0


# --- graceful skip -----------------------------------------------------------
def test_pipeline_skips_step_with_reason_when_prereq_absent(client, tmp_path, monkeypatch):
    from app.services import image_bank_service as svc
    log = []
    monkeypatch.setattr(svc, '_score_prereq', lambda: 'bank scoring extra not installed')
    monkeypatch.setattr(svc, '_caption_prereq', lambda: None)
    monkeypatch.setattr(svc, '_gpu_busy_reason', lambda: None)
    monkeypatch.setattr(svc, '_score_job', _fake_pass(log, 'score'))
    monkeypatch.setattr(svc, '_caption_job', _fake_pass(log, 'caption'))

    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    r = client.post(f'/api/bank/{bank_id}/pipeline', json={
        'steps': ['scan', 'score', 'caption'], 'reject_flags': [], 'resolve_dups': False})
    assert r.status_code == 202

    report = _report(client, bank_id)
    steps = {s['step']: s for s in report['steps']}
    assert steps['score']['status'] == 'skipped'
    assert 'bank scoring' in steps['score']['reason']
    # The pipeline CONTINUES past a skipped step.
    assert steps['scan']['status'] == 'done'
    assert steps['caption']['status'] == 'done'
    assert 'score' not in [n for n, _ in log]        # never ran
    assert 'caption' in [n for n, _ in log]


def test_pipeline_skips_gpu_step_when_gpu_busy(client, tmp_path, monkeypatch):
    from app.services import image_bank_service as svc
    log = []
    monkeypatch.setattr(svc, '_score_prereq', lambda: None)
    monkeypatch.setattr(svc, '_gpu_busy_reason', lambda: 'training is running on the GPU')
    monkeypatch.setattr(svc, '_score_job', _fake_pass(log, 'score'))

    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    client.post(f'/api/bank/{bank_id}/pipeline', json={
        'steps': ['scan', 'score'], 'reject_flags': [], 'resolve_dups': False})
    report = _report(client, bank_id)
    score = next(s for s in report['steps'] if s['step'] == 'score')
    assert score['status'] == 'skipped'
    assert 'GPU' in score['reason'] or 'training' in score['reason']
    assert log == []


# --- survivors only ----------------------------------------------------------
def test_pipeline_heavy_passes_only_see_survivors(client, tmp_path, monkeypatch):
    """auto-reject runs BEFORE the heavy passes, so by the time score/faces run
    the rejected images are already out — the costly work never pays for them."""
    from app.services import image_bank_service as svc
    log = []
    monkeypatch.setattr(svc, '_score_prereq', lambda: None)
    monkeypatch.setattr(svc, '_faces_prereq', lambda: None)
    monkeypatch.setattr(svc, '_gpu_busy_reason', lambda: None)
    monkeypatch.setattr(svc, '_score_job', _fake_pass(log, 'score'))
    monkeypatch.setattr(svc, '_faces_job', _fake_pass(log, 'faces'))

    # 3 flats (uniform → rejected) + 2 photos (survive).
    files = {f'flat{i}.jpg': _flat(value=100 + i) for i in range(3)}
    files.update({'p1.jpg': _photo(), 'p2.jpg': _photo(300)})
    bank_id, _src = _mkbank(client, tmp_path, files)
    client.post(f'/api/bank/{bank_id}/pipeline', json={
        'steps': ['scan', 'auto_reject', 'score', 'faces'],
        'reject_flags': ['uniform'], 'resolve_dups': False})

    # Every heavy pass saw only the survivors (< the 5 total).
    assert log, 'heavy passes ran'
    for name, survivors in log:
        assert survivors <= 2, f'{name} saw {survivors} rows — should skip rejects'


# --- cancel ------------------------------------------------------------------
def test_pipeline_cancel_midway_records_remaining_as_cancelled(client, tmp_path, monkeypatch):
    from app.services import image_bank_service as svc
    from app.services import bank_jobs
    log = []

    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _photo()})

    # Cancel this bank's live job from inside the score pass.
    def score_factory(*_a, **_k):
        def run(job):
            log.append('score')
            bank_jobs.cancel(bank_id)
        return run

    monkeypatch.setattr(svc, '_score_prereq', lambda: None)
    monkeypatch.setattr(svc, '_watermark_prereq', lambda: None)
    monkeypatch.setattr(svc, '_gpu_busy_reason', lambda: None)
    monkeypatch.setattr(svc, '_score_job', score_factory)
    monkeypatch.setattr(svc, '_watermark_job', _fake_pass(log, 'watermark'))

    client.post(f'/api/bank/{bank_id}/pipeline', json={
        'steps': ['scan', 'score', 'watermark'], 'reject_flags': [], 'resolve_dups': False})

    assert 'watermark' not in log                    # never reached after cancel
    report = _report(client, bank_id)
    assert report['cancelled'] is True
    steps = {s['step']: s for s in report['steps']}
    assert steps['scan']['status'] == 'done'
    assert steps['score']['status'] == 'done'        # it executed, then cancelled
    assert steps['watermark']['status'] == 'cancelled'


# --- validation --------------------------------------------------------------
def test_pipeline_empty_steps_is_400(client, tmp_path):
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    r = client.post(f'/api/bank/{bank_id}/pipeline', json={'steps': []})
    assert r.status_code == 400


def test_pipeline_unknown_bank_is_400(client):
    r = client.post('/api/bank/999999/pipeline', json={'steps': ['scan']})
    assert r.status_code == 400
