"""Checkpoint-by-checkpoint render timelines and their safe GIF export."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image


def test_wsgi_file_close_releases_a_response_slot_exactly_once():
    from app.routes.training import _CloseCallbackFile

    wrapped = BytesIO(b'timeline')
    releases = []
    leased = _CloseCallbackFile(wrapped, lambda: releases.append('released'))

    assert leased.read() == b'timeline'
    leased.close()
    leased.close()  # send_file + call_on_close may both close the same object

    assert wrapped.closed is True
    assert releases == ['released']


def test_three_wsgi_gif_downloads_release_the_real_route_lease(client, app,
                                                                monkeypatch):
    from werkzeug.test import EnvironBuilder

    from app.extensions import db
    from app.services import checkpoint_timeline as timeline

    with app.app_context():
        dataset_id = _create(client)
        record = _record(db, dataset_id)
        _image(db, dataset_id, record.id, 100, filename='wsgi-a.png')
        _image(db, dataset_id, record.id, 200, filename='wsgi-b.png')
        record_id = record.id
    series_id = client.get(
        f'/api/train/run/{record_id}/timeline').get_json()['series'][0]['id']

    class TwoSlotGate:
        def __init__(self):
            self.active = 0
            self.releases = 0

        def acquire(self):
            if self.active >= 2:
                return False
            self.active += 1
            return True

        def release(self):
            self.active -= 1
            self.releases += 1

    gate = TwoSlotGate()
    monkeypatch.setattr(timeline, 'acquire_gif_response_slot', gate.acquire)
    monkeypatch.setattr(timeline, 'release_gif_response_slot', gate.release)

    path = f'/api/train/run/{record_id}/timeline/{series_id}/gif'
    for _index in range(3):
        builder = EnvironBuilder(path=path, method='GET')
        environ = builder.get_environ()
        result = {}

        def start_response(status, headers, exc_info=None):
            result['status'] = status
            result['headers'] = headers
            return lambda _data: None

        app_iter = app.wsgi_app(environ, start_response)
        try:
            body = b''.join(app_iter)
        finally:
            app_iter.close()
            builder.close()

        assert result['status'].startswith('200 ')
        assert body.startswith(b'GIF8')
        assert gate.active == 0

    assert gate.releases == 3


def _create(client, name='Timeline', trigger='timeline'):
    return client.post('/api/dataset/create',
                       json={'name': name, 'trigger_word': trigger}).get_json()['id']


def _record(db, dataset_id):
    from app.models import TrainingRunRecord
    row = TrainingRunRecord(dataset_id=dataset_id, family='krea', source='local',
                            fingerprint='timeline', version=1)
    db.session.add(row)
    db.session.commit()
    return row


SETTINGS = {
    'prompt': 'portrait of timeline',
    'seed': 123456,
    'strength': 0.85,
    'z_model': 'krea-base.safetensors',
    'aspect': '4:3',
    'cfg': 3.5,
    'steps': 28,
    'steps2': 12,
    'negative': 'blur',
    'sampler': 'euler',
    'scheduler': 'simple',
    'extra_loras': '[{"filename":"detail.safetensors","strength":0.2}]',
    'krea_rebalance': 1.15,
    'weight_dtype': 'fp8_e4m3fn',
    'enhancer_strength': 0.35,
    'detail_amount': 0.2,
    'resolution_tier': 'hq',
    'resolution_multiplier': 1.2,
    'init_image': 'init.png',
    'denoise': 0.72,
}


def _image(db, dataset_id, record_id, step, *, run_id='launch-a',
           filename=None, on_disk=True, color=(120, 80, 40), size=(48, 32),
           status='done', checkpoint=None, **changes):
    from app.models import LoraTestImage
    from app.services.dataset_storage import ensure_dataset_dir
    values = dict(SETTINGS)
    values.update(changes)
    filename = filename if filename is not None else f'{run_id}-{step}-{len(checkpoint or "")}.png'
    row = LoraTestImage(
        dataset_id=dataset_id,
        record_id=record_id,
        step=step,
        run_id=run_id,
        checkpoint=checkpoint or f'lora-{step}.safetensors',
        filename=filename,
        status=status,
        **values,
    )
    db.session.add(row)
    db.session.commit()
    if on_disk and filename:
        path = Path(ensure_dataset_dir(dataset_id)) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', size, color).save(path)
    return row


def test_series_never_mix_records_launches_or_render_conditions(client, app):
    from app.extensions import db
    from app.services import checkpoint_timeline as timeline
    with app.app_context():
        dataset_id = _create(client)
        record = _record(db, dataset_id)
        foreign_record = _record(db, dataset_id)

        for step in (100, 200):
            _image(db, dataset_id, record.id, step, run_id='launch-a',
                   filename=f'a-{step}.png')
            _image(db, dataset_id, record.id, step, run_id='launch-b',
                   filename=f'b-{step}.png')
            _image(db, dataset_id, record.id, step, run_id='launch-a',
                   prompt='a different prompt', filename=f'config-{step}.png')
            _image(db, dataset_id, foreign_record.id, step, run_id='launch-a',
                   filename=f'foreign-{step}.png')

        result = timeline.checkpoint_timeline(record.id)
        again = timeline.checkpoint_timeline(record.id)

    assert result['count'] == result['shown'] == 3
    assert result['candidate_count'] == result['frames_shown'] == 6
    assert result['excluded'] == 0 and result['truncated'] is False
    assert [s['id'] for s in result['series']] == [s['id'] for s in again['series']]
    assert all(len(s['id']) == 64 for s in result['series'])
    assert {(s['run_id'], s['conditions']['prompt']) for s in result['series']} == {
        ('launch-a', SETTINGS['prompt']),
        ('launch-b', SETTINGS['prompt']),
        ('launch-a', 'a different prompt'),
    }
    for series in result['series']:
        assert set(series['conditions']) == set(timeline.CONDITION_FIELDS)
        assert series['conditions']['dataset_id'] == dataset_id
        # Inference `steps` is a condition; checkpoint `step` is the axis.
        assert series['conditions']['steps'] == 28
        assert series['steps'] == [100, 200]
        assert all(frame['record_id'] == record.id for frame in series['frames'])
        assert all(frame['dataset_id'] == dataset_id for frame in series['frames'])
        assert series['created_at']


def test_real_create_run_launch_id_feeds_a_checkpoint_timeline(client, app,
                                                                monkeypatch):
    from app.config import LOCAL_USER
    from app.extensions import db
    from app.models import LoraTestImage
    from app.services import lora_test_studio as studio
    from app.services import checkpoint_timeline as timeline
    from app.services.dataset_storage import ensure_dataset_dir

    with app.app_context():
        dataset_id = _create(client, name='Producer timeline', trigger='producer')
        record = _record(db, dataset_id)
        checkpoints = [
            'z image\\producer_000000100.safetensors',
            'z image\\producer_000000200.safetensors',
        ]
        monkeypatch.setattr(studio, 'gpu_busy_reason', lambda: None)
        monkeypatch.setattr(studio, '_active_run_count', lambda *_args: 0)
        monkeypatch.setattr(studio, 'list_test_checkpoints',
                            lambda *_args: [{'filename': cp} for cp in checkpoints])
        monkeypatch.setattr(studio, 'permanent_lora_candidates', lambda *_args: [])
        monkeypatch.setattr(studio, 'get_zimage_models',
                            lambda: ['zmodel.safetensors'])
        monkeypatch.setattr(studio, '_preflight_checkpoint_arch',
                            lambda *_args, **_kwargs: None)
        monkeypatch.setattr(studio, '_preflight_run',
                            lambda *_args, **_kwargs: None)
        monkeypatch.setattr(studio, '_target_node_classes', lambda: None)
        monkeypatch.setattr(studio, '_build_cell_workflow',
                            lambda *_args, **_kwargs: {'1': {}})
        monkeypatch.setattr(studio, '_enqueue_cell',
                            lambda *_args, job_id=None, **_kwargs: job_id)

        origins = {
            checkpoints[0]: {'record_id': record.id, 'step': 100},
            checkpoints[1]: {'record_id': record.id, 'step': 200},
        }
        launched = studio.create_run(
            LOCAL_USER, dataset_id, checkpoints, [0.8],
            studio.StudioGenSettings(seed=77, z_model='zmodel.safetensors',
                                     prompt='same render across checkpoints'),
            origins=origins)
        rows = (LoraTestImage.query.filter(
            LoraTestImage.id.in_(launched['ids'])).order_by(LoraTestImage.step).all())
        assert len(rows) == 2
        assert {row.run_id for row in rows} == {launched['run_id']}
        folder = Path(ensure_dataset_dir(dataset_id))
        for row in rows:
            row.status = 'done'
            row.filename = f'producer-{row.step}.png'
            Image.new('RGB', (32, 32), (row.step, 20, 30)).save(folder / row.filename)
        db.session.commit()

        result = timeline.checkpoint_timeline(record.id)

    assert result['count'] == 1
    assert result['series'][0]['run_id'] == launched['run_id']
    assert result['series'][0]['steps'] == [100, 200]


def test_legacy_nulls_missing_files_and_incomplete_series_are_honest(client, app):
    from app.extensions import db
    from app.models import LoraTestImage
    from app.services import checkpoint_timeline as timeline
    with app.app_context():
        dataset_id = _create(client)
        record = _record(db, dataset_id)
        base = dict(dataset_id=dataset_id, record_id=record.id,
                    checkpoint='legacy.safetensors', strength=1.0)
        db.session.add_all([
            LoraTestImage(**base, filename=None, step=100, run_id='legacy',
                          status='done'),
            LoraTestImage(**base, filename='no-step.png', step=None,
                          run_id='legacy', status='done'),
            LoraTestImage(**base, filename='no-run.png', step=100, run_id=None,
                          status='done'),
            LoraTestImage(**base, filename='failed.png', step=100,
                          run_id='legacy', status='failed'),
        ])
        db.session.commit()
        _image(db, dataset_id, record.id, 100, filename='gone.png',
               on_disk=False, run_id='usable')
        _image(db, dataset_id, record.id, 200, filename='lonely.png',
               run_id='other')
        result = timeline.checkpoint_timeline(record.id)

    # The four legacy/non-done rows are outside the declared source filter.
    assert result['candidate_count'] == 2
    assert result['count'] == 0 and result['series'] == []
    assert result['excluded'] == 2
    assert result['excluded_counts']['missing_or_unsafe_file'] == 1
    assert result['excluded_counts']['insufficient_series'] == 1
    assert result['excluded_counts']['insufficient_series_frames'] == 1


def test_newest_duplicate_wins_and_frames_are_in_ascending_step_order(client, app):
    from app.extensions import db
    from app.services import checkpoint_timeline as timeline
    with app.app_context():
        dataset_id = _create(client)
        record = _record(db, dataset_id)
        _image(db, dataset_id, record.id, 300, filename='300.png')
        old = _image(db, dataset_id, record.id, 100, filename='100-old.png')
        _image(db, dataset_id, record.id, 200, filename='200.png')
        newest = _image(db, dataset_id, record.id, 100,
                        filename='100-new.png', checkpoint='renamed-checkpoint.safetensors')
        result = timeline.checkpoint_timeline(record.id)

    series = result['series'][0]
    assert series['steps'] == [100, 200, 300]
    assert [frame['step'] for frame in series['frames']] == [100, 200, 300]
    assert series['frames'][0]['id'] == newest.id
    assert series['frames'][0]['id'] != old.id
    assert series['frame_count'] == series['shown'] == 3
    assert result['excluded_counts']['duplicate_steps'] == 1
    assert result['excluded'] == 1


def test_all_caps_are_applied_and_reported_without_hiding_counts(client, app,
                                                                 monkeypatch):
    from app.extensions import db
    from app.services import checkpoint_timeline as timeline
    monkeypatch.setattr(timeline, 'CANDIDATE_CAP', 7)
    monkeypatch.setattr(timeline, 'SERIES_CAP', 1)
    monkeypatch.setattr(timeline, 'FRAME_CAP', 3)
    with app.app_context():
        dataset_id = _create(client)
        record = _record(db, dataset_id)
        for run_id in ('older', 'newer'):
            for step in (10, 20, 30, 40):
                _image(db, dataset_id, record.id, step, run_id=run_id,
                       filename=f'{run_id}-{step}.png')
        result = timeline.checkpoint_timeline(record.id)

    assert result['candidate_count'] == 8 and result['candidates_scanned'] == 7
    assert result['count'] == 2 and result['shown'] == 1
    assert result['frame_count'] == 7 and result['frames_shown'] == 3
    assert result['series'][0]['run_id'] == 'newer'
    assert result['series'][0]['steps'] == [10, 20, 40]
    assert result['series'][0]['frame_count'] == 4
    assert result['series'][0]['truncated'] is True
    assert result['excluded'] == 5
    assert result['excluded_counts']['beyond_candidate_cap'] == 1
    assert result['excluded_counts']['beyond_series_cap'] == 1
    assert result['excluded_counts']['beyond_series_cap_frames'] == 3
    assert result['excluded_counts']['beyond_frame_cap'] == 1
    assert result['truncated'] is True


def test_timeline_route_returns_the_series_contract(client, app, monkeypatch):
    from app.extensions import db
    from app.services import checkpoint_timeline as timeline_service
    with app.app_context():
        dataset_id = _create(client)
        record = _record(db, dataset_id)
        _image(db, dataset_id, record.id, 500, filename='500.png')
        _image(db, dataset_id, record.id, 1000, filename='1000.png')
        record_id = record.id

    response = client.get(f'/api/train/run/{record_id}/timeline')
    assert response.status_code == 200
    body = response.get_json()
    assert body['record_id'] == record_id
    assert body['series'][0]['steps'] == [500, 1000]
    frame = body['series'][0]['frames'][0]
    assert frame['url'].endswith(f'/frame/{frame["id"]}')

    def forbidden_scan(_record_id):
        raise AssertionError('one preview must not rebuild the whole timeline')

    monkeypatch.setattr(timeline_service, '_build_timeline', forbidden_scan)
    preview = client.get(frame['url'])
    assert preview.status_code == 200
    assert preview.mimetype == 'image/webp'
    with Image.open(BytesIO(preview.data)) as image:
        assert image.format == 'WEBP'
        assert max(image.size) <= 1280
    preview.close()
    series_id = body['series'][0]['id']
    assert client.get(
        f'/api/train/run/{record_id + 1}/timeline/{series_id}/frame/{frame["id"]}'
    ).status_code == 404
    assert client.get(
        f'/api/train/run/{record_id}/timeline/{"0" * 64}/frame/{frame["id"]}'
    ).status_code == 404
    assert client.get(
        f'/api/train/run/{record_id}/timeline/{series_id}/frame/{frame["id"] + 999}'
    ).status_code == 404


def test_gif_route_is_animated_bounded_letterboxed_and_really_blended(client, app):
    from app.extensions import db
    with app.app_context():
        dataset_id = _create(client)
        record = _record(db, dataset_id)
        _image(db, dataset_id, record.id, 100, filename='red.png',
               color=(255, 0, 0), size=(1200, 600))
        _image(db, dataset_id, record.id, 200, filename='blue.png',
               color=(0, 0, 255), size=(600, 1200))
        record_id = record.id

    timeline = client.get(f'/api/train/run/{record_id}/timeline').get_json()
    series_id = timeline['series'][0]['id']
    response = client.get(
        f'/api/train/run/{record_id}/timeline/{series_id}/gif'
        '?max_edge=99999&fade_frames=999&duration_ms=-50')

    assert response.status_code == 200
    assert response.mimetype == 'image/gif'
    assert 'attachment' in response.headers['Content-Disposition'].lower()
    with Image.open(BytesIO(response.data)) as gif:
        assert gif.is_animated
        assert gif.n_frames == 5  # 2 keyframes + the clamped 3 real fades.
        assert max(gif.size) <= 768
        assert gif.size == (768, 768)  # mixed aspects are contained, not stretched
        pixels = []
        for index in range(gif.n_frames):
            gif.seek(index)
            rgb = gif.convert('RGB')
            pixels.append(rgb.getpixel((gif.width // 2, gif.height // 2)))
        assert pixels[0][0] > 240 and pixels[0][2] < 20
        assert pixels[-1][2] > 240 and pixels[-1][0] < 20
        assert any(red > 20 and blue > 20 for red, _green, blue in pixels[1:-1])
    response.close()


def test_database_filename_cannot_escape_the_real_dataset_directory(client, app):
    from app.extensions import db
    from app.services import checkpoint_timeline as timeline
    from app.services.dataset_storage import ensure_dataset_dir
    with app.app_context():
        dataset_id = _create(client)
        record = _record(db, dataset_id)
        dataset_dir = Path(ensure_dataset_dir(dataset_id))
        outside = dataset_dir.parent / 'outside.png'
        Image.new('RGB', (16, 16), (0, 255, 0)).save(outside)
        _image(db, dataset_id, record.id, 100, filename='../outside.png',
               on_disk=False)
        _image(db, dataset_id, record.id, 200, filename='..\\outside.png',
               on_disk=False)
        _image(db, dataset_id, record.id, 300, filename=str(outside),
               on_disk=False)
        record_id = record.id
        result = timeline.checkpoint_timeline(record_id)

    assert result['candidate_count'] == result['excluded'] == 3
    assert result['excluded_counts']['missing_or_unsafe_file'] == 3
    assert result['series'] == []
    response = client.get(
        f'/api/train/run/{record_id}/timeline/{"0" * 64}/gif'
        f'?path={outside}')
    assert response.status_code == 404


def test_gif_rejects_an_oversized_source_before_full_decode(client, app,
                                                             monkeypatch):
    from app.extensions import db
    from app.services.dataset_storage import ensure_dataset_dir
    with app.app_context():
        dataset_id = _create(client)
        record = _record(db, dataset_id)
        _image(db, dataset_id, record.id, 100, filename='normal.png')
        _image(db, dataset_id, record.id, 200, filename='oversized.png',
               on_disk=False)
        oversized = Path(ensure_dataset_dir(dataset_id)) / 'oversized.png'
        # One-bit pixels keep this fixture tiny while its declared dimensions
        # exceed the decoder budget that protects the threaded Flask process.
        Image.new('1', (5000, 4000), 1).save(oversized)
        record_id = record.id

    real_load = Image.Image.load

    def guarded_load(image, *args, **kwargs):
        if image.width == 5000 and image.height == 4000:
            raise AssertionError('oversized pixels must be rejected before load')
        return real_load(image, *args, **kwargs)

    monkeypatch.setattr(Image.Image, 'load', guarded_load)

    timeline = client.get(f'/api/train/run/{record_id}/timeline').get_json()
    series_id = timeline['series'][0]['id']
    oversized_frame = next(
        frame for frame in timeline['series'][0]['frames'] if frame['step'] == 200)
    assert client.get(oversized_frame['url']).status_code == 404
    response = client.get(
        f'/api/train/run/{record_id}/timeline/{series_id}/gif')
    assert response.status_code == 404
    assert 'fewer than two readable frames' in response.get_json()['error']


def test_unknown_series_id_is_a_404(client, app):
    from app.extensions import db
    with app.app_context():
        dataset_id = _create(client)
        record_id = _record(db, dataset_id).id
    response = client.get(
        f'/api/train/run/{record_id}/timeline/{"f" * 64}/gif')
    assert response.status_code == 404
    assert 'not found' in response.get_json()['error']


def test_record_id_outside_sqlite_range_is_a_404_on_both_routes(client):
    too_large = 1 << 63
    series_id = 'f' * 64
    assert client.get(f'/api/train/run/{too_large}/timeline').status_code == 404
    assert client.get(
        f'/api/train/run/{too_large}/timeline/{series_id}/gif').status_code == 404


def test_gif_busy_gate_is_non_blocking_and_retryable(client, monkeypatch):
    from app.services import checkpoint_timeline as timeline

    class BusyGate:
        def acquire(self, blocking=False):
            assert blocking is False
            return False

        def release(self):  # pragma: no cover - acquisition deliberately fails
            raise AssertionError('an unacquired gate must not be released')

    monkeypatch.setattr(timeline, '_GIF_RENDER_GATE', BusyGate())
    response = client.get(f'/api/train/run/1/timeline/{"f" * 64}/gif')
    assert response.status_code == 429
    assert response.headers['Retry-After'] == '1'


def test_malformed_series_id_short_circuits_timeline_scan(client, monkeypatch):
    from app.services import checkpoint_timeline as timeline

    def forbidden_scan(_record_id):
        raise AssertionError('malformed digest must not scan database or files')

    monkeypatch.setattr(timeline, '_build_timeline', forbidden_scan)
    response = client.get('/api/train/run/1/timeline/not-a-sha256/gif')
    assert response.status_code == 404


def test_gif_output_cap_returns_413(client, app, monkeypatch):
    from app.extensions import db
    from app.services import checkpoint_timeline as timeline
    with app.app_context():
        dataset_id = _create(client)
        record = _record(db, dataset_id)
        _image(db, dataset_id, record.id, 100, filename='cap-a.png')
        _image(db, dataset_id, record.id, 200, filename='cap-b.png')
        record_id = record.id
    series_id = client.get(
        f'/api/train/run/{record_id}/timeline').get_json()['series'][0]['id']
    monkeypatch.setattr(timeline, 'GIF_OUTPUT_BYTE_CAP', 1)
    response = client.get(
        f'/api/train/run/{record_id}/timeline/{series_id}/gif')
    assert response.status_code == 413
