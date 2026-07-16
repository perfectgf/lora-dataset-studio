"""Scrape DIRECT → dataset CONCEPT (scrape_import_urls).

Flux AUTONOME : les images scannées sélectionnées sont téléchargées directement
dans le dataset. Vérifie : filtres qualité (côté court ≥768, ratio ≤3:1), dedup
dHash intra-batch ET vs images existantes, mapping des raisons de download, et la
route (concept-only, cap, compteurs skipped). Le download réseau est monkeypatché.

Porté de l'app source, adapté à notre extraction mono-utilisateur : LOCAL_USER,
racine d'images via la fixture `app`, route sous app.routes.datasets.
"""
import io
import json
import zipfile
from unittest.mock import patch

import pytest
from PIL import Image

from app.models import FaceDatasetImage
from app.scrape.sources.base import Match
from app.scrape.sources.pexels import PexelsSource
from app.services import face_dataset_service as svc
from app.config import LOCAL_USER


def _img_bytes(w=1280, h=960, fmt='JPEG', grad=None):
    """grad='ltr'|'rtl' → gradient horizontal BASSE FRÉQUENCE. Indispensable pour
    les tests dHash : un motif fin se fait moyenner par le resize 9×8 en aplat
    uniforme → collisions. Un gradient survit au downscale (ltr→0, rtl→1)."""
    if grad:
        ramp = list(range(0, 256, 32))
        if grad == 'rtl':
            ramp = ramp[::-1]
        small = Image.new('L', (8, 8))
        small.putdata([ramp[x] for _ in range(8) for x in range(8)])
        im = small.resize((w, h), Image.BILINEAR).convert('RGB')
    else:
        im = Image.new('RGB', (w, h), (120, 40, 40))
    b = io.BytesIO()
    im.save(b, fmt)
    return b.getvalue()


def _concept(user_id=LOCAL_USER):
    return svc.create_dataset(user_id, 'CIM', 'cim_act', kind='concept',
                              concept_desc='an ice cream cone being licked')


def _item(url):
    return {'url': url, 'title': ''}


def _pexels_item(photo_id='123'):
    return {
        'url': f'https://images.pexels.com/photos/{photo_id}/photo.jpeg',
        'title': 'A Pexels photo',
        'platform': 'pexels',
        'source_url': f'https://www.pexels.com/photo/example-{photo_id}/',
        'photographer': f'Photographer {photo_id}',
        'photographer_url': f'https://www.pexels.com/@photographer-{photo_id}/',
    }


def _fake_downloader(by_url):
    """Retourne un _download_scrape_item bouchonné depuis {url: bytes|None}.
    None → échec réseau ('errors')."""
    def _dl(item):
        data = by_url.get(item['url'])
        return ('ok', data) if data is not None else ('errors', None)
    return _dl


# --- dHash -------------------------------------------------------------------
def test_dhash_dupe_and_distinct():
    as_jpg = Image.open(io.BytesIO(_img_bytes(grad='ltr')))
    as_png = Image.open(io.BytesIO(_img_bytes(grad='ltr', fmt='PNG')))
    other = Image.open(io.BytesIO(_img_bytes(grad='rtl')))
    assert svc._hamming(svc._dhash(as_jpg), svc._dhash(as_png)) <= svc.SCRAPE_DHASH_MAX_DISTANCE
    assert svc._hamming(svc._dhash(as_jpg), svc._dhash(other)) > svc.SCRAPE_DHASH_MAX_DISTANCE


# --- Service : happy path + filtres ------------------------------------------
def test_scrape_import_happy_path(app):
    with app.app_context():
        c = _concept()
        by_url = {'http://x/a.jpg': _img_bytes(grad='ltr'),
                  'http://x/b.jpg': _img_bytes(grad='rtl')}
        with patch.object(svc, '_download_scrape_item', _fake_downloader(by_url)):
            res = svc.scrape_import_urls(LOCAL_USER, c.id, [_item('http://x/a.jpg'), _item('http://x/b.jpg')])
        assert res['imported'] == 2
        assert all(v == 0 for v in res['skipped'].values()), res['skipped']
        rows = FaceDatasetImage.query.filter_by(dataset_id=c.id).all()
        assert len(rows) == 2 and all(r.source == 'import' and r.status == 'keep' for r in rows)


def test_pexels_scan_import_payload_and_backup_preserve_provenance(app, monkeypatch):
    with app.app_context():
        c = _concept()
        monkeypatch.setenv('PEXELS_API_KEY', 'integration-test-key')
        api_photo = {
            'id': 31415,
            'url': 'https://www.pexels.com/photo/example-31415/',
            'photographer': 'Photographer 31415',
            'photographer_url': 'https://www.pexels.com/@photographer-31415/',
            'alt': 'A Pexels photo',
            'src': {
                'original': 'https://images.pexels.com/photos/31415/photo.jpeg',
                'medium': 'https://images.pexels.com/photos/31415/medium.jpeg',
            },
        }
        with patch('app.scrape.sources.pexels._request_json', return_value=(
                {'photos': [api_photo], 'next_page': None}, None)):
            scanned, error = PexelsSource().scan(Match(
                url='https://www.pexels.com/search/portrait/'))
        assert error is None and len(scanned) == 1
        item = scanned[0]
        data = _img_bytes(grad='ltr')
        with patch.object(svc, '_download_scrape_item',
                          _fake_downloader({item['url']: data})):
            res = svc.scrape_import_urls(LOCAL_USER, c.id, [item])

        assert res['imported'] == 1
        expected = {key: item[key] for key in (
            'platform', 'source_url', 'photographer', 'photographer_url')}
        row = FaceDatasetImage.query.filter_by(dataset_id=c.id).one()
        assert json.loads(row.source_metadata) == expected
        assert svc.dataset_payload(LOCAL_USER, c.id)['images'][0]['source_metadata'] == expected

        backup = svc.build_backup_zip(LOCAL_USER, c.id)
        with zipfile.ZipFile(io.BytesIO(backup)) as archive:
            archived = json.loads(archive.read('images.json'))
        assert archived[0]['source_metadata'] == expected

        restored = svc.import_backup_zip(LOCAL_USER, backup)
        restored_payload = svc.dataset_payload(LOCAL_USER, restored.id)
        assert restored_payload['images'][0]['source_metadata'] == expected


def test_spoofed_pexels_links_are_dropped_without_blocking_import(app):
    with app.app_context():
        c = _concept()
        item = _pexels_item('2718')
        item['source_url'] = 'https://evil.example/pexels-lookalike'
        data = _img_bytes(grad='ltr')
        with patch.object(svc, '_download_scrape_item',
                          _fake_downloader({item['url']: data})):
            res = svc.scrape_import_urls(LOCAL_USER, c.id, [item])

        assert res['imported'] == 1
        row = FaceDatasetImage.query.filter_by(dataset_id=c.id).one()
        assert row.source_metadata is None
        assert svc.dataset_payload(LOCAL_USER, c.id)['images'][0]['source_metadata'] is None


def test_source_metadata_column_is_present(app):
    from sqlalchemy import text
    with app.app_context():
        columns = {row[1] for row in svc.db.session.execute(
            text('PRAGMA table_info(face_dataset_image)'))}
    assert 'source_metadata' in columns


def test_scrape_import_filters(app):
    with app.app_context():
        c = _concept()
        by_url = {
            'http://x/low.jpg': _img_bytes(700, 500),          # côté court < 768
            'http://x/wide.jpg': _img_bytes(3000, 800),        # ratio 3.75
            'http://x/dup1.jpg': _img_bytes(grad='ltr'),
            'http://x/dup2.jpg': _img_bytes(grad='ltr', fmt='PNG'),
            'http://x/dead.jpg': None,                          # échec réseau
        }
        urls = ['low', 'wide', 'dup1', 'dup2', 'dead']
        items = [_item(f'http://x/{u}.jpg') for u in urls]
        with patch.object(svc, '_download_scrape_item', _fake_downloader(by_url)):
            res = svc.scrape_import_urls(LOCAL_USER, c.id, items)
        assert res['imported'] == 1  # dup1 seul survit
        assert res['skipped'] == {'duplicates': 1, 'low_res': 1, 'extreme_ratio': 1,
                                  'not_image': 0, 'errors': 1}


def test_scrape_import_dedup_vs_existing(app):
    with app.app_context():
        c = _concept()
        data = _img_bytes(grad='ltr')
        ids, _ = svc.import_images(LOCAL_USER, c.id, [data], crop=False)  # déjà présente (webp normalisé)
        assert len(ids) == 1
        with patch.object(svc, '_download_scrape_item', _fake_downloader({'http://x/again.jpg': data})):
            res = svc.scrape_import_urls(LOCAL_USER, c.id, [_item('http://x/again.jpg')])
        assert res['imported'] == 0 and res['skipped']['duplicates'] == 1


def test_scrape_import_maps_not_image(app):
    with app.app_context():
        c = _concept()

        def _dl(item):
            return ('not_image', None)
        with patch.object(svc, '_download_scrape_item', _dl):
            res = svc.scrape_import_urls(LOCAL_USER, c.id, [_item('http://x/a.gif')])
        assert res['imported'] == 0 and res['skipped']['not_image'] == 1


def test_scrape_import_ignores_itemless(app):
    with app.app_context():
        c = _concept()
        res = svc.scrape_import_urls(LOCAL_USER, c.id, [{}, {'title': 'x'}, None])
        assert res['imported'] == 0


def test_small_rescue_is_opt_in_and_does_not_preflight_when_disabled(app):
    with app.app_context():
        c = _concept()
        data = _img_bytes(700, 500)
        with patch.object(svc, '_download_scrape_item',
                          _fake_downloader({'http://x/small.jpg': data})), \
             patch('app.services.klein_edit_helper.klein_missing_assets') as preflight:
            res = svc.scrape_import_urls(
                LOCAL_USER, c.id, [_item('http://x/small.jpg')])
        assert res['imported'] == 0
        assert res['rescue_queued'] == res['rescue_failed'] == 0
        assert res['skipped']['low_res'] == 1
        preflight.assert_not_called()
        assert FaceDatasetImage.query.filter_by(dataset_id=c.id).count() == 0


def test_small_rescue_preserves_original_and_queues_empty_prompt(app):
    with app.app_context():
        c = _concept()
        data = _img_bytes(700, 500)
        calls = []

        def enqueue(**kwargs):
            calls.append(kwargs)
            return 'small-rescue-job'

        real_get = svc.cfg.get

        def config_get(key, default=None):
            return '' if key == 'klein.small_image_prompt' else real_get(key, default)

        with patch.object(svc, '_download_scrape_item',
                          _fake_downloader({'http://x/small.jpg': data})), \
             patch('app.services.klein_edit_helper.klein_missing_assets', return_value=[]), \
             patch('app.services.klein_edit_helper.enqueue_klein_edit', side_effect=enqueue), \
             patch.object(svc.cfg, 'get', side_effect=config_get):
            res = svc.scrape_import_urls(
                LOCAL_USER, c.id, [_item('http://x/small.jpg')], rescue_small=True)

        assert res['imported'] == 0
        assert res['rescue_queued'] == 1 and res['rescue_failed'] == 0
        assert res['skipped']['low_res'] == 0
        source = FaceDatasetImage.query.filter_by(
            dataset_id=c.id, derivation_kind=svc.SMALL_IMAGE_SOURCE).one()
        candidate = FaceDatasetImage.query.filter_by(
            dataset_id=c.id, derivation_kind=svc.KLEIN_SMALL_IMAGE).one()
        assert source.status == candidate.status == 'pending'
        assert source.filename and candidate.filename is None
        assert candidate.parent_image_id == source.id
        assert candidate.job_id == 'small-rescue-job'
        assert candidate.variation_prompt == ''
        assert calls[0]['edit_prompt'] == ''
        assert calls[0]['source_path'] == svc._img_path(source)
        with open(svc._img_path(source), 'rb') as fh:
            assert fh.read() == data


def test_small_rescue_enqueue_failure_keeps_original_and_failed_candidate(app):
    with app.app_context():
        c = _concept()
        data = _img_bytes(700, 500)
        with patch.object(svc, '_download_scrape_item',
                          _fake_downloader({'http://x/small.jpg': data})), \
             patch('app.services.klein_edit_helper.klein_missing_assets', return_value=[]), \
             patch('app.services.klein_edit_helper.enqueue_klein_edit',
                   side_effect=RuntimeError('ComfyUI offline')):
            res = svc.scrape_import_urls(
                LOCAL_USER, c.id, [_item('http://x/small.jpg')], rescue_small=True)
        assert res['rescue_queued'] == 0 and res['rescue_failed'] == 1
        source = FaceDatasetImage.query.filter_by(
            dataset_id=c.id, derivation_kind=svc.SMALL_IMAGE_SOURCE).one()
        candidate = FaceDatasetImage.query.filter_by(
            dataset_id=c.id, derivation_kind=svc.KLEIN_SMALL_IMAGE).one()
        assert source.status == 'pending' and source.filename
        assert candidate.status == 'failed'
        assert 'ComfyUI offline' in candidate.fail_reason
        resolved = svc.resolve_small_image_rescue(
            LOCAL_USER, c.id, candidate.id, 'original')
        assert resolved['source']['status'] == 'keep'
        assert resolved['candidate']['status'] == 'reject'


def test_small_rescue_hd_duplicate_wins_before_klein_preflight(app):
    with app.app_context():
        c = _concept()
        low = _img_bytes(700, 500, grad='ltr')
        high = _img_bytes(1280, 960, grad='ltr')
        by_url = {'http://x/low.jpg': low, 'http://x/high.jpg': high}
        with patch.object(svc, '_download_scrape_item', _fake_downloader(by_url)), \
             patch('app.services.klein_edit_helper.klein_missing_assets') as preflight:
            res = svc.scrape_import_urls(
                LOCAL_USER, c.id,
                [_item('http://x/low.jpg'), _item('http://x/high.jpg')],
                rescue_small=True)
        assert res['imported'] == 1 and res['rescue_queued'] == 0
        assert res['skipped']['duplicates'] == 1
        preflight.assert_not_called()


def test_rescue_sort_keeps_metadata_attached_to_winning_pexels_bytes(app):
    with app.app_context():
        c = _concept()
        low_item = _pexels_item('100')
        high_item = _pexels_item('200')
        low = _img_bytes(700, 500, grad='ltr')
        high = _img_bytes(1280, 960, grad='ltr')
        by_url = {low_item['url']: low, high_item['url']: high}
        with patch.object(svc, '_download_scrape_item', _fake_downloader(by_url)), \
             patch('app.services.klein_edit_helper.klein_missing_assets') as preflight:
            res = svc.scrape_import_urls(
                LOCAL_USER, c.id, [low_item, high_item], rescue_small=True)

        assert res['imported'] == 1 and res['skipped']['duplicates'] == 1
        metadata = svc.dataset_payload(LOCAL_USER, c.id)['images'][0]['source_metadata']
        assert metadata['source_url'] == high_item['source_url']
        assert metadata['photographer'] == high_item['photographer']
        preflight.assert_not_called()


def test_small_rescue_source_and_candidate_keep_pexels_provenance(app):
    with app.app_context():
        c = _concept()
        item = _pexels_item('8080')
        data = _img_bytes(700, 500)
        with patch.object(svc, '_download_scrape_item',
                          _fake_downloader({item['url']: data})), \
             patch('app.services.klein_edit_helper.klein_missing_assets', return_value=[]), \
             patch('app.services.klein_edit_helper.enqueue_klein_edit',
                   return_value='pexels-rescue-job'):
            res = svc.scrape_import_urls(
                LOCAL_USER, c.id, [item], rescue_small=True)

        assert res['rescue_queued'] == 1
        rows = FaceDatasetImage.query.filter_by(dataset_id=c.id).all()
        assert len(rows) == 2
        expected = {key: item[key] for key in (
            'platform', 'source_url', 'photographer', 'photographer_url')}
        assert all(json.loads(row.source_metadata) == expected for row in rows)
        payload = svc.dataset_payload(LOCAL_USER, c.id)
        assert all(image['source_metadata'] == expected for image in payload['images'])


def test_resolve_small_rescue_is_atomic_idempotent_and_non_reversible(app):
    with app.app_context():
        c = _concept()
        source = FaceDatasetImage(
            dataset_id=c.id, source='import', filename='source.jpg', status='pending',
            derivation_kind=svc.SMALL_IMAGE_SOURCE)
        svc.db.session.add(source)
        svc.db.session.flush()
        candidate = FaceDatasetImage(
            dataset_id=c.id, source='generated', filename='result.png', status='pending',
            derivation_kind=svc.KLEIN_SMALL_IMAGE, parent_image_id=source.id)
        svc.db.session.add(candidate)
        svc.db.session.commit()

        first = svc.resolve_small_image_rescue(
            LOCAL_USER, c.id, candidate.id, 'klein')
        assert first['source']['status'] == 'reject'
        assert first['candidate']['status'] == 'keep'
        same = svc.resolve_small_image_rescue(
            LOCAL_USER, c.id, candidate.id, 'klein')
        assert same == first
        with pytest.raises(RuntimeError, match='already resolved as klein'):
            svc.resolve_small_image_rescue(
                LOCAL_USER, c.id, candidate.id, 'original')


def test_late_rescue_callbacks_cannot_overwrite_rejected_choice(app):
    with app.app_context():
        c = _concept()
        source = FaceDatasetImage(
            dataset_id=c.id, source='import', filename='source.jpg', status='pending',
            derivation_kind=svc.SMALL_IMAGE_SOURCE)
        svc.db.session.add(source)
        svc.db.session.flush()
        candidate = FaceDatasetImage(
            dataset_id=c.id, source='generated', status='pending', job_id='late-job',
            derivation_kind=svc.KLEIN_SMALL_IMAGE, parent_image_id=source.id)
        svc.db.session.add(candidate)
        svc.db.session.commit()

        svc.resolve_small_image_rescue(LOCAL_USER, c.id, candidate.id, 'original')
        svc.link_completed_dataset_image('late-job', None, failed=True,
                                         reason='cancel raced')
        svc.db.session.refresh(candidate)
        assert candidate.status == 'reject' and candidate.filename is None


def test_late_rescue_callback_ignores_activity_sync_failure(app, monkeypatch):
    with app.app_context():
        c = _concept()
        source = FaceDatasetImage(
            dataset_id=c.id, source='import', filename='source.jpg', status='keep',
            derivation_kind=svc.SMALL_IMAGE_SOURCE)
        svc.db.session.add(source)
        svc.db.session.flush()
        candidate = FaceDatasetImage(
            dataset_id=c.id, source='generated', status='reject', job_id='late-sync-job',
            derivation_kind=svc.KLEIN_SMALL_IMAGE, parent_image_id=source.id)
        svc.db.session.add(candidate)
        svc.db.session.commit()

        def broken_sync(_dataset_id):
            raise RuntimeError('activity registry unavailable')

        monkeypatch.setattr(svc, '_sync_generate_activity', broken_sync)
        # Must not propagate to job_queue._dispatch_completion and must not alter
        # either terminal status.
        svc.link_completed_dataset_image(
            'late-sync-job', 'ignored-result.png', failed=False)
        svc.db.session.refresh(source)
        svc.db.session.refresh(candidate)
        assert source.status == 'keep'
        assert candidate.status == 'reject' and candidate.filename is None
        svc.link_completed_dataset_image('late-job', 'late-result.png', failed=False)
        svc.db.session.refresh(candidate)
        assert candidate.status == 'reject' and candidate.filename is None


def test_backup_restores_all_small_rescue_pair_states_and_remaps_parents(app):
    with app.app_context():
        c = _concept()
        pairs = {}
        for label, source_status, candidate_status, ready in (
                ('queued', 'pending', 'pending', False),
                ('failed', 'pending', 'failed', False),
                ('ready', 'pending', 'pending', True),
                ('resolved', 'keep', 'reject', False)):
            source_name = f'{label}-source.jpg'
            with open(f'{svc._dataset_dir(c.id)}/{source_name}', 'wb') as fh:
                fh.write(_img_bytes(700, 500))
            source = FaceDatasetImage(
                dataset_id=c.id, source='import', filename=source_name,
                status=source_status, derivation_kind=svc.SMALL_IMAGE_SOURCE,
                variation_label=f'{label}-source')
            svc.db.session.add(source)
            svc.db.session.flush()
            candidate_name = f'{label}-result.png' if ready else None
            if candidate_name:
                with open(f'{svc._dataset_dir(c.id)}/{candidate_name}', 'wb') as fh:
                    fh.write(_img_bytes(1024, 1024, fmt='PNG'))
            candidate = FaceDatasetImage(
                dataset_id=c.id, source='generated', filename=candidate_name,
                status=candidate_status, derivation_kind=svc.KLEIN_SMALL_IMAGE,
                parent_image_id=source.id, variation_label=label,
                job_id=f'{label}-machine-job',
                fail_reason='boom' if label == 'failed' else None)
            svc.db.session.add(candidate)
            pairs[label] = (source, candidate)
        svc.db.session.commit()

        restored = svc.import_backup_zip(
            LOCAL_USER, svc.build_backup_zip(LOCAL_USER, c.id))
        rows = FaceDatasetImage.query.filter_by(dataset_id=restored.id).all()
        assert len(rows) == 8
        by_label = {row.variation_label: row for row in rows}
        for label in pairs:
            source = by_label[f'{label}-source']
            candidate = by_label[label]
            assert candidate.parent_image_id == source.id
            assert candidate.job_id is None
        assert by_label['queued'].status == 'failed'
        assert 'in flight when this backup was created' in by_label['queued'].fail_reason
        assert by_label['failed'].status == 'failed'
        assert by_label['failed'].fail_reason == 'boom'
        assert by_label['ready'].status == 'pending' and by_label['ready'].filename
        assert by_label['resolved-source'].status == 'keep'
        assert by_label['resolved'].status == 'reject'


# --- Downloader (SSRF + type) ------------------------------------------------
def test_download_scrape_item_rejects_private_host(app):
    with app.app_context():
        # _validate_public_http_url doit refuser un host qui résout en IP privée.
        reason, data = svc._download_scrape_item({'url': 'http://127.0.0.1/a.jpg'})
        assert reason == 'errors' and data is None


# --- Route -------------------------------------------------------------------
def test_route_accepts_character_datasets(client, app):
    """The concept-only gate is GONE: character datasets scrape too (images
    import full-frame, the user crops tiles manually afterwards)."""
    with app.app_context():
        p = svc.create_dataset(LOCAL_USER, 'Emma', 'z')  # character
        did = p.id
    fake = {'imported': 1, 'skipped': {'duplicates': 0, 'low_res': 0, 'extreme_ratio': 0,
                                       'not_image': 0, 'errors': 0}}
    with patch('app.routes.datasets.svc.scrape_import_urls', return_value=fake):
        resp = client.post(f'/api/dataset/{did}/scrape-import',
                           json={'items': [{'url': 'http://x/a.jpg'}]})
    assert resp.status_code == 200
    assert resp.get_json()['imported'] == 1


def test_route_validates_payload_and_cap(client, app):
    with app.app_context():
        c = _concept()
        did = c.id
    assert client.post(f'/api/dataset/{did}/scrape-import', json={}).status_code == 400
    too_many = [{'url': f'http://x/{i}.jpg'} for i in range(svc.SCRAPE_IMPORT_MAX + 1)]
    assert client.post(f'/api/dataset/{did}/scrape-import',
                       json={'items': too_many}).status_code == 400


def test_route_happy_path_reports_counters(client, app):
    with app.app_context():
        c = _concept()
        did = c.id
    fake = {'imported': 2, 'skipped': {'duplicates': 1, 'low_res': 0, 'extreme_ratio': 0,
                                       'not_image': 0, 'errors': 0}}
    with patch('app.routes.datasets.svc.scrape_import_urls', return_value=fake) as m:
        resp = client.post(f'/api/dataset/{did}/scrape-import',
                           json={'items': [{'url': 'http://x/a.jpg'}, {'url': 'http://x/b.jpg'}]})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['ok'] is True and body['imported'] == 2 and body['skipped']['duplicates'] == 1
    m.assert_called_once()


def test_route_forwards_rescue_small_and_rejects_non_boolean(client, app):
    with app.app_context():
        c = _concept()
        did = c.id
    items = [{'url': 'http://x/a.jpg'}]
    fake = {'imported': 0, 'rescue_queued': 1, 'rescue_failed': 0,
            'skipped': {'duplicates': 0, 'low_res': 0, 'extreme_ratio': 0,
                        'not_image': 0, 'errors': 0}}
    with patch('app.routes.datasets.svc.scrape_import_urls', return_value=fake) as mocked:
        resp = client.post(f'/api/dataset/{did}/scrape-import',
                           json={'items': items, 'rescue_small': True})
    assert resp.status_code == 200 and resp.get_json()['rescue_queued'] == 1
    mocked.assert_called_once_with(LOCAL_USER, did, items, rescue_small=True)
    bad = client.post(f'/api/dataset/{did}/scrape-import',
                      json={'items': items, 'rescue_small': 'yes'})
    assert bad.status_code == 400


def test_route_resolves_small_image_pair(client, app):
    with app.app_context():
        c = _concept()
        source = FaceDatasetImage(
            dataset_id=c.id, source='import', filename='source.jpg', status='pending',
            derivation_kind=svc.SMALL_IMAGE_SOURCE)
        svc.db.session.add(source)
        svc.db.session.flush()
        candidate = FaceDatasetImage(
            dataset_id=c.id, source='generated', filename='result.png', status='pending',
            derivation_kind=svc.KLEIN_SMALL_IMAGE, parent_image_id=source.id)
        svc.db.session.add(candidate)
        svc.db.session.commit()
        did, cid = c.id, candidate.id
    resp = client.post(
        f'/api/dataset/{did}/small-image-rescue/{cid}/resolve',
        json={'choice': 'original'})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['ok'] is True
    assert body['source']['status'] == 'keep'
    assert body['candidate']['status'] == 'reject'
