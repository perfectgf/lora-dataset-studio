"""Settings › Storage: the "what lives where" map and the relocation guard.

The map is only useful if it tells the truth about paths WITHOUT walking them
(the tab must open instantly on a 100 GB install), and the relocation is only
safe if an unusable target is refused BEFORE anything is saved and nothing is
ever moved without being asked.
"""
import os



def test_map_lists_every_category_without_walking_the_disk(app, monkeypatch):
    from app.services import storage_locations as sl
    from app.services import lora_training as lt
    walked = []
    monkeypatch.setattr(lt, '_dir_size', lambda p: walked.append(p) or 0)
    with app.app_context():
        rows = sl.locations()
    keys = [r['key'] for r in rows]
    for expected in ('datasets', 'cloud_runs', 'checkpoints', 'trash',
                     'run_archive', 'dist'):
        assert expected in keys
    assert walked == []                       # mounting the tab costs no walk
    by_key = {r['key']: r for r in rows}
    # the two new roots are relocatable and default to the data dir
    assert by_key['checkpoints']['relocatable'] is True
    assert by_key['checkpoints']['configured'] == ''
    assert by_key['checkpoints']['path'].endswith('checkpoints')
    # the trash is real but not something you point elsewhere from here
    assert by_key['trash']['relocatable'] is False
    # every row explains what is inside, in the user's terms
    assert all(r['holds'] for r in rows)


def test_sizes_are_a_separate_explicit_call(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import storage_locations as sl
    with app.app_context():
        root = cfg.checkpoints_root()
        (root / 'run_1').mkdir(parents=True, exist_ok=True)
        (root / 'run_1' / 'a.safetensors').write_bytes(b'x' * 4096)
        sizes = sl.sizes(['checkpoints'])
        assert sizes == {'checkpoints': 4096}
        assert sl.sizes(['nope']) == {}


def test_target_validation_refuses_what_would_break(app, tmp_path):
    from app.services import storage_locations as sl
    with app.app_context():
        assert sl.validate_target('trash', str(tmp_path))['ok'] is False
        assert sl.validate_target('checkpoints', 'relative/path')['ok'] is False
        # blank means "back to the default", which is always allowed
        blank = sl.validate_target('checkpoints', '')
        assert blank['ok'] is True and blank['default'] is True
        # a real, writable folder passes and reports whether it is empty
        good = sl.validate_target('checkpoints', str(tmp_path / 'newstore'))
        assert good['ok'] is True and good['empty'] is True
        assert os.path.isdir(good['path'])
        assert good.get('free_bytes', 0) > 0
        (tmp_path / 'newstore' / 'stuff.bin').write_bytes(b'x')
        assert sl.validate_target('checkpoints', str(tmp_path / 'newstore'))['empty'] is False


def test_moving_into_a_child_of_the_current_folder_is_refused(app, tmp_path):
    from app import config as cfg
    from app.services import storage_locations as sl
    with app.app_context():
        cfg.save_config({'paths': {'checkpoints_dir': str(tmp_path / 'store')}})
        res = sl.validate_target('checkpoints', str(tmp_path / 'store' / 'inner'))
        assert res['ok'] is False and 'inside' in res['reason']


def test_move_copies_everything_before_removing_the_source(app, tmp_path):
    from app.services import storage_locations as sl
    with app.app_context():
        src = tmp_path / 'src_store'
        (src / 'run_3').mkdir(parents=True)
        (src / 'run_3' / 'a.safetensors').write_bytes(b'x' * 2048)
        from app import config as cfg
        cfg.save_config({'paths': {'checkpoints_dir': str(src)}})
        dest = tmp_path / 'moved_store'
        job = sl.start_move('checkpoints', str(dest))
        for _ in range(200):
            state = sl.move_progress(job['job_id'])
            if state['phase'] in ('done', 'error'):
                break
            import time
            time.sleep(0.05)
        assert state['phase'] == 'done', state.get('error')
        assert (dest / 'run_3' / 'a.safetensors').read_bytes() == b'x' * 2048
        assert not src.exists()
        assert state['files_total'] == 1 and state['bytes_total'] == 2048


def test_routes_expose_the_map_and_refuse_a_bad_target(client, app):
    body = client.get('/api/storage/locations').get_json()
    assert any(r['key'] == 'checkpoints' for r in body['locations'])
    sizes = client.get('/api/storage/sizes?keys=checkpoints').get_json()
    assert sizes['ok'] is True and 'checkpoints' in sizes['sizes']
    bad = client.post('/api/storage/validate',
                      json={'key': 'checkpoints', 'path': 'nope'}).get_json()
    assert bad['ok'] is False
    assert client.post('/api/storage/move',
                       json={'key': 'checkpoints', 'path': 'nope'}).status_code == 400
    assert client.post('/api/storage/adopt-checkpoints').get_json()['ok'] is True
