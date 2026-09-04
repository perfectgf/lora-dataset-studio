"""✨ Neural render (DLSS 5) — the rules around the model, without the model.

Nothing here touches a GPU, a driver or the 165 MB file: the child process is
replaced by a stand-in wherever a render is asked for, and the rest is the
part this app owns — the dials and their ranges, the width floor, the file
probe and its sentences, the pinned install, the in-place pass with its
write-once backup, and the restore that undoes it.
"""
import hashlib
import io
import json
import re
import zipfile

import pytest

from app.services import neural_render as nr


# ── dials ───────────────────────────────────────────────────────────────────

def test_defaults_and_ranges():
    p = nr.normalize_params({})
    assert p == {'tone': 1.0, 'structure': 1.0, 'automask': False, 'temporal': 'auto',
                 'scene_cut': nr.SCENE_CUT_DEFAULT, 'strength': 1.0, 'passes': 1, 'scale': 1}
    assert nr.normalize_params({'tone': '0', 'structure': 2, 'automask': 1, 'temporal': 'ON'})[
        'temporal'] == 'on'
    for bad in ({'tone': 2.5}, {'structure': -0.1}, {'tone': 'x'}, {'temporal': 'maybe'},
                {'scene_cut': 3}):
        with pytest.raises(nr.NeuralRenderError):
            nr.normalize_params(bad)


def test_the_inert_dials_are_not_accepted_as_dials():
    """intensity / skin / preset / style changed nothing through the bridge
    (bit-identical output, swept both ways) — so they are not part of the
    contract, and a request naming them is simply ignored, never honoured."""
    p = nr.normalize_params({'intensity': 2, 'skin': 2, 'preset': 0, 'style': 6})
    assert 'intensity' not in p and 'skin' not in p and 'preset' not in p and 'style' not in p


def test_temporal_width_floor_is_the_measured_one_in_both_halves():
    """704 px: 700 fails, 704 passes on the driver's Optical Flow session, height
    irrelevant. The parent gates and the child re-checks — one number, pinned
    here in both files so they cannot drift."""
    assert nr.TEMPORAL_MIN_WIDTH == 704
    src = (nr.cfg.BACKEND_DIR / 'infer' / 'dlss5nr_infer.py').read_text(encoding='utf-8')
    m = re.search(r'^TEMPORAL_MIN_WIDTH = (\d+)', src, re.M)
    assert m and int(m.group(1)) == nr.TEMPORAL_MIN_WIDTH
    js = (nr.cfg.BACKEND_DIR.parent / 'frontend' / 'src' / 'components' / 'videobank'
          / 'neuralRenderParams.js').read_text(encoding='utf-8')
    m = re.search(r'export const TEMPORAL_MIN_WIDTH = (\d+)', js)
    assert m and int(m.group(1)) == nr.TEMPORAL_MIN_WIDTH


def test_auto_falls_back_below_the_floor_and_on_refuses():
    assert nr.decide_temporal('auto', 1024) == (True, 'temporal mode')
    assert nr.decide_temporal('auto', 512)[0] is False
    assert nr.decide_temporal('off', 4096) == (False, 'still mode')
    assert nr.decide_temporal('auto', None)[0] is True          # unknown width: let the child decide
    with pytest.raises(nr.NeuralRenderError, match='704'):
        nr.decide_temporal('on', 700)
    assert nr.decide_temporal('auto', 1024, nvof=False)[0] is False
    with pytest.raises(nr.NeuralRenderError, match='Optical Flow'):
        nr.decide_temporal('on', 1024, nvof=False)


# ── status: every absence is a sentence ─────────────────────────────────────

def _runtime(tmp_path, bridge=True, shim=True, model=True, small=False):
    root = tmp_path / 'rt'
    (root / 'caller').mkdir(parents=True)
    if bridge:
        (root / nr.BRIDGE_FILE).write_bytes(b'MZ')
    if shim:
        (root / nr.SHIM_FILE).write_bytes(b'MZ')
    if model:
        size = 1024 if small else nr.MODEL_MIN_BYTES
        with open(root / nr.MODEL_FILE, 'wb') as fh:
            fh.truncate(size)
    return root


def test_status_ready_only_when_everything_is_there(tmp_path):
    root = _runtime(tmp_path)
    ok = nr.status(root, os_name='nt', driver={'ngx': True, 'nvof': True}, worker_ok=True, ffmpeg_ok=True)
    assert ok['ready'] and ok['missing'] == [] and ok['driver_nvof']
    assert ok['runtime_dir'] == str(root)


def test_status_names_what_is_missing_in_words(tmp_path):
    root = _runtime(tmp_path, bridge=False, model=False)
    st = nr.status(root, os_name='nt', driver={'ngx': True, 'nvof': False}, worker_ok=True, ffmpeg_ok=True)
    assert not st['ready']
    assert any('bridge' in m and 'Setup' in m for m in st['missing'])
    assert any(nr.MODEL_FILE in m and str(root) in m for m in st['missing'])
    # Linux/Docker: the OS line, and nothing that pretends a driver could fix it.
    st = nr.status(root, os_name='posix', driver={'ngx': False, 'nvof': False}, worker_ok=False, ffmpeg_ok=True)
    assert st['missing'][0].startswith('Windows')
    assert not any('driver' in m for m in st['missing'])
    # A driver-less Windows machine.
    st = nr.status(_runtime(tmp_path / 'b'), os_name='nt', driver={'ngx': False, 'nvof': False}, worker_ok=True, ffmpeg_ok=True)
    assert any('NVIDIA display driver' in m for m in st['missing'])


def test_a_render_process_without_numpy_is_named_and_sent_to_setup(tmp_path):
    st = nr.status(_runtime(tmp_path), os_name='nt', driver={'ngx': True, 'nvof': True}, worker_ok=False, ffmpeg_ok=True)
    assert not st['ready'] and not st['worker']
    assert any('numpy' in m and 'Setup' in m for m in st['missing'])


def test_a_forwarder_under_the_models_name_is_called_out(tmp_path):
    """The classic trap: the 108 KB forwarder from a game mod, saved under the
    model's name. Present is not enough — the size says which file it is."""
    st = nr.status(_runtime(tmp_path, small=True), os_name='nt', driver={'ngx': True, 'nvof': True}, worker_ok=True, ffmpeg_ok=True)
    assert not st['ready']
    assert any('not the model' in m for m in st['missing'])


def test_the_probe_is_the_status(app):
    from app import capabilities
    with app.app_context():
        st = capabilities.probe_dlss5nr()
    assert set(st) >= {'ready', 'missing', 'runtime_dir', 'model_file'}


# ── the install: pinned bytes or nothing ────────────────────────────────────

def _fake_zip(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_install_refuses_bytes_that_are_not_the_pinned_release(app, tmp_path, monkeypatch):
    monkeypatch.setattr(nr, 'runtime_dir', lambda create=False: tmp_path / 'rt')
    data = _fake_zip({m: b'x' for m in nr.BRIDGE_RELEASE['members']})
    log = []
    rc = nr.install_bridge(log=log.append, fetch=lambda url: data)
    assert rc == 1
    assert any('refused' in line and 'pinned' in line for line in log)
    assert not (tmp_path / 'rt' / nr.BRIDGE_FILE).exists()


def test_install_unpacks_the_two_dlls_when_the_bytes_match(app, tmp_path, monkeypatch):
    monkeypatch.setattr(nr, 'runtime_dir', lambda create=False: tmp_path / 'rt')
    data = _fake_zip({m: b'MZ' + m.encode() for m in nr.BRIDGE_RELEASE['members']})
    monkeypatch.setitem(nr.BRIDGE_RELEASE, 'size', len(data))
    monkeypatch.setitem(nr.BRIDGE_RELEASE, 'sha256', hashlib.sha256(data).hexdigest())
    log = []
    assert nr.install_bridge(log=log.append, fetch=lambda url: data) == 0
    assert (tmp_path / 'rt' / nr.BRIDGE_FILE).is_file()
    assert (tmp_path / 'rt' / nr.SHIM_FILE).is_file()
    assert any(nr.MODEL_FILE in line and 'does not download' in line for line in log)


def test_the_install_action_is_wired_and_opt_in():
    from app import setup_installer
    assert 'dlss5nr_bridge' in setup_installer.INSTALL_ACTIONS
    assert setup_installer._WORKERS['dlss5nr_bridge'] is setup_installer._run_dlss5nr_bridge
    assert setup_installer._action_needed('dlss5nr_bridge', {}) is False
    assert nr.BRIDGE_RELEASE['url'] in setup_installer.manual_command('dlss5nr_bridge')


# ── the child's command line ────────────────────────────────────────────────

def test_worker_argv_carries_every_dial_and_the_mode(monkeypatch):
    monkeypatch.setattr(nr, 'worker_python', lambda: 'py.exe')
    params = nr.normalize_params({'tone': 0.5, 'structure': 1.5, 'automask': True})
    argv = nr.worker_argv('a.mp4', 'b.mp4', params, temporal_on=True, ffmpeg='ff.exe')
    joined = ' '.join(argv)
    assert 'dlss5nr_infer.py' in joined
    for flag, value in (('--tone', '0.5'), ('--structure', '1.5'), ('--automask', '1'),
                        ('--temporal', 'on'), ('--ffmpeg', 'ff.exe'), ('--src', 'a.mp4'),
                        ('--dst', 'b.mp4')):
        assert argv[argv.index(flag) + 1] == value


# ── dataset clips: in place, original kept, restorable ──────────────────────

def _dataset(app, tmp_path, names=('clip_0001.mp4', 'clip_0002.mp4')):
    from app.config import LOCAL_USER
    from app.extensions import db
    from app.models import VideoDataset, VideoDatasetClip
    out = tmp_path / 'ds'
    out.mkdir()
    with app.app_context():
        ds = VideoDataset(user_id=LOCAL_USER, name='set', target_profile='wan22',
                          output_dir=str(out))
        db.session.add(ds)
        db.session.flush()
        ids = []
        for n in names:
            (out / n).write_bytes(b'ORIGINAL ' + n.encode())
            (out / n).with_suffix('.txt').write_text('a caption', encoding='utf-8')
            row = VideoDatasetClip(dataset_id=ds.id, filename=n)
            db.session.add(row)
            db.session.flush()
            ids.append(row.id)
        db.session.commit()
        return ds.id, ids, out


def _stub_render(monkeypatch, marker=b'RENDERED'):
    calls = []

    def fake(src, dst, params, on_progress=None, cancel=None, timeout_s=None):
        calls.append((src, dst, dict(params)))
        with open(dst, 'wb') as fh:
            fh.write(marker + b' from ' + open(src, 'rb').read())
        return {'frames': 3, 'mode_note': 'still mode'}
    monkeypatch.setattr(nr, 'render_video', fake)
    monkeypatch.setattr(nr, 'status', lambda root=None, os_name=None, driver=None: {
        'ready': True, 'missing': [], 'driver_nvof': True})
    return calls


def _run_job_inline(monkeypatch):
    """bank_jobs runs the pass on a thread; here it runs now, on this one."""
    from app.services import bank_jobs

    def start(app, key, kind, fn, total=0, reserve_ids=None, reservation=None):
        job = {'kind': kind, 'done': 0, 'total': total, 'error': None, 'cancelled': False,
               'finished': False, 'detail': '', 'started_at': 0, '_touched': 0}
        fn(job)
        return job
    monkeypatch.setattr(bank_jobs, 'start', start)
    monkeypatch.setattr(bank_jobs, 'set_cancel_hook', lambda job, hook: None)
    monkeypatch.setattr(bank_jobs, 'progress', lambda job, **kw: job.update({k: v for k, v in kw.items() if v is not None}))
    monkeypatch.setattr(bank_jobs, 'cancelled', lambda job: False)
    monkeypatch.setattr(bank_jobs, 'fail', lambda job, msg: job.__setitem__('error', msg))
    monkeypatch.setattr(bank_jobs, 'running', lambda key: False)


def test_render_replaces_the_clip_keeps_the_original_and_never_stacks(app, tmp_path, monkeypatch):
    from app.config import LOCAL_USER
    ds_id, ids, out = _dataset(app, tmp_path)
    calls = _stub_render(monkeypatch)
    _run_job_inline(monkeypatch)
    with app.app_context():
        assert nr.rendered_clip_ids(LOCAL_USER, ds_id) == []
        res = nr.start_dataset_render(app, LOCAL_USER, ds_id, [ids[0]], {'tone': 0})
        assert res['queued'] == 1 and res['params']['tone'] == 0.0
        # The file the trainer reads IS the render; the caption did not move.
        assert (out / 'clip_0001.mp4').read_bytes().startswith(b'RENDERED from ORIGINAL clip_0001')
        assert (out / 'clip_0001.txt').read_text(encoding='utf-8') == 'a caption'
        assert (out / 'clip_0002.mp4').read_bytes() == b'ORIGINAL clip_0002.mp4'
        # The original sits OUTSIDE the dataset folder — nothing new in the flat folder.
        assert sorted(p.name for p in out.iterdir()) == ['clip_0001.mp4', 'clip_0001.txt',
                                                         'clip_0002.mp4', 'clip_0002.txt']
        backup = nr.backup_dir(ds_id) / 'clip_0001.mp4'
        assert backup.read_bytes() == b'ORIGINAL clip_0001.mp4'
        assert nr.rendered_clip_ids(LOCAL_USER, ds_id) == [ids[0]]
        # A second render reads the ORIGINAL again (the backup), never the render.
        nr.start_dataset_render(app, LOCAL_USER, ds_id, [ids[0]], {'tone': 2})
        assert calls[-1][0] == str(backup)
        assert (out / 'clip_0001.mp4').read_bytes() == b'RENDERED from ORIGINAL clip_0001.mp4'
        assert backup.read_bytes() == b'ORIGINAL clip_0001.mp4'


def test_restore_puts_the_original_back_and_forgets_the_render(app, tmp_path, monkeypatch):
    from app.config import LOCAL_USER
    ds_id, ids, out = _dataset(app, tmp_path)
    _stub_render(monkeypatch)
    _run_job_inline(monkeypatch)
    with app.app_context():
        nr.start_dataset_render(app, LOCAL_USER, ds_id, [], {})     # empty ids = every clip
        assert sorted(nr.rendered_clip_ids(LOCAL_USER, ds_id)) == sorted(ids)
        assert nr.restore_dataset_clips(LOCAL_USER, ds_id, [ids[1]]) == {'restored': 1}
        assert (out / 'clip_0002.mp4').read_bytes() == b'ORIGINAL clip_0002.mp4'
        assert not (nr.backup_dir(ds_id) / 'clip_0002.mp4').exists()
        assert nr.rendered_clip_ids(LOCAL_USER, ds_id) == [ids[0]]
        assert nr.restore_dataset_clips(LOCAL_USER, ds_id, []) == {'restored': 1}
        assert nr.rendered_clip_ids(LOCAL_USER, ds_id) == []


def test_a_failed_clip_leaves_the_original_in_place_and_is_reported(app, tmp_path, monkeypatch):
    from app.config import LOCAL_USER
    ds_id, ids, out = _dataset(app, tmp_path)
    _stub_render(monkeypatch)
    _run_job_inline(monkeypatch)

    def boom(src, dst, params, **kw):
        raise nr.NeuralRenderError('the model refused the frame')
    monkeypatch.setattr(nr, 'render_video', boom)
    with app.app_context():
        from app.services import bank_jobs
        seen = {}
        monkeypatch.setattr(bank_jobs, 'fail', lambda job, msg: seen.setdefault('msg', msg))
        nr.start_dataset_render(app, LOCAL_USER, ds_id, [ids[0]], {})
        assert (out / 'clip_0001.mp4').read_bytes() == b'ORIGINAL clip_0001.mp4'
        assert 'refused the frame' in seen['msg']
        assert not list(out.glob('.nr-*'))          # no half-written file left behind


def test_the_pass_refuses_when_the_lane_is_not_set_up(app, tmp_path, monkeypatch):
    from app.config import LOCAL_USER
    ds_id, ids, _ = _dataset(app, tmp_path)
    monkeypatch.setattr(nr, 'status', lambda root=None, os_name=None, driver=None: {
        'ready': False, 'missing': ['your own copy of nvngx_dlssnr.dll, placed in X'], 'driver_nvof': True})
    with app.app_context():
        with pytest.raises(nr.NeuralRenderError, match='nvngx_dlssnr.dll'):
            nr.start_dataset_render(app, LOCAL_USER, ds_id, ids, {})


# ── routes ──────────────────────────────────────────────────────────────────

def test_dataset_routes(app, client, tmp_path, monkeypatch):
    ds_id, ids, out = _dataset(app, tmp_path)
    _stub_render(monkeypatch)
    _run_job_inline(monkeypatch)
    assert client.get('/api/video-dataset/999/neural-render').status_code == 404
    r = client.get(f'/api/video-dataset/{ds_id}/neural-render')
    assert r.status_code == 200
    body = r.get_json()
    assert body['rendered_ids'] == [] and 'ready' in body['status'] and body['job'] is None
    r = client.post(f'/api/video-dataset/{ds_id}/neural-render', json={'ids': 'nope'})
    assert r.status_code == 400
    r = client.post(f'/api/video-dataset/{ds_id}/neural-render', json={'ids': [ids[0]], 'tone': 0.2})
    assert r.status_code == 200 and r.get_json()['queued'] == 1
    assert client.get(f'/api/video-dataset/{ds_id}/neural-render').get_json()['rendered_ids'] == [ids[0]]
    r = client.post(f'/api/video-dataset/{ds_id}/neural-render/restore', json={'ids': []})
    assert r.status_code == 200 and r.get_json()['restored'] == 1
    assert (out / 'clip_0001.mp4').read_bytes() == b'ORIGINAL clip_0001.mp4'
    r = client.post(f'/api/video-dataset/{ds_id}/neural-render', json={'ids': [], 'tone': 9})
    assert r.status_code == 400 and 'tone' in r.get_json()['error']


def test_the_capability_payload_carries_the_lane(app, monkeypatch):
    """Setup and both verbs receive the complete Neural status, including why
    the lane is unavailable, regardless of how the probes are scheduled."""
    from app import capabilities
    status = {'ready': False, 'missing': ['model missing'],
              'runtime_dir': 'test-runtime', 'model_file': nr.MODEL_FILE}
    monkeypatch.setattr(nr, 'status', lambda: status)
    monkeypatch.setattr(capabilities, '_http_ok', lambda *a, **k: False)
    monkeypatch.setattr(capabilities, '_import_ok', lambda *a, **k: False)
    monkeypatch.setattr(capabilities, 'gpu_vram_gb', lambda: None)
    monkeypatch.setattr(capabilities.ffmpeg_tools, 'ffmpeg_ready',
                        lambda: {'ok': False, 'reason': 'not installed'})
    with app.app_context():
        payload = capabilities.probe(force=True)
    assert payload['dlss5nr'] == status


# ── the child's stderr pump ─────────────────────────────────────────────────

def test_the_child_drains_ffmpeg_stderr_so_a_chatty_encoder_cannot_block():
    """A pipe nobody reads fills at 4 KB on Windows and the writer blocks; ffmpeg
    crosses that after ~35 s of libx264 output (measured). The pump keeps the
    tail and never lets the child stall — proven with a writer that emits far
    more than a pipe holds."""
    import importlib.util
    import subprocess
    import sys
    spec = importlib.util.spec_from_file_location(
        'dlss5nr_infer', str(nr.cfg.BACKEND_DIR / 'infer' / 'dlss5nr_infer.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    code = ("import sys" + chr(10)
            + "for i in range(4000): sys.stderr.write('line %d ' % i + 'x' * 60 + chr(10))" + chr(10)
            + "sys.stdout.write('ok')")
    proc = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    tail = []
    pump = mod.pump_stderr(proc.stderr, tail, limit=10)
    out = proc.stdout.read()
    assert proc.wait(timeout=30) == 0, 'the writer must finish — it would hang on an undrained pipe'
    pump.join(timeout=10)
    assert out == b'ok'
    assert len(tail) == 10 and tail[-1].startswith('line 3999')


def test_a_missing_ffmpeg_is_named_and_sent_to_setup(tmp_path):
    st = nr.status(_runtime(tmp_path), os_name='nt', driver={'ngx': True, 'nvof': True},
                   worker_ok=True, ffmpeg_ok=False)
    assert not st['ready'] and not st['ffmpeg']
    assert any('ffmpeg' in m and 'Setup' in m for m in st['missing'])


def test_the_childs_last_words_survive_os_exit():
    """os._exit skips every buffer flush; the protocol's lines only reach the
    parent because emit() flushes each one. Exercised with the real module:
    a child that emits, then fails through os._exit, is read whole."""
    import subprocess
    import sys
    script = str(nr.cfg.BACKEND_DIR / 'infer' / 'dlss5nr_infer.py')
    code = chr(10).join([
        'import importlib.util, sys',
        'spec = importlib.util.spec_from_file_location("w", sys.argv[1])',
        'm = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)',
        'm.emit("frame", done=3)',
        'm.fail("boom")',
    ])
    proc = subprocess.run([sys.executable, '-c', code, script], capture_output=True, text=True, timeout=60)
    lines = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    assert proc.returncode == 1
    assert lines == [{'event': 'frame', 'done': 3}, {'event': 'error', 'message': 'boom'}]


def test_every_additive_indexed_column_has_its_index_line(app):
    """The migration that ADDS a column never creates its index; only
    _INDEX_ADDITIONS does. The existing contract checks that list against the
    models one way; this is the other way, so the next index=True column added
    through _SCHEMA_ADDITIONS cannot ship index-less on every older database
    (vfi_of did, for a week)."""
    from app import _INDEX_ADDITIONS, _SCHEMA_ADDITIONS
    from app.extensions import db
    indexed = set(_INDEX_ADDITIONS)
    missing = []
    with app.app_context():
        for table, col, _type in _SCHEMA_ADDITIONS:
            model_table = db.metadata.tables.get(table)
            if model_table is None or col not in model_table.c:
                continue
            if model_table.c[col].index and (table, col) not in indexed:
                missing.append((table, col))
    assert not missing, f'index=True columns added by migration but never indexed on old databases: {missing}'


def test_backups_leave_with_their_clips_and_with_the_dataset(app, client, tmp_path, monkeypatch):
    ds_id, ids, out = _dataset(app, tmp_path)
    _stub_render(monkeypatch)
    _run_job_inline(monkeypatch)
    r = client.post(f'/api/video-dataset/{ds_id}/neural-render', json={'ids': []})
    assert r.status_code == 200
    root = nr.backup_dir(ds_id)
    # Each kept original travels with the record of the dials that replaced it.
    assert sorted(p.name for p in root.iterdir()) == ['clip_0001.mp4', 'clip_0001.mp4.nr.json',
                                                      'clip_0002.mp4', 'clip_0002.mp4.nr.json']
    r = client.post(f'/api/video-dataset/{ds_id}/clips/remove', json={'ids': [ids[0]]})
    assert r.status_code == 200 and r.get_json()['removed'] == 1
    assert sorted(p.name for p in root.iterdir()) == ['clip_0002.mp4', 'clip_0002.mp4.nr.json']
    assert client.delete(f'/api/video-dataset/{ds_id}').status_code == 200
    assert not root.exists()


def test_the_original_of_a_rendered_clip_is_served_and_absent_otherwise(app, client, tmp_path, monkeypatch):
    """The side-by-side player needs the bytes the render replaced. They exist
    only for a rendered clip (the backup IS the state), so a clip that plays
    its own original answers 404 rather than serving itself twice."""
    ds_id, ids, out = _dataset(app, tmp_path)
    _stub_render(monkeypatch)
    _run_job_inline(monkeypatch)
    assert client.get(f'/api/video-dataset/{ds_id}/clip/{ids[0]}/original').status_code == 404
    assert client.post(f'/api/video-dataset/{ds_id}/neural-render', json={'ids': [ids[0]]}).status_code == 200
    r = client.get(f'/api/video-dataset/{ds_id}/clip/{ids[0]}/original')
    assert r.status_code == 200 and r.data == b'ORIGINAL clip_0001.mp4'
    assert r.headers.get('Accept-Ranges') == 'bytes'
    assert client.get(f'/api/video-dataset/{ds_id}/clip/{ids[1]}/original').status_code == 404
    assert client.get(f'/api/video-dataset/999/clip/{ids[0]}/original').status_code == 404


# ── the levers the model does not expose ────────────────────────────────────

def test_strength_passes_and_scale_are_validated_and_carried():
    p = nr.normalize_params({'strength': 2.5, 'passes': '3', 'scale': 2})
    assert (p['strength'], p['passes'], p['scale']) == (2.5, 3, 2)
    for bad in ({'strength': 3.5}, {'strength': -1}, {'passes': 0}, {'passes': 4}, {'scale': 3}, {'scale': 'x'}):
        with pytest.raises(nr.NeuralRenderError):
            nr.normalize_params(bad)
    argv = nr.worker_argv('a.mp4', 'b.mp4', p, temporal_on=False, ffmpeg='ff.exe')
    for flag, value in (('--strength', '2.5'), ('--passes', '3'), ('--scale', '2')):
        assert argv[argv.index(flag) + 1] == value


def test_extra_passes_force_still_mode_and_refuse_an_explicit_temporal():
    assert nr.decide_temporal('auto', 1024, passes=2) == (False, 'still mode (extra passes)')
    assert nr.decide_temporal('auto', 1024, passes=1) == (True, 'temporal mode')
    with pytest.raises(nr.NeuralRenderError, match='exclude'):
        nr.decide_temporal('on', 1024, passes=3)


def test_the_child_scales_up_for_the_model_and_back_for_the_file():
    """A 2x render works on 2w x 2h and DELIVERS w x h: a dataset's clips keep
    the size the target profile gave them, whatever the working size."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'dlss5nr_infer', str(nr.cfg.BACKEND_DIR / 'infer' / 'dlss5nr_infer.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.decode_filter(1024, 576, 1024, 576) == 'null'
    assert mod.decode_filter(1024, 576, 2048, 1152) == 'scale=2048:1152:flags=lanczos'
    assert mod.encode_filter(1024, 576, 1024, 576) == []
    assert mod.encode_filter(2048, 1152, 1024, 576) == ['-vf', 'scale=1024:576:flags=lanczos']
    import numpy as np
    fin = np.full((2, 2, 3), 0.5, np.float32)
    fout = np.full((2, 2, 3), 0.6, np.float32)
    assert mod.apply_strength(np, fin, fout, 1.0) is fout
    assert float(mod.apply_strength(np, fin, fout, 2.0)[0, 0, 0]) == pytest.approx(0.7)
    assert float(mod.apply_strength(np, fin, fout, 0.0)[0, 0, 0]) == pytest.approx(0.5)
    assert float(mod.apply_strength(np, fin, np.full((2, 2, 3), 0.9, np.float32), 3.0)[0, 0, 0]) == 1.0


def test_a_dataset_render_keeps_its_dials_beside_the_original(app, client, tmp_path, monkeypatch):
    """The record lives next to the kept original, outside the dataset folder
    (a trainer must never find a .json there), is published with the state,
    and leaves with the backup on restore."""
    ds_id, ids, out = _dataset(app, tmp_path)
    _stub_render(monkeypatch)
    _run_job_inline(monkeypatch)
    r = client.post(f'/api/video-dataset/{ds_id}/neural-render', json={'ids': [ids[0]], 'strength': 1.5, 'temporal': 'off'})
    assert r.status_code == 200
    assert sorted(p.name for p in out.iterdir()) == ['clip_0001.mp4', 'clip_0001.txt', 'clip_0002.mp4', 'clip_0002.txt']
    state = client.get(f'/api/video-dataset/{ds_id}/neural-render').get_json()
    rec = state['rendered_params'][str(ids[0])]
    assert rec['strength'] == 1.5 and rec['temporal'] == 'off' and rec['temporal_used'] is False
    assert str(ids[1]) not in state['rendered_params']
    assert client.post(f'/api/video-dataset/{ds_id}/neural-render/restore', json={'ids': [ids[0]]}).status_code == 200
    assert not nr.sidecar_path(ds_id, 'clip_0001.mp4').exists()
    assert client.get(f'/api/video-dataset/{ds_id}/neural-render').get_json()['rendered_params'] == {}
