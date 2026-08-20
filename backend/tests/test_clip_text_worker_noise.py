"""The CLIP text worker must survive a stdout that is not only its own.

Reported from a real install: 🎨 Medium died with "the text encoder produced no
result — check the ✨ Score interpreter" on a machine whose ✨ Score interpreter
was provably fine (the pass that produced the embeddings had just run). The
parent read ONE line of the child's stdout and called json.loads on it, so any
first-run banner — a weights download, an interpreter greeting, a dependency
announcing itself — became "no result". Text SEARCH takes the same road and was
broken by the same line.

These tests run a REAL subprocess (a stub interpreter, no model is ever loaded)
because the property under test lives in the pipe: a mocked encoder cannot print
a banner. What they pin:

  * a noisy worker still ENCODES — both through 🎨 Medium's prototype matrix and
    through the 🔎 text-search route;
  * when the worker really fails, the message QUOTES what it printed, on stdout
    and on stderr, instead of blaming a component it never looked at;
  * that quote is paste-safe: a home-dir path in the child's chatter is redacted
    before it reaches a message a user pastes into a public thread.
"""
import hashlib
import os
from pathlib import Path
import sys

import pytest
from PIL import Image

np = pytest.importorskip('numpy')


# --- stub interpreters ---------------------------------------------------------
# A worker that WORKS but talks first. Every line before the JSON is exactly the
# kind of thing a fresh ML environment prints on its first run.
CHATTY_WORKER = r'''
import json, sys

def vec(text):
    v = [0.0] * 768
    v[sum(ord(c) for c in text) % 768] = 1.0
    return v

raw = sys.stdin.readline()
print('[stub] first run: fetching open_clip ViT-L-14 weights')
print('Downloading (1.71G): 100%')
print(json.dumps({'ok': True, 'ready': True, 'dim': 768}))
sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    print('[stub] chatter before the answer')
    print(json.dumps({'ok': True, 'vector': vec(str(msg.get('text') or ''))}))
    sys.stdout.flush()
'''

# A worker that genuinely fails: it says why on stderr (which the app used to
# send to DEVNULL) and leaks a home-dir path on stdout (which must be redacted).
BROKEN_WORKER = r'''
import sys

raw = sys.stdin.readline()
print(r'[stub] cache dir C:\Users\someone\AppData\Local\open_clip')
print('[stub] no JSON is coming')
sys.stdout.flush()
sys.stderr.write('ModuleNotFoundError: No module named open_clip\n')
sys.stderr.flush()
sys.exit(1)
'''


def _install_worker(monkeypatch, tmp_path, source, name):
    """Point the encoder at a stub interpreter script run by THIS python.

    Nothing is faked inside the process: a real child, a real pipe, a real
    readline — the only thing replaced is the model."""
    from app import config as cfg
    from app.services import clip_text_encoder
    script = tmp_path / name
    script.write_text(source, encoding='utf-8')
    cfg.save_config({'bank_scoring': {'python': sys.executable}})
    monkeypatch.setattr(clip_text_encoder, '_SCRIPT', str(script))
    monkeypatch.setattr(clip_text_encoder, 'unavailable_reason', lambda: None)
    clip_text_encoder.release()      # never inherit a warm worker from a neighbour
    return clip_text_encoder


# --- bank fixtures (same shape as the text-search suite) -----------------------
def _mkbank(client, tmp_path, names, name='B'):
    src = tmp_path / 'src'
    for rel in names:
        os.makedirs(os.path.dirname(str(src / rel)), exist_ok=True)
        Image.new('RGB', (64, 64), (128, 128, 128)).save(str(src / rel))
    r = client.post('/api/bank/create', json={'name': name, 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


def _unit(idx):
    v = np.zeros(768, dtype='float32')
    v[idx] = 1.0
    return v


def _stub_index(text):
    """The coordinate CHATTY_WORKER puts its 1.0 on — so a test can build an
    image embedding that really is the nearest neighbour of that query."""
    return sum(ord(c) for c in text) % 768


def _write_score_cache(app, bank_id, embs_by_name):
    with app.app_context():
        from app.models import BankImage
        from app.services import image_bank_service as banks
        bank = banks.get_bank(_uid(), bank_id)
        rows = {os.path.basename(r.relpath): r
                for r in BankImage.query.filter_by(bank_id=bank_id).all()}
        paths, states, arr, sigs, hashes = [], [], [], [], []
        for nm, e in embs_by_name.items():
            p = banks.abs_image_path(bank, rows[nm])
            paths.append(p)
            states.append('ok')
            arr.append(np.asarray(e, dtype='float32'))
            st = os.stat(p)
            sigs.append(f'{st.st_size}:{st.st_mtime_ns}')
            hashes.append(np.frombuffer(
                hashlib.sha256(Path(p).read_bytes()).digest(), dtype='uint8'))
        cache_path = banks._score_cache_path(bank_id)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(cache_path),
            paths=np.array(paths), states=np.array(states),
            aes=np.array([float('nan')] * len(paths), dtype='float32'),
            nsfw=np.array([float('nan')] * len(paths), dtype='float32'),
            embs=np.stack(arr).astype('float32'), sigs=np.array(sigs),
            hashes=np.stack(hashes).astype('uint8'))


def _uid():
    from app.config import LOCAL_USER
    return LOCAL_USER


# --- the property: a chatty worker still encodes -------------------------------
def test_banner_before_the_json_still_encodes(app, tmp_path, monkeypatch):
    """A line of chatter ahead of every JSON answer — start AND query — and the
    encode still succeeds. This is the reported bug, at its narrowest."""
    with app.app_context():
        enc = _install_worker(monkeypatch, tmp_path, CHATTY_WORKER, 'chatty.py')
        vec, cached = enc.encode_query('a red car')
    assert cached is False
    assert vec.shape == (768,)
    # The vector is the child's, not a stand-in: it carries the stub's coordinate.
    assert int(np.argmax(vec)) == _stub_index('a red car')
    assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-5)


def test_borrowed_clip_worker_ignores_the_user_site(app, tmp_path, monkeypatch):
    """The readiness probe and the actual long-lived worker must use the same
    isolated Python. This prevents a user's global package from poisoning a
    healthy ComfyUI embedded environment only when the worker starts.

    BORROWED is the word that does the work, and it is not decoration. The
    stub above runs under THIS interpreter, which is the app's own — and there
    the flag is deliberately withheld, because on a system-Python install the
    user site-packages is where pip actually put torch (services/infer_env).
    So the real launch is exercised on our own Python, and the borrowed contract
    is asserted on the argv builder both halves go through."""
    import subprocess

    from app.services import infer_env
    seen = []
    real_popen = subprocess.Popen

    def capture(command, *args, **kwargs):
        seen.append(list(command))
        return real_popen(command, *args, **kwargs)

    with app.app_context():
        enc = _install_worker(monkeypatch, tmp_path, CHATTY_WORKER, 'isolated.py')
        monkeypatch.setattr(enc.subprocess, 'Popen', capture)
        vec, _ = enc.encode_query('isolated')
        enc.release()
        assert vec.shape == (768,)
        # Our own interpreter: launched as it stands, and it still answers.
        assert seen and seen[0][1] != '-s'
        # A borrowed one: the flag, on the same builder the worker uses.
        assert infer_env.worker_argv(
            r'C:\ComfyUI\python_embeded\python.exe',
            enc._SCRIPT)[1] == '-s'


def test_medium_prototypes_survive_a_chatty_worker(app, tmp_path, monkeypatch):
    """🎨 Medium's prototype matrix — the exact frame in the bug report — builds
    normally over a worker that prints a banner on every answer."""
    with app.app_context():
        _install_worker(monkeypatch, tmp_path, CHATTY_WORKER, 'chatty.py')
        from app.services import image_bank_service as banks
        names, P = banks._medium_prototype_matrix()
    assert names, 'the medium pass must have buckets to classify into'
    assert P.shape == (len(names), 768)
    assert np.isfinite(P).all()


def test_text_search_survives_a_chatty_worker(client, app, tmp_path, monkeypatch):
    """The OTHER surface fed by the same encoder: 🔎 text search returns a real
    ranking through a noisy worker, instead of an announced 503."""
    query = 'a red car'
    idx = _stub_index(query)
    bank_id, _ = _mkbank(client, tmp_path, ['hit.jpg', 'other.jpg'])
    _write_score_cache(app, bank_id, {'hit.jpg': _unit(idx),
                                      'other.jpg': _unit((idx + 7) % 768)})
    with app.app_context():
        _install_worker(monkeypatch, tmp_path, CHATTY_WORKER, 'chatty.py')
    r = client.post(f'/api/bank/{bank_id}/search-text', json={'query': query, 'n': 1})
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body['results'][0]['score'] == pytest.approx(1.0, abs=1e-3)
    with app.app_context():
        from app.models import BankImage
        best = BankImage.query.filter_by(id=body['image_ids'][0]).one()
    assert os.path.basename(best.relpath) == 'hit.jpg'


# --- the property: a real failure says WHAT it read ----------------------------
def _assert_says_what_it_read(msg):
    assert '[stub] no JSON is coming' in msg, msg
    # stderr used to go to DEVNULL — the cause was thrown away before anyone
    # could read it.
    assert 'No module named open_clip' in msg, msg
    # Paste-safe: the home-dir path the child printed is redacted, not echoed.
    assert 'someone' not in msg, msg
    assert '~' in msg, msg
    # And it no longer sends the user to a component it never looked at.
    assert 'Score interpreter' not in msg, msg


def test_failure_message_quotes_the_worker_on_the_medium_path(app, tmp_path, monkeypatch):
    with app.app_context():
        enc = _install_worker(monkeypatch, tmp_path, BROKEN_WORKER, 'broken.py')
        from app.services import image_bank_service as banks
        with pytest.raises(enc.TextEncodeError) as excinfo:
            banks._medium_prototype_matrix()
    _assert_says_what_it_read(str(excinfo.value))


def test_failure_message_quotes_the_worker_on_the_search_path(client, app, tmp_path,
                                                              monkeypatch):
    bank_id, _ = _mkbank(client, tmp_path, ['hit.jpg'])
    _write_score_cache(app, bank_id, {'hit.jpg': _unit(3)})
    with app.app_context():
        _install_worker(monkeypatch, tmp_path, BROKEN_WORKER, 'broken.py')
    r = client.post(f'/api/bank/{bank_id}/search-text', json={'query': 'a red car'})
    assert r.status_code == 503, r.get_json()
    _assert_says_what_it_read(r.get_json()['error'])
