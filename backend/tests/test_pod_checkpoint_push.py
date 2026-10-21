"""The resumable checkpoint push: slicing, skipping, refusing, assembling.

No pod is rented here. Two things ARE really exercised rather than mocked,
because they are the two that would fail silently in production:

* the streamed multipart body goes through a real HTTP server on loopback, so
  the framing (Content-Length, not chunked) and the received byte count are
  facts, not assumptions about what `requests` does with an iterable;
* the pod-side programs are executed as real subprocesses against a real
  directory, so "the program text is valid Python and its arithmetic is right"
  is proven rather than asserted. They are shipped as SOURCE TEXT to a machine
  nobody can attach a debugger to; a typo in them is not a thing to discover
  there.
"""

from public_dense_test_io import no_dense_provider_io  # noqa: F401
from email.parser import BytesParser
from email.policy import default
import http.server
import json
import os
import posixpath
import shlex
import subprocess
import sys
import threading

import pytest

from lds_cloud_training import dense_pod_hub as hub
from lds_cloud_training import pod_checkpoint_push as push
from lds_cloud_training.aitoolkit_remote import RemoteAiToolkit, RemoteError

pytestmark = pytest.mark.plugins('cloud_training')


# --- slicing arithmetic ---------------------------------------------------------

def test_slices_cover_the_file_exactly_and_do_not_overlap():
    total = 26 * 10 ** 9
    plan = push.plan_slices(total, 2 * 1024 ** 3)
    assert sum(s['length'] for s in plan) == total
    assert plan[0]['offset'] == 0
    for a, b in zip(plan, plan[1:]):
        assert a['offset'] + a['length'] == b['offset']
    assert [s['index'] for s in plan] == list(range(1, len(plan) + 1))


def test_an_empty_file_still_gets_one_slice():
    """Zero slices would make 'everything already landed' and 'there is nothing
    to send' the same state, and the assembly would create no file at all."""
    assert push.plan_slices(0) == [{'index': 1, 'offset': 0, 'length': 0}]


def test_slice_names_survive_the_route_filename_sanitiser():
    """The pod route rewrites anything outside [A-Za-z0-9._-]. A suffix that got
    rewritten would land every slice under one mangled name."""
    name = push.slice_name('Krea_000001500.safetensors', 7)
    assert name == 'Krea_000001500.safetensors.p0007'
    assert all(c.isalnum() or c in '._-' for c in name)


def test_the_command_length_does_not_grow_with_the_file():
    """The slice LIST is never sent — the program derives names from a count.
    A 26 GB transfer and a 260 GB one must produce the same command length, or
    the vast command ceiling becomes a size limit nobody documented."""
    small = push.build_assemble_command('/s', '/d/x.safetensors', 'x.safetensors',
                                        13, 26 * 10 ** 9)
    big = push.build_assemble_command('/s', '/d/x.safetensors', 'x.safetensors',
                                      130, 260 * 10 ** 9)
    assert abs(len(big) - len(small)) < 20
    assert len(big) < push.MAX_COMMAND_CHARS


def test_a_job_name_carrying_quotes_cannot_become_code():
    paths = push.pod_paths('/workspace/datasets', "run'; rm -rf /; '")
    assert "'" not in paths['staging']
    cmd = push.build_probe_command(paths['staging'], '/t/j/k.safetensors')
    # Everything user-influenced rides inside one quoted argv entry.
    assert cmd.count("'") % 2 == 0
    assert 'rm -rf' not in cmd


# --- the disk refusal and its arithmetic ----------------------------------------

def test_disk_refusal_budgets_the_file_plus_ONE_slice_not_the_file_twice():
    """The assembly deletes each slice as it appends it, so the peak is the
    finished file plus one slice. Budgeting 2x would refuse pods that fit."""
    total = 26 * 10 ** 9
    short = push.disk_shortfall(30 * 10 ** 9, total, 2 * 1024 ** 3)
    assert short is None
    assert push.disk_shortfall(2 * total, total, 2 * 1024 ** 3) is None


def test_disk_refusal_shows_its_arithmetic():
    short = push.disk_shortfall(20 * 10 ** 9, 26 * 10 ** 9, 2 * 1024 ** 3)
    assert short['need_bytes'] == 26 * 10 ** 9 + 2 * 1024 ** 3
    assert short['short_bytes'] == short['need_bytes'] - short['free_bytes']


def test_slices_already_on_the_pod_are_not_asked_for_again():
    total = 26 * 10 ** 9
    tight = push.disk_shortfall(6 * 10 ** 9, total, 2 * 1024 ** 3,
                                already_bytes=24 * 10 ** 9)
    assert tight is None


def test_unmeasurable_free_space_never_blocks():
    """Same rule the local-disk forecast already follows: a volume we could not
    measure is not evidence of a shortage."""
    assert push.disk_shortfall(-1, 26 * 10 ** 9) is None


# --- the pod-side programs, really run ------------------------------------------

def _run_program(program, *args):
    r = subprocess.run([sys.executable, '-c', program, *[str(a) for a in args]],
                       capture_output=True, text=True, timeout=120)
    parsed = hub.parse_result(r.stdout)
    assert parsed is not None, f'no result line: {r.stdout}\n{r.stderr}'
    return parsed


def _seed(staging, name, blobs):
    os.makedirs(staging, exist_ok=True)
    for i, blob in enumerate(blobs, start=1):
        with open(os.path.join(staging, push.slice_name(name, i)), 'wb') as fh:
            fh.write(blob)


def test_assembly_rebuilds_the_file_byte_for_byte_and_clears_the_slices(tmp_path):
    payload = os.urandom(9000)
    blobs = [payload[i:i + 1000] for i in range(0, 9000, 1000)]
    staging = str(tmp_path / 'staging')
    dest = str(tmp_path / 'job' / 'model.safetensors')
    _seed(staging, 'model.safetensors', blobs)

    out = _run_program(push.ASSEMBLE_PROGRAM, staging, dest, 'model.safetensors',
                       len(blobs), len(payload))
    assert out['ok'] is True and out['bytes'] == len(payload)
    with open(dest, 'rb') as fh:
        assert fh.read() == payload
    # Slices AND the manifest are gone: 26 GB of debris on a pod that is about
    # to train is the difference between fitting and not fitting.
    assert not os.path.isdir(staging)
    assert not os.path.exists(dest + push.MANIFEST_SUFFIX)


def test_an_interrupted_assembly_resumes_at_the_slice_it_stopped_on(tmp_path):
    payload = os.urandom(4000)
    blobs = [payload[i:i + 1000] for i in range(0, 4000, 1000)]
    staging = str(tmp_path / 'staging')
    dest = str(tmp_path / 'job' / 'model.safetensors')
    _seed(staging, 'model.safetensors', blobs)

    # Simulate a run that appended two slices and died: the first two are
    # consumed, the manifest says so, the destination holds their bytes.
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, 'wb') as fh:
        fh.write(payload[:2000])
    with open(dest + push.MANIFEST_SUFFIX, 'w') as fh:
        json.dump({'done': 2}, fh)
    for i in (1, 2):
        os.remove(os.path.join(staging, push.slice_name('model.safetensors', i)))

    out = _run_program(push.ASSEMBLE_PROGRAM, staging, dest, 'model.safetensors',
                       4, len(payload))
    assert out['ok'] is True
    assert out['appended'] == 2, 'only the remaining slices should be appended'
    with open(dest, 'rb') as fh:
        assert fh.read() == payload


def test_a_manifest_that_disagrees_with_the_file_restarts_instead_of_appending(tmp_path):
    """The manifest is a claim; the file on disk is the fact. Appending to a
    file of unknown length is how a checkpoint becomes garbage that still
    loads, so a disagreement must restart, not continue."""
    payload = os.urandom(4000)
    blobs = [payload[i:i + 1000] for i in range(0, 4000, 1000)]
    staging = str(tmp_path / 'staging')
    dest = str(tmp_path / 'job' / 'model.safetensors')
    _seed(staging, 'model.safetensors', blobs)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, 'wb') as fh:
        fh.write(b'\x00' * 137)                     # a truncated earlier attempt
    with open(dest + push.MANIFEST_SUFFIX, 'w') as fh:
        json.dump({'done': 2}, fh)                  # claims two slices landed

    out = _run_program(push.ASSEMBLE_PROGRAM, staging, dest, 'model.safetensors',
                       4, len(payload))
    assert out['ok'] is True
    with open(dest, 'rb') as fh:
        assert fh.read() == payload


def test_assembly_refuses_when_the_slices_do_not_add_up(tmp_path):
    payload = os.urandom(3000)
    blobs = [payload[i:i + 1000] for i in range(0, 3000, 1000)]
    blobs[1] = blobs[1][:400]                       # one truncated slice
    staging = str(tmp_path / 'staging')
    dest = str(tmp_path / 'job' / 'model.safetensors')
    _seed(staging, 'model.safetensors', blobs)

    out = _run_program(push.ASSEMBLE_PROGRAM, staging, dest, 'model.safetensors',
                       3, len(payload))
    assert out['ok'] is False
    assert 'expected' in (out['error'] or '')
    assert not os.path.exists(dest), 'a refused assembly must land nothing'


def test_assembly_names_the_slices_that_are_missing(tmp_path):
    staging = str(tmp_path / 'staging')
    dest = str(tmp_path / 'job' / 'model.safetensors')
    _seed(staging, 'model.safetensors', [b'a' * 10, b'b' * 10])

    out = _run_program(push.ASSEMBLE_PROGRAM, staging, dest, 'model.safetensors',
                       3, 30)
    assert out['ok'] is False
    assert 'model.safetensors.p0003' in (out['error'] or '')


def test_the_probe_reports_slices_destination_and_free_space(tmp_path):
    staging = str(tmp_path / 'staging')
    dest = str(tmp_path / 'job' / 'model.safetensors')
    _seed(staging, 'model.safetensors', [b'a' * 10, b'b' * 20])
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, 'wb') as fh:
        fh.write(b'z' * 7)

    out = _run_program(push.PROBE_PROGRAM, staging, dest)
    assert out['ok'] is True
    assert out['slices'] == {'model.safetensors.p0001': 10,
                             'model.safetensors.p0002': 20}
    assert out['dest_bytes'] == 7
    assert out['free_bytes'] > 0


def test_the_probe_on_a_pod_that_has_nothing_yet(tmp_path):
    out = _run_program(push.PROBE_PROGRAM, str(tmp_path / 'nope'),
                       str(tmp_path / 'job' / 'model.safetensors'))
    assert out['ok'] is True
    assert out['slices'] == {}
    assert out['dest_bytes'] == -1
    assert out['free_bytes'] > 0, ('free space is probed on the nearest EXISTING '
                                   'parent, so a destination directory that does '
                                   'not exist yet still yields a number')


# --- the streamed body, over a real socket --------------------------------------

class _Recorder(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def do_POST(self):
        n = int(self.headers.get('Content-Length') or 0)
        body = b''
        while len(body) < n:
            block = self.rfile.read(min(1 << 16, n - len(body)))
            if not block:
                break
            body += block
        self.server.seen.append({'headers': dict(self.headers), 'body': body})
        parts = BytesParser(policy=default).parsebytes(
            b'Content-Type: ' + self.headers['Content-Type'].encode()
            + b'\r\n\r\n' + body)
        fields = {p.get_param('name', header='content-disposition'): p
                  for p in parts.iter_parts()}
        dataset = fields['datasetName'].get_payload(decode=True).decode()
        destination = posixpath.normpath(posixpath.join('/datasets', dataset))
        # Mirror the current upstream route, not the old path.join loophole.
        if posixpath.dirname(destination) != '/datasets' or destination == '/datasets':
            payload = b'{"error":"Invalid dataset name"}'
            status = 400
        else:
            payload = json.dumps(self.server.answer).encode()
            status = self.server.status
            root = getattr(self.server, 'dataset_root', None)
            if root is not None and status == 200:
                folder = root / posixpath.basename(destination)
                folder.mkdir(parents=True, exist_ok=True)
                part = fields['files']
                (folder / part.get_filename()).write_bytes(part.get_payload(decode=True))
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


@pytest.fixture
def pod_route():
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _Recorder)
    srv.seen, srv.answer, srv.status = [], {'ok': True}, 200
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield srv
    finally:
        srv.shutdown()


def _remote(srv):
    return RemoteAiToolkit(f'http://127.0.0.1:{srv.server_address[1]}', 'tok')


def test_a_slice_is_framed_by_content_length_never_chunked(pod_route, tmp_path):
    """A declared length is the shape ai-toolkit's `request.formData()` is known
    to accept. Falling back to Transfer-Encoding: chunked would be a silent bet
    on a parser we do not control."""
    path = tmp_path / 'f.bin'
    path.write_bytes(os.urandom(300_000))
    _remote(pod_route).upload_file_slice('/w/datasets', '/w/datasets/_lds_push_j',
                                         'f.bin.p0001', str(path))
    head = pod_route.seen[0]['headers']
    assert head.get('Transfer-Encoding') is None
    assert int(head['Content-Length']) == len(pod_route.seen[0]['body'])


def test_a_slice_sends_exactly_its_own_bytes_and_the_right_destination(pod_route,
                                                                       tmp_path):
    payload = os.urandom(50_000)
    path = tmp_path / 'f.bin'
    path.write_bytes(payload)
    sent = _remote(pod_route).upload_file_slice(
        '/w/datasets', '/w/datasets/_lds_push_job', 'f.bin.p0002', str(path),
        offset=20_000, length=10_000)
    assert sent == 10_000
    body = pod_route.seen[0]['body']
    assert payload[20_000:30_000] in body
    assert payload[:20_000] not in body
    assert b'name="datasetName"' in body and b'_lds_push_job' in body
    assert b'filename="f.bin.p0002"' in body


def test_a_slice_running_past_the_end_of_the_file_is_clamped(pod_route, tmp_path):
    path = tmp_path / 'f.bin'
    path.write_bytes(b'x' * 1000)
    sent = _remote(pod_route).upload_file_slice(
        '/w/d', '/w/d/s', 'f.bin.p0001', str(path), offset=900, length=5000)
    assert sent == 100


def test_a_non_200_from_the_route_is_a_loud_failure(pod_route, tmp_path):
    pod_route.status = 507
    path = tmp_path / 'f.bin'
    path.write_bytes(b'x' * 100)
    with pytest.raises(RemoteError):
        _remote(pod_route).upload_file_slice('/w/d', '/w/d/s', 'f.p0001', str(path))


def test_seed_checkpoint_still_sends_the_whole_file(pod_route, tmp_path):
    """Whole-file uploads of helpers and tokens retain the streamed body."""
    payload = os.urandom(40_000)
    path = tmp_path / 'lora.safetensors'
    path.write_bytes(payload)
    _remote(pod_route).seed_checkpoint('/w/datasets', '/w/datasets/_lds_hub',
                                       'job_000000500.safetensors', str(path))
    body = pod_route.seen[0]['body']
    assert payload in body
    assert b'filename="job_000000500.safetensors"' in body
    assert b'_lds_hub' in body


@pytest.mark.parametrize('dest', [
    '/w/training/job', '/w/datasets/_lds_push/job', '/w/datasets',
    '/w/datasets/../training/job',
])
def test_upload_rejects_destinations_outside_a_direct_child(pod_route, tmp_path, dest):
    path = tmp_path / 'checkpoint.safetensors'
    path.write_bytes(b'weights')
    with pytest.raises(RemoteError, match='direct child'):
        _remote(pod_route).upload_file_slice('/w/datasets', dest, path.name, str(path))
    assert pod_route.seen == []


def test_checkpoint_staging_is_an_allowed_dataset_directory():
    paths = push.pod_paths('/w/datasets', 'lds9_lora')
    assert posixpath.dirname(paths['staging']) == paths['root']


@pytest.mark.parametrize('slice_bytes,expected_slices', [(10_000, 1), (1000, 3)])
def test_checkpoint_reaches_training_folder_through_strict_upload_route(
        pod_route, tmp_path, monkeypatch, slice_bytes, expected_slices):
    """Real multipart HTTP plus the real pod programs, with disposable files."""
    datasets = tmp_path / 'datasets'
    pod_route.dataset_root = datasets
    payload = os.urandom(2500)
    source = tmp_path / 'source.safetensors'
    source.write_bytes(payload)
    output = tmp_path / 'output' / 'new_job'
    remote_name = 'new_job_000002000.safetensors'

    def run_on_test_pod(remote, *, command, **kwargs):
        args = shlex.split(command(None))
        assert args[:2] == ['python', '-c']
        return _run_program(args[2], *args[3:])

    monkeypatch.setattr(hub, 'run_program', run_on_test_pod)
    result = push.push_checkpoint(
        _remote(pod_route), instance_id='test-pod', local_path=str(source),
        dest_dir=output.as_posix(), remote_name=remote_name,
        datasets_folder=datasets.as_posix(), job_name='new_job',
        tmp_dir=str(tmp_path), slice_bytes=slice_bytes)
    assert result['sent_slices'] == expected_slices
    assert len(pod_route.seen) == expected_slices
    assert (output / remote_name).read_bytes() == payload


def test_a_body_refuses_to_be_sent_twice(tmp_path):
    """Iterating again would send a SHORT body under a full Content-Length — a
    redirect or a replaying retry would corrupt the slice silently."""
    from lds_cloud_training.aitoolkit_remote import _StreamedPart
    path = tmp_path / 'f.bin'
    path.write_bytes(b'x' * 100)
    part = _StreamedPart('files', 'f.bin', str(path))
    assert b''.join(part)
    with pytest.raises(RemoteError):
        b''.join(part)


def test_a_file_that_shrinks_mid_send_fails_instead_of_padding(tmp_path):
    """Padding to Content-Length would land a file of the right SIZE and the
    wrong BYTES, which ai-toolkit's auto-resume would happily train from."""
    from lds_cloud_training.aitoolkit_remote import _StreamedPart
    path = tmp_path / 'f.bin'
    path.write_bytes(b'x' * 100_000)
    part = _StreamedPart('files', 'f.bin', str(path))
    it = iter(part)
    next(it)                                    # the multipart preamble
    path.write_bytes(b'x' * 10)                 # truncated under us
    with pytest.raises(RemoteError, match='shrank'):
        list(it)


# --- the push itself: what gets skipped, sent and refused -----------------------

class _FakeRemote:
    """The two seams push_checkpoint uses, and nothing else."""

    def __init__(self, staging_state, free_bytes=10 ** 12, dest_bytes=-1):
        self.sent = []
        self.state = dict(staging_state)
        self.free_bytes = free_bytes
        self.dest_bytes = dest_bytes
        self.fail_slices = set()

    def get_settings(self):
        return {'DATASETS_FOLDER': '/w/datasets', 'TRAINING_FOLDER': '/w/training'}

    def upload_file_slice(self, root, dest, name, path, *, offset=0, length=None,
                          **kw):
        if name in self.fail_slices:
            raise RemoteError('proxy cut the stream')
        self.sent.append({'name': name, 'offset': offset, 'length': length})
        self.state[name] = length


def _push(remote, monkeypatch, tmp_path, size, **kw):
    src = tmp_path / 'master.safetensors'
    src.write_bytes(b'm' * size)
    results = []

    def fake_run(_remote, *, command, **_kw):
        text = command(None) if callable(command) else command
        if 'shutil' in text:            # the probe
            return {'ok': True, 'slices': dict(remote.state),
                    'dest_bytes': remote.dest_bytes,
                    'free_bytes': remote.free_bytes, 'assembled': 0}
        results.append(text)            # the assembly
        return {'ok': True, 'bytes': size, 'appended': len(remote.state)}

    monkeypatch.setattr(hub, 'run_program', fake_run)
    out = push.push_checkpoint(
        remote, instance_id='1', local_path=str(src),
        dest_dir='/w/training/job', remote_name='master.safetensors',
        datasets_folder='/w/datasets', job_name='job', tmp_dir=str(tmp_path),
        slice_bytes=1000, **kw)
    return out, results


def test_a_fresh_push_sends_every_slice(monkeypatch, tmp_path):
    remote = _FakeRemote({})
    out, _ = _push(remote, monkeypatch, tmp_path, 3500)
    assert out['slices'] == 4 and out['sent_slices'] == 4
    assert [s['offset'] for s in remote.sent] == [0, 1000, 2000, 3000]
    assert [s['length'] for s in remote.sent] == [1000, 1000, 1000, 500]


def test_a_resumed_push_skips_what_already_landed(monkeypatch, tmp_path):
    """The whole point. An app restart mid-transfer must not re-send 20 GB."""
    remote = _FakeRemote({'master.safetensors.p0001': 1000,
                          'master.safetensors.p0002': 1000})
    out, _ = _push(remote, monkeypatch, tmp_path, 3500)
    assert out['skipped_slices'] == 2 and out['sent_slices'] == 2
    assert [s['name'] for s in remote.sent] == ['master.safetensors.p0003',
                                                'master.safetensors.p0004']


def test_a_slice_of_the_WRONG_size_is_resent_not_trusted(monkeypatch, tmp_path):
    """A short slice is a cut transfer and a long one belongs to another file.
    Both would assemble into a checkpoint of the right length and wrong bytes."""
    remote = _FakeRemote({'master.safetensors.p0001': 999,     # truncated
                          'master.safetensors.p0002': 4000})   # foreign
    out, _ = _push(remote, monkeypatch, tmp_path, 3500)
    assert out['sent_slices'] == 4


def test_progress_counts_only_slices_that_LANDED(monkeypatch, tmp_path):
    """Bytes that reached the wire but not the pod's disk are not progress, and
    a freeze watchdog that counts them keeps a dead transfer alive."""
    remote = _FakeRemote({'master.safetensors.p0001': 1000})
    seen = []
    _push(remote, monkeypatch, tmp_path, 3000,
          on_progress=lambda done, total: seen.append((done, total)))
    assert seen[0] == (1000, 3000), 'the already-landed slice counts immediately'
    assert seen[-1] == (3000, 3000)
    assert seen == sorted(seen), 'a byte counter must never go backwards'


def test_a_push_onto_a_pod_that_already_has_the_whole_file_sends_nothing(
        monkeypatch, tmp_path):
    remote = _FakeRemote({}, dest_bytes=3000)
    out, assemblies = _push(remote, monkeypatch, tmp_path, 3000)
    assert out['reused'] is True and remote.sent == [] and assemblies == []


def test_a_full_pod_is_refused_with_its_arithmetic_before_a_byte_is_sent(
        monkeypatch, tmp_path):
    remote = _FakeRemote({}, free_bytes=2500)
    with pytest.raises(push.PodPushError) as err:
        _push(remote, monkeypatch, tmp_path, 3000)
    assert remote.sent == [], 'the refusal must come before the transfer'
    for fragment in ('free', 'Short by', 'Hugging Face'):
        assert fragment in str(err.value)


def test_a_slice_that_will_not_go_through_keeps_what_already_did(monkeypatch,
                                                                tmp_path):
    remote = _FakeRemote({})
    remote.fail_slices = {'master.safetensors.p0002'}
    with pytest.raises(push.PodPushError) as err:
        _push(remote, monkeypatch, tmp_path, 3000)
    assert 'stays on the pod' in str(err.value)
    assert 'resumes' in str(err.value) or 'picks up' in str(err.value)
    # The first slice was sent and is deliberately NOT cleaned up.
    assert remote.sent[0]['name'] == 'master.safetensors.p0001'


def test_a_transient_slice_failure_is_retried_rather_than_fatal(monkeypatch,
                                                               tmp_path):
    """Some vast hosts' proxies cut long streams; the download side needs up to
    400 resumed connections for the same reason."""
    remote = _FakeRemote({})
    calls = {'n': 0}
    real = remote.upload_file_slice

    def flaky(*a, **kw):
        calls['n'] += 1
        if calls['n'] == 1:
            raise RemoteError('proxy cut the stream')
        return real(*a, **kw)

    remote.upload_file_slice = flaky
    out, _ = _push(remote, monkeypatch, tmp_path, 1500)
    assert out['sent_slices'] == 2


def test_the_assembly_is_told_the_exact_count_and_total(monkeypatch, tmp_path):
    remote = _FakeRemote({})
    _out, assemblies = _push(remote, monkeypatch, tmp_path, 3500)
    assert len(assemblies) == 1
    assert " '4' " in assemblies[0] and "'3500'" in assemblies[0]
