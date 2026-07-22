"""Self-updater service: git-behind status + apply (pull/deps), and the packaged
release-ZIP update path (download mocked, but a REAL forged mini-ZIP is extracted
and swapped in a tmp tree). git and the network are fully mocked — no real pull,
no real download, no restart (schedule_restart is never called here)."""
import json
import zipfile
from pathlib import Path

from app.services import updater


class _R:
    def __init__(self, stdout='', stderr='', rc=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, rc


def _patch_git(monkeypatch, resp):
    monkeypatch.setattr(updater, 'is_git_checkout', lambda root=None: True)
    monkeypatch.setattr(updater, '_git', lambda root, *a, **k: resp(a))


def test_status_none_when_not_a_git_checkout(monkeypatch):
    monkeypatch.setattr(updater, 'is_git_checkout', lambda root=None: False)
    assert updater.git_update_status() is None


def test_status_reports_commits_behind(monkeypatch):
    def resp(a):
        if a[:2] == ('rev-parse', '--abbrev-ref'):
            return _R('main\n')
        if a[0] == 'fetch':
            return _R()
        if a[0] == 'rev-list':
            return _R('3\n')
        if a[0] == 'rev-parse' and a[-1] == 'HEAD':
            return _R('aaaaaaa\n')
        if a[0] == 'rev-parse':
            return _R('bbbbbbb\n')      # origin/main short sha
        return _R()
    _patch_git(monkeypatch, resp)
    s = updater.git_update_status()
    assert s['is_git'] and s['behind'] == 3 and s['update_available'] is True
    assert s['current_sha'] == 'aaaaaaa' and s['remote_sha'] == 'bbbbbbb'
    # commit links so the user can read what the pending update contains
    assert s['repo'] and s['repo'] in s['commits_url']
    assert s['compare_url'].endswith('/compare/aaaaaaa...bbbbbbb')


def test_status_no_compare_url_when_up_to_date(monkeypatch):
    def resp(a):
        if a[:2] == ('rev-parse', '--abbrev-ref'):
            return _R('main\n')
        if a[0] == 'fetch':
            return _R()
        if a[0] == 'rev-list':
            return _R('0\n')
        return _R('sha\n')
    _patch_git(monkeypatch, resp)
    s = updater.git_update_status()
    # up to date -> no incoming range to compare, but the history link stays
    assert 'compare_url' not in s
    assert s['commits_url'].endswith('/commits/main')


def test_status_up_to_date(monkeypatch):
    def resp(a):
        if a[:2] == ('rev-parse', '--abbrev-ref'):
            return _R('main\n')
        if a[0] == 'fetch':
            return _R()
        if a[0] == 'rev-list':
            return _R('0\n')
        return _R('sha\n')
    _patch_git(monkeypatch, resp)
    assert updater.git_update_status()['update_available'] is False


def test_apply_manual_when_not_git(monkeypatch):
    monkeypatch.setattr(updater, 'is_git_checkout', lambda root=None: False)
    r = updater.apply_update()
    assert r['ok'] is False and r['manual'] is True and 'releases' in r['url']


def test_apply_no_change(monkeypatch):
    def resp(a):
        if a[:2] == ('rev-parse', '--abbrev-ref'):
            return _R('main\n')
        if a[0] == 'rev-parse':
            return _R('samesha\n')      # before == after
        if a[0] == 'pull':
            return _R('Already up to date.\n')
        return _R()
    _patch_git(monkeypatch, resp)
    r = updater.apply_update()
    assert r['ok'] is True and r['changed'] is False


def test_apply_changed_no_deps(monkeypatch):
    state = {'pulled': False}

    def resp(a):
        if a[:2] == ('rev-parse', '--abbrev-ref'):
            return _R('main\n')
        if a[0] == 'rev-parse' and a[-1] == 'HEAD':
            return _R('bbbbbbb\n' if state['pulled'] else 'aaaaaaa\n')
        if a[0] == 'pull':
            state['pulled'] = True
            return _R('Updating aaaaaaa..bbbbbbb\n')
        if a[0] == 'diff':
            return _R('backend/app/services/foo.py\n')
        return _R()
    _patch_git(monkeypatch, resp)
    r = updater.apply_update()
    assert r['ok'] and r['changed'] is True and r['deps_changed'] is False


def test_apply_defers_requirements_install_until_restart(monkeypatch):
    state = {'pulled': False}

    def resp(a):
        if a[:2] == ('rev-parse', '--abbrev-ref'):
            return _R('main\n')
        if a[0] == 'rev-parse' and a[-1] == 'HEAD':
            return _R('bbbbbbb\n' if state['pulled'] else 'aaaaaaa\n')
        if a[0] == 'pull':
            state['pulled'] = True
            return _R('Updating\n')
        if a[0] == 'diff':
            return _R('backend/requirements.txt\nbackend/app/x.py\n')
        return _R()
    _patch_git(monkeypatch, resp)
    pip_calls = []
    monkeypatch.setattr(updater.subprocess, 'run',
                        lambda *a, **k: pip_calls.append(a[0]) or _R())
    r = updater.apply_update()
    assert r['changed'] is True and r['deps_changed'] is True
    assert pip_calls == []   # never mutate imported packages in the live Flask process


def test_restart_helper_installs_dependencies_after_port_is_free(tmp_path):
    command = updater._dependency_install_command(tmp_path, 'python-test')
    assert command == ['python-test', '-m', 'pip', 'install', '-q', '-r',
                       str(tmp_path / 'backend' / 'requirements.txt')]

    helper = updater._restart_helper_code(
        'python-test', 'run-test.py', str(tmp_path), 5050,
        install_requirements=True, root=tmp_path,
    )
    compile(helper, '<restart-helper>', 'exec')
    assert repr(command) in helper
    assert helper.index('s.bind') < helper.index('subprocess.run(dependency_command')
    assert helper.index('subprocess.run(dependency_command') < helper.index('subprocess.Popen')


def test_apply_pull_failure_is_reported_not_raised(monkeypatch):
    def resp(a):
        if a[:2] == ('rev-parse', '--abbrev-ref'):
            return _R('main\n')
        if a[0] == 'rev-parse':
            return _R('aaaaaaa\n')
        if a[0] == 'pull':
            return _R('', 'error: Your local changes would be overwritten', rc=1)
        return _R()
    _patch_git(monkeypatch, resp)
    r = updater.apply_update()
    assert r['ok'] is False and 'log' in r
    reason = r['reason'].lower()
    assert 'fast-forward' in reason and 'local changes' in reason
    # It must NEVER suggest re-cloning: by default the user's datasets, database
    # and config.json (API keys) live inside this very folder, ignored by git, so
    # "start over with a fresh clone" reads as "delete all your work".
    assert 'clone' not in reason
    assert 'git fetch origin' in reason        # the recovery that keeps them


# --- Release-ZIP update path (packaged install, no .git) ----------------------

class _FakeResp:
    """Minimal stand-in for requests.Response (json feed only)."""
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body or {}

    def json(self):
        return self._body


def _make_app_tree(root: Path, *, version_marker: str, requirements: str = 'flask\n'):
    """Write a tiny but structurally-real app tree under `root`."""
    (root / 'backend' / 'app').mkdir(parents=True)
    (root / 'backend' / 'run.py').write_text('# run\n', encoding='utf-8')
    (root / 'backend' / 'app' / '__init__.py').write_text(version_marker, encoding='utf-8')
    (root / 'backend' / 'requirements.txt').write_text(requirements, encoding='utf-8')
    (root / 'frontend' / 'dist').mkdir(parents=True)
    (root / 'frontend' / 'dist' / 'index.html').write_text(version_marker, encoding='utf-8')
    (root / 'start.bat').write_text(version_marker, encoding='utf-8')


def _forge_release_zip(tmp_path: Path, *, requirements: str = 'flask\n') -> Path:
    """A real ZIP shaped like the release artifact: everything wrapped in a single
    top-level folder, carrying NEW app content. Returns the zip path."""
    stage = tmp_path / 'stage' / 'LoRA-Dataset-Studio-windows'
    _make_app_tree(stage, version_marker='NEW', requirements=requirements)
    (stage / 'build_info.json').write_text('{"version":"9999.01.01"}', encoding='utf-8')
    zip_path = tmp_path / 'release.zip'
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for p in stage.rglob('*'):
            zf.write(p, p.relative_to(stage.parent))
    return zip_path


def test_latest_release_parses_version_and_prefers_windows_zip(monkeypatch):
    import requests
    body = {'tag_name': 'v9999.01.01', 'html_url': 'https://x/r',
            'assets': [
                {'name': 'notes.txt', 'browser_download_url': 'https://x/notes'},
                {'name': 'LoRA-Dataset-Studio-linux.zip', 'browser_download_url': 'https://x/lin'},
                {'name': 'LoRA-Dataset-Studio-windows.zip', 'browser_download_url': 'https://x/win'},
            ]}
    monkeypatch.setattr(requests, 'get', lambda *a, **k: _FakeResp(200, body))
    rel = updater.latest_release()
    assert rel['version'] == '9999.01.01' and rel['tag'] == 'v9999.01.01'
    assert rel['zip_url'] == 'https://x/win'      # windows asset wins over the linux one


def test_latest_release_degrades_offline(monkeypatch):
    import requests
    def boom(*a, **k):
        raise requests.ConnectionError('offline')
    monkeypatch.setattr(requests, 'get', boom)
    assert 'reason' in updater.latest_release()


def test_staged_app_root_handles_wrapped_and_flat(tmp_path):
    wrapped = tmp_path / 'w'
    _make_app_tree(wrapped / 'LoRA-Dataset-Studio', version_marker='x')
    assert updater._staged_app_root(wrapped).name == 'LoRA-Dataset-Studio'
    flat = tmp_path / 'f'
    _make_app_tree(flat, version_marker='x')
    assert updater._staged_app_root(flat) == flat


def test_plan_update_items_skips_protected_and_narrows_frontend(tmp_path):
    staged = tmp_path / 's'
    _make_app_tree(staged, version_marker='x')
    (staged / 'data').mkdir()            # protected — must be ignored even if present
    (staged / 'config.json').write_text('{}', encoding='utf-8')
    items = updater.plan_update_items(staged)
    assert 'backend' in items and 'frontend/dist' in items and 'start.bat' in items
    assert 'frontend' not in items       # narrowed to frontend/dist
    assert 'data' not in items and 'config.json' not in items


def test_requirements_changed(tmp_path):
    staged, live = tmp_path / 'staged', tmp_path / 'live'
    _make_app_tree(staged, version_marker='x', requirements='flask\nnumpy\n')
    _make_app_tree(live, version_marker='x', requirements='flask\n')
    assert updater._requirements_changed(staged, live) is True
    _make_app_tree(tmp_path / 'same2', version_marker='x', requirements='flask\n')
    assert updater._requirements_changed(tmp_path / 'same2', live) is False


def test_apply_release_files_swaps_and_backs_up(tmp_path):
    staged, repo, backup = tmp_path / 'staged', tmp_path / 'repo', tmp_path / 'bak'
    _make_app_tree(staged, version_marker='NEW')
    _make_app_tree(repo, version_marker='OLD')
    # user state that must survive untouched
    (repo / 'config.json').write_text('user-config', encoding='utf-8')
    (repo / 'data').mkdir(); (repo / 'data' / 'app.db').write_text('rows', encoding='utf-8')
    (repo / '.venv').mkdir(); (repo / '.venv' / 'python.exe').write_text('runtime', encoding='utf-8')

    swapped = updater.apply_release_files(staged, repo, backup)

    assert 'backend' in swapped and 'frontend/dist' in swapped
    assert (repo / 'backend' / 'run.py').read_text() == '# run\n'
    assert (repo / 'backend' / 'app' / '__init__.py').read_text() == 'NEW'   # replaced
    assert (backup / 'backend' / 'app' / '__init__.py').read_text() == 'OLD'  # old kept
    # preserved user state / runtime
    assert (repo / 'config.json').read_text() == 'user-config'
    assert (repo / 'data' / 'app.db').read_text() == 'rows'
    assert (repo / '.venv' / 'python.exe').read_text() == 'runtime'


def test_apply_release_files_rolls_back_on_failure(tmp_path, monkeypatch):
    staged, repo, backup = tmp_path / 'staged', tmp_path / 'repo', tmp_path / 'bak'
    _make_app_tree(staged, version_marker='NEW')
    _make_app_tree(repo, version_marker='OLD')

    # Fail the move of the SECOND item so the first is already switched in.
    real_move = updater.shutil.move
    calls = {'n': 0}

    def flaky_move(src, dst):
        # only count moves INTO the repo (new -> live), not the backup moves
        if Path(dst).parent == repo or Path(dst) == repo / 'frontend' / 'dist':
            calls['n'] += 1
            if calls['n'] == 2:
                raise OSError('simulated mid-swap failure')
        return real_move(src, dst)

    monkeypatch.setattr(updater.shutil, 'move', flaky_move)
    try:
        updater.apply_release_files(staged, repo, backup)
        assert False, 'expected the swap to raise'
    except OSError:
        pass
    # every original is back in place, byte for byte
    assert (repo / 'backend' / 'app' / '__init__.py').read_text() == 'OLD'
    assert (repo / 'frontend' / 'dist' / 'index.html').read_text() == 'OLD'
    assert (repo / 'start.bat').read_text() == 'OLD'


def _run_zip_update(tmp_path, monkeypatch, *, latest='9999.01.01', requirements='flask\n',
                    zip_requirements='flask\n'):
    """Drive apply_zip_update against a tmp install with the network mocked and a
    REAL forged ZIP extracted + swapped. Returns (result, repo_root)."""
    repo = tmp_path / 'install'
    _make_app_tree(repo, version_marker='OLD', requirements=requirements)
    (repo / 'config.json').write_text('mine', encoding='utf-8')
    monkeypatch.setenv('LDS_DATA_DIR', str(repo / 'data'))
    zip_path = _forge_release_zip(tmp_path, requirements=zip_requirements)
    monkeypatch.setattr(updater, 'latest_release',
                        lambda *a, **k: {'version': latest, 'tag': f'v{latest}',
                                         'zip_url': 'https://x/win', 'html_url': 'https://x/r'})
    monkeypatch.setattr(updater, '_download_file',
                        lambda url, dest, **k: updater.shutil.copyfile(zip_path, dest))
    # apply_zip_update chdir()s off the CWD before the swap; keep the test process put.
    chdirs = []
    monkeypatch.setattr(updater.os, 'chdir', lambda p: chdirs.append(p))
    res = updater.apply_zip_update(root=repo)
    return res, repo, chdirs


def test_apply_zip_update_downloads_extracts_and_swaps(tmp_path, monkeypatch):
    res, repo, chdirs = _run_zip_update(tmp_path, monkeypatch,
                                        zip_requirements='flask\ntorch\n')
    assert res['ok'] and res['changed'] and res['to'] == '9999.01.01'
    assert res['deps_changed'] is True         # requirements differ -> pip on restart
    assert chdirs == [str(repo)]               # moved off the CWD before swapping
    # new app content is live, user config preserved
    assert (repo / 'backend' / 'app' / '__init__.py').read_text() == 'NEW'
    assert (repo / 'frontend' / 'dist' / 'index.html').read_text() == 'NEW'
    assert (repo / 'config.json').read_text() == 'mine'
    # a backup of the old files exists for recovery
    assert (Path(res['backup']) / 'backend' / 'app' / '__init__.py').read_text() == 'OLD'


def test_apply_zip_update_no_op_when_not_newer(tmp_path, monkeypatch):
    res, repo, _ = _run_zip_update(tmp_path, monkeypatch, latest='2000.01.01')
    assert res['ok'] and res['changed'] is False
    assert (repo / 'backend' / 'app' / '__init__.py').read_text() == 'OLD'  # nothing swapped


def test_apply_zip_update_rolls_back_and_reports_on_swap_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(updater, 'apply_release_files',
                        lambda *a, **k: (_ for _ in ()).throw(OSError('disk full mid-swap')))
    res, repo, _ = _run_zip_update(tmp_path, monkeypatch)
    assert res['ok'] is False and 'rolled back' in res['reason']
    assert (repo / 'backend' / 'app' / '__init__.py').read_text() == 'OLD'


def test_download_file_reports_progress(tmp_path, monkeypatch):
    import requests

    class _Stream:
        headers = {'Content-Length': '6'}
        def raise_for_status(self): pass
        def iter_content(self, chunk_size=1): return [b'ab', b'cd', b'ef']
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(requests, 'get', lambda *a, **k: _Stream())
    seen = []
    dest = tmp_path / 'out.bin'
    updater._download_file('https://x/z', dest, on_progress=lambda d, t: seen.append((d, t)))
    assert dest.read_bytes() == b'abcdef'
    assert seen == [(2, 6), (4, 6), (6, 6)]      # cumulative bytes, known total


def test_apply_zip_update_emits_phases(tmp_path, monkeypatch):
    repo = tmp_path / 'install'
    _make_app_tree(repo, version_marker='OLD')
    monkeypatch.setenv('LDS_DATA_DIR', str(repo / 'data'))
    zip_path = _forge_release_zip(tmp_path)
    rel = {'version': '9999.01.01', 'zip_url': 'https://x/z', 'zip_size': 123}
    monkeypatch.setattr(updater.os, 'chdir', lambda p: None)

    def fake_download(url, dest, **kw):
        updater.shutil.copyfile(zip_path, dest)
        if kw.get('on_progress'):
            kw['on_progress'](123, 123)          # simulate the download reporting bytes
    monkeypatch.setattr(updater, '_download_file', fake_download)

    phases = []
    updater.apply_zip_update(root=repo, release=rel,
                             on_progress=lambda ph, d=0, t=0: phases.append(ph))
    assert phases[0] == 'downloading' and 'extracting' in phases and 'installing' in phases


def test_start_zip_update_no_op_when_not_newer(tmp_path, monkeypatch):
    monkeypatch.setattr(updater, 'latest_release',
                        lambda *a, **k: {'version': '2000.01.01', 'zip_url': 'https://x/z'})
    r = updater.start_zip_update(root=tmp_path)
    assert r['ok'] is True and r['changed'] is False    # decided inline, no thread


def test_start_zip_update_runs_worker_to_restarting(tmp_path, monkeypatch):
    import time
    repo = tmp_path / 'install'
    _make_app_tree(repo, version_marker='OLD')
    (repo / 'config.json').write_text('mine', encoding='utf-8')
    monkeypatch.setenv('LDS_DATA_DIR', str(repo / 'data'))
    zip_path = _forge_release_zip(tmp_path)
    monkeypatch.setattr(updater, 'latest_release',
                        lambda *a, **k: {'version': '9999.01.01', 'tag': 'v9999.01.01',
                                         'zip_url': 'https://x/win', 'zip_size': 99})
    monkeypatch.setattr(updater, '_download_file',
                        lambda url, dest, **k: updater.shutil.copyfile(zip_path, dest))
    monkeypatch.setattr(updater.os, 'chdir', lambda p: None)
    restarts = []
    monkeypatch.setattr(updater, 'schedule_restart', lambda *a, **k: restarts.append(k))

    r = updater.start_zip_update(root=repo)
    assert r['async'] is True and r['to'] == '9999.01.01' and r['total'] == 99

    for _ in range(100):                                # wait for the worker (<~5 s)
        if updater.zip_update_progress().get('phase') in ('restarting', 'error', 'done'):
            break
        time.sleep(0.05)
    prog = updater.zip_update_progress()
    assert prog['phase'] == 'restarting'
    assert (repo / 'backend' / 'app' / '__init__.py').read_text() == 'NEW'   # swapped
    assert (repo / 'config.json').read_text() == 'mine'                      # preserved
    assert restarts == [{'install_requirements': False}]


def test_apply_zip_update_rejects_archive_without_backend(tmp_path, monkeypatch):
    repo = tmp_path / 'install'
    _make_app_tree(repo, version_marker='OLD')
    monkeypatch.setenv('LDS_DATA_DIR', str(repo / 'data'))
    # a ZIP that does NOT contain backend/
    bogus = tmp_path / 'bogus.zip'
    with zipfile.ZipFile(bogus, 'w') as zf:
        zf.writestr('readme.txt', 'not an app')
    monkeypatch.setattr(updater, 'latest_release',
                        lambda *a, **k: {'version': '9999.01.01', 'zip_url': 'https://x/z'})
    monkeypatch.setattr(updater, '_download_file',
                        lambda url, dest, **k: updater.shutil.copyfile(bogus, dest))
    res = updater.apply_zip_update(root=repo)
    assert res['ok'] is False and 'missing backend' in res['reason']
    assert (repo / 'backend' / 'app' / '__init__.py').read_text() == 'OLD'
