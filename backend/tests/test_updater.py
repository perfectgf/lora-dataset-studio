"""Self-updater service: git-behind status + apply (pull/deps). git is fully mocked —
no network, no real pull, no restart (schedule_restart is never called here)."""
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


def test_apply_reinstalls_when_requirements_change(monkeypatch):
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
    assert pip_calls and 'pip' in ' '.join(pip_calls[0])   # pip install was invoked


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
    assert r['ok'] is False and 'failed' in r['reason'].lower() and 'log' in r
