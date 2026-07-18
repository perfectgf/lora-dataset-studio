"""🖥️ Server-side folder picker — native dialog (mocked) + read-only listing.

The native dialog spawns PowerShell on a real desktop, which we never want in a
test run, so open_native_folder_dialog is monkeypatched at the route boundary.
The listing endpoint is exercised for real against tmp_path directories."""
import os

import pytest

from app.services import folder_picker


# --- POST /api/system/pick-folder (native dialog, mocked) --------------------
def test_pick_folder_returns_chosen_path(client, monkeypatch):
    monkeypatch.setattr(folder_picker, 'open_native_folder_dialog',
                        lambda initial=None: r'D:\some\folder')
    r = client.post('/api/system/pick-folder', json={'initial': ''})
    assert r.status_code == 200
    assert r.get_json() == {'available': True, 'path': r'D:\some\folder'}


def test_pick_folder_cancelled(client, monkeypatch):
    monkeypatch.setattr(folder_picker, 'open_native_folder_dialog',
                        lambda initial=None: None)
    r = client.post('/api/system/pick-folder', json={})
    assert r.status_code == 200
    assert r.get_json() == {'available': True, 'cancelled': True}


def test_pick_folder_unavailable_is_200_not_error(client, monkeypatch):
    """A headless / Linux server answers 200 with available:false so the UI
    falls back to the in-app browser instead of showing an error toast."""
    def _boom(initial=None):
        raise folder_picker.NativePickerUnavailable('native folder dialog is Windows-only')
    monkeypatch.setattr(folder_picker, 'open_native_folder_dialog', _boom)
    r = client.post('/api/system/pick-folder', json={})
    assert r.status_code == 200
    body = r.get_json()
    assert body['available'] is False
    assert 'reason' in body


def test_pick_folder_forwards_initial(client, monkeypatch):
    seen = {}
    def _capture(initial=None):
        seen['initial'] = initial
        return None
    monkeypatch.setattr(folder_picker, 'open_native_folder_dialog', _capture)
    client.post('/api/system/pick-folder', json={'initial': '  C:\\pics  '})
    assert seen['initial'] == 'C:\\pics'  # route trims whitespace


# --- native dialog helper (no subprocess) ------------------------------------
def test_native_dialog_unavailable_off_windows(monkeypatch):
    monkeypatch.setattr(os, 'name', 'posix')
    with pytest.raises(folder_picker.NativePickerUnavailable):
        folder_picker.open_native_folder_dialog()


def test_native_dialog_available_precheck(monkeypatch):
    monkeypatch.setattr(os, 'name', 'posix')
    assert folder_picker.native_dialog_available() is False


class _FakeProc:
    def __init__(self, returncode=0, stdout=b'', stderr=b''):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_native_dialog_runs_file_not_stdin(monkeypatch, tmp_path):
    """The dialog must launch via a temp `-File` with the initial path in the
    environment — NOT `-Command -` with the script piped on stdin, which never
    actually opens the dialog. Locks in that regression."""
    monkeypatch.setattr(os, 'name', 'nt')
    monkeypatch.setattr(folder_picker, '_powershell_exe', lambda: 'powershell')
    seen = {}

    def fake_run(cmd, **kw):
        seen['cmd'] = cmd
        seen['env'] = kw.get('env') or {}
        # The temp .ps1 must still exist at launch time (deleted only after).
        seen['script_exists'] = os.path.isfile(cmd[-1])
        return _FakeProc(stdout='D:\\chosen'.encode('utf-8'))

    monkeypatch.setattr(folder_picker.subprocess, 'run', fake_run)
    out = folder_picker.open_native_folder_dialog(initial='C:\\pics')

    assert out == 'D:\\chosen'
    assert '-File' in seen['cmd'] and '-Command' not in seen['cmd']
    assert seen['cmd'][-1].endswith('.ps1')
    assert seen['env'].get('LDS_PICKER_INITIAL') == 'C:\\pics'
    assert seen['script_exists'] is True
    assert not os.path.isfile(seen['cmd'][-1])  # temp script cleaned up


def test_native_dialog_nonzero_exit_is_unavailable(monkeypatch):
    """A crashing script (no desktop) -> NativePickerUnavailable, and a localized
    non-UTF-8 stderr must not crash the decode."""
    monkeypatch.setattr(os, 'name', 'nt')
    monkeypatch.setattr(folder_picker, '_powershell_exe', lambda: 'powershell')
    monkeypatch.setattr(folder_picker.subprocess, 'run',
                        lambda cmd, **kw: _FakeProc(returncode=1, stderr=b'\xae bad'))
    with pytest.raises(folder_picker.NativePickerUnavailable):
        folder_picker.open_native_folder_dialog()


# --- GET /api/system/list-folders --------------------------------------------
def test_list_roots_when_no_path(client):
    r = client.get('/api/system/list-folders')
    assert r.status_code == 200
    body = r.get_json()
    assert body['is_root'] is True
    assert body['parent'] is None
    assert isinstance(body['entries'], list) and len(body['entries']) >= 1


def test_list_subfolders_lists_dirs_only(client, tmp_path):
    root = tmp_path / 'browse'
    (root / 'alpha').mkdir(parents=True)
    (root / 'Beta').mkdir()
    (root / 'a_file.txt').write_text('x')  # must NOT appear
    r = client.get('/api/system/list-folders', query_string={'path': str(root)})
    assert r.status_code == 200
    body = r.get_json()
    names = [e['name'] for e in body['entries']]
    assert names == ['alpha', 'Beta']  # dirs only, case-insensitive sort
    assert body['is_root'] is False
    assert body['parent'] == os.path.dirname(os.path.abspath(str(root)))


def test_list_subfolders_missing_path_400(client, tmp_path):
    r = client.get('/api/system/list-folders',
                   query_string={'path': str(tmp_path / 'nope')})
    assert r.status_code == 400
    assert 'error' in r.get_json()


def test_list_subfolders_collapses_traversal(client, tmp_path):
    """abspath() normalizes '..' so a nested '../..' path resolves cleanly to an
    existing directory rather than being served literally."""
    nested = tmp_path / 'x' / 'y'
    nested.mkdir(parents=True)
    weird = str(nested / '..' / '..')
    r = client.get('/api/system/list-folders', query_string={'path': weird})
    assert r.status_code == 200
    assert r.get_json()['path'] == os.path.abspath(str(tmp_path))
