"""Server-side folder selection for the Browse… field.

Both mechanisms operate on the machine that RUNS the server — the folder the app
triages lives there, not on the browser's machine — so a browser <input> can't
reach it:

  • open_native_folder_dialog(): pops the OS-native "choose a folder" dialog on
    the server's own desktop. Windows only, via a short PowerShell script running
    FolderBrowserDialog in its OWN -STA process — no message-pump conflict with
    the Flask worker thread. Returns the chosen path, None when the user
    cancelled, or raises NativePickerUnavailable when there is no desktop to draw
    on (headless box, Linux vast.ai instance, PowerShell missing, timeout). The UI
    silently falls back to the in-app browser then.

  • list_subfolders(): read-only enumeration of a directory's immediate
    SUBFOLDERS (never files — nothing sensitive is ever streamed) plus the drive
    list for the roots view. Backs the in-app folder browser used from the LAN /
    tablet / Linux, where a server-side native dialog makes no sense.
"""
import logging
import os
import shutil
import string
import subprocess
import tempfile

logger = logging.getLogger(__name__)

# Give the human time to actually browse and pick before we give up on the
# native dialog and let the UI fall back to the in-app browser.
NATIVE_DIALOG_TIMEOUT = 180


class NativePickerUnavailable(RuntimeError):
    """The server has no usable native folder dialog (not Windows, no desktop,
    PowerShell missing, or the dialog timed out)."""


# PowerShell that shows a folder dialog and writes ONLY the chosen path to stdout
# (empty when cancelled). Runs under -STA (FolderBrowserDialog requires a
# single-threaded apartment); a TopMost owner form pulls it above the console.
# The initial path arrives via the LDS_PICKER_INITIAL env var, NOT stdin: this
# script is launched with `-File`, and stdin under `-Command -` would be consumed
# as the command text (which also stops ShowDialog from ever blocking). Output is
# forced to UTF-8 so paths with accents survive the trip back to Python.
_PS_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Windows.Forms | Out-Null
$dlg = New-Object System.Windows.Forms.FolderBrowserDialog
$dlg.Description = 'LoRA Dataset Studio - choose a folder'
$dlg.ShowNewFolderButton = $true
$initial = $env:LDS_PICKER_INITIAL
if ($initial -and (Test-Path -LiteralPath $initial -PathType Container)) {
  $dlg.SelectedPath = $initial
}
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.ShowInTaskbar = $false
$owner.Opacity = 0
try {
  if ($dlg.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {
    [Console]::Out.Write($dlg.SelectedPath)
  }
} finally {
  $owner.Dispose()
}
"""


def native_dialog_available():
    """Cheap pre-check: Windows + a PowerShell on PATH. The real proof is only
    known once we try (there may be no interactive desktop), but this rejects the
    obvious non-Windows / vast.ai case without spawning a process."""
    return os.name == 'nt' and bool(_powershell_exe())


def _powershell_exe():
    # Windows PowerShell (powershell.exe) defaults to -STA; pwsh (7+) is MTA and
    # would need -STA passed anyway — we pass it explicitly below, so either works.
    return shutil.which('powershell') or shutil.which('pwsh')


def open_native_folder_dialog(initial=None):
    """Show the server-side native folder dialog. Returns the selected path, or
    None if the user cancelled. Raises NativePickerUnavailable when there is no
    native dialog to show (see module docstring)."""
    exe = _powershell_exe()
    if os.name != 'nt' or not exe:
        raise NativePickerUnavailable('native folder dialog is Windows-only')
    # Run the script from a temp -File (NOT `-Command -`): piping the script over
    # stdin makes PowerShell treat all of stdin as the command and ShowDialog
    # returns immediately without ever painting. The initial path rides in via an
    # env var; stderr is captured as bytes and decoded leniently so a localized
    # (non-UTF-8) PowerShell error never crashes the reader thread.
    env = dict(os.environ, LDS_PICKER_INITIAL=(initial or ''))
    tmp = tempfile.NamedTemporaryFile(
        'w', suffix='.ps1', delete=False, encoding='utf-8')
    try:
        tmp.write(_PS_SCRIPT)
        tmp.close()
        proc = subprocess.run(
            [exe, '-NoProfile', '-STA', '-ExecutionPolicy', 'Bypass',
             '-File', tmp.name],
            env=env, capture_output=True, timeout=NATIVE_DIALOG_TIMEOUT)
    except subprocess.TimeoutExpired as e:
        raise NativePickerUnavailable('the folder dialog timed out') from e
    except OSError as e:  # PowerShell vanished between the which() and the run
        raise NativePickerUnavailable(f'could not launch the dialog: {e}') from e
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    if proc.returncode != 0:
        # A non-zero exit here means the script itself blew up (e.g. no desktop /
        # WinForms unavailable on a service session) — treat as unavailable so the
        # UI falls back rather than surfacing a raw PowerShell trace.
        logger.info('native folder dialog unavailable: %s',
                    (proc.stderr or b'').decode('utf-8', 'replace').strip()[:200])
        raise NativePickerUnavailable('the native folder dialog is unavailable')
    path = (proc.stdout or b'').decode('utf-8', 'replace').strip()
    return path or None


def list_drives():
    """Root entries for the browser: drive letters on Windows, '/' on POSIX."""
    if os.name == 'nt':
        drives = []
        for letter in string.ascii_uppercase:
            root = f'{letter}:\\'
            if os.path.isdir(root):
                drives.append({'name': root, 'path': root})
        return drives
    return [{'name': '/', 'path': '/'}]


def list_subfolders(path=None):
    """Read-only listing for the in-app folder browser.

    path falsy  -> the roots view (drives / '/'), no parent.
    path given  -> that directory's immediate SUBFOLDERS only (never files),
                   with the parent for an "up" step. Path is normalized (abspath
                   collapses any '..'); unreadable entries are skipped, not fatal.
    """
    if not path or not str(path).strip():
        return {'path': None, 'parent': None, 'is_root': True,
                'entries': list_drives()}

    p = os.path.abspath(os.path.expanduser(str(path).strip()))
    if not os.path.isdir(p):
        raise ValueError('That folder does not exist.')

    entries = []
    with os.scandir(p) as it:
        for e in it:
            try:
                if e.is_dir():
                    entries.append({'name': e.name, 'path': e.path})
            except OSError:
                continue  # a junction/symlink we can't stat — skip, don't fail
    entries.sort(key=lambda d: d['name'].lower())

    parent = os.path.dirname(p)
    # At a filesystem root, dirname() is a no-op (returns p); surface the roots
    # view as the parent instead so "up" from C:\ lands on the drive list.
    at_root = parent == p
    return {'path': p, 'parent': None if at_root else parent,
            'is_root': False, 'entries': entries}
