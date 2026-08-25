"""A BROKEN interpreter path must fail the capability probe FAST, never hang.

The probes run whatever python the user configured or a fixture fabricated. On
Windows, CreateProcess on an invalid image (a stub, a truncated download, a
text file named python.exe) does not always return an error: the
invalid-16-bit-image path raises a MODAL MessageBox inside the PARENT's
CreateProcess and waits for a click that a server, a hidden console or a CI
runner can never deliver — and whether the dialog appears depends on the error
mode inherited from whoever launched the process. Caught live: the whole suite
frozen inside `_import_ok` over a 4-byte fake python.exe, native stack ending
MessageBoxW <- RaiseInvalid16BitExeError <- CreateProcessInternalW.
app/__init__ now sets SetErrorMode so the call fails with a code instead; this
test pins that with a wall-clock bound on exactly the file that froze it.
"""
import time

from app import capabilities


def test_a_fake_interpreter_fails_fast_instead_of_hanging(tmp_path):
    fake = tmp_path / 'python.exe'
    fake.write_text('fake')
    t0 = time.monotonic()
    ok = capabilities._import_ok(str(fake), 'import torch', timeout=30)
    elapsed = time.monotonic() - t0
    assert ok is False, 'a text file is not an importing interpreter'
    assert elapsed < 10, f'the broken-exe probe took {elapsed:.1f}s — a dialog is blocking again'
