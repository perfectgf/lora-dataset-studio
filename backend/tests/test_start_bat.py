"""start.bat, read the way cmd runs it.

Issue #77, second report (v2026.11.11): `.\\start.bat` typed in a PowerShell 7 tab
went straight back to the prompt, with no message and no app. Reproduced with a
`python` on PATH that is a batch file, as pyenv-win installs it (shims\\python.bat):
cmd never comes back from a batch file started from another one without CALL, so
start.bat ended inside its own check of that python. Only the venv's python.exe is
known to be an executable; every other interpreter start.bat runs goes through CALL.
"""
import re
from pathlib import Path

START_BAT = Path(__file__).resolve().parents[2] / 'start.bat'
INTERPRETERS = {'py', 'python', 'python3', '%py%'}


def _code_lines():
    for line in START_BAT.read_text(encoding='utf-8').splitlines():
        code = line.strip()
        if code and not re.match(r'(?i)(rem\b|::)', code):
            yield code


def _commands():
    """(first word, text) of every command a line can start: quoted strings are
    blanked, then the line is cut wherever cmd may begin a new command."""
    for code in _code_lines():
        for part in re.split(r'\(|\)|&&|\|\||&', re.sub(r'"[^"]*"', '""', code)):
            words = part.split()
            if words:
                yield words[0].lower(), part.strip()


def test_every_interpreter_but_the_venv_python_is_started_with_call():
    commands = list(_commands())
    bare = [text for first, text in commands if first in INTERPRETERS]
    assert not bare, f'a .bat/.cmd shim started like this ends start.bat on the spot: {bare}'
    called = [text for first, text in commands
              if first == 'call' and text.split()[1].lower() in INTERPRETERS]
    assert len(called) >= 3, called      # py -3.x, python, %PY% -m venv


def test_every_failure_waits_for_a_key_before_the_window_closes():
    """A double-clicked start.bat runs in a window that closes when it ends: a
    failure that exits straight away is a window that flashes and vanishes."""
    lines = list(_code_lines())
    silent = [lines[i - 1] + ' / ' + line for i, line in enumerate(lines)
              if re.search(r'(?i)\bexit /b\b(?!\s+0\b)', line) and lines[i - 1].lower() != 'pause']
    assert not silent, silent
