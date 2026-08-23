"""backend/extensions/ is a local-only drop-in dir. Three independent locks
keep it out of anything published: .gitignore (repo), robocopy /XD (bundle
build), check_release_artifacts (final ZIP). Each lock gets its own test so a
regression names the layer that broke.
"""
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_backend_extensions_is_gitignored():
    proc = subprocess.run(
        ['git', 'check-ignore', '-q', 'backend/extensions/anything/__init__.py'],
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, 'backend/extensions/ must be gitignored'


def test_release_zip_build_excludes_the_extensions_dir():
    text = (REPO_ROOT / 'packaging' / 'build_release_zip.ps1').read_text(encoding='utf-8')
    xd_lines = [l for l in text.splitlines() if '/XD' in l]
    assert xd_lines, 'expected a robocopy /XD exclusion list'
    assert any(re.search(r'/XD.*\bextensions\b', l) for l in xd_lines), \
        'robocopy must exclude backend/extensions from the release bundle'
