"""Enforce what a published release may contain: no EXE, and no personal data.

With no argument, this checks every workflow that can publish a GitHub release.
Paths passed on the command line are inspected too; ZIP members are read
without extracting them. The release workflow and CI both invoke this script.

WHY THE ARCHIVE AND NOT THE REPO. `backend/tests/test_no_personal_data.py`
guards the sources, but it walks `git ls-files` and skips `frontend/dist/`, so
it is blind to three things at once: untracked files, ignored files, and the
compiled bundle everybody actually downloads. That blindness has already cost
something — a published release carried `backend/.pytest_cache/`, because the
packaging script excluded `tests` but not the cache directory beside it.

Reading the final ZIP closes all three holes in one pass, because the ZIP is
the artifact, not a proxy for it. The patterns are imported from
`scripts/privacy_patterns.py`, never copied: two tables that drift apart are
worse than one table.
"""

from __future__ import annotations

import os
import re
import sys
import zipfile
from fnmatch import fnmatch
from pathlib import Path

if __package__:
    from . import privacy_patterns as privacy
else:                                     # `python scripts/check_release_artifacts.py`
    import privacy_patterns as privacy


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
PUBLISH_MARKERS = (
    "gh release create",
    "gh release upload",
    "softprops/action-gh-release",
    "actions/upload-release-asset",
    "ncipollo/release-action",
)
DIST_GLOB = re.compile(r"(?i)(?:packaging[\\/])?dist[\\/][^\r\n\"'`]*\*")

# A member bigger than this is not read. Nothing in a source bundle is a 32 MB
# text file, so hitting the cap means something unreviewable is being shipped:
# it is reported rather than skipped, because a scan that silently gives up is
# indistinguishable from a scan that passed.
MAX_SCANNED_MEMBER_BYTES = 32 * 1024 * 1024

# Local-only directories that must never reach a published archive, whatever
# the packaging script's exclusion list forgot. Matched on normalized member
# paths, at any depth (a bundle may nest the repo under a top-level folder).
FORBIDDEN_MEMBER_DIRS = ("backend/extensions/",)

# THE EXCEPTION LIST — read this before adding to it.
#
# Each entry is (member glob, pattern label, exact matched text). All three
# must match for a finding to be dropped, so an exception can never widen
# beyond the one string it was written for.
#
# It is EMPTY, and that is a measured result, not an oversight: a real build of
# the current bundle scans 189 of its 189 files — minified JS, CSS, LICENSE and
# all — and produces zero findings. Two shapes people expect to see here do not
# need an entry, and adding one would be wrong:
#   * the UI placeholders `C:\path\to\...`, `C:\ComfyUI`, `C:\ai-toolkit` — the
#     Windows-path pattern only fires on a `…:\Users\<account>` path, because a
#     drive letter plus a generic folder identifies nobody;
#   * help text starting with `Examples:` — it is prose, matching nothing.
# Token-shaped strings that spell out that they are fake (`hf_…LEAKED…`, a
# placeholder someone is meant to overwrite) are handled once, for every
# scanner, by FAKE_SECRET_HINTS in privacy_patterns.py.
#
# If a release ever fails here, the first question is whether the string really
# is harmless. Only then add an entry, with a comment saying why.
ARCHIVE_EXCEPTIONS: tuple[tuple[str, str, str], ...] = ()


def _release_workflows() -> list[Path]:
    workflows: list[Path] = []
    for pattern in ("*.yml", "*.yaml"):
        for path in WORKFLOW_DIR.glob(pattern):
            text = path.read_text(encoding="utf-8")
            if any(marker in text.casefold() for marker in PUBLISH_MARKERS):
                workflows.append(path)
    return sorted(set(workflows))


def check_workflows() -> list[str]:
    errors: list[str] = []
    workflows = _release_workflows()
    if not workflows:
        return ["No GitHub release-publishing workflow was found."]

    for path in workflows:
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT)
        if ".exe" in text.casefold():
            errors.append(f"{relative}: release workflow references a forbidden EXE")
        if DIST_GLOB.search(text):
            errors.append(
                f"{relative}: broad dist wildcard could attach an EXE; list each release asset explicitly"
            )
    return errors


def check_artifact(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"Artifact does not exist: {path}"]
    if path.is_file() and path.suffix.casefold() == ".exe":
        return [f"Forbidden release artifact: {path}"]

    if path.is_dir():
        for candidate in path.rglob("*"):
            if candidate.is_file() and candidate.suffix.casefold() == ".exe":
                errors.append(f"Forbidden executable in release directory: {candidate}")
            rel = candidate.relative_to(path).as_posix()
            normalized = rel + "/" if candidate.is_dir() else rel
            if any(d in normalized for d in FORBIDDEN_MEMBER_DIRS):
                errors.append(
                    f"Local-only directory in {path}: {rel} — "
                    f"backend/extensions/ must never ship"
                )
        return errors

    if path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if Path(member).suffix.casefold() == ".exe":
                    errors.append(f"Forbidden executable in {path}: {member}")
                normalized = member.replace("\\", "/")
                if any(f"{d}" in normalized for d in FORBIDDEN_MEMBER_DIRS):
                    errors.append(
                        f"Local-only directory in {path}: {member} — "
                        f"backend/extensions/ must never ship"
                    )
            errors.extend(scan_archive_privacy(archive, path))
    return errors


def _excused(member: str, label: str, found: str) -> bool:
    return any(
        fnmatch(member, glob) and label == excused_label and found == text
        for glob, excused_label, text in ARCHIVE_EXCEPTIONS
    )


def name_list_path() -> str:
    """Where the forbidden-identifier list lives — never inside the repo."""
    return os.environ.get("LDS_PRIVACY_NAMES") or str(ROOT / ".privacy-names")


def scan_archive_privacy(archive: zipfile.ZipFile, path: Path) -> list[str]:
    """Read every textual member of an OPEN archive and report personal data.

    This is the half that sees what `git ls-files` cannot: whatever ended up in
    the ZIP, tracked or not, source or built.
    """
    errors: list[str] = []
    names = privacy.name_pattern(privacy.read_name_list(name_list_path()))
    if names is None:
        # Never a silent skip: a scan that quietly ran half of itself is how a
        # name reached fifteen releases in the first place.
        print(
            "NOTE: no forbidden-name list (LDS_PRIVACY_NAMES or .privacy-names); "
            "scanning patterns only",
            file=sys.stderr,
        )

    for info in archive.infolist():
        member = info.filename
        if info.is_dir() or not privacy.is_scannable(member):
            continue
        if info.file_size > MAX_SCANNED_MEMBER_BYTES:
            errors.append(
                f"{path.name}!{member}: text member too large to inspect "
                f"({info.file_size} bytes) — it must not ship unreviewed"
            )
            continue
        body = archive.read(member).decode("utf-8", "replace")

        for line, label, found in privacy.scan_text(body):
            if _excused(member, label, found):
                continue
            errors.append(
                f"{path.name}!{member}:{line} — {label}: {found[:60]}"
            )
        if names is not None:
            for match in names.finditer(body):
                line = body[: match.start()].count("\n") + 1
                errors.append(
                    f"{path.name}!{member}:{line} — a forbidden identifier: "
                    f"{match.group(0)}"
                )
    return errors


def main(argv: list[str]) -> int:
    errors = check_workflows()
    for value in argv:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        errors.extend(check_artifact(candidate))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    checked = "release workflows"
    verdict = "contain no publishable EXE"
    if argv:
        checked += " and " + ", ".join(argv)
        verdict += ", no machine path, no personal address and no token"
    print(f"OK: {checked} {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
