"""Enforce the repository policy that published releases never contain an EXE.

With no argument, this checks every workflow that can publish a GitHub release.
Paths passed on the command line are inspected too; ZIP members are checked
without extracting them. The release workflow and CI both invoke this script.
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path


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
        return errors

    if path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if Path(member).suffix.casefold() == ".exe":
                    errors.append(f"Forbidden executable in {path}: {member}")
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
    if argv:
        checked += " and " + ", ".join(argv)
    print(f"OK: {checked} contain no publishable EXE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
