from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts import check_release_artifacts as policy


class ReleaseArtifactPolicyTests(unittest.TestCase):
    def test_release_workflow_rejects_executable_attachment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflows = root / ".github" / "workflows"
            workflows.mkdir(parents=True)
            (workflows / "release.yml").write_text(
                'run: gh release create "v1" "packaging/dist/unstable.exe"\n',
                encoding="utf-8",
            )

            with patch.object(policy, "ROOT", root), patch.object(
                policy, "WORKFLOW_DIR", workflows
            ):
                errors = policy.check_workflows()

        self.assertTrue(any("forbidden EXE" in error for error in errors), errors)

    def test_release_workflow_rejects_broad_dist_glob(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflows = root / ".github" / "workflows"
            workflows.mkdir(parents=True)
            (workflows / "release.yml").write_text(
                'run: gh release create "v1" packaging/dist/*\n', encoding="utf-8"
            )

            with patch.object(policy, "ROOT", root), patch.object(
                policy, "WORKFLOW_DIR", workflows
            ):
                errors = policy.check_workflows()

        self.assertTrue(any("broad dist wildcard" in error for error in errors), errors)

    def test_zip_rejects_executable_member_and_accepts_start_bat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forbidden = root / "forbidden.zip"
            allowed = root / "allowed.zip"
            with zipfile.ZipFile(forbidden, "w") as archive:
                archive.writestr("app/Launcher.EXE", b"not really an executable")
            with zipfile.ZipFile(allowed, "w") as archive:
                archive.writestr("app/start.bat", "@echo off\n")

            self.assertTrue(policy.check_artifact(forbidden))
            self.assertEqual(policy.check_artifact(allowed), [])


if __name__ == "__main__":
    unittest.main()
