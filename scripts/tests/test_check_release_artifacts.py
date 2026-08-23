from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts import check_release_artifacts as policy


# Every forbidden fixture below is ASSEMBLED at runtime, never written as one
# literal. This file is itself scanned by backend/tests/test_no_personal_data.py,
# and the old scanner only stayed green by exempting the file that held its own
# fixtures — an exemption is a hole. Splitting the string leaves no hole: the
# pattern needs the pieces adjacent, and in this source they are not.
MACHINE_PATH = "C:" + chr(92) + "Users" + chr(92) + "acct7x"
PERSONAL_EMAIL = "someone.real" + "@" + "mailbox-provider.net"
TAILNET_IP = "100." + "101.102.103"
OPENAI_KEY = "sk-" + "R7qWmZ2xTb9LdKfE4vHnAcYu"
HF_TOKEN = "hf_" + "QvNbT4mLpZ2rWkXd9sYeAuHc"
GITHUB_PAT = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"


def _zip_with(directory: Path, name: str, members: dict[str, str]) -> Path:
    path = directory / name
    with zipfile.ZipFile(path, "w") as archive:
        for member, text in members.items():
            archive.writestr(member, text)
    return path


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

            with self._no_name_list():
                self.assertTrue(policy.check_artifact(forbidden))
                self.assertEqual(policy.check_artifact(allowed), [])

    # --- the archive privacy scan -----------------------------------------
    #
    # The counter-proof matters more than the happy path: a scan that rejects
    # nothing is indistinguishable from a scan that is broken.

    def _no_name_list(self):
        """Pin the name list to a path that does not exist.

        Otherwise a developer's own .privacy-names would decide whether these
        tests pass, and CI (which has none) would be testing something else.
        """
        return patch.dict("os.environ", {"LDS_PRIVACY_NAMES": "/nonexistent-name-list"})

    def test_archive_is_rejected_for_each_forbidden_shape(self):
        cases = {
            "a Windows user path": ("backend/app/config.py",
                                    f'CACHE = r"{MACHINE_PATH}\\lds"\n'),
            "an email address": ("README.md", f"Contact: {PERSONAL_EMAIL}\n"),
            "an OpenAI-shaped key": ("backend/app/keys.py", f'KEY = "{OPENAI_KEY}"\n'),
            "a Hugging Face token": (".env.example", f"HF_TOKEN={HF_TOKEN}\n"),
            "a GitHub token": ("scripts/bootstrap.ps1", f"$t = '{GITHUB_PAT}'\n"),
            "a tailnet address": ("config.example.json",
                                  '{"host": "' + TAILNET_IP + '"}\n'),
        }
        for label, (member, body) in cases.items():
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as directory:
                    path = _zip_with(Path(directory), "release.zip", {member: body})
                    with self._no_name_list():
                        errors = policy.check_artifact(path)
                self.assertTrue(
                    any(label in error and member in error for error in errors),
                    f"{label} was not reported: {errors}",
                )

    def test_archive_scan_sees_what_git_ls_files_cannot(self):
        """The point of scanning the ZIP: these two members are invisible to
        the tracked-file test — one is a build output it skips, the other is an
        ignored directory that reached a published release."""
        for member in ("frontend/dist/assets/index-Ab12Cd.js",
                       "backend/.pytest_cache/v/cache/nodeids.json"):
            with self.subTest(member=member):
                with tempfile.TemporaryDirectory() as directory:
                    path = _zip_with(Path(directory), "release.zip",
                                     {member: f'{{"path": "{MACHINE_PATH}"}}'})
                    with self._no_name_list():
                        errors = policy.check_artifact(path)
                self.assertTrue(
                    any("a Windows user path" in error for error in errors), errors
                )

    def test_clean_archive_passes_with_ui_placeholders(self):
        """The bundle legitimately ships `C:\\path\\to\\...` placeholders, help
        text starting with `Examples:`, documentation addresses and the edges of
        the tailnet block. A guard that fails a release over these gets turned
        off, so this is a first-class requirement, not politeness."""
        members = {
            "frontend/dist/assets/index-Ab12Cd.js": (
                'placeholder:"C:\\\\path\\\\to\\\\ComfyUI",'
                'hint:"Examples: C:\\\\ComfyUI, C:\\\\ai-toolkit, '
                'C:\\\\Users\\\\<your account>\\\\models"'
            ),
            "README.md": (
                "Report a bug to noreply@lora-dataset-studio.dev, or read the "
                "spoofing fixture https://pexels.com@evil.example/ ; the tailnet "
                "block starts at 100.64.0.0 and Tailscale answers on "
                "100.100.100.100.\n"
            ),
            "config.example.json": '{"comfy_path": "C:\\\\Users\\\\YourName\\\\ComfyUI"}',
            # Long enough to match the token shape; kept only because it says
            # out loud that it is a placeholder.
            ".env.example": "HF_TOKEN=hf_ReplaceThisPlaceholderTokenValue\n",
            "start.bat": "@echo off\r\n",
            "backend/app/version.py": "APP_VERSION = '2026.08.03.1'\n",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = _zip_with(Path(directory), "release.zip", members)
            with self._no_name_list():
                errors = policy.check_artifact(path)
        self.assertEqual(errors, [])

    def test_forbidden_identifier_in_archive_is_reported_when_a_list_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = root / "names.txt"
            names.write_text("# one per line\nRosalind\n", encoding="utf-8")
            path = _zip_with(root, "release.zip",
                            {"README.md": "Written by Rosalind, one evening.\n"})
            with patch.dict("os.environ", {"LDS_PRIVACY_NAMES": str(names)}):
                errors = policy.check_artifact(path)
        self.assertTrue(
            any("a forbidden identifier: Rosalind" in error for error in errors),
            errors,
        )

    def test_unscannably_large_text_member_is_reported_not_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _zip_with(Path(directory), "release.zip", {"huge.json": "{}"})
            with patch.object(policy, "MAX_SCANNED_MEMBER_BYTES", 1), \
                    self._no_name_list():
                errors = policy.check_artifact(path)
        self.assertTrue(any("too large to inspect" in error for error in errors), errors)

    def test_binary_members_are_not_scanned(self):
        """A PNG full of random bytes must not be decoded into false findings."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "release.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("frontend/dist/logo.png",
                                 bytes(range(256)) * 40 + MACHINE_PATH.encode())
            with self._no_name_list():
                errors = policy.check_artifact(path)
        self.assertEqual(errors, [])

    def test_an_archive_carrying_backend_extensions_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / 'bundle.zip'
            with zipfile.ZipFile(bundle, 'w') as zf:
                zf.writestr('backend/run.py', 'print("ok")\n')
                zf.writestr('backend/extensions/something/__init__.py',
                             'def register(app, csrf):\n    pass\n')
            with self._no_name_list():
                errors = policy.check_artifact(bundle)
        self.assertTrue(any('backend/extensions' in e for e in errors), errors)

    def test_a_directory_carrying_backend_extensions_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / 'bundle'
            ext_dir = bundle / 'backend' / 'extensions' / 'something'
            ext_dir.mkdir(parents=True)
            (bundle / 'backend' / 'run.py').write_text('print("ok")\n', encoding='utf-8')
            (ext_dir / '__init__.py').write_text(
                'def register(app, csrf):\n    pass\n', encoding='utf-8'
            )
            with self._no_name_list():
                errors = policy.check_artifact(bundle)
        self.assertTrue(any('backend/extensions' in e for e in errors), errors)


if __name__ == "__main__":
    unittest.main()
