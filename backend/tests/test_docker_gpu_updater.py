"""Local-only contract tests for the portable Docker updater.

Every update uses a forged ZIP, a disposable installation and a fake BAT
launcher. Nothing in this module calls Docker, GitHub, Ollama or the user's
actual installation.
"""

from __future__ import annotations

import os
import hashlib
import json
import shutil
import stat
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
UPDATER = REPO_ROOT / "scripts" / "update-docker-gpu.ps1"
GPU_BAT = REPO_ROOT / "update-docker-gpu.bat"
GENERIC_BAT = REPO_ROOT / "update-docker.bat"
POWERSHELL = shutil.which("powershell.exe")
COMSPEC = os.environ.get("COMSPEC") or shutil.which("cmd.exe")
# Seconds allowed for one cmd.exe -> .bat -> .bat -> PowerShell chain. Cold, on
# a hosted CI runner shared with seven other xdist workers, 20 s was crossed
# twice in two days; this guards against a hang, not against a slow start.
BAT_TIMEOUT = 120
GIT = shutil.which("git.exe") or shutil.which("git")
TEST_COMMIT = "1" * 40

pytestmark = pytest.mark.skipif(
    not POWERSHELL, reason="Windows PowerShell is required for updater tests"
)

PROTECTED_NAMES = {
    ".env",
    ".git",
    ".python",
    ".venv",
    "venv",
    "config.json",
    "data",
    "data-docker",
    "run",
    "basedir",
    "data-docker-gpu",
    "bank-images",
    "ollama-data",
    ".docker-launch-settings",
    ".docker-compose.external-comfy.override.yml",
    ".docker-gpu-settings.env",
    ".docker-gpu-settings",
    ".lds-update.lock",
}


def _write(path: Path, data: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        data = data.encode("utf-8")
    path.write_bytes(data)


def _ascii_harness(target_name: str, argument: str) -> bytes:
    """A .cmd that calls a SIBLING script, with no non-ASCII byte in the file.

    cmd.exe decodes a .bat/.cmd in the *console* codepage, which is not a
    constant: it is 65001 (UTF-8) in a terminal that ran `chcp 65001` and 850
    on a default French install. Encoding the file in `GetOEMCP()` -- what the
    `oem` codec gives -- therefore guesses right only when the two happen to
    agree. Measured on a machine with GetOEMCP=850 and console CP=65001: the
    accented install path was written as cp850 `\\x82`, cmd read it as UTF-8,
    got an invalid lead byte and answered "path not found" (exit 1). The
    hostile argument never reached the .bat, so the rejection this test exists
    for was never exercised -- a recurring false red that looked like a
    product failure and was the harness every time.

    The cure is to stop baking the path into the file: `%~dp0` expands to the
    harness's OWN directory at run time, whatever its bytes are, so dropping
    the harness NEXT TO the target makes the script pure ASCII and identical
    under every codepage. The accented directory still gets exercised -- it is
    where both files live, and it reaches cmd through argv, which Python hands
    to Windows as native UTF-16.
    """
    return (
        "@echo off\r\n"
        f'call "%~dp0{target_name}" "{argument}"\r\n'
        "exit /b %ERRORLEVEL%\r\n"
    ).encode("ascii")


def _fake_launcher(label: str) -> bytes:
    return (
        "@echo off\r\n"
        "setlocal DisableDelayedExpansion\r\n"
        "set /a COUNT=0\r\n"
        'if exist "%LDS_TEST_COUNT%" set /p COUNT=<"%LDS_TEST_COUNT%"\r\n'
        "set /a COUNT+=1\r\n"
        '>"%LDS_TEST_COUNT%" echo %COUNT%\r\n'
        f'>>"%LDS_TEST_LOG%" echo {label} %*\r\n'
        'if "%LDS_TEST_FAIL_FIRST%"=="1" if "%COUNT%"=="1" exit /b 23\r\n'
        'if "%LDS_TEST_ALWAYS_FAIL%"=="1" exit /b 24\r\n'
        "exit /b 0\r\n"
    ).encode("ascii")


def _archive_payload() -> dict[str, bytes]:
    payload = {
        "docker-compose.gpu.yml": b"services:\n  studio: {}\n",
        "Dockerfile.gpu": b"FROM scratch\n",
        "start-docker-gpu.bat": _fake_launcher("LEGACY-NEW"),
        "start-docker.bat": _fake_launcher("STUDIO-NEW"),
        "start-docker-studio.bat": b"@echo off\r\nexit /b 0\r\n",
        "update-docker-gpu.bat": b"@echo off\r\nrem UPDATED GPU BAT\r\n",
        "update-docker.bat": b"@echo off\r\nrem UPDATED GENERIC BAT\r\n",
        "backend/run.py": b"# NEW RUN\n",
        "backend/new_module.py": b"# NEW MODULE\n",
        "frontend/dist/index.html": b"<h1>NEW</h1>\n",
        "packaging/docker/studio_launch.sh": b"#!/bin/sh\n# NEW\n",
        "scripts/update-docker-gpu.ps1": b"# NEW UPDATER SCRIPT\n",
        "scripts/new-helper.ps1": b"# NEW HELPER\n",
        "brand-new-code.txt": b"CREATED BY UPDATE\n",
        ".gitignore": b".env\nrun/\nbasedir/\ndata*/\nbank-images/\nollama-data/\n"
        b".docker-launch-settings\n.docker-compose.external-comfy.override.yml\n"
        b".docker-gpu-settings*\nconfig.json\n.venv/\n.python/\n",
        # A hostile/unexpected source archive may carry local-looking state.
        # The updater must ignore every one of these top-level entries.
        ".env": b"ARCHIVE_SECRET=must-not-win\n",
        "config.json": b'{"archive": true}\n',
        ".docker-launch-settings": b"LAST_LAUNCHER=studio\n",
        ".docker-gpu-settings.env": b"ARCHIVE GPU SETTINGS\n",
        ".docker-gpu-settings": b"ARCHIVE GPU SETTINGS 2\n",
        ".docker-compose.external-comfy.override.yml": (
            b"# path-base64: QzpcQ29tZnlVSSDOqA==\n"
            b"services:\n  studio:\n    volumes: []\n"
        ),
        "data-docker/config.json": (
            b'{"deployment_mode":"archive-must-not-replace-studio"}\n'
        ),
        "data-docker-gpu/config.json": (
            b'{"deployment_mode":"archive-must-not-replace-gpu"}\n'
        ),
    }
    for folder in (
        "run",
        "basedir",
        "data",
        "data-docker",
        "data-docker-gpu",
        "bank-images",
        "ollama-data",
        ".venv",
        ".python",
    ):
        payload[f"{folder}/nested/archive.bin"] = b"ARCHIVE MUST NOT REPLACE STATE"
    return payload


def _write_archive(
    directory: Path,
    payload: dict[str, bytes] | None = None,
    *,
    extra: list[tuple[str | zipfile.ZipInfo, bytes]] | None = None,
) -> Path:
    archive = directory / "archive locale ü !.zip"
    files = payload or _archive_payload()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for relative, data in files.items():
            bundle.writestr(f"lora-dataset-studio-source/{relative}", data)
        for name, data in extra or []:
            bundle.writestr(name, data)
    return archive


def _make_install(
    root: Path,
    *,
    launcher_mode: str | None = None,
    deployment_mode: str = "docker",
    real_updater: bool = False,
) -> None:
    code = {
        "docker-compose.gpu.yml": b"services:\n  old: {}\n",
        "Dockerfile.gpu": b"FROM old\n",
        "start-docker-gpu.bat": _fake_launcher("LEGACY-OLD"),
        "backend/run.py": b"# OLD RUN\n",
        "backend/obsolete.py": b"# MUST RETURN ON ROLLBACK\n",
        "frontend/dist/index.html": b"<h1>OLD</h1>\n",
        "packaging/docker/studio_launch.sh": b"#!/bin/sh\n# OLD\n",
        "scripts/old-helper.ps1": b"# OLD HELPER\n",
        "obsolete-root-code/old.txt": b"# OBSOLETE ROOT CODE\n",
        "update-docker-gpu.bat": b"@echo off\r\nrem OLD GPU BAT\r\n",
        "update-docker.bat": b"@echo off\r\nrem OLD GENERIC BAT\r\n",
    }
    if real_updater:
        code["scripts/update-docker-gpu.ps1"] = UPDATER.read_bytes()
        code["update-docker-gpu.bat"] = GPU_BAT.read_bytes()
        code["update-docker.bat"] = GENERIC_BAT.read_bytes()
    else:
        code["scripts/update-docker-gpu.ps1"] = b"# OLD UPDATER\n"
    if launcher_mode is not None:
        assert launcher_mode in {"studio", "gpu"}
        code["start-docker.bat"] = _fake_launcher("STUDIO-OLD")
    assert deployment_mode in {"none", "host", "docker"}
    for relative, data in code.items():
        _write(root / relative, data)

    state_files = {
        ".env": b"\xffAPI_SECRET=keep-me-byte-for-byte\r\n",
        "config.json": b'{"local": true, "secret": "keep"}\r\n',
        ".docker-gpu-settings.env": b"GPU_CHOICE=legacy\r\n",
        ".docker-gpu-settings": b"opaque legacy choice\x00\xff",
    }
    if launcher_mode is not None:
        state_files[".docker-launch-settings"] = (
            f"LAST_LAUNCHER={launcher_mode}\r\n"
        ).encode("ascii")
    studio_deployment = deployment_mode if launcher_mode == "studio" else "none"
    gpu_deployment = (
        deployment_mode if launcher_mode in {None, "gpu"} else "none"
    )
    state_files["data-docker/config.json"] = (
        "{\r\n"
        f'  "deployment_mode": "{studio_deployment}",\r\n'
        '  "opaque_local_value": "studio préservé !"\r\n'
        "}\r\n"
    ).encode("utf-8")
    state_files["data-docker-gpu/config.json"] = (
        "{\r\n"
        f'  "deployment_mode": "{gpu_deployment}",\r\n'
        '  "opaque_local_value": "gpu préservé !"\r\n'
        "}\r\n"
    ).encode("utf-8")
    if launcher_mode == "studio":
        state_files[".docker-compose.external-comfy.override.yml"] = (
            b"# path-base64: QzpcTW9kZWxlcyDDqCBtb2k=\r\n"
            b"services:\r\n"
            b"  studio:\r\n"
            b"    volumes:\r\n"
            b"      - type: bind\r\n"
            b"        source: 'C:\\\\Modeles \xc3\xa0 moi'\r\n"
            b"        target: /comfy/mnt\r\n"
        )
    for relative, data in state_files.items():
        _write(root / relative, data)
    for index, folder in enumerate(
        (
            "run",
            "basedir",
            "data",
            "data-docker",
            "data-docker-gpu",
            "bank-images",
            "ollama-data",
            ".venv",
            ".python",
        )
    ):
        _write(root / folder / "nested" / "state.bin", bytes([index, 0, 255]) + folder.encode())


def _snapshot_named_state(root: Path) -> dict[str, object]:
    snapshot: dict[str, object] = {}
    for name in sorted(PROTECTED_NAMES - {".git"}):
        path = root / name
        if not path.exists():
            snapshot[name] = None
        elif path.is_file():
            snapshot[name] = path.read_bytes()
        else:
            snapshot[name] = {
                item.relative_to(path).as_posix(): item.read_bytes()
                for item in sorted(path.rglob("*"))
                if item.is_file()
            }
    return snapshot


def _snapshot_code(root: Path) -> dict[str, bytes]:
    return {
        item.relative_to(root).as_posix(): item.read_bytes()
        for item in sorted(root.rglob("*"))
        if item.is_file()
        and item.relative_to(root).parts[0] not in PROTECTED_NAMES
        and not item.name.startswith(".lds-update-")
    }


def _run_updater(
    workspace: Path,
    install: Path,
    *,
    archive: Path | None = None,
    channel: str = "main",
    launcher: Path | None = None,
    runner: Path = UPDATER,
    metadata: Path | None = None,
    git_remote: Path | None = None,
    fail_first: bool = False,
    always_fail: bool = False,
    test_mode: bool = True,
    test_commit: str = TEST_COMMIT,
    test_fault: str | None = None,
    test_signal: Path | None = None,
    test_hold_ms: int = 0,
    max_archive_bytes: int = 0,
    max_entry_bytes: int = 0,
    max_expanded_bytes: int = 0,
    required_free_bytes: int = 0,
    docker: Path | None = None,
    extra_env: dict[str, str] | None = None,
    force_public: bool = False,
) -> tuple[subprocess.CompletedProcess[str], list[str], Path]:
    runtime_temp = workspace / "température runner !"
    runtime_temp.mkdir(parents=True, exist_ok=True)
    log = workspace / "launcher log.txt"
    count = workspace / "launcher count.txt"
    args = [
        POWERSHELL,
        "-NoLogo",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(runner),
        "-InstallRoot",
        str(install),
        "-Channel",
        channel,
    ]
    if test_mode:
        args += ["-TestMode"]
        if test_commit:
            args += ["-TestCommit", test_commit]
    if archive is not None:
        args += ["-ArchivePath", str(archive)]
    if launcher is not None:
        args += ["-LauncherPath", str(launcher)]
    if metadata is not None:
        args += ["-ReleaseMetadataPath", str(metadata)]
    if git_remote is not None:
        args += ["-GitRemote", str(git_remote)]
    if test_fault is not None:
        args += ["-TestFault", test_fault]
    if test_signal is not None:
        args += ["-TestSignalPath", str(test_signal)]
    if test_hold_ms:
        args += ["-TestHoldMilliseconds", str(test_hold_ms)]
    if max_archive_bytes:
        args += ["-TestMaxArchiveBytes", str(max_archive_bytes)]
    if max_entry_bytes:
        args += ["-TestMaxEntryBytes", str(max_entry_bytes)]
    if max_expanded_bytes:
        args += ["-TestMaxExpandedBytes", str(max_expanded_bytes)]
    if required_free_bytes:
        args += ["-TestRequiredFreeBytes", str(required_free_bytes)]
    if docker is not None:
        args += ["-DockerPath", str(docker)]
    if force_public:
        args += ["-ForcePublic"]
    env = os.environ.copy()
    env.pop("LDS_DNS", None)
    env.update(
        {
            "TEMP": str(runtime_temp),
            "TMP": str(runtime_temp),
            "LDS_TEST_LOG": str(log),
            "LDS_TEST_COUNT": str(count),
            "LDS_TEST_FAIL_FIRST": "1" if fail_first else "0",
            "LDS_TEST_ALWAYS_FAIL": "1" if always_fail else "0",
            **(extra_env or {}),
        }
    )
    completed = subprocess.run(
        args,
        cwd=str(workspace),
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=90,
        check=False,
    )
    lines = log.read_text(encoding="ascii").splitlines() if log.exists() else []
    return completed, lines, runtime_temp


def _assert_clean_transaction_dirs(root: Path, runtime_temp: Path) -> None:
    assert not list(root.glob(".lds-update-*"))
    assert not list(runtime_temp.iterdir())


def _assert_confirmed_rebuild_commits_transaction(
    root: Path, runtime_temp: Path
) -> None:
    """--update-rebuild returns 0 only once Docker reports the container
    healthy, so a zero exit commits. Anything left behind would be recovered by
    the NEXT run and roll a working install back to the previous code."""
    assert not list(root.glob(".lds-update-*"))
    assert not (root / ".lds-update.lock").exists()
    assert not list(runtime_temp.iterdir())


def test_static_contract_is_portable_stable_by_default_and_single_logic():
    bat = GPU_BAT.read_text(encoding="utf-8")
    alias = GENERIC_BAT.read_text(encoding="utf-8")
    script = UPDATER.read_text(encoding="utf-8")

    assert 'set "LDS_UPDATE_CHANNEL=stable"' in bat
    assert 'if /I "!LDS_UPDATE_REQUEST!"=="main"' in bat
    assert 'if /I "!LDS_UPDATE_REQUEST!"=="v2"' in bat
    assert 'pushd "%~dp0"' in bat
    assert '-InstallRoot "%~dp0."' in bat
    assert "EnableExtensions DisableDelayedExpansion" in bat
    assert "Unknown channel: %~1" not in bat
    assert "setlocal DisableDelayedExpansion" in alias
    assert (
        alias.strip().splitlines()[-1]
        == '@"%~dp0update-docker-gpu.bat" "%~1" "%~2"'
    )
    assert alias.count("update-docker-gpu.bat") == 1
    # Test-only switches must be unreachable from either public entry point.
    for entry in (bat, alias):
        assert "TestMode" not in entry
        assert "RunningFromTemp" not in entry
        assert "LDS_UPDATE_TEST_" not in entry

    assert script.startswith("#requires -Version 5.1")
    assert "api.github.com/repos/$Repo/releases/latest" in script
    assert "api.github.com/repos/$Repo/commits/$escapedReference" in script
    assert 'Value = "https://codeload.github.com/$Repo/zip/$commit"' in script
    assert "codeload.github.com/$Repo/zip/refs/heads/main" not in script
    assert "'User-Agent' = $script:UserAgent" in script
    assert "No fallback to main" in script
    assert "'--update-rebuild'" in script
    assert "RunningFromTemp" in script and "Invoke-SelfBootstrap" in script
    assert "deployment_mode" not in script
    assert "core.quotePath=false" in script
    assert "'ls-files', '-z'" in script
    assert "'ls-tree', '-r', '-z'" in script
    assert "git pull --ff-only" in script
    assert "cryptographic signature" in script
    assert "stack Docker healthy" not in script
    assert "Rebuild started. Check that" in bat
    for mutation in (
        "@('reset'",
        "@('checkout'",
        "@('merge'",
        "'fetch', '--force'",
        "update-ref",
    ):
        assert mutation not in script
    assert "TestMode" not in bat
    assert "LDS_UPDATE_TEST_" not in bat
    for protected in PROTECTED_NAMES - {".git"}:
        assert f"'{protected}'" in script
    lowered = script.lower()
    for forbidden in ("docker system prune", "docker image rm", "docker volume rm"):
        assert forbidden not in lowered


@pytest.mark.parametrize('channel', ['v2', 'main'])
def test_commit_channel_resolves_v2_including_legacy_alias(tmp_path, channel):
    # Execute the actual resolver with a local commit lookup, without running
    # the updater or contacting GitHub. A renamed branch must not strand users.
    harness = tmp_path / 'resolve.ps1'
    harness.write_text('''param($Source, $Channel)
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Source, [ref]$null, [ref]$null)
$function = $ast.Find({ param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $node.Name -eq 'Resolve-ArchiveSource'
}, $false)
Invoke-Expression $function.Extent.Text
function Assert-RepositoryName { param($Repo) }
function Get-ImmutableCommit { param($Repo, $Reference)
    if ($Reference -ne 'v2') { throw "Unexpected branch: $Reference" }
    return ('1' * 40)
}
Resolve-ArchiveSource -SelectedChannel $Channel -Repo 'sample/project' | ConvertTo-Json -Compress
''', encoding='utf-8')
    result = subprocess.run([POWERSHELL, '-NoProfile', '-File', str(harness),
                             str(UPDATER), channel], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    source = json.loads(result.stdout)
    assert source['Tag'] == 'v2'
    assert source['Commit'] == TEST_COMMIT


@pytest.mark.skipif(not COMSPEC, reason="cmd.exe is required")
def test_bat_alias_handles_special_path_and_rejects_hostile_argument(tmp_path):
    install = tmp_path / "test neuf é !"
    scripts = install / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(GPU_BAT, install / GPU_BAT.name)
    shutil.copy2(GENERIC_BAT, install / GENERIC_BAT.name)
    # Windows PowerShell 5.1 writes ANSI by default, which mangles the accented
    # install path this test deliberately uses. Pin UTF-8 without BOM instead.
    probe = (
        "param([string]$Channel,[string]$InstallRoot)\n"
        "[IO.File]::WriteAllText((Join-Path $InstallRoot 'bat-result.txt'), "
        "($Channel + '|' + $InstallRoot), (New-Object Text.UTF8Encoding($false)))\n"
        "exit 0\n"
    )
    _write(scripts / UPDATER.name, probe)

    # cmd.exe -> update-docker.bat -> update-docker-gpu.bat -> PowerShell, cold,
    # on a hosted runner that is also running seven other xdist workers: CI
    # measured that chain past 20 s twice in two days (2026-08-21, 2026-08-22)
    # and called it a failure of THIS test. The budget guards against a hang,
    # not a slow start, so it is generous — a genuine hang still trips it.
    ok = subprocess.run(
        [COMSPEC, "/d", "/c", str(install / GENERIC_BAT.name), "main"],
        input="\n",
        capture_output=True,
        text=True,
        errors="replace",
        timeout=BAT_TIMEOUT,
        check=False,
    )
    assert ok.returncode == 0
    written = (install / "bat-result.txt").read_text(encoding="utf-8")
    assert written.startswith("main|")
    assert str(install) in written

    (install / "bat-result.txt").unlink()
    # The harness lives NEXT TO the .bat it calls and reaches it through %~dp0,
    # so the accented install path never has to be encoded into the file --
    # see _ascii_harness for the false red that cost.
    harness = install / "hostile argument harness.cmd"
    _write(harness, _ascii_harness(
        GENERIC_BAT.name, "main & echo PWNED_SENTINEL"))
    rejected = subprocess.run(
        [COMSPEC, "/d", "/c", str(harness)],
        input="\n",
        capture_output=True,
        text=True,
        errors="replace",
        timeout=BAT_TIMEOUT,
        check=False,
    )
    assert rejected.returncode == 2
    assert "PWNED_SENTINEL" not in (rejected.stdout + rejected.stderr)
    assert not (install / "bat-result.txt").exists()

    extra = subprocess.run(
        [COMSPEC, "/d", "/c", str(install / GENERIC_BAT.name), "main", "extra"],
        input="\n",
        capture_output=True,
        text=True,
        errors="replace",
        timeout=BAT_TIMEOUT,
        check=False,
    )
    assert extra.returncode == 2
    assert "Too many arguments" in extra.stdout
    assert not (install / "bat-result.txt").exists()


@pytest.mark.skipif(not COMSPEC, reason="cmd.exe is required")
def test_public_bat_ignores_environment_test_injections(tmp_path):
    install = tmp_path / "public bat é !"
    scripts = install / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(GPU_BAT, install / GPU_BAT.name)
    shutil.copy2(GENERIC_BAT, install / GENERIC_BAT.name)
    archive = _write_archive(tmp_path)
    fake = tmp_path / "fake docker launcher.cmd"
    _write(fake, _fake_launcher("MUST-NOT-RUN"))
    probe = (
        "param([string]$Channel,[string]$InstallRoot)\n"
        "$unexpected = @($args) -join '|'\n"
        "Set-Content -LiteralPath (Join-Path $InstallRoot 'bat-result.txt') "
        "-Value ($Channel + '|' + $unexpected) -NoNewline\n"
        "exit 0\n"
    )
    _write(scripts / UPDATER.name, probe)
    env = os.environ.copy()
    env.update(
        {
            "LDS_UPDATE_TEST_ARCHIVE_PATH": str(archive),
            "LDS_UPDATE_TEST_LAUNCHER_PATH": str(fake),
        }
    )
    completed = subprocess.run(
        [COMSPEC, "/d", "/c", str(install / GENERIC_BAT.name), "main"],
        input="\n",
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=BAT_TIMEOUT,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (install / "bat-result.txt").read_text() == "main|"
    assert not (tmp_path / "launcher log.txt").exists()


def test_injection_parameters_require_explicit_test_mode(tmp_path):
    install = tmp_path / "no implicit injection !"
    _make_install(install)
    before_code = _snapshot_code(install)
    archive = _write_archive(tmp_path)

    result, launches, runtime_temp = _run_updater(
        tmp_path,
        install,
        archive=archive,
        test_mode=False,
        test_commit="",
    )

    assert result.returncode == 1
    assert "refused without explicit -TestMode" in (result.stdout + result.stderr)
    assert launches == []
    assert _snapshot_code(install) == before_code
    _assert_clean_transaction_dirs(install, runtime_temp)


def test_zip_success_preserves_all_state_and_legacy_launcher_choice(tmp_path):
    install = tmp_path / "test neuf é !"
    _make_install(install)
    before_state = _snapshot_named_state(install)
    archive = _write_archive(tmp_path)

    result, launches, runtime_temp = _run_updater(tmp_path, install, archive=archive)

    assert result.returncode == 0, result.stdout + result.stderr
    assert launches == ["LEGACY-NEW --update-rebuild"]
    assert _snapshot_named_state(install) == before_state
    assert not (install / ".docker-launch-settings").exists()
    assert (install / "backend" / "run.py").read_bytes() == b"# NEW RUN\n"
    assert not (install / "backend" / "obsolete.py").exists()
    assert not (install / "obsolete-root-code").exists()
    assert (install / "brand-new-code.txt").read_bytes() == b"CREATED BY UPDATE\n"
    _assert_confirmed_rebuild_commits_transaction(install, runtime_temp)


@pytest.mark.parametrize("launcher_mode", ["studio", "gpu"])
@pytest.mark.parametrize("deployment_mode", ["none", "host", "docker"])
def test_launcher_and_opaque_deployment_config_matrix_are_preserved(
    tmp_path, launcher_mode, deployment_mode
):
    install = tmp_path / f"{launcher_mode} {deployment_mode} é !"
    _make_install(
        install, launcher_mode=launcher_mode, deployment_mode=deployment_mode
    )
    config_dir = "data-docker" if launcher_mode == "studio" else "data-docker-gpu"
    config_path = install / config_dir / "config.json"
    config_before = config_path.read_bytes()
    assert f'"deployment_mode": "{deployment_mode}"'.encode() in config_before
    before_state = _snapshot_named_state(install)
    archive = _write_archive(tmp_path)

    result, launches, runtime_temp = _run_updater(tmp_path, install, archive=archive)

    assert result.returncode == 0, result.stdout + result.stderr
    expected = "STUDIO-NEW" if launcher_mode == "studio" else "LEGACY-NEW"
    assert launches == [f"{expected} --update-rebuild"]
    assert config_path.read_bytes() == config_before
    assert _snapshot_named_state(install) == before_state
    _assert_confirmed_rebuild_commits_transaction(install, runtime_temp)


@pytest.mark.parametrize(
    "marker_bytes",
    [
        pytest.param(
            b"LAST_LAUNCHER=studio\r\n"
            b"EXTRA=host\r\n",
            id="multiple-data-lines",
        ),
        pytest.param(
            b"LAST_LAUNCHER=studio\r\n\r\n",
            id="extra-empty-line",
        ),
        pytest.param(b"LAST_LAUNCHER=other\r\n", id="unknown-launcher"),
        pytest.param(b"\xef\xbb\xbfLAST_LAUNCHER=gpu\r\n", id="utf8-bom"),
    ],
)
def test_malformed_launcher_settings_abort_before_overlay(tmp_path, marker_bytes):
    install = tmp_path / "malformed settings !"
    _make_install(install, launcher_mode="studio", deployment_mode="host")
    _write(install / ".docker-launch-settings", marker_bytes)
    before_code = _snapshot_code(install)
    before_state = _snapshot_named_state(install)
    archive = _write_archive(tmp_path)

    result, launches, runtime_temp = _run_updater(tmp_path, install, archive=archive)

    assert result.returncode == 1
    assert "Invalid settings launcher" in (result.stdout + result.stderr)
    assert launches == []
    assert _snapshot_code(install) == before_code
    assert _snapshot_named_state(install) == before_state
    _assert_clean_transaction_dirs(install, runtime_temp)


def test_launcher_settings_directory_is_not_treated_as_legacy_mode(tmp_path):
    install = tmp_path / "settings directory !"
    _make_install(install)
    (install / ".docker-launch-settings").mkdir()
    before_code = _snapshot_code(install)
    before_state = _snapshot_named_state(install)
    archive = _write_archive(tmp_path)

    result, launches, runtime_temp = _run_updater(tmp_path, install, archive=archive)

    assert result.returncode == 1
    assert "folder or reparse point" in (result.stdout + result.stderr)
    assert launches == []
    assert _snapshot_code(install) == before_code
    assert _snapshot_named_state(install) == before_state
    _assert_clean_transaction_dirs(install, runtime_temp)


def test_external_comfy_override_and_referenced_host_folder_are_never_touched(
    tmp_path,
):
    install = tmp_path / "studio external comfy !"
    _make_install(install, launcher_mode="studio", deployment_mode="host")
    external_comfy = tmp_path / "ComfyUI hôte é !"
    _write(external_comfy / "models" / "model.safetensors", b"HOST MODEL BYTES")
    _write(external_comfy / "custom_nodes" / "node.py", b"# HOST NODE\n")
    host_before = {
        item.relative_to(external_comfy).as_posix(): item.read_bytes()
        for item in sorted(external_comfy.rglob("*"))
        if item.is_file()
    }
    override = install / ".docker-compose.external-comfy.override.yml"
    _write(
        override,
        (
            "# updater must never parse or follow this source path\r\n"
            "services:\r\n"
            "  studio:\r\n"
            "    volumes:\r\n"
            "      - type: bind\r\n"
            f"        source: '{external_comfy}'\r\n"
            "        target: /comfy/mnt\r\n"
        ),
    )
    override_before = override.read_bytes()
    archive = _write_archive(tmp_path)

    result, launches, runtime_temp = _run_updater(tmp_path, install, archive=archive)

    assert result.returncode == 0, result.stdout + result.stderr
    assert launches == ["STUDIO-NEW --update-rebuild"]
    assert override.read_bytes() == override_before
    assert {
        item.relative_to(external_comfy).as_posix(): item.read_bytes()
        for item in sorted(external_comfy.rglob("*"))
        if item.is_file()
    } == host_before
    _assert_confirmed_rebuild_commits_transaction(install, runtime_temp)


def test_real_updater_self_updates_from_temp_copy(tmp_path):
    install = tmp_path / "self update é !"
    _make_install(install, real_updater=True)
    archive = _write_archive(tmp_path)
    external = tmp_path / "faux launcher.cmd"
    _write(external, _fake_launcher("OVERRIDE"))

    result, launches, runtime_temp = _run_updater(
        tmp_path,
        install,
        archive=archive,
        launcher=external,
        runner=install / "scripts" / UPDATER.name,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert launches == ["OVERRIDE --update-rebuild"]
    assert (install / "scripts" / UPDATER.name).read_bytes() == b"# NEW UPDATER SCRIPT\n"
    assert (install / GPU_BAT.name).read_bytes() == b"@echo off\r\nrem UPDATED GPU BAT\r\n"
    assert (install / GENERIC_BAT.name).read_bytes() == (
        b"@echo off\r\nrem UPDATED GENERIC BAT\r\n"
    )
    _assert_confirmed_rebuild_commits_transaction(install, runtime_temp)


def test_failed_build_rolls_back_old_code_and_removes_new_files(tmp_path):
    install = tmp_path / "rollback é !"
    _make_install(install, launcher_mode="studio", deployment_mode="host")
    before_code = _snapshot_code(install)
    before_state = _snapshot_named_state(install)
    archive = _write_archive(tmp_path)

    result, launches, runtime_temp = _run_updater(
        tmp_path, install, archive=archive, fail_first=True
    )

    assert result.returncode == 1
    assert launches == [
        "STUDIO-NEW --update-rebuild",
        "STUDIO-OLD --update-rebuild",
    ]
    assert _snapshot_code(install) == before_code
    assert _snapshot_named_state(install) == before_state
    assert not (install / "brand-new-code.txt").exists()
    assert not (install / "backend" / "new_module.py").exists()
    assert (install / "obsolete-root-code" / "old.txt").read_bytes() == (
        b"# OBSOLETE ROOT CODE\n"
    )
    assert "Old code restored" in (result.stdout + result.stderr)
    _assert_clean_transaction_dirs(install, runtime_temp)


def test_invalid_previous_launcher_aborts_before_code_switch(tmp_path):
    install = tmp_path / "invalid old launcher !"
    _make_install(install, launcher_mode="studio", deployment_mode="host")
    previous = install / "start-docker.bat"
    previous.unlink()
    previous.mkdir()
    before_code = _snapshot_code(install)
    before_state = _snapshot_named_state(install)
    archive = _write_archive(tmp_path)

    result, launches, runtime_temp = _run_updater(tmp_path, install, archive=archive)

    assert result.returncode == 1
    assert "Required launcher not found" in (result.stdout + result.stderr)
    assert launches == []
    assert _snapshot_code(install) == before_code
    assert _snapshot_named_state(install) == before_state
    _assert_clean_transaction_dirs(install, runtime_temp)


def test_interrupted_overlay_is_recovered_before_next_archive_validation(tmp_path):
    install = tmp_path / "crash recovery é !"
    _make_install(install, launcher_mode="studio", deployment_mode="host")
    before_code = _snapshot_code(install)
    before_state = _snapshot_named_state(install)
    archive = _write_archive(tmp_path)

    crashed, launches, first_temp = _run_updater(
        tmp_path,
        install,
        archive=archive,
        test_fault="overlay-after-first-switch",
    )

    assert crashed.returncode == 86, crashed.stdout + crashed.stderr
    assert launches == []
    assert list(install.glob(".lds-update-journal-*.json"))

    retry_workspace = tmp_path / "retry"
    retry_workspace.mkdir()
    invalid_payload = _archive_payload()
    del invalid_payload["backend/run.py"]
    invalid_archive = _write_archive(retry_workspace, invalid_payload)
    retried, retry_launches, retry_temp = _run_updater(
        retry_workspace,
        install,
        archive=invalid_archive,
    )

    output = retried.stdout + retried.stderr
    assert retried.returncode == 1
    assert "Recovering interrupted transaction" in output
    assert "missing sentinel 'backend/run.py'" in output
    assert retry_launches == []
    assert _snapshot_code(install) == before_code
    assert _snapshot_named_state(install) == before_state
    assert not list(install.glob(".lds-update-*"))
    _assert_clean_transaction_dirs(install, retry_temp)

    # The killed child cannot run its temp cleanup; remove only its disposable
    # pytest-owned directory after the recovery assertions.
    for child in list(first_temp.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    _assert_clean_transaction_dirs(install, first_temp)


def test_second_update_does_not_recover_and_undo_the_first(tmp_path):
    """A confirmed rebuild must leave no journal behind. While it did, the next
    run treated the previous success as an interrupted transaction and restored
    the code it had just replaced."""
    install = tmp_path / "consecutive updates é !"
    _make_install(install, launcher_mode="gpu")
    archive = _write_archive(tmp_path)

    first, first_launches, first_temp = _run_updater(
        tmp_path, install, archive=archive
    )
    assert first.returncode == 0, first.stdout + first.stderr
    assert first_launches == ["LEGACY-NEW --update-rebuild"]
    assert (install / "backend" / "run.py").read_bytes() == b"# NEW RUN\n"
    _assert_confirmed_rebuild_commits_transaction(install, first_temp)

    second_workspace = tmp_path / "second"
    second_workspace.mkdir()
    newer = _archive_payload()
    newer["backend/run.py"] = b"# NEWER RUN\n"
    second_archive = _write_archive(second_workspace, newer)
    second, second_launches, second_temp = _run_updater(
        second_workspace, install, archive=second_archive
    )

    output = second.stdout + second.stderr
    assert second.returncode == 0, output
    assert "Recovering interrupted transaction" not in output
    assert second_launches == ["LEGACY-NEW --update-rebuild"]
    assert (install / "backend" / "run.py").read_bytes() == b"# NEWER RUN\n"
    _assert_confirmed_rebuild_commits_transaction(install, second_temp)


def test_exclusive_lock_rejects_a_concurrent_update(tmp_path):
    install = tmp_path / "concurrent install é !"
    _make_install(install)
    archive = _write_archive(tmp_path)
    signal = tmp_path / "lock-held.signal"
    first_temp = tmp_path / "first runtime temp"
    first_temp.mkdir()
    first_log = tmp_path / "first launcher.log"
    first_count = tmp_path / "first launcher.count"
    args = [
        POWERSHELL,
        "-NoLogo",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(UPDATER),
        "-InstallRoot",
        str(install),
        "-Channel",
        "main",
        "-TestMode",
        "-TestCommit",
        TEST_COMMIT,
        "-TestFault",
        "hold-lock",
        "-TestSignalPath",
        str(signal),
        "-TestHoldMilliseconds",
        "4000",
    ]
    env = os.environ.copy()
    env.update(
        {
            "TEMP": str(first_temp),
            "TMP": str(first_temp),
            "LDS_TEST_LOG": str(first_log),
            "LDS_TEST_COUNT": str(first_count),
            "LDS_TEST_FAIL_FIRST": "0",
            "LDS_TEST_ALWAYS_FAIL": "0",
        }
    )
    first = subprocess.Popen(
        args,
        cwd=str(tmp_path),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    try:
        deadline = time.monotonic() + 15
        while not signal.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert signal.exists(), "the first updater never acquired its lock"

        second_workspace = tmp_path / "second updater"
        second_workspace.mkdir()
        second, launches, second_temp = _run_updater(
            second_workspace,
            install,
            archive=archive,
        )
        assert second.returncode == 1
        assert "Another update" in (second.stdout + second.stderr)
        assert launches == []
        assert not list(second_temp.iterdir())
    finally:
        stdout, stderr = first.communicate(timeout=20)
    assert first.returncode == 1, stdout + stderr
    assert "TestFault hold-lock" in (stdout + stderr)
    _assert_clean_transaction_dirs(install, first_temp)
    assert not list(second_temp.iterdir())


def test_local_archive_size_limit_uses_actual_downloaded_bytes(tmp_path):
    install = tmp_path / "archive byte limit"
    _make_install(install)
    before_code = _snapshot_code(install)
    archive = _write_archive(tmp_path)

    result, launches, runtime_temp = _run_updater(
        tmp_path,
        install,
        archive=archive,
        max_archive_bytes=128,
    )

    assert result.returncode == 1
    assert "Local archive exceeds the limit" in (result.stdout + result.stderr)
    assert launches == []
    assert _snapshot_code(install) == before_code
    _assert_clean_transaction_dirs(install, runtime_temp)


def test_disk_space_preflight_fails_before_archive_copy(tmp_path):
    install = tmp_path / "disk preflight"
    _make_install(install)
    before_code = _snapshot_code(install)
    archive = _write_archive(tmp_path)

    result, launches, runtime_temp = _run_updater(
        tmp_path,
        install,
        archive=archive,
        required_free_bytes=(2**63 - 1),
    )

    assert result.returncode == 1
    assert "Insufficient disk space" in (result.stdout + result.stderr)
    assert launches == []
    assert _snapshot_code(install) == before_code
    _assert_clean_transaction_dirs(install, runtime_temp)


def _forge_central_uncompressed_size(
    archive: Path, member_suffix: bytes, declared_size: int
) -> None:
    data = bytearray(archive.read_bytes())
    position = 0
    found = False
    signature = b"PK\x01\x02"
    while True:
        position = data.find(signature, position)
        if position < 0:
            break
        name_length, extra_length, comment_length = struct.unpack_from(
            "<HHH", data, position + 28
        )
        name_start = position + 46
        name = bytes(data[name_start : name_start + name_length])
        if name.endswith(member_suffix):
            struct.pack_into("<I", data, position + 24, declared_size)
            found = True
            break
        position = name_start + name_length + extra_length + comment_length
    assert found, f"central entry not found: {member_suffix!r}"
    archive.write_bytes(data)


def test_streaming_entry_limit_ignores_forged_central_length(tmp_path):
    install = tmp_path / "forged zip length"
    _make_install(install)
    before_code = _snapshot_code(install)
    payload = _archive_payload()
    payload["backend/forged-large.bin"] = b"Z" * 4096
    archive = _write_archive(tmp_path, payload)
    _forge_central_uncompressed_size(archive, b"backend/forged-large.bin", 1)

    result, launches, runtime_temp = _run_updater(
        tmp_path,
        install,
        archive=archive,
        max_entry_bytes=1024,
    )

    assert result.returncode == 1
    assert "actual per-entry limit" in (result.stdout + result.stderr)
    assert launches == []
    assert _snapshot_code(install) == before_code
    _assert_clean_transaction_dirs(install, runtime_temp)


def test_streaming_global_expansion_limit_counts_all_entries(tmp_path):
    install = tmp_path / "global zip limit"
    _make_install(install)
    before_code = _snapshot_code(install)
    archive = _write_archive(tmp_path)

    result, launches, runtime_temp = _run_updater(
        tmp_path,
        install,
        archive=archive,
        max_expanded_bytes=256,
    )

    assert result.returncode == 1
    assert "actual global expanded-size limit" in (result.stdout + result.stderr)
    assert launches == []
    assert _snapshot_code(install) == before_code
    _assert_clean_transaction_dirs(install, runtime_temp)


def test_missing_sentinel_fails_before_install_or_launcher(tmp_path):
    install = tmp_path / "missing sentinel !"
    _make_install(install)
    before_code = _snapshot_code(install)
    payload = _archive_payload()
    del payload["packaging/docker/studio_launch.sh"]
    archive = _write_archive(tmp_path, payload)

    result, launches, runtime_temp = _run_updater(tmp_path, install, archive=archive)

    assert result.returncode == 1
    assert "missing sentinel" in (result.stdout + result.stderr)
    assert launches == []
    assert _snapshot_code(install) == before_code
    _assert_clean_transaction_dirs(install, runtime_temp)


def test_path_traversal_zip_is_rejected_before_extraction_or_install(tmp_path):
    install = tmp_path / "traversal target"
    _make_install(install)
    before_code = _snapshot_code(install)
    archive = _write_archive(
        tmp_path,
        extra=[("lora-dataset-studio-source/../../escaped.txt", b"PWNED")],
    )

    result, launches, runtime_temp = _run_updater(tmp_path, install, archive=archive)

    assert result.returncode == 1
    assert "ZIP traversal/segment refused" in (result.stdout + result.stderr)
    assert launches == []
    assert _snapshot_code(install) == before_code
    assert not list(tmp_path.rglob("escaped.txt"))
    _assert_clean_transaction_dirs(install, runtime_temp)


def test_symlink_zip_entry_is_rejected(tmp_path):
    install = tmp_path / "symlink target"
    _make_install(install)
    link = zipfile.ZipInfo("lora-dataset-studio-source/backend/linked.py")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive = _write_archive(tmp_path, extra=[(link, b"../../outside")])

    result, launches, runtime_temp = _run_updater(tmp_path, install, archive=archive)

    assert result.returncode == 1
    assert "ZIP link or special type refused" in (result.stdout + result.stderr)
    assert launches == []
    _assert_clean_transaction_dirs(install, runtime_temp)


def test_stable_without_release_fails_clearly_without_main_fallback(tmp_path):
    install = tmp_path / "no release"
    _make_install(install)
    metadata = tmp_path / "release.json"
    metadata.write_text("{}", encoding="utf-8")

    result, launches, runtime_temp = _run_updater(
        tmp_path, install, channel="stable", metadata=metadata
    )

    output = result.stdout + result.stderr
    assert result.returncode == 1
    assert "No usable stable GitHub Release" in output
    assert "No fallback to main" in output
    assert launches == []
    _assert_clean_transaction_dirs(install, runtime_temp)


def test_local_archive_injection_requires_an_immutable_test_commit(tmp_path):
    install = tmp_path / "missing immutable test commit"
    _make_install(install)
    before_code = _snapshot_code(install)
    archive = _write_archive(tmp_path)

    result, launches, runtime_temp = _run_updater(
        tmp_path,
        install,
        archive=archive,
        test_commit="",
    )

    assert result.returncode == 1
    assert "TestCommit is required" in (result.stdout + result.stderr)
    assert launches == []
    assert _snapshot_code(install) == before_code
    _assert_clean_transaction_dirs(install, runtime_temp)


def test_stable_local_fixture_is_tied_to_one_exact_commit(tmp_path):
    install = tmp_path / "stable exact sha"
    _make_install(install)
    archive = _write_archive(tmp_path)
    metadata = tmp_path / "release.json"
    metadata.write_text('{"tag_name":"v9.8.7"}', encoding="utf-8")
    exact_commit = "a" * 40

    result, launches, runtime_temp = _run_updater(
        tmp_path,
        install,
        archive=archive,
        channel="stable",
        metadata=metadata,
        test_commit=exact_commit,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Selected immutable commit: {exact_commit}" in result.stdout
    assert launches == ["LEGACY-NEW --update-rebuild"]
    _assert_confirmed_rebuild_commits_transaction(install, runtime_temp)


def _git_run(root: Path, *args: str) -> str:
    completed = subprocess.run(
        [GIT, "-C", str(root), *args],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout.strip()


def _make_git_install(
    workspace: Path, *, track_protected_env_in_current: bool = False
) -> tuple[Path, str]:
    install = workspace / "git checkout é !"
    _make_install(install)
    ignore = "\n".join(
        sorted(name + ("/" if (install / name).is_dir() else "") for name in PROTECTED_NAMES)
    )
    _write(install / ".gitignore", ignore + "\n")
    _git_run(install, "init")
    _git_run(install, "config", "user.name", "Updater Tests")
    _git_run(install, "config", "user.email", "updater-tests@example.invalid")
    _git_run(install, "add", ".")
    _git_run(install, "commit", "-m", "old")
    _git_run(install, "branch", "-M", "main")
    if track_protected_env_in_current:
        _git_run(install, "add", "-f", ".env")
        _git_run(install, "commit", "-m", "unsafe protected state")
    old = _git_run(install, "rev-parse", "HEAD")
    assert _git_run(install, "status", "--porcelain") == ""
    return install, old


@pytest.mark.skipif(not GIT, reason="git is required for checkout-path tests")
def test_clean_git_checkout_fails_closed_without_mutating_head(tmp_path):
    install, old = _make_git_install(tmp_path)
    before_state = _snapshot_named_state(install)
    before_code = _snapshot_code(install)

    result, launches, runtime_temp = _run_updater(tmp_path, install)

    output = result.stdout + result.stderr
    assert result.returncode == 1
    assert launches == []
    assert "git pull --ff-only" in output
    assert "start-docker-gpu.bat' --update-rebuild" in output
    assert "no Git file was modified" in output
    assert _git_run(install, "symbolic-ref", "--short", "HEAD") == "main"
    assert _git_run(install, "rev-parse", "HEAD") == old
    assert _git_run(install, "status", "--porcelain") == ""
    assert _snapshot_code(install) == before_code
    assert _snapshot_named_state(install) == before_state
    _assert_clean_transaction_dirs(install, runtime_temp)


@pytest.mark.skipif(not GIT, reason="git is required for checkout-path tests")
def test_dirty_git_checkout_aborts_and_preserves_user_change(tmp_path):
    install, old = _make_git_install(tmp_path)
    _write(install / "backend" / "run.py", b"# LOCAL USER CHANGE\n")

    result, launches, runtime_temp = _run_updater(tmp_path, install)

    assert result.returncode == 1
    assert "Modified Git checkout (initial state)" in (result.stdout + result.stderr)
    assert "git pull --ff-only" in (result.stdout + result.stderr)
    assert launches == []
    assert _git_run(install, "rev-parse", "HEAD") == old
    assert (install / "backend" / "run.py").read_bytes() == b"# LOCAL USER CHANGE\n"
    _assert_clean_transaction_dirs(install, runtime_temp)


@pytest.mark.skipif(not GIT, reason="git is required for checkout-path tests")
def test_git_head_tracking_protected_state_is_rejected(tmp_path):
    install, old = _make_git_install(
        tmp_path, track_protected_env_in_current=True
    )
    before_state = _snapshot_named_state(install)

    result, launches, runtime_temp = _run_updater(tmp_path, install)

    assert result.returncode == 1
    assert "protected local state tracked by Git" in (result.stdout + result.stderr)
    assert launches == []
    assert _git_run(install, "symbolic-ref", "--short", "HEAD") == "main"
    assert _git_run(install, "rev-parse", "HEAD") == old
    assert _snapshot_named_state(install) == before_state
    _assert_clean_transaction_dirs(install, runtime_temp)


@pytest.mark.skipif(not GIT, reason="git is required for checkout-path tests")
def test_git_ignored_protected_unicode_path_is_preserved(tmp_path):
    install, old = _make_git_install(tmp_path)
    ignored = install / "data-docker" / "préservé local !.bin"
    _write(ignored, b"LOCAL IGNORED STATE")

    result, launches, runtime_temp = _run_updater(tmp_path, install)

    output = result.stdout + result.stderr
    assert result.returncode == 1
    assert "git pull --ff-only" in output
    assert "ignored file outside protected state" not in output
    assert launches == []
    assert _git_run(install, "rev-parse", "HEAD") == old
    assert ignored.read_bytes() == b"LOCAL IGNORED STATE"
    _assert_clean_transaction_dirs(install, runtime_temp)


@pytest.mark.skipif(not GIT, reason="git is required for checkout-path tests")
def test_git_ignored_nonprotected_unicode_path_is_rejected(tmp_path):
    install, old = _make_git_install(tmp_path)
    exclude = install / ".git" / "info" / "exclude"
    with exclude.open("a", encoding="utf-8") as stream:
        stream.write("\nignored-extra/\n")
    ignored = install / "ignored-extra" / "nom é local !.bin"
    _write(ignored, b"UNEXPECTED IGNORED CODE")

    result, launches, runtime_temp = _run_updater(tmp_path, install)

    output = result.stdout + result.stderr
    assert result.returncode == 1
    assert "ignored file outside protected state" in output
    assert launches == []
    assert _git_run(install, "rev-parse", "HEAD") == old
    assert ignored.read_bytes() == b"UNEXPECTED IGNORED CODE"
    _assert_clean_transaction_dirs(install, runtime_temp)


# --- Patreon channel: disposable fetch container, then the same transaction ---

INSTALLED = "2026.09.29"
PATREON = "2026.10.20"
PATREON_MARKER = "patreon-channel.json"

FAKE_DOCKER = r'''
import hashlib, json, os, sys
from pathlib import Path

args = sys.argv[1:]
with open(os.environ["LDS_TEST_DOCKER_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps(args) + "\n")
scenario = os.environ.get("LDS_TEST_DOCKER_SCENARIO", "fetch")
if args[:2] == ["image", "inspect"]:
    if scenario == "no-image":
        print("Error: No such image: " + args[-1], file=sys.stderr)
        sys.exit(1)
    print("sha256:" + hashlib.sha256(args[-1].encode()).hexdigest())
    sys.exit(0)
volumes = [args[i + 1] for i, value in enumerate(args) if value == "--volume"]
out = Path(next(v for v in volumes if v.endswith(":/out:rw"))[:-len(":/out:rw")])
if scenario == "current":
    print("No newer LDS release on the Patreon channel.")
    sys.exit(3)
if scenario == "membership":
    print("[ERROR] This update is available to Patreon members.", file=sys.stderr)
    sys.exit(4)
if scenario == "silent":  # An image from before fetch-source existed.
    sys.exit(0)
archive = Path(os.environ["LDS_TEST_DOCKER_ARCHIVE"]).read_bytes()
(out / "lds-core-source.zip").write_bytes(archive)
manifest = {"format": "lds-core-source/1", "version": os.environ["LDS_TEST_DOCKER_VERSION"],
            "commit": "a" * 40, "archive": "lds-core-source.zip", "length": len(archive),
            "sha256": hashlib.sha256(archive).hexdigest()}
if scenario == "tampered":
    manifest["sha256"] = hashlib.sha256(archive + b"x").hexdigest()
if scenario == "extra":
    (out / "node.py").write_bytes(b"print('not part of a release')\n")
(out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
print("Fetched LDS v" + manifest["version"])
'''


def _patreon_setup(workspace: Path, *, launcher_mode: str | None = None,
                   archive_version: str = PATREON) -> tuple[Path, Path, Path]:
    install = workspace / "install patreon é !"
    _make_install(install, launcher_mode=launcher_mode)
    _write(install / "backend" / "app" / "version.py", f"APP_VERSION = '{INSTALLED}'\n")
    payload = _archive_payload()
    payload["backend/app/version.py"] = f"APP_VERSION = '{archive_version}'\n".encode()
    archive = _write_archive(workspace, payload)
    docker = workspace / "fake docker.cmd"
    _write(workspace / "fake_docker.py", FAKE_DOCKER)
    _write(docker, b'@echo off\r\n"%LDS_TEST_PYTHON%" "%~dp0fake_docker.py" %*\r\nexit /b %ERRORLEVEL%\r\n')
    return install, archive, docker


def _run_patreon(workspace: Path, install: Path, archive: Path, docker: Path, *,
                 scenario: str = "fetch", version: str = PATREON, **options):
    calls = workspace / "docker calls.jsonl"
    result, launches, runtime_temp = _run_updater(
        workspace, install, channel="patreon", docker=docker, extra_env={
            "LDS_TEST_PYTHON": sys.executable, "LDS_TEST_DOCKER_LOG": str(calls),
            "LDS_TEST_DOCKER_SCENARIO": scenario, "LDS_TEST_DOCKER_ARCHIVE": str(archive),
            "LDS_TEST_DOCKER_VERSION": version}, **options)
    recorded = ([json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()]
                if calls.exists() else [])
    return result, launches, runtime_temp, recorded


@pytest.mark.parametrize("launcher_mode", [None, "gpu", "studio"])
def test_patreon_channel_fetches_in_a_disposable_container_then_applies(tmp_path, launcher_mode):
    install, archive, docker = _patreon_setup(tmp_path, launcher_mode=launcher_mode)
    if launcher_mode == "gpu":  # Compose reads ${LDS_DNS:-1.1.1.1} from .env.
        with (install / ".env").open("ab") as env:
            env.write(b"LDS_DNS=9.9.9.9 # local resolver\r\n")
    before_state = _snapshot_named_state(install)

    result, launches, runtime_temp, calls = _run_patreon(tmp_path, install, archive, docker)

    assert result.returncode == 0, result.stdout + result.stderr
    studio = launcher_mode == "studio"
    assert launches == ["STUDIO-NEW --update-rebuild" if studio else "LEGACY-NEW --update-rebuild"]
    assert (install / "backend" / "app" / "version.py").read_text() == f"APP_VERSION = '{PATREON}'\n"
    assert (install / "brand-new-code.txt").read_bytes() == b"CREATED BY UPDATE\n"
    data = install / ("data-docker" if studio else "data-docker-gpu")
    # The lane's data folder records the channel, so public channels refuse.
    after_state = _snapshot_named_state(install)
    marker = json.loads(after_state[data.name].pop(PATREON_MARKER).decode("utf-8"))
    assert marker == {"format": "lds-patreon-channel/1", "version": PATREON, "commit": "a" * 40}
    assert after_state == before_state
    _assert_confirmed_rebuild_commits_transaction(install, runtime_temp)

    image = "lora-dataset-studio" if studio else "lora-dataset-studio-gpu"
    inspect, run = calls
    assert inspect == ["image", "inspect", "--format", "{{.Id}}", image]
    volumes = [run[index + 1] for index, value in enumerate(run) if value == "--volume"]
    # Exactly two mounts: data read-only, and a fresh folder of this run only.
    assert len(volumes) == 2 and volumes[0] == f"{data}:/data:ro"
    out = Path(volumes[1][:-len(":/out:rw")])
    assert out.name == "patreon" and out.parent.parent == runtime_temp
    dns = [] if studio else ["--dns", "9.9.9.9" if launcher_mode == "gpu" else "1.1.1.1"]
    assert run == [
        "run", "--rm", "--pull", "never", "--no-healthcheck", "--user", "0:0",
        "--entrypoint", "/usr/local/bin/python" if studio else "/app/.venv/bin/python",
        "--workdir", "/app/backend", "--env", "PYTHONPATH=", "--env", "PYTHONHOME=",
        "--volume", volumes[0], "--volume", volumes[1], *dns,
        "sha256:" + hashlib.sha256(image.encode()).hexdigest(),
        "-m", "app.services.patreon_core", "fetch-source",
        "--out", "/out", "--data", "/data", "--newer-than", INSTALLED]


@pytest.mark.parametrize("scenario,version,message", [
    ("tampered", PATREON, "does not match the SHA-256"),
    ("extra", PATREON, "exactly one release archive"),
    ("silent", PATREON, "exactly one release archive"),
    ("membership", PATREON, "Patreon did not allow this download"),
    ("fetch", "2026.09.01", "not newer than the installed"),
    ("fetch", "2026.10.25", "does not contain the release of its manifest"),
    ("no-image", PATREON, "could not find the image"),
])
def test_patreon_channel_applies_nothing_it_cannot_verify(tmp_path, scenario, version, message):
    install, archive, docker = _patreon_setup(tmp_path)
    before_code = _snapshot_code(install)

    result, launches, runtime_temp, calls = _run_patreon(
        tmp_path, install, archive, docker, scenario=scenario, version=version)

    assert result.returncode == 1, result.stdout + result.stderr
    assert message in result.stdout + result.stderr
    assert launches == []
    assert _snapshot_code(install) == before_code
    _assert_clean_transaction_dirs(install, runtime_temp)
    assert [call[0] for call in calls] == (["image"] if scenario == "no-image" else ["image", "run"])
    assert not list(install.glob(f"data-docker*/{PATREON_MARKER}"))


def test_patreon_channel_up_to_date_changes_nothing(tmp_path):
    install, archive, docker = _patreon_setup(tmp_path)
    before_code = _snapshot_code(install)

    result, launches, runtime_temp, _ = _run_patreon(tmp_path, install, archive, docker, scenario="current")

    assert result.returncode == 3, result.stdout + result.stderr
    assert f"LDS v{INSTALLED} is up to date on the Patreon channel" in result.stdout
    assert launches == []
    assert _snapshot_code(install) == before_code
    _assert_clean_transaction_dirs(install, runtime_temp)


@pytest.mark.skipif(not GIT, reason="git is required for checkout-path tests")
def test_patreon_channel_refuses_a_git_checkout_before_docker(tmp_path):
    install, old = _make_git_install(tmp_path)
    _, archive, docker = _patreon_setup(tmp_path / "unused")

    result, launches, runtime_temp, calls = _run_patreon(tmp_path, install, archive, docker)

    assert result.returncode == 1
    assert "patreon channel updates ZIP installations only" in result.stdout + result.stderr
    assert launches == [] and calls == []
    assert _git_run(install, "rev-parse", "HEAD") == old
    _assert_clean_transaction_dirs(install, runtime_temp)


@pytest.mark.parametrize("archive_version,applied", [("2026.09.01", False), (INSTALLED, True)])
def test_github_channels_never_install_an_older_version(tmp_path, archive_version, applied):
    """After a Patreon update the public release can be older: double-clicking
    update-docker.bat must not silently take the installation back."""
    install = tmp_path / "patreon installed é !"
    _make_install(install)
    _write(install / "backend" / "app" / "version.py", f"APP_VERSION = '{INSTALLED}'\n")
    payload = _archive_payload()
    payload["backend/app/version.py"] = f"APP_VERSION = '{archive_version}'\n".encode()
    before_code = _snapshot_code(install)

    result, launches, runtime_temp = _run_updater(tmp_path, install, archive=_write_archive(tmp_path, payload))

    if applied:  # Same version: reapplying stays possible.
        assert result.returncode == 0, result.stdout + result.stderr
        assert launches == ["LEGACY-NEW --update-rebuild"]
    else:
        assert result.returncode == 1
        assert "older than the installed" in result.stdout + result.stderr
        assert launches == [] and _snapshot_code(install) == before_code
    _assert_clean_transaction_dirs(install, runtime_temp)


@pytest.mark.skipif(not COMSPEC, reason="cmd.exe is required")
def test_bat_forwards_patreon_and_reports_an_up_to_date_installation(tmp_path):
    install = tmp_path / "bat patreon é !"
    (install / "scripts").mkdir(parents=True)
    shutil.copy2(GPU_BAT, install / GPU_BAT.name)
    shutil.copy2(GENERIC_BAT, install / GENERIC_BAT.name)
    probe = (
        "param([string]$Channel,[string]$InstallRoot)\n"
        "[IO.File]::WriteAllText((Join-Path $InstallRoot 'bat-result.txt'), $Channel)\n"
        "exit 3\n"
    )
    _write(install / "scripts" / UPDATER.name, probe)
    completed = subprocess.run(
        [COMSPEC, "/d", "/c", str(install / GENERIC_BAT.name), "patreon"],
        input="\n", capture_output=True, text=True, errors="replace",
        timeout=BAT_TIMEOUT, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "already up to date" in completed.stdout
    assert "Rebuild started" not in completed.stdout
    assert (install / "bat-result.txt").read_text() == "patreon"


@pytest.mark.skipif(not COMSPEC, reason="cmd.exe is required")
@pytest.mark.parametrize("updater_code, returncode, message", [
    (0, 0, "Rebuild started"),
    (3, 0, "already up to date"),
    (1, 1, "did not complete"),
    (4, 1, "did not complete"),
    (-1, 1, "did not complete"),
])
def test_bat_exit_code_follows_the_updater(tmp_path, updater_code, returncode, message):
    """A caller reads the exit code, not the text. An exit /b inside a nested
    block that more commands follow left the process code at 0: on a real
    install (2026-10-18) a refused update printed "did not complete" and
    returned 0."""
    install = tmp_path / "bat exit é !"
    (install / "scripts").mkdir(parents=True)
    shutil.copy2(GPU_BAT, install / GPU_BAT.name)
    shutil.copy2(GENERIC_BAT, install / GENERIC_BAT.name)
    _write(install / "scripts" / UPDATER.name, f"exit {updater_code}\n")
    completed = subprocess.run(
        [COMSPEC, "/d", "/c", str(install / GENERIC_BAT.name), "stable"],
        input="\n", capture_output=True, text=True, errors="replace",
        timeout=BAT_TIMEOUT, check=False)
    assert completed.returncode == returncode, completed.stdout + completed.stderr
    assert message in completed.stdout


# --- The channel marker: public channels never replace a Patreon release ----

def _marker_bytes(version: str = PATREON) -> bytes:
    return json.dumps({"format": "lds-patreon-channel/1", "version": version,
                       "commit": "b" * 40}).encode("utf-8")


@pytest.mark.parametrize("previous", [None, _marker_bytes("2026.10.01")])
def test_a_rejected_patreon_rebuild_restores_the_channel_marker(tmp_path, previous):
    install, archive, docker = _patreon_setup(tmp_path)
    marker = install / "data-docker-gpu" / PATREON_MARKER
    if previous is not None:
        _write(marker, previous)
    before_code = _snapshot_code(install)

    result, launches, runtime_temp, _ = _run_patreon(tmp_path, install, archive, docker, fail_first=True)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "Old code restored" in result.stdout + result.stderr
    assert launches == ["LEGACY-NEW --update-rebuild", "LEGACY-OLD --update-rebuild"]
    assert _snapshot_code(install) == before_code
    assert (marker.read_bytes() if marker.exists() else None) == previous


@pytest.mark.parametrize("lane", ["data-docker-gpu", "data-docker"])
def test_public_channels_refuse_to_replace_a_patreon_installation(tmp_path, lane):
    """Either lane's data folder counts: both lanes share this code folder."""
    install = tmp_path / "patreon marker é !"
    _make_install(install)
    _write(install / lane / PATREON_MARKER, _marker_bytes())
    before_code = _snapshot_code(install)

    result, launches, runtime_temp = _run_updater(tmp_path, install, archive=_write_archive(tmp_path))

    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert f"LDS v{PATREON} from the Patreon channel" in output
    assert "update-docker.bat patreon" in output and "update-docker.bat main force" in output
    assert launches == [] and _snapshot_code(install) == before_code
    _assert_clean_transaction_dirs(install, runtime_temp)


def test_force_installs_the_public_release_and_leaves_the_patreon_channel(tmp_path):
    install = tmp_path / "patreon force é !"
    _make_install(install)
    # A damaged marker still counts; force removes every lane's marker.
    _write(install / "data-docker-gpu" / PATREON_MARKER, b"{")
    _write(install / "data-docker" / PATREON_MARKER, _marker_bytes())

    result, launches, runtime_temp = _run_updater(
        tmp_path, install, archive=_write_archive(tmp_path), force_public=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert launches == ["LEGACY-NEW --update-rebuild"]
    assert (install / "brand-new-code.txt").read_bytes() == b"CREATED BY UPDATE\n"
    assert not list(install.glob(f"data-docker*/{PATREON_MARKER}"))
    _assert_confirmed_rebuild_commits_transaction(install, runtime_temp)


def test_force_is_refused_on_the_patreon_channel(tmp_path):
    install, archive, docker = _patreon_setup(tmp_path)

    result, launches, runtime_temp, calls = _run_patreon(tmp_path, install, archive, docker, force_public=True)

    assert result.returncode == 1
    assert "applies to the stable, v2 and main channels only" in result.stdout + result.stderr
    assert launches == [] and calls == []


@pytest.mark.skipif(not COMSPEC, reason="cmd.exe is required")
def test_bat_forwards_force_only_as_the_second_argument(tmp_path):
    install = tmp_path / "bat force é !"
    (install / "scripts").mkdir(parents=True)
    shutil.copy2(GPU_BAT, install / GPU_BAT.name)
    shutil.copy2(GENERIC_BAT, install / GENERIC_BAT.name)
    probe = (
        "param([string]$Channel,[string]$InstallRoot,[switch]$ForcePublic)\n"
        "[IO.File]::WriteAllText((Join-Path $InstallRoot 'bat-result.txt'), "
        "($Channel + '|' + $ForcePublic + '|' + (@($args) -join ',')))\n"
        "exit 0\n"
    )
    _write(install / "scripts" / UPDATER.name, probe)
    result = install / "bat-result.txt"

    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        # An inherited variable never forces: only the argument does.
        env = {**os.environ, "LDS_UPDATE_FORCE": "-ForcePublic"}
        return subprocess.run(
            [COMSPEC, "/d", "/c", str(install / GENERIC_BAT.name), *arguments],
            input="\n", capture_output=True, text=True, errors="replace", env=env,
            timeout=BAT_TIMEOUT, check=False)

    assert run("stable", "force").returncode == 0
    assert result.read_text() == "stable|True|"
    result.unlink()
    assert run("v2").returncode == 0
    assert result.read_text() == "v2|False|"
    result.unlink()
    refused = run("v2", "forced")
    assert refused.returncode == 2 and "Too many arguments" in refused.stdout
    assert not result.exists()


def test_versions_compare_as_numbers_on_every_channel(tmp_path):
    """2026.10.11.10 is newer than 2026.10.11.9, which an ordinal comparison
    denies: neither a public channel nor the Patreon channel may refuse it."""
    public = tmp_path / "public"
    install = public / "numeric é !"
    _make_install(install)
    _write(install / "backend" / "app" / "version.py", "APP_VERSION = '2026.10.11.9'\n")
    payload = _archive_payload()
    payload["backend/app/version.py"] = b"APP_VERSION = '2026.10.11.10'\n"
    public.mkdir(exist_ok=True)
    result, launches, _ = _run_updater(public, install, archive=_write_archive(public, payload))
    assert result.returncode == 0, result.stdout + result.stderr
    assert launches == ["LEGACY-NEW --update-rebuild"]

    member = tmp_path / "patreon"
    install, archive, docker = _patreon_setup(member, archive_version="2026.10.11.10")
    _write(install / "backend" / "app" / "version.py", "APP_VERSION = '2026.10.11.9'\n")
    result, launches, _, _ = _run_patreon(member, install, archive, docker, version="2026.10.11.10")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (install / "backend" / "app" / "version.py").read_text() == "APP_VERSION = '2026.10.11.10'\n"
