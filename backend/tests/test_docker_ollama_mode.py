"""Behavioral tests for the secret-safe Ollama deployment-mode helper."""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / 'scripts' / 'docker-ollama-mode.ps1'
POWERSHELL = shutil.which('powershell.exe')


def run_helper(config, *extra, action='Read'):
    if POWERSHELL is None:
        pytest.skip('Windows PowerShell 5.1 is unavailable')
    result = subprocess.run(
        [
            POWERSHELL,
            '-NoLogo',
            '-NoProfile',
            '-NonInteractive',
            '-ExecutionPolicy',
            'Bypass',
            '-File',
            str(HELPER),
            '-Action',
            action,
            '-ConfigPath',
            str(config),
            *extra,
        ],
        capture_output=True,
        text=True,
        encoding='utf-8',
        timeout=15,
        check=False,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr
    return dict(
        line.split('=', 1)
        for line in result.stdout.splitlines()
        if '=' in line
    ), result


@pytest.mark.parametrize(
    ('payload', 'mode'),
    [
        ({}, 'unset'),
        ({'ollama': {}}, 'unset'),
        ({'ollama': {'deployment_mode': None}}, 'unset'),
        ({'ollama': {'deployment_mode': 'unconfigured'}}, 'unset'),
        ({'ollama': {'deployment_mode': 'none'}}, 'none'),
        ({'ollama': {'deployment_mode': 'host'}}, 'host'),
        ({'ollama': {'deployment_mode': 'docker'}}, 'docker'),
    ],
)
def test_reads_only_the_allowed_persistent_mode(tmp_path, payload, mode):
    config = tmp_path / 'config.json'
    payload.setdefault('secret_token', 'must-never-be-printed')
    config.write_text(json.dumps(payload), encoding='utf-8')

    values, result = run_helper(config)

    assert values == {'STATE': 'VALID', 'MODE': mode, 'PROVIDER': 'ollama'}
    assert 'must-never-be-printed' not in result.stdout
    assert 'must-never-be-printed' not in result.stderr


def test_missing_config_is_unconfigured_and_not_created(tmp_path):
    config = tmp_path / 'missing.json'

    values, _ = run_helper(config)

    assert values == {'STATE': 'VALID', 'MODE': 'unset', 'PROVIDER': 'ollama'}
    assert not config.exists()


@pytest.mark.parametrize(
    'contents',
    [
        '{bad json',
        '[]',
        json.dumps({'ollama': 'not-an-object'}),
        json.dumps({'ollama': {'deployment_mode': 'surprise'}}),
        json.dumps({'ollama': {'deployment_mode': 7}}),
    ],
)
def test_invalid_data_fails_closed_without_echoing_it(tmp_path, contents):
    config = tmp_path / 'config.json'
    config.write_text(contents, encoding='utf-8')

    values, result = run_helper(config)

    assert values == {'STATE': 'INVALID', 'MODE': 'unset', 'PROVIDER': 'ollama'}
    assert contents not in result.stdout


@pytest.mark.parametrize(
    ('payload', 'provider'),
    [
        ({}, 'ollama'),
        ({'local_llm': {}}, 'ollama'),
        ({'local_llm': {'provider': None}}, 'ollama'),
        ({'local_llm': {'provider': 'ollama'}}, 'ollama'),
        ({'local_llm': {'provider': 'LMStudio'}}, 'lmstudio'),
        ({'local_llm': {'provider': ' lmstudio '}}, 'lmstudio'),
        # Anything the launcher does not know reads as Ollama rather than as a
        # refusal: an unknown provider is a config the app will fix, not a
        # reason to leave the user without a Studio.
        ({'local_llm': {'provider': 'surprise'}}, 'ollama'),
        ({'local_llm': 'not-an-object'}, 'ollama'),
    ],
)
def test_reports_which_local_llm_the_sidecar_question_is_about(
        tmp_path, payload, provider):
    config = tmp_path / 'config.json'
    payload.setdefault('secret_token', 'must-never-be-printed')
    config.write_text(json.dumps(payload), encoding='utf-8')

    values, result = run_helper(config)

    assert values['PROVIDER'] == provider
    assert 'must-never-be-printed' not in result.stdout


def test_an_lm_studio_install_still_reports_a_stale_ollama_mode_truthfully(tmp_path):
    """The two answers stay separate, because they are two questions.

    `deployment_mode` can read 'docker' from an Ollama session months ago. The
    helper must not rewrite that to 'none' to be helpful -- the launcher is the
    one that decides not to start the sidecar, and it can only decide that if
    it is told what is actually persisted.
    """
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({
        'local_llm': {'provider': 'lmstudio'},
        'ollama': {'deployment_mode': 'docker'},
    }), encoding='utf-8')

    values, _ = run_helper(config)

    assert values == {'STATE': 'VALID', 'MODE': 'docker', 'PROVIDER': 'lmstudio'}


def test_wait_ends_at_once_when_the_user_does_not_run_ollama(tmp_path):
    """Otherwise the BAT window blocks for its full timeout on a dead question.

    Setup stops offering the deployment cards under LM Studio, so
    `deployment_mode` will never leave 'unset' -- the wait would run to the end
    of its 15 minutes every single launch.
    """
    if POWERSHELL is None:
        pytest.skip('Windows PowerShell 5.1 is unavailable')
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'local_llm': {'provider': 'lmstudio'}}),
                      encoding='utf-8')

    started = time.monotonic()
    values, _ = run_helper(config, '-TimeoutSeconds', '30', '-PollSeconds', '2',
                           action='Wait')
    elapsed = time.monotonic() - started

    assert values == {'STATE': 'VALID', 'MODE': 'unset', 'PROVIDER': 'lmstudio'}
    assert elapsed < 20, 'the wait ran on past a question that cannot be answered'


def test_wait_is_bounded(tmp_path):
    if POWERSHELL is None:
        pytest.skip('Windows PowerShell 5.1 is unavailable')
    config = tmp_path / 'missing.json'

    result = subprocess.run(
        [
            POWERSHELL,
            '-NoLogo',
            '-NoProfile',
            '-NonInteractive',
            '-ExecutionPolicy',
            'Bypass',
            '-File',
            str(HELPER),
            '-Action',
            'Wait',
            '-ConfigPath',
            str(config),
            '-TimeoutSeconds',
            '1',
            '-PollSeconds',
            '1',
        ],
        capture_output=True,
        text=True,
        encoding='utf-8',
        timeout=5,
        check=False,
    )

    assert result.returncode == 0
    assert 'STATE=TIMEOUT' in result.stdout
    assert 'MODE=unset' in result.stdout
