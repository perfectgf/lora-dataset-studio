"""Honest diagnosis of a LOCAL training run that died.

Two pure helpers — no GPU, no ai-toolkit, no filesystem — so both are testable
with plain strings and simulated probe payloads:

* `extract_error_excerpt` picks the part of a training log that EXPLAINS the
  failure (the last traceback, else the last real error line) instead of the
  raw tail. ai-toolkit's very first lines are usually a harmless
  `FutureWarning` from huggingface_hub, and a run that dies before writing
  anything else made the panel show that warning in red as if it were the
  cause. When nothing in the log looks like an error we say so, in plain
  words, instead of dressing the last lines up as one.

* `torch_arch_verdict` turns a torch/GPU probe into a verdict on whether the
  installed PyTorch actually ships kernels for this GPU. The RTX 50 trap:
  Blackwell is compute capability 12.0 (`sm_120`) while stable PyTorch wheels
  only carry kernels up to `sm_90`. `torch.cuda.is_available()` still returns
  True, the GPU is named correctly, buckets and datasets are built — then the
  FIRST real kernel launch dies with "no kernel image is available for
  execution on the device" and ai-toolkit exits 1 with nothing useful said.

* `torch_cuda_verdict` answers the question BEFORE that one: will ai-toolkit
  train on the GPU at all? It takes its device from Hugging Face Accelerate
  (`self.device = str(self.accelerator.device)` in BaseSDTrainProcess since
  2025-01-25, before Krea 2 existed), never from the job config, and Accelerate
  resolves to the CPU without a word when torch cannot see a card — a CPU-only
  wheel (a plain `pip install torch` on Windows), a CUDA build newer than the
  NVIDIA driver, a card hidden by CUDA_VISIBLE_DEVICES — or when its own
  switches point there (ACCELERATE_USE_CPU, ACCELERATE_TORCH_DEVICE), which the
  probe reads back directly. The run then "works": the card stays empty, RAM
  fills up, the ETA reads in the hundreds of hours. Nothing fails, which is
  the whole problem.

* `interpreter_verdict` (+ `missing_module_in_log`, `is_windows_store_python`)
  answers the question the panel never used to answer: WHICH Python was used.
  An interpreter that exists and runs but has no torch is the worst shape of
  all — everything looks configured, and the failure blames something else.

All of them return "I don't know" (kind `none` / `None`) rather than guessing:
the absence of information is never reported as a diagnosis.

Surfaced by the failure block in TrainingPanel, by the training preflight, by
the launch gate and by the Settings ▸ Local tools Test button. The Blackwell
trap was reported by wannadecryptor (Discord, RTX 5070); the silent-CPU trap by
acontentsheltie (Discord, RTX 3090).
"""
import re
from pathlib import Path

from ..utils.redact import redact_tokens, redact_user_paths

# How many lines of the log the excerpt may show. Small enough to stay readable
# on a phone, big enough for a real traceback.
MAX_EXCERPT_LINES = 14

_TRACEBACK_MARK = 'traceback (most recent call last)'

# What "something actually went wrong" looks like in a training log. Ordered
# from most to least specific; any hit makes a line a candidate cause.
_ERROR_RE = re.compile(r"""(?xi)
    \bno\s+kernel\s+image\b                 # the Blackwell / wrong-arch death
  | \bout\s+of\s+memory\b
  | \b[A-Za-z_][\w.]*Error\b                # RuntimeError, OSError, torch.cuda.OutOfMemoryError
  | \b[A-Za-z_][\w.]*Exception\b
  | \berror\b\s*[:=]                        # "CUDA error: ...", "error: ..."
  | \btraceback\b
  | \bassert(?:ion)?\s+(?:failed|error)\b
  | \b(?:killed|aborted|segmentation\s+fault)\b
  | \bfailed\b
""")

# A line that is merely a warning is NEVER a candidate cause, whatever error-ish
# words it happens to contain — that is the whole point of this module.
_WARNING_RE = re.compile(r'\b\w*Warning\b|\bwarnings?\.warn\b|^\s*warn(?:ing)?\b', re.IGNORECASE)


def _clean_lines(log_text):
    """Log text -> non-empty, path-redacted lines. tqdm rewrites its progress bar
    with \\r, so carriage returns are treated as line breaks: only the last state
    of a bar survives, instead of one mile-long line."""
    if not log_text:
        return []
    text = redact_tokens(redact_user_paths(str(log_text)))
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return [line.rstrip() for line in text.split('\n') if line.strip()]


def _cap(lines, max_lines):
    """Keep an excerpt readable: first line + tail, with an explicit marker for
    what was dropped (never a silent truncation)."""
    if len(lines) <= max_lines:
        return list(lines)
    # `max(1, …)`: at max_lines 2 the naive `- 2` gives 0, and `lines[-0:]` is the
    # WHOLE list — a silent no-op cap. Small callers must still get a real cap.
    tail = max(1, max_lines - 2)
    omitted = len(lines) - 1 - tail
    return lines[:1] + [f'... ({omitted} more lines — open training.log for the full run) ...'] \
        + lines[-tail:]


def extract_error_excerpt(log_text, max_lines=MAX_EXCERPT_LINES) -> dict:
    """What of this log explains the failure.

    Returns {'kind', 'text', 'headline'} where `kind` is:
      * 'traceback' — the last Python traceback in the log (the strongest signal);
      * 'error'     — the last non-warning line that reads like an error, with a
                      couple of lines of context around it;
      * 'none'      — nothing in the log looks like an error. `text` is then the
                      plain tail, and the caller MUST present it as context, not
                      as a cause (see `headline`: empty).
    `headline` is the one-line summary (the exception line / the error line),
    empty when kind is 'none'. Everything is already path-redacted."""
    lines = _clean_lines(log_text)
    if not lines:
        return {'kind': 'none', 'text': '', 'headline': ''}

    # 1) The LAST traceback wins: a run can survive an earlier caught one.
    start = None
    for i, line in enumerate(lines):
        if _TRACEBACK_MARK in line.lower():
            start = i
    if start is not None:
        block = lines[start:]
        return {'kind': 'traceback', 'text': '\n'.join(_cap(block, max_lines)),
                'headline': block[-1].strip()}

    # 2) Otherwise the last error-looking line that is NOT a warning.
    idx = None
    for i, line in enumerate(lines):
        if _WARNING_RE.search(line):
            continue
        if _ERROR_RE.search(line):
            idx = i
    if idx is not None:
        block = lines[max(0, idx - 2):idx + 3]
        return {'kind': 'error', 'text': '\n'.join(_cap(block, max_lines)),
                'headline': lines[idx].strip()}

    # 3) Nothing. Say nothing — the caller shows the tail as context only.
    return {'kind': 'none', 'text': '\n'.join(lines[-max_lines:]), 'headline': ''}


# --- gated Hugging Face repo: 401 is NOT 403 -----------------------------------
# Every family with a license-gated base (Krea 2, FLUX.1-dev, FLUX.2 Klein) dies
# the same way when the download is refused, and huggingface_hub prints ONE
# sentence for both cases: "You must have access to it and be authenticated to
# access it". Those are two different problems with two opposite fixes, and
# reading the sentence instead of the status code has already produced a wrong
# public answer (reported by SurpassHR on GitHub, training on Krea 2):
#
#   401 Unauthorized -> the request carried NO valid token. Hugging Face cannot
#                       even tell who is asking. Fix: give the app a token.
#   403 Forbidden    -> the token IS valid; that account has not accepted the
#                       licence / is not on the authorized list. Fix: accept the
#                       licence on the model page. Another token changes nothing.
_GATED_MARK_RE = re.compile(
    r'gatedrepoerror|cannot access gated repo|is restricted|'
    r'must have access to it and be authenticated', re.IGNORECASE)
_HTTP_STATUS_RE = re.compile(r'\b(401|403)\b[^\n]{0,40}?client\s*error', re.IGNORECASE)
# "Access to model krea/Krea-2-Turbo is restricted" — the most reliable shape.
_REPO_TEXT_RE = re.compile(
    r'access to (?:model|repo(?:sitory)?)\s+([\w.\-]+/[\w.\-]+)', re.IGNORECASE)
# Fallback: the resolve URL. `settings`/`docs`/... are site pages, not repos.
_REPO_URL_RE = re.compile(r'huggingface\.co/([\w.\-]+/[\w.\-]+)', re.IGNORECASE)
_NOT_A_REPO_OWNER = {'settings', 'docs', 'api', 'join', 'login', 'pricing', 'blog'}

GATED_401_TITLE = 'Hugging Face saw no valid token — this is not a licence problem'
GATED_403_TITLE = 'Your token works — the licence has not been accepted yet'


def gated_repo_verdict(log_text, token_configured=None) -> dict | None:
    """Did this run die on a gated Hugging Face repo, and WHY exactly?

    Returns None when the log shows no gated-repo refusal (never a guess).
    Otherwise {'status': 401|403, 'repo', 'url', 'title', 'message'} — already
    path-redacted and token-redacted, and never echoing a token itself.

    `token_configured` (optional) is whether the app HAS a Hugging Face token
    saved: with one, a 401 means the token was rejected (expired/revoked/typo),
    which is a different sentence from "you have no token".
    """
    lines = _clean_lines(log_text)
    if not lines:
        return None
    text = '\n'.join(lines)
    if not _GATED_MARK_RE.search(text):
        return None
    m = _HTTP_STATUS_RE.search(text)
    if not m:
        return None
    status = int(m.group(1))

    repo = ''
    hit = _REPO_TEXT_RE.search(text)
    if hit:
        repo = hit.group(1)
    else:
        for cand in _REPO_URL_RE.findall(text):
            if cand.split('/', 1)[0].lower() not in _NOT_A_REPO_OWNER:
                repo = cand
                break
    name = repo or 'the base model'
    url = f'https://huggingface.co/{repo}' if repo else 'the model page on huggingface.co'

    if status == 403:
        return {
            'status': 403, 'repo': repo, 'url': url, 'title': GATED_403_TITLE,
            'message': (
                f'Hugging Face recognised the token but that account has not been granted '
                f'access to {name} (HTTP 403 Forbidden). Adding another token will not help. '
                f'Open {url} while signed in with the SAME account the token belongs to, '
                f'accept the model licence, wait for it to show as granted, then train again.'),
        }
    if token_configured:
        detail = ('A Hugging Face token IS saved in Settings ▸ API keys, so that token was '
                  'rejected — typically expired, revoked, or missing read access. Create a '
                  'fresh read token at https://huggingface.co/settings/tokens, paste it over '
                  'the old one, then train again.')
    else:
        detail = ('No Hugging Face token is saved in Settings ▸ API keys. Create a read token '
                  'at https://huggingface.co/settings/tokens and paste it there, then train '
                  'again. (A token from `hf auth login` is picked up too, but only when the '
                  'CLI ran as the same user as this app.)')
    return {
        'status': 401, 'repo': repo, 'url': url, 'title': GATED_401_TITLE,
        'message': (
            f'The download of {name} was refused as NOT AUTHENTICATED (HTTP 401). Hugging Face '
            f'could not tell who was asking, so this says nothing about whether you have been '
            f'granted access — asking for access again will not fix it. {detail}'),
    }


# --- the interpreter itself ----------------------------------------------------
# The worst failure shape of all, because everything LOOKS configured: a path
# that exists, runs, and has none of ai-toolkit's dependencies. Every run then
# dies on `ModuleNotFoundError: No module named 'torch'` — a sentence that says
# nothing about WHICH Python was used, so the search goes everywhere except the
# one setting at fault. Reported in full by strouder (GitHub #19): a
# `aitoolkit.python` pointing at the Windows Store python.exe stub, with a
# perfectly good `venv\Scripts\python.exe` sitting next to run.py the whole time.
#
# The remedy is not a stricter filter — conda, uv, portable `python_embeded` and
# plain system Pythons are all legitimate here, and no name test tells them from
# a stub. It is to SAY WHICH PATH, and to say it before the run rather than after.
_MISSING_MODULE_RE = re.compile(
    r"ModuleNotFoundError:\s*No module named ['\"]([\w.]+)['\"]")

# `…\AppData\Local\Microsoft\WindowsApps\python.exe` is, on a default Windows 11,
# the App Execution Alias that opens the Microsoft Store — it answers `python
# --version`-ish prompts and has no site-packages anyone can install into for
# ai-toolkit. It is NOT forbidden (a real Store Python does install there), it is
# merely named as the likely explanation once the import has already failed.
_WINDOWS_STORE_RE = re.compile(
    r'[\\/]Microsoft[\\/]WindowsApps[\\/][^\\/]*python[^\\/]*$', re.IGNORECASE)

WINDOWS_STORE_NOTE = (
    'That path is the Windows Store "App Execution Alias" that Windows 11 puts on '
    'PATH by default. It is usually a stub that opens the Microsoft Store rather '
    'than a real Python, and nothing can be installed into it — which is exactly '
    'what an ai-toolkit that "runs but has no torch" looks like.')

INTERPRETER_TITLE = 'The Python configured for ai-toolkit cannot import torch'

# The title above was a CONSTANT, shown for whatever module the log named: a
# venv missing `dotenv` or `oyaml` was announced as "cannot import torch" while
# torch imported fine. The headline is the first thing the failure panel renders
# and the only line most people read, so it stated the one thing that was not
# true. Two shapes now, because the two failures have different remedies:
# torch missing is the interpreter (GitHub #19, strouder — a Windows Store stub
# configured while a working venv sat next to run.py); anything else missing is
# the RIGHT interpreter with an incomplete install, and no amount of repointing
# fixes it.
DEPENDENCY_TITLE = 'The ai-toolkit venv is missing a package ai-toolkit needs'


def interpreter_title(module='torch') -> str:
    """Which of the two headlines this missing module deserves."""
    return INTERPRETER_TITLE if (module or 'torch') == 'torch' else DEPENDENCY_TITLE


def missing_module_in_log(log_text) -> str:
    """The module named by the LAST `ModuleNotFoundError` in a training log, or
    '' when the log shows none. A fact read off the log — never a guess."""
    lines = _clean_lines(log_text)
    found = ''
    for line in lines:
        m = _MISSING_MODULE_RE.search(line)
        if m:
            found = m.group(1)
    return found


def is_windows_store_python(path) -> bool:
    """Does this path look like the Windows Store python stub? Shape only; the
    caller must already know the import failed before it says anything."""
    return bool(path) and bool(_WINDOWS_STORE_RE.search(str(path).replace('/', '\\')))


def interpreter_verdict(python, torch_ok, alternative='', module='torch',
                        aitoolkit_dir=None) -> dict | None:
    """Why a training run cannot start with the interpreter that is configured.

    Returns None whenever `torch_ok` is not a proven False — True (fine) and
    None (probe did not answer: cold-import timeout, no interpreter) are both
    "nothing to say", never a refusal.

    `alternative` is a DIFFERENT interpreter already known to work (typically the
    `venv/` next to run.py that an explicit `aitoolkit.python` is shadowing);
    pass '' when there is none. Both paths are home-redacted for pasting.

    Returns {'python', 'module', 'windows_store', 'alternative', 'title',
    'message'}."""
    if torch_ok is not False:
        return None
    raw = str(python or '').strip()
    store = is_windows_store_python(raw)
    shown = redact_user_paths(raw) or '(no interpreter configured)'
    alt = redact_user_paths(str(alternative or '').strip())
    parts = [
        f'ai-toolkit is set to run with {shown}, and that interpreter cannot '
        f'`import {module}`. Training would die immediately on '
        f'"ModuleNotFoundError: No module named \'{module}\'".',
    ]
    if store:
        parts.append(WINDOWS_STORE_NOTE)
    if alt:
        parts.append(
            f'A working interpreter WAS found in your ai-toolkit folder: {alt} — it '
            f'imports {module} fine. Put that path in Settings ▸ Local tools ▸ '
            '"Python interpreter", or clear that field entirely and the app will '
            'find it by itself.')
    elif module == 'torch':
        parts.append(
            'Point Settings ▸ Local tools ▸ "Python interpreter" at the Python you '
            'actually installed ai-toolkit\'s requirements into (its venv, or your '
            'conda / uv / portable environment), or clear that field to let the app '
            'auto-detect a venv next to run.py.')
    else:
        # NOT the interpreter. `import torch` works in this very Python, so it is
        # the one ai-toolkit was installed into — it is the install that is
        # short. The old copy sent these users to the interpreter setting, which
        # is a dead end when there is no other interpreter to name, and never
        # printed the one line that fixes it. Reported by acontentsheltie
        # (Discord): three PowerShell commands from another chatbot to get a run
        # to start at all.
        parts.append(
            f'This is not the wrong interpreter: torch imports in that same Python, so '
            f'it is the one ai-toolkit was installed into — the install is incomplete, '
            f'and `{module}` is simply not in it. Reinstalling ai-toolkit\'s '
            f'dependencies into that venv is the fix; changing the interpreter is not.')
    # Said in every shape, because the OPPOSITE used to be said: the panel offered
    # "the base model needs a Hugging Face token" for this exact failure, and the
    # search went everywhere but the setting at fault.
    parts.append('This is not a Hugging Face token problem and not a missing base '
                 'model — nothing was downloaded, the interpreter never got that far.')
    command = ('' if module == 'torch'
               else dependency_repair_command(aitoolkit_dir, python))
    return {'python': shown, 'module': module, 'windows_store': store,
            'alternative': alt, 'title': interpreter_title(module),
            'command': command,
            'message': ' '.join(parts) + fix_line(command)}


# --- the Hugging Face fast-download accelerator --------------------------------
# `HF_HUB_ENABLE_HF_TRANSFER=1` swaps huggingface_hub's plain HTTP download for a
# Rust accelerator that must be installed separately (`hf_transfer`, and on newer
# hubs the Xet backend `hf_xet`). With the flag on and the package missing, or
# with the accelerator failing mid-transfer, downloads die — and they die looking
# EXACTLY like a network problem, so people go and check their connection, their
# firewall and their proxy. Reported by bobba84 (GitHub #18): a ComfyUI install
# without `hf_xet`, fixed by setting the variable to 0.
#
# We do NOT set this variable anywhere: the app launches ai-toolkit with
# `dict(os.environ, …)`, so it can only arrive from the machine — a shell profile,
# ai-toolkit's own `.env`, or a ComfyUI launcher. We cannot fix somebody else's
# environment; we can stop it from being mistaken for a network fault.
_HF_TRANSFER_MARK_RE = re.compile(
    r'hf_hub_enable_hf_transfer|hf_transfer|hf_xet|xetdownloaderror|'
    r'consider disabling', re.IGNORECASE)
# Something must have actually gone WRONG. The deprecation FutureWarning about
# this very variable is the single most common line in an ai-toolkit log — firing
# on it would recreate the exact "a warning shown as a cause" bug this module was
# written to kill.
_HF_TRANSFER_FAIL_RE = re.compile(
    r'consider disabling hf_hub_enable_hf_transfer|'
    r'package is not available|'
    r'error while downloading|'
    r'\b(?:hf_transfer|hf_xet)\b[^\n]{0,80}?(?:not available|not installed|'
    r'importerror|failed)|'
    r'(?:importerror|runtimeerror|xetdownloaderror)[^\n]{0,80}?'
    r'\b(?:hf_transfer|hf_xet)\b', re.IGNORECASE)

HF_TRANSFER_TITLE = 'The Hugging Face fast-download accelerator failed — not your network'


def hf_transfer_verdict(log_text) -> dict | None:
    """Did this download die because of the `HF_HUB_ENABLE_HF_TRANSFER` accelerator?

    Returns None whenever the log shows no such failure — including a log that
    merely mentions the variable in its deprecation warning. Otherwise
    {'title', 'message'}, path- and token-redacted."""
    lines = _clean_lines(log_text)
    if not lines:
        return None
    text = '\n'.join(line for line in lines if not _WARNING_RE.search(line))
    if not (_HF_TRANSFER_MARK_RE.search(text) and _HF_TRANSFER_FAIL_RE.search(text)):
        return None
    return {
        'title': HF_TRANSFER_TITLE,
        'message': (
            'This download used the optional Hugging Face fast-download accelerator '
            '(`HF_HUB_ENABLE_HF_TRANSFER=1`), which needs the `hf_transfer` / `hf_xet` '
            'package installed in the SAME environment that downloads. When it is '
            'missing or fails, the transfer aborts with an error that reads like a '
            'connection problem — your network is probably fine. The app never sets '
            'that variable: it comes from your shell, from ai-toolkit\'s `.env`, or '
            'from a ComfyUI launcher. Two fixes, either one works: set '
            '`HF_HUB_ENABLE_HF_TRANSFER=0` to fall back to the plain (slower, very '
            'reliable) HTTP download, or install the accelerator with '
            '`pip install hf_xet` in that environment and try again.'),
    }


# --- torch build vs GPU architecture -------------------------------------------
# `sm_120` -> (12, 0); `sm_86` -> (8, 6). CUDA cubins are binary-compatible
# INSIDE a major version (an sm_86 kernel runs on an sm_89 RTX 4090, which is
# why stable wheels ship no sm_89 and the 4090 works anyway) but never across
# majors — an sm_90 build has nothing to run on Blackwell (major 12).
# `compute_XX` (PTX) entries are deliberately ignored: forward JIT across a major
# generation is not what stable wheels actually deliver, and every field report
# of an RTX 50 on a stable build is the same hard "no kernel image" failure.
_SM_RE = re.compile(r'^sm_(\d+)(\d)[a-z]*$', re.IGNORECASE)

# The first architecture whose only remedy is the cu128 wheel index (Blackwell,
# RTX 50-series). Below that we describe the mismatch but invent no command.
BLACKWELL_MAJOR = 12
CU128_INDEX_URL = 'https://download.pytorch.org/whl/cu128'
CU130_INDEX_URL = 'https://download.pytorch.org/whl/cu130'
# Which index serves which torch. Measured 2026-09-07 with `pip index versions
# torch --index-url …`: cu128 stops at 2.11.0 (torchvision 0.26.0), cu130 carries
# 2.12 and up (2.14.0 at the time). A pin the index cannot serve is "No matching
# distribution found" — worse than no pin — so the split lives here, in one place.
_CU128_LAST_TORCH = (2, 11)
_FINAL_RELEASE_RE = re.compile(r'^\d+(\.\d+)*$')


def torch_arch_verdict(info, venv_python=None) -> dict | None:
    """Does the installed torch ship kernels this GPU can run?

    `info` is the raw probe payload (see capabilities.aitoolkit_torch_info):
    {'torch', 'cuda', 'capability': [major, minor], 'arch_list': [...],
     'device_name'}. Returns None whenever the answer is UNKNOWN — probe absent,
     no GPU, unparseable arch list. None is never a claim of incompatibility.

    Returns {'supported', 'sm', 'gpu', 'torch', 'built_up_to', 'blackwell',
    'message', 'command'} otherwise. `command` is '' when we have no remedy we
    can honestly name."""
    if not isinstance(info, dict) or info.get('error'):
        return None
    cap = info.get('capability')
    if not (isinstance(cap, (list, tuple)) and len(cap) == 2):
        return None
    try:
        major, minor = int(cap[0]), int(cap[1])
    except (TypeError, ValueError):
        return None
    built = []
    for arch in (info.get('arch_list') or []):
        m = _SM_RE.match(str(arch).strip())
        if m:
            built.append((int(m.group(1)), int(m.group(2))))
    if not built:
        return None

    sm = f'sm_{major}{minor}'
    supported = any(bmaj == major and bmin <= minor for bmaj, bmin in built)
    top = max(built)
    gpu = (info.get('device_name') or '').strip() or 'this GPU'
    torch_version = (info.get('torch') or '').strip() or 'the installed PyTorch'
    verdict = {'supported': supported, 'sm': sm, 'gpu': gpu, 'torch': torch_version,
               'built_up_to': f'sm_{top[0]}{top[1]}',
               'blackwell': major >= BLACKWELL_MAJOR, 'message': '', 'command': ''}
    if supported:
        verdict['message'] = (f'{gpu} is compute capability {major}.{minor} ({sm}) and '
                              f'PyTorch {torch_version} ships kernels for it.')
        return verdict

    verdict['message'] = (
        f'{gpu} is compute capability {major}.{minor} ({sm}), but the PyTorch installed in '
        f'the ai-toolkit venv ({torch_version}) only ships GPU kernels up to '
        f'sm_{top[0]}{top[1]}. CUDA still reports the card as available and training starts '
        'normally, then dies at the first real GPU computation with "no kernel image is '
        'available for execution on the device".')
    if major >= BLACKWELL_MAJOR:
        verdict['command'] = torch_reinstall_command(venv_python)
    elif major < min(bmaj for bmaj, _ in built):
        # The other end of the wheel: a card OLDER than anything this build
        # still carries (a GTX 10-series against a build that starts at sm_70).
        # Reachable since the probe runs on every card; a certain death with no
        # way out named is worse than silence, so name the only two there are.
        verdict['message'] += (
            f' This build is newer than the card: an older PyTorch that still ships {sm} '
            'kernels, or a newer card, are the only ways out.')
    return verdict


def _pypi_version(raw) -> str:
    """'2.13.0+cpu' -> '2.13.0': the local tag names the build being replaced.
    '' for a dev / pre-release ('2.14.0.dev20260901', '2.10.0a0'): the stable
    indexes do not carry those (nightlies live under /whl/nightly/), so a pin
    on one is certain to fail — the caller falls back to no pin at all."""
    v = str(raw or '').strip().split('+', 1)[0]
    return v if _FINAL_RELEASE_RE.match(v) else ''


def _version_tuple(v: str):
    try:
        return tuple(int(x) for x in v.split('.')[:2])
    except ValueError:
        return None


def torch_index_url(torch_version=None) -> str:
    """The PyTorch index that actually serves this torch version as a CUDA
    build: cu130 from 2.12 up, cu128 below (and when the version is unknown —
    the Blackwell remedy pins nothing and takes the newest cu128 pair)."""
    t = _version_tuple(_pypi_version(torch_version))
    return CU130_INDEX_URL if (t and t > _CU128_LAST_TORCH) else CU128_INDEX_URL


def torch_reinstall_command(venv_python=None, torch_version=None,
                            torchvision_version=None, index_url=None) -> str:
    """The one paste-safe pip line that swaps the venv's torch for a CUDA build.

    `--no-deps` is not optional. `--index-url` REPLACES PyPI, and together with
    `--force-reinstall` pip re-resolves torch's whole dependency tree from the
    PyTorch index — numpy included, which that index serves in versions old
    enough to break every extension compiled against numpy 2 ("numpy.dtype size
    changed, expected 96, got 88"). Learned on the first Blackwell remedy the
    day after it went out (wannadecryptor). Safe on Windows: the CUDA wheels
    there bundle their DLLs and declare no nvidia-* packages (METADATA read,
    2026-09-07).

    Versions are pinned to the pair the venv already resolved everything else
    against (torchcodec, torchvision and quanto match a torch VERSION, not a
    CUDA flavour) when the probe read both and both are final releases; with
    either unknown, nothing is pinned and pip takes the newest pair on the
    index, which at least agrees with itself — pinning one half alone would be
    the worst of both. The index follows the version (`torch_index_url`): a pin
    the index does not carry is "No matching distribution found"."""
    tv, tvv = _pypi_version(torch_version), _pypi_version(torchvision_version)
    pair = f'torch=={tv} torchvision=={tvv}' if (tv and tvv) else 'torch torchvision'
    index = index_url or torch_index_url(torch_version)
    return (f'{_shell_interpreter(venv_python)} -m pip install --force-reinstall --no-deps '
            f'{pair} --index-url {index}')


def _shell_interpreter(venv_python) -> str:
    """The interpreter as ONE token a shell actually runs, account name hidden.

    Two things were wrong with the obvious `"<path>"`: PowerShell — the default
    shell of Windows Terminal, so the shell the Windows audience pastes into —
    parses a quoted string in command position as an EXPRESSION and stops on
    the next token ("Unexpected token '-m'"); and the `~` that redact_user_paths
    substitutes for the profile is expanded by no Windows shell at all (cmd:
    "syntax incorrect"; PowerShell: "module '~' could not be loaded"). Both
    measured on a live machine. What runs, also measured:

    * Windows, venv under the profile: `& "$env:USERPROFILE\\…"` — the call
      operator makes the string a command, the variable hides the account name
      and PowerShell expands it inside double quotes. PowerShell only, which the
      "Fix (PowerShell)" label says.
    * Windows, anywhere else: the bare path (PowerShell, cmd and bash alike),
      or `& "…"` when it holds a space.
    * POSIX: `~/…` UNQUOTED, so the shell expands it; quoted only for a space.
    """
    raw = str(venv_python or '').strip()
    if not raw:
        return '<ai-toolkit venv python>'
    path = redact_user_paths(raw)
    windows = '\\' in path or bool(re.match(r'^[A-Za-z]:', path))
    if windows:
        if path.startswith('~'):
            return f'& "$env:USERPROFILE{path[1:]}"'
        return f'& "{path}"' if ' ' in path else path
    if path.startswith('~') or ' ' not in path:
        return path
    return f'"{path}"'


def fix_line(command: str) -> str:
    """' Fix: <line>' to append to a message — or ' Fix (PowerShell): <line>'
    when the line only runs there (the `&` call-operator form). '' when there
    is no command to name."""
    if not command:
        return ''
    return (' Fix (PowerShell): ' if command.startswith('& ') else ' Fix: ') + command


# ai-toolkit's own installer, added upstream on 2026-07-27. It reads the NVIDIA
# driver and installs the CUDA build that driver can serve, so it repairs a
# half-installed venv AND a CPU-only torch in one pass.
_MANAGER_ENTRY = ('manager', '__main__.py')


def has_aitoolkit_manager(aitoolkit_dir) -> bool:
    """Does this checkout carry `python -m manager`? Presence on disk, never a
    date: a checkout older than 2026-07-27 has no `manager/`, and naming the
    command there would answer "No module named manager" — a dead remedy handed
    to exactly the oldest installs, the ones most likely to be incomplete."""
    raw = str(aitoolkit_dir or '').strip()
    if not raw:
        return False
    try:
        return (Path(raw) / _MANAGER_ENTRY[0] / _MANAGER_ENTRY[1]).is_file()
    except (OSError, ValueError):
        return False


def dependency_repair_command(aitoolkit_dir=None, venv_python=None) -> str:
    """One line that puts ai-toolkit's own dependencies back into its venv.

    With the manager: `python -m manager install`, run FROM the checkout. It is
    the only remedy that also gets the torch build right.

    Without it, `pip install -r requirements.txt` — and that line is deliberately
    NOT sold as the whole answer by the callers, because it is the very command
    that produces the silent-CPU trap on Windows: `torch` is absent from
    ai-toolkit's requirements yet pulled in by five of them (open_clip_torch,
    lycoris-lora, optimum-quanto, pytorch-wavelets, pytorch_fid), and the wheel
    PyPI serves on Windows is CPU-only. The CPU-only branch of
    torch_cuda_verdict catches that on the next launch and hands over the pip
    line for the CUDA build, so the loop still closes — but the message says so
    rather than letting the user discover it."""
    if has_aitoolkit_manager(aitoolkit_dir):
        return 'python -m manager install'
    raw = str(aitoolkit_dir or '').strip()
    if not raw:
        return ''
    req = redact_user_paths(str(Path(raw) / 'requirements.txt'))
    return f'{_shell_interpreter(venv_python)} -m pip install -r "{req}"'


def torch_cuda_verdict(info, venv_python=None, aitoolkit_dir=None) -> dict | None:
    """Can the PyTorch installed in the ai-toolkit venv see the GPU at all?

    `info` is the raw probe payload (see capabilities.aitoolkit_torch_info).
    Returns None whenever the answer is UNKNOWN — probe absent, an older payload
    without the `cuda_available` field, a probe that errored. None is never a
    claim: on a machine with no NVIDIA card the probe does not even run.

    Otherwise {'available', 'cpu_build', 'torch', 'cuda', 'reason', 'message',
    'command'}. `available` False is the silent-CPU trap: ai-toolkit takes its
    device from Hugging Face Accelerate, never from the job config, and
    Accelerate resolves to the CPU without a word when torch cannot see a card
    — or when its own switches send it there (ACCELERATE_USE_CPU,
    ACCELERATE_TORCH_DEVICE, a `.env` in the ai-toolkit folder), which the
    probe reads back as `accelerator_device`. The run starts, the card stays
    empty, system RAM fills and the ETA reads in the hundreds of hours.

    `command` is the pip line that installs the CUDA build of the SAME torch,
    on the index that serves that version — for a CPU-only wheel. A CUDA build
    the driver cannot serve gets no pip line: the same version on another CUDA
    flavour is not on the indexes (cu128 stops at 2.11, measured), so the
    honest remedy is the driver, and the message says which one. `reason`
    quotes torch's own words when it gave any (the driver-too-old warning),
    path- and token-redacted.

    Reported by acontentsheltie (Discord, RTX 3090): three training logs where
    nothing ever touched the card — latent caching never past image 1, VRAM at
    0.8 GB throughout. (Not the 3.6 s per quantised block in the same logs:
    measured, the device changes that loop by 0.06 s; it is the disk paging
    the weights in at 0.2 GB/s.)"""
    if not isinstance(info, dict) or info.get('error') or 'cuda_available' not in info:
        return None
    if info.get('cuda_available') and not info.get('capability'):
        return None      # "available" with no device to open: a payload at odds with itself
    torch_version = (info.get('torch') or '').strip()
    # '' when the probe did not say: every sentence below drops the version
    # fragment rather than reading "torch the installed PyTorch".
    versioned = f' ({torch_version})' if torch_version else ''
    cuda = (str(info.get('cuda') or '').strip()) or None
    accel = str(info.get('accelerator_device') or '').strip()
    # '' on a payload from before this field existed (a cached probe answer, an
    # older install): the branch below never fires then, and the old
    # "unknown keeps the green" behaviour is what remains.
    accel_error = redact_user_paths(redact_tokens(
        ' '.join(str(info.get('accelerator_error') or '').split())))[:300]
    repair = dependency_repair_command(aitoolkit_dir, venv_python)
    verdict = {'available': bool(info.get('cuda_available')), 'cpu_build': cuda is None,
               'torch': torch_version, 'cuda': cuda, 'reason': '', 'message': '',
               'command': ''}
    consequence = ('ai-toolkit takes its device from Hugging Face Accelerate, which '
                   'falls back to the CPU without a word: the run starts, the card stays '
                   'empty, system RAM fills up and the ETA runs into the hundreds of hours.')
    if verdict['available'] and accel and not accel.startswith('cuda'):
        # torch sees the card; Accelerate — the one ai-toolkit listens to — was
        # told to look elsewhere. No pip line: nothing is installed wrong.
        verdict['available'] = False
        verdict['reason'] = f'accelerator device {accel}'
        verdict['message'] = (
            f'The PyTorch installed in the ai-toolkit venv{versioned} sees the GPU, but '
            f'Hugging Face Accelerate resolves to "{accel}" in that environment. '
            f'{consequence} Something points Accelerate away from the card: '
            'ACCELERATE_USE_CPU or ACCELERATE_TORCH_DEVICE in the environment, or a '
            'line in the .env file of the ai-toolkit folder. Remove it, then launch again.')
        return verdict
    if verdict['available'] and not accel and accel_error:
        # Accelerate did not answer AND said why. Until this branch existed the
        # payload carried `accelerator_device: null` for both "accelerate is not
        # installed" and "Accelerator() raises", the empty string fell through as
        # nothing-to-say, and a venv that cannot train was declared ready — Test
        # green, launch gate open (measured on a venv with a working torch and no
        # accelerate). UNKNOWN keeps the green everywhere else in this file
        # because there the unknown is harmless; here it IS the symptom, so it is
        # only ever raised on a reason the probe brought back.
        verdict['available'] = False
        verdict['reason'] = accel_error
        remedy = ('Reinstall ai-toolkit\'s dependencies in that venv.' if repair
                  else 'Reinstall ai-toolkit\'s dependencies in that venv, following '
                       'its README.')
        if repair and not repair.startswith('python -m manager'):
            # The pip fallback is the command that CREATES the CPU-only wheel on
            # Windows; say so here rather than let the next launch teach it.
            remedy += (' That line installs the packages only — on Windows it also '
                       'pulls a CPU-only PyTorch, and the app will hand you the CUDA '
                       'line on the next launch.')
        verdict['command'] = repair
        verdict['message'] = (
            f'The PyTorch installed in the ai-toolkit venv{versioned} sees the GPU, but '
            f'Hugging Face Accelerate cannot be asked which device to use in that '
            f'environment: {accel_error}. ai-toolkit reads its device from '
            f'`Accelerator()` and from nothing else, so the run would die on that same '
            f'call. {remedy}')
        return verdict
    if verdict['available']:
        gpu = (info.get('device_name') or '').strip() or 'the GPU'
        verdict['message'] = (f'PyTorch{" " + torch_version if torch_version else ""} in the '
                              f'ai-toolkit venv sees {gpu}.')
        return verdict
    reason = redact_user_paths(redact_tokens(
        ' '.join(str(info.get('cuda_reason') or '').split())))
    verdict['reason'] = reason
    if cuda is None:
        verdict['message'] = (
            f'The PyTorch installed in the ai-toolkit venv{versioned} is a '
            f'CPU-only build: it has no CUDA at all, so it cannot see the GPU. '
            f'{consequence} Install the CUDA build of the same PyTorch, then launch '
            'again.')
        verdict['command'] = torch_reinstall_command(
            venv_python, torch_version=info.get('torch'),
            torchvision_version=info.get('torchvision'))
    else:
        why = (f' torch says: "{reason}".' if reason else
               ' Usually the NVIDIA driver is older than that CUDA release, or the '
               'card is hidden from this process (CUDA_VISIBLE_DEVICES).')
        built = (f' ({torch_version}, built for CUDA {cuda})' if torch_version
                 else f' (built for CUDA {cuda})')
        needs = (' (CUDA 13.0 needs an R580 driver or newer; CUDA 12.8, R570)'
                 if cuda.startswith(('13.', '12.8')) else '')
        verdict['message'] = (
            f'The PyTorch installed in the ai-toolkit venv{built} cannot open the GPU '
            f'on this machine.{why} {consequence} Update the NVIDIA driver to one that '
            f'supports CUDA {cuda}{needs}, then launch again.')
    return verdict
