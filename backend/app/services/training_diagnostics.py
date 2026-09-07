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

* `torch_cuda_verdict` answers the question BEFORE that one: can the installed
  PyTorch see the GPU at all? A CPU-only wheel (a plain `pip install torch` on
  Windows), a CUDA build newer than the NVIDIA driver, or a card hidden by
  CUDA_VISIBLE_DEVICES all make `torch.cuda.is_available()` False — and
  ai-toolkit, which takes its device from Hugging Face Accelerate, then trains
  on the CPU without saying so: the card stays empty, RAM fills up, the ETA
  reads in the hundreds of hours. Nothing fails, which is the whole problem.

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


def interpreter_verdict(python, torch_ok, alternative='', module='torch') -> dict | None:
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
    else:
        parts.append(
            'Point Settings ▸ Local tools ▸ "Python interpreter" at the Python you '
            'actually installed ai-toolkit\'s requirements into (its venv, or your '
            'conda / uv / portable environment), or clear that field to let the app '
            'auto-detect a venv next to run.py.')
    # Said in every shape, because the OPPOSITE used to be said: the panel offered
    # "the base model needs a Hugging Face token" for this exact failure, and the
    # search went everywhere but the setting at fault.
    parts.append('This is not a Hugging Face token problem and not a missing base '
                 'model — nothing was downloaded, the interpreter never got that far.')
    return {'python': shown, 'module': module, 'windows_store': store,
            'alternative': alt, 'title': INTERPRETER_TITLE,
            'message': ' '.join(parts)}


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
    return verdict


def _pypi_version(raw) -> str:
    """'2.13.0+cpu' -> '2.13.0': the local tag names the build being replaced."""
    v = str(raw or '').strip()
    return v.split('+', 1)[0] if v else ''


def torch_reinstall_command(venv_python=None, torch_version=None,
                            torchvision_version=None, index_url=CU128_INDEX_URL) -> str:
    """The one paste-safe pip line that swaps the venv's torch for a CUDA build.

    `--no-deps` is not optional. `--index-url` REPLACES PyPI, and together with
    `--force-reinstall` pip re-resolves torch's whole dependency tree from the
    PyTorch index — numpy included, which that index serves in versions old
    enough to break every extension compiled against numpy 2 ("numpy.dtype size
    changed, expected 96, got 88"). Learned on the first Blackwell remedy the
    day after it went out (wannadecryptor).

    Versions are pinned to the pair the venv already resolved everything else
    against (torchcodec, torchvision and quanto match a torch VERSION, not a
    CUDA flavour) when the probe read both; with either unknown, nothing is
    pinned and pip takes the newest pair on the index, which at least agrees
    with itself — pinning one half alone would be the worst of both."""
    python = redact_user_paths(str(venv_python).strip()) if venv_python else ''
    exe = f'"{python}"' if python else '<ai-toolkit venv python>'
    tv, tvv = _pypi_version(torch_version), _pypi_version(torchvision_version)
    pair = f'torch=={tv} torchvision=={tvv}' if (tv and tvv) else 'torch torchvision'
    return (f'{exe} -m pip install --force-reinstall --no-deps {pair} '
            f'--index-url {index_url}')


def torch_cuda_verdict(info, venv_python=None) -> dict | None:
    """Can the PyTorch installed in the ai-toolkit venv see the GPU at all?

    `info` is the raw probe payload (see capabilities.aitoolkit_torch_info).
    Returns None whenever the answer is UNKNOWN — probe absent, an older payload
    without the `cuda_available` field, a probe that errored. None is never a
    claim: on a machine with no NVIDIA card the probe does not even run.

    Otherwise {'available', 'cpu_build', 'torch', 'cuda', 'reason', 'message',
    'command'}. `available` False is the silent-CPU trap: ai-toolkit takes its
    device from Hugging Face Accelerate, which picks the CPU when
    `torch.cuda.is_available()` answers False and says nothing about it (the
    job config's `device: cuda:0` decides nothing), so the run starts, the card
    stays empty, system RAM fills and the ETA reads in the hundreds of hours.
    `command` is the pip line that installs the CUDA build of the same torch;
    for a CUDA build the driver cannot serve, updating the driver is the other
    way out and the message says so. `reason` quotes torch's own words when it
    gave any (the driver-too-old warning), path- and token-redacted.

    Reported by acontentsheltie (Discord, RTX 3090): three training logs where
    nothing ever touched the card — quantisation at 3.6 s per block instead of
    about one, latent caching never past image 1, VRAM at 0.8 GB throughout."""
    if not isinstance(info, dict) or info.get('error') or 'cuda_available' not in info:
        return None
    torch_version = (info.get('torch') or '').strip() or 'the installed PyTorch'
    cuda = (str(info.get('cuda') or '').strip()) or None
    verdict = {'available': bool(info.get('cuda_available')), 'cpu_build': cuda is None,
               'torch': torch_version, 'cuda': cuda, 'reason': '', 'message': '',
               'command': ''}
    if verdict['available']:
        gpu = (info.get('device_name') or '').strip() or 'the GPU'
        verdict['message'] = f'PyTorch {torch_version} in the ai-toolkit venv sees {gpu}.'
        return verdict
    consequence = ('ai-toolkit picks its device with torch.cuda.is_available() and, '
                   'when that answers False, trains on the CPU without saying so: the '
                   'card stays empty, system RAM fills up and the ETA runs into the '
                   'hundreds of hours.')
    reason = redact_user_paths(redact_tokens(
        ' '.join(str(info.get('cuda_reason') or '').split())))
    verdict['reason'] = reason
    if cuda is None:
        verdict['message'] = (
            f'The PyTorch installed in the ai-toolkit venv ({torch_version}) is a '
            f'CPU-only build: it has no CUDA at all, so it cannot see the GPU. '
            f'{consequence} Install the CUDA build of the same PyTorch, then launch '
            'again.')
    else:
        why = (f' torch says: "{reason}".' if reason else
               ' Usually the NVIDIA driver is older than that CUDA release, or the '
               'card is hidden from this process (CUDA_VISIBLE_DEVICES).')
        verdict['message'] = (
            f'The PyTorch installed in the ai-toolkit venv ({torch_version}, built for '
            f'CUDA {cuda}) cannot open the GPU on this machine.{why} {consequence} '
            'Update the NVIDIA driver, or install the CUDA 12.8 build of the same '
            'PyTorch, then launch again.')
    verdict['command'] = torch_reinstall_command(
        venv_python, torch_version=info.get('torch'),
        torchvision_version=info.get('torchvision'))
    return verdict
