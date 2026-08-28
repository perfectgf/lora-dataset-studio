"""📊 Machine load in five numbers — CPU, RAM, GPU, VRAM, GPU temperature.

What this is FOR: the header and the Canvas show these while a generation or a
training run is going, so "is the machine actually working, or is it stuck?" is
answered by looking, not by opening Task Manager on the server's desktop.

Three deliberate properties, because a widget that polls forever is a widget
that costs something:

* **Every field is optional.** No NVIDIA card, no `nvidia-smi`, psutil missing
  from a slimmed-down install — the key is simply ABSENT from the payload and
  the widget draws one number less. Never 0, never -1: a zero would read as
  "the GPU is idle" on a machine that has no GPU at all.
* **One cache for every client.** Two browser tabs and a phone on the LAN
  polling every 5 s would fork `nvidia-smi` three times over. The reading is
  shared and at most ~3 s old, under a lock, so N clients cost what one costs.
* **Nothing blocks.** `cpu_percent(interval=None)` measures the window since
  the previous call instead of sleeping through a fresh one; the counter is
  primed at import so the first real request already has a window to report.

The `nvidia-smi` invocation mirrors `capabilities.gpu_vram_gb` (same flags,
same timeout, same CREATE_NO_WINDOW so no console window flashes on the
server's desktop) — that call is cached 10 min for a total that never changes,
which is why the live one cannot simply reuse it.
"""
import logging
import subprocess
import threading
import time

logger = logging.getLogger(__name__)

# Short on purpose: long enough that a burst of clients costs one probe, short
# enough that the number moves while you watch it.
_TTL = 3.0

_lock = threading.Lock()
_cache = {'ts': 0.0, 'data': None}

_MIB_PER_GB = 1024.0
_BYTES_PER_GB = 1024 ** 3


def _psutil():
    """psutil, or None when this install does not have it. It IS in
    requirements.txt — but a user-assembled venv is not a promise, and a
    missing dependency must cost two numbers, not a 500."""
    try:
        import psutil
        return psutil
    except Exception:       # pragma: no cover - depends on the install
        return None


# ⚠️ `psutil.cpu_percent(interval=None)` CANNOT be used from a web server, and
# the way it fails is silent. It is differential — it reports the window since
# the previous call — and psutil keeps that previous reading keyed by THREAD
# (`_last_cpu_times[threading.current_thread().ident]`). Flask serves each
# request on a worker thread from a pool, so nearly every poll is that thread's
# FIRST call and answers a flat 0.0. Measured here before it was fixed: the
# endpoint returned `cpu_percent: 0` forever on a machine sitting at 20-30%,
# while the exact same code in a one-thread script reported 12, 16, 27, 20.
#
# So the delta is kept HERE, in one module-level pair of readings guarded by the
# same lock as the cache — thread-independent by construction, and free: no
# sleep, no blocking sample on the request path except the very first one, which
# has no previous reading to subtract.
_FIRST_SAMPLE_S = 0.12
_cpu_prev = None


def _cpu_percent(ps):
    """System-wide busy %, over the window since the previous reading.

    Same arithmetic psutil itself uses (total minus idle minus iowait over
    total), so the number matches what `cpu_percent` would say on a machine
    where it worked.
    """
    global _cpu_prev
    prev, _cpu_prev = _cpu_prev, ps.cpu_times()
    if prev is None:
        # No window to measure yet. One short blocking sample beats opening the
        # widget on "CPU 0%" while a training run is saturating the machine.
        return ps.cpu_percent(interval=_FIRST_SAMPLE_S)
    deltas = [max(0.0, now - was) for now, was in zip(_cpu_prev, prev)]
    total = sum(deltas)
    if total <= 0:
        return 0.0
    fields = _cpu_prev._fields
    idle = sum(d for name, d in zip(fields, deltas) if name in ('idle', 'iowait'))
    return max(0.0, min(100.0, (total - idle) / total * 100.0))


def _gpu_sample():
    """(util %, VRAM used GB, VRAM total GB, temp °C or None) for GPU 0, or
    None when unknown.

    None covers every "there is no answer here" case at once: no NVIDIA card,
    nvidia-smi absent from PATH, a driver that hung, a container without the
    device. Callers drop the GPU fields entirely rather than guessing.
    """
    try:
        proc = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if getattr(proc, 'returncode', 1) != 0:
        return None
    lines = [ln for ln in (proc.stdout or '').strip().splitlines() if ln.strip()]
    if not lines:
        return None
    parts = [p.strip() for p in lines[0].split(',')]
    if len(parts) < 3:
        return None
    try:
        # "[N/A]" is what nvidia-smi prints for utilization on some laptop and
        # virtualised GPUs — float() raises and the whole sample is dropped,
        # which is the honest outcome.
        sample = (round(float(parts[0])),
                  round(float(parts[1]) / _MIB_PER_GB, 1),
                  round(float(parts[2]) / _MIB_PER_GB, 1))
    except ValueError:
        return None
    # Temperature is judged on its own: "[N/A]" here (some virtualised and
    # laptop GPUs) must cost the one number it is, not the whole sample.
    try:
        temp = round(float(parts[3])) if len(parts) > 3 else None
    except ValueError:
        temp = None
    return sample + (temp,)


def _collect():
    """One fresh reading. Every probe is independent: a broken GPU probe must
    not cost the CPU and RAM numbers, and vice versa."""
    out = {}
    ps = _psutil()
    if ps is not None:
        try:
            out['cpu_percent'] = round(_cpu_percent(ps))
        except Exception:   # pragma: no cover - defensive
            logger.debug('cpu_percent unavailable', exc_info=True)
        try:
            vm = ps.virtual_memory()
            # total - available, not `vm.used`: on Windows `used` excludes the
            # cache the OS will hand back on demand, and the widget's job is to
            # answer "how much of this machine is left", which is `available`.
            out['ram_used_gb'] = round((vm.total - vm.available) / _BYTES_PER_GB, 1)
            out['ram_total_gb'] = round(vm.total / _BYTES_PER_GB, 1)
        except Exception:   # pragma: no cover - defensive
            logger.debug('virtual_memory unavailable', exc_info=True)
    gpu = _gpu_sample()
    if gpu is not None:
        out['gpu_percent'], out['vram_used_gb'], out['vram_total_gb'] = gpu[:3]
        if gpu[3] is not None:
            out['gpu_temp_c'] = gpu[3]
    return out


def machine_stats(force=False):
    """The payload behind GET /api/system/stats — at most ~3 s old, shared by
    every client. Keys are absent when the machine cannot answer them."""
    now = time.time()
    with _lock:
        cached = _cache['data']
        if not force and cached is not None and (now - _cache['ts']) < _TTL:
            return dict(cached)
        data = _collect()
        _cache.update(ts=time.time(), data=data)
        return dict(data)


def reset_cache(forget_cpu=False):
    """Tests only — drop the shared reading so the next call really probes.
    `forget_cpu` also drops the CPU baseline, i.e. back to a cold process."""
    global _cpu_prev
    with _lock:
        _cache.update(ts=0.0, data=None)
        if forget_cpu:
            _cpu_prev = None
