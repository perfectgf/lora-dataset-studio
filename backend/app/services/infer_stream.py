"""Run one of the `backend/infer/*.py` helper scripts in the dedicated ML
interpreter and READ ITS PROGRESS WHILE IT RUNS.

Extracted verbatim from face_similarity._run_scorer, which solved this first for
the face-scoring pass: `subprocess.run()` only hands the output back once the
child has exited, so every `[tag] i/N` line those scripts print all along was
unreadable until it no longer mattered — the bar sat at 0 for the whole pass and
then jumped to N/N, which reads exactly like a hang.

Both the scoring pass and the concept face-mask preview now share this one
mechanism instead of growing a second copy of it. Callers own their own line
grammar (one regex each); this module owns the plumbing:

* Popen, not run() — stdout is read here, stderr is drained by a thread (a full
  pipe would deadlock the child), both bounded: only the last few stderr lines
  are kept, for the error message.
* A watchdog bounds the run rather than letting it hang forever — and, when the
  caller gave an `on_stop`, it ASKS the child to wind up first and only kills it
  if that goes unanswered, so a budget that turned out too small costs the tail
  of the pass instead of all of it.
* A callback that raises must never take the pass down with it: progress is a
  display concern, the work is the work.
* Optionally, a STOP that asks before it kills (see `should_stop`).
"""
from __future__ import annotations
import logging
import subprocess
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)

# How long we wait for the child to die after a timeout kill before giving up on
# joining its reader thread. Short: the pipes are closed by then.
_JOIN_GRACE_S = 5

# After a stop is REQUESTED, how long the child may still take to wind up and
# print what it found before it is killed instead. Generous on purpose: the whole
# value of asking is the final JSON line, and losing it to save a few seconds
# would defeat the mechanism. It only ever elapses when the child is somewhere it
# cannot poll (model load, first-run download) — where there is nothing to lose.
_STOP_GRACE_S = 90
_STOP_POLL_S = 0.25

# Enough stderr to identify a crash (for a Python traceback the last non-empty
# line is the `SomeError: ...` one, which is what a human wants to read) without
# buffering a chatty run.
_TAIL_LINES = 5


def stderr_tail(lines) -> str:
    """Last non-empty stderr line — the useful half of a child's traceback."""
    return next((ln.strip() for ln in reversed(list(lines or ())) if ln.strip()), '')


def run_infer_script(python, script, payload, timeout, on_line=None,
                     should_stop=None, on_stop=None, stop_grace=_STOP_GRACE_S,
                     poll_s=_STOP_POLL_S):
    """Run ``python script``, feed it ``payload`` on stdin, stream its stderr
    lines to ``on_line`` as they arrive.

    Returns ``(stdout, stderr_lines, returncode, timed_out)``. ``timed_out`` says
    the BUDGET elapsed, not that nothing came back: the watchdog asks before it
    kills, so a child that wound up in time still hands over its stdout. Callers
    read the output first and use the flag to label it, not to discard it.

    STOPPING — ``should_stop()`` is polled while the child runs. On its first
    true, ``on_stop()`` is called ONCE (the caller's hook: these scripts are
    asked to wind up by a sentinel file, which the caller owns because it also
    owns the payload that named it) and the child is then given ``stop_grace``
    seconds to exit on its own before being killed.

    Asking before killing is the whole point. These children hold their results
    in memory until they print one final JSON line, so a kill discards
    everything computed so far — the exact waste a Stop button is supposed to
    avoid. The kill remains as a BACKSTOP for the case the child cannot answer:
    a stop asked while it is deep inside a model load or a several-hundred-
    megabyte download reaches no polling point, and there a kill costs nothing
    because nothing has been computed yet.
    """
    proc = subprocess.Popen(
        [python, script], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace',
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    lines: deque = deque(maxlen=_TAIL_LINES)

    def _drain():
        for line in proc.stderr:
            line = line.strip()
            if not line:
                continue
            lines.append(line)
            if on_line:
                try:
                    on_line(line)
                except Exception:
                    logger.debug('infer progress callback failed', exc_info=True)

    reader = threading.Thread(target=_drain, daemon=True)
    reader.start()
    state = {'timed_out': False}
    # Set in the `finally` below, so both the stop poller and the watchdog can
    # tell "the run is over" from "still waiting".
    stop_done = threading.Event()

    def _expire():
        """The budget elapsed. ASK first, kill second — the same bargain as the
        stop path, for the same reason: the child holds everything it computed in
        memory until its final JSON line, so killing it outright is what turns
        "this needed more time" into "the whole pass is lost". A child that
        cannot answer is killed after the grace, and there it has nothing to lose
        anyway (it is inside a model load or a several-hundred-megabyte
        download). Runs on the Timer's own thread, so waiting here blocks
        nothing."""
        state['timed_out'] = True
        if on_stop:
            try:
                on_stop()
            except Exception:
                logger.debug('infer on_stop failed', exc_info=True)
            deadline = time.monotonic() + max(0, stop_grace)
            while time.monotonic() < deadline:
                if proc.poll() is not None or stop_done.wait(poll_s):
                    logger.warning('infer child wound up after its %ss budget elapsed',
                                   int(timeout))
                    return
        logger.warning('infer child exceeded its %ss budget and could not wind up '
                       '— killing', int(timeout))
        try:
            proc.kill()
        except OSError:
            pass

    watchdog = threading.Timer(max(1, int(timeout)), _expire)
    watchdog.daemon = True
    watchdog.start()

    # Stop poller. Runs only when a caller asked for one, so every existing call
    # site keeps exactly the behaviour it had.
    stopper = None
    if should_stop:
        def _watch_stop():
            asked = False
            while not stop_done.wait(poll_s):
                if proc.poll() is not None:
                    return
                if not asked:
                    try:
                        if not should_stop():
                            continue
                    except Exception:
                        logger.debug('infer should_stop failed', exc_info=True)
                        continue
                    asked = True
                    deadline = time.monotonic() + max(0, stop_grace)
                    if on_stop:
                        try:
                            on_stop()
                        except Exception:
                            logger.debug('infer on_stop failed', exc_info=True)
                elif time.monotonic() >= deadline:
                    # It never reached a polling point. Nothing it holds can be
                    # recovered anyway, so stop waiting for a line that is not
                    # coming.
                    logger.info('infer child did not wind up in %ss — killing', stop_grace)
                    try:
                        proc.kill()
                    except OSError:
                        pass
                    return
        stopper = threading.Thread(target=_watch_stop, daemon=True)
        stopper.start()
    try:
        try:
            proc.stdin.write(payload)
            proc.stdin.close()
        except OSError:
            pass        # the child died early — the caller reports it from rc
        stdout = proc.stdout.read()
        proc.wait()
    finally:
        watchdog.cancel()
        stop_done.set()
        if stopper:
            stopper.join(timeout=_JOIN_GRACE_S)
        reader.join(timeout=_JOIN_GRACE_S)
    return stdout, lines, proc.returncode, state['timed_out']
