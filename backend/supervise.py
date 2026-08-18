"""Keep the backend alive across a death Python cannot catch.

A native crash — an access violation inside a C extension, or an antivirus hook
faulting in the middle of one — kills the interpreter outright. No `try`
survives it, no traceback reaches the log, and until this module existed the app
simply stayed down until someone relaunched it by hand.

This is deliberately a SEPARATE, tiny process: everything heavy lives in the
child, so the supervisor itself has almost no surface to crash on.

WHICH LAUNCH PATHS USE IT — the three differ, and the difference is deliberate:

  · start.bat (the Windows double-click) runs this. `LDS_SUPERVISE=0` opts out.
  · Docker already supervises itself and must NOT get a second one:
    packaging/docker/studio_launch.sh wraps the backend in a `while true` loop
    that honours the same exit 75, because ComfyUI is the foreground process
    there and nothing else would restart the studio. Compose adds
    `restart: unless-stopped` above that.
  · packaging/launcher.py (the portable bundle) does NOT, and this is the one
    real gap. Pointing it here would be a one-line change and a worse bug: its
    Quit button calls `proc.terminate()` on whatever it spawned, so it would
    kill the supervisor and leave the backend running as an orphan — an
    invisible server holding the port, with no window left to stop it. Giving
    the bundle crash recovery means teaching the LAUNCHER to relaunch and to
    say so in its Tk status window (it is frozen `--noconsole`, so the notices
    below would go nowhere), not inserting this process underneath it.

The restart contract is NOT invented here: `LDS_RESTART_MODE=supervisor` already
exists in updater.py and is what Docker uses, so an in-app "Update & restart"
ends in exit 75 rather than spawning its own detached helper. Two relaunchers on
one port is the failure that contract exists to prevent.
"""
import os
import subprocess
import sys
import time

CLEAN_EXIT = 0
RESTART_EXIT = 75          # the launcher's "relaunch me" signal (Docker contract)
CONTROL_C_EXIT = 3221225786   # 0xC000013A, Windows STATUS_CONTROL_C_EXIT
RELAUNCH_DELAY = 2.0       # let the port free before binding it again
HEALTHY_UPTIME = 60.0      # served this long => the next death is a fresh one
MAX_RAPID_RESTARTS = 5     # consecutive deaths with no healthy uptime


def run_backend(python, script, *, env=None, spawn=None):
    """Start the backend and wait for it, returning its exit code.

    The child is told the supervisor owns relaunches, so ``schedule_restart``
    exits 75 instead of spawning its own detached helper — two relaunchers would
    put two backends on one port.
    """
    child_env = dict(os.environ if env is None else env)
    child_env['LDS_RESTART_MODE'] = 'supervisor'
    runner = spawn or (lambda command, env: subprocess.call(command, env=env))
    return runner([python, script], child_env)


def console_notice(message, *, stream=None):
    """Put a line on the console AND push it out.

    The flush is the point: this process outlives every message it writes, so a
    buffered notice (stdout redirected to a file, which Python block-buffers)
    would never be seen — the supervisor would relaunch the app in total silence
    and the crash would go unreported.
    """
    out = stream if stream is not None else sys.stdout
    out.write(f'{message}\n')
    try:
        out.flush()
    except (AttributeError, ValueError):    # a closed or exotic stream
        pass


def backend_launcher(python, script, *, env=None, spawn=None):
    """Build the ``run_once`` callable ``supervise`` drives.

    Carries the one piece of state a relaunch must not repeat: opening a browser
    tab. The first launch may open one; every relaunch after it stays silent.
    """
    state = dict(os.environ if env is None else env)

    def once():
        code = run_backend(python, script, env=state, spawn=spawn)
        state['LDS_OPEN_BROWSER'] = '0'
        return code

    return once


def supervise(run_once, *, sleep=time.sleep, now=time.monotonic,
              notify=console_notice):
    """Run the backend, relaunching it when it ends in a way that asks for it.

    ``run_once`` starts the backend and returns its exit code once it ends.
    Returns the exit code the supervisor itself should end on.
    """
    rapid_deaths = 0
    while True:
        started = now()
        try:
            code = run_once()
        except KeyboardInterrupt:
            # Ctrl+C reaches this process as well as the child. Ending quietly
            # keeps the console free of a traceback the user did not cause.
            return CLEAN_EXIT
        # A stop the user asked for, however it is spelled.
        if code in (CLEAN_EXIT, CONTROL_C_EXIT):
            return CLEAN_EXIT
        # Uptime is the whole guard: a backend that served for a while and then
        # died hit a runtime fault and deserves another life, while one that
        # dies on every boot is broken in a way relaunching cannot fix.
        if now() - started >= HEALTHY_UPTIME:
            rapid_deaths = 0
        else:
            rapid_deaths += 1
            if rapid_deaths >= MAX_RAPID_RESTARTS:
                notify(f'[LDS] the app stopped {rapid_deaths} times right after '
                       f'starting (last exit code {code}). Not restarting it again '
                       f'— the console above should say why.')
                return code
        if code == RESTART_EXIT:
            notify('[LDS] restarting...')
        else:
            notify(f'[LDS] the app stopped unexpectedly (crash, exit code {code}) '
                   f'— restarting it. If a long pass was running, reopen the app '
                   f'and start it again; work already committed is kept.')
        sleep(RELAUNCH_DELAY)


def main():
    """Launcher entry point: supervise ``run.py`` sitting next to this file."""
    here = os.path.dirname(os.path.abspath(__file__))
    return supervise(backend_launcher(sys.executable, os.path.join(here, 'run.py')))


if __name__ == '__main__':
    sys.exit(main())
