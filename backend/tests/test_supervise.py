"""The launcher must outlive a backend that dies without warning.

A native crash (an access violation inside a C extension or an antivirus hook,
which no Python `except` can catch) takes the whole process down mid-pass. Until
now `start.bat` ran `run.py` directly, so that death was final: the app stayed
down until someone noticed and double-clicked again. Observed live on a bank
watermark pass over ~35 000 images — four crashes in one morning, each one
needing a manual relaunch.

The supervisor turns that into a hiccup. It also has to know the difference
between the three ways the backend can end, because they need opposite answers:

  * exit 0   -> the user asked it to stop. Stay stopped.
  * exit 75  -> "Update & restart" asked for a relaunch (`LDS_RESTART_MODE=
                supervisor`, already the Docker contract). Relaunch.
  * anything else -> it died. Relaunch, but not forever: a backend that crashes
                at boot must not become an infinite respawn loop.
"""
import supervise


class FakeClock:
    """Monotonic time the test advances by hand — the supervisor's restart
    budget is a function of UPTIME, so the clock has to be steerable."""

    def __init__(self):
        self.value = 1000.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def make_runner(exit_codes, clock=None, uptime=0.0):
    """A fake backend that exits with each given code in turn."""
    calls = []

    def run_once():
        calls.append(clock() if clock else len(calls))
        if clock and uptime:
            clock.advance(uptime)
        return exit_codes[len(calls) - 1] if len(calls) <= len(exit_codes) else 0

    run_once.calls = calls
    return run_once


def test_a_clean_exit_is_not_relaunched():
    """Quitting on purpose must stay quit — a supervisor that relaunches a
    deliberate shutdown makes the app impossible to close."""
    runner = make_runner([0])

    result = supervise.supervise(runner, sleep=lambda _s: None)

    assert len(runner.calls) == 1
    assert result == 0


def test_an_update_restart_relaunches_the_backend():
    """`Update & restart` ends the process with 75 and expects the launcher to
    bring it back. Without this the update would land and the app never return."""
    runner = make_runner([supervise.RESTART_EXIT, 0])

    supervise.supervise(runner, sleep=lambda _s: None)

    assert len(runner.calls) == 2


def test_a_backend_that_dies_instantly_gives_up_instead_of_looping_forever():
    """The failure mode a supervisor INTRODUCES: a backend broken at boot (bad
    config, missing dependency, a port it can never bind) dies in a second, and
    a naive loop respawns it forever — burning the CPU and hiding the error
    behind a scrolling console. After a few deaths with no healthy uptime, the
    supervisor must stop and let the exit code speak."""
    clock = FakeClock()
    calls = []

    def run_once():
        calls.append(clock())
        if len(calls) > 20:
            raise RuntimeError('supervisor never gave up — infinite respawn loop')
        clock.advance(0.5)          # dies half a second after every launch
        return 3221225477           # 0xC0000005, a Windows access violation

    result = supervise.supervise(run_once, sleep=lambda _s: None, now=clock)

    assert len(calls) <= 6, f'gave up after {len(calls)} launches, expected a small budget'
    assert result == 3221225477


def test_a_crash_after_serving_normally_does_not_spend_the_budget():
    """The case this whole module exists for: an app that runs fine for hours and
    then dies mid-pass. Each of those deaths is a fresh one, so the give-up
    budget must not accumulate across them — otherwise the fifth crash of the
    week leaves the user with an app that refuses to come back."""
    clock = FakeClock()
    calls = []
    # Four fast deaths bring the budget to the edge, then ONE healthy run has to
    # wipe it — otherwise the very next fast death trips the give-up. Without
    # the interleaving the counter is never incremented and the test proves
    # nothing (verified by mutation: it survived the reset being deleted).
    lifetimes = [0.5, 0.5, 0.5, 0.5, 3600.0, 0.5, 0.5, 0.5, 0.5]

    def run_once():
        index = len(calls)
        calls.append(clock())
        if index >= len(lifetimes):
            return 0                # the run that finally stays up
        clock.advance(lifetimes[index])
        return 3221225477

    supervise.supervise(run_once, sleep=lambda _s: None, now=clock)

    assert len(calls) == len(lifetimes) + 1, (
        f'gave up after {len(calls)} launches — a healthy run must reset the budget')


def test_the_child_is_told_the_supervisor_owns_the_relaunch():
    """`updater.schedule_restart` has two modes. Left to itself it spawns a
    DETACHED helper that waits for the port and starts a new backend — which,
    under a supervisor that ALSO relaunches, means two backends racing for one
    port. Setting LDS_RESTART_MODE=supervisor switches it to a plain exit 75 and
    leaves us the only relauncher. Getting this wrong is silent until the day an
    update produces two servers and a lost database write."""
    seen = {}

    def fake_spawn(command, env):
        seen['command'] = command
        seen['env'] = env
        return 0

    supervise.run_backend('py.exe', 'run.py', env={'EXISTING': 'kept'},
                          spawn=fake_spawn)

    assert seen['env']['LDS_RESTART_MODE'] == 'supervisor'
    assert seen['env']['EXISTING'] == 'kept', 'the child must inherit the launcher env'
    assert seen['command'] == ['py.exe', 'run.py']


def test_a_crash_is_announced_rather_than_papered_over():
    """A supervisor that relaunches in silence is worse than none: the app looks
    fine, the crash never gets reported, and the bug lives forever. Every
    unexpected death has to say so, with the exit code that identifies it."""
    notices = []
    runner = make_runner([3221225477, 0])

    supervise.supervise(runner, sleep=lambda _s: None, notify=notices.append)

    assert notices, 'the crash was swallowed'
    assert '3221225477' in ' '.join(notices)


def test_an_intentional_restart_is_not_reported_as_a_crash():
    """Update & restart is a normal event. Calling it a crash would train the
    user to ignore the message that matters."""
    notices = []
    runner = make_runner([supervise.RESTART_EXIT, 0])

    supervise.supervise(runner, sleep=lambda _s: None, notify=notices.append)

    assert not any('crash' in n.lower() for n in notices), notices


def test_ctrl_c_closes_the_app_instead_of_restarting_it():
    """Ctrl+C in the console reaches the CHILD too, which then dies with a
    non-zero code. Treated as a crash, that makes the app immortal: every
    attempt to close it brings it straight back. The supervisor has to read
    Windows' control-C exit status as "the user asked to stop"."""
    runner = make_runner([supervise.CONTROL_C_EXIT, 0])

    supervise.supervise(runner, sleep=lambda _s: None, notify=lambda _m: None)

    assert len(runner.calls) == 1, 'Ctrl+C must not be answered with a relaunch'


def test_interrupting_the_supervisor_itself_stops_everything():
    """The same keystroke also raises KeyboardInterrupt in this process. It must
    end the loop rather than escape as a traceback over the user's console."""
    def run_once():
        raise KeyboardInterrupt

    result = supervise.supervise(run_once, sleep=lambda _s: None,
                                 notify=lambda _m: None)

    assert result == supervise.CLEAN_EXIT


def test_the_browser_opens_on_the_first_launch_only():
    """`start.bat` asks the backend to open a tab. Left alone, every relaunch
    would open another one — so a crash loop that used to leave the app down now
    leaves the user with a pile of windows instead. Only the first launch gets to
    open one."""
    launches = []

    def fake_spawn(_command, child_env):
        launches.append(child_env.get('LDS_OPEN_BROWSER'))
        return 3221225477 if len(launches) < 3 else 0

    launch = supervise.backend_launcher('py.exe', 'run.py',
                                        env={'LDS_OPEN_BROWSER': '1'},
                                        spawn=fake_spawn)
    supervise.supervise(launch, sleep=lambda _s: None, notify=lambda _m: None)

    assert launches == ['1', '0', '0'], launches


def test_the_crash_notice_is_flushed_and_not_left_in_a_buffer():
    """Proven necessary by a live run: the supervisor relaunched the backend
    correctly and printed NOTHING. Python block-buffers stdout as soon as it is
    redirected to a file (start.bat piped to a log, a service wrapper), so the
    notice sat in a buffer of a process that never exits. Asserting `notify` was
    CALLED measured a proxy; what matters is that the bytes leave."""
    class RecordingStream:
        def __init__(self):
            self.written = []
            self.flushed_after = None

        def write(self, text):
            self.written.append(text)

        def flush(self):
            self.flushed_after = len(self.written)

    stream = RecordingStream()

    supervise.console_notice('boom', stream=stream)

    assert ''.join(stream.written).startswith('boom')
    assert stream.flushed_after == len(stream.written), 'written but never flushed'


# --- which launch paths supervise, pinned so the answer cannot drift ------------

def _repo_file(rel):
    from pathlib import Path
    return (Path(__file__).resolve().parents[2] / rel).read_text(encoding='utf-8')


def test_the_windows_double_click_runs_the_supervisor_and_can_opt_out():
    bat = _repo_file('start.bat')
    assert 'backend\supervise.py' in bat
    assert 'LDS_SUPERVISE' in bat, 'there must be a way to run the backend directly'


def test_docker_is_not_given_a_second_supervisor():
    """It already loops on its own, honouring the same exit 75. A second one
    would be two relaunchers on one port — the exact failure LDS_RESTART_MODE
    exists to prevent."""
    launch = _repo_file('packaging/docker/studio_launch.sh')
    assert 'while true' in launch and '-eq 75' in launch
    assert 'supervise.py' not in launch


def test_the_portable_bundle_stays_out_until_its_quit_button_can_cope():
    """Not an oversight — a documented gap. packaging/launcher.py terminates
    whatever it spawned when you press Quit, so putting the supervisor there
    would orphan the backend: an invisible server holding the port with no
    window left to stop it. Flip this test the day the launcher learns to
    relaunch AND report in its own status window."""
    launcher = _repo_file('packaging/launcher.py')
    assert 'supervise.py' not in launcher
    assert 'proc.terminate()' in launcher, 'the reason above rests on this line'
    # …and the reason is written where the next reader will look.
    assert 'packaging/launcher.py' in _repo_file('backend/supervise.py')
