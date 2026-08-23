"""What "about two hours left" is allowed to mean.

TIME IS INJECTED, ALWAYS. Not one test here sleeps: a rate test that sleeps is a
rate test that goes flaky the moment the machine is busy, and a machine running
several bank passes is exactly when this code is read. Every scenario below is a
list of (instant, done) handed straight to the estimator.

The scenarios are the measured ones, not invented ones:
* ✨ Score resuming from cache — `resuming — 20847 of 21220 already cached`,
  twenty thousand rows in about two seconds, then 373 images on the GPU for an
  hour and a half.
* a steady GPU pass at 6.4 images/s, which predicted 1 h 27 and took 1 h 27.
* the style grouping, which publishes done=0/total=0 for 181 s.
"""
import pytest

from app.services import bank_jobs, job_eta


def feed(state, events, total, start=None):
    """Replay (instant, done) into the estimator; return the last read."""
    for at, done in events:
        job_eta.observe(state, done, total, at)
    return job_eta.read(state, events[-1][0] if start is None else start)


def steady(rate, count, total, first=0, t0=0.0):
    """`count` items arriving at `rate` per second."""
    return [(t0 + i / rate, first + i) for i in range(count + 1)]


def test_says_nothing_before_the_window_floor():
    """Twelve seconds of a pass is not a speed, however many items it saw."""
    state = job_eta.new_state()
    result = feed(state, steady(6.4, 76, 37800, first=12939), 37800)
    assert result[0] == job_eta.ETA_ESTIMATING
    assert result[1] is None


def test_a_steady_pass_publishes_and_is_right():
    state = job_eta.new_state()
    events = steady(6.4, 860, 37800, first=12939)
    result = feed(state, events, 37800)
    assert result[0] == job_eta.ETA_READY
    remaining = 37800 - events[-1][1]
    truth = remaining / 6.4
    assert result[1] == pytest.approx(truth, rel=0.05)


def test_the_estimate_waits_for_the_stability_streak():
    """A number is not published the instant it can be computed.

    The window floor is 20 s; the streak is another 15. Publishing at 20 s would
    put "3 hours" on screen off two dozen samples, and the whole value of the
    figure is that it does not then become "20 minutes".
    """
    state = job_eta.new_state()
    events = steady(6.4, 860, 37800, first=12939)
    at_25 = [e for e in events if e[0] <= 25.0]
    assert feed(job_eta.new_state(), at_25, 37800)[0] == job_eta.ETA_ESTIMATING
    at_40 = [e for e in events if e[0] <= 40.0]
    assert feed(state, at_40, 37800)[0] == job_eta.ETA_READY


def resume_from_cache():
    """The measured ✨ Score resume: a burst, then the real work."""
    events = [(0.0, 0)]
    for i in range(1, 11):                    # 20 847 rows over two seconds
        events.append((i * 0.2, round(20847 * i / 10)))
    rate = 373 / 5220.0                       # 373 images in 1 h 27
    at, done = 2.0, 20847
    while at < 400.0:
        at += 1 / rate
        done += 1
        events.append((at, done))
    return events, rate


def test_a_cache_burst_never_promises_a_few_seconds():
    """THE trap. An average since the start says 1.8 s here; the truth is 5122.

    While the burst is still inside the window the answer must be silence, not a
    number computed across two speeds that belongs to neither.
    """
    events, _ = resume_from_cache()
    state = job_eta.new_state()
    # Asserted at EVERY report, not at one convenient instant: the estimate only
    # has to be wrong once to teach the user the figure means nothing. Ninety
    # seconds is well past the point where the window is long enough and full
    # enough to compute something — a build with no burst guard speaks here.
    for at, done in events:
        if at > 90.0:
            break
        job_eta.observe(state, done, 21220, at)
        assert job_eta.read(state, at)[0] != job_eta.ETA_READY, (
            f'promised a duration at t={at:.1f} across a cache burst')

    # What the naive average would have said in that window, for the record.
    elapsed, done = [e for e in events if e[0] <= 90.0][-1]
    naive = (21220 - done) / (done / elapsed)
    assert naive < 5.0


def test_after_the_burst_leaves_the_window_the_estimate_is_the_real_one():
    events, rate = resume_from_cache()
    state = job_eta.new_state()
    result = feed(state, events, 21220)
    assert result[0] == job_eta.ETA_READY
    truth = (21220 - events[-1][1]) / rate
    assert result[1] == pytest.approx(truth, rel=0.1)
    # ...and it is an hour-plus, not the couple of seconds the average promised.
    assert result[1] > 3600


def test_nothing_to_count_gets_silence_not_a_guess():
    """The style grouping publishes done=0/total=0 for three minutes."""
    state = job_eta.new_state()
    result = feed(state, [(0.0, 0), (60.0, 0), (181.0, 0)], 0)
    assert result[0] == job_eta.ETA_NONE
    assert result[1] is None


def test_a_phase_change_clears_the_window_and_scopes_the_answer():
    """✨ Score: inference, then ~21 000 rows written, then style grouping.

    Seconds per inference say nothing about seconds per row. A duration measured
    across that boundary is wrong by construction, so the window is dropped —
    and from then on the answer is explicitly about the current step.
    """
    state = job_eta.new_state()
    feed(state, steady(2.0, 300, 400), 400)
    assert job_eta.read(state, 150.0)[0] == job_eta.ETA_READY
    assert job_eta.read(state, 150.0)[2] == 'job'

    job_eta.observe(state, 0, 21220, 151.0)      # the write-back begins
    ready, seconds, scope = job_eta.read(state, 151.0)
    assert ready == job_eta.ETA_ESTIMATING
    assert seconds is None
    assert scope == 'phase'


def test_a_phase_with_the_same_total_still_resets_when_done_goes_back():
    """Two phases can share a total; the counter restarting is the giveaway."""
    state = job_eta.new_state()
    feed(state, steady(2.0, 300, 500), 500)
    assert job_eta.read(state, 150.0)[0] == job_eta.ETA_READY
    job_eta.observe(state, 0, 500, 151.0)
    assert job_eta.read(state, 151.0) == (job_eta.ETA_ESTIMATING, None, 'phase')


def test_the_detail_churning_mid_phase_does_not_reset_anything():
    """The framing pass rewrites its detail when Ollama's window expires and
    again when it recovers. Phase detection reads the NUMBERS, never that text —
    which is why detail is not even passed in here."""
    state = job_eta.new_state()
    feed(state, steady(2.0, 300, 5000), 5000)
    assert job_eta.read(state, 150.0)[0] == job_eta.ETA_READY
    assert state['phases'] == 1


def test_a_wedged_pass_stops_promising():
    state = job_eta.new_state()
    events = steady(1.0, 200, 4000)
    assert feed(state, events, 4000)[0] == job_eta.ETA_READY
    # A literal silence, not `STALE_SAMPLE_SECONDS + 1`: expressing the wait in
    # terms of the constant means widening the constant also widens the test,
    # and the assertion follows the bug instead of catching it.
    assert job_eta.read(state, 400.0)[0] == job_eta.ETA_ESTIMATING


def test_a_slowing_pass_grows_its_estimate_instead_of_holding_the_old_one():
    """The window is the only source; a pass that halves in speed says so."""
    state = job_eta.new_state()
    events = steady(4.0, 400, 5000)
    feed(state, events, 5000)
    fast = job_eta.read(state, events[-1][0])[1]
    slow_start, done = events[-1]
    slow = [(slow_start + i / 1.0, done + i) for i in range(1, 121)]
    feed(state, slow, 5000)
    result = job_eta.read(state, slow[-1][0])
    assert result[0] == job_eta.ETA_READY
    assert result[1] > fast * 2


def test_a_pass_that_decays_to_a_crawl_keeps_refreshing_its_promise():
    """Below the item floor the estimate must still MOVE.

    The floor is there so the first number is not published off a burst. Applied
    after publication it would do the opposite of its job: a pass falling to two
    items a minute would stop qualifying, and the twenty-minute promise made
    while it was fast would sit on screen through the hour that followed.
    """
    state = job_eta.new_state()
    events = steady(4.0, 400, 5000)
    feed(state, events, 5000)
    fast = job_eta.read(state, events[-1][0])[1]
    at, done = events[-1]
    crawl = [(at + i * 25.0, done + i) for i in range(1, 13)]   # ~2.4 items/min
    feed(state, crawl, 5000)
    result = job_eta.read(state, crawl[-1][0])
    assert result[0] == job_eta.ETA_READY
    assert result[1] > fast * 10, (
        f'still promising {result[1]:.0f} s after slowing 100x (was {fast:.0f})')


def test_a_finished_job_carries_no_estimate(app):
    """"About twenty minutes left" under a completed pass is the loudest way to
    say the number means nothing."""
    bank_jobs.reset()
    bank_jobs.start(app, 8811, 'scan', lambda j: bank_jobs.progress(
        j, done=5, total=10), total=10)
    snap = bank_jobs.get(8811)
    assert snap['finished'] is True
    assert snap['eta_state'] == job_eta.ETA_NONE
    assert snap['eta_seconds'] is None
    bank_jobs.reset()


def test_the_snapshot_carries_the_estimate_fields(app):
    """The three keys the bank payload ships, whatever the pass."""
    bank_jobs.reset()
    holder = {}

    def work(job):
        bank_jobs.progress(job, done=0, total=100, detail='scanning')
        holder['snap'] = bank_jobs.get(8812)

    bank_jobs.start(app, 8812, 'scan', work, total=100)
    snap = holder['snap']
    assert snap['eta_state'] == job_eta.ETA_ESTIMATING
    assert snap['eta_seconds'] is None
    assert snap['eta_scope'] == 'job'
    bank_jobs.reset()


def test_bump_feeds_the_estimator(app):
    """Most passes never call progress() per item — they call bump()."""
    bank_jobs.reset()
    seen = {}

    def work(job):
        for _ in range(3):
            bank_jobs.bump(job)
        seen['samples'] = len(job['_eta']['samples'])
        seen['done'] = job['_eta']['done']

    bank_jobs.start(app, 8813, 'faces', work, total=50)
    assert seen['done'] == 3
    assert seen['samples'] >= 1
    bank_jobs.reset()
