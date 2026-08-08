"""Make the video extra READY for a test that is not about the video extra.

Every `/api/video-bank/...` route that decodes, encodes or detects opens with the
same gate: `capabilities.probe_video()`, and a 503 carrying `detail` when a piece
is missing. That gate reads the MACHINE — PyAV importable, ffmpeg on PATH,
transnetv2 in the scoring interpreter — so a route test's verdict depends on what
the box running it happens to have installed.

On the maintainer's machine everything is present and the tests pass. CI installs
`requirements-dev.txt` and nothing else, so the gate fires first and five tests
that never mention the video extra fail on it: a 202 arrives as 503, a 404 as 503,
and the one test that asserts its OWN 503 gets the wrong sentence. That surfaced
on a release tag, because CI path-gates the heavy job and a small commit never
runs it.

`pytest.importorskip` would be the wrong tool: it would delete the coverage in the
one environment where it is cheapest to run. These tests do not need PyAV, ffmpeg
or a GPU — every service call underneath is already stubbed. They only need the
gate to stop answering for the machine.

Tests that are ABOUT a missing piece override this with their own `monkeypatch`
(see test_video_advisory_routes.py, which sets `decode: False` to assert the very
sentence this fixture suppresses). Last patch wins, so nothing here weakens them.
"""
import pytest


def detect_source_stub(clips, probs=None, fps=30.0):
    """A `video_bank_service._detect_source` stand-in for a test whose subject
    is anything BUT the detector.

    Built here rather than inline in each test file so the shape of that seam
    lives in ONE place: it grew a probability vector once, and six copies of a
    hand-written stub would have gone on agreeing with the old shape while
    production had already moved past it."""
    def run(_path, fps_native=None, **_options):
        return {'clips': list(clips), 'probs': probs,
                'fps_native': fps_native or fps,
                'frame_count': len(probs['single']) if probs else None}
    return run


@pytest.fixture(autouse=True)
def video_extra_ready(monkeypatch):
    """Report every video piece as installed, for this module only."""
    monkeypatch.setattr('app.capabilities.probe_video', lambda: {
        'ok': True, 'decode': True, 'detect': True, 'encode': True,
        'detail': 'video extra ready',
    })
