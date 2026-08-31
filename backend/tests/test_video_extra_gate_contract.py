"""Every route test that talks to /api/video-bank/ must neutralise the machine gate.

`capabilities.probe_video()` reads the MACHINE (PyAV, ffmpeg, the scoring
interpreter), and every heavy video route opens with it. A route test that does
not import `_video_extra.video_extra_ready` (or patch `probe_video` itself)
passes on the maintainer's box and 503s on CI — which is exactly how the
v2026.08.31 release tag went red on one test that never mentioned the video
extra. Twice is a pattern; this pins it as text so the third time fails here,
in seconds, instead of thirty-five minutes into a release job.
"""
import pathlib
import re

HERE = pathlib.Path(__file__).parent
# Only the routes whose handler opens with `capabilities.probe_video()` — read
# off routes/video_bank.py: measure, embed, caption, watermark, safezone, camera,
# aicheck. create/refresh/clips/dedup/promote/... carry no machine gate, and a
# contract that flagged them would teach people to ignore it.
ROUTE = re.compile(r"""/api/video-bank/[^'"]*/(measure|embed|caption|watermark|safezone|camera|aicheck)['"]""")
NEUTRALISED = re.compile(r"from _video_extra import|probe_video")


def test_every_video_bank_route_test_neutralises_the_machine_gate():
    offenders = []
    for path in sorted(HERE.glob('test_*.py')):
        if path.name == pathlib.Path(__file__).name:
            continue
        src = path.read_text(encoding='utf-8')
        if ROUTE.search(src) and not NEUTRALISED.search(src):
            offenders.append(path.name)
    assert not offenders, (
        'these test files post to a machine-gated /api/video-bank/ route without importing '
        '_video_extra.video_extra_ready (or patching capabilities.probe_video): '
        + ', '.join(offenders))
