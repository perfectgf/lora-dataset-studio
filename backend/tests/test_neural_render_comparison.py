"""⇔ The comparison as ONE file — the command, and what must not be in it.

No ffmpeg runs here: what is worth pinning is the argv, because two of its flags
are contracts rather than choices. `-map_metadata -1` is the one that matters
most — a studio clip carries ComfyUI's entire workflow in its `comment` tag,
absolute paths included, and this file is built to be handed to other people.
"""
import pytest

from app.services import neural_render as nr


def argv(**kw):
    kw.setdefault('left_label', 'Original')
    kw.setdefault('right_label', 'Neural render (DLSS 5)')
    kw.setdefault('ffmpeg', 'ffmpeg')
    return nr.comparison_argv('left.mp4', 'right.mp4', 'out.mp4', **kw)


def test_the_export_carries_no_metadata_from_the_source_clip():
    """The privacy contract of this file, and the reason it exists as a test:
    without the flag, ffmpeg copies the source's tags — and a studio clip's
    comment is the whole ComfyUI workflow, machine paths and all."""
    a = argv()
    assert '-map_metadata' in a
    assert a[a.index('-map_metadata') + 1] == '-1'


def test_the_two_clips_are_stacked_and_the_left_one_brings_the_sound():
    a = argv()
    graph = a[a.index('-filter_complex') + 1]
    assert 'hstack=inputs=2' in graph
    # `?` so a clip with no audio track is not a failure.
    assert '0:a?' in a


def test_the_pair_ends_with_the_SHORTER_clip_and_that_lives_on_the_filter():
    """`-shortest` alone does not do this, which is what the first version of
    this file asserted and the commit message claimed. Measured on a 5.17 s
    against a 2 s clip: the output ran the full 5.17 s with the short pane
    frozen on its last frame, because hstack pads the short input before the
    muxer — where `-shortest` lives — ever sees it."""
    a = argv()
    graph = a[a.index('-filter_complex') + 1]
    assert 'hstack=inputs=2:shortest=1' in graph, \
        'without shortest=1 on the filter the short side freezes to the end'
    # Kept for the audio stream, which the muxer option does reach.
    assert '-shortest' in a
    # Both shapes carry it — labelled and unlabelled.
    plain = argv(font=None)[argv(font=None).index('-filter_complex') + 1]
    assert plain == '[0:v][1:v]hstack=inputs=2:shortest=1[v]'


def test_a_windows_font_path_survives_all_three_parsers():
    """Measured on the bundled ffmpeg 7.1: `C\\\\:/…` parses, `C\\:/…` does not
    (`No option name near '/Windows/…'`). Backslashes become forward slashes,
    and a path without a colon comes back untouched."""
    assert nr.graph_value(r'C:\Windows\Fonts\arial.ttf') == r'C\\:/Windows/Fonts/arial.ttf'
    assert nr.graph_value('/usr/share/fonts/TTF/DejaVuSans.ttf') == \
        '/usr/share/fonts/TTF/DejaVuSans.ttf'


def test_without_a_font_the_panes_are_unlabelled_rather_than_unbuilt():
    """No font ships with this app, so the labels are a bonus: a machine with
    none of the candidates gets the comparison, just without captions."""
    plain = argv(font=None)[argv(font=None).index('-filter_complex') + 1]
    assert plain == '[0:v][1:v]hstack=inputs=2:shortest=1[v]'
    assert 'drawtext' not in plain
    labelled = argv(font='/fonts/x.ttf')[argv(font='/fonts/x.ttf').index('-filter_complex') + 1]
    assert labelled.count('drawtext') == 2
    assert "text='Original'" in labelled


def test_a_label_never_ends_its_own_option():
    """The labels are ours, not the user's, so the three characters that would
    break out of the option are dropped rather than escaped."""
    graph = nr.label_filter("it's:a\\ label", '/fonts/x.ttf')
    assert "text='its a label'" in graph
    assert "\\" not in graph.split("text='")[1].split("'")[0]


def test_a_missing_clip_is_a_sentence_not_a_traceback(tmp_path):
    only = tmp_path / 'one.mp4'
    only.write_bytes(b'not really a video')
    with pytest.raises(nr.NeuralRenderError, match='right clip'):
        nr.build_comparison(str(only), str(tmp_path / 'gone.mp4'),
                            left_label='Original', right_label='Neural render (DLSS 5)')
    with pytest.raises(nr.NeuralRenderError, match='left clip'):
        nr.build_comparison(str(tmp_path / 'gone.mp4'), str(only),
                            left_label='Original', right_label='Neural render (DLSS 5)')


def test_the_font_list_is_only_files_that_exist(monkeypatch):
    monkeypatch.setattr(nr.os.path, 'isfile', lambda p: p == nr.FONT_CANDIDATES[-1])
    assert nr.comparison_font() == nr.FONT_CANDIDATES[-1]
    monkeypatch.setattr(nr.os.path, 'isfile', lambda p: False)
    assert nr.comparison_font() is None


# ── the two routes: one per surface, refusing for the same reason ───────────

def test_the_studio_route_refuses_a_clip_that_is_not_a_render(app, client, tmp_path,
                                                              monkeypatch):
    from app.extensions import db
    from app.models import VideoTestClip
    from app.services import video_test_studio as vts
    monkeypatch.setattr(vts, 'clips_dir', lambda create=True: str(tmp_path))
    with app.app_context():
        plain = VideoTestClip(status='done', filename='plain.mp4', mode='i2v')
        db.session.add(plain)
        db.session.commit()
        plain_id = plain.id
        orphan = VideoTestClip(status='done', filename='render.mp4', mode='i2v',
                               nr_of=plain_id + 999)
        db.session.add(orphan)
        db.session.commit()
        orphan_id = orphan.id
    # A clip nobody rendered has no second side.
    res = client.get(f'/api/video-studio/clip/{plain_id}/comparison')
    assert res.status_code == 404 and 'not a neural render' in res.get_json()['error']
    # A render whose source row is gone says THAT, rather than half a comparison.
    res = client.get(f'/api/video-studio/clip/{orphan_id}/comparison')
    assert res.status_code == 404 and 'came from is gone' in res.get_json()['error']


def test_the_dataset_route_refuses_a_clip_that_plays_its_original(client, monkeypatch):
    """Same 404 and the same reason as the /original route it sits beside: no
    backup means the clip on disk IS the original."""
    monkeypatch.setattr(nr, 'original_clip_path', lambda *a, **k: None)
    res = client.get('/api/video-dataset/1/clip/1/comparison')
    assert res.status_code == 404 and 'nothing to compare' in res.get_json()['error']


# ── one encode at a time ────────────────────────────────────────────────────

def test_a_second_comparison_is_refused_while_one_is_encoding(tmp_path, monkeypatch):
    """Six of these at once took 17.4 s each instead of 2.1 s (measured): six
    ffmpeg processes divide a machine rather than share it, and each finished
    file waits in memory. So the second caller is told to come back — the answer
    the timeline GIF next door already gives."""
    clip = tmp_path / 'a.mp4'
    clip.write_bytes(b'x')
    monkeypatch.setattr(nr.ffmpeg_tools, 'ffmpeg_path', lambda: 'ffmpeg')   # the runner has none
    assert nr._COMPARISON_GATE.acquire(blocking=False)
    try:
        with pytest.raises(nr.ComparisonBusyError, match='try again'):
            nr.build_comparison(str(clip), str(clip), left_label='a', right_label='b')
    finally:
        nr._COMPARISON_GATE.release()
    # …and the gate is free again: a refusal must not consume the slot.
    assert nr._COMPARISON_GATE.acquire(blocking=False)
    nr._COMPARISON_GATE.release()


def test_the_gate_is_released_even_when_ffmpeg_fails(tmp_path, monkeypatch):
    """The slot is held in a `finally`, so a broken encode cannot wedge the
    route shut until a restart."""
    clip = tmp_path / 'a.mp4'
    clip.write_bytes(b'not a video')
    monkeypatch.setattr(nr.ffmpeg_tools, 'ffmpeg_path', lambda: 'ffmpeg')
    with pytest.raises(nr.NeuralRenderError):
        nr.build_comparison(str(clip), str(clip), left_label='a', right_label='b',
                            ffmpeg='definitely-not-an-encoder')
    assert nr._COMPARISON_GATE.acquire(blocking=False), 'the slot leaked on failure'
    nr._COMPARISON_GATE.release()


def test_busy_and_too_large_are_their_own_answers(client, monkeypatch):
    """429 with a Retry-After, and 413 — not a flat 400 that reads as "your
    request was wrong" for two conditions that are neither."""
    monkeypatch.setattr(nr, 'original_clip_path', lambda *a, **k: 'orig.mp4')
    from app.services import video_bank_service as svc
    monkeypatch.setattr(svc, 'dataset_clip_media_path', lambda *a, **k: 'render.mp4')

    def busy(*a, **k):
        raise nr.ComparisonBusyError('another comparison is being built')
    monkeypatch.setattr(nr, 'build_comparison', busy)
    res = client.get('/api/video-dataset/1/clip/1/comparison')
    assert res.status_code == 429 and res.headers.get('Retry-After') == '5'

    def big(*a, **k):
        raise nr.ComparisonTooLargeError('the comparison came out at 900 MB')
    monkeypatch.setattr(nr, 'build_comparison', big)
    res = client.get('/api/video-dataset/1/clip/1/comparison')
    assert res.status_code == 413


def test_a_temp_dir_that_cannot_be_made_does_not_keep_the_slot(tmp_path, monkeypatch):
    """The slot leaked here for one commit: `mkdtemp` sat between the acquire and
    the try, so a full disk walked out holding it and every later export answered
    429 until a restart. On this machine C: goes under 10 GB regularly, so this
    is a Tuesday, not a thought experiment."""
    clip = tmp_path / 'a.mp4'
    clip.write_bytes(b'x')

    def no_space(*a, **k):
        raise OSError(28, 'No space left on device')
    monkeypatch.setattr(nr.ffmpeg_tools, 'ffmpeg_path', lambda: 'ffmpeg')   # the runner has none
    monkeypatch.setattr(nr.tempfile, 'mkdtemp', no_space)
    with pytest.raises(OSError):
        nr.build_comparison(str(clip), str(clip), left_label='a', right_label='b')
    monkeypatch.undo()
    # The next caller must not be told the machine is busy.
    assert nr._COMPARISON_GATE.acquire(blocking=False), \
        'the slot was never released — every later export would answer 429'
    nr._COMPARISON_GATE.release()
