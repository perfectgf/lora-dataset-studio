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
    # The pair may differ in length; the file ends with the shorter.
    assert '-shortest' in a


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
    assert plain == '[0:v][1:v]hstack=inputs=2[v]'
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
