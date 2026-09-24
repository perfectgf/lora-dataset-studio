"""Completing a clip must not extract or publish unused still images."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from lds_video import video_test_studio as studio


@pytest.mark.parametrize('status,parent', [('done', None), ('done', 4), ('failed', None)])
def test_completion_only_joins_continuations(monkeypatch, status, parent):
    clip = SimpleNamespace(status=status, continues_of=parent)
    monkeypatch.setitem(sys.modules, 'lds_video.models', SimpleNamespace(
        db=SimpleNamespace(session=SimpleNamespace(get=lambda *args: clip)), VideoTestClip=object))
    extract = Mock(side_effect=AssertionError('completion extracted a frame'))
    publish = Mock(side_effect=AssertionError('completion opened the image library'))
    join = Mock()
    monkeypatch.setattr(studio, 'last_frame_png', extract)
    monkeypatch.setattr(studio, 'frames_dataset_id', publish)
    monkeypatch.setattr(studio, '_join_continuation', join)
    studio.postprocess_completed_clip(8)
    extract.assert_not_called()
    publish.assert_not_called()
    assert join.call_args_list == ([((8,), {})] if parent and status == 'done' else [])


def test_explicit_frame_request_extracts_once_then_reuses_cache(monkeypatch, tmp_path):
    from lds_sdk import video_host

    (tmp_path / 'clip.mp4').write_bytes(b'video fixture')
    clip = SimpleNamespace(user_id='local', status='done', filename='clip.mp4')
    query = SimpleNamespace(filter_by=lambda **kwargs: SimpleNamespace(first=lambda: clip))
    monkeypatch.setitem(sys.modules, 'lds_video.models', SimpleNamespace(VideoTestClip=SimpleNamespace(query=query)))
    monkeypatch.setitem(sys.modules, 'lds_sdk.video_host.config', SimpleNamespace(LOCAL_USER='local'))
    monkeypatch.setattr(video_host, 'ffmpeg_tools', SimpleNamespace(ffmpeg_path=lambda: 'ffmpeg'), raising=False)
    monkeypatch.setattr(studio, 'clips_dir', lambda: tmp_path)
    monkeypatch.setattr(studio, 'last_frame_command', lambda _exe, _src, dst: dst)

    def extract(destination, **kwargs):
        from pathlib import Path
        Path(destination).write_bytes(b'frame fixture')
        return SimpleNamespace(returncode=0)

    run = Mock(side_effect=extract)
    monkeypatch.setattr(studio, '_run_ffmpeg', run)
    first = studio.last_frame_png(8)
    assert studio.last_frame_png(8) == first
    assert run.call_count == 1
