"""Registration and callbacks with explicit inert host interfaces."""
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import lds_video

ROOT = Path(__file__).resolve().parents[1]


def test_registration_owns_only_public_video_and_never_live(monkeypatch):
    manifest = json.loads((ROOT / 'plugin.json').read_text(encoding='utf8'))
    installs, downloads, probes, blueprints, handlers, hooks = {}, {}, {}, [], {}, {}
    for name, attrs in {
        'lds_video.downloads': {'H3_DOWNLOADS': {a: {'url': 'https://example.invalid/model', 'dest': 'models/file', 'min_free_gb': 1, 'min_bytes': 1} for a in manifest['owns']['install_actions'] if a.startswith('h3_')}},
        'lds_video.neural_render': {'install_bridge': lambda **kw: None},
        'lds_video.probes': {'PROBES': {k: lambda: False for k in manifest['owns']['probes']}, 'video_host_ready': lambda: False},
        'lds_video.routes.video_bank': {'bp': object()},
        'lds_video.routes.video_datasets': {'bp': object()},
        'lds_video.routes.video_studio': {'bp': object()},
        'lds_video.video_reference_catalog': {'reference_downloads': lambda: {}},
        'lds_video.video_references': {'MAX_UPLOAD_BYTES': 1024},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    ctx = SimpleNamespace(dir=ROOT,
        register_node_pack=lambda key: installs.update({key: {'node_pack': True}}),
        register_request_limit=lambda *args: None,
        register_install_action=lambda key, **kw: installs.update({key: kw}),
        register_model_download=lambda key, **kw: downloads.update({key: kw}),
        register_probe=lambda key, fn: probes.update({key: fn}),
        register_blueprint=lambda bp, **kw: blueprints.append(kw['url_prefix']),
        register_job_handler=lambda key, fn, **kw: handlers.update({key: fn}),
        register_hook=lambda key, fn: hooks.update({key: fn}))
    lds_video.register(ctx)
    assert set(installs) | set(downloads) == set(manifest['owns']['install_actions'])
    assert set(probes) == set(manifest['owns']['probes'])
    assert list(handlers) == ['is_video_test']
    assert set(hooks) == {'job_queue.keep_inputs', 'job_queue.unlinked_results'}
    assert blueprints == ['/api', '/api', '/api/video-studio']
    assert manifest['requires'] == []
    assert all('live' not in key for key in installs)


def test_video_completion_and_cleanup_use_own_clip_service(monkeypatch):
    calls = []
    service = ModuleType('lds_video.video_test_studio')
    service.link_completed_clip = lambda *args, **kw: calls.append((args, kw))
    service.keep_frames_of_existing_clips = lambda: (2, {'frame-retained'})
    monkeypatch.setitem(sys.modules, service.__name__, service)
    monkeypatch.setattr(lds_video, 'video_test_studio', service, raising=False)
    lds_video._video_test_done('job', 'clip.mp4', failed=True, reason='failure', metadata={'ignored': True})
    assert calls == [(('job', 'clip.mp4'), {'failed': True, 'reason': 'failure'})]
    assert lds_video._keep_inputs({'other-frame'}) == {'other-frame', 'frame-retained'}
