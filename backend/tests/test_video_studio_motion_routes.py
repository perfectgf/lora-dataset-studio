"""The ✨ Motion endpoints, seen from the panel that calls them.

The service has its own file; what only the route can answer is whether every
piece the panel sends actually ARRIVES — the frame, the instruction, the chosen
model, the clip length. A parameter that never reaches the service is invisible
to a service test and fails silently in front of the user, which is how the
enrichment button spent a day rewriting prompts with no idea what picture they
animate, and how ✨ Auto wrote the same beat for a 1 s clip and a 15 s one.
"""


def test_the_panel_s_pieces_reach_the_writer(client, monkeypatch):
    """image + instruction + model + the clip length, from ✨ Auto."""
    from app.services import video_motion_prompt as vmp
    seen = {}

    def fake(image_name, instruction=None, model=None, seconds=None, shots=1):
        seen.update(image=image_name, instruction=instruction, model=model,
                    seconds=seconds, shots=shots)
        return 'She lifts her gaze slowly toward the lens.'

    monkeypatch.setattr(vmp, 'suggest_from_frame', fake)
    r = client.post('/api/video-studio/motion/suggest',
                    json={'image': 'staged_1.png', 'instruction': 'make her jump',
                          'model': 'qwen3-vl:8b', 'seconds': 15.04})
    assert r.status_code == 200
    assert r.get_json()['prompt'].startswith('She lifts her gaze')
    assert seen == {'image': 'staged_1.png', 'instruction': 'make her jump',
                    'model': 'qwen3-vl:8b', 'seconds': 15.04, 'shots': 1}
    # A body without a length paces nothing — never a default the dials do
    # not show.
    client.post('/api/video-studio/motion/suggest', json={'image': 'staged_1.png'})
    assert seen['seconds'] is None


def test_the_enrichment_is_told_which_frame_the_clip_starts_from(client, monkeypatch):
    """Without the frame, "make her look out of the window" invents a window."""
    from app.services import video_motion_prompt as vmp
    seen = {}

    def fake(prompt, image=None, model=None, seconds=None, shots=1):
        seen.update(prompt=prompt, image=image, model=model, seconds=seconds,
                    shots=shots)
        return 'She turns her head slowly to the left.'

    monkeypatch.setattr(vmp, 'enhance', fake)
    r = client.post('/api/video-studio/motion/enhance',
                    json={'prompt': 'she turns', 'image': 'staged_1.png',
                          'model': 'qwen3-vl:8b', 'seconds': 2.29})
    assert r.status_code == 200
    assert seen == {'prompt': 'she turns', 'image': 'staged_1.png',
                    'model': 'qwen3-vl:8b', 'seconds': 2.29, 'shots': 1}
    # t2v has no frame, and that is not an error — the text alone is enriched.
    client.post('/api/video-studio/motion/enhance',
                json={'prompt': 'she turns', 'image': None})
    assert seen['image'] is None


def test_enrich_at_launch_writes_from_what_the_launch_carries(client, monkeypatch):
    """The launch body has no `seconds`, only the frame count that will render;
    the route converts it with the readback's own arithmetic (N-1 intervals at
    the target's fps) so the launch and the ✨ Enrich button pace the same
    clip. And the frame is named to the writer only when it will be animated:
    a text-to-video launch that still carries a stale staged name must not
    produce a prompt that references <Picture 1>."""
    from app.services import video_motion_prompt as vmp
    from app.services import video_test_studio as vts
    monkeypatch.setattr('app.capabilities.probe',
                        lambda *a, **k: {'comfyui': {'reachable': True}})
    monkeypatch.setattr('app.capabilities.probe_comfyui',   # the runner has no ComfyUI
                        lambda: {'ok': True, 'status': 'ok', 'detail': '', 'hint': ''})
    launched = {}
    monkeypatch.setattr(vts, 'enqueue_clip',
                        lambda user, **kw: launched.update(kw) or
                        {'clip_id': 1, 'seed': 1, 'frames': kw.get('frames')})
    seen = {}

    def fake(prompt, image=None, model=None, seconds=None, shots=1):
        seen.update(prompt=prompt, image=image, seconds=seconds, shots=shots)
        return 'integrated_multimodal_description: [Shot 1] She turns slowly.'

    monkeypatch.setattr(vmp, 'enhance', fake)
    r = client.post('/api/video-studio/generate',
                    json={'mode': 'i2v', 'image': 'staged_1.png', 'prompt': 'she turns',
                          'enhance': True, 'frames': 56})
    assert r.status_code == 200, r.get_json()
    assert seen['image'] == 'staged_1.png'
    assert abs(seen['seconds'] - 55 / 24) < 1e-9        # 56 frames at 24 fps
    # The clip row records the prompt that ran, not the one that was typed —
    # headed once for the encoder (the writer's answer here carries no header
    # of its own; the route adds it at generation).
    assert launched['prompt'] == vmp.inject_alignment_header(
        'integrated_multimodal_description: [Shot 1] She turns slowly.')

    client.post('/api/video-studio/generate',
                json={'mode': 't2v', 'image': 'staged_1.png', 'prompt': 'she turns',
                      'enhance': True, 'frames': 362})
    assert seen['image'] is None
    assert abs(seen['seconds'] - 361 / 24) < 1e-9


def test_a_refusal_arrives_as_a_sentence_not_a_stack_trace(client, monkeypatch):
    from app.services import video_motion_prompt as vmp

    def refuse(*a, **kw):
        raise ValueError('no local model to write it with — Ollama: not running')

    monkeypatch.setattr(vmp, 'suggest_from_frame', refuse)
    monkeypatch.setattr(vmp, 'enhance', refuse)
    for path, body in (('suggest', {'image': 'a.png'}), ('enhance', {'prompt': 'she turns'})):
        r = client.post(f'/api/video-studio/motion/{path}', json=body)
        assert r.status_code == 400
        assert 'Ollama: not running' in r.get_json()['error']


def test_the_model_window_lists_the_providers_own_and_saves_a_choice(client, monkeypatch):
    from app.services import video_motion_prompt as vmp
    monkeypatch.setattr(vmp, 'model_choices',
                        lambda: {'provider': 'ollama', 'label': 'Ollama',
                                 'reachable': True, 'current': '',
                                 'models': ['qwen3-vl:8b', 'llava:13b']})
    r = client.get('/api/video-studio/motion/models')
    assert r.status_code == 200
    assert r.get_json()['models'] == ['qwen3-vl:8b', 'llava:13b']

    saved = {}
    monkeypatch.setattr(vmp, 'set_model', lambda name: saved.setdefault('name', name) or name)
    r = client.put('/api/video-studio/motion/model', json={'model': 'qwen3-vl:8b'})
    assert r.status_code == 200
    assert r.get_json()['model'] == 'qwen3-vl:8b'
    assert saved['name'] == 'qwen3-vl:8b'


def test_the_fence_and_a_transport_failure_arrive_as_409s_not_as_a_bare_500(
        client, monkeypatch):
    """Both providers. The Ollama fence and the LM Studio one (a subclass, by
    design) carry the code the panel keys its banner and "unload" offer on; a
    plain transport failure is a 409 sentence without the code. Measured
    before the fix: the routes caught ValueError/TypeError only, so the fence
    — a RuntimeError — fell through as a 500 with nothing to show."""
    from app.services import video_motion_prompt as vmp
    from app.services.vision_lmstudio import LocalLmStudioFenceError
    from app.services.vision_ollama import LocalOllamaFenceError

    def refuse_with(exc):
        def _refuse(*a, **kw):
            raise exc
        return _refuse

    for exc in (LocalOllamaFenceError('the GPU is already in use outside LDS'),
                LocalLmStudioFenceError('the GPU is already in use outside LDS')):
        monkeypatch.setattr(vmp, 'suggest_from_frame', refuse_with(exc))
        monkeypatch.setattr(vmp, 'enhance', refuse_with(exc))
        for path, body in (('suggest', {'image': 'a.png'}),
                           ('enhance', {'prompt': 'she turns'})):
            r = client.post(f'/api/video-studio/motion/{path}', json=body)
            assert r.status_code == 409, (path, exc)
            assert r.get_json()['code'] == 'ollama_fence_blocked'
            assert 'already in use outside LDS' in r.get_json()['error']

    plain = refuse_with(RuntimeError('LM Studio did not answer'))
    monkeypatch.setattr(vmp, 'suggest_from_frame', plain)
    monkeypatch.setattr(vmp, 'enhance', plain)
    for path, body in (('suggest', {'image': 'a.png'}),
                       ('enhance', {'prompt': 'she turns'})):
        r = client.post(f'/api/video-studio/motion/{path}', json=body)
        assert r.status_code == 409
        assert 'code' not in r.get_json()
        assert 'LM Studio did not answer' in r.get_json()['error']


def test_a_launch_whose_enrichment_failed_still_launches_and_says_so(
        client, monkeypatch):
    """The user asked for a clip, not an essay: the fence (or any failure) on
    the enrichment must not refuse the launch — but the answer carries why it
    ran the un-enriched prompt, so the panel can say it instead of a clip
    that silently ignored the checkbox."""
    from app.services import video_motion_prompt as vmp
    from app.services import video_test_studio as vts
    from app.services.vision_ollama import LocalOllamaFenceError
    monkeypatch.setattr('app.capabilities.probe',
                        lambda *a, **k: {'comfyui': {'reachable': True}})
    monkeypatch.setattr('app.capabilities.probe_comfyui',   # the runner has no ComfyUI
                        lambda: {'ok': True, 'status': 'ok', 'detail': '', 'hint': ''})
    launched = {}
    monkeypatch.setattr(vts, 'enqueue_clip',
                        lambda user, **kw: launched.update(kw) or
                        {'clip_id': 1, 'seed': 1, 'frames': kw.get('frames')})

    def fenced(*a, **kw):
        raise LocalOllamaFenceError('the GPU is already in use outside LDS')

    monkeypatch.setattr(vmp, 'enhance', fenced)
    r = client.post('/api/video-studio/generate',
                    json={'mode': 'i2v', 'image': 'staged_1.png', 'prompt': 'she turns',
                          'enhance': True, 'frames': 56})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()['enrich_skipped'] == 'the GPU is already in use outside LDS'
    assert launched['prompt'] == vmp.inject_alignment_header('she turns')
    # The answer names the prompt that ran — the text as typed, header and
    # all — so a batch of start frames can send the rest of its clips on the
    # same one (the header injection is its own fixed point, so the same
    # text sent back is headed once, not twice).
    assert r.get_json()['prompt'] == launched['prompt']
    assert vmp.inject_alignment_header(r.get_json()['prompt']) == launched['prompt']

    # A launch whose enrichment worked carries no such key at all, and names
    # the rewrite: the second clip of a batch runs it with `enhance` dropped,
    # since the vision window refuses once this clip sits in the queue.
    monkeypatch.setattr(vmp, 'enhance', lambda *a, **kw: 'She turns slowly toward the lens.')
    r = client.post('/api/video-studio/generate',
                    json={'mode': 'i2v', 'image': 'staged_1.png', 'prompt': 'she turns',
                          'enhance': True, 'frames': 56})
    assert r.status_code == 200
    assert 'enrich_skipped' not in r.get_json()
    assert r.get_json()['prompt'] == launched['prompt']
    assert 'She turns slowly toward the lens.' in r.get_json()['prompt']


def test_the_enhancement_says_when_it_had_nothing_to_add(client, monkeypatch):
    """The model can hand the text back as it was — which, in the field, looks
    exactly like a request that worked. The flag is what lets the panel tell
    the two apart. (An UNUSABLE answer is not this case: the service raises
    and the route says so — the test below.)"""
    from app.services import video_motion_prompt as vmp
    monkeypatch.setattr(vmp, 'enhance', lambda prompt, **kw: prompt)
    r = client.post('/api/video-studio/motion/enhance', json={'prompt': '  she turns  '})
    assert r.status_code == 200
    assert r.get_json() == {'ok': True, 'prompt': '  she turns  ', 'unchanged': True}

    monkeypatch.setattr(vmp, 'enhance',
                        lambda prompt, **kw: 'She turns her head slowly to the left.')
    r = client.post('/api/video-studio/motion/enhance', json={'prompt': 'she turns'})
    assert r.get_json()['unchanged'] is False


def test_the_shot_count_reaches_the_writer_on_all_three_gestures(client, monkeypatch):
    """`shots` is in every body the panel sends; a route that read it on two of
    the three gestures would cut a 3-shot ✨ Auto into one shot at launch."""
    from app.services import video_motion_prompt as vmp
    from app.services import video_test_studio as vts
    monkeypatch.setattr('app.capabilities.probe',
                        lambda *a, **k: {'comfyui': {'reachable': True}})
    monkeypatch.setattr('app.capabilities.probe_comfyui',   # the runner has no ComfyUI
                        lambda: {'ok': True, 'status': 'ok', 'detail': '', 'hint': ''})
    monkeypatch.setattr(vts, 'enqueue_clip',
                        lambda user, **kw: {'clip_id': 1, 'seed': 1, 'frames': kw.get('frames')})
    seen = []
    monkeypatch.setattr(vmp, 'suggest_from_frame',
                        lambda image, **kw: seen.append(kw['shots']) or 'She turns slowly.')
    monkeypatch.setattr(vmp, 'enhance',
                        lambda prompt, **kw: seen.append(kw['shots']) or 'She turns slowly.')
    client.post('/api/video-studio/motion/suggest', json={'image': 'a.png', 'shots': 3})
    client.post('/api/video-studio/motion/enhance', json={'prompt': 'she turns', 'shots': 3})
    client.post('/api/video-studio/generate',
                json={'mode': 't2v', 'prompt': 'she turns', 'enhance': True,
                      'frames': 241, 'shots': 3})
    assert seen == [3, 3, 3]


def test_the_three_writers_run_inside_the_gpu_exclusive_vision_window(client, monkeypatch):
    """Parity with the image studio's twins (`/api/studio/describe` and
    `/enhance-prompt` both enter the window): a writer that ran outside it
    would fight a queued clip for VRAM — and H3 alone fills most of the card.
    The plain launch owes no window: it goes to ComfyUI's queue, whose worker
    already waits on `vision_in_progress`. Measured before the fix: no
    `gpu_exclusive` anywhere in the route module."""
    import contextlib

    from app.routes import video_studio as vsr
    from app.services import video_motion_prompt as vmp
    from app.services import video_test_studio as vts
    monkeypatch.setattr('app.capabilities.probe',
                        lambda *a, **k: {'comfyui': {'reachable': True}})
    monkeypatch.setattr('app.capabilities.probe_comfyui',   # the runner has no ComfyUI
                        lambda: {'ok': True, 'status': 'ok', 'detail': '', 'hint': ''})
    monkeypatch.setattr(vts, 'enqueue_clip',
                        lambda user, **kw: {'clip_id': 1, 'seed': 1, 'frames': kw.get('frames')})
    entered, state = [], {'inside': False}

    @contextlib.contextmanager
    def window(flag_ttl=300):
        entered.append(flag_ttl)
        state['inside'] = True
        try:
            yield
        finally:
            state['inside'] = False

    monkeypatch.setattr(vsr, 'gpu_exclusive_vision_window', window)
    wrote = []

    def writer(*a, **kw):
        # Inside, not merely around: the write is what the window protects.
        wrote.append(state['inside'])
        return 'She turns slowly.'

    monkeypatch.setattr(vmp, 'suggest_from_frame', writer)
    monkeypatch.setattr(vmp, 'enhance', writer)
    client.post('/api/video-studio/motion/suggest', json={'image': 'a.png'})
    client.post('/api/video-studio/motion/enhance', json={'prompt': 'she turns'})
    client.post('/api/video-studio/generate',
                json={'mode': 't2v', 'prompt': 'she turns', 'enhance': True, 'frames': 56})
    assert wrote == [True, True, True]
    # The same patience as the image studio's twins, on all three.
    assert entered == [600, 600, 600]

    r = client.post('/api/video-studio/generate',
                    json={'mode': 't2v', 'prompt': 'she turns', 'frames': 56})
    assert r.status_code == 200
    assert entered == [600, 600, 600], 'a plain launch must not take the vision window'


def test_a_queued_clip_refuses_the_buttons_with_its_reason_and_lets_the_launch_through(
        client, monkeypatch):
    """The REAL window, with ComfyUI reporting work in its queue (a clip
    already rendering is exactly that): the ✨ buttons answer 503 with the
    reason in `detail` — the bare "GPU busy" alone does not say what to wait
    for — and the launch still launches, un-enriched and saying why."""
    from app.services import video_motion_prompt as vmp
    from app.services import video_test_studio as vts
    from app.job_queue import queue_manager
    monkeypatch.setattr('app.capabilities.probe',
                        lambda *a, **k: {'comfyui': {'reachable': True}})
    monkeypatch.setattr('app.capabilities.probe_comfyui',   # the runner has no ComfyUI
                        lambda: {'ok': True, 'status': 'ok', 'detail': '', 'hint': ''})
    launched = {}
    monkeypatch.setattr(vts, 'enqueue_clip',
                        lambda user, **kw: launched.update(kw) or
                        {'clip_id': 1, 'seed': 1, 'frames': kw.get('frames')})
    monkeypatch.setattr(queue_manager, 'has_comfyui_work', lambda: True)
    calls = []
    monkeypatch.setattr(vmp, 'suggest_from_frame',
                        lambda *a, **kw: calls.append('suggest') or 'She turns.')
    monkeypatch.setattr(vmp, 'enhance',
                        lambda *a, **kw: calls.append('enhance') or 'She turns.')

    for path, body in (('suggest', {'image': 'a.png'}),
                       ('enhance', {'prompt': 'she turns'})):
        r = client.post(f'/api/video-studio/motion/{path}', json=body)
        assert r.status_code == 503, (path, r.get_json())
        assert r.get_json()['error'] == 'GPU busy'
        assert 'queued or active work' in r.get_json()['detail']

    r = client.post('/api/video-studio/generate',
                    json={'mode': 'i2v', 'image': 'staged_1.png', 'prompt': 'she turns',
                          'enhance': True, 'frames': 56})
    assert r.status_code == 200, r.get_json()
    assert 'queued or active work' in r.get_json()['enrich_skipped']
    assert launched['prompt'] == vmp.inject_alignment_header('she turns')
    # The refusal happened BEFORE the writer, not around a wasted call.
    assert calls == []


def test_an_unusable_enrichment_is_a_sentence_on_the_button_and_a_reason_at_launch(
        client, monkeypatch):
    """The service refuses an unusable answer in words. The button shows the
    sentence (a 409, the field untouched); the launch still launches with the
    typed prompt and carries the same words as the reason."""
    from app.services import video_motion_prompt as vmp
    from app.services import video_test_studio as vts

    def unusable(*a, **kw):
        raise RuntimeError('The model answered nothing usable as a prompt — your text '
                           'is unchanged; try again, or pick another model under ⚙.')
    monkeypatch.setattr(vmp, 'enhance', unusable)
    r = client.post('/api/video-studio/motion/enhance', json={'prompt': 'she turns'})
    assert r.status_code == 409
    assert 'nothing usable' in r.get_json()['error']

    monkeypatch.setattr('app.capabilities.probe',
                        lambda *a, **k: {'comfyui': {'reachable': True}})
    monkeypatch.setattr('app.capabilities.probe_comfyui',   # the runner has no ComfyUI
                        lambda: {'ok': True, 'status': 'ok', 'detail': '', 'hint': ''})
    launched = {}
    monkeypatch.setattr(vts, 'enqueue_clip',
                        lambda user, **kw: launched.update(kw) or
                        {'clip_id': 1, 'seed': 1, 'frames': kw.get('frames')})
    r = client.post('/api/video-studio/generate',
                    json={'mode': 't2v', 'prompt': 'she turns', 'enhance': True, 'frames': 56})
    assert r.status_code == 200, r.get_json()
    assert 'nothing usable' in r.get_json()['enrich_skipped']
    assert launched['prompt'] == 'she turns'


def test_a_launch_heads_an_image_to_video_prompt_once_and_unnames_the_picture_in_text_only(
        client, monkeypatch):
    """The reference writer's rule, at generation: a prompt typed by hand gets
    the official I2V header in code; one that already carries it — written by
    ✨, or reused from a clip — is never headed twice. And a text-to-video
    launch is the mirror: a prompt written for a start frame, then launched
    without one, would name a picture the encoder is not given."""
    from app.services import video_motion_prompt as vmp
    from app.services import video_test_studio as vts
    monkeypatch.setattr('app.capabilities.probe',
                        lambda *a, **k: {'comfyui': {'reachable': True}})
    monkeypatch.setattr('app.capabilities.probe_comfyui',   # the runner has no ComfyUI
                        lambda: {'ok': True, 'status': 'ok', 'detail': '', 'hint': ''})
    launched = {}
    monkeypatch.setattr(vts, 'enqueue_clip',
                        lambda user, **kw: launched.update(kw) or
                        {'clip_id': 1, 'seed': 1, 'frames': kw.get('frames')})

    r = client.post('/api/video-studio/generate',
                    json={'mode': 'i2v', 'image': 'staged_1.png',
                          'prompt': 'she turns toward the window', 'frames': 56})
    assert r.status_code == 200, r.get_json()
    headed = launched['prompt']
    assert vmp.has_alignment_header(headed)
    assert headed.endswith('she turns toward the window')

    # Reused as it ran: still one header.
    client.post('/api/video-studio/generate',
                json={'mode': 'i2v', 'image': 'staged_1.png', 'prompt': headed, 'frames': 56})
    assert launched['prompt'] == headed
    assert launched['prompt'].count('is fully referenced') == 1

    # The same text launched as text-to-video names no picture.
    written = vmp.inject_alignment_header(
        'integrated_multimodal_description: [Shot 1] <Picture 1> turns toward the window.')
    client.post('/api/video-studio/generate',
                json={'mode': 't2v', 'prompt': written, 'frames': 56})
    assert not vmp.has_alignment_header(launched['prompt'])
    assert 'Picture 1' not in launched['prompt']
    assert 'turns toward the window' in launched['prompt']


def test_a_typed_prompt_in_the_headers_english_launches_whole_and_nothing_launches_empty(
        client, monkeypatch):
    """Refuted through this route: the header was known by a PHRASE, and
    "is fully referenced" / "align with the target video" are prompt
    English. A text-to-video prompt typed with one reached the sampler
    amputated — 76 of 128 characters, "sunset.from" — or EMPTY, the
    emptiness check sitting before the rewrite; an image-to-video one was
    not given the official header at all. The header is known by its shape
    now, and the text is judged again after the rewrite."""
    from app.services import video_motion_prompt as vmp
    from app.services import video_test_studio as vts
    monkeypatch.setattr('app.capabilities.probe',
                        lambda *a, **k: {'comfyui': {'reachable': True}})
    monkeypatch.setattr('app.capabilities.probe_comfyui',   # the runner has no ComfyUI
                        lambda: {'ok': True, 'status': 'ok', 'detail': '', 'hint': ''})
    launched = {}
    monkeypatch.setattr(vts, 'enqueue_clip',
                        lambda user, **kw: launched.update(kw) or
                        {'clip_id': 1, 'seed': 1, 'frames': kw.get('frames')})
    typed = [
        'Wide shot of a studio where every prop is fully referenced',
        'A woman walks along a beach at sunset, the golden light is fully referenced '
        'to that hour, and she turns slowly toward the camera',
        'A woman walks along a beach at sunset. The mural behind her is fully '
        'referenced from a 1970s tourism poster. She turns toward the camera.',
        'Grade the shot so the tones align with the target video, then hold.',
        "The colours align with the target video's palette throughout the clip.",
        'A slow push in. Everything must align with the target video reference.',
        'For the target video, at 3 seconds she turns toward the camera and the light settles.',
        'A slow push in.\nFor the target video, at 2 seconds in, the camera settles on her face.\n'
        'She turns away.',
        'For the target video, at 3 seconds she turns toward the camera. The picture '
        'on the wall behind her falls.',
    ]
    for p in typed:
        r = client.post('/api/video-studio/generate',
                        json={'mode': 't2v', 'prompt': p, 'frames': 56})
        assert r.status_code == 200, (p, r.get_json())
        assert launched['prompt'] == p
        r = client.post('/api/video-studio/generate',
                        json={'mode': 'i2v', 'image': 'staged_1.png', 'prompt': p, 'frames': 56})
        assert r.status_code == 200, (p, r.get_json())
        assert launched['prompt'] == vmp._ALIGNMENT_HEADER + '\n\n' + p
    # A prompt that is nothing but the header, or labels around the identity
    # sentence, is nothing once they are set aside: refused on both modes,
    # never launched.
    launched.clear()
    for p in (vmp._ALIGNMENT_HEADER,
              vmp._ALIGNMENT_HEADER + '\n\nintegrated_multimodal_description: [Shot 1] '
              + vmp._IDENTITY_SENTENCE + '\noverall_soundscape: rain'):
        r = client.post('/api/video-studio/generate',
                        json={'mode': 't2v', 'prompt': p, 'frames': 56})
        assert r.status_code == 400, (p, r.get_json())
        assert 'no motion' in r.get_json()['error']
        r = client.post('/api/video-studio/generate',
                        json={'mode': 'i2v', 'image': 'staged_1.png', 'prompt': p, 'frames': 56})
        assert r.status_code == 400, (p, r.get_json())
    assert not launched
    # The end-frame line of the reference writer, pasted with its motion, is
    # replaced by this launch's header — the picture is the first frame here.
    pasted = ('How the reference pictures align with the target video — <Picture 1> '
              '(from [Shot 1]) aligns with the 5.00-second mark of the target video.\n\n'
              'She turns toward the camera.')
    client.post('/api/video-studio/generate',
                json={'mode': 'i2v', 'image': 'staged_1.png', 'prompt': pasted, 'frames': 56})
    assert launched['prompt'] == vmp._ALIGNMENT_HEADER + '\n\nShe turns toward the camera.'
    # The official line glued mid-line, or behind a no-break space, is one
    # header wherever it starts: replaced on image-to-video — never stacked
    # under a second — and gone on text-to-video, the sentences around it kept.
    for p in ('She turns slowly. ' + vmp._ALIGNMENT_HEADER + ' Then she smiles.',
              '\u00a0' + vmp._ALIGNMENT_HEADER + '\n\nShe turns slowly. Then she smiles.'):
        client.post('/api/video-studio/generate',
                    json={'mode': 'i2v', 'image': 'staged_1.png', 'prompt': p, 'frames': 56})
        assert launched['prompt'] == vmp._ALIGNMENT_HEADER + '\n\nShe turns slowly. Then she smiles.'
        client.post('/api/video-studio/generate',
                    json={'mode': 't2v', 'prompt': p, 'frames': 56})
        assert launched['prompt'] == 'She turns slowly. Then she smiles.'


# --- ✨ /motion/write-batch : N pictures, ONE vision window --------------------
#
# The route exists for the ORDER of GPU work, not for convenience: entering the
# vision window makes ComfyUI drop its models, so the next clip reloads the
# video model (tens of GB for H3). Writing per picture through the single-frame
# routes pays that per picture. These pin the two things only the route can get
# wrong: one window for the whole strip, and one bad frame not costing the rest.

def test_one_window_writes_for_the_whole_strip(client, monkeypatch):
    from app.services import video_motion_prompt as vmp
    from app.routes import video_studio as vs
    windows = []

    class _W:
        def __init__(self, **kw):
            windows.append(kw)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(vs, 'gpu_exclusive_vision_window', lambda **kw: _W(**kw))
    monkeypatch.setattr(vmp, 'suggest_from_frame', lambda name, **kw: f'motion for {name}')
    r = client.post('/api/video-studio/motion/write-batch',
                    json={'images': ['a.png', 'b.png', 'c.png'], 'seconds': 5.0})
    assert r.status_code == 200
    body = r.get_json()
    assert [x['prompt'] for x in body['results']] == [
        'motion for a.png', 'motion for b.png', 'motion for c.png']
    # ONE window for three pictures — the reload this replaces.
    assert len(windows) == 1, f'expected a single vision window, got {len(windows)}'
    # …and a TTL that covers the batch, not one click.
    assert windows[0]['flag_ttl'] >= 600


def test_a_typed_prompt_enriches_every_frame_instead_of_proposing(client, monkeypatch):
    """The panel's own rule, applied N times: typed motion → enrich each frame
    from it; nothing typed → propose from the frame alone. Getting this backwards
    would silently throw away what the user wrote."""
    from app.services import video_motion_prompt as vmp
    calls = []
    monkeypatch.setattr(vmp, 'enhance',
                        lambda text, **kw: (calls.append(('enhance', text, kw.get('image'))) or 'rich'))
    monkeypatch.setattr(vmp, 'suggest_from_frame',
                        lambda name, **kw: (calls.append(('suggest', name)) or 'fresh'))
    client.post('/api/video-studio/motion/write-batch',
                json={'images': ['a.png', 'b.png'], 'prompt': 'she turns'})
    assert [c[0] for c in calls] == ['enhance', 'enhance']
    assert [c[2] for c in calls] == ['a.png', 'b.png'], 'each frame anchors its own rewrite'
    calls.clear()
    client.post('/api/video-studio/motion/write-batch', json={'images': ['a.png']})
    assert [c[0] for c in calls] == ['suggest']


def test_one_unwritable_frame_does_not_cost_the_others(client, monkeypatch):
    from app.services import video_motion_prompt as vmp

    def flaky(name, **kw):
        if name == 'bad.png':
            raise ValueError('the model could not describe that start frame')
        return f'motion for {name}'

    monkeypatch.setattr(vmp, 'suggest_from_frame', flaky)
    r = client.post('/api/video-studio/motion/write-batch',
                    json={'images': ['a.png', 'bad.png', 'c.png']})
    assert r.status_code == 200, 'a frame that failed is not a failed batch'
    body = r.get_json()
    assert body['written'] == 2 and body['failed'] == 1
    got = {x['image']: x for x in body['results']}
    assert 'could not describe' in got['bad.png']['error']
    assert 'prompt' not in got['bad.png']
    assert [x['index'] for x in body['results']] == [0, 1, 2]


def test_an_empty_or_oversized_strip_is_refused_before_the_gpu(client):
    r = client.post('/api/video-studio/motion/write-batch', json={'images': []})
    assert r.status_code == 400
    from app.routes.video_studio import MAX_WRITE_BATCH
    r = client.post('/api/video-studio/motion/write-batch',
                    json={'images': [f'{i}.png' for i in range(MAX_WRITE_BATCH + 1)]})
    assert r.status_code == 400
    assert str(MAX_WRITE_BATCH) in r.get_json()['error']
