"""The ✨ Motion endpoints, seen from the panel that calls them.

The service has its own file; what only the route can answer is whether every
piece the panel sends actually ARRIVES — the frame, the instruction, the chosen
model. A parameter that never reaches the service is invisible to a service
test and fails silently in front of the user, which is how the enrichment
button spent a day rewriting prompts with no idea what picture they animate.
"""


def test_the_panel_s_three_pieces_reach_the_writer(client, monkeypatch):
    """image + instruction + model, from ✨ Auto."""
    from app.services import video_motion_prompt as vmp
    seen = {}

    def fake(image_name, instruction=None, model=None):
        seen.update(image=image_name, instruction=instruction, model=model)
        return 'She lifts her gaze slowly toward the lens.'

    monkeypatch.setattr(vmp, 'suggest_from_frame', fake)
    r = client.post('/api/video-studio/motion/suggest',
                    json={'image': 'staged_1.png', 'instruction': 'make her jump',
                          'model': 'qwen3-vl:8b'})
    assert r.status_code == 200
    assert r.get_json()['prompt'].startswith('She lifts her gaze')
    assert seen == {'image': 'staged_1.png', 'instruction': 'make her jump',
                    'model': 'qwen3-vl:8b'}


def test_the_enrichment_is_told_which_frame_the_clip_starts_from(client, monkeypatch):
    """Without the frame, "make her look out of the window" invents a window."""
    from app.services import video_motion_prompt as vmp
    seen = {}

    def fake(prompt, image=None, model=None):
        seen.update(prompt=prompt, image=image, model=model)
        return 'She turns her head slowly to the left.'

    monkeypatch.setattr(vmp, 'enhance', fake)
    r = client.post('/api/video-studio/motion/enhance',
                    json={'prompt': 'she turns', 'image': 'staged_1.png',
                          'model': 'qwen3-vl:8b'})
    assert r.status_code == 200
    assert seen == {'prompt': 'she turns', 'image': 'staged_1.png',
                    'model': 'qwen3-vl:8b'}
    # t2v has no frame, and that is not an error — the text alone is enriched.
    client.post('/api/video-studio/motion/enhance',
                json={'prompt': 'she turns', 'image': None})
    assert seen['image'] is None


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
