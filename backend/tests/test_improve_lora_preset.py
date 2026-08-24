"""✨ The improve pass chains a generation-LoRA preset — by GLOBAL setting.

`klein.improve_lora_preset` names one of klein.generation_lora_presets, the way
the improve INSTRUCTION is already global (identity_prompts.klein_improve): one
answer to "what will improve run with", honoured by the single pass, the 🔄
re-run and the batch, because all three drain through the same enqueue profile.

What is pinned here:

  * the profile carries the RESOLVED rows, so the choice reaches
    enqueue_klein_edit through the same splat as every other improve knob;
  * fail-closed all the way down — '' and a stale name mean "no preset",
    never a blocked pass;
  * SeedVR2 is untouched: a restoration chains nothing, and its candidates
    must not carry a preset they never ran;
  * provenance — a ◉ Canvas/Gallery candidate records the rows that DID decide
    it in `extra_loras` (the studio's own {filename, strength} shape, which is
    what the lightbox "Made with" panel reads).
"""
import io
import json
import os

import pytest
from PIL import Image

PRESETS = [{'name': 'Detail', 'loras': [
    {'file': 'klein/detail.safetensors', 'strength': 0.7},
    {'file': 'klein/skin.safetensors', 'strength': 0.4},
]}]


def _set_improve_preset(name, presets=PRESETS):
    from app import config as cfg
    cfg.save_config({'klein': {'generation_lora_presets': presets,
                               'improve_lora_preset': name}})


def _png():
    buf = io.BytesIO()
    Image.new('RGB', (64, 48), (30, 60, 90)).save(buf, 'PNG')
    return buf.getvalue()


def _board_image(svc):
    from app.extensions import db
    from app.models import LoraTestImage
    ds = svc.create_dataset('local', 'Improve preset', 'presettrigger')
    os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
    with open(os.path.join(svc._dataset_dir(ds.id), 'render.png'), 'wb') as fh:
        fh.write(_png())
    row = LoraTestImage(dataset_id=ds.id, checkpoint='z image\\Lola-1200.safetensors',
                        strength=0.9, status='done', filename='render.png',
                        record_id=7, step=1200, run_id='run-a', seed=4242,
                        prompt='a portrait')
    db.session.add(row)
    db.session.commit()
    return ds, row


# --- the profile -------------------------------------------------------------

def test_the_enqueue_profile_carries_the_resolved_rows(app):
    from app.services import face_dataset_service as svc
    with app.app_context():
        _set_improve_preset('Detail')
        profile = svc._improve_enqueue_profile(None)
        assert profile['generation_loras'] == PRESETS[0]['loras']


def test_no_pick_and_a_stale_name_both_mean_no_preset(app):
    from app.services import face_dataset_service as svc
    with app.app_context():
        _set_improve_preset('')
        assert svc._improve_enqueue_profile(None)['generation_loras'] == []
        _set_improve_preset('Renamed-away')
        assert svc._improve_enqueue_profile(None)['generation_loras'] == []


def test_the_klein_hand_off_forwards_the_rows_to_enqueue_klein_edit(app, monkeypatch):
    """The splat is the contract: _enqueue_improve passes the profile through
    **kwargs, so a renamed key would silently stop reaching the graph. This
    asserts the value ARRIVES, not just that the profile holds it."""
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh
    seen = {}

    def fake_enqueue(user_id, source_filename, edit_prompt, **kw):
        seen.update(kw)
        return 'job-1'

    monkeypatch.setattr(keh, 'enqueue_klein_edit', fake_enqueue)
    with app.app_context():
        _set_improve_preset('Detail')

        class Src:
            filename = 'render.png'
            dataset_id = 1
            id = 1
        svc._enqueue_improve('klein', user_id='local', source=Src(),
                             source_path='unused.png', prompt='p', label='l',
                             dataset=None)
        assert seen['generation_loras'] == PRESETS[0]['loras']


# --- the setting round-trips through /api/settings ---------------------------

def test_the_setting_saves_and_echoes_like_every_other_klein_knob(client, app):
    """The lightbox picker writes through PUT /api/settings (partial, deep-
    merged) — the same contract the improve instruction uses. What the server
    echoes is what every mounted picker then shows."""
    r = client.put('/api/settings', json={
        'config': {'klein': {'generation_lora_presets': PRESETS,
                             'improve_lora_preset': 'Detail'}}})
    assert r.status_code == 200
    assert r.get_json()['config']['klein']['improve_lora_preset'] == 'Detail'
    # …and a later unrelated partial save does not clobber it.
    client.put('/api/settings', json={'config': {'klein': {'improve_steps': 6}}})
    d = client.get('/api/settings').get_json()
    assert d['config']['klein']['improve_lora_preset'] == 'Detail'


# --- provenance on the canvas/gallery candidate ------------------------------

def test_a_klein_candidate_records_the_rows_that_decided_it(app, monkeypatch):
    from app.extensions import db
    from app.models import LoraTestImage
    from app.services import face_dataset_service as svc
    from app.services import lora_test_studio as lts
    monkeypatch.setattr(svc, '_enqueue_improve',
                        lambda *a, **k: 'job-1')
    monkeypatch.setattr(svc, '_improve_preflight', lambda engine: None)
    with app.app_context():
        _set_improve_preset('Detail')
        ds, row = _board_image(svc)
        out = lts.improve_canvas_image('local', row.id, engine='klein')
        candidate = db.session.get(LoraTestImage, out['candidate_id'])
        assert json.loads(candidate.extra_loras) == [
            {'filename': 'klein/detail.safetensors', 'strength': 0.7},
            {'filename': 'klein/skin.safetensors', 'strength': 0.4},
        ]


@pytest.mark.parametrize('engine,preset', [
    ('klein', ''),          # no pick — nothing ran, nothing stored
    ('seedvr2', 'Detail'),  # a restoration chains nothing, whatever is set
])
def test_candidates_never_claim_a_preset_that_did_not_run(app, monkeypatch,
                                                          engine, preset):
    from app.extensions import db
    from app.models import LoraTestImage
    from app.services import face_dataset_service as svc
    from app.services import lora_test_studio as lts
    monkeypatch.setattr(svc, '_enqueue_improve', lambda *a, **k: 'job-1')
    monkeypatch.setattr(svc, '_improve_preflight', lambda engine: None)
    with app.app_context():
        _set_improve_preset(preset)
        ds, row = _board_image(svc)
        out = lts.improve_canvas_image('local', row.id, engine=engine)
        candidate = db.session.get(LoraTestImage, out['candidate_id'])
        assert candidate.extra_loras is None
