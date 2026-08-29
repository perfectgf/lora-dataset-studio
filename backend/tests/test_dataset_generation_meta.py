"""⚙ generation_meta — the dataset's own "made with" stamp.

Three layers, cheapest proof first:

  * the helpers: what is stamped is exactly what was known (None dropped,
    nothing left -> NULL, bad JSON parses to None instead of crashing a
    viewer on a hand-edited database);
  * the SOURCE contract: every place that creates a PENDING GENERATED
    FaceDatasetImage stamps generation_meta in the same constructor call.
    This is the anti-forget guard: the gap being closed existed precisely
    because five lanes were written on five different days — a sixth lane
    added without a stamp must fail here BY NAME, not ship a new island of
    rows that know nothing about themselves;
  * one lane end-to-end (the Klein variations), enqueue mocked: the stamp
    round-trips through the column and the parser with the engine's facts.
"""
import re

import pytest

from app.services import face_dataset_service as svc


# --- the helpers --------------------------------------------------------------

def test_the_stamp_claims_only_what_was_known():
    raw = svc._generation_meta_json(engine='klein', base_model=None, steps=4,
                                    loras=[], seed=0)
    assert raw is not None
    import json
    parsed = json.loads(raw)
    assert parsed == {'engine': 'klein', 'steps': 4, 'seed': 0}, (
        'None and empty values must be dropped; a real 0 (a seed) must not be')
    assert svc._generation_meta_json(engine=None, loras=[]) is None, (
        'nothing known stores NULL, never an empty shell')


def test_the_parser_never_crashes_a_viewer():
    assert svc.parsed_generation_meta(None) is None
    assert svc.parsed_generation_meta('') is None
    assert svc.parsed_generation_meta('{not json') is None
    assert svc.parsed_generation_meta('[1,2]') is None
    assert svc.parsed_generation_meta('{"engine": "krea"}') == {'engine': 'krea'}


# --- the source contract ------------------------------------------------------

def _generated_constructor_calls(source: str):
    """Every FaceDatasetImage(...) call text whose kwargs mark it as a PENDING
    GENERATED row — the shape ⏹ Stop targets and the generating lanes create."""
    calls = []
    for m in re.finditer(r'FaceDatasetImage\(', source):
        depth, i = 1, m.end()
        while depth and i < len(source):
            if source[i] == '(':
                depth += 1
            elif source[i] == ')':
                depth -= 1
            i += 1
        call = source[m.start():i]
        if "source='generated'" in call and "status='pending'" in call:
            calls.append(call)
    return calls


def test_every_generating_lane_stamps_generation_meta():
    import inspect
    source = inspect.getsource(svc)
    calls = _generated_constructor_calls(source)
    assert len(calls) >= 6, (
        'expected the six generating lanes (klein / krea / api variations, '
        f'improve, camera, small-image rescue) — found {len(calls)}; '
        'if one moved, update this floor')
    for call in calls:
        assert 'generation_meta=' in call, (
            'a pending GENERATED row is created without its made-with stamp:\n'
            + call[:400])


# --- one lane end-to-end ------------------------------------------------------

def test_the_klein_lane_round_trips_its_facts(app, monkeypatch):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Meta', 'metatrig')
        ds.ref_filename = 'ref.png'
        svc.db.session.commit()
        monkeypatch.setattr(svc, '_ref_path', lambda _ds: __file__)
        monkeypatch.setattr('app.services.klein_edit_helper.enqueue_klein_edit',
                            lambda **kw: 'job-meta')
        monkeypatch.setattr('app.services.klein_edit_helper.klein_missing_assets',
                            lambda: [])
        ids = svc.generate_variations(
            LOCAL_USER, ds.id,
            [{'label': 'Front portrait', 'prompt': 'a portrait', 'framing': 'face'}],
            1)
        row = svc.db.session.get(FaceDatasetImage, ids[0])
        meta = svc.parsed_generation_meta(row.generation_meta)
        assert meta and meta['engine'] == 'klein'
        assert meta['steps'] > 0
        assert 'aspect' in meta
