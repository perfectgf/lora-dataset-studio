"""A caption on a dataset frozen by an exclusive operation is refused first.

A copy to or from a Bank, a training export or a backup freezes the dataset.
The caption pass refused such a dataset — but only once it was inside the GPU
vision window, which had already asked ComfyUI to unload its models for a pass
that could not run. The route now refuses before the window opens.
"""
import pytest


def test_a_frozen_dataset_refuses_a_caption_before_comfyui_is_asked_anything(
        app, client, monkeypatch):
    from app.services import dataset_activity
    ds_id = client.post('/api/dataset/create',
                        json={'name': 'Frozen', 'trigger_word': 'frozen'}).get_json()['id']
    monkeypatch.setattr('app.utils.comfyui.free_comfyui_vram',
                        lambda *a, **k: pytest.fail('ComfyUI was asked to unload for a refused caption'))
    token = dataset_activity.begin_exclusive(ds_id, 'backup', detail='creating portable backup')
    try:
        r = client.post(f'/api/dataset/{ds_id}/caption', json={})
        assert r.status_code == 409, r.get_json()
        assert 'backed up' in r.get_json()['error']
    finally:
        dataset_activity.end(token)
