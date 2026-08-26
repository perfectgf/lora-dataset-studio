"""📷 Camera angles — the vocabulary, the guards, and the route.

What is asserted here is what a second reader would plausibly get wrong:

  * the prompt grammar belongs to the LoRA. A token that reads better in
    English is a token the adapter never saw, and the model then answers it the
    way any edit model does — by turning the SUBJECT while the camera stays put.
    That produces a fine picture at the wrong angle, filed under the name of the
    angle that was asked for, and nothing downstream can detect it;
  * an unknown pose is refused rather than dropped, for the same reason;
  * the lane refuses BEFORE creating rows when the weights are absent — the
    Klein lane already paid for the opposite (a dataset full of failed tiles);
  * a camera view cannot be re-shot from another angle: the second pass would
    re-invent what the first pass invented and sell it as the original scene;
  * the VAE is NOT a download of this lane's own — it is the Krea 2 lane's file,
    and the Setup key this lane reports must be the button that installs it.
"""
import pytest

from app.services import camera_angles as ca


# --- the grammar --------------------------------------------------------------

def test_the_prompt_is_the_loras_published_grammar():
    assert ca.TRIGGER == '<sks>'
    assert ca.pose_prompt('front', 'eye', 'medium') == \
        '<sks> front view eye-level shot medium shot'
    assert ca.pose_prompt('back_left', 'low', 'wide') == \
        '<sks> back-left quarter view low-angle shot wide shot'
    assert ca.pose_prompt('right', 'high', 'close') == \
        '<sks> right side view high-angle shot close-up'


def test_ninety_six_poses_and_every_one_of_them_builds():
    assert ca.POSE_COUNT == 96 == len(ca.AZIMUTHS) * len(ca.ELEVATIONS) * len(ca.DISTANCES)
    seen = set()
    for a in ca.AZIMUTHS:
        for e in ca.ELEVATIONS:
            for d in ca.DISTANCES:
                prompt = ca.pose_prompt(a['id'], e['id'], d['id'])
                assert prompt.startswith('<sks> ')
                seen.add(ca.pose_id(a['id'], e['id'], d['id']))
    assert len(seen) == 96


def test_an_unknown_component_raises_instead_of_silently_dropping_a_token():
    for args in (('nope', 'eye', 'medium'), ('front', 'nope', 'medium'),
                 ('front', 'eye', 'nope')):
        with pytest.raises(ValueError, match=ca.UNKNOWN_POSE):
            ca.pose_prompt(*args)


def test_pose_ids_parse_back_and_a_broken_one_returns_none():
    assert ca.parse_pose('right/low/wide') == ('right', 'low', 'wide')
    for bad in (None, 42, '', 'right', 'right/low', 'a/b/c', 'right/low/wide/extra'):
        assert ca.parse_pose(bad) is None, bad


def test_the_reference_pose_is_the_pictures_own_viewpoint():
    assert ca.REFERENCE_POSE == ('front', 'eye', 'medium')
    assert ca.is_reference_pose('front/eye/medium') is True
    assert ca.is_reference_pose('back/eye/medium') is False


# --- the selection ------------------------------------------------------------

def test_a_repeated_pose_is_one_picture_not_two():
    # Charging the GPU for a duplicate is a bug the user cannot see.
    assert ca.normalize_requested(
        ['right/low/medium', 'right/low/medium', 'back/eye/wide']) == [
        ('right', 'low', 'medium'), ('back', 'eye', 'wide')]


def test_the_selection_refuses_to_be_empty_or_unbounded():
    for empty in ([], (), 'not a list', None):
        with pytest.raises(ValueError, match=ca.NO_VIEWS_PICKED):
            ca.normalize_requested(empty)
    with pytest.raises(ValueError, match=ca.UNKNOWN_POSE):
        ca.normalize_requested(['front/eye/medium', 'nope'])
    too_many = [ca.pose_id(a['id'], 'eye', 'medium') for a in ca.AZIMUTHS] * 2
    # Deduplicated first: eight uniques is UNDER the ceiling, so a doubled list
    # of the same eight must pass rather than read as sixteen.
    assert len(ca.normalize_requested(too_many)) == 8
    over = [ca.pose_id(a['id'], e['id'], 'medium')
            for a in ca.AZIMUTHS for e in ca.ELEVATIONS][:ca.MAX_VIEWS_PER_RUN + 1]
    with pytest.raises(ValueError, match='more than'):
        ca.normalize_requested(over)


def test_the_catalog_carries_everything_the_picker_draws():
    cat = ca.catalog()
    assert cat['pose_count'] == 96
    assert cat['trigger'] == '<sks>'
    assert cat['reference_pose'] == 'front/eye/medium'
    assert cat['max_views'] == ca.MAX_VIEWS_PER_RUN
    for key, expected in (('azimuths', 8), ('elevations', 4), ('distances', 3)):
        assert len(cat[key]) == expected
        for entry in cat[key]:
            assert {'id', 'token', 'label'} <= set(entry)


# --- the weights --------------------------------------------------------------

def test_the_vae_is_the_krea_lanes_download_not_a_second_copy(app):
    """One file, one Setup button. A second key for the same bytes would offer
    the same gigabyte twice and let two copies drift apart."""
    from app import setup_installer
    from app.services import qwen_camera_helper as qch
    assert qch.CAMERA_VAE_ACTION == 'krea_vae'
    assert qch.CAMERA_VAE_ACTION in qch.CAMERA_REQUIRED
    assert 'camera_vae' not in setup_installer._MODEL_DOWNLOADS
    krea = setup_installer._MODEL_DOWNLOADS['krea_vae']['dest']
    assert krea[-1] == 'qwen_image_vae.safetensors'


def test_every_camera_asset_can_be_installed_from_setup(app):
    from app import setup_installer
    from app.services import qwen_camera_helper as qch
    for action in setup_installer._CAMERA_DOWNLOADS:
        assert action in setup_installer.INSTALL_ACTIONS, action
    for action in qch.CAMERA_REQUIRED + qch.CAMERA_RECOMMENDED:
        assert action in setup_installer.INSTALL_ACTIONS, action


def test_the_text_encoder_is_never_confused_with_the_other_two_qwens(app):
    """models/text_encoders holds three Qwen encoders and they are NOT
    interchangeable. A resolver matching a bare 'qwen' picks the wrong one and
    the sampler dies on a shape mismatch — or worse, does not."""
    from app import setup_installer
    dest = setup_installer._CAMERA_DOWNLOADS['camera_text_encoder']['dest'][-1]
    assert dest == 'qwen_2.5_vl_7b_fp8_scaled.safetensors'
    klein = setup_installer._KLEIN_DOWNLOADS['klein_text_encoder']['dest'][-1]
    krea = setup_installer._KREA_DOWNLOADS['krea_text_encoder']['dest'][-1]
    assert len({dest, klein, krea}) == 3


def test_a_missing_required_asset_names_the_setup_buttons(app, monkeypatch):
    from app.services import qwen_camera_helper as qch
    monkeypatch.setattr(qch, 'resolve_camera_unet', lambda: None)
    monkeypatch.setattr(qch, 'resolve_camera_lora', lambda: ('x', None))
    monkeypatch.setattr(qch, 'resolve_camera_text_encoder', lambda: 'te.safetensors')
    monkeypatch.setattr(qch, 'resolve_camera_vae', lambda: 'vae.safetensors')
    monkeypatch.setattr(qch, 'resolve_camera_speed_lora', lambda: ('y', 'path'))
    assert qch.camera_missing_assets() == ['camera_model', 'camera_lora']
    assert qch.camera_ready() is False


def test_the_speed_lora_alone_missing_does_not_block_the_lane(app, monkeypatch):
    """It buys steps, not correctness: absent, the graph runs at 20 instead of
    4. Gating the lane on it would refuse to render over a speed-up."""
    from app.services import qwen_camera_helper as qch
    monkeypatch.setattr(qch, 'resolve_camera_unet', lambda: 'm.safetensors')
    monkeypatch.setattr(qch, 'resolve_camera_lora', lambda: ('a', 'path'))
    monkeypatch.setattr(qch, 'resolve_camera_text_encoder', lambda: 'te.safetensors')
    monkeypatch.setattr(qch, 'resolve_camera_vae', lambda: 'vae.safetensors')
    monkeypatch.setattr(qch, 'resolve_camera_speed_lora', lambda: ('s', None))
    assert qch.camera_missing_assets() == ['camera_speed_lora']
    assert qch.camera_ready() is True
    assert qch.STEPS_WITHOUT_SPEED_LORA > qch.STEPS_WITH_SPEED_LORA


def test_the_shipped_workflow_still_has_the_nodes_the_helper_edits(app):
    import json
    from app.services import qwen_camera_helper as qch
    graph = json.loads(qch.WORKFLOW_CAMERA_PATH.read_text(encoding='utf-8'))
    for node in qch._REQUIRED_NODES:
        assert node in graph, node
    # The three the helper writes into by name.
    assert graph['112']['class_type'] == 'TextEncodeQwenImageEditPlus'
    assert graph['109']['class_type'] == 'LoraLoaderModelOnly'
    assert graph['106']['class_type'] == 'KSampler'


# --- the route ----------------------------------------------------------------

def _dataset(client, name='Nova'):
    return client.post('/api/dataset/create',
                       json={'name': name, 'trigger_word': 'nova'}).get_json()['id']


def _row(dataset_id, **kw):
    from app.extensions import db
    from app.models import LoraTestImage
    img = LoraTestImage(dataset_id=dataset_id, checkpoint='a.safetensors', strength=1.0,
                        status=kw.pop('status', 'done'),
                        filename=kw.pop('filename', 'x.png'), **kw)
    db.session.add(img)
    db.session.commit()
    return img


def test_the_catalog_route_serves_the_vocabulary_and_the_readiness(client, app):
    body = client.get('/api/camera/catalog').get_json()
    assert body['pose_count'] == 96
    assert body['trigger'] == '<sks>'
    assert isinstance(body['ready'], bool)
    assert isinstance(body['missing'], list)


def test_a_camera_view_cannot_be_re_shot_from_another_angle(client, app):
    from app.services import lora_test_studio as lts
    ds = _dataset(client)
    with app.app_context():
        row = _row(ds, derivation_kind=lts.CAMERA_ANGLE, camera_pose='right/eye/medium')
        with pytest.raises(ValueError, match=ca.ALREADY_DERIVED):
            lts.camera_views_for_canvas_image('local', row.id, ['back/eye/medium'])


def test_an_improve_result_IS_allowed_as_a_source(client, app, monkeypatch, tmp_path):
    """✨ is the same scene from the same viewpoint, only cleaner — the best
    source this lane can be handed, not a compounded guess.

    The first guard refused every `derivation_kind`, and one look at a real
    library settled it: the newest six tiles were all improve results, so the
    verb was greyed out on exactly the pictures people keep."""
    from app.services import lora_test_studio as lts
    from app.services import qwen_camera_helper as qch
    ds = _dataset(client)
    with app.app_context():
        row = _row(ds, derivation_kind='canvas_image_improve')
        (tmp_path / 'x.png').write_bytes(b'x')
        monkeypatch.setattr(lts.fds, '_dataset_dir', lambda _id: str(tmp_path))
        # It must get PAST the derivation guard; the weights preflight is what
        # stops it here, which is proof the source itself was accepted.
        monkeypatch.setattr(qch, 'camera_missing_assets', lambda: ['camera_model'])
        with pytest.raises(qch.CameraModelsMissing):
            lts.camera_views_for_canvas_image('local', row.id, ['back/eye/medium'])


def test_an_unfinished_render_is_refused_before_anything_is_queued(client, app):
    from app.services import lora_test_studio as lts
    ds = _dataset(client)
    with app.app_context():
        row = _row(ds, status='pending', filename=None)
        with pytest.raises(ValueError, match=ca.SOURCE_NOT_DONE):
            lts.camera_views_for_canvas_image('local', row.id, ['back/eye/medium'])


def test_missing_weights_refuse_BEFORE_any_row_is_created(client, app, monkeypatch, tmp_path):
    """The Klein lane already paid for the opposite: a preflight that ran too
    late left the dataset full of failed tiles."""
    from app.models import LoraTestImage
    from app.services import lora_test_studio as lts
    from app.services import qwen_camera_helper as qch
    ds = _dataset(client)
    with app.app_context():
        row = _row(ds)
        src = tmp_path / 'x.png'
        src.write_bytes(b'x')
        monkeypatch.setattr(lts.fds, '_dataset_dir', lambda _id: str(tmp_path))
        monkeypatch.setattr(qch, 'camera_missing_assets',
                            lambda: ['camera_model', 'camera_lora'])
        before = LoraTestImage.query.count()
        with pytest.raises(qch.CameraModelsMissing):
            lts.camera_views_for_canvas_image('local', row.id, ['back/eye/medium'])
        assert LoraTestImage.query.count() == before


def test_the_route_turns_missing_weights_into_an_actionable_409(client, app,
                                                                monkeypatch, tmp_path):
    from app.services import lora_test_studio as lts
    from app.services import qwen_camera_helper as qch
    ds = _dataset(client)
    with app.app_context():
        row = _row(ds)
        row_id = row.id
    (tmp_path / 'x.png').write_bytes(b'x')
    monkeypatch.setattr(lts.fds, '_dataset_dir', lambda _id: str(tmp_path))
    monkeypatch.setattr(qch, 'camera_missing_assets', lambda: ['camera_model'])
    r = client.post(f'/api/canvas/image/{row_id}/camera',
                    json={'poses': ['back/eye/medium']})
    assert r.status_code == 409
    body = r.get_json()
    assert body['ok'] is False
    assert body['camera_missing'] == ['camera_model']
    # The string is what a person reads in a toast: it must stand on its own.
    assert 'Camera angles' in body['error']


def test_the_route_refuses_an_unknown_pose(client, app, monkeypatch, tmp_path):
    from app.services import lora_test_studio as lts
    ds = _dataset(client)
    with app.app_context():
        row = _row(ds)
        row_id = row.id
    (tmp_path / 'x.png').write_bytes(b'x')
    monkeypatch.setattr(lts.fds, '_dataset_dir', lambda _id: str(tmp_path))
    r = client.post(f'/api/canvas/image/{row_id}/camera', json={'poses': ['nope']})
    assert r.status_code >= 400
    assert ca.UNKNOWN_POSE in (r.get_json() or {}).get('error', '')


def test_an_unknown_image_is_a_404_not_a_crash(client, app):
    r = client.post('/api/canvas/image/999999/camera',
                    json={'poses': ['back/eye/medium']})
    assert r.status_code == 404
