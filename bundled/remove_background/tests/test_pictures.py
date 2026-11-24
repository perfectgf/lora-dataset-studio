"""Import, cut out in steps, paint, cancel and recover through the registered HTTP API."""
import io
import json
from pathlib import Path
import sys

import pytest
from flask import Flask
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'backend'), str(ROOT / 'bundled/remove_background')]
from app.plugins.api import PluginContext
from app.plugins.manifest import load_manifest
from app.plugins.registry import PluginRecord, PluginRegistry
from lds_remove_background import assets, engine, graphs, register, routes, studio

URL = '/api/remove-background/pictures'


@pytest.fixture
def product(tmp_path, monkeypatch):
    app = Flask(__name__)
    app.config.update(TESTING=True, MAX_CONTENT_LENGTH=64 * 1024 * 1024)
    directory = ROOT / 'bundled/remove_background'
    manifest = load_manifest(directory)
    registry = PluginRegistry()
    registry.records[manifest.id] = PluginRecord(manifest.id, manifest, str(directory), True, state='loaded')
    assert registry.claim_manifest(manifest) is None
    app.extensions['lds_plugins'] = registry
    register(PluginContext(app, registry, manifest, tmp_path / 'plugin-data'))
    output, input_dir = tmp_path / 'output', tmp_path / 'input'
    output.mkdir()
    input_dir.mkdir()
    monkeypatch.setattr(routes, 'require_idle_comfy', lambda: None)
    monkeypatch.setattr(routes.config, 'local_user', lambda: 'test-user')
    monkeypatch.setattr(engine.config, 'comfyui_dir', lambda kind: str(output if kind == 'output' else input_dir))
    nodes = graphs.FIND_NODES | graphs.MATTE_NODES | graphs.FINE_NODES
    monkeypatch.setattr(engine.local_render, 'fetch_object_info_classes', lambda: set(nodes))
    monkeypatch.setattr(assets, 'inspect_models',
                        lambda: {slot: {'file': spec['filename'], 'state': 'ready'} for slot, spec in assets.MODELS.items()})
    monkeypatch.setattr(engine.comfy, 'ensure_input_usable', str)

    def stage(path, name, folder):
        (Path(folder) / name).write_bytes(Path(path).read_bytes())
        return str(Path(folder) / name)
    monkeypatch.setattr(engine.comfy, 'stage_input_image', stage)
    jobs = {}

    def enqueue(plugin, kind, **kwargs):
        assert plugin == 'remove_background' and kind == 'is_remove_background'
        jobs[kwargs['job_id']] = {**kwargs, 'status': 'pending', 'result_filename': None, 'error': None}
        return kwargs['job_id']
    monkeypatch.setattr(engine.comfy, 'add_plugin_job', enqueue)
    monkeypatch.setattr(studio.comfy, 'plugin_job', lambda _plugin, ident, _user: jobs.get(ident))
    monkeypatch.setattr(studio, 'send_to_trash', lambda path, context='': __import__('shutil').rmtree(path))

    def answer(job_id, image, *, failed=False, reason=None):
        """The queue finished a step: its file is in ComfyUI's output folder."""
        name = None
        if image is not None:
            name = f'LDS_RemoveBackground_{job_id}_00001_.png'
            image.save(output / name)
        jobs[job_id].update(status='failed' if failed else 'completed', result_filename=name)
        callback = registry.job_handlers['is_remove_background'][1]
        with app.app_context():
            callback(job_id, name, failed=failed, reason=reason, metadata=jobs[job_id]['metadata'])
    product = app.test_client()
    product.jobs, product.answer, product.output, product.input = jobs, answer, output, input_dir
    return product


def add(client, size=(600, 400), box=(250, 150, 350, 250)):
    picture = Image.new('RGB', size, (30, 90, 200))
    ImageDraw.Draw(picture).ellipse(box, fill=(220, 180, 40))
    source = io.BytesIO()
    picture.save(source, 'PNG')
    source.seek(0)
    response = client.post(URL, data={'image': (source, 'source.png')})
    assert response.status_code == 201, response.get_json()
    return response.get_json()['picture']


def shape(size, box):
    mask = Image.new('RGB', size, (0, 0, 0))
    ImageDraw.Draw(mask).ellipse(box, fill=(255, 255, 255))
    return mask


def last(client):
    return list(client.jobs)[-1]


def staged(client, job_id):
    return Image.open(client.input / client.jobs[job_id]['metadata']['staged_inputs'][0])


def test_a_described_object_is_found_then_cut_from_close_up(product):
    picture = add(product)
    response = product.post(f'{URL}/{picture["id"]}/cut', json={'keep': 'describe', 'kind': 'photo', 'phrase': 'keep the ball'})
    assert response.status_code == 202 and response.json['picture']['run']['stage'] == 'find'
    find = last(product)
    assert product.jobs[find]['workflow_data']['3']['inputs']['text'] == 'ball: 16'
    assert product.post(f'{URL}/{picture["id"]}/cut', json={'keep': 'subject', 'kind': 'photo'}).status_code == 400
    product.answer(find, shape((600, 400), (250, 150, 350, 250)))
    cut = last(product)
    assert cut != find and product.jobs[cut]['workflow_data']['2']['inputs']['bg_removal_name'] == 'birefnet.safetensors'
    assert staged(product, cut).size[0] < 200, 'the matting model sees the object from close up'
    run = product.get(f'{URL}/{picture["id"]}').json['picture']['run']
    assert run['stage'] == 'cut' and 'files' not in run
    size = staged(product, cut).size
    product.answer(cut, shape(size, (24, 24, size[0] - 24, size[1] - 24)))
    done = product.get(f'{URL}/{picture["id"]}').json['picture']
    assert done['run']['status'] == 'done' and done['mask']['matted'] is True and done['mask']['words'] == ['ball']
    assert list(product.output.iterdir()) == [], 'the working files are read and gone'
    mask = Image.open(io.BytesIO(product.get(f'{URL}/{picture["id"]}/mask').data))
    assert mask.mode == 'L' and mask.size == (600, 400)
    assert mask.getpixel((300, 200)) == 255 and mask.getpixel((20, 20)) == 0
    cutout = Image.open(io.BytesIO(product.get(f'{URL}/{picture["id"]}/cutout?clean=1').data))
    assert cutout.mode == 'RGBA' and cutout.getpixel((20, 20)) == (0, 0, 0, 0)
    white = Image.open(io.BytesIO(product.get(f'{URL}/{picture["id"]}/cutout?background=white').data))
    assert white.mode == 'RGB' and white.getpixel((20, 20)) == (255, 255, 255)
    assert product.get(f'{URL}/{picture["id"]}/cutout?background=purple').status_code == 400


def test_a_small_main_subject_is_looked_at_twice_and_a_large_one_once(product):
    picture = add(product)
    product.post(f'{URL}/{picture["id"]}/cut', json={'keep': 'subject', 'kind': 'photo'})
    first = last(product)
    product.answer(first, shape((600, 400), (250, 150, 350, 250)))
    second = last(product)
    assert second != first
    assert product.get(f'{URL}/{picture["id"]}').json['picture']['run']['stage'] == 'closer'
    product.answer(second, shape(staged(product, second).size, (20, 20, 120, 120)))
    assert product.get(f'{URL}/{picture["id"]}').json['picture']['run']['status'] == 'done'

    large = add(product, box=(10, 10, 590, 390))
    product.post(f'{URL}/{large["id"]}/cut', json={'keep': 'subject', 'kind': 'graphic'})
    only = last(product)
    assert product.jobs[only]['workflow_data']['2']['inputs']['bg_removal_name'] == 'lucida.safetensors'
    product.answer(only, shape((600, 400), (10, 10, 590, 390)))
    assert last(product) == only
    assert product.get(f'{URL}/{large["id"]}').json['picture']['mask']['kind'] == 'graphic'


def test_clicks_follow_the_picture_sent_and_a_selection_bounds_the_matte(product):
    picture = add(product, size=(3000, 2000), box=(1200, 700, 1800, 1300))
    body = {'keep': 'points', 'kind': 'fine', 'points': [{'x': 1500, 'y': 1000, 'keep': True}, {'x': 100, 'y': 100, 'keep': False}]}
    assert product.post(f'{URL}/{picture["id"]}/cut', json=body).status_code == 202
    inputs = product.jobs[last(product)]['workflow_data']['4']['inputs']
    assert json.loads(inputs['positive_coords']) == [{'x': 768, 'y': 512}], 'the sent picture is half the size'
    assert json.loads(inputs['negative_coords']) == [{'x': 51, 'y': 51}]
    product.answer(last(product), shape((1536, 1024), (614, 358, 922, 666)))
    fine = product.jobs[last(product)]['workflow_data']
    assert (fine['5']['inputs']['width'], fine['5']['inputs']['height']) == staged(product, last(product)).size
    product.post(f'{URL}/{picture["id"]}/cancel', json={})

    other = add(product)
    drawn = Image.new('RGBA', (300, 200), (0, 0, 0, 0))
    ImageDraw.Draw(drawn).rectangle((100, 50, 200, 150), fill=(255, 255, 255, 255))
    stream = io.BytesIO()
    drawn.save(stream, 'PNG')
    stream.seek(0)
    response = product.post(f'{URL}/{other["id"]}/cut', data={'keep': 'selection', 'kind': 'photo', 'selection': (stream, 's.png')})
    assert response.status_code == 202, response.get_json()
    job = last(product)
    everything = Image.new('RGB', staged(product, job).size, (255, 255, 255))
    product.answer(job, everything)
    mask = Image.open(io.BytesIO(product.get(f'{URL}/{other["id"]}/mask').data))
    assert mask.getpixel((300, 200)) == 255 and mask.getpixel((560, 380)) == 0, 'nothing outside the selection'


@pytest.mark.parametrize('body', [
    {'keep': 'describe', 'kind': 'photo', 'phrase': '  the '},
    {'keep': 'points', 'kind': 'photo', 'points': []},
    {'keep': 'points', 'kind': 'photo', 'points': [{'x': 5, 'y': 5, 'keep': False}]},
    {'keep': 'points', 'kind': 'photo', 'points': [{'x': 9999, 'y': 5, 'keep': True}]},
    {'keep': 'points', 'kind': 'photo', 'points': [{'x': True, 'y': 5, 'keep': True}]},
    {'keep': 'selection', 'kind': 'photo'},
    {'keep': 'everything', 'kind': 'photo'},
    {'keep': 'subject', 'kind': 'cartoon'},
])
def test_an_incomplete_request_queues_nothing(product, body):
    picture = add(product)
    response = product.post(f'{URL}/{picture["id"]}/cut', json=body)
    assert response.status_code == 400 and response.json['error']
    assert product.jobs == {} and list(product.input.iterdir()) == []


def test_missing_models_and_old_comfyui_say_what_to_do(product, monkeypatch):
    picture = add(product)
    report = {slot: {'file': spec['filename'], 'state': 'ready'} for slot, spec in assets.MODELS.items()}
    report['sam'] = {'file': None, 'state': 'missing'}
    monkeypatch.setattr(assets, 'inspect_models', lambda: report)
    refused = product.post(f'{URL}/{picture["id"]}/cut', json={'keep': 'describe', 'kind': 'photo', 'phrase': 'ball'})
    assert refused.status_code == 409 and refused.json['missing'] == ['remove_bg_sam']
    assert product.post(f'{URL}/{picture["id"]}/cut', json={'keep': 'subject', 'kind': 'photo'}).status_code == 202
    monkeypatch.setattr(engine.local_render, 'fetch_object_info_classes', lambda: set(graphs.FIND_NODES))
    other = add(product, size=(500, 400))
    refused = product.post(f'{URL}/{other["id"]}/cut', json={'keep': 'subject', 'kind': 'photo'})
    assert refused.status_code == 409 and 'RemoveBackground' in refused.json['error']
    monkeypatch.setattr(engine.local_render, 'fetch_object_info_classes', lambda: None)
    assert 'Start it' in product.post(f'{URL}/{other["id"]}/cut', json={'keep': 'subject', 'kind': 'photo'}).json['error']


def test_nothing_found_failures_and_cancelled_steps_leave_the_picture_usable(product, monkeypatch):
    picture = add(product)
    product.post(f'{URL}/{picture["id"]}/cut', json={'keep': 'describe', 'kind': 'photo', 'phrase': 'unicorn'})
    product.answer(last(product), Image.new('RGB', (600, 400), (0, 0, 0)))
    run = product.get(f'{URL}/{picture["id"]}').json['picture']['run']
    assert run['status'] == 'failed' and 'unicorn' in run['error']
    product.post(f'{URL}/{picture["id"]}/cut', json={'keep': 'subject', 'kind': 'photo'})
    product.answer(last(product), None, failed=True, reason='Model could not load')
    assert product.get(f'{URL}/{picture["id"]}').json['picture']['run']['error'] == 'Model could not load'
    # A step ComfyUI finished while nobody listened is read when the picture is opened again.
    product.post(f'{URL}/{picture["id"]}/cut', json={'keep': 'subject', 'kind': 'photo'})
    job = last(product)
    shape((600, 400), (20, 20, 580, 380)).save(product.output / 'late.png')
    product.jobs[job].update(status='completed', result_filename='late.png')
    assert product.get(f'{URL}/{picture["id"]}').json['picture']['run']['status'] == 'done'
    # A cancelled run ignores the answer that still comes, and its file does not stay.
    product.post(f'{URL}/{picture["id"]}/cut', json={'keep': 'subject', 'kind': 'photo'})
    job = last(product)
    monkeypatch.setattr(studio.comfy, 'cancel_plugin_job', lambda *_: 'cancelled')
    version = product.get(f'{URL}/{picture["id"]}').json['picture']['mask']['version']
    assert product.post(f'{URL}/{picture["id"]}/cancel', json={}).json['picture']['run']['status'] == 'cancelled'
    product.answer(job, shape((600, 400), (0, 0, 50, 50)))
    record = product.get(f'{URL}/{picture["id"]}').json['picture']
    assert record['run']['status'] == 'cancelled' and record['mask']['version'] == version
    assert list(product.output.iterdir()) == []


def test_painting_survives_until_reset_or_a_new_answer(product):
    picture = add(product)
    assert product.post(f'{URL}/{picture["id"]}/paint', data={}).status_code == 400, 'there is no mask to paint on yet'
    product.post(f'{URL}/{picture["id"]}/cut', json={'keep': 'subject', 'kind': 'photo'})
    product.answer(last(product), shape((600, 400), (20, 20, 580, 380)))

    def paint():
        strokes = Image.new('LA', (600, 400), (0, 0))
        ImageDraw.Draw(strokes).rectangle((280, 180, 320, 220), fill=(0, 255))
        stream = io.BytesIO()
        strokes.save(stream, 'PNG')
        stream.seek(0)
        return product.post(f'{URL}/{picture["id"]}/paint', data={'strokes': (stream, 'strokes.png')})

    def centre():
        return Image.open(io.BytesIO(product.get(f'{URL}/{picture["id"]}/mask').data)).getpixel((300, 200))
    assert centre() == 255
    painted = paint().json['picture']['mask']
    assert painted['painted'] is True and painted['version'] == 2 and centre() == 0
    assert product.get(f'{URL}/{picture["id"]}/paint').status_code == 200
    reset = product.post(f'{URL}/{picture["id"]}/paint', data={}).json['picture']['mask']
    assert reset['painted'] is False and centre() == 255
    paint()
    product.post(f'{URL}/{picture["id"]}/cut', json={'keep': 'subject', 'kind': 'photo'})
    product.answer(last(product), shape((600, 400), (20, 20, 580, 380)))
    assert centre() == 255 and product.get(f'{URL}/{picture["id"]}').json['picture']['mask']['painted'] is False
    oversized = io.BytesIO()
    Image.new('LA', (4000, 10)).save(oversized, 'PNG')
    oversized.seek(0)
    assert product.post(f'{URL}/{picture["id"]}/paint', data={'strokes': (oversized, 's.png')}).status_code == 400


def test_pictures_stay_in_their_own_folder(product, tmp_path):
    picture = add(product)
    again = add(product)
    assert again['id'] == picture['id'], 'the same picture is found again, with its work'
    assert len(product.get(URL).json['pictures']) == 1
    for ident in ('..', 'x' * 32, '0' * 31, picture['id'].upper() + 'g'):
        assert product.get(f'{URL}/{ident}').status_code == 404
    assert product.get(f'{URL}/{"0" * 32}/mask').status_code == 404
    assert product.get(f'{URL}/{picture["id"]}/mask').status_code == 404, 'no mask before a cut-out'
    assert 'digest' not in product.get(f'{URL}/{picture["id"]}').json['picture']
    for data in (b'not a picture', b''):
        assert product.post(URL, data={'image': (io.BytesIO(data), 'x.png')}).status_code == 400
    assert product.post(URL, json={'surface': 'elsewhere', 'image_id': 1}).status_code == 400
    assert product.post(f'{URL}/{picture["id"]}/dataset', json={'background': 'white'}).status_code == 400
    product.post(f'{URL}/{picture["id"]}/cut', json={'keep': 'subject', 'kind': 'photo'})
    assert product.delete(f'{URL}/{picture["id"]}').status_code == 400, 'not while a cut-out runs on it'
    product.answer(last(product), shape((600, 400), (20, 20, 580, 380)))
    assert product.delete(f'{URL}/{picture["id"]}').status_code == 200
    assert product.get(URL).json['pictures'] == []


def test_a_transparent_upload_is_read_on_white(product):
    source = io.BytesIO()
    Image.new('RGBA', (40, 30), (255, 0, 0, 0)).save(source, 'PNG')
    source.seek(0)
    picture = product.post(URL, data={'image': (source, 'clear.png')}).json['picture']
    assert Image.open(io.BytesIO(product.get(f'{URL}/{picture["id"]}/view').data)).convert('RGB').getpixel((5, 5)) == (255, 255, 255)
