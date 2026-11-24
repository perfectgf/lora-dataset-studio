"""Pixels and graphs: what the models answer becomes one alpha mask, without a GPU."""
import json
from pathlib import Path
import sys

import pytest
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'backend'), str(ROOT / 'bundled/remove_background')]

from lds_remove_background import graphs, masks


def disc(size, box, blur=0):
    mask = Image.new('L', size, 0)
    ImageDraw.Draw(mask).ellipse(box, fill=255)
    return mask.filter(ImageFilter.GaussianBlur(blur)) if blur else mask


def test_sam_decides_the_shape_and_the_matte_its_edge():
    target = disc((400, 400), (100, 100, 300, 300))
    # The matte also took a table the object stands on, and has a softer edge everywhere.
    matte = disc((400, 400), (98, 98, 302, 302), blur=3)
    ImageDraw.Draw(matte).rectangle((0, 280, 400, 400), fill=255)
    alpha, matted = masks.follow_target(target, matte)
    assert matted
    assert alpha.getpixel((200, 200)) == 255
    assert alpha.getpixel((40, 360)) == 0 and alpha.getpixel((200, 390)) == 0, 'the table is not part of the object'
    assert 0 < alpha.getpixel((200, 99)) < 255, 'a soft edge comes from the matte'


def test_a_part_the_matte_lost_is_kept_and_thin_strands_survive():
    target = disc((400, 400), (100, 100, 300, 300))
    matte = disc((400, 400), (98, 98, 302, 302), blur=2)
    ImageDraw.Draw(matte).rectangle((150, 180, 250, 260), fill=0)    # a dark sleeve against a dark wall
    ImageDraw.Draw(matte).line((200, 100, 203, 94), fill=200, width=2)  # one strand of hair, just outside SAM's edge
    alpha, _ = masks.follow_target(target, matte)
    assert alpha.getpixel((200, 220)) > 240
    assert alpha.getpixel((202, 96)) > 60


def test_see_through_parts_of_the_object_stay_see_through():
    target = disc((400, 400), (100, 100, 300, 300))
    matte = disc((400, 400), (98, 98, 302, 302), blur=1)
    ImageDraw.Draw(matte).rectangle((160, 160, 240, 240), fill=110)   # glass
    alpha, _ = masks.follow_target(target, matte)
    assert 90 < alpha.getpixel((200, 200)) < 130


def test_a_matte_of_something_else_leaves_the_whole_shape_to_sam():
    target = disc((400, 400), (100, 100, 200, 200))
    alpha, matted = masks.follow_target(target, disc((400, 400), (250, 250, 390, 390)))
    assert matted is False
    assert alpha.getpixel((150, 150)) == 255 and alpha.getpixel((320, 320)) == 0
    assert masks.follow_target(Image.new('L', (40, 40), 0), disc((40, 40), (5, 5, 30, 30)))[1] is False


def test_a_second_look_only_when_the_subject_leaves_room():
    assert masks.close_up(disc((1000, 1000), (400, 400, 600, 600))) is not None
    assert masks.close_up(disc((1000, 1000), (50, 50, 950, 950))) is None
    assert masks.close_up(Image.new('L', (100, 100), 0)) is None
    box = masks.close_up(disc((1000, 1000), (0, 0, 200, 200)))
    assert box[:2] == (0, 0), 'the margin stays on the picture'


def test_the_brush_touches_only_what_was_painted_whatever_its_size():
    mask = disc((800, 600), (200, 150, 600, 450), blur=4)
    strokes = Image.new('LA', (400, 300), (0, 0))
    ImageDraw.Draw(strokes).rectangle((10, 10, 40, 40), fill=(255, 255))    # keep
    ImageDraw.Draw(strokes).rectangle((190, 140, 210, 160), fill=(0, 255))  # remove
    painted = masks.painted(mask, strokes)
    assert painted.getpixel((50, 50)) == 255 and painted.getpixel((400, 300)) == 0
    assert painted.getpixel((300, 400)) == mask.getpixel((300, 400))


def test_nothing_of_the_background_is_left_under_transparent_pixels():
    picture = Image.new('RGB', (300, 300), (10, 200, 30))
    ImageDraw.Draw(picture).ellipse((80, 80, 220, 220), fill=(200, 20, 20))
    alpha = disc((300, 300), (80, 80, 220, 220), blur=3)
    result = masks.cut_out(picture, alpha, clean=True)
    assert result.mode == 'RGBA' and result.getpixel((5, 5)) == (0, 0, 0, 0)
    edge = next(x for x in range(221, 240) if 20 < alpha.getpixel((x, 150)) < 120)
    red, green, _blue, _alpha = result.getpixel((edge, 150))
    assert red > green, 'the green background is out of the soft edge'
    raw = masks.cut_out(picture, alpha, clean=False).getpixel((edge, 150))
    assert raw[1] > raw[0]
    white = masks.cut_out(picture, alpha, background='white', clean=True)
    assert white.mode == 'RGB' and white.getpixel((5, 5)) == (255, 255, 255)


def test_cleaning_edges_keeps_the_colour_of_wide_glass():
    picture = Image.new('RGB', (600, 600), (40, 160, 120))
    ImageDraw.Draw(picture).rectangle((200, 40, 400, 140), fill=(90, 60, 20))
    alpha = Image.new('L', (600, 600), 0)
    ImageDraw.Draw(alpha).rectangle((200, 40, 400, 140), fill=255)   # a solid cap
    ImageDraw.Draw(alpha).rectangle((150, 141, 450, 560), fill=110)  # a see-through bottle
    cleaned = masks.clean_edges(picture, alpha)
    assert cleaned.getpixel((300, 400)) == (40, 160, 120)


def test_stray_dots_go_and_the_outline_stays():
    mask = disc((500, 500), (100, 100, 400, 400))
    ImageDraw.Draw(mask).rectangle((20, 20, 22, 22), fill=255)
    cleaned = masks.without_specks(mask)
    assert cleaned.getpixel((21, 21)) == 0
    assert cleaned.getpixel((250, 101)) == mask.getpixel((250, 101)) == 255
    dot = Image.new('L', (500, 500), 0)
    dot.putpixel((10, 10), 255)
    assert masks.without_specks(dot).getpixel((10, 10)) == 255, 'a shape that is all dots is kept'


@pytest.mark.parametrize('typed,found', [
    ('keep the man, remove everything else', ['man']),
    ('woman in red, dog', ['woman in red', 'dog']),
    ('Only a (big) dog: 3.', ['big dog 3']),
    ('the  ', []),
    ('dog, Dog, dog', ['dog']),
    ('couple without the background', ['couple']),
])
def test_an_instruction_becomes_the_phrases_sam_reads(typed, found):
    assert graphs.phrases(typed) == found


def test_every_graph_uses_only_the_nodes_it_declares():
    files = {'rmbg': 'r', 'rmbg_text_encoder': 't', 'rmbg_vae': 'v'}
    built = {
        'find words': (graphs.find('a.png', 'sam', 'p', words=['dog', 'cat']), graphs.FIND_NODES),
        'find points': (graphs.find('a.png', 'sam', 'p', points=[{'x': 1, 'y': 2, 'keep': True},
                                                                  {'x': 3, 'y': 4, 'keep': False}]), graphs.FIND_NODES),
        'matte': (graphs.matte('a.png', 'birefnet', 'p'), graphs.MATTE_NODES),
        'fine': (graphs.fine_matte('a.png', files, (1024, 1024), 'p'), graphs.FINE_NODES),
    }
    for name, (graph, allowed) in built.items():
        assert {node['class_type'] for node in graph.values()} <= allowed, name
        saved = [node for node in graph.values() if node['class_type'] == 'SaveImage']
        assert len(saved) == 1 and saved[0]['inputs']['filename_prefix'] == 'p', name
        for node in graph.values():
            for value in node['inputs'].values():
                if isinstance(value, list):
                    assert value[0] in graph, name
    assert built['find words'][0]['3']['inputs']['text'] == 'dog: 16, cat: 16'
    points = built['find points'][0]['4']['inputs']
    assert json.loads(points['positive_coords']) == [{'x': 1, 'y': 2}]
    assert json.loads(points['negative_coords']) == [{'x': 3, 'y': 4}]


def test_rmbg_gets_a_size_its_node_does_not_crop():
    for width, height in ((1000, 1000), (4000, 900), (333, 1777), (1, 9000)):
        size = graphs.fine_size(width, height)
        assert all(side % 16 == 0 and 16 <= side <= 2048 for side in size)
    assert graphs.fitted(300, 200, 2048) == (300, 200), 'a small picture is never enlarged'
    assert graphs.fitted(4096, 1024, 2048) == (2048, 512)
