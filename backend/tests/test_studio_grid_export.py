"""Tests for the « Export grid » composition (studio_grid_export).

Pure-PIL rendering is exercised from fixture tiles (tiny solid-color PNGs), and
the DB collector/orchestrator from real LoraTestImage rows + on-disk files. No
ComfyUI, no network — the composition is deterministic."""
import os

from PIL import Image


def _tile(path, w, h, color):
    Image.new('RGB', (w, h), color).save(path)
    return path


def _blocks_from_dir(d, *, rows=2, cols=3, aspect='16:9', tw=64, th=36, hole=False):
    """Build a render-ready block structure backed by fixture tiles on disk."""
    os.makedirs(d, exist_ok=True)
    row_dicts = []
    for r in range(rows):
        cells = []
        for c in range(cols):
            if hole and r == rows - 1 and c == cols - 1:
                cells.append(None)          # a missing cell → blank placeholder
                continue
            p = _tile(os.path.join(d, f'{r}_{c}.png'), tw, th, (40 + r * 30, 60, 90 + c * 20))
            cells.append(p)
        row_dicts.append({'label': f'ckpt · {2000 + r * 500} steps', 'cells': cells})
    return [{'header': f'FORMAT {aspect} · CFG 1.0 · 12 STEPS',
             'col_labels': [f'{c / 2:.1f}' for c in range(cols)], 'rows': row_dicts}]


# --- pure renderer ------------------------------------------------------------
def test_render_grid_image_composes_dark_canvas(tmp_path):
    from app.services import studio_grid_export as sge
    blocks = _blocks_from_dir(str(tmp_path / 'fix'), rows=3, cols=4, hole=True)
    img, downscaled = sge.render_grid_image('trigger', 'Z-Image · seed 42', blocks,
                                            cell_size=512)
    assert img.mode == 'RGB'
    # A real grid of 3×4 512px-ish tiles is comfortably over 1500px wide (4 cols +
    # label column) and 900px tall (3 rows of 16:9 tiles + banner).
    assert img.size[0] > 1500 and img.size[1] > 900
    assert not downscaled
    # Top-left pixel is the dark background (banner area), not white.
    assert img.getpixel((2, 2)) == sge._BG


def test_render_caps_canvas_and_flags_downscale(tmp_path):
    from app.services import studio_grid_export as sge
    blocks = _blocks_from_dir(str(tmp_path / 'fix'), rows=3, cols=5)
    img, downscaled = sge.render_grid_image('t', 's', blocks, cell_size=512, max_side=600)
    assert downscaled is True
    assert max(img.size) <= 600


def test_render_prompt_grows_the_banner(tmp_path):
    """Including the prompt adds banner lines → a taller canvas. Proves the
    include-prompt path actually bakes the text in."""
    from app.services import studio_grid_export as sge
    blocks = _blocks_from_dir(str(tmp_path / 'fix'), rows=1, cols=2)
    no_prompt, _ = sge.render_grid_image('t', 's', blocks, prompt=None, cell_size=512)
    with_prompt, _ = sge.render_grid_image('t', 's', blocks,
                                           prompt='a woman in a field', cell_size=512)
    assert with_prompt.size[1] > no_prompt.size[1]


def test_render_missing_file_falls_back_to_placeholder(tmp_path):
    from app.services import studio_grid_export as sge
    blocks = [{'header': 'FORMAT 1:1 · 8 STEPS', 'col_labels': ['1.0'],
               'rows': [{'label': 'x', 'cells': [str(tmp_path / 'does_not_exist.png')]}]}]
    img, _ = sge.render_grid_image('t', 's', blocks, cell_size=256)  # must not raise
    assert img.mode == 'RGB'


def test_render_empty_blocks_raises(app):
    from app.services import studio_grid_export as sge
    import pytest
    with pytest.raises(ValueError):
        sge.render_grid_image('t', 's', [], cell_size=512)


def test_fmt_strength_matches_frontend():
    from app.services.studio_grid_export import _fmt_strength
    assert _fmt_strength(1.0) == '1.0'
    assert _fmt_strength(0.5) == '0.5'
    assert _fmt_strength(1.4) == '1.4'
    assert _fmt_strength(0.0) == '0.0'
    assert _fmt_strength(2.0) == '2.0'
    # Extended (> 2.0) strengths render as clean generic labels, no clamp/rounding.
    assert _fmt_strength(2.5) == '2.5'
    assert _fmt_strength(3.5) == '3.5'
    assert _fmt_strength(4.0) == '4.0'


# --- DB collector -------------------------------------------------------------
def _make_run(app, tmp_path, *, run_seed, checkpoints, strengths, aspect='16:9',
              prompt='portrait', rating_of=None):
    """Create done LoraTestImage rows + fixture files for a run. Returns dataset id."""
    from app.services import face_dataset_service as svc
    from app.models import LoraTestImage
    from app.config import LOCAL_USER
    ds = svc.create_dataset(LOCAL_USER, f'DS{run_seed}', 'troubeau')
    ds_dir = svc._dataset_dir(ds.id)
    os.makedirs(ds_dir, exist_ok=True)
    for cp in checkpoints:
        for s in strengths:
            fn = f'{os.path.basename(cp)}_{s}_{run_seed}.png'
            _tile(os.path.join(ds_dir, fn), 64, 36, (80, 90, 120))
            rating = 1 if (rating_of and rating_of == (cp, s)) else 0
            svc.db.session.add(LoraTestImage(
                dataset_id=ds.id, checkpoint=cp, strength=s, aspect=aspect,
                filename=fn, status='done', rating=rating, run_seed=run_seed,
                seed=run_seed, prompt=prompt, cfg=1.0, steps=12))
    svc.db.session.commit()
    return ds.id


def test_collect_grid_builds_rows_cols_blocks(app, tmp_path):
    from app.services import studio_grid_export as sge
    from app.config import LOCAL_USER
    with app.app_context():
        cks = ['z image\\lora_troubeau_000002000.safetensors',
               'z image\\lora_troubeau_000002500.safetensors']
        ds_id = _make_run(app, tmp_path, run_seed=555, checkpoints=cks,
                          strengths=[0.5, 1.0, 1.4])
        grid = sge.collect_grid(LOCAL_USER, ds_id)
        assert grid['family'] == 'zimage'
        assert grid['run_seed'] == 555
        assert len(grid['blocks']) == 1
        block = grid['blocks'][0]
        assert block['col_labels'] == ['0.5', '1.0', '1.4']
        assert len(block['rows']) == 2                      # two checkpoints
        assert grid['n_cells'] == 6                         # 2 × 3, all filled
        # Row labels are the human LoRA labels (trigger · steps …), not raw filenames.
        assert all('troubeau' in r['label'] for r in block['rows'])
        assert 'FORMAT 16:9' in block['header'] and 'CFG 1.0' in block['header']


def test_collect_grid_renders_extended_strength_columns(app, tmp_path):
    """A run swept beyond 2.0 (progressive-disclosure « + » range) exports its
    extended columns generically — 2.5 / 3.5 / 4.0 show up as real grid headers."""
    from app.services import studio_grid_export as sge
    from app.config import LOCAL_USER
    with app.app_context():
        cks = ['z image\\lora_troubeau_000002000.safetensors']
        ds_id = _make_run(app, tmp_path, run_seed=777, checkpoints=cks,
                          strengths=[1.0, 2.5, 3.5, 4.0])
        grid = sge.collect_grid(LOCAL_USER, ds_id)
        block = grid['blocks'][0]
        assert block['col_labels'] == ['1.0', '2.5', '3.5', '4.0']
        assert grid['n_cells'] == 4


def test_collect_grid_unknown_dataset_returns_none(app):
    from app.services import studio_grid_export as sge
    from app.config import LOCAL_USER
    with app.app_context():
        assert sge.collect_grid(LOCAL_USER, 999999) is None


def test_collect_grid_empty_run_raises(app):
    from app.services import studio_grid_export as sge, face_dataset_service as svc
    from app.config import LOCAL_USER
    import pytest
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Empty', 'troubeau')
        with pytest.raises(sge.GridExportEmpty):
            sge.collect_grid(LOCAL_USER, ds.id)


def test_collect_grid_default_picks_most_recent_run(app, tmp_path):
    from app.services import studio_grid_export as sge
    from app.config import LOCAL_USER
    with app.app_context():
        # Two runs on the SAME dataset: the later rows (higher id) = run 222.
        from app.services import face_dataset_service as svc
        from app.models import LoraTestImage
        ds = svc.create_dataset(LOCAL_USER, 'Multi', 'troubeau')
        ds_dir = svc._dataset_dir(ds.id)
        os.makedirs(ds_dir, exist_ok=True)
        cp = 'z image\\lora_troubeau_000002000.safetensors'
        for run_seed in (111, 222):
            for s in (0.5, 1.0):
                fn = f'r{run_seed}_{s}.png'
                _tile(os.path.join(ds_dir, fn), 64, 36, (10, 20, 30))
                svc.db.session.add(LoraTestImage(
                    dataset_id=ds.id, checkpoint=cp, strength=s, aspect='16:9',
                    filename=fn, status='done', run_seed=run_seed, seed=run_seed,
                    prompt='p', cfg=1.0, steps=12))
            svc.db.session.commit()
        grid = sge.collect_grid(LOCAL_USER, ds.id)             # default = most recent
        assert grid['run_seed'] == 222
        grid_a = sge.collect_grid(LOCAL_USER, ds.id, run_seed=111)  # explicit older run
        assert grid_a['run_seed'] == 111


def test_collect_grid_prefers_liked_representative(app, tmp_path):
    from app.services import studio_grid_export as sge, face_dataset_service as svc
    from app.models import LoraTestImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Rep', 'troubeau')
        ds_dir = svc._dataset_dir(ds.id)
        os.makedirs(ds_dir, exist_ok=True)
        cp = 'z image\\lora_troubeau_000002000.safetensors'
        for fn, rating in (('plain.png', 0), ('liked.png', 1)):
            _tile(os.path.join(ds_dir, fn), 64, 36, (10, 20, 30))
            svc.db.session.add(LoraTestImage(
                dataset_id=ds.id, checkpoint=cp, strength=1.0, aspect='16:9',
                filename=fn, status='done', rating=rating, run_seed=7, seed=7,
                prompt='p', cfg=1.0, steps=12))
        svc.db.session.commit()
        grid = sge.collect_grid(LOCAL_USER, ds.id)
        cell_path = grid['blocks'][0]['rows'][0]['cells'][0]
        assert os.path.basename(cell_path) == 'liked.png'


def test_collect_grid_aspect_filter_isolates_one_format(app, tmp_path):
    from app.services import studio_grid_export as sge, face_dataset_service as svc
    from app.models import LoraTestImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Fmt', 'troubeau')
        ds_dir = svc._dataset_dir(ds.id)
        os.makedirs(ds_dir, exist_ok=True)
        cp = 'z image\\lora_troubeau_000002000.safetensors'
        for aspect in ('16:9', '9:16'):
            fn = f'{aspect.replace(":", "x")}.png'
            _tile(os.path.join(ds_dir, fn), 64, 36, (10, 20, 30))
            svc.db.session.add(LoraTestImage(
                dataset_id=ds.id, checkpoint=cp, strength=1.0, aspect=aspect,
                filename=fn, status='done', run_seed=9, seed=9, prompt='p',
                cfg=1.0, steps=12))
        svc.db.session.commit()
        assert len(sge.collect_grid(LOCAL_USER, ds.id, aspect='all')['blocks']) == 2
        one = sge.collect_grid(LOCAL_USER, ds.id, aspect='9:16')
        assert len(one['blocks']) == 1 and '9:16' in one['blocks'][0]['header']


# --- orchestrator: encode + options -------------------------------------------
def test_export_grid_returns_jpeg_by_default(app, tmp_path):
    from app.services import studio_grid_export as sge
    from app.config import LOCAL_USER
    with app.app_context():
        ds_id = _make_run(app, tmp_path, run_seed=1, strengths=[0.5, 1.0],
                          checkpoints=['z image\\lora_troubeau_000002000.safetensors'])
        data, mime, meta = sge.export_grid(LOCAL_USER, ds_id)
        assert mime == 'image/jpeg'
        assert data[:3] == b'\xff\xd8\xff'                 # JPEG magic
        assert meta['format'] == 'jpg' and meta['n_cells'] == 2
        assert meta['download_name'].endswith('.jpg')


def test_export_grid_png_option(app, tmp_path):
    from app.services import studio_grid_export as sge
    from app.config import LOCAL_USER
    with app.app_context():
        ds_id = _make_run(app, tmp_path, run_seed=2, strengths=[1.0],
                          checkpoints=['z image\\lora_troubeau_000002000.safetensors'])
        data, mime, meta = sge.export_grid(LOCAL_USER, ds_id, fmt='png')
        assert mime == 'image/png' and data[:4] == b'\x89PNG'
        assert meta['download_name'].endswith('.png')


def test_export_grid_prompt_off_by_default(app, tmp_path):
    """Default must NOT bake the prompt in (prompts can be personal/NSFW): the
    default image equals the include_prompt=False one and is shorter than the
    opt-in one."""
    from app.services import studio_grid_export as sge
    from app.config import LOCAL_USER
    with app.app_context():
        ds_id = _make_run(app, tmp_path, run_seed=3, strengths=[0.5, 1.0], prompt='a secret prompt',
                          checkpoints=['z image\\lora_troubeau_000002000.safetensors'])
        _, _, meta_default = sge.export_grid(LOCAL_USER, ds_id)
        _, _, meta_off = sge.export_grid(LOCAL_USER, ds_id, include_prompt=False)
        _, _, meta_on = sge.export_grid(LOCAL_USER, ds_id, include_prompt=True)
        assert meta_default['height'] == meta_off['height']
        assert meta_on['height'] > meta_off['height']       # prompt adds banner lines


def test_export_grid_cell_size_clamped_to_allowed_cran(app, tmp_path):
    from app.services import studio_grid_export as sge
    from app.config import LOCAL_USER
    with app.app_context():
        ds_id = _make_run(app, tmp_path, run_seed=4, strengths=[1.0],
                          checkpoints=['z image\\lora_troubeau_000002000.safetensors'])
        # An absurd cell_size clamps to the nearest allowed cran (512/768), not literal.
        small, _, _ = sge.export_grid(LOCAL_USER, ds_id, cell_size=10)
        big, _, _ = sge.export_grid(LOCAL_USER, ds_id, cell_size=9000)
        assert len(small) > 0 and len(big) > 0


def test_export_grid_empty_run_raises_grid_export_empty(app):
    from app.services import studio_grid_export as sge, face_dataset_service as svc
    from app.config import LOCAL_USER
    import pytest
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Empty2', 'troubeau')
        with pytest.raises(sge.GridExportEmpty):
            sge.export_grid(LOCAL_USER, ds.id)
