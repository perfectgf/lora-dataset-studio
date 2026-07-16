"""Compose a LoRA Test-Studio results grid into ONE shareable image (« Export grid »).

The Studio shows a run as a grid: rows = checkpoints, columns = strengths, grouped
into FORMAT blocks (16:9, 9:16…) whose header reads « FORMAT 16:9 · CFG 1.0 · 12
STEPS ». This module renders that exact grid — labels included — into a single dark,
sober image (the classic XY plot, but clean) ready to post on Civitai/Reddit.

Two layers, so the pure PIL composition is testable without a DB:
  - ``render_grid_image(title, subtitle, blocks, …)`` — pure composition from
    label strings + on-disk image paths (or None for an empty cell). No DB.
  - ``collect_grid(user_id, dataset_id, …)`` — reads the LoraTestImage rows of ONE
    run (mirrors ``studio_payload``'s family scoping + variant grouping) and resolves
    each (checkpoint × strength) cell to a representative image path.
  - ``export_grid(user_id, dataset_id, …)`` — orchestrates: collect → render →
    encode (JPEG q90 by default, PNG optional). Returns (bytes, mime, meta).

No new dependency: PIL (Pillow) is already used by the app. The final canvas is
capped (``MAX_CANVAS_SIDE`` px per side) so a big run (« 97 img ») downscales
gracefully instead of producing a multi-hundred-megapixel PNG.
"""
from __future__ import annotations

import io
import os

from PIL import Image, ImageDraw, ImageFont

from . import face_dataset_service as fds
from . import lora_test_studio as lts
from ..models import LoraTestImage
from ..utils.comfyui import FAMILY_LABELS, family_of_lora, format_trained_lora_label

# --- Palette (sober dark, coherent with the app's graphite surface) -----------
_BG = (13, 15, 19)          # page background (near-black)
_PANEL = (22, 25, 31)       # empty-cell backdrop
_BORDER = (44, 49, 59)      # thin tile separator
_TEXT = (231, 233, 238)     # primary text
_MUTED = (150, 158, 168)    # secondary text
_HEADER = (185, 191, 255)   # block header (indigo tint)
_ACCENT = (139, 147, 255)   # accent line

# Hard cap on the final image's longest side (px). Beyond this the whole canvas is
# downscaled (LANCZOS) — a large sweep stays under a sane pixel budget for upload.
MAX_CANVAS_SIDE = 8000
# Allowed long-side per tile (2 crans, as specified). Anything else clamps here.
CELL_SIZES = (512, 768)
DEFAULT_CELL_SIZE = 512
# Prompt is truncated to this many chars when the user opts to include it.
_PROMPT_MAX_CHARS = 200

FOOTER_TEXT = 'Made with LoRA Dataset Studio'


class GridExportEmpty(Exception):
    """The requested run has no completed tile to compose (unknown/empty run) →
    the route answers 409 (nothing to export yet)."""


def _fmt_strength(s) -> str:
    """Mirror of the frontend ``fmt`` (utils/studioFormat.js): 2 decimals, but
    « 1.0 » stays readable (1.00→1.0, 0.50→0.5, 1.40→1.4, 0.0→0.0)."""
    try:
        v = f'{float(s):.2f}'.rstrip('0')
    except (TypeError, ValueError):
        return str(s)
    if v.endswith('.'):
        v += '0'
    return v


def _font(size: float) -> ImageFont.FreeTypeFont:
    """Scalable default font (DejaVu Sans, bundled inside Pillow — no system font
    dependency, works on Linux cloud + Windows). Supports « · » and accents."""
    return ImageFont.load_default(size=max(8, int(round(size))))


# --- DB → structured grid -----------------------------------------------------
def _variant_header(z_model_label, aspect, cfg, steps, steps2) -> str:
    """« [model · ] FORMAT 16:9 · CFG 1.0 · 12 STEPS » — same fields as the
    frontend variant caption, uppercased for the banner."""
    parts = []
    if z_model_label:
        parts.append(z_model_label)
    parts.append(f'FORMAT {aspect or "—"}')
    if cfg is not None:
        parts.append(f'CFG {_fmt_strength(cfg)}')
    if steps is not None:
        parts.append(f'{steps}{"/" + str(steps2) if steps2 is not None else ""} STEPS')
    return ' · '.join(parts)


def _pick_representative(cells: list) -> "LoraTestImage | None":
    """One image per (checkpoint × strength) cell: prefer a liked (👍) tile, else
    the most recent completed one — deterministic, matches « show the winner »."""
    done = [c for c in cells if c.status == 'done' and c.filename]
    if not done:
        return None
    liked = [c for c in done if c.rating == 1]
    pool = liked or done
    return max(pool, key=lambda c: c.id)


def collect_grid(user_id, dataset_id, *, family=None, run_seed=None, prompt=None,
                 aspect=None) -> dict | None:
    """Structure ONE run's completed cells into render-ready blocks.

    Mirrors ``studio_payload``: scopes to the effective family (deduced from each
    checkpoint's folder), then to a single run (``run_seed`` + ``prompt``; the
    frontend groups a run by ``run_seed ?? seed`` + prompt). ``run_seed=None`` picks
    the most recent run (the one the Studio shows by default). ``aspect`` (or 'all')
    keeps only that format block. Returns None if the dataset is unknown; raises
    ``GridExportEmpty`` when the run has no tile to show.

    Shape::

        {'title', 'subtitle', 'family', 'aspect', 'run_seed', 'prompt',
         'n_cells', 'blocks': [{'header','col_labels','rows':[{'label','cells'}]}]}
    """
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        return None
    eff = lts._resolve_family(ds, family, lts.available_families(ds))
    rows_all = (LoraTestImage.query.filter_by(dataset_id=dataset_id)
                .order_by(LoraTestImage.id.asc()).all())
    # Family scope (from the checkpoint's folder) + only usable tiles (done + file).
    done = [r for r in rows_all
            if (family_of_lora(r.checkpoint) or 'zimage') == eff
            and r.status == 'done' and r.filename]
    if not done:
        raise GridExportEmpty('no completed test image to export for this pipeline')

    def _run_key(r):
        return r.run_seed if r.run_seed is not None else r.seed

    if run_seed is None:
        latest = max(done, key=lambda r: r.id)   # newest completed tile = current run
        run_seed = _run_key(latest)
        if prompt is None:
            prompt = latest.prompt
    try:
        run_seed = int(run_seed)
    except (TypeError, ValueError):
        raise ValueError(f'invalid run_seed: {run_seed!r}')

    sel = [r for r in done if _run_key(r) == run_seed
           and (prompt is None or (r.prompt or '') == (prompt or ''))]
    want_aspect = (aspect or 'all')
    if want_aspect != 'all':
        sel = [r for r in sel if (r.aspect or '') == want_aspect]
    if not sel:
        raise GridExportEmpty('this run has no completed image to export')

    # Group into variant blocks exactly like the frontend (z_model, aspect, cfg,
    # steps, steps2), rows = checkpoint, cols = strength.
    def _vkey(r):
        return (r.z_model or '', r.aspect or '', r.cfg, r.steps, r.steps2)

    blocks_map: dict = {}
    for r in sel:
        blocks_map.setdefault(_vkey(r), []).append(r)

    ds_dir = fds._dataset_dir(dataset_id)
    blocks = []
    n_cells = 0
    for vkey in sorted(blocks_map, key=lambda k: (str(k[0]), str(k[1]),
                                                  k[2] or 0, k[3] or 0, k[4] or 0)):
        brows = blocks_map[vkey]
        z_model, b_aspect, cfg, steps, steps2 = vkey
        z_label = (lts._basename(z_model).rsplit('.', 1)[0] if z_model else None)
        strengths = sorted({r.strength for r in brows})
        # checkpoints (rows): unique, sorted by label numerically like the grid.
        cp_label = {}
        for r in brows:
            cp_label.setdefault(r.checkpoint,
                                format_trained_lora_label(r.checkpoint, eff)
                                or lts._basename(r.checkpoint).rsplit('.', 1)[0])
        checkpoints = sorted(cp_label, key=lambda c: _natural_key(cp_label[c]))
        by_cell: dict = {}
        for r in brows:
            by_cell.setdefault((r.checkpoint, r.strength), []).append(r)
        row_dicts = []
        for cp in checkpoints:
            cells = []
            for s in strengths:
                rep = _pick_representative(by_cell.get((cp, s), []))
                if rep:
                    cells.append(os.path.join(ds_dir, rep.filename))
                    n_cells += 1
                else:
                    cells.append(None)
            row_dicts.append({'label': cp_label[cp], 'cells': cells})
        blocks.append({
            'header': _variant_header(z_label, b_aspect, cfg, steps, steps2),
            'col_labels': [_fmt_strength(s) for s in strengths],
            'rows': row_dicts,
        })

    # Banner title/subtitle.
    trigger = (ds.trigger_word or ds.name or f'dataset {dataset_id}').strip()
    fam_label = FAMILY_LABELS.get(eff, eff)
    single_model = None
    z_models = {r.z_model for r in sel}
    if len(z_models) == 1:
        only = next(iter(z_models))
        single_model = lts._basename(only).rsplit('.', 1)[0] if only else None
    sub_parts = [fam_label]
    if single_model:
        sub_parts.append(single_model)
    sub_parts.append(f'seed {run_seed}')
    return {
        'title': trigger,
        'subtitle': ' · '.join(sub_parts),
        'family': eff,
        'aspect': want_aspect,
        'run_seed': run_seed,
        'prompt': prompt,
        'n_cells': n_cells,
        'blocks': blocks,
    }


def _natural_key(s: str):
    """Split a label into text/number chunks so 'v2' < 'v10' (numeric-aware sort,
    like the frontend's localeCompare(..., {numeric:true}))."""
    import re
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', s or '')]


# --- Pure PIL composition -----------------------------------------------------
def _fit_lines(draw, text, font, max_w, max_lines=2):
    """Wrap `text` to at most `max_lines` lines within `max_w`, truncating the last
    with « … »."""
    words = (text or '').split(' ')
    lines, cur = [], ''
    for w in words:
        trial = (cur + ' ' + w).strip()
        if not cur or draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = w
            if len(lines) >= max_lines:
                break
    if len(lines) < max_lines and cur:
        lines.append(cur)
    return [_truncate(draw, ln, font, max_w) for ln in lines[:max_lines]] or ['']


def _truncate(draw, text, font, max_w):
    """Trim `text` (adding « … ») until it fits within `max_w`."""
    if draw.textlength(text, font=font) <= max_w:
        return text
    ell = '…'
    out = text
    while out and draw.textlength(out + ell, font=font) > max_w:
        out = out[:-1]
    return (out + ell) if out else ell


def _load_rgb(path, cache):
    """Open an image once (RGB), memoized. None if it can't be read (missing/corrupt
    tile → empty placeholder cell, never a crash)."""
    if path in cache:
        return cache[path]
    img = None
    try:
        with Image.open(path) as im:
            img = im.convert('RGB')
    except Exception:
        img = None
    cache[path] = img
    return img


def _plan(blocks, cell_long, draw, cache):
    """Compute geometry for a candidate tile long-side. Returns a dict with the cell
    box per block, the shared label-column width, per-block sizes and total (W, H)."""
    scale = cell_long / 512.0
    M = max(16, round(28 * scale))
    gap = max(4, round(10 * scale))
    pad = max(6, round(14 * scale))

    f_title = _font(46 * scale)
    f_sub = _font(25 * scale)
    f_hdr = _font(29 * scale)
    f_lbl = _font(26 * scale)
    f_foot = _font(18 * scale)
    lbl_line_h = (f_lbl.getbbox('Ayg')[3] - f_lbl.getbbox('Ayg')[1]) + round(4 * scale)
    hdr_h = (f_hdr.getbbox('Ayg')[3] - f_hdr.getbbox('Ayg')[1]) + round(6 * scale)
    colhdr_h = lbl_line_h + round(6 * scale)

    # Per-block cell box (from the first readable image; all tiles in a block share
    # the same aspect ratio). Fall back to a square if none can be read.
    block_geo = []
    for b in blocks:
        cw = ch = cell_long
        for row in b['rows']:
            found = False
            for p in row['cells']:
                im = _load_rgb(p, cache) if p else None
                if im:
                    iw, ih = im.size
                    r = cell_long / max(iw, ih)
                    cw, ch = max(1, round(iw * r)), max(1, round(ih * r))
                    found = True
                    break
            if found:
                break
        block_geo.append({'cw': cw, 'ch': ch, 'n_cols': len(b['col_labels']),
                          'n_rows': len(b['rows'])})

    # Shared label column width: longest row label, capped, wrapped to ≤2 lines.
    label_max_w = round(cell_long * 1.7)
    longest = 0
    for b in blocks:
        for row in b['rows']:
            longest = max(longest, draw.textlength(row['label'], font=f_lbl))
    Lw = min(label_max_w, max(round(cell_long * 0.55), int(longest) + 2 * pad))

    content_w = 0
    for g in block_geo:
        content_w = max(content_w, Lw + gap + g['n_cols'] * (g['cw'] + gap))

    return {
        'scale': scale, 'M': M, 'gap': gap, 'pad': pad,
        'f_title': f_title, 'f_sub': f_sub, 'f_hdr': f_hdr, 'f_lbl': f_lbl, 'f_foot': f_foot,
        'lbl_line_h': lbl_line_h, 'hdr_h': hdr_h, 'colhdr_h': colhdr_h,
        'block_geo': block_geo, 'Lw': Lw, 'content_w': content_w, 'cell_long': cell_long,
    }


def render_grid_image(title, subtitle, blocks, *, prompt=None, footer_text=FOOTER_TEXT,
                      cell_size=DEFAULT_CELL_SIZE, max_side=MAX_CANVAS_SIDE):
    """Compose the grid into a single dark image. Pure: `blocks` carry label strings
    and on-disk image paths (or None for a missing cell) — no DB access. Returns
    (PIL.Image RGB, downscaled: bool).

    `blocks` = [{'header': str, 'col_labels': [str], 'rows': [{'label': str,
    'cells': [path|None]}]}]. Text scales with `cell_size` so labels stay legible
    once Reddit/Civitai downscale the image. The final canvas is capped to
    `max_side` px per side (whole-image LANCZOS downscale)."""
    if not blocks:
        raise ValueError('nothing to render (no block)')
    cache: dict = {}
    measure_img = Image.new('RGB', (1, 1))
    draw0 = ImageDraw.Draw(measure_img)

    # Pick a tile size that keeps the canvas under the cap (bound memory up front),
    # then a final safety downscale covers any residual overflow from text/margins.
    cell_long = int(cell_size)
    plan = _plan(blocks, cell_long, draw0, cache)
    W, H, _layout = _compute_dims(title, subtitle, prompt, footer_text, plan, draw0)
    downscaled = False
    if max(W, H) > max_side:
        factor = max_side / float(max(W, H))
        new_long = max(96, int(cell_long * factor))
        if new_long < cell_long:
            cell_long = new_long
            downscaled = True
            plan = _plan(blocks, cell_long, draw0, cache)
            W, H, _layout = _compute_dims(title, subtitle, prompt, footer_text, plan, draw0)

    img = Image.new('RGB', (max(1, W), max(1, H)), _BG)
    draw = ImageDraw.Draw(img)
    _paint(draw, img, title, subtitle, prompt, footer_text, blocks, plan, _layout, cache)

    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        downscaled = True
    return img, downscaled


def _compute_dims(title, subtitle, prompt, footer_text, plan, draw):
    """Total (W, H) + a layout dict of y-anchors, given a plan (fixed cell size).
    Prompt wrapping is computed here (and reused by _paint via the layout dict) so
    the measured height matches exactly what gets drawn."""
    M, scale = plan['M'], plan['scale']
    gap = plan['gap']
    f_title, f_sub, f_foot = plan['f_title'], plan['f_sub'], plan['f_foot']
    hdr_h, colhdr_h = plan['hdr_h'], plan['colhdr_h']
    title_h = f_title.getbbox('Ayg')[3] - f_title.getbbox('Ayg')[1]
    sub_h = f_sub.getbbox('Ayg')[3] - f_sub.getbbox('Ayg')[1]

    # Banner: title, subtitle, optional prompt line(s), an accent rule beneath.
    banner_top = M
    y = banner_top + title_h + round(6 * scale) + sub_h
    prompt_lines = []
    if prompt:
        prompt_lines = _fit_lines(draw, f'“{prompt}”', f_sub, max(plan['content_w'], 1), max_lines=2)
        y += len(prompt_lines) * (sub_h + round(10 * scale))
    rule_y = y + round(10 * scale)
    body_top = rule_y + round(16 * scale)

    # Blocks stacked vertically.
    y = body_top
    for g in plan['block_geo']:
        y += hdr_h + round(6 * scale)          # block header
        y += colhdr_h                          # strength labels row
        y += g['n_rows'] * (g['ch'] + gap)     # tile rows
        y += round(22 * scale)                 # gap between blocks
    H = y + M

    title_w = draw.textlength(title or '', font=f_title)
    sub_w = draw.textlength(subtitle or '', font=f_sub)
    foot_w = draw.textlength(footer_text or '', font=f_foot)
    prompt_w = max((draw.textlength(pl, font=f_sub) for pl in prompt_lines), default=0)
    top_line_w = title_w + round(24 * scale) + foot_w   # title + footer share the top line
    W = 2 * M + max(plan['content_w'], top_line_w, sub_w, prompt_w)
    layout = {'banner_top': banner_top, 'prompt_lines': prompt_lines,
              'rule_y': rule_y, 'body_top': body_top}
    return int(W), int(H), layout


def _paint(draw, img, title, subtitle, prompt, footer_text, blocks, plan, layout, cache):
    M, gap, pad, scale = plan['M'], plan['gap'], plan['pad'], plan['scale']
    f_title, f_sub, f_hdr = plan['f_title'], plan['f_sub'], plan['f_hdr']
    f_lbl, f_foot = plan['f_lbl'], plan['f_foot']
    Lw = plan['Lw']
    title_h = f_title.getbbox('Ayg')[3] - f_title.getbbox('Ayg')[1]

    # --- Banner ---------------------------------------------------------------
    y = layout['banner_top']
    draw.text((M, y), title or '', font=f_title, fill=_TEXT,
              stroke_width=max(1, round(scale)), stroke_fill=_TEXT)
    # Footer (discreet promo) top-right, baseline-aligned with the title.
    if footer_text:
        fw = draw.textlength(footer_text, font=f_foot)
        draw.text((img.size[0] - M - fw, y + title_h * 0.35), footer_text,
                  font=f_foot, fill=_MUTED)
    y += title_h + round(6 * scale)
    sub_h = f_sub.getbbox('Ayg')[3] - f_sub.getbbox('Ayg')[1]
    draw.text((M, y), subtitle or '', font=f_sub, fill=_MUTED)
    y += sub_h
    for pl in layout['prompt_lines']:
        y += round(2 * scale)
        draw.text((M, y + round(8 * scale)), pl, font=f_sub, fill=_MUTED)
        y += sub_h + round(8 * scale)
    # Accent rule under the banner.
    draw.rectangle([M, layout['rule_y'], img.size[0] - M, layout['rule_y'] + max(2, round(2 * scale))],
                   fill=_ACCENT)

    # --- Blocks ---------------------------------------------------------------
    y = layout['body_top']
    hdr_h, colhdr_h = plan['hdr_h'], plan['colhdr_h']
    for b, g in zip(blocks, plan['block_geo']):
        cw, ch = g['cw'], g['ch']
        draw.text((M, y), b['header'], font=f_hdr, fill=_HEADER)
        y += hdr_h + round(6 * scale)
        grid_x0 = M + Lw + gap
        # Strength column labels, centered over each column.
        for i, cl in enumerate(b['col_labels']):
            cx = grid_x0 + i * (cw + gap) + cw / 2
            draw.text((cx, y + colhdr_h / 2), cl, font=f_lbl, fill=_TEXT, anchor='mm')
        y += colhdr_h
        # Rows.
        for row in b['rows']:
            row_top = y
            row_cy = row_top + ch / 2
            lines = _fit_lines(draw, row['label'], f_lbl, Lw - pad, max_lines=2)
            total_h = len(lines) * plan['lbl_line_h']
            ly = row_cy - total_h / 2
            for ln in lines:
                draw.text((M, ly), ln, font=f_lbl, fill=_TEXT, anchor='la')
                ly += plan['lbl_line_h']
            for i in range(g['n_cols']):
                x0 = grid_x0 + i * (cw + gap)
                box = [x0, row_top, x0 + cw, row_top + ch]
                path = row['cells'][i] if i < len(row['cells']) else None
                im = _load_rgb(path, cache) if path else None
                if im is not None:
                    tile = im.copy()
                    tile.thumbnail((cw, ch), Image.Resampling.LANCZOS)
                    ox = x0 + (cw - tile.size[0]) // 2
                    oy = row_top + (ch - tile.size[1]) // 2
                    img.paste(tile, (ox, oy))
                    draw.rectangle(box, outline=_BORDER, width=max(1, round(scale)))
                else:
                    # Missing cell (checkpoint not tested at this strength) — a clean
                    # blank panel keeps the XY grid aligned without a noisy glyph.
                    draw.rectangle(box, fill=_PANEL, outline=_BORDER, width=max(1, round(scale)))
            y = row_top + ch + gap
        y += round(22 * scale)


# --- Orchestration ------------------------------------------------------------
def _sanitize_name(s: str) -> str:
    keep = ''.join(c if (c.isalnum() or c in '-_') else '_' for c in (s or ''))
    return keep.strip('_') or 'grid'


def export_grid(user_id, dataset_id, *, family=None, run_seed=None, prompt=None,
                aspect=None, include_prompt=False, cell_size=None, fmt='jpeg',
                footer=True) -> tuple[bytes, str, dict]:
    """Collect ONE run's grid, render it, encode it. Returns (bytes, mime, meta).

    meta = {downscaled, width, height, n_cells, n_blocks, download_name, format}.
    Raises ValueError (→400) on bad params, GridExportEmpty (→409) on an empty/
    unknown run; returns None-collect (unknown dataset) as ValueError to the route
    which has already 404'd. `include_prompt` defaults False (prompts can be
    personal/NSFW); when True the prompt is truncated to _PROMPT_MAX_CHARS."""
    grid = collect_grid(user_id, dataset_id, family=family, run_seed=run_seed,
                        prompt=prompt, aspect=aspect)
    if grid is None:
        raise ValueError('dataset not found')

    try:
        cs = int(cell_size) if cell_size is not None else DEFAULT_CELL_SIZE
    except (TypeError, ValueError):
        cs = DEFAULT_CELL_SIZE
    cs = min(CELL_SIZES, key=lambda v: abs(v - cs))  # clamp to an allowed cran

    shown_prompt = None
    if include_prompt and grid.get('prompt'):
        p = grid['prompt'].strip()
        shown_prompt = (p[:_PROMPT_MAX_CHARS].rstrip() + '…') if len(p) > _PROMPT_MAX_CHARS else p

    image, downscaled = render_grid_image(
        grid['title'], grid['subtitle'], grid['blocks'],
        prompt=shown_prompt, footer_text=(FOOTER_TEXT if footer else None),
        cell_size=cs)

    ext = 'png' if str(fmt).lower() == 'png' else 'jpg'
    mime = 'image/png' if ext == 'png' else 'image/jpeg'
    buf = io.BytesIO()
    if ext == 'png':
        image.save(buf, format='PNG', optimize=True)
    else:
        image.save(buf, format='JPEG', quality=90, optimize=True, progressive=True)
    data = buf.getvalue()

    asp_tag = _sanitize_name(grid['aspect']) if grid['aspect'] and grid['aspect'] != 'all' else 'all'
    name = f'lora-grid_{_sanitize_name(grid["title"])}_{asp_tag}_seed{grid["run_seed"]}.{ext}'
    meta = {'downscaled': downscaled, 'width': image.size[0], 'height': image.size[1],
            'n_cells': grid['n_cells'], 'n_blocks': len(grid['blocks']),
            'download_name': name, 'format': ext}
    return data, mime, meta
