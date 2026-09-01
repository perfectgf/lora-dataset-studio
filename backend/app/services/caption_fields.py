"""The structured tail of a video caption — parsed, never trusted.

C12-C (2026-09-01): the captioner is asked for its 150-200-word paragraph AND,
after a line holding only ``---``, five labelled lines — Subject, Motion,
Setting, Style, Short. The paragraph is what trains today; the fields are what
lets a target with a published budget be served something SHORTER than the
paragraph without anyone cutting a sentence in half (see
video_bank_service.compose_sidecar_text), and what a later UI can show as
facets.

STDLIB ONLY, ON PURPOSE. The inference worker runs in an interpreter this
project does not own (ComfyUI's python_embeded, the Score interpreter) and
imports this file by path to count tokens on the PROSE alone — so nothing here
may import the app, Flask or the database.

The parse is deliberately forgiving and never raises: a model that forgets the
separator, mislabels a line or writes the fields in prose gets its whole text
kept as the caption and ``None`` for the fields. A caption is never lost to a
formatting slip — that would be the empty-sidecar failure in a new coat.
"""
from __future__ import annotations

import re

FIELD_KEYS = ('subject', 'motion', 'setting', 'style', 'short')

# `---` alone on a line, any surrounding blank lines. The LAST such separator
# wins: prose may legitimately contain a dash line? It does not in practice,
# but taking the last one means a stray early dash cannot eat the paragraph.
_SEP = re.compile(r'\n[ \t]*-{3,}[ \t]*\n')
_LABEL = re.compile(r'^\s*\**\s*(subject|motion|setting|style|short)\s*\**\s*:\s*(.*)$',
                    re.I)


def split_caption_fields(raw: str) -> tuple[str, dict | None]:
    """(prose, fields) — fields is a dict with the five keys (missing ones
    None) when a labelled block was found, else None and the whole text is the
    prose. Prose is returned stripped, with any labelled block removed."""
    text = str(raw or '').replace('\r\n', '\n').strip()
    if not text:
        return '', None
    parts = _SEP.split('\n' + text + '\n')
    if len(parts) >= 2:
        prose, tail = '\n'.join(parts[:-1]).strip(), parts[-1]
        fields = _parse_block(tail)
        if fields:
            return prose, fields
    # No separator, or a separator followed by nothing labelled: look for a
    # labelled block starting at the first "Subject:" line instead.
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if _LABEL.match(line) and _LABEL.match(line).group(1).lower() == 'subject':
            fields = _parse_block('\n'.join(lines[i:]))
            if fields:
                return '\n'.join(lines[:i]).strip(), fields
            break
    return text, None


def _parse_block(block: str) -> dict | None:
    out = {}
    for line in str(block).split('\n'):
        m = _LABEL.match(line)
        if not m:
            continue
        key, value = m.group(1).lower(), m.group(2).strip().strip('*').strip()
        if value and key not in out:
            out[key] = value
    if not out:
        return None
    return {k: out.get(k) for k in FIELD_KEYS}


def fields_to_prose(fields: dict | None) -> str:
    """The short form a budgeted target receives instead of the paragraph:
    subject, motion, setting, style — each a sentence, in the order a reader
    needs them, with the model's own words and nothing added."""
    if not fields:
        return ''
    bits = []
    for key in ('subject', 'motion', 'setting', 'style'):
        v = (fields.get(key) or '').strip()
        if v:
            bits.append(v if v.endswith(('.', '!', '?')) else v + '.')
    return ' '.join(bits)
