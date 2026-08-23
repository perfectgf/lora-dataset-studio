"""🎬 Scenes — the ONE definition of what a caption becomes when it is read as a
prompt, shared by the two surfaces that offer scenes.

A Bank and a Dataset both hold one caption per image, and read in row order both
are a SEQUENCE (a storyboard, a shoot, a chapter). The generation panels offer
either as ordered passes of the 📝 prompt axis, so the two must produce byte-
identical cards for the same caption: the ceiling, the cut, the framing fallback
and the label are here and NOWHERE else.

Why a module of its own rather than a helper on one of the two services: the
first copy of this logic already lived in ``image_bank_service``, and the repo
has paid for exactly that shape before (the face pass's size gate diverged
between Bank and Dataset and shipped that way). ``test_scene_caption_parity.py``
reads both sides against this module so they cannot drift apart in silence.

Pure Python — no models, no session, no Flask. Import it from anywhere.
"""

# The dataset shot importer's own prompt ceiling (frontend/src/utils/shotImport.js),
# restated here so a scene can never carry a line that surface would refuse.
SCENE_MAX_PROMPT = 500

# The four framings a scene card may carry. A row classified as anything else —
# 'unknown', or never classified at all — rides as 'body' rather than being
# dropped: refusing a page would silently remove a beat from the MIDDLE of a
# sequence, which is the one failure a reader cannot spot.
SCENE_FRAMINGS = ('face', 'bust', 'body', 'back')
SCENE_DEFAULT_FRAMING = 'body'

# Labels carry the sequence number ("Scene 3 — page_003.jpg") so the reading
# order stays visible wherever the card lands, and stay short enough to sit in a
# card without wrapping the panel.
SCENE_MAX_LABEL = 80


def scene_prompt(caption) -> str:
    """One import-safe prompt line: collapsed whitespace, word-boundary cut."""
    text = ' '.join(str(caption or '').split())
    if len(text) <= SCENE_MAX_PROMPT:
        return text
    cut = text.rfind(' ', 0, SCENE_MAX_PROMPT - 1)
    return text[:cut if cut > 0 else SCENE_MAX_PROMPT - 1] + '…'


def scene_framing(framing) -> str:
    """The card's framing: the row's own when it is one of the four, else 'body'."""
    return framing if framing in SCENE_FRAMINGS else SCENE_DEFAULT_FRAMING


def scene_label(index: int, stem: str) -> str:
    """"Scene 3 — page_003.jpg" — ``index`` is 0-based over the KEPT scenes, so
    the numbering follows the sequence a user reads, not the row ids behind it."""
    label = f'Scene {index + 1} — {stem}'
    return label if len(label) <= SCENE_MAX_LABEL else label[:SCENE_MAX_LABEL - 1] + '…'
