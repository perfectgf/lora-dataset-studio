"""Dataset coverage — what the set never showed.

Sits directly under the composition meter and answers the question the meter
cannot: the meter counts face / bust / body / back against a target, so a set can
be fully green while every single image is the same person, front-on, in one
outfit, under one light. That set trains a LoRA that reproduces exactly that and
generalises to nothing, and nothing in the app said a word about it.

This composes what already exists rather than adding a model:

  * the stored `framing` column (📐 the vision classifier already wrote it) —
    reused here only to say when it is MISSING, since the bar above already
    renders the counts;
  * the captions the caption pass already wrote, scanned by the pure lexicon in
    ``caption_coverage`` for camera view, camera height, lighting, setting,
    outfit and expression.

No new inference, no GPU, no network — one query and pure string work. The whole
point is that it costs nothing to look, so it can be looked at constantly.

Kind-aware: character / concept / style are judged on different axes (see
``caption_coverage.KIND_AXES``), because "only one outfit" is a defect for a
character and meaningless for a style.
"""
from __future__ import annotations

from ..models import FaceDatasetImage
from . import caption_coverage

# The pool the panel reads is deliberately the SAME one the composition meter
# above it counts (everything not rejected and not failed). Two adjacent readouts
# disagreeing about which images they describe is worse than either being wrong.
_EXCLUDED = ('reject', 'failed')


def _pool(dataset_id):
    return (FaceDatasetImage.query
            .filter(FaceDatasetImage.dataset_id == dataset_id,
                    FaceDatasetImage.status.notin_(_EXCLUDED))
            .all())


def coverage(user_id, dataset_id) -> dict | None:
    """Read-only variety report for the dataset, or None when it is gone.

    Never mutates anything. Returns the per-axis distributions the panel renders
    plus generated advice, plus the two honesty counters (`uncaptioned`,
    `unframed`) that let the UI say *why* an axis is blank instead of drawing an
    empty bar and implying an absence it never measured.
    """
    from .face_dataset_service import get_dataset

    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return None

    imgs = _pool(dataset_id)
    kind = (getattr(ds, 'kind', None) or 'character').lower()
    # The ids ride along so each bucket can say WHICH images it counted, which is
    # what lets a chip become a filter on the grid ("show me those 3 profiles")
    # instead of a number you then hunt for by eye. They describe THIS pool, so a
    # chip filter shows exactly what the panel counted — never a rejected image
    # the panel deliberately left out.
    report = caption_coverage.analyse([i.caption for i in imgs], kind=kind,
                                      ids=[i.id for i in imgs])

    # Framing is NOT re-reported bucket by bucket (the composition bar above owns
    # that); what the bar cannot say is how many images it silently ignored.
    known = sum(1 for i in imgs if i.framing in ('face', 'bust', 'body', 'back'))
    report['unframed'] = len(imgs) - known
    report['framed'] = known

    notes = caption_coverage.advice(report)
    # The framing gap is prepended rather than folded into the lexicon advice: it
    # is about a DIFFERENT pass, and it is the one the user can act on in one
    # click via the classify button that already sits next to this panel.
    if imgs and report['unframed']:
        notes.insert(0, {
            'tone': 'info',
            'text': f'{report["unframed"]} of {len(imgs)} images have no shot type yet — '
                    f'they count for nothing in the composition bar above. Run the '
                    f'📐 classify pass to bring them in.'})
    report['advice'] = notes
    report['dataset_id'] = ds.id
    return report
