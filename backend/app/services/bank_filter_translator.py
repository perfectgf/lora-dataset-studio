"""Turn a sentence like "I want an amateur dataset" into the bank's OWN filter.

The model does not look at a single image and does not choose which images you
get. It moves your controls, and the grid's existing counters then say — measured,
without the model — how many images that lands on. That split is the whole design:
a wrong translation costs you one glance at chips you can edit, not a silent
selection of 3 000 pictures you have to trust.

WHY A TRANSLATOR AND NOT A JUDGE. Asking a vision model to rate 25 000 images is
hours of GPU behind the exclusive fence, and it would be WORSE: the aesthetic head
is trained for exactly that judgement and already ran. Everything a request like
"amateur" needs — medium, the aesthetic flag, resolution tiers, noise — is measured
and indexed. The only thing missing is the sentence-to-facet step, which is small,
cheap, and verifiable.

THE VOCABULARY IS IMPORTED, NEVER RETYPED. Every allowed value comes from
`image_bank_service` at call time. A copied list is a list that drifts, and a
drifted list here would let the model emit a facet value the grid silently ignores
— the user would read a confident summary over an unchanged grid.

WHAT IT REFUSES TO DO, on purpose:

* No negated CLIP phrase. MEASURED on this app's own encoder: "a woman without a
  bikini" returns 60% bikinis against a 10.1% baseline, and "an abstract 3d shape
  without blue" ranked blue images 1st-4th. Negation does not weaken a concept, it
  SUMMONS it. A request to exclude something becomes the word-exclude box (which
  guarantees absence) or is reported as unsupported — never a "without" sent to
  CLIP.
* No invented axis. A request that needs what the picture CONTAINS ("women
  outdoors") has nothing to land on while captions cover ~1% of a bank and framing
  ~0.03%; saying so is the useful answer. A plausible filter over axes that cannot
  express the request would return 3 000 convincing, unrelated images.
* No style facet while the style partition is degenerate. Single-link clustering
  collapses (measured: one group holding 24 928 of 24 931 images), so filtering by
  it would look like a choice and be none.
"""
from __future__ import annotations

import json
import re


def _vocab():
    """The facet vocabularies, read from the service that owns them (see module
    docstring: imported, never retyped). Imported inside the function so this
    module stays importable by a test that never touches the database."""
    from . import image_bank_service as svc
    from .image_provenance import ORIGINS
    return {
        'status': ('keep', 'pending', 'reject'),
        'flag': tuple(svc._QUALITY_FLAGS) + tuple(svc._SCORE_FLAGS),
        'medium': tuple(svc.MEDIUM_KEYS),
        'framing': tuple(svc._FRAMING_KEYS),
        'angle': tuple(svc.ANGLES) + ('unknown',),
        'origin': tuple(ORIGINS),
        'res_bucket': tuple(b[0] for b in svc._RES_BUCKETS),
    }


def _sorts():
    from . import image_bank_service as svc
    return tuple(svc.GRID_SORTS)


# Free-text facets: not a closed vocabulary, but still constrained (see _clean_text).
_TEXT_FACETS = ('search', 'exclude')

# A request whose only expressible part would be one of these is not a translation,
# it is a no-op dressed as one. Used to decide `refused`.
_TRIVIAL = frozenset({'status'})

# Negation markers. The check is deliberately crude and English-only because the
# app's prompt is English and the cost of a false positive (one phrase moved to
# `unsupported`) is far below the cost of a false negative (a query that returns
# the exact thing the user asked to avoid).
_NEGATION = re.compile(r'\b(without|no|not|non|never|except|excluding|avoid)\b', re.I)


def _clean_text(value):
    """A CLIP phrase the encoder can actually rank on, or None with a reason.

    Rejects negation outright — see the module docstring for the measurement. Also
    caps the length: the encoder truncates at 77 tokens and a paragraph silently
    becomes its first sentence, which is a filter that does not say what it does."""
    s = ' '.join(str(value or '').split())
    if not s:
        return None, 'empty'
    if _NEGATION.search(s):
        return None, (f'"{s}" is phrased as an exclusion — CLIP ranks a negated '
                      'phrase HIGHER on the thing it names, so this would return '
                      'more of what you asked to avoid, not less')
    if len(s) > 200:
        return None, f'"{s[:60]}…" is too long to rank on (the encoder reads ~77 tokens)'
    return s, None


def axis_stats(bank_id) -> dict:
    """What this bank actually holds, per facet value. Handed to the model so it
    reasons over THIS bank instead of inventing absolutes — and so a value that
    exists nowhere here is visibly worth zero before it is ever proposed."""
    from . import image_bank_service as svc
    return {
        'medium': svc._medium_counts(bank_id),
        'framing': svc._framing_counts(bank_id),
        'angle': svc._angle_counts(bank_id),
        'origin': svc._origin_counts(bank_id),
        'res_bucket': svc._res_bucket_counts(bank_id),
    }


def coverage_note(bank_id) -> list:
    """The axes that are too sparse to translate against, named with their numbers.

    This is what lets the app answer "I cannot express that" with a reason instead
    of a shrug. An axis measured on 0.03% of a bank is not a filter, and a model
    told only its NAME would happily use it."""
    from . import image_bank_service as svc
    from ..models import BankImage
    total = BankImage.query.filter_by(bank_id=bank_id).count()
    out = []
    if not total:
        return out
    captioned = (BankImage.query.filter_by(bank_id=bank_id)
                 .filter(BankImage.caption.isnot(None))
                 .filter(BankImage.caption != '').count())
    for label, n in (('captions', captioned),
                     ('framing', sum(v for k, v in svc._framing_counts(bank_id).items()
                                     if k != 'unknown'))):
        if n * 20 < total:            # under 5% — cannot carry a selection
            out.append(f'{label}: only {n} of {total} images have it')
    return out


def build_prompt(sentence: str, stats: dict, coverage: list) -> str:
    """The whole instruction. Values are listed WITH their counts so an empty
    bucket is visibly empty, and the refusal path is spelled out as a first-class
    answer rather than a fallback."""
    vocab = _vocab()
    lines = ['You translate a request about a photo collection into a FILTER.',
             'You never choose images. You only set filter fields.', '',
             'Allowed fields and their ONLY allowed values:']
    for key in ('status', 'flag'):
        lines.append(f'  {key}: {", ".join(vocab[key])}')
    for key in ('medium', 'framing', 'angle', 'origin', 'res_bucket'):
        counts = stats.get(key) or {}
        shown = ', '.join(f'{v} ({counts.get(v, 0)} images)' for v in vocab[key])
        lines.append(f'  {key}: {shown}')
    lines += [
        '  search: a SHORT positive phrase describing what to rank first.',
        '  sort: one of ' + ', '.join(_sorts()[:12]) + ', …', '',
        'Rules:',
        '- Use ONLY the values listed. Never invent a field or a value.',
        '- NEVER phrase `search` as an exclusion ("without X", "no X"). The ranker '
        'returns MORE of a negated thing, not less.',
        '- A value with 0 images is worth nothing — do not pick it.',
        '- If part of the request cannot be expressed with these fields, put that '
        'part in "unsupported" and leave the filter alone. That is a correct '
        'answer, not a failure.',
    ]
    if coverage:
        lines.append('- These axes are too sparse to use here: ' + '; '.join(coverage))
    lines += ['', 'Answer with JSON only:',
              '{"filter": {...}, "sort": "..." or null, '
              '"understood": ["short phrase", ...], "unsupported": ["...", ...]}',
              '', f'Request: {sentence}']
    return '\n'.join(lines)


def _json_block(raw: str):
    """The first JSON object in the reply. Models wrap JSON in prose or a fence
    often enough that demanding a bare object would fail on a correct answer."""
    s = str(raw or '')
    i, j = s.find('{'), s.rfind('}')
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(s[i:j + 1])
    except (ValueError, TypeError):
        return None


def parse_reply(raw, stats: dict) -> dict:
    """Validate the model's answer against the real vocabulary.

    Anything unknown is DROPPED and named in `dropped` — never kept, never silently
    discarded. A summary that lists a field the grid ignores is the failure mode
    this function exists to prevent: the user would read a confident sentence over
    an unchanged grid."""
    vocab = _vocab()
    data = _json_block(raw)
    if not isinstance(data, dict):
        return {'filter': {}, 'sort': None, 'understood': [], 'unsupported': [],
                'dropped': ['the model did not answer with JSON'], 'refused': True}

    raw_filter = data.get('filter')
    raw_filter = raw_filter if isinstance(raw_filter, dict) else {}
    out, dropped = {}, []

    for key, value in raw_filter.items():
        if key in vocab:
            allowed = vocab[key]
            if value in allowed:
                counts = stats.get(key)
                # Kept even at zero: the grid's own counter is the honest place to
                # learn a bucket is empty, and dropping it here would hide that the
                # model DID understand the request.
                if counts is not None and not counts.get(value):
                    dropped.append(f'{key}={value} exists in this bank but holds 0 images')
                out[key] = value
            else:
                dropped.append(f'{key}={value!r} is not a value this bank uses')
        elif key in _TEXT_FACETS:
            cleaned, why = _clean_text(value)
            if cleaned:
                out[key] = cleaned
            elif why != 'empty':
                dropped.append(why)
        else:
            dropped.append(f'{key!r} is not a filter this bank has')

    sort = data.get('sort')
    sort = sort if sort in _sorts() else None
    if data.get('sort') and sort is None:
        dropped.append(f'sort={data.get("sort")!r} is not one of the grid sorts')

    understood = [str(x) for x in (data.get('understood') or []) if str(x).strip()][:6]
    unsupported = [str(x) for x in (data.get('unsupported') or []) if str(x).strip()][:6]

    # Refused when nothing meaningful came back. `status` alone does not count:
    # "only the kept ones" is a click, not a translation, and presenting it as one
    # would make the feature look like it worked on a request it did not read.
    meaningful = set(out) - _TRIVIAL
    return {'filter': out, 'sort': sort, 'understood': understood,
            'unsupported': unsupported, 'dropped': dropped,
            'refused': not meaningful}


def translate(bank_id, sentence: str, *, generate=None) -> dict:
    """Sentence -> validated filter. Never applies anything: the caller hands the
    result to the UI, which moves the visible controls and lets the grid's own
    counters say how many images that lands on.

    `strict=True` on the generator so a refusal keeps its wording. This surface has
    no degraded mode — the batch captioner can fall back on a long caption it
    already has, but here an empty string would arrive as "the model returned
    nothing", sending the user to check a Settings value that was never the cause
    (the local fence holding a model loaded outside the app is the usual one)."""
    text = ' '.join(str(sentence or '').split())
    if not text:
        raise ValueError('describe what you want in a sentence')
    if len(text) > 400:
        raise ValueError('keep the request to a sentence or two')

    if generate is None:
        from .vision_llm import generate_text as generate

    stats = axis_stats(bank_id)
    coverage = coverage_note(bank_id)
    raw = generate(build_prompt(text, stats, coverage),
                   num_predict=400, strict=True)
    out = parse_reply(raw, stats)
    # Echoed back so the UI can show WHAT was asked next to what was understood.
    # Without it a summary read minutes later is a claim with no subject.
    out['request'] = text
    out['coverage'] = coverage
    return out
