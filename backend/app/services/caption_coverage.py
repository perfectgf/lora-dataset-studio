"""Reading VARIETY out of captions that were already written.

The composition meter answers "how many face / bust / body / back shots?".
It cannot answer the question that actually decides whether a LoRA generalises:
*what did the set never show?* A character shot twenty times from the front, in
one outfit, under one light, trains a LoRA that can only ever produce that — and
nothing in the app said so, because every number on screen was green.

This module is the cheap half of that answer. It does NOT run a model: it scans
the captions the caption pass ALREADY wrote, with a hand-written lexicon, and
reports which buckets of each axis were mentioned and which never were. Pure
functions over strings — deterministic, instant, unit-testable, no GPU, no
network, and it degrades to an honest "captions missing" instead of empty bars.

What that buys, and what it costs, stated once here so the UI can repeat it:

  * A caption mentions what the captioner found WORTH SAYING. "profile" absent
    from every caption is strong evidence there are no profile shots; it is not
    proof. We phrase findings as observations about the captions, never as
    verdicts about the pixels.
  * Negation is not parsed. "not smiling" counts as a smile mention. Captioners
    write positively, so this is rare — but it is a real false positive and the
    panel says the readout is caption-based.
  * A bucket the lexicon has no word for is invisible. The lexicon is
    deliberately broad and dull rather than clever.

Bucket ids are part of the API payload the frontend keys on — treat them as
stored labels: never rename one without an alias.
"""
from __future__ import annotations

import re

# Two ways an axis can be short, because they are genuinely different defects.
#
#   'buckets'  — the named values matter individually. Nobody needs an overhead
#                shot, but a character with no profile view has a hole a viewer
#                can point at. Advice names the missing CORE buckets.
#   'variety'  — no single value matters; the COUNT does. One outfit across the
#                whole set bakes that outfit into the trigger word, whichever
#                outfit it happens to be. Advice reports how many distinct
#                values showed up, never which one is "missing".
MODE_BUCKETS = 'buckets'
MODE_VARIETY = 'variety'


def _kw(*words):
    """Compile a keyword list into one word-boundary regex.

    Spaces and hyphens in a phrase match any run of space/hyphen/underscore, so
    'three quarter' also catches 'three-quarter' and 'three_quarter' — captioners
    are inconsistent about exactly that and it is not worth three entries each.
    """
    # Split on separators FIRST, then escape each piece: re.escape() turns '-'
    # into '\-', and substituting the separator class over that leaves a stray
    # backslash which silently makes 't-shirt' match the literal '['.
    parts = [r'[\s\-_]+'.join(re.escape(p) for p in re.split(r'[\s\-_]+', w.lower().strip()) if p)
             for w in words]
    return re.compile(r'\b(?:' + '|'.join(p for p in parts if p) + r')\b', re.IGNORECASE)


class _Bucket:
    __slots__ = ('id', 'label', 'core', 'rx')

    def __init__(self, bid, label, words, core=False):
        self.id = bid
        self.label = label
        self.core = core
        self.rx = _kw(*words)


class _Axis:
    __slots__ = ('id', 'label', 'mode', 'buckets', 'want', 'hint')

    def __init__(self, aid, label, mode, buckets, want=0, hint=''):
        self.id = aid
        self.label = label
        self.mode = mode
        self.buckets = buckets
        # For MODE_VARIETY: how many distinct values a healthy set shows. Below
        # this the axis is called thin. Ignored for MODE_BUCKETS.
        self.want = want
        self.hint = hint


# --- the lexicon ------------------------------------------------------------
# Deliberately plain vocabulary: these are the words captioners actually emit,
# not the words a photographer would choose. Adding a synonym is cheap and safe;
# adding a bucket changes the payload shape, so it needs a test.

AXES = {
    'view': _Axis(
        'view', 'Camera view', MODE_BUCKETS, hint='Which side of the subject the set ever saw',
        buckets=[
            _Bucket('frontal', 'frontal', core=True, words=[
                'front view', 'frontal', 'facing the camera', 'facing camera',
                'facing forward', 'looking at the camera', 'looking at the viewer',
                'looking directly at', 'faces the camera']),
            _Bucket('three_quarter', 'three-quarter', core=True, words=[
                'three quarter', 'three quarters', '3/4 view', '3/4',
                'quarter view', 'angled toward', 'angled towards',
                'turned slightly', 'slightly turned', 'head turned']),
            _Bucket('profile', 'profile', core=True, words=[
                'profile', 'side view', 'from the side', 'side on',
                'looking to the side', 'looking off to', 'sideways']),
            _Bucket('from_behind', 'from behind', words=[
                'from behind', 'back view', 'rear view', 'back to the camera',
                'seen from behind', 'viewed from behind', 'facing away',
                'her back', 'his back', 'their back', 'over the shoulder',
                'over her shoulder', 'over his shoulder']),
        ]),
    'angle': _Axis(
        'angle', 'Camera height', MODE_BUCKETS, hint='Eye level only is the default trap',
        buckets=[
            _Bucket('eye_level', 'eye level', words=[
                'eye level', 'eye height', 'straight on', 'head on']),
            _Bucket('low', 'low angle', words=[
                'low angle', 'from below', 'looking up at', 'worms eye',
                'shot from below', 'upward angle']),
            _Bucket('high', 'high angle', words=[
                'high angle', 'from above', 'looking down at', 'downward angle',
                'shot from above']),
            _Bucket('overhead', 'overhead', words=[
                'overhead', 'top down', 'birds eye', 'directly above']),
        ]),
    'lighting': _Axis(
        'lighting', 'Lighting', MODE_BUCKETS, hint='One light baked in is the classic LoRA smell',
        buckets=[
            _Bucket('daylight', 'daylight', core=True, words=[
                'daylight', 'natural light', 'sunlight', 'sunny', 'bright day',
                'sunlit', 'outdoor light', 'in the sun', 'midday']),
            _Bucket('golden', 'golden hour', words=[
                'golden hour', 'sunset', 'sunrise', 'dusk', 'warm light',
                'warm glow', 'orange light']),
            _Bucket('indoor', 'indoor light', core=True, words=[
                'indoor lighting', 'artificial light', 'lamp', 'lamplight',
                'ceiling light', 'warm indoor', 'household light', 'candlelight']),
            _Bucket('studio', 'studio light', words=[
                'studio lighting', 'studio light', 'softbox', 'ring light',
                'flash', 'strobe', 'evenly lit', 'professional lighting']),
            _Bucket('night', 'night / low light', words=[
                'night', 'nighttime', 'at night', 'dark', 'dim', 'low light',
                'moonlight', 'neon', 'shadowy']),
            _Bucket('backlit', 'backlit / rim', words=[
                'backlit', 'back lit', 'rim light', 'rim lighting', 'silhouette',
                'against the light', 'halo of light']),
            _Bucket('overcast', 'soft / overcast', words=[
                'overcast', 'cloudy', 'diffused light', 'soft light', 'shade',
                'in the shade', 'grey sky', 'gray sky']),
        ]),
    'setting': _Axis(
        'setting', 'Setting', MODE_BUCKETS, hint='Where the subject was ever placed',
        buckets=[
            _Bucket('indoor', 'indoor', core=True, words=[
                'indoors', 'indoor', 'inside', 'room', 'bedroom', 'living room',
                'kitchen', 'bathroom', 'office', 'interior', 'hallway', 'couch',
                'sofa', 'bed']),
            _Bucket('outdoor', 'outdoor', core=True, words=[
                'outdoors', 'outdoor', 'outside', 'garden', 'park', 'forest',
                'field', 'mountain', 'nature', 'trees', 'grass', 'sky']),
            _Bucket('urban', 'urban / street', words=[
                'street', 'city', 'urban', 'sidewalk', 'alley', 'building',
                'downtown', 'rooftop', 'cafe', 'shop']),
            _Bucket('studio_bg', 'studio / plain backdrop', words=[
                'plain background', 'white background', 'grey background',
                'gray background', 'black background', 'studio backdrop',
                'seamless background', 'solid background', 'neutral background']),
            _Bucket('water', 'beach / water', words=[
                'beach', 'ocean', 'sea', 'pool', 'poolside', 'lake', 'river',
                'shore', 'sand', 'waves']),
            _Bucket('vehicle', 'vehicle', words=[
                'car', 'inside a car', 'vehicle', 'train', 'bus', 'motorcycle',
                'driver seat', 'passenger seat']),
        ]),
    'outfit': _Axis(
        'outfit', 'Outfit', MODE_VARIETY, want=3,
        hint='One outfit across the set gets learned as part of the subject',
        buckets=[
            _Bucket('casual', 'casual', words=[
                'casual', 't-shirt', 'tshirt', 'tee', 'jeans', 'hoodie',
                'sweater', 'sweatshirt', 'shorts', 'tank top', 'blouse',
                'cardigan', 'leggings']),
            _Bucket('formal', 'formal', words=[
                'formal', 'suit', 'blazer', 'tuxedo', 'gown', 'evening dress',
                'cocktail dress', 'shirt and tie', 'necktie', 'dress shirt']),
            _Bucket('dress', 'dress / skirt', words=[
                'dress', 'skirt', 'sundress', 'minidress', 'maxi dress']),
            _Bucket('swim', 'swimwear / lingerie', words=[
                'bikini', 'swimsuit', 'swimwear', 'lingerie', 'bra', 'underwear',
                'panties', 'bodysuit', 'corset']),
            _Bucket('sport', 'sportswear', words=[
                'sportswear', 'athletic', 'gym', 'workout', 'yoga pants',
                'sports bra', 'tracksuit', 'jersey', 'running shorts']),
            _Bucket('outerwear', 'outerwear', words=[
                'coat', 'jacket', 'trench coat', 'parka', 'raincoat', 'scarf',
                'leather jacket', 'puffer']),
            _Bucket('uniform', 'uniform / costume', words=[
                'uniform', 'costume', 'armor', 'armour', 'robe', 'kimono',
                'cosplay', 'apron', 'scrubs', 'military']),
            _Bucket('nude', 'nude', words=[
                'nude', 'naked', 'topless', 'undressed', 'bare chest',
                'no clothes', 'unclothed']),
        ]),
    'expression': _Axis(
        'expression', 'Expression', MODE_VARIETY, want=2,
        hint='A single expression trains a mask, not a face',
        buckets=[
            _Bucket('neutral', 'neutral', words=[
                'neutral expression', 'blank expression', 'expressionless',
                'calm expression', 'relaxed expression', 'impassive']),
            _Bucket('smile', 'smiling', words=[
                'smiling', 'smile', 'grinning', 'grin', 'cheerful', 'happy']),
            _Bucket('laugh', 'laughing', words=[
                'laughing', 'laugh', 'giggling', 'beaming']),
            _Bucket('serious', 'serious', words=[
                'serious', 'stern', 'frowning', 'frown', 'intense expression',
                'determined', 'confident expression', 'smirk', 'smirking']),
            _Bucket('surprise', 'surprised', words=[
                'surprised', 'shocked', 'wide eyed', 'mouth open', 'gasping',
                'astonished']),
            _Bucket('pensive', 'pensive / sad', words=[
                'pensive', 'thoughtful', 'sad', 'melancholy', 'wistful',
                'sombre', 'somber', 'crying', 'tearful']),
        ]),
}

# Which axes each dataset kind is actually judged on.
#
# `character`  — everything: the invariant is one identity, so view, light,
#                setting, outfit and expression must all vary around it.
# `concept`    — the concept is the invariant and the PERSON is the nuisance
#                variable; outfit/setting/light variety is exactly what stops
#                the concept binding to one wearer. Expression is dropped: a
#                concept LoRA has no face to over-fit, and reporting it would be
#                noise dressed as advice.
# `style`      — the style is the invariant. Judging it on outfits or
#                expressions would be inventing a defect; what matters is that
#                the style was shown over different content and light.
KIND_AXES = {
    'character': ['view', 'angle', 'lighting', 'setting', 'outfit', 'expression'],
    'concept': ['view', 'angle', 'lighting', 'setting', 'outfit'],
    'style': ['lighting', 'setting', 'view'],
}


def axes_for_kind(kind: str | None):
    """The axis ids judged for this dataset kind (unknown/NULL kind → character)."""
    return KIND_AXES.get((kind or 'character').lower(), KIND_AXES['character'])


def scan_caption(text: str) -> dict:
    """{axis_id: set(bucket_ids)} mentioned by ONE caption. Every axis, always —
    the caller narrows to the kind; keeping this total makes it testable alone."""
    hits = {aid: set() for aid in AXES}
    if not text:
        return hits
    for aid, axis in AXES.items():
        for b in axis.buckets:
            if b.rx.search(text):
                hits[aid].add(b.id)
    return hits


def analyse(captions, kind=None, ids=None) -> dict:
    """Per-axis coverage over an iterable of caption strings.

    `captions` is every caption in the pool INCLUDING the empty ones — the count
    of missing captions is half the honesty of this panel, so it must not be
    filtered out by the caller and silently forgotten.

    `ids` is optional and positional to `captions`: pass it and every bucket also
    reports WHICH images landed in it (`image_ids`), so the panel can hand the
    grid a filter instead of only a number. It is gathered in the SAME pass as
    the counts on purpose — a second walk over the captions would be a second
    place for the lexicon to be applied, and the day the two disagreed the panel
    would show a count its own list could not reproduce.
    """
    caps = list(captions or [])
    total = len(caps)
    if ids is not None:
        ids = list(ids)
        if len(ids) != total:
            raise ValueError('ids must be positional to captions '
                             f'({len(ids)} ids for {total} captions)')
    # Pair BEFORE dropping the empties: filtering first and zipping after would
    # line each id up with the wrong caption the moment one image has none.
    pairs = [(ids[i] if ids is not None else None, c)
             for i, c in enumerate(caps) if (c or '').strip()]
    written = [c for _, c in pairs]
    counts = {aid: {b.id: 0 for b in AXES[aid].buckets} for aid in AXES}
    members = ({aid: {b.id: [] for b in AXES[aid].buckets} for aid in AXES}
               if ids is not None else None)
    for img_id, c in pairs:
        for aid, bids in scan_caption(c).items():
            for bid in bids:
                counts[aid][bid] += 1
                if members is not None:
                    members[aid][bid].append(img_id)

    out_axes = []
    for aid in axes_for_kind(kind):
        axis = AXES[aid]
        rows = []
        for b in axis.buckets:
            n = counts[aid][b.id]
            row = {'id': b.id, 'label': b.label, 'count': n, 'core': b.core,
                   # "thin" only means something once there is a set to be
                   # thin against: one profile shot out of six is fine.
                   'thin': 0 < n <= 1 and len(written) >= 8}
            if members is not None:
                row['image_ids'] = members[aid][b.id]
            rows.append(row)
        present = [r for r in rows if r['count'] > 0]
        out_axes.append({
            'id': axis.id, 'label': axis.label, 'mode': axis.mode,
            'hint': axis.hint, 'want': axis.want,
            'buckets': rows,
            'present': len(present),
            'missing_core': [r['label'] for r in rows if r['core'] and r['count'] == 0],
        })

    return {
        'kind': (kind or 'character').lower(),
        'total': total,
        'captioned': len(written),
        'uncaptioned': total - len(written),
        'axes': out_axes,
    }


def advice(report: dict) -> list:
    """Honest, actionable sentences from `analyse()`. `{'tone','text'}` each,
    warnings first. Pure function of the report — this is the logic worth
    proving, so it never touches the DB or the request."""
    out = []
    total, captioned = report['total'], report['captioned']

    if total == 0:
        return [{'tone': 'info',
                 'text': 'No images yet — the variety read appears once the dataset has some.'}]
    if captioned == 0:
        return [{'tone': 'info',
                 # Wording stays surface-neutral: this module now backs BOTH the
                 # dataset panel (where the counts above are the composition bar)
                 # and the bank one (where they are the framing chips).
                 'text': 'No captions yet, so variety cannot be read. Run the caption '
                         'pass and this panel fills in — until then only the counts '
                         'above are known.'}]
    if report['uncaptioned']:
        out.append({'tone': 'info',
                    'text': f'{report["uncaptioned"]} of {total} images have no caption yet — '
                            f'everything below is read from the other {captioned}.'})

    # Too small a sample and every axis looks "missing" for the wrong reason.
    if captioned < 5:
        out.append({'tone': 'info',
                    'text': f'Only {captioned} captions to read — too few to call anything '
                            f'missing yet.'})
        return out

    for axis in report['axes']:
        if axis['mode'] == MODE_BUCKETS:
            gone = axis['missing_core']
            if gone:
                out.append({'tone': 'warn',
                            'text': f'No {" or ".join(gone)} mentioned in any caption — '
                                    f'generate or import a few ({axis["label"].lower()}).'})
            elif axis['present'] == 0:
                # NOT the same defect as "they are all the same", and saying so
                # was a lie the panel could tell with every chip on zero: no
                # caption named an angle at all, which means unmeasured, not
                # uniform. An axis nobody described cannot be judged.
                out.append({'tone': 'info',
                            'text': f'{axis["label"]}: no caption names one, so this cannot '
                                    f'be judged.'})
            elif axis['present'] == 1:
                only = next(b['label'] for b in axis['buckets'] if b['count'] > 0)
                out.append({'tone': 'warn',
                            'text': f'{axis["label"]}: every caption that says so describes '
                                    f'the same one ({only}) — {axis["hint"].lower()}.'})
        else:
            if axis['present'] == 0:
                out.append({'tone': 'info',
                            'text': f'{axis["label"]}: no caption names one, so this cannot '
                                    f'be judged.'})
            elif axis['present'] < axis['want']:
                named = [b['label'] for b in axis['buckets'] if b['count'] > 0]
                out.append({'tone': 'warn',
                            'text': f'Only {axis["present"]} {axis["label"].lower()} '
                                    f'type{"" if axis["present"] == 1 else "s"} across the set '
                                    f'({", ".join(named)}) — {axis["hint"].lower()}.'})

    thin = [(a['label'], b['label']) for a in report['axes'] for b in a['buckets'] if b['thin']]
    if thin:
        out.append({'tone': 'info',
                    'text': 'Mentioned only once: '
                            + ', '.join(f'{lbl} ({ax.lower()})' for ax, lbl in thin[:4])
                            + ' — one image is not enough for the model to generalise it.'})

    if not any(a['tone'] == 'warn' for a in out):
        out.append({'tone': 'info',
                    'text': 'Nothing obviously missing — the captions describe a varied set.'})
    out.sort(key=lambda a: 0 if a['tone'] == 'warn' else 1)
    return out
