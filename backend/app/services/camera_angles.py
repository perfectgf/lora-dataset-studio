"""📷 Camera angles — move the CAMERA around a picture's subject.

WHY THIS IS NOT A VARIATION SHOT, even though both produce "another angle".
The dataset's shot catalog (services/face_variations.VARIATION_CATALOG) has an
`angle` group — "three-quarter left view", "profile right". Those are SUBJECT
poses: an edit model satisfies them by turning the person, and the room behind
stays exactly where it was. Measured on this repo's own Klein lane (2026-08-25,
one reference, seed held constant, every phrasing tried — English, an explicit
"only the photographer moves", the Chinese cinematography terms): the backdrop
never reprojected. The windows stayed left, the armchair stayed right. Klein
rotates the SUBJECT. Asked hard enough to move the photographer, it drew a
photographer into the room.

That is a real feature — it is just a different one. This module is the other
one: the same instant seen from somewhere else, backdrop included. It runs on
Qwen-Image-Edit-2511 with fal.ai's Multiple-Angles LoRA, which was trained on
3000+ gaussian-splatting renders — pairs where the subject CANNOT move and the
background MUST change, because the geometry made them. That inversion of the
bias is the whole reason for a second engine; a LoRA on Klein would not have
bought it (the only one that exists for Klein 9B was tried and changed nothing).

WHAT THIS MODULE IS. The pose vocabulary and the prompt grammar, and nothing
else — no I/O, no ComfyUI, no database. The grammar is the LoRA's own published
one and is NOT ours to improvise:

    <sks> [azimuth] [elevation] [distance]

96 poses = 8 azimuths x 4 elevations x 3 distances. The descriptors are copied
character-for-character from the model card; a synonym that reads better in
English ("side view" for "right side view") is a token the LoRA never saw.
`test_camera_angles.py` pins every one of them for that reason.

WHICH SURFACES (the Bank/Dataset parity rule, answered rather than skipped).
The verb lives on the 🖼 Gallery / ◉ Canvas (`lora_test_image`) and on the
DATASET (`face_dataset_image`) — two routes, because the tables have
independent id spaces and a Bank id sent to either would re-shoot a real but
unrelated picture. The dataset lane adds the one thing the gallery cannot:
the pose feeds the CAPTION (pose_caption_phrase below), because a back view
left undescribed binds "back-facing" to the trigger word.

THE BANK DOES NOT CARRY THIS VERB, deliberately. The Bank's promise is that it
holds REAL material — scraped and imported photographs — and that promise is
what makes it trustworthy as a source. A camera view is the model's hypothesis
about a scene, and landing hypotheses in the reservoir of real would quietly
break the one distinction the whole curation flow leans on. The path that
respects it already exists and costs one extra gesture: promote the Bank image
into a dataset, then 📷 from there — the view is then born as a dataset
CANDIDATE, reviewed like every other generated tile, never filed as real.

⚠️ The ids below are written into user databases (a produced picture stores the
pose it was asked for) and into localStorage on the frontend. Renaming one
strands every row already there — they only ever change WITH an alias path,
same rule as the shot catalog's labels.
"""

# The LoRA's trigger. Present in every prompt it was trained on; without it the
# adapter is loaded and inert — the exact failure mode this repo already met on
# the video side, where an orbit preset kept its trigger word while the LoRA
# checkbox was off and the model animated the subject instead of the camera.
TRIGGER = '<sks>'

# --- Azimuth: where the camera stands, around the subject ---------------------
# 0 deg is the reference photograph's own viewpoint, not a compass bearing: the
# LoRA re-renders relative to the picture it is given. That is what makes
# `front view` a CONTROL — asking for it returns the picture you started from
# (measured: a background delta of 3.9 against a 64-80 range for a real move),
# and it is why the picker offers it rather than hiding it.
# `label` carries no degrees on purpose: it is composed into a tile caption
# ("Back-left · High · Wide"), and three angles in one line is noise where the
# dial already shows the geometry.
AZIMUTHS = (
    {'id': 'front',       'degrees': 0,   'token': 'front view',                'label': 'Front'},
    {'id': 'front_right', 'degrees': 45,  'token': 'front-right quarter view',  'label': 'Front-right'},
    {'id': 'right',       'degrees': 90,  'token': 'right side view',           'label': 'Right side'},
    {'id': 'back_right',  'degrees': 135, 'token': 'back-right quarter view',   'label': 'Back-right'},
    {'id': 'back',        'degrees': 180, 'token': 'back view',                 'label': 'Back'},
    {'id': 'back_left',   'degrees': 225, 'token': 'back-left quarter view',    'label': 'Back-left'},
    {'id': 'left',        'degrees': 270, 'token': 'left side view',            'label': 'Left side'},
    {'id': 'front_left',  'degrees': 315, 'token': 'front-left quarter view',   'label': 'Front-left'},
)

# --- Elevation: how high the camera is ----------------------------------------
# The one axis Klein could also do, and the one the model card singles out as
# this LoRA's strength (proper -30 support). Kept in the same order as the
# picker draws it, low to high.
ELEVATIONS = (
    {'id': 'low',      'degrees': -30, 'token': 'low-angle shot',  'label': 'Low', 'hint': 'camera below, looking up'},
    {'id': 'eye',      'degrees': 0,   'token': 'eye-level shot',  'label': 'Eye level', 'hint': 'camera at subject height'},
    {'id': 'elevated', 'degrees': 30,  'token': 'elevated shot',   'label': 'Elevated', 'hint': 'camera slightly above'},
    {'id': 'high',     'degrees': 60,  'token': 'high-angle shot', 'label': 'High', 'hint': 'camera high, looking down'},
)

# --- Distance: how far the camera is ------------------------------------------
# ⚠️ HONEST NOTE, measured before shipping: this axis is the loose one. Several
# poses asked at `medium shot` came back tighter than the reference. It is not a
# focal length, it is a hint the model mostly honours — which is why the picker
# says so instead of letting a user discover it on a dataset.
DISTANCES = (
    {'id': 'close',  'factor': 0.6, 'token': 'close-up',    'label': 'Close-up'},
    {'id': 'medium', 'factor': 1.0, 'token': 'medium shot', 'label': 'Medium'},
    {'id': 'wide',   'factor': 1.8, 'token': 'wide shot',   'label': 'Wide'},
)

POSE_COUNT = len(AZIMUTHS) * len(ELEVATIONS) * len(DISTANCES)   # 96

# The pose a bare picture is assumed to already be at. Two things read it: the
# picker opens with this ring lit as "you are here", and `is_reference_pose`
# below keeps the UI honest about a request that would re-render what you have.
REFERENCE_POSE = ('front', 'eye', 'medium')

# The only ceiling is the vocabulary itself: 96 distinct poses exist, and a
# request repeating one asked for one picture.
#
# There WAS an arbitrary 12 here, on the reasoning that a button which can spend
# twenty minutes of GPU should not do it quietly. The reasoning was half right
# and the remedy was wrong: eight sides at two distances is 16 — an ordinary,
# obviously reasonable request — and the cap refused it. A limit that blocks the
# normal case to prevent a rare one is not a safeguard, it is a bug with a
# justification. What the button actually owed was to say what it will cost, not
# to decide it: the picker states the view count and the minutes BEFORE the
# click, loudly past LONG_RUN_SECONDS, and every queued view is cancellable one
# by one from the system queue.
MAX_VIEWS_PER_RUN = POSE_COUNT

_BY_ID = {
    'azimuth': {a['id']: a for a in AZIMUTHS},
    'elevation': {e['id']: e for e in ELEVATIONS},
    'distance': {d['id']: d for d in DISTANCES},
}

# Refusals, worded as the surface should say them (the frontend mirrors these in
# utils/cameraAngles.js so a picture explains itself BEFORE the click).
NO_VIEWS_PICKED = 'pick at least one camera position'
# Only reachable from a malformed request: after de-duplication a selection
# cannot exceed the number of poses that exist.
TOO_MANY_VIEWS = f'there are only {POSE_COUNT} camera positions'
UNKNOWN_POSE = 'that camera position does not exist'
SOURCE_GONE = 'that image is no longer in the library'
SOURCE_FILE_GONE = 'that image file is no longer on disk'
SOURCE_NOT_DONE = 'this image is still rendering'
ALREADY_DERIVED = 'a camera view cannot itself be re-shot from another angle'


def pose_id(azimuth, elevation, distance):
    """The stable id of one pose. Stored on the produced row."""
    return f'{azimuth}/{elevation}/{distance}'


def parse_pose(value):
    """`'right/low/medium'` -> `('right', 'low', 'medium')`, or None.

    Returns None rather than raising for ANY malformed input: this parses
    values that arrive from a request body and from old database rows, and the
    caller's job is to refuse with UNKNOWN_POSE, not to crash on a typo.
    """
    if not isinstance(value, str):
        return None
    parts = value.split('/')
    if len(parts) != 3:
        return None
    az, el, di = (p.strip() for p in parts)
    if (az not in _BY_ID['azimuth'] or el not in _BY_ID['elevation']
            or di not in _BY_ID['distance']):
        return None
    return (az, el, di)


def is_reference_pose(pose):
    """True when this pose IS the picture's own viewpoint.

    Not refused — asking for it is the control that proves the lane works, and
    a user comparing engines wants it. The surface merely says what it is.
    """
    return parse_pose(pose) == REFERENCE_POSE if isinstance(pose, str) else tuple(pose) == REFERENCE_POSE


def pose_prompt(azimuth, elevation, distance):
    """The prompt for one pose, in the LoRA's published grammar.

    Raises ValueError on an unknown component — a silently-dropped token would
    produce a picture at some OTHER angle and store it under the name of the one
    that was asked for, which is worse than an error.
    """
    try:
        a = _BY_ID['azimuth'][azimuth]
        e = _BY_ID['elevation'][elevation]
        d = _BY_ID['distance'][distance]
    except KeyError:
        raise ValueError(UNKNOWN_POSE)
    return f"{TRIGGER} {a['token']} {e['token']} {d['token']}"


def pose_caption_phrase(pose):
    """The training-caption fragment for a pose — or None when it adds nothing.

    THE POINT: what a caption does not describe binds to the trigger word. A
    back view left uncaptioned teaches the LoRA that the character IS
    back-facing; describing the angle is what keeps it promptable. And the
    angle is the one thing the captioner cannot be trusted to see — measured on
    this repo's own probes, VLMs mis-describe viewpoints even when they are
    told — while WE know it exactly, because it was requested.

    Azimuth and elevation only, never the distance: framing is visible in the
    picture and the captioner already describes it; the camera position behind
    the picture is what it cannot know.

    None for the front/eye components (the reference pose is how ordinary
    photos already look — "seen from the front" on every image is noise, and a
    caption of pure noise would still block nothing).
    """
    parsed = parse_pose(pose) if isinstance(pose, str) else (
        tuple(pose) if pose else None)
    if parsed is None:
        return None
    az, el, _di = parsed
    parts = []
    if az != 'front':
        parts.append({
            'front_right': 'seen from the front-right',
            'right': 'seen from the right side',
            'back_right': 'seen from the back-right',
            'back': 'seen from behind',
            'back_left': 'seen from the back-left',
            'left': 'seen from the left side',
            'front_left': 'seen from the front-left',
        }[az])
    if el != 'eye':
        parts.append({
            'low': 'low camera angle',
            'elevated': 'slightly elevated camera angle',
            'high': 'high camera angle',
        }[el])
    return ', '.join(parts) or None


def pose_label(azimuth, elevation, distance):
    """What the tile says under the picture. Human, not the model's tokens."""
    a = _BY_ID['azimuth'].get(azimuth)
    e = _BY_ID['elevation'].get(elevation)
    d = _BY_ID['distance'].get(distance)
    if not (a and e and d):
        raise ValueError(UNKNOWN_POSE)
    return f"{a['label']} · {e['label']} · {d['label']}"


def catalog():
    """Everything the picker needs, in ONE payload.

    The frontend holds its own copy of the vocabulary (it has to — the dial is
    drawn from the degrees), and `test_camera_catalog_contract.py` reads both
    sides so the two cannot drift.
    """
    return {
        'trigger': TRIGGER,
        'azimuths': [dict(a) for a in AZIMUTHS],
        'elevations': [dict(e) for e in ELEVATIONS],
        'distances': [dict(d) for d in DISTANCES],
        'pose_count': POSE_COUNT,
        'max_views': MAX_VIEWS_PER_RUN,
        'reference_pose': pose_id(*REFERENCE_POSE),
    }


def normalize_requested(poses):
    """Validate a requested list of pose ids -> list of tuples, de-duplicated.

    Order is the caller's, minus repeats: a picker that sends the same pose
    twice asked for one picture, not two, and charging the GPU for the
    duplicate would be a bug the user cannot see.
    """
    if not isinstance(poses, (list, tuple)):
        raise ValueError(NO_VIEWS_PICKED)
    out, seen = [], set()
    for raw in poses:
        parsed = parse_pose(raw)
        if parsed is None:
            raise ValueError(UNKNOWN_POSE)
        if parsed in seen:
            continue
        seen.add(parsed)
        out.append(parsed)
    if not out:
        raise ValueError(NO_VIEWS_PICKED)
    if len(out) > MAX_VIEWS_PER_RUN:
        raise ValueError(TOO_MANY_VIEWS)
    return out
