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

WHICH SURFACE, AND WHY ONLY ONE (the Bank/Dataset parity rule, answered rather
than skipped). This verb lands on the 🖼 Gallery and the ◉ Canvas, which are the
two views of `lora_test_image` — the table whose rows carry a `record_id`/`step`,
so a produced view appears next to the picture it was made from with no second
delivery path to invent. The Bank keeps `BankImage` rows and a dataset keeps
`FaceDatasetImage` ones: three tables, three INDEPENDENT id spaces, three
completion callbacks. Sending a Bank id to this route would not 404 — it would
re-shoot a real but unrelated picture, which is the exact bug ✨ improve's own
route note describes. So the other two surfaces are a deliberate NOT-YET, not an
oversight: each needs its own route, its own ownership check and its own
ingestion, and the pose vocabulary in this module is already the shared part.

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

# How many views one press may queue. Not a technical ceiling — 96 poses at
# ~12 s each is twenty minutes of GPU, and a Generate button that can silently
# spend that is the kind of control this repo has refused before. The picker
# counts up to it and says what it will cost.
MAX_VIEWS_PER_RUN = 12

_BY_ID = {
    'azimuth': {a['id']: a for a in AZIMUTHS},
    'elevation': {e['id']: e for e in ELEVATIONS},
    'distance': {d['id']: d for d in DISTANCES},
}

# Refusals, worded as the surface should say them (the frontend mirrors these in
# utils/cameraAngles.js so a picture explains itself BEFORE the click).
NO_VIEWS_PICKED = 'pick at least one camera position'
TOO_MANY_VIEWS = f'that is more than {MAX_VIEWS_PER_RUN} views in one go'
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
