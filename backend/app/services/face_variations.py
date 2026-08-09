"""Pure module: face-dataset variation catalog, composition presets, vision prompts.

No DB, no Flask -> trivially unit-tested. The catalog drives the Klein fan-out;
the presets target a balanced training composition (see the design spec).
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import re
import zlib

# Verrou d'identité renforcé (deep-research 2026-06-14, source primaire Google AI) :
# nommer les traits + interdire l'embellissement améliore la cohérence du visage.
# NB : la qualité de la photo de référence reste le facteur déterminant.
IDENTITY_GUARD = (
    "This is the SAME person as the reference image. Preserve their facial identity "
    "EXACTLY: same eye shape and color, nose, jawline, lips, skin tone and texture, "
    "and face proportions. Do NOT beautify, slim, age, or alter the face. Use the "
    "reference ONLY to lock the facial identity: take the clothing/outfit and the "
    "facial expression from the description below, and do NOT copy the outfit or the "
    "expression shown in the reference image. "
    "SFW, realistic photographic portrait.")

# Variante multi-références (Nano Banana) : avec un guard au singulier le modèle
# peut s'ancrer sur une seule image ; on lui dit EXPLICITEMENT que toutes les refs
# montrent la même personne et qu'il doit s'appuyer sur chacune d'elles.
IDENTITY_GUARD_MULTI = (
    "ALL the reference images show the SAME person (different angles, expressions or "
    "framings). Use EVERY reference image together to lock the identity. Preserve their "
    "facial identity EXACTLY: same eye shape and color, nose, jawline, lips, skin tone "
    "and texture, and face proportions. Do NOT beautify, slim, age, or alter the face. "
    "Use the reference images ONLY to lock the facial identity: take the clothing/outfit "
    "and the facial expression from the description below, and do NOT copy the outfit or "
    "the expression shown in the reference images. "
    "SFW, realistic photographic portrait.")

# Klein restage + face-identity block (see wrap_variation_klein). Held as a named
# constant so it can be the DEFAULT of the editable klein_identity override, byte
# for byte what the wrapper used to inline. The nsfw-dependent tail ({ending}) is
# NOT part of it — that stays a separate SFW/nudity clamp the wrapper appends.
IDENTITY_GUARD_KLEIN = (
    "Restage the shot to match this description — change the pose, camera angle, "
    "framing, clothing and facial expression accordingly; do not copy the "
    "composition, the outfit or the facial expression of the reference image (use "
    "it only for the facial identity). "
    "Keep the facial identity exactly the same: same eye shape and color, nose, "
    "jawline, lips, skin tone and texture, and face proportions. Do not beautify "
    "or alter the face. Sharp focus, natural skin texture with visible pores, "
    "realistic lighting with soft shadows, high detail.")

# Fixed instruction for the manual "Klein upscale & improve" action. Lives here
# (not in face_dataset_service) so all four editable identity/quality prompts share
# ONE default registry; face_dataset_service re-imports it under the same name so
# `svc.KLEIN_IMAGE_IMPROVE_PROMPT` (persisted-in-tests) keeps resolving.
KLEIN_IMAGE_IMPROVE_PROMPT = (
    'add detailed texture, add sharp details, add candid shot, add soft focus effect')


# --- Editable identity / quality prompts (feature request @bbsorry / 雨田壹) ---
# The four identity "locks" above were hardcoded and invisible. They are now
# overridable (config identity_prompts.*) with a Settings UI + "Restore default",
# PER SUBJECT TYPE — see identity_prompt_config_key for the layout.
# get_identity_prompt returns the override ONLY when it is set
# to non-blank text, otherwise the shipped constant — so the default path stays
# byte-identical to the pre-feature behaviour (the reproducibility invariant the
# existing wrapper tests lock). The config read is lazy: this module stays
# import-pure (no Flask), and a caller with no config configured gets the default.
IDENTITY_PROMPT_KINDS = ('face_single', 'face_multi', 'klein_identity', 'klein_improve')

# The identity locks were only ONE of the six sources the local-edit prompt is
# assembled from. The other five were hardcoded and shipped in every prompt with
# no way to see or change them: the markings hold order, the two directives baked
# into every human shot (outfit / expression), the concrete-garment palette, the
# per-framing detail block and the photographic rendering tail. They ride the
# SAME mechanism as the four above — one config key, blank means "shipped
# default", non-blank wins — deliberately: a second override system would be a
# second set of rules to learn and a second place for the default path to rot.
#
# `outfit_palette` is a LIST stored as TEXT, one garment per line. Same field,
# same "Restore default", same blank-means-default contract; `outfit_palette()`
# parses it and falls back to the shipped tuple when the user leaves nothing
# usable behind.
PROMPT_PART_KINDS = ('markings_lock', 'outfit_vary', 'expression_neutral',
                     'outfit_palette', 'render_tail_sfw', 'render_tail_nsfw',
                     'framing_face', 'framing_bust', 'framing_body', 'framing_back')

# The framings that own a detail block. An unknown/absent framing yields no block
# at all (historical behaviour), so this set is also what stops a bogus framing
# from reaching identity_prompt_default and raising KeyError.
PROMPT_FRAMINGS = ('face', 'bust', 'body', 'back')
_IDENTITY_PROMPT_DEFAULTS = {
    'face_single': IDENTITY_GUARD,
    'face_multi': IDENTITY_GUARD_MULTI,
    'klein_identity': IDENTITY_GUARD_KLEIN,
    'klein_improve': KLEIN_IMAGE_IMPROVE_PROMPT,
}


# --- Subject type (Human / Animal / Creature / Object / Other / Anime) --------
# The identity locks above are written for a HUMAN subject ("facial identity …
# eye shape, jawline, lips, skin tone"). A dataset now declares WHAT its subject
# is, so the generation prompts stop assuming a person. `subject_type` is a
# per-dataset dimension ORTHOGONAL to `kind` (character/concept/style): a specific
# dog is character+animal, "dogs in general" is concept+animal. 'human' is the
# default and every human default/prompt below stays byte-identical to the
# pre-feature behaviour (a legacy dataset stores the column as NULL -> 'human').
# APPEND-ONLY. `subject_type` is a STORED column (face_dataset.subject_type,
# VARCHAR(16)) and this tuple is also the order the selector renders in, so a new
# type goes at the END: renaming or reordering would re-point values that are
# already in users' databases. 'anime' (added last) is the drawn-character type —
# see IDENTITY_GUARD_ANIME for why it could not just be 'human'.
SUBJECT_TYPES = ('human', 'animal', 'creature', 'object', 'other', 'anime')


def normalize_subject_type(value) -> str:
    """Whitelist a stored/incoming subject type; anything unknown or blank -> the
    'human' default, so a legacy dataset (column NULL) behaves exactly as before."""
    v = (value or '').strip().lower()
    return v if v in SUBJECT_TYPES else 'human'


# Non-human identity locks — SAME STRUCTURE as the human guards (name the defining
# traits, forbid restyling, reference locks identity ONLY, pose/setting come from
# the description) but the traits that DEFINE the subject differ by type. First
# honest drafts, to be refined — not an exact science. `klein_improve` is a
# subject-agnostic quality instruction, so it is NOT redeclared per subject.
IDENTITY_GUARD_ANIMAL = (
    "This is the SAME animal as the reference image. Preserve its identity EXACTLY: "
    "same species and breed, fur/coat colour, pattern and markings, eye colour, ear "
    "and muzzle shape, and body build. Do NOT restyle, change the breed or alter its "
    "markings. Use the reference ONLY to lock the animal's identity: take the pose, "
    "setting and framing from the description below, and do NOT copy the pose or "
    "background shown in the reference image. Realistic photograph.")
IDENTITY_GUARD_ANIMAL_MULTI = (
    "ALL the reference images show the SAME animal (different angles or poses). Use "
    "EVERY reference image together to lock its identity. Preserve its identity EXACTLY: "
    "same species and breed, fur/coat colour, pattern and markings, eye colour, ear and "
    "muzzle shape, and body build. Do NOT restyle, change the breed or alter its markings. "
    "Take the pose, setting and framing from the description below, and do NOT copy the "
    "pose or background shown in the reference images. Realistic photograph.")
IDENTITY_GUARD_ANIMAL_KLEIN = (
    "Restage the shot to match this description — change the pose, camera angle, framing "
    "and setting accordingly; do not copy the composition or background of the reference "
    "image (use it only for the animal's identity). Keep the animal's identity exactly the "
    "same: same species and breed, fur/coat colour, pattern and markings, eye colour, ear "
    "and muzzle shape, and body build. Do not restyle or alter its markings. Sharp focus, "
    "natural fur/skin texture, realistic lighting with soft shadows, high detail.")

IDENTITY_GUARD_OBJECT = (
    "This is the SAME object as the reference image. Preserve its identity EXACTLY: same "
    "shape and silhouette, colour, material and finish, proportions, and any logos, text or "
    "distinctive markings. Do NOT redesign, restyle or recolour it. Use the reference ONLY "
    "to lock the object's identity: take the angle, setting and framing from the description "
    "below, and do NOT copy the background shown in the reference image. Realistic "
    "photograph.")
IDENTITY_GUARD_OBJECT_MULTI = (
    "ALL the reference images show the SAME object (different angles or lighting). Use EVERY "
    "reference image together to lock its identity. Preserve its identity EXACTLY: same shape "
    "and silhouette, colour, material and finish, proportions, and any logos, text or "
    "distinctive markings. Do NOT redesign, restyle or recolour it. Take the angle, setting "
    "and framing from the description below, and do NOT copy the background shown in the "
    "reference images. Realistic photograph.")
IDENTITY_GUARD_OBJECT_KLEIN = (
    "Restage the shot to match this description — change the camera angle, framing and setting "
    "accordingly; do not copy the composition or background of the reference image (use it only "
    "for the object's identity). Keep the object's identity exactly the same: same shape and "
    "silhouette, colour, material and finish, proportions, and any logos, text or distinctive "
    "markings. Do not redesign or recolour it. Sharp focus, accurate materials and reflections, "
    "realistic lighting, high detail.")

IDENTITY_GUARD_CREATURE = (
    "This is the SAME creature as the reference image. Preserve its identity EXACTLY: same body "
    "form and silhouette, skin/scale/fur colour and texture, distinctive features (horns, wings, "
    "tail, markings), and proportions. Do NOT beautify, redesign or alter its distinctive "
    "features. Use the reference ONLY to lock the creature's identity: take the pose, setting and "
    "framing from the description below, and do NOT copy the pose or background shown in the "
    "reference image. Realistic, detailed render.")
IDENTITY_GUARD_CREATURE_MULTI = (
    "ALL the reference images show the SAME creature (different angles or poses). Use EVERY "
    "reference image together to lock its identity. Preserve its identity EXACTLY: same body form "
    "and silhouette, skin/scale/fur colour and texture, distinctive features (horns, wings, tail, "
    "markings), and proportions. Do NOT beautify, redesign or alter its distinctive features. Take "
    "the pose, setting and framing from the description below, and do NOT copy the pose or "
    "background shown in the reference images. Realistic, detailed render.")
IDENTITY_GUARD_CREATURE_KLEIN = (
    "Restage the shot to match this description — change the pose, camera angle, framing and "
    "setting accordingly; do not copy the composition or background of the reference image (use it "
    "only for the creature's identity). Keep the creature's identity exactly the same: same body "
    "form and silhouette, skin/scale/fur colour and texture, distinctive features (horns, wings, "
    "tail, markings), and proportions. Do not redesign or alter its distinctive features. Sharp "
    "focus, natural texture, realistic lighting with soft shadows, high detail.")

IDENTITY_GUARD_OTHER = (
    "This is the SAME subject as the reference image. Preserve its identity EXACTLY: same overall "
    "shape, colours, textures, proportions and any distinctive markings or features. Do NOT "
    "redesign, restyle or alter its defining details. Use the reference ONLY to lock the subject's "
    "identity: take the angle/pose, setting and framing from the description below, and do NOT copy "
    "the background shown in the reference image. Realistic photograph.")
IDENTITY_GUARD_OTHER_MULTI = (
    "ALL the reference images show the SAME subject (different angles or lighting). Use EVERY "
    "reference image together to lock its identity. Preserve its identity EXACTLY: same overall "
    "shape, colours, textures, proportions and any distinctive markings or features. Do NOT "
    "redesign, restyle or alter its defining details. Take the angle/pose, setting and framing "
    "from the description below, and do NOT copy the background shown in the reference images. "
    "Realistic photograph.")
IDENTITY_GUARD_OTHER_KLEIN = (
    "Restage the shot to match this description — change the angle, framing and setting "
    "accordingly; do not copy the composition or background of the reference image (use it only "
    "for the subject's identity). Keep the subject's identity exactly the same: same overall "
    "shape, colours, textures, proportions and any distinctive markings or features. Do not "
    "redesign or alter its defining details. Sharp focus, realistic lighting, high detail.")

# --- Anime / drawn character --------------------------------------------------
# The one subject type whose lock INVERTS a rule every other type takes for
# granted. For a human, an animal or a product the rendering is a constant we can
# safely hardcode ("realistic photograph") and the identity lives in physical
# traits — skin texture, coat markings, material finish. A drawn character has
# none of that: it is defined by its DESIGN (hair colour and shape, eye shape and
# iris colour, the signature outfit, the accessories, the marks) and by the ART
# STYLE it is drawn in. So this lock has to do two things no other lock does:
#   1. name design traits instead of physical ones, and
#   2. ACTIVELY FORBID the photorealism the other locks demand. Merely omitting
#      "realistic photograph" is not enough: the edit engines default to photo,
#      and the reference is the only thing holding the style — say it out loud.
# The signature OUTFIT is deliberately listed as identity, the exact opposite of
# the human path (which bakes OUTFIT_VARY into every shot so clothing does not
# bind to the person). For a character, the costume IS the character.
IDENTITY_GUARD_ANIME = (
    "This is the SAME anime character as the reference image. Preserve their character "
    "design EXACTLY: same hair colour, hairstyle and hair silhouette (bangs, ahoge, braids, "
    "twintails), same eye shape and iris colour, same skin tone, same signature outfit and "
    "its colours, the same accessories (ribbon, hairpin, glasses, earrings, headphones, "
    "weapon, animal ears or tail) and the same distinctive marks (mole, scar, facial "
    "marking). The ART STYLE IS PART OF THE CHARACTER: keep the same drawn anime rendering — "
    "same line work, cel shading and colour palette. Do NOT redesign the character and do "
    "NOT turn it into a photograph, a 3D render or a real person: no photographic skin "
    "texture, no pores, no film grain, no lens artefacts. Use the reference ONLY to lock "
    "the character design: take the pose, the expression, the framing and the setting from "
    "the description below, and do NOT copy the pose or the background shown in the "
    "reference image. Anime illustration, SFW.")
IDENTITY_GUARD_ANIME_MULTI = (
    "ALL the reference images show the SAME anime character (different angles, expressions "
    "or framings). Use EVERY reference image together to lock the character design. Preserve "
    "it EXACTLY: same hair colour, hairstyle and hair silhouette (bangs, ahoge, braids, "
    "twintails), same eye shape and iris colour, same skin tone, same signature outfit and "
    "its colours, the same accessories (ribbon, hairpin, glasses, earrings, headphones, "
    "weapon, animal ears or tail) and the same distinctive marks (mole, scar, facial "
    "marking). The ART STYLE IS PART OF THE CHARACTER: keep the same drawn anime rendering — "
    "same line work, cel shading and colour palette. Do NOT redesign the character and do "
    "NOT turn it into a photograph, a 3D render or a real person: no photographic skin "
    "texture, no pores, no film grain, no lens artefacts. Take the pose, the expression, the "
    "framing and the setting from the description below, and do NOT copy the pose or the "
    "background shown in the reference images. Anime illustration, SFW.")
IDENTITY_GUARD_ANIME_KLEIN = (
    "Restage the shot to match this description — change the pose, camera angle, framing and "
    "setting accordingly; do not copy the composition or the background of the reference "
    "image (use it only for the character design). Keep the character design exactly the "
    "same: same hair colour, hairstyle and hair silhouette, same eye shape and iris colour, "
    "same skin tone, same signature outfit and its colours, the same accessories and the "
    "same distinctive marks. Keep the same drawn anime art style: same line work, cel "
    "shading and colour palette. Do not redesign the character and do not turn it into a "
    "photograph, a 3D render or a real person — no photographic skin texture, no pores, no "
    "film grain. Clean line art, flat cel shading, crisp details.")

# Per-subject default registry. 'human' points at the ORIGINAL dict so the human
# path is byte-identical; the others compose their guards with the shared,
# subject-agnostic klein_improve instruction.
_IDENTITY_DEFAULTS_BY_SUBJECT = {
    'human': _IDENTITY_PROMPT_DEFAULTS,
    'animal': {'face_single': IDENTITY_GUARD_ANIMAL, 'face_multi': IDENTITY_GUARD_ANIMAL_MULTI,
               'klein_identity': IDENTITY_GUARD_ANIMAL_KLEIN, 'klein_improve': KLEIN_IMAGE_IMPROVE_PROMPT},
    'object': {'face_single': IDENTITY_GUARD_OBJECT, 'face_multi': IDENTITY_GUARD_OBJECT_MULTI,
               'klein_identity': IDENTITY_GUARD_OBJECT_KLEIN, 'klein_improve': KLEIN_IMAGE_IMPROVE_PROMPT},
    'creature': {'face_single': IDENTITY_GUARD_CREATURE, 'face_multi': IDENTITY_GUARD_CREATURE_MULTI,
                 'klein_identity': IDENTITY_GUARD_CREATURE_KLEIN, 'klein_improve': KLEIN_IMAGE_IMPROVE_PROMPT},
    'other': {'face_single': IDENTITY_GUARD_OTHER, 'face_multi': IDENTITY_GUARD_OTHER_MULTI,
              'klein_identity': IDENTITY_GUARD_OTHER_KLEIN, 'klein_improve': KLEIN_IMAGE_IMPROVE_PROMPT},
    'anime': {'face_single': IDENTITY_GUARD_ANIME, 'face_multi': IDENTITY_GUARD_ANIME_MULTI,
              'klein_identity': IDENTITY_GUARD_ANIME_KLEIN, 'klein_improve': KLEIN_IMAGE_IMPROVE_PROMPT},
}


def identity_prompt_default(kind: str, subject_type: str = 'human') -> str:
    """The shipped (hardcoded) default for an identity-prompt kind — what a
    Settings "Restore default" returns to. `subject_type` selects the human /
    animal / object / creature / other lock; the human table is byte-identical to
    the original. Raises KeyError on an unknown kind."""
    table = _IDENTITY_DEFAULTS_BY_SUBJECT.get(normalize_subject_type(subject_type),
                                              _IDENTITY_PROMPT_DEFAULTS)
    return table[kind]


def identity_prompt_defaults(subject_type: str = 'human') -> dict:
    """All four shipped defaults as a fresh {kind: text} dict — read-only view
    surfaced in the settings payload so the UI can SHOW the effective default
    prompt (and let the user copy it into the field to edit), instead of an
    empty box with a generic "leave blank" placeholder. These are code
    constants, not secrets, so they are safe to return verbatim."""
    st = normalize_subject_type(subject_type)
    return dict(_IDENTITY_DEFAULTS_BY_SUBJECT.get(st, _IDENTITY_PROMPT_DEFAULTS))


def identity_prompt_defaults_by_subject() -> dict:
    """Every subject type's defaults at once — {subject_type: {kind: text}}. The
    Settings screen edits the prompts OUT of any dataset context, so it needs all
    five sets to show the right default next to whichever subject the user picks."""
    return {st: dict(table) for st, table in _IDENTITY_DEFAULTS_BY_SUBJECT.items()}


# Kinds whose override is scoped PER SUBJECT TYPE. `klein_improve` is deliberately
# NOT one of them: it is a subject-agnostic quality instruction ("add texture and
# detail"), identical in every default table, so splitting it per subject would
# multiply a setting nobody would want to keep in sync.
#
# The render tail and the framing detail join them: the anime tail says "Anime
# illustration, same art style as the reference" where every photographic subject
# says "Professional realistic photograph", and the framing blocks are already six
# different tables. One shared text for all six would be a regression.
# The other four parts (markings lock, the two directives, the garment palette)
# stay FLAT/global on purpose: `_augment_prompt` only ever runs on the human
# catalog, so the directives and the garments have no non-human meaning to split,
# and the markings order is one sentence about not inventing detail.
PER_SUBJECT_PROMPT_KINDS = ('face_single', 'face_multi', 'klein_identity',
                            'render_tail_sfw', 'render_tail_nsfw',
                            'framing_face', 'framing_bust', 'framing_body', 'framing_back')


def identity_prompt_config_key(kind: str, subject_type: str = 'human') -> str:
    """Where the override for (kind, subject_type) lives in the config.

    HUMAN keeps the ORIGINAL flat key `identity_prompts.<kind>` — never renamed,
    never migrated. That is where every override written before this fix landed
    (the editable-prompt UI shipped human-only text and no subject selector), so
    reading it as the human override preserves it exactly: a user who tuned the
    lock for people keeps it applying to people, with no migration step that could
    lose it. Non-human subjects get their OWN branch,
    `identity_prompts.by_subject.<subject_type>.<kind>`, with NO fallback to the
    flat key — that fallback IS the bug (reported by ashish.sinha): a prompt
    written while looking at an Animal dataset was stored globally and then rode
    on human generations, producing tails and extra limbs."""
    st = normalize_subject_type(subject_type)
    if st == 'human' or kind not in PER_SUBJECT_PROMPT_KINDS:
        return f'identity_prompts.{kind}'
    return f'identity_prompts.by_subject.{st}.{kind}'


def read_prompt_override(identity_prompts, kind: str, subject_type: str = 'human'):
    """The override for (kind, subject_type) inside a PLAIN `identity_prompts`
    dict — the same tree `config.get('identity_prompts.…')` walks, but read from a
    dict handed in by a caller instead of from the saved config. Mirrors
    identity_prompt_config_key exactly (human = flat legacy key, others under
    `by_subject`); returns None for anything missing or malformed, so a
    hand-edited config file or a truncated request body degrades to the default
    instead of raising."""
    if not isinstance(identity_prompts, dict):
        return None
    st = normalize_subject_type(subject_type)
    if st == 'human' or kind not in PER_SUBJECT_PROMPT_KINDS:
        return identity_prompts.get(kind)
    by = identity_prompts.get('by_subject')
    if not isinstance(by, dict):
        return None
    node = by.get(st)
    return node.get(kind) if isinstance(node, dict) else None


# --- Preview overrides (Settings ▸ Image engines composed preview) ------------
# The Settings screen saves on an explicit button, so a preview that read the
# SAVED config would show the previous text while the user edits — the one moment
# the preview exists for. Instead the screen posts its in-flight `identity_prompts`
# object and the preview composes with it, through the REAL wrappers: the point of
# the panel is to show the string the engine receives, and a second composition
# path in JavaScript would drift from this one on the first change made here.
#
# A ContextVar (not a module global) so a preview request can never leak its
# unsaved text into a generation running on another thread. Default None = "no
# preview in flight", which is every code path except the preview route.
_PREVIEW_OVERRIDES = contextvars.ContextVar('lds_preview_prompt_overrides', default=None)


@contextlib.contextmanager
def preview_prompt_overrides(identity_prompts):
    """Compose prompts against `identity_prompts` (an unsaved Settings tree)
    instead of the saved config, for the duration of the block."""
    token = _PREVIEW_OVERRIDES.set(identity_prompts if isinstance(identity_prompts, dict) else None)
    try:
        yield
    finally:
        _PREVIEW_OVERRIDES.reset(token)


def get_identity_prompt(kind: str, subject_type: str = 'human') -> str:
    """Effective identity/quality prompt for `kind` and `subject_type`: the user's
    Settings override FOR THAT SUBJECT TYPE when it holds non-blank text, else the
    subject-type default (human = byte-identical to the hardcoded constant). Lazy,
    defensive config read so the no-override path a unit test exercises returns the
    default unchanged even outside a Flask app.

    The override is scoped per subject type — see identity_prompt_config_key for
    the storage layout and why human stays on the legacy flat key."""
    default = identity_prompt_default(kind, subject_type)
    pending = _PREVIEW_OVERRIDES.get()
    if pending is not None:
        # A preview shows the editor's state, WHOLE: no fall-through to the saved
        # config, or a field the user just cleared would still look overridden.
        override = read_prompt_override(pending, kind, subject_type)
        return override if isinstance(override, str) and override.strip() else default
    try:
        from .. import config as cfg
        override = cfg.get(identity_prompt_config_key(kind, subject_type))
    except Exception:
        return default
    if isinstance(override, str) and override.strip():
        return override
    return default


# --- Prompt suffixes (community feature request) -----------------------------
# A FREE creative direction the user attaches to the DATASET (global text and/or a
# per-framing map {face,bust,body,back}) that rides on every generated variation.
# Applied at WRAP time ONLY: the stored variation_prompt stays raw, so a later
# regenerate re-applies the CURRENT suffix exactly once (baking it into the stored
# prompt would double-apply on regeneration). The suffix always lands in the
# DESCRIPTIVE portion of the wrapper — never ahead of (or inside) the identity
# lock, which stays byte-identical.
def _append_suffix(prompt: str, suffix: str) -> str:
    """Splice the creative-direction suffix into the descriptive prompt text.
    Empty/blank suffix -> the prompt comes back byte-identical (the no-suffix
    regression invariant). Trailing '.'/',' are trimmed on both sides so the
    join always reads as one clean comma-separated description."""
    s = (suffix or '').strip().rstrip('.,').strip()
    if not s:
        return prompt
    p = (prompt or '').rstrip().rstrip('.,').rstrip()
    return f'{p}, {s}'


def compose_prompt_suffix(global_suffix, framing_suffixes=None, framing=None) -> str:
    """Effective suffix for ONE shot. `framing_suffixes` is the per-framing map
    {face,bust,body,back} — as a dict or the JSON string stored on the dataset
    row (defensively parsed). Composition order: the per-framing suffix FIRST
    (the more specific direction sits closest to the shot description), then the
    global one, comma-joined; an exact duplicate collapses to one. Returns ''
    when nothing applies. Pure — no DB, no Flask."""
    m = framing_suffixes
    if isinstance(m, str):
        try:
            m = json.loads(m)
        except (ValueError, TypeError):
            m = None
    per = ''
    if isinstance(m, dict) and framing:
        v = m.get(framing)
        per = v.strip() if isinstance(v, str) else ''
    g = (global_suffix or '').strip()
    if per and g and per.lower() == g.lower():
        g = ''
    parts = [x for x in (per.rstrip('.,').strip(), g.rstrip('.,').strip()) if x]
    return ', '.join(parts)


def wrap_variation(prompt: str, ref_count: int = 1, suffix: str = '',
                   subject_type: str = 'human') -> str:
    """Guard-FIRST wrapper (API engines). The identity guard stays the very first
    thing the model reads; the dataset suffix extends the descriptive tail AFTER
    it (appended to the creative prompt), so the lock is never diluted. `subject_type`
    picks the human/animal/object/creature/other identity lock (default 'human' =
    byte-identical to the historical output)."""
    guard = get_identity_prompt('face_multi' if ref_count > 1 else 'face_single', subject_type)
    return f"{guard} {apply_directive_overrides(_append_suffix(prompt, suffix))}"


# Enrichissement PAR CADRAGE pour Klein (étude prompts 2026-07-10, sources :
# guide fal.ai Flux2-klein + guide BFL FLUX.2) : Klein veut des descriptions
# CONCRÈTES et détaillées (hiérarchie sujet → cadre → technique) — les tags
# télégraphiques du catalogue suffisent aux moteurs API (qui brodent seuls)
# mais SOUS-spécifient Klein, qui comble les trous arbitrairement.
_KLEIN_FRAMING_DETAIL = {
    'face': ('Close-up head-and-shoulders portrait: the face fills most of the frame, '
             'both eyes in crisp focus, 85mm portrait lens look with gentle background '
             'separation.'),
    'bust': ('Half-length portrait from the waist up: torso and shoulders naturally '
             'posed, hands relaxed if visible, 50mm lens look.'),
    'body': ('Full-length shot: the ENTIRE body visible from head to toe including the '
             'feet, natural standing distance, 35mm lens look, the figure well '
             'proportioned within the frame.'),
    'back': ('Seen from behind: back to the camera, head direction natural, full or '
             'three-quarter figure.'),
}

# Klein subject noun + per-subject framing detail. The human map above is reused
# verbatim for 'human' so the human Klein prompt stays byte-identical; the others
# are terse, subject-appropriate framing hints (Klein under-fills bare tags). The
# face/bust/body/back keys stay the internal framing enum (composition/aspect are
# shared) — only the WORDING adapts to the subject.
_KLEIN_SUBJECT_NOUN = {'human': 'person', 'animal': 'animal', 'creature': 'creature',
                       'object': 'object', 'other': 'subject', 'anime': 'character'}

# THE MEDIUM AND THE RENDERING TAIL ARE PART OF THE PROMPT, AND THEY WERE HARDCODED.
# `wrap_variation_klein` used to open with "Create a new photograph of the same …"
# and close with "Professional realistic photograph, SFW." Neither string belongs to
# the editable identity guard, so for the five photographic subject types nobody ever
# had to think about them — and for 'anime' they would have overruled the anime lock
# TWICE in the same prompt, at the two positions an instruction model weighs most
# (the opening command and the final style tag). They are per-subject now. Every
# pre-existing type reads the DEFAULT tuple, byte for byte what it used to inline, so
# no existing dataset's prompt changes by a single character.
_KLEIN_MEDIUM = {'anime': 'illustration'}
_KLEIN_MEDIUM_DEFAULT = 'photograph'
_KLEIN_RENDER_TAIL_DEFAULT = (
    "Explicit nudity is allowed; render natural, anatomically correct forms. "
    "Professional realistic photograph.",
    "Professional realistic photograph, SFW.")
_KLEIN_RENDER_TAIL = {
    'anime': ("Explicit nudity is allowed; render natural, anatomically correct forms in the "
              "same drawn anime art style. Anime illustration, same art style as the reference.",
              "Anime illustration, same art style as the reference, SFW."),
}
_KLEIN_FRAMING_DETAIL_ANIMAL = {
    'face': 'Close-up of the head filling the frame, both eyes in sharp focus, gentle background separation.',
    'bust': 'Half-body framing: head, chest and front legs, natural pose.',
    'body': 'Full body from head to tail including the feet/paws, the whole animal well proportioned in the frame.',
    'back': 'Seen from behind: hindquarters and tail visible, head direction natural.',
}
_KLEIN_FRAMING_DETAIL_OBJECT = {
    'face': 'Tight detail crop: texture, material and markings in sharp focus.',
    'bust': 'Medium framing of the object from a slight angle, well lit.',
    'body': 'The ENTIRE object in frame, front or three-quarter angle, well proportioned and evenly lit.',
    'back': 'Rear view of the object, plain background.',
}
_KLEIN_FRAMING_DETAIL_CREATURE = {
    'face': 'Close-up head-and-shoulders framing, face and distinctive features in sharp focus.',
    'bust': 'Half-length framing from the waist up, distinctive features visible.',
    'body': 'Full figure from head to feet, the whole creature well proportioned in the frame.',
    'back': 'Seen from behind: back, silhouette and back-facing features visible.',
}
_KLEIN_FRAMING_DETAIL_OTHER = {
    'face': 'Tight detail crop, texture and detail in sharp focus.',
    'bust': 'Medium framing of the subject from a slight angle.',
    'body': 'The ENTIRE subject in frame, well proportioned and evenly lit.',
    'back': 'Seen from behind: rear view of the subject.',
}
# Anime framing hints use DRAWN conventions, not photographic ones: "bust-up",
# "cowboy shot" (knee-up) and "full body" are the vocabulary of the medium, and a
# "85mm portrait lens look" would reintroduce exactly the photographic intent the
# anime lock forbids.
_KLEIN_FRAMING_DETAIL_ANIME = {
    'face': 'Close-up of the face filling the frame, both eyes drawn in full detail, clean line work.',
    'bust': 'Bust-up framing from the chest up, the collar and shoulders of the signature outfit visible.',
    'body': 'Full body from head to feet including the shoes, the whole character well proportioned in the frame.',
    'back': 'Seen from behind: the back of the head, the hair silhouette and the back of the outfit visible.',
}
_KLEIN_FRAMING_DETAIL_BY_SUBJECT = {
    'human': _KLEIN_FRAMING_DETAIL,
    'animal': _KLEIN_FRAMING_DETAIL_ANIMAL,
    'object': _KLEIN_FRAMING_DETAIL_OBJECT,
    'creature': _KLEIN_FRAMING_DETAIL_CREATURE,
    'other': _KLEIN_FRAMING_DETAIL_OTHER,
    'anime': _KLEIN_FRAMING_DETAIL_ANIME,
}


def wrap_variation_klein(prompt: str, nsfw: bool = False, framing: str | None = None,
                         suffix: str = '', subject_type: str = 'human',
                         label: str = '') -> str:
    """Klein (FLUX.2, Kontext-lineage) is an INSTRUCTION-edit model: it follows
    imperative edit commands (the consistency LoRA's own usage example is "Turn
    this cat into a dog"). The API-engine wrapper above — preservation order
    FIRST, descriptive tags after — reads as "change nothing", so Klein returned
    a near-copy of the reference (live repro 2026-07-10: every variation looked
    like a plain upscale). Structure follows the fal.ai/BFL edit guidance:
      1. direct command first (the change),
      2. the FULL intended result (framing-specific detail — Klein under-fills
         terse tag prompts, unlike the API engines which embellish on their own),
      3. restage + identity constraints,
      4. photographic/technical tail.
    NEGATIVE PROMPTS: dead end at CFG 1 (guidance-distilled model — the sampler
    ignores the negative conditioning entirely; ComfyUI-NAG would be needed to
    restore them). All steering therefore lives in the POSITIVE prompt.
    `nsfw=True` (local Klein only — the route refuses NSFW on API engines) drops
    the SFW clamp and allows explicit nudity with natural anatomy.
    `suffix` (dataset prompt-suffix) joins the DESCRIPTIVE portion (2. — appended
    to the creative prompt, before the framing detail): instruction-first means
    the description IS the command, so the suffix steers the intended result and
    never touches the restage/identity constraints that follow. Empty suffix ->
    byte-identical output.
    `label` (the variation label) picks this shot's concrete garment — see the
    two Krea-born fixes below, both MEASURED on Klein before being applied here,
    which is why Klein passes the same two flags as Krea.
    """
    return _compose_edit_prompt(prompt, nsfw=nsfw, framing=framing, suffix=suffix,
                                subject_type=subject_type, label=label,
                                concrete_outfit=True, markings_lock=True)


# --- Anti-fuite tenue / expression (constat terrain 2026-07-14) ---------------
# Les moteurs d'édition (Nano Banana, ChatGPT-image, Klein) PRÉSERVENT ce qu'on ne
# contredit pas explicitement. Symptômes réels rapportés par le propriétaire :
#   1) sur les plans buste, le modèle reprend la MÊME tenue que la réf → la tenue se
#      lie à l'identité dans le LoRA ;
#   2) l'expression de la réf (sourire, grimace) se propage à TOUS les plans.
# Deux corrections complémentaires, au bon niveau :
#   • WRAPPERS (IDENTITY_GUARD / IDENTITY_GUARD_MULTI / wrap_variation_klein) : la réf
#     ne sert QU'À l'identité du visage ; tenue + expression viennent de la description,
#     jamais copiées de la réf. Directive GÉNÉRALE → couvre aussi les prompts édités /
#     custom, les deux familles de moteurs et la régénération.
#   • CATALOGUE : chaque entrée SANS tenue / expression explicite reçoit une cible
#     CONCRÈTE mais variée (les modèles d'édition suivent mieux une consigne « porte X »
#     qu'un vide qu'ils comblent par la réf). Baker la directive dans le TEXTE du prompt
#     la propage partout (API + Klein + persistance variation_prompt + régénération).
OUTFIT_VARY = ('wearing a different casual everyday outfit, varied in style and colour '
               '(not the outfit from the reference image)')
EXPRESSION_NEUTRAL = ('a calm neutral facial expression, not copying the expression from '
                      'the reference image')

# Détecteurs « le texte nomme-t-il DÉJÀ une tenue / une expression ? » (mots entiers).
# Servent à n'ajouter la directive par défaut qu'aux entrées qui n'en portent pas —
# les entrées à tenue nommée (veste, robe, bikini…) ou expression nommée (sourire,
# sérieux…) gardent la leur. OUTFIT_VARY contient « outfit » et EXPRESSION_NEUTRAL
# « expression » → la passe d'augmentation est idempotente.
_HAS_OUTFIT = re.compile(
    r'\b(outfit|top|clothes|clothing|jacket|dress|bikini|swimsuit|swimwear|sportswear|'
    r'leggings|jeans|lingerie|towel|shirt|blouse|coat|skirt|gown|suit)\b', re.I)
# NB: 'neutral' is deliberately NOT an expression token — it's ambiguous with the
# frequent 'neutral background/studio' phrasing. A shot whose only expression cue is a
# bare 'neutral' therefore GAINS the explicit EXPRESSION_NEUTRAL directive (which also
# adds the 'not copying the reference' anti-leak clause a bare 'neutral' lacks).
_HAS_EXPRESSION = re.compile(
    r'\b(expression|smil\w*|serious|laugh\w*|surprised|pensive|grin\w*|'
    r'frown\w*|smirk\w*|pout\w*)\b', re.I)


# --- Krea 2 Identity Edit wrapper --------------------------------------------
# MEASURED on a live install (2026-07-25), single reference photo, NO character
# LoRA:
#   ✅ identity holds by itself — forehead/neck tattoos, five piercings, hoop
#      earrings, body morphology, all carried from the one reference;
#   ✅ the requested FRAMING is honoured (a real full-length shot when asked),
#      which the API-engine wrapper never managed on this model — that wrapper
#      leads with preservation ("change nothing"), and Krea answered with a
#      close-up whatever the shot asked for. So Krea reuses the KLEIN wrapper's
#      shape: imperative command first, full intended result, then the identity
#      lock. Same model class (instruction edit + reference grounding), same
#      winning structure.
#   ⚠️ the OUTFIT never changed, and the TATTOOS were REDRAWN each time (same
#      spirit, different design) — the two things this wrapper has to fix.
#
# Krea reuses the `klein_identity` editable lock deliberately: it is the "local
# edit engine" identity lock, already per-subject-type and already overridable in
# Settings. A second copy of the same sentence would be one more thing for the
# user to keep in sync, for no behavioural gain.

# The model PRESERVES anything it is not positively ordered to change, so a
# NEGATION is a no-op on it. These rules turn the catalog's negative outfit
# phrasings into a concrete positive garment (see krea_edit_helper.outfit_for:
# deterministic per shot label, so outfits vary across the dataset but a
# regenerate of one shot reproduces its own). Applied at WRAP time only — the
# stored `variation_prompt` keeps the raw catalog text, so the other engines are
# untouched and a regenerate re-applies the current rules exactly once.
#
# ALSO APPLIED TO KLEIN since 2026-07-27, on measurement, and to the two LOCAL
# edit engines ONLY. What Klein actually does with the negation (5 pairs, same
# seed, one factor changed):
#   • it is NOT copying the reference garment — the negation IS followed, and the
#     "colour leaks in tight shots" reading did NOT reproduce: the two fresh bust
#     shots came back beige and grey-green knits, muted neutrals, not the
#     reference's olive. That symptom is written off.
#   • what DOES collapse is the lower half: all three wide shots answered the
#     negation with blue jeans and pale sneakers, whatever the top. A dataset of
#     that teaches the LoRA "this person wears blue jeans".
#   • the concrete garment was obeyed exactly in 5/5 and broke the collapse in
#     3/3 (grey denim / black trousers / black trousers) with three visibly
#     different tops. That is the reason it ships here — not the colour claim.
# The API engines (Nano Banana, ChatGPT, OpenRouter) are deliberately NOT
# included: nothing has been measured on them, and a Klein result is not
# evidence about a different model family.
_KREA_OUTFIT_GENERIC = (
    re.compile(r'\bcasual clothes different from the reference outfit\b', re.I),
    re.compile(r'\bdifferent outfit\b', re.I),
)
# Leftover negations after a NAMED garment ("wearing a jacket different from the
# reference outfit") — the garment already carries the intent, the negation only
# spends tokens on something this model ignores.
_KREA_NEGATION = re.compile(r',?\s*different from the reference (?:image|outfit)\b', re.I)
# EXPRESSION_NEUTRAL carries the same kind of dead negation. Not separately
# measured — same mechanism, and dropping the clause loses nothing positive.
_KREA_EXPRESSION_NEGATION = re.compile(
    r',?\s*not copying the expression from the reference image\b', re.I)

# Permanent markings were redrawn on every render (measured). A dataset built
# from that teaches the LoRA an AVERAGE tattoo — the exact failure mode a
# character LoRA must not have.
#
# ⚠️ The first version of this hold order ENUMERATED what to preserve ("tattoos
# with the same design..., the same scars, moles and piercings"). It was assumed
# to "cost nothing when the subject has no markings". It cost a great deal: a
# text encoder does not bind "reproduce X as in the reference" — it reads the
# word `tattoos`, and paints tattoos. Reported within hours on subjects who have
# none. Naming a feature summons it; this is the oldest trap in prompting, and
# the enumeration walked straight into it.
#
# So the order now holds the SKIN, without naming a single feature. What is on
# the reference is carried by the reference conditioning, not by this sentence —
# whose only job is to forbid invention and redrawing.
#
# The re-worded order had only ever been checked in TEXT (the word was gone from
# the prompt). It was verified IN IMAGE on 2026-07-27, on Klein, before being
# extended to it — three shots on a subject with no markings at all, with and
# without the sentence, same seed: nothing invented in 3/3, the pairs are
# near-identical. And on the tattooed subject it earns its place: on the outdoor
# bust the forehead piece VANISHED without it and is fully there with it, same
# seed; the jaw work is preserved on both sides on the close-up; the wide shot is
# a draw. Hence: applied to Klein too, and to the local edit engines only.
KREA_MARKINGS_LOCK = (
    "Keep the skin exactly as it is in the reference image: do not add anything "
    "to it, and do not redraw, restyle, move or remove what is already there. ")


# Concrete garments the negation is replaced BY. The pick is deterministic on the
# shot label via crc32 — NOT hash(), whose PYTHONHASHSEED randomisation would give
# the same shot a different outfit on every app restart. Three properties at once:
# outfits genuinely differ across the dataset, one shot regenerates identically,
# and no randomness ever leaks into a stored prompt.
#
# SIZE IS MEASURED, not decorative. With crc32 modulo N, the load histogram over
# the shipped catalog depends only on N, so the count was chosen by measuring it:
# at 12 garments, 41 eligible shots collapsed onto 11 distinct ones with the
# worst garment carrying 6 (a dataset that trades one uniform for another); at 25
# they spread over 23 distinct ones with the worst carrying 4. Neighbouring sizes
# are NOT monotonically better (24 peaks at 5, 22 at 6) — hence the pinned
# distribution test, which fails if a catalog change re-concentrates the picks.
# Every entry differs from the others on at least two of colour / cut / sleeve
# length / material, so two shots that DO share a garment still look apart, and
# all of them are neutral, everyday and plausible from a face crop to a full
# body. Nothing here may name a summonable skin feature (see _SUMMONABLE in the
# tests): 'scarf' contains 'scar'.
KREA_OUTFIT_PALETTE = (
    'a plain white cotton t-shirt',
    'a red knit sweater',
    'a navy blue zip hoodie',
    'a beige linen shirt',
    'a black fitted turtleneck',
    'an olive green utility jacket',
    'a grey marl sweatshirt',
    'a burgundy long-sleeve top',
    'a light blue denim shirt',
    'a mustard yellow cardigan',
    'a dark green flannel shirt',
    'a cream ribbed knit top',
    'a charcoal wool blazer',
    'a rust orange corduroy shirt',
    'a teal cotton polo shirt',
    'a soft pink oversized sweatshirt',
    'a striped navy and white long-sleeve top',
    'a camel wool coat',
    'a black leather biker jacket',
    'a lilac short-sleeve blouse',
    'a chocolate brown suede jacket',
    'a sky blue gingham shirt',
    'a plum velvet top',
    'a silver grey satin blouse',
    'a coral sleeveless linen top',
)


# --- Editable defaults for the five parts that were hardcoded ----------------
# Registered into the SAME per-subject tables the four identity locks live in, so
# `get_identity_prompt`, the Settings payload, the config-key layout and "Restore
# default" all work on them with no second mechanism. Done here, after the
# constants exist, rather than at the table literal 400 lines above — the tables
# are built before the framing/tail/directive constants are defined.
#   • the two SUBJECT-scoped parts (render tail, framing detail) take that
#     subject's own text, so anime keeps its illustration tail and its drawn
#     framing vocabulary;
#   • the flat ones repeat identically in every table — that is what makes them
#     read the flat legacy-style key through identity_prompt_config_key.
# The markings lock is registered STRIPPED: its constant carries a trailing space
# because it used to be concatenated raw, and a trailing space in a textarea is
# invisible and impossible to preserve through an edit. `_compose_edit_prompt`
# re-adds the separator, so the composed bytes do not move.
def _register_prompt_part_defaults():
    for st, table in _IDENTITY_DEFAULTS_BY_SUBJECT.items():
        nsfw_tail, sfw_tail = _KLEIN_RENDER_TAIL.get(st, _KLEIN_RENDER_TAIL_DEFAULT)
        detail = _KLEIN_FRAMING_DETAIL_BY_SUBJECT.get(st, _KLEIN_FRAMING_DETAIL)
        table['markings_lock'] = KREA_MARKINGS_LOCK.strip()
        table['outfit_vary'] = OUTFIT_VARY
        table['expression_neutral'] = EXPRESSION_NEUTRAL
        table['outfit_palette'] = '\n'.join(KREA_OUTFIT_PALETTE)
        table['render_tail_sfw'] = sfw_tail
        table['render_tail_nsfw'] = nsfw_tail
        for f in PROMPT_FRAMINGS:
            table[f'framing_{f}'] = detail.get(f, '')


_register_prompt_part_defaults()


def outfit_palette(subject_type: str = 'human') -> tuple:
    """The concrete garments in use — the user's list when they wrote one, else
    the shipped tuple. Stored as one garment per line; blank lines and stray
    whitespace are dropped, and a palette the user emptied entirely falls back to
    the shipped one rather than producing a prompt with no garment in it at all.

    ⚠️ `krea_outfit_for` picks by crc32(label) % len(palette), so the LENGTH of
    this list decides which shot gets which garment. Adding or removing one entry
    reshuffles every shot's outfit — same shots, different clothes. That is
    surfaced in the field's help text, because it looks like a bug otherwise."""
    raw = get_identity_prompt('outfit_palette', subject_type)
    items = tuple(line.strip() for line in (raw or '').splitlines() if line.strip())
    return items or KREA_OUTFIT_PALETTE


def apply_directive_overrides(text: str) -> str:
    """Swap the SHIPPED outfit / expression directives for the user's overrides,
    at wrap time.

    They cannot be applied where they are injected: `_augment_prompt` bakes them
    into the catalog at IMPORT time, with no Flask app and no config yet, and the
    augmented text is then PERSISTED in `variation_prompt`. Doing the swap here
    instead has the property that matters — an edit reaches datasets that were
    built before it, on their next generation, instead of only new ones.
    No override (or a prompt that never carried the directive) -> the text comes
    back byte-identical."""
    out = text or ''
    for kind, shipped in (('outfit_vary', OUTFIT_VARY),
                          ('expression_neutral', EXPRESSION_NEUTRAL)):
        current = get_identity_prompt(kind)
        if current != shipped and shipped in out:
            out = out.replace(shipped, current)
    return out


def krea_outfit_for(label: str) -> str:
    """A concrete garment for a shot label — stable across runs and processes."""
    palette = outfit_palette()
    key = (label or '').encode('utf-8', 'replace')
    return palette[zlib.crc32(key) % len(palette)]


def krea_outfit_directive(prompt: str, label: str = '') -> str:
    """Rewrite the catalog's NEGATIVE outfit/expression clauses into positive
    ones for Krea. Idempotent (the substituted text contains none of the
    patterns) and a no-op on any prompt that never carried them — which is every
    non-human catalog, since `_augment_prompt` only runs on the human one."""
    p = prompt or ''
    garment = f'wearing {krea_outfit_for(label)}'
    p = p.replace(OUTFIT_VARY, garment)
    for rx in _KREA_OUTFIT_GENERIC:
        p = rx.sub(garment, p)
    p = _KREA_NEGATION.sub('', p)
    p = _KREA_EXPRESSION_NEGATION.sub('', p)
    return p


def wrap_variation_krea(prompt: str, nsfw: bool = False, framing: str | None = None,
                         suffix: str = '', subject_type: str = 'human',
                         label: str = '') -> str:
    """Full Krea 2 Identity Edit prompt for one shot.

    Same four-part structure as `wrap_variation_klein` (command → full intended
    result → identity lock → rendering tail), plus the two Krea-specific fixes:
    the outfit negation becomes a concrete garment, and permanent markings get an
    explicit hold order. `label` selects that garment deterministically — pass
    the variation label so the same shot always renders the same outfit.

    The outfit and permanent-marking fixes are also used by Klein, but Krea has
    one deliberately local addition: conflicting pose cards receive a final,
    catalog-owned priority instruction.  The named entry point keeps that engine
    boundary explicit rather than letting a Krea tuning silently affect Klein.
    """
    return _compose_edit_prompt(prompt, nsfw=nsfw, framing=framing, suffix=suffix,
                                 subject_type=subject_type, label=label,
                                 concrete_outfit=True, markings_lock=True,
                                 card_pose_priority=True)


# Krea 2 is especially literal about the prompt tail.  The otherwise useful
# generic human details say "both eyes in crisp focus" for a face and "natural
# standing distance" for a body.  Those clauses directly fight a profile card
# and a sitting/lying/walking card respectively, then pull the edit back toward
# the reference composition.  Do not try to rewrite the card itself: it is a
# persisted user-facing instruction.  Instead, omit the generic block for the
# conflicting cards and repeat the card's instruction as the final imperative.
#
# This is intentionally Krea-only.  Klein was separately measured with its
# framing details, and API engines use `wrap_variation`, not this local-edit
# assembly path.
_KREA_PROFILE_VIEW = re.compile(
    r'\b(?:left|right)\s+(?:profile|side)(?:\s+view)?\b|'
    r'\b(?:profile|side)(?:\s+view)?\s+(?:left|right)\b|'
    r'\b(?:strict\s+)?profile(?:\s+view)?\b|\bside[- ]view\b', re.I)
_KREA_THREE_QUARTER_VIEW = re.compile(
    r'\b(?:three[- ]quarter|3\s*(?:/|-)\s*4)\b', re.I)
_KREA_NON_STANDING_BODY_POSE = re.compile(
    r'\b(?:sitting|seated|sit|lying|reclining|kneeling|crouching|squatting|'
    r'walking|running|jumping|dancing)\b', re.I)
KREA_CARD_POSE_PRIORITY_PREFIX = 'Mandatory shot requirement:'
KREA_CARD_POSE_PRIORITY_SUFFIX = (
    'Render exactly this pose, camera angle and framing; do not substitute the '
    'reference pose or a different view.')
KREA_TRUE_PROFILE_REQUIREMENT = (
    'For a profile shot, rotate the face to a true 90-degree side view with only '
    'the nearer eye visible.')
KREA_CARD_POSE_PRIORITY_GENERIC = (
    f'{KREA_CARD_POSE_PRIORITY_PREFIX} honor exactly the pose, camera angle and '
    f'framing already stated in the shot description. {KREA_CARD_POSE_PRIORITY_SUFFIX}')
_KREA_TRUSTED_TAIL_MAX_CHARS = 600
# The final priority instruction must stay server-owned.  These replace the
# catalog's default variation clauses without consulting the editable outfit
# palette (whose text is valid in the normal description, but must not appear
# after identity/SFW locks).
_KREA_CATALOG_OUTFIT = 'wearing a varied casual outfit'
_KREA_CATALOG_EXPRESSION = 'a calm neutral facial expression'


def krea_card_pose_priority(card_prompt: str = '', *, true_profile: bool = False) -> str:
    """Return Krea's final pose imperative without moving free-form text after SFW.

    A concrete prompt is only accepted here after `_krea_builtin_card_prompt`
    proved it is one of the shipped catalog cards.  Custom and edited prompts get
    the bounded generic form: their actual words stay in the ordinary description
    before the identity and rendering locks.
    """
    shot = (card_prompt or '').strip().rstrip('.')
    if not shot:
        priority = KREA_CARD_POSE_PRIORITY_GENERIC
    else:
        priority = (
            f'{KREA_CARD_POSE_PRIORITY_PREFIX} {shot}. '
            f'{KREA_CARD_POSE_PRIORITY_SUFFIX}')
    # ``true_profile`` comes from a pose classifier before the locks, but only
    # selects this static instruction; no free-form request text moves after SFW.
    return (f'{priority} {KREA_TRUE_PROFILE_REQUIREMENT}'
            if true_profile else priority)


def _krea_card_pose_needs_priority(prompt: str, framing: str | None,
                                   subject_type: str) -> bool:
    """Whether Krea's generic human framing would contradict this shot card."""
    # The non-human framing details do not contain the human-only "both eyes" /
    # "standing" assumptions.  Never strip useful animal/creature/anime detail.
    if normalize_subject_type(subject_type) != 'human':
        return False
    text = prompt or ''
    return (
        framing == 'face' and _krea_true_profile_view(text)
    ) or (
        framing == 'body' and bool(_KREA_NON_STANDING_BODY_POSE.search(text))
    )


def _krea_true_profile_view(text: str) -> bool:
    """A real side profile, never a three-quarter view with both eyes visible."""
    return (bool(_KREA_PROFILE_VIEW.search(text or ''))
            and not bool(_KREA_THREE_QUARTER_VIEW.search(text or '')))


def _krea_builtin_card_prompt(prompt: str, label: str, subject_type: str) -> str:
    """Return a bounded catalog card for a final Krea imperative, if exact.

    A regenerate can keep its old label while its prompt is edited, and imported
    shots use arbitrary user text.  Neither is allowed after identity/SFW locks.
    Only the unmodified human catalog card is safe to repeat at the tail.
    """
    if normalize_subject_type(subject_type) != 'human':
        return ''
    catalog_prompt = prompt_by_label(label or '')
    if not catalog_prompt or not isinstance(prompt, str):
        return ''
    if prompt.strip() != catalog_prompt.strip():
        return ''
    # Keep the final catalog instruction positive without consulting an editable
    # setting.  The normal description may still use the selected outfit palette,
    # but this tail is deliberately static and safe to place after identity/SFW.
    card = (catalog_prompt
            .replace(OUTFIT_VARY, _KREA_CATALOG_OUTFIT)
            .replace(EXPRESSION_NEUTRAL, _KREA_CATALOG_EXPRESSION))
    return card if len(card) <= _KREA_TRUSTED_TAIL_MAX_CHARS else ''


def _compose_edit_prompt(prompt: str, *, nsfw: bool, framing, suffix: str,
                          subject_type: str, label: str,
                          concrete_outfit: bool, markings_lock: bool,
                          card_pose_priority: bool = False) -> str:
    """The ONE assembly of a local-edit prompt, shared by Klein and Krea — the two
    wrappers had drifted into two copies of the same six-part concatenation.  The
    engine-specific flags (name a concrete garment / hold the skin / prioritize a
    catalog pose) keep one engine's correction from silently moving the other.

    Klein's order remains load-bearing and unchanged: command + description,
    framing detail, markings hold, identity lock, rendering tail.  On Krea only,
    a profile or non-standing body card suppresses the contradictory generic
    detail and finishes with a card-priority imperative.  Every other Krea call
    keeps the historical assembly.  Every part is read through
    get_identity_prompt, so the normal path remains byte-identical to the
    hardcoded version — including the single spaces between parts, which is why
    the markings lock is re-joined with an explicit space instead of relying on
    the trailing one its constant carries (a textarea cannot hold a trailing
    space the user can see)."""
    st = normalize_subject_type(subject_type)
    noun = _KLEIN_SUBJECT_NOUN.get(st, 'subject')
    detail = (get_identity_prompt(f'framing_{framing}', st)
              if framing in PROMPT_FRAMINGS else '').strip()
    medium = _KLEIN_MEDIUM.get(st, _KLEIN_MEDIUM_DEFAULT)
    ending = get_identity_prompt('render_tail_nsfw' if nsfw else 'render_tail_sfw', st)
    body = apply_directive_overrides(_append_suffix(prompt, suffix))
    if concrete_outfit:
        # AFTER the overrides, never before: the substitution keys on the SHIPPED
        # outfit directive, so running it first consumed that text and the user's
        # own directive silently never landed. This order means "a concrete
        # garment replaces the default negation, and yields to anything you wrote
        # yourself" — the only reading under which the Settings box does what it
        # says on the two engines that name a garment.
        body = krea_outfit_directive(body, label)
    # The final Krea-only imperative never repeats free-form prompt/suffix text
    # after the identity and rendering locks.  A built-in, unchanged card may be
    # repeated because it is a short server-owned instruction; edited/custom
    # cards get the bounded generic imperative instead.
    prioritize_card = card_pose_priority and _krea_card_pose_needs_priority(
        prompt, framing, st)
    true_profile = (prioritize_card and framing == 'face'
                    and _krea_true_profile_view(prompt))
    final_card = (_krea_builtin_card_prompt(prompt, label, st)
                  if prioritize_card else '')
    if prioritize_card:
        # A user override of the generic framing text must not revive the same
        # contradiction.  The card itself remains the sole composition source.
        detail = ''
    lock = get_identity_prompt('markings_lock', st).strip() if markings_lock else ''
    return (
        f"Create a new {medium} of the same {noun} as the reference image: {body}. "
        + (f"{detail} " if detail else "")
        + (f"{lock} " if lock else "")
        + f"{get_identity_prompt('klein_identity', st)} {ending}"
        + (f" {krea_card_pose_priority(final_card, true_profile=true_profile)}"
           if prioritize_card else ""))


def _e(i, axis, framing, label, prompt, co=False, cb=False, aspect=None):
    return {'id': i, 'axis': axis, 'framing': framing, 'label': label,
            'prompt': prompt, 'changes_outfit': co, 'changes_bg': cb, 'aspect': aspect}


def _augment_prompt(entry, *, allow_outfit=True):
    """Bake the default outfit-variation + neutral-expression directives into an
    entry's prompt when it does not already specify them (see OUTFIT_VARY /
    EXPRESSION_NEUTRAL). Skips the outfit clause when framing='back' would still get
    it? No — outfit is visible from behind, so back shots DO get the outfit clause;
    only the expression clause is skipped for 'back' (no face). `allow_outfit=False`
    (NSFW nude/lingerie states) skips the outfit clause entirely: the described state
    of (un)dress IS the intent, not a leak — injecting a 'casual outfit' would fight it."""
    p = entry['prompt']
    add = []
    if allow_outfit and not _HAS_OUTFIT.search(p):
        add.append(OUTFIT_VARY)
    if entry['framing'] != 'back' and not _HAS_EXPRESSION.search(p):
        add.append(EXPRESSION_NEUTRAL)
    if add:
        entry['prompt'] = p + ', ' + ', '.join(add)
    return entry


VARIATION_CATALOG = [
    _e('face_front_neutral', 'expression', 'face', 'Face front, neutral',
       'close-up portrait, front view, neutral expression, soft light, plain neutral background', cb=True),
    _e('face_front_smile', 'expression', 'face', 'Face front, smile',
       'close-up portrait, front view, slight smile, soft window light, blurred home interior background', cb=True),
    _e('face_34l_smile', 'angle', 'face', 'Face 3/4 left, smile',
       'close-up portrait, three-quarter left view, smiling'),
    _e('face_34l_serious', 'angle', 'face', 'Face 3/4 left, serious',
       'close-up portrait, three-quarter left view, serious expression'),
    _e('face_34r_laugh', 'angle', 'face', 'Face 3/4 right, laugh',
       'close-up portrait, three-quarter right view, laughing'),
    _e('face_34r_soft', 'angle', 'face', 'Face 3/4 right, gentle',
       'close-up portrait, three-quarter right view, gentle expression'),
    _e('face_profile_l', 'angle', 'face', 'Profile left',
       'close-up portrait, left profile view, neutral'),
    _e('face_profile_r', 'angle', 'face', 'Profile right',
       'close-up portrait, right profile view, neutral'),
    _e('face_profile_l_smile', 'angle', 'face', 'Profile left, smile',
       'close-up portrait, strict left profile view, slight smile, soft window light, blurred background', cb=True),
    _e('face_profile_r_smile', 'angle', 'face', 'Profile right, smile',
       'close-up portrait, strict right profile view, slight smile, soft window light, blurred background', cb=True),
    _e('face_profile_l_serious', 'angle', 'face', 'Profile left, serious',
       'close-up portrait, strict left profile view, serious expression, even studio light, plain background', cb=True),
    _e('face_profile_r_serious', 'angle', 'face', 'Profile right, serious',
       'close-up portrait, strict right profile view, serious expression, even studio light, plain background', cb=True),
    _e('face_profile_l_look_up', 'angle', 'face', 'Profile left, looking up',
       'close-up portrait, strict left profile view, head tilted slightly upward, eyes looking up, pensive expression, soft daylight, blurred outdoor background', cb=True),
    _e('face_profile_r_look_up', 'angle', 'face', 'Profile right, looking up',
       'close-up portrait, strict right profile view, head tilted slightly upward, eyes looking up, pensive expression, soft daylight, blurred outdoor background', cb=True),
    _e('face_profile_l_rim_light', 'lighting', 'face', 'Profile left, rim light',
       'close-up portrait, strict left profile view, neutral expression, cinematic rim light, dark blurred background', cb=True),
    _e('face_profile_r_rim_light', 'lighting', 'face', 'Profile right, rim light',
       'close-up portrait, strict right profile view, neutral expression, cinematic rim light, dark blurred background', cb=True),
    _e('face_window', 'lighting', 'face', 'Face, window light',
       'close-up portrait, front view, soft window light, blurred background', cb=True),
    _e('face_studio', 'lighting', 'face', 'Face, studio',
       'close-up portrait, studio lighting, plain background', cb=True),
    _e('face_golden', 'lighting', 'face', 'Face, golden hour',
       'close-up portrait, three-quarter view, warm golden hour light, outdoor', cb=True),
    _e('face_surprise', 'expression', 'face', 'Face, surprise',
       'close-up portrait, front view, surprised expression'),
    _e('face_look_up', 'angle', 'face', 'Face, looking up',
       'close-up portrait, looking slightly upward, soft daylight, outdoor blurred background', cb=True),
    _e('face_look_down', 'angle', 'face', 'Face, looking down',
       'close-up portrait, looking slightly downward, pensive, indoor blurred background', cb=True),
    _e('bust_front', 'framing', 'bust', 'Bust, front',
       'upper body portrait, front view, neutral, wearing a casual top different from the reference outfit',
       co=True, cb=True),
    _e('bust_34', 'framing', 'bust', 'Bust, three-quarter',
       'upper body portrait, three-quarter view, smiling, different outfit, indoor', co=True, cb=True),
    _e('bust_outdoor', 'background', 'bust', 'Bust, outdoor',
       'upper body portrait, front view, outdoor park background', cb=True),
    _e('bust_studio', 'background', 'bust', 'Bust, studio',
       'upper body portrait, three-quarter view, studio backdrop', cb=True),
    _e('bust_jacket', 'outfit', 'bust', 'Bust, jacket',
       'upper body portrait, wearing a jacket different from the reference outfit, urban background',
       co=True, cb=True),
    _e('bust_evening', 'outfit', 'bust', 'Bust, evening outfit',
       'upper body portrait, elegant evening look, different from the reference outfit, dim ambient light',
       co=True, cb=True),
    _e('body_stand_front', 'framing', 'body', 'Body standing, front',
       'full body shot, standing, front view, casual clothes different from the reference outfit, street',
       co=True, cb=True),
    _e('body_stand_34', 'framing', 'body', 'Body standing, three-quarter',
       'full body shot, standing, three-quarter view, different outfit, outdoor', co=True, cb=True),
    _e('body_sit', 'framing', 'body', 'Body sitting',
       'full body shot, sitting on a chair, relaxed, indoor', co=True, cb=True),
    _e('body_walk', 'framing', 'body', 'Body walking',
       'full body shot, walking, dynamic pose, city background', co=True, cb=True),
    _e('body_cafe', 'background', 'body', 'Body, café',
       'full body shot, standing in a cafe, warm light', co=True, cb=True),
    _e('body_beach', 'background', 'body', 'Body, beach (clothed)',
       'full body shot, standing on a beach, summer casual clothes different from the reference outfit, daylight',
       co=True, cb=True),
    _e('back_34', 'framing', 'back', 'Back, three-quarter',
       'full body shot, three-quarter back view, showing hairstyle and silhouette', co=True, cb=True),
    _e('body_wide_env', 'framing', 'body', 'Body, wide urban shot',
       'full body shot, wide environmental framing, subject off-center, lots of background, urban plaza',
       co=True, cb=True, aspect='16:9'),
    _e('body_walk_wide', 'framing', 'body', 'Body walking, wide shot',
       'full body shot, walking across a wide street, dynamic, cinematic wide framing',
       co=True, cb=True, aspect='16:9'),
    _e('body_land_outdoor', 'framing', 'body', 'Body, outdoor landscape',
       'full body shot, standing outdoors, wide natural landscape background, daylight',
       co=True, cb=True, aspect='4:3'),
    _e('body_sit_terrace', 'framing', 'body', 'Body sitting, wide terrace',
       'full body shot, sitting on a cafe terrace, wide framing, warm light',
       co=True, cb=True, aspect='4:3'),
    _e('body_field_wide', 'framing', 'body', 'Body, wide open field',
       'full body shot, standing in an open field, wide nature background, soft daylight',
       co=True, cb=True, aspect='16:9'),
    _e('bust_land', 'framing', 'bust', 'Bust, landscape framing',
       'upper body portrait, landscape framing, environment visible on the sides, outdoor',
       cb=True, aspect='4:3'),
    # --- Body emphasis (fidélité corps) : silhouette RÉELLEMENT visible mais dans
    # le registre AUTORISÉ des moteurs API (vêtements ajustés, maillot de bain en
    # contexte plage/piscine, tenue de sport, robe moulante, contre-jour). Pas de
    # contournement de filtre : pour du contenu explicite → Klein en local.
    _e('bust_fitted_top', 'outfit', 'bust', 'Bust, fitted top',
       'upper body portrait, fitted ribbed knit top, natural relaxed pose, soft indoor light',
       co=True, cb=True),
    _e('bust_summer_dress', 'outfit', 'bust', 'Bust, summer dress',
       'upper body portrait, fitted summer dress with thin straps, golden hour light, outdoor',
       co=True, cb=True),
    _e('bust_swim', 'outfit', 'bust', 'Bust, swimsuit (beach)',
       'upper body portrait, wearing a bikini top, sunny beach in the background, bright '
       'daylight, natural relaxed pose', co=True, cb=True),
    _e('body_bodycon', 'outfit', 'body', 'Body, bodycon dress',
       'full body shot, elegant fitted bodycon evening dress, standing, upscale hotel lobby, '
       'warm ambient light', co=True, cb=True),
    _e('body_athletic', 'outfit', 'body', 'Body, sportswear',
       'full body shot, athletic sportswear, fitted leggings and sports top, gym setting, '
       'confident stance', co=True, cb=True),
    _e('body_swim_beach', 'outfit', 'body', 'Body, bikini beach',
       'full body shot, wearing a bikini, standing on a sunny beach, natural relaxed pose, '
       'bright daylight', co=True, cb=True, aspect='3:4'),
    _e('body_swim_pool', 'outfit', 'body', 'Body, swimsuit pool',
       'full body shot, one-piece swimsuit, standing at the edge of a swimming pool, summer '
       'daylight', co=True, cb=True, aspect='3:4'),
    _e('body_jeans_fit', 'outfit', 'body', 'Body, fitted jeans',
       'full body shot, fitted high-waisted jeans and tucked-in top, urban street, daylight',
       co=True, cb=True),
    _e('body_silhouette', 'lighting', 'body', 'Body, backlit silhouette',
       'full body shot, backlit near a large window, figure outlined by rim light, elegant '
       'fitted dress, moody interior', co=True, cb=True),
    # Gros plans VISAGE en formats variés (preset visage-centré) : la robustesse de
    # format sur le visage lui-même, sans plan corps (corps reste générique).
    _e('face_land', 'framing', 'face', 'Face, landscape framing',
       'close-up portrait, three-quarter view, landscape framing, face to one side with environment, outdoor',
       cb=True, aspect='4:3'),
    _e('face_tall', 'framing', 'face', 'Face, tall framing',
       'close-up portrait, front view, tall vertical framing, head and shoulders, soft natural light',
       cb=True, aspect='9:16'),
    _e('face_wide', 'framing', 'face', 'Face, cinematic framing',
       'close-up portrait, wide cinematic framing, face off-center, blurred background',
       cb=True, aspect='16:9'),
]

# --- Catalogue NSFW (moteur Klein LOCAL uniquement) --------------------------
# Plans corps non censurés pour la fidélité corporelle : jamais envoyés aux
# moteurs API (route + service refusent), générés par le Klein local qui n'a pas
# de filtre. Le registre reste "état + pose + décor" (lingerie/topless/nu) — pas
# d'acte : c'est un dataset de PERSONNAGE, l'acte appartient au prompt d'usage.
# Le caption doit décrire l'état (nude/lingerie) pour qu'il reste promptable et
# ne se lie pas au trigger (principe d'inversion).
NSFW_VARIATION_CATALOG = [
    _e('nsfw_bust_lingerie', 'nsfw', 'bust', 'Bust, lingerie',
       'bust shot, wearing delicate lace lingerie, bedroom, soft window light',
       co=True, cb=True),
    _e('nsfw_bust_topless', 'nsfw', 'bust', 'Bust, topless',
       'bust shot, topless, bare chest, neutral indoor background, natural light',
       co=True, cb=True),
    _e('nsfw_bust_towel', 'nsfw', 'bust', 'Bust, towel',
       'bust shot, wrapped in a bath towel, bare shoulders, bathroom, soft light',
       co=True, cb=True),
    _e('nsfw_body_lingerie', 'nsfw', 'body', 'Body, lingerie standing',
       'full body shot, standing, matching lace lingerie set, bedroom interior, soft light',
       co=True, cb=True, aspect='3:4'),
    _e('nsfw_body_nude_stand', 'nsfw', 'body', 'Body, nude standing',
       'full body shot, standing fully nude, natural anatomy, relaxed pose, neutral studio '
       'background, soft even light', co=True, cb=True, aspect='3:4'),
    _e('nsfw_body_nude_34', 'nsfw', 'body', 'Body, nude three-quarter',
       'full body shot, three-quarter view, fully nude, natural anatomy, standing by a large '
       'window, soft daylight', co=True, cb=True, aspect='3:4'),
    _e('nsfw_body_nude_sit', 'nsfw', 'body', 'Body, nude sitting on bed',
       'full body shot, sitting nude on the edge of a bed, relaxed natural pose, warm bedroom '
       'light', co=True, cb=True, aspect='3:4'),
    _e('nsfw_body_nude_lying', 'nsfw', 'body', 'Body, nude lying',
       'full body shot, lying nude on a bed on her side, natural anatomy, soft morning light',
       co=True, cb=True, aspect='4:3'),
    _e('nsfw_body_shower', 'nsfw', 'body', 'Body, nude shower',
       'full body shot, nude in the shower, wet skin and hair, water droplets, glass and tile '
       'background', co=True, cb=True, aspect='9:16'),
    _e('nsfw_back_nude', 'nsfw', 'back', 'Back, nude',
       'full body shot from behind, standing nude, back and buttocks visible, natural anatomy, '
       'neutral background', co=True, cb=True, aspect='3:4'),
]

# Bake the default outfit-variation / neutral-expression directives into every entry
# that doesn't already specify one (see _augment_prompt). Done in place, AFTER both
# catalogs are built, so `prompt_by_label` and the /variations route serve the fixed
# text and it lands in `variation_prompt` at generation time. NSFW entries keep their
# described state of (un)dress (allow_outfit=False) but still get a neutral expression.
for _entry in VARIATION_CATALOG:
    _augment_prompt(_entry)
for _entry in NSFW_VARIATION_CATALOG:
    _augment_prompt(_entry, allow_outfit=False)
del _entry

_NSFW_LABELS = {e['label'] for e in NSFW_VARIATION_CATALOG}


# --- Non-human catalogs (subject_type) ---------------------------------------
# These are authored COMPLETE: the outfit/expression augmentation above is a
# HUMAN concern (an animal wears no outfit, an object has no expression), so it is
# NEVER run on them. Every label is GLOBALLY UNIQUE across all catalogs so the
# by-label resolvers (prompt_by_label / aspect_for_label / is_nsfw_label) keep
# working on the union with no subject_type threading — enforced by
# test_labels_globally_unique. The `framing` stays the internal enum
# (face/bust/body/back) so composition, aspect and the stored column are shared;
# only the LABEL/PROMPT wording adapts (the UI relabels the group headers per
# type). First honest drafts, to be refined. No NSFW catalog for these types.
ANIMAL_CATALOG = [
    _e('animal_head_front', 'angle', 'face', 'Animal head, front',
       'close-up photo of the animal, head and shoulders, front view, looking at the camera, '
       'soft even light, plain neutral background', cb=True),
    _e('animal_head_34', 'angle', 'face', 'Animal head, three-quarter',
       'close-up photo of the animal, head, three-quarter view, soft daylight, blurred background', cb=True),
    _e('animal_head_profile_l', 'angle', 'face', 'Animal head, profile left',
       'close-up photo of the animal, head, left profile view, natural outdoor light', cb=True),
    _e('animal_head_profile_r', 'angle', 'face', 'Animal head, profile right',
       'close-up photo of the animal, head, right profile view, natural outdoor light', cb=True),
    _e('animal_head_up', 'angle', 'face', 'Animal head, looking up',
       'close-up photo of the animal, head, looking slightly upward, alert, outdoor blurred background', cb=True),
    _e('animal_head_studio', 'lighting', 'face', 'Animal head, studio',
       'close-up photo of the animal, head, studio lighting, plain seamless background', cb=True),
    _e('animal_head_land', 'framing', 'face', 'Animal head, landscape framing',
       'close-up photo of the animal, head, landscape framing, environment to one side, outdoor',
       cb=True, aspect='4:3'),
    _e('animal_half_front', 'framing', 'bust', 'Animal half-body, front',
       'half-body photo of the animal, front view, sitting, natural indoor light', cb=True),
    _e('animal_half_34', 'framing', 'bust', 'Animal half-body, three-quarter',
       'half-body photo of the animal, three-quarter view, outdoor park background', cb=True),
    _e('animal_body_stand_side', 'framing', 'body', 'Animal body, standing side',
       'full body photo of the animal, standing, side profile, the whole body from head to tail visible, '
       'natural setting', cb=True),
    _e('animal_body_stand_front', 'framing', 'body', 'Animal body, standing front',
       'full body photo of the animal, standing, front view, the entire body visible, plain background', cb=True),
    _e('animal_body_lying', 'framing', 'body', 'Animal body, lying down',
       'full body photo of the animal, lying down, relaxed, indoor floor, soft light', cb=True),
    _e('animal_body_walk', 'framing', 'body', 'Animal body, walking',
       'full body photo of the animal, walking, dynamic side view, outdoor natural ground',
       cb=True, aspect='16:9'),
    _e('animal_body_run', 'framing', 'body', 'Animal body, running',
       'full body photo of the animal, running, dynamic action, side view, outdoor field',
       cb=True, aspect='16:9'),
    _e('animal_body_outdoor', 'framing', 'body', 'Animal body, outdoor landscape',
       'full body photo of the animal, standing outdoors, wide natural landscape background, daylight',
       cb=True, aspect='4:3'),
    _e('animal_back', 'framing', 'back', 'Animal, from behind',
       'full body photo of the animal seen from behind, hindquarters and tail visible, natural setting', cb=True),
    # --- Head: remaining angles, states and light directions -------------------
    _e('animal_head_down', 'angle', 'face', 'Animal head, looking down',
       'close-up photo of the animal, head lowered, looking downward, calm, soft indoor light', cb=True),
    _e('animal_head_tilt', 'angle', 'face', 'Animal head, tilted',
       'close-up photo of the animal, head tilted to one side, curious, ears forward, soft daylight', cb=True),
    _e('animal_head_low', 'angle', 'face', 'Animal head, low angle',
       'close-up photo of the animal, head seen from below, low camera angle, imposing, plain background', cb=True),
    _e('animal_head_top', 'angle', 'face', 'Animal head, from above',
       'close-up photo of the animal, head seen from directly above, looking up at the camera, floor background', cb=True),
    _e('animal_head_open_mouth', 'expression', 'face', 'Animal head, mouth open',
       'close-up photo of the animal, head, mouth open, tongue and teeth visible, relaxed, outdoor daylight', cb=True),
    _e('animal_head_ears', 'expression', 'face', 'Animal head, ears alert',
       'close-up photo of the animal, head, ears raised and alert, attentive gaze, plain background', cb=True),
    _e('animal_head_eyes', 'framing', 'face', 'Animal eyes, extreme close-up',
       'extreme close-up photo of the animal, eyes and surrounding fur in sharp focus, shallow depth of field', cb=True),
    _e('animal_head_golden', 'lighting', 'face', 'Animal head, golden hour',
       'close-up photo of the animal, head, warm golden hour light from the side, outdoor, blurred background', cb=True),
    _e('animal_head_window', 'lighting', 'face', 'Animal head, window light',
       'close-up photo of the animal, head, soft window light indoors, blurred room background', cb=True),
    _e('animal_head_backlit', 'lighting', 'face', 'Animal head, backlit',
       'close-up photo of the animal, head, backlit with a bright rim of light around the fur, dark background', cb=True),
    _e('animal_head_night', 'lighting', 'face', 'Animal head, low light',
       'close-up photo of the animal, head, dim night light, dark background, eyes catching the light', cb=True),
    _e('animal_head_overcast', 'lighting', 'face', 'Animal head, overcast light',
       'close-up photo of the animal, head, flat overcast daylight, even and shadowless, outdoor', cb=True),
    _e('animal_head_wide', 'framing', 'face', 'Animal head, cinematic framing',
       'close-up photo of the animal, head off-center, wide cinematic framing, blurred environment',
       cb=True, aspect='16:9'),
    _e('animal_head_tall', 'framing', 'face', 'Animal head, tall framing',
       'close-up photo of the animal, head, tall vertical framing, plain background, soft light',
       cb=True, aspect='9:16'),
    _e('animal_detail_fur', 'framing', 'face', 'Animal detail, coat texture',
       'extreme close-up photo of the animal, fur or skin texture and its markings, sharp focus, soft even light', cb=True),
    _e('animal_detail_paw', 'framing', 'face', 'Animal detail, paw',
       'close-up photo of the animal, one paw or foot, sharp focus, natural ground, soft light', cb=True),
    _e('animal_detail_tail', 'framing', 'face', 'Animal detail, tail',
       'close-up photo of the animal, tail, sharp focus, natural setting, soft daylight', cb=True),
    # --- Half body -------------------------------------------------------------
    _e('animal_half_side', 'framing', 'bust', 'Animal half-body, side',
       'half-body photo of the animal, side view, head and chest visible, plain background, even light', cb=True),
    _e('animal_half_studio', 'lighting', 'bust', 'Animal half-body, studio',
       'half-body photo of the animal, studio lighting, plain seamless background', cb=True),
    _e('animal_half_golden', 'lighting', 'bust', 'Animal half-body, golden hour',
       'half-body photo of the animal, three-quarter view, warm golden hour light, outdoor', cb=True),
    _e('animal_half_lying', 'framing', 'bust', 'Animal half-body, resting',
       'half-body photo of the animal lying down with its head raised, front legs visible, indoor floor, soft light', cb=True),
    _e('animal_half_look_up', 'angle', 'bust', 'Animal half-body, looking up',
       'half-body photo of the animal sitting and looking upward, alert, blurred outdoor background', cb=True),
    # --- Full body: poses ------------------------------------------------------
    _e('animal_body_stand_34', 'framing', 'body', 'Animal body, standing three-quarter',
       'full body photo of the animal, standing, three-quarter view, the whole body visible, natural setting', cb=True),
    _e('animal_body_sit_front', 'framing', 'body', 'Animal body, sitting front',
       'full body photo of the animal, sitting, front view, the entire body visible, plain background', cb=True),
    _e('animal_body_sit_side', 'framing', 'body', 'Animal body, sitting side',
       'full body photo of the animal, sitting, side profile, the whole body visible, natural ground', cb=True),
    _e('animal_body_jump', 'framing', 'body', 'Animal body, jumping',
       'full body photo of the animal mid-jump, all four legs off the ground, dynamic action, outdoor',
       cb=True, aspect='16:9'),
    _e('animal_body_play', 'framing', 'body', 'Animal body, playing',
       'full body photo of the animal playing, lively pose, outdoor grass, daylight', cb=True),
    _e('animal_body_sleep', 'framing', 'body', 'Animal body, sleeping',
       'full body photo of the animal curled up asleep, eyes closed, soft indoor light, blanket or floor', cb=True),
    _e('animal_body_drink', 'framing', 'body', 'Animal body, drinking',
       'full body photo of the animal drinking, head lowered to a bowl or water, side view, natural light', cb=True),
    _e('animal_body_stretch', 'framing', 'body', 'Animal body, stretching',
       'full body photo of the animal stretching, elongated pose, side view, indoor floor, soft light', cb=True),
    # --- Full body: settings and light ----------------------------------------
    _e('animal_body_studio', 'lighting', 'body', 'Animal body, studio',
       'full body photo of the animal, standing, studio lighting, plain seamless background, the whole body visible', cb=True),
    _e('animal_body_golden', 'lighting', 'body', 'Animal body, golden hour',
       'full body photo of the animal standing outdoors, warm golden hour backlight, long shadows',
       cb=True, aspect='4:3'),
    _e('animal_body_night', 'lighting', 'body', 'Animal body, night',
       'full body photo of the animal outdoors at night, cool dim light, dark background', cb=True),
    _e('animal_body_snow', 'background', 'body', 'Animal body, snow',
       'full body photo of the animal walking in snow, white winter landscape, cold daylight',
       cb=True, aspect='16:9'),
    _e('animal_body_water', 'background', 'body', 'Animal body, water',
       'full body photo of the animal at the edge of water, wet ground reflecting the light, outdoor daylight',
       cb=True, aspect='4:3'),
    _e('animal_body_forest', 'background', 'body', 'Animal body, forest',
       'full body photo of the animal in a forest, trees and undergrowth around it, dappled light',
       cb=True, aspect='4:3'),
    _e('animal_body_field', 'background', 'body', 'Animal body, tall grass',
       'full body photo of the animal standing in tall grass, meadow, warm daylight', cb=True, aspect='4:3'),
    _e('animal_body_urban', 'background', 'body', 'Animal body, city street',
       'full body photo of the animal on a city pavement, urban background, daylight',
       cb=True, aspect='16:9'),
    _e('animal_body_indoor', 'background', 'body', 'Animal body, indoors',
       'full body photo of the animal on a sofa or indoor floor, home interior, warm lamp light', cb=True),
    _e('animal_body_low', 'angle', 'body', 'Animal body, low angle',
       'full body photo of the animal from a very low camera angle at ground level, the whole body visible, outdoor', cb=True),
    _e('animal_body_top', 'angle', 'body', 'Animal body, from above',
       'full body photo of the animal from directly above, top-down view, floor or ground background', cb=True),
    # --- From behind -----------------------------------------------------------
    _e('animal_back_34', 'framing', 'back', 'Animal, rear three-quarter',
       'full body photo of the animal from a rear three-quarter angle, back and one flank visible, natural setting', cb=True),
    _e('animal_back_walk', 'framing', 'back', 'Animal, walking away',
       'full body photo of the animal walking away from the camera, seen from behind, outdoor path',
       cb=True, aspect='16:9'),
]
CREATURE_CATALOG = [
    _e('creature_face_front', 'expression', 'face', 'Creature face, front',
       'close-up portrait of the creature, front view, neutral expression, soft light, plain background', cb=True),
    _e('creature_face_34l', 'angle', 'face', 'Creature face, three-quarter left',
       'close-up portrait of the creature, three-quarter left view, natural light'),
    _e('creature_face_34r', 'angle', 'face', 'Creature face, three-quarter right',
       'close-up portrait of the creature, three-quarter right view, natural light'),
    _e('creature_face_profile', 'angle', 'face', 'Creature face, profile',
       'close-up portrait of the creature, profile view, dramatic side light', cb=True),
    _e('creature_face_wide', 'framing', 'face', 'Creature face, cinematic framing',
       'close-up of the creature, wide cinematic framing, face off-center, blurred background',
       cb=True, aspect='16:9'),
    _e('creature_bust_front', 'framing', 'bust', 'Creature bust, front',
       'upper body shot of the creature, front view, plain background', cb=True),
    _e('creature_bust_34', 'framing', 'bust', 'Creature bust, three-quarter',
       'upper body shot of the creature, three-quarter view, environment background', cb=True),
    _e('creature_body_stand', 'framing', 'body', 'Creature body, standing front',
       'full body shot of the creature, standing, front view, the entire body visible, neutral setting', cb=True),
    _e('creature_body_34', 'framing', 'body', 'Creature body, standing three-quarter',
       'full body shot of the creature, standing, three-quarter view, natural environment', cb=True),
    _e('creature_body_action', 'framing', 'body', 'Creature body, action pose',
       'full body shot of the creature, dynamic action pose, dramatic environment',
       cb=True, aspect='16:9'),
    _e('creature_body_outdoor', 'framing', 'body', 'Creature body, outdoor landscape',
       'full body shot of the creature, standing outdoors, wide landscape background, daylight',
       cb=True, aspect='4:3'),
    _e('creature_back', 'framing', 'back', 'Creature, from behind',
       'full body shot of the creature seen from behind, back silhouette and back-facing features visible', cb=True),
    # --- Face: states, angles, light -------------------------------------------
    _e('creature_face_menacing', 'expression', 'face', 'Creature face, menacing',
       'close-up portrait of the creature, front view, menacing expression, hard directional light, dark background', cb=True),
    _e('creature_face_calm', 'expression', 'face', 'Creature face, calm',
       'close-up portrait of the creature, three-quarter view, calm and still expression, soft daylight', cb=True),
    _e('creature_face_roar', 'expression', 'face', 'Creature face, roaring',
       'close-up portrait of the creature, mouth wide open roaring, teeth visible, dramatic light', cb=True),
    _e('creature_face_low', 'angle', 'face', 'Creature face, low angle',
       'close-up of the creature seen from below, low heroic camera angle, sky or ceiling behind', cb=True),
    _e('creature_face_top', 'angle', 'face', 'Creature face, high angle',
       'close-up of the creature seen from above, high camera angle looking down at it, ground background', cb=True),
    _e('creature_face_rim', 'lighting', 'face', 'Creature face, rim light',
       'close-up portrait of the creature, cinematic rim light outlining its silhouette, dark background', cb=True),
    _e('creature_face_fire', 'lighting', 'face', 'Creature face, firelight',
       'close-up portrait of the creature lit by warm flickering firelight from below, deep shadows', cb=True),
    _e('creature_face_mist', 'lighting', 'face', 'Creature face, mist',
       'close-up portrait of the creature in cold mist, diffuse light, pale desaturated background', cb=True),
    _e('creature_face_eyes', 'framing', 'face', 'Creature eyes, extreme close-up',
       'extreme close-up of the creature, eyes and the surrounding skin in sharp focus, shallow depth of field', cb=True),
    _e('creature_detail_skin', 'framing', 'face', 'Creature detail, skin texture',
       'extreme close-up of the creature, skin, scale or fur texture and its markings, sharp focus, soft light', cb=True),
    _e('creature_detail_hand', 'framing', 'face', 'Creature detail, hand or claw',
       'close-up of the creature, one hand, claw or limb extremity, sharp focus, neutral background', cb=True),
    _e('creature_face_tall', 'framing', 'face', 'Creature face, tall framing',
       'close-up portrait of the creature, tall vertical framing, head and shoulders, plain background',
       cb=True, aspect='9:16'),
    # --- Bust ------------------------------------------------------------------
    _e('creature_bust_side', 'framing', 'bust', 'Creature bust, side',
       'upper body shot of the creature, side view, plain background, even light', cb=True),
    _e('creature_bust_studio', 'lighting', 'bust', 'Creature bust, studio',
       'upper body shot of the creature, studio lighting, plain seamless background', cb=True),
    _e('creature_bust_dark', 'lighting', 'bust', 'Creature bust, low key',
       'upper body shot of the creature, low key lighting, most of the frame in shadow, single light source', cb=True),
    _e('creature_bust_gear', 'outfit', 'bust', 'Creature bust, gear visible',
       'upper body shot of the creature, its armour, harness or gear clearly visible, three-quarter view', cb=True),
    # --- Full body: poses ------------------------------------------------------
    _e('creature_body_walk', 'framing', 'body', 'Creature body, walking',
       'full body shot of the creature walking, side view, the entire body visible, natural environment',
       cb=True, aspect='16:9'),
    _e('creature_body_crouch', 'framing', 'body', 'Creature body, crouching',
       'full body shot of the creature crouched low, coiled and ready to move, the whole body visible', cb=True),
    _e('creature_body_leap', 'framing', 'body', 'Creature body, leaping',
       'full body shot of the creature mid-leap, airborne, dynamic action, dramatic environment',
       cb=True, aspect='16:9'),
    _e('creature_body_rest', 'framing', 'body', 'Creature body, resting',
       'full body shot of the creature at rest, seated or lying, calm, the entire body visible', cb=True),
    _e('creature_body_scale', 'framing', 'body', 'Creature body, size reference',
       'full body shot of the creature next to an ordinary doorway or vehicle that gives its true scale, wide shot',
       cb=True, aspect='16:9'),
    # --- Full body: settings and light -----------------------------------------
    _e('creature_body_studio', 'lighting', 'body', 'Creature body, studio',
       'full body shot of the creature, standing, studio lighting, plain seamless background, the whole body visible', cb=True),
    _e('creature_body_ruins', 'background', 'body', 'Creature body, ruins',
       'full body shot of the creature among stone ruins, overgrown architecture, moody daylight',
       cb=True, aspect='4:3'),
    _e('creature_body_forest', 'background', 'body', 'Creature body, forest',
       'full body shot of the creature in a dense forest, dappled light through the trees', cb=True, aspect='4:3'),
    _e('creature_body_night', 'lighting', 'body', 'Creature body, night',
       'full body shot of the creature at night, cool moonlight, dark environment, the whole body still readable', cb=True),
    _e('creature_body_low', 'angle', 'body', 'Creature body, low angle',
       'full body shot of the creature from a low camera angle at ground level, towering over the viewer', cb=True),
    # --- From behind -----------------------------------------------------------
    _e('creature_back_34', 'framing', 'back', 'Creature, rear three-quarter',
       'full body shot of the creature from a rear three-quarter angle, back and one side visible, natural setting', cb=True),
    _e('creature_back_walk', 'framing', 'back', 'Creature, walking away',
       'full body shot of the creature walking away from the camera, seen from behind, wide environment',
       cb=True, aspect='16:9'),
]
OBJECT_CATALOG = [
    _e('object_full_front', 'framing', 'body', 'Object, full front',
       'product photo of the object, front view, the entire object in frame, plain seamless background, '
       'even studio light', cb=True),
    _e('object_full_34', 'framing', 'body', 'Object, full three-quarter',
       'product photo of the object, three-quarter angle, the full object visible, soft studio light, '
       'plain background', cb=True),
    _e('object_full_side', 'framing', 'body', 'Object, full side',
       'product photo of the object, side view, full profile, plain background, even light', cb=True),
    _e('object_context', 'framing', 'body', 'Object, in context',
       'photo of the object in a natural setting, realistic environment, soft daylight', cb=True),
    _e('object_outdoor', 'framing', 'body', 'Object, outdoor',
       'photo of the object outdoors, natural daylight, simple background', cb=True, aspect='4:3'),
    _e('object_hero', 'lighting', 'body', 'Object, studio hero',
       'product hero photo of the object, three-quarter angle, gradient studio background, dramatic light',
       cb=True, aspect='4:3'),
    _e('object_detail', 'framing', 'face', 'Object, detail',
       'close-up detail photo of the object, showing texture and material, sharp focus, soft light', cb=True),
    _e('object_detail_marking', 'framing', 'face', 'Object, detail marking',
       'close-up detail photo of the object focusing on its logo, text or distinctive marking, sharp focus', cb=True),
    _e('object_top', 'angle', 'bust', 'Object, top-down',
       'product photo of the object from a top-down angle, plain background, even light', cb=True),
    _e('object_low', 'angle', 'bust', 'Object, low angle',
       'product photo of the object from a low angle, plain background, studio light', cb=True),
    _e('object_hand', 'framing', 'bust', 'Object, in hand',
       'photo of the object held in a hand for scale, neutral background, soft light', cb=True),
    _e('object_back', 'angle', 'back', 'Object, rear view',
       'product photo of the object from behind, rear view, plain background, even light', cb=True),
    # --- Angles ----------------------------------------------------------------
    _e('object_full_45', 'angle', 'body', 'Object, elevated three-quarter',
       'product photo of the object from an elevated three-quarter angle, the full object visible, plain background', cb=True),
    _e('object_bottom', 'angle', 'bust', 'Object, underside',
       'product photo of the underside of the object, base and markings visible, plain background, even light', cb=True),
    _e('object_back_34', 'angle', 'back', 'Object, rear three-quarter',
       'product photo of the object from a rear three-quarter angle, back and one side visible, plain background', cb=True),
    # --- Studio light ----------------------------------------------------------
    _e('object_studio_soft', 'lighting', 'body', 'Object, soft studio light',
       'product photo of the object, large softbox lighting, very soft shadows, white seamless background', cb=True),
    _e('object_studio_hard', 'lighting', 'body', 'Object, hard light',
       'product photo of the object, single hard light source, sharp defined shadow on the surface, plain background', cb=True),
    _e('object_backlit', 'lighting', 'body', 'Object, backlit',
       'product photo of the object backlit, bright rim along its edges, dark gradient background', cb=True),
    _e('object_night', 'lighting', 'body', 'Object, night lighting',
       'photo of the object at night, coloured ambient and neon light, dark surroundings', cb=True),
    # --- In the real world -----------------------------------------------------
    _e('object_table', 'background', 'body', 'Object, on a wooden table',
       'photo of the object resting on a wooden table, home interior, soft daylight from a window', cb=True),
    _e('object_floor', 'background', 'body', 'Object, on the ground',
       'photo of the object on the ground outdoors, natural surface, overcast daylight', cb=True),
    _e('object_shelf', 'background', 'body', 'Object, on a shelf',
       'photo of the object on a shelf among ordinary items, indoor ambient light', cb=True),
    _e('object_city', 'background', 'body', 'Object, city background',
       'photo of the object outdoors with a city street behind it, daylight, shallow depth of field',
       cb=True, aspect='16:9'),
    _e('object_in_use', 'framing', 'body', 'Object, in use',
       'photo of the object being used as intended, hands present but the object stays the focus, natural light', cb=True),
    _e('object_scale', 'framing', 'bust', 'Object, size reference',
       'photo of the object beside an everyday item that gives its true scale, plain background, even light', cb=True),
    _e('object_medium_34', 'framing', 'bust', 'Object, medium three-quarter',
       'medium photo of the object from a three-quarter angle, part of the surroundings visible, soft light', cb=True),
    # --- Details ---------------------------------------------------------------
    _e('object_detail_material', 'framing', 'face', 'Object, detail material',
       'extreme close-up photo of the object, its material and surface finish, sharp focus, raking light', cb=True),
    _e('object_detail_edge', 'framing', 'face', 'Object, detail seam',
       'close-up photo of the object focusing on a seam, joint or edge where two parts meet, sharp focus', cb=True),
    _e('object_detail_wear', 'framing', 'face', 'Object, detail wear',
       'close-up photo of the object showing wear, scratches or patina from use, sharp focus, soft light', cb=True),
    _e('object_detail_top', 'framing', 'face', 'Object, detail from above',
       'close-up photo of the top surface of the object seen from directly above, sharp focus, even light', cb=True),
]
OTHER_CATALOG = [
    _e('other_full_front', 'framing', 'body', 'Subject, full front',
       'photo of the subject, front view, the entire subject in frame, plain neutral background, even light', cb=True),
    _e('other_full_34', 'framing', 'body', 'Subject, full three-quarter',
       'photo of the subject, three-quarter angle, the full subject visible, soft light, plain background', cb=True),
    _e('other_full_side', 'framing', 'body', 'Subject, full side',
       'photo of the subject, side view, full profile, plain background', cb=True),
    _e('other_context', 'framing', 'body', 'Subject, in context',
       'photo of the subject in a natural setting, realistic environment, daylight', cb=True, aspect='4:3'),
    _e('other_outdoor', 'framing', 'body', 'Subject, outdoor',
       'photo of the subject outdoors, natural daylight, wide background', cb=True, aspect='16:9'),
    _e('other_detail', 'framing', 'face', 'Subject, detail',
       'close-up detail photo of the subject, showing texture and detail, sharp focus, soft light', cb=True),
    _e('other_detail_2', 'framing', 'face', 'Subject, detail angle',
       'close-up of the subject from a different angle, sharp focus, soft light', cb=True),
    _e('other_medium', 'framing', 'bust', 'Subject, medium',
       'medium shot of the subject, three-quarter angle, simple background, even light', cb=True),
    _e('other_top', 'angle', 'bust', 'Subject, elevated angle',
       'photo of the subject from an elevated angle, plain background', cb=True),
    _e('other_back', 'angle', 'back', 'Subject, rear view',
       'photo of the subject from behind, rear view, plain background', cb=True),
    _e('other_full_back34', 'angle', 'back', 'Subject, rear three-quarter',
       'photo of the subject from a rear three-quarter angle, back and one side visible, plain background', cb=True),
    _e('other_low', 'angle', 'bust', 'Subject, low angle',
       'photo of the subject from a low camera angle, plain background, even light', cb=True),
    _e('other_medium_side', 'framing', 'bust', 'Subject, medium side',
       'medium shot of the subject from the side, simple background, soft light', cb=True),
    _e('other_scale', 'framing', 'bust', 'Subject, size reference',
       'photo of the subject beside an everyday item that gives its true scale, plain background', cb=True),
    _e('other_studio', 'lighting', 'body', 'Subject, studio',
       'photo of the subject, studio lighting, plain seamless background, the entire subject visible', cb=True),
    _e('other_studio_hard', 'lighting', 'body', 'Subject, hard light',
       'photo of the subject, single hard light source, sharp defined shadows, plain background', cb=True),
    _e('other_backlit', 'lighting', 'body', 'Subject, backlit',
       'photo of the subject backlit, bright rim along its edges, dark gradient background', cb=True),
    _e('other_night', 'lighting', 'body', 'Subject, night',
       'photo of the subject at night, dim ambient light, dark surroundings', cb=True),
    _e('other_indoor', 'background', 'body', 'Subject, indoors',
       'photo of the subject in an ordinary interior, home or workshop background, warm ambient light', cb=True),
    _e('other_wide', 'framing', 'body', 'Subject, wide shot',
       'wide photo of the subject off-center with plenty of environment around it, cinematic framing',
       cb=True, aspect='16:9'),
    _e('other_detail_3', 'framing', 'face', 'Subject, detail material',
       'extreme close-up of the subject, its material and surface finish, sharp focus, raking light', cb=True),
    _e('other_detail_marking', 'framing', 'face', 'Subject, detail marking',
       'close-up of the subject focusing on a distinctive marking, text or feature, sharp focus', cb=True),
]

# --- Anime catalog ------------------------------------------------------------
# Authored from the framing vocabulary of the MEDIUM, not of photography. Anime
# and its booru tagging have their own shot language — bust-up, cowboy shot
# (knee-up), full body, character-sheet turnaround, expression sheet — and those
# are the crops a drawn character is actually published in, so they are the crops
# a LoRA should see. Nothing here says "photo", "lens", "depth of field" or
# "studio lighting": the identity lock spends its whole budget forbidding
# photorealism and a catalog that whispered "close-up photo of" would undo it.
#
# Two deliberate departures from the human catalog:
#   • the SIGNATURE OUTFIT is named as identity in almost every shot, where the
#     human catalog bakes OUTFIT_VARY ("a different casual outfit") into every
#     shot that names none. A real person's clothes must NOT bind to them; a
#     character's costume is half of what makes them recognisable. Two shots do
#     offer an alternate outfit — as an explicit, opt-in card, not the default.
#   • the CHARACTER SHEET shots (front / side / back on a plain background) have
#     no photographic equivalent. They are how character designs are published
#     and they give the LoRA clean, unambiguous silhouette coverage.
# `_augment_prompt` is NEVER run here (same rule as every non-human catalog).
ANIME_CATALOG = [
    # --- Face: angles ---------------------------------------------------------
    _e('anime_face_front', 'angle', 'face', 'Anime face, front',
       'close-up of the anime character, front view, neutral expression, clean line art, '
       'flat cel shading, plain background', cb=True),
    _e('anime_face_34l', 'angle', 'face', 'Anime face, three-quarter left',
       'close-up of the anime character, three-quarter left view, soft even colour, simple background', cb=True),
    _e('anime_face_34r', 'angle', 'face', 'Anime face, three-quarter right',
       'close-up of the anime character, three-quarter right view, soft even colour, simple background', cb=True),
    _e('anime_face_profile_l', 'angle', 'face', 'Anime face, profile left',
       'close-up of the anime character, left profile view, the hair silhouette clearly readable', cb=True),
    _e('anime_face_profile_r', 'angle', 'face', 'Anime face, profile right',
       'close-up of the anime character, right profile view, the hair silhouette clearly readable', cb=True),
    _e('anime_face_low', 'angle', 'face', 'Anime face, from below',
       'close-up of the anime character seen from below, low angle looking up at the face, sky behind', cb=True),
    _e('anime_face_high', 'angle', 'face', 'Anime face, from above',
       'close-up of the anime character seen from above, high angle looking down at the face', cb=True),
    _e('anime_face_tilt', 'angle', 'face', 'Anime face, head tilted',
       'close-up of the anime character, head tilted to one side, three-quarter view, plain background', cb=True),
    # --- Face: expressions (the anime expression sheet) -----------------------
    _e('anime_face_smile', 'expression', 'face', 'Anime face, smile',
       'close-up of the anime character, front view, warm smile, eyes slightly narrowed, simple background', cb=True),
    _e('anime_face_laugh', 'expression', 'face', 'Anime face, laughing',
       'close-up of the anime character, laughing with the mouth open and the eyes closed happily', cb=True),
    _e('anime_face_angry', 'expression', 'face', 'Anime face, angry',
       'close-up of the anime character, angry expression, brows drawn down, mouth set, tense', cb=True),
    _e('anime_face_surprised', 'expression', 'face', 'Anime face, surprised',
       'close-up of the anime character, surprised expression, eyes wide, mouth slightly open', cb=True),
    _e('anime_face_sad', 'expression', 'face', 'Anime face, sad',
       'close-up of the anime character, sad expression, eyes downcast, subdued colour', cb=True),
    _e('anime_face_blush', 'expression', 'face', 'Anime face, blushing',
       'close-up of the anime character, embarrassed, blush across the cheeks, looking away', cb=True),
    _e('anime_face_closed_eyes', 'expression', 'face', 'Anime face, eyes closed',
       'close-up of the anime character, eyes gently closed, calm and serene expression', cb=True),
    _e('anime_face_smirk', 'expression', 'face', 'Anime face, confident smirk',
       'close-up of the anime character, confident smirk, one brow raised, looking at the viewer', cb=True),
    # --- Face: detail and light ----------------------------------------------
    _e('anime_face_eyes', 'framing', 'face', 'Anime detail, eyes',
       'extreme close-up of the anime character, both eyes and the iris pattern drawn in full detail, '
       'highlights and eyelashes crisp', cb=True),
    _e('anime_face_hair', 'framing', 'face', 'Anime detail, hair',
       'close-up of the anime character, the hairstyle and its strands, colour and shape clearly readable, '
       'plain background', cb=True),
    _e('anime_face_accessory', 'framing', 'face', 'Anime detail, head accessory',
       'close-up of the anime character, the ribbon, hairpin, glasses or headwear of the signature outfit '
       'in full detail', cb=True),
    _e('anime_face_backlit', 'lighting', 'face', 'Anime face, rim light',
       'close-up of the anime character, strong rim light outlining the hair and shoulders, dark background', cb=True),
    _e('anime_face_sunset', 'lighting', 'face', 'Anime face, sunset light',
       'close-up of the anime character in warm orange sunset light, long shadows, golden sky behind', cb=True),
    _e('anime_face_night', 'lighting', 'face', 'Anime face, night lighting',
       'close-up of the anime character at night, cool blue and neon light on the face, dark background', cb=True),
    _e('anime_face_tall', 'framing', 'face', 'Anime face, tall framing',
       'close-up of the anime character, tall vertical framing, head and shoulders, plain background',
       cb=True, aspect='9:16'),
    _e('anime_face_wide', 'framing', 'face', 'Anime face, cinematic framing',
       'close-up of the anime character, wide cinematic framing, face off-centre, simple background',
       cb=True, aspect='16:9'),
    # --- Bust-up and cowboy shot ---------------------------------------------
    _e('anime_bust_front', 'framing', 'bust', 'Anime bust-up, front',
       'bust-up shot of the anime character from the chest up, front view, wearing the signature outfit, '
       'plain background', cb=True),
    _e('anime_bust_34', 'framing', 'bust', 'Anime bust-up, three-quarter',
       'bust-up shot of the anime character from the chest up, three-quarter view, wearing the signature '
       'outfit, simple background', cb=True),
    _e('anime_bust_side', 'framing', 'bust', 'Anime bust-up, side',
       'bust-up shot of the anime character from the side, wearing the signature outfit, plain background', cb=True),
    _e('anime_bust_arms_crossed', 'framing', 'bust', 'Anime bust-up, arms crossed',
       'upper body shot of the anime character with the arms crossed, confident, wearing the signature '
       'outfit, simple background', cb=True),
    _e('anime_bust_hand_face', 'framing', 'bust', 'Anime bust-up, hand near face',
       'upper body shot of the anime character with one hand raised near the face, fingers clearly drawn, '
       'wearing the signature outfit', cb=True),
    _e('anime_bust_collar', 'outfit', 'bust', 'Anime bust-up, outfit detail',
       'upper body shot of the anime character showing the collar, emblem and upper details of the '
       'signature outfit, three-quarter view, plain background', cb=True),
    _e('anime_cowboy_front', 'framing', 'bust', 'Anime cowboy shot, front',
       'cowboy shot of the anime character from the knees up, front view, standing, wearing the signature '
       'outfit, simple background', cb=True),
    _e('anime_cowboy_34', 'framing', 'bust', 'Anime cowboy shot, three-quarter',
       'cowboy shot of the anime character from the knees up, three-quarter view, relaxed stance, wearing '
       'the signature outfit', cb=True),
    _e('anime_bust_outdoor', 'background', 'bust', 'Anime bust-up, outdoors',
       'upper body shot of the anime character outdoors, wearing the signature outfit, sunny sky and '
       'simple scenery behind', cb=True),
    # --- Full body: character sheet ------------------------------------------
    _e('anime_sheet_front', 'framing', 'body', 'Anime character sheet, front',
       'full body character reference of the anime character, front view, standing straight, arms slightly '
       'away from the body, wearing the signature outfit, plain white background, even flat colour',
       cb=True, aspect='9:16'),
    _e('anime_sheet_side', 'framing', 'body', 'Anime character sheet, side',
       'full body character reference of the anime character, side view, standing straight, wearing the '
       'signature outfit, plain white background, even flat colour', cb=True, aspect='9:16'),
    # --- Full body: poses -----------------------------------------------------
    _e('anime_body_stand_front', 'framing', 'body', 'Anime full body, standing front',
       'full body shot of the anime character standing, front view, the entire body visible from head to '
       'shoes, wearing the signature outfit, simple background', cb=True),
    _e('anime_body_stand_34', 'framing', 'body', 'Anime full body, standing three-quarter',
       'full body shot of the anime character standing, three-quarter view, the whole body visible, '
       'wearing the signature outfit', cb=True),
    _e('anime_body_walk', 'framing', 'body', 'Anime full body, walking',
       'full body shot of the anime character walking, side view, mid stride, wearing the signature outfit', cb=True),
    _e('anime_body_run', 'framing', 'body', 'Anime full body, running',
       'full body shot of the anime character running, dynamic pose, hair and the signature outfit trailing '
       'with the motion', cb=True, aspect='16:9'),
    _e('anime_body_jump', 'framing', 'body', 'Anime full body, mid-air',
       'full body shot of the anime character mid-jump, airborne, dynamic action pose, wearing the '
       'signature outfit', cb=True, aspect='16:9'),
    _e('anime_body_action', 'framing', 'body', 'Anime full body, action pose',
       'full body shot of the anime character in a dramatic action pose, dynamic perspective, wearing the '
       'signature outfit', cb=True, aspect='16:9'),
    _e('anime_body_sit', 'framing', 'body', 'Anime full body, sitting',
       'full body shot of the anime character sitting, legs visible, relaxed, wearing the signature outfit, '
       'simple interior background', cb=True),
    _e('anime_body_lying', 'framing', 'body', 'Anime full body, lying down',
       'full body shot of the anime character lying down seen from above, the whole body visible, wearing '
       'the signature outfit', cb=True, aspect='16:9'),
    _e('anime_body_low', 'angle', 'body', 'Anime full body, low angle',
       'full body shot of the anime character from a low angle at ground level, towering perspective, '
       'wearing the signature outfit', cb=True),
    # --- Full body: settings and light ---------------------------------------
    _e('anime_body_city', 'background', 'body', 'Anime full body, city street',
       'full body shot of the anime character standing on a city street, buildings and signage behind, '
       'daylight, wearing the signature outfit', cb=True, aspect='4:3'),
    _e('anime_body_indoor', 'background', 'body', 'Anime full body, indoors',
       'full body shot of the anime character in an ordinary interior room, warm ambient colour, wearing '
       'the signature outfit', cb=True),
    _e('anime_body_nature', 'background', 'body', 'Anime full body, outdoors',
       'full body shot of the anime character outdoors among trees and grass, soft daylight, wearing the '
       'signature outfit', cb=True, aspect='4:3'),
    _e('anime_body_night_neon', 'lighting', 'body', 'Anime full body, night neon',
       'full body shot of the anime character at night, neon signs casting coloured light, dark street, '
       'wearing the signature outfit', cb=True, aspect='16:9'),
    # --- Alternate outfits: OPT-IN, never the default ------------------------
    _e('anime_alt_casual', 'outfit', 'body', 'Anime full body, alternate casual outfit',
       'full body shot of the anime character wearing an alternate casual everyday outfit instead of the '
       'signature one, the character design otherwise unchanged, standing, simple background', cb=True),
    _e('anime_alt_seasonal', 'outfit', 'body', 'Anime full body, alternate seasonal outfit',
       'full body shot of the anime character wearing an alternate seasonal outfit (coat and scarf, or '
       'summer clothes) instead of the signature one, the character design otherwise unchanged', cb=True),
    # --- From behind ----------------------------------------------------------
    _e('anime_back_stand', 'framing', 'back', 'Anime, from behind',
       'full body shot of the anime character seen from behind, standing, the back of the hair and of the '
       'signature outfit clearly visible, plain background', cb=True),
    _e('anime_back_34', 'framing', 'back', 'Anime, rear three-quarter',
       'full body shot of the anime character from a rear three-quarter angle, back and one side visible, '
       'wearing the signature outfit', cb=True),
    _e('anime_back_shoulder', 'framing', 'back', 'Anime, looking over the shoulder',
       'shot of the anime character from behind looking back over one shoulder at the viewer, the face '
       'partly visible, wearing the signature outfit', cb=True),
    _e('anime_back_walk', 'framing', 'back', 'Anime, walking away',
       'full body shot of the anime character walking away from the viewer, seen from behind, wide scenery',
       cb=True, aspect='16:9'),
    _e('anime_sheet_back', 'framing', 'back', 'Anime character sheet, back',
       'full body character reference of the anime character, back view, standing straight, wearing the '
       'signature outfit, plain white background, even flat colour', cb=True, aspect='9:16'),
]

# subject_type -> catalog / nsfw / by-id. 'human' reuses the existing objects so
# the human path is byte-identical. `_ALL_CATALOGS` is the union every by-label
# resolver searches (labels are globally unique across it).
_SUBJECT_CATALOGS = {
    'human': VARIATION_CATALOG, 'animal': ANIMAL_CATALOG, 'creature': CREATURE_CATALOG,
    'object': OBJECT_CATALOG, 'other': OTHER_CATALOG, 'anime': ANIME_CATALOG,
}
_SUBJECT_NSFW = {'human': NSFW_VARIATION_CATALOG}   # only human has an uncensored catalog
_ALL_CATALOGS = (VARIATION_CATALOG + NSFW_VARIATION_CATALOG + ANIMAL_CATALOG
                 + CREATURE_CATALOG + OBJECT_CATALOG + OTHER_CATALOG + ANIME_CATALOG)


# Legacy label aliases (old French persisted key -> current English catalog label).
# The catalog labels used to be French and are stored verbatim on every generated
# row (FaceDatasetImage.variation_label) AND inside dataset backups. Regeneration,
# the NSFW/Klein-only guard and the aspect-ratio resolver all look a stored label
# up against the live catalog, so translating the labels would ORPHAN every dataset
# created before the migration (wrong prompt fallback, NSFW shots leaking to API
# engines, lost aspect overrides). Every by-label lookup routes the incoming label
# through this map first, so old rows keep resolving exactly as they used to.
# One entry per translated label; keys are the pre-migration French strings, values
# must each be a current catalog label (guarded by test_legacy_aliases_resolve).
LEGACY_LABEL_ALIASES = {
    # Face
    'Visage face, neutre': 'Face front, neutral',
    'Visage face, sourire': 'Face front, smile',
    'Visage 3/4 gauche, sourire': 'Face 3/4 left, smile',
    'Visage 3/4 gauche, serieux': 'Face 3/4 left, serious',
    'Visage 3/4 droite, rire': 'Face 3/4 right, laugh',
    'Visage 3/4 droite, doux': 'Face 3/4 right, gentle',
    'Profil gauche': 'Profile left',
    'Profil droite': 'Profile right',
    'Profil gauche, sourire': 'Profile left, smile',
    'Profil droite, sourire': 'Profile right, smile',
    'Profil gauche, serieux': 'Profile left, serious',
    'Profil droite, serieux': 'Profile right, serious',
    'Profil gauche, regard haut': 'Profile left, looking up',
    'Profil droite, regard haut': 'Profile right, looking up',
    'Profil gauche, lumiere cinema': 'Profile left, rim light',
    'Profil droite, lumiere cinema': 'Profile right, rim light',
    'Visage, lumiere fenetre': 'Face, window light',
    'Visage, studio': 'Face, studio',
    'Visage, golden hour': 'Face, golden hour',
    'Visage, surprise': 'Face, surprise',
    'Visage, regard haut': 'Face, looking up',
    'Visage, regard bas': 'Face, looking down',
    # Bust
    'Buste face': 'Bust, front',
    'Buste 3/4': 'Bust, three-quarter',
    'Buste exterieur': 'Bust, outdoor',
    'Buste studio': 'Bust, studio',
    'Buste, veste': 'Bust, jacket',
    'Buste, tenue soiree': 'Bust, evening outfit',
    'Buste, cadre paysage': 'Bust, landscape framing',
    'Buste, haut ajusté': 'Bust, fitted top',
    "Buste, robe d'été": 'Bust, summer dress',
    'Buste, maillot (plage)': 'Bust, swimsuit (beach)',
    # Body
    'Corps debout face': 'Body standing, front',
    'Corps debout 3/4': 'Body standing, three-quarter',
    'Corps assis': 'Body sitting',
    'Corps en marche': 'Body walking',
    'Corps, cafe': 'Body, café',
    'Corps, plage (habille)': 'Body, beach (clothed)',
    'Corps, plan large urbain': 'Body, wide urban shot',
    'Corps en marche, large': 'Body walking, wide shot',
    'Corps, paysage exterieur': 'Body, outdoor landscape',
    'Corps assis, terrasse large': 'Body sitting, wide terrace',
    'Corps, champ large': 'Body, wide open field',
    'Corps, robe moulante': 'Body, bodycon dress',
    'Corps, tenue de sport': 'Body, sportswear',
    'Corps, bikini plage': 'Body, bikini beach',
    'Corps, maillot piscine': 'Body, swimsuit pool',
    'Corps, jean ajusté': 'Body, fitted jeans',
    'Corps, silhouette contre-jour': 'Body, backlit silhouette',
    # Back
    'Dos 3/4': 'Back, three-quarter',
    # Face formats
    'Visage, cadre paysage': 'Face, landscape framing',
    'Visage, cadre vertical': 'Face, tall framing',
    'Visage, cadre cinema': 'Face, cinematic framing',
    # NSFW catalog (local Klein only)
    'Buste, lingerie': 'Bust, lingerie',
    'Buste, topless': 'Bust, topless',
    'Buste, serviette': 'Bust, towel',
    'Corps, lingerie debout': 'Body, lingerie standing',
    'Corps, nu debout': 'Body, nude standing',
    'Corps, nu trois-quarts': 'Body, nude three-quarter',
    'Corps, nu assis lit': 'Body, nude sitting on bed',
    'Corps, nu allongé': 'Body, nude lying',
    'Corps, nu douche': 'Body, nude shower',
    'Dos, nu': 'Back, nude',
}


def canonical_label(label):
    """Resolve a stored variation label to its current catalog label. Pre-migration
    rows (and backups) persisted the French labels; those strings are still what the
    DB hands back on regeneration, so every by-label lookup passes through here first.
    Current English labels, 🔞 custom-prompt labels and empty/None all pass through
    unchanged (they are absent from LEGACY_LABEL_ALIASES)."""
    return LEGACY_LABEL_ALIASES.get(label, label)


def is_nsfw_label(label) -> bool:
    """True when a variation label belongs to the NSFW catalog or carries the 🔞
    custom-prompt prefix — drives the Klein-only guard and the NSFW wrapper on
    regeneration (the DB row only stores the label). A legacy French NSFW label is
    canonicalised first so pre-migration nude shots stay fail-closed on local Klein."""
    label = canonical_label(label)
    return bool(label) and (label in _NSFW_LABELS or label.startswith('🔞'))


# Préréglage face-heavy (deep-research 2026-06-14) : majorité de visages — c'est là
# que se joue la cohérence d'identité — et ≤4 plein-pied (le reste du catalogue
# body/cafe/beach reste sélectionnable manuellement). 14 visage / 6 buste / 4 corps / 1 dos.
_BALANCED_25 = [
    'face_front_neutral', 'face_front_smile', 'face_34l_smile', 'face_34l_serious',
    'face_34r_laugh', 'face_34r_soft', 'face_profile_l', 'face_profile_r',
    'face_window', 'face_studio', 'face_golden', 'face_surprise',
    'face_look_up', 'face_look_down',
    'bust_front', 'bust_34', 'bust_outdoor', 'bust_studio', 'bust_jacket', 'bust_evening',
    'body_stand_front', 'body_stand_34', 'body_sit', 'body_walk',
    'back_34',
]
_ZIMAGE_12 = [
    'face_front_neutral', 'face_front_smile', 'face_34l_smile', 'face_34r_laugh',
    'face_profile_l', 'face_window', 'face_golden', 'face_surprise',
    'bust_front', 'bust_34', 'body_stand_front', 'body_sit',
]
_BALANCED_MULTIFORMAT = _BALANCED_25 + [
    'body_wide_env', 'body_walk_wide', 'body_land_outdoor',
    'body_sit_terrace', 'body_field_wide', 'bust_land',
]
# Visage-centré : QUE du visage + buste, en formats variés, ZÉRO plan corps. Pour un
# LoRA où l'identité (visage) prime et où le corps doit rester générique/pilotable
# (ne pas l'entraîner = ne pas le graver). 17 visage / 7 buste, formats 1:1/3:4/4:3/9:16/16:9.
_FACE_FOCUSED = [
    'face_front_neutral', 'face_front_smile', 'face_34l_smile', 'face_34l_serious',
    'face_34r_laugh', 'face_34r_soft', 'face_profile_l', 'face_profile_r',
    'face_window', 'face_studio', 'face_golden', 'face_surprise',
    'face_look_up', 'face_look_down', 'face_land', 'face_tall', 'face_wide',
    'bust_front', 'bust_34', 'bust_outdoor', 'bust_studio', 'bust_jacket', 'bust_evening',
    'bust_land',
]
# Plein-pied fiable (deep-research 2026-06-16) : pour un LoRA qui doit rendre le
# CORPS de façon robuste (le perso casse en paysage/pied). On prend TOUT le catalogue
# corps (11) + dos, et un noyau visage/buste resserré pour rester ~50/50 — entraîner
# surtout sur des plans corps dégraderait le visage (identité qui dérive). ZÉRO
# nouvelle variation : tout est déjà dans le catalogue. 10 visage / 4 buste / 11 corps / 1 dos.
_FULLBODY_FOCUSED = [
    'face_front_neutral', 'face_front_smile', 'face_34l_smile', 'face_34r_laugh',
    'face_34r_soft', 'face_profile_l', 'face_window', 'face_golden', 'face_studio',
    'face_surprise',
    'bust_front', 'bust_34', 'bust_outdoor', 'bust_jacket',
    'body_stand_front', 'body_stand_34', 'body_sit', 'body_walk', 'body_cafe',
    'body_beach', 'body_wide_env', 'body_walk_wide', 'body_land_outdoor',
    'body_sit_terrace', 'body_field_wide',
    'back_34',
]
# Body-emphasis (fidélité corps, 25 = 8 visage / 8 buste / 8 corps / 1 dos — aligné
# sur la cible de composition body-fidelity 8/8/8/2, le dos se génère en x2) : les
# plans buste/corps privilégient les tenues qui MONTRENT la silhouette (ajusté,
# maillot, sport, moulant, contre-jour) tout en restant dans le registre accepté
# par les moteurs API. Le visage garde son noyau identité.
_BODY_EMPHASIS = [
    'face_front_neutral', 'face_front_smile', 'face_34l_smile', 'face_34r_laugh',
    'face_profile_l', 'face_window', 'face_golden', 'face_studio',
    'bust_front', 'bust_34', 'bust_fitted_top', 'bust_summer_dress', 'bust_swim',
    'bust_outdoor', 'bust_jacket', 'bust_evening',
    'body_stand_front', 'body_stand_34', 'body_bodycon', 'body_athletic',
    'body_swim_beach', 'body_swim_pool', 'body_jeans_fit', 'body_silhouette',
    'back_34',
]
_PRESETS = {'balanced_25': _BALANCED_25, 'zimage_12': _ZIMAGE_12,
            'balanced_multiformat': _BALANCED_MULTIFORMAT, 'face_focused': _FACE_FOCUSED,
            'fullbody_focused': _FULLBODY_FOCUSED, 'body_emphasis': _BODY_EMPHASIS}
_BY_ID = {e['id']: e for e in VARIATION_CATALOG}

# Non-human presets. The frontend renders these from `preset_meta_for` (returned
# by the /variations route) so it doesn't need to know their keys ahead of time;
# the human path keeps its own hardcoded PRESET_META untouched (so the human
# /variations response stays byte-identical).
#
# These used to be `[e['id'] for e in <CATALOG>]` — every shot of the type. That
# was defensible on a 12-shot first draft; on the catalogs below it would mean a
# single click queueing 59 images (and, on an API engine, billing them). So each
# preset is now CURATED: a deliberate composition, the way the human presets are.
# The KEYS are unchanged (`animal_balanced`…) — only their contents grew, which
# is the safe direction: preset keys are transient, but the shot IDS they list are
# persisted in the user's saved presets (`datasetCustomPresetsV1.selectedIds`), so
# no existing id is ever renamed or dropped.
_ANIMAL_BALANCED = [
    'animal_head_front', 'animal_head_34', 'animal_head_profile_l', 'animal_head_up',
    'animal_head_tilt', 'animal_head_studio', 'animal_head_golden', 'animal_detail_fur',
    'animal_half_front', 'animal_half_34', 'animal_half_side', 'animal_half_studio',
    'animal_half_look_up',
    'animal_body_stand_side', 'animal_body_stand_front', 'animal_body_stand_34',
    'animal_body_sit_front', 'animal_body_lying', 'animal_body_walk', 'animal_body_run',
    'animal_body_outdoor', 'animal_body_studio',
    'animal_back', 'animal_back_34',
]
_ANIMAL_HEAD = [
    'animal_head_front', 'animal_head_34', 'animal_head_profile_l', 'animal_head_profile_r',
    'animal_head_up', 'animal_head_down', 'animal_head_tilt', 'animal_head_studio',
    'animal_head_window', 'animal_head_golden', 'animal_head_backlit', 'animal_head_eyes',
]
_ANIMAL_FULLBODY = [
    'animal_body_stand_side', 'animal_body_stand_front', 'animal_body_stand_34',
    'animal_body_sit_front', 'animal_body_lying', 'animal_body_walk', 'animal_body_run',
    'animal_body_jump', 'animal_body_outdoor', 'animal_body_studio',
    'animal_back', 'animal_back_34',
]
_CREATURE_BALANCED = [
    'creature_face_front', 'creature_face_34l', 'creature_face_34r', 'creature_face_profile',
    'creature_face_calm', 'creature_detail_skin',
    'creature_bust_front', 'creature_bust_34', 'creature_bust_side', 'creature_bust_studio',
    'creature_body_stand', 'creature_body_34', 'creature_body_walk', 'creature_body_crouch',
    'creature_body_rest', 'creature_body_action', 'creature_body_outdoor', 'creature_body_studio',
    'creature_back', 'creature_back_34',
]
_CREATURE_FACE = [
    'creature_face_front', 'creature_face_34l', 'creature_face_34r', 'creature_face_profile',
    'creature_face_calm', 'creature_face_menacing', 'creature_face_roar', 'creature_face_rim',
    'creature_face_low', 'creature_face_eyes',
]
_CREATURE_FULLBODY = [
    'creature_body_stand', 'creature_body_34', 'creature_body_walk', 'creature_body_action',
    'creature_body_leap', 'creature_body_crouch', 'creature_body_rest', 'creature_body_outdoor',
    'creature_back', 'creature_back_34',
]
_OBJECT_BALANCED = [
    'object_full_front', 'object_full_34', 'object_full_side', 'object_full_45',
    'object_hero', 'object_context', 'object_outdoor', 'object_studio_soft',
    'object_top', 'object_low', 'object_hand', 'object_medium_34',
    'object_detail', 'object_detail_marking', 'object_detail_material', 'object_detail_edge',
    'object_back', 'object_back_34',
]
_OBJECT_STUDIO = [
    'object_full_front', 'object_full_34', 'object_full_side', 'object_full_45',
    'object_hero', 'object_studio_soft', 'object_studio_hard', 'object_backlit',
    'object_top', 'object_back',
]
_OBJECT_CONTEXT = [
    'object_context', 'object_outdoor', 'object_table', 'object_floor', 'object_shelf',
    'object_city', 'object_in_use', 'object_hand', 'object_scale', 'object_night',
]
_OTHER_BALANCED = [
    'other_full_front', 'other_full_34', 'other_full_side', 'other_context',
    'other_outdoor', 'other_studio', 'other_indoor',
    'other_medium', 'other_medium_side', 'other_top', 'other_low',
    'other_detail', 'other_detail_2', 'other_detail_3',
    'other_back', 'other_full_back34',
]
_OTHER_QUICK = [
    'other_full_front', 'other_full_34', 'other_full_side', 'other_medium',
    'other_detail', 'other_detail_2', 'other_context', 'other_back',
]
_ANIME_BALANCED = [
    'anime_face_front', 'anime_face_34l', 'anime_face_profile_l', 'anime_face_smile',
    'anime_face_angry', 'anime_face_eyes',
    'anime_bust_front', 'anime_bust_34', 'anime_bust_collar', 'anime_cowboy_front',
    'anime_sheet_front', 'anime_sheet_side', 'anime_body_stand_front', 'anime_body_stand_34',
    'anime_body_walk', 'anime_body_action', 'anime_body_sit', 'anime_body_city',
    'anime_back_stand', 'anime_sheet_back',
]
_ANIME_FACE = [
    'anime_face_front', 'anime_face_34l', 'anime_face_34r', 'anime_face_profile_l',
    'anime_face_smile', 'anime_face_laugh', 'anime_face_angry', 'anime_face_surprised',
    'anime_face_blush', 'anime_face_closed_eyes', 'anime_face_eyes', 'anime_face_hair',
]
_ANIME_FULLBODY = [
    'anime_sheet_front', 'anime_sheet_side', 'anime_sheet_back',
    'anime_body_stand_front', 'anime_body_stand_34', 'anime_body_walk', 'anime_body_run',
    'anime_body_action', 'anime_body_sit', 'anime_back_stand', 'anime_back_34',
]
# The character-sheet preset has no equivalent in any other subject type: three
# clean turnaround views plus the neutral crops, on a plain background. It is the
# cheapest set that teaches a LoRA a character's silhouette and outfit without
# any scenery to memorise alongside it.
_ANIME_SHEET = [
    'anime_sheet_front', 'anime_sheet_side', 'anime_sheet_back',
    'anime_face_front', 'anime_face_profile_l', 'anime_bust_front', 'anime_bust_side',
    'anime_face_hair', 'anime_face_accessory', 'anime_bust_collar',
]
_SUBJECT_PRESETS = {
    'human': _PRESETS,
    'animal': {'animal_balanced': _ANIMAL_BALANCED, 'animal_head_focused': _ANIMAL_HEAD,
               'animal_fullbody_focused': _ANIMAL_FULLBODY},
    'creature': {'creature_balanced': _CREATURE_BALANCED, 'creature_face_focused': _CREATURE_FACE,
                 'creature_fullbody_focused': _CREATURE_FULLBODY},
    'object': {'object_balanced': _OBJECT_BALANCED, 'object_studio': _OBJECT_STUDIO,
               'object_context': _OBJECT_CONTEXT},
    'other': {'other_balanced': _OTHER_BALANCED, 'other_quick': _OTHER_QUICK},
    'anime': {'anime_balanced': _ANIME_BALANCED, 'anime_face_focused': _ANIME_FACE,
              'anime_fullbody_focused': _ANIME_FULLBODY, 'anime_character_sheet': _ANIME_SHEET},
}
# Preset display metadata surfaced ONLY for non-human types (human uses the
# frontend's hardcoded PRESET_META, keeping the human payload byte-identical).
_SUBJECT_PRESET_META = {
    'animal': [{'key': 'animal_balanced', 'name': 'Balanced',
                'hint': 'A balanced spread of head, half-body, full-body and rear shots for an animal.'},
               {'key': 'animal_head_focused', 'name': 'Head focused',
                'hint': 'Head angles, expressions and light directions — for a face the model must nail.'},
               {'key': 'animal_fullbody_focused', 'name': 'Full body focused',
                'hint': 'Standing, sitting, moving and rear shots — for the whole silhouette and proportions.'}],
    'creature': [{'key': 'creature_balanced', 'name': 'Balanced',
                  'hint': 'Face, bust, full-body and rear shots for a creature or fictional character.'},
                 {'key': 'creature_face_focused', 'name': 'Face focused',
                  'hint': 'Angles, expressions and dramatic light on the face alone.'},
                 {'key': 'creature_fullbody_focused', 'name': 'Full body focused',
                  'hint': 'Poses, action and rear views — for anatomy and silhouette.'}],
    'object': [{'key': 'object_balanced', 'name': 'Balanced',
                'hint': 'Front, angle, detail and rear views for an object or product.'},
               {'key': 'object_studio', 'name': 'Studio',
                'hint': 'Clean product shots on a plain background, several angles and light setups.'},
               {'key': 'object_context', 'name': 'In context',
                'hint': 'The object in real settings and in use — teaches it apart from its backdrop.'}],
    'other': [{'key': 'other_balanced', 'name': 'Balanced',
               'hint': 'Angles, framings and detail shots for any subject.'},
              {'key': 'other_quick', 'name': 'Quick set',
               'hint': 'Eight shots to test a subject before committing to a full run.'}],
    'anime': [{'key': 'anime_balanced', 'name': 'Balanced',
               'hint': 'Face, bust-up, cowboy shot, full body and rear views for a drawn character.'},
              {'key': 'anime_face_focused', 'name': 'Face & expressions',
               'hint': 'Angles plus the expression sheet — smile, angry, surprised, blush, eyes closed.'},
              {'key': 'anime_fullbody_focused', 'name': 'Full body focused',
               'hint': 'Standing, walking, running and action poses — for the silhouette and proportions.'},
              {'key': 'anime_character_sheet', 'name': 'Character sheet',
               'hint': 'Front, side and back turnaround on a plain background, plus hair, eyes and '
                       'outfit details — teaches the design with no scenery to memorise.'}],
}
_SUBJECT_BY_ID = {st: {e['id']: e for e in cat} for st, cat in _SUBJECT_CATALOGS.items()}


def variation_catalog(subject_type: str = 'human'):
    """The SFW variation catalog for a subject type ('human' = the historical one)."""
    return _SUBJECT_CATALOGS.get(normalize_subject_type(subject_type), VARIATION_CATALOG)


# --- Composed-prompt preview -------------------------------------------------
# The prompt an engine actually receives is ~1000 characters assembled from six
# sources, and nothing in the app ever showed it. Editing one of those sources
# without seeing the whole was editing blind — and reading one required writing a
# throwaway script. This composes a real shot through the REAL wrappers, so the
# preview cannot drift from what generation does: any future change to the
# assembly shows up here for free, and a preview that were re-implemented in the
# UI would have shown yesterday's structure forever.
#
# It is a PURE text composition: no model, no GPU, no network, nothing enqueued.
PREVIEW_ENGINES = ('nanobanana', 'chatgpt', 'openrouter', 'klein', 'krea')

# The engines that go through wrap_variation (guard-first), i.e. the API family.
_API_PREVIEW_ENGINES = ('nanobanana', 'chatgpt', 'openrouter')


def preview_shot(subject_type: str = 'human', framing: str = 'bust', nsfw: bool = False):
    """A REAL catalog entry to preview with — the first one matching `framing`
    (and the NSFW catalog when asked), so the preview shows the directives that
    are actually baked into shipped shots rather than a made-up sentence. Falls
    back to the first entry of whatever catalog is non-empty, and finally to a
    minimal synthetic entry, so no subject type can make the preview blank."""
    st = normalize_subject_type(subject_type)
    pools = ([nsfw_variation_catalog(st), variation_catalog(st)] if nsfw
             else [variation_catalog(st)])
    for pool in pools:
        for entry in pool:
            if entry.get('framing') == framing:
                return entry
    for pool in pools:
        if pool:
            return pool[0]
    return {'id': 'preview', 'label': 'Preview', 'framing': framing,
            'prompt': 'a portrait, front view'}


def compose_preview(engine: str, subject_type: str = 'human', framing: str = 'bust',
                    nsfw: bool = False, suffix: str = '', overrides=None) -> dict:
    """The exact text `engine` would be sent for one shot, plus what it was made
    from. `overrides` is an UNSAVED `identity_prompts` tree (see
    preview_prompt_overrides) — pass None to preview the saved configuration.

    Returns {engine, subject_type, framing, nsfw, shot_id, shot_label,
             shot_prompt, prompt, length}. Unknown engine -> Klein, the same
    fallback activeExtraRefPromptKey uses on the client, so the two surfaces
    cannot disagree about what a legacy engine id means."""
    eng = str(engine or '').strip().lower()
    st = normalize_subject_type(subject_type)
    fr = framing if framing in PROMPT_FRAMINGS else 'bust'
    entry = preview_shot(st, fr, nsfw)
    raw = entry.get('prompt', '')
    label = entry.get('label', '')
    with preview_prompt_overrides(overrides):
        if eng in _API_PREVIEW_ENGINES:
            text = wrap_variation(raw, ref_count=1, suffix=suffix, subject_type=st)
        elif eng == 'krea':
            text = wrap_variation_krea(raw, nsfw=nsfw, framing=fr, suffix=suffix,
                                       subject_type=st, label=label)
        else:
            eng = eng if eng == 'klein' else 'klein'
            text = wrap_variation_klein(raw, nsfw=nsfw, framing=fr, suffix=suffix,
                                        subject_type=st)
    return {'engine': eng, 'subject_type': st, 'framing': fr, 'nsfw': bool(nsfw),
            'shot_id': entry.get('id', ''), 'shot_label': label,
            'shot_prompt': raw, 'prompt': text, 'length': len(text)}


def nsfw_variation_catalog(subject_type: str = 'human'):
    """The uncensored catalog for a subject type — only 'human' has one; every other
    type returns [] (no NSFW body catalog)."""
    return _SUBJECT_NSFW.get(normalize_subject_type(subject_type), [])


def presets_for(subject_type: str = 'human') -> dict:
    """The {preset_name: [ids]} map for a subject type."""
    return _SUBJECT_PRESETS.get(normalize_subject_type(subject_type), {})


def preset_meta_for(subject_type: str = 'human') -> list:
    """Preset display metadata ({key,name,hint}) for a NON-human subject type; []
    for 'human' (the frontend owns the human preset labels)."""
    return _SUBJECT_PRESET_META.get(normalize_subject_type(subject_type), [])


def select_preset(name: str, subject_type: str = 'human'):
    st = normalize_subject_type(subject_type)
    by_id = _SUBJECT_BY_ID.get(st, _BY_ID)
    return [by_id[i] for i in presets_for(st).get(name, []) if i in by_id]


# --- Custom shot catalogs (imported from JSON) --------------------------------
# Idea by ashish.sinha (Discord): instead of typing 30-40 shots by hand, export
# the catalog, have an LLM write more, import the result. The shots live in the
# config (`custom_shots`), per subject type — NOT in localStorage: a catalog that
# vanishes when the browser is cleared, after paying an LLM to write it, is a
# feature that punishes its user, and the config is what makes the same shots
# visible from a second device and part of the full backup.
#
# The frontend validates on import (`shotImport.js`) and never sends the aspect
# (it is resolved server-side per label), so a custom shot is stored exactly as
# {id, label, prompt, framing, nsfw?}. This sanitizer is the second line: config
# .json is a plain file a user can hand-edit, and every shot that reaches it must
# still satisfy the invariants — enum framing, and above all a label that does
# NOT shadow a built-in one (see the by-label resolvers below).
MAX_CUSTOM_SHOTS_PER_SUBJECT = 300
_MAX_CUSTOM_LABEL = 80
_MAX_CUSTOM_PROMPT = 500


def all_catalog_labels() -> list:
    """Every label the by-label resolvers can already answer for: the union of all
    catalogs PLUS the legacy French aliases (still stored on pre-migration rows and
    routed through `canonical_label`). This is the reserved set an imported shot may
    never re-use — a collision would silently resolve to the built-in entry."""
    return sorted({e['label'] for e in _ALL_CATALOGS} | set(LEGACY_LABEL_ALIASES))


def sanitize_custom_shots(raw) -> dict:
    """{subject_type: [shot]} keeping only well-formed, non-colliding shots.
    Never raises: a malformed config must degrade to 'no custom shots', not 500
    the workspace."""
    if not isinstance(raw, dict):
        return {}
    reserved = {lbl.strip().lower() for lbl in all_catalog_labels()}
    out = {}
    for subject, shots in raw.items():
        st = normalize_subject_type(subject)
        if not isinstance(shots, list):
            continue
        seen_ids, seen_labels, kept = set(), set(), []
        for shot in shots[:MAX_CUSTOM_SHOTS_PER_SUBJECT]:
            if not isinstance(shot, dict):
                continue
            sid = shot.get('id')
            label = shot.get('label')
            prompt = shot.get('prompt')
            framing = shot.get('framing')
            if not all(isinstance(v, str) and v.strip() for v in (sid, label, prompt, framing)):
                continue
            sid, label, prompt = sid.strip(), label.strip(), prompt.strip()
            framing = framing.strip().lower()
            if framing not in ASPECT_BY_FRAMING:
                continue
            if len(label) > _MAX_CUSTOM_LABEL or len(prompt) > _MAX_CUSTOM_PROMPT:
                continue
            key = label.lower()
            if key in reserved or key in seen_labels or sid in seen_ids:
                continue
            seen_ids.add(sid)
            seen_labels.add(key)
            kept.append({'id': sid, 'label': label, 'prompt': prompt, 'framing': framing,
                         **({'nsfw': True} if shot.get('nsfw') is True else {}),
                         'imported': True})
        out.setdefault(st, []).extend(kept)
    return {st: shots for st, shots in out.items() if shots}


def prompt_by_label(label):
    """Raw catalog prompt for a display label (fallback for pre-migration rows).
    Searches the SFW catalog then the NSFW one (regenerate needs both). The label is
    canonicalised first so a legacy French label still recovers its prompt."""
    label = canonical_label(label)
    return next((e['prompt'] for e in _ALL_CATALOGS if e['label'] == label), None)


# Aspect ratio par cadrage (deep-research 2026-06-14) : forcer tout en carré
# letterboxe les plans corps (bandes noires apprises par le LoRA). On demande à
# Nano Banana un ratio adapté ; ai-toolkit gère le bucketing non-carré.
ASPECT_BY_FRAMING = {'face': '1:1', 'bust': '3:4', 'body': '3:4', 'back': '3:4'}


def aspect_for_framing(framing: str) -> str:
    return ASPECT_BY_FRAMING.get(framing, '1:1')


def aspect_for_entry(entry) -> str:
    """Ratio d'une ENTRÉE de catalogue : override explicite, sinon défaut du cadrage."""
    return entry.get('aspect') or aspect_for_framing(entry.get('framing'))


def aspect_for_label(label, framing='face') -> str:
    """Ratio résolu PAR LABEL sur le catalogue serveur (autoritatif) — le frontend
    n'envoie pas l'aspect, et la régénération n'a que la ligne DB. Retrouve l'entrée
    par son label → son override ; label inconnu → fallback cadrage. Le label est
    d'abord canonicalisé pour qu'un ancien label français retrouve son override."""
    label = canonical_label(label)
    e = next((x for x in _ALL_CATALOGS if x['label'] == label), None)
    return aspect_for_entry(e) if e else aspect_for_framing(framing)


def composition_counts(entries):
    out = {'face': 0, 'bust': 0, 'body': 0, 'back': 0}
    for e in entries:
        out[e['framing']] = out.get(e['framing'], 0) + 1
    return out


CAPTION_PROMPT = (
    "Caption Type: Straightforward.\n\n"
    "ABSOLUTE RULE - the subject's physical identity is already known and must NEVER "
    "appear in the caption. Never mention, in any form: hair (its length, colour, style, "
    "texture, or how it falls - e.g. do NOT write \"long hair\", \"hair falls around the "
    "shoulders\", \"hair tied back\", \"ponytail\"), face shape, facial features, eye "
    "colour, eyebrows, nose, lips, jawline, skin tone or texture, freckles, age, gender, "
    "body build, or ethnicity. If a person is present, refer to them only as \"the subject\".\n\n"
    "You MUST still describe: the subject's expression and gaze as actions or states ONLY "
    "(smiling, laughing, surprised, eyes closed, looking at the viewer); pose and body "
    "position; clothing and accessories with their colours; the setting or location; and "
    "the lighting and mood.\n\n"
    "Output ONE caption as flowing natural-language prose, beginning with the shot type and "
    "framing (close-up, three-quarter shot, full-body, wide), then the pose, then the "
    "expression, then the clothing and accessories, then the setting, then the lighting and "
    "mood. Output only the caption itself - no preamble, no \"Here is\", no quotation marks, "
    "no commentary.")

# JoyCaption et le fallback Qwen3-VL partagent ce prompt POSITIF + mode entrainé
# "Straightforward". Validé empiriquement (24/31 fuites -> 0/31). La consigne negative
# precedente etait ignoree par JoyCaption ("not a general instruction follower").
JOYCAPTION_PROMPT = CAPTION_PROMPT


# Neutral DESCRIPTIVE caption — used by the image bank (and the launch-all pipeline),
# NOT by dataset training. A bank has no trigger word and nothing to protect, so unlike
# the dataset prompts this one omits NOTHING: it names everything visible (subjects and
# their appearance, clothing colours, objects, setting, mood) precisely because the
# caption doubles as SEARCH text — "show me every image with a red dress" only works if
# "red dress" actually made it into the caption. One compact paragraph of plain prose.
DESCRIPTIVE_CAPTION_PROMPT = (
    "Caption Type: Straightforward.\n\n"
    "Describe this image plainly and completely, so the description can be searched "
    "later. Name what is actually visible: the subjects and their appearance, their "
    "pose and expression, the clothing and accessories WITH their colours, any notable "
    "objects, the setting or location, and the overall lighting and mood.\n\n"
    "Output ONE caption as flowing natural-language prose. Output only the caption "
    "itself — no preamble, no \"Here is\", no quotation marks, no commentary.")


# Dataset STYLE : l'invariant du set est le RENDU (esthétique, médium, palette, trait…),
# qui doit être absorbé par le LoRA — donc jamais décrit. Règle miroir du concept :
# ce qui est captionné reste contrôlable par le prompt, ce qui est tu est absorbé.
# On décrit donc le CONTENU librement (sujets, scène, composition — l'identité est
# conservée, les sujets varient) et on tait tout vocabulaire de style/rendu.
CAPTION_PROMPT_STYLE = (
    "Caption Type: Straightforward.\n\n"
    "This is one image from a STYLE training set: every image shares the same artistic "
    "style, and that style must NEVER be described - no words about the medium, technique, "
    "rendering, color palette, line work, brushwork, film grain, aesthetic or art movement. "
    "Caption only the CONTENT, as if the image were a plain photograph of the scene.\n\n"
    "Describe freely: the subjects present and their appearance, pose and expression, "
    "clothing, the setting and objects, the composition and framing, the time of day. "
    "One compact paragraph of plain prose. No preamble, no quotes, no lists."
)

CAPTION_PROMPT_STYLE_BOORU = (
    "Caption Type: Booru tag list.\n\n"
    "This is one image from a STYLE training set: every image shares the same artistic "
    "style, and that style must NEVER be tagged - no medium, technique, rendering, "
    "palette, aesthetic or art-movement tags (no 'oil painting', 'anime style', "
    "'watercolor', 'monochrome', 'sketch', etc.). Tag only the CONTENT.\n\n"
    "Output a single line of comma-separated booru tags covering: subject count and "
    "type, appearance, pose, expression, clothing, objects, setting, framing. "
    "Lowercase, underscores for spaces, no preamble."
)


def caption_prompt_for_style(mode) -> str:
    """The caption prompt for a STYLE dataset: content-only (the style is absorbed
    by omission), prose vs booru by model family."""
    return CAPTION_PROMPT_STYLE_BOORU if mode == 'booru' else CAPTION_PROMPT_STYLE


# Dataset CONCEPT (logique INVERSÉE) : l'invariant du set n'est plus l'identité mais
# l'acte/effet récurrent qu'on OMET pour qu'il se lie au trigger. On décrit donc tout —
# personnes, pose, cadrage, lumière, décor — SAUF l'acte central répété. Le captioneur
# reçoit la description EXACTE du concept ({concept}, saisie à la création du dataset) pour
# savoir précisément quoi taire, plutôt que de deviner l'action dominante. Aucun post-filtre
# d'identité (on GARDE l'identité).
CAPTION_PROMPT_CONCEPT = (
    "Caption Type: Straightforward.\n\n"
    "This is one image from a CONCEPT training set. The single element every image in the "
    "set shares is: {concept}. Describe the whole scene EXCEPT that shared element - simply "
    "leave it unmentioned, as if it were not there. Never name it, and never describe the "
    "act, object, device or surface that shows it.\n\n"
    "Describe, in full and freely (nothing about the people is hidden): the people present "
    "and their appearance (hair, face, body, skin, marks), their pose and body position, "
    "their expression and gaze, any clothing or state of undress and its colours, the "
    "setting or location, the framing (close-up, three-quarter, full-body, from above, "
    "from below), and the lighting and mood. Write as a neutral outside observer of the "
    "person and their surroundings - describe what is in the scene, not how the picture was "
    "taken.\n\n"
    "Never transcribe any watermark, website URL, studio name, or text printed on the "
    "image.\n\n"
    "Output ONE caption as flowing natural-language prose, beginning with the shot type "
    "and framing, then the people and pose, then expression, then clothing/setting, then "
    "lighting and mood - but leaving the shared concept itself UNSPOKEN. Output only the "
    "caption itself - no preamble, no reasoning, no \"Here is\", no quotation marks, no "
    "commentary.")


# Passe de RAFFINAGE concept (Joy→Qwen) : JoyCaption est très détaillé mais LITTÉRAL —
# il NOMME l'acte/les fluides/le watermark (ce qui, pour un concept, doit rester tu
# pour se lier au trigger). Qwen relit la caption Joy + l'image et RÉÉCRIT en retirant
# uniquement le focal explicite + le texte incrusté, en gardant tout le contexte riche.
# => détail de JoyCaption + adhérence de Qwen (mesuré : Joy nomme le concept ~4/4).
CAPTION_REFINE_CONCEPT_PROMPT = (
    "Below is a draft caption describing this exact image:\n\n"
    "\"\"\"\n{existing}\n\"\"\"\n\n"
    "Rewrite it as ONE clean caption for a CONCEPT training set. The single recurring "
    "concept this set teaches is: {concept}.\n\n"
    "KEEP every contextual detail already present: the people and their appearance (hair, "
    "face, body, skin, freckles), their pose and body position, expression and gaze, any "
    "clothing or state of undress and its colours, the setting or location, the camera "
    "angle and framing, and the lighting and mood.\n\n"
    "But you MUST REMOVE, and never restate:\n"
    "1. The concept itself - {concept} - and any word, substance, effect, action or graphic "
    "focal detail that names or describes it, in ANY phrasing. Do NOT replace it with "
    "euphemisms or vague allusions either (words like 'organ', 'genitalia', 'member', "
    "'intimate act', 'sexual act'): leave it entirely undescribed, as if the caption were "
    "unaware of it, and describe only the people, their positions, hands, faces and the "
    "scene.\n"
    "2. Any watermark, website URL, studio name, or text printed on the image.\n\n"
    "Rephrase around the removed elements so the prose stays natural - do NOT mention that "
    "anything was removed.\n\n"
    "Output ONLY the rewritten caption as flowing prose - no preamble, no \"Here is\", no "
    "quotation marks, no commentary.")


# Expansion de la ban-list concept : à partir de la description du concept, le LLM liste
# les mots/locutions qu'un captioneur emploierait pour le NOMMER (synonymes, argot, formes
# verbales). Sert au DÉTECTEUR de fuite (regex), pas au prompt de caption — la littérature
# sur le negative prompting montre que lister les mots interdits dans le prompt de
# GÉNÉRATION amorce l'effet « éléphant rose » ; la robustesse vient de la vérification en
# sortie + correction ciblée. Format JSON objet (le grammar-mode d'Ollama produit un objet
# plus fiablement qu'un tableau nu). Accolades DOUBLÉES → survivent au .format(concept=…).
# Loop-resistant on purpose: the earlier version listed residue examples ("glistening,
# dripping, sticky, white substance") and asked for 8-25 terms — the abliterated Qwen
# latched onto the examples and looped combinatorially ("mirror selfie shot",
# "self-portrait photograph"…) past the token budget, leaving an UNCLOSED array that
# json.loads rejected → empty ban-list → the concept leaked into every caption. So: no
# seeding examples, "each term once, then STOP", 6-15 terms, and an explicit ban on
# listing the PEOPLE/body/clothing (which must stay DESCRIBED, never scrubbed).
EXPAND_CONCEPT_TERMS_PROMPT = (
    "Ignore the attached image entirely. You are building a caption BLOCKLIST for a "
    "CONCEPT training set.\n"
    "Concept: \"{concept}\".\n"
    "List the words and short phrases (max 3 words) a photo captioner would use to NAME "
    "this concept itself, or the object, device, surface or action that shows it - plus "
    "close synonyms and singular/plural forms.\n"
    "Rules: ONLY words that specifically point to the concept. Do NOT list the people, "
    "their body, skin, clothing, colours, pose, expression or setting. Each term at most "
    "ONCE - never repeat a word or pad with combinations. Give 6 to 15 terms, then STOP.\n"
    "Output ONLY a JSON object: {{\"terms\": [\"term one\", \"term two\"]}}")


# Réécriture CORRECTIVE après détection de fuite : on nomme les mots EXACTS qui ont fui
# (feedback ciblé ≫ instruction générique). Placeholders : existing / concept / leaked.
CAPTION_LEAK_FIX_PROMPT = (
    "Below is a caption for this exact image:\n\n"
    "\"\"\"\n{existing}\n\"\"\"\n\n"
    "This caption is for a CONCEPT training set where the concept must stay UNSPOKEN. "
    "The concept is: {concept}.\n"
    "The caption accidentally uses these forbidden words: {leaked}. They MUST disappear.\n"
    "Rewrite the caption keeping every other detail (people and their appearance, pose, "
    "expression, clothing, setting, camera angle and framing, lighting) but remove the "
    "forbidden words WITHOUT replacing them by synonyms, euphemisms or vague allusions "
    "that still name or hint at the concept (no 'organ', 'genitalia', 'member', 'intimate "
    "act', 'sexual act' or similar): leave the thing entirely undescribed, as if the "
    "caption were unaware of it. Do not mention that anything was removed.\n"
    "Output ONLY the rewritten caption as flowing prose - no preamble, no \"Here is\", no "
    "quotation marks, no commentary.")


# --- Mode FIDÉLITÉ CORPS (fidelity='body') -------------------------------------
# Pour un LoRA qui doit reproduire AUSSI la morphologie, les marques corporelles
# PERMANENTES (tatouages, cicatrices, taches de naissance, piercings) sont de
# l'identité au même titre que le visage : les décrire dans la caption les lierait
# aux mots au lieu du trigger. Blocs AJOUTÉS aux prompts de base (la morphologie —
# body build, breast size… — y est déjà bannie).
BODY_FIDELITY_PROSE_SUFFIX = (
    "\n\nBODY-FIDELITY RULE - this subject's BODY is part of the learned identity. "
    "Additionally NEVER mention: tattoos, scars, birthmarks, moles, piercings or any "
    "permanent body marking; body proportions or measurements; breast/chest size; "
    "muscle definition. Clothing, pose and framing must still be fully described.")

BODY_FIDELITY_BOORU_SUFFIX = (
    "\n\nBODY-FIDELITY RULE - additionally never tag permanent body markings or "
    "proportions: no tattoo, scar, birthmark, mole, piercing, abs, muscular or "
    "measurement tags. Clothing, pose and framing tags stay required.")


def caption_prompt_for(mode, body=False) -> str:
    """The caption prompt for a character dataset: prose vs booru, with the extra
    body-identity ban block when the dataset targets full-body fidelity."""
    base = CAPTION_PROMPT_BOORU if mode == 'booru' else JOYCAPTION_PROMPT
    if not body:
        return base
    return base + (BODY_FIDELITY_BOORU_SUFFIX if mode == 'booru' else BODY_FIDELITY_PROSE_SUFFIX)


# Detecteur INDICATIF de VRAIS descripteurs d'identite (cheveux/peau/couleur d'yeux/
# forme de visage/traits). Ne flague PAS "the face" (lumiere) ni "eyes open/looking"
# (expression) — calibre empiriquement sur 31 captions reelles.
_IDENTITY_LEAK = re.compile(
    r'\bhair\b'
    r'|\bcomplexion\b|\bfreckles?\b|\bjawline\b|\beyebrows?\b|\bfacial\s+features?\b'
    r'|\bskin\b'
    r'|\b(?:blue|brown|green|hazel|grey|gray|dark|light|pale|amber)\s+eyes\b'
    r'|\b(?:round|oval|square|angular|heart-shaped|long|narrow|wide|slim|chubby)\s+face\b',
    re.I)

# Marques corporelles permanentes = identité en mode body-fidelity (détection + drop).
_BODY_LEAK = re.compile(
    r'\btattoos?\b|\btattooed\b|\bscars?\b|\bscarred\b|\bbirthmarks?\b|\bmoles?\b'
    r'|\bpiercings?\b|\bpierced\b', re.I)


def caption_has_identity_leak(caption, body=False) -> bool:
    """True si la caption mentionne un VRAI trait d'identite. Detecteur SEUL (badge).
    body=True (fidélité corps) flague AUSSI les marques corporelles permanentes."""
    if not caption:
        return False
    return bool(_IDENTITY_LEAK.search(caption) or (body and _BODY_LEAK.search(caption)))


# Post-filtre : drop les PHRASES decrivant un trait d'identite. Avec le prompt
# "Straightforward", la rare fuite est isolee dans sa propre phrase -> suppression
# propre (pas de casse grammaticale). NE drop PAS expression ("eyes closed") ni
# lumiere ("shadow on the face").
_DROP_SENT = re.compile(
    r'\bhair\b|\bcomplexion\b|\bfreckles?\b|\bjawline\b|\beyebrows?\b|\bfacial\s+features?\b'
    r'|\bskin\s+(?:tone|texture)\b', re.I)


def drop_identity_sentences(caption, body=False) -> str:
    """Retire les phrases d'identite isolees d'une caption (post-captioning).
    body=True retire aussi les phrases décrivant une marque corporelle permanente."""
    parts = re.split(r'(?<=[.!?])\s+', caption or '')
    kept = [s for s in parts if s.strip() and not _DROP_SENT.search(s)
            and not (body and _BODY_LEAK.search(s))]
    return ' '.join(kept).strip()


# --- STYLE medium lead-in (kind=style) ----------------------------------------
# Style twin of drop_identity_sentences. CAPTION_PROMPT_STYLE forbids naming the
# medium, but that rule is NEGATIVE and JoyCaption is "not a general instruction
# follower" (same reason CAPTION_PROMPT had to be rewritten positive, above), so
# captions keep opening with "A digital illustration of …". The medium then binds
# to those words instead of being absorbed by the LoRA — exactly what a style set
# is trying to avoid — so the opener is cut here, in post.
#
# Only the LEAD-IN is touched. A medium named INSIDE the scene is content ("a
# painting hangs on the wall"), and the whole filter is built to leave it alone:
# it matches at position 0 only, and only when the medium noun is followed by
# "of" or a depicting verb, which is what makes it an opener rather than a
# subject. The residual false positive is "An oil painting of a stormy sea hangs
# on the wall" — a lead-in shape describing an object — which _LEAD_IN_OBJECT
# catches by looking for a hanging/mounting verb landing on a wall or an easel.
#
# The vocabulary below is the TEXT lexicon; the bank owns a separate medium
# vocabulary (image_bank_service.MEDIUM_PROTOTYPES) that is a CLIP classifier
# tuned against measured cosine margins and must not be edited to serve a text
# filter. test_style_caption_lead_in.py ties the two so they cannot drift.
STYLE_MEDIUM_HEADS = (
    'illustration', 'illustrations', 'drawing', 'drawings', 'sketch', 'sketches',
    'painting', 'paintings', 'render', 'renders', 'rendering', 'renderings',
    'artwork', 'artworks', 'art', 'image', 'images', 'picture', 'pictures',
    'photo', 'photos', 'photograph', 'photographs', 'snapshot', 'snapshots',
    'comic', 'cartoon', 'manga', 'anime',
)

# Modifiers that may sit between the article and the head noun. A CLOSED list on
# purpose: allowing "any adjective" would swallow real openers such as "A young
# woman with red hair …", where the first words are the content itself.
STYLE_MEDIUM_MODS = (
    'digital', 'traditional', 'hand-drawn', 'hand', 'drawn', 'painted', 'rendered',
    'illustrated', 'sketched', 'coloured', 'colored', 'colorful', 'colourful',
    'monochrome', 'black-and-white', 'greyscale', 'grayscale', 'sepia',
    'detailed', 'highly', 'stylized', 'stylised', 'simple', 'rough', 'quick',
    'vintage', 'retro', 'classic', 'candid', 'group', 'studio', 'professional',
    'anime', 'anime-style', 'manga', 'manga-style', 'cartoon', 'cartoonish',
    'comic', 'comic-book', 'book', 'pencil', 'charcoal', 'ink', 'pen',
    'watercolor', 'watercolour', 'oil', 'acrylic', 'gouache', 'pastel', 'chalk',
    'crayon', 'airbrush', 'linocut', 'woodcut', 'collage',
    '3d', 'cgi', 'computer-generated', 'photorealistic', 'realistic',
    'semi-realistic', 'real-life', 'fantasy', 'concept', 'line', 'vector',
    'pixel', 'low-poly', 'cel-shaded', 'painterly', 'sketchy', 'minimalist',
    'abstract', 'surreal',
)

# "This is …", "The image shows …" — the wrapper a captioner puts in front of the
# medium. Stripped only as part of a medium lead-in, never on its own.
_LEAD_PREFIX = (r'(?:this|that|it)\s+is|here\s+is'
                r'|(?:this|the)\s+(?:image|picture|artwork|illustration)\s+'
                r'(?:is|shows|depicts|features|presents)')
_LEAD_VERB = (r'shows?|showing|depicts?|depicting|features?|featuring'
              r'|portrays?|portraying|captures?|capturing|presents?|presenting'
              r'|illustrates?|illustrating|displays?|displaying')


def _alt(words) -> str:
    # longest-first so 'artwork' wins over 'art', 'anime-style' over 'anime'…
    return '|'.join(re.escape(w) for w in sorted(words, key=len, reverse=True))


# The article is INDEFINITE only. A lead-in announces the medium ("A digital
# illustration of…", "Digital art of…"); a definite one points at something already
# in the scene ("The artwork of the poster behind her reads FESTIVAL"), which is
# content. "The image is a…" still works — that "the" belongs to the prefix.
_LEAD_STEM = (rf'^\W*(?:(?:{_LEAD_PREFIX})\s+)?(?:an?\s+)?'
              rf'(?:(?:{_alt(STYLE_MEDIUM_MODS)})[-\s]+){{0,4}}'
              rf'(?:{_alt(STYLE_MEDIUM_HEADS)})\b')
# "<medium> of <content>" / "<medium> shows <content>"
_STYLE_LEAD_IN = re.compile(rf'{_LEAD_STEM}(?:\s+of\s+|\s+(?:{_LEAD_VERB})\s+)', re.I)
# "<medium>." as a whole opening sentence, the rest of the caption following it
_STYLE_LEAD_SENTENCE = re.compile(rf'{_LEAD_STEM}\s*[.!](?:\s+|$)', re.I)
# the object-in-the-scene rescue described above
_LEAD_IN_OBJECT = re.compile(
    r'^[^.!?]{0,80}?\b(?:hangs?|hanging|hung|leans?|leaning|propped|mounted'
    r'|framed|displayed)\b[^.!?]{0,30}?\b(?:wall|walls|easel|frame|mantel|mantelpiece'
    r'|shelf|gallery)\b', re.I)


def drop_style_lead_in(caption) -> str:
    """Retire l'amorce de médium d'une caption de dataset STYLE, en réparant la
    grammaire : « A digital illustration of a young woman… » -> « A young woman… ».
    Ne touche QUE l'amorce ; un médium cité dans la scène est du contenu."""
    text = (caption or '').strip()
    if not text:
        return ''
    m = _STYLE_LEAD_IN.match(text) or _STYLE_LEAD_SENTENCE.match(text)
    if not m:
        return text
    rest = text[m.end():].lstrip()
    # Nothing but the medium statement, or the medium is the scene's own object:
    # returning the caption untouched is always the safe side of this filter.
    if not rest or _LEAD_IN_OBJECT.match(rest):
        return text
    return rest[0].upper() + rest[1:]


# --- CONCEPT leak detection (kind=concept) ----------------------------------
# A concept LoRA teaches a recurring element (a pose, an act, an effect) that must bind
# to the TRIGGER word, never to caption words. A caption "leaks" when it NAMES that
# element. Unlike identity (a FIXED vocabulary: hair/skin/eyes), the concept vocabulary is
# PER-DATASET, so the lexicon is DERIVED from the dataset's own concept_desc — never a
# hard-coded list:
#   1. the meaningful words of concept_desc itself (singular/plural tolerated by the regex);
#   2. the cached LLM ban-list (ds.concept_terms), when present;
#   3. the basic lexical FIELD of any body region the description ANCHORS — and only then.
#      "leg behind head position" anchors the lower-limb family, so the periphrases a
#      captioner reaches for ("knees lifted", "feet raised", "thighs") are caught even
#      though the description never spells them out. A concept about "a mirror selfie"
#      anchors NO body family, so leg words are never added: the field is scoped to the
#      anchors actually present, not sprayed onto every concept.
# This is exactly why the leg_behind incident leaked: the ban-list was the 4 words of the
# description; "knees/feet/thighs/lifted/raised" were never listed, so the omission net
# had nothing to catch — and the aggregate badge FORCED 0 for concept datasets, hiding it.
_CONCEPT_LEAK_STOP = frozenset((
    'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'by', 'with', 'to', 'from',
    'that', 'this', 'as', 'is', 'are', 'his', 'her', 'their', 'its', 'it', 'one', 'both',
    'act', 'shown', 'worn', 'being', 'person', 'subject', 'focal', 'point', 'visible',
    'bare', 'exposed', 'full', 'close', 'closeup', 'wearing', 'showing'))

# (anchor tokens that may appear in concept_desc) -> (lexical field added to the lexicon,
#  human label for the omission hint). A family fires only if >=1 anchor is a TOKEN of the
#  description, so the field is scoped to the body region the concept is actually about.
_BODY_FAMILIES = (
    (frozenset({'leg', 'legs', 'knee', 'knees', 'thigh', 'thighs', 'foot', 'feet',
                'calf', 'calves', 'shin', 'shins', 'ankle', 'ankles', 'hamstring'}),
     ('leg', 'legs', 'knee', 'knees', 'thigh', 'thighs', 'foot', 'feet',
      'calf', 'calves', 'shin', 'shins', 'ankle', 'ankles'),
     'the legs, knees, thighs, feet or ankles'),
    (frozenset({'arm', 'arms', 'elbow', 'elbows', 'wrist', 'wrists', 'hand', 'hands',
                'forearm', 'forearms', 'palm', 'palms'}),
     ('arm', 'arms', 'elbow', 'elbows', 'wrist', 'wrists', 'hand', 'hands',
      'forearm', 'forearms'),
     'the arms, elbows, wrists or hands'),
    (frozenset({'head', 'neck', 'nape', 'chin'}),
     ('head', 'neck', 'nape'),
     'the head or neck'),
    (frozenset({'hip', 'hips', 'waist', 'torso', 'back', 'spine', 'pelvis'}),
     ('hip', 'hips', 'waist', 'torso', 'spine'),
     'the hips, waist or torso'),
)

# Posture verbs: a POSE concept binds the ARRANGEMENT, not merely the body part, so these
# are added when a body family fires OR the description itself names a pose/position. Kept
# to unambiguous posture verbs — recall matters far more than a rare over-scrub here: an
# UNDER-detected pose (the incident) binds the concept to words and kills the LoRA, while
# an over-scrubbed clause only trims a caption the trigger already carries.
_POSE_ANCHOR = frozenset({
    'pose', 'posed', 'poses', 'position', 'positions', 'positioned', 'positioning',
    'posture', 'postured', 'arranged', 'contorted', 'contortion', 'bent', 'folded',
    'raised', 'lifted', 'extended', 'spread', 'split', 'splits', 'stretched', 'curled',
    'arched', 'crossed', 'tucked', 'elevated', 'splayed', 'kneeling', 'squatting'})
_POSE_FIELD = ('lifted', 'raised', 'extended', 'bent', 'folded', 'crossed', 'tucked',
               'splayed', 'elevated', 'spread', 'straightened', 'curled', 'arched',
               'positioned')


def _concept_desc_tokens(text) -> list:
    return [w for w in re.split(r'[^a-z]+', (text or '').lower()) if w]


def concept_lexical_field(concept_desc) -> list:
    """The derived body/pose lexical field for a concept: the union of every body family
    the description ANCHORS, plus the posture-verb field when the concept is a pose.
    Empty for a concept that names no body region (e.g. a photographic 'mirror selfie').
    Pure & deterministic — never a per-all-concepts hard-coded vocabulary."""
    toks = set(_concept_desc_tokens(concept_desc))
    if not toks:
        return []
    field, limb_fired = set(), False
    for anchors, terms, _label in _BODY_FAMILIES:
        if toks & anchors:
            field.update(terms)
            limb_fired = True
    if limb_fired or (toks & _POSE_ANCHOR):
        field.update(_POSE_FIELD)
    return sorted(field)


def _norm_concept_terms(concept_terms) -> list:
    """Accept a list, a JSON string (as stored on ds.concept_terms), or None -> clean
    list of strings."""
    if not concept_terms:
        return []
    if isinstance(concept_terms, str):
        try:
            concept_terms = json.loads(concept_terms)
        except (ValueError, TypeError):
            return []
    return [t for t in concept_terms if isinstance(t, str)] if isinstance(concept_terms, list) else []


def concept_leak_terms(concept_desc, concept_terms=None) -> list:
    """The full concept-leak lexicon: meaningful words of concept_desc + the cached LLM
    ban-list (concept_terms) + the derived body/pose field. Deterministic; the detection
    counterpart of the identity regex."""
    terms = {w for w in _concept_desc_tokens(concept_desc)
             if len(w) >= 3 and w not in _CONCEPT_LEAK_STOP}
    for t in _norm_concept_terms(concept_terms):
        t = t.strip().lower()
        if len(t) >= 3 and t not in _CONCEPT_LEAK_STOP:
            terms.add(t)
    terms.update(concept_lexical_field(concept_desc))
    return sorted(terms)


def _concept_leak_re(terms):
    """Leak regex over a term list: word boundaries, space/hyphen interchangeable,
    plural/-s/-es/-ing/-ed tolerated. None if the list is empty."""
    pats = []
    for t in terms or []:
        t = (t or '').strip().lower()
        if len(t) < 3:
            continue
        p = re.escape(t).replace(r'\ ', r'[\s-]+').replace(r'\-', r'[\s-]+')
        pats.append(p)
    if not pats:
        return None
    return re.compile(r'\b(?:' + '|'.join(pats) + r')(?:e?s|ing|ed)?\b', re.I)


def caption_concept_leaks(caption, concept_desc, concept_terms=None) -> list:
    """The forbidden concept terms actually PRESENT in `caption` (deduped, sorted). Empty
    = clean. Drives the honest badge, the per-image flag, and the targeted-rewrite
    feedback. Pure — no model, no I/O."""
    if not caption:
        return []
    leak_re = _concept_leak_re(concept_leak_terms(concept_desc, concept_terms))
    if not leak_re:
        return []
    return sorted({m.group(0).lower() for m in leak_re.finditer(caption)})


def caption_has_concept_leak(caption, concept_desc, concept_terms=None) -> bool:
    """True if `caption` names the concept (kind=concept). Detector SEUL (badge), the
    concept-side twin of caption_has_identity_leak."""
    return bool(caption_concept_leaks(caption, concept_desc, concept_terms))


def drop_concept_sentences(caption, concept_desc, concept_terms=None) -> str:
    """Concept analogue of drop_identity_sentences: drop whole sentences that name the
    concept. Sentence-level safety mirror (the service clause-scrub is finer-grained)."""
    leak_re = _concept_leak_re(concept_leak_terms(concept_desc, concept_terms))
    if not leak_re:
        return (caption or '').strip()
    parts = re.split(r'(?<=[.!?])\s+', caption or '')
    kept = [s for s in parts if s.strip() and not leak_re.search(s)]
    return ' '.join(kept).strip()


def concept_omission_hint(concept_desc) -> str:
    """A SPECIFIC negative clause for the caption prompt, derived from the concept. The
    generic 'describe their pose and body position' instruction CONTRADICTS a pose concept
    (the pose IS the concept), so we name the exact body regions to leave unstated. Empty
    when the concept anchors no body region — the base prompt's 'leave {concept}
    unmentioned' already suffices, and the historical prompt stays byte-identical."""
    toks = set(_concept_desc_tokens(concept_desc))
    if not toks:
        return ''
    labels = [label for anchors, _terms, label in _BODY_FAMILIES if toks & anchors]
    if not labels:
        return ''
    parts = labels[0] if len(labels) == 1 else ', nor of '.join(labels)
    return (' In particular, do NOT describe the position or arrangement of ' + parts +
            ': never say they are lifted, raised, extended, bent, folded, crossed, spread '
            'or in any specific position - that exact pose is captured by the trigger word '
            'ALONE. Describe the person, clothing, expression and setting normally, but '
            'leave how the body is positioned entirely unstated.')


def caption_prompt_for_concept(concept_desc) -> str:
    """The concept caption prompt with a dynamic, concept-specific omission clause folded
    into the opening instruction. For a non-body concept the clause is empty and the
    prompt is byte-identical to the historical CAPTION_PROMPT_CONCEPT.format()."""
    desc = (concept_desc or '').strip()
    base = CAPTION_PROMPT_CONCEPT.format(concept=desc)
    hint = concept_omission_hint(desc)
    if not hint:
        return base
    # Splice the specific negative right after the opening omission sentence ("…never
    # describe the act, object, device or surface that shows it.") so it sits beside the
    # general rule and OVERRIDES the later generic "describe their pose" line.
    anchor = 'that shows it.'
    idx = base.find(anchor)
    if idx == -1:
        return base + '\n\n' + hint.strip()
    cut = idx + len(anchor)
    return base[:cut] + hint + base[cut:]


# --- Mode BOORU (datasets SDXL booru-native type bigLove) --------------------
# Les fine-tunes SDXL booru se promptent en tags danbooru (virgules) ; la prose est
# un mismatch de style (recherche 2026-06-14). On demande à JoyCaption le mode
# "Booru tag list" en EXCLUANT l'identité (même principe que la prose : l'identité
# se lie au trigger, pas aux mots).
CAPTION_PROMPT_BOORU = (
    "Caption Type: Booru tag list.\n\n"
    "ABSOLUTE RULE - the subject's physical identity is already known and must NEVER be "
    "tagged. Do NOT output any tag describing: hair (length/colour/style - e.g. long_hair, "
    "blonde_hair, ponytail, bangs, braid), eye colour (blue_eyes, brown_eyes, ...), face "
    "shape, facial features (eyebrows, eyelashes, lips, nose, jawline, freckles, moles), "
    "skin tone or texture, age, gender or count (1girl, 1boy, solo, woman, man, female, "
    "male), or body build (breast size, curvy, petite, muscular, thick thighs).\n\n"
    "DO output comma-separated booru/danbooru tags for ONLY: expression and gaze "
    "(smile, open_mouth, looking_at_viewer, closed_eyes, wink); pose and framing "
    "(standing, sitting, upper_body, cowboy_shot, full_body, portrait, from_side, "
    "from_above); clothing and accessories with their colours; the setting or location; "
    "and the lighting and mood. Output ONLY the comma-separated tag list - no preamble, "
    "no sentences, no quotation marks.")

# Tags booru d'IDENTITÉ à filtrer en post-traitement (le filtre prose par PHRASES est
# inutilisable sur des tags virgule). On drop par sous-chaîne, par valeur exacte, et un
# cas spécial 'eyes' (garder l'expression closed_eyes/wink, drop la couleur).
_IDENTITY_TAG_CONTAINS = (
    'hair', 'bangs', 'braid', 'ponytail', 'twintail', 'sideburn', 'eyebrow', 'eyelash',
    'freckle', 'complexion', 'jawline',
)
_IDENTITY_TAG_EXACT = frozenset({
    '1girl', '1boy', '2girls', '3girls', 'multiple_girls', 'multiple_boys',
    'solo', 'solo_focus', 'girl', 'boy', 'woman', 'man', 'female', 'male',
    'mature_female', 'milf', 'child', 'loli', 'shota', 'teenage', 'old',
    'aged_down', 'aged_up', 'bun', 'bald', 'mole', 'mole_under_eye',
    'breasts', 'large_breasts', 'medium_breasts', 'small_breasts', 'huge_breasts',
    'gigantic_breasts', 'flat_chest', 'curvy', 'thick_thighs', 'wide_hips',
    'petite', 'muscular', 'plump', 'skinny', 'lips', 'thick_lips', 'nose',
    'dark_skin', 'pale_skin', 'tan', 'tanlines', 'dark-skinned_female',
    'dark-skinned_male', 'pointy_ears',
})


# Marques corporelles permanentes (mode body-fidelity) — par sous-chaîne : couvre
# tattoo/arm_tattoo/tattooed, scar/scar_on_face, piercing/ear_piercing…
_BODY_TAG_CONTAINS = ('tattoo', 'scar', 'birthmark', 'piercing', 'pierced')


def _is_identity_tag(tag, body=False) -> bool:
    t = (tag or '').strip().lower().replace(' ', '_')
    if not t:
        return False
    if t in _IDENTITY_TAG_EXACT:
        return True
    if 'eyes' in t:  # garde l'EXPRESSION (closed_eyes, wink), drop la couleur (blue_eyes)
        return not any(k in t for k in ('closed', 'wink', 'half'))
    if body and any(sub in t for sub in _BODY_TAG_CONTAINS):
        return True
    return any(sub in t for sub in _IDENTITY_TAG_CONTAINS)


def drop_identity_tags(caption, body=False) -> str:
    """Retire les tags booru d'identité d'une caption en liste de tags (mode booru),
    pendant booru de drop_identity_sentences (mode prose). body=True retire aussi
    les marques corporelles permanentes (fidélité corps)."""
    if not caption:
        return ''
    kept = [t.strip() for t in caption.split(',') if t.strip() and not _is_identity_tag(t, body=body)]
    return ', '.join(kept).strip()


def caption_style(text) -> str:
    """Heuristique PURE : 'booru' (liste de tags virgule courts) vs 'prose' (phrases).
    Sert au garde-fou de cohérence caption↔type au lancement de l'entraînement."""
    t = (text or '').strip()
    if not t:
        return 'prose'
    segs = [s.strip() for s in t.split(',') if s.strip()]
    if len(segs) < 3:
        return 'prose'
    avg_words = sum(len(s.split()) for s in segs) / len(segs)
    sentence_punct = t.count('.') + t.count('!') + t.count('?')
    # Beaucoup de segments courts + quasi pas de ponctuation de phrase = tags booru.
    return 'booru' if (avg_words <= 3.0 and sentence_punct <= 1) else 'prose'


HEAD_BBOX_PROMPT = (
    "Locate the MAIN person's HEAD (face + hair) in this image. Output ONLY a minified "
    'JSON object with the bounding box on a 0-1000 grid: {"y1":top,"x1":left,"y2":bottom,"x2":right}. '
    "Include the whole head and hair, tight but complete. Output the JSON only.")

WATERMARK_BBOX_PROMPT = (
    "Look for an OVERLAID WATERMARK on this photo: a logo, a website URL, a social-media "
    "@username/handle, or studio/site text that was ADDED ON TOP of the picture after it "
    "was taken (often semi-transparent, in a corner, along an edge, or tiled). Do NOT "
    "report text that is PHYSICALLY PART OF THE SCENE (shop signs, street signs, clothing "
    "prints, book/product labels, tattoos) — only the overlay added onto the image. "
    'Output ONLY a minified JSON object. If an overlaid watermark is present: '
    '{"present":true,"y1":top,"x1":left,"y2":bottom,"x2":right} — the bounding box of the '
    "watermark on a 0-1000 grid (top-left origin, tight but complete). If there is NO "
    'overlaid watermark: {"present":false}. Output the JSON only.')

CLASSIFY_PROMPT = (
    "Classify this portrait photo. Output ONLY a minified JSON object: "
    '{"framing":"face|bust|body|back","angle":"front|three-quarter|profile|back",'
    '"expression":"one word"}. framing=face for a close-up of the head, bust for upper body, '
    "body for full body, back if seen from behind. Output the JSON only.")
