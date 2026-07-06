"""Pure module: face-dataset variation catalog, composition presets, vision prompts.

No DB, no Flask -> trivially unit-tested. The catalog drives the Klein fan-out;
the presets target a balanced training composition (see the design spec).
"""
from __future__ import annotations

import re

# Verrou d'identité renforcé (deep-research 2026-06-14, source primaire Google AI) :
# nommer les traits + interdire l'embellissement améliore la cohérence du visage.
# NB : la qualité de la photo de référence reste le facteur déterminant.
IDENTITY_GUARD = (
    "This is the SAME person as the reference image. Preserve their facial identity "
    "EXACTLY: same eye shape and color, nose, jawline, lips, skin tone and texture, "
    "and face proportions. Do NOT beautify, slim, age, or alter the face. "
    "SFW, realistic photographic portrait.")

# Variante multi-références (Nano Banana) : avec un guard au singulier le modèle
# peut s'ancrer sur une seule image ; on lui dit EXPLICITEMENT que toutes les refs
# montrent la même personne et qu'il doit s'appuyer sur chacune d'elles.
IDENTITY_GUARD_MULTI = (
    "ALL the reference images show the SAME person (different angles, expressions or "
    "framings). Use EVERY reference image together to lock the identity. Preserve their "
    "facial identity EXACTLY: same eye shape and color, nose, jawline, lips, skin tone "
    "and texture, and face proportions. Do NOT beautify, slim, age, or alter the face. "
    "SFW, realistic photographic portrait.")


def wrap_variation(prompt: str, ref_count: int = 1) -> str:
    guard = IDENTITY_GUARD_MULTI if ref_count > 1 else IDENTITY_GUARD
    return f"{guard} {prompt}"


def _e(i, axis, framing, label, prompt, co=False, cb=False, aspect=None):
    return {'id': i, 'axis': axis, 'framing': framing, 'label': label,
            'prompt': prompt, 'changes_outfit': co, 'changes_bg': cb, 'aspect': aspect}


VARIATION_CATALOG = [
    _e('face_front_neutral', 'expression', 'face', 'Visage face, neutre',
       'close-up portrait, front view, neutral expression, soft light, plain neutral background', cb=True),
    _e('face_front_smile', 'expression', 'face', 'Visage face, sourire',
       'close-up portrait, front view, slight smile, soft window light, blurred home interior background', cb=True),
    _e('face_34l_smile', 'angle', 'face', 'Visage 3/4 gauche, sourire',
       'close-up portrait, three-quarter left view, smiling'),
    _e('face_34l_serious', 'angle', 'face', 'Visage 3/4 gauche, serieux',
       'close-up portrait, three-quarter left view, serious expression'),
    _e('face_34r_laugh', 'angle', 'face', 'Visage 3/4 droite, rire',
       'close-up portrait, three-quarter right view, laughing'),
    _e('face_34r_soft', 'angle', 'face', 'Visage 3/4 droite, doux',
       'close-up portrait, three-quarter right view, gentle expression'),
    _e('face_profile_l', 'angle', 'face', 'Profil gauche',
       'close-up portrait, left profile view, neutral'),
    _e('face_profile_r', 'angle', 'face', 'Profil droite',
       'close-up portrait, right profile view, neutral'),
    _e('face_window', 'lighting', 'face', 'Visage, lumiere fenetre',
       'close-up portrait, front view, soft window light, blurred background', cb=True),
    _e('face_studio', 'lighting', 'face', 'Visage, studio',
       'close-up portrait, studio lighting, plain background', cb=True),
    _e('face_golden', 'lighting', 'face', 'Visage, golden hour',
       'close-up portrait, three-quarter view, warm golden hour light, outdoor', cb=True),
    _e('face_surprise', 'expression', 'face', 'Visage, surprise',
       'close-up portrait, front view, surprised expression'),
    _e('face_look_up', 'angle', 'face', 'Visage, regard haut',
       'close-up portrait, looking slightly upward, soft daylight, outdoor blurred background', cb=True),
    _e('face_look_down', 'angle', 'face', 'Visage, regard bas',
       'close-up portrait, looking slightly downward, pensive, indoor blurred background', cb=True),
    _e('bust_front', 'framing', 'bust', 'Buste face',
       'upper body portrait, front view, neutral, casual top', co=True, cb=True),
    _e('bust_34', 'framing', 'bust', 'Buste 3/4',
       'upper body portrait, three-quarter view, smiling, different outfit, indoor', co=True, cb=True),
    _e('bust_outdoor', 'background', 'bust', 'Buste exterieur',
       'upper body portrait, front view, outdoor park background', cb=True),
    _e('bust_studio', 'background', 'bust', 'Buste studio',
       'upper body portrait, three-quarter view, studio backdrop', cb=True),
    _e('bust_jacket', 'outfit', 'bust', 'Buste, veste',
       'upper body portrait, wearing a jacket, urban background', co=True, cb=True),
    _e('bust_evening', 'outfit', 'bust', 'Buste, tenue soiree',
       'upper body portrait, elegant evening outfit, dim ambient light', co=True, cb=True),
    _e('body_stand_front', 'framing', 'body', 'Corps debout face',
       'full body shot, standing, front view, casual clothes, street', co=True, cb=True),
    _e('body_stand_34', 'framing', 'body', 'Corps debout 3/4',
       'full body shot, standing, three-quarter view, different outfit, outdoor', co=True, cb=True),
    _e('body_sit', 'framing', 'body', 'Corps assis',
       'full body shot, sitting on a chair, relaxed, indoor', co=True, cb=True),
    _e('body_walk', 'framing', 'body', 'Corps en marche',
       'full body shot, walking, dynamic pose, city background', co=True, cb=True),
    _e('body_cafe', 'background', 'body', 'Corps, cafe',
       'full body shot, standing in a cafe, warm light', co=True, cb=True),
    _e('body_beach', 'background', 'body', 'Corps, plage (habille)',
       'full body shot, standing on a beach, summer casual clothes, daylight', co=True, cb=True),
    _e('back_34', 'framing', 'back', 'Dos 3/4',
       'full body shot, three-quarter back view, showing hairstyle and silhouette', co=True, cb=True),
    _e('body_wide_env', 'framing', 'body', 'Corps, plan large urbain',
       'full body shot, wide environmental framing, subject off-center, lots of background, urban plaza',
       co=True, cb=True, aspect='16:9'),
    _e('body_walk_wide', 'framing', 'body', 'Corps en marche, large',
       'full body shot, walking across a wide street, dynamic, cinematic wide framing',
       co=True, cb=True, aspect='16:9'),
    _e('body_land_outdoor', 'framing', 'body', 'Corps, paysage exterieur',
       'full body shot, standing outdoors, wide natural landscape background, daylight',
       co=True, cb=True, aspect='4:3'),
    _e('body_sit_terrace', 'framing', 'body', 'Corps assis, terrasse large',
       'full body shot, sitting on a cafe terrace, wide framing, warm light',
       co=True, cb=True, aspect='4:3'),
    _e('body_field_wide', 'framing', 'body', 'Corps, champ large',
       'full body shot, standing in an open field, wide nature background, soft daylight',
       co=True, cb=True, aspect='16:9'),
    _e('bust_land', 'framing', 'bust', 'Buste, cadre paysage',
       'upper body portrait, landscape framing, environment visible on the sides, outdoor',
       cb=True, aspect='4:3'),
    # Gros plans VISAGE en formats variés (preset visage-centré) : la robustesse de
    # format sur le visage lui-même, sans plan corps (corps reste générique).
    _e('face_land', 'framing', 'face', 'Visage, cadre paysage',
       'close-up portrait, three-quarter view, landscape framing, face to one side with environment, outdoor',
       cb=True, aspect='4:3'),
    _e('face_tall', 'framing', 'face', 'Visage, cadre vertical',
       'close-up portrait, front view, tall vertical framing, head and shoulders, soft natural light',
       cb=True, aspect='9:16'),
    _e('face_wide', 'framing', 'face', 'Visage, cadre cinema',
       'close-up portrait, wide cinematic framing, face off-center, blurred background',
       cb=True, aspect='16:9'),
]

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
_PRESETS = {'balanced_25': _BALANCED_25, 'zimage_12': _ZIMAGE_12,
            'balanced_multiformat': _BALANCED_MULTIFORMAT, 'face_focused': _FACE_FOCUSED,
            'fullbody_focused': _FULLBODY_FOCUSED}
_BY_ID = {e['id']: e for e in VARIATION_CATALOG}


def select_preset(name: str):
    return [_BY_ID[i] for i in _PRESETS.get(name, []) if i in _BY_ID]


def prompt_by_label(label):
    """Raw catalog prompt for a display label (fallback for pre-migration rows)."""
    return next((e['prompt'] for e in VARIATION_CATALOG if e['label'] == label), None)


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
    par son label → son override ; label inconnu → fallback cadrage."""
    e = next((x for x in VARIATION_CATALOG if x['label'] == label), None)
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


# Dataset CONCEPT (logique INVERSÉE) : l'invariant du set n'est plus l'identité mais
# l'acte/effet récurrent qu'on OMET pour qu'il se lie au trigger. On décrit donc tout —
# personnes, pose, cadrage, lumière, décor — SAUF l'acte central répété. Le captioneur
# ne connaît pas le concept : on lui demande d'ignorer l'action explicite dominante et
# de décrire le contexte autour. Aucun post-filtre d'identité (on GARDE l'identité).
CAPTION_PROMPT_CONCEPT = (
    "Caption Type: Straightforward.\n\n"
    "This is a CONCEPT training image: the images in this set all share ONE recurring "
    "explicit action, act, or visual effect. That single shared element is the concept "
    "being learned and must NEVER be named or described — do NOT write the act, the body "
    "fluids, the penetration, the specific sexual/graphic focal detail, or any word that "
    "labels what is happening at the focal point. Describe everything AROUND it instead.\n\n"
    "You MUST describe, in full and freely (identity is NOT hidden here): the people "
    "present and their appearance (hair, face, body, skin), their pose and body position, "
    "their expression and gaze, any clothing or state of undress and its colours, the "
    "setting or location, the camera angle and framing (close-up, three-quarter, "
    "full-body, from above, from below, point of view), and the lighting and mood.\n\n"
    "Output ONE caption as flowing natural-language prose, beginning with the shot type "
    "and framing, then the people and pose, then expression, then clothing/setting, then "
    "lighting and mood — but leaving the shared concept itself UNSPOKEN. Output only the "
    "caption itself - no preamble, no \"Here is\", no quotation marks, no commentary.")


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


def caption_has_identity_leak(caption) -> bool:
    """True si la caption mentionne un VRAI trait d'identite. Detecteur SEUL (badge)."""
    return bool(caption and _IDENTITY_LEAK.search(caption))


# Post-filtre : drop les PHRASES decrivant un trait d'identite. Avec le prompt
# "Straightforward", la rare fuite est isolee dans sa propre phrase -> suppression
# propre (pas de casse grammaticale). NE drop PAS expression ("eyes closed") ni
# lumiere ("shadow on the face").
_DROP_SENT = re.compile(
    r'\bhair\b|\bcomplexion\b|\bfreckles?\b|\bjawline\b|\beyebrows?\b|\bfacial\s+features?\b'
    r'|\bskin\s+(?:tone|texture)\b', re.I)


def drop_identity_sentences(caption) -> str:
    """Retire les phrases d'identite isolees d'une caption (post-captioning)."""
    parts = re.split(r'(?<=[.!?])\s+', caption or '')
    kept = [s for s in parts if s.strip() and not _DROP_SENT.search(s)]
    return ' '.join(kept).strip()


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


def _is_identity_tag(tag) -> bool:
    t = (tag or '').strip().lower().replace(' ', '_')
    if not t:
        return False
    if t in _IDENTITY_TAG_EXACT:
        return True
    if 'eyes' in t:  # garde l'EXPRESSION (closed_eyes, wink), drop la couleur (blue_eyes)
        return not any(k in t for k in ('closed', 'wink', 'half'))
    return any(sub in t for sub in _IDENTITY_TAG_CONTAINS)


def drop_identity_tags(caption) -> str:
    """Retire les tags booru d'identité d'une caption en liste de tags (mode booru),
    pendant booru de drop_identity_sentences (mode prose)."""
    if not caption:
        return ''
    kept = [t.strip() for t in caption.split(',') if t.strip() and not _is_identity_tag(t)]
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

CLASSIFY_PROMPT = (
    "Classify this portrait photo. Output ONLY a minified JSON object: "
    '{"framing":"face|bust|body|back","angle":"front|three-quarter|profile|back",'
    '"expression":"one word"}. framing=face for a close-up of the head, bust for upper body, '
    "body for full body, back if seen from behind. Output the JSON only.")
