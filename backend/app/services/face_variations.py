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


def wrap_variation_klein(prompt: str, nsfw: bool = False, framing: str | None = None) -> str:
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
    the SFW clamp and allows explicit nudity with natural anatomy."""
    detail = _KLEIN_FRAMING_DETAIL.get(framing or '', '')
    ending = ("Explicit nudity is allowed; render natural, anatomically correct forms. "
              "Professional realistic photograph.") if nsfw else \
             "Professional realistic photograph, SFW."
    return (
        f"Create a new photograph of the same person as the reference image: {prompt}. "
        + (f"{detail} " if detail else "")
        + "Restage the shot to match this description — change the pose, camera angle and "
          "framing accordingly; do not copy the composition of the reference image. "
          "Keep the facial identity exactly the same: same eye shape and color, nose, "
          "jawline, lips, skin tone and texture, and face proportions. Do not beautify "
          "or alter the face. Sharp focus, natural skin texture with visible pores, "
          f"realistic lighting with soft shadows, high detail. {ending}")


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
    # --- Body emphasis (fidélité corps) : silhouette RÉELLEMENT visible mais dans
    # le registre AUTORISÉ des moteurs API (vêtements ajustés, maillot de bain en
    # contexte plage/piscine, tenue de sport, robe moulante, contre-jour). Pas de
    # contournement de filtre : pour du contenu explicite → Klein en local.
    _e('bust_fitted_top', 'outfit', 'bust', 'Buste, haut ajusté',
       'upper body portrait, fitted ribbed knit top, natural relaxed pose, soft indoor light',
       co=True, cb=True),
    _e('bust_summer_dress', 'outfit', 'bust', 'Buste, robe d\'été',
       'upper body portrait, fitted summer dress with thin straps, golden hour light, outdoor',
       co=True, cb=True),
    _e('bust_swim', 'outfit', 'bust', 'Buste, maillot (plage)',
       'upper body portrait, wearing a bikini top, sunny beach in the background, bright '
       'daylight, natural relaxed pose', co=True, cb=True),
    _e('body_bodycon', 'outfit', 'body', 'Corps, robe moulante',
       'full body shot, elegant fitted bodycon evening dress, standing, upscale hotel lobby, '
       'warm ambient light', co=True, cb=True),
    _e('body_athletic', 'outfit', 'body', 'Corps, tenue de sport',
       'full body shot, athletic sportswear, fitted leggings and sports top, gym setting, '
       'confident stance', co=True, cb=True),
    _e('body_swim_beach', 'outfit', 'body', 'Corps, bikini plage',
       'full body shot, wearing a bikini, standing on a sunny beach, natural relaxed pose, '
       'bright daylight', co=True, cb=True, aspect='3:4'),
    _e('body_swim_pool', 'outfit', 'body', 'Corps, maillot piscine',
       'full body shot, one-piece swimsuit, standing at the edge of a swimming pool, summer '
       'daylight', co=True, cb=True, aspect='3:4'),
    _e('body_jeans_fit', 'outfit', 'body', 'Corps, jean ajusté',
       'full body shot, fitted high-waisted jeans and tucked-in top, urban street, daylight',
       co=True, cb=True),
    _e('body_silhouette', 'lighting', 'body', 'Corps, silhouette contre-jour',
       'full body shot, backlit near a large window, figure outlined by rim light, elegant '
       'fitted dress, moody interior', co=True, cb=True),
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

# --- Catalogue NSFW (moteur Klein LOCAL uniquement) --------------------------
# Plans corps non censurés pour la fidélité corporelle : jamais envoyés aux
# moteurs API (route + service refusent), générés par le Klein local qui n'a pas
# de filtre. Le registre reste "état + pose + décor" (lingerie/topless/nu) — pas
# d'acte : c'est un dataset de PERSONNAGE, l'acte appartient au prompt d'usage.
# Le caption doit décrire l'état (nude/lingerie) pour qu'il reste promptable et
# ne se lie pas au trigger (principe d'inversion).
NSFW_VARIATION_CATALOG = [
    _e('nsfw_bust_lingerie', 'nsfw', 'bust', 'Buste, lingerie',
       'bust shot, wearing delicate lace lingerie, bedroom, soft window light',
       co=True, cb=True),
    _e('nsfw_bust_topless', 'nsfw', 'bust', 'Buste, topless',
       'bust shot, topless, bare chest, neutral indoor background, natural light',
       co=True, cb=True),
    _e('nsfw_bust_towel', 'nsfw', 'bust', 'Buste, serviette',
       'bust shot, wrapped in a bath towel, bare shoulders, bathroom, soft light',
       co=True, cb=True),
    _e('nsfw_body_lingerie', 'nsfw', 'body', 'Corps, lingerie debout',
       'full body shot, standing, matching lace lingerie set, bedroom interior, soft light',
       co=True, cb=True, aspect='3:4'),
    _e('nsfw_body_nude_stand', 'nsfw', 'body', 'Corps, nu debout',
       'full body shot, standing fully nude, natural anatomy, relaxed pose, neutral studio '
       'background, soft even light', co=True, cb=True, aspect='3:4'),
    _e('nsfw_body_nude_34', 'nsfw', 'body', 'Corps, nu trois-quarts',
       'full body shot, three-quarter view, fully nude, natural anatomy, standing by a large '
       'window, soft daylight', co=True, cb=True, aspect='3:4'),
    _e('nsfw_body_nude_sit', 'nsfw', 'body', 'Corps, nu assis lit',
       'full body shot, sitting nude on the edge of a bed, relaxed natural pose, warm bedroom '
       'light', co=True, cb=True, aspect='3:4'),
    _e('nsfw_body_nude_lying', 'nsfw', 'body', 'Corps, nu allongé',
       'full body shot, lying nude on a bed on her side, natural anatomy, soft morning light',
       co=True, cb=True, aspect='4:3'),
    _e('nsfw_body_shower', 'nsfw', 'body', 'Corps, nu douche',
       'full body shot, nude in the shower, wet skin and hair, water droplets, glass and tile '
       'background', co=True, cb=True, aspect='9:16'),
    _e('nsfw_back_nude', 'nsfw', 'back', 'Dos, nu',
       'full body shot from behind, standing nude, back and buttocks visible, natural anatomy, '
       'neutral background', co=True, cb=True, aspect='3:4'),
]

_NSFW_LABELS = {e['label'] for e in NSFW_VARIATION_CATALOG}


def is_nsfw_label(label) -> bool:
    """True when a variation label belongs to the NSFW catalog or carries the 🔞
    custom-prompt prefix — drives the Klein-only guard and the NSFW wrapper on
    regeneration (the DB row only stores the label)."""
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


def select_preset(name: str):
    return [_BY_ID[i] for i in _PRESETS.get(name, []) if i in _BY_ID]


def prompt_by_label(label):
    """Raw catalog prompt for a display label (fallback for pre-migration rows).
    Searches the SFW catalog then the NSFW one (regenerate needs both)."""
    return next((e['prompt'] for e in VARIATION_CATALOG + NSFW_VARIATION_CATALOG
                 if e['label'] == label), None)


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
    e = next((x for x in VARIATION_CATALOG + NSFW_VARIATION_CATALOG
              if x['label'] == label), None)
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
