/** English display map for the backend's French variation catalog labels
 * (backend/app/services/face_variations.py — VARIATION_CATALOG).
 *
 * The French strings are DB/lookup keys (FaceDatasetImage.variation_label,
 * prompt_by_label()) and must never change; this map is display-only, used
 * where a label is rendered/announced in the UI.
 */
const LABELS = {
  'Visage face, neutre': 'Face front, neutral',
  'Visage face, sourire': 'Face front, smile',
  'Visage 3/4 gauche, sourire': 'Face three-quarter left, smile',
  'Visage 3/4 gauche, serieux': 'Face three-quarter left, serious',
  'Visage 3/4 droite, rire': 'Face three-quarter right, laughing',
  'Visage 3/4 droite, doux': 'Face three-quarter right, soft expression',
  'Profil gauche': 'Left profile',
  'Profil droite': 'Right profile',
  'Profil gauche, sourire': 'Left profile, smile',
  'Profil droite, sourire': 'Right profile, smile',
  'Profil gauche, serieux': 'Left profile, serious',
  'Profil droite, serieux': 'Right profile, serious',
  'Profil gauche, regard haut': 'Left profile, looking up',
  'Profil droite, regard haut': 'Right profile, looking up',
  'Profil gauche, lumiere cinema': 'Left profile, cinematic rim light',
  'Profil droite, lumiere cinema': 'Right profile, cinematic rim light',
  'Visage, lumiere fenetre': 'Face, window light',
  'Visage, studio': 'Face, studio lighting',
  'Visage, golden hour': 'Face, golden hour',
  'Visage, surprise': 'Face, surprised',
  'Visage, regard haut': 'Face, looking up',
  'Visage, regard bas': 'Face, looking down',
  'Buste face': 'Bust, front',
  'Buste 3/4': 'Bust, three-quarter',
  'Buste exterieur': 'Bust, outdoor',
  'Buste studio': 'Bust, studio',
  'Buste, veste': 'Bust, jacket',
  'Buste, tenue soiree': 'Bust, evening outfit',
  'Corps debout face': 'Body standing, front',
  'Corps debout 3/4': 'Body standing, three-quarter',
  'Corps assis': 'Body sitting',
  'Corps en marche': 'Body walking',
  'Corps, cafe': 'Body, café',
  'Corps, plage (habille)': 'Body, beach (clothed)',
  'Dos 3/4': 'Back, three-quarter',
  'Corps, plan large urbain': 'Body, wide urban shot',
  'Corps en marche, large': 'Body walking, wide shot',
  'Corps, paysage exterieur': 'Body, outdoor landscape',
  'Corps assis, terrasse large': 'Body sitting, wide terrace',
  'Corps, champ large': 'Body, wide open field',
  'Buste, cadre paysage': 'Bust, landscape framing',
  'Visage, cadre paysage': 'Face, landscape framing',
  'Visage, cadre vertical': 'Face, vertical framing',
  'Visage, cadre cinema': 'Face, cinematic wide framing',
  // NSFW catalog (local Klein only)
  'Buste, lingerie': 'Bust, lingerie',
  'Buste, topless': 'Bust, topless',
  'Buste, serviette': 'Bust, bath towel',
  'Corps, lingerie debout': 'Body standing, lingerie',
  'Corps, nu debout': 'Body standing, nude',
  'Corps, nu trois-quarts': 'Body three-quarter, nude',
  'Corps, nu assis lit': 'Body sitting on bed, nude',
  'Corps, nu allongé': 'Body lying down, nude',
  'Corps, nu douche': 'Body in the shower, nude',
  'Dos, nu': 'Back, nude',
};

/** Maps a backend catalog label to its English display text. Falls back to
 * the raw value (or '') for unknown/legacy/empty labels, so pre-migration
 * DB rows still render instead of going blank. */
export function displayLabel(label) {
  return LABELS[label] || label || '';
}
