// react-frontend/src/components/dataset/studio/loraStack.js
/**
 * Logique PURE du mode « pile » (🧬 Blend) du Test Studio — extraite du JSX pour
 * être testable sous `node --test` (le runner ne parse pas le JSX). Le ◉ LoRA
 * Canvas monte ce MÊME module (cf. utils/canvasGeneration) : un seul clamp, une
 * seule règle « ≥2 LoRA d'une seule famille », donc les deux écrans ne peuvent
 * pas répondre deux choses.
 *
 * ⚠️ Le mode s'affiche « 🧬 Blend » depuis le 03/08/2026 et s'affichait
 * « 🧬 Combine » avant. La valeur du mode, la clé d'API et les noms exportés
 * ici gardent le mot `combine` EXPRÈS : ils sont stockés (localStorage) ou
 * publics (POST), et un libellé ne renomme pas une donnée.
 *
 * Deux modes, une seule sélection de LoRA :
 *   - 'compare' : chaque LoRA coché est testé SEUL, une colonne par LoRA
 *                 (comportement historique) ; l'axe strengths balaye chacun.
 *   - 'combine' : (affiché « 🧬 Blend ») les LoRA cochés sont chargés ENSEMBLE
 *                 dans la même génération, chacun à SON poids ; l'axe strengths
 *                 n'a plus de sens et disparaît de l'UI comme du payload.
 *
 * Les poids vivent hors de la sélection (le LoraPicker n'en connaît pas) : une map
 * `{ "<dataset_id>:<checkpoint>": poids }`, pour qu'un poids réglé survive au
 * décochage/recochage d'un autre LoRA et à un changement de checkpoint.
 */

/* Extension spelled out: this module is imported directly by `node --test`,
   whose ESM resolver does not add it the way Vite does. */
import { runCost } from './runCost.js';

export const COMBINE_MIN_WEIGHT = 0;
/* Plafond d'un poids de blend.
 *
 * 2.0 pendant longtemps, et c'était un plafond de CONFORT déguisé en limite
 * technique : rien côté ComfyUI n'interdit de charger un LoRA au-delà, et les
 * usages où ça sert existent (un LoRA entraîné faible qu'on pousse, un style
 * qu'on veut écrasant, un slider poussé à fond). Le curseur s'arrêtait donc
 * avant la zone que les gens allaient chercher à la main dans un workflow.
 *
 * 5.0 est le nouveau plafond, et il reste un plafond : au-delà, ce n'est plus
 * « fort », c'est du bruit — et une valeur illisible/aberrante doit encore être
 * ramenée dans une plage que le backend accepte. ⚠️ Le MÊME nombre est écrit
 * dans backend/app/services/lora_test_studio.COMBINE_MAX_WEIGHT : les deux
 * clampent, et deux plafonds différents feraient mentir le curseur (l'UI
 * annoncerait 4.5, le serveur rendrait 2.0 sans rien dire).
 *
 * La zone > 2 est délibérément atteignable au CURSEUR *et* au clavier : le
 * champ à droite accepte la saisie libre au centième, comme les axes du Test
 * Studio, parce qu'un pas de 0.05 pour aller de 1 à 5 est une invitation à
 * abandonner. */
export const COMBINE_MAX_WEIGHT = 5;
export const COMBINE_DEFAULT_WEIGHT = 1;

/** Une saisie CLAVIER de poids, ramenée dans la plage — ou `null` si elle ne
 *  veut rien dire encore.
 *
 *  `null` n'est pas un échec : c'est « ne touche pas au poids ». Un champ vidé
 *  pour être retapé, un `-` seul, un `1.` en cours de frappe passent tous par
 *  ici, et remplacer ça par 0 (ou par 1) ferait sauter le curseur sous les
 *  doigts à chaque caractère. Seule une valeur finie est clampée et rendue. */
export function clampBlendWeight(raw) {
  const n = Number(raw);
  if (raw === '' || raw == null || !Number.isFinite(n)) return null;
  return Math.round(Math.min(COMBINE_MAX_WEIGHT, Math.max(COMBINE_MIN_WEIGHT, n)) * 100) / 100;
}

/** Clé stable d'un LoRA sélectionné dans la map de poids. */
export const stackKey = (sel) => `${sel?.dataset_id}:${sel?.checkpoint}`;

/** Poids retenu pour une sélection : clampé 0..COMBINE_MAX_WEIGHT, arrondi au
 *  centième, 1 par défaut. */
export function stackWeight(weights, sel) {
  const raw = Number((weights || {})[stackKey(sel)]);
  if (!Number.isFinite(raw)) return COMBINE_DEFAULT_WEIGHT;
  return Math.round(Math.min(COMBINE_MAX_WEIGHT, Math.max(COMBINE_MIN_WEIGHT, raw)) * 100) / 100;
}

/**
 * Le mode combine exige ≥2 LoRA d'une MÊME famille. Retourne null si tout va bien,
 * sinon le message (anglais) à afficher — le backend refuse aussi, mais l'utilisateur
 * doit le savoir AVANT de dépenser du GPU.
 */
export function combineBlocker(selection) {
  const sel = selection || [];
  if (sel.length < 2) return 'Check at least two LoRAs to blend them.';
  const families = [...new Set(sel.map((s) => s.family || s.train_type || 'zimage'))];
  if (families.length > 1) {
    return `Blending needs one family: ${families.join(' + ')} use different base `
      + 'models and workflows. Uncheck one of them.';
  }
  return null;
}

/**
 * Corps `selections` du POST /api/studio/run. En combine, chaque entrée porte son
 * `weight` (le backend chaîne les LoRA au-delà du premier dans le même graphe et
 * injecte TOUS les triggers) ; en comparaison le payload reste celui d'avant, sans
 * `weight`, pour ne rien changer aux runs existants.
 */
export function buildSelectionsPayload(selection, { combine = false, weights = {}, sets = {} } = {}) {
  return (selection || []).map((s) => {
    if (!combine) return { dataset_id: s.dataset_id, checkpoint: s.checkpoint };
    const list = stackWeightList(weights, sets, s);
    return {
      dataset_id: s.dataset_id,
      checkpoint: s.checkpoint,
      // `weight` (scalaire) reste envoyé EN PLUS de `weights` : cette app se met à
      // jour par `git pull`, donc un frontend neuf peut tourner quelques minutes
      // contre un backend qui ne connaît pas encore le balayage. Il rend alors la
      // tête de liste — une image au lieu de N, dégradé et juste, pas cassé.
      weight: list[0],
      weights: list,
    };
  });
}

/* ---------------------------------------------------------------------------
   🧬 BLEND SWEEP — plusieurs poids COCHÉS par LoRA → un lot de combinaisons.

   Le curseur donne UN poids par LoRA : une pile = une image, et comparer
   « 0.8/0.6 » à « 0.6/0.8 » demandait deux lancements et deux attentes. Les
   cases de poids remplacent ça par un balayage : chaque LoRA porte un ENSEMBLE
   de poids, et le lancement rend le PRODUIT CARTÉSIEN des combinaisons, une
   cellule chacune, dans un seul run.

   Règle du curseur, explicite parce qu'elle décide de tout le reste :
   AUCUNE case cochée = le curseur gouverne (comportement d'avant, au centième
   près, valeurs hors grille comprises). Dès qu'une case est cochée, ce sont LES
   CASES qui gouvernent — une seule case = une seule configuration, donc
   exactement une image, comme avant.

   Pas de plafond dur, délibérément : `build_matrix` côté serveur porte la même
   règle écrite (« PAS de plafond sur le nombre de cellules : la file est
   sérielle et l'utilisateur voit le compte + l'estimation de durée avant de
   lancer »). Un blend sweep est un sweep comme l'axe strengths, il obéit donc à
   la même règle plutôt qu'à une seconde. Au-delà de BLEND_WARN_CELLS on AVERTIT
   (le coût est réel), on n'interdit pas. Passer à un plafond dur = utiliser
   cette constante comme borne au lieu d'un seuil, à UN endroit.
   --------------------------------------------------------------------------- */

/** Grille de cases proposée sous chaque curseur. Des valeurs rondes qui couvrent
 *  la plage utile d'un blend ; le curseur reste là pour le hors-grille. */
export const BLEND_WEIGHT_CHIPS = [0.4, 0.6, 0.8, 1.0];

/** Seuil d'AVERTISSEMENT (pas de refus) sur le nombre d'images d'un lancement.
 *  Aligné sur `MAX_TEST_IMAGES` du backend, qui n'était jusqu'ici qu'un nombre
 *  annoncé au frontend sans que rien ne le vérifie. */
export const BLEND_WARN_CELLS = 24;

/** Les poids COCHÉS d'une sélection : clampés, arrondis, dédupliqués, triés.
 *  Vide = aucune case cochée. */
export function stackWeightSet(sets, sel) {
  const raw = (sets || {})[stackKey(sel)];
  if (!Array.isArray(raw)) return [];
  const out = [];
  for (const v of raw) {
    const n = Number(v);
    if (!Number.isFinite(n)) continue;
    const w = Math.round(Math.min(COMBINE_MAX_WEIGHT, Math.max(COMBINE_MIN_WEIGHT, n)) * 100) / 100;
    if (!out.includes(w)) out.push(w);
  }
  return out.sort((a, b) => a - b);
}

/** Les poids que CE LoRA va réellement balayer : ses cases si elle en a, sinon
 *  la valeur du curseur. Jamais vide — un LoRA contribue toujours un poids. */
export function stackWeightList(weights, sets, sel) {
  const chosen = stackWeightSet(sets, sel);
  return chosen.length ? chosen : [stackWeight(weights, sel)];
}

/** Toutes les combinaisons de poids, produit cartésien dans l'ordre de la
 *  sélection, le DERNIER LoRA variant le plus vite (l'ordre de lecture d'un
 *  tableau : on parcourt les poids du second en gardant le premier). Chaque
 *  combinaison est un tableau de poids aligné sur `selection`. */
export function blendCombinations(selection, { weights = {}, sets = {} } = {}) {
  const lists = (selection || []).map((s) => stackWeightList(weights, sets, s));
  if (!lists.length) return [];
  return lists.reduce((acc, list) => acc.flatMap((combo) => list.map((w) => [...combo, w])), [[]]);
}

/** Combien de configurations le lancement va rendre (hors seeds et axe batch). */
export function blendConfigCount(selection, opts = {}) {
  return (selection || []).length
    ? (selection || []).reduce(
      (n, s) => n * stackWeightList(opts.weights || {}, opts.sets || {}, s).length, 1)
    : 0;
}

const fmtWeight = (w) => {
  const n = Number(w);
  if (!Number.isFinite(n)) return '?';
  return String(Math.round(n * 100) / 100);
};

/** L'étiquette d'UNE combinaison, telle qu'elle est montrée sur sa cellule :
 *  « margot 0.8 × telegram 0.6 ». Le nom vient de la sélection (`lora_label`),
 *  le poids de la combinaison — les deux surfaces alimentent `lora_label`, donc
 *  l'étiquette est fabriquée ici et pas deux fois. */
export function blendComboLabel(selection, combo) {
  return (selection || [])
    .map((s, i) => `${s.lora_label || s.checkpoint || `LoRA ${i + 1}`} ${fmtWeight((combo || [])[i])}`)
    .join(' × ');
}

/** Ce que le bouton doit dire du COÛT avant de lancer, et quand s'en inquiéter.
 *  `warn` ne bloque JAMAIS : il colore et il chiffre. */
export function blendSweepCost({ configCount, count = 1, batchMult = 1,
  secondsPerImage = null }) {
  const cells = Math.max(0, Number(configCount) || 0)
    * Math.max(1, Number(count) || 0) * Math.max(1, Number(batchMult) || 1);
  // Même estimation que SeedControls : deux compteurs qui annonceraient deux
  // durées pour un seul lancement seraient pires que zéro. Les deux passent
  // désormais par `runCost`, donc par le rythme MESURÉ de la machine quand il
  // existe — le « 12 s » d'avant était le chiffre d'une 4090 servi à tout le
  // monde, et il mentait d'un facteur cinq sur une carte lente.
  const cost = runCost(cells, secondsPerImage);
  return {
    configs: Math.max(0, Number(configCount) || 0),
    cells,
    warn: cells > BLEND_WARN_CELLS,
    measured: cost.measured,
    label: cost.label,
    minutes: Math.ceil(cost.seconds / 60),
  };
}

/**
 * Nombre de cellules annoncé AVANT lancement. Une pile combinée valait UNE
 * configuration ; avec des cases de poids elle en vaut `configCount` (le produit
 * cartésien), d'où `configCount × count × batchMult`. `configCount` absent =
 * 1 configuration, c'est-à-dire exactement le comportement d'avant les cases.
 */
/* `axisTotal` = le produit des axes de rendu (CFG × steps × 2e passe) que le
   panneau propose désormais aussi en comparaison/blend. 1 par défaut : un appelant
   qui ne balaye aucun de ces axes obtient le compte d'avant, à l'identique. */
export function cellCount({ selectionCount, strengthCount, count, batchMult = 1,
  combine = false, configCount = 1, axisTotal = 1 }) {
  const n = Math.max(0, Number(count) || 0);
  const mult = Math.max(1, Number(batchMult) || 1);
  const axes = Math.max(1, Number(axisTotal) || 1);
  if (combine) {
    return selectionCount >= 2
      ? Math.max(0, Number(configCount) || 0) * n * mult * axes : 0;
  }
  return Math.max(0, selectionCount) * Math.max(0, strengthCount) * n * mult * axes;
}
