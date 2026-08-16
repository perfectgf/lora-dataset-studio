/* WHAT EACH BANK PASS IS, told in the three blocks its launch window shows.
 *
 * WHY THREE BLOCKS AND NOT ONE "SETTINGS" PANEL. A window called "settings for
 * this pass" would be a lie, and it was measured: of the 12 knobs in 🎚 Filter
 * thresholds, the 🔎 Scan quality pass reads exactly ONE (dup_distance) — its
 * measuring code (services/image_quality.py) does not even import the config.
 * The other eleven decide how the GRID IS SORTED AND FLAGGED, and they re-sort on
 * save without any pass running. Mixing the two in one window would teach people
 * that nudging a slider costs a rescan, which is the opposite of true.
 *
 * So every window says three separate things:
 *   THIS RUN                 — where it runs and how big that is (the scope).
 *   SETTINGS THIS PASS READS — only what the CALCULATION consumes, and where
 *                              those values live (they are global — the same for
 *                              every bank — unless said otherwise).
 *   NOT DECIDED HERE         — what is re-read when the grid is sorted or
 *                              flagged, retunable with no pass at all.
 *
 * AND A FOURTH THING WHEN IT IS TRUE: `caveats`, for what the scope does NOT
 * cover. 🔎 Scan's duplicate re-grouping always covers the whole bank whatever
 * the scope; 🎨 Medium also runs chained inside ✨ Score with the default scope.
 * Both are stated rather than left for someone to discover.
 *
 * SCOPES THAT WOULD LIE ARE OFFERED DISABLED, WITH THE REASON. ✨ Score, 👥 Group
 * by person and ✂ Find crops & variants each produce a PARTITION of the whole
 * bank, renumbered on every run: handed a subset they would number a
 * sub-population from 1 and drop those ids on top of unrelated groups already in
 * the database. A wrong group is worse than a missing option — but a missing
 * option is worse than a disabled one, because the maintainer concluded a shipped
 * feature did not exist twice over from exactly that kind of silence.
 *
 * Plain .js (no JSX) so `node --test` can execute all of it.
 */
import { PASS_SCOPE_OPTIONS } from './bankPassScope.js';
import { normalizeSemanticEngine, semanticEngineLabel } from './bankSemanticEngine.js';

/* The reason a whole-bank pass refuses a scope. One sentence, said the same way
   everywhere, because it is the same fact three times. */
const PARTITION_REASON = (what) =>
  `${what} is one numbering of the WHOLE bank, recomputed from scratch on every `
  + 'run. A pass handed part of the bank would number that part from 1 and land '
  + 'those ids on top of unrelated groups already saved — so this pass covers '
  + 'everything, always.';

const SEMANTIC_INDEX_REASON = 'The semantic index is one cache for the WHOLE Bank. '
  + 'Search, similarity, diversity and crops/variants must all read the same '
  + 'embedding space, so selections and status scopes do not apply.';

/**
 * The passes, keyed by the id used in `payload.pass_scopes` and in the tests.
 *
 * `scopes: true`  — every option in PASS_SCOPE_OPTIONS is offered.
 * `scopes: string`— none are: the string is the reason, shown next to the one
 *                   immutable THIS RUN line.
 * `selection`     — true, or the reason it is refused.
 * `redo`          — the "run it again on rows that already have a result" line,
 *                   which is where "Rescan all" went: it was never an action of
 *                   its own, it was a SCOPE wearing a button's clothes.
 * `countable`     — false when no honest number exists (see ✂ below).
 */
export const BANK_PASSES = {
  scan: {
    id: 'scan',
    label: '🔎 Scan quality',
    verb: '🔎 Scan',
    endpoint: 'scan',
    what: 'Measures sharpness, noise, flatness, size and effective detail, hashes '
      + 'every image, then groups the exact duplicates. CPU only.',
    scopes: true,
    selection: true,
    redo: {
      key: 'rescan',
      label: 'Also re-measure images that were already scanned',
      note: 'This is where “Rescan all” went. It was never a second pass — it was '
        + 'this scope, with a button of its own.',
    },
    settings: [
      { name: 'Duplicate distance (🎚 Filter thresholds ▸ dup_distance)',
        note: 'The only stored setting this pass reads, and it is read by the '
          + 'duplicate GROUPING at the end, not by the measuring.' },
    ],
    notHere: [
      'Sharpness / noise / flat / small / detail thresholds — those decide which '
        + 'measured images get FLAGGED and how the grid sorts. Change them and the '
        + 'grid re-sorts on save; no rescan is needed.',
    ],
    caveats: [
      'The scope chooses what is MEASURED. The duplicate grouping at the end of '
        + 'the pass always covers the whole bank, because it works from the stored '
        + 'hashes and renumbers all of them together.',
    ],
    binCost: 'a rejected image is re-read from disk and re-measured',
  },

  score: {
    id: 'score',
    label: '✨ Score',
    verb: '✨ Score',
    endpoint: 'score',
    what: 'Rates every image for aesthetics (1–10), flags NSFW and groups the bank '
      + 'by visual style — one CLIP pass, resumed from its cache.',
    scopes: PARTITION_REASON('The style grouping'),
    selection: PARTITION_REASON('The style grouping'),
    fixedScopeLine: 'The whole bank except the bin — every kept and undecided image.',
    redo: {
      key: 'rescore',
      label: 'Throw the cached embeddings away and recompute everything',
      note: 'For a new model, or scores you no longer trust. Costs a full pass; '
        + 'without it the run resumes and skips what is already computed.',
    },
    settings: [
      { name: 'Style threshold (🎚 Filter thresholds ▸ style_threshold)',
        note: 'Decides how tight the style groups are.' },
      { name: 'Bank scoring models folder + Python (Setup ▸ Quality tools)' },
      { name: 'Scoring device (CPU / GPU, resolved from this machine)',
        note: 'On the GPU the pass takes the exclusive window: ComfyUI is unloaded '
          + 'and training cannot start for its duration.' },
    ],
    notHere: [
      'Aesthetic minimum and NSFW maximum — they only decide which scored images '
        + 'get flagged and how the grid sorts. Retunable with no pass at all.',
    ],
    caveats: [],
  },

  semantic_index: {
    id: 'semantic_index',
    label: '🧠 Build semantic index',
    verb: '🧠 Build semantic index',
    endpoint: 'semantic-index',
    what: 'Builds or resumes the selected semantic engine cache for search, '
      + 'similarity, diversity, balanced sampling and crops/variants.',
    scopes: SEMANTIC_INDEX_REASON,
    selection: SEMANTIC_INDEX_REASON,
    fixedScopeLine: 'Every image in the Bank — one shared semantic space.',
    countable: false,
    redo: {
      key: 'rescan',
      explicit: true,
      label: 'Rebuild this engine’s semantic index from scratch',
      note: 'Explicit reindex only. The other engine cache and the CLIP data '
        + 'owned by ✨ Score are preserved.',
    },
    settings: [
      { name: 'This Bank’s selected semantic engine' },
      { name: 'Managed model + Python (Setup ▸ Quality tools)' },
    ],
    notHere: [
      'Which engine this Bank uses — choose it in Analyze before opening this window.',
      '✨ Score’s aesthetic, NSFW, visual-style and 🎨 Medium results — they stay on CLIP.',
    ],
    caveats: [
      'A normal run resumes and fills only missing or stale rows. Tick rebuild only '
        + 'when you deliberately want to recompute this engine’s entire cache.',
    ],
  },

  faces: {
    id: 'faces',
    label: '👥 Group by person',
    verb: '👥 Group',
    endpoint: 'faces',
    what: 'Embeds the dominant face of every image and clusters the bank by person '
      + '— no reference photo needed.',
    scopes: PARTITION_REASON('A person clustering'),
    selection: PARTITION_REASON('A person clustering'),
    fixedScopeLine: 'The whole bank except the bin, minus the folders you declared '
      + 'to be a single person (those are skipped — that is what the declaration '
      + 'is for).',
    redo: null,
    settings: [
      { name: 'Face threshold (🎚 Filter thresholds ▸ face_threshold)',
        note: 'How sure the detector must be before a face counts.' },
      { name: 'Face scoring models folder, Python and device (Setup ▸ Quality tools)' },
    ],
    notHere: [
      'Which folders count as “one person” — that is 👤 Single person here, on the '
        + 'subfolder panel, and it takes effect without a pass.',
    ],
    caveats: [],
  },

  angles: {
    id: 'angles',
    label: '⤢ Measure head angles',
    verb: '⤢ Measure',
    endpoint: 'angles',
    what: 'Re-runs the face detector on images that were face-scanned by a build '
      + 'that computed the head angle and threw it away. Writes the angle and '
      + 'nothing else.',
    scopes: true,
    selection: true,
    redo: null,
    settings: [
      { name: 'Face scoring models folder, Python and device (Setup ▸ Quality tools)' },
      { name: 'Time estimate — NOT configurable',
        note: 'The ~2 s per image the estimate uses is a measured constant in the '
          + 'code, not a setting.' },
    ],
    notHere: [
      'The ⤢ angle filter buckets (front / three-quarter / profile) read the '
        + 'measured value; they need no re-measure.',
    ],
    caveats: [
      'This lane has always reached the bin — its pool never filtered on status at '
        + 'all. Leaving the scope alone keeps that exactly; picking one narrows it '
        + 'for the first time.',
    ],
    binCost: 'the face detector runs again on each rejected image',
  },

  watermark: {
    id: 'watermark',
    label: '🚩 Find watermarks',
    verb: '🚩 Scan',
    endpoint: 'watermark',
    what: 'Looks for overlaid watermarks and logos, and locates them so the '
      + 'cleaning levels below can crop or repaint them.',
    scopes: true,
    selection: true,
    redo: {
      key: 'rescan',
      label: 'Also re-check images that were already scanned',
      note: 'Images you dismissed as “not watermarked” are never re-examined, '
        + 'whichever line you pick — that is why the two numbers here are not the '
        + 'pile size.',
    },
    settings: [],   // filled at render time: the ROUTE decides (see below)
    notHere: [
      'Which watermarked images get rejected, and the ✂ crop / 🖌 repaint levels — '
        + 'those are their own actions on the watermark panel.',
    ],
    caveats: [],
    binCost: 'each rejected image is put through the detector',
  },

  /* THE TWO LEVELS THAT PRODUCE A NEW IMAGE.
   *
   * Every other entry here computes a verdict: a wrong scope costs time. These
   * two build a cleaned copy of each image they touch — a real bank offered
   * "✂ Auto-crop (16 052)" and "🧽 Inpaint (16 507)" from a single click each,
   * with no way to say WHICH images. That is why they now open a window too.
   *
   * They are NOT in BANK_PASS_ORDER on purpose: their buttons live on the 🚩
   * Watermarks panel, inside the funnel they belong to. A second copy in the
   * pass row would read as two different actions.
   *
   * THEIR POOL IS NOT A PILE. Both act on flagged images that carry something to
   * act on — a stored box or a mask you drew — so a scope INTERSECTS that set and
   * can never widen it. The numbers on the scope lines come from that same pool,
   * per pile, measured server-side from the clause the run itself filters on.
   */
  watermark_crop: {
    id: 'watermark_crop',
    label: '✂ Auto-crop watermarks',
    verb: '✂ Crop',
    endpoint: 'watermark/crop',
    what: 'Cuts off the border strip holding the mark, on the flagged images whose '
      + 'mark sits in a border. CPU only — no model, no GPU, and no invented pixel.',
    scopes: true,
    selection: true,
    redo: null,
    settings: [],
    notHere: [
      'WHERE each mark sits — that is 🚩 Find watermarks, and this level only '
        + 'routes on the box it stored.',
      'Images whose mask you drew by hand in ▶ Review: a crop cannot express '
        + 'several zones, or a zone on the subject, so those are 🧽 Inpaint’s.',
    ],
    caveats: [
      'Your own files are never written to — that is what makes this safe to try. '
        + 'The crop lands in the bank’s own copy, and ↩ Undo cleaning throws every '
        + 'one of those copies away and re-flags the images. Two things undo does '
        + 'not reach: an image you already promoted (that copy was written into '
        + 'the dataset) and an image whose source file changed on disk since the '
        + 'clean — that one keeps its cleaned copy and has to be re-scanned.',
      'Undo is bank-wide, not per run: it restores every cleaned image, including '
        + 'ones cleaned before this one. It also drops the measurements taken from '
        + 'the cleaned pixels, so ✨ Score has to pass over those rows again.',
      'The count on each line is the pool this level WALKS, not what it will '
        + 'change: a mark that is not in a border stays flagged for 🧽 Inpaint. '
        + 'The number on the ✂ button itself is that narrower, routed figure.',
    ],
    binCost: 'each rejected image is decoded and re-encoded into the bank’s working copy',
  },

  watermark_inpaint: {
    id: 'watermark_inpaint',
    label: '🧽 Repaint watermarks',
    verb: '🧽 Repaint',
    endpoint: 'watermark/inpaint',
    what: 'Repaints the marks a crop cannot remove, on every image still flagged. '
      + 'LaMa is fast and invents little; Klein is slower and also clears a mark '
      + 'sitting ON the subject.',
    scopes: true,
    selection: true,
    redo: null,
    settings: [
      { name: 'Level 3 engine — LaMa or Klein (picked on this panel, for this run)' },
      { name: 'LaMa models folder, Python and device (Setup ▸ Quality tools)' },
      { name: 'Klein weights + ComfyUI (Setup ▸ Generation models), when Klein is picked',
        note: 'A bank has no dataset to inherit a Klein model from, so this pass '
          + 'resolves it automatically — the panel names the one that will run.' },
    ],
    notHere: [
      'WHERE each mark sits — 🚩 Find watermarks stores the box, and ▶ Review is '
        + 'where you redraw it.',
      'Whether a watermarked image is rejected at all — that is the 🚩 flag’s own '
        + 'chip in the grid.',
    ],
    caveats: [
      'Your own files are never written to. The repaint lands in the bank’s own '
        + 'copy, and ↩ Undo cleaning throws every one of those copies away and '
        + 're-flags the images. Two things undo does not reach: an image you '
        + 'already promoted (that copy was written into the dataset) and an image '
        + 'whose source file changed on disk since the clean — that one keeps its '
        + 'cleaned copy and has to be re-scanned.',
      'Undo is bank-wide, not per run: it restores every cleaned image, including '
        + 'ones cleaned before this one. It also drops the measurements taken from '
        + 'the cleaned pixels, so ✨ Score has to pass over those rows again.',
      'With LaMa, a mark ON the subject is left flagged rather than smeared — '
        + 'switch the engine to Klein for those. An emptied mask repaints nothing, '
        + 'on purpose.',
    ],
    binCost: 'each rejected image costs a full repaint, the slowest step of the funnel',
  },

  /* ✨ THE THIRD PASS THAT PRODUCES A NEW IMAGE, and the only one that spends
   * GPU-minutes PER IMAGE. Asked for by nofaceman on Discord, in the same breath
   * as the ✂ crop: the Bank holds the filtering and the curation, so being sent
   * to a Dataset just to upscale — and exported back out to curate the result —
   * is a detour around the screen you are already on.
   *
   * IT IS NOT THE DATASET PASS OF THE SAME NAME, in the one way that matters to
   * the person pressing the button: a dataset improve creates a separate
   * candidate you review next to the original, a BANK improve replaces what the
   * bank shows. A bank IS the review — so ↩ Revert is the "no thanks", exactly
   * like the watermark clean it sits next to.
   *
   * NOT IN BANK_PASS_ORDER, for the same reason as the two cleaning levels: its
   * button lives on the ✂ Edits panel, with the ↩ Revert that takes it back.
   */
  improve: {
    id: 'improve',
    label: '✨ Upscale & improve',
    verb: '✨ Improve',
    endpoint: 'improve',
    what: 'Re-renders each image at a higher resolution, in the Bank itself. Klein '
      + 'rewrites detail from a prompt (sharper, and skin and colour can shift); '
      + 'SeedVR2 resolves detail and leaves the original look alone. GPU, one '
      + 'ComfyUI round-trip per image.',
    scopes: true,
    selection: true,
    // No "do it again" lane on purpose: an improved image LEAVES the pool, and
    // the way back in is ↩ Revert — one click, and it says what it takes back.
    // A tick box here would be a second, quieter way to spend the same
    // GPU-minutes twice on the same image.
    redo: null,
    settings: [
      { name: 'Engine — Klein or SeedVR2 (picked on this panel, for this run)',
        note: 'The Settings default (Settings ▸ Engines) governs the single-✨ '
          + 'surfaces; a batch always states the engine on the button you press.' },
      { name: 'Klein weights + ComfyUI (Setup ▸ Generation models), when Klein runs',
        note: 'A bank has no dataset to inherit a Klein model from, so this pass '
          + 'resolves it automatically — the panel names the one that will run.' },
      { name: 'The improve instruction (Settings ▸ Engines ▸ Upscale & improve)',
        note: 'Klein only. SeedVR2 sends no instruction at all.' },
    ],
    notHere: [
      'WHICH images deserve it — that is the curation you already do in the grid, '
        + 'and it is why the scope below matters more here than anywhere else.',
      'The training resolution. A bank sits UPSTREAM of that choice: what an image '
        + 'is finally scaled to is decided when a dataset imports it.',
    ],
    caveats: [
      'Your own files are never written to. The result is a copy kept by the app, '
        + 'and ↩ Revert on the ✂ Edits panel throws it away and brings the original '
        + 'framing back — the rotation it absorbed included.',
      'It replaces what the Bank SHOWS for that image, so everything downstream — '
        + 'the grid, ▶ Review and what gets promoted — uses the improved version. '
        + 'There is no candidate to validate: the bank is the review.',
      'Every measurement taken from the old pixels is cleared, so ✨ Score, 📐 '
        + 'Framing and the rest pass over those images again. That is deliberate: '
        + 'scores read off the version before the upscale would describe an image '
        + 'this bank no longer holds.',
      'One ComfyUI round-trip per image, minutes apiece on a modest card. ⏹ Stop '
        + 'ends the run between two images and keeps what is already done.',
    ],
    binCost: 'each rejected image costs a full GPU render, the most expensive line in the app',
  },

  framing: {
    id: 'framing',
    label: '📐 Classify framing',
    verb: '📐 Classify',
    endpoint: 'framing',
    what: 'Tags each shot face / bust / body / back with the vision model, feeding '
      + 'the 📐 filter chips and the coverage advice.',
    scopes: true,
    selection: true,
    redo: {
      key: 'rescan',
      label: 'Also re-classify images that already have a framing',
    },
    settings: [
      { name: 'Vision model + Ollama URL (Settings ▸ Local tools)' },
      { name: 'Vision concurrency (Settings ▸ Local tools)' },
      { name: 'The prompt — NOT configurable',
        note: 'It is the same fixed classifier prompt the datasets use.' },
    ],
    notHere: [
      'The coverage advice and the framing target — they read the stored framing '
        + 'and recompute on the spot.',
    ],
    caveats: [],
    binCost: 'each rejected image costs one vision-model call',
  },

  medium: {
    id: 'medium',
    label: '🎨 Classify medium',
    verb: '🎨 Classify',
    endpoint: 'medium',
    what: 'Sorts scored images into photograph / anime / 3D render / illustration '
      + '— or an honest “unsure” — by reading the embeddings ✨ Score already '
      + 'cached. No image is looked at again and the GPU stays free.',
    scopes: true,
    selection: true,
    redo: {
      key: 'rescan',
      label: 'Also re-classify images that already have a verdict',
    },
    settings: [
      { name: 'The decision margins — NOT configurable',
        note: 'They are fixed constants in the code (0.005 for photographs, 0.030 '
          + 'for the rest). An empty block here would read as “we did not look”.' },
      { name: 'CLIP text encoder (its phrase cache is written once per install)' },
    ],
    notHere: [
      'The 🎨 medium chips re-read the stored verdict and its margin, so a borderline '
        + 'call can be reconsidered without running this again.',
    ],
    caveats: [
      '✨ Score runs this pass automatically when it finishes, always with the '
        + 'default scope — a scope picked here belongs to this run only and is '
        + 'stored nowhere.',
      // The dependency, stated where the scope is chosen. ✨ Score has NO scope
      // control at all (its style grouping is one numbering of the bank, so it
      // always runs on kept + undecided), which means the bin cannot be scored
      // by asking for it — the images there carry an embedding only if they
      // were scored BEFORE being rejected. Aiming this pass at the bin without
      // that is a run that classifies nothing, and the scope lines now count
      // exactly how many of the images they offer are in that state.
      'This pass reads the embeddings ✨ Score cached and computes none of its '
        + 'own. ✨ Score never runs on the bin, so a rejected image has an '
        + 'embedding only if it was scored before you rejected it — aimed at '
        + 'the bin, this pass answers for those and skips the rest.',
    ],
    binFree: true,
  },

  semantic_dedup: {
    id: 'semantic_dedup',
    label: '✂ Find crops & variants',
    verb: '✂ Find crops & variants',
    endpoint: 'semantic-dedup',
    what: 'Groups crops and re-compressed variants of the SAME shot that the exact '
      + 'hash misses, from this Bank’s selected semantic engine. No GPU after the index exists.',
    scopes: PARTITION_REASON('The “same shot” grouping'),
    selection: PARTITION_REASON('The “same shot” grouping'),
    fixedScopeLine: 'Every image the selected semantic engine has indexed — rejected '
      + 'ones included. That pool lives in the engine cache, so this window cannot '
      + 'put a number on it without inventing one.',
    countable: false,
    redo: null,
    settings: [
      { name: 'The selected engine’s same-shot threshold',
        note: 'CLIP and SigLIP 2 use separate values; their cosine scores are not '
          + 'numerically interchangeable.' },
    ],
    notHere: [
      'How a group is resolved (keep best / keep first) — that is the ✂ Same shot '
        + 'chip’s own action.',
    ],
    caveats: [],
  },

  caption: {
    id: 'caption',
    label: '🏷️ Caption',
    verb: '🏷️ Caption',
    endpoint: 'caption',
    what: 'Describes each image so it becomes searchable, and the description rides '
      + 'along when you promote it to a dataset.',
    scopes: true,
    selection: true,
    // Caption's "redo" is 🔄 Re-caption: a destructive second launch with its own
    // confirmation and its own two figures, so it is NOT a tick box here.
    redo: null,
    settings: [
      { name: 'Captioning engine (Settings ▸ Captioning & quality)',
        note: 'Overridden for this run by the Engine picker below.' },
      { name: 'Vision model + Ollama URL (Settings ▸ Local tools)',
        note: 'Overridden for this run by the Model picker below.' },
      { name: 'Vision concurrency (Settings ▸ Local tools)' },
    ],
    notHere: [
      'The 🔍 search and the 🏷️ tag chips read the captions that exist; they never '
        + 'ask for a pass.',
    ],
    caveats: [],
    binCost: 'each rejected image costs a full captioning call, the slowest of them all',
  },
};

/** Every pass id, in the order the panel lists its buttons. */
export const BANK_PASS_ORDER = ['scan', 'faces', 'score', 'medium', 'framing',
  'semantic_index', 'semantic_dedup', 'watermark', 'angles', 'caption'];

export function bankPass(passId, { semanticEngine = 'clip' } = {}) {
  const spec = BANK_PASSES[passId] || null;
  if (!spec || (passId !== 'semantic_index' && passId !== 'semantic_dedup')) return spec;
  const engine = normalizeSemanticEngine(semanticEngine);
  const label = semanticEngineLabel(engine);
  if (passId === 'semantic_index') {
    return {
      ...spec,
      label: `🧠 Build ${label} semantic index`,
      verb: `🧠 Build ${label} semantic index`,
      what: `Builds or resumes the whole-Bank ${label} cache for search, similarity, `
        + 'diversity, balanced sampling and crops/variants.',
      settings: [
        { name: `${label} — this Bank’s selected semantic engine` },
        ...spec.settings.slice(1),
      ],
    };
  }
  return {
    ...spec,
    what: `Groups crops and re-compressed variants of the SAME shot that the exact `
      + `hash misses, from the ${label} semantic index. No GPU after the index exists.`,
    fixedScopeLine: `Every image ${label} has indexed — rejected ones included. `
      + 'That pool lives in the engine cache, so this window cannot put a number '
      + 'on it without inventing one.',
    settings: engine === 'siglip2' ? [
      { name: 'SigLIP 2 same-shot threshold '
          + '(bank_semantic.siglip2_semantic_dup_threshold)',
        note: 'A separate conservative starting value, not a transplanted CLIP '
          + 'cutoff. Review the proposed groups; LDS has not yet calibrated a '
          + 'universal SigLIP 2 boundary across multiple real Banks.' },
    ] : [
      { name: 'CLIP same-shot threshold '
          + '(🎚 Filter thresholds ▸ semantic_dup_threshold)',
        note: 'Overridable for one run from the 🎚 panel.' },
      { name: 'Style threshold (🎚 Filter thresholds ▸ style_threshold)',
        note: 'CLIP-only comparison-blocking optimisation. Set looser than the '
          + 'same-shot threshold and the pass falls back to comparing everything.' },
    ],
    notHere: engine === 'siglip2'
      ? [...spec.notHere,
        'CLIP style clusters — SigLIP 2 never reuses them to block comparisons.']
      : spec.notHere,
  };
}

/**
 * Is this scope available for this pass, and why not?
 * @returns {{ok: boolean, reason: string}}
 *
 * NEVER returns "absent". An option a pass cannot honour is rendered disabled
 * with this reason next to it — the whole point of the disabled state is that the
 * user can see the capability was considered and read the objection, instead of
 * concluding the feature does not exist.
 */
export function passScopeAvailability(passId, scopeId) {
  const spec = bankPass(passId);
  if (!spec) return { ok: false, reason: 'Unknown pass.' };
  if (spec.scopes === true) return { ok: true, reason: '' };
  return { ok: false, reason: String(spec.scopes) };
}

/** Can a selection drive this pass, and why not? */
export function passSelectionAvailability(passId) {
  const spec = bankPass(passId);
  if (!spec) return { ok: false, reason: 'Unknown pass.' };
  if (spec.selection === true) return { ok: true, reason: '' };
  return { ok: false, reason: String(spec.selection) };
}

/** The scope lines a dialog draws: every option, each with its availability. */
export function passScopeRows(passId) {
  return PASS_SCOPE_OPTIONS.map((o) => ({
    ...o, ...passScopeAvailability(passId, o.id),
  }));
}

/** Which watermark route will actually run, and the settings IT reads.
 *
 *  The pass picks between two engines at launch, and the settings are not the
 *  same for both — so the window names the one that is going to run rather than
 *  listing both and letting the user guess which half applies. */
export function watermarkSettings(detectorReady) {
  if (detectorReady) {
    return {
      route: 'The dedicated watermark detector (SigLIP2) — installed, so this is '
        + 'the route this run will take. ~0.14 s per image.',
      settings: [
        { name: 'Detector threshold (Settings ▸ Captioning & quality ▸ watermark_detect)' },
        { name: 'Detector models folder, Python and device (Setup ▸ Quality tools)' },
      ],
    };
  }
  return {
    route: 'The vision model (Ollama) — the detector extra is not installed, so '
      + 'this run falls back to the route that has always worked. ~1.7 s per image.',
    settings: [
      { name: 'Vision model + Ollama URL (Settings ▸ Local tools)' },
      { name: 'Vision concurrency (Settings ▸ Local tools)' },
      { name: 'The prompt — NOT configurable' },
    ],
  };
}
