/* The Video Test Studio's request shapes, kept out of the components.
 *
 * Two reasons this is its own module rather than inline `fetch` calls:
 *
 *  - the payload has ten fields and four of them change what ComfyUI actually
 *    computes (turbo, sparse, the base swap, the upscale). A mistake there does
 *    not throw — it renders a perfectly good clip that answers a different
 *    question than the one that was asked. `node --test` can check this shape;
 *    it cannot check a component;
 *  - the panel and the history read the same clip objects, and the labels below
 *    are what stop "sparse: max" from being written three different ways.
 */

export const VIDEO_STUDIO_BASE = '/api/video-studio';

export const optionsUrl = () => `${VIDEO_STUDIO_BASE}/options`;
export const lorasUrl = () => `${VIDEO_STUDIO_BASE}/loras`;
export const deployUrl = () => `${VIDEO_STUDIO_BASE}/deploy`;

/** Where a LoRA the user already has is brought into the picker. */
export const loraImportUrl = () => '/api/video-studio/lora/import';

/** ↗ Smooth a finished clip — RIFE interpolation, as a new clip. */
export const clipVfiUrl = (id) => `/api/video-studio/clip/${id}/vfi`;
/** ↗ The rates Smooth can make of a clip. RIFE interpolates by a WHOLE factor,
 *  so the choices are the source rate times 2, 3 and 4 — 48, 72, 96 fps for a
 *  clip authored at 24 — never an arbitrary number (that would mean dropping
 *  frames unevenly afterwards). `cost` is relative to the ×2 pass: the work
 *  grows with the frames written between each pair (1, 2, 3). */
export const SMOOTH_MULTIPLIERS = [2, 3, 4];
export function smoothTargets(clip) {
  const fps = Number(clip?.fps) > 0 ? Number(clip.fps) : 24;
  const frames = Number(clip?.frames) > 0 ? Number(clip.frames) : null;
  return SMOOTH_MULTIPLIERS.map((m) => ({
    multiplier: m,
    fps: Math.round(fps * m),
    frames: frames ? frames * m : null,
    cost: m - 1,
  }));
}

/** ✨ Neural render a finished clip — DLSS 5 Neural Rendering, as a new clip. */
export const clipNeuralRenderUrl = (id) => `/api/video-studio/clip/${id}/neural-render`;

/** ✨ The Motion field's two helpers: propose from the start frame, or enrich
 * what is already written. Both answer with a prompt the user can still edit —
 * neither is a launch. */
export const motionSuggestUrl = () => '/api/video-studio/motion/suggest';
export const motionEnhanceUrl = () => '/api/video-studio/motion/enhance';

/** ⚙ The model that writes the motion — listed, and chosen. */
export const motionModelsUrl = () => '/api/video-studio/motion/models';
export const motionModelUrl = () => '/api/video-studio/motion/model';
export const sourceUrl = () => `${VIDEO_STUDIO_BASE}/source`;
export const generateUrl = () => `${VIDEO_STUDIO_BASE}/generate`;
/** The history, newest first: one page of `limit`, `before` (a clip id) for the
 * page after it. The server appends the SOURCE of every listed render, so the
 * pair a comparison needs is always on screen together. */
/** The newest page REPLACES what it covers and KEEPS what it does not.
 *  `keepOlderThan` is the boundary of the page PROPER (the server's
 *  `oldest_id`), never the oldest id on the page: a source that rode along
 *  with its render is older by construction, and taking it as the boundary
 *  dropped every loaded clip between the two at every poll. Deleted rows
 *  leave through the page they belonged to, which the fresh page no longer
 *  carries. */
export function mergeClipPages(prev, fresh, keepOlderThan) {
  const byId = new Map();
  (fresh || []).forEach((c) => byId.set(c.id, c));
  (prev || []).forEach((c) => { if (c.id < keepOlderThan && !byId.has(c.id)) byId.set(c.id, c); });
  return [...byId.values()].sort((a, b) => b.id - a.id);
}

export const clipsUrl = (limit = 24, before = null) =>
  `${VIDEO_STUDIO_BASE}/clips?limit=${limit}${before ? `&before=${before}` : ''}`;
export const clipUrl = (id) => `${VIDEO_STUDIO_BASE}/clip/${id}`;
export const clipVideoUrl = (id) => `${VIDEO_STUDIO_BASE}/clip/${id}/video`;
export const clipRateUrl = (id) => `${VIDEO_STUDIO_BASE}/clip/${id}/rate`;
/** ⬇ A render and its source as ONE side-by-side mp4 (404 when the clip is
 * not a render, or its source is gone). */
export const clipComparisonUrl = (id) => `${VIDEO_STUDIO_BASE}/clip/${id}/comparison`;

/* The sparse levels, in the order they cost adherence. The wording says what
 * each one DOES to the picture rather than naming a budget: "0.3 video budget"
 * is the node's vocabulary, not a user's, and the only decision here is how
 * much prompt fidelity to trade for speed. */
export const SPARSE_CHOICES = [
  { value: '', label: 'Off', hint: 'Dense attention — the reference render.' },
  { value: 'conservative', label: 'Conservative',
    hint: 'Faster, edges of the schedule kept dense. Safest with a prompt that matters.' },
  { value: 'default', label: 'Default',
    hint: 'The node author\'s defaults — roughly 1.6× faster.' },
  { value: 'max', label: 'Max',
    hint: 'Sparse on every pass, including the one that sets the composition. '
        + 'Fastest, and the prompt is followed less closely.' },
];

/* Whether a clip is still on its way. One predicate, so the poller, the button
 * and the tile cannot disagree about what "running" means. */
export const isRunning = (clip) => !!clip && (clip.status === 'pending');

/** The body of POST /generate.
 *
 * Only what was actually chosen is sent: an option left off is ABSENT rather
 * than `false`, so the server's defaults stay the single definition of "off"
 * and a future default change does not have to be mirrored here.
 *
 * `image` is dropped in t2v even when one was picked earlier, because the
 * server would otherwise be handed a start frame for a mode that has no start
 * frame — the kind of mismatch that gets answered with a clip nobody can
 * explain rather than an error.
 */
export function buildGeneratePayload(state) {
  const s = state || {};
  const mode = s.mode === 't2v' ? 't2v' : 'i2v';
  const body = { mode, prompt: (s.prompt || '').trim() };
  if (mode === 'i2v') {
    if (s.image) body.image = s.image;
    if (s.ratio) body.ratio = s.ratio;
  } else if (s.aspect) {
    body.aspect = s.aspect;
  }
  if (s.lora) {
    body.lora = s.lora;
    body.lora_strength = Number(s.loraStrength ?? 1);
    if (s.runId) body.run_id = s.runId;
    if (s.datasetId) body.dataset_id = s.datasetId;
  }
  if (s.frames) body.frames = Number(s.frames);
  if (s.megapixels) body.megapixels = Number(s.megapixels);
  if (s.seed !== '' && s.seed !== null && s.seed !== undefined) {
    body.seed = Number(s.seed);
  }
  if (s.steps) body.steps = Number(s.steps);
  // ✨ Enrich at launch: the SERVER rewrites the motion and records what ran,
  // so a clip never claims a prompt that is not the one it was made from.
  if (s.enhance) body.enhance = true;
  // ⚡ The acceleration by name; `turbo` rides along for the older servers'
  // boolean when the name is larryvrh's.
  if (s.accel) {
    body.accel = s.accel;
    if (s.accel === 'turbo') body.turbo = true;
  } else if (s.turbo) {
    body.turbo = true;
  }
  if (s.eros) body.eros = true;
  if (s.sparse) body.sparse = s.sparse;
  if (s.latentUpscale) body.latent_upscale = true;
  return body;
}

/** ⚡ The Render panel's acceleration choices, as the server names them —
 *  this list is only the shape shown before the options arrive (and in tests);
 *  the server's `accelerations` carries availability, arena rank and hint. */
export const ACCELERATIONS = [
  { id: 'turbo', label: 'larryvrh Turbo v4', arena: '#1 · I2V 1103 / T2V 1110', steps: 6 },
  { id: 'parasyte', label: 'Parasyte Turbo', arena: '#2 · I2V 1106 / T2V 1094', steps: 6 },
  { id: 'dareties', label: 'DARE-TIES merge', arena: '#3 · I2V 1107 / T2V 1085', steps: 6 },
];
export const accelLabel = (id) => (ACCELERATIONS.find((a) => a.id === id) || {}).label
  || (id ? String(id) : '');
/** The acceleration a clip ran with: the stored name, or `turbo` from the
 *  flag on rows older than the choice. */
export const clipAccel = (clip) => (clip?.accel || (clip?.turbo ? 'turbo' : ''));
/** What the panel should hold once the server said what is on this machine:
 *  the current pick if it is available (or unknown), else the first available
 *  one, else the dense base. `null` availability (probe unreachable) keeps
 *  the pick — an unknown is not a no. */
export function pickAvailableAccel(current, accelerations) {
  const rows = Array.isArray(accelerations) ? accelerations : [];
  if (!current) return '';
  const row = rows.find((a) => a.id === current);
  if (!row || row.available !== false) return current;
  const other = rows.find((a) => a.available === true);
  return other ? other.id : '';
}

/** How long the clip will be, in seconds, at the target's own fps.
 * N frames span N-1 intervals — the same arithmetic the training lane uses, so
 * a 39-frame clip reads as the same duration in both places. */
export function clipSeconds(frames, fps) {
  if (!frames || !fps) return null;
  return Math.round(((frames - 1) / fps) * 100) / 100;
}

/** The one-line summary under a finished clip.
 *
 * Ordered by how much each thing changed the render: the base first (it is a
 * different model), then the LoRA and its strength, then the accelerators.
 * Options that were off contribute nothing — a row of "turbo: no, sparse: no"
 * is noise in a list whose entire job is showing what differed.
 */
export function clipSummary(clip) {
  if (!clip) return '';
  const bits = [];
  if (clip.eros) bits.push('🔥 10Eros');
  if (clip.lora) {
    const name = String(clip.lora).replace(/\\/g, '/').split('/').pop()
      .replace(/\.safetensors$/i, '');
    bits.push(`${name} @ ${clip.lora_strength ?? 1}`);
  } else {
    bits.push('no LoRA');
  }
  const accel = clipAccel(clip);
  if (accel) bits.push(accel === 'turbo' ? '⚡ turbo' : `⚡ ${accelLabel(accel)}`);
  if (clip.sparse) bits.push(`sparse ${clip.sparse}`);
  if (clip.latent_upscale) bits.push('🔬 upscale');
  bits.push(`${clip.steps} steps`);
  if (clip.seed !== null && clip.seed !== undefined) bits.push(`seed ${clip.seed}`);
  return bits.join(' · ');
}

/** Every legal clip length the STUDIO may generate, not the ones training uses.
 *
 * The dropdown was built from `frame_choices` — the TRAINING catalogue, which
 * stops at 209 frames because that is where training clip lengths stop being
 * useful. The model renders to about 15 s, the server has always accepted it
 * (FRAMES_MIN/FRAMES_MAX = 22/362, and its own comment says capping the studio
 * at 8.7 s would be "a reason that has nothing to do with the studio") — the
 * list on screen was simply the wrong table. Generated from the bounds and the
 * VAE's own rule (≡ 5 mod 17), so there is one source of truth and no third
 * copy of the ladder.
 */
export function studioFrameChoices(options) {
  const min = Number(options?.frames_min) || 22;
  const max = Number(options?.frames_max) || 362;
  const out = [];
  for (let f = min - ((min - 5) % 17 || 0); f <= max; f += 17) {
    if (f >= min && f % 17 === 5 % 17) out.push(f);
  }
  // The floor the catalogue offers is 22 (5 mod 17) and stays first even when
  // the arithmetic above starts one rung higher.
  if (out[0] !== min && min % 17 === 5) out.unshift(min);
  return out.length ? out : (options?.frame_choices || [39, 56, 107]);
}

// ⏱ The launch advice, as two sentences. The server decides WHETHER to speak
// (video_test_studio.launch_advice: the flag missing, a ComfyUI that knows it,
// a machine whose RAM cannot hold the weights); this only phrases what it
// sent, and never spells a flag of its own — every name comes from the payload,
// so a second flag on the server side needs no change here.
export function launchAdviceLines(advice) {
  if (!advice || !advice.flag) return null;
  const { flag, add, remove } = advice;
  const title = remove
    ? `ComfyUI is running with ${remove}, which switches off the loader ${flag} relies on`
    : `ComfyUI is running without ${flag}`;
  let change;
  if (remove && add) change = `Remove ${remove} and add ${flag}`;
  else if (remove) change = `Remove ${remove} (${flag} is already on the command line)`;
  else change = `Add ${flag}`;
  return {
    title,
    action: `${change} on the command that starts ComfyUI, then start it again.`,
  };
}

// ⏱ Render time as a person reads it: "24 s", "5 min 48 s", "2 min", "1 h 12 min".
// The number is the queue's own measurement (claim → settled, model loading
// included); null for anything that is not a positive number, so a card never
// prints "rendered in null" for a clip the queue could not time. A measured
// fraction of a second reads "1 s" — a real measurement is rounded, never hidden.
export function renderTimeLabel(seconds) {
  const s = Number(seconds);
  if (!Number.isFinite(s) || s <= 0) return null;
  const t = Math.max(1, Math.round(s));
  if (t < 60) return `${t} s`;
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const r = t % 60;
  if (h) return m ? `${h} h ${m} min` : `${h} h`;
  return r ? `${m} min ${r} s` : `${m} min`;
}
