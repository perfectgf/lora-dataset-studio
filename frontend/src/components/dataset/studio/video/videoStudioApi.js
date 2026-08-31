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
export const sourceUrl = () => `${VIDEO_STUDIO_BASE}/source`;
export const generateUrl = () => `${VIDEO_STUDIO_BASE}/generate`;
export const clipsUrl = (limit = 24) => `${VIDEO_STUDIO_BASE}/clips?limit=${limit}`;
export const clipUrl = (id) => `${VIDEO_STUDIO_BASE}/clip/${id}`;
export const clipVideoUrl = (id) => `${VIDEO_STUDIO_BASE}/clip/${id}/video`;
export const clipRateUrl = (id) => `${VIDEO_STUDIO_BASE}/clip/${id}/rate`;

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
  if (s.turbo) body.turbo = true;
  if (s.eros) body.eros = true;
  if (s.sparse) body.sparse = s.sparse;
  if (s.latentUpscale) body.latent_upscale = true;
  return body;
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
  if (clip.turbo) bits.push('⚡ turbo');
  if (clip.sparse) bits.push(`sparse ${clip.sparse}`);
  if (clip.latent_upscale) bits.push('🔬 upscale');
  bits.push(`${clip.steps} steps`);
  if (clip.seed !== null && clip.seed !== undefined) bits.push(`seed ${clip.seed}`);
  return bits.join(' · ');
}
