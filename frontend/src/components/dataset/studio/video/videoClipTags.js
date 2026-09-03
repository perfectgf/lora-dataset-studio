/** The facts of a rendered clip as SEPARATE tags, for a pill row.
 *
 * `clipSummary` (videoStudioApi) joins the same facts into one emoji line for
 * places that have a single line of text. A card has room for a tag row and
 * carries icons of its own, so here the emoji go and each fact stands alone —
 * ordered by how much it changed the render: base first, LoRA and strength,
 * then the accelerators, then the numbers that make a run repeatable. */
import { neuralRenderTags } from '../../../videobank/neuralRenderParams.js';
import { accelLabel, clipAccel } from './videoStudioApi.js';

export function clipTags(clip) {
  if (!clip) return [];
  const tags = [];
  if (clip.eros) tags.push('10Eros base');
  if (clip.lora) {
    const name = String(clip.lora).replace(/\\/g, '/').split('/').pop()
      .replace(/\.safetensors$/i, '');
    tags.push(`${name} @ ${clip.lora_strength ?? 1}`);
  } else {
    tags.push('no LoRA');
  }
  const accel = clipAccel(clip);
  if (accel) tags.push(accel === 'turbo' ? 'turbo' : accelLabel(accel));
  if (clip.sparse) tags.push(`sparse ${clip.sparse}`);
  if (clip.latent_upscale) tags.push('upscale ×2');
  // ✨ A neural-rendered clip: same settings as its source, different pixels —
  // and the dials that made it, which are the only ones that differ.
  if (clip.nr_of) tags.push('neural render', ...neuralRenderTags(clip.nr_params));
  // ↗ A smoothed clip has the same settings as its source and is NOT the same
  // artefact — without this the pair is two identical-looking cards.
  if (clip.vfi_of) tags.push(`smoothed → ${Math.round(clip.fps || 0)} fps`);
  // ⏭ A continuation: joined behind its parent, or left as the part when the
  // join failed — `joined` is null while the part still renders, and that is
  // no verdict, so the pill says nothing about it yet.
  if (clip.continues_of) tags.push(clip.joined === false ? `continues #${clip.continues_of} (not joined)` : `continues #${clip.continues_of}`);
  if (clip.steps) tags.push(`${clip.steps} steps`);
  if (clip.seed !== null && clip.seed !== undefined) tags.push(`seed ${clip.seed}`);
  return tags;
}
