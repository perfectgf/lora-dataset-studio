/** The facts of a rendered clip as SEPARATE tags, for a pill row.
 *
 * `clipSummary` (videoStudioApi) joins the same facts into one emoji line for
 * places that have a single line of text. A card has room for a tag row and
 * carries icons of its own, so here the emoji go and each fact stands alone —
 * ordered by how much it changed the render: base first, LoRA and strength,
 * then the accelerators, then the numbers that make a run repeatable. */
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
  if (clip.turbo) tags.push('turbo');
  if (clip.sparse) tags.push(`sparse ${clip.sparse}`);
  if (clip.latent_upscale) tags.push('upscale ×2');
  if (clip.steps) tags.push(`${clip.steps} steps`);
  if (clip.seed !== null && clip.seed !== undefined) tags.push(`seed ${clip.seed}`);
  return tags;
}
