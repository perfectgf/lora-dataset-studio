// Setup's view of the plugin, read from its `remove_background` probe. Pure.
export const BASE_MODELS = [
  { action: 'remove_bg_birefnet', slot: 'birefnet', label: 'BiRefNet, cuts out photos (444 MB)' },
  { action: 'remove_bg_sam', slot: 'sam', label: 'SAM 3.1, finds what you describe or click (1.75 GB)' },
  { action: 'remove_bg_lucida', slot: 'lucida', label: 'Lucida, cuts out graphics and glass (885 MB)' },
]

export function setupRows(caps) {
  return [{ label: 'Remove background',
    what: 'Keep one subject of a picture and drop the rest, in your ComfyUI',
    ok: caps?.remove_background?.ok === true, topic: 'remove-background-setup' }]
}

export function modelRows(caps) {
  const ready = caps?.remove_background?.models || {}
  const folder = !!caps?.comfyui?.dir_valid
  return BASE_MODELS.map(({ action, slot, label }) => ({
    action, label, present: ready[slot] === true, available: folder,
    hint: folder ? '' : 'Choose the ComfyUI folder in Local tools first.',
  }))
}
