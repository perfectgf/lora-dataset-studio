// What the editor offers, in the words it uses. Pure: node tests import it.
export const KEEPS = [
  { id: 'subject', label: 'Main subject', hint: 'The models decide what the subject is.' },
  { id: 'selection', label: 'Inside selection', hint: 'Draw roughly around what stays. Hair may pass a little beyond your line.' },
  { id: 'describe', label: 'Describe it', hint: 'A few words, such as “woman in red” or “dog, ball”. Everything that matches is kept.' },
  { id: 'points', label: 'Click on it', hint: 'Click on what stays. Switch to Leave out, or hold Alt, to click on what goes.' },
]

export const KINDS = [
  { id: 'photo', label: 'Photo', model: 'BiRefNet', hint: 'Quick cut-outs of people, animals and products.' },
  { id: 'fine', label: 'Photo, finest edges', model: 'RMBG 3.0', hint: 'Single strands of hair and fur. Slower. Non-commercial licence.' },
  { id: 'graphic', label: 'Graphic, logo or glass', model: 'Lucida', hint: 'Logos, text, drawings and see-through objects.' },
]

export const BACKGROUNDS = [
  { id: 'transparent', label: 'Transparent', css: '' },
  { id: 'white', label: 'White', css: '#ffffff' },
  { id: 'grey', label: 'Grey', css: '#808080' },
  { id: 'black', label: 'Black', css: '#000000' },
]

const STAGES = {
  find: 'Finding what stays…',
  cut: 'Cutting the edge…',
  closer: 'Looking closer at the edge…',
}

export function runLabel(run) {
  if (!run) return ''
  if (run.status === 'queued') return run.stage === 'closer' || (run.stage === 'cut' && run.keep !== 'subject' && run.keep !== 'selection')
    ? STAGES[run.stage] : 'Waiting for the GPU…'
  return STAGES[run.stage] || 'Working…'
}

/** What is missing before a cut-out can run, as one sentence, or ''. */
export function blocked(status, keep, kind) {
  if (!status) return 'Checking the models…'
  if (!status.configured) return 'Connect your ComfyUI folder in Settings first.'
  if (!status.reachable) return 'ComfyUI does not answer. Start it, then try again.'
  const parts = [status.kinds[kind], ...(keep === 'describe' || keep === 'points' ? [status.find] : [])]
  const nodes = [...new Set(parts.flatMap(part => part.missing_nodes || []))]
  return nodes.length ? `Update ComfyUI and restart it when idle. Missing native nodes: ${nodes.join(', ')}.` : ''
}

/** The model slots one cut-out needs and that are not usable yet. */
export function missingModels(status, keep, kind) {
  if (!status) return []
  const slots = [...status.kinds[kind].models, ...(keep === 'describe' || keep === 'points' ? status.find.models : [])]
  return slots.filter(slot => status.models[slot].state !== 'ready')
}

/** A filled-in request, or the reason it cannot start yet. */
export function notReadyToCut({ keep, phrase, points, hasSelection }) {
  if (keep === 'selection' && !hasSelection) return 'Draw around what stays on the picture.'
  if (keep === 'describe' && !phrase.trim()) return 'Say in a few words what stays.'
  if (keep === 'points' && !points.some(point => point.keep)) return 'Click on what stays on the picture.'
  return ''
}

const STORE = 'lds.remove_background.options'

export function readOptions(storage) {
  try {
    const saved = JSON.parse(storage?.getItem(STORE) || '{}')
    return {
      kind: KINDS.some(item => item.id === saved.kind) ? saved.kind : 'photo',
      background: BACKGROUNDS.some(item => item.id === saved.background) ? saved.background : 'transparent',
      clean: saved.clean !== false,
    }
  } catch { return { kind: 'photo', background: 'transparent', clean: true } }
}

export function saveOptions(storage, options) {
  try { storage?.setItem(STORE, JSON.stringify(options)) } catch { /* the choice lasts for this visit only */ }
}
