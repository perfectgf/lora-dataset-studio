/* The collapsible GROUPS a long settings section is organised into, and the
 * little state around them. PURE JS — `node --test` covers every decision.
 *
 * WHY GROUPS AND NOT SUB-PAGES. The Image engines section had grown into a
 * single wall of eleven cards — API keys next to Klein pins next to LoRA
 * presets next to the improve prompt — and "where is the thing I came for"
 * was answered by scrolling (reported from a tablet, mid preset editing).
 * Groups keep every deep-link alive for free: each group is a NATIVE
 * <details>, and help/revealTarget.openCollapsedAncestors already opens a
 * collapsed <details> on the way to a ?focus= field — so Settings search and
 * every "Open in Settings →" link land INSIDE the right group with zero new
 * wiring. Sub-pages would have needed an alias for every existing link.
 *
 * The <details> stay UNCONTROLLED on purpose: the reveal helper flips
 * `open` on the DOM node directly, and a React-controlled `open` prop would
 * fight it on the next render. The initial state is read once at mount; the
 * prop value then never changes, so React never writes over a user's toggle.
 */

/** The Image engines groups, in display order. `id` is stored in localStorage
 *  and used in DOM anchors — never rename one without an alias. */
import { BarChart3, Bot, Box, Cloud, Drama, Dumbbell, Eraser, ImageDown, KeyRound, Map, Package, PenLine, Puzzle, SlidersHorizontal, SlidersVertical, Trash2, ZoomIn } from 'lucide-react';
export const ENGINES_GROUPS = [
  { id: 'engines-keys', title: 'Engines & API keys', icon: KeyRound,
    blurb: 'Which engines are on, which one opens preselected, their API keys and models.' },
  { id: 'klein', title: 'Klein (local)', icon: SlidersHorizontal,
    blurb: 'Model file pins and generation quality for the local Klein engine.' },
  { id: 'krea', title: 'Krea 2 Edit (local)', icon: SlidersVertical,
    blurb: 'The second local engine — base model, identity LoRA and its dials.' },
  { id: 'lora-presets', title: 'Generation LoRA presets', icon: Puzzle,
    blurb: 'Named LoRA chains you pick per run — one list per local engine.' },
  { id: 'seedvr2', title: 'Upscaling — SeedVR2', icon: ZoomIn,
    blurb: 'The restoration upscaler: tiling, resolution and VRAM behaviour.' },
  { id: 'prompts', title: 'Prompts & improve tuning (advanced)', icon: PenLine,
    blurb: 'Identity prompts per subject type, the improve instruction and its strength knobs.' },
]

/** 🖥️ Local tools — three tools, three long cards; the group is the tool. */
export const LOCAL_TOOLS_GROUPS = [
  { id: 'comfyui', title: 'ComfyUI', icon: Puzzle,
    blurb: 'The local generation backend — API URL, install folder, model paths.' },
  { id: 'ollama', title: 'Ollama', icon: Bot,
    blurb: 'Local vision & text models — captions, descriptions, prompt help.' },
  { id: 'aitoolkit', title: 'ai-toolkit', icon: Dumbbell,
    blurb: 'The local trainer — install folder and its Python.' },
]

/** ✍️ Captioning & quality. */
export const CAPTIONING_GROUPS = [
  { id: 'import', title: 'Import & image size', icon: ImageDown,
    blurb: 'What happens to a photo as it enters — stored format, resolution budgets.' },
  { id: 'captioning', title: 'Captioning', icon: PenLine,
    blurb: 'Which captioner writes the training captions, and how.' },
  { id: 'watermarks', title: 'Watermark inpainting', icon: Eraser,
    blurb: 'The clean-watermarks pass — engine and behaviour.' },
  { id: 'quality', title: 'Quality scoring & triage', icon: BarChart3,
    blurb: 'Face similarity and the bank triage thresholds.' },
]

/** 🏋️ Training. */
export const TRAINING_GROUPS = [
  { id: 'defaults', title: 'Defaults', icon: SlidersHorizontal,
    blurb: 'The model family new runs start on.' },
  { id: 'cloud', title: 'Cloud training (vast.ai)', icon: Cloud,
    blurb: 'The API key that unlocks cloud runs, and every guardrail on spend and stalls.' },
  { id: 'masking', title: 'Concept face masking', icon: Drama,
    blurb: 'Masking faces out of concept training.' },
]

/** 💾 Storage. */
export const STORAGE_GROUPS = [
  { id: 'overview', title: 'What lives where', icon: Map,
    blurb: 'Every folder the app writes to, with sizes on demand.' },
  { id: 'locations', title: 'Movable folders', icon: Package,
    blurb: 'The three relocatable roots — dataset images, cloud staging, checkpoints.' },
  { id: 'housekeeping', title: 'Cleanup & trash', icon: Trash2,
    blurb: 'Cloud-run housekeeping, the trash, and the run image archive.' },
  { id: 'models', title: 'Model files', icon: Box,
    blurb: 'fp8 quantization and the Hugging Face cache.' },
]

/** Which sections carry groups at all. Scraping, Server, Maintenance and the
 *  Overview keep their flat handful of cards ON PURPOSE: a summary over one
 *  or two cards is navigation for a hallway with one door. */
export const SECTION_GROUPS = {
  engines: ENGINES_GROUPS,
  'local-tools': LOCAL_TOOLS_GROUPS,
  captioning: CAPTIONING_GROUPS,
  training: TRAINING_GROUPS,
  storage: STORAGE_GROUPS,
}

/** The DOM id a group's <details> carries — the TOC and tests address it. */
export function groupDomId(sectionId, groupId) {
  return `settings-group-${sectionId}-${groupId}`
}

const storageKey = (sectionId) => `settingsGroupsOpen.${sectionId}`

/** Which groups start OPEN, read once at mount. Defaults to all collapsed —
 *  the summary at the top of the section is the map, and a wall that opens
 *  fully unfolded is the exact screen this replaces. Storage failures (private
 *  window, blocked site data) mean "all collapsed", never a crash. */
export function readOpenGroups(storage, sectionId) {
  try {
    const raw = storage && storage.getItem(storageKey(sectionId))
    const list = raw ? JSON.parse(raw) : []
    return new Set(Array.isArray(list) ? list.filter((v) => typeof v === 'string') : [])
  } catch {
    return new Set()
  }
}

/** Persist one toggle. Write-through and forgiving for the same reasons. */
export function storeGroupToggle(storage, sectionId, groupId, open) {
  try {
    const cur = readOpenGroups(storage, sectionId)
    if (open) cur.add(groupId)
    else cur.delete(groupId)
    storage.setItem(storageKey(sectionId), JSON.stringify([...cur]))
  } catch { /* a private window loses the convenience, not the section */ }
}
