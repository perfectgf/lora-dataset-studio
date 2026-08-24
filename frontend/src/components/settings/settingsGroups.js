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
export const ENGINES_GROUPS = [
  { id: 'engines-keys', title: 'Engines & API keys', icon: '🔑',
    blurb: 'Which engines are on, which one opens preselected, their API keys and models.' },
  { id: 'klein', title: 'Klein (local)', icon: '🎛️',
    blurb: 'Model file pins and generation quality for the local Klein engine.' },
  { id: 'krea', title: 'Krea 2 Edit (local)', icon: '🎚️',
    blurb: 'The second local engine — base model, identity LoRA and its dials.' },
  { id: 'lora-presets', title: 'Generation LoRA presets', icon: '🧩',
    blurb: 'Named LoRA chains you pick per run — one list per local engine.' },
  { id: 'seedvr2', title: 'Upscaling — SeedVR2', icon: '🔍',
    blurb: 'The restoration upscaler: tiling, resolution and VRAM behaviour.' },
  { id: 'prompts', title: 'Prompts & improve tuning (advanced)', icon: '✍️',
    blurb: 'Identity prompts per subject type, the improve instruction and its strength knobs.' },
]

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
