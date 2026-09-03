/* 🌳 Pure helpers of the video workspace's ◉ Graph — the bridge between the
   lineage TREE the server answers (the image graph's shape: nodes keyed by
   record_id, pills under each) and the STEP model every video verb is
   decided by (videoCheckpoints.stepActionModel). One conversion, in one
   place, so a pill on the graph and a row in the list offer exactly the same
   actions for the same save — a verb that differed between the two would be
   the drift CLAUDE.md's parity rule is about, inside a single section. */
import { stepActionModel, stepKey } from './videoCheckpoints.js'

/* The two row styles the image lane's checkpoint popover draws with
   (CheckpointActionsPopover.jsx), shared by the list AND the graph popover of
   this lane: a verb learned on one surface has to LOOK like itself on the
   other. Here rather than in a component, because both components import
   this module and neither should import the other. */
export const ROW_CLS = 'flex items-center gap-1.5 rounded-md border px-2 py-1 text-[0.6875rem] font-medium '
  + 'disabled:cursor-not-allowed disabled:opacity-60'
export const MUTED_CLS = 'rounded-md border border-border bg-app/40 px-2 py-1 text-content-subtle text-[0.625rem]'

/** The GROUP a tree node stands for, in the list's vocabulary. */
export function nodeGroup(node) {
  const local = node?.source === 'local'
  return {
    key: local ? 'local' : `cloud-${node?.run_id}`,
    lane: local ? 'local' : 'cloud',
    run_id: local ? null : (node?.run_id ?? null),
    active: !!node?.active,
    status: node?.status ?? null,
    run_name: node?.run_name ?? null,
    parent_run_id: node?.parent_record_id ?? null,
    steps: node?.checkpoints || [],
  }
}

/** The STEP a pill stands for. `testable` is the tree's word for "every file
 * of this step is in ComfyUI" (the shared pill renderer reads it); `files`
 * carry each file's own deployed state, which is what ⏏ needs. */
export function pillStep(pill) {
  return {
    step: pill?.step ?? null,
    final: !!pill?.final,
    deployed: pill?.testable === true,
    files: pill?.files || [],
  }
}

export function pillKey(node, pill) {
  return stepKey(nodeGroup(node), pillStep(pill))
}

/** What the graph popover offers for a pill — the list's decisions, verbatim. */
export function pillActionModel(datasetId, node, pill, ctx = {}) {
  return stepActionModel(datasetId, nodeGroup(node), pillStep(pill), ctx)
}

/** What a pill's preview means here: the training sample ai-toolkit rendered
 * at that step (prompt 0), and how many samples the step has. The shared pill
 * reads `{status, url, count}`; a pill with no sample gets null and draws its
 * plain box. */
export function pillPreview(pill) {
  if (!pill?.preview_url && !pill?.preview_status) return null
  return { status: pill.preview_status || null, url: pill.preview_url || null,
    count: Number(pill.preview_count) || (pill.preview_url ? 1 : 0) }
}

/** The deploy sentence of a pill's title on this lane — true of ITS verbs: the
 * image pill's says "tick it and 🎨 Generate will deploy it first", and there
 * is no such bar here. */
export function videoDeployHint(pill) {
  if (pill?.present === false) return ' — this save is no longer on disk'
  if (pill?.testable === true) return ' — deployed to ComfyUI (the Video Studio lists it)'
  return ' — not deployed — 📦 Deploy from its actions to test it in the Studio'
}

/** The samples of ONE step, prompt order — what the sample lightbox lists. */
export function samplesOfStep(samples, step) {
  return (samples || [])
    .filter((s) => step != null && Number(s.step) === Number(step))
    .sort((a, b) => (a.prompt_idx ?? 0) - (b.prompt_idx ?? 0))
}

/** The one-line summary the graph's fold shows: runs, saves, previews. */
export function graphSummary(tree) {
  const nodes = tree?.nodes || []
  const saves = nodes.reduce((n, node) => n + (node.checkpoints?.length || 0), 0)
  const previews = nodes.reduce((n, node) => n
    + (node.checkpoints || []).reduce((m, p) => m + (Number(p.preview_count) || 0), 0), 0)
  const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`
  return `${plural(nodes.length, 'run')} · ${plural(saves, 'save')} · ${plural(previews, 'preview')}`
}

/* Why the graph has no "Generate previews" bar YET, unlike the image one. Not
   a difference of nature — the refuter checked: the Video Studio's
   `enqueue_clip` can already render one clip for one deployed save, and the
   image bar queues its previews the same way — but the fan-out over selected
   saves is not written. Said on screen as work not done, so nobody reads the
   missing bar as a rule. */
export const PREVIEWS_NOTE = 'Previews are the samples ai-toolkit rendered while training '
  + '(one per prompt, every save). Rendering new ones from a deployed save is not wired '
  + 'here yet — pick the save in the Studio\'s Video tab.'

export const EMPTY_GRAPH_NOTE = 'No run to draw yet — the graph appears once a training has saved something.'
