/* 📦 Pure brain of the video workspace's Checkpoints & LoRAs section — every
   decision the list makes about a SAVE, JSX-free so `node --test` can pin them.

   The image lane's checkpoint popover (components/dataset/checkpointPopover.js)
   decides per FILE; this one decides per STEP, and that is the whole reason it
   is a second file rather than a second caller of the first: a Wan 2.2
   checkpoint is two files at one step, and a per-file model offers half a LoRA
   without noticing. Same verbs, same rule — an action is live, or shown with
   its real reason, or absent; never a button that fails in silence — same
   wording where the behaviour is the same, and a DIFFERENT sentence where it is
   not (a local video run cannot pick a step to continue from; see
   CONTINUE_LOCAL_REASON), which is CLAUDE.md's parity rule made literal. */
import { deleteDestination, isRecoverable } from '../../utils/deletionWording.js'
import { stepLabel } from './videoCloudStatus.js'
import { videoDatasetCheckpointUrl, videoDatasetLocalCheckpointUrl } from './videoBankApi.js'

export const EMPTY_NOTE = 'No checkpoints yet — train this set on this PC or in the '
  + 'cloud, and every save comes back here, step by step.'

/* Why a LOCAL save offers no ▶ Continue: ai-toolkit resumes from whatever it
   finds in the run folder, so the next local launch continues from the NEWEST
   save whatever row the click came from. Offering the button would do
   something other than what it says. */
export const CONTINUE_LOCAL_REASON = 'Resumes from its newest save on the next '
  + 'local launch — the run folder is the resume state, so no step is picked here'
export const ACTIVE_CLOUD_REASON = 'This run is still on its pod — stop it first'
export const ACTIVE_LOCAL_REASON = 'Training is running and still writing these '
  + 'saves — stop it first'
export const NO_LORAS_ROOT_REASON = 'ComfyUI\'s loras folder is not configured — '
  + 'nothing to deploy into'
export const HAND_PLACED_REASON = 'Deployed by hand — remove it from ComfyUI\'s '
  + 'loras folder yourself'

/** The groups the section renders, in order: the local run first (it is this
 * machine), then the cloud runs newest first — the server's own order. A group
 * with no step is not a group: the empty state is one sentence, not a header
 * over nothing. */
export function checkpointGroups(payload) {
  const out = []
  const local = payload?.local
  if (local?.steps?.length) {
    out.push({
      key: 'local', lane: 'local', run_id: null, active: !!local.active,
      run_name: local.run_name, folder: local.folder, parent_run_id: null,
      status: local.active ? 'training' : 'done', steps: local.steps,
    })
  }
  for (const g of payload?.cloud || []) {
    if (g?.steps?.length) out.push({ key: `cloud-${g.run_id}`, lane: 'cloud', ...g })
  }
  return out
}

export function groupTitle(group) {
  if (group.lane === 'local') return `On this PC — ${group.run_name || 'local run'}`
  const from = group.parent_run_id ? ` — continued from #${group.parent_run_id}` : ''
  return `Cloud run #${group.run_id}${from}`
}

// Relative "15m ago" from a naive-UTC backend timestamp — the Runs page's rule,
// so a group header reads exactly like its Runs row.
export function timeAgo(iso, now = Date.now()) {
  if (!iso) return ''
  const t = new Date(/[Z+]/.test(iso) ? iso : `${iso}Z`).getTime()
  if (Number.isNaN(t)) return ''
  const s = Math.max(0, (now - t) / 1000)
  if (s < 60) return 'just now'
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`
  return `${Math.floor(s / 86400)}d ago`
}

export function groupSub(group, now = Date.now()) {
  if (group.lane === 'local') {
    return group.active ? 'training now — saves still being written' : 'this machine\'s run folder'
  }
  const bits = [group.status || 'unknown']
  if (group.gpu) bits.push(group.gpu)
  if (group.price_per_hour != null) bits.push(`$${Number(group.price_per_hour).toFixed(2)}/h`)
  const when = timeAgo(group.finished_at || group.created_at, now)
  if (when) bits.push(when)
  return bits.join(' · ')
}

export function fmtSize(bytes) {
  const n = Number(bytes)
  if (!Number.isFinite(n) || n <= 0) return ''
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(1)} GB`
  if (n >= 1024 ** 2) return `${Math.round(n / 1024 ** 2)} MB`
  return `${Math.max(1, Math.round(n / 1024))} KB`
}

/** The short name a ⬇ link shows: the expert of a Wan pair, else the file. */
export function fileShortName(filename, fileCount = 1) {
  const name = String(filename || '')
  if (fileCount > 1) {
    if (/_high_noise\.safetensors$/i.test(name)) return 'high noise'
    if (/_low_noise\.safetensors$/i.test(name)) return 'low noise'
  }
  return name
}

export function stepKey(group, step) {
  return `${group.key}:${step.final ? 'final' : step.step}`
}

/** One ⬇ per FILE — both halves of a pair side by side is what the loaders
 * expect; the URL comes from the lane the save belongs to. */
export function downloadLinks(datasetId, group, step) {
  const files = step.files || []
  return files.map((f) => ({
    filename: f.filename,
    short: fileShortName(f.filename, files.length),
    size: f.size,
    url: group.lane === 'cloud'
      ? videoDatasetCheckpointUrl(datasetId, group.run_id, f.filename)
      : videoDatasetLocalCheckpointUrl(datasetId, f.filename),
  }))
}

/**
 * Everything a step's row renders, decided in one place.
 *
 *   { key, label, files, deployed,
 *     continue: {ok:true}|{reason},
 *     deploy:   {ok:true, folder}|{reason}|null   (null when deployed),
 *     undeploy: {ok:true}|{reason}|null           (null when not deployed),
 *     del:      {ok:true, label, title}|{reason},
 *     details:  bool }
 *
 * `ctx.deleteMode` is the server's `delete_mode`: the 🗑 title names the real
 * destination through the app-wide wording, never a "trash" typed here.
 */
export function stepActionModel(datasetId, group, step, ctx = {}) {
  const { canDeploy = true, deployFolder = 'h3/lds', deleteMode = 'app_trash' } = ctx
  const files = step.files || []
  const deployed = !!step.deployed && files.length > 0
  const cloud = group.lane === 'cloud'

  let cont
  if (!cloud) cont = { reason: CONTINUE_LOCAL_REASON }
  else if (group.active) cont = { reason: ACTIVE_CLOUD_REASON }
  else cont = { ok: true }

  let deploy = null
  let undeploy = null
  if (deployed) {
    undeploy = files.every((f) => f.undeployable) ? { ok: true } : { reason: HAND_PLACED_REASON }
  } else {
    deploy = canDeploy ? { ok: true, folder: deployFolder } : { reason: NO_LORAS_ROOT_REASON }
  }

  let del
  if (group.active) del = { reason: cloud ? ACTIVE_CLOUD_REASON : ACTIVE_LOCAL_REASON }
  else {
    del = {
      ok: true,
      label: files.length > 1 ? 'Delete the training saves' : 'Delete the training save',
      title: `Move every file of this step to ${deleteDestination(deleteMode)}`
        + (isRecoverable(deleteMode) ? ' — recoverable until you empty it' : ''),
    }
  }

  return {
    key: stepKey(group, step), label: stepLabel(step),
    files: downloadLinks(datasetId, group, step),
    deployed, continue: cont, deploy, undeploy, del, details: cloud,
  }
}

const quoted = (files) => (files || []).map((f) => `“${f.filename}”`).join(' + ')

/** The 🗑 confirmation. Names every file the click moves and the destination —
 * from the app-wide wording, never a sentence of this file's own. */
export function describeStepDelete(group, step, mode) {
  const files = step.files || []
  const many = files.length > 1
  const where = deleteDestination(mode)
  const lines = [
    `DELETE THE TRAINING SAVE${many ? 'S' : ''} — ${quoted(files)} (${stepLabel(step)})?`, '',
    many
      ? 'These are the run\'s own checkpoint files — both experts of the pair go together, never half.'
      : 'This is the run\'s own checkpoint file, not a ComfyUI copy.',
    isRecoverable(mode)
      ? `${many ? 'They go' : 'It goes'} to ${where} — recoverable until you empty it.`
      : `${many ? 'They go' : 'It goes'} to ${where}.`,
  ]
  if (deployed(step)) {
    lines.push('', 'The copy deployed into ComfyUI is a separate file and is KEPT — use ⏏ Undeploy for that one.')
  }
  return lines.join('\n')
}

const deployed = (step) => !!step?.deployed

export function describeUndeploy(step, mode = 'app_trash') {
  const files = (step.files || []).filter((f) => f.deployed_as)
  return [
    `UNDEPLOY — REMOVE FROM COMFYUI — ${quoted(files)} (${stepLabel(step)})?`, '',
    `Only the copy in ComfyUI's loras folder goes to ${deleteDestination(mode)}.`,
    'The training save is KEPT — this step offers to deploy again right after.',
  ].join('\n')
}

/** The run-level 🗑 the training block used to carry, moved here unchanged:
 * this one removes the run's files and its history line for good (the server
 * deletes the store directory by name — no trash), and says so. */
export function runDeleteConfirmation(group) {
  const n = (group.steps || []).reduce((sum, s) => sum + (s.files?.length || 0), 0)
  return `Delete run #${group.run_id} and its ${n} LoRA file(s) from disk?\n\n`
    + 'The dataset and its clips are untouched — only this run’s '
    + 'checkpoints and its history line go. This cannot be undone.'
}

export function deleteReport(res = {}) {
  const removed = res.removed || []
  const kept = res.files_kept || []
  const where = deleteDestination(res.delete_mode)
  const head = removed.length
    ? `Moved ${removed.length} file${removed.length === 1 ? '' : 's'} to ${where}.`
    : 'Nothing was moved.'
  if (!kept.length) return head
  return `${head} ${kept.length} file${kept.length === 1 ? '' : 's'} kept — held open by `
    + `another program: ${kept.join(', ')}.`
}

export function deployReport(res = {}) {
  const names = (res.deployed || []).map((n) => String(n).split(/[\\/]/).pop())
  return `Deployed → ${res.folder || 'h3/lds'}: ${names.join(' + ')}. The Video Test Studio lists it now.`
}

export function undeployReport(step) {
  const files = (step?.files || []).filter((f) => f.deployed_as)
  return `Removed from ComfyUI: ${files.map((f) => f.filename).join(' + ')}. The training save is kept.`
}

/** The ▶ Continue body. `from_step` is the harvested step the server seeds the
 * new pod with — for a FINAL save that is the run's total step count, which is
 * the number the listing reports it at. */
export function continueBody(group, step, extraSteps) {
  const extra = Math.max(1, Math.floor(Number(extraSteps) || 0))
  return { run_id: group.run_id, extra_steps: extra, from_step: step.step }
}

const fmtWhen = (iso) => {
  if (!iso) return null
  const t = new Date(/[Z+]/.test(iso) ? iso : `${iso}Z`)
  return Number.isNaN(t.getTime()) ? String(iso) : t.toLocaleString()
}

/** ⓘ label/value rows of one cloud run, in reading order; a value the run
 * does not carry is not a row. */
export function detailsRows(d = {}) {
  const p = d.params || {}
  const yesNo = (v) => (v == null ? null : (v ? 'yes' : 'no'))
  const rows = [
    ['Status', [d.status, d.phase_detail].filter(Boolean).join(' — ') || null],
    ['GPU', d.gpu ? `${d.gpu}${d.price_per_hour != null ? ` · $${Number(d.price_per_hour).toFixed(2)}/h` : ''}` : null],
    ['Requested GPU class', p.requested_gpu ?? null],
    ['Started', fmtWhen(d.created_at)],
    ['Finished', fmtWhen(d.finished_at)],
    ['Continued from', d.parent_run_id != null ? `run #${d.parent_run_id}` : null],
    ['Resumed at step', p.resume_step ?? null],
    ['Steps', p.steps ?? null],
    ['Target', p.target_profile ?? null],
    ['Frames per clip', p.frames ?? null],
    ['Image-to-video', yesNo(p.do_i2v)],
    ['Low VRAM', yesNo(p.low_vram)],
    ['Distillation', p.distillation ?? null],
    ['Base model', p.base_model || null],
    ['Sample prompts', Array.isArray(p.sample_prompts) ? String(p.sample_prompts.length) : null],
    ['Saves on this machine', d.saves ?? null],
    ['Error', d.error || null],
  ]
  return rows.filter(([, v]) => v != null && v !== '').map(([k, v]) => [k, String(v)])
}
