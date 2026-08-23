import { useCallback, useEffect, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { Card, INPUT_CLASS } from './primitives'
import ResetToDefault from './ResetToDefault'
import HfStorageCard from './HfStorageCard'
import Fp8QuantizeTool from '../dataset/Fp8QuantizeTool'
import {
  formatSize, locationRows, moveLabel, movePercent, relocationChoices,
} from './storageLocations.js'

/* Settings › Storage — everything that answers "where does this live and how
   much of my disk does it take".

   It exists because those answers were scattered: the trash was in Maintenance,
   the Hugging Face allowance in Training, the dataset root in a card called
   "Data", and the two directories that actually fill a drive — the cloud run
   staging and the checkpoints — were hard-coded and invisible. When C: fills up,
   "put it on the other drive" was a config.json edit and a manual copy.

   Two rules run through the whole tab:
     - sizes are measured on demand, never on mount (walking a 127 GB tree is not
       something a tab is allowed to do while you are reading it);
     - a location change never moves files by itself. You choose. */

/* One relocatable root. Type a path, check it, then pick what happens to the
   files already there — the two answers are spelled out before anything runs. */
function LocationEditor({
  id, storageKey, label, help, section, field, current, sizeBytes,
  config, setField, configDefaults, saveConfigPatch, toast, onChanged,
}) {
  const stored = (config[section] || {})[field] || ''
  const [draft, setDraft] = useState(stored)
  const [check, setCheck] = useState(null)
  const [checking, setChecking] = useState(false)
  const [job, setJob] = useState(null)

  useEffect(() => { setDraft(stored) }, [stored])

  const validate = async () => {
    setChecking(true)
    try {
      setCheck(await postJson('/api/storage/validate', { key: storageKey, path: draft }))
    } catch (e) {
      setCheck({ ok: false, reason: e.message || 'Could not check that folder.' })
    } finally {
      setChecking(false)
    }
  }

  const persist = async (value) => {
    // Saved through the explicit patch, not setField + a section save: the state
    // update is not visible to this callback, so that version saved nothing at
    // all while reporting success (caught headless before it shipped).
    await saveConfigPatch(section, { [field]: value })
    onChanged?.()
  }

  const adopt = async () => {
    try {
      await persist(check?.default ? '' : (check?.path || draft))
      toast?.success(check?.default ? 'Back to the default folder.' : 'New folder in use.')
      setCheck(null)
    } catch (e) {
      toast?.error(e.message || 'Could not save the new location.')
    }
  }

  const move = async () => {
    let started
    try {
      started = await postJson('/api/storage/move', { key: storageKey, path: check.path })
    } catch (e) {
      toast?.error(e.message || 'Could not start the move.')
      return
    }
    setJob({ phase: 'scanning' })
    for (let i = 0; i < 100000; i += 1) {
      await new Promise((r) => setTimeout(r, 700))
      let state
      try {
        state = (await apiFetch(`/api/storage/move/progress?job_id=${started.job_id}`,
          { background: true })).job
      } catch { continue }
      setJob(state)
      if (state?.phase === 'error') { toast?.error(state.error || 'The move failed.'); return }
      if (state?.phase === 'done') break
    }
    // The config only points at the new folder once every byte is there.
    try {
      await persist(check.path)
      toast?.success('Files moved and the new folder is in use.')
      setCheck(null)
    } catch (e) {
      toast?.error(e.message || 'Files moved, but the location could not be saved.')
    }
  }

  const choices = relocationChoices({ validation: check, currentSize: sizeBytes })
  const busy = job && ['scanning', 'copying'].includes(job.phase)
  return (
    <Card title={label} help={help}>
      <p className="break-all text-xs text-content-subtle">
        <span className="text-content-muted">In use now:</span> {current || '—'}
      </p>
      <div>
        <label htmlFor={id} className="block text-sm font-medium text-content">
          Folder (leave empty for the default)
        </label>
        {/* Column on a phone: a path field and two buttons never share 400 px. */}
        <div className="mt-1 flex flex-col gap-2 sm:flex-row">
          <input id={id} type="text" value={draft} disabled={busy}
            onChange={(e) => { setDraft(e.target.value); setCheck(null) }}
            placeholder="Defaults to the app’s data folder"
            className={`${INPUT_CLASS} sm:flex-1`} />
          <button type="button" onClick={validate} disabled={checking || busy}
            className="shrink-0 rounded-md border border-border-strong px-3 py-1.5 text-sm font-medium text-content hover:bg-surface-raised disabled:opacity-50">
            {checking ? 'Checking…' : 'Check folder'}
          </button>
        </div>
        <div className="mt-1">
          <ResetToDefault label={label} section={section} field={field}
            config={config} configDefaults={configDefaults} setField={setField} />
        </div>
      </div>

      {check && !check.ok && (
        <p className="rounded-lg border border-rose-500/40 bg-rose-500/10 p-2 text-xs text-rose-200">
          <span aria-hidden>⚠</span> {check.reason}
        </p>
      )}

      {choices.length > 0 && !busy && (
        <div className="space-y-2 rounded-lg border border-border bg-surface-raised p-3">
          <p className="text-xs text-content-muted">
            {check.default
              ? 'This goes back to the folder inside the app’s data directory.'
              : `${check.path} is writable${check.empty === false ? ' and already has files in it' : ''}.`}
          </p>
          {choices.map((choice) => (
            <div key={choice.id} className="flex flex-col gap-1 sm:flex-row sm:items-start sm:justify-between">
              <p className="min-w-0 text-xs text-content-subtle">{choice.detail}</p>
              <button type="button" disabled={choice.disabled}
                onClick={() => (choice.id === 'move' ? move() : adopt())}
                className="shrink-0 self-start rounded-md border border-border-strong px-3 py-1.5 text-xs font-medium text-content hover:bg-surface-raised disabled:opacity-40">
                {choice.label}
              </button>
            </div>
          ))}
        </div>
      )}

      {job && (
        <div className="space-y-1" role="status" aria-live="polite">
          <p className="text-xs text-content-muted">{moveLabel(job)}</p>
          {busy && (
            <div className="h-1.5 w-full max-w-sm overflow-hidden rounded-full bg-surface-raised">
              <div className="h-full rounded-full bg-gradient-primary transition-[width] duration-300"
                style={{ width: `${movePercent(job) ?? 15}%` }} />
            </div>
          )}
        </div>
      )}
    </Card>
  )
}

/* The cleanup that used to destroy weights, and the folders it used to ignore.
   Both live here rather than on the Runs hub: this is the disk tab. */
function CloudRunHousekeeping({ toast, onChanged }) {
  const [orphans, setOrphans] = useState(null)
  const [busy, setBusy] = useState('')

  const scan = async () => {
    setBusy('scan')
    try {
      setOrphans((await apiFetch('/api/dataset/train/cloud/orphans')).orphans || [])
    } catch (e) {
      toast?.error(e.message || 'Could not scan the run folders.')
    } finally {
      setBusy('')
    }
  }

  const purge = async () => {
    const total = (orphans || []).reduce((n, o) => n + (o.size_bytes || 0), 0)
    const withCkpt = (orphans || []).filter((o) => o.checkpoints > 0).length
    if (!window.confirm(
      `Move ${orphans.length} unclaimed run folder(s) (${formatSize(total)}) to the trash?\n\n`
      + (withCkpt ? `${withCkpt} of them still hold a checkpoint — those are moved into the checkpoint store first, never deleted.\n\n` : '')
      + 'Files go to the trash on this same disk; the space comes back when you empty it.',
    )) return
    setBusy('purge')
    try {
      const res = await postJson('/api/dataset/train/cloud/purge-orphans', {})
      toast?.success(`${res.purged_dirs} folder(s) moved to the trash`
        + (res.rescued_checkpoints ? `, ${res.rescued_checkpoints} checkpoint(s) rescued` : '')
        + '.')
      await scan()
      onChanged?.()
    } catch (e) {
      toast?.error(e.message || 'Could not clean those folders.')
    } finally {
      setBusy('')
    }
  }

  const adopt = async () => {
    setBusy('adopt')
    try {
      const res = await postJson('/api/storage/adopt-checkpoints', {})
      toast?.success(res.moved
        ? `${res.moved} checkpoint(s) moved out of the run folders and into the store.`
        : 'Every checkpoint is already in the store.')
      onChanged?.()
    } catch (e) {
      toast?.error(e.message || 'Could not move the checkpoints.')
    } finally {
      setBusy('')
    }
  }

  return (
    <Card title="Cloud run housekeeping"
      help="Cleaning a finished run trashes its dataset copy, its sample images and its logs — never a checkpoint. Everything goes to the trash on the same disk, so the space only comes back when you empty it.">
      <div className="flex flex-wrap items-center gap-2">
        <button id="storage-scan-orphans" type="button" onClick={scan} disabled={!!busy}
          className="rounded-md border border-border-strong px-3 py-1.5 text-sm font-medium text-content hover:bg-surface-raised disabled:opacity-50">
          {busy === 'scan' ? 'Scanning…' : 'Find unclaimed run folders'}
        </button>
        <button id="storage-adopt-checkpoints" type="button" onClick={adopt} disabled={!!busy}
          className="rounded-md border border-border px-3 py-1.5 text-sm font-medium text-content hover:bg-surface-raised disabled:opacity-50">
          {busy === 'adopt' ? 'Moving…' : 'Move stray checkpoints into the store'}
        </button>
      </div>
      {orphans && orphans.length === 0 && (
        <p className="text-xs text-content-muted">
          Every run folder on disk belongs to a run this app knows about.
        </p>
      )}
      {orphans && orphans.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm text-content">
            {orphans.length} folder(s) no run points at —{' '}
            <span className="font-semibold tabular-nums">
              {formatSize(orphans.reduce((n, o) => n + (o.size_bytes || 0), 0))}
            </span>.
          </p>
          <ul className="space-y-1 text-xs text-content-subtle">
            {orphans.map((o) => (
              <li key={o.name} className="break-all">
                {o.name} — {formatSize(o.size_bytes)}
                {o.checkpoints > 0 && (
                  <span className="text-amber-300">
                    {' '}· holds {o.checkpoints} checkpoint(s), which will be rescued
                  </span>
                )}
              </li>
            ))}
          </ul>
          <button type="button" onClick={purge} disabled={!!busy}
            className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-1.5 text-sm font-medium text-red-300 disabled:opacity-40">
            {busy === 'purge' ? 'Cleaning…' : 'Move them to the trash'}
          </button>
        </div>
      )}
    </Card>
  )
}

/* App-wide trash: everything the app "deletes" (checkpoints, cloud staging,
   deployed LoRAs) is MOVED here — this card is the only place bytes actually
   die. Size fetched once on mount (no poll). */
function TrashCard({ reloadKey }) {
  const [size, setSize] = useState(null)
  const [busy, setBusy] = useState(false)
  const [opening, setOpening] = useState(false)
  useEffect(() => {
    let alive = true
    apiFetch('/api/trash')
      .then((d) => { if (alive) setSize(d?.size_bytes ?? null) })
      .catch(() => { /* best-effort */ })
    return () => { alive = false }
  }, [reloadKey])
  const openFolder = async () => {
    setOpening(true)
    try {
      const d = await postJson('/api/trash/open', {})
      if (!d?.ok) window.alert(d?.error || 'Could not open the trash folder.')
    } catch {
      window.alert('Could not open the trash folder.')
    } finally {
      setOpening(false)
    }
  }
  const empty = async () => {
    if (!window.confirm('Permanently delete everything in the trash?\n\nThis is the ONLY destructive action — deleted checkpoints cannot be recovered afterwards.')) return
    setBusy(true)
    try {
      const d = await postJson('/api/trash/empty', {})
      if (d?.ok) setSize(0)
    } finally {
      setBusy(false)
    }
  }
  return (
    <Card title="Trash" help="Everything the app deletes (checkpoints, cloud staging, deployed LoRAs) is moved here first — emptying it is the only action that actually destroys files. It sits on the same disk, so cleaning does not give space back until you empty it.">
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-sm text-content">
          <span aria-hidden>🗑</span> Trash size:{' '}
          <span className="font-semibold tabular-nums">{size == null ? '…' : formatSize(size)}</span>
        </span>
        <button type="button" onClick={openFolder} disabled={opening}
          title="Open the trash folder in the file explorer"
          className="rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm font-medium text-content disabled:opacity-40">
          {opening ? 'Opening…' : '📂 Open folder'}
        </button>
        <button type="button" onClick={empty} disabled={busy || !size}
          className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-1.5 text-sm font-medium text-red-300 disabled:opacity-40">
          {busy ? 'Emptying…' : 'Empty trash'}
        </button>
      </div>
    </Card>
  )
}

/* The run image archive: a deduplicated copy of every image a training run was
   launched on, so a comparison can still SHOW an image that has since been
   deleted from its dataset. Content-addressed, so an unchanged dataset costs
   nothing on its second launch — but it is still bytes, so its size is visible
   and clearable here rather than growing invisibly. */
function RunArchiveCard() {
  const [info, setInfo] = useState(null)
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    let alive = true
    apiFetch('/api/run-archive')
      .then((d) => { if (alive) setInfo(d || null) })
      .catch(() => { /* best-effort */ })
    return () => { alive = false }
  }, [])
  const clear = async () => {
    if (!window.confirm('Delete every archived training image?\n\nYour runs, their settings and their captions are kept — you just lose the ability to look at images that have since been deleted from their dataset.')) return
    setBusy(true)
    try {
      const d = await postJson('/api/run-archive/clear', {})
      if (d?.ok) setInfo((v) => ({ ...(v || {}), size_bytes: 0 }))
    } finally {
      setBusy(false)
    }
  }
  return (
    <Card title="Run image archive" help="When a training run is launched, a deduplicated copy of the images it trains on is kept so that comparing two runs can still show an image you have since deleted. Only new or edited images are copied, and the archive stops growing at its ceiling.">
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-sm text-content">
          <span aria-hidden>🗂</span> Archive size:{' '}
          <span className="font-semibold tabular-nums">
            {info == null ? '…' : formatSize(info.size_bytes || 0)}
          </span>
          {info?.max_bytes ? (
            <span className="text-content-subtle"> / {formatSize(info.max_bytes)} ceiling</span>
          ) : null}
        </span>
        {info && !info.enabled && (
          <span className="text-xs text-content-subtle">Archiving is turned off.</span>
        )}
        <button type="button" onClick={clear} disabled={busy || !info?.size_bytes}
          className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-1.5 text-sm font-medium text-red-300 disabled:opacity-40">
          {busy ? 'Clearing…' : 'Clear archive'}
        </button>
      </div>
    </Card>
  )
}

export default function StorageSection({
  config, setField, configDefaults, saveConfigPatch, toast,
}) {
  const [locations, setLocations] = useState([])
  const [sizes, setSizes] = useState({})
  const [measuring, setMeasuring] = useState(false)
  const [reloadKey, setReloadKey] = useState(0)

  const loadLocations = useCallback(async () => {
    try {
      setLocations((await apiFetch('/api/storage/locations')).locations || [])
    } catch { /* the editors below still work off the config */ }
  }, [])
  useEffect(() => { loadLocations() }, [loadLocations, reloadKey])

  const measure = async () => {
    setMeasuring(true)
    try {
      setSizes((await apiFetch('/api/storage/sizes')).sizes || {})
    } catch (e) {
      toast?.error(e.message || 'Could not measure the folders.')
    } finally {
      setMeasuring(false)
    }
  }

  const rows = locationRows(locations, sizes)
  const byKey = Object.fromEntries(rows.map((r) => [r.key, r]))
  const changed = () => { setReloadKey((k) => k + 1); setSizes({}) }
  // Shared props of the three relocatable roots. The DOM `id` stays a LITERAL on
  // each element below (the help registry's focus targets are matched against
  // the JSX source, and a computed id is invisible to it).
  const shared = (key) => ({
    current: byKey[key]?.path,
    sizeBytes: byKey[key]?.sizeBytes,
    config, setField, configDefaults, saveConfigPatch, toast, onChanged: changed,
  })

  return (
    <div className="space-y-6">
      <Card title="What lives where"
        help="Every folder this app writes to, with the drive it sits on. Sizes are measured only when you ask — walking a hundred gigabytes of datasets is not something a page should do while you read it.">
        <div className="flex flex-wrap items-center gap-3">
          <button id="storage-measure" type="button" onClick={measure} disabled={measuring}
            className="rounded-md border border-border-strong px-3 py-1.5 text-sm font-medium text-content hover:bg-surface-raised disabled:opacity-50">
            {measuring ? 'Measuring…' : '📏 Measure everything'}
          </button>
          <span className="text-xs text-content-subtle">
            {Object.keys(sizes).length ? 'Measured just now.' : 'Not measured yet.'}
          </span>
        </div>
        {/* The table scrolls inside its own box: a long Windows path must never
            make the whole page scroll sideways on a phone. */}
        <div className="-mx-2 overflow-x-auto px-2">
          <ul className="min-w-[18rem] space-y-2">
            {rows.map((row) => (
              <li key={row.key} className="rounded-lg border border-border bg-surface-raised p-3">
                <div className="flex flex-col gap-1 sm:flex-row sm:items-baseline sm:justify-between">
                  <p className="text-sm font-medium text-content">
                    {row.label}
                    {row.relocatable && (
                      <span className="ml-2 rounded border border-border px-1 text-[0.625rem] uppercase tracking-wide text-content-subtle">
                        movable
                      </span>
                    )}
                  </p>
                  <p className="shrink-0 text-sm tabular-nums text-content">{row.sizeLabel}</p>
                </div>
                <p className="mt-0.5 break-all text-xs text-content-subtle">
                  {row.path || 'not configured'}{row.exists ? '' : ' (not created yet)'}
                </p>
                <p className="mt-0.5 text-xs text-content-muted">{row.holds}</p>
                {row.volumeLabel && (
                  <p className="mt-0.5 text-[0.6875rem] text-content-subtle">{row.volumeLabel}</p>
                )}
              </li>
            ))}
          </ul>
        </div>
      </Card>

      <LocationEditor id="dataset-images-root" storageKey="datasets"
        label="Dataset images root" section="paths" field="dataset_images_root"
        help="Where dataset images live on disk — usually the biggest folder of all."
        {...shared('datasets')} />

      <LocationEditor id="cloud-runs-dir" storageKey="cloud_runs"
        label="Cloud run staging" section="paths" field="cloud_runs_dir"
        help="Working files of cloud training runs — the exported dataset copy, the sample images and the logs. This is the folder that grows to tens of gigabytes; cleaning a finished run empties it without ever touching a checkpoint."
        {...shared('cloud_runs')} />

      <LocationEditor id="checkpoints-dir" storageKey="checkpoints"
        label="Checkpoint store" section="paths" field="checkpoints_dir"
        help="Where the .safetensors your cloud runs produce are kept for good. No cleanup in the app ever removes a file from here — only you can, from the Checkpoints panel or by emptying the trash."
        {...shared('checkpoints')} />

      <CloudRunHousekeeping toast={toast} onChanged={changed} />
      <TrashCard reloadKey={reloadKey} />
      <RunArchiveCard />

      {/* The SAME component the full-model recipe card renders — imported, not
          copied, so the refusals (already quantized, LoRA, overwriting the
          source) and the read-back verification can never differ between the two
          doors. It is here because its first door is inside a dense dataset's
          recipe, which somebody who downloaded a 26 GB model from Hugging Face
          and has no dataset at all never opens — and "this file is too big" is a
          disk question, asked on this tab. `framed={false}` drops its own accent
          box and title: the Card below already says both. */}
      <Card id="storage-fp8-quantize" title="Quantize an existing model to fp8"
        help="A full-precision model — downloaded from Hugging Face, or delivered by a full-model run — is about 2.5× the size ComfyUI needs to generate with it. Point this at one and it writes the ~10 GB fp8 version next to the original, on this machine, without ever modifying the source.">
        <Fp8QuantizeTool framed={false} />
      </Card>

      <HfStorageCard config={config} setField={setField} configDefaults={configDefaults} />
    </div>
  )
}
