import { useCallback, useEffect, useState } from 'react'
import { apiFetch, del, postJson } from '../api/fetchClient'
import { useToast } from '../components/common/Toast'
import { HelpBadge } from '../help/HelpMode'
import BankWorkspace from '../components/bank/BankWorkspace'
import FolderPickerField from '../components/common/FolderPicker'
import { hiddenCount, previewSlots } from '../components/bank/bankPreview'
import { bankListSyncToast } from '../components/bank/bankSync'
import { overlapNotice } from '../components/bank/bankOverlap'
import { datasetFolderNotice } from '../utils/pathRelation'
import FolderSyncNote from '../components/bank/FolderSyncNote'
import FolderCheckLine from '../components/bank/FolderCheckLine'
import RelocateBankDialog from '../components/bank/RelocateBankDialog'
import BankScrapePanel from '../components/bank/BankScrapePanel'
import BankLaneTabs from '../components/videobank/BankLaneTabs'
import { bankListOverview } from '../components/bank/bankOverview.js'

const CURRENT_KEY = 'bankCurrentId'

/** The card's thumbnail strip: the bank's first few images, so a list of banks
 * reads at a glance instead of as a wall of folder paths. Clicking a thumbnail
 * opens the bank, like the title and the Open button. Thumbnails are served by
 * the same route the workspace grid uses (generated on demand when the bank was
 * never scanned) and load lazily, so an off-screen card costs nothing. */
function BankPreviewStrip({ bank, onOpen }) {
  if (!bank.preview_ids?.length) return null
  const extra = hiddenCount(bank.total, bank.preview_ids)
  return (
    <div className="relative grid grid-cols-5 gap-1">
      {previewSlots(bank.preview_ids).map((id, i) => (
        <div key={id ?? `empty-${i}`}
          className="aspect-square overflow-hidden rounded border border-border bg-surface-raised">
          {id != null && (
            <button type="button" onClick={onOpen} tabIndex={-1} aria-hidden="true"
              className="block h-full w-full">
              <img src={`/api/bank/${bank.id}/thumb/${id}`} alt="" loading="lazy"
                onError={(e) => { e.currentTarget.style.visibility = 'hidden' }}
                className="h-full w-full object-cover" />
            </button>
          )}
        </div>
      ))}
      {extra > 0 && (
        <span className="pointer-events-none absolute bottom-1 right-1 rounded bg-black/60 px-1 text-[0.625rem] font-semibold text-white">
          +{extra}
        </span>
      )}
    </div>
  )
}

const BANK_STATUS_TONE = {
  keep: 'bg-emerald-400', pending: 'bg-amber-300', reject: 'bg-rose-400',
}

function BankListSummary({ bank }) {
  const summary = bankListOverview(bank)
  return (
    <div className="space-y-1.5">
      {summary.total > 0 ? (
        <div className="flex h-2 overflow-hidden rounded-full bg-surface-raised" role="img"
          aria-label={summary.status.map((row) => `${row.label}: ${row.value}, ${row.percent}%`).join('; ')}>
          {summary.status.filter((row) => row.value > 0).map((row) => (
            <span key={row.id} className={BANK_STATUS_TONE[row.id]}
              style={{ width: `${row.widthPercent}%`, minWidth: '1px' }} />
          ))}
        </div>
      ) : summary.total === 0
        ? <p className="text-[11px] text-content-subtle">No images.</p>
        : <p className="text-[11px] text-amber-300/90">Curation totals unavailable.</p>}
      <ul className="flex flex-wrap gap-x-2 gap-y-0.5 text-[11px] text-content-muted">
        {summary.status.map((row) => (
          <li key={row.id} className="tabular-nums">{row.label} <span className="text-content">{row.value ?? '—'}</span>
            {row.percent != null && <span className="text-content-subtle"> · {row.percent}%</span>}</li>
        ))}
      </ul>
      <div className="flex items-center gap-2 text-[11px]">
        <span className="text-content-muted">Quality</span>
        <span className={summary.scanPercent == null ? 'text-amber-300/90' : 'text-content-subtle'}>
          {summary.scanText}
        </span>
      </div>
    </div>
  )
}

/** 🗃️ Image bank — triage a big unsorted folder BEFORE it becomes datasets.
 * List view (create/open/delete banks) + per-bank workspace. The bank
 * references the folder in place: nothing is copied until promotion, and the
 * source files are never modified. */
export default function BankPage() {
  const toast = useToast()
  const [banks, setBanks] = useState(null)
  const [currentId, setCurrentId] = useState(() => {
    try { return Number(localStorage.getItem(CURRENT_KEY)) || null } catch { return null }
  })
  const [name, setName] = useState('')
  const [folder, setFolder] = useState('')
  const [creating, setCreating] = useState(false)
  const [relocating, setRelocating] = useState(null)   // the bank being repointed
  // Dataset storage folders, so a folder that belongs to a dataset can be named
  // as such WHILE it is typed. The server refuses it either way — this only
  // spares the round-trip and the "why not?" (see utils/pathRelation.js).
  const [datasets, setDatasets] = useState([])

  // ⚠️ Plain loads do NOT re-walk the source folders any more: doing that cost a
  // full disk inventory of the whole library on every navigation to this page
  // (690-1 190 ms on a real 8-bank / 86 493-image library). `rescan` is the 🔄
  // button, and it is the only caller that asks the server to walk.
  const refresh = useCallback(async ({ rescan = false } = {}) => {
    try {
      const d = await apiFetch(`/api/banks${rescan ? '?rescan=1' : ''}`)
      setBanks(d.banks || [])
      if (!rescan) return
      // A walk just happened: say what it found, so the counters never move
      // without an explanation — and say so even when it found nothing, because
      // silence after a click reads as a broken button.
      const note = bankListSyncToast(d.banks)
      if (note) toast[note.type](note.text)
      else toast.success('Source folders checked — no new image found.')
    } catch (e) {
      toast.error(e?.message || 'Could not load the banks.')
      if (!rescan) setBanks([])
    }
  }, [toast])

  const [rescanning, setRescanning] = useState(false)
  const rescan = async () => {
    if (rescanning) return
    setRescanning(true)
    try { await refresh({ rescan: true }) } finally { setRescanning(false) }
  }

  useEffect(() => { if (currentId == null) refresh() }, [currentId, refresh])

  // Best effort: a failed list just means no live hint, never a broken form.
  useEffect(() => {
    if (currentId != null) return undefined
    let alive = true
    apiFetch('/api/dataset/list')
      .then((d) => { if (alive) setDatasets(d.datasets || []) })
      .catch(() => { if (alive) setDatasets([]) })
    return () => { alive = false }
  }, [currentId])

  const folderNotice = datasetFolderNotice(folder, datasets)

  const open = (id) => {
    try { localStorage.setItem(CURRENT_KEY, String(id)) } catch { /* ignore */ }
    setCurrentId(id)
  }
  const close = () => {
    try { localStorage.removeItem(CURRENT_KEY) } catch { /* ignore */ }
    setCurrentId(null)
  }

  const create = async (e) => {
    e.preventDefault()
    // A bank over a dataset's folder would share the dataset's LIVE files; the
    // server refuses it, and so does the form (the notice says what to do).
    if (creating || folderNotice) return
    setCreating(true)
    try {
      const d = await postJson('/api/bank/create', { name, folder })
      toast.success(`Bank created — ${d.added} image(s) inventoried.`)
      // Nested folders mean two banks over the same files. Harmless while
      // triaging, destructive at 🗑 Delete rejected — said once, up front.
      const overlap = overlapNotice(d.overlaps)
      if (overlap) toast.warning(overlap, 12000)
      setName(''); setFolder('')
      open(d.id)
    } catch (err) {
      toast.error(err?.message || 'Could not create the bank.')
    } finally {
      setCreating(false)
    }
  }

  const remove = async (bank) => {
    if (!window.confirm(`Remove the bank “${bank.name}”?\n\nOnly the triage data (decisions, scores, thumbnails) is deleted — the source folder and its images are NOT touched.`)) return
    try {
      await del(`/api/bank/${bank.id}`)
      toast.success('Bank removed — source folder untouched.')
      refresh()
    } catch (e) {
      toast.error(e?.message || 'Could not remove the bank.')
    }
  }

  if (currentId != null) {
    return <BankWorkspace bankId={currentId} onBack={close} onGone={close} />
  }

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-center gap-2">
        <h1 className="text-xl font-bold text-content">🗃️ Image bank</h1>
        <HelpBadge topic="page-bank" />
        {/* The kind of bank you are making, said WHERE you make one. Until now a
            .mp4 dropped in this folder was skipped in silence — this is the only
            place someone with a folder of rushes would ever have looked. */}
        <BankLaneTabs className="w-full sm:ml-auto sm:w-auto" />
      </header>
      <p className="text-sm text-content-muted max-w-3xl">
        Point the app at a big unsorted folder (a Telegram export, a scrape dump…) and triage it
        into dataset-ready selections: a quality pass flags blur/noise/flat/small shots and groups
        near-duplicates, the face pass sorts the dump by person — then you promote the keepers
        into a dataset. No pass ever modifies your files; the one thing that adds to the folder
        is a scrape you send to this bank yourself.
      </p>

      <form onSubmit={create}
        className="flex flex-wrap items-end gap-3 rounded-lg border border-border bg-surface p-4">
        <div className="grow min-w-40">
          <label htmlFor="bank-name" className="block text-sm font-medium text-content">Name</label>
          <input id="bank-name" value={name} onChange={(e) => setName(e.target.value)}
            placeholder="Telegram export 07/2026" required
            className="mt-1 w-full rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content" />
        </div>
        <div className="grow-[3] min-w-64">
          <FolderPickerField id="bank-folder" label="Folder on this computer"
            value={folder} onChange={setFolder} required
            placeholder="C:\path\to\unsorted-images (subfolders included)" />
        </div>
        <button type="submit" disabled={creating || !!folderNotice}
          title={folderNotice ? 'That folder belongs to a dataset' : undefined}
          className="rounded-md bg-gradient-primary px-4 py-2 text-sm font-semibold text-white disabled:opacity-50">
          {creating ? 'Inventorying…' : '➕ Create bank'}
        </button>
        {/* basis-full: its own row inside the wrapping flex form, so the sentence
            never squeezes the fields — including at 400 px. */}
        {folderNotice && (
          <p role="alert"
            className="basis-full rounded-md border border-rose-500/70 bg-rose-500/15 p-3 text-sm text-rose-100">
            ⛔ {folderNotice.text}
          </p>
        )}
      </form>

      {/* Second way in: the scraper's own destination. A bank no longer needs a
          folder you prepared by hand — you can fill one straight from the web. */}
      <BankScrapePanel banks={banks} onDone={() => refresh()} />

      <FolderCheckLine banks={banks} busy={rescanning} onRescan={rescan} />

      {banks == null ? (
        <p className="text-sm text-content-muted">Loading…</p>
      ) : banks.length === 0 ? (
        <p className="text-sm text-content-muted">
          No bank yet — create one above to start triaging a folder.
        </p>
      ) : (
        // grid-cols-1 (= minmax(0,1fr)), NOT the implicit auto column: an auto
        // column is sized on max-content, so the unbreakable source PATH inside
        // a card stretched it past the viewport and scrolled the whole page
        // sideways on a phone — with `truncate` never getting a chance to fire.
        <ul className="grid gap-3 grid-cols-1 sm:grid-cols-2 xl:grid-cols-3">
          {banks.map((b) => (
            <li key={b.id}
              className="flex min-w-0 flex-col gap-2 rounded-lg border border-border bg-surface p-4">
              <div className="flex min-w-0 items-center gap-2">
                <button type="button" onClick={() => open(b.id)}
                  className="min-w-0 truncate text-left text-base font-semibold text-content hover:underline">
                  {b.name}
                </button>
                {b.activity && !b.activity.finished && (
                  <span className="text-xs text-amber-300">⏳ {b.activity.kind}…</span>
                )}
                <button type="button" onClick={() => setRelocating(b)}
                  aria-label={`Move the folder of bank ${b.name}`}
                  title="Moved this folder to another disk? Point the bank at its new location."
                  className="ml-auto px-1.5 text-content-subtle hover:text-content">📦</button>
                <button type="button" onClick={() => remove(b)} aria-label={`Remove bank ${b.name}`}
                  className="px-1.5 text-content-subtle hover:text-rose-300">✕</button>
              </div>
              <p className="truncate font-mono text-xs text-content-subtle" title={b.source_path}>
                {b.source_path}
              </p>
              <BankPreviewStrip bank={b} onOpen={() => open(b.id)} />
              <BankListSummary bank={b} />
              <FolderSyncNote sync={b.folder_sync}
                onRelocate={() => setRelocating(b)} />
              <button type="button" onClick={() => open(b.id)}
                className="self-start rounded-md border border-border bg-surface-raised px-3 py-1 text-xs font-semibold text-content hover:bg-surface">
                Open →
              </button>
            </li>
          ))}
        </ul>
      )}

      {relocating && (
        <RelocateBankDialog bankId={relocating.id} bankName={relocating.name}
          sourcePath={relocating.source_path}
          onClose={() => setRelocating(null)} onDone={() => refresh()} />
      )}
    </div>
  )
}
