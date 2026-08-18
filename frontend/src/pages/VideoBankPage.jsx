import { useCallback, useEffect, useState } from 'react'
import { apiFetch, del, postJson } from '../api/fetchClient'
import { useToast } from '../components/common/Toast'
import { HelpBadge } from '../help/HelpMode'
import FolderPickerField from '../components/common/FolderPicker'
import BankLaneTabs from '../components/videobank/BankLaneTabs'
import VideoBankScrapePanel from '../components/videobank/VideoBankScrapePanel'
import VideoBankWorkspace from '../components/videobank/VideoBankWorkspace'
import VideoCapabilityStrip from '../components/videobank/VideoCapabilityStrip'
import { countsSummary } from '../components/videobank/videoBankStatus'

const CURRENT_KEY = 'videoBankCurrentId'

/** 🎬 Video bank — triage a folder of RUSHES before it becomes a training set.
 *
 * WHY THIS IS A SEPARATE KIND OF BANK. Drop a .mp4 into an image bank today and
 * it is skipped in silence: no row, no warning, nothing to click. That is the
 * defect this lane exists to close, and closing it properly needs a different
 * unit of work — an image bank triages FILES, this one triages SHOTS, of which
 * one file yields hundreds.
 *
 * Nothing is copied and nothing is encoded here: the bank stores the bounds of
 * each detected shot. Video is only ever written at promotion, for the shots you
 * kept — which is why a bank of 340 shots costs no disk at all until you have
 * decided which 128 of them are worth an encode.
 */
export default function VideoBankPage() {
  const toast = useToast()
  const [banks, setBanks] = useState(null)
  const [capability, setCapability] = useState(null)
  const [currentId, setCurrentId] = useState(() => {
    try { return Number(localStorage.getItem(CURRENT_KEY)) || null } catch { return null }
  })
  const [name, setName] = useState('')
  const [folder, setFolder] = useState('')
  const [creating, setCreating] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const d = await apiFetch('/api/video-banks')
      setBanks(d.banks || [])
    } catch (e) {
      toast.error(e?.message || 'Could not load the video banks.')
      setBanks([])
    }
  }, [toast])

  useEffect(() => { if (currentId == null) refresh() }, [currentId, refresh])

  // The capability probe lives on a BANK's payload, so the list page has no
  // bank to read it from. Borrow the first one's — and stay silent when there is
  // none: telling someone with no bank that ffmpeg is missing is a lecture, not
  // a warning. The create form still works either way; only promotion needs it.
  useEffect(() => {
    if (currentId != null || !banks?.length) { setCapability(null); return undefined }
    let alive = true
    apiFetch(`/api/video-bank/${banks[0].id}`, { background: true })
      .then((d) => { if (alive) setCapability(d.capability || null) })
      .catch(() => { if (alive) setCapability(null) })
    return () => { alive = false }
  }, [banks, currentId])

  // useCallback, and not for micro-optimisation: `close` is handed to the
  // workspace as `onGone`, which ends up in the dependency list of its 2 s poll.
  // A fresh identity every render would tear the interval down and rebuild it on
  // every render, and a bank that re-renders faster than the poll would then
  // never actually poll.
  const open = useCallback((id) => {
    try { localStorage.setItem(CURRENT_KEY, String(id)) } catch { /* ignore */ }
    setCurrentId(id)
  }, [])
  const close = useCallback(() => {
    try { localStorage.removeItem(CURRENT_KEY) } catch { /* ignore */ }
    setCurrentId(null)
  }, [])

  const create = async (e) => {
    e.preventDefault()
    if (creating) return
    setCreating(true)
    try {
      const d = await postJson('/api/video-bank/create', { name, folder })
      toast.success(`Video bank created — ${d.added} file(s) inventoried.`)
      setName(''); setFolder('')
      open(d.id)
    } catch (err) {
      toast.error(err?.message || 'Could not create the video bank.')
    } finally {
      setCreating(false)
    }
  }

  const remove = async (bank) => {
    // eslint-disable-next-line no-alert
    if (!window.confirm(`Remove the video bank “${bank.name}”?\n\nOnly the triage data (detected shots, decisions, thumbnails) is deleted — your video files and any dataset already built from them are NOT touched.`)) return
    try {
      await del(`/api/video-bank/${bank.id}`)
      toast.success('Video bank removed — your files are untouched.')
      refresh()
    } catch (e) {
      toast.error(e?.message || 'Could not remove the video bank.')
    }
  }

  if (currentId != null) {
    return <VideoBankWorkspace bankId={currentId} onBack={close} onGone={close} />
  }

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-center gap-2">
        <h1 className="flex items-center gap-2 text-xl font-bold text-content">
          🎬 Video bank
          <span className="rounded-full border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-amber-600 dark:text-amber-400">
            Beta
          </span>
        </h1>
        <HelpBadge topic="page-video-bank" />
        <BankLaneTabs className="w-full sm:ml-auto sm:w-auto" />
      </header>
      <p className="max-w-3xl text-sm text-content-muted">
        Point the app at a folder of rushes and turn it into a video training set: each
        file is cut at its shot boundaries, you keep the shots worth training on, and
        only those get encoded — at the length and frame rate your target model
        actually demands. Nothing is copied and nothing is re-encoded until you promote,
        so a bank of hundreds of shots costs no disk space at all.
      </p>

      <VideoCapabilityStrip capability={capability} />

      <form onSubmit={create}
        className="flex flex-wrap items-end gap-3 rounded-lg border border-border bg-surface p-4">
        <div className="grow min-w-40">
          <label htmlFor="video-bank-name" className="block text-sm font-medium text-content">Name</label>
          <input id="video-bank-name" value={name} onChange={(e) => setName(e.target.value)}
            placeholder="City rushes 08/2026" required
            className="mt-1 w-full rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content" />
        </div>
        <div className="grow-[3] min-w-64">
          <FolderPickerField id="video-bank-folder" label="Folder on this computer"
            value={folder} onChange={setFolder} required
            placeholder="path\to\rushes (subfolders included)"
            hint="Reads .mp4, .mov, .mkv, .webm and .avi. No pass ever modifies your files: scanning, cutting and building all write elsewhere. The one thing that adds to this folder is a scrape you send to this bank yourself." />
        </div>
        <button type="submit" disabled={creating}
          className="rounded-md bg-gradient-primary px-4 py-2 text-sm font-semibold text-white disabled:opacity-50">
          {creating ? 'Inventorying…' : '➕ Create video bank'}
        </button>
      </form>

      {/* Second way in: the scraper's own destination. A video bank no longer
          needs a folder of rushes you assembled by hand — you can fill one
          straight from the web. A new bank gets a folder of its own; an existing
          one receives the clips into the folder it already follows, and the
          picker names that folder when it is one of yours. */}
      <VideoBankScrapePanel banks={banks} onDone={() => refresh()} />

      {banks == null ? (
        <p className="text-sm text-content-muted">Loading…</p>
      ) : banks.length === 0 ? (
        <p className="text-sm text-content-muted">
          No video bank yet — create one above to cut a folder of rushes into shots.
        </p>
      ) : (
        /* grid-cols-1 (= minmax(0,1fr)), NOT the implicit auto column: an auto
           column is sized on max-content, so the unbreakable source PATH inside a
           card stretches it past the viewport and scrolls the whole page sideways
           on a phone — with `truncate` never getting a chance to fire. */
        <ul className="grid gap-3 grid-cols-1 sm:grid-cols-2">
          {banks.map((b) => (
            <li key={b.id}
              className="flex min-w-0 flex-col gap-2 rounded-lg border border-border bg-surface p-4">
              <div className="flex min-w-0 items-center gap-2">
                <button type="button" onClick={() => open(b.id)}
                  className="min-w-0 truncate text-left text-base font-semibold text-content hover:underline">
                  {b.name}
                </button>
                <button type="button" onClick={() => remove(b)}
                  aria-label={`Remove video bank ${b.name}`}
                  className="ml-auto px-1.5 text-content-subtle hover:text-rose-300">✕</button>
              </div>
              <p className="truncate font-mono text-xs text-content-subtle" title={b.source_path}>
                {b.source_path}
              </p>
              <p className="text-xs text-content-muted">{countsSummary(b.counts)}</p>
              <button type="button" onClick={() => open(b.id)}
                className="self-start rounded-md border border-border bg-surface-raised px-3 py-1 text-xs font-semibold text-content hover:bg-surface">
                Open →
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
