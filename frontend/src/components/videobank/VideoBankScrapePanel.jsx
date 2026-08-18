import { useState } from 'react'
import { postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import { HelpBadge } from '../../help/HelpMode'
import ConceptSourcesPanel from '../dataset/ConceptSourcesPanel'
import {
  runVideoBankScrapeImport,
  scrapableVideoBanks,
  summarizeVideoBankScrapeImport,
  videoBankScrapeDestination,
  videoBankScrapeNextStep,
} from './videoBankScrapeImport'

/**
 * 🕸 Scrape the web into a video bank — the scraper's third destination.
 *
 * The scan endpoint has always returned video items; nothing consumed them, so
 * the only way to triage a scraped clip was to download it by hand, drop it in a
 * folder and point a bank at that folder. Same scan UI as the two image
 * destinations, one extra question: which bank receives the clips.
 *
 * WHY "ADD TO AN EXISTING BANK" ONLY LISTS SOME OF THEM. A video bank promises
 * never to write into the folder it points at — that is what makes it safe to
 * run over an archive of originals. So a scrape lands in a folder the app owns,
 * and a bank created from your own rushes is not a destination. The server says
 * which is which (`scrapable` on the bank row); the picker only shows those.
 *
 * Collapsed by default: the page's first job is still to open a bank.
 */
export default function VideoBankScrapePanel({ banks, onDone }) {
  const toast = useToast()
  const [open, setOpen] = useState(false)
  const [mode, setMode] = useState('new')          // 'new' | 'existing'
  const [name, setName] = useState('')
  const [bankId, setBankId] = useState('')
  const [busy, setBusy] = useState(false)
  const eligible = scrapableVideoBanks(banks)

  const handleImport = async (items) => {
    const destination = videoBankScrapeDestination({ mode, name, bankId, banks })
    if (!destination) {
      toast.error(mode === 'existing'
        ? 'Pick which bank receives the videos.'
        : 'Name the bank that will receive the videos.')
      return { ok: false }
    }
    setBusy(true)
    try {
      const res = await runVideoBankScrapeImport({
        items, destination, post: (url, body) => postJson(url, body),
        onBatch: ({ index, count, total }) => {
          if (count > 1) toast.info(`Downloading batch ${index + 1} of ${count} (${total} picked)…`)
        },
      })
      if (!res.ok) {
        toast.error(res.error || 'Could not scrape into the video bank.')
        if (res.saved) toast.warning(`${summarizeVideoBankScrapeImport(res)} before the failure.`)
      } else {
        const next = videoBankScrapeNextStep(res)
        toast.success(`${res.created ? 'Bank created — ' : ''}${summarizeVideoBankScrapeImport(res)}.${next ? ` ${next}` : ''}`)
        // Resume the SAME bank on the next import instead of creating another.
        if (res.bankId) { setMode('existing'); setBankId(String(res.bankId)) }
      }
      await onDone?.()
      return res
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="rounded-lg border border-border bg-surface">
      <button type="button" onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-4 py-3 text-left">
        <span aria-hidden>🕸</span>
        <span className="text-sm font-semibold text-content">Scrape the web into a video bank</span>
        <span className="hidden text-[0.6875rem] text-content-subtle sm:inline">
          no folder to prepare — the clips land in a bank ready to cut
        </span>
        <HelpBadge topic="video-bank-scrape" />
        <span aria-hidden className="ml-auto text-content-subtle">{open ? '▾' : '▸'}</span>
      </button>

      {open && (
        <div className="flex flex-col gap-3 border-t border-border p-3 sm:p-4">
          {/* Destination first: it decides whether this scrape starts a pile or
              grows one. Wraps to one column at 400 px. */}
          <div className="flex flex-col gap-2 rounded-lg border border-border bg-white/[0.03] p-3">
            <span className="text-[0.6875rem] font-semibold uppercase tracking-wide text-content-subtle">
              Destination
            </span>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm text-content">
              <label className="flex items-center gap-1.5">
                <input type="radio" name="video-bank-scrape-dest" value="new"
                  checked={mode === 'new'} onChange={() => setMode('new')}
                  className="accent-indigo-500" />
                New bank
              </label>
              <label className={`flex items-center gap-1.5 ${eligible.length ? '' : 'opacity-50'}`}>
                <input type="radio" name="video-bank-scrape-dest" value="existing"
                  disabled={!eligible.length}
                  checked={mode === 'existing'} onChange={() => setMode('existing')}
                  className="accent-indigo-500" />
                Add to a scraped bank
              </label>
            </div>
            {mode === 'new' ? (
              <label className="flex flex-col gap-1">
                <span className="text-xs text-content-muted">Name</span>
                <input value={name} onChange={(e) => setName(e.target.value)}
                  aria-label="Name of the new video bank"
                  placeholder="Scraped clips 08/2026"
                  className="w-full rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content" />
              </label>
            ) : (
              <label className="flex flex-col gap-1">
                <span className="text-xs text-content-muted">Bank</span>
                <select value={bankId} onChange={(e) => setBankId(e.target.value)}
                  aria-label="Video bank that receives the clips"
                  className="w-full rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content">
                  <option value="">Choose a bank…</option>
                  {eligible.map((b) => (
                    <option key={b.id} value={b.id}>
                      {b.name} ({b.counts?.sources ?? 0})
                    </option>
                  ))}
                </select>
              </label>
            )}
            <p className="text-[0.6875rem] leading-relaxed text-content-subtle">
              Clips are stored exactly as downloaded, then cut into shots by the bank&rsquo;s
              own passes — length, motion and sharpness stay for you to judge there.
              Only banks created by a scrape can receive one: a bank you pointed at your
              own footage is never written into.
            </p>
          </div>

          <ConceptSourcesPanel destination="video-bank" stateKey="video-bank"
            onImport={handleImport} busy={busy} />
        </div>
      )}
    </section>
  )
}
