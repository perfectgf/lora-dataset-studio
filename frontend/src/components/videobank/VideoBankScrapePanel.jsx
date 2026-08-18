import { useState } from 'react'
import { postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import { HelpBadge } from '../../help/HelpMode'
import ConceptSourcesPanel from '../dataset/ConceptSourcesPanel'
import {
  findVideoBank,
  runVideoBankScrapeImport,
  scrapableVideoBanks,
  summarizeVideoBankScrapeImport,
  videoBankScrapeDestination,
  videoBankScrapeFolderNotice,
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
 * "ADD TO AN EXISTING BANK" LISTS THEM ALL, your own folders included. The first
 * cut of this panel only offered banks the app had itself created, on the ground
 * that a video bank never writes into the folder it points at. That answered a
 * question nobody had asked — "may the app write here?" — instead of the one
 * they had: put these clips in THAT bank. Picking the bank is the consent, so
 * there is no toggle beside it; what replaces the refusal is one sentence, at
 * the moment of choosing, naming the folder the clips will be added to.
 *
 * (The only bank still missing from the list is one whose folder belongs to a
 * dataset — `scrapable` on the bank row, decided server-side.)
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
  const folderNotice = mode === 'existing'
    ? videoBankScrapeFolderNotice(findVideoBank(banks, bankId))
    : ''

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
              <label className={`flex items-center gap-1.5 ${eligible.length ? '' : 'opacity-50'}`}
                title={eligible.length || !(banks || []).length ? undefined
                  : 'Your video banks sit on dataset folders, which a scrape can never write into.'}>
                <input type="radio" name="video-bank-scrape-dest" value="existing"
                  disabled={!eligible.length}
                  checked={mode === 'existing'} onChange={() => setMode('existing')}
                  className="accent-indigo-500" />
                Add to an existing bank
              </label>
            </div>
            {/* The ONE refusal that survived 863cbb56 was also the one left
                mute: banks exist, none can receive, and the radio just greyed
                out. Same rule as everywhere else in that commit — what replaces
                a refusal is a sentence at the moment of choosing. */}
            {!eligible.length && (banks || []).length > 0 && (
              <p className="text-[0.6875rem] leading-relaxed text-content-subtle">
                Your existing banks sit on a dataset&rsquo;s own folder, which a scrape
                can never write into — create a new bank instead.
              </p>
            )}
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
            {/* The honesty that replaced a refusal — shown only when the chosen
                bank follows a folder of the user's own, because that is the only
                case where a download lands somewhere they also use themselves.
                `break-all` on the path: a Windows path has no spaces to wrap at
                and would otherwise push the panel sideways at 400 px. */}
            {folderNotice && (
              <p className="rounded-md border border-amber-400/40 bg-amber-500/10 px-2 py-1.5 text-[0.6875rem] leading-relaxed text-amber-100 break-all">
                {folderNotice}
              </p>
            )}
            <p className="text-[0.6875rem] leading-relaxed text-content-subtle">
              Clips are stored exactly as downloaded, then cut into shots by the bank&rsquo;s
              own passes — length, motion and sharpness stay for you to judge there.
              Whichever bank you pick receives them: the clips are added to the folder
              that bank follows, alongside whatever is already in it.
            </p>
          </div>

          <ConceptSourcesPanel destination="video-bank" stateKey="video-bank"
            onImport={handleImport} busy={busy} />
        </div>
      )}
    </section>
  )
}
