import { useState } from 'react'
import { Globe } from 'lucide-react';
import { postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import { HelpBadge } from '../../help/HelpMode'
import ConceptSourcesPanel from '../dataset/ConceptSourcesPanel'
import {
  bankScrapeDestination,
  runBankScrapeImport,
  summarizeBankScrapeImport,
} from './bankScrapeImport'

/**
 * 🕸 Scrape into a bank — the scraper's second destination.
 *
 * The scraper used to have exactly one outlet: straight into a dataset, through
 * filters that exist for TRAINING (short side ≥ 768 px, ratio ≤ 3:1, perceptual
 * de-duplication). Getting a scrape into a bank meant scraping into a throwaway
 * dataset first and importing it back — by which point those filters had already
 * thrown away the very images the triage tools exist to judge.
 *
 * Same scan UI, same picking, one extra question: which bank receives them. A new
 * bank gets a folder of its own under the app's bank sources; an existing one is
 * simply added to (a bank follows a LIVE folder, so a second scrape resumes the
 * pile instead of replacing it).
 *
 * Collapsed by default: the page's first job is still to open a bank.
 */
export default function BankScrapePanel({ banks, onDone }) {
  const toast = useToast()
  const [open, setOpen] = useState(false)
  const [mode, setMode] = useState('new')          // 'new' | 'existing'
  const [name, setName] = useState('')
  const [bankId, setBankId] = useState('')
  const [busy, setBusy] = useState(false)
  const known = banks || []

  const handleImport = async (items) => {
    const destination = bankScrapeDestination({ mode, name, bankId })
    if (!destination) {
      toast.error(mode === 'existing'
        ? 'Pick which bank receives the images.'
        : 'Name the bank that will receive the images.')
      return { ok: false }
    }
    setBusy(true)
    try {
      const res = await runBankScrapeImport({
        items, destination, post: (url, body) => postJson(url, body),
        onBatch: ({ index, count, total }) => {
          if (count > 1) toast.info(`Downloading batch ${index + 1} of ${count} (${total} picked)…`)
        },
      })
      if (!res.ok) {
        toast.error(res.error || 'Could not scrape into the bank.')
        if (res.saved) toast.warning(`${summarizeBankScrapeImport(res)} before the failure.`)
      } else {
        toast.success(`${res.created ? 'Bank created — ' : ''}${summarizeBankScrapeImport(res)}.`)
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
        <Globe aria-hidden="true" className="h-4 w-4" />
        <span className="text-sm font-semibold text-content">Scrape the web into a bank</span>
        <span className="hidden text-[0.6875rem] text-content-subtle sm:inline">
          no folder to prepare — the images land in a bank ready to triage
        </span>
        <HelpBadge topic="bank-scrape" />
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
                <input type="radio" name="bank-scrape-dest" value="new"
                  checked={mode === 'new'} onChange={() => setMode('new')}
                  className="accent-indigo-500" />
                New bank
              </label>
              <label className={`flex items-center gap-1.5 ${known.length ? '' : 'opacity-50'}`}>
                <input type="radio" name="bank-scrape-dest" value="existing"
                  disabled={!known.length}
                  checked={mode === 'existing'} onChange={() => setMode('existing')}
                  className="accent-indigo-500" />
                Add to an existing bank
              </label>
            </div>
            {mode === 'new' ? (
              <label className="flex flex-col gap-1">
                <span className="text-xs text-content-muted">Name</span>
                <input value={name} onChange={(e) => setName(e.target.value)}
                  aria-label="Name of the new bank"
                  placeholder="Scraped portraits 07/2026"
                  className="w-full rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content" />
              </label>
            ) : (
              <label className="flex flex-col gap-1">
                <span className="text-xs text-content-muted">Bank</span>
                <select value={bankId} onChange={(e) => setBankId(e.target.value)}
                  aria-label="Bank that receives the images"
                  className="w-full rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content">
                  <option value="">Choose a bank…</option>
                  {known.map((b) => (
                    <option key={b.id} value={b.id}>{b.name} ({b.total})</option>
                  ))}
                </select>
              </label>
            )}
            <p className="text-[0.6875rem] leading-relaxed text-content-subtle">
              Images are stored exactly as downloaded. Small shots, near-duplicates and
              framing stay for the bank&rsquo;s own passes to judge — that is the point of
              triaging here rather than importing straight into a dataset.
            </p>
          </div>

          <ConceptSourcesPanel destination="bank" stateKey="bank"
            onImport={handleImport} busy={busy} />
        </div>
      )}
    </section>
  )
}
