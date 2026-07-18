import { useCallback, useEffect, useState } from 'react'
import { apiFetch, del, postJson } from '../api/fetchClient'
import { useToast } from '../components/common/Toast'
import { HelpBadge } from '../help/HelpMode'
import BankWorkspace from '../components/bank/BankWorkspace'

const CURRENT_KEY = 'bankCurrentId'

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

  const refresh = useCallback(async () => {
    try {
      const d = await apiFetch('/api/banks')
      setBanks(d.banks || [])
    } catch (e) {
      toast.error(e?.message || 'Could not load the banks.')
      setBanks([])
    }
  }, [toast])

  useEffect(() => { if (currentId == null) refresh() }, [currentId, refresh])

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
    if (creating) return
    setCreating(true)
    try {
      const d = await postJson('/api/bank/create', { name, folder })
      toast.success(`Bank created — ${d.added} image(s) inventoried.`)
      setName(''); setFolder('')
      open(d.id)
    } catch (err) {
      toast.error(err?.message || 'Could not create the bank.')
    } finally {
      setCreating(false)
    }
  }

  const remove = async (bank) => {
    // eslint-disable-next-line no-alert
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
      <header className="flex items-center gap-2">
        <h1 className="text-xl font-bold text-content">🗃️ Image bank</h1>
        <HelpBadge topic="page-bank" />
      </header>
      <p className="text-sm text-content-muted max-w-3xl">
        Point the app at a big unsorted folder (a Telegram export, a scrape dump…) and triage it
        into dataset-ready selections: a quality pass flags blur/noise/flat/small shots and groups
        near-duplicates, the face pass sorts the dump by person — then you promote the keepers
        into a dataset. The folder itself is never modified.
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
          <label htmlFor="bank-folder" className="block text-sm font-medium text-content">
            Folder on this computer
          </label>
          <input id="bank-folder" value={folder} onChange={(e) => setFolder(e.target.value)}
            placeholder="C:\path\to\unsorted-images (subfolders included)" required
            className="mt-1 w-full rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content font-mono" />
        </div>
        <button type="submit" disabled={creating}
          className="rounded-md bg-gradient-primary px-4 py-2 text-sm font-semibold text-white disabled:opacity-50">
          {creating ? 'Inventorying…' : '➕ Create bank'}
        </button>
      </form>

      {banks == null ? (
        <p className="text-sm text-content-muted">Loading…</p>
      ) : banks.length === 0 ? (
        <p className="text-sm text-content-muted">
          No bank yet — create one above to start triaging a folder.
        </p>
      ) : (
        <ul className="grid gap-3 sm:grid-cols-2">
          {banks.map((b) => (
            <li key={b.id}
              className="flex flex-col gap-2 rounded-lg border border-border bg-surface p-4">
              <div className="flex items-center gap-2">
                <button type="button" onClick={() => open(b.id)}
                  className="text-left text-base font-semibold text-content hover:underline">
                  {b.name}
                </button>
                {b.activity && !b.activity.finished && (
                  <span className="text-xs text-amber-300">⏳ {b.activity.kind}…</span>
                )}
                <button type="button" onClick={() => remove(b)} aria-label={`Remove bank ${b.name}`}
                  className="ml-auto px-1.5 text-content-subtle hover:text-rose-300">✕</button>
              </div>
              <p className="truncate font-mono text-xs text-content-subtle" title={b.source_path}>
                {b.source_path}
              </p>
              <p className="text-xs text-content-muted">
                {b.total} image(s) · {b.scanned} scanned · <span className="text-emerald-300">{b.keep} kept</span> · <span className="text-rose-300">{b.reject} rejected</span>
              </p>
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
