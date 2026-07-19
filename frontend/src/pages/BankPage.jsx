import { useCallback, useEffect, useState } from 'react'
import { apiFetch, del, postJson } from '../api/fetchClient'
import { useToast } from '../components/common/Toast'
import { HelpBadge } from '../help/HelpMode'
import { useI18n } from '../i18n/I18nContext'
import BankWorkspace from '../components/bank/BankWorkspace'
import FolderPickerField from '../components/common/FolderPicker'

const CURRENT_KEY = 'bankCurrentId'

/** 🗃️ Image bank — triage a big unsorted folder BEFORE it becomes datasets.
 * List view (create/open/delete banks) + per-bank workspace. The bank
 * references the folder in place: nothing is copied until promotion, and the
 * source files are never modified. */
export default function BankPage() {
  const toast = useToast()
  const { t } = useI18n()
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
      toast.error(e?.message || t('bank.page.loadFailed'))
      setBanks([])
    }
  }, [toast, t])

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
      toast.success(t('bank.page.created', { count: d.added }))
      setName(''); setFolder('')
      open(d.id)
    } catch (err) {
      toast.error(err?.message || t('bank.page.createFailed'))
    } finally {
      setCreating(false)
    }
  }

  const remove = async (bank) => {
    // eslint-disable-next-line no-alert
    if (!window.confirm(t('bank.removeConfirm', { name: bank.name }))) return
    try {
      await del(`/api/bank/${bank.id}`)
      toast.success(t('bank.page.removed'))
      refresh()
    } catch (e) {
      toast.error(e?.message || t('bank.page.removeFailed'))
    }
  }

  if (currentId != null) {
    return <BankWorkspace bankId={currentId} onBack={close} onGone={close} />
  }

  return (
    <div className="space-y-6">
      <header className="flex items-center gap-2">
        <h1 className="text-xl font-bold text-content">🗃️ {t('bank.page.title')}</h1>
        <span className="px-1.5 py-0.5 rounded border border-amber-400/50 bg-amber-500/10 text-amber-300 text-[0.625rem] font-semibold uppercase tracking-wide">{t('common.beta')}</span>
        <HelpBadge topic="page-bank" />
      </header>
      <p className="text-sm text-content-muted max-w-3xl">
        {t('bank.page.description')}
      </p>

      <form onSubmit={create}
        className="flex flex-wrap items-end gap-3 rounded-lg border border-border bg-surface p-4">
        <div className="grow min-w-40">
          <label htmlFor="bank-name" className="block text-sm font-medium text-content">{t('bank.page.name')}</label>
          <input id="bank-name" value={name} onChange={(e) => setName(e.target.value)}
            placeholder={t('bank.page.namePlaceholder')} required
            className="mt-1 w-full rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content" />
        </div>
        <div className="grow-[3] min-w-64">
          <FolderPickerField id="bank-folder" label={t('bank.page.folder')}
            value={folder} onChange={setFolder} required
            placeholder={t('bank.page.folderPlaceholder')} />
        </div>
        <button type="submit" disabled={creating}
          className="rounded-md bg-gradient-primary px-4 py-2 text-sm font-semibold text-white disabled:opacity-50">
          {creating ? t('bank.page.inventorying') : `➕ ${t('bank.page.create')}`}
        </button>
      </form>

      {banks == null ? (
        <p className="text-sm text-content-muted">{t('common.loading')}</p>
      ) : banks.length === 0 ? (
        <p className="text-sm text-content-muted">
          {t('bank.page.empty')}
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
                  <span className="text-xs text-amber-300">
                    ⏳ {t(`bank.activity.${['scan', 'faces', 'score', 'watermark', 'promote'].includes(b.activity.kind)
                      ? b.activity.kind : 'job'}`)}…
                  </span>
                )}
                <button type="button" onClick={() => remove(b)}
                  aria-label={t('bank.page.removeLabel', { name: b.name })}
                  className="ml-auto px-1.5 text-content-subtle hover:text-rose-300">✕</button>
              </div>
              <p className="truncate font-mono text-xs text-content-subtle" title={b.source_path}>
                {b.source_path}
              </p>
              <p className="text-xs text-content-muted">
                {t('bank.counts.images', { count: b.total })} · {t('bank.counts.scanned', { count: b.scanned })} ·{' '}
                <span className="text-emerald-300">{t('bank.counts.kept', { count: b.keep })}</span> ·{' '}
                <span className="text-rose-300">{t('bank.counts.rejected', { count: b.reject })}</span>
              </p>
              <button type="button" onClick={() => open(b.id)}
                className="self-start rounded-md border border-border bg-surface-raised px-3 py-1 text-xs font-semibold text-content hover:bg-surface">
                {t('bank.page.open')} →
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
