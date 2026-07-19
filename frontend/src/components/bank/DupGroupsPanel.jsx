import { useCallback, useEffect, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import { useI18n } from '../../i18n/I18nContext'

const GROUPS_PAGE = 25

/** Near-duplicate resolution: one card per unresolved group. "Keep best"
 * keeps the highest-resolution/sharpest member, "Keep first" the oldest by
 * import order; clicking a member keeps THAT one. Losers are rejected (a
 * reversible status) — nothing is ever deleted from disk. */
export default function DupGroupsPanel({ bankId, live, onChanged }) {
  const toast = useToast()
  const { t } = useI18n()
  const [data, setData] = useState(null)
  const [offset, setOffset] = useState(0)
  const [busy, setBusy] = useState(false)

  const refresh = useCallback(async (off = offset) => {
    try {
      const d = await apiFetch(`/api/bank/${bankId}/dup-groups?offset=${off}&limit=${GROUPS_PAGE}`)
      setData(d); setOffset(off)
    } catch (e) {
      toast.error(e?.message || t('bank.duplicates.loadFailed'))
    }
  }, [bankId, offset, toast, t])

  useEffect(() => { refresh(0) // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bankId])

  const resolve = async (body, okMsg) => {
    if (busy) return
    setBusy(true)
    try {
      const d = await postJson(`/api/bank/${bankId}/dups/resolve`, body)
      toast.success(okMsg || t('bank.duplicates.resolved', {
        groups: d.resolved, rejected: d.rejected,
      }))
      await refresh(0)
      onChanged?.()
    } catch (e) {
      toast.error(e?.message || t('bank.duplicates.resolveFailed'))
    } finally {
      setBusy(false)
    }
  }

  if (data == null) return <p className="text-sm text-content-muted">{t('bank.duplicates.loading')}</p>
  if (data.total === 0) {
    return (
      <p className="text-sm text-content-muted">
        {t('bank.duplicates.empty')}
      </p>
    )
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2">
        <span className="text-sm font-semibold text-content">
          ≈ {t('bank.duplicates.unresolved', { count: data.total })}
        </span>
        <span className="text-xs text-content-subtle">
          {t('bank.duplicates.reversible')}
        </span>
        <span className="ml-auto" />
        <button type="button" disabled={busy || live}
          onClick={() => resolve({ strategy: 'best' })}
          title={t('bank.duplicates.resolveAllBestTitle')}
          className="rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-white disabled:opacity-50">
          {t('bank.duplicates.resolveAllBest')}
        </button>
        <button type="button" disabled={busy || live}
          onClick={() => resolve({ strategy: 'first' })}
          title={t('bank.duplicates.resolveAllFirstTitle')}
          className="rounded-md border border-border bg-surface-raised px-3 py-1 text-xs font-semibold text-content disabled:opacity-50 hover:bg-surface">
          {t('bank.duplicates.resolveAllFirst')}
        </button>
      </div>

      <ul className="space-y-3">
        {data.groups.map((g) => (
          <li key={g.group} className="rounded-lg border border-border bg-surface p-3">
            <div className="mb-2 flex items-center gap-2 text-xs text-content-muted">
              <span className="font-semibold text-content">{t('bank.duplicates.group', { id: g.group })}</span>
              <span>{t('bank.duplicates.clickToKeep', { count: g.images.length })}</span>
              <span className="ml-auto" />
              <button type="button" disabled={busy || live}
                onClick={() => resolve({ strategy: 'best', group: g.group })}
                className="rounded-md border border-border px-2 py-0.5 text-content hover:bg-surface-raised disabled:opacity-50">
                {t('bank.duplicates.keepBest')}
              </button>
              <button type="button" disabled={busy || live}
                onClick={() => resolve({ strategy: 'first', group: g.group })}
                className="rounded-md border border-border px-2 py-0.5 text-content hover:bg-surface-raised disabled:opacity-50">
                {t('bank.duplicates.keepFirst')}
              </button>
            </div>
            <ul className="flex flex-wrap gap-2">
              {g.images.map((img) => (
                <li key={img.id} className="w-32">
                  <button type="button" disabled={busy || live}
                    onClick={() => resolve({ keep_ids: [img.id] },
                      t('bank.duplicates.keptOne', {
                        name: img.name, count: g.images.length - 1,
                      }))}
                    title={t('bank.duplicates.keepThisTitle', {
                      size: `${img.width || '?'}×${img.height || '?'}`,
                      sharpness: img.blur_score != null ? Math.round(img.blur_score) : '?',
                    })}
                    className={`relative block w-full overflow-hidden rounded-lg border ${img.id === g.best_id
                      ? 'border-emerald-400 ring-1 ring-emerald-400' : 'border-border'} ${img.status === 'reject' ? 'opacity-50' : ''}`}>
                    <img src={`/api/bank/${bankId}/thumb/${img.id}`} alt={img.name}
                      loading="lazy" className="h-24 w-full object-cover" />
                    {img.id === g.best_id && (
                      <span className="absolute left-1 top-1 rounded bg-emerald-500/90 px-1 text-[10px] font-bold text-white">{t('bank.duplicates.best')}</span>
                    )}
                  </button>
                  <p className="mt-0.5 truncate text-[10px] text-content-subtle" title={img.name}>
                    {img.width || '?'}×{img.height || '?'} · {img.name}
                  </p>
                </li>
              ))}
            </ul>
          </li>
        ))}
      </ul>

      {data.total > GROUPS_PAGE && (
        <nav className="flex items-center gap-3 text-sm" aria-label={t('bank.duplicates.pages')}>
          <button type="button" disabled={offset === 0}
            onClick={() => refresh(Math.max(0, offset - GROUPS_PAGE))}
            className="rounded-md border border-border px-2 py-1 text-content disabled:opacity-40">← {t('bank.pagination.prev')}</button>
          <span className="text-content-muted">
            {t('bank.duplicates.range', {
              from: offset + 1, to: Math.min(offset + GROUPS_PAGE, data.total), total: data.total,
            })}
          </span>
          <button type="button" disabled={offset + GROUPS_PAGE >= data.total}
            onClick={() => refresh(offset + GROUPS_PAGE)}
            className="rounded-md border border-border px-2 py-1 text-content disabled:opacity-40">{t('bank.pagination.next')} →</button>
        </nav>
      )}
    </div>
  )
}
