import { Link, useLocation } from 'react-router'
import { useUsageStatistics } from './UsageStatisticsContext'

export default function UsageStatisticsBanner() {
  const { state, loading, busy, error, setEnabled } = useUsageStatistics()
  const { pathname } = useLocation()
  if (!state?.configured || state.choice !== null || loading || pathname === '/settings/maintenance') return null
  return (
    <aside aria-label="Optional usage statistics" className="shrink-0 border-b border-border bg-surface px-4 py-3">
      <div className="mx-auto flex max-w-5xl flex-wrap items-center gap-x-6 gap-y-3">
        <div className="min-w-0 flex-1 basis-72">
          <p className="text-sm font-medium text-content">Help improve LDS with optional usage statistics</p>
          <p className="mt-1 text-xs leading-relaxed text-content-muted">
            Share feature outcomes, failure stages, model families, captioning engines, local/cloud categories and activity with the LDS maintainer via PostHog Cloud EU.
            No images, prompts, captions or file names. Off until you choose; change your mind in Settings.{' '}
            <Link to="/guide/settings-reference?h=usage-statistics" className="text-primary underline">What is shared</Link>
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2" aria-busy={busy}>
          <button type="button" disabled={busy} onClick={() => setEnabled(true)}
            className="min-h-10 rounded-md border border-border-strong px-3 py-2 text-sm font-medium text-content hover:bg-surface-raised disabled:opacity-50">
            Share usage statistics
          </button>
          <button type="button" disabled={busy} onClick={() => setEnabled(false)}
            className="min-h-10 rounded-md border border-border-strong px-3 py-2 text-sm font-medium text-content hover:bg-surface-raised disabled:opacity-50">
            No thanks
          </button>
          {busy && <span role="status" className="text-xs text-content-muted">Saving…</span>}
        </div>
        {error && <p role="alert" className="w-full text-sm text-rose-400">{error}</p>}
      </div>
    </aside>
  )
}
