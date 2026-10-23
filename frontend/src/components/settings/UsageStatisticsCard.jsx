import { Link } from 'react-router'
import { HelpBadge } from '../../help/HelpMode'
import { useUsageStatistics } from '../../usage/UsageStatisticsContext'
import { Card } from './primitives'

export default function UsageStatisticsCard() {
  const { state, loading, busy, error, refresh, setEnabled } = useUsageStatistics()
  const enabled = state?.enabled === true
  return (
    <Card id="usage-statistics" title={<span className="inline-flex items-center gap-2">
      Optional usage statistics <HelpBadge topic="settings-usage-statistics" />
    </span>} help="Help the LDS maintainer prioritize improvements and find unreliable features. Sharing is off by default and never required to use LDS.">
      <div className="space-y-3 text-sm text-content-muted">
        <p>
          If you agree, LDS shares a random installation ID, activity dates and event times, coarse feature names,
          operation outcomes, error categories and failure stages, model families, captioning engines,
          local/cloud operation categories, LDS version, operating system and approximate durations
          with the maintainer via <strong className="font-medium text-content">PostHog Cloud EU</strong>.
        </p>
        <p>
          Images, prompts, captions, names or paths of files, tokens, logs and screen recordings are never included.
          The ID is pseudonymous: these statistics count participating installations, not identifiable people.
        </p>
        <p>These expanded categories require a new choice. An earlier refusal stays saved.</p>
        <p>
          PostHog's free plan lists one year of event retention. Its retention rollout can leave data stored longer;
          LDS does not guarantee an automatic deletion date.
        </p>
        <p>
          Your choice applies to everyone using this LDS installation. Turning sharing off saves immediately and
          clears pending statistics; it does not recall data already sent.
        </p>
        <Link to="/guide/settings-reference?h=usage-statistics" className="inline-block text-primary underline">
          Read what is collected and how it works
        </Link>
      </div>
      <div className="space-y-3" aria-busy={busy || loading}>
        <p role="status" className="text-sm font-medium text-content">
          {loading ? 'Checking your choice…'
            : !state ? 'Sharing status unavailable'
              : !state.configured ? 'Usage sharing is unavailable on this installation. No statistics are sent.'
                : enabled ? 'Usage sharing is on'
                  : state.choice === 'disabled' ? 'Usage sharing is off — your choice is saved'
                    : 'Usage sharing is off — you have not chosen yet'}
        </p>
        {state?.configured && (
          <div className="flex flex-wrap items-center gap-2">
            <button type="button" disabled={busy || loading} onClick={() => setEnabled(!enabled)}
              className="min-h-10 rounded-md border border-border-strong px-3 py-2 text-sm font-medium text-content hover:bg-surface-raised disabled:opacity-50">
              {busy ? 'Saving…' : enabled ? 'Turn off sharing' : 'Share usage statistics'}
            </button>
            {state.choice === null && <button type="button" disabled={busy || loading} onClick={() => setEnabled(false)}
              className="min-h-10 rounded-md border border-border-strong px-3 py-2 text-sm font-medium text-content hover:bg-surface-raised disabled:opacity-50">
              No thanks
            </button>}
          </div>
        )}
        {error && <div className="space-y-2">
          <p role="alert" className="text-sm text-rose-400">{error}</p>
          <button type="button" onClick={refresh} disabled={busy || loading}
            className="min-h-10 rounded-md border border-border-strong px-3 py-2 text-sm text-content hover:bg-surface-raised disabled:opacity-50">
            Check again
          </button>
        </div>}
      </div>
    </Card>
  )
}
