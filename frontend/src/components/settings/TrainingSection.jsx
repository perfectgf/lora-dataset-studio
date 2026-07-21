import { useEffect, useState } from 'react'
import { INPUT_CLASS, Card, SecretField } from './primitives'

// Keep in sync with backend TRAIN_TYPES (face_dataset_service.py) — 'flux' had
// been forgotten here when the FLUX.1 family landed (fixed alongside flux2klein).
const FAMILY_OPTIONS = ['zimage', 'sdxl', 'krea', 'flux', 'flux2klein']

/* First-time walkthrough for renting cloud GPUs — collapsed by default so the
   card stays compact for users who already have a key. */
function VastKeyGuide() {
  const link = 'font-medium text-sky-300 underline hover:text-sky-200'
  return (
    <details className="mb-2 rounded-lg border border-border bg-surface px-3 py-2 open:pb-3">
      <summary className="cursor-pointer select-none text-xs font-medium text-content">
        <span aria-hidden>📖</span> How to get a vast.ai API key (≈2 minutes)
      </summary>
      <ol className="mt-2 list-decimal space-y-1.5 pl-5 text-xs text-content-muted">
        <li>
          Create a free account at{' '}
          <a href="https://cloud.vast.ai/" target="_blank" rel="noreferrer" className={link}>cloud.vast.ai</a>
          {' '}(email or Google sign-in).
        </li>
        <li>
          Add credit: open{' '}
          <a href="https://cloud.vast.ai/billing/" target="_blank" rel="noreferrer" className={link}>Billing</a>
          {' '}in the left sidebar and click <strong>Add Credit</strong> — $5 is plenty to
          start (a typical training run costs ~$1–2, billed by vast.ai, not by this app).
        </li>
        <li>
          Open{' '}
          <a href="https://cloud.vast.ai/manage-keys/" target="_blank" rel="noreferrer" className={link}>Keys</a>
          {' '}(left sidebar, under Account) and copy your API key — create one first if
          the list is empty.
        </li>
        <li>
          Paste the key in the field below and press <strong>Test</strong> — it saves the
          key automatically and should answer “connected as &lt;your account&gt;”.
        </li>
      </ol>
    </details>
  )
}

const VAST_SECRET = {
  key: 'VAST_API_KEY', label: 'vast.ai API key', testTarget: 'vast',
  help: 'Enables cloud GPU training: the app rents a GPU for the run and shuts it down when done (typical run: $1-2). Get a key at cloud.vast.ai → Keys.',
  guide: <VastKeyGuide />,
}

function CloudOfferFilter({ id, label, help, checked, onChange }) {
  return (
    <div className="flex items-start justify-between gap-4 rounded-lg border border-border bg-surface-raised px-3 py-2.5">
      <div>
        <p id={`${id}-label`} className="text-sm font-medium text-content">{label}</p>
        <p id={`${id}-help`} className="mt-0.5 text-xs text-content-muted">{help}</p>
      </div>
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={checked}
        aria-labelledby={`${id}-label`}
        aria-describedby={`${id}-help`}
        onClick={() => onChange(!checked)}
        className={`relative h-6 w-11 shrink-0 rounded-full transition-colors ${checked ? 'bg-emerald-500' : 'border border-border-strong bg-surface'}`}
      >
        <span
          aria-hidden
          className={`absolute top-0.5 h-5 w-5 rounded-full bg-white transition-transform ${checked ? 'translate-x-5' : 'translate-x-0.5'}`}
        />
      </button>
    </div>
  )
}

/* Cloud training limits: concurrency cap, offer price ceiling, monthly budget
   and the stall watchdog timeout. Fetches the cloud status ONCE on mount for
   the "Spent this month" info line — no poll, this page is not a dashboard. */
function CloudTrainingCard({ config, setField }) {
  const [spend, setSpend] = useState(null)
  const verifiedOnly = config.cloud?.verified_only ?? true
  const secureCloudOnly = config.cloud?.secure_cloud_only ?? false
  useEffect(() => {
    let alive = true
    // Raw fetch (not apiFetch): this info line is best-effort — a transient
    // 500 must not fire the global error toast over a cosmetic detail.
    fetch('/api/dataset/train/cloud/status', { credentials: 'include' })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (alive && d && typeof d.month_spend === 'number') setSpend(d.month_spend) })
      .catch(() => { /* info line is best-effort */ })
    return () => { alive = false }
  }, [])
  return (
    <Card title="Cloud training" help="vast.ai GPU rental guardrails — how many training pods may run at once, the offer price ceiling, your monthly spend limit, and how long a run may go without step progress before it is rescued and killed.">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label htmlFor="cloud-max-concurrent-runs" className="block text-sm font-medium text-content">
            Max simultaneous cloud runs
          </label>
          <input
            id="cloud-max-concurrent-runs"
            type="number"
            min="1"
            max="10"
            step="1"
            value={config.cloud?.max_concurrent_runs ?? 1}
            onChange={(e) => setField('cloud', 'max_concurrent_runs', parseInt(e.target.value) || 1)}
            className={INPUT_CLASS}
          />
        </div>
        <div>
          <label htmlFor="cloud-max-price-per-hour" className="block text-sm font-medium text-content">
            Max price per hour ($)
          </label>
          <input
            id="cloud-max-price-per-hour"
            type="number"
            min="0.1"
            max="5"
            step="0.05"
            value={config.cloud?.max_price_per_hour ?? 0.8}
            onChange={(e) => setField('cloud', 'max_price_per_hour', Math.max(0.1, parseFloat(e.target.value) || 0.1))}
            className={INPUT_CLASS}
          />
        </div>
        <div>
          <label htmlFor="cloud-monthly-budget" className="block text-sm font-medium text-content">
            Monthly budget ($, 0 = unlimited)
          </label>
          <input
            id="cloud-monthly-budget"
            type="number"
            min="0"
            step="1"
            value={config.cloud?.monthly_budget_usd ?? 0}
            onChange={(e) => setField('cloud', 'monthly_budget_usd', parseFloat(e.target.value) || 0)}
            className={INPUT_CLASS}
          />
        </div>
        <div>
          <label htmlFor="cloud-stall-timeout" className="block text-sm font-medium text-content">
            Stall timeout (minutes)
          </label>
          <input
            id="cloud-stall-timeout"
            type="number"
            min="5"
            max="240"
            step="1"
            value={config.cloud?.stall_timeout_minutes ?? 30}
            onChange={(e) => setField('cloud', 'stall_timeout_minutes', parseInt(e.target.value) || 30)}
            className={INPUT_CLASS}
          />
        </div>
        <div>
          <label htmlFor="cloud-unreachable-grace" className="block text-sm font-medium text-content">
            Unreachable grace (minutes)
          </label>
          <input
            id="cloud-unreachable-grace"
            type="number"
            min="1"
            max="60"
            step="1"
            value={config.cloud?.unreachable_grace_minutes ?? 6}
            onChange={(e) => setField('cloud', 'unreachable_grace_minutes', parseInt(e.target.value) || 6)}
            className={INPUT_CLASS}
          />
          <p className="mt-1 text-[0.6875rem] text-content-subtle">
            How long a mid-run pod may stay unreachable (a vast.ai network blip) before the run is given up and retried on a fresh host. Raise it if healthy runs die with "pod unreachable".
          </p>
        </div>
        <div>
          <label htmlFor="cloud-min-reliability" className="block text-sm font-medium text-content">
            Min host reliability
          </label>
          <input
            id="cloud-min-reliability"
            type="number"
            min="0.9"
            max="0.999"
            step="0.005"
            value={config.cloud?.min_reliability ?? 0.98}
            onChange={(e) => setField('cloud', 'min_reliability', Math.min(0.999, Math.max(0.9, parseFloat(e.target.value) || 0.98)))}
            className={INPUT_CLASS}
          />
          <p className="mt-1 text-[0.6875rem] text-content-subtle">
            Lower it (e.g. 0.95) to surface cheaper hosts in the GPU picker — at a higher risk of a pod that never boots (≈ a few wasted cents, auto-cleaned).
          </p>
        </div>
      </div>
      <div className="space-y-2">
        <p className="text-sm font-medium text-content">GPU offer filters</p>
        <div className="grid gap-2 lg:grid-cols-2">
          <CloudOfferFilter
            id="cloud-verified-only"
            label="Verified hosts only"
            help="Only show hosts verified by vast.ai. Recommended; turning this off can reveal more or cheaper offers, with more host risk."
            checked={verifiedOnly}
            onChange={(value) => setField('cloud', 'verified_only', value)}
          />
          <CloudOfferFilter
            id="cloud-secure-cloud-only"
            label="Secure Cloud only"
            help="Only show offers marked Secure Cloud by vast.ai. This excludes Community Cloud machines, so availability may be lower and prices higher."
            checked={secureCloudOnly}
            onChange={(value) => setField('cloud', 'secure_cloud_only', value)}
          />
        </div>
      </div>
      {spend != null && (
        <p className="text-xs text-content-muted">Spent this month: ${spend.toFixed(2)}</p>
      )}
    </Card>
  )
}

export default function TrainingSection(props) {
  const { config, setField } = props
  return (
    <div className="space-y-6">
      <Card title="Defaults" help="Preselected model family for new training runs — each dataset can still override it.">
        <div>
          <label htmlFor="training-default-family" className="block text-sm font-medium text-content">Default training family</label>
          <select
            id="training-default-family"
            value={config.training.default_family}
            onChange={(e) => setField('training', 'default_family', e.target.value)}
            className={INPUT_CLASS}
          >
            {FAMILY_OPTIONS.map((f) => <option key={f} value={f}>{f}</option>)}
          </select>
        </div>
      </Card>

      <Card title="Cloud GPU (vast.ai)" help="No local GPU? The app can rent one per run — the key below unlocks the ☁️ Train in cloud button.">
        <SecretField field={VAST_SECRET} {...props} />
      </Card>

      <CloudTrainingCard config={config} setField={setField} />
    </div>
  )
}
