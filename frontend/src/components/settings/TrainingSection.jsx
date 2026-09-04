import { useEffect, useState } from 'react'
import { INPUT_CLASS, Card, SecretField } from './primitives'
import { SettingsGroup, SettingsGroupsToc, useSettingsGroupProps } from './SettingsGroupsView'
import { TRAINING_GROUPS } from './settingsGroups'
import ResetToDefault from './ResetToDefault'
import { defaultValueAt } from './settingDefaults.js'
import { VAST_CONSOLE_URL, VAST_REFERRAL_ID, vastSignupUrl } from '../../utils/vastReferral.js'

// Keep in sync with backend TRAIN_TYPES (face_dataset_service.py) — 'flux' had
// been forgotten here when the FLUX.1 family landed (fixed alongside flux2klein).
const FAMILY_OPTIONS = ['zimage', 'sdxl', 'krea', 'flux', 'flux2klein', 'anima']

/* First-time walkthrough for renting cloud GPUs — collapsed by default so the
   card stays compact for users who already have a key. Step 1 is the ONE link
   in the app that may carry the project's vast.ai referral id (the account is
   created there; see utils/vastReferral.js). When it does, the disclosure sits
   right under the steps with the untagged link beside it — Billing and Keys
   stay bare. Exported for tests/vast-key-guide-render.test.mjs. */
export function VastKeyGuide({ referralId = VAST_REFERRAL_ID } = {}) {
  const link = 'font-medium text-sky-300 underline hover:text-sky-200'
  const signup = vastSignupUrl(referralId)
  const referral = signup !== VAST_CONSOLE_URL
  return (
    <details className="mb-2 rounded-lg border border-border bg-surface px-3 py-2 open:pb-3">
      <summary className="cursor-pointer select-none text-xs font-medium text-content">
        <span aria-hidden>📖</span> How to get a vast.ai API key (≈2 minutes)
      </summary>
      <ol className="mt-2 list-decimal space-y-1.5 pl-5 text-xs text-content-muted">
        <li>
          Create a free account at{' '}
          <a href={signup} target="_blank" rel="noreferrer" className={link}>cloud.vast.ai</a>
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
      {referral && (
        <p className="mt-2 text-xs text-content-muted">
          Step 1’s link is a referral link: open a vast.ai account through it and vast.ai
          pays this project 3% of what you spend there. It costs you nothing — the prices
          are identical either way, and nothing in the app behaves differently. Prefer not
          to?{' '}
          <a href={VAST_CONSOLE_URL} target="_blank" rel="noreferrer" className={link}>cloud.vast.ai</a>
          {' '}works exactly the same.
        </p>
      )}
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
          className={`absolute top-0.5 left-0.5 h-5 w-5 rounded-full bg-white transition-transform ${checked ? 'translate-x-5' : 'translate-x-0'}`}
        />
      </button>
    </div>
  )
}

/* Cloud training limits: concurrency cap, offer price ceiling, monthly budget
   and the stall watchdog timeout. Fetches the cloud status ONCE on mount for
   the "Spent this month" info line — no poll, this page is not a dashboard. */
function CloudTrainingCard({ config, setField, configDefaults }) {
  // Every shipped value below is read from the server payload (config_defaults),
  // never retyped: these guardrails move between releases.
  const dflt = (key) => defaultValueAt(configDefaults, 'cloud', key)
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
    <Card title="Cloud training" help="vast.ai GPU rental guardrails — how many training pods may run at once, the offer price ceiling, your monthly spend limit, and the watchdogs that end a run that has stopped making progress. A pod that is still downloading its base model IS making progress: those budgets are judged on its byte counter, not on the training step it has not reached yet.">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
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
            value={config.cloud?.max_concurrent_runs ?? dflt('max_concurrent_runs')}
            onChange={(e) => setField('cloud', 'max_concurrent_runs', parseInt(e.target.value) || dflt('max_concurrent_runs'))}
            className={INPUT_CLASS}
          />
          <ResetToDefault label="Max simultaneous cloud runs" section="cloud" field="max_concurrent_runs"
            config={config} configDefaults={configDefaults} setField={setField} />
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
            value={config.cloud?.max_price_per_hour ?? dflt('max_price_per_hour')}
            onChange={(e) => setField('cloud', 'max_price_per_hour', Math.max(0.1, parseFloat(e.target.value) || 0.1))}
            className={INPUT_CLASS}
          />
          <ResetToDefault label="Max price per hour" section="cloud" field="max_price_per_hour"
            config={config} configDefaults={configDefaults} setField={setField} />
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
            value={config.cloud?.monthly_budget_usd ?? dflt('monthly_budget_usd')}
            onChange={(e) => setField('cloud', 'monthly_budget_usd', parseFloat(e.target.value) || 0)}
            className={INPUT_CLASS}
          />
          <ResetToDefault label="Monthly budget" section="cloud" field="monthly_budget_usd"
            config={config} configDefaults={configDefaults} setField={setField} />
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
            value={config.cloud?.stall_timeout_minutes ?? dflt('stall_timeout_minutes')}
            onChange={(e) => setField('cloud', 'stall_timeout_minutes', parseInt(e.target.value) || dflt('stall_timeout_minutes'))}
            className={INPUT_CLASS}
          />
          <ResetToDefault label="Stall timeout" section="cloud" field="stall_timeout_minutes"
            config={config} configDefaults={configDefaults} setField={setField} />
        </div>
        <div>
          <label htmlFor="cloud-first-step-timeout" className="block text-sm font-medium text-content">
            First-step timeout (minutes)
          </label>
          <input
            id="cloud-first-step-timeout"
            type="number"
            min="5"
            max="240"
            step="1"
            value={config.cloud?.first_step_timeout_minutes ?? dflt('first_step_timeout_minutes')}
            onChange={(e) => setField('cloud', 'first_step_timeout_minutes', parseInt(e.target.value) || dflt('first_step_timeout_minutes'))}
            className={INPUT_CLASS}
          />
          <p className="mt-1 text-[0.6875rem] text-content-subtle">
            Before training starts, the pod downloads its base model. This is the idle budget for that phase — the clock restarts every time the pod reports more downloaded bytes, so a slow-but-working download is never cut. Only a pod that reports nothing at all for this long is terminated.
          </p>
          <ResetToDefault label="First-step timeout" section="cloud" field="first_step_timeout_minutes"
            config={config} configDefaults={configDefaults} setField={setField} />
        </div>
        <div>
          <label htmlFor="cloud-first-step-download-budget" className="block text-sm font-medium text-content">
            Base-model download ceiling (minutes, 0 = none)
          </label>
          <input
            id="cloud-first-step-download-budget"
            type="number"
            min="0"
            max="480"
            step="1"
            value={config.cloud?.first_step_download_budget_minutes ?? dflt('first_step_download_budget_minutes')}
            onChange={(e) => setField('cloud', 'first_step_download_budget_minutes', Math.max(0, parseInt(e.target.value, 10) || 0))}
            className={INPUT_CLASS}
          />
          <p className="mt-1 text-[0.6875rem] text-content-subtle">
            The hard ceiling on that same phase. A host too slow to ever finish would otherwise keep its download alive — and your rental with it — until the runtime cap. Past this, the pod is terminated even though it is still downloading. Set 0 to rely on the runtime cap alone.
          </p>
          <ResetToDefault label="Base-model download ceiling" section="cloud" field="first_step_download_budget_minutes"
            config={config} configDefaults={configDefaults} setField={setField} />
        </div>
        <div>
          <label htmlFor="cloud-max-runtime" className="block text-sm font-medium text-content">
            Max runtime (minutes)
          </label>
          <input
            id="cloud-max-runtime"
            type="number"
            min="30"
            max="1440"
            step="10"
            value={config.cloud?.max_runtime_minutes ?? dflt('max_runtime_minutes')}
            onChange={(e) => setField('cloud', 'max_runtime_minutes', parseInt(e.target.value) || dflt('max_runtime_minutes'))}
            className={INPUT_CLASS}
          />
          <p className="mt-1 text-[0.6875rem] text-content-subtle">
            The last backstop on the bill: past this, the pod is terminated whatever it is doing, and the newest checkpoint is rescued first. Enforced from outside the run too, so it holds even if the run's own supervision dies.
          </p>
          <ResetToDefault label="Max runtime" section="cloud" field="max_runtime_minutes"
            config={config} configDefaults={configDefaults} setField={setField} />
        </div>
        <div>
          <label htmlFor="cloud-freeze-watchdog" className="block text-sm font-medium text-content">
            Freeze watchdog (minutes, 0 = warn only)
          </label>
          <input
            id="cloud-freeze-watchdog"
            type="number"
            min="0"
            max="480"
            step="1"
            value={config.cloud?.freeze_watchdog_minutes ?? dflt('freeze_watchdog_minutes')}
            onChange={(e) => setField('cloud', 'freeze_watchdog_minutes', Math.max(0, parseInt(e.target.value, 10) || 0))}
            className={INPUT_CLASS}
          />
          <p className="mt-1 text-[0.6875rem] text-content-subtle">
            Last-resort net when a training run stops reporting altogether (a restart, a connection wedged against the pod): the pod is terminated from outside the run, so it can't keep billing unnoticed. Checkpoints already downloaded are kept. Set 0 to only get the warning on the run card. Booting and downloading are never cut by this; the dataset upload has its own setting below.
          </p>
          <ResetToDefault label="Freeze watchdog" section="cloud" field="freeze_watchdog_minutes"
            config={config} configDefaults={configDefaults} setField={setField} />
        </div>
        <div>
          <label htmlFor="cloud-upload-stall" className="block text-sm font-medium text-content">
            Dataset upload stall (minutes, 0 = never cut)
          </label>
          <input
            id="cloud-upload-stall"
            type="number"
            min="0"
            max="480"
            step="1"
            value={config.cloud?.upload_stall_minutes ?? dflt('upload_stall_minutes')}
            onChange={(e) => setField('cloud', 'upload_stall_minutes', Math.max(0, parseInt(e.target.value, 10) || 0))}
            className={INPUT_CLASS}
          />
          <p className="mt-1 text-[0.6875rem] text-content-subtle">
            This is <strong>not</strong> a time limit on the upload — a large dataset is allowed to take as long as it needs, and the run card shows the files and gigabytes going across. It is how long the machine may sit with <strong>no data at all</strong> arriving before the run is given up and the pod released, so a wedged transfer stops billing in minutes instead of hours. Set 0 to never cut. Turning the freeze watchdog off turns this off too.
          </p>
          <ResetToDefault label="Dataset upload stall" section="cloud" field="upload_stall_minutes"
            config={config} configDefaults={configDefaults} setField={setField} />
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
            value={config.cloud?.unreachable_grace_minutes ?? dflt('unreachable_grace_minutes')}
            onChange={(e) => setField('cloud', 'unreachable_grace_minutes', parseInt(e.target.value) || dflt('unreachable_grace_minutes'))}
            className={INPUT_CLASS}
          />
          <p className="mt-1 text-[0.6875rem] text-content-subtle">
            How long a mid-run pod may stay unreachable (a vast.ai network blip) before the run is given up and retried on a fresh host. Raise it if healthy runs die with "pod unreachable".
          </p>
          <ResetToDefault label="Unreachable grace" section="cloud" field="unreachable_grace_minutes"
            config={config} configDefaults={configDefaults} setField={setField} />
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
            value={config.cloud?.min_reliability ?? dflt('min_reliability')}
            onChange={(e) => setField('cloud', 'min_reliability', Math.min(0.999, Math.max(0.9, parseFloat(e.target.value) || dflt('min_reliability'))))}
            className={INPUT_CLASS}
          />
          <p className="mt-1 text-[0.6875rem] text-content-subtle">
            Lower it (e.g. 0.95) to surface cheaper hosts in the GPU picker — at a higher risk of a pod that never boots (≈ a few wasted cents, auto-cleaned).
          </p>
          <ResetToDefault label="Min host reliability" section="cloud" field="min_reliability"
            config={config} configDefaults={configDefaults} setField={setField} />
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


/* Concept face masking (issue #15, reported by shivdbz2010 on GitHub). Both knobs
   are exposed because nobody has measured the right value — no public A/B of a
   concept LoRA trained with vs without face masking exists — so a frozen number
   would be a guess dressed as a default. Every shipped value is read from the
   server payload; a literal here would drift the day the default moves. */
function ConceptFaceMaskCard({ config, setField, configDefaults }) {
  const dflt = (key) => defaultValueAt(configDefaults, 'face_mask', key)
  return (
    <Card title="Concept face masking"
      help="Used only by Concept datasets that turned it on in Advanced training options. It weighs the faces down in the training loss so the concept learns the act, not the identities in your photos. It does not alter your images.">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div>
          <label htmlFor="face-mask-expand" className="block text-sm font-medium text-content">
            Head coverage (face box ×)
          </label>
          <input
            id="face-mask-expand"
            type="number"
            min="1"
            max="3"
            step="0.1"
            value={config.face_mask?.expand ?? dflt('expand')}
            onChange={(e) => setField('face_mask', 'expand', parseFloat(e.target.value) || dflt('expand'))}
            className={INPUT_CLASS}
          />
          <p className="mt-1 text-xs text-content-muted">
            Face detection returns a box from the eyes to the chin. This grows it into a head:
            higher covers hair and jaw, lower stays tight on the face. Preview it on your own
            images from the training panel — the right value depends on how your shots are framed.
          </p>
          <ResetToDefault label="Head coverage" section="face_mask" field="expand"
            config={config} configDefaults={configDefaults} setField={setField} />
        </div>
        <div>
          <label htmlFor="face-mask-min-weight" className="block text-sm font-medium text-content">
            Loss weight kept on faces
          </label>
          <input
            id="face-mask-min-weight"
            type="number"
            min="0.05"
            max="1"
            step="0.05"
            value={config.face_mask?.min_weight ?? dflt('min_weight')}
            onChange={(e) => setField('face_mask', 'min_weight', parseFloat(e.target.value) || dflt('min_weight'))}
            className={INPUT_CLASS}
          />
          <p className="mt-1 text-xs text-content-muted">
            How much the masked area still counts. Lower pushes the identity out harder.
            It does not go to zero on purpose: an area worth nothing is not ignored, it is
            unpenalised — the model can put anything there at no cost, and reports of
            degraded anatomy start right below this floor.
          </p>
          <ResetToDefault label="Loss weight kept on faces" section="face_mask" field="min_weight"
            config={config} configDefaults={configDefaults} setField={setField} />
        </div>
      </div>
    </Card>
  )
}

export default function TrainingSection(props) {
  const { config, setField, configDefaults } = props
  // Summary + collapsible groups — same shells as Image engines. The vast.ai
  // KEY and the cloud GUARDRAILS live in ONE group on purpose: they are the
  // two halves of "training on a rented GPU", and they used to sit with the
  // masking card between them.
  const [defaultsGroup, cloudGroup, maskingGroup] = TRAINING_GROUPS
  const groupProps = useSettingsGroupProps('training')
  return (
    <div className="space-y-4">
      <SettingsGroupsToc sectionId="training" groups={TRAINING_GROUPS} />

      <SettingsGroup {...groupProps(defaultsGroup)}>
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
      </SettingsGroup>

      <SettingsGroup {...groupProps(cloudGroup)}>
      <Card title="Cloud GPU (vast.ai)" help="No local GPU? The app can rent one per run — the key below unlocks the ☁️ Train in cloud button.">
        <SecretField field={VAST_SECRET} {...props} />
      </Card>

      <CloudTrainingCard config={config} setField={setField} configDefaults={configDefaults} />
      </SettingsGroup>

      <SettingsGroup {...groupProps(maskingGroup)}>
      <ConceptFaceMaskCard config={config} setField={setField} configDefaults={configDefaults} />
      </SettingsGroup>
    </div>
  )
}
