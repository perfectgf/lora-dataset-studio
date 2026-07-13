import { useState } from 'react'
import { postJson } from '../../api/fetchClient'

export const INPUT_CLASS =
  'mt-1 w-full rounded-md border border-border-strong bg-surface-raised px-3 py-2 text-sm text-content ' +
  'placeholder:text-content-subtle focus:border-primary focus:outline-none'

/* Section heading: a small mono "rack tag" eyebrow above the title keeps every
   settings/guide section labeled the same way without shouting. */
export function SectionHeader({ eyebrow, title, description }) {
  return (
    <div>
      <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-content-subtle">{eyebrow}</p>
      <h1 className="mt-1 text-xl font-semibold text-content">{title}</h1>
      {description && <p className="mt-1 text-sm text-content-muted">{description}</p>}
    </div>
  )
}

// Status is never color-only: an explicit glyph + text label carries the
// meaning, color is a reinforcing cue on top.
export function StatusBadge({ ok, okLabel = 'Configured', missingLabel = 'Not set' }) {
  return (
    <span className={`inline-flex items-center gap-1 text-xs font-medium ${ok ? 'text-emerald-400' : 'text-content-subtle'}`}>
      <span aria-hidden="true">{ok ? '✓' : '✗'}</span>
      {ok ? okLabel : missingLabel}
    </span>
  )
}

export function TestResult({ result }) {
  if (!result) return null
  return (
    <p className={`text-xs ${result.ok ? 'text-emerald-400' : 'text-rose-400'}`}>
      <span aria-hidden="true">{result.ok ? '✓' : '✗'}</span> {result.detail}
    </p>
  )
}

export function TestButton({ target, onResult, beforeTest }) {
  const [busy, setBusy] = useState(false)
  const run = async () => {
    setBusy(true)
    try {
      // Secret fields pass beforeTest to persist the value still sitting in the
      // write-only input: the probe reads the SAVED key, so testing an unsaved
      // paste would always answer "key missing".
      if (beforeTest) await beforeTest()
      onResult(await postJson(`/api/settings/test/${target}`, {}))
    } catch (e) {
      onResult({ ok: false, detail: e.message || 'Test failed' })
    } finally {
      setBusy(false)
    }
  }
  return (
    <button
      type="button"
      onClick={run}
      disabled={busy}
      className="shrink-0 rounded-md border border-border-strong px-3 py-1.5 text-xs font-medium text-content hover:bg-surface-raised disabled:opacity-50"
    >
      {busy ? 'Testing…' : 'Test'}
    </button>
  )
}

export function Card({ title, help, children }) {
  return (
    <section className="rounded-xl border border-border bg-surface p-5">
      <h2 className="text-base font-semibold text-content">{title}</h2>
      {help && <p className="mt-1 text-sm text-content-muted">{help}</p>}
      <div className="mt-4 space-y-4">{children}</div>
    </section>
  )
}

export function TextField({ id, label, value, onChange, placeholder, help }) {
  return (
    <div>
      <label htmlFor={id} className="block text-sm font-medium text-content">{label}</label>
      {help && <p className="mb-1 text-xs text-content-muted">{help}</p>}
      <input
        id={id}
        type="text"
        value={value ?? ''}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className={INPUT_CLASS}
      />
    </div>
  )
}

/* One saved-secret row: write-only password input + presence badge + optional
   Test (persists the pending paste first) + Remove. `field` comes from a
   SECRET_FIELDS-style descriptor: { key, label, testTarget, help, guide? }. */
export function SecretField({
  field, secretsPresence, secretInputs, setSecretInputs,
  testResults, recordTestResult, saveSecretIfPending, handleDeleteSecret,
}) {
  const f = field
  return (
    <div className="flex items-end gap-3">
      <div className="flex-1">
        <div className="flex items-center justify-between">
          <label htmlFor={f.key} className="block text-sm font-medium text-content">{f.label}</label>
          <StatusBadge ok={!!secretsPresence[f.key]} />
        </div>
        <p className="mb-1 text-xs text-content-muted">{f.help}</p>
        {f.guide}
        <input
          id={f.key}
          type="password"
          autoComplete="off"
          value={secretInputs[f.key] ?? ''}
          onChange={(e) => setSecretInputs((prev) => ({ ...prev, [f.key]: e.target.value }))}
          placeholder={secretsPresence[f.key] ? 'Already set — enter a new value to replace it' : 'Not set'}
          className={INPUT_CLASS}
        />
        {f.testTarget && <TestResult result={testResults[f.testTarget]} />}
      </div>
      {f.testTarget && (
        <TestButton target={f.testTarget} beforeTest={() => saveSecretIfPending(f.key)}
          onResult={(r) => recordTestResult(f.testTarget, r)} />
      )}
      {secretsPresence[f.key] && (
        <button
          type="button"
          onClick={() => handleDeleteSecret(f.key, f.label)}
          title={`Remove the saved ${f.label}`}
          className="shrink-0 rounded-md border border-rose-500/40 px-3 py-1.5 text-xs font-medium text-rose-300 hover:bg-rose-500/10"
        >
          Remove
        </button>
      )}
    </div>
  )
}
