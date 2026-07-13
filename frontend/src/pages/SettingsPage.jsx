import { useEffect, useState, useCallback, useMemo } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { apiFetch, putJson, del } from '../api/fetchClient'
import { useToast } from '../components/common/Toast'
import { useCapabilities } from '../context/CapabilitiesContext'
import { SETTINGS_SECTIONS, sectionStatus } from '../components/settings/registry'
import { SectionHeader } from '../components/settings/primitives'
import OverviewSection from '../components/settings/OverviewSection'
import EnginesSection from '../components/settings/EnginesSection'
import LocalToolsSection from '../components/settings/LocalToolsSection'
import CaptioningSection from '../components/settings/CaptioningSection'
import TrainingSection from '../components/settings/TrainingSection'
import ServerSection from '../components/settings/ServerSection'
import MaintenanceSection from '../components/settings/MaintenanceSection'

const SECTION_COMPONENTS = {
  overview: OverviewSection,
  engines: EnginesSection,
  'local-tools': LocalToolsSection,
  captioning: CaptioningSection,
  training: TrainingSection,
  server: ServerSection,
  maintenance: MaintenanceSection,
}

/* Sidebar LED: the section's live health, so the rail doubles as a status map.
   Never color-only — an sr-only label spells the state out. */
function StatusLed({ status }) {
  if (!status) return null
  const cls = status === 'ready' ? 'bg-emerald-400'
    : status === 'partial' ? 'bg-amber-400'
    : 'bg-white/15'
  const label = status === 'ready' ? 'configured'
    : status === 'partial' ? 'partly configured'
    : 'not configured'
  return (
    <span className="ml-auto flex items-center pl-2">
      <span aria-hidden className={`h-1.5 w-1.5 rounded-full ${cls}`} />
      <span className="sr-only">({label})</span>
    </span>
  )
}

export default function SettingsPage() {
  const toast = useToast()
  const { caps, refresh } = useCapabilities()
  const { section } = useParams()
  const navigate = useNavigate()
  const [config, setConfig] = useState(null)
  // Snapshot of the last-persisted config: the floating save bar appears when
  // the edited config drifts from it (or a secret paste is pending).
  const [savedConfig, setSavedConfig] = useState(null)
  const [runtime, setRuntime] = useState({ host: null, port: null })
  const [secretsPresence, setSecretsPresence] = useState({})
  const [secretInputs, setSecretInputs] = useState({})
  const [testResults, setTestResults] = useState({})
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const data = await apiFetch('/api/settings')
      setConfig(data.config)
      setSavedConfig(data.config)
      setRuntime(data.runtime || { host: null, port: null })
      setSecretsPresence(data.secrets)
    } catch (e) {
      toast.error(`Failed to load settings: ${e.message}`)
    } finally {
      setLoading(false)
    }
  }, [toast])

  useEffect(() => { load() }, [load])

  const setField = (section, key, value) => {
    setConfig((prev) => ({ ...prev, [section]: { ...prev[section], [key]: value } }))
  }

  const recordTestResult = (target, result) => {
    setTestResults((prev) => ({ ...prev, [target]: result }))
  }

  const toggleEngine = (id) => {
    setConfig((prev) => {
      const enabled = prev.engines.enabled || []
      const next = enabled.includes(id) ? enabled.filter((e) => e !== id) : [...enabled, id]
      return { ...prev, engines: { ...prev.engines, enabled: next } }
    })
  }

  // Clear a saved API key. Explicit action — the write-only field can't wipe a key
  // by going blank — so confirm, delete server-side, then refresh presence + caps
  // so any engine that depended on it flips to unavailable right away.
  const handleDeleteSecret = async (key, label) => {
    if (!window.confirm(`Remove the saved ${label}? Any engine that uses it stops working until you add a new key.`)) return
    try {
      const data = await del(`/api/settings/secret/${key}`)
      setSecretsPresence(data.secrets)
      setSecretInputs((prev) => { const next = { ...prev }; delete next[key]; return next })
      await refresh(true)
      toast.success(`${label} removed.`)
    } catch (e) {
      toast.error(`Remove failed: ${e.message}`)
    }
  }

  // Save a single secret field's pending value (used by the Test button so
  // "paste key -> Test" just works without a separate Save click). No-op when
  // the field is empty; a failed save throws so the test reports it instead of
  // probing a key that never landed.
  const saveSecretIfPending = async (key) => {
    const pending = (secretInputs[key] || '').trim()
    if (!pending) return
    const data = await putJson('/api/settings', { secrets: { [key]: pending } })
    setSecretsPresence(data.secrets)
    setSecretInputs((prev) => { const next = { ...prev }; delete next[key]; return next })
    await refresh(true)
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      // Only send secret fields the user actually typed into — the fields
      // stay blank on load, so an untouched field must never overwrite an
      // already-saved key with an empty value. Trim: a pasted key with
      // trailing whitespace/newline would otherwise corrupt the Bearer header.
      const secrets = Object.fromEntries(
        Object.entries(secretInputs)
          .map(([k, v]) => [k, (v || '').trim()])
          .filter(([, v]) => v)
      )
      const data = await putJson('/api/settings', { config, secrets })
      setConfig(data.config)
      setSavedConfig(data.config)
      setRuntime(data.runtime || { host: null, port: null })
      setSecretsPresence(data.secrets)
      setSecretInputs({})
      // force=true: /api/capabilities caches probes for 30s server-side, so a
      // plain refresh() could leave onboarding/studio_visible stale right
      // after the config that determines them just changed.
      await refresh(true)
      toast.success('Settings saved.')
      return true
    } catch (e) {
      toast.error(`Save failed: ${e.message}`)
      return false
    } finally {
      setSaving(false)
    }
  }

  // Dirty = the edited config drifted from the saved snapshot, or a secret is
  // typed but not yet persisted. Drives the floating save bar + the tab-close
  // guard. (Section switches keep this page mounted, so edits survive them.)
  const dirty = useMemo(() => !!(config && savedConfig
      && (JSON.stringify(config) !== JSON.stringify(savedConfig)
          || Object.values(secretInputs).some((v) => (v || '').trim()))),
    [config, savedConfig, secretInputs])

  useEffect(() => {
    if (!dirty) return undefined
    const warn = (e) => { e.preventDefault(); e.returnValue = '' }
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [dirty])

  const discard = () => {
    setConfig(savedConfig)
    setSecretInputs({})
  }

  if (loading || !config) {
    return <p className="text-content-muted">Loading settings…</p>
  }

  const sectionProps = {
    config, setField, secretsPresence, secretInputs, setSecretInputs,
    testResults, recordTestResult, saveSecretIfPending, handleDeleteSecret,
    toggleEngine, handleSave, saving, runtime, caps, refreshCaps: refresh, toast,
  }

  const activeId = SECTION_COMPONENTS[section] ? section : 'overview'
  const active = SETTINGS_SECTIONS.find((s) => s.id === activeId)
  const ActiveSection = SECTION_COMPONENTS[activeId]

  const navItem = (s, chip) => {
    const isActive = s.id === activeId
    const base = chip
      ? `flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full border px-3 py-1.5 text-xs font-medium ${
          isActive ? 'border-border-strong bg-surface-raised text-content' : 'border-border text-content-muted hover:text-content'}`
      : `relative flex w-full items-center gap-2.5 rounded-md px-3 py-2 text-left text-sm font-medium ${
          isActive ? 'bg-surface-raised text-content' : 'text-content-muted hover:bg-surface hover:text-content'}`
    return (
      <button key={s.id} type="button" onClick={() => navigate(`/settings/${s.id}`)}
        aria-current={isActive ? 'page' : undefined} className={base}>
        {!chip && isActive && (
          <span aria-hidden className="absolute bottom-1.5 left-0 top-1.5 w-0.5 rounded bg-gradient-primary" />
        )}
        <span aria-hidden>{s.icon}</span>
        <span>{s.title}</span>
        {!chip && <StatusLed status={sectionStatus(s.id, caps)} />}
      </button>
    )
  }

  return (
    <div>
      <div className="lg:grid lg:grid-cols-[230px_minmax(0,1fr)] lg:items-start lg:gap-8">
        <aside>
          {/* Mobile: horizontal chip rail */}
          <nav aria-label="Settings sections" className="-mx-4 flex gap-2 overflow-x-auto px-4 pb-3 lg:hidden">
            {SETTINGS_SECTIONS.map((s) => navItem(s, true))}
          </nav>
          {/* Desktop: sticky LED rail */}
          <nav aria-label="Settings sections" className="hidden lg:sticky lg:top-20 lg:block">
            <p className="px-3 pb-2 font-mono text-[11px] uppercase tracking-[0.18em] text-content-subtle">Settings</p>
            <div className="flex flex-col gap-0.5">
              {SETTINGS_SECTIONS.map((s) => navItem(s, false))}
            </div>
          </nav>
        </aside>

        <div className="mt-2 space-y-6 lg:mt-0">
          <SectionHeader eyebrow={active.eyebrow} title={active.title} description={active.description} />
          <ActiveSection {...sectionProps} />
        </div>
      </div>

      {/* Floating save bar — only exists while there is something to save, so
          the page never shows a dead "Save" button. */}
      {dirty && (
        <div role="status"
          className="fixed inset-x-0 bottom-4 z-40 mx-auto flex w-fit max-w-[calc(100vw-2rem)] items-center gap-3 rounded-full border border-border bg-surface-overlay/95 px-4 py-2 shadow-lg backdrop-blur">
          <span aria-hidden className="text-amber-400">●</span>
          <span className="text-sm text-content">Unsaved changes</span>
          <button type="button" onClick={discard}
            className="rounded-full border border-border-strong px-3 py-1 text-xs font-medium text-content hover:bg-surface-raised">
            Discard
          </button>
          <button type="button" onClick={handleSave} disabled={saving}
            className="rounded-full bg-gradient-primary px-4 py-1 text-xs font-semibold text-white disabled:opacity-50">
            {saving ? 'Saving…' : 'Save changes'}
          </button>
        </div>
      )}
    </div>
  )
}
