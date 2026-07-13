import { useEffect, useState, useCallback } from 'react'
import { apiFetch, putJson, del } from '../api/fetchClient'
import { useToast } from '../components/common/Toast'
import { useCapabilities } from '../context/CapabilitiesContext'
import EnginesSection from '../components/settings/EnginesSection'
import LocalToolsSection from '../components/settings/LocalToolsSection'
import CaptioningSection from '../components/settings/CaptioningSection'
import TrainingSection from '../components/settings/TrainingSection'
import ServerSection from '../components/settings/ServerSection'
import MaintenanceSection from '../components/settings/MaintenanceSection'

export default function SettingsPage() {
  const toast = useToast()
  const { caps, refresh } = useCapabilities()
  const [config, setConfig] = useState(null)
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

  if (loading || !config) {
    return <p className="text-content-muted">Loading settings…</p>
  }

  const sectionProps = {
    config, setField, secretsPresence, secretInputs, setSecretInputs,
    testResults, recordTestResult, saveSecretIfPending, handleDeleteSecret,
    toggleEngine, handleSave, saving, runtime, caps, refreshCaps: refresh, toast,
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-4">
        <h1 className="text-xl font-semibold text-content">Settings</h1>
        <button
          type="button"
          onClick={handleSave}
          disabled={saving}
          className="rounded-lg bg-gradient-primary px-4 py-2 text-sm font-semibold text-white transition-transform hover:-translate-y-px disabled:opacity-50"
        >
          {saving ? 'Saving…' : 'Save changes'}
        </button>
      </div>

      {!caps.configured && (
        <div role="status" className="rounded-xl border border-primary/40 bg-primary/10 p-4 text-sm text-content">
          <p className="font-medium">Let's get you set up.</p>
          <p className="mt-1 text-content-muted">
            Add at least one image API key to start. Add ComfyUI + ai-toolkit for local generation & training.
          </p>
        </div>
      )}

      <EnginesSection {...sectionProps} />
      <LocalToolsSection {...sectionProps} />
      <CaptioningSection {...sectionProps} />
      <TrainingSection {...sectionProps} />
      <ServerSection {...sectionProps} />
      <MaintenanceSection {...sectionProps} />

      <div className="flex justify-end">
        <button
          type="button"
          onClick={handleSave}
          disabled={saving}
          className="rounded-lg bg-gradient-primary px-4 py-2 text-sm font-semibold text-white transition-transform hover:-translate-y-px disabled:opacity-50"
        >
          {saving ? 'Saving…' : 'Save changes'}
        </button>
      </div>
    </div>
  )
}
