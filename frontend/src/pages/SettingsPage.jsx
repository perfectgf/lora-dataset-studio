import { useEffect, useState, useCallback, useMemo, useRef } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { apiFetch, putJson, del } from '../api/fetchClient'
import { useToast } from '../components/common/Toast'
import { useCapabilities } from '../context/CapabilitiesContext'
import { SETTINGS_SECTIONS, sectionStatus, matchesQuery } from '../components/settings/registry'
import { SectionHeader } from '../components/settings/primitives'
import { HelpBadge } from '../help/HelpMode'
import { searchHelpTopics, helpTopics } from '../help/helpRegistry'
import { openCollapsedAncestors, resolveFocusTarget } from '../help/revealTarget'
import { buildGuideTextIndex, matchGuideAnchors } from '../help/guideTextIndex'
import settingsReferenceRaw from '../../../docs/guide/settings-reference.md?raw'
import OverviewSection from '../components/settings/OverviewSection'
import EnginesSection from '../components/settings/EnginesSection'
import ScrapingSection from '../components/settings/ScrapingSection'
import LocalToolsSection from '../components/settings/LocalToolsSection'
import CaptioningSection from '../components/settings/CaptioningSection'
import TrainingSection from '../components/settings/TrainingSection'
import ServerSection from '../components/settings/ServerSection'
import MaintenanceSection from '../components/settings/MaintenanceSection'

const SECTION_COMPONENTS = {
  overview: OverviewSection,
  engines: EnginesSection,
  scraping: ScrapingSection,
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
  const [query, setQuery] = useState('')

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

  // Persist ONE config section before a card's Test probe reads the SAVED value —
  // the config counterpart of saveSecretIfPending. The /api/settings/test/<target>
  // probes read config from disk, so testing a freshly-typed ComfyUI/Ollama/
  // ai-toolkit path/URL without this always answered "not configured".
  //  - No-op when that section is unchanged (nothing to save).
  //  - PUTs ONLY this section, never the other still-being-edited sections — so a
  //    Test click can't silently flush unrelated in-progress edits (mirrors how
  //    SecretField persists only its own key).
  //  - Reconciles just this section's local value + the saved snapshot with the
  //    server's canonical result (the ComfyUI base_dir auto-correction may rewrite
  //    what was typed) so the field shows what actually landed and the save bar's
  //    dirty flag clears for this section without a redundant second save.
  // Throws on failure so the Test button reports it (same contract as SecretField).
  const saveConfigSection = async (section) => {
    const current = config?.[section]
    if (!current || JSON.stringify(current) === JSON.stringify(savedConfig?.[section])) return
    const data = await putJson('/api/settings', { config: { [section]: current } })
    setConfig((prev) => ({ ...prev, [section]: data.config[section] }))
    setSavedConfig(data.config)
    setRuntime(data.runtime || { host: null, port: null })
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

  // Deep-link focus: /settings/<section>?focus=<domId> scrolls to that field and
  // flashes a ring. Depends on `config` so it re-runs once settings finish
  // loading — the field only exists after the active section has rendered.
  // The reveal helper first opens any collapsed <details> the field sits in
  // (e.g. the ai-toolkit overrides), or, when the field itself is gated behind a
  // switch that hasn't rendered it yet (the access token behind LAN +
  // require-token), rings that gate instead so the deep-link never dead-ends.
  const [searchParams] = useSearchParams()
  const focusId = searchParams.get('focus')
  useEffect(() => {
    if (!focusId || loading || !config) return undefined
    const found = resolveFocusTarget(focusId)
    if (!found) return undefined
    openCollapsedAncestors(found.el)
    found.el.scrollIntoView({ behavior: 'smooth', block: 'center' })
    const ring = ['ring-2', 'ring-indigo-400/70', 'ring-offset-2', 'ring-offset-app', 'rounded-md']
    found.el.classList.add(...ring)
    const t = setTimeout(() => found.el.classList.remove(...ring), 2000)
    return () => clearTimeout(t)
  }, [focusId, section, loading, config])

  // Enriched search index: besides the section rail, individual settings are
  // matched by their tutorial TEXT (docs/guide/settings-reference.md, indexed by
  // H2) so e.g. "crop" surfaces the watermark auto-crop setting via its docs.
  const guideIndex = useMemo(() => buildGuideTextIndex(settingsReferenceRaw), [])
  const settingResults = useMemo(() => {
    const q = query.trim()
    if (!q) return []
    const seen = new Set()
    const out = []
    const consider = (t) => {
      if ((t.kind === 'setting' || t.kind === 'action') && !seen.has(t.id)) {
        seen.add(t.id); out.push(t)
      }
    }
    for (const t of searchHelpTopics(q)) consider(t)          // label / keyword / id
    const anchors = matchGuideAnchors(guideIndex, q)          // tutorial text
    for (const t of helpTopics) {
      if (t.guide.chapter === 'settings-reference' && anchors.has(t.guide.anchor)) consider(t)
    }
    return out
  }, [query, guideIndex])
  // Keyboard cursor across the flat result list (sections then settings).
  const [activeResult, setActiveResult] = useState(-1)
  useEffect(() => { setActiveResult(-1) }, [query])

  const discard = () => {
    setConfig(savedConfig)
    setSecretInputs({})
  }

  if (loading || !config) {
    return <p className="text-content-muted">Loading settings…</p>
  }

  const sectionProps = {
    config, setField, secretsPresence, secretInputs, setSecretInputs,
    testResults, recordTestResult, saveSecretIfPending, saveConfigSection, handleDeleteSecret,
    toggleEngine, handleSave, saving, runtime, caps, refreshCaps: refresh, toast,
  }

  const activeId = SECTION_COMPONENTS[section] ? section : 'overview'
  const active = SETTINGS_SECTIONS.find((s) => s.id === activeId)
  const ActiveSection = SECTION_COMPONENTS[activeId]

  // Search filters the rail by title/keywords; the active section always stays
  // listed so the visible content is never orphaned from its nav item.
  const visibleSections = SETTINGS_SECTIONS.filter(
    (s) => s.id === activeId || matchesQuery(s, query)
  )

  const q = query.trim()
  // Flat, keyboard-navigable result list: matched sections first, then settings.
  const flatResults = q
    ? [...visibleSections.map((s) => ({ type: 'section', id: s.id })),
       ...settingResults.map((t) => ({ type: 'setting', topic: t }))]
    : []
  const goToSetting = (t) => {
    const { route, focus } = t.app
    if (!focus) { navigate(route); return }              // action topics carry no focus
    navigate(`${route}${route.includes('?') ? '&' : '?'}focus=${focus}`)
  }
  const activateResult = (r) => {
    if (!r) return
    if (r.type === 'section') navigate(`/settings/${r.id}`)
    else goToSetting(r.topic)
    setActiveResult(-1)
  }
  const onSearchKeyDown = (e) => {
    if (!flatResults.length) return
    if (e.key === 'ArrowDown') { e.preventDefault(); setActiveResult((i) => Math.min(flatResults.length - 1, i + 1)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActiveResult((i) => Math.max(0, i - 1)) }
    else if (e.key === 'Enter') { e.preventDefault(); activateResult(flatResults[activeResult >= 0 ? activeResult : 0]) }
    else if (e.key === 'Escape') { setQuery('') }
  }

  const navItem = (s, chip, resultIdx = null) => {
    const isActive = s.id === activeId
    const activeKb = resultIdx !== null && resultIdx === activeResult
    const base = chip
      ? `flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full border px-3 py-1.5 text-xs font-medium ${
          isActive ? 'border-border-strong bg-surface-raised text-content' : 'border-border text-content-muted hover:text-content'}`
      : `relative flex w-full items-center gap-2.5 rounded-md px-3 py-2 text-left text-sm font-medium ${
          isActive ? 'bg-surface-raised text-content' : 'text-content-muted hover:bg-surface hover:text-content'} ${
          activeKb ? 'ring-1 ring-inset ring-indigo-400/50' : ''}`
    return (
      <button key={s.id} type="button" onClick={() => navigate(`/settings/${s.id}`)}
        onMouseEnter={resultIdx !== null ? () => setActiveResult(resultIdx) : undefined}
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

  const settingResultItem = (t, resultIdx) => {
    const activeKb = resultIdx === activeResult
    return (
      <button key={t.id} type="button" onClick={() => goToSetting(t)}
        onMouseEnter={() => setActiveResult(resultIdx)}
        className={`flex w-full items-center gap-2 rounded-md px-3 py-1.5 text-left text-sm ${
          activeKb ? 'bg-surface text-content ring-1 ring-inset ring-indigo-400/50'
            : 'text-content-muted hover:bg-surface hover:text-content'}`}>
        <span aria-hidden className="text-content-subtle">›</span>
        <span className="truncate">{t.title}</span>
        <span className="ml-auto shrink-0 text-[10px] uppercase tracking-wide text-content-subtle">{t.kind}</span>
      </button>
    )
  }

  return (
    <div>
      <div className="lg:grid lg:grid-cols-[230px_minmax(0,1fr)] lg:items-start lg:gap-8">
        <aside>
          {/* Mobile: horizontal chip rail */}
          <nav aria-label="Settings sections" className="-mx-4 flex gap-2 overflow-x-auto px-4 pb-3 lg:hidden">
            {visibleSections.map((s) => navItem(s, true))}
          </nav>
          {/* Desktop: sticky LED rail */}
          <nav aria-label="Settings sections" className="hidden lg:sticky lg:top-20 lg:block">
            <p className="px-3 pb-2 font-mono text-[11px] uppercase tracking-[0.18em] text-content-subtle">Settings</p>
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={onSearchKeyDown}
              placeholder="Find a setting…"
              aria-label="Find a setting"
              className="mb-2 w-full rounded-md border border-border bg-surface px-3 py-1.5 text-xs text-content placeholder:text-content-subtle focus:border-primary focus:outline-none"
            />
            {q ? (
              <div className="space-y-3">
                <div>
                  <p className="px-3 pb-1 text-[11px] uppercase tracking-wide text-content-subtle" role="status">
                    Sections ({visibleSections.length})
                  </p>
                  <div className="flex flex-col gap-0.5">
                    {visibleSections.map((s, i) => navItem(s, false, i))}
                  </div>
                </div>
                {settingResults.length > 0 && (
                  <div>
                    <p className="px-3 pb-1 text-[11px] uppercase tracking-wide text-content-subtle">
                      Settings ({settingResults.length})
                    </p>
                    <div className="flex flex-col gap-0.5">
                      {settingResults.map((t, j) => settingResultItem(t, visibleSections.length + j))}
                    </div>
                  </div>
                )}
                {visibleSections.length === 0 && settingResults.length === 0 && (
                  <p className="px-3 text-xs text-content-subtle">No matches.</p>
                )}
              </div>
            ) : (
              <div className="flex flex-col gap-0.5">
                {visibleSections.map((s) => navItem(s, false))}
              </div>
            )}
          </nav>
        </aside>

        <div className="mt-2 space-y-6 lg:mt-0">
          <SectionHeader eyebrow={active.eyebrow} title={active.title} description={active.description}
            badge={<HelpBadge topic={`settings-${activeId}`} />} />
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
