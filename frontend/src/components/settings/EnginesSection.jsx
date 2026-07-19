import { useEffect, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { INPUT_CLASS, Card, StatusBadge, SecretField } from './primitives'
import KleinLoraCombobox, { useKleinGenerationLoras } from './KleinLoraCombobox'
import { useI18n } from '../../i18n/I18nContext'

const ENGINE_SECRETS = [
  { key: 'GEMINI_API_KEY', labelKey: 'geminiLabel', testTarget: 'gemini', helpKey: 'geminiHelp' },
  { key: 'OPENAI_API_KEY', labelKey: 'openaiLabel', testTarget: 'openai', helpKey: 'openaiHelp' },
]

const ENGINE_OPTIONS = [
  { id: 'nanobanana', label: 'Nano Banana (Gemini)' },
  { id: 'chatgpt', label: 'ChatGPT (gpt-image-2)' },
  { id: 'klein', label: 'Klein (ComfyUI, local)' },
]

/* Optional generation-LoRA PRESETS for the local Klein engine (Idea by
   @waltm — Discord feature request): named combinations of user-pointed LoRA
   files (any files, any purpose — texture, anatomy, style…). Inside a preset
   the rows chain after the consistency LoRA in LIST ORDER (file + strength,
   reorderable, capped at 8). Per run the workspace's 🖥️ Klein tuning panel
   just PICKS a preset ("None" by default) — the choice carries the intent,
   there is no automatic gating. The app never ships or hardcodes a LoRA name. */
const MAX_GENERATION_LORAS = 8        // mirrors backend klein_edit_helper caps
const MAX_GENERATION_LORA_PRESETS = 12

const SMALL_BTN = 'grid h-6 w-6 place-items-center rounded border border-border text-xs ' +
  'text-content-muted hover:bg-surface-raised disabled:opacity-30'
const TEXT_BTN = 'rounded-md border border-border-strong px-2 py-1 text-xs font-medium ' +
  'text-content hover:bg-surface-raised disabled:opacity-50'

/** Fresh name not colliding with the existing presets ("Preset 2", "x (copy)"…). */
function freeName(presets, base) {
  const taken = new Set(presets.map((p) => (p?.name || '').trim()))
  if (!taken.has(base)) return base
  for (let n = 2; ; n += 1) {
    const cand = `${base} ${n}`
    if (!taken.has(cand)) return cand
  }
}

function KleinLoraPresetCard({ preset, index, presets, save, loraScan }) {
  const { t } = useI18n()
  const rows = Array.isArray(preset?.loras) ? preset.loras : []
  const patchPreset = (p) => save(presets.map((x, j) => (j === index ? { ...x, ...p } : x)))
  const patchRow = (i, p) => patchPreset({ loras: rows.map((r, j) => (j === i ? { ...r, ...p } : r)) })
  const moveRow = (i, dir) => {
    const j = i + dir
    if (j < 0 || j >= rows.length) return
    const next = [...rows]
    ;[next[i], next[j]] = [next[j], next[i]]
    patchPreset({ loras: next })
  }
  return (
    <div className="rounded-lg border border-border p-3 space-y-2">
      <div className="flex items-center gap-2">
        <input
          type="text" aria-label={t('settings.engines.presets.nameLabel', { number: index + 1 })}
          value={preset?.name || ''}
          onChange={(e) => patchPreset({ name: e.target.value })}
          placeholder={t('settings.engines.presets.namePlaceholder')}
          className={`${INPUT_CLASS} mt-0 font-medium`}
        />
        <button type="button" className={TEXT_BTN}
          disabled={presets.length >= MAX_GENERATION_LORA_PRESETS}
          onClick={() => save([...presets,
            { ...preset, name: freeName(presets, `${(preset?.name || t('settings.engines.presets.defaultName')).trim()
              || t('settings.engines.presets.defaultName')} ${t('settings.engines.presets.copySuffix')}`),
              loras: rows.map((r) => ({ ...r })) }])}
          title={t('settings.engines.presets.duplicateTitle')}>
          {t('settings.engines.presets.duplicate')}
        </button>
        <button type="button" className={`${TEXT_BTN} hover:bg-red-500/15 hover:text-red-300`}
          onClick={() => save(presets.filter((_, j) => j !== index))}
          title={t('settings.engines.presets.deleteTitle')}>
          {t('common.remove')}
        </button>
      </div>
      {rows.length === 0 && (
        <p className="text-xs text-content-muted">{t('settings.engines.presets.empty')}</p>
      )}
      {rows.map((row, i) => {
        const strength = Number.isFinite(Number(row?.strength)) ? Number(row.strength) : 0.6
        return (
          <div key={i} className="flex items-center gap-2 flex-wrap">
            <span className="text-xs text-content-muted w-4 shrink-0" aria-hidden="true">{i + 1}.</span>
            <KleinLoraCombobox
              ariaLabel={t('settings.engines.presets.fileLabel', { preset: index + 1, row: i + 1 })}
              value={row?.file || ''}
              onChange={(next) => patchRow(i, { file: next })}
              {...loraScan}
            />
            <label className="flex items-center gap-1.5 text-xs text-content-muted">
              <span className="whitespace-nowrap">{strength.toFixed(2)}</span>
              <input
                type="range" min={0} max={1.5} step={0.05} value={strength}
                aria-label={t('settings.engines.presets.strengthLabel', { preset: index + 1, row: i + 1 })}
                onChange={(e) => patchRow(i, { strength: Number(e.target.value) })}
                className="w-28 accent-indigo-500"
              />
            </label>
            <button type="button" onClick={() => moveRow(i, -1)} disabled={i === 0}
              aria-label={t('settings.engines.presets.moveUpLabel', { row: i + 1, preset: index + 1 })}
              title={t('settings.engines.presets.earlier')} className={SMALL_BTN}>↑</button>
            <button type="button" onClick={() => moveRow(i, 1)} disabled={i === rows.length - 1}
              aria-label={t('settings.engines.presets.moveDownLabel', { row: i + 1, preset: index + 1 })}
              title={t('settings.engines.presets.later')} className={SMALL_BTN}>↓</button>
            <button type="button" onClick={() => patchPreset({ loras: rows.filter((_, j) => j !== i) })}
              aria-label={t('settings.engines.presets.removeLabel', { row: i + 1, preset: index + 1 })}
              title={t('settings.engines.presets.removeTitle')}
              className={`${SMALL_BTN} hover:bg-red-500/15 hover:text-red-300`}>✕</button>
          </div>
        )
      })}
      <div className="flex items-center gap-3">
        <button
          type="button" className={TEXT_BTN}
          onClick={() => patchPreset({ loras: [...rows, { file: '', strength: 0.6 }] })}
          disabled={rows.length >= MAX_GENERATION_LORAS}
        >
          ＋ {t('settings.engines.presets.addLora')}
        </button>
        <span className="text-xs text-content-muted">
          {t('settings.engines.presets.chainCount', { count: rows.length, max: MAX_GENERATION_LORAS })}
        </span>
      </div>
    </div>
  )
}

function KleinLorasCard({ config, setField }) {
  const { t } = useI18n()
  const presets = Array.isArray(config.klein?.generation_lora_presets)
    ? config.klein.generation_lora_presets : []
  const save = (next) => setField('klein', 'generation_lora_presets', next)
  // ONE scan of ComfyUI's loras folder, shared by every row's picker (never one
  // fetch per row). Degrades to free-text on any failure — see the hook.
  const loraScan = useKleinGenerationLoras()
  return (
    <Card
      id="klein-generation-lora-presets"
      title={t('settings.engines.presets.title')}
      help={t('settings.engines.presets.help', {
        loras: MAX_GENERATION_LORAS,
        presets: MAX_GENERATION_LORA_PRESETS,
      })}
    >
      {presets.length === 0 && (
        <p className="text-sm text-content-muted">{t('settings.engines.presets.none')}</p>
      )}
      {presets.map((preset, i) => (
        <KleinLoraPresetCard key={i} preset={preset} index={i} presets={presets} save={save} loraScan={loraScan} />
      ))}
      <div className="flex items-center gap-3">
        <button
          type="button" className={TEXT_BTN}
          onClick={() => save([...presets, {
            name: freeName(presets, t('settings.engines.presets.myPreset')), loras: [],
          }])}
          disabled={presets.length >= MAX_GENERATION_LORA_PRESETS}
        >
          ＋ {t('settings.engines.presets.new')}
        </button>
        <span className="text-xs text-content-muted">{presets.length}/{MAX_GENERATION_LORA_PRESETS}</span>
      </div>
    </Card>
  )
}

const CHATGPT_AUTH_OPTIONS = [
  { id: 'auto', labelKey: 'auto' },
  { id: 'api', labelKey: 'api' },
  { id: 'subscription', labelKey: 'subscription' },
]

/* ChatGPT subscription (Codex OAuth) — EXPERIMENTAL lane. Device-code login:
   the user opens the verification URL from ANY device and types the one-time
   code; we poll the backend until it reports connected. */
function ChatgptSubscriptionCard({ caps, config, setField, refreshCaps, toast }) {
  const { t } = useI18n()
  const sub = caps.chatgpt_subscription || {}
  const [device, setDevice] = useState(null)     // {verification_url, user_code}
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!device) return undefined
    const id = setInterval(async () => {
      try {
        const r = await apiFetch('/api/settings/chatgpt-oauth/poll')
        if (r.status === 'connected') {
          setDevice(null)
          toast.success(t('settings.engines.subscription.connectedToast'))
          await refreshCaps(true)
        } else if (r.status === 'error') {
          setDevice(null)
          setError(r.detail || t('settings.engines.subscription.loginFailed'))
        }
      } catch { /* transient — keep polling */ }
    }, 3000)
    return () => clearInterval(id)
  }, [device, refreshCaps, t, toast])

  const start = async () => {
    setBusy(true); setError(null)
    try {
      const r = await postJson('/api/settings/chatgpt-oauth/start', {})
      setDevice(r)
    } catch (e) {
      setError(e.message || t('settings.engines.subscription.startFailed'))
    } finally {
      setBusy(false)
    }
  }

  const importCodex = async () => {
    setBusy(true); setError(null)
    try {
      await postJson('/api/settings/chatgpt-oauth/import-codex', {})
      setDevice(null)
      toast.success(t('settings.engines.subscription.importedToast'))
      await refreshCaps(true)
    } catch (e) {
      setError(e.message || t('settings.engines.subscription.importFailed'))
    } finally {
      setBusy(false)
    }
  }

  const disconnect = async () => {
    setBusy(true); setError(null)
    try {
      await postJson('/api/settings/chatgpt-oauth/logout', {})
      toast.success(t('settings.engines.subscription.disconnectedToast'))
      await refreshCaps(true)
    } catch (e) {
      setError(e.message || t('settings.engines.subscription.disconnectFailed'))
    } finally {
      setBusy(false)
    }
  }

  const btn = 'rounded-md border border-border-strong px-3 py-1.5 text-xs font-medium ' +
    'text-content hover:bg-surface-raised disabled:opacity-50'

  return (
    <Card
      title={t('settings.engines.subscription.title')}
      help={t('settings.engines.subscription.help')}
    >
      <div className="flex items-center justify-between">
        <StatusBadge ok={!!sub.connected}
          okLabel={sub.email
            ? t('settings.engines.subscription.connectedAs', { email: sub.email })
            : t('settings.engines.subscription.connected')}
          missingLabel={t('settings.engines.subscription.notConnected')} />
        <div className="flex gap-2">
          {!sub.connected && (
            <button type="button" onClick={start} disabled={busy || !!device} className={btn}>
              {device
                ? t('settings.engines.subscription.waiting')
                : t('settings.engines.subscription.connect')}
            </button>
          )}
          {!sub.connected && sub.codex_cli_detected && (
            <button type="button" onClick={importCodex} disabled={busy || !!device} className={btn}>
              {t('settings.engines.subscription.importCodex')}
            </button>
          )}
          {sub.connected && (
            <button type="button" onClick={disconnect} disabled={busy} className={btn}>
              {t('settings.engines.subscription.disconnect')}
            </button>
          )}
        </div>
      </div>

      {device && (
        <div role="status" className="rounded-lg border border-primary/40 bg-primary/10 p-3 text-sm text-content">
          <p>{t('settings.engines.subscription.openUrl')}{' '}
            <a href={device.verification_url} target="_blank" rel="noreferrer"
              className="font-medium underline">{device.verification_url}</a>
          </p>
          <p className="mt-1">{t('settings.engines.subscription.enterCode')}</p>
          <p className="mt-1 select-all font-mono text-lg font-semibold tracking-widest">{device.user_code}</p>
        </div>
      )}

      {error && <p className="text-xs text-rose-400"><span aria-hidden="true">✗</span> {error}</p>}

      <div>
        <label htmlFor="chatgpt-auth-mode" className="block text-sm font-medium text-content">
          {t('settings.engines.subscription.authMode')}
        </label>
        <select
          id="chatgpt-auth-mode"
          value={config.engines.chatgpt_auth || 'auto'}
          onChange={(e) => setField('engines', 'chatgpt_auth', e.target.value)}
          className={INPUT_CLASS}
        >
          {CHATGPT_AUTH_OPTIONS.map((o) => (
            <option key={o.id} value={o.id}>
              {t(`settings.engines.subscription.authOptions.${o.labelKey}`)}
            </option>
          ))}
        </select>
        <p className="mt-1 text-xs text-content-muted">
          {t('settings.engines.subscription.quotaHelp')}
        </p>
      </div>
    </Card>
  )
}

export default function EnginesSection(props) {
  const { t } = useI18n()
  const { config, setField, toggleEngine, caps, refreshCaps, toast } = props
  return (
    <div className="space-y-6">
      <Card title={t('settings.engines.apiKeysTitle')} help={t('settings.engines.apiKeysHelp')}>
        {ENGINE_SECRETS.map((f) => <SecretField key={f.key} field={{
          ...f,
          label: t(`settings.engines.${f.labelKey}`),
          help: t(`settings.engines.${f.helpKey}`),
        }} {...props} />)}
      </Card>

      <ChatgptSubscriptionCard caps={caps} config={config} setField={setField} refreshCaps={refreshCaps} toast={toast} />

      <Card title={t('settings.engines.enginesTitle')} help={t('settings.engines.enginesHelp')}>
        <div>
          <label htmlFor="engine-default" className="block text-sm font-medium text-content">
            {t('settings.engines.defaultEngine')}
          </label>
          <select
            id="engine-default"
            value={config.engines.default}
            onChange={(e) => setField('engines', 'default', e.target.value)}
            className={INPUT_CLASS}
          >
            {ENGINE_OPTIONS.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
          </select>
        </div>

        <fieldset id="engines-enabled" className="scroll-mt-24">
          <legend className="mb-1 block text-sm font-medium text-content">
            {t('settings.engines.enabledEngines')}
          </legend>
          <div className="flex flex-col gap-2">
            {ENGINE_OPTIONS.map((o) => (
              <label key={o.id} htmlFor={`engine-enabled-${o.id}`} className="flex items-center gap-2 text-sm text-content">
                <input
                  id={`engine-enabled-${o.id}`}
                  type="checkbox"
                  checked={(config.engines.enabled || []).includes(o.id)}
                  onChange={() => toggleEngine(o.id)}
                  className="h-4 w-4 rounded border-border-strong"
                />
                {o.label}
              </label>
            ))}
          </div>
        </fieldset>
      </Card>

      <KleinLorasCard config={config} setField={setField} />
    </div>
  )
}
