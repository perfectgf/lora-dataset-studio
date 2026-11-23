import { useCallback, useEffect, useState } from 'react'
import { apiFetch } from '../../api/fetchClient'
import { Card } from './primitives'
import ModelFilePicker from './ModelFilePicker'
import InstallRunner from '../setup/InstallRunner'
import ComfyNodeRepair from '../setup/ComfyNodeRepair'

/** The same resolved files and install actions used by Studio's preflight. */
export default function TrainedImageModelsCard({ config, setField }) {
  const [families, setFamilies] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const load = useCallback(async (force = false) => {
    setLoading(true)
    setError('')
    try {
      const data = await apiFetch(`/api/comfy/trained-image-models${force ? '?force=1' : ''}`)
      setFamilies(data.families || [])
    } catch (e) {
      setError(e.message || 'Could not check image generation models.')
    } finally {
      setLoading(false)
    }
  }, [])
  useEffect(() => { load() }, [load])
  const refresh = () => load(true)
  return (
    <Card title="Test Studio image models" help="Prepare Z-Image, FLUX.1, Anima and Qwen-Image 2.1 for generation. Downloads start only when you choose Install.">
      <div id="studio-models" className="scroll-mt-24 space-y-3">
        <p className="text-xs text-content-muted">
          Leave a file blank to detect it automatically. Existing compatible files are reused,
          including shared model folders. Save changed file selections before checking again.
        </p>
        <button type="button" onClick={refresh} disabled={loading}
          className="min-h-10 rounded-md border border-border px-3 py-1.5 text-xs text-content disabled:opacity-50">
          {loading ? 'Checking models…' : 'Check models again'}
        </button>
        {error && <p role="alert" className="text-sm text-red-400">{error}</p>}
        {families.map((family) => {
          const values = config.studio_models?.[family.family] || {}
          const missing = new Set(family.install_actions || [])
          return (
            <details key={family.family} className="rounded-lg border border-border p-3">
              <summary className="min-h-10 cursor-pointer text-sm font-medium text-content">
                {family.label}
                <span className="ml-2 text-xs font-normal text-content-muted">
                  {family.ready ? 'Ready' : family.pin_warnings?.length || family.config_error ? 'Check file selection'
                    : !family.models_ready ? 'Model files needed'
                    : !family.nodes_checked ? 'Start ComfyUI to check' : 'ComfyUI update needed'}
                </span>
              </summary>
              <div className="space-y-4 pt-2">
                {family.config_error && <p role="alert" className="text-sm text-red-400">{family.config_error}</p>}
                {family.missing_nodes?.length > 0 && (
                  <ComfyNodeRepair nodes={family.missing_nodes} onRefresh={refresh} />
                )}
                {!family.nodes_checked && <p className="text-xs text-content-muted">
                  ComfyUI did not answer the node check. Model files alone do not confirm readiness.
                </p>}
                {family.slots.map((slot) => (
                  <div key={slot.key}>
                    {/* A slot that is not configurable (Z-Image) shows the file its own resolvers found. */}
                    {slot.configurable === false && <p className="mb-1 text-sm text-content">{slot.label}</p>}
                    {slot.configurable === false && <p className="break-all text-xs text-content-muted">
                      {family.assets?.[slot.key] ? `Found: ${family.assets[slot.key]}` : `Not found · ${slot.filename}`}
                    </p>}
                    {slot.configurable !== false && <label htmlFor={`studio-${family.family}-${slot.key}`} className="mb-1 block text-sm text-content">
                      {slot.label}
                    </label>}
                    {slot.configurable !== false && <ModelFilePicker id={`studio-${family.family}-${slot.key}`}
                      ariaLabel={`${family.label} ${slot.label}`}
                      value={values[slot.key] || ''}
                      onChange={(value) => setField('studio_models', family.family, { ...values, [slot.key]: value })}
                      placeholder={`Auto · ${family.assets?.[slot.key] || slot.filename}`}
                      files={slot.files || []} folder={`ComfyUI models/${slot.kind}`}
                      loading={loading} error={false} rescan={refresh} rescanning={loading} />}
                    {family.invalid_assets?.filter((item) => item.slot === slot.key).map((item) => (
                      <p key={item.asset} className="mt-1 text-xs text-red-400">{item.reason}</p>
                    ))}
                    {family.pin_warnings?.filter((item) => item.slot === slot.key).map((item) => (
                      <p key={item.slot} className="mt-1 text-xs text-amber-400">{item.message}</p>
                    ))}
                    {slot.configurable !== false && values[slot.key] && <button type="button"
                      onClick={() => setField('studio_models', family.family, { ...values, [slot.key]: '' })}
                      className="min-h-10 text-xs text-primary underline">Use automatic detection</button>}
                    {missing.has(slot.action) && <div className="mt-2 space-y-1">
                      <InstallRunner action={slot.action} buttonLabel={`Install ${slot.label}`} onDone={refresh} />
                      <a href={slot.source_url} target="_blank" rel="noreferrer"
                        className="text-xs text-primary underline">Model details and license</a>
                    </div>}
                  </div>
                ))}
              </div>
            </details>
          )
        })}
      </div>
    </Card>
  )
}
