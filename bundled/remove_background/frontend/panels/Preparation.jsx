import { useCallback, useEffect, useState } from 'react'
import { HelpBadge } from '@lds/plugin-sdk'
import { Card, InstallRunner, SettingsLink } from '@lds/plugin-sdk/ui'
import { readStatus } from '../lib/api.js'

/** Settings and Setup: what is ready, and the optional heavy matting model. */
export default function Preparation({ caps = {}, onDone }) {
  const [status, setStatus] = useState(null)
  const [error, setError] = useState('')
  const refresh = useCallback(() => readStatus().then(data => { setStatus(data); setError('') })
    .catch(problem => setError(problem.message || 'The models could not be checked.')), [])
  useEffect(() => { refresh() }, [refresh, caps])
  const done = () => { refresh(); onDone?.() }
  const absent = slot => status && status.models[slot].state !== 'ready'
  const nodes = [...new Set(Object.values(status?.kinds || {}).flatMap(item => item.missing_nodes).concat(status?.find.missing_nodes || []))]
  return <Card id="remove-background-preparation" title="Prepare Remove Background"
    help="Every model runs in your own ComfyUI on its native nodes. Files that are already there are reused.">
    <p role="status" className="m-0 text-sm text-content-muted">
      {error || caps.remove_background?.detail || 'Checking the models…'}<HelpBadge topic="remove-background-setup" className="ml-1" />
    </p>
    {!caps.comfyui?.dir_valid && <p className="m-0 text-sm"><SettingsLink section="local-tools" focus="comfyui-base-dir" tone="warning">Connect your local ComfyUI folder</SettingsLink></p>}
    {nodes.length > 0 && <p className="m-0 break-words text-sm text-amber-200">Update ComfyUI with its normal updater and restart it when idle. Missing native nodes: {nodes.join(', ')}.</p>}
    <div className="space-y-2">
      <h4 className="m-0 text-sm font-semibold text-content">RMBG 3.0, the finest edges (optional)</h4>
      <p className="m-0 text-xs text-content-subtle">Keeps single strands of hair and fur. Its weights are for non-commercial use only, unless you have a licence from Bria.{' '}
        <a href="https://huggingface.co/briaai/RMBG-3.0-Comfy" target="_blank" rel="noreferrer" className="text-primary underline">CC BY-NC 4.0</a></p>
      {absent('rmbg') && <InstallRunner action="remove_bg_rmbg" buttonLabel="Install RMBG 3.0 (8.41 GB)" onDone={done} />}
      {absent('rmbg_text_encoder') && <InstallRunner action="remove_bg_rmbg_text_encoder" buttonLabel="Install its text encoder (4.86 GB)" onDone={done} />}
      {absent('rmbg_vae') && <InstallRunner action="remove_bg_rmbg_vae" buttonLabel="Install its VAE (336 MB)" onDone={done} />}
      {status && !absent('rmbg') && !absent('rmbg_text_encoder') && !absent('rmbg_vae') && <p className="m-0 text-sm text-content-muted">Installed.</p>}
    </div>
  </Card>
}
