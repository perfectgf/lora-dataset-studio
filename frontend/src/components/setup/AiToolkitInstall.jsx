import { useCallback, useEffect, useState } from 'react'
import { apiFetch } from '../../api/fetchClient'
import InstallRunner from './InstallRunner'

const PLAN_URL = '/api/setup/aitoolkit-install/plan'

const STEP_NAMES = { source: 'download', python: 'Python environment', torch: 'PyTorch',
  deps: 'requirements', check: 'GPU check', config: 'settings' }

/** One-click AI Toolkit install for the Setup training step. The backend plan is
 *  the dry run: GPU and driver verdict, CUDA build, free space, source and the
 *  exact commands. Install stays disabled until that plan says "ready". */
export default function AiToolkitInstall({ onInstalled, onUseFolder }) {
  const [typed, setTyped] = useState(null)       // null = the folder the plan proposes
  const [plan, setPlan] = useState(null)
  const [error, setError] = useState('')
  const [revision, setRevision] = useState(0)
  const [showCommands, setShowCommands] = useState(false)

  useEffect(() => {
    let live = true
    const query = typed && typed.trim() ? `?dest=${encodeURIComponent(typed.trim())}` : ''
    const timer = setTimeout(async () => {
      try {
        const next = await apiFetch(`${PLAN_URL}${query}`)
        if (live) { setPlan(next); setError('') }
      } catch (e) {
        if (live) setError(e.message || 'Could not check this machine.')
      }
    }, typed === null ? 0 : 400)
    return () => { live = false; clearTimeout(timer) }
  }, [typed, revision])

  const recheck = useCallback(() => setRevision((n) => n + 1), [])
  const folder = typed ?? plan?.dest ?? ''

  if (error) {
    return (
      <div className="rounded-md border border-border bg-surface-raised p-3 text-xs text-content-muted">
        <p role="alert">{error}</p>
        <button type="button" onClick={recheck} className="mt-2 text-primary underline">Check again</button>
      </div>
    )
  }
  if (!plan) {
    return <p className="text-xs text-content-subtle" role="status">Checking this machine for AI Toolkit…</p>
  }
  if (plan.state === 'unsupported' || plan.state === 'installed') {
    return (
      <p className="rounded-md border border-border bg-surface-raised px-3 py-2 text-xs text-content-muted">
        {plan.reason}
      </p>
    )
  }

  const ready = plan.state === 'ready'
  const short = (plan.commit || '').slice(0, 7)
  const free = plan.disk?.free_gb
  return (
    <section aria-label="Install AI Toolkit" data-help="setup.aitoolkit_install"
      className="space-y-3 rounded-md border border-primary/40 bg-primary/5 p-3 text-sm text-content">
      <div>
        <p className="font-semibold">Install AI Toolkit in one click</p>
        <p className="mt-1 text-xs leading-relaxed text-content-muted">
          LDS downloads AI Toolkit {plan.version}, creates its own Python environment with the
          CUDA build of PyTorch that AI Toolkit documents, checks that it sees your GPU and
          selects it in Settings ▸ Local tools. It needs an NVIDIA GPU and installs into a new
          folder; an AI Toolkit you already use is never changed.
        </p>
      </div>
      <label className="block space-y-1 text-xs">
        <span className="font-medium text-content">Install into</span>
        <input value={folder} onChange={(e) => setTyped(e.target.value)} spellCheck={false}
          className="min-h-10 w-full rounded-md border border-border bg-surface px-2 font-mono text-content" />
        <span className="block text-content-subtle">
          A new or empty folder, outside the LDS folder. Spaces are fine. The default sits next to
          LDS, on the drive you chose for it.
        </span>
      </label>
      <ul className="space-y-1 text-xs text-content-muted">
        <li>PyTorch {plan.torch?.torch}{plan.cuda_build ? ` for CUDA (${plan.cuda_build})` : ''}
          {plan.gpus?.length ? ` · ${plan.gpus.join(', ')}` : ''}</li>
        <li>{plan.source === 'git'
          ? `Downloaded with Git (branch main, at ${short})`
          : `Git was not found: LDS downloads a fixed copy of AI Toolkit (${short})`}</li>
        <li>{plan.python?.download
          ? `LDS first downloads its own Python ${plan.python.version}`
          : plan.python ? `Python ${plan.python.version} creates the environment` : 'No usable Python found'}</li>
        <li>Needs about {plan.disk?.need_gb} GB on {plan.disk?.drive}
          {free != null ? ` · ${free} GB free` : ''}</li>
      </ul>
      {plan.pinokio && (
        <p className="text-xs text-content-muted">
          Pinokio can also install AI Toolkit from its Discover page. To use that copy instead,
          select its folder below.
        </p>
      )}
      {plan.resume && (
        <p className="text-xs text-content-muted" role="status">
          An unfinished install is in this folder
          {plan.resume.done?.length ? ` (done: ${plan.resume.done.map((s) => STEP_NAMES[s] || s).join(', ')})` : ''}.
          Installing again continues from there.
        </p>
      )}
      {!ready && plan.reason && (
        <p role="alert" className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
          {plan.reason}
        </p>
      )}
      {plan.state === 'existing' && onUseFolder && (
        <button type="button" onClick={() => onUseFolder(plan.dest)}
          className="min-h-10 rounded-lg bg-gradient-primary px-4 py-1.5 text-xs font-semibold text-gray-950">
          Use this folder
        </button>
      )}
      <InstallRunner action="aitoolkit_install"
        buttonLabel={plan.resume ? 'Resume install' : 'Install AI Toolkit'}
        busyLabel="Installing AI Toolkit…" body={{ dest: folder }}
        disabled={!ready && plan.state !== 'running'}
        cancellable cancelLabel="Cancel install"
        cancelledNote="Install cancelled. Completed steps are kept: click Resume install to continue."
        onDone={() => { recheck(); onInstalled?.() }} />
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
        <button type="button" onClick={recheck} className="text-primary underline">Check again</button>
        <button type="button" aria-expanded={showCommands} onClick={() => setShowCommands((v) => !v)}
          className="text-primary underline">
          {showCommands ? 'Hide the commands' : 'Show the exact commands'}
        </button>
      </div>
      {showCommands && (
        <div className="space-y-2 text-xs text-content-muted">
          <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-all rounded-md border border-border bg-surface-raised p-2 text-[11px]">
            {(plan.commands || []).join('\n')}
          </pre>
          <p>{plan.update_note}</p>
        </div>
      )}
    </section>
  )
}
