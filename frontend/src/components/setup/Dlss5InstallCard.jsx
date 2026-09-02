import { useEffect, useRef, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import { HelpBadge } from '../../help/HelpMode'

const POLL_MS = 1200
const ACTION = 'dlss5nr_bridge'

// ✨ DLSS 5 Neural Rendering — NVIDIA's model over finished video clips.
//
// TWO HALVES, AND ONLY ONE OF THEM IS A BUTTON
// --------------------------------------------
// The BRIDGE is two small MIT-licensed DLLs from an open-source project; this
// app downloads them from a release pinned by size and SHA-256 (a re-uploaded
// asset is refused, never trusted). The MODEL — nvngx_dlssnr.dll — is NVIDIA's.
// It is not downloaded, not linked, not looked for anywhere but the folder
// named below: the user brings their own copy, and the card says so in the
// same breath as the button, because a card with a button and no sentence
// about the model would leave someone with a working bridge and nothing to run.
//
// A capability gated by the OPERATING SYSTEM, the first in this app: the model
// is a Direct3D 12 library. On Linux, in Docker or on a non-NVIDIA card the
// card does not hide — it says why, so a ✗ never reads as "something to fix".
export default function Dlss5InstallCard({ caps, onDone }) {
  const toast = useToast()
  const st = caps?.dlss5nr || null
  const [phase, setPhase] = useState('idle')
  const [log, setLog] = useState([])
  const [copied, setCopied] = useState(false)
  const timer = useRef(null)
  const mounted = useRef(true)

  const poll = () => {
    apiFetch(`/api/setup/install/${ACTION}/status`).then((r) => {
      if (!mounted.current) return
      setLog(r.log || [])
      if (r.state === 'success' || r.state === 'error' || r.state === 'cancelled') {
        setPhase('done')
        onDone?.()
        if (r.state === 'success') toast.success('Neural rendering bridge installed.')
        else toast.warning('The bridge install did not finish — see the log below.')
      } else {
        timer.current = setTimeout(poll, POLL_MS)
      }
    }).catch(() => {
      if (mounted.current) timer.current = setTimeout(poll, POLL_MS)
    })
  }

  useEffect(() => {
    mounted.current = true
    apiFetch(`/api/setup/install/${ACTION}/status`).then((r) => {
      if (!mounted.current) return
      if (r.state === 'running' || r.state === 'queued') { setPhase('running'); setLog(r.log || []); poll() }
    }).catch(() => { /* not attached — stay idle */ })
    return () => { mounted.current = false; clearTimeout(timer.current) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const start = async () => {
    setPhase('running'); setLog([])
    try {
      await postJson(`/api/setup/install/${ACTION}`, {})
      poll()
    } catch (e) {
      setPhase('idle')
      toast.error(e.message || 'Could not start the bridge download.')
    }
  }

  const copyPath = async () => {
    try {
      await navigator.clipboard.writeText(st?.runtime_dir || '')
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      toast.warning('Could not copy — the path is written below.')
    }
  }

  if (!st) return null
  const unfixable = !st.os_ok || !st.driver_ngx
  const Row = ({ ok, children }) => (
    <li className="flex items-start gap-2 text-sm">
      <span aria-hidden="true" className={ok ? 'text-emerald-400' : 'text-content-subtle'}>{ok ? '✓' : '○'}</span>
      <span className={ok ? 'text-content' : 'text-content-muted'}>{children}</span>
    </li>
  )

  return (
    <section className="rounded-xl border border-border bg-surface p-4 sm:p-5">
      <div className="flex flex-wrap items-start justify-between gap-x-3 gap-y-1">
        <h3 className="text-base font-semibold text-content">
          DLSS 5 Neural Rendering — optional, video clips
          <HelpBadge topic="setup-dlss5-install" className="ml-2" />
        </h3>
        {st.ready && <span className="text-xs font-semibold text-emerald-400">ready</span>}
      </div>
      <p className="mt-1 text-sm text-content-muted">
        NVIDIA&apos;s DLSS 5 model re-renders a finished clip&apos;s materials and lighting.
        Windows and an NVIDIA GPU only; the two bridge DLLs are installed here, the model file is yours to place.
      </p>

      <ul className="mt-3 flex flex-col gap-1">
        <Row ok={st.os_ok}>Windows {st.os_ok ? '' : '— the model is a Direct3D 12 library and runs nowhere else (no Linux, no Docker)'}</Row>
        <Row ok={st.driver_ngx}>NVIDIA display driver {st.driver_ngx ? '' : '— its NGX runtime was not found on this machine'}</Row>
        <Row ok={st.worker}>Video extra in the app&apos;s interpreter (numpy for the render process){st.worker ? '' : ' — install it from the list below'}</Row>
        <Row ok={st.bridge}>Neural rendering bridge v{st.bridge_version} (MIT, from {st.bridge_url?.replace('https://', '')})</Row>
        <Row ok={st.model}>
          Your <code className="rounded bg-surface-raised px-1 font-mono text-xs">{st.model_file}</code>
          {st.model ? '' : ' — not found in the folder below'}
        </Row>
      </ul>

      {!unfixable && !st.bridge && (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <button type="button" onClick={start} disabled={phase === 'running'}
            className="min-h-10 rounded-md border border-border-strong bg-surface-raised px-3 py-1.5 text-sm font-semibold text-content hover:bg-surface disabled:opacity-50 lg:min-h-0">
            {phase === 'running' ? 'Downloading…' : 'Install the bridge (≈ 0.2 MB)'}
          </button>
        </div>
      )}

      {!unfixable && (
        <div className="mt-3 rounded-lg border border-border bg-surface-raised p-3 text-sm text-content-muted">
          <p>
            Place your copy of <code className="font-mono text-xs text-content">{st.model_file}</code> in this folder, then reopen this screen:
          </p>
          <div className="mt-1 flex flex-wrap items-center gap-2">
            <code className="max-w-full break-all font-mono text-xs text-content">{st.runtime_dir}</code>
            <button type="button" onClick={copyPath} aria-label="Copy the folder path"
              className="min-h-10 rounded border border-border px-2 py-0.5 text-xs text-content-muted hover:text-content lg:min-h-0">
              {copied ? 'Copied' : 'Copy path'}
            </button>
          </div>
          <p className="mt-2 text-xs text-content-subtle">
            This app does not download the model and offers no link to it: it is NVIDIA&apos;s, shipped for the RTX 50 series.
            A real one is about 165 MB — a file of the same name under 1 MB is a forwarder from a game mod, not the model.
            The model decides on which GPU it runs; a refusal is shown, in its own words, on the first clip you render.
          </p>
        </div>
      )}

      {log.length > 0 && (
        <pre className="mt-3 max-h-40 overflow-auto rounded-md border border-border bg-surface-raised p-2 font-mono text-[0.6875rem] text-content-muted">
          {log.slice(-20).join('\n')}
        </pre>
      )}
    </section>
  )
}
