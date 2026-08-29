import { useEffect, useRef, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'

const ACTION = 'krea_sampler_nodes'
const POLL_MS = 1200

/* The optional preset sampler, offered inside the Krea card but deliberately NOT
   part of its "Install everything".
 *
 * WHY IT IS NOT IN THE GROUP
 * --------------------------
 * `_INSTALL_GROUPS['krea']` is the answer to "what does this engine need?", and
 * the honest answer does not include this. Krea renders perfectly without it; the
 * preset sampler is a different way to sample, not a missing piece. Listing it
 * with the weights would tell every user they are missing something they are not,
 * and would put a red ✗ on an engine that works.
 *
 * WHY IT STILL NEEDS A BUTTON
 * ---------------------------
 * Because the alternative is a dropdown entry that fails at run time with no way
 * to repair it — a capability the app probes and cannot install is a dead end by
 * construction. This is that button, three lines from the option it enables.
 *
 * Unlike every other install on this screen there is no download: the node ships
 * inside the app, and installing it means copying it into the user's ComfyUI. So
 * the only states worth distinguishing are copied / not copied / copied-but-
 * ComfyUI-hasn't-restarted, which is what the three branches below are. */
export default function KreaSamplerInstallRow({ caps, onDone }) {
  const toast = useToast()
  const [state, setState] = useState(null)
  const timer = useRef(null)
  const mounted = useRef(true)

  const cu = caps?.comfyui || {}
  const installed = !!cu.krea_sampler_nodes_installed
  const missing = (cu.krea_sampler_nodes_missing || []).length > 0

  useEffect(() => {
    mounted.current = true
    return () => { mounted.current = false; clearTimeout(timer.current) }
  }, [])

  const poll = () => {
    apiFetch(`/api/setup/install/${ACTION}/status`).then((s) => {
      if (!mounted.current) return
      setState(s)
      if (s.state === 'success' || s.state === 'error') {
        onDone?.()
        if (s.state === 'success') {
          toast.success('Preset sampler installed. Restart ComfyUI to load it.')
        } else {
          // The log's last line is the reason (a foreign folder, a read-only
          // ComfyUI). Showing the bare word "error" would send someone hunting.
          toast.error((s.log || []).slice(-1)[0] || 'The preset sampler could not be installed.')
        }
      } else {
        timer.current = setTimeout(poll, POLL_MS)
      }
    }).catch(() => {
      if (mounted.current) timer.current = setTimeout(poll, POLL_MS)
    })
  }

  const start = async () => {
    setState({ state: 'running' })
    try {
      setState(await postJson(`/api/setup/install/${ACTION}`, {}))
      poll()
    } catch (e) {
      setState(null)
      toast.error(e.message || 'Could not start the install.')
    }
  }

  const running = state?.state === 'running' || state?.state === 'queued'
  const justInstalled = state?.state === 'success'

  return (
    <div className="mt-4 border-t border-border pt-3">
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
        <span className="text-sm font-medium text-content">Preset sampler — optional</span>
        {(installed || justInstalled) ? (
          <span className={missing && !justInstalled ? 'text-xs text-amber-400' : 'text-xs text-emerald-400'}>
            {missing || justInstalled ? '✓ installed — restart ComfyUI to load it' : '✓ installed'}
          </span>
        ) : (
          <button
            type="button"
            onClick={start}
            disabled={running}
            className="rounded-md border border-border px-3 py-1 text-xs font-medium text-content hover:border-primary disabled:opacity-50"
          >
            {running ? 'Installing…' : 'Install'}
          </button>
        )}
      </div>
      <p className="mt-1 text-xs text-content-subtle">
        Adds a second way to sample Krea renders, built for its 8-step turbo setting:
        a scheduled multistep correction plus explicit control of the final step. It
        appears in the Studio&rsquo;s Sampler menu once installed. Krea works without it —
        the <span className="whitespace-nowrap">&ldquo;neutral&rdquo;</span> preset is
        plain Euler, there as the reference to compare the others against.
      </p>
    </div>
  )
}
