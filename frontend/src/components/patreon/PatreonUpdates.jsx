import PinokioUpdateInstructions from '../common/PinokioUpdateInstructions'
import { patreonUpdateView } from '../settings/updateStatus'
import { PatreonAccessActions, usePatreonAccess } from './PatreonAccess'
import { PATREON_ACCESS_SUMMARY, membershipLabel } from './patreonModel'

/* Patreon channel without an active membership: say a release exists, offer
   Connect / Join, and leave the installed version alone. The plugin cards'
   membership features reuse it, with the Store's plugin-admin token. */
export function PatreonJoin({ access, onChanged, adminToken = '' }) {
  const state = usePatreonAccess(adminToken, access)
  // A refresh or a confirmed code can make the update available: re-read the channel.
  const confirm = async (code) => {
    const done = await state.confirm(code)
    if (done) onChanged?.()
    return done
  }
  return (
    <div className="space-y-2">
      <p className="text-xs text-content-muted">{state.access ? membershipLabel(state.access) : 'Checking Patreon…'}</p>
      <PatreonAccessActions state={{ ...state, confirm, load: async () => { await state.load(); onChanged?.() } }} compact />
    </div>
  )
}

function Notes({ release }) {
  return release?.notes?.length > 0 ? (
    <ul className="list-disc space-y-1 pl-5 text-content-muted">
      {release.notes.map((note, index) => <li key={index}>{note}</li>)}
    </ul>
  ) : null
}

/* The Patreon block of Settings › Maintenance › Updates. LDS updates now come
   through Patreon; the installed version keeps working whatever the
   membership, and this block only offers newer versions.

   `patreon` is GET /api/update/patreon; `installed` the apply answer of an
   update that still needs a restart (Pinokio, or a failed automatic one). */
export default function PatreonUpdates({ patreon, installed = null, error = '', applying = false, onApply, onChanged }) {
  const view = patreonUpdateView(patreon)
  let body = null
  if (applying) {
    body = null
  } else if (installed) {
    body = (
      <div className="space-y-2">
        <p className="text-content"><span aria-hidden>✓</span> LDS v{installed.version} is installed.</p>
        {installed.install_mode === 'pinokio'
          ? <PinokioUpdateInstructions />
          : <p className="text-content-muted">{installed.reason || 'Restart LDS to finish the update.'}</p>}
      </div>
    )
  } else if (view.kind === 'install') {
    body = (
      <div className="space-y-2">
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-content"><span aria-hidden>⬆</span> {view.text}</span>
          <button type="button" onClick={onApply}
            className="rounded-md bg-gradient-primary px-3 py-1.5 text-sm font-semibold text-gray-950 transition-transform hover:-translate-y-px">
            {patreon.install_mode === 'pinokio' ? 'Install update' : 'Update & restart'}
          </button>
        </div>
        <Notes release={patreon.release} />
      </div>
    )
  } else if (view.kind === 'host') {
    // Docker: the container never replaces its own image; the host updater
    // does, and it cannot see LDS's work, unlike the in-app updates.
    body = (
      <div className="space-y-2">
        <p className="text-content"><span aria-hidden>⬆</span> {view.text}</p>
        <p className="text-content-muted">
          Docker installs it from the host, for an LDS installation from the Windows ZIP only. Stop running
          work first (generation, training, captioning): the update rebuilds and restarts LDS without waiting
          for it. Then, in the LDS folder, run:
        </p>
        <code className="block overflow-x-auto whitespace-nowrap rounded bg-app/70 px-2 py-1 text-xs text-content">
          {view.command}
        </code>
        <Notes release={patreon.release} />
      </div>
    )
  } else if (view.kind === 'join') {
    body = (
      <div className="space-y-2">
        <p className="text-content"><span aria-hidden>⬆</span> {view.text} Your installed version keeps working.</p>
        <Notes release={patreon.release} />
        <PatreonJoin access={patreon.access && { ...patreon.access, can_manage: patreon.can_manage }}
          onChanged={onChanged} />
      </div>
    )
  } else if (view.kind === 'blocked') {
    body = (
      <div className="space-y-1">
        <p className="text-content"><span aria-hidden>⬆</span> {view.text}</p>
        <p className="text-content-muted">{view.reason}</p>
      </div>
    )
  } else if (['current', 'restart', 'unavailable'].includes(view.kind)) {
    body = (
      <p className={view.kind === 'current' ? 'text-emerald-400' : 'text-content-muted'}>
        {view.kind === 'current' && <span aria-hidden>✓ </span>}{view.text}
      </p>
    )
  }
  return (
    <div className="space-y-2 border-t border-border pt-3 text-sm" data-patreon-updates>
      <p className="text-content-muted">{PATREON_ACCESS_SUMMARY}</p>
      {body}
      {error && <p role="alert" className="text-amber-500">{error}</p>}
    </div>
  )
}
