import { versionLabel } from '../../utils/versionLabel.js'

/* Pure helpers for the mode-aware Updates card. Kept out of the component so the
   mode/label/progress logic is unit-testable without a DOM (see updateStatus.test.js).

   Install modes:
   - 'docker'      a container image: the host checkout must be pulled and the
                   image rebuilt; files inside the running container are never
                   replaced by the in-app updater.
   - 'pinokio'     launched by the Pinokio launcher: the tree is a git checkout
                   and updates the same way, but the RESTART belongs to Pinokio.
                   Updating here would relaunch the server detached from the
                   launcher, which would then show the app as stopped while an
                   untracked one held the port.
   - 'git'         a git checkout: "Update & restart" fast-forwards (unchanged).
   - 'zip'         a packaged install whose latest release ships a ZIP asset: the
                   button downloads + swaps the release, with a progress bar.
   - 'unavailable' non-git and no downloadable release: don't promise an update
                   the app can't perform — link out to the releases page instead. */

// Fallback for older GPU servers. New servers supply their lane's commands.
export const DOCKER_UPDATE_COMMANDS = Object.freeze([
  'git pull',
  'docker compose -f docker-compose.gpu.yml up -d --build',
])

export const DOCKER_UPDATE_GUIDE_URL =
  'https://github.com/perfectgf/lora-dataset-studio/blob/v2/docs/guide/docker.md#updates-and-restarts'

export function dockerUpdateCommands(status) {
  const commands = status?.instructions
  return Array.isArray(commands) && commands.length > 0
    && commands.every((command) => typeof command === 'string' && command.trim())
    ? commands : DOCKER_UPDATE_COMMANDS
}

export function isDockerInstall(s) {
  return s?.install_mode === 'docker'
}

// Pinokio's own three clicks. Not shell commands — the user never opens a
// terminal in this install shape, so naming `git pull` here would send them
// looking for one. Pinokio's Update tab runs the same `git pull --ff-only`.
export const PINOKIO_UPDATE_STEPS = Object.freeze([
  'Stop the app in Pinokio',
  'Click the Update tab',
  'Click Start',
])

export const PINOKIO_UPDATE_GUIDE_URL =
  'https://github.com/perfectgf/lora-dataset-studio#option-5--pinokio-one-click-any-os'

export function isPinokioInstall(s) {
  return s?.install_mode === 'pinokio'
}

export function formatMB(bytes) {
  if (!bytes || bytes <= 0) return ''
  const mb = bytes / 1e6
  return `${mb >= 100 ? Math.round(mb) : mb.toFixed(1)} MB`
}

export function installMode(s) {
  if (!s) return 'unknown'
  // Docker wins even if a defensive/future payload accidentally also reports a
  // writable git checkout or ZIP asset. Never surface the in-app apply action in
  // a container just because /app happens to contain .git metadata.
  if (isDockerInstall(s)) return 'docker'
  // Before the is_git branch ON PURPOSE: a Pinokio install IS a git checkout,
  // and answering 'git' would put back the button that strands the launcher.
  if (isPinokioInstall(s)) return 'pinokio'
  if (s.ok === false) return 'unknown'
  if (s.is_git) return 'git'
  if (s.can_apply) return 'zip'
  return 'unavailable'
}

/* A ZIP or Docker installation on the Patreon channel (GET /api/update/check
   reports channel 'patreon'): a newer public release is not offered, since it
   would replace the Patreon version. Say it exists and name the explicit way
   back: a button for a ZIP install (can_force), the host command for Docker. */
export function publicReleaseNotice(s) {
  if (!s || s.channel !== 'patreon' || !s.public_update_available) return null
  const release = s.latest ? `Public release v${s.latest}` : 'A newer public release'
  return {
    text: `${release} is also available. This installation runs a Patreon release; installing the `
      + 'public release replaces it and removes the Patreon changes.',
    canForce: Boolean(s.can_force) && !isDockerInstall(s),
    command: isDockerInstall(s) ? 'update-docker.bat stable force' : null,
  }
}

/* Headline for a ZIP-mode update, e.g. "Update to v2026.07.19 (download ~42 MB)".
   The size hint is omitted when the release didn't report an asset size. */
export function zipUpdateHeadline(s) {
  const v = s && s.latest ? `Update to v${s.latest}` : 'Update available'
  const size = s && s.zip_size ? formatMB(s.zip_size) : ''
  return size ? `${v} (download ~${size})` : v
}

/* Percent complete for the download phase, or null when the total is unknown
   (server sent no Content-Length) — the UI then shows an indeterminate bar. */
export function progressPercent(p) {
  if (!p || !p.total || p.total <= 0) return null
  return Math.max(0, Math.min(100, Math.round((p.downloaded || 0) / p.total * 100)))
}

/* A failed read of GET /api/update/patreon, as a status the block can show:
   the block says so instead of disappearing. */
export function patreonCheckFailure(error) {
  const message = typeof error?.body?.error === 'string' ? error.body.error : ''
  return { status: 'unavailable',
    message: message || 'The Patreon update channel could not be checked. Check again in a moment.' }
}

export function membersUpdateAvailable(p) {
  return p?.status === 'ready' && !!(p.entitled || p.can_apply) && !!p.update_available
}

/** The public check alone cannot establish whether a members installation is current. */
export function updateCheckFeedback(s, p) {
  if (p?.status === 'ready' && (p.restart_required || p.update_available)) {
    return { type: membersUpdateAvailable(p) && !p.restart_required ? 'success' : 'info',
      message: patreonUpdateView(p).text }
  }
  if (s?.update_available) {
    return { type: 'success', message: `Update available — v${s.latest || s.remote_sha || 'new'}` }
  }
  if (!s?.ok) return { type: 'error', message: s?.reason || 'Could not check for updates.' }
  if (!p) return { type: 'info', message: 'Checking Patreon updates…' }
  if (p.status !== 'ready' && (p.status !== 'not_configured' || s.channel === 'patreon')) {
    return { type: 'warning', message: p.message || 'The Patreon update channel could not be checked. Try again.' }
  }
  return { type: 'info', message: `You're up to date — ${versionLabel(s)}` }
}

export function latestUpdateLabel(s, p) {
  if (p?.status === 'ready' && p.release?.version
      && (p.update_available || p.on_channel || s?.channel === 'patreon')) {
    return `v${p.release.version} (Patreon)`
  }
  if (s?.channel === 'patreon') return p ? 'Patreon check unavailable' : 'Checking Patreon…'
  if (s?.remote_sha) {
    const behind = typeof s.behind === 'number' && s.behind > 0
      ? ` (+${s.behind} commit${s.behind === 1 ? '' : 's'})` : ''
    return `${s.remote_sha}${behind}`
  }
  if (s?.latest) return `v${s.latest}`
  return s?.update_available ? 'update available' : s?.ok ? 'No newer public build reported' : 'Not checked'
}

export const PATREON_ADMIN_REASON = 'Install it from the computer that runs LDS: '
  + 'installing an update needs local administration.'

/* The Patreon channel block of the Updates card, from GET /api/update/patreon.
   kind: hidden | unavailable | none | current | restart | install | host | join | blocked.
   Never offers an install the server said it cannot apply (can_apply), nor one
   this browser may not request (can_manage false: Docker bridge or LAN). `host`:
   Docker, where the host updater installs the release (host_command). */
export function patreonUpdateView(p) {
  if (!p || p.status === 'not_configured') return { kind: 'hidden' }
  if (p.status !== 'ready') {
    return { kind: 'unavailable', text: p.message || 'The Patreon update channel could not be checked.' }
  }
  if (p.restart_required) {
    return { kind: 'restart', text: 'A Patreon update is installed. Restart LDS to finish it.' }
  }
  const version = p.release?.version ? `LDS v${p.release.version}` : 'A new LDS version'
  if (!p.update_available) {
    return p.on_channel
      ? { kind: 'current', text: `Up to date on the Patreon channel${p.release?.version ? ` (v${p.release.version})` : ''}.` }
      : { kind: 'none' }
  }
  if (p.blocked_reason) return { kind: 'blocked', text: `${version} is available to Patreon members first.`, reason: p.blocked_reason }
  if (p.can_apply && p.can_manage === false) {
    return { kind: 'blocked', text: `${version} is available with your Patreon membership.`, reason: PATREON_ADMIN_REASON }
  }
  if (p.can_apply) return { kind: 'install', text: `${version} is available with your Patreon membership.` }
  if (p.entitled && typeof p.host_command === 'string' && p.host_command) {
    return { kind: 'host', text: `${version} is available with your Patreon membership.`, command: p.host_command }
  }
  if (p.access?.patreon?.status === 'verification_required') {
    return { kind: 'blocked', text: `${version} is available to Patreon members first.`,
      reason: 'Patreon could not verify your membership right now. Check again in a few minutes.' }
  }
  if (!p.entitled) return { kind: 'join', text: `${version} is available to Patreon members (LDS Member) first.` }
  return { kind: 'blocked', text: `${version} is available to Patreon members first.`,
    reason: 'This installation cannot apply it right now. Check again later.' }
}

/* Human phase line for the progress area. Returns null for phases the card
   renders elsewhere (idle/done) so the caller can branch on that. */
export function progressLabel(p) {
  const phase = p && p.phase
  if (phase === 'downloading') {
    const pct = progressPercent(p)
    const dl = formatMB(p.downloaded)
    if (p.total) return `⬇ Downloading… ${pct == null ? '' : `${pct}% `}(${dl || '0 MB'} / ${formatMB(p.total)})`
    return `⬇ Downloading… ${dl || ''}`.trim()
  }
  if (phase === 'extracting') return '📦 Extracting the update…'
  if (phase === 'installing') return '🔧 Installing the new files…'
  if (phase === 'restarting') return '↻ Restarting the app…'
  return null
}
