/* Pure helpers for the mode-aware Updates card. Kept out of the component so the
   mode/label/progress logic is unit-testable without a DOM (see updateStatus.test.js).

   Install modes:
   - 'git'         a git checkout: "Update & restart" fast-forwards (unchanged).
   - 'zip'         a packaged install whose latest release ships a ZIP asset: the
                   button downloads + swaps the release, with a progress bar.
   - 'unavailable' non-git and no downloadable release: don't promise an update
                   the app can't perform — link out to the releases page instead. */

export function formatMB(bytes) {
  if (!bytes || bytes <= 0) return ''
  const mb = bytes / 1e6
  return `${mb >= 100 ? Math.round(mb) : mb.toFixed(1)} MB`
}

export function installMode(s) {
  if (!s || s.ok === false) return 'unknown'
  if (s.is_git) return 'git'
  if (s.can_apply) return 'zip'
  return 'unavailable'
}

/* Headline for a ZIP-mode update, e.g. "Update to v2026.07.19 (download ~42 MB)".
   The size hint is omitted when the release didn't report an asset size. */
export function zipUpdateHeadline(s, t) {
  const v = s && s.latest
    ? (t ? t('settings.maintenance.updateToVersion', { version: s.latest }) : `Update to v${s.latest}`)
    : (t ? t('settings.maintenance.updateAvailable') : 'Update available')
  const size = s && s.zip_size ? formatMB(s.zip_size) : ''
  return size
    ? (t ? t('settings.maintenance.downloadSize', { headline: v, size }) : `${v} (download ~${size})`)
    : v
}

/* Percent complete for the download phase, or null when the total is unknown
   (server sent no Content-Length) — the UI then shows an indeterminate bar. */
export function progressPercent(p) {
  if (!p || !p.total || p.total <= 0) return null
  return Math.max(0, Math.min(100, Math.round((p.downloaded || 0) / p.total * 100)))
}

/* Human phase line for the progress area. Returns null for phases the card
   renders elsewhere (idle/done) so the caller can branch on that. */
export function progressLabel(p, t) {
  const phase = p && p.phase
  if (phase === 'downloading') {
    const pct = progressPercent(p)
    const dl = formatMB(p.downloaded)
    if (p.total) {
      const progress = pct == null ? '' : `${pct}% `
      return t
        ? t('settings.maintenance.downloadingProgress', {
          progress, downloaded: dl || '0 MB', total: formatMB(p.total),
        })
        : `⬇ Downloading… ${progress}(${dl || '0 MB'} / ${formatMB(p.total)})`
    }
    return t
      ? t('settings.maintenance.downloadingUnknown', { downloaded: dl || '' }).trim()
      : `⬇ Downloading… ${dl || ''}`.trim()
  }
  if (phase === 'extracting') return t ? t('settings.maintenance.extractingUpdate') : '📦 Extracting the update…'
  if (phase === 'installing') return t ? t('settings.maintenance.installingFiles') : '🔧 Installing the new files…'
  if (phase === 'restarting') return t ? t('settings.maintenance.restartingApp') : '↻ Restarting the app…'
  return null
}
