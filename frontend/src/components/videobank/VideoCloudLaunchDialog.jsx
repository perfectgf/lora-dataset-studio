import { useEffect, useState } from 'react'
import { apiFetch } from '../../api/fetchClient'
import SettingsLink from '../common/SettingsLink'
import CloudTierEstimate from '../shared/CloudTierEstimate'
import { launchButtonLabel } from '../../utils/launchProgress'
import {
  launchFooterLine, launchStatusLine, offersEmptyMessage, videoOffersUrl,
} from './videoCloudLaunch'

/** ☁ The launch window of a video dataset's cloud run — the image lane's
 * CloudLaunchDialog, for video.
 *
 * WHAT WAS THERE BEFORE: a `<select>` of GPU names next to the button, filled
 * on demand. It rented a pod on a click with no confirmation, and the only
 * place the cost appeared was inside the option text of a closed dropdown. The
 * image lane has had the dialog for weeks: one radio per GPU class, its price
 * per hour, the ROUGH duration and total for THIS set, a warning when the run
 * would outlive the runtime cap, the month's spend against the budget, and a
 * button that says what it is doing while the reservation runs. Two surfaces
 * of one product may not differ on the screen that spends money.
 *
 * The estimate line is the SHARED component — the same rules that decide when
 * the image dialog admits it has no number decide it here.
 *
 * What is deliberately NOT ported, and why: the image dialog's custom-base
 * push gate and Hugging Face delivery checks. A video run trains from the
 * target profile's official weights, pulled by the pod itself; there is no
 * user-owned base to upload and no delivery repository to write. Rendering
 * those sections here would be furniture about a problem this lane does not
 * have.
 */
export default function VideoCloudLaunchDialog({ ds, steps, cloudStatus, onClose, onLaunch }) {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [data, setData] = useState(null)       // {tiers, steps, frames, max_price_per_hour, max_runtime_minutes}
  const [selected, setSelected] = useState(null)
  const [launching, setLaunching] = useState(false)
  // Seconds since the click — the launch POST freezes the dataset and checks
  // the target's weights, so a motionless button reads as a hang.
  const [launchElapsed, setLaunchElapsed] = useState(0)

  useEffect(() => {
    let alive = true
    setLoading(true); setError(null); setData(null); setSelected(null)
    apiFetch(videoOffersUrl(ds.id, steps))
      .then((body) => {
        if (!alive) return
        setData(body)
        if (body?.tiers?.length) setSelected(body.tiers[0].gpu_name)
      })
      .catch((e) => { if (alive) setError(e?.message || 'Could not load GPU offers.') })
      .finally(() => { if (alive) setLoading(false) })
    return () => { alive = false }
  }, [ds.id, steps])

  const go = async () => {
    if (!selected) return
    setLaunching(true)
    setLaunchElapsed(0)
    const started = Date.now()
    const tick = setInterval(() => setLaunchElapsed(Math.round((Date.now() - started) / 1000)), 1000)
    try {
      const launched = await onLaunch(selected)      // owns its own toasts
      if (launched) onClose()
    } finally {
      clearInterval(tick)
      setLaunching(false)
    }
  }

  const tiers = data?.tiers || []

  return (
    <div role="dialog" aria-modal="true" aria-label="Choose cloud GPU speed"
      data-probe-chrome="cloud-launch" data-probe-layer
      className="fixed inset-0 z-50 grid place-items-center bg-black/60 p-4"
      onKeyDown={(e) => { if (e.key === 'Escape' && !launching) onClose() }}>
      <div className="flex w-full max-w-lg flex-col gap-3 rounded-xl border border-border bg-surface-overlay p-4">
        <h3 className="m-0 text-sm font-bold text-content">
          <span aria-hidden>☁️</span> Choose GPU speed for this run
        </h3>

        {loading && <p className="m-0 text-sm text-content-muted">Loading live GPU offers…</p>}
        {error && (
          <p className="m-0 text-sm text-red-300">
            ⚠ {error}{' '}
            <SettingsLink section="training" tone="warning">Cloud settings</SettingsLink>
          </p>
        )}
        {!loading && !error && tiers.length === 0 && (
          <p className="m-0 text-sm text-content-muted">
            {offersEmptyMessage(data)}{' '}
            <SettingsLink section="training" focus="cloud-max-price-per-hour">
              increase the price cap in Settings
            </SettingsLink>.
          </p>
        )}

        {tiers.length > 0 && (
          <div className="flex max-h-[50vh] flex-col gap-1.5 overflow-y-auto">
            {tiers.map((t) => (
              <label key={t.gpu_name}
                className={`flex cursor-pointer items-center gap-3 rounded-lg border px-3 py-2 transition-colors ${
                  selected === t.gpu_name
                    ? 'border-sky-400/70 bg-sky-500/10'
                    : 'border-border bg-surface hover:bg-surface-raised'}`}>
                <input type="radio" name="video-gpu-tier" className="accent-sky-400"
                  checked={selected === t.gpu_name}
                  onChange={() => setSelected(t.gpu_name)} />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm font-semibold text-content">
                    {t.gpu_name}
                    {t.gpu_ram_gb ? <span className="font-normal text-content-subtle"> · {t.gpu_ram_gb} GB</span> : null}
                  </span>
                  <CloudTierEstimate tier={t} maxRuntimeMinutes={data?.max_runtime_minutes} />
                </span>
              </label>
            ))}
          </div>
        )}

        <p className="m-0 text-[0.6875rem] text-content-subtle">
          {launchFooterLine(ds, data, steps, cloudStatus)}
        </p>

        {launching && (
          <p aria-live="polite"
            className="m-0 rounded-lg border border-sky-400/35 bg-sky-500/[0.08] px-3 py-2 text-[0.75rem] leading-relaxed text-sky-100">
            {launchStatusLine()}
          </p>
        )}

        <div className="flex items-center gap-2">
          <button type="button" onClick={go} disabled={!selected || launching}
            className="min-h-10 rounded-lg bg-gradient-primary px-3 py-1.5 text-sm font-semibold text-gray-950 disabled:opacity-40 lg:min-h-0">
            {launchButtonLabel({ launching, elapsedSeconds: launchElapsed, fullMode: false })}
          </button>
          <button type="button" onClick={onClose} disabled={launching}
            className="ml-auto min-h-10 rounded-lg px-3 py-1.5 text-sm text-content-muted hover:text-content disabled:opacity-40 lg:min-h-0">
            Cancel
          </button>
        </div>
      </div>
    </div>
  )
}
