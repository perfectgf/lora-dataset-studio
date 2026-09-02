/**
 * ☁ The video cloud launch window, RENDERED — in the states that spend money.
 *
 * `renderToStaticMarkup` runs no effects, so the offers fetch never fires: the
 * dialog renders its LOADING state, which is exactly the first thing a user sees
 * and the one an undefined identifier would throw in. The tier list itself is
 * exercised through the shared CloudTierEstimate, mounted directly with the
 * three shapes the server really answers with (a rough estimate, no estimate,
 * over the cap).
 *
 * ⚠️ Mounts components: needs frontend/node_modules (react + esbuild).
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import { createElement, renderToStaticMarkup } from './support/mountJsx.mjs'

const { MemoryRouter } = await import('react-router')
const { default: VideoCloudLaunchDialog } =
  await import('../src/components/videobank/VideoCloudLaunchDialog.jsx')
const { default: CloudTierEstimate } =
  await import('../src/components/shared/CloudTierEstimate.jsx')

const DS = { id: 9, target_label: 'Wan 2.2 T2V A14B', frames: 81, clips: 6 }
const render = (Component, props) => renderToStaticMarkup(
  createElement(MemoryRouter, null, createElement(Component, props)))

test('the launch window opens on its loading state, as a layer, with the footer already honest', () => {
  const html = render(VideoCloudLaunchDialog, {
    ds: DS, steps: 2000, cloudStatus: { monthly_budget: 50, month_spend: 3.2 },
    onClose: () => {}, onLaunch: async () => true,
  })
  assert.match(html, /role="dialog"/)
  assert.match(html, /data-probe-chrome="cloud-launch" data-probe-layer/,
    'the window covers the page by design — a layer, never budgeted chrome')
  assert.match(html, /Loading live GPU offers/)
  // The money line is on screen from the first frame, before any tier loads.
  assert.match(html, /2000 steps · Wan 2\.2 T2V A14B · 81 frames · 6 clips · this month: \$3\.20 of \$50\.00/)
  assert.match(html, /Time &amp; cost are rough/)
  // Nothing to launch yet: the button is there and disabled.
  assert.match(html, /<button[^>]*disabled[^>]*>☁️ Rent &amp; train<\/button>/)
})

test('the shared estimate line says the three things the server can answer', () => {
  const rough = renderToStaticMarkup(createElement(CloudTierEstimate, {
    tier: { gpu_name: 'RTX 4090', dph_total: 0.42, est_minutes: 95, est_cost: 0.71,
      estimate_status: 'rough', exceeds_cap: false },
    maxRuntimeMinutes: 480,
  }))
  assert.match(rough, /\$0\.420\/h · ~1 h 35 min · ≈ \$0\.71 total/)
  assert.doesNotMatch(rough, /runtime cap/)

  const none = renderToStaticMarkup(createElement(CloudTierEstimate, {
    tier: { gpu_name: 'A100', dph_total: 1.1, est_minutes: null, est_cost: null,
      estimate_status: 'unavailable', exceeds_cap: null },
  }))
  assert.match(none, /\$1\.100\/h/)
  assert.match(none, /duration and cost unavailable/, 'no number is invented for an off-grid frame count')

  const over = renderToStaticMarkup(createElement(CloudTierEstimate, {
    tier: { gpu_name: 'RTX 3090', dph_total: 0.2, est_minutes: 600, est_cost: 2.0,
      estimate_status: 'rough', exceeds_cap: true },
    maxRuntimeMinutes: 480,
  }))
  assert.match(over, /~10 h · ≈ \$2\.00 total/)
  assert.match(over, /Longer than the 8 h runtime cap/, 'the cap warning is said BEFORE the click')
  assert.match(over, /saved LoRA checkpoints are rescued/)
})
