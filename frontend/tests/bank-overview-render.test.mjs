import assert from 'node:assert/strict'
import test from 'node:test'

import { render } from './support/mountJsx.mjs'

// The JSX loader is registered while mountJsx is evaluated, so this import has
// to stay dynamic (see support/mountJsx.mjs).
const { default: BankOverview } =
  await import('../src/components/bank/BankOverview.jsx')

test('the loaded Bank overview renders expanded with an accessible controlled region', () => {
  const html = render(BankOverview, {
    payload: {
      counts: { total: 3, keep: 1, pending: 1, reject: 1, scanned: 3 },
      res_buckets: { res_1_2: 3 },
    },
  })

  assert.match(html, /<button[^>]*aria-expanded="true"[^>]*aria-controls="([^"]+)"/)
  const controls = /aria-controls="([^"]+)"/.exec(html)?.[1]
  assert.ok(controls)
  assert.ok(html.includes(`id="${controls}"`), 'aria-controls points at the details region')
  assert.match(html, />3 images</)
  assert.match(html, /Pass coverage/)
  assert.match(html, /Structure/)
})

test('the unavailable Bank overview keeps the same usable header and total contract', () => {
  const html = render(BankOverview, { payload: null })

  assert.match(html, /<button[^>]*aria-expanded="true"[^>]*aria-controls=/)
  assert.match(html, /Bank overview/)
  assert.match(html, /Total unavailable/)
  assert.match(html, /role="status"/)
  assert.match(html, /Overview unavailable — waiting for bank data/)
  assert.doesNotMatch(html, /Pass coverage/)
})
