import assert from 'node:assert/strict'
import test from 'node:test'

import { createElement, render } from './support/mountJsx.mjs'

// The JSX loader is registered while mountJsx is evaluated, so these imports
// stay dynamic (see support/mountJsx.mjs).
const { PatreonAccessActions } = await import('../src/components/patreon/PatreonAccess.jsx')
const { default: PatreonUpdates } = await import('../src/components/patreon/PatreonUpdates.jsx')
const { InstallPlan } = await import('../src/pages/store/Catalog.jsx')
const { default: PluginCard } = await import('../src/pages/store/PluginCard.jsx')
const { default: SupportProjectNotice } = await import('../src/components/common/SupportProjectNotice.jsx')
const { MemoryRouter } = await import('react-router')

const JOIN = 'https://www.patreon.com/c/example'
const UNLINKED = { status: 'not_connected', patreon: { linked: false, status: 'inactive', tier: null,
  valid_until: null, join_url: JOIN, request_url: null } }
const PRO = { status: 'ready', patreon: { linked: true, status: 'active', tier: 'pro', valid_until: '2026-11-04T00:00:00Z',
  join_url: JOIN, request_url: 'https://requests.example.test/form' } }
const state = (access, extra = {}) => ({ access, busy: false, error: '', authorizeUrl: '',
  load() {}, connect() {}, disconnect() {}, ...extra })

test('membership actions render in every state', () => {
  const unlinked = render(PatreonAccessActions, { state: state(UNLINKED) })
  assert.match(unlinked, />Connect Patreon</)
  assert.match(unlinked, /Join on Patreon/)
  assert.match(unlinked, /Just joined\?/)
  const pro = render(PatreonAccessActions, { state: state(PRO) })
  assert.match(pro, /Request a custom plugin/)
  assert.match(pro, />Disconnect</)
  assert.doesNotMatch(pro, /Connect Patreon|Join on Patreon/)
  const lapsed = render(PatreonAccessActions, { state: state({ status: 'ready',
    patreon: { ...UNLINKED.patreon, linked: true } }) })
  assert.match(lapsed, /Connect Patreon again/)
  const pending = render(PatreonAccessActions, {
    state: state(UNLINKED, { authorizeUrl: 'https://www.patreon.com/oauth2/authorize?state=x' }) })
  assert.match(pending, /href="https:\/\/www\.patreon\.com\/oauth2\/authorize\?state=x"[^>]*>Continue on Patreon/)
})

test('the code of the Patreon page is asked after Continue on Patreon and while an approval waits', () => {
  const field = /Enter the code shown on the Patreon page[\s\S]*<input[^>]*>[\s\S]*>Confirm</
  assert.doesNotMatch(render(PatreonAccessActions, { state: state(UNLINKED) }), /Enter the code shown/)
  assert.match(render(PatreonAccessActions, {
    state: state(UNLINKED, { authorizeUrl: 'https://www.patreon.com/oauth2/authorize?state=x' }) }), field)
  // Back later, or after a reload: the approval still waits for its code.
  const waiting = { ...UNLINKED, patreon: { ...UNLINKED.patreon, pending: true } }
  assert.match(render(PatreonAccessActions, { state: state(waiting) }), field)
  const refused = render(PatreonAccessActions, { state: state(waiting, { error: 'This code has expired.' }) })
  assert.match(refused, /role="alert"[^>]*>This code has expired\./)
  // The Updates block shares these actions: the field appears there too.
  const updates = render(PatreonUpdates, { patreon: { status: 'ready', update_available: true, install_mode: 'git',
    release: { version: '2026.10.20' }, can_apply: false, entitled: false, access: waiting } })
  assert.match(updates, field)
})

test('without administration the membership still shows, and says where to connect', () => {
  const remote = render(PatreonAccessActions, { state: state({ ...UNLINKED, can_manage: false }) })
  assert.match(remote, /<button[^>]*disabled=""[^>]*>Connect Patreon</)
  assert.match(remote, /needs local administration/)
  assert.doesNotMatch(render(PatreonAccessActions, { state: state(UNLINKED) }), /needs local administration/)
  const unread = render(PatreonUpdates, { patreon: { status: 'unavailable', message: 'Patreon is offline.' } })
  assert.match(unread, /Patreon is offline\./)
})

test('the Patreon block of the Updates card renders each channel state', () => {
  const ready = { status: 'ready', update_available: true, install_mode: 'git',
    release: { version: '2026.10.20', notes: ['Members feature'] } }
  assert.match(render(PatreonUpdates, { patreon: { ...ready, can_apply: true, entitled: true } }), /Update &amp; restart/)
  assert.match(render(PatreonUpdates, { patreon: { ...ready, can_apply: true, entitled: true, install_mode: 'pinokio' } }),
    />Install update</)
  const join = render(PatreonUpdates, { patreon: { ...ready, can_apply: false, entitled: false, access: UNLINKED } })
  assert.match(join, /available to Patreon members/)
  assert.match(join, /Members feature/)
  assert.match(join, /Connect Patreon/)
  assert.match(render(PatreonUpdates, { patreon: { ...ready, install_mode: 'docker', blocked_reason: 'Docker limit.' } }),
    /Docker limit\./)
  assert.match(render(PatreonUpdates, { patreon: null, installed: { version: '2026.10.20', install_mode: 'pinokio' } }),
    /Click the Update tab/)
  assert.match(render(PatreonUpdates, { patreon: { status: 'not_configured' } }), /reach Patreon members \(LDS Member\) first, then everyone/)
})

test('the Docker host command names its scope and asks to stop running work first', () => {
  const html = render(PatreonUpdates, { patreon: { status: 'ready', update_available: true, install_mode: 'docker',
    release: { version: '2026.10.20' }, can_apply: false, entitled: true, host_command: 'update-docker.bat patreon' } })
  assert.match(html, /update-docker\.bat patreon/)
  assert.match(html, /installation from the Windows ZIP only/)
  assert.match(html, /Stop running work first/)
  // No promise that the host updater waits for running work: it cannot see it.
  assert.doesNotMatch(html, /nothing is applied while/i)
})

test('an install review asks for a Patreon membership, not a purchase', () => {
  const plan = { plan_id: 'x'.repeat(64), requested: 'manga', update_only: false,
    packages: [{ manifest: { id: 'manga', name: 'Manga Studio', version: '1.2.0' }, action: 'install' }],
    purchase_required: ['manga'], patreon_required: ['manga'], paid_packages: ['manga'], patreon_access: UNLINKED }
  const html = render(InstallPlan, { plan, busy: false, onConfirm() {}, onCancel() {}, onAcquire() {}, onRecheck() {} })
  assert.match(html, /Manga Studio is available to Patreon members/)
  assert.match(html, /Connect Patreon/)
  assert.match(html, /Check again/)
  assert.doesNotMatch(html, /Open purchases/)
})

test('Update all goes ahead without the members-only plugins it had to leave out', () => {
  const plan = { plan_id: 'x'.repeat(64), requested: ['tool'], update_only: true,
    packages: [{ manifest: { id: 'tool', name: 'Tool', version: '2.0.0' }, action: 'install', previous_version: '1.0.0' }],
    purchase_required: [], patreon_required: [], paid_packages: [], patreon_access: null,
    excluded: [{ id: 'manga', name: 'Manga Studio' }] }
  const html = render(InstallPlan, { plan, busy: false, onConfirm() {}, onCancel() {}, onAcquire() {}, onRecheck() {} })
  assert.match(html, /Not included: Manga Studio/)
  assert.match(html, /active Patreon membership/)
  assert.match(html, /<button[^>]*>Update all<\/button>/)
  assert.doesNotMatch(html, /<button[^>]*disabled=""[^>]*>Update all/)
})

test('a Patreon product is labelled and promises no screenshots', () => {
  const release = { manifest: { id: 'manga', name: 'Manga Studio', version: '1.2.0', description: 'Comics.' },
    price: { kind: 'paid', product: 'patreon_lds', provider: 'patreon' }, changelog: [],
    presentation: { schema_version: 1, images: [] }, compatibility_issues: [] }
  const html = render(PluginCard, { plugin: null, release, storeActions: null })
  assert.match(html, /For Patreon members/)
  assert.doesNotMatch(html, /Screenshots coming soon/)
})

test('a plugin card lists its membership features: what members have, or would get', () => {
  const release = { manifest: { id: 'cloud_training', name: 'Cloud training', version: '1.0.7', description: 'Train.' },
    price: { kind: 'free' }, changelog: [], presentation: { schema_version: 1, images: [] }, compatibility_issues: [] }
  const reserved = [{ version: '1.0.9', changelog: ['Train MOSS-Image LoRAs in the cloud.'], available: false }]
  const free = render(PluginCard, { plugin: null, release, membersReleases: reserved, patreonAccess: UNLINKED })
  assert.match(free, /Membership features/)
  assert.match(free, /Version 1\.0\.9 is available to Patreon members \(LDS Member\) first/)
  assert.match(free, /Train MOSS-Image LoRAs in the cloud\./)
  assert.match(free, />Connect Patreon</)
  assert.match(free, /Join on Patreon/)
  const members = render(PluginCard, { plugin: null, release, patreonAccess: PRO,
    membersReleases: [{ ...reserved[0], available: true }] })
  assert.match(members, /Version 1\.0\.9 is included with your members version of LDS\./)
  assert.doesNotMatch(members, /Connect Patreon|Join on Patreon/)
  const library = { ...release, manifest: { ...release.manifest, id: 'local.creative_library', name: 'Creative Library' },
    price: { kind: 'paid', product: 'patreon_lds', provider: 'patreon' }, changelog: ['Batch boxes.'],
    compatibility_issues: [{ field: 'compatibility.api', message: 'This plugin requires the plugin API >=1.26.' }] }
  const locked = render(PluginCard, { plugin: null, release: library, patreonAccess: UNLINKED })
  assert.match(locked, /This plugin is available to Patreon members \(LDS Member\), with the members version of LDS\./)
  assert.match(locked, /Batch boxes\./)
  assert.match(locked, /Join on Patreon/)
  const member = render(PluginCard, { plugin: null, release: library, patreonAccess: PRO })
  assert.match(member, /Included with your Patreon membership\./)
  assert.match(member, /Install the members version of LDS from Settings › Maintenance › Updates\./)
  assert.doesNotMatch(member, /Join on Patreon/)
  // Nothing reserved, or the channel unread: no block at all.
  assert.doesNotMatch(render(PluginCard, { plugin: null, release, membersReleases: [] }), /Membership features/)
})

test('the announcement names the LDS Member tier and no price', () => {
  const html = render(() => createElement(MemoryRouter, null, createElement(SupportProjectNotice)))
  assert.match(html, /New LDS versions reach Patreon members first/)
  assert.match(html, /Patreon members \(LDS Member\) first, then everyone/)
  assert.doesNotMatch(html, /LDS Pro|custom plugin/i)
  assert.match(html, /href="\/plugins\?tab=purchases"/)
  assert.doesNotMatch(html, /[$€£]/)
})
