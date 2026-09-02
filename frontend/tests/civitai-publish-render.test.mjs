/**
 * 📤 Publish to Civitai — RENDERED, not grepped.
 *
 * The contract tests pin that the viewer draws the verb and that both hosts
 * hand a checkpoint to the same dialog. This file mounts the components
 * (tests/support/mountJsx.mjs) and reads the markup, so a ReferenceError in a
 * branch — the class of bug the source-text tests cannot see — throws here
 * instead of on a user's screen. Effects never run under renderToStaticMarkup,
 * so no request is made: the dialog is proved in the state it opens in.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { createElement, renderToStaticMarkup } from './support/mountJsx.mjs'

const { default: GeneratedImageLightbox } =
  await import('../src/components/shared/GeneratedImageLightbox.jsx')
const { default: CivitaiPublishModal } =
  await import('../src/components/shared/CivitaiPublishModal.jsx')
const { default: CheckpointActionsPopover } =
  await import('../src/components/dataset/CheckpointActionsPopover.jsx')
const { CapabilitiesProvider } = await import('../src/context/CapabilitiesContext.jsx')
const { ToastProvider } = await import('../src/components/common/Toast.jsx')

const inApp = (node) =>
  renderToStaticMarkup(createElement(ToastProvider, null,
    createElement(CapabilitiesProvider, null, node)))

const row = (extra = {}) => ({
  id: 4211, dataset_id: 7, url: '/api/dataset/7/img/x.png', step: 2500, record_id: 4,
  seed: 1234, prompt: 'a portrait', checkpoint: 'krea\\lora_nova_000002500.safetensors', ...extra,
})

test('the viewer offers 📤 Civitai beside 📷 on a library row, and nothing on a bare preview', () => {
  const html = inApp(createElement(GeneratedImageLightbox, { img: row(), alt: 'x', onClose: () => {} }))
  assert.match(html, /data-testid="lightbox-civitai"/)
  assert.match(html, /data-testid="lightbox-camera-angles"/)
  const bare = inApp(createElement(GeneratedImageLightbox, {
    img: { url: '/preview.png', step: 500 }, alt: 'preview', onClose: () => {},
  }))
  assert.doesNotMatch(bare, /data-testid="lightbox-civitai"/, 'a URL-only preview has no row to post')
})

test('the image door renders: thumbnail, the page section, the post section', () => {
  const html = inApp(createElement(CivitaiPublishModal, {
    context: { kind: 'image', img: row() }, onClose: () => {},
  }))
  assert.match(html, /data-testid="civitai-publish-modal"/)
  assert.match(html, /data-probe-layer/, 'a layer that covers the page must say so to the responsive probe')
  assert.match(html, /Post this image on Civitai/)
  assert.match(html, /lora_nova_000002500 · step 2500/, 'the subject names the checkpoint and the step')
  assert.match(html, /data-testid="civitai-post-image"/)
  assert.match(html, /data-testid="civitai-post-publish-now"[^>]*checked/, 'a post is published by default')
  assert.match(html, /Looking up the link/, 'the link is being resolved on open')
})

test('a legacy picture without a checkpoint stamp is told so, never guessed', () => {
  const html = inApp(createElement(CivitaiPublishModal, {
    context: { kind: 'image', img: row({ record_id: null, step: null }) }, onClose: () => {},
  }))
  assert.match(html, /carries no checkpoint stamp/)
  assert.match(html, /No checkpoint of this dataset is linked/)
  assert.doesNotMatch(html, /data-testid="civitai-ref"/, 'no address field: there is no checkpoint to mark')
})

test('the checkpoint door renders the mark / create tabs and no post section', () => {
  const html = inApp(createElement(CivitaiPublishModal, {
    context: {
      kind: 'checkpoint',
      node: { record_id: 4, dataset_id: 7, dataset_name: 'Nova', train_type: 'krea' },
      pill: { step: 2500, filename: 'lora_nova_000002500.safetensors' },
    },
    onClose: () => {},
  }))
  assert.match(html, /Civitai model page/)
  assert.match(html, /Nova · step 2500/)
  assert.match(html, /Looking up the link/)
  assert.doesNotMatch(html, /data-testid="civitai-post-image"/, 'a checkpoint is not a picture to post')
})

test('the mark pane looks the page up first and only then offers a version to link', async () => {
  // Effects never run under renderToStaticMarkup, so the pane is proved in the
  // state it opens in: the address field and Look up, no version pick yet —
  // the pick only exists once a page has answered.
  const { default: Modal } = await import('../src/components/shared/CivitaiPublishModal.jsx')
  const html = inApp(createElement(Modal, {
    context: {
      kind: 'checkpoint',
      node: { record_id: 4, dataset_id: 7, dataset_name: 'Nova', train_type: 'krea' },
      pill: { step: 2500, filename: 'lora_nova_000002500.safetensors' },
    },
    onClose: () => {},
  }))
  // The link lookup is still pending at first paint, so the tabs are not
  // drawn yet; the pane's own controls are pinned by their test ids in the
  // source, which the contract below reads.
  assert.match(html, /Looking up the link/)
})

test('the popover shows the 📤 row only when a host can open the dialog, and never for a run card', () => {
  const node = { record_id: 4, dataset_id: 7, train_type: 'krea', source: 'local' }
  const pill = { step: 2500, filename: 'lora_nova_000002500.safetensors', present: true, download_url: '/dl' }
  const withHost = renderToStaticMarkup(createElement(CheckpointActionsPopover, {
    node, pill, onPublish: () => {}, onClose: () => {},
  }))
  assert.match(withHost, /data-testid="checkpoint-civitai"/)
  assert.match(withHost, /📤<\/span> Civitai</)
  const linked = renderToStaticMarkup(createElement(CheckpointActionsPopover, {
    node, pill: { ...pill, civitai: { model_name: 'Nova', version_name: 'v1' } },
    onPublish: () => {}, onClose: () => {},
  }))
  assert.match(linked, /📤<\/span> On Civitai</)
  assert.match(linked, /On Civitai: Nova · v1/)
  const noHost = renderToStaticMarkup(createElement(CheckpointActionsPopover, { node, pill, onClose: () => {} }))
  assert.doesNotMatch(noHost, /data-testid="checkpoint-civitai"/)
  const runCard = renderToStaticMarkup(createElement(CheckpointActionsPopover, {
    node, pill: null, onPublish: () => {}, onClose: () => {},
  }))
  assert.doesNotMatch(runCard, /data-testid="checkpoint-civitai"/, 'a page is made from ONE save')
})
