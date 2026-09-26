/* 🔍 The generated-image viewer OWNS its verbs — the hosts only supply context.

   Why this is a contract and not a convention: the viewer is mounted by four
   hosts, and when the verbs were host-wired the Canvas had ✦ Repair but no
   📷 Camera while the Gallery had 📷 but no ✦ — the same picture, the same
   library row, different powers depending on which screen you happened to
   open it from. Nobody decided that; three hosts were simply written on three
   different days. This file makes the next verb land everywhere at once, and
   makes a host that quietly re-wires one fail by name.

   Host wiring is read as source. The installed owners' actions are also
   rendered through the real SDK host; effects and keyboard events are not
   executed by that server render. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { render } from './support/mountJsx.mjs';
import { installRuntimeHost } from './support/runtimeHost.mjs';
import { registerBundledDescriptor } from './support/bundledDescriptors.mjs';
import { contributions, setEnabled } from '../src/plugins/registry.js';
import { contributionKey, createLayerTracker } from '../src/plugins/layerTracker.js';
import cameraDescriptor from '../../bundled/camera_angles/frontend/index.js';
import civitaiDescriptor from '../../bundled/civitai_publish/frontend/index.js';

test.beforeEach(installRuntimeHost);

const read = (p) => fs.readFileSync(path.join(process.cwd(), p), 'utf8');

const VIEWER = 'src/components/shared/GeneratedImageLightbox.jsx';
// Every component that renders <GeneratedImageLightbox. Grown on purpose when
// a new host appears — the point is that a NEW host gets every verb for free.
const HOSTS = [
  'src/pages/GalleryPage.jsx',
  '../bundled/canvas/frontend/components/canvas/LineageCanvas.jsx',
  'src/components/shared/CheckpointGalleryPanel.jsx',
  'src/components/dataset/PreviewLightbox.jsx',
  // The Test Studio (normal view AND run comparison) — it replaced its own
  // fifth lightbox with the shared viewer, exactly the move this contract
  // exists to make cheap: facts and verbs arrived for free, the host kept
  // only its ordered wrap-around set and the 👍/👎 verdict.
  'src/components/dataset/studio/StudioResultViewer.jsx',
];

const viewer = read(VIEWER);
const cameraAction = read('../bundled/camera_angles/frontend/panels/GalleryCameraAction.jsx');
const civitaiAction = read('../bundled/civitai_publish/frontend/panels/GalleryCivitaiAction.jsx');

test('the host list above is the real host list', () => {
  const dirs = ['src/pages', 'src/components', '../bundled/canvas/frontend'];
  const found = [];
  const walk = (dir) => {
    for (const e of fs.readdirSync(path.join(process.cwd(), dir), { withFileTypes: true })) {
      const rel = `${dir}/${e.name}`;
      if (e.isDirectory()) walk(rel);
      else if (/\.jsx?$/.test(e.name) && !/\.test\./.test(e.name)
        && read(rel).includes('<GeneratedImageLightbox')) found.push(rel);
    }
  };
  dirs.forEach(walk);
  assert.deepEqual(found.sort(), [...HOSTS].sort(),
    'a new GeneratedImageLightbox host appeared — add it to HOSTS here; it gets every verb for free');
});

test('the shared viewer offers ✦ and the active 📷 verb, gated by the row, not by the host', async () => {
  assert.match(viewer, /data-testid="lightbox-repair"/);
  assert.match(viewer, /<PluginSlot slot="lightbox.action" surface="gallery" img=\{img\}/);
  assert.match(viewer, /hasRow=\{hasRow\} disabled=\{busy\}/);
  assert.match(cameraAction, /data-testid="lightbox-camera-angles"/);
  assert.match(cameraAction, /import CameraAnglePicker from '.\/CameraAnglePicker'/);
  // The standard repair wiring lives here — a host passes onRepair only to
  // OVERRIDE it, never because the button would otherwise not exist.
  assert.match(viewer, /\/api\/studio\/image\/\$\{imageId\}\/repair/);
  assert.match(viewer, /repair\/undo/);
  // 📷 disabled-with-reason, not hidden: the refusal is the explanation.
  assert.match(cameraAction, /cameraRefusal\(img\)/);
  assert.deepEqual(contributions('lightbox.action', 'gallery'), [], 'absent camera owner has no verb');
  assert.equal(registerBundledDescriptor(cameraDescriptor), true);
  setEnabled(['camera_angles']);
  const [item] = contributions('lightbox.action', 'gallery');
  assert.equal(item.plugin, 'camera_angles');
  const { default: CameraAction } = await item.panels.gallery();
  const html = render(CameraAction, { img: { id: 7, status: 'done' }, hasRow: true });
  assert.match(html, /data-testid="lightbox-camera-angles"/);
  assert.doesNotMatch(html, /disabled=""/);
  const refused = render(CameraAction, {
    img: { id: 8, status: 'done', derivation_kind: 'camera_angle' }, hasRow: true,
  });
  assert.match(refused, /disabled=""/);
  assert.match(refused, /cannot itself be re-shot/);
  assert.equal(render(CameraAction, { img: { url: '/p.png' }, hasRow: false }), '');
  setEnabled([]);
  assert.deepEqual(contributions('lightbox.action', 'gallery'), [], 'disabled camera owner has no verb');
});

test('no host re-wires a viewer verb', () => {
  for (const h of HOSTS) {
    const src = read(h);
    assert.ok(!src.includes('lightbox-camera-angles'),
      `${h}: the camera button belongs to the viewer`);
    assert.ok(!src.includes('lightbox-repair'),
      `${h}: the repair button belongs to the viewer`);
    assert.ok(!src.includes("shared/CameraAnglePicker"),
      `${h}: the picker is mounted by the viewer, not the host`);
    assert.ok(!/onRepair(?:Undo)?=/.test(src),
      `${h}: standard repair is the viewer's — pass onRowChanged, not onRepair`);
  }
});

test('the shared picker serves both lightboxes and the standalone camera workspace', async () => {
  // All three surfaces use the same picker instead of maintaining separate controls.
  const all = [];
  const walk = (dir) => {
    for (const e of fs.readdirSync(path.join(process.cwd(), dir), { withFileTypes: true })) {
      const rel = `${dir}/${e.name}`;
      if (e.isDirectory()) walk(rel);
      else if (/\.jsx$/.test(e.name) && read(rel).includes('<CameraAnglePicker')) all.push(rel);
    }
  };
  walk('src');
  for (const entry of fs.readdirSync(path.join(process.cwd(), '../bundled'), { withFileTypes: true })) {
    const dir = `../bundled/${entry.name}/frontend`;
    if (entry.isDirectory() && fs.existsSync(path.join(process.cwd(), dir))) walk(dir);
  }
  assert.deepEqual(all.sort(), [
    '../bundled/camera_angles/frontend/panels/CameraStudio.jsx',
    '../bundled/camera_angles/frontend/panels/DatasetCameraAction.jsx',
    '../bundled/camera_angles/frontend/panels/GalleryCameraAction.jsx',
  ]);
  assert.match(read('src/components/dataset/DatasetLightbox.jsx'),
    /<PluginSlot slot="lightbox.action" surface="dataset" img=\{img\}/);
  assert.equal(registerBundledDescriptor(cameraDescriptor), true);
  setEnabled(['camera_angles']);
  for (const [surface, expected] of [
    ['dataset', '../../bundled/camera_angles/frontend/panels/DatasetCameraAction.jsx'],
    ['gallery', '../../bundled/camera_angles/frontend/panels/GalleryCameraAction.jsx'],
  ]) {
    const items = contributions('lightbox.action', surface);
    assert.equal(items.length, 1, `one camera mount on ${surface}`);
    const actual = await items[0].panels[surface]();
    assert.equal(actual.default, (await import(expected)).default);
  }
  setEnabled([]);
  for (const surface of ['dataset', 'gallery']) assert.deepEqual(contributions('lightbox.action', surface), []);
});

test('while the picker is open, the viewer stands down its window keys', () => {
  // The dataset lightbox lesson, applied here: Escape peels the picker first,
  // arrows must not walk the list under the open dial.
  assert.match(viewer, /if \(pluginLayersRef.current.any\(\)\) return;/);
  assert.match(viewer, /onLayer: pluginLayersRef.current.onLayerFor\(contributionKey\(item\)\)/);
  assert.match(cameraAction, /onLayer\?\.\(open\)/);
  assert.match(cameraAction, /return \(\) => onLayer\?\.\(false\)/);
  assert.match(cameraAction, /if \(!open\) return undefined;/);
  assert.match(cameraAction, /if \(e.key === 'Escape'\) \{ e.stopPropagation\(\); setOpen\(false\); \}/);
  assert.match(cameraAction, /addEventListener\('keydown', onKey, true\)/);
  assert.match(cameraAction, /removeEventListener\('keydown', onKey, true\)/);
  assert.equal(registerBundledDescriptor(cameraDescriptor), true);
  assert.equal(registerBundledDescriptor(civitaiDescriptor), true);
  setEnabled(['camera_angles', 'civitai_publish']);
  const items = contributions('lightbox.action', 'gallery');
  assert.equal(items.length, 2, 'camera and publishing coexist in the shared viewer');
  const tracker = createLayerTracker();
  const cameraLayer = tracker.onLayerFor(contributionKey(items.find(item => item.plugin === 'camera_angles')));
  const civitaiLayer = tracker.onLayerFor(contributionKey(items.find(item => item.plugin === 'civitai_publish')));
  cameraLayer(true);
  civitaiLayer(false);
  assert.equal(tracker.any(), true, 'another closed verb must not release the camera layer');
  cameraLayer(false);
  assert.equal(tracker.any(), false, 'closing the camera layer returns keys to the viewer');
});

test('📤 Civitai is the active viewer verb, addressed by the row\'s own checkpoint stamp', async () => {
  assert.match(civitaiAction, /data-testid="lightbox-civitai"/);
  assert.match(civitaiAction, /import CivitaiPublishModal from '.\/CivitaiPublishModal'/);
  // The image door: the modal is told it is about THIS row, nothing else.
  assert.match(civitaiAction, /<CivitaiPublishModal context=\{\{ kind: 'image', img \}\}/);
  // Disabled-with-reason, like 📷.
  assert.match(civitaiAction, /civitaiVerbRefusal\(img\)/);
  // A layer like ✦ / 📷 / ✨: the window keys stand down under it.
  assert.match(viewer, /if \(repairOpen \|\| improveOpen\) return;/);
  assert.match(viewer, /if \(pluginLayersRef.current.any\(\)\) return;/);
  assert.match(civitaiAction, /onLayer\?\.\(open\)/);
  assert.match(civitaiAction, /return \(\) => onLayer\?\.\(false\)/);
  for (const h of HOSTS) {
    const src = read(h);
    assert.ok(!src.includes('lightbox-civitai'), `${h}: the Civitai button belongs to the viewer`);
    // A host may open the CHECKPOINT door of the same dialog (the popover's
    // row); the IMAGE door is the viewer's alone.
    assert.ok(!src.includes("kind: 'image'"), `${h}: the image door of the Civitai dialog is the viewer's`);
  }
  assert.deepEqual(contributions('lightbox.action', 'gallery'), [], 'absent publisher has no verb');
  assert.equal(registerBundledDescriptor(civitaiDescriptor), true);
  setEnabled(['civitai_publish']);
  const [item] = contributions('lightbox.action', 'gallery');
  assert.equal(item.plugin, 'civitai_publish');
  const { default: CivitaiAction } = await item.panels.gallery();
  const html = render(CivitaiAction, { img: { id: 7 }, hasRow: true });
  assert.match(html, /data-testid="lightbox-civitai"/);
  assert.doesNotMatch(html, /disabled=""/);
  const refused = render(CivitaiAction, { img: { id: 'missing' }, hasRow: true });
  assert.match(refused, /disabled=""/);
  assert.match(refused, /no library entry to post/);
  assert.equal(render(CivitaiAction, { img: { url: '/p.png' }, hasRow: false }), '');
  assert.deepEqual(contributions('lightbox.action', 'dataset'), [], 'the publisher has no dataset door');
  setEnabled([]);
  assert.deepEqual(contributions('lightbox.action', 'gallery'), [], 'disabled publisher has no verb');
});
