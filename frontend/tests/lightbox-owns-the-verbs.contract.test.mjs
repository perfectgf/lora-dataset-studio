/* 🔍 The generated-image viewer OWNS its verbs — the hosts only supply context.

   Why this is a contract and not a convention: the viewer is mounted by four
   hosts, and when the verbs were host-wired the Canvas had ✦ Repair but no
   📷 Camera while the Gallery had 📷 but no ✦ — the same picture, the same
   library row, different powers depending on which screen you happened to
   open it from. Nobody decided that; three hosts were simply written on three
   different days. This file makes the next verb land everywhere at once, and
   makes a host that quietly re-wires one fail by name.

   Source-read, like every contract here: node --test parses no JSX. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';

const read = (p) => fs.readFileSync(path.join(process.cwd(), p), 'utf8');

const VIEWER = 'src/components/shared/GeneratedImageLightbox.jsx';
// Every component that renders <GeneratedImageLightbox. Grown on purpose when
// a new host appears — the point is that a NEW host gets every verb for free.
const HOSTS = [
  'src/pages/GalleryPage.jsx',
  'src/components/canvas/LineageCanvas.jsx',
  'src/components/shared/CheckpointGalleryPanel.jsx',
  'src/components/dataset/PreviewLightbox.jsx',
];

const viewer = read(VIEWER);

test('the host list above is the real host list', () => {
  const dirs = ['src/pages', 'src/components'];
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

test('the viewer draws ✦ and 📷 itself, gated by the row, not by the host', () => {
  assert.match(viewer, /data-testid="lightbox-repair"/);
  assert.match(viewer, /data-testid="lightbox-camera-angles"/);
  assert.match(viewer, /import CameraAnglePicker from '.\/CameraAnglePicker'/);
  // The standard repair wiring lives here — a host passes onRepair only to
  // OVERRIDE it, never because the button would otherwise not exist.
  assert.match(viewer, /\/api\/studio\/image\/\$\{imageId\}\/repair/);
  assert.match(viewer, /repair\/undo/);
  // 📷 disabled-with-reason, not hidden: the refusal is the explanation.
  assert.match(viewer, /cameraRefusal\(img\)/);
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

test('the picker has exactly two mounts: this viewer and the dataset lightbox', () => {
  // Two id spaces, two mounts — lora_test_image here, face_dataset_image in
  // DatasetLightbox. A third mount is a copy about to drift.
  const all = [];
  const walk = (dir) => {
    for (const e of fs.readdirSync(path.join(process.cwd(), dir), { withFileTypes: true })) {
      const rel = `${dir}/${e.name}`;
      if (e.isDirectory()) walk(rel);
      else if (/\.jsx$/.test(e.name) && read(rel).includes('<CameraAnglePicker')) all.push(rel);
    }
  };
  walk('src');
  assert.deepEqual(all.sort(), [
    'src/components/dataset/DatasetLightbox.jsx',
    'src/components/shared/GeneratedImageLightbox.jsx',
  ]);
});

test('while the picker is open, the viewer stands down its window keys', () => {
  // The dataset lightbox lesson, applied here: Escape peels the picker first,
  // arrows must not walk the list under the open dial.
  assert.match(viewer, /if \(cameraOpen\) \{\s*if \(e\.key === 'Escape'\) setCameraOpen\(false\);\s*return;/);
});
