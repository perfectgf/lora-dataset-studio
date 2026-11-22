import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { deriveSteps } from '../../hooks/useGuidedFlow.js';
import { sectionProgress } from './sectionProgress.js';

const image = (id, extra = {}) => ({ id, filename: `${id}.jpg`, status: 'keep', caption: 'a caption', ...extra });
const CAPS = { training_visible: true, studio_visible: true, face_scoring: true };

test('the two steps of Add images are judged together, and the figure is the generated count', () => {
  // reference set, nothing generated yet
  const empty = deriveSteps({ ref_filename: 'ref.jpg', images: [] }, CAPS, 0);
  const before = sectionProgress(empty.steps, empty.nextStep);
  assert.equal(before.add.done, false);
  assert.equal(before.add.next, true);
  assert.equal(before.add.short, '0/25');

  const filled = deriveSteps({ ref_filename: 'ref.jpg', images: [image(1), image(2)] }, CAPS, 0);
  const after = sectionProgress(filled.steps, filled.nextStep);
  assert.equal(after.add.done, true);
  assert.equal(after.add.short, '2/25');
  assert.equal(after.images.short, '2 kept');
  assert.equal(after.captions.short, '2/2');
});

test('exactly one section carries the next step, and it follows the path', () => {
  const noCaption = deriveSteps({ ref_filename: 'ref.jpg', images: [image(1, { caption: '' })] }, CAPS, 0);
  const progress = sectionProgress(noCaption.steps, noCaption.nextStep);
  assert.deepEqual(Object.keys(progress).filter((id) => progress[id].next), ['captions']);
  assert.equal(progress.captions.done, false);
  assert.equal(progress.images.done, true);

  const trained = deriveSteps({ ref_filename: 'ref.jpg', images: [image(1)] }, CAPS, 3);
  const later = sectionProgress(trained.steps, trained.nextStep);
  assert.deepEqual(Object.keys(later).filter((id) => later[id].next), ['studio']);
  assert.equal(later.training.done, true);
  assert.equal(later.checkpoints.short, '3');
});

test('an optional step neither blocks its section nor becomes the next step', () => {
  // face scoring is the only step of Curation, and it is optional
  const path = deriveSteps({ ref_filename: 'ref.jpg', images: [image(1)] }, CAPS, 0);
  const progress = sectionProgress(path.steps, path.nextStep);
  assert.equal(progress.curation.next, false);
  assert.equal(progress.curation.done, false);
  const scored = deriveSteps({ ref_filename: 'ref.jpg', images: [image(1, { face_state: 'ok' })] }, CAPS, 0);
  assert.equal(sectionProgress(scored.steps, scored.nextStep).curation.done, true);
});

test('a section without a step has no state at all', () => {
  const path = deriveSteps({ ref_filename: 'ref.jpg', images: [image(1)] }, CAPS, 0);
  const progress = sectionProgress(path.steps, path.nextStep);
  assert.equal(progress.scrape, undefined);
  assert.equal(progress.export, undefined);            // training is available: Export is not a step
  const exportOnly = deriveSteps({ ref_filename: 'ref.jpg', images: [image(1)] }, { studio_visible: true }, 0);
  assert.equal(sectionProgress(exportOnly.steps, exportOnly.nextStep).export.next, true);
  assert.deepEqual(sectionProgress(undefined, null), {});
});

test('a step this install cannot do is reported with its reason', () => {
  const path = deriveSteps({ ref_filename: 'ref.jpg', images: [image(1)] }, { training_visible: true }, 1);
  const progress = sectionProgress(path.steps, path.nextStep);
  assert.equal(progress.studio.unavailable, true);
  assert.equal(progress.studio.hint, 'Configure ComfyUI in Settings');
  assert.equal(progress.studio.next, false);
});

test('the rail leads a step that needs ComfyUI to the ComfyUI API setting', () => {
  // What the PROGRESS checklist did with a gear link, kept when the rail took over its job.
  const workspace = readFileSync(new URL('./DatasetWorkspace.jsx', import.meta.url), 'utf8');
  assert.match(workspace,
    /\{blocked && \(\s*<Link to="\/settings\/local-tools\?focus=comfyui-api-url" title=\{blocked\.hint\}/);
});
