/** Every screen that STARTS Klein work names the model it will run on.
 *
 * The reported gap was two questions in one, and only the first got answered
 * first: "let me choose the model" (a dataset setting) and "tell me which model
 * is running" (a line of text). The second is the half that works everywhere —
 * on a one-model install, and on the bank, which has no dataset to choose with.
 *
 * Three lanes were still silent after the choice shipped: the local reference
 * edit, the rescue of scraped images under 768 px, and the watermark inpaint
 * (dataset AND bank). They render the shared <KleinModelSetting> now, and this
 * pins it: a surface that drops the component fails here rather than quietly
 * going back to running an unnamed model. `node --test` cannot parse JSX, so
 * the assertion is textual — the same technique as SettingsLink.test.js.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const read = (p) => fs.readFileSync(new URL(p, import.meta.url), 'utf8');

// file → whether that surface also lets the model be CHOSEN (it has a dataset).
const SURFACES = {
  '../dataset/KleinImproveNote.jsx': true,      // ✨ Upscale & improve (shipped first)
  '../dataset/VariationCatalog.jsx': true,      // Klein generation
  '../dataset/ReferenceEditModal.jsx': true,    // local reference edit
  '../dataset/ConceptSourcesPanel.jsx': true,   // rescue of small scraped images
  '../dataset/DatasetWorkspace.jsx': true,      // 🧽 Clean, bulk
  '../dataset/WatermarkReviewLightbox.jsx': true,  // 🧽 Clean, one image
  '../bank/BankWatermarkPanel.jsx': false,      // bank inpaint — naming only
  '../bank/BankEditPanel.jsx': false,           // bank ✨ improve — naming only
};

for (const [file, scoped] of Object.entries(SURFACES)) {
  test(`${file} names the Klein model it will run on`, () => {
    const src = read(file);
    assert.match(src, /import KleinModelSetting from/,
      'the surface no longer imports the shared model line');
    assert.match(src, /<KleinModelSetting\b/);
    if (scoped) {
      // Dataset-scoped: the line names the DATASET's model and can change it.
      assert.match(src, /<KleinModelSetting datasetId=/);
    } else {
      // Bank: there is no dataset to inherit from, and adding a picker here
      // would be a second place to choose a model for the same UNETLoader.
      assert.doesNotMatch(src, /<KleinModelSetting[^>]*datasetId=/);
    }
  });
}

test('the dead per-browser picker is gone, not merely unused', () => {
  // Last writer of editPage_flux2KleinModel_v1. Nothing imported it once the
  // choice moved onto the dataset; a component that still WRITES a key the app
  // now only reads as a legacy suggestion is a trap for the next reader.
  assert.equal(fs.existsSync(new URL('./Flux2KleinModelPicker.jsx', import.meta.url)),
    false);
});
