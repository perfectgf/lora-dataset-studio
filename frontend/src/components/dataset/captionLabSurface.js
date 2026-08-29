// The 🧪 Caption Lab's two surfaces, in ONE place.
//
// The bench itself (CaptionLab.jsx) is surface-agnostic: it composes candidates and
// reads results. What it cannot know is WHERE to run one, what Stop talks to, and what
// "use this config" means here — and those three are the only things that differ between
// a Dataset and a Bank. Naming them once, side by side, is what keeps the two from
// drifting: the backend already shares one definition of a candidate
// (face_dataset_service.preview_caption_path, pinned by test_bank_caption_lab.py), and
// this is its front-of-house half.
import { postJson } from '../../api/fetchClient';

/* WHAT AN IMAGE IS CALLED is not the same field on the two surfaces, and assuming it was
   is how the Bank shipped a dead filter: a dataset row carries `filename`, a bank row
   carries `name` (the basename of its relpath, image_bank_service.py:1401), and
   `filename` appears nowhere in a bank payload. One permissive resolver rather than a
   key spelled in three places. It lives HERE, beside the other two-surface differences,
   so a host can import it without dragging the lazily-loaded picker into its chunk. */
export const imageDisplayName = (img) => img.filename || img.name || String(img.id);

/** The Dataset bench: a dataset OWNS a caption method, stored on the row. */
export function datasetLabSurface({ datasetId, imageId }) {
  return {
    kind: 'dataset',
    preview: (body) => postJson(
      `/api/dataset/${datasetId}/image/${imageId}/caption/preview`, body),
    cancel: () => postJson(`/api/dataset/${datasetId}/caption/cancel`, {}),
    applyConfig: (config) => postJson(`/api/dataset/${datasetId}/caption/options`, config),
    applyLabel: '⚙️ Make default',
    applyTitle: "Store this config as the dataset's default caption method",
    applyDone: 'Saved as this dataset’s default caption method',
  };
}

/** The Bank bench.
 *
 *  THE LABEL IS DIFFERENT ON PURPOSE, and it is not a parity gap — it is the rule.
 *  A bank has no caption_options row to persist to (useCaptionOptions.js says so in as
 *  many words): its engine, vision model, register and length are picked PER RUN. So the
 *  winning config is loaded into those dials, which keeps the promise the dataset button
 *  makes — "the next caption pass uses this" — in the mechanics this surface actually
 *  has. CLAUDE.md's wording rule cuts both ways: identical behaviour deserves
 *  recognisable wording, and DIFFERENT behaviour must not wear the same label. Calling
 *  this one "Make default" would promise a persistence the Bank does not have.
 */
export function bankLabSurface({ bankId, imageId, onApplyRunConfig }) {
  return {
    kind: 'bank',
    preview: (body) => postJson(
      `/api/bank/${bankId}/image/${imageId}/caption/preview`, body),
    // The Bank has ONE Stop for whatever holds it, and the bench takes the bank lease
    // under the same 'caption' kind as the batch pass — so this is that same Stop.
    cancel: () => postJson(`/api/bank/${bankId}/cancel`, {}),
    applyConfig: async (config) => { onApplyRunConfig(config); },
    applyLabel: '⚙️ Use for the next run',
    applyTitle: 'Load this config into the caption dials above. A bank picks its engine, '
      + 'model, register and length per run rather than storing them, so this is what '
      + '"make it the default" means here.',
    applyDone: 'Loaded into the caption dials for the next run',
  };
}
