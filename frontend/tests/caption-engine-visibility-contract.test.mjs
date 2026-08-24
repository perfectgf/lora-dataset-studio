/* The caption engine has to be visible WHERE THE RESULT IS, not only in Settings.
 *
 * `captioning.backend` defaults to 'auto', which chains JoyCaption and the Ollama
 * vision model — two different writing styles, switched between without a word. The
 * Settings copy describes the INTENT; nothing described the run. So the backend now
 * counts who wrote each stored caption and the caption route returns it, and the two
 * surfaces that show a caption result have to actually use it: the toast that reports
 * the pass, and a line under the caption buttons that survives the toast.
 *
 * Read as SOURCE, and the limitation is stated rather than hidden: DatasetWorkspace
 * and useDataset cannot be mounted here without a dataset payload, a jobs context and
 * a capabilities provider that fetches. What this pins is the WIRING — that the
 * response field reaches both surfaces and is scoped to the right dataset. The
 * wording itself is rendered from utils/captionEngines.js, which IS executed, by
 * src/utils/captionEngines.test.js.
 */
import assert from 'node:assert/strict'
import { readSource } from './support/readSource.mjs'
import test from 'node:test'

const read = (rel) => readSource(`src/${rel}`)

test('every caption pass carries the engines from the response into the session state', () => {
  const src = read('hooks/useDataset.js')
  // Three entry points write captions: caption, re-caption, and the targeted
  // re-caption from the leak panel. A pass that forgets to report is a pass whose
  // result silently loses its author.
  const stores = src.match(/setLastCaptionRun\(\{ datasetId: (?:currentId|run\.datasetId), captioned: d\.captioned, engines: d\.engines \}\)/g) || []
  assert.equal(stores.length, 3, 'caption, recaption and recaptionImages must each report');
  // The toast that already reports the count now names the writer too.
  const suffixed = src.match(/captionResultSuffix\(d\.engines\)/g) || []
  assert.equal(suffixed.length, 3)
  assert.match(src, /lastCaptionRun,/, 'the hook must expose it to the workspace')
})

test('the line is scoped to the dataset it describes', () => {
  const src = read('components/dataset/DatasetWorkspace.jsx')
  // Without the id check, switching datasets would leave the previous dataset's
  // pass described under another dataset's buttons — a new lie replacing the old
  // silence.
  assert.match(src, /ds\.lastCaptionRun\.datasetId === d\.id/)
  assert.match(src, /captionEnginesSummary\(ds\.lastCaptionRun\.engines\)/)
  // Shown in the Captions section, under the buttons that produced it.
  const captions = src.slice(src.indexOf('id="gf-captions"'))
  assert.ok(captions.indexOf('lastCaptionEngines') > -1
    && captions.indexOf('lastCaptionEngines') < captions.indexOf('id="ds-captions-generate"'),
  'the engine line belongs in the Captions section, above the generate row')
  // 400px: the sentence names two engines and must be allowed to wrap.
  assert.match(src, /break-words[^"]*text-\[0\.75rem\][^"]*text-content-muted" title=\{CAPTION_ENGINE_WHY\}|title=\{CAPTION_ENGINE_WHY\}[\s\S]{0,200}break-words/)
})
