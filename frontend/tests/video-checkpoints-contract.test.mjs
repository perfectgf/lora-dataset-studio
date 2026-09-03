/**
 * 📦 Checkpoints & LoRAs of a video dataset — the rules that hold the section
 * together, pinned at the SOURCE because `renderToStaticMarkup` runs none of
 * them:
 *  · ONE list. The training block used to render the cloud files itself; a
 *    second copy of the list is how one surface grows a verb the other lacks.
 *  · Every confirmation sentence comes from videoCheckpoints.js, and the
 *    destination inside it from utils/deletionWording.js — the wording the
 *    clips' removal already uses. A "Trash" typed by hand in the JSX is exactly
 *    what shipped wrong once (a sentence promising the app's Trash on an
 *    install whose default is the OS recycle bin).
 *  · ▶ rents a pod, so it owes the licence question and the confirmations
 *    loop, like every other pod-renting POST of the lane.
 *  · The rail's two new sections point at anchors the workspace renders.
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import { readSource } from './support/readSource.mjs'

const codeOnly = (text) => text
  .replace(/\{\/\*[\s\S]*?\*\/\}/g, '')
  .replace(/\/\*[\s\S]*?\*\//g, '')
  .replace(/^\s*\/\/.*$/gm, '')
const manager = codeOnly(readSource('src/components/videobank/VideoCheckpointManager.jsx'))
const block = codeOnly(readSource('src/components/videobank/VideoTrainingBlock.jsx'))
const workspace = codeOnly(readSource('src/components/videobank/VideoDatasetWorkspace.jsx'))
const model = readSource('src/components/videobank/videoCheckpoints.js')
const clips = readSource('src/components/videobank/videoDatasetClips.js')

test('the training block renders no checkpoint file any more — the section is the one list', () => {
  assert.ok(!/videoDatasetCheckpointUrl|videoDatasetLocalCheckpointUrl/.test(block),
    'the block must not build download links')
  assert.ok(!/vds-training-checkpoints|deleteRun/.test(block))
  assert.match(block, /onSaveCount\?\.\(saveCount\)/, 'the block reports its save count to the section')
  assert.match(workspace, /<VideoCheckpointManager ds=\{ds\} refreshKey=\{saveCount\}/)
})

test('every confirmation the section asks is a sentence of the model, never typed in the JSX', () => {
  const confirms = [...manager.matchAll(/window\.confirm\(([^)]*\()/g)].map((m) => m[1])
  assert.ok(confirms.length >= 3, `expected the three confirmations, found ${confirms.length}`)
  for (const c of confirms) {
    assert.match(c, /^(describeStepDelete|describeUndeploy|runDeleteConfirmation)\($/,
      `a confirmation is not a model sentence: ${c}`)
  }
  assert.ok(!/Trash(?!2)|recoverable|Recycle Bin/.test(manager),   // Trash2 is the icon
    'the destination wording lives in the model, through deletionWording.js')
  assert.match(model, /deleteDestination\(mode\)/)
  assert.match(model, /isRecoverable\(mode\)/)
  // Both deletes of the workspace read the same helper — one destination, two verbs.
  assert.match(clips, /deleteDestination\(/)
})

test('▶ Continue from here owes the licence question and the confirmations loop', () => {
  const at = manager.indexOf('const confirmContinue = (g, s) => {')
  assert.notEqual(at, -1)
  const body = manager.slice(at, manager.indexOf('if (err && !payload)'))
  const ack = body.indexOf('ensureLicenceAck(ds')
  const post = body.indexOf("postWithConfirmations((b) => postJson(url, b),\n        continueBody(g, s, extraSteps), 'Launch anyway (force)')")
  assert.ok(ack !== -1 && post !== -1 && ack < post, 'licence ack first, then the confirmations loop')
  assert.match(body, /if \(d === null\) return/, 'a declined question rents nothing and says nothing')
  // No bare postJson may rent a pod: the continue URL appears only inside the loop.
  const bare = [...manager.matchAll(/await postJson\(([^,]+),/g)].map((m) => m[1].trim())
  assert.ok(!bare.some((u) => /Continue/.test(u)), `a bare postJson rents a pod: ${bare}`)
})

test('the rail\'s Checkpoints and Studio sections land on anchors the workspace renders', async () => {
  const { VIDEO_DATASET_SECTIONS } = await import('../src/components/videobank/videoDatasetSections.js')
  const ids = VIDEO_DATASET_SECTIONS.map((s) => s.id)
  assert.deepEqual(ids.slice(-2), ['checkpoints', 'studio'], 'the two sections close the rail, like the image one')
  for (const id of ['checkpoints', 'studio']) {
    const section = VIDEO_DATASET_SECTIONS.find((s) => s.id === id)
    assert.equal(section.when, undefined, `${id} is always in the rail`)
    for (const p of section.panels) assert.ok(workspace.includes(`id="${p.targetId}"`), `${p.targetId} missing`)
  }
  assert.match(workspace, /<Link to="\/studio\?lane=video"/, 'the Studio launcher opens the Video tab')
})

test('the ◉ Graph is a second view of the SAME saves: same handlers, same model, no image route', async () => {
  const graph = codeOnly(readSource('src/components/videobank/VideoLineageGraph.jsx'))
  // The manager hands the graph the very functions the list rows call.
  assert.match(manager, /<VideoLineageGraph datasetId=\{ds\.id\} tree=\{tree\} busy=\{busy\}/)
  assert.match(manager, /onDeploy=\{deploy\} onUndeploy=\{undeploy\} onDelete=\{remove\}\s+onContinue=\{continueFrom\}/)
  // The popover's decisions come from the list's model through the bridge.
  assert.match(graph, /a=\{pillActionModel\(datasetId, openCk\.node, openCk\.pill, ctx\)\}/)
  assert.ok(!/\/api\//.test(graph), 'the graph builds no URL of its own — the model does')
  // Nothing under the video lane ever addresses an IMAGE dataset BY ID: the
  // two tables share one id space, and `/api/dataset/<id>/…` with a video id is
  // somebody else's dataset. (The id-less cloud status route is shared on purpose.)
  const { readdirSync } = await import('node:fs')
  const dir = new URL('../src/components/videobank/', import.meta.url)
  for (const f of readdirSync(dir)) {
    if (!/\.jsx?$/.test(f) || /\.test\./.test(f)) continue
    const src = codeOnly(readSource(`src/components/videobank/${f}`))
    assert.ok(!/\/api\/dataset\/(\$\{|\d)/.test(src), `${f} addresses an image dataset BY ID`)
  }
})

test('the deploy folder on screen is the server\'s, with the app\'s own subfolder as the only default', () => {
  assert.match(manager, /deployFolder: payload\?\.deploy_folder \|\| 'h3\/lds'/)
  assert.ok(!/Deploy → h3/.test(manager), 'the folder is never typed into the label')
})
