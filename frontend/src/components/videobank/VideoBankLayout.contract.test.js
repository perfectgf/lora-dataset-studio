/**
 * The video bank wears the image bank's Encre shell — pinned so it cannot
 * quietly drift back into a lane with its own clothes.
 *
 * Born of the maintainer's 2026-09-01 objection, verbatim: « J'ai l'impression
 * que tu travailles sur la partie vidéo comme si c'était quelque chose de
 * différent… Tu utilises une UI différente, des méthodes différentes. » He was
 * right, measurably: zero imports from the shared atoms, a pre-Encre vertical
 * stack, hand-rolled chips. These assertions are the mirror of
 * BankOverviewLayout.contract.test.js, at the strength that matters: the SAME
 * decision module, the SAME atoms, the SAME shell literals — two lanes, two
 * component trees (deliberately), ONE form.
 */
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const read = (name) => readFileSync(new URL(`./${name}`, import.meta.url), 'utf8')
const workspace = read('VideoBankWorkspace.jsx')
const rail = read('VideoFilterRail.jsx')
const passes = read('VideoPassesPanel.jsx')

test('the video lane decides its layout with bankLayout.js, never a copy of it', () => {
  // One decision module for both lanes: when the rail is a column, what the
  // preference key is, what the ⚙ button says. A videobank-local fork of any
  // of these is the drift this contract exists to refuse.
  assert.match(workspace, /from '\.\.\/bank\/bankLayout\.js'/)
  assert.match(workspace, /loadRailOpen|railIsColumn/)
  assert.doesNotMatch(workspace, /RAIL_SIDE_BY_SIDE|const railIsColumn =/)
})

test('the shell is the image lane’s: rail beside the grid, drawer below', () => {
  assert.match(workspace, /lg:grid-cols-\[17rem_minmax\(0,1fr\)\]/)
  assert.match(workspace, /railOpen && railIsColumnNow/)
  // The drawer really covers the grid instead of squeezing it…
  assert.match(workspace, /railOpen && !railIsColumnNow/)
  // …and as a column the rail is pinned, or it scrolls away after one screen.
  assert.match(workspace, /lg:sticky/)
})

test('the shared atoms dress every surface — no hand-rolled twins', () => {
  assert.match(workspace, /from '\.\.\/bank\/BankAtoms\.jsx'/)
  assert.match(rail, /\{ Chip, FilterGroup, GroupLabel \} from '\.\.\/bank\/BankAtoms\.jsx'/)
  assert.match(passes, /\{ GroupLabel, PassButton \} from '\.\.\/bank\/BankAtoms\.jsx'/)
  // The drawer is a sheet of glass over the grid, not an opaque card — the
  // exact pin the image rail carries, for the exact same reason.
  assert.match(rail, /bg-surface-overlay/)
})

test('the passes are a panel on demand, and the probe can see all of it', () => {
  assert.match(workspace, /id="video-passes-panel"/)
  assert.match(workspace, /data-probe-reading/)
  assert.match(workspace, /aria-controls="video-passes-panel"/)
  assert.match(workspace, /id="video-filter-rail"/)
  assert.match(workspace, /aria-controls="video-filter-rail"/)
  assert.match(workspace, /data-probe-chrome="header"/)
  // The rail names itself to the probe in both of its lives.
  assert.match(rail, /data-probe-panel="rail"/)
})

test('what left the workspace arrived somewhere a user can still reach', () => {
  // The refactor moved surfaces, it must not lose them: every pass button, the
  // search box, the thresholds, the files list and the shot-cut dial live on.
  for (const [file, needle] of [
    [passes, /'probe', 'detect', 'thumbs', 'measure', 'embed'/],
    [passes, /VideoShotCutsPanel/],
    [rail, /VideoClipSearchBox/],
    [rail, /VideoThresholdsPanel/],
    [rail, /VideoSourceList/],
    [workspace, /PASS_LABELS\.pipeline/],
    [workspace, /PASS_LABELS\.promote/],
  ]) {
    assert.match(file, needle)
  }
})
