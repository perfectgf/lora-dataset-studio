import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { canRegenerateGeneric } from './improveRerun.js';
import { REVIEW_SHORTCUT_HINT } from '../shared/reviewShortcuts.js';

const lightbox = readFileSync(new URL('./DatasetLightbox.jsx', import.meta.url), 'utf8');
const workspace = readFileSync(new URL('./DatasetWorkspace.jsx', import.meta.url), 'utf8');
const hook = readFileSync(new URL('../../hooks/useDataset.js', import.meta.url), 'utf8');
const grid = readFileSync(new URL('./DatasetGrid.jsx', import.meta.url), 'utf8');
const settings = readFileSync(new URL('../settings/ScrapingSection.jsx', import.meta.url), 'utf8');
const attribution = readFileSync(new URL('./PexelsAttribution.jsx', import.meta.url), 'utf8');
const sourceAttribution = readFileSync(new URL('./SourceAttribution.jsx', import.meta.url), 'utf8');

test('lightbox exposes an accessible responsive image improvement action', () => {
  // ONE button per engine, from the shared pure module — the labels, the
  // per-engine disabled reasons and the trade-off sentences are asserted in
  // utils/improveEngines.test.js, so this file only pins the wiring.
  assert.match(lightbox, /improveButtons\.map\(\(btn\) => \(/);
  assert.match(lightbox, /lightboxImproveButtons\(\{/);
  assert.match(lightbox, /aria-busy=\{improvementActive\}/);
  assert.match(lightbox, /w-full sm:w-auto/);
  // The engine pressed travels to the handler; a single-✨ surface passes none
  // and the improve.engine setting decides instead.
  assert.match(lightbox, /onImprove\(img\.id, engineId\)/);
  // Klein's amber note follows KLEIN's button, never SeedVR2's: it is about
  // Klein's instruction pulling drawn skin towards realism, and SeedVR2 sends
  // no instruction at all.
  assert.match(lightbox, /\{rail && btn\.showKleinNote &&/);
  assert.match(lightbox, /\{!rail && improveButtons\.some\(\(b\) => b\.showKleinNote\)/);
});

// The comparison is what makes an improvement judgeable: before this, the
// lightbox showed the RESULT alone and the original had to be remembered.
// node --test cannot render JSX, so the contract is asserted on the source.
test('a derived image can be inspected next to the original it came from', () => {
  // Both panes live in ONE grid whose cells are equal, and both images are
  // object-contain: identical box, identical scale — the only honest way to
  // compare an upscale against its source (Klein rescales to a pixel budget and
  // keeps the aspect ratio, so equal boxes means identical framing).
  assert.match(lightbox, /grid-rows-2 grid-cols-1 sm:grid-rows-1 sm:grid-cols-2/);
  // h-full w-full, never max-h/max-w: an <img> at its intrinsic size is capped
  // but never scaled UP, so a small original rendered smaller than the result
  // that filled its pane — two different scales, the exact dishonesty this mode
  // exists to remove. Caught in a headless capture, pinned here.
  assert.match(lightbox, /className="h-full w-full select-none object-contain"/);
  assert.doesNotMatch(lightbox, /max-h-full max-w-full select-none object-contain/);
  // Real button, pressed state carried by aria (not colour alone).
  assert.match(lightbox, /aria-pressed=\{compareMode === 'derived'\}/);
  assert.match(lightbox, /Compare with original/);
  assert.match(lightbox, /Exit comparison/);
  // Full-width control at phone width, like the other lightbox actions.
  assert.match(lightbox, /w-full sm:w-auto[^]{0,400}Compare with original/);
  // Each pane names its side in TEXT — from whichever descriptor is on screen.
  assert.match(lightbox, /activeCompare\.beforeLabel/);
  assert.match(lightbox, /activeCompare\.afterLabel/);
  // Zoom is not silently broken: comparison says, in the same hint slot, that
  // 100 % lives outside the comparison.
  assert.match(lightbox, /exit comparison to zoom/i);
  // A vanished original explains itself instead of leaving a dead button.
  assert.match(lightbox, /compare && !compare\.available/);
  assert.match(lightbox, /\{compare\.reason\}/);
});

// The second comparison: "is this still the same person?" — the question a
// plainly generated variation raises, which the derived comparison could never
// answer because a generated row has no `derivation_kind` and therefore no
// button at all. Decisions live in utils/referenceCompare.js (unit-tested);
// this file pins the wiring, since node --test cannot render JSX.
test('any dataset image can be inspected next to the reference photo', () => {
  assert.match(lightbox, /describeReferenceComparison\(img, refFilename\)/);
  // Its own button, next to the other one — not a mode picker that would hide
  // one question behind the other on an improved image.
  assert.match(lightbox, /Compare with reference/);
  assert.match(lightbox, /aria-pressed=\{compareMode === 'reference'\}/);
  assert.match(lightbox, /w-full sm:w-auto[^]{0,400}Compare with reference/);
  // ONE state for both readings ⇒ entering either leaves the other; two
  // booleans would have allowed two pairs of panes at once. And it lives in the
  // id-STAMPED per-image slot, not in a useState of its own: that is what makes
  // ⟩ leave the comparison behind with the image it belonged to, instead of
  // carrying an "Original" pane onto a picture whose parent is someone else's.
  assert.match(lightbox, /full, compareMode, improving, actionsOpen, repairOpen, deciding,\n  \} = lightboxImageState\(/);
  assert.doesNotMatch(lightbox, /useState\((true|false|'none')\)/);
  assert.match(lightbox,
    /patchImageState\(\{\s*full: false,\s*compareMode: compareMode === mode \? 'none' : mode,/);
  // The reference is NOT a payload row: its own cache nonce, same /img endpoint.
  assert.match(lightbox, /compareMode === 'reference' \? refNonce : parentNonce/);
  // The rail/bar rule only needs to know a comparison is open, whichever it is.
  assert.match(lightbox, /comparing: compareMode !== 'none'/);
  // "same scale" is TRUE of the derived pair and FALSE here (a square head
  // reference against a body plan): the hint is per mode, not shared.
  assert.match(lightbox, /compareMode === 'derived' && inCompare/);
  assert.match(lightbox, /different framings — each pane fits its own image/);
  // A reference recorded without a usable file states why, like the other mode.
  assert.match(lightbox, /refCompare && !refCompare\.available/);
  assert.match(lightbox, /\{refCompare\.reason\}/);
});

/* The dataset lightbox is where a picture is actually LOOKED at, and until now
   the only thing it could not do there was judge it: the verdict had to be taken
   on the thumbnail behind the overlay. It now offers the Bank's review bar —
   same three verdicts, same glyphs, same keys — wired to the dataset's OWN
   status. node --test cannot render JSX, so the wiring is pinned as text; the
   keyboard grammar itself is unit-tested in shared/reviewShortcuts.test.js. */
test('the dataset lightbox offers Keep / Reject / Skip on the Bank keys', () => {
  assert.match(lightbox, /✓ Keep<ShortcutKey>K<\/ShortcutKey>/);
  assert.match(lightbox, /✕ Reject<ShortcutKey>R<\/ShortcutKey>/);
  assert.match(lightbox, /⏭ Skip<ShortcutKey>S<\/ShortcutKey>/);
  // The grammar is READ, never re-decided here — that is what stops the two
  // review surfaces from drifting apart one refactor at a time.
  assert.match(lightbox, /from '\.\.\/shared\/reviewShortcuts'/);
  assert.match(lightbox, /const action = reviewKeyAction\(e\)/);
  assert.match(lightbox, /action === 'keep' \|\| action === 'reject'/);
  assert.match(lightbox, /decide\(action\)/);
  // K/R are the SAME status words the grid tile writes — no second notion of
  // "kept" for captioning, export and training to disagree with.
  assert.match(lightbox, /await onStatus\(imageId, status\)/);
  assert.match(workspace, /onStatus=\{viewImgLive\._rescueReviewPreview \? null : ds\.setStatus\}/);
  // Printed, not folklore — from the same constant the handler is built on.
  assert.match(lightbox, /\{REVIEW_SHORTCUT_HINT\} · ← → move without deciding · Esc close/);
  assert.equal(REVIEW_SHORTCUT_HINT, 'K keep · R reject · S skip');
});

test('a verdict advances only once the write has landed, and skip touches nothing', () => {
  const decide = lightbox.match(/const decide = useCallback\(([\s\S]*?)\n  \}, \[/)[1];
  // The move is INSIDE the try, after the await: advancing first and posting
  // afterwards is how a decision is silently dropped on a slow disk.
  assert.match(decide, /await onStatus\(imageId, status\);\s*\n[\s\S]*?if \(after && onNavigate\) onNavigate\(after\)/);
  // The next image is captured BEFORE the write — refreshing under a status
  // filter can retire this very image from the shown list.
  assert.match(decide, /const after = nextImage;[\s\S]*await onStatus/);
  // A second K while the first is in flight must not fire a second write.
  assert.match(decide, /if \(!onStatus \|\| deciding \|\| busy \|\| imageId == null\) return/);
  // ⏭ Skip is nothing but "next": no status call anywhere in it.
  const skip = lightbox.match(/const skipImage = useCallback\(([\s\S]*?)\n  \}, \[/)[1];
  assert.doesNotMatch(skip, /onStatus/);
  assert.match(skip, /onNavigate\(nextImage\)/);
});

test('the lightbox says which verdict the image already carries', () => {
  // Three buttons that also CHANGE the state cannot be the only reading of it —
  // "did my K land?" is otherwise unanswerable on the last image of a list.
  assert.match(lightbox, /keep: \{ text: '✓ kept'/);
  assert.match(lightbox, /reject: \{ text: '✕ rejected'/);
  assert.match(lightbox, /pending: \{ text: '· undecided'/);
  assert.match(lightbox, /VERDICT_CHIP\[img\.status\]/);
  // Same green/red as the Bank's review chips: one colour code across the app.
  assert.match(lightbox, /bg-emerald-500\/25 text-emerald-200/);
  assert.match(lightbox, /bg-rose-500\/25 text-rose-200/);
});

test('workspace hands the lightbox the reference photo and its own nonce', () => {
  assert.match(workspace, /refFilename=\{d\.ref_filename \|\| ''\}/);
  assert.match(workspace, /refNonce=\{ds\.refNonce \|\| 0\}/);
});

test('workspace feeds the lightbox the resolved parent of a derived image', () => {
  assert.match(workspace, /describeDerivedComparison/);
  assert.match(workspace, /compare=\{viewImgComparison\}/);
  assert.match(workspace, /parentNonce=/);
});

test('workspace guards rescue rows and detects a pending improvement child', () => {
  assert.match(workspace, /!viewImgLive\._rescueReviewPreview/);
  assert.match(workspace, /!isSmallImageRescueRow\(viewImgLive\)/);
  assert.match(workspace, /viewImgLive\.derivation_kind !== 'klein_image_improve'/);
  assert.match(workspace, /image\.derivation_kind === 'klein_image_improve'/);
  assert.match(workspace, /image\.parent_image_id === viewImgLive\.id/);
  assert.match(workspace, /const viewImgImproving[\s\S]*image\.status === 'pending'[\s\S]*\)\) : false/);
  assert.match(workspace, /const viewImgImprovementReady[\s\S]*image\.status === 'pending'[\s\S]*!!image\.filename/);
  // Engine readiness is read from capabilities INSIDE the lightbox now, per
  // engine — a `kleinAvailable` prop would be a second source of truth for
  // one of the two engines and none for the other.
  assert.match(workspace, /onImprove=\{canImproveViewImg/);
  assert.match(workspace, /ds\.improveImage\(imageId, \{ engine \}\)/);
});

test('dataset hook starts improvement, reports the preserved original, then refreshes', () => {
  assert.match(hook, /`\/api\/dataset\/image\/\$\{imageId\}\/improve`,/);
  // The engine rides along when the caller names one (the lightbox's two
  // buttons); absent, the server falls back to the improve.engine setting.
  assert.match(hook, /engine \? \{ engine \} : \{\}/);
  assert.match(hook, /original stays intact while a separate 2 MP candidate is generated for validation/);
  assert.match(hook, /Could not start image improvement/);
  assert.match(hook, /resolveSmallImageRescue, improveImage, reimproveImage, improveBatch, classify/);
});

test('the bulk improvement is ONE call that starts a server job, not a per-image loop', () => {
  assert.match(hook, /`\/api\/dataset\/\$\{currentId\}\/improve\/batch`,/);
  assert.match(hook, /\{ image_ids: ids, engine \}/);
  // The engine the user pressed rides along; absent, the server falls back to the
  // improve.engine setting (which is what the single-tile pass uses).
  assert.match(grid, /onImproveBatch\(eligible\.map\(\(image\) => image\.id\), engineId\)/);
  // No client-side sequential driver survives: that loop is what walked into the
  // fan-out cap and made ⏹ Stop powerless.
  assert.doesNotMatch(grid, /runSequentialKleinImprove/);
  // Progress is read from the server activity, so it survives a reload.
  assert.match(grid, /improveBatchLabel\(activity\)/);
  // ⏹ Stop generation stays reachable (and enabled) for a running batch. The
  // enabled-ness itself is decided by isStopGenerationBlocked (unit-tested in
  // scraperState.test.js) — this earlier inline expression only exempted
  // 'improve', which left the button dead for every plain generation batch.
  assert.match(workspace, /pending > 0 \|\| act\?\.kind === 'improve'/);
  assert.match(workspace, /disabled=\{isStopGenerationBlocked\(\{[\s\S]{0,120}?busy: ds\.busy, activity: act/);
});

test('settings separates scraper rescue instructions from manual lightbox improvement', () => {
  assert.match(settings, /title="Klein rescue — small scraped images"/);
  assert.match(settings, /automatic rescue of scraped images under 768 px/);
  // The point is that the two flows are distinct and the manual one is elsewhere —
  // asserted on that meaning, not on a fixed sentence. The old wording claimed the
  // manual pass had a fixed profile, which stopped being true once its strength and
  // step count became editable.
  assert.match(settings, /manual Upscale & improve is a different flow/);
  assert.match(settings, /Settings ▸ Image engines/);
  // the rescue card points at the separate manual "Identity & Klein prompts" card
  assert.match(settings, /separate from the manual .Klein upscale &amp; improve. prompt/i);
});

test('manual improvement candidates cannot use the unrelated generic regenerate path', () => {
  const gridItem = readFileSync(new URL('./DatasetGridItem.jsx', import.meta.url), 'utf8');
  // The guard moved into improveRerun.js (testable in node --test, which cannot
  // parse JSX) when the tile gained its own 🔄✨ re-run of the improve pass. Same
  // meaning, asserted at both ends: the tile delegates, and the decision refuses.
  assert.match(gridItem, /const isImageImproveCandidate = isImageImproveRow\(img\)/);
  assert.match(gridItem, /const canRegenerate = canRegenerateGeneric\(img, \{ isRescueDerived \}\)/);
  assert.match(gridItem, /if \(!isImageImproveCandidate && img\.status !== 'reject'/);
  assert.equal(canRegenerateGeneric({ source: 'generated', filename: 'a.png', status: 'keep',
    derivation_kind: 'klein_image_improve', parent_image_id: 2 }), false);
});

test('curation grid and lightbox render provenance through ONE dispatching component', () => {
  // Two sources used to mean two stacked, independently-fail-closed components
  // rendered at every site — twelve more recorded as debt would have meant
  // twelve stacked components at each. Both render sites now feed the SAME
  // `img.source_metadata` to a single dispatcher instead.
  const gridItem = readFileSync(new URL('./DatasetGridItem.jsx', import.meta.url), 'utf8');
  assert.match(gridItem, /<SourceAttribution metadata=\{img\.source_metadata\}/);
  assert.match(lightbox, /<SourceAttribution metadata=\{img\.source_metadata\}/);
  assert.doesNotMatch(gridItem, /<PexelsAttribution\b|<WebImageSource\b/);
  assert.doesNotMatch(lightbox, /<PexelsAttribution\b|<WebImageSource\b/);

  // The dispatcher is a lookup table keyed by platform, fail-closed on both an
  // unrecognized platform (no entry => no renderer => null) and, per-platform,
  // whatever its own renderer's payload validation still refuses.
  assert.match(sourceAttribution, /pexels:\s*PexelsAttribution/);
  assert.match(sourceAttribution, /websearch:\s*WebImageSource/);
  assert.match(sourceAttribution, /if \(!Renderer\) return null;/);

  // The Pexels wording/links guarantee itself still lives in PexelsAttribution.jsx
  // — moved under the dispatcher, not narrowed by it.
  assert.match(attribution, /Photo by\{' '\}/);
  assert.match(attribution, /rel="noopener noreferrer"/);
  assert.match(attribution, /attribution\.photographerUrl/);
  assert.match(attribution, /attribution\.sourceUrl/);
});
