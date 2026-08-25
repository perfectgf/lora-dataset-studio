import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
  HONESTY_NOTE, MERGE_DRAFT_KEY, MERGE_OPEN_KEY, MERGE_RUNNING_STATES,
  PRECISION_NOTE, TURBO_NOTE, canAskPlan, carriedOverNote, clearMergeDraft,
  fmtDuration, fmtGB, initialMergeBase, loadMergeDraft, loadMergeOpen,
  loraPayload, newLoraRow, pct, planHeadline, saveMergeDraft, saveMergeOpen,
  weightHint,
} from './loraMerge.js';

/** A localStorage stand-in. `raise` makes every call throw, which is what a
 *  browser with storage disabled actually does — it is not an absent object. */
const fakeStore = (seed = {}, raise = false) => ({
  data: { ...seed },
  getItem(key) { if (raise) throw new Error('denied'); return key in this.data ? this.data[key] : null; },
  setItem(key, value) { if (raise) throw new Error('denied'); this.data[key] = String(value); },
  removeItem(key) { if (raise) throw new Error('denied'); delete this.data[key]; },
});

const plan = (over = {}) => ({
  ok: true,
  base_name: 'krea2_raw_bf16.safetensors',
  destination_name: 'krea2_raw_bf16_merged_20260804-081624.safetensors',
  destination_dir: 'A:\\ComfyUI\\models\\unet\\Krea',
  family_label: 'Krea 2',
  merged_tensors: 256,
  base_tensors: 430,
  output_bytes: 26.28e9,
  required_bytes: 28.28e9,
  free_bytes: 51.8e9,
  estimated_seconds: 119,
  carried_over: [],
  carried_over_bytes: 0,
  loras: [{ name: 'mine.safetensors', weight: 0.8, rank: 32, has_alpha: false }],
  on_failure: 'The merge writes to a .part file and only renames it when it finishes.',
  ...over,
});

// --- rows ---------------------------------------------------------------------

test('rows get distinct ids so removing one never removes its neighbour', () => {
  const a = newLoraRow();
  const b = newLoraRow();
  assert.notEqual(a.id, b.id);
  assert.equal(a.weight, 1, 'a fresh row means "exactly as trained"');
});

test('only rows that name a file are sent, and weights are numbers', () => {
  const rows = [newLoraRow('mine.safetensors', '0.8'), newLoraRow('  '),
    newLoraRow(' turbo.safetensors ', 1)];
  assert.deepEqual(loraPayload(rows), [
    { path: 'mine.safetensors', weight: 0.8 },
    { path: 'turbo.safetensors', weight: 1 },
  ]);
});

test('the weight leaves as a JSON number, never a locale-formatted string', () => {
  // The machine this was built on is fr-FR: the weight field DISPLAYS "0,8".
  // <input type="number"> still hands JavaScript "0.8", so what matters is that
  // we send Number(...) and not the raw display — pinned here because the bug it
  // would cause (float("0,8") on the server) is invisible in an en-US locale and
  // breaks every French, German, Spanish and Brazilian user.
  const body = JSON.stringify({ loras: loraPayload([newLoraRow('a.safetensors', '0.8')]) });
  assert.match(body, /"weight":0\.8/);
  assert.doesNotMatch(body, /0,8/);
  // and the type really is a number, not a numeric string
  assert.equal(typeof loraPayload([newLoraRow('a.safetensors', '0.8')])[0].weight, 'number');
});

test('an empty weight becomes something the server can refuse, not a crash', () => {
  // Number('') is 0 and Number('abc') is NaN; NaN serialises to null. Both are
  // refusals the plan states in words — neither is an exception.
  assert.equal(loraPayload([newLoraRow('a.safetensors', '')])[0].weight, 0);
  assert.equal(JSON.stringify(loraPayload([newLoraRow('a.safetensors', 'abc')])),
    '[{"path":"a.safetensors","weight":null}]');
});

test('the check button stays disabled until there is something to check', () => {
  assert.equal(canAskPlan('', [newLoraRow('mine.safetensors')]), false);
  assert.equal(canAskPlan('base.safetensors', [newLoraRow('')]), false);
  assert.equal(canAskPlan('base.safetensors', [newLoraRow('mine.safetensors')]), true);
});

// --- formatting ---------------------------------------------------------------

test('sizes read as sizes, and an unknown one is not "0 GB"', () => {
  assert.equal(fmtGB(26.28e9), '26.3 GB');
  assert.equal(fmtGB(null), '—');
  assert.equal(fmtGB(0), '—');
});

test('durations round to something a human acts on', () => {
  assert.equal(fmtDuration(0), '');
  assert.equal(fmtDuration(45), 'under a minute');
  assert.equal(fmtDuration(119), 'about 2 minutes');
  assert.equal(fmtDuration(3600), 'about 1 h');
  assert.equal(fmtDuration(5400), 'about 1 h 30 min');
});

test('progress never leaves the 0-100 range whatever the server says', () => {
  assert.equal(pct(0, 0), 0);
  assert.equal(pct(215, 430), 50);
  assert.equal(pct(999, 430), 100);
  assert.equal(pct(-5, 430), 0);
});

// --- the sentences ------------------------------------------------------------

test('the headline names the file, the size and the folder', () => {
  const line = planHeadline(plan());
  assert.match(line, /Folds 1 LoRA into krea2_raw_bf16\.safetensors/);
  assert.match(line, /krea2_raw_bf16_merged_20260804-081624\.safetensors \(26\.3 GB\)/);
  assert.match(line, /ComfyUI/);
  assert.equal(planHeadline({ ok: false }), '');
});

test('the headline pluralises when several LoRAs are stacked', () => {
  const line = planHeadline(plan({
    loras: [{ name: 'a.safetensors', weight: 1 }, { name: 'b.safetensors', weight: 0.5 }],
  }));
  assert.match(line, /Folds 2 LoRAs/);
});

test('tensors we do not understand are DISCLOSED by name, not refused', () => {
  // The real case: a community Krea 2 file carries ~75 MB of an image in two
  // tensors hiding under the legitimate `last.` prefix. Refusing the whole merge
  // over them would help nobody; dropping them silently would be worse.
  const note = carriedOverNote(plan({
    carried_over: [{ name: 'last.down.weight', bytes: 37.7e6 },
      { name: 'last.up.weight', bytes: 37.7e6 }],
    carried_over_bytes: 75.4e6,
  }));
  assert.match(note, /2 tensors are not part of the Krea 2 layout/);
  assert.match(note, /last\.down\.weight, last\.up\.weight/);
  assert.match(note, /copied over unchanged/);
  assert.match(note, /nothing is dropped/);
});

test('a clean base says nothing at all rather than reassuring at length', () => {
  assert.equal(carriedOverNote(plan()), '');
  assert.equal(carriedOverNote({ ok: true }), '');
});

test('a long list of passengers is truncated but its count stays true', () => {
  const rows = Array.from({ length: 7 }, (_v, i) => ({ name: `x.${i}.weight`, bytes: 10 }));
  const note = carriedOverNote(plan({ carried_over: rows, carried_over_bytes: 70 }));
  assert.match(note, /^7 tensors are not part/);
  assert.match(note, /and 3 more/);
});

test('the weight field guides before the server refuses', () => {
  assert.equal(weightHint(1), '');
  assert.match(weightHint(0), /contributes nothing/);
  assert.match(weightHint(1.5), /harder than it was trained/);
  assert.match(weightHint(-0.5), /subtracts the LoRA/);
});

// --- the honesty this feature exists to keep ---------------------------------

test('the app never calls a merged model a trained one', () => {
  // On the model sites this exact object is routinely published as a "finetune".
  // That is the vocabulary this feature refuses, in the screen AND in the file.
  assert.match(HONESTY_NOTE, /not a model that was trained/);
  assert.match(HONESTY_NOTE, /metadata/);
  for (const text of [HONESTY_NOTE, TURBO_NOTE, PRECISION_NOTE]) {
    assert.doesNotMatch(text, /\bfinetune\b/i);
  }
});

test('the Turbo transplant is offered WITH the reserve we owe it', () => {
  assert.match(TURBO_NOTE, /have not tested it ourselves/);
  assert.match(TURBO_NOTE, /approximation/);
});

test('merging into a quantized file is explained, not just forbidden', () => {
  assert.match(PRECISION_NOTE, /full-precision \(bf16\)/);
  assert.match(PRECISION_NOTE, /compounds/);
});

test('only "running" keeps the poller alive', () => {
  assert.deepEqual(MERGE_RUNNING_STATES, ['running']);
  for (const done of ['done', 'error', 'cancelled', undefined]) {
    assert.equal(MERGE_RUNNING_STATES.includes(done), false);
  }
});

// --- the draft: what a window resize used to eat ------------------------------
//
// The tool renders inside TrainingPanel's CheckpointPortal, whose host node
// comes and goes with the layout. React swaps between a portal and a plain
// subtree there, and that swap UNMOUNTS everything inside: a phone turned to
// landscape lost the path just typed. These helpers are what survives it, so
// they are held to the rule that they may degrade but never throw — they run
// during state initialisation, where an exception costs the whole panel.

test('the draft key and the disclosure key are the ones on user machines', () => {
  // Stored on every user's browser: renaming either silently discards a draft
  // and re-closes a tool somebody had left open. There is no alias path here.
  assert.equal(MERGE_DRAFT_KEY, 'loraMergeDraft_v1');
  assert.equal(MERGE_OPEN_KEY, 'loraMergeOpen_v1');
});

test('what was typed comes back after the remount', () => {
  const store = fakeStore();
  saveMergeDraft({
    base: 'D:\\models\\krea2_raw_bf16.safetensors',
    rows: [newLoraRow('mine.safetensors', '0.8'), newLoraRow('turbo.safetensors', 1)],
  }, store);
  const back = loadMergeDraft(store);
  assert.equal(back.base, 'D:\\models\\krea2_raw_bf16.safetensors');
  assert.deepEqual(back.rows.map((row) => row.path), ['mine.safetensors', 'turbo.safetensors']);
  // and it still serialises to what the server accepts, weights included
  assert.deepEqual(loraPayload(back.rows), [
    { path: 'mine.safetensors', weight: 0.8 },
    { path: 'turbo.safetensors', weight: 1 },
  ]);
});

test('rows read back get FRESH ids, never the stored ones', () => {
  // A stored id could collide with a row added later in the session, and then
  // "Remove" on one would take its twin with it.
  const store = fakeStore();
  saveMergeDraft({ rows: [newLoraRow('a.safetensors')] }, store);
  const [restored] = loadMergeDraft(store).rows;
  assert.notEqual(restored.id, newLoraRow('b.safetensors').id);
});

test('no localStorage at all is a blank form, not a crash', () => {
  assert.deepEqual(loadMergeDraft(null).rows.map((r) => r.path), ['']);
  assert.equal(loadMergeDraft(null).base, '');
  assert.equal(loadMergeOpen(null), false);
  // and writing into nothing is a no-op, not a thrown keystroke
  assert.doesNotThrow(() => saveMergeDraft({ base: 'x', rows: [] }, null));
  assert.doesNotThrow(() => saveMergeOpen(true, null));
  assert.doesNotThrow(() => clearMergeDraft(null));
});

test('a storage that REFUSES every call degrades quietly', () => {
  // Private browsing and a full quota both throw on the call rather than
  // leaving `localStorage` undefined — the case an `if (localStorage)` misses.
  const store = fakeStore({}, true);
  assert.equal(loadMergeDraft(store).base, '');
  assert.equal(loadMergeOpen(store), false);
  assert.doesNotThrow(() => saveMergeDraft({ base: 'x', rows: [] }, store));
  assert.doesNotThrow(() => saveMergeOpen(true, store));
  assert.doesNotThrow(() => clearMergeDraft(store));
});

test('corrupt JSON is a blank form', () => {
  const store = fakeStore({ [MERGE_DRAFT_KEY]: '{"base":"a.safetensors",' });
  const back = loadMergeDraft(store);
  assert.equal(back.base, '');
  assert.equal(back.rows.length, 1);
});

test('a shape from another era is a blank form, whatever it is', () => {
  for (const stored of ['null', '"a string"', '42', '[1,2,3]', '{"rows":"nope"}',
    '{"base":{"path":"x"}}', '{}']) {
    const back = loadMergeDraft(fakeStore({ [MERGE_DRAFT_KEY]: stored }));
    assert.equal(back.base, '', `base survived ${stored}`);
    assert.equal(back.rows.length, 1, `rows survived ${stored}`);
    assert.equal(back.rows[0].path, '');
  }
});

test('junk rows are dropped one by one, the good ones are kept', () => {
  const store = fakeStore({
    [MERGE_DRAFT_KEY]: JSON.stringify({
      base: 'base.safetensors',
      rows: [null, 'a string', { weight: 1 }, { path: 'good.safetensors', weight: 0.5 },
        { path: 'noweight.safetensors' }],
    }),
  });
  const back = loadMergeDraft(store);
  assert.deepEqual(back.rows.map((row) => row.path), ['good.safetensors', 'noweight.safetensors']);
  assert.equal(back.rows[0].weight, 0.5);
  assert.equal(back.rows[1].weight, 1, 'a row with no weight means "as trained"');
});

test('an absurd row count is capped rather than rendered', () => {
  const rows = Array.from({ length: 500 }, (_v, i) => ({ path: `l${i}.safetensors`, weight: 1 }));
  const store = fakeStore({ [MERGE_DRAFT_KEY]: JSON.stringify({ base: '', rows }) });
  assert.equal(loadMergeDraft(store).rows.length, 24);
});

test('a base handed in by a full-model card WINS over the draft', () => {
  // Opening the tool from a card must merge into THAT model. Restoring a path
  // typed somewhere else there would write the wrong 26 GB file.
  const draft = { base: 'D:\\other\\something_else.safetensors', rows: [] };
  assert.equal(initialMergeBase('D:\\models\\this_run.safetensors', draft),
    'D:\\models\\this_run.safetensors');
  // …and with no card, the draft is exactly what the field starts on
  assert.equal(initialMergeBase('', draft), 'D:\\other\\something_else.safetensors');
  assert.equal(initialMergeBase(undefined, undefined), '');
});

test('starting a merge forgets the draft', () => {
  const store = fakeStore();
  saveMergeDraft({ base: 'a.safetensors', rows: [newLoraRow('b.safetensors')] }, store);
  clearMergeDraft(store);
  assert.equal(store.getItem(MERGE_DRAFT_KEY), null);
  assert.equal(loadMergeDraft(store).base, '');
});

test('the disclosure remembers being open, and only "open" counts as open', () => {
  const store = fakeStore();
  assert.equal(loadMergeOpen(store), false, 'closed is the default');
  saveMergeOpen(true, store);
  assert.equal(loadMergeOpen(store), true);
  saveMergeOpen(false, store);
  assert.equal(loadMergeOpen(store), false);
  for (const junk of ['true', 'yes', '', '{}']) {
    assert.equal(loadMergeOpen(fakeStore({ [MERGE_OPEN_KEY]: junk })), false,
      `"${junk}" must not read as open`);
  }
});

// --- the component's own source: invariants worth pinning --------------------

const tool = readFileSync(
  fileURLToPath(new URL('./LoraMergeTool.jsx', import.meta.url)), 'utf8');

test('the poller never treats apiFetch as if it resolved a Response', () => {
  // apiFetch RESOLVES THE PARSED BODY. `.then((r) => r.json())` throws a
  // TypeError the `.catch()` swallows, which once left a panel stuck on a
  // progress bar while the job finished perfectly.
  assert.doesNotMatch(tool, /apiFetch\([^)]*\)[\s\S]{0,80}\.json\(\)/);
});

test('the merge is a two-step: nothing starts without a plan on screen', () => {
  // `start` is only reachable from <LoraMergePlan onStart=...>, which renders
  // only when plan.ok. A button wired straight to /api/tools/lora-merge would
  // be a 26 GB write on one click.
  const starts = [...tool.matchAll(/onClick=\{start\}/g)];
  assert.equal(starts.length, 0, 'no top-level control may call start directly');
  assert.match(tool, /<LoraMergePlan[^>]*onStart=\{start\}/);
  assert.match(tool, /if \(!plan\?\.ok\) return null;/);
});

test('the tool survives 400 px: paths wrap, rows stack, nothing scrolls sideways', () => {
  assert.ok(tool.includes('break-all'), 'file names must be allowed to wrap');
  assert.ok((tool.match(/flex-col gap-1 sm:flex-row|flex-col gap-2 sm:flex-row/g) || []).length >= 2,
    'the LoRA rows and the action row stack on a phone and go inline from sm up');
  assert.ok((tool.match(/flex-wrap/g) || []).length >= 2);
  assert.ok(!tool.includes('overflow-x-scroll'));
  assert.ok(!tool.includes('whitespace-nowrap'));
  assert.ok(tool.includes('w-full sm:'), 'inputs are full width on a phone');
});

test('every input carries a label a screen reader can read', () => {
  // Cut each tag at its own `/>`, not at the first `>`: an `onChange` arrow
  // function puts a `>` inside the tag, which a naive `[^>]*` stops on and then
  // happily "passes" every input by never seeing its aria-label.
  const inputs = [...tool.matchAll(/<input\b/g)]
    .map((m) => tool.slice(m.index, tool.indexOf('/>', m.index) + 2));
  assert.ok(inputs.length >= 3, 'the base field, a LoRA path and a weight');
  for (const input of inputs) {
    assert.ok(/aria-label=|id="lora-merge-base"/.test(input),
      `an input with no accessible name: ${input.slice(0, 60)}`);
  }
});

test('a refusal is shown inline as an alert, never swallowed', () => {
  assert.match(tool, /role="alert"[\s\S]{0,120}plan\.error/);
});

test('an instance opened from a card never writes the shared draft', () => {
  // Both instances can be mounted at once: the card's tool would otherwise
  // overwrite, on mount, the path somebody typed in the always-present one.
  assert.match(tool, /const ownsDraft = !base;/);
  assert.match(tool, /if \(ownsDraft\) saveMergeDraft\(/);
  assert.match(tool, /if \(ownsDraft\) clearMergeDraft\(\)/);
});

test('the base prop only overwrites the field when it names something', () => {
  // `useEffect(() => setBasePath(base), [base])` also runs on mount, where an
  // empty prop would wipe the draft that had just been restored.
  assert.match(tool, /useEffect\(\(\) => \{ if \(base\) setBasePath\(base\); \}, \[base\]\);/);
});

const panel = readFileSync(
  fileURLToPath(new URL('./TrainingPanel.jsx', import.meta.url)), 'utf8');

test('the merge disclosure is controlled, so a remount cannot close it', () => {
  // `open` on a <details> is DOM state. The block sits inside CheckpointPortal,
  // and every swap of that portal's host unmounts it — which is how rotating a
  // phone closed the tool on top of emptying it.
  const summary = panel.indexOf('Merge a LoRA into a base checkpoint');
  assert.notEqual(summary, -1, 'the merge disclosure is gone');
  const block = panel.slice(panel.lastIndexOf('<details', summary), summary);
  assert.match(block, /open=\{mergeOpen\}/);
  assert.match(block, /onClick=\{toggleMerge\}/);
  // and the toggle persists, otherwise a reload still forgets it
  assert.match(panel, /const \[mergeOpen, setMergeOpen\] = useState\(loadMergeOpen\)/);
  assert.match(panel, /saveMergeOpen\(next\)/);
});
