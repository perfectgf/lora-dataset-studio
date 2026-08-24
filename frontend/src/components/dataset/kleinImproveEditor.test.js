/* The in-place improve-instruction editor: what it reads, what it writes, and
   when it writes it. Every assertion here is on a VALUE, never on source text. */
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  IMPROVE_SCOPE_NOTE, createImproveSaver, effectiveImprovePrompt,
  improveEditorState, improveSettingsPatch,
} from './kleinImproveEditor.js';
import { readImproveInstruction } from './kleinImproveHint.js';

const SHIPPED = 'add detailed texture, add sharp details, add candid shot, add soft focus effect';

const payload = (identityPrompts = {}) => ({
  config: { identity_prompts: identityPrompts },
  identity_prompt_defaults: { klein_improve: SHIPPED },
});

// ---- reading ---------------------------------------------------------------

test('a payload that has not arrived yields loaded:false, never a guess', () => {
  for (const bad of [null, undefined, 'nope', 42]) {
    assert.deepEqual(improveEditorState(bad),
      { loaded: false, stored: '', shipped: '', enabled: true,
        loraPreset: '', loraPresets: [] });
  }
});

test('the LoRA-preset half reads from the same payload as the instruction', () => {
  const p = payload({});
  p.config.klein = {
    improve_lora_preset: 'Detail',
    generation_lora_presets: [
      { name: 'Detail', loras: [{ file: 'klein/d.safetensors', strength: 0.7 }] },
      { name: 'Skin', loras: [{ file: 'klein/s.safetensors', strength: 0.4 }] },
      { name: '', loras: [] },              // junk a hand-edited config can hold
    ],
  };
  const s = improveEditorState(p);
  assert.equal(s.loraPreset, 'Detail');
  assert.deepEqual(s.loraPresets, ['Detail', 'Skin']);
});

test('a stale preset pick is KEPT, so the user can see it and clear it', () => {
  // The backend resolves it fail-closed to "none"; hiding it here would leave
  // an invisible setting nobody can unset.
  const p = payload({});
  p.config.klein = { improve_lora_preset: 'Renamed-away', generation_lora_presets: [] };
  assert.equal(improveEditorState(p).loraPreset, 'Renamed-away');
});

test('stored and shipped stay SEPARATE — the box shows one and saves the other', () => {
  const s = improveEditorState(payload({ klein_improve: 'flat anime shading' }));
  assert.equal(s.stored, 'flat anime shading');
  assert.equal(s.shipped, SHIPPED, 'the shipped text is still reachable for Reset');
  assert.equal(s.loaded, true);
});

test('no override reads as stored:"" — which is what "follow the default" IS', () => {
  const s = improveEditorState(payload({}));
  assert.equal(s.stored, '');
  assert.equal(effectiveImprovePrompt(s), SHIPPED);
});

test('the toggle is on unless it is explicitly false', () => {
  assert.equal(improveEditorState(payload({})).enabled, true);
  assert.equal(improveEditorState(payload({ klein_improve_enabled: false })).enabled, false);
  // Anything else (a legacy string, a missing key) must not read as "off": the
  // backend contract is `!== false`, and reading it stricter would silently tell
  // the user no instruction is sent while one still is.
  assert.equal(improveEditorState(payload({ klein_improve_enabled: 'yes' })).enabled, true);
});

test('CONTRACT: the box and the quoted line agree on what is in force', () => {
  // Two readers of the same contract (the editor and the hint) must never
  // disagree, or the panel would quote one sentence and edit another.
  for (const ip of [
    {},
    { klein_improve: '' },
    { klein_improve: '   ' },
    { klein_improve: 'keep it a drawing' },
    { klein_improve: 'keep it a drawing', klein_improve_enabled: false },
    { klein_improve_enabled: false },
  ]) {
    const p = payload(ip);
    const s = improveEditorState(p);
    assert.equal(effectiveImprovePrompt(s), readImproveInstruction(p).prompt,
      `effective prompt disagrees for ${JSON.stringify(ip)}`);
    assert.equal(s.enabled, readImproveInstruction(p).enabled);
  }
});

test('a whitespace-only override still means "follow the default"', () => {
  assert.equal(effectiveImprovePrompt({ stored: '  \n ', shipped: SHIPPED }), SHIPPED);
});

// ---- writing ---------------------------------------------------------------

test('the patch is PARTIAL — a toggle save never rewrites the prompt', () => {
  assert.deepEqual(improveSettingsPatch({ enabled: false }),
    { config: { identity_prompts: { klein_improve_enabled: false } } });
  assert.deepEqual(improveSettingsPatch({ prompt: 'x' }),
    { config: { identity_prompts: { klein_improve: 'x' } } });
});

test('both fields ride in ONE patch when both are pending', () => {
  assert.deepEqual(improveSettingsPatch({ prompt: 'x', enabled: true }),
    { config: { identity_prompts: { klein_improve: 'x', klein_improve_enabled: true } } });
});

test('"back to the shipped text" is written as an EMPTY value, not a copy', () => {
  // The whole default contract: storing the default text would pin the user to
  // today's wording forever.
  assert.deepEqual(improveSettingsPatch({ prompt: '' }),
    { config: { identity_prompts: { klein_improve: '' } } });
});

test('a preset-only save touches klein and NOTHING else', () => {
  assert.deepEqual(improveSettingsPatch({ loraPreset: 'Detail' }),
    { config: { klein: { improve_lora_preset: 'Detail' } } });
  // Clearing it back to "none" is an empty string, same as every other
  // follow-the-default contract here.
  assert.deepEqual(improveSettingsPatch({ loraPreset: '' }),
    { config: { klein: { improve_lora_preset: '' } } });
});

test('instruction and preset ride in ONE patch when both are pending', () => {
  assert.deepEqual(improveSettingsPatch({ prompt: 'x', loraPreset: 'Skin' }),
    { config: { identity_prompts: { klein_improve: 'x' },
      klein: { improve_lora_preset: 'Skin' } } });
});

test('the scope note actually says the change is app-wide', () => {
  assert.match(IMPROVE_SCOPE_NOTE, /every/i);
  assert.match(IMPROVE_SCOPE_NOTE, /Settings/);
});

// ---- the saver -------------------------------------------------------------

/** A controllable clock: fires nothing until `run()` is called. */
function fakeTimers() {
  const queued = new Map();
  let next = 1;
  return {
    setTimeoutFn: (fn) => { queued.set(next, fn); return next++; },
    clearTimeoutFn: (id) => queued.delete(id),
    run() { const fns = [...queued.values()]; queued.clear(); fns.forEach((f) => f()); },
    get count() { return queued.size; },
  };
}

test('a burst of keystrokes becomes ONE write carrying the LAST value', () => {
  const t = fakeTimers();
  const sent = [];
  const s = createImproveSaver((p) => sent.push(p), { ...t });
  s.schedule('prompt', 'a');
  s.schedule('prompt', 'ab');
  s.schedule('prompt', 'abc');
  assert.deepEqual(sent, [], 'nothing goes out mid-sentence');
  t.run();
  assert.deepEqual(sent, [{ prompt: 'abc' }]);
});

test('a pending sentence rides along with the toggle instead of being lost', () => {
  const t = fakeTimers();
  const sent = [];
  const s = createImproveSaver((p) => sent.push(p), { ...t });
  s.schedule('prompt', 'keep it a drawing');
  s.schedule('enabled', false);
  s.flush();
  assert.deepEqual(sent, [{ prompt: 'keep it a drawing', enabled: false }]);
});

test('flush on unmount saves the last keystroke — the case that loses work', () => {
  const t = fakeTimers();
  const sent = [];
  const s = createImproveSaver((p) => sent.push(p), { ...t });
  s.schedule('prompt', 'half a sen');
  s.flush();                     // the lightbox closes here
  assert.deepEqual(sent, [{ prompt: 'half a sen' }]);
  t.run();                       // and the dead timer must not fire a second write
  assert.equal(sent.length, 1);
});

test('flush with nothing pending sends nothing at all', () => {
  const t = fakeTimers();
  const sent = [];
  const s = createImproveSaver((p) => sent.push(p), { ...t });
  s.flush();
  s.flush();
  assert.deepEqual(sent, [], 'unmounting a note nobody touched must not PUT');
});

test('cancel drops the pending patch', () => {
  const t = fakeTimers();
  const sent = [];
  const s = createImproveSaver((p) => sent.push(p), { ...t });
  s.schedule('prompt', 'x');
  assert.deepEqual(s.pending, { prompt: 'x' });
  s.cancel();
  assert.equal(s.pending, null);
  t.run();
  assert.deepEqual(sent, []);
});
