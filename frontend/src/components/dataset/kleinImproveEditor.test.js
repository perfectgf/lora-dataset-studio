/* The in-place improve-instruction editor: what it reads, what it writes, and
   when it writes it. Every assertion here is on a VALUE, never on source text. */
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  IMPROVE_SCOPE_NOTE, PRESET_LORA_STRENGTH_MAX, clampPresetStrength, createImproveSaver,
  effectiveImprovePrompt, improveEditorState, improveSettingsPatch,
  presetChainRows, withPresetRowStrength,
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
        loraPreset: '', loraPresets: [], presets: [], consistencyLora: '',
        megapixels: 2 });
  }
});

test('the output budget reads from the same payload, junk degrading to the shipped 2', () => {
  const p = payload({});
  p.config.klein = { improve_megapixels: 4.5 };
  assert.equal(improveEditorState(p).megapixels, 4.5);
  p.config.klein = { improve_megapixels: 'huge' };
  assert.equal(improveEditorState(p).megapixels, 2);
  assert.equal(improveEditorState(payload({})).megapixels, 2);
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

test('the output budget saves clamped to the Settings bounds, half-typed never writes', () => {
  assert.deepEqual(improveSettingsPatch({ megapixels: '4' }),
    { config: { klein: { improve_megapixels: 4 } } });
  // The same clamp the Settings card applies — two editors, one range.
  assert.deepEqual(improveSettingsPatch({ megapixels: 99 }),
    { config: { klein: { improve_megapixels: 8 } } });
  assert.deepEqual(improveSettingsPatch({ megapixels: 0 }),
    { config: { klein: { improve_megapixels: 0.5 } } });
  // An emptied box mid-typing is not a settings write.
  assert.deepEqual(improveSettingsPatch({ megapixels: '' }),
    { config: {} });
  // …and it shares the klein object with a pending preset pick.
  assert.deepEqual(improveSettingsPatch({ megapixels: 2.5, loraPreset: 'Skin' }),
    { config: { klein: { improve_lora_preset: 'Skin', improve_megapixels: 2.5 } } });
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

// ---- the picked preset's chain --------------------------------------------

/* A config as Settings really writes one: two presets, one of them holding a
   row a half-finished edit left blank. That blank row is the whole reason the
   drawn list and the stored list are joined by INDEX and not by position. */
const PRESETS = [
  { name: 'Real', loras: [
    { file: 'klein/details.safetensors', strength: 0.15 },
    { file: '', strength: 0.6 },                              // an empty slot
    { file: 'klein/realistic.safetensors', strength: 0.25 },
  ] },
  { name: 'Soft', loras: [{ file: 'klein/soft.safetensors', strength: 0.9 }] },
];

const withKlein = (klein) => ({
  config: { identity_prompts: {}, klein },
  identity_prompt_defaults: { klein_improve: SHIPPED },
});

test('the panel reads the preset DEFINITIONS, not only their names', () => {
  const st = improveEditorState(withKlein({
    generation_lora_presets: PRESETS, consistency_lora: 'klein/consistency.safetensors',
  }));
  assert.deepEqual(st.loraPresets, ['Real', 'Soft'], 'the picker still lists names');
  assert.equal(st.presets.length, 2, 'and the rows behind them are available to draw');
  assert.equal(st.presets[0].loras.length, 3,
    'kept AS STORED — sanitizing here would drop the blank row from the array '
    + 'this panel writes back');
  assert.equal(st.consistencyLora, 'klein/consistency.safetensors');
});

test('a preset with no name, or junk in the list, cannot crash the panel', () => {
  for (const bad of [null, 'nope', 42, {}, [1, 2]]) {
    const st = improveEditorState(withKlein({ generation_lora_presets: bad }));
    assert.ok(Array.isArray(st.presets));
    assert.deepEqual(presetChainRows(st.presets, 'Real'), []);
  }
});

test('the drawn rows skip empty slots but keep their real index', () => {
  const rows = presetChainRows(PRESETS, 'Real');
  assert.deepEqual(rows.map((r) => r.file),
    ['klein/details.safetensors', 'klein/realistic.safetensors']);
  assert.deepEqual(rows.map((r) => r.index), [0, 2],
    'the blank row still occupies index 1 in config — writing by the DRAWN '
    + 'position would move the second slider onto the wrong LoRA');
  assert.deepEqual(rows.map((r) => r.strength), [0.15, 0.25]);
});

test('a junk strength is drawn at the shipped 0.6 rather than an empty slider', () => {
  const rows = presetChainRows([{ name: 'x', loras: [{ file: 'a.safetensors' }] }], 'x');
  assert.equal(rows[0].strength, 0.6);
});

test('no pick, or a pick naming nothing, draws no chain', () => {
  assert.deepEqual(presetChainRows(PRESETS, ''), []);
  assert.deepEqual(presetChainRows(PRESETS, '   '), []);
  assert.deepEqual(presetChainRows(PRESETS, 'Deleted last week'), [],
    'a stale name is already reported by the picker; inventing rows under it '
    + 'would read as "broken" instead of "gone"');
});

test('moving one slider changes ONE number and carries everything else through', () => {
  const next = withPresetRowStrength(PRESETS, 'Real', 2, 0.75);
  assert.equal(next[0].loras[2].strength, 0.75);
  assert.equal(next[0].loras[0].strength, 0.15, 'the sibling row is untouched');
  assert.deepEqual(next[0].loras[1], { file: '', strength: 0.6 },
    'the empty slot survives a save it never asked for');
  assert.deepEqual(next[1], PRESETS[1], 'and so does every other preset');
  assert.notEqual(next, PRESETS, 'a new array — nothing is mutated in place');
  assert.deepEqual(PRESETS[0].loras[2], { file: 'klein/realistic.safetensors', strength: 0.25 });
});

test('the strength lands inside the range Settings and the backend both enforce', () => {
  assert.equal(withPresetRowStrength(PRESETS, 'Soft', 0, 99)[1].loras[0].strength,
    PRESET_LORA_STRENGTH_MAX);
  assert.equal(withPresetRowStrength(PRESETS, 'Soft', 0, -3)[1].loras[0].strength, 0);
  assert.equal(withPresetRowStrength(PRESETS, 'Soft', 0, '0.35')[1].loras[0].strength, 0.35,
    'an input event carries a STRING; a preset full of strings is a preset the '
    + 'server has to guess about');
});

test('an impossible write returns the list untouched instead of guessing', () => {
  for (const args of [['', 0, 0.5], ['Gone', 0, 0.5], ['Soft', 9, 0.5],
    ['Soft', -1, 0.5], ['Soft', 0, 'abc']]) {
    assert.equal(withPresetRowStrength(PRESETS, ...args), PRESETS,
      `${JSON.stringify(args)} should have changed nothing`);
  }
});

test('the presets ride in the same partial PUT as the rest of the panel', () => {
  const next = withPresetRowStrength(PRESETS, 'Soft', 0, 0.4);
  assert.deepEqual(improveSettingsPatch({ presets: next }),
    { config: { klein: { generation_lora_presets: next } } },
    'no identity_prompts section: a slider must not rewrite the instruction');
  // Coalesced with a keystroke, exactly as the saver merges them.
  assert.deepEqual(improveSettingsPatch({ prompt: 'x', presets: next }), {
    config: {
      identity_prompts: { klein_improve: 'x' },
      klein: { generation_lora_presets: next },
    },
  });
});

test('a patch with no presets key touches no preset', () => {
  assert.equal('klein' in improveSettingsPatch({ prompt: 'x' }).config, false);
  for (const bad of [null, 'nope', 42, undefined]) {
    const cfg = improveSettingsPatch({ presets: bad }).config;
    assert.equal(cfg.klein?.generation_lora_presets, undefined,
      'a malformed list must never be PUT over a real one');
  }
});

test('a strength stored out of range is DRAWN at what the pass will actually run', () => {
  // A hand-edited config, or a number copied from the Krea card whose ceiling
  // is 6: the engine clamps to 1.5, and a panel showing "3.00" beside the
  // picture would be naming a strength nothing uses.
  const rows = presetChainRows([{ name: 'x', loras: [
    { file: 'a.safetensors', strength: 3 }, { file: 'b.safetensors', strength: -2 },
  ] }], 'x');
  assert.deepEqual(rows.map((r) => r.strength), [PRESET_LORA_STRENGTH_MAX, 0]);
  assert.equal(clampPresetStrength('0.35'), 0.35);
  assert.equal(clampPresetStrength('nope'), 0.6);
});

test('rows are labelled by the part that differs, and keep the path when it does not', () => {
  // `truncate` cuts from the END, so at 360 px two files of one folder became
  // the same ellipsis. The filename is what tells them apart…
  const one = presetChainRows([{ name: 'x', loras: [
    { file: 'klein/very-long-name-alpha.safetensors', strength: 0.5 },
    { file: 'klein/very-long-name-beta.safetensors', strength: 0.5 },
  ] }], 'x');
  assert.deepEqual(one.map((r) => r.label),
    ['very-long-name-alpha.safetensors', 'very-long-name-beta.safetensors']);
  // …unless the filename is what they SHARE, in which case the folder is.
  const two = presetChainRows([{ name: 'x', loras: [
    { file: 'klein/detail.safetensors', strength: 0.5 },
    { file: 'flux\\detail.safetensors', strength: 0.5 },
  ] }], 'x');
  assert.deepEqual(two.map((r) => r.label),
    ['klein/detail.safetensors', 'flux\\detail.safetensors']);
  // The stored value is never rewritten by how it is displayed.
  assert.deepEqual(one.map((r) => r.file),
    ['klein/very-long-name-alpha.safetensors', 'klein/very-long-name-beta.safetensors']);
});

test('storing what is already stored is not a write', () => {
  // The caller schedules a save on whatever comes back, so the identity check
  // IS the guard against a request spent on nothing.
  assert.equal(withPresetRowStrength(PRESETS, 'Soft', 0, 0.9), PRESETS);
  assert.equal(withPresetRowStrength(PRESETS, 'Soft', 0, '0.9'), PRESETS);
  const maxed = [{ name: 'Soft', loras: [{ file: 'a.safetensors', strength: 1.5 }] }];
  assert.equal(withPresetRowStrength(maxed, 'Soft', 0, 9), maxed,
    'a value clamped ONTO the one already stored is still nothing to store');
  assert.notEqual(withPresetRowStrength(PRESETS, 'Soft', 0, 0.85), PRESETS);
});
