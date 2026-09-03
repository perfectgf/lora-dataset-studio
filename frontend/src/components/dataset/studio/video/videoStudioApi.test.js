import test from 'node:test';
import assert from 'node:assert/strict';

import {
  ACCELERATIONS, accelLabel, clipAccel, pickAvailableAccel,
  mergeClipPages,
  smoothTargets,
  buildGeneratePayload, clipSeconds, clipSummary, isRunning, launchAdviceLines, renderTimeLabel,
  SPARSE_CHOICES, studioFrameChoices,
}  from './videoStudioApi.js';

test('an option left off is absent from the payload, never false', () => {
  const body = buildGeneratePayload({ mode: 'i2v', prompt: ' she turns ', image: 'a.png' });
  assert.equal(body.prompt, 'she turns');
  assert.equal(body.image, 'a.png');
  for (const key of ['turbo', 'eros', 'sparse', 'latent_upscale', 'lora']) {
    assert.ok(!(key in body), `${key} should not be sent when it is off`);
  }
});

test('t2v drops the start image and keeps the aspect instead', () => {
  const body = buildGeneratePayload({
    mode: 't2v', prompt: 'a street at night', image: 'left-over.png',
    ratio: 1.77, aspect: 'portrait',
  });
  assert.ok(!('image' in body), 't2v must not carry a start frame');
  assert.ok(!('ratio' in body));
  assert.equal(body.aspect, 'portrait');
});

test('a LoRA carries its strength and its provenance', () => {
  const body = buildGeneratePayload({
    mode: 'i2v', prompt: 'p', image: 'a.png',
    lora: 'h3/lds/jessy.safetensors', loraStrength: 1.3, runId: 174, datasetId: 8,
  });
  assert.equal(body.lora, 'h3/lds/jessy.safetensors');
  assert.equal(body.lora_strength, 1.3);
  assert.equal(body.run_id, 174);
  assert.equal(body.dataset_id, 8);
});

test('seed 0 is sent — it is a seed, not an empty field', () => {
  const body = buildGeneratePayload({ mode: 't2v', prompt: 'p', seed: 0 });
  assert.equal(body.seed, 0);
  assert.ok(!('seed' in buildGeneratePayload({ mode: 't2v', prompt: 'p', seed: '' })));
});

test('every sparse choice is a level the server accepts', () => {
  // The server normalises anything it does not know to OFF, silently — which is
  // the right server behaviour and the wrong thing to discover from a render.
  const accepted = new Set(['', 'default', 'conservative', 'max']);
  for (const c of SPARSE_CHOICES) {
    assert.ok(accepted.has(c.value), `unknown sparse level "${c.value}"`);
    assert.ok(c.label && c.hint, `sparse level "${c.value}" needs a label and a hint`);
  }
});

test('clip length counts intervals, not frames', () => {
  assert.equal(clipSeconds(121, 24), 5);       // the lane's own cross-check
  assert.equal(clipSeconds(0, 24), null);
  assert.equal(clipSeconds(56, 0), null);
});

test('the summary names what differed and stays quiet about what did not', () => {
  const line = clipSummary({
    lora: 'h3\\lds\\jessy_2000.safetensors', lora_strength: 1.3, turbo: true,
    sparse: 'conservative', steps: 6, seed: 42, latent_upscale: false, eros: false,
  });
  assert.match(line, /jessy_2000 @ 1\.3/);
  assert.match(line, /⚡ turbo/);
  assert.match(line, /sparse conservative/);
  assert.ok(!line.includes('upscale'), 'an option that was off must not be listed');
  assert.ok(!line.includes('10Eros'));
  assert.match(clipSummary({ steps: 20, seed: 1 }), /no LoRA/);
});

test('running is one predicate', () => {
  assert.equal(isRunning({ status: 'pending' }), true);
  assert.equal(isRunning({ status: 'done' }), false);
  assert.equal(isRunning(null), false);
});

// --- the sampling steps, once they became reachable (2026-09-01) --------------

test('an explicit step count travels; auto sends nothing at all', () => {
  // The server has always accepted `steps` and always let it win over turbo's
  // own six — the panel simply never offered the dial, so the one number that
  // trades time for fidelity was the one nobody could turn.
  const withSteps = buildGeneratePayload({
    mode: 't2v', prompt: 'a street', turbo: true, steps: 12,
  })
  assert.equal(withSteps.steps, 12)
  assert.equal(withSteps.turbo, true)
  // Auto is the ABSENCE of the key: the server then applies the count for the
  // mode in force, and nothing here claims a choice nobody made.
  const auto = buildGeneratePayload({ mode: 't2v', prompt: 'a street', turbo: true, steps: '' })
  assert.equal('steps' in auto, false)
})

// --- the studio's own clip lengths, not training's (2026-09-01) ---------------

test('the length list reaches the model, not the training catalogue', () => {
  // The dropdown was built from `frame_choices` — the TRAINING ladder, which
  // stops at 209 frames (8.67s) because that is where training lengths stop
  // being useful. The server has always accepted up to 362 (15.04s) and says
  // so in its own comment; the list on screen was the wrong table.
  const l = studioFrameChoices({ frames_min: 22, frames_max: 362 })
  assert.equal(l[0], 22)
  assert.equal(l[l.length - 1], 362)
  // 362 frames at 24 fps is 15.04s — the model's own reach.
  assert.equal(clipSeconds(362, 24), 15.04)
  // Every rung is legal for H3's VAE: 17 pixel frames per chunk, so ≡ 5 mod 17.
  assert.ok(l.every((f) => f % 17 === 5))
  // No duplicates, ascending.
  assert.deepEqual([...l].sort((a, b) => a - b), l)
  assert.equal(new Set(l).size, l.length)
})

test('the length list falls back rather than inventing lengths', () => {
  const fallback = studioFrameChoices({ frame_choices: [39, 56] })
  assert.ok(fallback.length > 0)
  assert.ok(fallback.every((f) => f % 17 === 5 || [39, 56].includes(f)))
})

test('the launch carries the enrich flag only when it is asked for', () => {
  // ✨ Enrich at launch is done SERVER-side so the clip records what actually
  // ran; the payload's job is only to say whether it was asked for.
  const on = buildGeneratePayload({ mode: 't2v', prompt: 'she turns', enhance: true })
  assert.equal(on.enhance, true)
  const off = buildGeneratePayload({ mode: 't2v', prompt: 'she turns' })
  assert.equal('enhance' in off, false)
})

test('the launch advice phrases exactly what the server sent, flag names included', () => {
  const base = { flag: '--fast-disk', ram_total_gb: 47.7, weights_gb: 43 }
  const add = launchAdviceLines({ ...base, add: true, remove: null })
  assert.equal(add.title, 'ComfyUI is running without --fast-disk')
  assert.match(add.action, /^Add --fast-disk on the command that starts ComfyUI, then start it again\.$/)

  const both = launchAdviceLines({ ...base, add: true, remove: '--disable-dynamic-vram' })
  assert.match(both.title, /running with --disable-dynamic-vram, which switches off the loader --fast-disk relies on/)
  assert.match(both.action, /^Remove --disable-dynamic-vram and add --fast-disk on the command/)

  // The flag already on the line: the card must not ask to add it again.
  const only = launchAdviceLines({ ...base, add: false, remove: '--disable-dynamic-vram' })
  assert.match(only.action, /^Remove --disable-dynamic-vram \(--fast-disk is already on the command line\) on the command/)
  assert.doesNotMatch(only.action, /and add/)

  // Nothing sent, nothing said.
  assert.equal(launchAdviceLines(null), null)
  assert.equal(launchAdviceLines({}), null)
})

test('the render time reads the way a person says it, and is null for anything else', () => {
  assert.equal(renderTimeLabel(24.4), '24 s')
  assert.equal(renderTimeLabel(59.6), '1 min')          // rounds to 60, and 60 is a minute
  assert.equal(renderTimeLabel(348.03), '5 min 48 s')
  assert.equal(renderTimeLabel(120), '2 min')
  assert.equal(renderTimeLabel(0.4), '1 s')             // a measured fraction is rounded up, never hidden
  assert.equal(renderTimeLabel(3600), '1 h')
  assert.equal(renderTimeLabel(5400), '1 h 30 min')
  assert.equal(renderTimeLabel(28800), '8 h')
  assert.equal(renderTimeLabel(3661), '1 h 1 min')       // seconds drop past the hour
  for (const junk of [null, undefined, 0, -3, 'abc', NaN, Infinity]) {
    assert.equal(renderTimeLabel(junk), null, String(junk))
  }
})

test('smooth offers whole factors of the source rate, with frames and relative cost', () => {
  const t = smoothTargets({ fps: 24, frames: 124 });
  assert.deepEqual(t.map((x) => [x.multiplier, x.fps, x.frames, x.cost]),
    [[2, 48, 248, 1], [3, 72, 372, 2], [4, 96, 496, 3]]);
  // A clip that never stored its rate is an H3 clip: 24 fps authored.
  assert.deepEqual(smoothTargets({}).map((x) => x.fps), [48, 72, 96]);
  assert.equal(smoothTargets({ fps: 30 })[0].frames, null, 'no frame count → no count promised');
  assert.equal(smoothTargets({ fps: 30 })[1].fps, 90);
});

test('a poll keeps the loaded older clips: the boundary is the page proper, not a source that rode along', () => {
  const clip = (id, extra = {}) => ({ id, ...extra });
  // Older pages the user asked for, down to clip 40 — 41 is the clip just smoothed.
  const prev = [79, 78, 77, 75, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40].map((id) => clip(id));
  // The first page proper is 79..50; 41 rides along because 79 was smoothed from it.
  const fresh = [79, 78, 77, 75, 50, 41].map((id) => clip(id, id === 79 ? { vfi_of: 41 } : {}));
  const kept = mergeClipPages(prev, fresh, 50).map((c) => c.id);
  assert.deepEqual(kept, [79, 78, 77, 75, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40]);
  // What the old boundary (the oldest id ON the page: 41) did — every loaded clip between the two vanished.
  assert.deepEqual(mergeClipPages(prev, fresh, 41).map((c) => c.id), [79, 78, 77, 75, 50, 41, 40]);
  // A row deleted inside the page proper (78, between 79 and the boundary 50) leaves with the
  // page; a clip older than the boundary (30) is kept, the fresh page never carried it.
  assert.deepEqual(mergeClipPages([clip(79), clip(78), clip(30)], [clip(79), clip(50)], 50).map((c) => c.id), [79, 50, 30]);
});

test('the acceleration travels by name, and larryvrh keeps the older boolean beside it', () => {
  assert.deepEqual(ACCELERATIONS.map((a) => a.id), ['turbo', 'parasyte', 'dareties']);
  const base = { prompt: 'p', mode: 't2v' };
  assert.equal(buildGeneratePayload({ ...base, accel: 'parasyte' }).accel, 'parasyte');
  assert.equal(buildGeneratePayload({ ...base, accel: 'parasyte' }).turbo, undefined);
  assert.equal(buildGeneratePayload({ ...base, accel: 'turbo' }).turbo, true);
  assert.equal(buildGeneratePayload({ ...base, turbo: true }).turbo, true, 'a caller without the name still speaks the flag');
  assert.equal(buildGeneratePayload({ ...base }).accel, undefined);
  assert.equal(clipAccel({ accel: 'dareties', turbo: false }), 'dareties');
  assert.equal(clipAccel({ turbo: true }), 'turbo', 'a row older than the choice');
  assert.equal(clipAccel({}), '');
  assert.equal(accelLabel('parasyte'), 'Parasyte Turbo');
  assert.match(clipSummary({ accel: 'dareties', steps: 6 }), /⚡ DARE-TIES merge/);
  assert.match(clipSummary({ turbo: true, steps: 6 }), /⚡ turbo/);
});

test('the pick follows what the machine holds: unavailable falls to the first available, unknown stays', () => {
  const rows = [{ id: 'turbo', available: false }, { id: 'parasyte', available: true }, { id: 'dareties', available: false }];
  assert.equal(pickAvailableAccel('turbo', rows), 'parasyte');
  assert.equal(pickAvailableAccel('parasyte', rows), 'parasyte');
  assert.equal(pickAvailableAccel('turbo', [{ id: 'turbo', available: null }]), 'turbo', 'a probe that could not run is not a no');
  assert.equal(pickAvailableAccel('turbo', [{ id: 'turbo', available: false }]), '', 'nothing available: the dense base');
  assert.equal(pickAvailableAccel('', rows), '');
});

test('a continuation travels in the launch and reads on the summary', () => {
  assert.equal(buildGeneratePayload({ mode: 'i2v', prompt: 'p', image: 'a.png', continues: 41 }).continues, 41)
  assert.equal(buildGeneratePayload({ mode: 'i2v', prompt: 'p', image: 'a.png' }).continues, undefined)
  assert.match(clipSummary({ continues_of: 41, steps: 6 }), /⏭ continues #41/)
})
