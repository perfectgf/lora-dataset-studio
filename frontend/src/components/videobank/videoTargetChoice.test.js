import test from 'node:test'
import assert from 'node:assert/strict'

import {
  frameOptions, clipSeconds, frameOptionLabel, defaultFrames, needsManualFrames,
  sizeOptions, targetWarnings, targetBadge, promoteProblem, promotePayload,
  datasetScaleNote,
  promoteScopeLabel,
  insetProblem, insetHint, insetOutcome,
  capProblem, capHint, capBalanceNote,
  lengthSuggestion, lengthSuggestionNote,
} from './videoTargetChoice.js'

// Verbatim shapes of GET /api/video/targets — kept as fixtures rather than
// invented, so a catalogue change that breaks the picker breaks a test.
const WAN14B = {
  key: 'wan22_14b', label: 'Wan 2.1 / 2.2 14B', fps: 16,
  frame_choices: [17, 25, 33, 49, 65, 81, 97, 121], frame_default: 81,
  default_seconds: 5, size_multiple: 16,
  recommended_sizes: [[832, 480], [480, 832], [1280, 720], [720, 1280]],
  keep_audio: false, caption_style: 'freeform',
  training_verified: true, licence_note: null,
}
const TI2V5B = {
  key: 'wan22_ti2v5b', label: 'Wan 2.2 TI2V-5B', fps: 24,
  frame_choices: [25, 49, 81, 97, 121], frame_default: 121, default_seconds: 5,
  size_multiple: 32, recommended_sizes: [[1280, 704], [704, 1280]],
  keep_audio: false, caption_style: 'freeform',
  training_verified: false, licence_note: null,
}
const MINIMAX = {
  key: 'minimax_h3', label: 'MiniMax H3', fps: 24,
  frame_choices: [39, 56, 73, 90, 107, 124], frame_default: 124,
  size_multiple: 32, recommended_sizes: [[1344, 768]],
  keep_audio: true, caption_style: 'paragraph_with_audio',
  training_verified: false,
  licence_note: 'MiniMax H3 Community License grants NO rights in the EU, UK, '
    + 'South Korea or USA — and the restriction covers the outputs, not just '
    + 'the model. Check your territory first.',
}
const GENERIC = {
  key: 'generic', label: 'Generic / other', fps: null, frame_choices: [],
  frame_default: null, size_multiple: null, recommended_sizes: [],
  keep_audio: false, caption_style: 'freeform',
  training_verified: false, licence_note: null,
}

// ---- the two fields that save a week ---------------------------------------

test('a target with no known trainer SAYS SO at the picker', () => {
  // Exactly one of the four clears this bar. Someone who discovers it after
  // cutting and captioning has lost the week.
  const w = targetWarnings(TI2V5B)
  assert.ok(w.some((x) => x.key === 'unverified'))
  assert.match(w.find((x) => x.key === 'unverified').text, /No LoRA trainer is known/)
  assert.equal(targetBadge(TI2V5B).text, 'Not trainable yet')
})

test('the trainable target carries no scare text', () => {
  assert.deepEqual(targetWarnings(WAN14B), [])
  assert.equal(targetBadge(WAN14B).text, 'Trainable')
})

test('the MiniMax licence names the excluded territories AND the outputs', () => {
  // The restriction reaching the outputs is the part people get wrong — keeping
  // the training private is not a way around it.
  const licence = targetWarnings(MINIMAX).find((w) => w.key === 'licence')
  assert.ok(licence, 'the licence note must be surfaced')
  assert.equal(licence.tone, 'danger')
  for (const territory of ['EU', 'UK', 'South Korea', 'USA']) {
    assert.ok(licence.text.includes(territory), `territory ${territory} not named`)
  }
  assert.match(licence.text, /outputs/)
})

test('the licence outranks the missing trainer in the warning order', () => {
  // A licence that grants nothing where you live costs more than a trainer that
  // does not exist yet, and both apply to MiniMax.
  const keys = targetWarnings(MINIMAX).map((w) => w.key)
  assert.ok(keys.indexOf('licence') < keys.indexOf('unverified'))
})

test('a joint audio-video target says the soundtrack is kept', () => {
  assert.ok(targetWarnings(MINIMAX).some((w) => w.key === 'audio'))
  assert.ok(!targetWarnings(WAN14B).some((w) => w.key === 'audio'))
})

// ---- lengths: the menu, never a seconds field -------------------------------

test('the length options come from frame_choices and nowhere else', () => {
  assert.deepEqual(frameOptions(WAN14B).map((o) => o.frames), WAN14B.frame_choices)
  assert.deepEqual(frameOptions(MINIMAX).map((o) => o.frames), MINIMAX.frame_choices)
})

test('seconds are DISPLAYED per option, computed at the target’s own rate', () => {
  // The cross-check that (frames-1)/fps is the right arithmetic: both Wan
  // variants land on exactly 5.00 s at their own rate — 81 @ 16 and 121 @ 24.
  assert.equal(clipSeconds(WAN14B, 81), 5)
  assert.equal(clipSeconds(TI2V5B, 121), 5)
  assert.equal(frameOptionLabel(WAN14B, 81), '81 frames — 5.00s')
})

test('a target with no fps shows frames alone rather than a wrong duration', () => {
  assert.equal(clipSeconds(GENERIC, 60), null)
  assert.equal(frameOptionLabel(GENERIC, 60), '60 frames')
})

test('a catalogue with no verified lengths asks for a frame count, not seconds', () => {
  // "Generic / other" imposes nothing. Falling back to Wan's menu here would
  // offer 4n+1 lengths for a model nobody said anything about.
  assert.deepEqual(frameOptions(GENERIC), [])
  assert.equal(needsManualFrames(GENERIC), true)
  assert.equal(needsManualFrames(WAN14B), false)
  assert.equal(defaultFrames(GENERIC), null)
  assert.match(promoteProblem({ name: 'x', target: GENERIC, frames: null }),
    /type a frame count/)
})

test('the default length is the catalogue’s own', () => {
  assert.equal(defaultFrames(WAN14B), 81)
  assert.equal(defaultFrames(TI2V5B), 121)
})

// ---- sizes ------------------------------------------------------------------

test('“keep the source size” is always the first size option', () => {
  const opts = sizeOptions(WAN14B)
  assert.equal(opts[0].key, 'source')
  assert.equal(opts[0].width, null)
  assert.deepEqual(opts.slice(1).map((o) => o.key),
    ['832x480', '480x832', '1280x720', '720x1280'])
})

test('a target with no recommended sizes still offers the source size', () => {
  assert.deepEqual(sizeOptions(GENERIC).map((o) => o.key), ['source'])
})

// ---- the payload ------------------------------------------------------------

test('the source size sends NO width and NO height', () => {
  const body = promotePayload({
    name: ' wan set ', targetKey: 'wan22_14b', frames: 81,
    size: { width: null, height: null }, ids: [],
  })
  assert.deepEqual(body, { name: 'wan set', target_profile: 'wan22_14b', frames: 81 })
})

test('a lone width is never sent — the server reads the PAIR or nothing', () => {
  // width without height is silently ignored server-side, so the user would get
  // a source-size cut while the dialog said 832 wide.
  const body = promotePayload({
    name: 'x', targetKey: 'wan22_14b', frames: 81, size: { width: 832, height: null },
  })
  assert.ok(!('width' in body) && !('height' in body))
})

test('an EMPTY selection omits ids, because empty means "every kept clip"', () => {
  // The server's convention. Sending `ids: []` and expecting "nothing" would
  // promote the whole bank.
  assert.ok(!('ids' in promotePayload({ name: 'x', targetKey: 'k', frames: 81, ids: [] })))
  assert.deepEqual(
    promotePayload({ name: 'x', targetKey: 'k', frames: 81, ids: [3, 7] }).ids, [3, 7])
})

test('the button says which clips it is about to encode', () => {
  assert.equal(promoteScopeLabel(0, 128), 'all 128 kept clips')
  assert.equal(promoteScopeLabel(1, 128), '1 selected clip')
  assert.equal(promoteScopeLabel(12, 128), '12 selected clips')
  assert.equal(promoteScopeLabel(0, 1), 'all 1 kept clip')
})

// ---- refusals ---------------------------------------------------------------

test('the dialog refuses what the server would refuse, before the round trip', () => {
  assert.match(promoteProblem({ name: '  ', target: WAN14B, frames: 81 }), /Name the dataset/)
  assert.match(promoteProblem({ name: 'x', target: null, frames: 81 }), /Pick a target/)
  assert.match(promoteProblem({ name: 'x', target: WAN14B, frames: null }), /Pick a clip length/)
  assert.match(promoteProblem({ name: 'x', target: GENERIC, frames: 12.5 }),
    /whole number of frames/)
  assert.equal(promoteProblem({ name: 'x', target: WAN14B, frames: 81 }), null)
})

test('the source-size option warns when the target caps the canvas area', () => {
  // 1920x1088 is a clean multiple of 32 and still out of spec for MiniMax H3 —
  // "keep the source's size" is the one path that can smuggle it through, so it
  // carries the warning at the point of choice. The backend refuses too; this
  // hint is what keeps the refusal from being a surprise.
  const h3 = { key: 'minimax_h3', recommended_sizes: [[1344, 768]],
               max_pixels: 1032192 }
  const source = sizeOptions(h3).find((o) => o.key === 'source')
  assert.match(source.hint, /cap|larger|rescal/i)
})

test('no canvas cap, no hint on the source-size option', () => {
  const source = sizeOptions(WAN14B).find((o) => o.key === 'source')
  assert.equal(source.hint, undefined)
})

// --- the edge inset (wave 4) --------------------------------------------------

test('no inset means the payload does not mention one', () => {
  // Default zero is the whole point: an existing recipe must export exactly what
  // it exported yesterday, and a body carrying edge_inset_s: 0 invites a future
  // reader to "clean it up" to a non-zero default.
  const body = promotePayload({ name: 'a', targetKey: 'wan22_14b', frames: 81 });
  assert.ok(!('edge_inset_s' in body));
  assert.ok(!('edge_inset_s' in promotePayload({
    name: 'a', targetKey: 'wan22_14b', frames: 81, edgeInsetS: 0 })));
});

test('an inset rides in the payload in seconds', () => {
  const body = promotePayload({ name: 'a', targetKey: 'wan22_14b', frames: 81,
    edgeInsetS: 0.25 });
  assert.equal(body.edge_inset_s, 0.25);
});

test('a nonsense inset is refused before the request, with a reason', () => {
  // A negative inset would EXTEND every clip into the neighbouring shot. The
  // server refuses it too; catching it here is what puts the reason next to the
  // field instead of in a red banner after a round trip.
  assert.match(insetProblem(-0.1), /negative|extend/i);
  assert.match(insetProblem(60), /too (long|much)|cap/i);
  assert.equal(insetProblem(0), '');
  assert.equal(insetProblem(0.25), '');
  assert.equal(insetProblem(''), '');
});

test('the inset explains what it costs before the click', () => {
  assert.equal(insetHint(0), '');
  assert.match(insetHint(0.25), /0\.5\s*s/);       // 0.25 off EACH end
  assert.match(insetHint(0.25), /drop|short/i);
});

test('the promotion report names the clips the inset itself removed', () => {
  // "Too short for 81 frames" and "too short once your trim was applied" are the
  // same sentence to a user and two different problems — only the second is
  // fixed by lowering the knob.
  const note = insetOutcome({ edge_inset_s: 0.25, inset_would_drop: 7 });
  assert.match(note, /7/);
  assert.match(note, /0\.25/);
  assert.equal(insetOutcome({ edge_inset_s: 0.25, inset_would_drop: 0 }), '');
  assert.equal(insetOutcome({ edge_inset_s: 0, inset_would_drop: 0 }), '');
});

// --- the per-source cap (wave 4 finishing) ------------------------------------

test('no cap means the payload does not mention one', () => {
  const body = promotePayload({ name: 'a', targetKey: 'wan22_14b', frames: 81 });
  assert.ok(!('max_per_source' in body));
  assert.ok(!('max_per_source' in promotePayload({
    name: 'a', targetKey: 'wan22_14b', frames: 81, maxPerSource: '' })));
});

test('a cap rides in the payload as a whole number', () => {
  const body = promotePayload({ name: 'a', targetKey: 'wan22_14b', frames: 81,
    maxPerSource: '4' });
  assert.equal(body.max_per_source, 4);
});

test('a cap that is not a whole number of clips is refused with a reason', () => {
  // Refused here as well as on the server, so the reason sits next to the field
  // rather than arriving as a banner after a round trip.
  assert.match(capProblem(0), /at least 1|empty/i);
  assert.match(capProblem(-2), /at least 1|empty/i);
  assert.match(capProblem(2.5), /whole/i);
  assert.match(capProblem('lots'), /number/i);
  assert.equal(capProblem(''), '');
  assert.equal(capProblem(3), '');
});

test('the cap hint says which clips it keeps, because it is not a sample', () => {
  // The service keeps each source's EARLIEST clips — stable and explainable.
  // A user who assumes "a random pick" gets a different dataset than they think.
  assert.match(capHint(), /earliest|first/i);
});

test('a lopsided result suggests the cap, with the real share', () => {
  // The composition comes back AFTER the job starts, so this cannot be a
  // pre-flight suggestion — it is what the result says about what just happened.
  const note = capBalanceNote({ sources: 3, top_source_share: 0.6 }, null);
  assert.match(note, /60%/);
  assert.match(note, /cap|max/i);
});

test('a balanced result says nothing, and neither does an already capped one', () => {
  assert.equal(capBalanceNote({ sources: 5, top_source_share: 0.25 }, null), '');
  // Already capped: the user made the call, repeating the advice is nagging.
  assert.equal(capBalanceNote({ sources: 3, top_source_share: 0.6 }, 2), '');
  // One source is not an imbalance, it is the whole bank.
  assert.equal(capBalanceNote({ sources: 1, top_source_share: 1 }, null), '');
});

test('exact sizes lead the size menu and say so; stated sizes name their cost', () => {
  const target = {
    max_pixels: 768 * 1344,
    exact_sizes: [[1024, 576], [768, 768]],
    recommended_sizes: [[1344, 768], [768, 768]],
  }
  const labels = sizeOptions(target).map((s) => s.label)
  assert.ok(labels[1].includes('1024 × 576 — trains exactly as cut'))
  assert.ok(labels.some((l) => l.includes("model's stated size")))
  // the duplicate (768x768) appears once, as exact
  assert.equal(labels.filter((l) => l.includes('768 × 768')).length, 1)
  // a target with no exact sizes keeps its plain labels — no cost invented
  const plain = sizeOptions({ recommended_sizes: [[1280, 704]] })
  assert.equal(plain[1].label, '1280 × 704')
})

test('the dataset scale note tells a novice where their clip count sits', () => {
  assert.equal(datasetScaleNote(0), null)
  assert.equal(datasetScaleNote('nope'), null)
  assert.equal(datasetScaleNote(5).tone, 'warning')
  assert.ok(datasetScaleNote(12).text.includes('50-200'))
  assert.equal(datasetScaleNote(80).tone, 'good')
  assert.ok(datasetScaleNote(300).text.includes('curation'))
})


// --- what a clip length costs, from the shots actually kept (2026-09-01) ------------

test('lengthSuggestion counts how many kept shots survive each length', () => {
  // 24 fps: 39 frames = 1.583s, 209 frames = 8.667s.
  const spans = [2, 3, 4, 5, 9, 10]
  const options = [{ frames: 39 }, { frames: 107 }, { frames: 209 }]
  const s = lengthSuggestion(spans, options, 24)
  assert.equal(s.total, 6)
  // Every shot clears 39 frames; only the two long ones clear 209.
  assert.equal(s.rows.find((r) => r.frames === 39).fits, 6)
  assert.equal(s.rows.find((r) => r.frames === 209).fits, 2)
  // The LONGEST length that still keeps 80% — 107 frames (4.42s) keeps 3/6, so
  // the answer here is 39.
  assert.equal(s.frames, 39)
})

test('lengthSuggestion prefers the longest length that still keeps most shots', () => {
  const spans = [9, 9.5, 10, 11, 12]
  const s = lengthSuggestion(spans, [{ frames: 39 }, { frames: 209 }], 24)
  assert.equal(s.frames, 209)          // all five clear 8.67s
  assert.equal(s.share, 1)
})

test('lengthSuggestion says nothing rather than something wrong', () => {
  assert.equal(lengthSuggestion([], [{ frames: 39 }], 24), null)
  assert.equal(lengthSuggestion([1, 2], [], 24), null)
  assert.equal(lengthSuggestion([1, 2], [{ frames: 39 }], 0), null)
  assert.equal(lengthSuggestion(null, null, null), null)
})

test('the note compares the CURRENT choice with the suggestion', () => {
  const s = lengthSuggestion([9, 9.5, 10, 11, 12], [{ frames: 39 }, { frames: 209 }], 24)
  // Same choice as the suggestion: one sentence, no comparison.
  assert.match(lengthSuggestionNote(s, 209), /209 frames/)
  assert.doesNotMatch(lengthSuggestionNote(s, 209), /your 209/)
  // A different choice: what it costs, next to it.
  const note = lengthSuggestionNote(s, 39)
  assert.match(note, /your 39 frames/)
  assert.match(note, /100%/)
  assert.equal(lengthSuggestionNote(null, 39), '')
})
