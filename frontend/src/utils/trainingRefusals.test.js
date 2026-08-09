/* The confirm-and-resubmit contract shared by every launch/relaunch lane.

   Written after GitHub #23 (1Tomber): ↻ Retry posted, the server refused with a
   confirmable UNCAPTIONED: 400, and the click produced NOTHING — no toast, no
   dialog, no job, only an uncaught promise rejection in the console. A refusal
   the user cannot see is worse than a refusal, so these tests pin the two halves
   of the contract: a confirmable refusal must ASK, and anything else must be
   rethrown so the caller can SAY it. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import {
  CONFIRMABLE_REFUSALS,
  RETRY_CONFIRMABLE_REFUSALS,
  confirmableRetryFlag,
  postWithConfirmations,
} from './trainingRefusals.js';

/* window.confirm is the user's answer; scripted per test. Async on purpose: a
   sync `finally` would restore window BEFORE an awaited body ever asked. */
async function withConfirm(answers, run) {
  const asked = [];
  const previous = globalThis.window;
  globalThis.window = {
    confirm: (message) => {
      asked.push(message);
      return answers.length ? answers.shift() : false;
    },
  };
  try {
    return await run(asked);
  } finally {
    globalThis.window = previous;
  }
}

const refusal = (message) => {
  const e = new Error(message);
  e.status = 400;
  return e;
};

const UNCAPTIONED = 'UNCAPTIONED: 1 kept image(s) have no caption (including '
  + 'whitespace). Captions are strongly recommended — confirm explicitly to train anyway.';

// --- the reported scenario, end to end -------------------------------------

test('a confirmed UNCAPTIONED refusal is resubmitted with the force flag', async () => {
  await withConfirm([true], async (asked) => {
    const sent = [];
    const post = async (body) => {
      sent.push(body);
      if (!body.allow_uncaptioned) throw refusal(UNCAPTIONED);
      return { ok: true, started: true };
    };
    const out = await postWithConfirmations(post, { record_id: 42 },
      'Retry anyway (force)', RETRY_CONFIRMABLE_REFUSALS);
    assert.deepEqual(out, { ok: true, started: true });
    assert.deepEqual(sent, [
      { record_id: 42 },
      { record_id: 42, allow_uncaptioned: true },
    ]);
    // the dialog showed the server's sentence WITHOUT the machine marker
    assert.equal(asked.length, 1);
    assert.ok(!asked[0].includes('UNCAPTIONED:'));
    assert.ok(asked[0].includes('1 kept image(s) have no caption'));
    assert.ok(asked[0].endsWith('Retry anyway (force)?'));
  });
});

test('declining a confirm returns null — the decline IS the answer, not an error', async () => {
  await withConfirm([false], async () => {
    const post = async () => { throw refusal(UNCAPTIONED); };
    assert.equal(
      await postWithConfirmations(post, { record_id: 1 }, 'Retry anyway (force)'),
      null);
  });
});

// --- the anti-silence half --------------------------------------------------

test('a NON-confirmable refusal is rethrown so the caller can surface it', async () => {
  await withConfirm([], async (asked) => {
    const post = async () => { throw refusal('a training is already in progress'); };
    await assert.rejects(
      () => postWithConfirmations(post, {}, 'Retry anyway (force)',
        RETRY_CONFIRMABLE_REFUSALS),
      /already in progress/);
    assert.deepEqual(asked, []);          // no dialog for a non-confirmable refusal
  });
});

test('a server that IGNORES the confirmed flag is reported, never re-asked forever', async () => {
  // The infinite-dialog trap: an endpoint that does not read the flag would
  // return the same marker for ever. One confirm, then the truth.
  await withConfirm([true, true, true], async (asked) => {
    let calls = 0;
    const post = async () => { calls += 1; throw refusal(UNCAPTIONED); };
    await assert.rejects(
      () => postWithConfirmations(post, {}, 'Retry anyway (force)',
        RETRY_CONFIRMABLE_REFUSALS),
      /no caption/);
    assert.equal(calls, 2);               // original + one confirmed resubmit
    assert.equal(asked.length, 1);
  });
});

test('several guards in sequence each get their own confirm', async () => {
  await withConfirm([true, true], async (asked) => {
    const post = async (body) => {
      if (!body.allow_uncaptioned) throw refusal(UNCAPTIONED);
      if (!body.allow_caption_mismatch) {
        // Kept byte-representative of what the backend actually emits: the
        // family name is interpolated (`this {label} dataset`), so a Klein user
        // reads "FLUX.2 Klein", not a frozen "Z-Image". The parser under test
        // keys on the `MISMATCH_CAPTION: ` PREFIX alone — see the assertion
        // below, which is what makes this fixture's wording free to track the
        // backend instead of pinning it.
        throw refusal('MISMATCH_CAPTION: this FLUX.2 Klein dataset has booru TAG captions.');
      }
      return { ok: true };
    };
    const out = await postWithConfirmations(post, {}, 'Retry anyway (force)',
      RETRY_CONFIRMABLE_REFUSALS);
    assert.deepEqual(out, { ok: true });
    assert.equal(asked.length, 2);
  });
});

test('a mismatch refusal is recognised by its PREFIX, whatever family it names', async () => {
  // The backend interpolates the family label into this message. If the parser
  // ever keyed on a family name instead of the marker, adding a family would
  // silently turn a confirmable refusal into a hard error — so pin the
  // independence rather than the sentence.
  for (const family of ['Z-Image', 'FLUX.2 Klein', 'Krea 2', 'SDXL', '']) {
    await withConfirm([true], async () => {
      const post = async (body) => {
        if (!body.allow_caption_mismatch) {
          throw refusal(`MISMATCH_CAPTION: this ${family} dataset has booru TAG captions.`);
        }
        return { ok: true };
      };
      const out = await postWithConfirmations(post, {}, 'Retry anyway (force)',
        RETRY_CONFIRMABLE_REFUSALS);
      assert.deepEqual(out, { ok: true }, family);
    });
  }
});

// --- the resolving-client dialect -------------------------------------------
/* hooks/useDataset's postJson NEVER throws: a refusal resolves as
   {ok:false, error}. Fed to a loop that only listened for rejections, a 409
   PARALLEL_RUN: refusal sailed through as a success — the dataset panel
   toasted "Cloud run created — follow its launch on the Runs page" while the
   server had refused and created nothing (reported with screenshots, one
   POST → 409 in the log and no retry). The loop must speak both dialects. */

test('a refusal RESOLVED as {ok:false} still asks and resubmits with the flag', async () => {
  await withConfirm([true], async (asked) => {
    const sent = [];
    const post = async (body) => {
      sent.push(body);
      if (!body.allow_parallel_run) {
        return { ok: false, error: 'PARALLEL_RUN: this dataset already has an active krea cloud run (#150) — launching another one rents a second pod, billed separately. Launch anyway?' };
      }
      return { ok: true, run_id: 151 };
    };
    const out = await postWithConfirmations(post, { steps: 3250 }, 'Train anyway (force)');
    assert.deepEqual(out, { ok: true, run_id: 151 });
    assert.deepEqual(sent, [
      { steps: 3250 },
      { steps: 3250, allow_parallel_run: true },
    ]);
    assert.equal(asked.length, 1);
    assert.ok(!asked[0].includes('PARALLEL_RUN:'));
  });
});

test('a NON-confirmable {ok:false} resolution is thrown, never returned as a success', async () => {
  await withConfirm([], async (asked) => {
    const post = async () => ({ ok: false, error: 'cloud run limit reached (4/4 active)' });
    await assert.rejects(
      () => postWithConfirmations(post, {}, 'Train anyway (force)'),
      /limit reached/);
    assert.deepEqual(asked, []);
  });
});

test('declining a resolved-dialect confirm returns null too', async () => {
  await withConfirm([false], async () => {
    const post = async () => ({ ok: false, error: 'PARALLEL_RUN: sibling #150 …' });
    assert.equal(await postWithConfirmations(post, {}, 'Train anyway (force)'), null);
  });
});

// --- the guard family -------------------------------------------------------

test('the retry lane can answer every pre-flight guard Start can answer', () => {
  const flags = RETRY_CONFIRMABLE_REFUSALS.map(([, flag]) => flag);
  assert.deepEqual(flags.slice().sort(), [
    'allow_caption_mismatch',
    'allow_caption_quality',
    // Full-model cloud runs only: the private Hugging Face storage forecast.
    // It rides the shared list because the ceiling it compares against is an
    // ESTIMATE — every lane that can start a run must be able to say "anyway".
    'allow_hf_storage',
    // ... and its twin about this machine's disk, for the same reason: the
    // delivery lands here first and its size is an estimate.
    'allow_local_disk',
    'allow_not_ready',
    // A second cloud run of the same family on the same dataset — the retry
    // lane's sibling can hit this too (auto-retry replays it).
    'allow_parallel_run',
    'allow_uncaptioned',
    'allow_unverified_weights',
  ]);
});

test('the readiness floor stays OUT of the shared list (the panel has its own checkbox)', () => {
  // Adding NOT_READY to the shared list would make the dataset panel ask twice
  // for something its preparation card already asks once.
  assert.ok(!CONFIRMABLE_REFUSALS.some(([marker]) => marker.startsWith('NOT_READY')));
  assert.ok(RETRY_CONFIRMABLE_REFUSALS.some(([marker]) => marker.startsWith('NOT_READY')));
});

test('confirmableRetryFlag defaults to the shared list when none is passed', async () => {
  await withConfirm([true], () => {
    assert.equal(confirmableRetryFlag(UNCAPTIONED, 'Train anyway'), 'allow_uncaptioned');
    assert.equal(confirmableRetryFlag('NOT_READY: only 3 kept image(s)', 'Train anyway'), null);
  });
});

// --- the call site ----------------------------------------------------------

const page = fs.readFileSync(new URL('../pages/CloudRunsPage.jsx', import.meta.url), 'utf8');

test('every mutating handler of the Runs hub says something when it is refused', () => {
  // fetchClient's postJson REJECTS on a 400 and shows nothing of its own for
  // that status, so a handler without a catch is a button that does nothing.
  // These three had `try { … } finally { … }` (or no try at all) before #23.
  for (const handler of [
    /const retry = async \(run\) => \{[\s\S]*?\n  \};/,
    /const stop = async \(run\) => \{[\s\S]*?\n  \};/,
  ]) {
    const body = page.match(handler)?.[0];
    assert.ok(body, `handler not found: ${handler}`);
    assert.match(body, /\} catch \(e\) \{[\s\S]*?toast\.error\(/);
  }
  // the global 🧹 purge posts inside a try whose catch toasts
  assert.match(page, /cloud\/purge', \{\}\);[\s\S]{0,600}?\} catch \(e\) \{\s*\n\s*toast\.error\(/);
});

test('retry answers confirmable refusals through the shared loop, not a bare post', () => {
  const body = page.match(/const retry = async \(run\) => \{[\s\S]*?\n  \};/)[0];
  assert.match(body, /postWithConfirmations\(/);
  assert.match(body, /RETRY_CONFIRMABLE_REFUSALS/);
  assert.match(body, /'Retry anyway \(force\)'/);
  // a decline must return quietly — not fall through to the success toast
  assert.match(body, /if \(!d\) return;/);
  // and the dead `d.ok === false` branch is gone: this client throws, never
  // returns an ok:false envelope
  assert.ok(!/d\.ok === false/.test(body));
});
