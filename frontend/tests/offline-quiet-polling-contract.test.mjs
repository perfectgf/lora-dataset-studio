/* Contract: the network layer stays quiet about background polls, the toast
 * queue merges repeats, and the offline state is rendered somewhere.
 *
 * These are source assertions on purpose. The logic itself is unit-tested
 * (utils/toastQueue.test.js, utils/connectionStatus.test.js,
 * components/bank/progressPresence.test.js) — what THIS file protects is the
 * WIRING, which lives in JSX `node --test` cannot execute. A rewrite of
 * Toast.jsx that quietly goes back to `[...prev, toast]` would pass every unit
 * test and re-ship the ten stacked banners.
 */
import test from 'node:test';
import { readSource } from './support/readSource.mjs'
import assert from 'node:assert/strict';

const read = (rel) => readSource(`src/${rel}`)

test('apiFetch accepts a background flag and strips it from the fetch init', () => {
  const s = read('api/fetchClient.js');
  assert.match(s, /const \{ background = false, \.\.\.init \} = options/,
    'background must be destructured out — it is not a valid RequestInit key');
  assert.match(s, /fetchWithCsrfRetry\(url, init\)/);
});

test('a network failure goes through the connection store, not straight to a toast', () => {
  const s = read('api/fetchClient.js');
  assert.match(s, /if \(reportRequestFailure\(\{ background \}\)\) toastRef\?\.error\(/,
    'the toast must be gated on the store deciding this failure is worth announcing');
  assert.doesNotMatch(s, /catch \{\s*toastRef\?\.error\('Connection lost/,
    'the unconditional per-failure banner must be gone');
});

test('recovery is announced — including by the poll that noticed it', () => {
  const s = read('api/fetchClient.js');
  assert.match(s, /if \(reportRequestSuccess\(\)\) toastRef\?\.success\(CONNECTION_BACK_MESSAGE\)/);
  // Gating this on `!background` looks symmetric with the failure side and is
  // wrong: nobody clicks during an outage, so the poll is the ONLY thing that
  // ever sees the server come back. reportRequestSuccess() already fires on
  // exactly one edge per episode, so there is nothing here to spam.
  assert.doesNotMatch(s, /reportRequestSuccess\(\)[\s\S]{0,80}!background/);
});

test('an unhappy ANSWER to a background poll is as quiet as no answer at all', () => {
  const s = read('api/fetchClient.js');
  // The status branches below were reached whether or not the request was a
  // poll: a container answering 503 while it boots produced one "Server error"
  // toast per tick, which is the exact stack of banners this whole feature
  // exists to remove.
  assert.match(s, /\} else if \(background\) \{[\s\S]{0,600}?\} else if \(res\.status === 401\)/,
    'the background short-circuit must come BEFORE the status toasts');
  // …but after the CSRF branch, which only rewrites the thrown message.
  assert.match(s, /res\.status === 400 && !parsed[\s\S]{0,400}?\} else if \(background\)/,
    'a background CSRF 400 must still get the actionable message');
});

test('the toast provider merges through the queue instead of appending blindly', () => {
  const s = read('components/common/Toast.jsx');
  assert.match(s, /from '\.\.\/\.\.\/utils\/toastQueue'/);
  assert.match(s, /pushToast\(prev, \{ id, message, type, expiresAt \}\)/);
  assert.doesNotMatch(s, /setToasts\(\(prev\) => \[\.\.\.prev,/,
    'the naive append is what stacked ten identical banners');
});

test('the repeat counter is aria-hidden so a merged banner is announced once', () => {
  const s = read('components/common/Toast.jsx');
  assert.match(s, /<span aria-hidden="true"[\s\S]{0,220}\{t\.count\}×/,
    'a visible counter inside the live region would re-announce on every repeat');
});

test('the app renders one persistent offline indicator', () => {
  assert.match(read('App.jsx'), /<ConnectionBanner \/>/);
  const banner = read('components/common/ConnectionBanner.jsx');
  assert.match(banner, /role="status"/, 'must be announceable');
  assert.match(banner, /Offline — reconnecting…/);
  // No timer/counter in the banner: its text must be stable while it is up,
  // or the live region re-announces on every tick.
  assert.doesNotMatch(banner, /setInterval|offlineSince/);
});

test('the bank progress zone knows about the offline state', () => {
  const s = read('components/bank/BankWorkspace.jsx');
  // The bar itself is BankProgress.jsx now; the workspace still passes it the
  // connection state, which is the property this test protects.
  assert.match(read('components/bank/BankProgress.jsx'), /progressPresence\(activity, offline\)/);
  assert.match(s, /<ProgressBar activity=\{payload\?\.activity\} onCancel=\{cancelJob\} offline=\{!connection\.online\}/);
});

test('the polls that fire on a timer are marked background', () => {
  const marked = [
    // The bank's 2 s tick now asks /activity for the job instead of re-fetching
    // the whole workspace payload (which took 12.5 s on a 50 000-image bank and
    // took the Stop button down with it). BOTH calls it can make on a timer are
    // pinned here — the property has not moved, only the URL it fires at.
    ['components/bank/BankWorkspace.jsx',
      /apiFetch\(`\/api\/bank\/\$\{bankId\}\/activity`, \{ background: true \}\)/],
    ['components/bank/BankWorkspace.jsx', /refreshPayload\(\{ background: true \}\)/],
    ['hooks/useTrainingActivity.js', /'\/api\/train\/activity', \{ background: true \}/],
    ['components/settings/MaintenanceSection.jsx', /background: true/],
    // The ChatGPT device-code poll. It used to live inside EnginesSection; it
    // moved out whole when the Setup wizard started offering the same login, so
    // the property is pinned where the setInterval actually is.
    ['components/common/ChatgptSubscriptionConnect.jsx', /background: true/],
    ['components/dataset/CaptionOptionsPopover.jsx', /background: true/],
    ['App.jsx', /update\/check\?auto=1', \{ background: true \}/],
  ];
  for (const [file, re] of marked) assert.match(read(file), re, `${file} poll not marked background`);
});

test('the bank dashboard refresh is on a wall clock, not a per-tick counter', () => {
  // The poll effect lists the job's `detail` in its dependencies, and a running
  // pass rewrites `detail` every couple of seconds — so the effect re-subscribes
  // constantly. A counter declared inside it would reset before ever reaching N
  // and the bank-wide counts would then NEVER refresh while a job ran. Caught in
  // review of this very change; pinned so it cannot come back as a "simpler" N.
  const s = read('components/bank/BankWorkspace.jsx');
  assert.match(s, /const fullPayloadAt = useRef\(0\)/,
    'the full-refresh deadline must outlive the effect');
  assert.match(s, /Date\.now\(\) - fullPayloadAt\.current >= FULL_PAYLOAD_MS/);
  assert.doesNotMatch(s, /ticks % FULL_PAYLOAD/,
    'a tick counter inside the effect is reset by every detail change');
});
