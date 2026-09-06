import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { PROMPT_STORAGE, readPromptDraft, writePromptDraft } from './videoPromptDraft.js';

const memory = () => {
  const map = new Map();
  return { getItem: (k) => (map.has(k) ? map.get(k) : null), setItem: (k, v) => map.set(k, String(v)),
    removeItem: (k) => map.delete(k), map };
};

test('the motion comes back as typed, line breaks included', () => {
  const storage = memory();
  const typed = 'She turns her head and smiles…\n[Shot 2] At 00:05.000, the camera cuts to a close-up.';
  writePromptDraft(typed, storage);
  assert.equal(readPromptDraft(storage), typed);
  assert.equal(JSON.parse(storage.map.get(PROMPT_STORAGE)).prompt, typed);
});

test('a cleared field clears the draft, so a reload does not resurrect an old prompt', () => {
  const storage = memory();
  writePromptDraft('she turns her head', storage);
  writePromptDraft('', storage);
  assert.equal(storage.map.has(PROMPT_STORAGE), false);
  assert.equal(readPromptDraft(storage), '');
});

test('nothing saved, a malformed entry or a non-string prompt read as an empty field', () => {
  assert.equal(readPromptDraft(memory()), '');
  const broken = memory();
  broken.setItem(PROMPT_STORAGE, '{not json');
  assert.equal(readPromptDraft(broken), '');
  const wrongShape = memory();
  wrongShape.setItem(PROMPT_STORAGE, JSON.stringify({ prompt: ['not', 'text'] }));
  assert.equal(readPromptDraft(wrongShape), '');
  const unrelated = memory();
  unrelated.setItem(PROMPT_STORAGE, JSON.stringify('a bare string'));
  assert.equal(readPromptDraft(unrelated), '');
});

test('a browser without storage, or one that throws, neither crashes nor blocks the take', () => {
  assert.equal(readPromptDraft(null), '');
  assert.doesNotThrow(() => writePromptDraft('typed', null));
  const throwing = { getItem: () => { throw new Error('SecurityError'); }, setItem: () => { throw new Error('QuotaExceededError'); },
    removeItem: () => { throw new Error('SecurityError'); } };
  assert.equal(readPromptDraft(throwing), '');
  assert.doesNotThrow(() => writePromptDraft('typed', throwing));
  assert.doesNotThrow(() => writePromptDraft('', throwing));
  assert.doesNotThrow(() => writePromptDraft(undefined, memory()));
});

test('the studio reads the draft once at mount and writes the field on every change', () => {
  const src = readFileSync(new URL('./VideoTestStudio.jsx', import.meta.url), 'utf8');
  assert.match(src, /import \{ readPromptDraft, writePromptDraft \} from '\.\/videoPromptDraft'/);
  // The lazy initializer: React calls it with no argument, so it reads
  // globalThis.localStorage.
  assert.match(src, /const \[prompt, setPrompt\] = useState\(readPromptDraft\)/);
  assert.match(src, /writePromptDraft\(prompt\)[\s\S]{0,40}\[prompt\]\)/);
  assert.doesNotMatch(src, /const \[prompt, setPrompt\] = useState\(''\)/);
});
