import test from 'node:test';
import assert from 'node:assert/strict';

import {
  MAX_NAME, MAX_SUMMARY, PREVIEWS_FORMAT, memberPreviewsSnapshot, parseMemberPreviews, previewsAt,
  refreshMemberPreviews,
} from './memberPreviews.js';

const JOIN = 'https://www.patreon.com/c/example';
const item = (extra = {}) => ({ id: 'krea_enhancer', placement: 'studio.options.krea',
  name: 'Krea2T prompt enhancer', summary: 'One sentence.', url: JOIN, ...extra });
const parse = (...items) => parseMemberPreviews({ format: PREVIEWS_FORMAT, items }).items;

test('a preview carries text and a Patreon link only', () => {
  assert.deepEqual(parse(item()), [item()]);
});

test('an entry with any other key is left out', () => {
  for (const extra of [{ route: '/api/x' }, { html: '<b>x</b>' }, { script: 'x' }, { enabled: true }]) {
    assert.deepEqual(parse(item(extra)), [], Object.keys(extra)[0]);
  }
});

test('an entry without a link on https://www.patreon.com is left out', () => {
  for (const url of [undefined, 'http://www.patreon.com/c/x', 'https://patreon.com.example.org/c/x',
    'https://example.org/c/x', 'javascript:alert(1)', 'https://www.patreon.com:8443/c/x',
    'https://user@www.patreon.com/c/x', 'https://www.patreon.com/c/x#frag', 42]) {
    assert.deepEqual(parse(item({ url })), [], String(url));
  }
});

test('texts are plain single lines with capped lengths', () => {
  assert.deepEqual(parse(item({ name: 'x'.repeat(MAX_NAME + 1) })), []);
  assert.deepEqual(parse(item({ summary: 'x'.repeat(MAX_SUMMARY + 1) })), []);
  for (const bad of ['two\nlines', 'tab\there', `bidi${String.fromCodePoint(0x202e)}text`,
    `sep${String.fromCodePoint(0x2028)}x`, '   ']) {
    assert.deepEqual(parse(item({ summary: bad })), [], JSON.stringify(bad));
  }
  assert.deepEqual(parse(item({ id: 'Bad-Id' })), []);
});

test('an unknown place or format shows nothing; each place shows its own entries', () => {
  assert.deepEqual(parse(item({ placement: 'settings.later' })), []);
  assert.deepEqual(parseMemberPreviews({ format: 'lds-member-previews/2', items: [item()] }).items, []);
  const items = parse(item(), item({ id: 'studio_remake', placement: 'studio.options' }),
    item({ id: 'subject_crop', placement: 'bank.crop' }));
  assert.deepEqual(previewsAt(items, ['studio.options', 'studio.options.zimage']).map((i) => i.id), ['studio_remake']);
  assert.deepEqual(previewsAt(items, ['bank.crop']).map((i) => i.id), ['subject_crop']);
});

test('the screens show what the server answers, and nothing once a read fails', async () => {
  await refreshMemberPreviews(async () => ({ format: PREVIEWS_FORMAT, items: [item()] }));
  assert.equal(memberPreviewsSnapshot().items.length, 1);
  await refreshMemberPreviews(async () => { throw new Error('offline'); });
  assert.deepEqual(memberPreviewsSnapshot().items, []);
});
