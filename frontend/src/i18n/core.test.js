import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { formatMessage, getMessage, translate } from './core.js';

const en = JSON.parse(readFileSync(new URL('./locales/en.json', import.meta.url), 'utf8'));
const zhCN = JSON.parse(readFileSync(new URL('./locales/zh-CN.json', import.meta.url), 'utf8'));

function flatten(catalog, prefix = '', out = {}) {
  for (const [key, value] of Object.entries(catalog)) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (value && typeof value === 'object') flatten(value, path, out);
    else out[path] = String(value);
  }
  return out;
}

test('getMessage resolves nested language-pack keys', () => {
  assert.equal(getMessage({ nav: { datasets: 'Datasets' } }, 'nav.datasets'), 'Datasets');
  assert.equal(getMessage({}, 'nav.datasets'), undefined);
});

test('formatMessage interpolates named values and preserves unknown placeholders', () => {
  assert.equal(formatMessage('{count} items for {name}', { count: 3, name: 'Ada' }), '3 items for Ada');
  assert.equal(formatMessage('Hello {name}', {}), 'Hello {name}');
});

test('translate falls back to English and finally to the key', () => {
  const en = { common: { save: 'Save' } };
  assert.equal(translate({}, en, 'common.save'), 'Save');
  assert.equal(translate({}, en, 'missing.key'), 'missing.key');
});

test('English and Simplified Chinese packs contain the same keys and placeholders', () => {
  const english = flatten(en);
  const chinese = flatten(zhCN);
  assert.deepEqual(Object.keys(chinese).sort(), Object.keys(english).sort());
  for (const key of Object.keys(english)) {
    const placeholders = (text) => [...text.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort();
    assert.deepEqual(placeholders(chinese[key]), placeholders(english[key]), key);
  }
});
