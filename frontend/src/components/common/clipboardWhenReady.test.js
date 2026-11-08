import test from 'node:test';
import assert from 'node:assert/strict';
import { copyWhenReady } from './clipboardWhenReady.js';

// Swap in a fake browser clipboard for one test, then put the real globals back.
async function withBrowser({ clipboard, ClipboardItem }, run) {
  const navigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  const item = Object.getOwnPropertyDescriptor(globalThis, 'ClipboardItem');
  Object.defineProperty(globalThis, 'navigator', { value: { clipboard }, configurable: true, writable: true });
  Object.defineProperty(globalThis, 'ClipboardItem', { value: ClipboardItem, configurable: true, writable: true });
  try {
    await run();
  } finally {
    for (const [name, saved] of [['navigator', navigator], ['ClipboardItem', item]]) {
      if (saved) Object.defineProperty(globalThis, name, saved);
      else delete globalThis[name];
    }
  }
}

test('the clipboard write starts inside the click, before the report exists (#77)', async () => {
  const writes = [];
  class FakeItem { constructor(data) { this.data = data; } }
  await withBrowser({ clipboard: { write: async (items) => { writes.push(items); } }, ClipboardItem: FakeItem }, async () => {
    let finish;
    const report = new Promise((resolve) => { finish = resolve; });
    const copied = copyWhenReady(report);
    // Firefox checks the click's activation when write() is CALLED: it must already be.
    assert.equal(writes.length, 1);
    const [item] = writes[0];
    assert.ok(item instanceof FakeItem);
    assert.equal(typeof item.data['text/plain'].then, 'function', 'the content is still a promise');
    finish('LDS diagnostic report');
    await copied;
    assert.equal(await (await item.data['text/plain']).text(), 'LDS diagnostic report');
  });
});

test('a page without clipboard access rejects, so the report is shown to copy by hand', async () => {
  await withBrowser({ clipboard: undefined, ClipboardItem: undefined }, async () => {
    await assert.rejects(copyWhenReady(Promise.resolve('report')), /no clipboard access/);
  });
});
