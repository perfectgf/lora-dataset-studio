// Copies a text that is still being built without losing the click that asked for it.
//
// Firefox (Safari too) lets a page write the clipboard only during the user activation
// a click grants, about five seconds in Firefox. `await build()` then `writeText()`
// lands after it whenever building is slow, and Firefox refuses: "Clipboard write was
// blocked due to lack of user activation" (issue #77). A ClipboardItem whose content is
// still a promise reaches the browser at once, inside the click, and the browser fills
// it in when the text is ready (Firefox 127+, Chrome, Safari).
//
// Call it synchronously from the click handler, before any await.
export function copyWhenReady(textPromise) {
  const clipboard = globalThis.navigator?.clipboard;
  const text = Promise.resolve(textPromise);
  if (clipboard?.write && typeof globalThis.ClipboardItem === 'function') {
    const blob = text.then((value) => new Blob([value], { type: 'text/plain' }));
    return clipboard.write([new globalThis.ClipboardItem({ 'text/plain': blob })]);
  }
  // No ClipboardItem: writeText once the text exists, the most such a browser allows.
  // No clipboard at all (a LAN page over plain http is not a secure context): reject,
  // and the caller shows the text to copy by hand.
  return text.then((value) => {
    if (!clipboard?.writeText) throw new Error('This page has no clipboard access.');
    return clipboard.writeText(value);
  });
}
