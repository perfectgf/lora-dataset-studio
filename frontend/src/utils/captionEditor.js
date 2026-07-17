export function isCaptionSaveShortcut(event) {
  return event?.key === 'Enter' && Boolean(event.ctrlKey || event.metaKey);
}

export function captionCharacterLabel(caption) {
  const count = String(caption || '').length;
  return `${count} character${count === 1 ? '' : 's'}`;
}

// The ceiling an older backend hard-sliced stored captions at — mid-word, mid-sentence.
// Kept only to RECOGNISE those legacy truncations (the current backend caps far higher
// and cuts on a sentence boundary), so the UI can nudge the user to re-caption.
export const LEGACY_CAPTION_CEILING = 800;

// True when a caption looks like a legacy truncation: exactly 800 characters AND not
// ending on sentence-final punctuation (so it was almost certainly cut mid-thought).
// The lost text can't be recovered — re-captioning the image is the repair path.
export function isLikelyTruncatedCaption(caption) {
  const text = String(caption || '');
  if (text.length !== LEGACY_CAPTION_CEILING) return false;
  return !/[.!?]["'”’)\]]?$/.test(text.trimEnd());
}
