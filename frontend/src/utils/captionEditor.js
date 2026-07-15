export function isCaptionSaveShortcut(event) {
  return event?.key === 'Enter' && Boolean(event.ctrlKey || event.metaKey);
}

export function captionCharacterLabel(caption) {
  const count = String(caption || '').length;
  return `${count} character${count === 1 ? '' : 's'}`;
}
