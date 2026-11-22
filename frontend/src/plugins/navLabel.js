/** The words of a plugin's menu entry.
 *
 * Plugins declare their entry with a pictograph in front ('📷 Camera angles',
 * '◉ Canvas', '✨ DLSS 5'). In a row of five that was a flourish; in a menu
 * that lists every page of every installed plugin it is a column of unrelated
 * glyphs — each from a different emoji font, each a different colour — beside
 * workspaces that carry a drawn icon. The menu therefore shows the words only.
 *
 * Only the DISPLAY changes: the label a plugin declared is not rewritten, so
 * its What's-new entries, its card and its own page keep whatever they say.
 * Anything that is not a plain string (a plugin that hands over an element) is
 * returned untouched, and so is a label made of symbols only — a menu entry
 * with no words would be worse than one with a glyph. */
export function plainNavLabel(label) {
  if (typeof label !== 'string') return label
  // Everything before the first letter or digit: pictographs, their variation
  // selectors and joiners, dingbats, the space that follows them.
  const words = label.replace(/^[^\p{L}\p{N}]+/u, '').trim()
  return words || label
}
