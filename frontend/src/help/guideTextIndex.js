/* Index a guide chapter's markdown BY H2 SECTION so the Settings search can
   match a setting on the words of its tutorial, not just its label/keywords —
   e.g. typing "crop" surfaces the watermark auto-crop setting because the doc
   section that explains it mentions the word.

   PURE JS: it only splits text and computes anchors with the shared
   markdownHeadingId, so node --test can exercise it directly. SettingsPage
   imports docs/guide/settings-reference.md as ?raw and builds the index once. */
import { markdownHeadingId } from '../utils/headingId.js';

/** Parse markdown into [{ anchor, title, text }] — one entry per H2 section.
    `text` is the lowercased body of the section (headings + prose), so callers
    can do a plain substring match. Content before the first H2 is ignored. */
export function buildGuideTextIndex(markdown) {
  const lines = String(markdown || '').replace(/\r\n/g, '\n').split('\n');
  const sections = [];
  let current = null;
  let inFence = false;
  for (const line of lines) {
    if (line.startsWith('```')) inFence = !inFence;
    const h2 = !inFence && line.match(/^##\s+(.+)$/);
    if (h2) {
      const title = h2[1].replace(/[`*_]/g, '').trim();
      current = { anchor: markdownHeadingId(h2[1]), title, buf: [title] };
      sections.push(current);
      continue;
    }
    if (current) current.buf.push(line);
  }
  return sections.map(({ anchor, title, buf }) => ({
    anchor, title, text: buf.join('\n').toLowerCase(),
  }));
}

/** Set of anchors whose section text contains the (case-insensitive) query.
    Empty query → empty set. */
export function matchGuideAnchors(index, query) {
  const q = String(query || '').trim().toLowerCase();
  const hits = new Set();
  if (!q) return hits;
  for (const s of index) {
    if (s.text.includes(q)) hits.add(s.anchor);
  }
  return hits;
}
