/* Slugify a markdown heading into a stable DOM id / anchor.
   Extracted from components/common/Markdown.jsx so it can be shared by the help
   registry and imported from node --test WITHOUT pulling in JSX or Vite's ?raw
   loader. Markdown.jsx re-exports it (GuidePage imports it from there — that
   compat path is preserved). Keep the algorithm byte-for-byte: the help
   registry's anchors are computed against exactly this function, and the
   contract test recomputes chapter anchors with it — any drift breaks both. */
export const markdownHeadingId = (text) => String(text || '')
  .replace(/[`*_]/g, '')
  .toLocaleLowerCase()
  .replace(/[^a-z0-9]+/g, '-')
  .replace(/^-|-$/g, '');
