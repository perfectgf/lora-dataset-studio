// Data-driven section list for the dataset workspace sidebar — same pattern as
// the Settings rail (components/settings/registry.js): stable ids (deep-linked
// through ?section=), icon + title for the nav, and a mono eyebrow tag + short
// description for the section header. The list is identical for every dataset
// kind; only the CONTENT of a section branches on kind (e.g. "Add images" is
// reference+generation for a character, scraping+import for a concept/style).

export const WORKSPACE_SECTIONS = [
  { id: 'images', title: 'Images', icon: '🖼️', eyebrow: 'overview',
    description: 'Everything in the dataset — keep ✓ the good shots, reject ✕ the rest, edit a caption right on its tile.' },
  { id: 'add', title: 'Add images', icon: '📸', eyebrow: 'build',
    description: 'Generate AI variations from the reference — and mix in real photos (import or scrape).',
    conceptDescription: 'Scrape galleries or drop photos — a concept LoRA learns from real images.' },
  { id: 'curation', title: 'Curation', icon: '🧹', eyebrow: 'quality',
    description: 'Quality passes over the kept images — face resemblance, watermark find & clean, cleanup.' },
  { id: 'captions', title: 'Captions', icon: '✍️', eyebrow: 'text',
    description: 'Captions are what training reads each image by — generate them, watch for leaks, edit in bulk.' },
  { id: 'export', title: 'Import & export', icon: '📦', eyebrow: 'data',
    description: 'Merge an existing dataset in — or get this one out: training ZIP, portable backup, Hugging Face.' },
  { id: 'training', title: 'Training', icon: '🎓', eyebrow: 'train',
    description: 'Turn the kept & captioned images into a LoRA — locally or in the cloud.' },
];

// Which section hosts each jump anchor (gf-*). Consumed by jumpTo: switch to
// the section first, then scroll to the anchor inside it. Covers the guided
// checklist targets AND the backend preflight "Fix →" targets (gf-generate,
// gf-images — see lora_training.py).
export const SECTION_FOR_TARGET = {
  'gf-reference': 'add',
  'gf-generate': 'add',
  'gf-images': 'images',
  'gf-curation': 'curation',
  'gf-captions': 'captions',
  'gf-export': 'export',
  'gf-training': 'training',
};

export function isWorkspaceSection(id) {
  return WORKSPACE_SECTIONS.some((s) => s.id === id);
}
