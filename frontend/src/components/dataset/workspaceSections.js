// Data-driven section list for the dataset workspace sidebar — same pattern as
// the Settings rail (components/settings/registry.js): stable ids (deep-linked
// through ?section=), icon + title for the nav, and a mono eyebrow tag + short
// description for the section header. The list is identical for every dataset
// kind; only the CONTENT of a section branches on kind (e.g. "Add images" is
// reference+generation for a character, scraping+import for a concept/style).

export const WORKSPACE_SECTIONS = [
  { id: 'images', title: 'Images', icon: '🖼️', eyebrow: 'overview',
    description: 'Everything in the dataset — keep ✓ the good shots, reject ✕ the rest, edit a caption right on its tile.',
    panels: [
      { id: 'review', title: 'Review images', targetId: 'ds-images-review', when: 'always' },
      { id: 'bulk', title: 'Bulk actions', targetId: 'ds-images-bulk', when: 'hasSelectableImages' },
    ] },
  { id: 'add', title: 'Add images', icon: '📸', eyebrow: 'build',
    description: 'Generate AI variations from the reference — and mix in real photos (import or scrape).',
    conceptDescription: 'Scrape galleries or drop photos — a concept LoRA learns from real images.',
    panels: [
      { id: 'reference', title: 'Reference photo', targetId: 'ds-add-reference', when: 'character' },
      { id: 'generate', title: 'Generate variations', targetId: 'ds-add-generate', when: 'character' },
      { id: 'import', title: 'Import photos', targetId: 'ds-add-import', when: 'always' },
      { id: 'scraper', title: 'Scraper', targetId: 'ds-add-scraper', when: 'always',
        reveal: 'scraper', focusSelector: 'input[type="url"]' },
    ] },
  { id: 'curation', title: 'Curation', icon: '🧹', eyebrow: 'quality',
    description: 'Quality passes over the kept images — face resemblance, watermark find & clean, cleanup.',
    panels: [
      { id: 'small-image-rescue', title: 'Klein rescue review', targetId: 'ds-curation-small-image-rescue', when: 'smallImageRescue' },
      { id: 'face-analysis', title: 'Face analysis', targetId: 'ds-curation-face-analysis', when: 'character' },
      { id: 'watermarks', title: 'Watermarks', targetId: 'ds-curation-watermarks', when: 'always' },
      { id: 'review-flagged', title: 'Review flagged', targetId: 'ds-curation-review-flagged', when: 'watermarkDetected' },
      { id: 'rejected-cleanup', title: 'Rejected cleanup', targetId: 'ds-curation-rejected-cleanup', when: 'unused' },
    ] },
  { id: 'captions', title: 'Captions', icon: '✍️', eyebrow: 'text',
    description: 'Captions are what training reads each image by — generate them, watch for leaks, edit in bulk.',
    panels: [
      { id: 'generate', title: 'Generate captions', targetId: 'ds-captions-generate', when: 'always' },
      { id: 'leak-review', title: 'Leak review', targetId: 'ds-captions-leak-review', when: 'leakReview', reveal: 'caption-leak' },
      { id: 'tools', title: 'Caption tools', targetId: 'ds-captions-tools', when: 'hasCaptionedKept', reveal: 'caption-tools' },
    ] },
  { id: 'export', title: 'Import & export', icon: '📦', eyebrow: 'data',
    description: 'Merge an existing dataset in — or get this one out: training ZIP, portable backup, Hugging Face.',
    panels: [
      { id: 'import', title: 'Import dataset', targetId: 'ds-export-import', when: 'always' },
      { id: 'training-zip', title: 'Export training ZIP', targetId: 'ds-export-training-zip', when: 'always' },
      { id: 'backup', title: 'Portable backup', targetId: 'ds-export-backup', when: 'always' },
      { id: 'hugging-face', title: 'Publish to Hugging Face', targetId: 'ds-export-hugging-face', when: 'huggingFace' },
    ] },
  { id: 'training', title: 'Training', icon: '🎓', eyebrow: 'train',
    description: 'Turn the kept & captioned images into a LoRA — locally or in the cloud.',
    panels: [
      { id: 'launch', title: 'Training status & launch', targetId: 'ds-training-launch', when: 'always' },
      { id: 'advanced', title: 'Advanced options', targetId: 'ds-training-advanced', when: 'trainingVisible', reveal: 'training-advanced' },
      { id: 'queue', title: 'Training queue', targetId: 'ds-training-queue', when: 'trainingQueue' },
    ] },
  { id: 'checkpoints', title: 'Checkpoints & LoRAs', icon: '📦', eyebrow: 'results',
    description: 'Review training saves, pick an epoch, import LoRAs into ComfyUI, and clean up old files.',
    panels: [
      { id: 'manager', title: 'Checkpoint manager', targetId: 'ds-checkpoints-manager', when: 'trainingVisible' },
    ] },
  { id: 'studio', title: 'Studio', icon: '🎛️', eyebrow: 'test',
    description: 'Test the trained LoRA with saved winning settings in the dedicated Studio.',
    panels: [
      { id: 'launcher', title: 'LoRA testing studio', targetId: 'ds-studio-launcher', when: 'studioVisible' },
    ] },
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
  'gf-checkpoints': 'checkpoints',
  'gf-studio': 'studio',
};

export function isWorkspaceSection(id) {
  return WORKSPACE_SECTIONS.some((s) => s.id === id);
}
