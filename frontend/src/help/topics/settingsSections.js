/* One section of the help registry, moved verbatim (2026-08-24 split).
   ORDER MATTERS inside and across sections: helpRegistry.js concatenates
   the six section arrays in a fixed order, and for a given (chapter,
   anchor) the FIRST topic owns the "Open this screen →" button. */
export const SETTINGS_SECTION_TOPICS = [
  // ---- Settings: section-level topics (kind 'section') --------------------
  { id: 'settings-how', kind: 'section', title: 'How settings work',
    keywords: ['settings', 'how', 'save', 'secret', 'write-only', 'restart', 'config'],
    guide: { chapter: 'settings-reference', anchor: 'how-settings-work' },
    app: { route: '/settings' } },
  { id: 'settings-overview', kind: 'section', title: 'Settings · Overview',
    keywords: ['overview', 'status', 'summary', 'capabilities', 'ready', 'configured'],
    guide: { chapter: 'settings-reference', anchor: 'overview' },
    app: { route: '/settings/overview' } },
  { id: 'settings-engines', kind: 'section', title: 'Settings · Image engines',
    keywords: ['engine', 'engines', 'generation', 'gemini', 'openai', 'chatgpt', 'klein', 'nano banana', 'api key', 'lora', 'preset'],
    guide: { chapter: 'settings-reference', anchor: 'image-engines' },
    app: { route: '/settings/engines' } },
  { id: 'settings-scraping', kind: 'section', title: 'Settings · Scraping & sources',
    keywords: ['scraping', 'sources', 'reddit', 'civitai', 'pexels', 'scrape', 'import', 'rate limit', '429'],
    guide: { chapter: 'settings-reference', anchor: 'scraping-sources' },
    app: { route: '/settings/scraping' } },
  { id: 'settings-local-tools', kind: 'section', title: 'Settings · Local tools',
    keywords: ['local tools', 'comfyui', 'ollama', 'ai-toolkit', 'aitoolkit', 'integrations', 'path', 'url', 'hugging face'],
    guide: { chapter: 'settings-reference', anchor: 'local-tools' },
    app: { route: '/settings/local-tools' } },
  { id: 'settings-captioning', kind: 'section', title: 'Settings · Captioning & quality',
    keywords: ['captioning', 'quality', 'caption', 'joycaption', 'face score', 'threshold', 'watermark', 'similarity', 'bank', 'triage'],
    guide: { chapter: 'settings-reference', anchor: 'captioning-quality' },
    app: { route: '/settings/captioning' } },
  { id: 'settings-training', kind: 'section', title: 'Settings · Training',
    keywords: ['training', 'family', 'cloud', 'vast', 'gpu', 'budget', 'price', 'stall'],
    guide: { chapter: 'settings-reference', anchor: 'training' },
    app: { route: '/settings/training' } },
  { id: 'settings-server', kind: 'section', title: 'Settings · Server & access',
    keywords: ['server', 'access', 'port', 'lan', 'network', 'token', 'remote', 'phone'],
    guide: { chapter: 'settings-reference', anchor: 'server-access' },
    app: { route: '/settings/server' } },
  { id: 'settings-storage', kind: 'section', title: 'Settings · Storage',
    keywords: ['storage', 'disk', 'space', 'disk full', 'drive', 'another drive', 'move folder',
      'relocate', 'path', 'location', 'where are my files', 'c drive full', 'trash',
      'archive', 'checkpoint store', 'staging', 'cloud runs', 'free space', 'gb'],
    guide: { chapter: 'settings-reference', anchor: 'storage' },
    app: { route: '/settings/storage' } },
  { id: 'settings-maintenance', kind: 'section', title: 'Settings · Maintenance',
    keywords: ['maintenance', 'update', 'restart', 'log', 'diagnostic', 'version', 'bug report'],
    guide: { chapter: 'settings-reference', anchor: 'maintenance' },
    app: { route: '/settings/maintenance' } },
  { id: 'dataset-settings-modal', kind: 'section', title: 'Per-dataset settings',
    keywords: ['dataset settings', 'per-dataset', 'prompt suffix', 'framing', 'trigger',
      'override', 'modal', 'kind', 'character', 'concept', 'style'],
    guide: { chapter: 'settings-reference', anchor: 'per-dataset-settings' },
    app: { route: '/datasets' },
    tip: { trigger: 'dataset-settings-open',
      text: 'Prompt suffixes add a creative direction to every generated variation — globally or per framing.' } },
  // Changing a dataset's kind (character/concept/style) after creation. Shares the
  // section's anchor — listed AFTER it so the modal keeps the "Open this screen →"
  // button. No tip: the modal already fires one (dataset-settings-open) and a
  // second on the same surface would spam. (The tip total is contract-locked.)
  { id: 'dataset-kind-switch', kind: 'setting', title: 'Change the dataset kind',
    keywords: ['kind', 'change kind', 'switch kind', 'character', 'concept', 'style',
      'convert', 'caption strategy', 'trigger'],
    guide: { chapter: 'settings-reference', anchor: 'per-dataset-settings' },
    app: { route: '/datasets' } },
  // Same suffixes, second surface: the generation panel exposes them inline so
  // they can be tuned per batch. Listed AFTER dataset-settings-modal so the modal
  // keeps the anchor's "Open this screen →" button.
  { id: 'prompt-suffixes', kind: 'setting', title: 'Prompt suffixes (generation panel)',
    keywords: ['prompt suffix', 'suffixes', 'creative direction', 'framing', 'per batch',
      'per-batch', 'generation', 'face', 'bust', 'body', 'back'],
    guide: { chapter: 'settings-reference', anchor: 'per-dataset-settings' },
    app: { route: '/datasets?section=add' } },
  // Multi-engine generation: the cards are checkboxes and, from two engines on,
  // a mode decides whether the shots are SHARED between them (varied dataset,
  // same cost) or sent to ALL of them (compare, then triage — multiplies cost).
  // Two properties of the Gemini engine that change what you get, and that no
  // amount of settings can change back. Kept as its own topic rather than folded
  // into dataset-engine-mode: this one answers "why did I get fewer images than
  // I asked for", which is a question people arrive at already frustrated.
  { id: 'nanobanana-filter-and-synthid', kind: 'section',
    title: 'Nano Banana: the output filter, and SynthID',
    keywords: ['nano banana', 'nanobanana', 'gemini', 'google', 'refused', 'refusal',
      'blocked', 'content filter', 'safety', 'imagesafety', 'empty response',
      'missing images', 'fewer images', 'synthid', 'watermark', 'provenance',
      'nsfw', 'policy', 'bikini', 'lingerie'],
    guide: { chapter: 'settings-reference', anchor: 'image-engines' },
    app: { route: '/datasets?section=add' } },
  { id: 'dataset-engine-mode', kind: 'setting', title: 'Engines & how they share a batch',
    keywords: ['engine', 'engines', 'multiple engines', 'several engines', 'split',
      'all engines', 'compare engines', 'klein', 'krea', 'krea 2 edit', 'nano banana',
      'chatgpt', 'cost', 'mix', 'multi-engine', 'local engine'],
    guide: { chapter: 'settings-reference', anchor: 'image-engines' },
    app: { route: '/datasets?section=add' } },
  // Subject type: WHAT the dataset's subject is (human/animal/creature/object/
  // other). Steers the generation catalog + the identity lock so the prompts stop
  // assuming a person. Listed after prompt-suffixes so the modal keeps the anchor.
  { id: 'subject-type', kind: 'setting', title: 'Subject type (human, animal, anime, object…)',
    keywords: ['subject', 'subject type', 'animal', 'creature', 'object', 'pet', 'dog',
      'product', 'human', 'person', 'catalog', 'non-human', 'generation',
      'anime', 'manga', 'character', 'drawn', '2d', 'illustration', 'waifu', 'anima',
      'art style', 'cartoon'],
    guide: { chapter: 'settings-reference', anchor: 'per-dataset-settings' },
    app: { route: '/datasets?section=add' } },
  { id: 'settings-config-file', kind: 'section', title: 'Config-file-only settings',
    keywords: ['config', 'config.json', 'advanced', 'file only', 'hidden', 'manual'],
    guide: { chapter: 'settings-reference', anchor: 'config-file-only-settings' },
    app: { route: '/settings/maintenance' } },
];
