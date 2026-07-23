/* The help registry — the single pivot for the bidirectional Help mode.
   PURE JS (zero JSX, zero Vite ?raw imports) so node --test can import it and
   the contract test (tests/help-registry-contract.test.mjs) can validate every
   route/anchor/focus against the real markdownHeadingId, the settings registry,
   and the workspace sections.

   Each topic maps ONE thing in the app to ONE place in the guide:
     { id, kind, title, keywords, guide: { chapter, anchor }, app: { route, focus? }, tip? }
   - kind    'section' | 'setting' | 'action' | 'page'
   - guide   chapter ∈ {getting-started, using-the-app, dataset-guide,
             troubleshooting, getting-help, settings-reference}; anchor = the id
             of an H2 in that chapter, computed by markdownHeadingId.
   - app     route = HashRouter path (may carry a query); focus = optional DOM id
             of a field on the target screen (scroll + highlight on arrival).
   - tip     { trigger, text } — an optional one-time contextual hint.

   ORDER MATTERS: for a given (chapter, anchor) the FIRST topic in this array is
   the one whose screen the guide's "Open this screen →" button opens. Section /
   page topics are therefore listed BEFORE the field / action topics that share
   their anchor (e.g. workspace-images before workspace-add/curation/export). */

// settings-reference H2 anchor for each Settings section id.
const SETTINGS_ANCHOR = {
  overview: 'overview',
  engines: 'image-engines',
  scraping: 'scraping-sources',
  'local-tools': 'local-tools',
  captioning: 'captioning-quality',
  training: 'training',
  server: 'server-access',
  maintenance: 'maintenance',
};

// Build a kind:'setting' topic. All fields in a Settings section share the
// section's route and settings-reference anchor; only the DOM focus id differs.
const setting = (id, section, focus, title, keywords, tip) => ({
  id, kind: 'setting', title, keywords,
  guide: { chapter: 'settings-reference', anchor: SETTINGS_ANCHOR[section] },
  app: { route: `/settings/${section}`, focus },
  ...(tip ? { tip } : {}),
});

const action = (id, title, keywords, route, chapter, anchor, tip) => ({
  id, kind: 'action', title, keywords,
  guide: { chapter, anchor },
  app: { route },
  ...(tip ? { tip } : {}),
});

const TOPICS = [
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
  { id: 'settings-maintenance', kind: 'section', title: 'Settings · Maintenance',
    keywords: ['maintenance', 'update', 'restart', 'log', 'trash', 'data', 'storage', 'diagnostic', 'version'],
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
  { id: 'settings-config-file', kind: 'section', title: 'Config-file-only settings',
    keywords: ['config', 'config.json', 'advanced', 'file only', 'hidden', 'manual'],
    guide: { chapter: 'settings-reference', anchor: 'config-file-only-settings' },
    app: { route: '/settings/maintenance' } },

  // ---- Workspace: section-level topics (kind 'section') -------------------
  // workspace-images first so it owns the "Open this screen →" button for the
  // character-walkthrough anchor it shares with add / curation / export.
  { id: 'workspace-images', kind: 'section', title: 'Images',
    keywords: ['images', 'review', 'keep', 'reject', 'caption', 'tiles', 'overview'],
    guide: { chapter: 'using-the-app', anchor: 'the-character-walkthrough-reference-photo-trained-lora' },
    app: { route: '/datasets?section=images' } },
  { id: 'workspace-add', kind: 'section', title: 'Add images',
    keywords: ['add images', 'generate', 'reference', 'variations', 'import', 'photos'],
    guide: { chapter: 'using-the-app', anchor: 'the-character-walkthrough-reference-photo-trained-lora' },
    app: { route: '/datasets?section=add' } },
  { id: 'workspace-scrape', kind: 'section', title: 'Scrape',
    keywords: ['scrape', 'scan', 'gallery', 'url', 'source', 'import', 'concept'],
    guide: { chapter: 'using-the-app', anchor: 'concept-datasets-an-object-or-action-not-a-person' },
    app: { route: '/datasets?section=scrape&panel=scan' },
    tip: { trigger: 'add-images-visit',
      text: 'Scraping now lives in its own 🕸 Scrape section of the sidebar.' } },
  { id: 'workspace-curation', kind: 'section', title: 'Curation',
    keywords: ['curation', 'quality', 'face', 'watermark', 'clean', 'cleanup', 'rescue'],
    guide: { chapter: 'using-the-app', anchor: 'the-character-walkthrough-reference-photo-trained-lora' },
    app: { route: '/datasets?section=curation' } },
  { id: 'workspace-captions', kind: 'section', title: 'Captions',
    keywords: ['captions', 'caption', 'generate', 'leak', 'edit', 'bulk', 'text',
      'caption lab', 'compare', 'a/b', 'model', 'joycaption', 'ollama', 'vocabulary',
      'explicit', 'candidate', 'preview'],
    guide: { chapter: 'dataset-guide', anchor: '3-captions-the-make-or-break-step' },
    app: { route: '/datasets?section=captions' } },
  { id: 'workspace-export', kind: 'section', title: 'Import & export',
    keywords: ['export', 'import', 'training zip', 'backup', 'hugging face', 'merge', 'data',
      'import to bank', 'bank', 're-triage'],
    guide: { chapter: 'using-the-app', anchor: 'the-character-walkthrough-reference-photo-trained-lora' },
    app: { route: '/datasets?section=export' } },
  { id: 'workspace-training', kind: 'section', title: 'Training',
    keywords: ['training', 'train', 'lora', 'launch', 'cloud', 'local', 'preflight'],
    guide: { chapter: 'dataset-guide', anchor: '5-pre-flight-checklist' },
    app: { route: '/datasets?section=training' } },
  { id: 'workspace-checkpoints', kind: 'section', title: 'Checkpoints & LoRAs',
    keywords: ['checkpoints', 'lora', 'epoch', 'checkpoint', 'results', 'import', 'comfyui',
      'graph', 'lineage', 'runs graph', 'continue', 'download'],
    guide: { chapter: 'dataset-guide', anchor: '6-after-training-pick-the-right-checkpoint' },
    app: { route: '/datasets?section=checkpoints' } },
  { id: 'workspace-studio', kind: 'section', title: 'Studio',
    keywords: ['studio', 'test', 'lora', 'checkpoint', 'winning settings'],
    guide: { chapter: 'dataset-guide', anchor: '6-after-training-pick-the-right-checkpoint' },
    app: { route: '/datasets?section=studio' } },

  // ---- Page-level topics (kind 'page') -----------------------------------
  { id: 'page-datasets', kind: 'page', title: 'Datasets library',
    keywords: ['datasets', 'library', 'tiles', 'browse', 'home', 'filter', 'kind'],
    guide: { chapter: 'getting-started', anchor: 'around-the-app' },
    app: { route: '/datasets' },
    tip: { trigger: 'library-browse',
      text: 'Resize tiles S/M/L, collapse sections, and filter by kind.' } },
  action('library-backup', 'Back up everything',
    ['backup', 'back up', 'export everything', 'move machine', 'migrate', 'restore',
     'settings', 'config', 'archive', 'save all', 'new install',
     'trained loras', 'training history', 'include loras', 'not trained yet',
     'import backup', 'backup menu', 'restore backup', 'zip'],
    '/datasets', 'using-the-app', 'back-up-everything'),
  { id: 'page-bank', kind: 'page', title: 'Image bank (triage)',
    keywords: ['bank', 'triage', 'import', 'folder', 'browse', 'choose folder', 'path',
      'telegram', 'duplicates', 'blurry', 'quality', 'cluster', 'person', 'sort',
      'sort resolution', 'resolution', 'megapixels', 'largest', 'smallest',
      'resolution tier', 'resolution filter', 'filter by resolution', 'megapixel',
      'small images', 'thumbnails', 'low resolution', 'high resolution',
      'promote', 'unsorted',
      'aesthetic', 'score', 'nsfw', 'watermark', 'style', 'subfolder', 'keep best',
      'semantic', 'near-duplicate', 'crop', 'crops', 'variant', 'same shot',
      'caption', 'captions', 'search', 'find', 'tag', 'tags', 'describe',
      'launch all', 'pipeline', 'auto-reject', 'overnight', 'run everything',
      'one click', 'batch', 'chain',
      'framing', 'shot type', 'face', 'bust', 'body', 'back', 'full body',
      'close-up', 'back view', 'classify framing', 'composition',
      'coverage advice', 'balance', 'what to add', 'missing', 'thin', 'imbalance',
      'curate', 'curation', 'diverse', 'diversity', 'variety', 'coverage',
      'most varied', 'farthest point', 'similar', 'similarity', 'reference',
      'looks like', 'find similar', 'pick diverse', 'subset', 'trim down',
      'show selected', 'selected view', 'show all', 'see selection',
      'delete rejected', 'delete from disk', 'remove from disk', 'trash',
      'free up space', 'permanently delete', 'clean up rejected',
      'preview', 'previews', 'bank card', 'card preview', 'thumbnail strip',
      'which bank is which', 'recognise bank', 'cover'],
    guide: { chapter: 'using-the-app', anchor: 'the-image-bank-triage-a-big-folder' },
    app: { route: '/bank' } },
  { id: 'page-setup', kind: 'page', title: 'Setup wizard',
    keywords: ['setup', 'wizard', 'onboarding', 'install', 'install everything',
      'install all', 'connect', 'tools'],
    guide: { chapter: 'getting-started', anchor: 'the-setup-wizard' },
    app: { route: '/setup' } },
  { id: 'page-studio', kind: 'page', title: 'Test Studio',
    keywords: ['studio', 'test', 'lora', 'checkpoint', 'generate', 'compare'],
    guide: { chapter: 'dataset-guide', anchor: '6-after-training-pick-the-right-checkpoint' },
    app: { route: '/studio' } },
  { id: 'page-cloud', kind: 'page', title: 'Runs (cloud & local)',
    keywords: ['runs', 'cloud', 'vast', 'stuck', 'history', 'training', 'gpu',
      'lineage', 'tree', 'genealogy', 'graph', 'continue', 'resumed', 'branch', 'superseded', 'descend',
      'checkpoints', 'checkpoint', 'epoch', 'download', 'continue from here'],
    guide: { chapter: 'troubleshooting', anchor: 'a-cloud-run-seems-stuck' },
    app: { route: '/cloud' } },
  action('lineage-inspect-notes', 'Inspect a run & take notes',
    ['inspect run', 'run settings', 'settings used', 'lineage notes', 'config',
     'compare runs', 'note', 'annotate', 'lab', 'rank', 'learning rate',
     'which params', 'experiment'],
    '/cloud', 'dataset-guide', '6-after-training-pick-the-right-checkpoint'),
  action('lineage-compare-runs', 'Compare two runs side by side',
    ['compare runs', 'compare two runs', 'diff', 'difference', 'what changed',
     'side by side', 'shift click', 'lineage compare', 'ab compare', 'settings diff',
     'which setting changed', 'experiment', 'lab'],
    '/cloud', 'dataset-guide', '6-after-training-pick-the-right-checkpoint'),
  action('lineage-remove-gone-run', 'Remove a gone run from the graph',
    ['remove run', 'delete run', 'gone', 'tidy graph', 'clean up runs',
     'no checkpoints', 'clear run', 'lineage cleanup'],
    '/cloud', 'dataset-guide', '6-after-training-pick-the-right-checkpoint'),
  action('lineage-generate-previews', 'Generate a preview per checkpoint',
    ['generate preview', 'preview checkpoint', 'inline generation', 'sample image',
     'same prompt', 'same seed', 'epoch by epoch', 'compare checkpoints', 'strength 1.0',
     'test studio', 'experiment lab', 'lab', 'which epoch', 'best checkpoint',
     'big previews', 'large previews', 'big preview mode', 'comfyui grid', 'preview tiles'],
    '/cloud', 'dataset-guide', '6-after-training-pick-the-right-checkpoint'),
  action('lineage-import-checkpoint', 'Import a checkpoint from the graph',
    ['import checkpoint', 'deploy checkpoint', 'import from graph', 'loras folder',
     'deploy lora', 'use this checkpoint', 'graph import', 'pill import', 'comfyui',
     'view preview large', 'zoom preview', 'lightbox', 'graph view', 'default view'],
    '/datasets?section=checkpoints', 'dataset-guide', '6-after-training-pick-the-right-checkpoint'),
  action('lineage-delete-checkpoint', 'Remove a deployed LoRA or delete a training save',
    ['delete checkpoint', 'delete save', 'remove checkpoint', 'trash checkpoint',
     'remove from comfyui', 'undeploy lora', 'delete training save', 'free disk space',
     'too many epochs', 'graph delete', 'pill delete', 'does it delete my lora',
     'imported lora kept', 'best settings warning'],
    '/datasets?section=checkpoints', 'dataset-guide', '6-after-training-pick-the-right-checkpoint'),

  // ---- Settings: per-field topics (kind 'setting') -----------------------
  // engines
  setting('engines.default', 'engines', 'engine-default', 'Default engine',
    ['default engine', 'engine', 'preselect', 'nanobanana', 'nano banana', 'chatgpt', 'klein', 'gpt-image']),
  setting('engines.enabled', 'engines', 'engines-enabled', 'Enabled engines',
    ['enabled engines', 'engine', 'engines', 'show', 'hide', 'generate panel', 'nanobanana', 'chatgpt', 'klein']),
  setting('engines.chatgpt_auth', 'engines', 'chatgpt-auth-mode', 'ChatGPT engine auth',
    ['chatgpt', 'auth', 'subscription', 'api key', 'codex', 'oauth', 'openai']),
  setting('klein.generation_lora_presets', 'engines', 'klein-generation-lora-presets', 'Klein generation LoRA presets',
    ['lora', 'preset', 'presets', 'klein', 'generation', 'texture', 'anatomy', 'style', 'chain', 'nsfw'],
    { trigger: 'klein-tuning-open',
      text: 'Build named generation-LoRA presets in Settings → Image engines, then pick one per run.' }),
  setting('identity_prompts.face', 'engines', 'identity-prompts', 'Identity lock prompts (API engines)',
    ['identity', 'prompt', 'guard', 'lock', 'face', 'reference', 'beautify', 'preserve', 'consistency', 'edit prompt']),
  setting('identity_prompts.klein_identity', 'engines', 'identity-prompts', 'Klein identity prompt',
    ['identity', 'klein', 'restage', 'face', 'prompt', 'preserve', 'pose']),
  setting('identity_prompts.klein_improve', 'engines', 'identity-prompt-klein-improve', 'Klein improve prompt & toggle',
    ['klein', 'improve', 'upscale', 'enhance', 'prompt', 'texture', 'detail', 'toggle', 'disable']),
  setting('GEMINI_API_KEY', 'engines', 'GEMINI_API_KEY', 'Gemini API key',
    ['gemini', 'api key', 'nano banana', 'nanobanana', 'google', 'key']),
  setting('OPENAI_API_KEY', 'engines', 'OPENAI_API_KEY', 'OpenAI API key',
    ['openai', 'api key', 'chatgpt', 'gpt-image', 'gpt', 'key']),
  // scraping
  setting('REDDIT_CLIENT_ID', 'scraping', 'REDDIT_CLIENT_ID', 'Reddit client ID',
    ['reddit', 'client id', 'scrape', '429', 'rate limit', 'quota', 'key']),
  setting('CIVITAI_API_KEY', 'scraping', 'CIVITAI_API_KEY', 'Civitai API key',
    ['civitai', 'api key', 'nsfw', 'adult', 'scrape', 'key']),
  setting('PEXELS_API_KEY', 'scraping', 'PEXELS_API_KEY', 'Pexels API key',
    ['pexels', 'api key', 'scrape', 'stock', 'key']),
  setting('klein.small_image_prompt', 'scraping', 'klein-small-image-prompt', 'Klein rescue — small scraped images',
    ['klein', 'small image', 'rescue', 'upscale', 'improve', 'prompt', 'scrape']),
  // local-tools
  setting('comfyui.api_url', 'local-tools', 'comfyui-api-url', 'ComfyUI API URL',
    ['comfyui', 'api', 'url', 'klein', 'studio', 'local']),
  setting('comfyui.base_dir', 'local-tools', 'comfyui-base-dir', 'ComfyUI install directory',
    ['comfyui', 'directory', 'path', 'install', 'base dir', 'models', 'loras']),
  setting('HF_TOKEN', 'local-tools', 'HF_TOKEN', 'Hugging Face token',
    ['hugging face', 'hf', 'token', 'gated', 'klein', 'download', 'fp8', 'key']),
  setting('ollama.url', 'local-tools', 'ollama-url', 'Ollama URL',
    ['ollama', 'url', 'vision', 'caption', 'local']),
  setting('ollama.vision_model', 'local-tools', 'ollama-vision-model', 'Ollama vision model',
    ['ollama', 'vision', 'model', 'abliterated', 'caption', 'qwen', 'uncensored']),
  setting('aitoolkit.dir', 'local-tools', 'aitoolkit-dir', 'ai-toolkit directory',
    ['ai-toolkit', 'aitoolkit', 'directory', 'path', 'training', 'run.py']),
  setting('aitoolkit.python', 'local-tools', 'aitoolkit-python', 'ai-toolkit Python interpreter',
    ['ai-toolkit', 'aitoolkit', 'python', 'interpreter', 'venv', 'conda', 'uv']),
  setting('aitoolkit.datasets_dir', 'local-tools', 'aitoolkit-datasets-dir', 'ai-toolkit datasets directory',
    ['ai-toolkit', 'aitoolkit', 'datasets', 'directory', 'override', 'path']),
  setting('aitoolkit.output_dir', 'local-tools', 'aitoolkit-output-dir', 'ai-toolkit output directory',
    ['ai-toolkit', 'aitoolkit', 'output', 'directory', 'override', 'path']),
  setting('aitoolkit.hf_home', 'local-tools', 'aitoolkit-hf-home', 'ai-toolkit Hugging Face cache',
    ['ai-toolkit', 'aitoolkit', 'hugging face', 'hf home', 'cache', 'override', 'path']),
  // captioning
  setting('captioning.backend', 'captioning', 'captioning-backend', 'Captioning backend',
    ['caption', 'captioning', 'backend', 'joycaption', 'ollama', 'auto']),
  setting('watermark.device', 'captioning', 'watermark-device', 'Watermark processing device',
    ['watermark', 'device', 'gpu', 'cuda', 'cpu', 'inpaint', 'lama']),
  setting('watermark.allow_crop', 'captioning', 'watermark-allow-crop', 'Allow automatic crop',
    ['watermark', 'crop', 'allow crop', 'border', 'clean', 'lama', 'klein']),
  setting('face_scoring.green', 'captioning', 'face-threshold-green', 'Face score — green threshold',
    ['face', 'score', 'green', 'threshold', 'similarity', 'resemblance', 'insightface']),
  setting('face_scoring.orange', 'captioning', 'face-threshold-orange', 'Face score — orange threshold',
    ['face', 'score', 'orange', 'threshold', 'similarity', 'borderline']),
  setting('bank.sharpness_min', 'captioning', 'bank-sharpness-min', 'Bank — sharpness minimum',
    ['bank', 'triage', 'sharpness', 'blur', 'blurry', 'laplacian', 'focus', 'threshold']),
  setting('bank.noise_max', 'captioning', 'bank-noise-max', 'Bank — noise maximum',
    ['bank', 'triage', 'noise', 'noisy', 'grain', 'threshold']),
  setting('bank.uniformity_min', 'captioning', 'bank-uniformity-min', 'Bank — uniformity minimum',
    ['bank', 'triage', 'uniform', 'flat', 'empty', 'solid', 'threshold']),
  setting('bank.min_side', 'captioning', 'bank-min-side', 'Bank — minimum side',
    ['bank', 'triage', 'small', 'resolution', 'size', 'pixels', 'threshold']),
  setting('bank.dup_distance', 'captioning', 'bank-dup-distance', 'Bank — duplicate distance',
    ['bank', 'triage', 'duplicate', 'duplicates', 'dhash', 'hamming', 'near-duplicate', 'threshold']),
  setting('bank.face_threshold', 'captioning', 'bank-face-threshold', 'Bank — same-person similarity',
    ['bank', 'triage', 'person', 'cluster', 'face', 'similarity', 'group by person', 'threshold']),
  setting('bank.aesthetic_min', 'captioning', 'bank-aesthetic-min', 'Bank — aesthetic minimum',
    ['bank', 'triage', 'aesthetic', 'quality', 'laion', 'keep best', 'nice', 'threshold']),
  setting('bank.nsfw_max', 'captioning', 'bank-nsfw-max', 'Bank — NSFW maximum',
    ['bank', 'triage', 'nsfw', 'sfw', 'explicit', 'safe', 'threshold']),
  setting('bank.style_threshold', 'captioning', 'bank-style-threshold', 'Bank — same-style similarity',
    ['bank', 'triage', 'style', 'cluster', 'group by style', 'screenshot', 'meme', 'threshold']),
  setting('bank.semantic_dup_threshold', 'captioning', 'bank-semantic-dup-threshold', 'Bank — semantic duplicate similarity',
    ['bank', 'triage', 'semantic', 'duplicate', 'near-duplicate', 'crop', 'crops', 'variant',
     'same shot', 'embedding', 'clip', 'cosine', 'threshold']),
  // training
  setting('training.default_family', 'training', 'training-default-family', 'Default training family',
    ['training', 'family', 'default', 'zimage', 'sdxl', 'krea', 'flux']),
  setting('VAST_API_KEY', 'training', 'VAST_API_KEY', 'vast.ai API key',
    ['vast', 'api key', 'cloud', 'gpu', 'rent', 'budget', 'key']),
  setting('cloud.max_concurrent_runs', 'training', 'cloud-max-concurrent-runs', 'Max simultaneous cloud runs',
    ['cloud', 'concurrent', 'runs', 'vast', 'gpu', 'limit']),
  setting('cloud.max_price_per_hour', 'training', 'cloud-max-price-per-hour', 'Max price per hour',
    ['cloud', 'price', 'hour', 'budget', 'vast', 'cost', 'ceiling']),
  setting('cloud.monthly_budget_usd', 'training', 'cloud-monthly-budget', 'Monthly budget',
    ['cloud', 'budget', 'monthly', 'vast', 'cost', 'limit', 'spend']),
  setting('cloud.stall_timeout_minutes', 'training', 'cloud-stall-timeout', 'Stall timeout',
    ['cloud', 'stall', 'timeout', 'watchdog', 'vast', 'stuck']),
  setting('cloud.unreachable_grace_minutes', 'training', 'cloud-unreachable-grace', 'Unreachable grace',
    ['cloud', 'unreachable', 'grace', 'timeout', 'network', 'vast', 'blackout', 'reconnect']),
  setting('cloud.min_reliability', 'training', 'cloud-min-reliability', 'Min host reliability',
    ['cloud', 'reliability', 'host', 'vast', 'offer', 'cheap']),
  setting('cloud.verified_only', 'training', 'cloud-verified-only', 'Verified hosts only',
    ['cloud', 'verified', 'host', 'vast', 'offer', 'filter']),
  setting('cloud.secure_cloud_only', 'training', 'cloud-secure-cloud-only', 'Secure Cloud only',
    ['cloud', 'secure', 'community', 'vast', 'offer', 'filter']),
  // Dual captions is a per-run Advanced training option (not a global Setting),
  // so it points at the dataset guide's dedicated section rather than
  // settings-reference, and its route is the training workspace section. Its tip
  // surfaces it when the Advanced options are first opened.
  { id: 'training.dual_captions', kind: 'setting', title: 'Dual captions (long + short)',
    keywords: ['dual captions', 'long', 'short', 'short caption', 'caption', 'augmentation',
      'short_and_long', 'advanced', 'training'],
    guide: { chapter: 'dataset-guide', anchor: '7-dual-captions-long-short' },
    app: { route: '/datasets?section=training' },
    tip: { trigger: 'dual-captions-advanced',
      text: 'New: train each image on a long AND a short caption (Advanced options → Dual captions) so the LoRA leans less on any single wording.' } },
  // server
  setting('server.port', 'server', 'server-port', 'Server port',
    ['server', 'port', 'bind', 'network', '5050']),
  setting('server.lan', 'server', 'server-lan', 'Available on the local network',
    ['lan', 'network', 'remote', 'phone', 'wifi', 'host', 'bind']),
  setting('server.require_token', 'server', 'server-require-token', 'Require an access token',
    ['token', 'require', 'access', 'remote', 'phone', 'security', 'lan']),
  setting('server.access_token', 'server', 'server-token', 'Access token',
    ['token', 'access', 'remote', 'phone', 'password', 'qr']),
  // maintenance
  setting('paths.dataset_images_root', 'maintenance', 'dataset-images-root', 'Dataset images root',
    ['data', 'storage', 'path', 'dataset', 'images', 'root', 'location', 'disk']),

  // ---- Action topics (kind 'action') -------------------------------------
  action('action-watermark-clean', 'Find & clean watermarks',
    ['watermark', 'clean', 'find', 'lama', 'klein', 'crop', 'remove'],
    '/datasets?section=curation&panel=watermarks', 'settings-reference', 'captioning-quality',
    { trigger: 'watermark-batch-clean',
      text: 'Clean has two engines — LaMa (fast) and Klein (quality) — and auto-crop can be turned off.' }),
  action('action-grid-status-filter', 'Filter the grid by decision',
    ['filter', 'decision', 'undecided', 'awaiting', 'pending', 'kept', 'keep', 'rejected',
     'reject', 'improve', 'candidates', 'klein', 'isolate', 'triage', 'select all', 'grid'],
    '/datasets?section=images', 'dataset-guide', '2-how-many-images-and-which-ones'),
  action('action-caption-generate', 'Generate captions',
    ['caption', 'generate', 'joycaption', 'ollama', 'text'],
    '/datasets?section=captions&panel=generate', 'dataset-guide', '3-captions-the-make-or-break-step'),
  action('action-caption-options', 'Caption method options',
    ['caption', 'options', 'engine', 'model', 'ollama', 'pull', 'instructions', 'prompt',
     'method', 'vocabulary', 'explicit', 'clinical', 'nsfw', 'abliterated', 'uncensored'],
    '/datasets?section=captions&panel=generate', 'dataset-guide', '3-captions-the-make-or-break-step'),
  action('action-caption-stop', 'Stop a captioning batch',
    ['caption', 'stop', 'cancel', 'abort', 'interrupt', 'batch', 'graceful', 'halt'],
    '/datasets?section=captions&panel=generate', 'dataset-guide', '3-captions-the-make-or-break-step'),
  action('action-training-launch', 'Train the LoRA',
    ['train', 'training', 'launch', 'cloud', 'lora', 'start'],
    '/datasets?section=training&panel=launch', 'dataset-guide', '5-pre-flight-checklist'),
  action('training-continue-anyway', 'Continue anyway (train a not-ready dataset)',
    ['continue', 'anyway', 'not ready', 'blocker', 'override', 'too few', 'overfit', 'readiness', 'force'],
    '/datasets?section=training&panel=launch', 'dataset-guide', '5-pre-flight-checklist'),
  action('action-scrape-scan', 'Scan a gallery URL',
    ['scrape', 'scan', 'gallery', 'url', 'import', 'concept'],
    '/datasets?section=scrape&panel=scan', 'using-the-app', 'concept-datasets-an-object-or-action-not-a-person'),
  action('action-import-from-bank', 'Import images from a bank',
    ['bank', 'import from bank', 'promote', 'triaged', 'kept images', 'add images',
     'copy from bank', 'reuse bank', 'nothing to promote', 'already imported'],
    '/datasets?section=add', 'using-the-app', 'the-image-bank-triage-a-big-folder'),
  action('action-studio-open', 'Open Studio',
    ['studio', 'test', 'lora', 'checkpoint', 'open'],
    '/datasets?section=studio', 'dataset-guide', '6-after-training-pick-the-right-checkpoint'),
  action('continue-training', 'Continue a training run',
    ['continue', 'resume', 'more steps', 'epoch', 'checkpoint', 'restart', 'undercook', 'overcook',
     'learning rate', 'lr', 'half', 'tenth', 'gentle finish', 'polish', 'timestep', 'cadence',
     'lane', 'local', 'cloud', 'run it'],
    '/datasets?section=checkpoints', 'dataset-guide', '6-after-training-pick-the-right-checkpoint',
    { trigger: 'continue-any-epoch',
      text: 'Finished a run? ▶ Continue trains it further — for any number of steps, or resumed from an earlier, less-cooked epoch.' }),
  action('action-recaption-targeted', 'Re-caption leaking images',
    ['caption', 'recaption', 'leak', 'targeted', 'fix', 'review'],
    '/datasets?section=captions&panel=leak-review', 'dataset-guide', '3-captions-the-make-or-break-step',
    { trigger: 'leak-panel-visible',
      text: 'You can re-caption just one leaking image (or all of them) — no full re-run.' }),
  action('action-watermark-restore', 'Restore original',
    ['watermark', 'restore', 'original', 'undo', 'revert', 'clean'],
    '/datasets?section=curation&panel=review-flagged', 'settings-reference', 'captioning-quality',
    { trigger: 'watermark-clean-done',
      text: 'Not happy with a clean? Restore brings the original back — then try the other engine.' }),
];

Object.freeze(TOPICS);

const BY_ID = new Map(TOPICS.map((t) => [t.id, t]));

/** The frozen registry array (registry order preserved). */
export const helpTopics = TOPICS;

/** Look up a single topic by id, or undefined. */
export function getHelpTopic(id) {
  return BY_ID.get(id);
}

/** All topics whose guide.chapter === chapterId, in registry order. */
export function helpTopicsForChapter(chapterId) {
  return TOPICS.filter((t) => t.guide.chapter === chapterId);
}

/** Case-insensitive search over id / title / keywords. Registry order. */
export function searchHelpTopics(query) {
  const q = String(query || '').trim().toLowerCase();
  if (!q) return [];
  return TOPICS.filter((t) =>
    t.id.toLowerCase().includes(q)
    || t.title.toLowerCase().includes(q)
    || t.keywords.some((k) => k.toLowerCase().includes(q)));
}

/** All one-time tips, flattened: { topicId, trigger, text, guide }. */
export function helpTips() {
  return TOPICS.filter((t) => t.tip).map((t) => ({
    topicId: t.id, trigger: t.tip.trigger, text: t.tip.text, guide: t.guide,
  }));
}

/** Resolve a tip by its stable trigger string (or null). */
export function getHelpTip(trigger) {
  return helpTips().find((t) => t.trigger === trigger) || null;
}

/** The in-app HashRouter "to" for a topic's guide anchor. The Getting-help
    chapter lives at its own /help route, every other chapter under /guide. */
export function guideHref(chapter, anchor) {
  const base = chapter === 'getting-help' ? '/help' : `/guide/${chapter}`;
  return anchor ? `${base}?h=${anchor}` : base;
}

/** Same, for a topic. */
export function topicGuideHref(topic) {
  if (!topic) return null;
  return guideHref(topic.guide.chapter, topic.guide.anchor);
}
