/* One section of the help registry, moved verbatim (2026-08-24 split).
   ORDER MATTERS inside and across sections: helpRegistry.js concatenates
   the six section arrays in a fixed order, and for a given (chapter,
   anchor) the FIRST topic owns the "Open this screen →" button. */
import { setting } from '../topicBuilders.js';

export const SETTINGS_FIELD_TOPICS = [
  // ---- Settings: per-field topics (kind 'setting') -----------------------
  // engines
  setting('engines.default', 'engines', 'engine-default', 'Default engine',
    ['default engine', 'engine', 'preselect', 'nanobanana', 'nano banana', 'chatgpt', 'klein', 'gpt-image',
     'openrouter']),
  setting('engines.enabled', 'engines', 'engines-enabled', 'Enabled engines',
    ['enabled engines', 'engine', 'engines', 'show', 'hide', 'generate panel', 'nanobanana', 'chatgpt', 'klein',
     'openrouter']),

  setting('klein.unet', 'engines', 'klein-model-unet', 'Klein diffusion model (UNET) file',
    ['klein', 'unet', 'diffusion model', 'model file', 'path', 'override', 'pin', 'custom model',
     'unreadable', 'corrupt', 'says missing',
     // The field is a PICKER now, and a pin it cannot find stops the engine.
     'dropdown', 'list', 'picker', 'choose', 'select', 'not found', 'refuses to run',
     'engine will not start']),
  setting('klein.text_encoder', 'engines', 'klein-model-text_encoder', 'Klein text encoder file',
    ['klein', 'text encoder', 'clip', 'qwen', 'model file', 'path', 'override', 'pin']),
  setting('klein.vae', 'engines', 'klein-model-vae', 'Klein VAE file',
    ['klein', 'vae', 'model file', 'path', 'override', 'pin']),
  setting('klein.consistency_lora', 'engines', 'klein-model-consistency_lora',
    'Klein consistency LoRA file',
    ['klein', 'consistency', 'lora', 'model file', 'path', 'override', 'pin', 'structure',
     'anchor', 'composition']),
  setting('klein.generation_lora_presets', 'engines', 'klein-generation-lora-presets', 'Klein generation LoRA presets',
    ['lora', 'preset', 'presets', 'klein', 'generation', 'texture', 'anatomy', 'style', 'chain', 'nsfw',
     // The silently-dropped row: it names the consistency LoRA the graph already
     // loads, so the server skips it. These are the words for the symptom.
     'duplicate', 'skipped', 'ignored', 'row ignored', 'double', 'double-stack',
     'stacked twice', 'blocky', 'posterized', 'macro-blocking', 'consistency lora'],
    { trigger: 'klein-tuning-open',
      text: 'Build named generation-LoRA presets in Settings → Image engines, then pick one per run.' }),
  // The half of the preset feature that was missing: the run panel opened on
  // "None" on every visit, so a configured preset applied only when the user
  // remembered to re-pick it — and the keywords below are the words someone
  // writes when they discover, in a finished PNG's metadata, that none of their
  // LoRA lines were applied.
  setting('klein.default_generation_lora_preset', 'engines', 'klein-default-lora-preset',
    'Klein preset selected by default',
    ['klein', 'lora', 'preset', 'default preset', 'default', 'always', 'automatic',
     'applied', 'not applied', 'ignored', 'ignores my settings', 'nothing happens',
     'resets to none', 'none', 'every run', 'remember', 'preselect']),
  setting('krea.default_generation_lora_preset', 'engines', 'krea-default-lora-preset',
    'Krea 2 Edit preset selected by default',
    ['krea', 'krea 2', 'lora', 'preset', 'default preset', 'default', 'always',
     'automatic', 'applied', 'not applied', 'ignored', 'resets to none', 'none',
     'every run', 'preselect']),
  // Shared by BOTH local engines, so it is not a klein.* or krea.* topic and it
  // does not live in Settings: the dial is in the Generate-variations panel,
  // above the shot cards whose size it decides. Route points there.
  { id: 'variations.output_megapixels', kind: 'setting', title: 'Variation output size',
    keywords: ['size', 'output size', 'resolution', 'megapixels', 'mp', 'pixels',
      'dimensions', 'too small', 'smaller', 'bigger', 'larger', 'upscale',
      '2 mp', 'klein', 'krea', 'krea 2', 'variations', 'shots', 'aspect',
      'ratio', 'different sizes', 'mixed sizes', 'vram', 'faster'],
    guide: { chapter: 'settings-reference', anchor: 'image-engines' },
    app: { route: '/datasets?section=add&panel=generate' } },
  setting('klein.generation_steps', 'engines', 'klein-generation', 'Klein generation steps',
    ['klein', 'steps', 'sampler', 'generation', 'quality', 'slower', 'cleaner', 'sampling', '5 steps']),
  setting('klein.edit_base_lora_strength', 'engines', 'klein-generation',
    'Klein enhancement LoRA on edits',
    ['klein', 'lora', 'realistic', 'enhancement', 'detail', 'edit', 'conformity',
     'not following', 'ignores the prompt', 'drift', 'style', 'reference edit',
     'variations', 'regenerate', 'node 139']),
  // Krea 2 Identity Edit — the second LOCAL engine. `grounding_px` first: it is
  // THE consistency ↔ prompt dial, and a bare pixel count means nothing without
  // that sentence, so it carries the widest keyword set of the four.
  setting('krea.grounding_px', 'engines', 'krea-grounding', 'Krea 2 Edit reference grounding',
    ['krea', 'krea 2', 'grounding', 'grounding_px', 'consistency', 'likeness', 'resemblance',
     'prompt adherence', 'variety', 'identity', 'reference', 'dial', 'slider', 'local engine']),
  // The two calibration dials that had NO input on the Settings page until it
  // gained sliders for them. They now sit on BOTH surfaces — this card and the
  // workspace's "🧬 Krea 2 Edit tuning" panel — writing the same global key, so
  // these topics point at the Settings field like their two siblings above and
  // below. Before that they had to point at the workspace panel, because
  // pointing at a field that did not exist would have been worse.
  setting('krea.ref_boost', 'engines', 'krea-ref-boost', 'Krea 2 Edit reference pull',
    ['krea', 'krea 2', 'ref boost', 'ref_boost', 'reference pull', 'reference boost',
     'likeness', 'resemblance', 'does not look like', 'identity', 'weak likeness',
     'similarity', 'too different', 'face', 'calibration', 'slider', 'local engine']),
  setting('krea.identity_lora_strength', 'engines', 'krea-identity-lora-strength',
    'Krea 2 Edit identity LoRA strength',
    ['krea', 'krea 2', 'identity lora strength', 'identity_lora_strength',
     'lora strength', 'identity', 'weight', 'face transfer', 'likeness', 'posterized',
     'waxy', 'blocky', 'calibration', 'slider', 'local engine']),
  setting('krea.steps', 'engines', 'krea-steps', 'Krea 2 Edit sampler steps',
    ['krea', 'steps', 'sampler', 'quality', 'slower', 'local engine']),
  setting('krea.base_model', 'engines', 'krea-base-model', 'Krea 2 Edit base model',
    ['krea', 'base model', 'turbo', 'raw', 'checkpoint', 'unet', 'diffusion model',
     'noise', 'biglove', 'incompatible', 'local engine',
     // A GGUF quantised base is a dead end ComfyUI reports as a bare
     // "value_not_in_list" — these terms are what someone stuck on it searches for.
     'gguf', 'quant', 'quantised', 'quantized', 'q4_k_m', 'q8', 'value not in list',
     'not in list', 'not detecting', 'model not found', 'unet_name', 'safetensors',
     'dropdown', 'list', 'picker', 'choose', 'select', 'not found', 'refuses to run',
     'engine will not start',
     // Naming the ELECTED base on screen. Two Krea builds in one folder both read
     // as "turbo", the tie-break picks one, and until it was named the only way to
     // find out was a finished PNG's metadata.
     'which model', 'which base', 'wrong model', 'currently loading', 'elected',
     'auto', 'finetune', 'community model', 'two models', 'several builds']),
  setting('krea.identity_lora', 'engines', 'krea-identity-lora', 'Krea 2 Edit identity LoRA',
    ['krea', 'identity', 'edit lora', 'lora', 'krea2_identity_edit', 'civitai',
     'node pack', 'comfyui-krea2edit', 'missing', 'local engine',
     'dropdown', 'list', 'picker', 'choose', 'select', 'not found', 'refuses to run']),
  setting('krea.generation_lora_presets', 'engines', 'krea-generation-lora-presets',
    'Krea 2 Edit generation LoRA presets',
    ['krea', 'krea 2', 'lora', 'loras', 'generation lora', 'preset', 'presets',
     'always-on', 'always on', 'filter bypass', 'filterbypass', 'bypass', 'nsfw',
     'uncensored', 'style lora', 'detail slider', 'chain', 'stack', 'strength',
     'duplicate', 'skipped', 'ignored', 'row ignored', 'double', 'double-stack',
     'blocky', 'posterized', 'macro-blocking', 'identity lora']),
  setting('identity_prompts.face', 'engines', 'identity-prompts', 'Identity lock prompts (API engines)',
    ['identity', 'prompt', 'guard', 'lock', 'face', 'reference', 'beautify', 'preserve', 'consistency', 'edit prompt',
     'subject type', 'animal', 'per subject', 'leak', 'tails', 'extra limbs']),
  // Named for the ENGINE FAMILY, not for Klein: Krea 2 Edit reads this very
  // text, and a Krea user searching "krea identity prompt" found nothing.
  setting('identity_prompts.klein_identity', 'engines', 'identity-prompts', 'Local engines identity prompt (Klein & Krea 2)',
    ['identity', 'klein', 'krea', 'krea 2', 'local engines', 'restage', 'face', 'prompt', 'preserve', 'pose']),
  // The words Qeeyana (Reddit) actually used are in here verbatim: she had the
  // symptom ("anime looks realistic after the quality inpaint") and no path to
  // the cause, because the shipped instruction — "add detailed texture, add
  // sharp details, add candid shot, add soft focus effect" — is a photographic
  // recipe applied to every dataset. Searching her own sentence must land here.

  // The four knobs behind the lightbox's "Adjust improve strength →". They were
  // exposed as settings but never registered, so Help search could not reach them
  // and the link had nothing to aim at.

  // The five parts the local-edit prompt is ALSO built from. They used to be
  // hardcoded, so nobody could search for them; these are the words a user reaches
  // for when a generated shot is wrong ("why is everyone wearing jeans", "it added
  // a tattoo", "it never does a full body").
  setting('identity_prompts.render_tail', 'engines', 'prompt-part-render-tail', 'Klein/Krea rendering tail (SFW & uncensored)',
    ['render', 'tail', 'ending', 'photograph', 'realistic', 'sfw', 'nsfw', 'uncensored',
     'nudity', 'clamp', 'illustration', 'anime', 'style', 'klein', 'krea', 'prompt']),
  setting('identity_prompts.framing_detail', 'engines', 'prompt-part-framing', 'Shot detail per framing (face/bust/body/back)',
    ['framing', 'shot', 'detail', 'close-up', 'bust', 'full body', 'back', 'lens', '85mm',
     'composition', 'cropped', 'klein', 'krea', 'prompt', 'head to toe']),
  setting('identity_prompts.markings_lock', 'engines', 'prompt-part-global', 'Skin hold, outfit & expression directives, garment palette',
    ['markings', 'skin', 'tattoo', 'scar', 'mole', 'piercing', 'redraw', 'invent',
     'outfit', 'clothes', 'garment', 'palette', 'wardrobe', 'jeans', 'same outfit',
     'expression', 'smile', 'neutral', 'krea', 'klein', 'prompt', 'directive']),
  setting('prompt-preview', 'engines', 'prompt-preview', 'See the prompt an engine actually receives',
    ['prompt', 'preview', 'composed', 'what is sent', 'debug', 'full prompt', 'inspect',
     'klein', 'krea', 'nano banana', 'chatgpt', 'openrouter', 'characters']),






  // scraping

  setting('CIVITAI_API_KEY', 'scraping', 'CIVITAI_API_KEY', 'Civitai API key',
    ['civitai', 'api key', 'nsfw', 'adult', 'scrape', 'key', 'publish', 'post', 'prompt browser']),



  // local-tools
  setting('comfyui.repair_timeout_minutes', 'local-tools', 'klein-repair-timeout',
    'Klein repair and watermark cleaning time limit',
    ['klein', 'repair', 'brush', 'clean', 'timeout', 'five minutes', 'slow hardware']),
  setting('comfyui.improve_timeout_minutes', 'local-tools', 'bank-improve-timeout',
    'Bank upscale and improve wait', ['bank', 'upscale', 'improve', 'timeout']),
  setting('timeouts.processing_multiplier', 'local-tools', 'processing-timeout-multiplier',
    'Processing timeout multiplier', ['timeout', 'captioning', 'analysis', 'model loading', 'hardware load']),
  setting('timeouts.network_multiplier', 'local-tools', 'network-timeout-multiplier',
    'Network timeout multiplier', ['timeout', 'network', 'connection', 'download', 'service health']),
  setting('comfyui.api_url', 'local-tools', 'comfyui-api-url', 'ComfyUI API URL',
    ['comfyui', 'api', 'url', 'klein', 'studio', 'local']),
  setting('comfyui.base_dir', 'local-tools', 'comfyui-base-dir', 'ComfyUI install directory',
    ['comfyui', 'directory', 'path', 'install', 'base dir', 'models', 'loras',
     // ComfyUI Desktop keeps a SHARED models folder and one inside its install
     // directory, so pointing the API address at one install and the models
     // override at another is easy and silent — the app then lists models the
     // running ComfyUI does not serve.
     'comfyui desktop', 'desktop', 'two folders', 'shared models', 'multiple installs',
     'wrong install', 'value not in list', 'not in list', 'model not found',
     'models folder', 'override']),
  setting('comfyui.output_dir', 'local-tools', 'comfyui-output-dir', 'ComfyUI output folder override',
    ['comfyui', 'output', 'directory', 'folder', 'override', 'path', 'custom', 'output-directory']),
  setting('comfyui.input_dir', 'local-tools', 'comfyui-input-dir', 'ComfyUI input folder override',
    ['comfyui', 'input', 'directory', 'folder', 'override', 'path', 'custom', 'input-directory']),
  setting('comfyui.models_dir', 'local-tools', 'comfyui-models-dir', 'ComfyUI models folder override',
    ['comfyui', 'models', 'directory', 'folder', 'override', 'path', 'custom', 'models-directory']),
  setting('comfyui.loras_dir', 'local-tools', 'comfyui-loras-dir', 'ComfyUI LoRAs folder override',
    ['comfyui', 'loras', 'lora', 'directory', 'folder', 'override', 'path', 'custom']),
  setting('comfyui.object_info_timeout_s', 'local-tools', 'comfyui-object-info-timeout',
    'ComfyUI response timeout',
    ['comfyui', 'timeout', 'slow', 'timed out', 'timing out', 'not running',
     'isn\'t running', 'unreachable', 'krea', 'klein', 'object_info', 'nodes',
     'custom nodes', 'many nodes', 'response', 'wait', '8 seconds', 'hang']),
  setting('comfyui.local_queue_limit', 'local-tools', 'comfyui-local-queue-limit',
    'Local generation queue limit',
    ['comfyui', 'generation', 'queue', 'batch', '60 images', '300 images', '1000', 'overnight', 'limit']),
  setting('comfyui.generation_timeout_minutes', 'local-tools', 'comfyui-generation-timeout',
    'Generation time limit',
    ['comfyui', 'generation', 'timeout', '15 minutes', 'slow render', 'unlimited', 'overnight']),
  setting('HF_TOKEN', 'local-tools', 'HF_TOKEN', 'Hugging Face token',
    ['hugging face', 'hf', 'token', 'gated', 'klein', 'krea', 'flux', 'download',
     'fp8', 'key', '401', '403', 'unauthorized', 'restricted', 'authenticated',
     'access', 'licence', 'license', 'hf auth login', 'training']),
  setting('local_llm.provider', 'local-tools', 'local-llm-provider', 'Local LLM provider',
    ['local llm', 'provider', 'ollama', 'lm studio', 'lmstudio', 'backend', 'switch',
     'caption', 'vision', 'which']),
  setting('lmstudio.url', 'local-tools', 'lmstudio-url', 'LM Studio URL',
    ['lm studio', 'lmstudio', 'url', '1234', 'server', 'developer', 'local', 'v1']),
  setting('lmstudio.vision_model', 'local-tools', 'lmstudio-vision-model', 'LM Studio model',
    ['lm studio', 'lmstudio', 'model', 'loaded', 'vlm', 'vision', 'caption', 'jit', 'list', 'refresh', 'detected']),
  setting('lmstudio.vision_concurrency', 'local-tools', 'lmstudio-vision-concurrency',
    'Images analysed at once',
    ['lm studio', 'lmstudio', 'concurrency', 'parallel', 'at once', 'speed', 'bank']),
  setting('lmstudio.vision_keep_warm_seconds', 'local-tools', 'lmstudio-vision-keep-warm',
    'Keep the vision model warm',
    ['lm studio', 'lmstudio', 'keep warm', 'keep alive', 'unload', 'vram', 'ttl']),
  setting('ollama.url', 'local-tools', 'ollama-url', 'Ollama URL',
    ['ollama', 'url', 'vision', 'caption', 'local']),
  setting('ollama.vision_model', 'local-tools', 'ollama-vision-model', 'Ollama vision model',
    ['ollama', 'vision', 'model', 'abliterated', 'caption', 'qwen', 'uncensored', 'list', 'refresh', 'installed']),
  setting('ollama.vision_concurrency', 'local-tools', 'ollama-vision-concurrency', 'Images analysed at once',
    ['ollama', 'concurrency', 'parallel', 'at once', 'speed', 'faster', 'watermark',
     'framing', 'caption', 'bank', 'vision', 'slow']),
  setting('ollama.vision_keep_warm_seconds', 'local-tools', 'ollama-vision-keep-warm',
    'Keep the vision model warm',
    ['ollama', 'keep alive', 'keep warm', 'unload', 'vram', 'memory', 'reload',
     'cold', 'slow', 'crop', 'describe', 'vision', 'speed']),
  setting('aitoolkit.dir', 'local-tools', 'aitoolkit-dir', 'ai-toolkit directory',
    ['ai-toolkit', 'aitoolkit', 'directory', 'path', 'training', 'run.py']),
  setting('aitoolkit.python', 'local-tools', 'aitoolkit-python', 'ai-toolkit Python interpreter',
    ['ai-toolkit', 'aitoolkit', 'python', 'interpreter', 'venv', 'conda', 'uv',
     'torch', 'no module named torch', 'windows store', 'windowsapps', '3.11',
     'python version', 'architecture', 'which folder', 'ports']),
  setting('aitoolkit.datasets_dir', 'local-tools', 'aitoolkit-datasets-dir', 'ai-toolkit datasets directory',
    ['ai-toolkit', 'aitoolkit', 'datasets', 'directory', 'override', 'path']),
  setting('aitoolkit.output_dir', 'local-tools', 'aitoolkit-output-dir', 'ai-toolkit output directory',
    ['ai-toolkit', 'aitoolkit', 'output', 'directory', 'override', 'path']),
  setting('aitoolkit.hf_home', 'local-tools', 'aitoolkit-hf-home', 'ai-toolkit Hugging Face cache',
    ['ai-toolkit', 'aitoolkit', 'hugging face', 'hf home', 'cache', 'override', 'path']),
  // captioning
  setting('dataset_import.max_side', 'captioning', 'dataset-import-max-side',
    'Dataset import — stored resolution',
    ['import', 'resolution', 'size', 'pixels', '1024', '2048', 'downscale', 'resize',
     'normalize', 'normalized', 'shrink', 'original', 'full size', 'preserve', 'quality']),
  setting('dataset_import.encoding', 'captioning', 'dataset-import-encoding',
    'Dataset import — stored encoding',
    ['import', 'encoding', 'webp', 'quality', 'lossless', 'compression', 'artifacts',
     'q92', 'recompress', 'disk space', 'preserve originals', 'jpeg', 'jpg', 'png', 'bmp',
     'original file', 'auto head crop', 'derived']),
  setting('image_input.max_pixels', 'captioning', 'image-input-max-pixels',
    'Image size budget — maximum total pixels',
    ['input', 'budget', 'limit', 'safety', 'pixels', 'megapixels', 'mi-pixels', 'memory',
     'ram', 'decode', 'bomb', 'panorama', 'camera master', 'too large', 'rejects images',
     'reduce the image', '16777216', '8192', 'no limit', 'unlimited', 'oversized',
     'caption', 'captioning', 'joycaption', 'skipped', 'not captioned']),
  setting('image_input.max_side', 'captioning', 'image-input-max-side',
    'Image size budget — maximum side',
    ['input', 'budget', 'limit', 'safety', 'side', 'width', 'height', 'px per side',
     'panorama', 'wide', 'too large', 'rejects images', '8192', '16384', 'no limit',
     'unlimited', 'oversized', 'caption', 'captioning', 'joycaption']),
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
  setting('bank.detail_min', 'captioning', 'bank-detail-min', 'Bank — real-detail minimum',
    ['bank', 'triage', 'upscale', 'upscaled', 'enlarged', 'resize', 'resized', 'fake resolution',
     'effective resolution', 'real detail', 'soft', 'interpolated', 'threshold']),
  setting('bank.bars_max', 'captioning', 'bank-bars-max', 'Bank — black-bar maximum',
    ['bank', 'triage', 'letterbox', 'pillarbox', 'black bars', 'screenshot', 'video',
     'padding', 'threshold']),
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














  // The Hugging Face allowance is a DISK question, so its card moved to
  // Settings › Storage with the rest of them. Ids stay put: they are stored in
  // the "seen" maps of existing installs.




  // The fp8 tool's SECOND door, and the findable one. Its first
  // (training.fp8_quantize_local, below) sits inside a dense dataset's recipe
  // card — which the person this helps most, someone who downloaded a 26 GB
  // full model from Hugging Face and has no dataset, never opens. Same
  // component, same refusals; only the address differs, so it gets its own
  // topic rather than stealing the other one's.

  // The unlocked half of the full-model (dense) recipe. Per-dataset Advanced
  // controls, not global Settings — they point at the dataset guide's
  // full-model section. Grouped as one topic on purpose: they are one card, and
  // the question a user actually has ("what can I change here?") is answered by
  // the card, not by four separate entries.
  // Preview steps / CFG (GitHub #46). Its own topic rather than keywords on the
  // recipe above: the question is not "what may I change?" but "why do my
  // previews look like sketches?", and the answer is a property of the BASE —
  // which is also why the searched words are symptoms, not setting names.
  { id: 'training.preview_quality', kind: 'setting',
    title: 'Preview quality — steps and CFG',
    keywords: ['preview', 'previews', 'sample', 'samples', 'sample steps', 'steps',
      'cfg', 'guidance', 'guidance scale', 'sketch', 'sketchy', 'blurry preview',
      'unfinished', 'ugly previews', 'slow preview', 'distilled', 'turbo', 'raw',
      'sample_steps', 'sample_guidance', 'preview quality', 'test images'],
    guide: { chapter: 'dataset-guide', anchor: '11-preview-quality-steps-and-cfg' },
    app: { route: '/datasets?section=training' } },

  // Its own topic rather than more keywords above: the question here is not
  // "what may I change?" but "what am I actually fine-tuning?", and it now has
  // three answers instead of one. It is also where the two refusals that
  // REMAIN have to be explained, or a greyed control reads as a bug.

  // The three quality levers get their OWN topic rather than more keywords on
  // the one above, because the question behind them is different: not "what may
  // I change here?" but "what should I change, and what does it cost me?". The
  // cost half is the reason — images-per-step is the only dense setting billed
  // by the hour — and the absences (EMA, min-SNR) need somewhere to be
  // explained, or they read as things we forgot rather than things that break.




  // Two questions behind one word. The refusal topic keeps its id (in-app help
  // badges and bookmarked links resolve against it), but the title and keywords
  // now cover BOTH answers: a packed export is refused, a plain fp8 cast is
  // allowed and merely costly.
  { id: 'training.quantized_base_refused', kind: 'setting',
    title: 'Which quantized checkpoints can be trained on, and which cannot',
    keywords: ['quantized', 'quantised', 'fp8', 'int8', 'gguf', 'custom weights',
      'base', 'refused', 'inference only', 'training', 'bf16', 'fp16', 'error',
      'scaled fp8', 'scale_weight', 'comfy_quant', 'packed export', 'fp8 cast',
      'cannot be loaded', 'strict', 'state dict', 'degraded', 'precision'],
    guide: { chapter: 'dataset-guide', anchor: '10-full-model-recipe-what-you-can-change' },
    app: { route: '/datasets?section=training' } },
  // The Krea base LIST is a different question from the quantization verdict a
  // listed entry may carry ("where are my models?" vs "why is this one greyed
  // out?"), and it is searched with the family name, so it gets its own topic.
  { id: 'training.krea_installed_bases', kind: 'setting',
    title: 'Training Krea 2 on a checkpoint you already have',
    keywords: ['krea', 'krea 2', 'base', 'base model', 'checkpoint', 'unet',
      'diffusion_models', 'my model', 'installed', 'continue training', 'merge',
      'community model', 'full model', 'not listed', 'missing from the list',
      'custom weights', 'absolute path'],
    guide: { chapter: 'dataset-guide', anchor: '1-pick-your-model-family-first' },
    app: { route: '/datasets?section=training' } },
  // Dual captions is a per-run Advanced training option (not a global Setting),
  // so it points at the dataset guide's dedicated section rather than
  // settings-reference, and its route is the training workspace section. Its tip
  // surfaces it when the Advanced options are first opened.
  { id: 'training.dual_captions', kind: 'setting', title: 'Dual captions (long + short)',
    keywords: ['dual captions', 'long', 'short', 'short caption', 'caption', 'augmentation',
      'short_and_long', 'advanced', 'training', 'krea', 'anima', 'cache_text_embeddings'],
    guide: { chapter: 'dataset-guide', anchor: '7-dual-captions-long-short' },
    app: { route: '/datasets?section=training' },
    tip: { trigger: 'dual-captions-advanced',
      text: 'New: train each image on a long AND a short caption (Advanced options → Dual captions) so the LoRA leans less on any single wording.' } },
  // 🎲 Use dataset captions — an action on the Preview-prompts field, in both
  // the LoRA and the full-model recipe. It writes an existing setting
  // (sample_prompts), so it points at the same Training section of the
  // settings reference where that field is documented.
  { id: 'training.sample_prompts_from_dataset', kind: 'action',
    title: 'Use dataset captions as preview prompts',
    keywords: ['preview prompts', 'sample prompts', 'sample_prompts', 'captions',
      'dataset captions', 'random', 'draw', 're-roll', 'reroll', 'dice', 'fill',
      'defaults', 'generic', 'trigger', 'advanced', 'training'],
    guide: { chapter: 'settings-reference', anchor: 'training' },
    app: { route: '/datasets?section=training&panel=advanced' },
    tip: { trigger: 'sample-prompts-from-dataset',
      text: 'Preview images can show YOUR subject instead of generic defaults: 🎲 Use dataset captions, under Preview prompts, fills the prompts from your own captions.' } },
  // Expert controls, not global Settings: factor is meaningful only for a LoKr
  // network, and the Krea fields intentionally surface one reported community
  // starting point without claiming its result transfers to every dataset.
  { id: 'training.lokr_factor', kind: 'setting', title: 'LoKr decomposition factor',
    keywords: ['lokr', 'lo kr', 'factor', 'decomposition', 'network', 'adapter',
      'rank', 'alpha', 'advanced', 'training', 'auto', 'krea'],
    guide: { chapter: 'settings-reference', anchor: 'training' },
    app: { route: '/datasets?section=training&panel=advanced' } },
  { id: 'training.krea_community_recipe', kind: 'setting', title: 'Krea Raw LoKr community starting point',
    keywords: ['krea', 'krea 2', 'krea raw', 'lokr', 'likeness', 'community recipe',
      'balanced', 'content', 'style', 'differential guidance', 'guidance scale',
      'automagic2', 'sigmoid', 'reddit', 'advanced', 'training', 'preset'],
    guide: { chapter: 'settings-reference', anchor: 'training' },
    app: { route: '/datasets?section=training&panel=advanced' } },
  // Person masking (`masked`, background at 10 %) became a per-DATASET setting on
  // 28/07 — it used to be a per-BROWSER localStorage preference the server only saw
  // at launch. Same shape as Dual captions / Memory saving: a per-dataset training
  // option, so it points at the settings-reference Training section, and its route
  // is the training workspace section where the toggle lives.
  { id: 'training.masked', kind: 'setting', title: 'Masked training (background at 10%)',
    keywords: ['masked', 'mask', 'person mask', 'masked training', 'background',
      'bg 10%', 'rembg', 'subject', 'isolate', 'loss weight', 'identity', 'room',
      'advanced', 'training', 'not installed', 'missing', 'ml extras',
      // It moved: people searching for where their old browser toggle went must
      // land here, and so must the readiness row that now names it.
      'per browser', 'localstorage', 'preference', 'phone', 'other machine',
      'trains unmasked', 'readiness', 'preparation'],
    guide: { chapter: 'settings-reference', anchor: 'training' },
    // Deliberately NO one-time tip: the What's-new entry announces the move, and
    // the panel already shows a targeted notice to the only browsers it affects
    // (the ones that had turned masking off). A third surface would be nagging.
    app: { route: '/datasets?section=training' } },
  // WHICH Klein model runs — a per-DATASET setting since 28/07. Improve took no
  // model at all (the server resolved one silently) and generation's picker was a
  // per-BROWSER localStorage value that improve never read, so "which model made
  // this?" had no answer on any screen. One setting now serves both.
  { id: 'dataset.klein_model', kind: 'setting', title: 'Klein model for this dataset',
    keywords: ['klein', 'model', 'base model', 'unet', 'diffusion model', 'which model',
      'choose model', 'pick model', 'improve', 'upscale', 'upscale & improve',
      'generation', 'flux2', 'flux 2', 'kv', '9b', '4b', 'safetensors', 'auto',
      'auto-detected', 'detected', 'comfyui models', 'model missing', 'moved',
      'not on disk', 'per browser', 'localstorage',
      // 29/07: the setting reaches every Klein lane the dataset owns, and each of
      // those screens now NAMES the model — so each is a way people look for it.
      'reference edit', 'edit reference', 'rescue', 'small images', 'under 768',
      'watermark', 'watermark clean', 'inpaint', 'klein inpaint', 'bank'],
    guide: { chapter: 'settings-reference', anchor: 'image-engines' },
    app: { route: '/datasets' } },
  // The 🧽 Klein clean's three dials (2026-08-31). They are NOT on the Settings screen:
  // they live next to the Klein model choice in the bank's Level 3 panel and the
  // dataset's Clean bar, because that is where the result is judged — so like
  // dataset.klein_model above these are hand-built topics pointing at the workspace,
  // not `setting()` entries pointing at a Settings field that does not exist. The
  // guide chapter is the one that documents them (Captioning & quality ▸ Watermark
  // inpainting).
  { id: 'watermark_clean.klein_prompt', kind: 'setting',
    title: 'Prompt sent to Klein (watermark clean)',
    keywords: ['klein', 'prompt', 'instruction', 'remove watermark', 'what is sent',
      'sent to klein', 'wording', 'text', 'edit prompt', 'custom prompt', 'clean',
      'watermark', 'inpaint', 'bank', 'dataset',
      // The words somebody writes when the pass did not remove their mark and they are
      // looking for something — anything — to turn.
      'still there', 'not removed', 'survived', 'did not work', 'logo remains',
      'signature', 'tiled', 'reset to default', 'default prompt'],
    guide: { chapter: 'settings-reference', anchor: 'captioning-quality' },
    app: { route: '/datasets' } },
  { id: 'watermark_clean.klein_max_mp', kind: 'setting',
    title: 'Processing size for the Klein clean (megapixels)',
    keywords: ['klein', 'megapixel', 'megapixels', 'mp', 'processing size', 'resolution',
      'quality', 'detail', 'sharper', 'blurry', 'soft', 'downscaled', '2 mp', '4 mp',
      'clean', 'watermark', 'vram', 'out of memory', 'oom', 'slow', 'faster', 'time',
      'bank', 'dataset'],
    guide: { chapter: 'settings-reference', anchor: 'captioning-quality' },
    app: { route: '/datasets' } },
  { id: 'watermark_clean.klein_output', kind: 'setting',
    title: 'What size the cleaned file is written at',
    keywords: ['klein', 'output', 'write back', 'dimensions', 'size', 'resized',
      'smaller', 'shrunk', 'my images got smaller', 'changed size', 'render size',
      'original dimensions', 'resample', 'clean', 'watermark', 'bank', 'dataset'],
    guide: { chapter: 'settings-reference', anchor: 'captioning-quality' },
    app: { route: '/datasets' } },
  // Concept face masking (issue #15) is a per-DATASET Advanced training option,
  // so like Dual captions it points at the dataset guide rather than
  // settings-reference. Its two tuning knobs live in Settings > Training and are
  // covered by the settings topics below.
  { id: 'training.qwenimage21', kind: 'setting', title: 'Train a Qwen-Image 2.1 LoRA',
    keywords: ['qwen', 'qwen image', '2.1', 'qwen_image_2', 'training', 'model family',
      'local', 'cloud', 'convrot8', 'ai toolkit'],
    guide: { chapter: 'dataset-guide', anchor: '1-pick-your-model-family-first' },
    app: { route: '/datasets' } },
  setting('studio_models', 'engines', 'studio-models', 'Prepare Test Studio image models',
    ['z-image', 'zimage', 'z-image turbo', 'qwen', 'qwen-image 2.1', 'flux', 'anima', 'moss', 'moss-image', 'test image', 'studio', 'base model',
     'text encoder', 'vae', 'install models', 'missing model', 'train then generate',
     'missing nodes', 'repair ComfyUI', 'update ComfyUI', 'check nodes again',
     'install node pack', 'custom node pack', 'detail daemon', 'DetailDaemonSamplerNode', 'matplotlib']),
  { id: 'training.mask_faces', kind: 'setting', title: 'Mask faces (Concept datasets)',
    keywords: ['mask faces', 'face mask', 'masking', 'concept', 'identity', 'bleed',
      'identity bleed', 'face bleed', 'character lora', 'combine loras', 'act',
      'anonymise', 'anonymize', 'advanced', 'training',
      // The optional detector this option depends on. Searching any of these
      // must land here, because this is where it is now installed from — the
      // capability is stored as `face_scoring`, but nobody calls it that.
      'insightface', 'face detection', 'detector', 'install', 'face scoring',
      'not installed', 'missing', 'onnxruntime', 'ml extras',
      // The preview can be stopped and picked back up. Searching for the way out
      // of a long pass must land on the option that started it.
      'stop', 'cancel', 'resume', 'continue', 'interrupt', 'looking for faces'],
    guide: { chapter: 'dataset-guide', anchor: '8-concept-loras-keeping-faces-out' },
    app: { route: '/datasets?section=training' },
    tip: { trigger: 'mask-faces-advanced',
      text: 'New for Concept datasets: mask the faces while training so the concept learns the act, not the people in your photos.' } },
  setting('face_mask.expand', 'training', 'face-mask-expand', 'Head coverage (face box x)',
    ['face mask', 'head', 'coverage', 'expand', 'dilate', 'hair', 'jaw', 'concept',
     'mask faces', 'tight', 'wide']),
  setting('face_mask.min_weight', 'training', 'face-mask-min-weight',
    'Loss weight kept on faces',
    ['face mask', 'weight', 'loss', 'min weight', 'concept', 'mask faces', 'zero',
     'strength', 'how hard']),
  // Memory saving (quantisation + low-VRAM loading) is a per-run Advanced
  // training option like Dual captions, so it points at the settings-reference
  // section that documents the Advanced panel rather than a global Settings card.
  { id: 'training.memory_saving', kind: 'setting', title: 'Memory saving (quantisation, low VRAM)',
    keywords: ['quantise', 'quantize', 'quantisation', 'qfloat8', 'fp8', 'low vram', 'lowvram',
      'vram', 'memory', 'oom', 'out of memory', '5090', '4090', '24 gb', '32 gb', 'slow',
      'speed', 'precision', 'text encoder', 'advanced', 'training',
      // The cross-family trap: these three flags are global while their
      // calibrated default is per family, so people search for why a run that
      // "worked on Anima" crawls or dies on Krea 2 / FLUX.
      'model family', 'switched family', 'lora type', 'carried over', 'crawl'],
    guide: { chapter: 'settings-reference', anchor: 'training' },
    app: { route: '/datasets?section=training' },
    tip: { trigger: 'memory-saving-advanced',
      text: 'New: if your card is bigger than 24 GB you can switch quantisation and low-VRAM loading off (Advanced options → Memory saving) for a faster, more precise run.' } },
  { id: 'training.speed', kind: 'setting', title: 'Speed (batch size, quantisation backend, checkpointing, compile)',
    keywords: ['speed', 'faster', 'slow', 'slower', 's/it', 'seconds per step', 'step time',
      'batch', 'batch size', 'gradient checkpointing', 'checkpointing', 'compile',
      'torch.compile', 'quantisation backend', 'qtype', 'convrot8', 'int8', 'w8a8',
      'qfloat8', 'fp8', 'throughput', 'hours', 'eta', 'onetrainer', 'comparison',
      'advanced', 'training'],
    guide: { chapter: 'settings-reference', anchor: 'training' },
    app: { route: '/datasets?section=training' },
    tip: { trigger: 'training-speed-advanced',
      text: 'New: Advanced options → Speed. Batch size, the quantisation backend, gradient checkpointing and compile were fixed at the values that fit 24 GB — on a bigger card you can trade that memory back for time.' } },
  // server
  setting('server.port', 'server', 'server-port', 'Server port',
    ['server', 'port', 'bind', 'network', '5050']),
  setting('server.lan', 'server', 'server-lan', 'Available on the local network',
    ['lan', 'network', 'remote', 'phone', 'wifi', 'host', 'bind']),
  setting('server.require_token', 'server', 'server-require-token', 'Require an access token',
    ['token', 'require', 'access', 'remote', 'phone', 'security', 'lan']),
  setting('server.access_token', 'server', 'server-token', 'Access token',
    ['token', 'access', 'remote', 'phone', 'password', 'qr']),
  // storage
  setting('paths.dataset_images_root', 'storage', 'dataset-images-root', 'Dataset images root',
    ['data', 'storage', 'path', 'dataset', 'images', 'root', 'location', 'disk']),

  setting('paths.checkpoints_dir', 'storage', 'checkpoints-dir', 'Checkpoint store folder',
    ['checkpoint', 'store', 'safetensors', 'where are my checkpoints', 'lost checkpoint',
      'deleted checkpoint', 'move', 'another drive', 'durable', 'disk']),
];
