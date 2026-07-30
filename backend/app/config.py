"""Config core: layered config.json over DEFAULTS, secrets in .env."""
import copy, json, os, secrets as _secrets, threading, unicodedata
from pathlib import Path
from dotenv import load_dotenv

LOCAL_USER = 'local'

BACKEND_DIR = Path(__file__).resolve().parent.parent          # backend/
REPO_ROOT = BACKEND_DIR.parent

def _data_dir() -> Path:
    return Path(os.environ.get('LDS_DATA_DIR', str(REPO_ROOT / 'data')))

def data_dir() -> Path:
    """Public accessor for the app's writable data directory (created on demand).
    Where app-managed artefacts live that aren't user datasets — e.g. the dedicated
    Python env the watermark-inpainting installer auto-provisions (data/envs/…)."""
    d = _data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d

def _config_path() -> Path:
    return Path(os.environ.get('LDS_CONFIG', str(REPO_ROOT / 'config.json')))

ENV_PATH = Path(os.environ.get('LDS_ENV', str(REPO_ROOT / '.env')))
load_dotenv(ENV_PATH)

# REDDIT_CLIENT_ID / CIVITAI_API_KEY / PEXELS_API_KEY: scraping credentials
# (Settings > Scraping & sources). Sources read their env var at request time,
# and set_secrets() stamps os.environ on save, so changes apply without restart.
SECRET_KEYS = ('GEMINI_API_KEY', 'OPENAI_API_KEY', 'OPENROUTER_API_KEY', 'HF_TOKEN',
               'VAST_API_KEY', 'REDDIT_CLIENT_ID', 'CIVITAI_API_KEY', 'PEXELS_API_KEY')

# A Krea install saved by a previous release can carry its old *defaults* in
# config.json, so a changed DEFAULTS value alone would never reach it. This
# marker lets us distinguish that one-time profile migration from settings the
# user changes after the new profile is available.
KREA_CALIBRATION_VERSION = 3
_LEGACY_KREA_GROUNDING_PX = 1024.0
_LEGACY_KREA_REF_BOOST = 4.0
_PREVIOUS_KREA_GROUNDING_PX = 512.0
_PREVIOUS_KREA_REF_BOOST = 1.0
_PREVIOUS_KREA_STEPS = 10

DEFAULTS = {
    # host: '127.0.0.1' = this machine only ; '0.0.0.0' = reachable from the LAN
    # (phone, tablet, another PC) — the Settings "Server" card's LAN toggle just
    # flips this. Port defaults to 5050 to match start.bat's default bind (so the
    # Settings port field shows what's actually running, not a phantom mismatch).
    # require_token (default OFF): a home LAN is trusted, so LAN access is open by
    # default — no token to type on a phone. Turn it ON to demand a token from
    # remote devices (access_token is then generated + persisted here so it
    # survives restarts and is copyable from Settings). Loopback never needs it.
    'server': {'host': '127.0.0.1', 'port': 5050, 'require_token': False, 'access_token': ''},
    'paths': {'dataset_images_root': ''},                      # '' -> DATA_DIR/datasets
    'comfyui': {'api_url': 'http://127.0.0.1:8188', 'base_dir': '',
                'output_dir': '', 'input_dir': '', 'models_dir': '', 'loras_dir': '',
                # setup_skipped (default False): the user consciously chose "continue
                # without ComfyUI" in the Setup wizard. It ONLY makes the Setup step
                # render a neutral "skipped" instead of nagging; it never gates a
                # capability. Setting base_dir annuls it (see settings.put_settings and
                # the DERIVED comfyui.skipped in capabilities.probe), so it can never
                # mask a real error of a configured ComfyUI.
                'setup_skipped': False,
                # Seconds ComfyUI is allowed to spend ANSWERING the /object_info
                # enumeration (the heaviest probe in the app). It is a READ budget
                # only: the connection itself still has to be accepted in
                # `utils.comfyui._OBJECT_INFO_CONNECT_TIMEOUT` seconds, so a ComfyUI
                # that is genuinely OFF never costs this. This has to be a setting
                # rather than a constant because the /object_info payload grows with
                # the number of custom nodes and model files INSTALLED — the richer
                # the install, the longer it takes, which is exactly why the old
                # hardcoded 8 s broke the people who had invested the most in their
                # ComfyUI (reported by j_o_e_l. on Discord, who measured ~15 s on his
                # own install). Clamped to 5-300 by utils.comfyui.object_info_timeout().
                'object_info_timeout_s': 45},
    'ollama': {'url': 'http://127.0.0.1:11434', 'vision_model': 'huihui_ai/qwen3-vl-abliterated:8b-instruct',  # -instruct, NOT ':8b' (=thinking): see get_vision_model()
               # How many vision calls a bank pass keeps in flight. 4 is the
               # measured knee; see services/vision_pool.py for the numbers.
               'vision_concurrency': 4,
               # Seconds an ISOLATED vision call may keep the model resident when
               # nothing else wants the GPU (0 = always unload, the old
               # behaviour). See services/vision_keepalive.py.
               'vision_keep_warm_seconds': 120},
    'aitoolkit': {'dir': '', 'datasets_dir': '', 'output_dir': '', 'hf_home': '',
                  # Explicit interpreter for installs without venv/.venv
                  # (conda, uv, system python). Empty = auto-detect.
                  'python': ''},
    # `enabled` is the ENGINE CATALOG as well as the default selection: adding an
    # engine here is what makes it reach existing installs (see _merge_new_engines
    # and LEGACY_KNOWN_ENGINES below). `known` is not a setting — it is the ledger
    # of which engines the app offered the last time the user picked, written by
    # save_config; [] means "no ledger yet".
    'engines': {'default': 'chatgpt',
                'enabled': ['nanobanana', 'chatgpt', 'openrouter', 'klein', 'krea'],
                'known': [],
                # chatgpt_auth: 'auto' = subscription when connected, else API key.
                'chatgpt_auth': 'auto',            # auto|api|subscription
                # The Codex ROUTER model of the subscription lane — NOT the image
                # model. The image model of the API-key lane is
                # engines.chatgpt_image_model (below); the subscription lane
                # renders on whatever OpenAI's image_generation tool serves that
                # plan (gpt-image-2 today) and takes no model from us. Two
                # different settings — never merged.
                'chatgpt_subscription_model': 'gpt-5.4-mini',
                # OpenRouter reaches the SAME upstream models as the two other
                # API engines through one account, so the slug is free text: its
                # catalogue moves fast and a renamed/retired model must never
                # require a new release. Default = the Nano Banana weights.
                'openrouter_model': 'google/gemini-3-pro-image',
                # Nano Banana / ChatGPT image models. Free text for the same
                # reason as OpenRouter's, but DEFAULTED TO BLANK, unlike it:
                # NANOBANANA_MODEL and CHATGPT_IMAGE_MODEL environment variables
                # already existed and some installs set them. A literal default
                # here would be a non-blank cfg.get() that silently outranked
                # that env var. Blank means "not chosen", so the documented
                # order holds: setting > environment variable > built-in default
                # (services/nanobanana.DEFAULT_MODEL and
                # services/chatgpt_image.DEFAULT_IMAGE_MODEL).
                'nanobanana_model': '',
                'chatgpt_image_model': ''},
    'captioning': {'backend': 'auto'},                         # auto|joycaption|ollama|none
    # 📥 What happens to a photo the moment it enters a dataset. Until now this
    # was two hardcoded numbers with no sentence anywhere saying they existed
    # (reported by Qeeyana on Reddit: "images added to dataset are automatically
    # normalized to 1024. Why? Let me choose not to.").
    #
    # max_side: longest side kept, in px, when a WebP normalization mode is
    #   explicitly selected. 0 = store at the original size. Whatever the value,
    #   the ceiling below applies: it is a format limit, not a preference.
    # encoding: 'preserve' (the shipped default) keeps supported static source
    #   bytes exactly as supplied: JPEG, PNG, WebP or BMP. 'standard' (WebP q92),
    #   'high' (WebP q100) and 'lossless' (lossless WebP) remain opt-in legacy
    #   normalization modes. `max_side` deliberately does not affect `preserve`:
    #   training creates its own disposable PNG staging copy at launch, so an
    #   import must not throw away the user's master first.
    #
    # Applies to the dataset INGEST lanes only (photo import, kohya ZIP/folder
    # merge, scrape-to-dataset). It does not touch generated images, the ≤2048
    # copies handed to a generation API, or an image the user already curated.
    'dataset_import': {'max_side': 1024, 'encoding': 'preserve'},
    'training': {'default_family': 'zimage'},
    # Concept face masking (opt-in per dataset, Advanced training options). Both
    # knobs are exposed because NOBODY has measured the right value: no public A/B
    # of a concept LoRA trained with vs without face masking exists, so shipping a
    # frozen number would be a guess dressed as a default.
    #
    # `expand`: how far the detected FACE box is grown to become a HEAD box.
    # InsightFace returns eyes-to-chin; untouched it leaks jaw, hair and neck.
    # 2.0 is the only published default for this exact chain (ai-toolkit-perceptual's
    # face_suppression_expand, documented "1.8-2.0 = full head coverage").
    #
    # `min_weight`: the loss weight left INSIDE the mask (ai-toolkit maps black ->
    # mask_min_value). NOT zero, on purpose, and the floor below is not cosmetic:
    #   - a zero-weight region is not "ignored", it is unpenalised — the model may
    #     put anything there at no cost (OneTrainer discussion #347: phantom limbs,
    #     edge artefacts), and the only published sweep of this knob (SECourses, 9
    #     runs) reports "anatomically disproportional" output below 0.1;
    #   - ai-toolkit divides the mask by its own mean (SDTrainer), so an image
    #     masked edge to edge at exactly 0.0 divides by zero -> NaN loss -> dead run.
    'face_mask': {'expand': 2.0, 'min_weight': 0.1},
    # Cloud GPU training (vast.ai). Everything has a sane default: the only
    # required user input is the VAST_API_KEY secret. Values here are knobs
    # for power users / for adjusting after the real-world smoke test.
    'cloud': {
        # Official vast.ai "Ostris AI Toolkit" template (smoke-validated
        # 2026-07-12): publishes the UI behind the pod's Caddy proxy on 18675
        # and generates the per-instance auth token. Clearing this falls back
        # to a raw-image launch using `image`/`onstart` below.
        'template_hash': '471ed5903d8cdb8e63b0d0e50f6cd519',
        'ui_port': 18675,              # container port the UI is reachable on (Caddy proxy)
        'image': 'vastai/ostris-ai-toolkit:4625406-2026-07-12-cuda-12.9',  # raw-image fallback only
        'max_price_per_hour': 0.80,    # background safety cap on offer price, $/h
        'offer_scan_limit': 100,       # offers fetched when listing GPU speed tiers
        'pod_overhead_minutes': 35,    # boot+model download+quantize (measured ~40 min live), in cost estimates
        'max_concurrent_runs': 1,      # simultaneous cloud pods; raise in Settings
        'min_inet_down_mbps': 400,     # skip hosts too slow to pull the 7 GB image
        'min_disk_bw_mbps': 500,       # skip hosts too slow to EXTRACT it (frozen 'loading')
        'min_reliability': 0.98,       # vast reliability floor (0.95 let a dead host through)
        # Offer trust filters. verified_only=True preserves the historical
        # behaviour; Secure Cloud is Vast's `datacenter` tier and is opt-in
        # because it usually narrows the marketplace and raises the price.
        'verified_only': True,
        'secure_cloud_only': False,
        'host_blacklist_days': 3,      # skip hosts whose pod never became ready
        # A host killed while it was still VISIBLY booting (see boot_budget below)
        # was slow, not broken: it is skipped for hours, not days, so a bad night
        # on one uplink does not silently shrink the marketplace for three days.
        'slow_boot_blacklist_hours': 6,
        # Boot is guarded by the same two clocks as the pre-step-1 phase below.
        # ready_timeout is IDLE time — rearmed every time the pod shows a boot
        # fact it had never shown before (a new vast status, the UI port getting
        # published, a moving host progress line), so a pod honestly pulling a
        # 26 GB image is never cut; boot_budget is the ABSOLUTE ceiling on the
        # phase, so a host too slow to ever finish still dies fast (0 = none).
        'ready_timeout_minutes': 25,   # no boot progress at all past this -> kill
        'boot_budget_minutes': 90,     # hard ceiling on the whole boot phase
        'max_runtime_minutes': 480,    # safety net (stall watchdog is the first line): hard stop past this
        'stall_timeout_minutes': 30,   # no step progress past this -> rescue + kill
        # Before step 1 the pod is fetching the base model. Two clocks guard it:
        # first_step_timeout is IDLE time — rearmed every time the pod's log
        # reports more downloaded bytes, so an honestly slow download is never
        # cut; download_budget is the ABSOLUTE ceiling on that phase, so a host
        # too slow to ever finish still dies well before the runtime cap
        # (0 = no ceiling, runtime cap is then the only backstop).
        'first_step_timeout_minutes': 45,  # no step 1 AND no new bytes past this -> kill
        'first_step_download_budget_minutes': 180,  # hard ceiling on the pre-step-1 phase
        # Out-of-monitor freeze watchdog: a training run whose own monitor stopped
        # reporting for this long is terminated by the supervisor (0 = only warn
        # in the UI, never cut). Slow-by-design phases (boot/upload/download) are
        # never judged on this value -- they get a fixed 2 h floor.
        'freeze_watchdog_minutes': 45,
        'unreachable_grace_minutes': 6,  # tolerated mid-run network blackout before giving up on the pod
        'monthly_budget_usd': 0,       # 0 = unlimited; launches blocked past this
        'disk_gb': 60,                 # instance disk (base model + dataset + checkpoints)
        # min_vram_gb est PAR FAMILLE (pas par variante) : pour flux2klein on prend
        # 32 — le 9B (32-48 GB) est la voie cloud principale de cette famille, et un
        # pod 32 GB entraîne aussi le 4B sans problème (l'inverse serait faux).
        'min_vram_gb': {'zimage': 24, 'sdxl': 16, 'krea': 24, 'flux2klein': 32},
        'onstart': '',                 # raw-image fallback: optional startup command
    },
    'face_scoring': {'python': '', 'models_root': '', 'green': 0.50, 'orange': 0.45},
    # 🗃️ Image bank triage thresholds. Raw scores are persisted per image;
    # these thresholds only drive the FLAGS computed at read time — so tuning
    # them re-sorts an already-scanned bank instantly, no rescan needed.
    # sharpness_min: Laplacian variance below this = flagged blurry (the classic
    #   ~100 rule of thumb). noise_max: residual std above this = flagged noisy.
    # uniformity_min: grayscale std below this = flagged flat/uniform (solid
    #   colors, empty screenshots). dup_distance: dHash Hamming distance (same
    #   64-bit hash as dataset imports) at or under which two images group as
    #   near-duplicates. min_side: smaller side under this = flagged small
    #   (mirrors the dataset import guard: trainers only downscale).
    # face_threshold: cosine similarity at or above which two faces are the
    #   same person when clustering the bank by subject.
    # aesthetic_min: LAION aesthetic score (~1..10) below which an image is flagged
    #   'low_aesthetic' — the "keep the nice ones" cut of a mixed dump.
    # nsfw_max: NSFW probability (0..1) above which is_nsfw is flagged, to split a
    #   mixed SFW/NSFW dump.
    # style_threshold: cosine similarity on the CLIP image embeddings at or above
    #   which two images share a visual STYLE when clustering by style.
    # semantic_dup_threshold: cosine similarity on the SAME CLIP embeddings at or
    #   above which two scored images are flagged a SEMANTIC near-duplicate (stage 2:
    #   crops / re-compressed variants of the same shot a dHash misses). Higher than
    #   style_threshold on purpose — a crop is far closer than merely "same style".
    'bank': {'sharpness_min': 100.0, 'noise_max': 15.0, 'uniformity_min': 12.0,
             'dup_distance': 8, 'min_side': 768, 'face_threshold': 0.45,
             'aesthetic_min': 5.0, 'nsfw_max': 0.5, 'style_threshold': 0.6,
             'semantic_dup_threshold': 0.96,
             # detail_min: effective resolution (0..1 of the stored size) below
             #   which an image is flagged 'soft_detail' — its pixels promise more
             #   picture than they deliver. 0.72 was picked on a real 36 000-image
             #   bank: it selects the softest ~3%, and sits below the 10th
             #   percentile of images measured to be genuinely full-resolution, so
             #   a sharp photo does not trip it. Raise it to be pickier.
             'detail_min': 0.72,
             # bars_max: fraction of the frame allowed to be flat black letterbox
             #   before the 'bars' flag. 0.04 ~ a thin band; it caught ~4% of the
             #   reference bank (screenshots of videos, padded stills).
             'bars_max': 0.04},
    'masks': {'python': ''},
    # Bank ✨ Score pass interpreter (CLIP aesthetic/NSFW stack). Auto-provisioned
    # by the bank_scoring installer into its own venv — declared here so a
    # full-config Save round-trips it instead of failing "unknown config section".
    # text_search_idle_minutes: how long the 🔤 text-search encoder stays warm
    #   after its last query. Loading CLIP costs ~8 s; encoding a phrase costs
    #   ~20 ms — so the worker is kept alive to make a refine-and-retry session
    #   instant, and reaped afterwards because it holds ~2.4 GB of RAM. 0 means
    #   "never stay warm": every distinct query pays the ~8 s load, which is the
    #   right trade on a memory-tight machine.
    'bank_scoring': {'python': '', 'text_search_idle_minutes': 10},
    # Watermark inpainting (simple-lama-inpainting, extra ML). Dedicated key so a
    # user can override it, but defaults empty -> reuse the same ML interpreter as
    # rembg/insightface (masks.python) then sys.executable. Never imported in-process.
    # allow_crop (default True = the shipped behaviour): when False the auto-routing
    # NEVER crops a border mark — it repaints it instead (LaMa/Klein per the chosen
    # engine). A persisted user preference (Settings ▸ Watermark inpainting AND the
    # batch Clean bar both edit it); the review lightbox can still override it per image.
    'watermark': {'python': '', 'device': 'auto', 'allow_crop': True},  # auto|cuda|cpu
    # consistency_strength: the dx8152 LoRA anchors STRUCTURE (composition/
    # background), not the face — its own guide says start at 0.5 and that
    # 0.8-1.0 "can prevent edits from applying". 0.9 made every variation a
    # near-copy of the reference. 0 disables the LoRA entirely.
    'klein': {'consistency_lora': 'klein/Flux2-Klein-9B-consistency-V2.safetensors',
              'consistency_strength': 0.5,
              # Optional generation-LoRA PRESETS (Idea by @waltm — Discord
              # feature request): named combinations the user picks per run.
              # Each preset: {name, loras: [{file, strength}]} — loras is an
              # ORDERED list (list order = chain order after the consistency
              # LoRA on the local Klein edit graph), file is a loras-relative
              # name (like consistency_lora; the app never hardcodes one).
              # There is deliberately NO automatic per-LoRA gating: the chosen
              # preset carries the intent (make an "NSFW full" preset if you
              # want one). Caps: 8 LoRAs/preset, 12 presets
              # (klein_edit_helper.MAX_GENERATION_LORAS / _PRESETS). The older
              # generation_loras flat list and the very old ultra_real_lora /
              # nsfw_lora keys are migrated in by _migrate_klein_loras() and
              # then dropped.
              'generation_lora_presets': [],
              # Optional instruction for small scraped-image rescue only.
              # Empty is intentional: never invent a restoration prompt for the user.
              'small_image_prompt': '',
              # Sampler steps for Klein GENERATION (variations, regenerate, small-image
              # rescue). 5 = the value hardcoded in the shipped workflow (node 77), so an
              # untouched install renders exactly as before. More steps = slower, usually
              # a cleaner render; clamped to 50 (face_dataset_service._IMPROVE_MAX_STEPS).
              # Raised on request by ashish.sinha (Discord). Separate from improve_steps,
              # which drives the manual "Upscale & improve" pass only.
              'generation_steps': 5,
              # Manual "Upscale & improve" quality profile. Its INSTRUCTION was
              # already editable (identity_prompts.klein_improve) but the knobs
              # deciding how much the pass actually changes were hardcoded at the
              # call site — including BOTH LoRA strengths pinned to 0, which meant
              # the workflow's own realistic LoRA (0.8 in improve skin.json) never
              # applied at all. These defaults reproduce that exact historical
              # behaviour, so an untouched install renders byte-identically; raise
              # improve_base_lora_strength to actually let that LoRA work.
              'improve_steps': 4,
              'improve_base_lora_strength': 0.0,
              # Overrides klein.consistency_strength for THIS pass only. It is the
              # dx8152 consistency LoRA (anchors composition/background), NOT an
              # identity LoRA — clamped [0, 1.5] by enqueue_klein_edit.
              # 1.0 where generation defaults to 0.5: dx8152 warns that 0.8-1.0 "can
              # prevent edits from applying", which is a problem for a restaging and
              # exactly the point here — an improve pass must add detail WITHOUT
              # redrawing the composition. Tuned on real runs, not from the guide.
              'improve_consistency_strength': 1.0,
              # Total pixel budget the source is rescaled to before sampling, so it
              # is the output resolution. 2 = the value hardcoded in the workflow.
              'improve_megapixels': 2.0},
    # Krea 2 Identity Edit — the second LOCAL generation engine (services/
    # krea_edit_helper.py). Every value here is a RESOLUTION HINT or a sampler
    # knob, never a hardcoded machine path: blank/absent means "find it yourself"
    # (canonical filename first, then a narrow token match, across every
    # extra_model_paths root), which is what makes the engine work on installs
    # that look nothing like the developer's.
    'krea': {
        # Internal marker for the calibrated dataset-restaging defaults below.
        # It is intentionally not a user-facing setting.
        'calibration_version': KREA_CALIBRATION_VERSION,
        # Blank = auto-resolve a Krea 2 base under any 'krea'-named model folder,
        # preferring a Turbo then a Raw build. Set it to a filename to pin one.
        'base_model': '',
        # The edit LoRA the whole engine hangs on. Not found under this name ->
        # the resolver scans the loras roots for a krea2_identity_edit* file, so a
        # renamed download still works.
        'identity_lora': 'krea/krea2_identity_edit_v1_2.safetensors',
        # Optional always-on generation LoRAs, as NAMED presets (mirrors
        # klein.generation_lora_presets — the mechanism is @waltm's idea). Pure
        # user data: [{name, loras: [{file, strength}]}], empty by default, and
        # the ONLY source of truth for which files may chain and in what order.
        # Krea never had the legacy single-slot LoRA keys Klein carries, so there
        # is no migration and no save carve-out — _deep_merge preserves a list
        # the incoming partial doesn't mention.
        'generation_lora_presets': [],
        # THE consistency <-> prompt-adherence dial, in pixels: the resolution the
        # reference is shown to the vision text-encoder at. LOW = follows the
        # PROMPT (more variety, weaker likeness); HIGH = RESEMBLES the reference
        # (stronger likeness, but it starts copying the pose and the outfit you
        # asked it to change). 512 is the measured default for dataset poses: it
        # keeps identity while giving the prompt room to move the pose.
        'grounding_px': 512,
        # Pack reference workflow values, measured working. cfg is pinned at 1.0
        # in code (guidance-distilled model) and is deliberately NOT a setting.
        'steps': 8,
        'identity_lora_strength': 1.0,
        # How hard the source latent is pushed back into the model each step.
        'ref_boost': 0.25,
    },
    # Z-Image pipeline — the two loader refs the shipped Test Studio workflow used
    # to hardcode from the developer's own ComfyUI (reported by bobba84, GitHub #18).
    # BLANK = "find it yourself": services/zimage_model_resolver scans every
    # registered vae / text_encoders root, sub-folders included, case- and
    # separator-insensitively (z_ae, z ae, z-ae, ae.safetensors; qwen_3_4b in any
    # sub-folder). Set either to a filename to PIN it — a pinned value is used as-is
    # and is never second-guessed, which is also the escape hatch when a shared
    # ComfyUI carries several plausible files (a FLUX.1 `ae.safetensors`, say).
    'zimage': {'vae': '', 'text_encoder': ''},
    # Editable identity / quality prompts (feature request by @bbsorry / 雨田壹).
    # The identity "locks" that ride ahead of every generated variation used to be
    # hardcoded and invisible; these overrides expose them without touching the
    # reproducibility invariant. EACH string default is '' on purpose: blank means
    # "use the shipped default", so the no-override path stays byte-identical to
    # the historical hardcoded prompt (get_identity_prompt falls back to the
    # constant). A non-blank value wins. Keys:
    #   face_single  — API-engine identity guard, single reference (IDENTITY_GUARD)
    #   face_multi   — API-engine identity guard, multi reference (IDENTITY_GUARD_MULTI)
    #   klein_identity — Klein restage + face-identity block (wrap_variation_klein)
    #   klein_improve  — the fixed "Klein upscale & improve" instruction
    # klein_improve_enabled (default True): when False the manual "Klein upscale &
    # improve" applies NO prompt at all (pure upscale), instead of the default/override.
    # The four flat keys above are the HUMAN overrides and keep their historical
    # names/meaning (never renamed — they are in user config files since the feature
    # shipped). `by_subject` holds the non-human ones,
    # {animal|creature|object|other: {face_single|face_multi|klein_identity: text}},
    # each read with NO fallback to the flat key: an override written on an Animal
    # dataset must never ride on a human generation (reported by ashish.sinha).
    # Empty by default — a subject with no entry follows its shipped default.
    # The five OTHER prompt parts, hardcoded until this wave and shipped in every
    # local-edit prompt: markings_lock (the skin hold order), outfit_vary /
    # expression_neutral (the two directives baked into every human shot),
    # outfit_palette (the concrete garments, one per LINE), render_tail_sfw /
    # render_tail_nsfw (the photographic tail) and framing_face|bust|body|back
    # (the per-framing detail block). Same contract as the four above — '' means
    # the shipped default — and the same split: the tail and the framing blocks
    # are per subject (anime's tail asks for an illustration, not a photograph) so
    # they live under `by_subject` for non-human types; the rest are flat.
    'identity_prompts': {'face_single': '', 'face_multi': '', 'klein_identity': '',
                         'klein_improve': '', 'klein_improve_enabled': True,
                         'markings_lock': '', 'outfit_vary': '', 'expression_neutral': '',
                         'outfit_palette': '', 'render_tail_sfw': '', 'render_tail_nsfw': '',
                         'framing_face': '', 'framing_bust': '', 'framing_body': '',
                         'framing_back': '',
                         'by_subject': {}},
    # User shot catalogs imported from JSON, {subject_type: [{id,label,prompt,
    # framing,nsfw?}]} — idea by ashish.sinha (Discord): have an LLM write 40 shots
    # instead of typing them. Stored SERVER-side rather than in localStorage so a
    # catalog survives a browser wipe, shows up on the phone as well as the desktop
    # and rides along in the full backup. Written by the workspace's Import button
    # (validated client-side by shotImport.js) and re-checked on read by
    # face_variations.sanitize_custom_shots — this file is hand-editable, and a
    # label shadowing a built-in one would hijack prompt/aspect/NSFW resolution.
    'custom_shots': {},
    'updates': {'repo': 'perfectgf/lora-dataset-studio'},      # GitHub repo for the release feed
}

_lock = threading.Lock()
_cache = None


def defaults() -> dict:
    """A deep COPY of the shipped defaults, for callers that must show the user
    what a setting would be if they never touched it.

    Exposed over the API (`config_defaults` in the settings payload) so the
    Settings UI can offer a per-field "Reset to default" without ever holding a
    second copy of these numbers. A literal typed into the frontend would go
    stale the next time a default moves here, and the reset button would then
    quietly restore a value that is no longer the default — a lie the user
    cannot see. Derived, never duplicated: this is the SAME dict the merge in
    load_config() uses.

    A copy, not the live object: a caller mutating the returned tree (jsonify
    does not, but a future one might) must not rewrite the app's defaults."""
    return copy.deepcopy(DEFAULTS)

def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _number_is(value, expected):
    """Strict-enough numeric comparison for hand-editable JSON settings."""
    try:
        return float(value) == expected
    except (TypeError, ValueError):
        return False


def _migrate_krea_pose_profile(conf: dict, stored: dict, incoming: dict | None = None) -> dict:
    """Apply calibrated Krea pose defaults only to untouched old profiles.

    ``config.json`` overrides ``DEFAULTS``. The first Krea profile saved
    1024 / 4.0; calibration v2 then saved
    512 / 1.0 / 10.  Both are reference-dominated for diverse dataset poses. Only
    those exact shipped profiles are upgraded. Any other value is an intentional
    choice. An explicit save of a calibration knob is also intentional and gets
    the marker rather than being rewritten.
    """
    stored_krea = (stored or {}).get('krea')
    krea = conf.get('krea')
    if not isinstance(stored_krea, dict) or not isinstance(krea, dict):
        return conf
    try:
        stored_version = int(stored_krea.get('calibration_version', 0) or 0)
    except (TypeError, ValueError):
        stored_version = 0
    if stored_version >= KREA_CALIBRATION_VERSION:
        return conf

    incoming_krea = (incoming or {}).get('krea')
    explicit_calibration = isinstance(incoming_krea, dict) and any(
        key in incoming_krea for key in ('grounding_px', 'ref_boost', 'steps'))
    if explicit_calibration:
        krea['calibration_version'] = KREA_CALIBRATION_VERSION
        return conf

    is_v1_default = (
        stored_version < 2
        and _number_is(stored_krea.get('grounding_px'), _LEGACY_KREA_GROUNDING_PX)
        and _number_is(stored_krea.get('ref_boost'), _LEGACY_KREA_REF_BOOST)
        # Older config files did not record steps, so absence means their
        # shipped 10-step profile; a present non-10 value is a user choice.
        and _number_is(stored_krea.get('steps', _PREVIOUS_KREA_STEPS),
                       _PREVIOUS_KREA_STEPS))
    is_v2_default = (
        stored_version == 2
        and _number_is(stored_krea.get('grounding_px'), _PREVIOUS_KREA_GROUNDING_PX)
        and _number_is(stored_krea.get('ref_boost'), _PREVIOUS_KREA_REF_BOOST)
        and _number_is(stored_krea.get('steps', _PREVIOUS_KREA_STEPS),
                       _PREVIOUS_KREA_STEPS))
    if is_v1_default or is_v2_default:
        krea['grounding_px'] = DEFAULTS['krea']['grounding_px']
        krea['ref_boost'] = DEFAULTS['krea']['ref_boost']
        krea['steps'] = DEFAULTS['krea']['steps']
        krea['calibration_version'] = KREA_CALIBRATION_VERSION
    return conf

# --- engines added by an update --------------------------------------------
# _deep_merge REPLACES lists (it only recurses into dicts), which is right for
# every other list we store — but engines.enabled doubles as "which engines
# exist", so a config saved before an engine shipped pinned its owner to the old
# catalogue forever: the more someone used the app, the fewer new engines they
# got, with no hint one existed. New SCALAR keys never had this problem, they
# fall back to their default; this is a list-only failure mode.
#
# The whole difficulty is telling "this engine didn't exist when I saved" from
# "I unchecked this on purpose" — blindly adding back what's missing would undo
# an explicit choice, which is worse than the bug. So a save records the
# catalogue the choice was made from (engines.known), and only engines absent
# from that ledger are merged in on read. Configs written before the ledger
# existed have no such record, but we know from the shipping history exactly
# which engines they could have been offered:
LEGACY_KNOWN_ENGINES = ('nanobanana', 'chatgpt', 'klein')
# ^ never extend this tuple. A new engine goes in DEFAULTS['engines']['enabled']
# and nowhere else; adding it here would mean "everyone has already seen it",
# i.e. exactly the bug this fixes.
#
# Only ONE key in DEFAULTS is a list of choices (engines.enabled) — the other
# list, klein.generation_lora_presets, is pure user data with an empty default
# and nothing to merge. Hence a named, tested helper rather than a framework;
# a second list-of-choices key should reuse the same known/enabled shape.


def _clean_engines(seq):
    return [e for e in (seq or []) if isinstance(e, str) and e]


def _engine_catalog(*groups):
    """Every engine this build knows about, in DEFAULTS order, plus any extra
    (older or hand-written) names the caller passes — nothing is ever dropped."""
    out = list(DEFAULTS['engines']['enabled'])
    for group in groups:
        for e in _clean_engines(group):
            if e not in out:
                out.append(e)
    return out


def _merge_new_engines(conf: dict, user: dict) -> dict:
    """Add engines that appeared since the user's saved selection (in place).

    Read-time only — the config file is never rewritten, so the fix applies to
    every existing install without a migration, and a downgrade still finds what
    it wrote. `user` is the raw file: an absent engines.enabled means the user
    never expressed a choice and already sits on the full default catalogue.

    Doubles as the shape guard for this section — config.json is hand-editable
    and a string where a list belongs would otherwise reach every consumer."""
    eng = conf.get('engines')
    if not isinstance(eng, dict):
        eng = conf['engines'] = copy.deepcopy(DEFAULTS['engines'])
    if not isinstance(eng.get('enabled'), list):
        eng['enabled'] = list(DEFAULTS['engines']['enabled'])
    saved = ((user or {}).get('engines') or {})
    saved = saved.get('enabled') if isinstance(saved, dict) else None
    if not isinstance(saved, list):
        return conf
    enabled = _clean_engines(eng.get('enabled'))
    if not enabled:
        # An empty list reads as "no restriction" downstream (face_dataset_service);
        # filling it in would turn that into a real, restrictive selection.
        eng['enabled'] = enabled
        return conf
    known = ((user or {}).get('engines') or {}).get('known')
    known = _clean_engines(known) if isinstance(known, list) else []
    known = known or list(LEGACY_KNOWN_ENGINES)
    eng['enabled'] = enabled + [e for e in DEFAULTS['engines']['enabled']
                                if e not in known and e not in enabled]
    eng['known'] = _engine_catalog(known, eng['enabled'])
    return conf


def _stamp_known_engines(merged: dict, partial: dict) -> dict:
    """Record the catalogue a selection was made from, on the saves that carry
    one. Only then: a save of some unrelated section must not certify that its
    author ever saw the engines they don't have enabled."""
    incoming = (partial or {}).get('engines')
    eng = merged.get('engines')
    if not isinstance(incoming, dict) or 'enabled' not in incoming or not isinstance(eng, dict):
        return merged
    eng['known'] = _engine_catalog(eng.get('known'), eng.get('enabled'))
    return merged


MIGRATED_LORA_PRESET_NAME = 'My LoRAs'

def _migrate_klein_loras(conf: dict, convert: bool = True) -> dict:
    """Two-stage soft migration of the pre-preset generation-LoRA formats into
    klein.generation_lora_presets (in place):
      (a) the very old single-slot keys ultra_real_lora / nsfw_lora become rows
          of the intermediate flat list (keeping their configured strengths);
      (b) a non-empty flat `generation_loras` list becomes ONE named preset
          ('My LoRAs'); the per-row nsfw_only flag is dropped — presets carry
          the intent now.
    Every legacy key is then removed so it can't shadow the presets. Idempotent
    (the preset is only created once, by name) and applied on EVERY load — a
    config.json written by any older version keeps working — and on save,
    which purges the legacy keys from the file.
    `convert=False` drops the legacy keys WITHOUT converting them: used when a
    save explicitly carries `generation_lora_presets` (the client already
    speaks the preset format, so the presets are authoritative — otherwise
    deleting the migrated preset in Settings would resurrect it from the
    file's legacy keys)."""
    k = conf.get('klein')
    if not isinstance(k, dict):
        return conf
    # (a) single-slot keys -> intermediate flat rows
    lst = k.pop('generation_loras', None)
    lst = [dict(e) for e in lst if isinstance(e, dict)] if isinstance(lst, list) else []
    for file_key, strength_key in (('ultra_real_lora', 'ultra_real_strength'),
                                   ('nsfw_lora', 'nsfw_strength')):
        f = (k.pop(file_key, '') or '')
        f = f.strip() if isinstance(f, str) else ''
        s = k.pop(strength_key, None)
        if convert and f and not any(e.get('file') == f for e in lst):
            lst.append({'file': f,
                        'strength': float(s) if isinstance(s, (int, float)) else 0.6})
    # (b) flat rows -> one named preset (nsfw_only dropped on purpose)
    presets = k.get('generation_lora_presets')
    presets = [dict(p) for p in presets if isinstance(p, dict)] if isinstance(presets, list) else []
    if convert:
        rows = []
        for e in lst:
            f = e.get('file')
            f = f.strip() if isinstance(f, str) else ''
            if not f:
                continue
            s = e.get('strength')
            rows.append({'file': f,
                         'strength': float(s) if isinstance(s, (int, float)) else 0.6})
        if rows and not any(p.get('name') == MIGRATED_LORA_PRESET_NAME for p in presets):
            presets.append({'name': MIGRATED_LORA_PRESET_NAME, 'loras': rows})
    k['generation_lora_presets'] = presets
    return conf

def load_config(force=False) -> dict:
    global _cache
    with _lock:
        if _cache is not None and not force:
            return copy.deepcopy(_cache)
        user = {}
        p = _config_path()
        if p.exists():
            try:
                user = json.loads(p.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                user = {}
        if not isinstance(user, dict):
            user = {}
        _cache = _merge_new_engines(
            _migrate_klein_loras(
                _migrate_krea_pose_profile(_deep_merge(DEFAULTS, user), user)),
            user)
        return copy.deepcopy(_cache)

def save_config(partial: dict) -> dict:
    global _cache
    with _lock:
        p = _config_path()
        current = {}
        if p.exists():
            try:
                current = json.loads(p.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                current = {}
        # convert=False when this save explicitly carries the presets: the
        # client already speaks the preset format, so a legacy key left in the
        # file must not resurrect a preset the user just deleted — only purge.
        if not isinstance(current, dict):
            current = {}
        merged = _stamp_known_engines(_migrate_klein_loras(
            _migrate_krea_pose_profile(_deep_merge(current, partial or {}), current,
                                       partial or {}),
            convert='generation_lora_presets' not in ((partial or {}).get('klein') or {})),
            partial)
        tmp = p.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding='utf-8')
        tmp.replace(p)
        _cache = None
    return load_config()

def get(dotted: str, default=None):
    node = load_config()
    for part in dotted.split('.'):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node

def is_configured() -> bool:
    return _config_path().exists()

def secret(name: str):
    val = (os.environ.get(name) or '').strip()
    return val or None


# str.splitlines() recognizes more separators than CR/LF.  In particular, a
# vertical tab (and the Unicode line/paragraph separators) embedded in a value
# becomes a fresh NAME=value assignment the next time the file is rewritten.
# CR/LF remain valid delimiters *between* .env assignments; every other
# splitlines separator is rejected in an existing file before it can be
# normalized into a real newline.
_UNSAFE_ENV_FILE_SEPARATORS = frozenset(
    '\x00\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029'
)
_UNSAFE_SECRET_CATEGORIES = frozenset({'Cc', 'Cf', 'Zl', 'Zp'})


def _validated_secret_updates(d: dict) -> dict:
    updates = {}
    for name, value in (d or {}).items():
        if name not in SECRET_KEYS or not value:
            continue
        if (not isinstance(value, str)
                or any(unicodedata.category(ch) in _UNSAFE_SECRET_CATEGORIES
                       for ch in value)):
            raise ValueError(f"secret '{name}' must be a single line of text")
        updates[name] = value
    return updates


def _read_safe_env_lines() -> list[str]:
    if not ENV_PATH.exists():
        return []
    raw = ENV_PATH.read_text(encoding='utf-8')
    if any(ch in _UNSAFE_ENV_FILE_SEPARATORS for ch in raw):
        raise ValueError(
            'the existing .env contains an unsafe non-standard line separator'
        )
    return raw.splitlines()


def validate_secrets(d: dict) -> None:
    """Validate a prospective secret update without changing disk or process env.

    Routes call this before saving config.json so an invalid combined request is
    rejected atomically.  set_secrets repeats the checks as defence in depth for
    non-HTTP callers.
    """
    updates = _validated_secret_updates(d)
    if updates:
        _read_safe_env_lines()


def _quote_env_value(value: str) -> str:
    """Return a python-dotenv single-quoted value that round-trips exactly."""
    escaped = value.replace('\\', '\\\\').replace("'", "\\'")
    return f"'{escaped}'"


def set_secrets(d: dict) -> None:
    updates = _validated_secret_updates(d)
    if not updates:
        return
    lines = _read_safe_env_lines()
    for name, value in updates.items():
        lines = [l for l in lines if not l.startswith(f'{name}=')]
        lines.append(f'{name}={_quote_env_value(value)}')
    ENV_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    # Do not mutate the live process until persistence has succeeded.
    for name, value in updates.items():
        os.environ[name] = value
    load_dotenv(ENV_PATH, override=True)

def delete_secrets(names) -> None:
    """Remove saved secrets outright (clear a key). Separate from set_secrets,
    which SKIPS empty values on purpose so a blank field can't wipe a key by
    accident — deletion has to be an explicit action."""
    names = [n for n in (names or []) if n in SECRET_KEYS]
    if not names:
        return
    lines = _read_safe_env_lines()
    for name in names:
        lines = [l for l in lines if not l.startswith(f'{name}=')]
        os.environ.pop(name, None)   # load_dotenv won't unset a removed line, so drop it here
    ENV_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    load_dotenv(ENV_PATH, override=True)

_COMFY_DERIVED = {'output': ('output_dir', 'output'), 'input': ('input_dir', 'input'),
                  'models': ('models_dir', 'models'), 'loras': ('loras_dir', 'models/loras')}

# Stable display order for the four override fields (Settings, docs, API payload).
COMFY_DIR_KINDS = ('output', 'input', 'models', 'loras')


def resolve_comfyui_dir(kind: str, base_dir: str, explicit: str = ''):
    """Pure resolution of one ComfyUI folder: an explicit override wins, else it is
    derived from the install directory. Kept separate from `comfyui_dir` (which reads
    live config) so the Settings screen can PREVIEW the very same computation on
    unsaved field values — what the user is shown is then, by construction, what the
    app will use. Reported from Discord (vykas22): a ComfyUI launched with
    --input-directory/--output-directory looked like it was ignored, because the four
    override keys existed but had no field anywhere in the app.

    Whitespace-only is treated as empty: a stray space used to resolve to Path(' ')
    and silently shadow the derived folder."""
    _, sub = _COMFY_DERIVED[kind]
    explicit = (explicit or '').strip()
    if explicit:
        return Path(explicit)
    base = (base_dir or '').strip()
    return Path(base) / Path(sub) if base else None


def comfyui_dir(kind: str):
    key, _ = _COMFY_DERIVED[kind]
    return resolve_comfyui_dir(kind, get('comfyui.base_dir') or '',
                               get(f'comfyui.{key}') or '')

def aitoolkit_derived_python(root):
    """The interpreter an ai-toolkit checkout carries, ignoring any explicit
    `aitoolkit.python`. Both venv layouts exist: ai-toolkit's docs say `venv`,
    plenty of setups use `.venv`. Pick whichever actually exists; when neither
    does, return the historical default path so callers keep a concrete path to
    name in their "invalid" details (never None)."""
    root = Path(root)
    for env_dir in ('venv', '.venv'):
        p = (root / env_dir / 'Scripts' / 'python.exe' if os.name == 'nt'
             else root / env_dir / 'bin' / 'python')
        if p.exists():
            return p
    win = root / 'venv' / 'Scripts' / 'python.exe'
    return win if os.name == 'nt' else root / 'venv' / 'bin' / 'python'


def aitoolkit_path(kind: str):
    root = get('aitoolkit.dir') or ''
    if not root:
        return None
    root = Path(root)
    if kind == 'dir':
        return root
    if kind == 'datasets':
        return Path(get('aitoolkit.datasets_dir') or root / 'datasets')
    if kind == 'output':
        return Path(get('aitoolkit.output_dir') or root / 'output')
    if kind == 'hf_home':
        return Path(get('aitoolkit.hf_home') or root / 'hf-cache' / 'huggingface')
    if kind == 'venv_python':
        # An explicit interpreter wins — installs WITHOUT a venv folder exist
        # in the wild (conda, uv, system python; user-reported from Reddit).
        explicit = (get('aitoolkit.python') or '').strip()
        if explicit:
            return Path(explicit)
        return aitoolkit_derived_python(root)
    if kind == 'venv_python_derived':
        # What the app WOULD run without the explicit override. Only useful when
        # an explicit one is set and turns out to be broken: it is the working
        # interpreter we can then offer to switch to (GitHub #19, strouder —
        # a `aitoolkit.python` pointing at a torch-less Python silently beat a
        # perfectly good venv sitting right next to run.py).
        return aitoolkit_derived_python(root)
    if kind == 'jobs':
        return root / 'config' / 'generated'
    raise KeyError(kind)

def dataset_images_root() -> Path:
    p = get('paths.dataset_images_root') or ''
    root = Path(p) if p else _data_dir() / 'datasets'
    root.mkdir(parents=True, exist_ok=True)
    return root

def backups_dir() -> Path:
    """Where 'Back up everything' writes its master archives (created on demand).
    Always under the app's data dir — never the (possibly relocated) datasets
    root — so a full backup never lands inside the very tree it is archiving."""
    d = _data_dir() / 'backups'
    d.mkdir(parents=True, exist_ok=True)
    return d

def bank_sources_root() -> Path:
    """Image folders CREATED by "Import to bank" — a copy of a dataset's kept
    images, so the new bank OWNS its files instead of pointing at the dataset's
    live folder (curating one would otherwise mutate the other). Deliberately
    NOT banks_root(): that one holds working data only and its contract is that
    it never contains source images."""
    root = _data_dir() / 'bank_sources'
    root.mkdir(parents=True, exist_ok=True)
    return root

def banks_root() -> Path:
    """Working data of the 🗃️ image banks (thumbnails + face-embedding cache),
    one subfolder per bank — never the source images, which stay in the user's
    folder untouched."""
    root = _data_dir() / 'banks'
    root.mkdir(parents=True, exist_ok=True)
    return root

def secret_key() -> str:
    d = _data_dir(); d.mkdir(parents=True, exist_ok=True)
    f = d / 'secret_key'
    if not f.exists():
        f.write_text(_secrets.token_hex(32), encoding='utf-8')
    return f.read_text(encoding='utf-8').strip()
