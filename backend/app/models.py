import json
from datetime import datetime, timedelta
from .extensions import db
from sqlalchemy import Integer, String, Text, DateTime, Float

class FaceDataset(db.Model):
    """A named face-dataset for LoRA character training (one per character)."""
    __tablename__ = 'face_dataset'
    id = db.Column(Integer, primary_key=True)
    user_id = db.Column(String(36), nullable=False, index=True, default='local')
    name = db.Column(String(100), nullable=False)
    trigger_word = db.Column(String(60), nullable=False)
    ref_filename = db.Column(String(255), nullable=True)
    # Original PLEIN CADRE de la référence (aspect conservé, capé ~2048), gardé pour
    # que le recadrage manuel puisse RÉÉLARGIR au lieu de seulement resserrer le crop
    # déjà fait. ref_filename = le carré dérivé (auto head-crop ou recadrage manuel).
    ref_original_filename = db.Column(String(255), nullable=True)
    # Références ADDITIONNELLES (JSON list de filenames, cap côté service) : envoyées
    # en plus à Nano Banana pour renforcer la cohérence d'identité. La principale
    # (ref_filename) reste la seule source de Klein, du crop et du scoring InsightFace.
    ref_extra_filenames = db.Column(Text, nullable=True)
    # Réglages gagnants du Studio de test LoRA (JSON: {lora_filename, strength,
    # z_model, seed, decided_at}). Écrit par l'humain via « ★ Définir comme
    # meilleur réglage », jamais automatiquement.
    best_settings = db.Column(Text, nullable=True)
    # Entraînement sur base CUSTOM : modèle de base ComfyUI choisi (z_model value ;
    # None = officiel Z-Image-Turbo) + variante (turbo|base|deturbo) qui règle
    # l'adapter de de-distillation. Isole aussi le run d'entraînement par base.
    train_base_model = db.Column(Text, nullable=True)
    train_variant = db.Column(String(20), nullable=True)
    # « Custom weights… » (V1, local-only) : quand train_base_model est un chemin
    # ABSOLU vers un .safetensors, c'est un poids custom de la MÊME architecture
    # (krea/flux/flux2klein/sdxl). Overrides SDXL UNIQUEMENT (ai-toolkit ne les
    # honore top-level que pour SDXL) : chemin VAE et chemin/te repo-id du TE.
    train_vae_path = db.Column(Text, nullable=True)
    train_te_path = db.Column(Text, nullable=True)
    # Réglages ai-toolkit avancés éditables par dataset (JSON) : rank, resolution,
    # save_every. NULL = défauts family-aware. Cf. lora_training._train_settings.
    train_settings = db.Column(Text, nullable=True)
    # Slider LoRA mode (Beta, ai-toolkit `concept_slider` trainer). JSON:
    # {enabled, positive, negative, target_class, anchor, guidance, anchor_strength}.
    # NULL = mode off. Dedicated column (NOT train_settings) so applying a training
    # preset — which REPLACES train_settings — can never silently wipe a slider
    # setup. Cf. lora_training._slider_settings. Additive migration in create_app.
    train_slider = db.Column(Text, nullable=True)
    # Famille de modèle entraînée : 'zimage' (défaut/None) ou 'sdxl'. Pilote la
    # branche de build_job_config (arch/scheduler/base) et le dossier loras d'import.
    train_type = db.Column(String(16), nullable=True)
    # Nature du dataset : NULL/'character' (défaut historique) ou 'concept'. Orthogonale
    # à train_type — un concept s'entraîne sur n'importe quelle base. Inverse la logique
    # import/caption (cf face_dataset_service : is_concept). Colonne ajoutée après coup
    # → migration additive idempotente dans create_app (db.create_all n'ALTER jamais).
    kind = db.Column(String(16), nullable=True)
    # Cible de fidélité (datasets personnage) : NULL/'face' (historique) ou 'body'.
    # 'body' = le LoRA doit reproduire AUSSI la morphologie/les marques corporelles →
    # captions bannissent en plus tatouages/cicatrices/grains de beauté (ils se lient
    # au trigger), composition cible plus de bustes/corps, import plein cadre par défaut.
    fidelity = db.Column(String(8), nullable=True)
    # Concept datasets only: what recurring act/concept must be OMITTED from every
    # caption so it binds to the trigger (the inverse of a character LoRA). Feeds the
    # {concept} placeholder of the caption/refine/ban-list prompts. concept_terms
    # caches the LLM-expanded synonym ban-list (JSON list) used to detect & scrub
    # leaks — both are additive columns (migration in create_app).
    concept_desc = db.Column(Text, nullable=True)
    concept_terms = db.Column(Text, nullable=True)
    # Creative-direction PROMPT SUFFIXES (community feature request): free user
    # text that rides on every generated variation. prompt_suffix = global text;
    # prompt_suffixes = JSON map {face,bust,body,back} per framing. Applied at
    # WRAP time only (face_variations.compose_prompt_suffix) — NEVER baked into
    # the stored variation_prompt, otherwise a regenerate would double-apply it.
    # Additive columns (migration in create_app).
    prompt_suffix = db.Column(Text, nullable=True)
    prompt_suffixes = db.Column(Text, nullable=True)
    # Per-dataset CAPTION method options (JSON: {backend, ollama_model, instructions}).
    # NULL/empty = follow the GLOBAL captioning defaults (captioning.backend +
    # ollama.vision_model). A dedicated column (NOT train_settings) so a training
    # preset — which REPLACES train_settings — can never wipe the caption setup, and
    # so captioning never has to depend on the ai-toolkit gate. Read via
    # face_dataset_service.caption_options; additive migration in create_app.
    caption_options = db.Column(Text, nullable=True)
    created_at = db.Column(DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(DateTime, default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())

    def __repr__(self):
        return f'<FaceDataset {self.id} {self.name}>'


class FaceDatasetImage(db.Model):
    """One image of a face-dataset: a generated Klein variation or an imported real photo."""
    __tablename__ = 'face_dataset_image'
    id = db.Column(Integer, primary_key=True)
    dataset_id = db.Column(
        Integer, db.ForeignKey('face_dataset.id', ondelete='CASCADE'),
        nullable=False, index=True)
    filename = db.Column(String(255), nullable=True)            # null until the job completes
    source = db.Column(String(12), nullable=False, default='generated')  # generated|import
    framing = db.Column(String(12), nullable=True)              # face|bust|body|back|unknown
    variation_label = db.Column(String(120), nullable=True)
    status = db.Column(String(10), nullable=False, default='pending')    # pending|keep|reject|failed
    caption = db.Column(Text, nullable=True)                    # WITHOUT the trigger word
    # Optional SHORT variant of the caption, derived from the long one (Ollama, text-only)
    # for ai-toolkit's dual long+short captioning. Same kind strategy as the long (identity/
    # concept/aesthetic omitted) and stored WITHOUT the trigger; the trigger is prepended at
    # export like the long. Additive column (migration in create_app). NULL = no short yet.
    caption_short = db.Column(Text, nullable=True)
    job_id = db.Column(String(36), nullable=True, index=True)
    variation_prompt = db.Column(String(500), nullable=True)    # RAW catalog prompt (regenerate)
    klein_model = db.Column(String(255), nullable=True)         # UNET used (regenerate)
    # Provenance for derived dataset images. Small scraped sources rescued through
    # Klein keep their own row/file and the generated candidate points back to it;
    # both stay outside training until the user resolves the pair explicitly.
    parent_image_id = db.Column(Integer, nullable=True)
    derivation_kind = db.Column(String(32), nullable=True)
    # Ressemblance faciale vs la reference (face analyzer Lot A). face_score = cosinus
    # ArcFace brut (NULL si non note) ; face_state = scorable|no_face|low_det|too_small|
    # extreme_pose|unreadable|error. Score brut persiste -> seuils recalibrables cote UI.
    face_score = db.Column(Float, nullable=True)
    face_state = db.Column(String(16), nullable=True)
    # Pourquoi status='failed' : message d'erreur du moteur (API/sauvegarde/queue),
    # affiché sur la tuile — sinon l'échec est muet et l'utilisateur relance à
    # l'aveugle. Nettoyé au regenerate. Colonne additive (migration create_app).
    fail_reason = db.Column(Text, nullable=True)
    # Facteur d'agrandissement appliqué par le crop (head-crop auto à l'import OU
    # recadrage manuel) pour atteindre le carré 1024 : size / côté_de_la_box. NULL =
    # jamais croppé (import plein cadre) ou pas encore recalculé (anciennes lignes).
    # >1 = le crop était plus petit que 1024 et a été agrandi (LANCZOS) — ce pixel-là
    # est donc de la texture inventée, pas du détail réel, et sur-pèse la loss de
    # cette image proportionnellement à sa part du cadre. Colonne additive (migration
    # create_app). Alimente composition_upscaled (dataset_payload) pour repérer un
    # dataset trop chargé en gros plans fabriqués plutôt que natifs.
    upscale_ratio = db.Column(Float, nullable=True)
    # Watermark auto-correction (V1) : détection + suppression des watermarks INCRUSTÉS
    # (logo de site, URL, pseudo, texte de studio ajouté PAR-DESSUS la photo scrapée) —
    # sinon le LoRA les apprend. watermark_state : NULL (jamais scanné) | 'none' (propre)
    # | 'detected' (trouvé, pas encore traité / à revoir manuellement) | 'dismissed'
    # (l'utilisateur a jugé en review que c'est un FAUX positif → plus de 🚩, et les
    # prochains scans le sautent) | 'cleaned' (crop ou inpaint LaMa appliqué) | 'failed'.
    # watermark_bbox : JSON [x1,y1,x2,y2] normalisé
    # [0,1] du watermark (NULL si aucun). Les bbox VLM sont GROSSIÈRES → déjà élargies
    # d'une marge avant stockage. Colonnes additives (migration create_app).
    watermark_state = db.Column(String(16), nullable=True)
    watermark_bbox = db.Column(Text, nullable=True)
    # Correction manuelle : JSON list de bbox normalisées. NULL conserve le bbox
    # automatique comme source effective ; [] est un override explicite vide.
    watermark_regions = db.Column(Text, nullable=True)
    # Métadonnées de provenance génériques, sérialisées en JSON. La première
    # intégration prise en charge est Pexels : plateforme, page photo et crédit
    # photographe. Toute écriture passe par la validation stricte du service.
    source_metadata = db.Column(Text, nullable=True)
    created_at = db.Column(DateTime, default=db.func.current_timestamp())

    def __repr__(self):
        return f'<FaceDatasetImage {self.id} ds={self.dataset_id} {self.status}>'


class ImageBank(db.Model):
    """A triage bank: a big unsorted folder of images (a Telegram export, a
    scrape dump…) referenced IN PLACE — nothing is copied and the source files
    are NEVER written to. The bank layers quality scores, duplicate groups,
    face clusters and keep/reject decisions on top; promoting a selection
    COPIES those files into a dataset through the normal import path. New
    table — created by db.create_all(), no migration needed."""
    __tablename__ = 'image_bank'
    id = db.Column(Integer, primary_key=True)
    user_id = db.Column(String(36), nullable=False, index=True, default='local')
    name = db.Column(String(100), nullable=False)
    source_path = db.Column(Text, nullable=False)     # absolute folder, read-only
    created_at = db.Column(DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(DateTime, default=db.func.current_timestamp(),
                           onupdate=db.func.current_timestamp())
    # Persisted summary of the last "Launch all" pipeline run (JSON): per-step
    # done/skipped(reason)/error plus the headline counts, so the outcome is
    # still there when the user reopens the bank the next morning. NULL = no
    # pipeline has ever run. Additive column — the image_bank table shipped in
    # the Beta, so it needs the additive path (see _SCHEMA_ADDITIONS).
    pipeline_report = db.Column(Text, nullable=True)

    def __repr__(self):
        return f'<ImageBank {self.id} {self.name}>'


class BankImage(db.Model):
    """One file of an image bank. RAW quality scores are persisted (never the
    verdicts): blur/noise/uniformity flags are recomputed against the current
    thresholds at read time, so recalibrating a threshold re-sorts the bank
    without a rescan — same philosophy as face_score. dhash is stored (unlike
    dataset imports, which recompute) because a bank holds thousands of files:
    duplicate grouping needs an O(1) read, not an O(n) disk walk."""
    __tablename__ = 'bank_image'
    id = db.Column(Integer, primary_key=True)
    bank_id = db.Column(
        Integer, db.ForeignKey('image_bank.id', ondelete='CASCADE'),
        nullable=False, index=True)
    relpath = db.Column(Text, nullable=False)          # relative to bank.source_path
    file_size = db.Column(Integer, nullable=True)
    width = db.Column(Integer, nullable=True)
    height = db.Column(Integer, nullable=True)
    # Quality pass (pure-PIL, CPU). quality_state: NULL = not scanned yet |
    # 'ok' | 'unreadable'. Scores are raw metric values (see image_quality.py).
    quality_state = db.Column(String(12), nullable=True)
    blur_score = db.Column(Float, nullable=True)         # Laplacian variance (sharpness)
    noise_score = db.Column(Float, nullable=True)        # residual std vs Gaussian blur
    uniformity_score = db.Column(Float, nullable=True)   # grayscale std (low = flat)
    dhash = db.Column(String(16), nullable=True)         # 64-bit hex, same dHash as imports
    # Duplicate group id (bank-local, rebuilt at the end of every quality scan).
    # NULL = no near-duplicate found. This is the EXACT/resized dedup (stage 1):
    # same 64-bit dHash family (Hamming <= dup_distance).
    dup_group = db.Column(Integer, nullable=True, index=True)
    # Semantic near-duplicate group id (stage 2 — "same shot, different crop"):
    # cosine of the CLIP embeddings the ✨ Score pass cached >= semantic_dup_threshold.
    # Catches crops / re-compressed variants a dHash misses. Assigned by the
    # semantic-dedup pass over the scored images; NULL = no semantic near-dup /
    # the pass hasn't run. Distinct column from dup_group so the two stages
    # co-exist (an image can belong to both).
    semantic_dup_group = db.Column(Integer, nullable=True, index=True)
    # Subject pass (InsightFace subprocess). face_state mirrors the dataset
    # vocabulary (scorable|no_face|low_det|too_small|extreme_pose|unreadable|error);
    # face_cluster is a bank-local person-cluster id (1 = biggest cluster),
    # NULL = unclustered (no usable face or pass not run yet).
    face_state = db.Column(String(16), nullable=True)
    face_det = db.Column(Float, nullable=True)
    face_cluster = db.Column(Integer, nullable=True, index=True)
    # Scoring pass (V2, the "bank scoring" ML extra: CLIP ViT-L/14 + a tiny
    # aesthetic head + an NSFW classifier, one subprocess like the face pass).
    # RAW scores persist, VERDICTS are recomputed at read time against the 'bank'
    # thresholds — same philosophy as the quality scores, so retuning a threshold
    # re-sorts the bank with no rescan.
    #   aesthetic_score : LAION improved-aesthetic prediction, ~1..10 (higher = nicer).
    #   nsfw_score      : 0..1 probability the image is NSFW (is_nsfw = > threshold).
    #   style_cluster   : bank-local visual-STYLE cluster id (1 = biggest), from the
    #                     CLIP image embeddings (group screenshots/memes vs photoreal),
    #                     the "group by style" counterpart to face_cluster. NULL = the
    #                     scoring pass hasn't run / no usable embedding.
    aesthetic_score = db.Column(Float, nullable=True)
    nsfw_score = db.Column(Float, nullable=True)
    style_cluster = db.Column(Integer, nullable=True, index=True)
    # Watermark pass (V2): reuses the dataset Qwen3-VL overlaid-watermark detector.
    # NULL = not scanned | 'none' (clean) | 'detected' (an overlaid watermark/logo/
    # URL was found → the read-time 'watermark' flag) | 'error'. Detection only; the
    # bank never edits the source file (cleaning stays a dataset-side action).
    watermark_state = db.Column(String(16), nullable=True)
    # Caption pass — a plain DESCRIPTIVE caption (no trigger, no identity omission:
    # a bank has no trigger word and nothing to protect). It doubles as the bank's
    # search text (the search bar matches caption + relpath) AND rides along to the
    # dataset on promotion, so a promoted selection starts already captioned. NULL =
    # not captioned yet. Additive column — created by db.create_all(), no migration.
    caption = db.Column(Text, nullable=True)
    # Framing pass — the SAME face/bust/body/back classification the datasets use
    # (Qwen3-VL, CLASSIFY_PROMPT). face = head close-up | bust = upper body |
    # body = full body | back = seen from behind | 'unknown' = a parseable answer
    # that wasn't one of the four | NULL = not classified yet (retryable). Powers
    # the 📐 Framing filter chips AND the coverage advice. Additive column —
    # created by db.create_all(), no migration (see _SCHEMA_ADDITIONS).
    framing = db.Column(String(8), nullable=True, index=True)
    # Triage decision — same words as dataset images (pending|keep|reject).
    # reject_reason: blur|noise|uniform|small|duplicate|unreadable|manual
    #                |low_aesthetic|nsfw|watermark (the V2 score-derived flags).
    status = db.Column(String(10), nullable=False, default='pending', index=True)
    reject_reason = db.Column(String(16), nullable=True)
    # Set once the image has been promoted (copied) into a dataset — the funnel's
    # provenance, and the guard against promoting the same file twice by accident.
    promoted_dataset_id = db.Column(Integer, nullable=True)
    created_at = db.Column(DateTime, default=db.func.current_timestamp())

    def __repr__(self):
        return f'<BankImage {self.id} bank={self.bank_id} {self.status}>'


class LoraTestImage(db.Model):
    """One cell image of a LoRA Test-Studio run (checkpoint x strength grid).

    Generated via the Z-Image (ZTurbo) pipeline with a fixed seed; the file is
    moved into the per-dataset folder on completion (like the dataset fan-out)
    and hidden from the gallery history."""
    __tablename__ = 'lora_test_image'
    id = db.Column(Integer, primary_key=True)
    dataset_id = db.Column(
        Integer, db.ForeignKey('face_dataset.id', ondelete='CASCADE'),
        nullable=False, index=True)
    checkpoint = db.Column(String(255), nullable=False)   # LoRA filename ('z image\\Lola-1000.safetensors')
    strength = db.Column(Float, nullable=False)
    filename = db.Column(String(255), nullable=True)      # null until the job completes
    job_id = db.Column(String(36), nullable=True, index=True)
    rating = db.Column(Integer, nullable=False, default=0)  # 1 (like) | -1 (dislike) | 0 (unrated)
    seed = db.Column(db.BigInteger, nullable=True)
    # Seed de BASE du lancement : toutes les cellules d'un même « Lancer le test »
    # partagent ce run_seed (regroupe les N seeds d'un batch). null = anciens runs
    # (un seul seed/lancement) → on retombe sur `seed` côté UI.
    run_seed = db.Column(db.BigInteger, nullable=True)
    # Groupe toutes les cellules d'UN lancement ; une comparaison multi-LoRA a des
    # cellules de dataset_id différents partageant ce run_id. null = anciens runs
    # (backfillés par add_lora_test_run_id).
    run_id = db.Column(String(36), nullable=True, index=True)
    status = db.Column(String(10), nullable=False, default='pending')  # pending|done|failed|cancelled
    # Pourquoi status='failed' : raison réelle remontée du chemin de génération
    # ComfyUI (validation 400 « modèle/node introuvable », node error, timeout,
    # enqueue raté…) — affichée au survol de la tuile en échec. Sinon l'échec est
    # muet et l'utilisateur relance à l'aveugle (P0-b). Colonne additive (migration
    # create_app). Les cellules en échec sont exclues du classement (cf. cell_scores).
    error = db.Column(Text, nullable=True)
    # Réglages du run (pour afficher TOUS les paramètres du meilleur résultat).
    z_model = db.Column(String(255), nullable=True)   # modèle Z-Image de base
    aspect = db.Column(String(16), nullable=True)     # format d'image (9:16, 4:3, …)
    prompt = db.Column(db.Text, nullable=True)        # prompt de test utilisé
    cfg = db.Column(Float, nullable=True)             # CFG testé (axe optionnel)
    steps = db.Column(Integer, nullable=True)         # steps pass 1 (KSampler) ; axe optionnel
    steps2 = db.Column(Integer, nullable=True)        # SDXL : steps pass 2 (detail daemon, node 57) ; NULL = pass 1
    extra_loras = db.Column(Text, nullable=True)      # LoRA always-on (style/utilitaire) JSON [{filename,strength}] ; appliqués à CHAQUE cellule (hors batch)
    krea_rebalance = db.Column(Float, nullable=True)  # Krea node 30 (NSFW/texture rebalance) : NULL=défaut, ≤1=OFF, >1=ON@force
    # Parité Generate (2026-07-01) — réglages persistés par cellule pour un resume fidèle.
    negative = db.Column(Text, nullable=True)             # Z-Image : prompt négatif (node 5)
    sampler = db.Column(String(32), nullable=True)        # Krea : node 26 sampler_name
    scheduler = db.Column(String(32), nullable=True)      # Krea : node 26 scheduler
    weight_dtype = db.Column(String(24), nullable=True)   # Krea : node 20 précision UNET (weight_dtype)
    enhancer_strength = db.Column(Float, nullable=True)   # Krea2T-Enhancer : NULL=OFF, sinon force ON
    detail_amount = db.Column(Float, nullable=True)       # SDXL : DetailDaemon detail_amount (NULL=défaut)
    resolution_tier = db.Column(String(12), nullable=True)  # fast|standard|hq|max (compute_tier_dims) ; NULL=table fixe
    init_image = db.Column(String(255), nullable=True)    # Krea img2img : fichier init copié dans COMFYUI_INPUT_DIR
    denoise = db.Column(Float, nullable=True)             # Krea img2img : node 26 denoise
    # Scoring facial objectif (« best epoch », méthode jandordoe) : similarité
    # cosinus InsightFace vs la référence du dataset + état de scorabilité
    # ('scorable'/'no_face'/'low_det'/…). NULL = cellule pas encore scorée.
    face_score = db.Column(Float, nullable=True)
    face_state = db.Column(String(16), nullable=True)
    created_at = db.Column(DateTime, default=db.func.current_timestamp())

    def __repr__(self):
        return f'<LoraTestImage {self.id} ds={self.dataset_id} {self.checkpoint}@{self.strength} {self.status}>'


class JobQueueMixin:
    """Shared lifecycle helpers for job queue tables (image + video).

    Both ImageGenerationQueue and VideoGenerationQueue use the same
    status columns (status, started_at, completed_at, last_heartbeat,
    error_message, result_filename, comfyui_prompt_id). This mixin
    factorizes the update/stuck logic so both tables stay in sync.
    """

    def update_status(self, new_status, error_message=None,
                      result_filename=None, comfyui_prompt_id=None):
        """Update status with automatic timestamp + heartbeat management."""
        self.status = new_status

        if error_message is not None:
            self.error_message = error_message

        if result_filename is not None:
            self.result_filename = result_filename

        if comfyui_prompt_id is not None:
            self.comfyui_prompt_id = comfyui_prompt_id

        if new_status == 'processing':
            self.started_at = datetime.utcnow()
        elif new_status in ('completed', 'failed', 'cancelled'):
            self.completed_at = datetime.utcnow()

        self.last_heartbeat = datetime.utcnow()

    def is_stuck(self, timeout_minutes=10):
        """True if the job is in-progress but heartbeat is missing/stale."""
        if self.status not in ('processing', 'sent_to_comfy'):
            return False
        if not self.last_heartbeat:
            return True
        return datetime.utcnow() - self.last_heartbeat > timedelta(minutes=timeout_minutes)


class ImageGenerationQueue(JobQueueMixin, db.Model):
    """Modèle pour la file d'attente de génération d'images"""
    __tablename__ = 'image_generation_queue'

    id = db.Column(Integer, primary_key=True)
    job_id = db.Column(String(36), unique=True, nullable=False)
    user_id = db.Column(String(36), nullable=False, default='local')
    status = db.Column(String(20), nullable=False, default='pending')
    workflow_data = db.Column(Text, nullable=True)
    prompt = db.Column(Text, nullable=True)
    result_filename = db.Column(String(255), nullable=True)
    error_message = db.Column(Text, nullable=True)
    retry_count = db.Column(Integer, default=0, nullable=False)
    priority = db.Column(Integer, default=0, nullable=False)
    created_at = db.Column(DateTime, default=db.func.current_timestamp(), nullable=False)
    started_at = db.Column(DateTime, nullable=True)
    completed_at = db.Column(DateTime, nullable=True)
    last_heartbeat = db.Column(DateTime, nullable=True)
    comfyui_prompt_id = db.Column(String(100), nullable=True)
    worker_id = db.Column(String(36), nullable=True)  # Worker GPU qui traite ce job
    job_metadata = db.Column(Text, nullable=True)

    __table_args__ = (
        db.Index('idx_img_status', 'status'),
        db.Index('idx_img_user_id', 'user_id'),
        db.Index('idx_img_created_at', 'created_at'),
        db.Index('idx_img_priority_created', 'priority', 'created_at'),
    )

    def to_dict(self):
        """Convertit le job en dictionnaire pour l'API"""
        metadata = {}
        if self.job_metadata:
            try:
                metadata = json.loads(self.job_metadata)
            except json.JSONDecodeError:
                metadata = {}

        workflow_data = {}
        if self.workflow_data:
            try:
                workflow_data = json.loads(self.workflow_data)
            except json.JSONDecodeError:
                workflow_data = {}

        return {
            'job_id': self.job_id,
            'user_id': self.user_id,
            'status': self.status,
            'prompt': self.prompt,
            'result_filename': self.result_filename,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'worker_id': self.worker_id,
            'metadata': metadata,
            'workflow_data': workflow_data,
            'type': 'image'
        }

    def to_status_dict(self):
        """Version allégée de to_dict() pour le polling /status (sans workflow_data)"""
        metadata = {}
        if self.job_metadata:
            try:
                metadata = json.loads(self.job_metadata)
            except json.JSONDecodeError:
                metadata = {}

        return {
            'job_id': self.job_id,
            'user_id': self.user_id,
            'status': self.status,
            'prompt': self.prompt,
            'result_filename': self.result_filename,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'worker_id': self.worker_id,
            'metadata': metadata,
            'type': 'image'
        }


class SystemState(db.Model):
    __tablename__ = 'system_state'
    key = db.Column(db.String(64), primary_key=True)
    value = db.Column(db.Text)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class CloudTrainingRun(db.Model):
    """One cloud training run = one ephemeral vast.ai pod. Durable on purpose:
    boot-time reconciliation matches live vast instances (label 'lds-<id>')
    against these rows to kill orphaned pods — the expensive failure mode."""
    __tablename__ = 'cloud_training_run'
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, nullable=False)
    run_name = db.Column(db.String(255))          # local run identity (lt._run_name)
    job_name = db.Column(db.String(255))          # unique remote job/dataset name
    status = db.Column(db.String(32), default='preparing')
    phase_detail = db.Column(db.Text, default='')
    vast_instance_id = db.Column(db.String(32))
    vast_label = db.Column(db.String(64))
    gpu_name = db.Column(db.String(64))
    price_per_hour = db.Column(db.Float)
    remote_job_id = db.Column(db.String(64))
    base_url = db.Column(db.String(255))
    auth_token = db.Column(db.String(128))
    staging_dir = db.Column(db.Text)
    checkpoint_local_path = db.Column(db.Text)
    train_params = db.Column(db.Text)             # JSON: steps/variant/train_type/masked
    error = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    finished_at = db.Column(db.DateTime)


class TrainingRunRecord(db.Model):
    """Provenance registry: one row per training LAUNCH (local or cloud).
    Answers "which VERSION of the dataset produced this checkpoint, with what
    settings?" — nothing recorded local runs before this (files only), and no
    run recorded the dataset's state. `version` is a human counter per
    (dataset, family): a launch whose dataset fingerprint was never seen
    becomes v(max+1); re-running an unchanged dataset keeps its version.
    `manifest` (JSON [[image_id, caption_hash], ...]) lets the UI say WHAT
    changed since ("+2 images, 3 captions edited"), not just that it did.
    New table — created by db.create_all(), no migration of existing rows."""
    __tablename__ = 'training_run_record'
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, nullable=False, index=True)
    family = db.Column(db.String(16), nullable=False)
    source = db.Column(db.String(8), nullable=False)        # 'local' | 'cloud'
    cloud_run_id = db.Column(db.Integer)                    # FK-ish, cloud only
    base_model = db.Column(db.String(255), default='')      # '' = official base
    variant = db.Column(db.String(32))
    masked = db.Column(db.Boolean, default=True)
    steps = db.Column(db.Integer)
    fingerprint = db.Column(db.String(16), nullable=False)
    manifest = db.Column(db.Text)                           # JSON, see docstring
    # JSON launch_settings_snapshot: the EFFECTIVE ai-toolkit settings this
    # launch used (rank/alpha/resolution/optimizer/...) — shown per run on the
    # unified Runs page. NULL on pre-feature rows.
    settings = db.Column(db.Text)
    version = db.Column(db.Integer, nullable=False)
    # Lineage (genealogy tree). A CONTINUATION stamps the record it resumed from
    # (parent_record_id) and the step it resumed AT (resumed_from) — the durable,
    # unambiguous edge the Runs-hub tree draws, instead of parsing `_superseded`
    # folder names. Both NULL on a fresh launch and on every pre-feature row: a
    # record with no parent is a lineage ROOT. No auto-invention for legacy runs
    # (the resume link was never persisted before) — added idempotently at boot.
    parent_record_id = db.Column(db.Integer, index=True)
    resumed_from = db.Column(db.Integer)
    # How the parent edge above came to exist. NULL = persisted natively by the
    # continuation that drew it; 'backfill' = reconstructed at boot for a
    # continuation that ran before the edge was persisted (see lineage_backfill).
    # Kept distinct so a reconstructed edge stays auditable and reversible.
    lineage_origin = db.Column(db.String(16))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class TrainingPreset(db.Model):
    """Named, shareable snapshot of the ⚙️ advanced training settings.

    `settings` stores the RAW explicit keys (the same blob shape as
    face_dataset.train_settings) — validation happens at APPLY time through
    the per-key update_train_settings path, so a preset exported by a newer
    (or older) app version applies gracefully: unknown keys are ignored and
    invalid values reported, never fatal. Import/export is a JSON file built
    around this row. ``dataset_kind``/``variants`` scope new presets without
    invalidating legacy NULL rows; their columns are added idempotently at boot."""
    __tablename__ = 'training_preset'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False, unique=True)
    train_type = db.Column(db.String(16), nullable=False, default='zimage')
    # Scope metadata is nullable for presets created before family/kind guards.
    # ``variants`` is a JSON list (e.g. ["base"] or ["4b", "9b"]).
    dataset_kind = db.Column(db.String(16), nullable=True)
    variants = db.Column(db.Text, nullable=True)
    settings = db.Column(db.Text, nullable=False, default='{}')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
