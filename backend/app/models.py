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
    # Per-FAMILY memory of (base, variant). `train_base_model`/`train_variant`
    # above stay the ACTIVE selection — every reader keeps reading them — and
    # this JSON only remembers what each family was last set to:
    #   {"zimage": {"base": "z image\\merge.safetensors", "variant": "turbo"},
    #    "krea":   {"base": "", "variant": "base"}}
    # Switching train_type stashes the outgoing family's pair here and restores
    # the incoming one (cf. face_dataset_service.set_train_type). Without it a
    # base chosen for Z-Image stayed attached to a Krea run: the selector read
    # the family, the summary read the column, and both told the truth about
    # different things. Additive + nullable migration in create_app; NULL simply
    # means "nothing remembered yet", which is what every existing dataset is.
    train_family_bases = db.Column(Text, nullable=True)
    # Same idea, for the handful of `train_settings` keys whose MEANING is bound
    # to the family (lora_training._FAMILY_SCOPED_SETTING_KEYS — today
    # `timestep_type`). `train_settings` above stays the ACTIVE blob every reader
    # reads; this JSON only remembers what each family was last set to:
    #   {"zimage": {"timestep_type": "sigmoid"}, "flux2klein": {}}
    # An empty dict for a family means "that family rides its own default", which
    # is NOT the same as "never configured" (absent). Without it, a `sigmoid`
    # picked under Z-Image overwrote the canonical `weighted` of FLUX.2 Klein and
    # Anima on a family switch — silently, and it changes the LoRA that comes out.
    # Additive + nullable migration in create_app; NULL = nothing remembered yet.
    train_family_settings = db.Column(Text, nullable=True)
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
    # Nature des poids optimisés. `lora` est le comportement historique et reste
    # le défaut explicite des nouvelles lignes comme des bases migrées. Le MVP
    # dense (`full_transformer`) entraîne uniquement le transformer Krea 2 Raw et
    # n'est exécutable que dans la lane cloud (voir services.lora_training).
    training_mode = db.Column(String(32), nullable=False, default='lora',
                              server_default='lora')
    # Nature du dataset : NULL/'character' (défaut historique) ou 'concept'. Orthogonale
    # à train_type — un concept s'entraîne sur n'importe quelle base. Inverse la logique
    # import/caption (cf face_dataset_service : is_concept). Colonne ajoutée après coup
    # → migration additive idempotente dans create_app (db.create_all n'ALTER jamais).
    kind = db.Column(String(16), nullable=True)
    # WHAT the subject is: NULL/'human' (historique) or animal/creature/object/other.
    # ORTHOGONALE à `kind` (character/concept/style) — un chien précis = character+animal,
    # « les chiens en général » = concept+animal. Steers the generation catalog + the
    # identity lock so the prompts stop assuming a person; NULL behaves exactly as
    # 'human' (byte-identical). Colonne additive (migration in create_app).
    subject_type = db.Column(String(16), nullable=True)
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
    # Which Klein UNET this dataset's Klein work runs on — the BARE file name as the
    # picker lists it (the subfolder prefix is the resolver's job). NULL = auto: let
    # klein_edit_helper.resolve_klein_unet choose, which is byte-for-byte what every
    # dataset did before this column existed. Read by ✨ Upscale & improve (all three
    # lanes) and used as the default of the generation picker. A dedicated column
    # (NOT train_settings) so applying a training preset — which REPLACES that blob —
    # cannot wipe it, and because it is not a training setting at all. Additive
    # migration in create_app.
    klein_model = db.Column(String(255), nullable=True)
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
    # WHO wrote each of the two captions above. NULL = never recorded (what every
    # row that predates these columns carries, and what a caption-free row keeps) |
    # 'asserted' = a human typed, corrected or imported the text | 'joycaption' /
    # 'ollama' = the engine that produced it. Same two vocabularies, same reasons,
    # as face_cluster_origin and watermark_source on BankImage — and the same
    # consequence for the first: a FORCED caption pass SKIPS 'asserted' rows so it
    # never silently overwrites the user's own words. NULL is re-captionable on
    # purpose: sparing it would make Re-caption inert on every existing dataset,
    # and the screen counts it separately instead of calling it machine-written.
    # The short caption has its OWN column because it has its own writers on both
    # sides (the expanded editor types it; the dual-caption pass derives it), and
    # the two cross constantly — one column for two texts would mislabel one of
    # them. See services/caption_origin.py. Additive columns (see _SCHEMA_ADDITIONS).
    caption_origin = db.Column(String(16), nullable=True)
    caption_short_origin = db.Column(String(16), nullable=True)
    job_id = db.Column(String(36), nullable=True, index=True)
    variation_prompt = db.Column(String(500), nullable=True)    # RAW catalog prompt (regenerate)
    klein_model = db.Column(String(255), nullable=True)         # UNET used (regenerate)
    # Provenance for derived dataset images. Small scraped sources rescued through
    # Klein keep their own row/file and the generated candidate points back to it;
    # both stay outside training until the user resolves the pair explicitly.
    parent_image_id = db.Column(Integer, nullable=True)
    derivation_kind = db.Column(String(32), nullable=True)
    # The bank_image this row was promoted FROM (NULL for everything else). It is
    # what makes "already promoted into this dataset" a fact we can re-check
    # instead of a one-way flag: when the user deletes the image here, the row —
    # and with it the link — disappears, so the bank offers the image again.
    # Deliberately no ForeignKey, like parent_image_id: legacy databases cannot
    # gain one, and a dangling id must read as "not promoted", never as a boot
    # error. Additive column (migration in create_app).
    bank_image_id = db.Column(Integer, nullable=True, index=True)
    # Ressemblance faciale vs la reference (face analyzer Lot A). face_score = cosinus
    # ArcFace brut (NULL si non note) ; face_state = scorable|no_face|low_det|too_small|
    # extreme_pose|unreadable|error. Score brut persiste -> seuils recalibrables cote UI.
    face_score = db.Column(Float, nullable=True)
    face_state = db.Column(String(16), nullable=True)
    # Pourquoi status='failed' : message d'erreur du moteur (API/sauvegarde/queue),
    # affiché sur la tuile — sinon l'échec est muet et l'utilisateur relance à
    # l'aveugle. Nettoyé au regenerate. Colonne additive (migration create_app).
    fail_reason = db.Column(Text, nullable=True)
    # Ce que fail_reason raconte, en UN mot machine — pour COMPTER les échecs par
    # nature sans relire la phrase (une phrase se réécrit, une clé non) :
    #   'refused' = le fournisseur a répondu normalement et a refusé l'image
    #               (filtre de sortie) — ce n'est pas une panne ;
    #   'empty'   = 200 sans image, sans raison exploitable (moteurs qui ne
    #               savent pas distinguer refus et hoquet) ;
    #   'error'   = vraie panne (réseau, clé, quota, 5xx, sauvegarde, queue).
    # NULL = ligne d'avant la colonne, ou échec non classé : l'UI ne compte alors
    # que ce qu'elle sait, elle ne devine pas. Colonne additive (migration
    # create_app). Valeurs = clés stockées en base : ne jamais les renommer.
    fail_kind = db.Column(String(16), nullable=True)
    # De combien la box recadrée (head-crop auto à l'import OU recadrage manuel) est
    # en-dessous de la résolution d'entraînement : size / côté_de_la_box. NULL =
    # jamais croppé (import plein cadre) ou pas encore recalculé (anciennes lignes).
    # >1 = la box était plus petite que 1024. Deux producteurs, pixels différents,
    # MÊME sens et même remède :
    #   - import head-crop : la box est agrandie (LANCZOS) → texture inventée ;
    #   - recadrage manuel : depuis la fin de l'agrandissement, la tuile garde ses
    #     pixels et reste donc simplement sous la résolution d'entraînement.
    # Dans les deux cas le cadrage est « rempli en recadrant », pas par une vraie
    # prise native. Valeur et échelle INCHANGÉES (ne pas plafonner : le seuil
    # UPSCALE_WARN_THRESHOLD et les lignes existantes en dépendent). Colonne additive
    # (migration create_app). Alimente composition_upscaled (dataset_payload).
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
    # WHICH detector ruled on this row: 'detector' (the SigLIP2 cascade extra) |
    # 'vision' (the Ollama vision model) | NULL = unknown, never guessed. Same
    # vocabulary and same reason as bank_image.watermark_source, which came
    # first: a dataset can already hold verdicts from BOTH routes today —
    # promotion copies a bank's watermark_state across — and the two disagree at
    # the margins, so "why is this one flagged?" has to stay answerable per image
    # rather than per dataset. watermark_score is the cascade's 0..1 score
    # (NULL on the vision route, which has none). Additive columns (create_app
    # migration); rows written before them stay NULL and read as 'unknown'.
    watermark_source = db.Column(String(16), nullable=True)
    watermark_score = db.Column(Float, nullable=True)
    # Métadonnées de provenance génériques, sérialisées en JSON. La première
    # intégration prise en charge est Pexels : plateforme, page photo et crédit
    # photographe. Toute écriture passe par la validation stricte du service.
    source_metadata = db.Column(Text, nullable=True)
    # Cached CONTENT hash of the file (sha1 of its bytes) and the `size:mtime` it
    # was computed for. The run snapshot needs to know whether the PIXELS changed
    # between two trainings — a re-crop, a "Reset to auto", a rembg mask or a
    # scrubbed watermark all keep the same id and filename. Hashing every image
    # at every launch would put file I/O on the launch path, so it is computed
    # once and reused while the stat still matches. Purely derived: dropping both
    # values only costs one re-hash. Additive columns (migration in create_app).
    content_sig = db.Column(String(24), nullable=True)
    content_sig_stat = db.Column(String(40), nullable=True)
    # Versioned Bank-only analysis captured when a Bank image is promoted here.
    # Its fingerprint is of THIS Dataset file's bytes, so importing this image back
    # into a Bank can restore scores only while the image has not been edited.
    # User-facing metadata (caption, framing, watermark, provenance and status)
    # deliberately stays in its normal columns and is never rolled back from this
    # historical snapshot. Additive column (migration in create_app).
    bank_analysis_snapshot = db.Column(Text, nullable=True)
    # Bidirectional, bounded metadata envelope for fields that only one side of
    # Bank <-> Dataset exposes (short caption, generation provenance, Bank reject
    # reason, etc.). It is inert history unless the transfer validator explicitly
    # restores a compatible destination field.
    transfer_metadata = db.Column(Text, nullable=True)
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
    # Which image/text embedding space powers Bank search, similarity, diversity,
    # coverage and semantic duplicate grouping. CLIP remains the durable default
    # for every existing Bank; SigLIP2 uses its own cache and never replaces the
    # CLIP vectors needed by aesthetic, NSFW, style and medium analysis.
    semantic_engine = db.Column(String(16), nullable=False, default='clip',
                                server_default='clip')

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
    # cosine in the Bank's selected CLIP or SigLIP2 space, with an engine-specific
    # threshold. Catches crops / re-compressed variants a dHash misses. Assigned
    # by the semantic-dedup pass; NULL = no semantic near-dup / the pass has not
    # run for the active engine. Distinct from dup_group so both stages co-exist.
    semantic_dup_group = db.Column(Integer, nullable=True, index=True)
    # Durable per-engine semantic partitions. ``semantic_dup_group`` above is the
    # active projection consumed by every existing filter/payload; switching the
    # Bank swaps that projection without throwing either engine's work away.
    clip_semantic_dup_group = db.Column(Integer, nullable=True)
    siglip2_semantic_dup_group = db.Column(Integer, nullable=True)
    # Subject pass (InsightFace subprocess). face_state mirrors the dataset
    # vocabulary (scorable|no_face|low_det|too_small|extreme_pose|unreadable|error);
    # face_cluster is a bank-local person-cluster id (1 = biggest cluster),
    # NULL = unclustered (no usable face or pass not run yet).
    face_state = db.Column(String(16), nullable=True)
    face_det = db.Column(Float, nullable=True)
    face_cluster = db.Column(Integer, nullable=True, index=True)
    # WHERE the face_cluster id came from. NULL = the embeddings pass computed it
    # (the default, and what every row that predates this column carries).
    # 'asserted' = the user declared the image's SUBFOLDER to hold a single person
    # (services/folder_person.py), so the id was written with no inference at all.
    # The two live in the SAME id space on purpose: every reader (filters, coverage,
    # the person chips) keeps working unchanged, and a later cross-folder merge can
    # still join an asserted group to a computed one. The column exists so the
    # embeddings pass can tell them apart — it SKIPS asserted rows (that skip is the
    # compute the assertion saves) and never silently overwrites the user's word.
    face_cluster_origin = db.Column(String(10), nullable=True)
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
    # URL was found → the read-time 'watermark' flag) | 'error' | 'cleaned' (one of
    # the two cleaning levels produced a clean version) | 'dismissed' (the user ruled
    # the flag a false positive — skipped by both cleaning levels and by re-scans).
    # The SOURCE FILE is never edited: a cleaned image is a separate blob in the
    # bank's own working directory (see image_bank_service.clean_image_path).
    watermark_state = db.Column(String(16), nullable=True)
    # The detected mark's normalized bbox, JSON [x1,y1,x2,y2] in 0..1 — the SAME
    # shape the dataset stores. Persisted (the detector parses it anyway) because
    # the two cleaning levels route on it: without a bbox there is nothing to crop
    # or to repaint. NULL on a row scanned by a build that only kept the boolean —
    # such rows are re-picked by the next scan (see start_watermark).
    watermark_bbox = db.Column(Text, nullable=True)
    # Hand-drawn correction: JSON list of normalized bboxes, the SAME shape and the
    # same meaning as FaceDatasetImage.watermark_regions (one validator, one editor).
    # NULL keeps the detector's bbox as the effective mask; [] is an EXPLICIT empty
    # override ("there is nothing to repaint here") and never falls back to the box.
    # Both cleaning levels read this first — an edit the cleaner ignores would be
    # worse than no editor at all. Additive column (migration in create_app).
    watermark_regions = db.Column(Text, nullable=True)
    # How the cleaned blob was produced: NULL = none (no cleaned version on disk) |
    # 'crop' (level 1, PIL, invents no pixel) | 'lama' | 'klein' (level 2, inpaint).
    # Non-NULL is what makes the readers (promote, thumbnails, the file route)
    # prefer the cleaned blob over the untouched source.
    watermark_clean_method = db.Column(String(16), nullable=True)
    # WHICH detector produced watermark_state: NULL (a row scanned before this
    # column existed, or never scanned) | 'vision' (the Ollama vision model, the
    # route that has always existed and still runs when the extra is absent) |
    # 'detector' (the dedicated SigLIP2 + Grounding DINO extra). Persisted rather
    # than derived because a bank is scanned over weeks and can hold BOTH — and
    # the two disagree at the margins, so "why is this flagged?" has to be
    # answerable per image, not per bank. Additive column (see _SCHEMA_ADDITIONS).
    watermark_source = db.Column(String(16), nullable=True)
    # The detector's raw score in 0..1 for this image (NULL for a vision-model row:
    # that route produces a sentence, not a number). Kept because the flag itself
    # is a THRESHOLD applied to it — storing only the verdict would mean a rescan
    # of the whole bank to answer "what would 0.92 have flagged?", and the
    # threshold is the one knob this feature offers.
    watermark_score = db.Column(Float, nullable=True)
    # Caption pass — a plain DESCRIPTIVE caption (no trigger, no identity omission:
    # a bank has no trigger word and nothing to protect). It doubles as the bank's
    # search text (the search bar matches caption + relpath) AND rides along to the
    # dataset on promotion, so a promoted selection starts already captioned. NULL =
    # not captioned yet. Additive column — created by db.create_all(), no migration.
    caption = db.Column(Text, nullable=True)
    # WHO wrote that caption. NULL = never recorded (every row that predates this
    # column, and every uncaptioned row) | 'asserted' = a human wrote or corrected
    # it — usually in a Dataset, the only place this app offers a caption editor,
    # and carried here by the Dataset -> Bank import | 'joycaption' / 'ollama' =
    # the engine that produced it. Exactly the two vocabularies already on this
    # table (face_cluster_origin's 'asserted', watermark_source's engine name),
    # and 'asserted' keeps its meaning: 🔄 Re-caption SKIPS those rows rather than
    # overwriting a human's words with a model's. NULL rows ARE re-captioned (their
    # origin was never recorded and cannot be recovered), and the button counts
    # them apart from the machine-written ones instead of pretending they are the
    # same thing. See services/caption_origin.py. Additive column (_SCHEMA_ADDITIONS).
    caption_origin = db.Column(String(16), nullable=True)
    # Portable source provenance shared with Dataset images. The only currently
    # accepted shape is normalized Pexels attribution; services validate/canonicalize
    # it before writing. Keeping it on the Bank makes Bank -> Dataset -> Bank
    # round-trips retain the user's lawful source information.
    source_metadata = db.Column(Text, nullable=True)
    # Framing pass — the SAME face/bust/body/back classification the datasets use
    # (Qwen3-VL, CLASSIFY_PROMPT). face = head close-up | bust = upper body |
    # body = full body | back = seen from behind | 'unknown' = a parseable answer
    # that wasn't one of the four | NULL = not classified yet (retryable). Powers
    # the 📐 Framing filter chips AND the coverage advice. Additive column —
    # created by db.create_all(), no migration (see _SCHEMA_ADDITIONS).
    framing = db.Column(String(8), nullable=True, index=True)
    # Provenance pass — computed by the SAME pure-PIL quality scan (no numpy, no
    # model), so every install gets them. See services/image_provenance.py for
    # what each one measures and what it CANNOT tell.
    #   detail_ratio    : effective resolution, 0..1 of the stored size. 0.5 = half
    #                     the stored width is interpolation. RAW score — the
    #                     'soft_detail' verdict is recomputed at read time against
    #                     bank.detail_min, like every other bank score. NOT blur:
    #                     sharpness is a contrast figure, this is a SCALE. It still
    #                     cannot separate an enlargement from a soft photograph.
    #   bars_ratio      : fraction of the frame taken by flat black letterbox bars.
    #   jpeg_quality    : quality of the last JPEG save, 1..100, from the
    #                     quantization tables. A displayed FACT, never a flag.
    #   origin          : 'ai' | 'camera' | 'unknown' — THREE states, never two.
    #                     'unknown' is the normal case (scrapers and chat apps strip
    #                     metadata); it must never be read as "not AI".
    #   origin_evidence : short token naming what proved it ('png-prompt',
    #                     'exif-camera', ...). NEVER the metadata's content — a
    #                     prompt is user data and this column is shown in the UI.
    # NULL on any row scanned by a build that predates them; the next quality scan
    # picks those rows back up on its own (see _scan_pool).
    detail_ratio = db.Column(Float, nullable=True)
    bars_ratio = db.Column(Float, nullable=True)
    jpeg_quality = db.Column(Float, nullable=True)
    origin = db.Column(String(8), nullable=True, index=True)
    origin_evidence = db.Column(String(24), nullable=True)
    # Medium pass — WHAT THE PICTURE IS MADE OF, as opposed to `origin`, which
    # reads the file's metadata. The two answer different questions and must
    # never be conflated: an AI-generated photorealistic portrait is origin='ai'
    # AND medium='photo'; a scanned manga page is origin='unknown' AND
    # medium='anime'. Computed with zero-shot CLIP over the embedding the ✨ Score
    # pass ALREADY cached, so it costs no new image inference.
    #   medium         : 'photo' | 'anime' | 'render3d' | 'illustration'
    #                    | 'unsure' | NULL (never classified — the ✨ Score pass
    #                    has not reached this image, so there is no embedding to
    #                    read and the honest answer is "not scored yet").
    #                    'unsure' is a REAL, measured verdict, not a placeholder:
    #                    see MEDIUM_MARGIN_* in image_bank_service for the numbers
    #                    that make it the majority answer on ambiguous rows.
    #   medium_margin  : the cosine gap between the winning prototype and the
    #                    runner-up. Persisted because it is the only number that
    #                    says HOW MUCH the verdict was worth, it is what the
    #                    'unsure' cut is applied to, and it makes the pass
    #                    re-tunable with no recompute (same read-time-verdict
    #                    philosophy as the quality thresholds).
    # Synergy note: this column is what a future "is this dataset anime or
    # photographic?" check (the Anima lane) should read — it is the only
    # per-image medium signal the app persists.
    medium = db.Column(String(16), nullable=True, index=True)
    medium_margin = db.Column(Float, nullable=True)
    # Face pass — the head's YAW in degrees, signed, as InsightFace reports it
    # (negative = turned one way, positive = the other; the app only ever reads
    # |yaw|, because "turned left" and "turned right" are the same shot type for
    # a training set). It was ALREADY computed by every face pass — used once to
    # gate 'extreme_pose' and then thrown away — so persisting it costs one float
    # and no extra inference. NULL = the pass never ran on this row, ran before
    # this column existed, or found no face; the ⤢ Angle chips call all three
    # "not measured", never "frontal".
    face_yaw = db.Column(Float, nullable=True)
    # Full SHA-256 of the exact EFFECTIVE bytes (cleaned/rotated when selected)
    # described by Quality, Score, Face and Framing.  Watermark geometry is the
    # deliberate exception: it is measured and applied in raw-source coordinates
    # and therefore has its own authority below.
    analysis_fingerprint = db.Column(String(64), nullable=True)
    # SHA-256 of the raw, pre-rotation source used for watermark detection,
    # regions and cleaning. Historical watermark metadata may survive a baked
    # transfer, but it is actionable only while this still matches the raw file.
    watermark_fingerprint = db.Column(String(64), nullable=True)
    # Manual turn, in degrees CLOCKWISE: NULL/0 = untouched | 90 | 180 | 270.
    # (Idea by 1Tomber, GitHub #17.) A bank is a READ-ONLY view over the user's
    # own folder, so a rotation cannot rewrite their file — it is stored here and
    # applied by the ONE resolver every reader goes through
    # (image_bank_service.resolved_image_path), which materialises a turned copy
    # in the bank's own working directory. That makes the turn free of loss where
    # it matters: the source keeps its exact bytes, and the derived copy is always
    # rebuilt from that pristine source, so ANY angle costs exactly one re-encode
    # and four quarter turns cost zero (the row is back at 0 and the copy is
    # dropped). Additive column — existing banks carry NULL and behave as before.
    rotation = db.Column(Integer, nullable=True)
    # Triage decision — same words as dataset images (pending|keep|reject).
    # reject_reason: blur|noise|uniform|small|duplicate|unreadable|manual
    #                |low_aesthetic|nsfw|watermark (the V2 score-derived flags)
    #                |soft_detail|bars (the provenance pass).
    status = db.Column(String(10), nullable=False, default='pending', index=True)
    reject_reason = db.Column(String(16), nullable=True)
    # Set once the image has been promoted (copied) into a dataset — the funnel's
    # provenance, and the guard against promoting the same file twice by accident.
    promoted_dataset_id = db.Column(Integer, nullable=True)
    # Set once the image has been promoted (copied) into another BANK — the
    # second destination of ⬆ Promote, for isolating candidates out of a big
    # dump without committing them to a training container yet. A SEPARATE
    # column on purpose: promoted_dataset_id is stored in user databases and
    # read as "a dataset id" everywhere, so it is never re-pointed at a bank.
    # An image can have gone to both; the two answers stay independent.
    promoted_bank_id = db.Column(Integer, nullable=True)
    # Same portable envelope as FaceDatasetImage.transfer_metadata. A Bank does
    # not interpret Dataset-only values, but must carry them byte-for-metadata
    # through Bank -> Bank and back to a Dataset.
    transfer_metadata = db.Column(Text, nullable=True)
    created_at = db.Column(DateTime, default=db.func.current_timestamp())

    def __repr__(self):
        return f'<BankImage {self.id} bank={self.bank_id} {self.status}>'


class BankFolderPerson(db.Model):
    """"This subfolder is one person" — the user's assertion about a bank folder.

    Scraped sources are very often one folder per person, and making the face pass
    re-discover by inference what the folder name already said is the single most
    expensive way to learn nothing. One row here says it instead: every image of
    that subfolder gets a person id immediately, at zero GPU cost, and the
    embeddings pass then SKIPS those images entirely.

    Keyed on (bank_id, subfolder) — the TOP-LEVEL subfolder, exactly the string the
    Subfolder filter uses ('' = the bank root), so the assertion and the scoping
    facet always talk about the same set of files. It is a rule, not a stamp: it
    survives re-scans, and an image that lands in the folder later joins the group
    on insert. Revoking deletes the row and clears the ids it wrote, putting those
    images back in the way of normal clustering.

    NO relationship() to ImageBank on purpose (see the delete-500 lesson): the
    services delete these rows explicitly, children first."""
    __tablename__ = 'bank_folder_person'
    __table_args__ = (db.UniqueConstraint('bank_id', 'subfolder',
                                          name='uq_bank_folder_person'),)
    id = db.Column(Integer, primary_key=True)
    bank_id = db.Column(
        Integer, db.ForeignKey('image_bank.id', ondelete='CASCADE'),
        nullable=False, index=True)
    # Top-level subfolder name, '' for the bank root. Stored verbatim — it is a
    # user path fragment and is never shown outside this install.
    subfolder = db.Column(Text, nullable=False, default='')
    # The bank-local person id this folder owns. Allocated once, above every id in
    # use at that moment, and NEVER reallocated while the assertion stands — the
    # embeddings pass offsets its own ids past it so the two never collide.
    cluster_id = db.Column(Integer, nullable=False)
    # Last sample check, JSON: {checked_at, sample, scorable, largest, faces,
    # verdict, note}. NULL = never verified (the assertion is still in force —
    # verification informs, it never gates).
    sample_report = db.Column(Text, nullable=True)
    created_at = db.Column(DateTime, default=db.func.current_timestamp())

    def __repr__(self):
        return (f'<BankFolderPerson bank={self.bank_id} '
                f'{self.subfolder!r} #{self.cluster_id}>')


class BankFolderProbe(db.Model):
    """"This subfolder LOOKS like one person" — a MEASUREMENT about a folder.

    Deliberately a different table from BankFolderPerson, because it is a
    different kind of thing with a different lifetime. A probe is something the
    app found out by sampling ~15 images; an assertion is something the user
    declared. Conflating them is exactly the mistake this feature exists to
    avoid: a suggestion must never be able to become a grouping on its own.

    So this table only ever feeds a SUGGESTION. It groups nothing, it writes no
    face_cluster, and confirming it is a click the user makes.

    ``content_sig`` is what makes a probe expire honestly: a cheap DB-only digest
    of the sampled pool's row ids, effective-analysis identities, transform
    markers, and face threshold. Images added, removed, transformed or rebound
    since → the verdict describes a folder that no longer exists, and the UI
    marks it stale rather than suggest from stale evidence."""
    __tablename__ = 'bank_folder_probe'
    __table_args__ = (db.UniqueConstraint('bank_id', 'subfolder',
                                          name='uq_bank_folder_probe'),)
    id = db.Column(Integer, primary_key=True)
    bank_id = db.Column(
        Integer, db.ForeignKey('image_bank.id', ondelete='CASCADE'),
        nullable=False, index=True)
    subfolder = db.Column(Text, nullable=False, default='')
    # 'consistent' | 'mixed' | 'inconclusive' — the same three the single-folder
    # check produces, from the same function. One vocabulary, one meaning.
    verdict = db.Column(String(14), nullable=False)
    sample = db.Column(Integer, nullable=True)      # images actually embedded
    scorable = db.Column(Integer, nullable=True)    # of those, with a usable face
    largest = db.Column(Integer, nullable=True)     # biggest same-person group
    faces = db.Column(Integer, nullable=True)       # distinct people in the sample
    note = db.Column(Text, nullable=True)           # the sentence shown to the user
    content_sig = db.Column(String(40), nullable=True)
    checked_at = db.Column(DateTime, default=db.func.current_timestamp())

    def __repr__(self):
        return (f'<BankFolderProbe bank={self.bank_id} '
                f'{self.subfolder!r} {self.verdict}>')


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
    resolution_multiplier = db.Column(Float, nullable=True)  # multiplicateur linéaire du palier [1.0,1.9] ; NULL/1.0=palier inchangé (resume fidèle)
    init_image = db.Column(String(255), nullable=True)    # Krea img2img : fichier init copié dans COMFYUI_INPUT_DIR
    denoise = db.Column(Float, nullable=True)             # Krea img2img : node 26 denoise
    # Scoring facial objectif (« best epoch », méthode jandordoe) : similarité
    # cosinus InsightFace vs la référence du dataset + état de scorabilité
    # ('scorable'/'no_face'/'low_det'/…). NULL = cellule pas encore scorée.
    face_score = db.Column(Float, nullable=True)
    face_state = db.Column(String(16), nullable=True)
    # WHICH training checkpoint produced this image — the run's record id and the
    # step, written AT GENERATION TIME by every surface that launches (Test
    # Studio, LoRA Canvas, comparison grid). Deliberately not a ForeignKey: run
    # records outlive their dataset, exactly like canvas_node_position.
    #
    # This inverts a dependency. Membership used to be DERIVED, on every render,
    # from the deployed LoRA's filename through comfyui._parse_trained_stem — the
    # heuristic that already shipped a bug (multi-word triggers cut at the
    # underscore, 2026-07-17). The parse now survives only in the one-shot
    # backfill (services.checkpoint_link_backfill), never in the hot path.
    #
    # NULL means "not linked", never "unknown, guess it": a legacy row whose
    # filename carries no run tag stays NULL and is simply absent from the
    # checkpoint gallery, with a counter saying how many there are. Additive
    # columns → existing databases keep their rows (see _SCHEMA_ADDITIONS).
    record_id = db.Column(Integer, nullable=True, index=True)
    step = db.Column(Integer, nullable=True)
    created_at = db.Column(DateTime, default=db.func.current_timestamp())
    # ✨ A row that is NOT a Test Studio cell: the ✨ Upscale & improve pass run on
    # one of these images from the ◉ Canvas lightbox. The result has to live in
    # THIS table — `canvas_image_node.image_id` is a `lora_test_image.id`, so
    # anything the board must be able to pin has to be one — but it was never
    # generated by a checkpoint×strength sweep and must not be read as one.
    #
    # `derivation_kind` is therefore the EXCLUSION MARKER, not decoration. Every
    # query in lora_test_studio.py that means "the cells of this dataset" goes
    # through `_cells()`, which filters these rows out; an audit of the 21 read
    # sites found TEN that otherwise break — a pending improve counted as an
    # active test run and blocked new launches, a failed one was resumed as a
    # Z-Image cell, and the row entered cell_scores/face_ranking as a vote for
    # the checkpoint that did not make it. A test forbids bare `LoraTestImage.query`
    # in that module so the invariant is enforced rather than remembered.
    #
    # `run_id` stays NULL on these rows as well — that is what keeps them out of
    # checkpoint_timeline (which filters `run_id IS NOT NULL`), where a 2 MP
    # upscale spliced into an epoch-by-epoch morph would be a lie. `record_id`
    # and `step` ARE copied, on purpose: that is what puts the improvement in the
    # same checkpoint gallery as its source, next to it, and on the board.
    #
    # `parent_image_id` is the source row, deliberately not a ForeignKey (same
    # reason as `record_id`): a deleted source must leave a picture you can still
    # look at, never a database error. Additive columns → existing databases keep
    # their rows and read NULL everywhere (see _SCHEMA_ADDITIONS).
    parent_image_id = db.Column(Integer, nullable=True, index=True)
    derivation_kind = db.Column(String(32), nullable=True)

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
    # WHICH TABLE `dataset_id` points into: 'face_dataset' or 'video_dataset'.
    # The two share one integer space, so without this a video run and a face
    # dataset of the same id are indistinguishable and every consumer resolves
    # the wrong row in silence. NULL on every row that predates the column, and
    # read as 'face_dataset' — see services/cloud_run_dataset.table_of, which is
    # the only place allowed to answer this question.
    dataset_table = db.Column(db.String(32))
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
    # When the user asked this run to stop. Durable on purpose: an in-memory
    # threading.Event does not survive a restart and cannot be enforced by
    # anything but the monitor thread — which is exactly what may be dead.
    # The supervisor uses this to terminate a pod whose stop was never honoured.
    stop_requested_at = db.Column(db.DateTime)
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
    # JSON run_snapshot: everything ELSE that defined this launch and that the
    # manifest could only hash — the caption TEXT of every image, each image's
    # true content hash, the dataset's kind/fidelity/reference, and the machine
    # (ai-toolkit revision, torch/CUDA, GPU, identity of the base-model FILE).
    # Without it a comparison can say "3 captions changed" but never what they
    # said, and blames the dataset for a gap that came from a trainer upgrade.
    # NULL on every pre-feature row: that run PREDATES full snapshots, which the
    # diff states rather than inventing. Additive column (migration in create_app).
    snapshot = db.Column(db.Text)
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
    note = db.Column(db.Text)                                # free-form run note (Lab)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class CheckpointNote(db.Model):
    """A free-form note on one checkpoint of a run, keyed by (record_id, step).
    Checkpoints aren't their own rows, so notes live here. New table -> created
    by db.create_all(), no migration."""
    __tablename__ = 'checkpoint_note'
    id = db.Column(db.Integer, primary_key=True)
    record_id = db.Column(db.Integer, nullable=False, index=True)
    step = db.Column(db.Integer, nullable=False)
    note = db.Column(db.Text, nullable=False, default='')
    __table_args__ = (db.UniqueConstraint('record_id', 'step',
                                          name='uq_checkpoint_note'),)


class CheckpointPreview(db.Model):
    """The Lab's inline-generated preview for one checkpoint (record_id, step):
    a same-prompt/same-seed image so an experienced user can eyeball how the LoRA
    evolves epoch by epoch. Checkpoints aren't their own rows, so — like notes —
    the preview mapping lives here, keyed by (record_id, step). It does NOT store
    an image itself: the picture is produced by the reused Test-Studio engine and
    lands in the per-dataset folder as a LoraTestImage; `lora_test_image_id` points
    at that row so the node reads the (async) filename + status live.

    Previews ACCUMULATE. Until the LoRA Canvas there was a
    ``UniqueConstraint('record_id', 'step')`` here and regenerating a checkpoint
    REPLACED the pointer: the previous image stayed on disk but nothing pointed
    at it any more, so a second look at the same epoch quietly erased the first.
    A checkpoint now keeps every preview it was ever given, newest first, and the
    node shows the newest one with the count beside it. Lifting a constraint on
    SQLite means recreating the table — done once, row-count-guarded, in
    ``services.checkpoint_preview_migration``.

    New table -> created by db.create_all(); the constraint lift is the only
    migration."""
    __tablename__ = 'checkpoint_preview'
    id = db.Column(db.Integer, primary_key=True)
    record_id = db.Column(db.Integer, nullable=False, index=True)
    step = db.Column(db.Integer, nullable=False)
    dataset_id = db.Column(db.Integer, nullable=False, index=True)
    # The reused Test-Studio result row that carries the actual image (its
    # filename is null until the job completes); the node joins through it.
    lora_test_image_id = db.Column(db.Integer, nullable=True)
    prompt = db.Column(db.Text, nullable=False, default='')
    seed = db.Column(db.BigInteger, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    # No UniqueConstraint on (record_id, step) — see the class docstring. The
    # composite index replaces it: the reads are all "every preview of this
    # checkpoint", which is exactly what the old unique index used to serve.
    __table_args__ = (db.Index('ix_checkpoint_preview_record_step',
                               'record_id', 'step'),)


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


class CanvasNodePosition(db.Model):
    """Where the user DRAGGED one run's card on the ◉ LoRA Canvas.

    A display preference, never provenance: moving a card changes nothing about
    which run continued which — that stays derived from ``parent_record_id`` /
    ``resumed_from``. One row per (dataset, run); no row at all means "wherever
    the automatic tree puts it", which is why ``✦ Tidy up`` is simply deleting
    every row of the lane.

    The rows also do a second, less obvious job. The automatic layout re-centres
    a parent over its children, so a NEW fork would shove every ancestor
    sideways and quietly destroy an arrangement the user had built. As soon as a
    lane holds one dragged card, the canvas writes a row for every OTHER card of
    that lane too, at the position it already occupied — from then on a new run
    lands in free space and nothing already on the board moves.

    ⚠️ The ``dataset`` relationship is not decoration. Child models here declare
    only a table-level ForeignKey, and without a mapper-level relationship the
    unit of work has no ordering dependency: SQLAlchemy emits
    ``DELETE FROM face_dataset`` first and a legacy database whose FK lacks
    ON DELETE CASCADE answers HTTP 500. delete_dataset ALSO deletes these rows
    explicitly and flushes before the parent — belt and braces, because that bug
    has already shipped once in this project. New table -> created by
    db.create_all(), no migration."""
    __tablename__ = 'canvas_node_position'
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(
        db.Integer, db.ForeignKey('face_dataset.id', ondelete='CASCADE'),
        nullable=False, index=True)
    # The training run's record id (training_run_record.id). Deliberately NOT a
    # ForeignKey: run records OUTLIVE their dataset (history is kept on purpose,
    # see delete_dataset), so a constraint here would fight that.
    record_id = db.Column(db.Integer, nullable=False, index=True)
    x = db.Column(Float, nullable=False, default=0.0)
    y = db.Column(Float, nullable=False, default=0.0)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow,
                           onupdate=datetime.utcnow)
    dataset = db.relationship('FaceDataset')
    __table_args__ = (db.UniqueConstraint('dataset_id', 'record_id',
                                          name='uq_canvas_node_position'),)


class CanvasImageNode(db.Model):
    """🖼 A generated image PINNED onto the ◉ LoRA Canvas, as a node of its own.

    The board compares checkpoints; the thing being compared is the picture. A
    picture you can only see one at a time, in a modal, is a picture you cannot
    compare — so an image can be dropped onto the board, moved and resized like
    any other node, next to the pill that made it.

    Same table shape and the same reasoning as ``CanvasNodePosition``, one row
    per (dataset, image): coordinates are LANE-LOCAL, the same world units the
    card positions use, so both live in one coordinate system.

    ⚠️ Lane-local is the ANCHOR, not a cage — and unlike a card's, these
    coordinates may be NEGATIVE. A picture goes wherever it is dropped, above or
    left of its own lane included; only the point it is measured FROM is the
    lane's origin. That reference is what makes the number survive: a lane's
    position on the board is derived from which datasets are ticked in the
    canvas filter and from the height of every lane above it, so it moves by
    hundreds of world units whenever an unrelated dataset is unticked or gains a
    run. Board-absolute coordinates would leave every picture hovering over the
    wrong lane after one filter click, and would need a migration nobody could
    compute server-side (lane heights are laid out in the browser). Measured
    from its own lane, a picture travels with the run it is evidence about, and
    every row ever written keeps meaning exactly what it meant — no migration,
    on any database. ``cloud_training.CANVAS_IMAGE_REACH`` bounds the distance
    so one corrupt row cannot collapse ✦ Fit; ✦ Tidy up is the way back.

    ⚠️ ``visible`` is the whole feature, not a flag. Closing a pinned image must
    NOT forget where it was: re-opening it has to put it back exactly where and
    at the size it was closed at. So the close writes ``visible = False`` and
    keeps the geometry; only an explicit "forget" would delete the row, and
    nothing in the UI does that today.

    ``image_id`` is ``lora_test_image.id`` and is deliberately NOT a ForeignKey,
    for the same reason ``record_id`` above is not one: it is resolved on read
    and a row whose image no longer exists is DELETED there (see
    ``canvas_image_nodes``). A constraint would turn "the user deleted that
    render" into a database error instead of a node quietly leaving the board.

    The link to the source checkpoint is NOT stored: it is read off the image
    row (``record_id`` / ``step``), so a pinned image can never disagree with
    the gallery about which checkpoint produced it. New table -> created by
    db.create_all(), no migration."""
    __tablename__ = 'canvas_image_node'
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(
        db.Integer, db.ForeignKey('face_dataset.id', ondelete='CASCADE'),
        nullable=False, index=True)
    image_id = db.Column(db.Integer, nullable=False, index=True)
    x = db.Column(Float, nullable=False, default=0.0)
    y = db.Column(Float, nullable=False, default=0.0)
    w = db.Column(Float, nullable=False, default=260.0)
    h = db.Column(Float, nullable=False, default=260.0)
    visible = db.Column(db.Boolean, nullable=False, default=True)
    # 🖼🖼 Which side-by-side GROUP this picture is fused into, and where in it.
    # Dropping one pinned image onto another turns them into one node whose
    # pictures sit edge to edge; there is no limit on how many join, and
    # dragging one off the strip makes it a node of its own again.
    #
    # A group is deliberately NOT a row of its own. It is a membership carried
    # by pictures that stay ordinary nodes, because the geometry above is a
    # per-image promise ("close it, re-open it from its gallery, it comes back
    # where and how big you left it") and a container node would have to keep a
    # second copy of every member's box, free to disagree with this one. Joining
    # a group therefore never touches x/y/w/h — the strip is computed from them
    # at draw time (frontend utils/canvasImageGroups), and leaving one hands
    # them straight back.
    #
    # Both are NULLABLE and ADDITIVE (see _SCHEMA_ADDITIONS): a database that
    # predates them reads NULL everywhere and draws the board it always drew.
    group_id = db.Column(db.String(40), nullable=True)
    group_pos = db.Column(db.Integer, nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow,
                           onupdate=datetime.utcnow)
    dataset = db.relationship('FaceDataset')
    __table_args__ = (db.UniqueConstraint('dataset_id', 'image_id',
                                          name='uq_canvas_image_node'),)


class CanvasLayoutPreset(db.Model):
    """💾 A NAMED snapshot of the whole board's arrangement.

    ``CanvasNodePosition`` and ``CanvasImageNode`` hold ONE arrangement — the
    live one. That is right for a board you tidy as you go and wrong the moment
    the board is used for what it is good at: you spend twenty minutes laying
    two datasets' renders out to judge a likeness, then you need the board for
    something else, and the only two moves the app offered were "leave it there
    forever" or ✦ Tidy up, which throws it away. A preset is the third: keep
    this arrangement, use the board, put it back.

    ── Why one blob and not a second pair of tables ─────────────────────────
    ``payload`` is JSON: ``{"positions": {"<dataset id>": [{record_id,x,y}...]},
    "images": {"<dataset id>": [{image_id,x,y,w,h,visible,group_id,group_pos}...]}}``.
    Mirroring the two live tables would double every future change to them and
    would need the same ownership checks in two more places. More importantly a
    preset is a MEMORY, not state: a run deleted since the preset was saved
    simply has no card to put back, and a row with a foreign key would instead
    make that a database error or a cascade that quietly edits history. Restore
    therefore goes through the very same validated save functions the live board
    uses (``save_canvas_positions`` / ``save_canvas_image_nodes``), which drop
    what no longer exists — so a preset can never smuggle in a card, a picture
    or a dataset the user does not own, however it was written.

    Scoped by ``user_id`` rather than by dataset, because the board spans
    datasets — that is the whole point of it — and a preset that could only
    remember one lane would not be a preset of the board.

    New table -> created by db.create_all(), no migration."""
    __tablename__ = 'canvas_layout_preset'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(String(36), nullable=False, index=True, default='local')
    name = db.Column(db.String(80), nullable=False)
    payload = db.Column(db.Text, nullable=False, default='{}')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow,
                           onupdate=datetime.utcnow)
    __table_args__ = (db.UniqueConstraint('user_id', 'name',
                                          name='uq_canvas_layout_preset'),)


# ---------------------------------------------------------------------------
# 🎬 Video lane — a SEPARATE set of tables, on purpose.
#
# The image lane rests on one invariant: one row = one file. `BankImage.relpath`
# is relative to the bank's source_path, every reader resolves it to a path, the
# dedup hashes a file, the promotion copies a file. The video lane cannot hold
# that invariant — a two-hour rush is ONE file and four hundred training clips.
#
# Bolting temporal bounds onto BankImage would have meant a migration over tables
# that already carry tens of thousands of rows in user databases, plus every
# existing pass learning to say "not applicable" for video rows. These tables are
# NEW, so db.create_all() creates them and no migration is needed at all, and not
# one line of the image lane changes.
# ---------------------------------------------------------------------------


class VideoBank(db.Model):
    """A triage bank over a folder of SOURCE VIDEOS, referenced in place.

    Same promise as the image bank: the source folder is never written to. The
    video bank keeps that promise more literally than one might expect — it
    stores no media at all beyond thumbnails, because cutting a clip means
    re-encoding it, and we only pay that at promotion, for the clips actually
    kept. What lives here is bounds and decisions."""
    __tablename__ = 'video_bank'
    id = db.Column(Integer, primary_key=True)
    user_id = db.Column(String(36), nullable=False, index=True, default='local')
    name = db.Column(String(100), nullable=False)
    source_path = db.Column(Text, nullable=False)      # absolute folder, read-only
    # Persisted summary of the last pipeline run (JSON), same role as
    # ImageBank.pipeline_report: the outcome is still there tomorrow morning.
    pipeline_report = db.Column(Text, nullable=True)
    created_at = db.Column(DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(DateTime, default=db.func.current_timestamp(),
                           onupdate=db.func.current_timestamp())

    def __repr__(self):
        return f'<VideoBank {self.id} {self.name}>'


class VideoSource(db.Model):
    """One source FILE of a video bank — the level the image lane does not have.

    It exists because the facts a file carries (duration, native fps, resolution,
    codec) are not facts a clip carries, and because a probe that fails must fail
    once per file rather than once per clip. `fps_native` is the SOURCE's rate and
    is never what a clip is encoded at: the target profile decides that. Reading
    this column at encode time is precisely how a 16 fps target ends up with
    accelerated motion."""
    __tablename__ = 'video_source'
    id = db.Column(Integer, primary_key=True)
    bank_id = db.Column(
        Integer, db.ForeignKey('video_bank.id', ondelete='CASCADE'),
        nullable=False, index=True)
    relpath = db.Column(Text, nullable=False)          # relative to bank.source_path
    file_size = db.Column(Integer, nullable=True)
    duration_s = db.Column(Float, nullable=True)
    fps_native = db.Column(Float, nullable=True)       # the SOURCE's rate, never the target's
    width = db.Column(Integer, nullable=True)
    height = db.Column(Integer, nullable=True)
    codec = db.Column(String(24), nullable=True)
    # NULL = not probed yet | 'ok' | 'unreadable'. Mirrors BankImage.quality_state's
    # vocabulary so "the file is broken" reads the same in both lanes.
    probe_state = db.Column(String(12), nullable=True)
    # NULL = shot detection has not run on this file | 'ok' | 'error'. Separate from
    # probe_state because a file can be perfectly readable and still fail detection.
    detect_state = db.Column(String(12), nullable=True)
    created_at = db.Column(DateTime, default=db.func.current_timestamp())

    def __repr__(self):
        return f'<VideoSource {self.id} bank={self.bank_id} {self.relpath}>'


class VideoClip(db.Model):
    """One detected shot — the unit of work of the whole video lane.

    BOUNDS ARE PTS SECONDS, AND THAT IS THE CANONICAL FORM. The detector reasons
    in frame indices on a decoded stream, which is frame-exact and tempting, but
    scraped material is routinely variable-frame-rate: index n corresponds to no
    stable instant there, while a presentation timestamp does. `ffmpeg -ss` takes
    seconds too. The frame indices are kept because they are what the detector
    actually said, and they make a disagreement debuggable — but nothing cuts
    from them.

    Sub-second precision is not a nicety: rounding a bound to the nearest second
    on a 2-second clip moves a quarter of the sample."""
    __tablename__ = 'video_clip'
    id = db.Column(Integer, primary_key=True)
    bank_id = db.Column(
        Integer, db.ForeignKey('video_bank.id', ondelete='CASCADE'),
        nullable=False, index=True)
    source_id = db.Column(
        Integer, db.ForeignKey('video_source.id', ondelete='CASCADE'),
        nullable=False, index=True)
    start_s = db.Column(Float, nullable=False)
    end_s = db.Column(Float, nullable=False)
    start_frame = db.Column(Integer, nullable=True)    # informative; never cut from
    end_frame = db.Column(Integer, nullable=True)
    # Which detector drew this boundary: 'transnetv2' | 'manual'. Persisted rather
    # than derived because a bank is worked on over days and can hold both, and
    # "why is this cut here?" has to be answerable per clip.
    detector = db.Column(String(16), nullable=True)
    # NULL = no thumbnail yet | 'ok' | 'error'. The thumbnail is the ONLY media the
    # bank writes, and the gallery shows nothing else — there is deliberately no
    # per-clip preview file to hover, because that would mean encoding every clip
    # including the ones about to be rejected.
    thumb_state = db.Column(String(12), nullable=True)
    # Quality metrics (wave 2): the RAW per-clip summary the metrics scan wrote,
    # as JSON — motion_mean/motion_p95, luma_min/luma_mean, sharpness_p90,
    # freeze_ratio, sharpest_frame_s, metrics_state. RAW scores only, never
    # verdicts: flags are recomputed at read time against the bank's thresholds
    # (services/video_metrics.verdicts), so retuning a cut re-sorts the bank with
    # no rescan — the same philosophy as the image lane's quality columns, and it
    # matters more here because decoding is ~85 % of this lane's cost. One JSON
    # column rather than eight scalar ones: the summary is written and read as a
    # unit, and adding a metric to the scan must not need a migration.
    # NULL = the scan has not reached this clip. Additive — see _SCHEMA_ADDITIONS.
    metrics_json = db.Column(Text, nullable=True)
    # 🔎 Search pass: NULL = never embedded | 'ok' | 'unreadable'. Only the STATE
    # lives here; the vectors themselves are a .npz next to the bank's thumbnails
    # (services/video_clip_search), for the same reason the image lane keeps its
    # ✨ Score embeddings out of SQLite — three 768-float rows per clip over a bank
    # of thousands is a blob store, not a column, and every reader of it wants the
    # whole matrix at once rather than one row. The state is here because it is
    # what the resume contract and the counters read, and both must be answerable
    # without touching numpy. Additive — see _SCHEMA_ADDITIONS.
    embed_state = db.Column(String(12), nullable=True)
    # 🗣 Caption: what HAPPENS in this shot, in prose. Two things at once, and the
    # second is why it is a column rather than a search index: it is the text the
    # 🔎 hybrid search matches literally, AND it is what the promotion writes into
    # the clip's `.txt` sidecar — which IS the training prompt. An empty sidecar
    # is not a neutral default: ai-toolkit trains it as an empty prompt and says
    # nothing (see video_clip_export.write_sidecar).
    caption = db.Column(Text, nullable=True)
    # NULL = never captioned | 'ok' (generated) | 'edited' (a human corrected it,
    # and a bulk re-run must not overwrite that) | 'error'. The 'edited' state is
    # the whole reason this is not derivable from `caption is not None`: a
    # generated caption is a draft, and losing a correction to the next pass is
    # the one thing that would stop anyone from ever editing one.
    caption_state = db.Column(String(12), nullable=True)
    # WHICH checkpoint wrote this caption. NULL for a caption a human typed and
    # for a clip that has none — nothing wrote those, so nothing is claimed.
    #
    # A column rather than a key inside a JSON blob because the question it
    # answers is per-ROW and asked at read time: two checkpoints do not produce
    # comparable captions, and changing the setting mid-corpus is exactly what
    # making it configurable invites. A bank captioned half by one and half by
    # another has to stay readable, which means every row saying which — and a
    # JSON field would make "show me what the old model wrote" a full-table
    # parse instead of a filter.
    caption_model = db.Column(String(120), nullable=True)
    # And WHICH prompt style produced it. Same reasoning as the model, and the
    # measurement says it matters more: the prompt turned out to dominate how
    # plainly a caption describes its footage, so two styles do not write
    # comparable captions any more than two checkpoints do. NULL for a human's
    # caption and for a failed one.
    caption_style = db.Column(String(16), nullable=True)
    # Triage decision — the same three words as the image lane.
    status = db.Column(String(10), nullable=False, default='pending', index=True)
    reject_reason = db.Column(String(16), nullable=True)
    # Set once the clip has been encoded into a video dataset. A REAL foreign key
    # with SET NULL, unlike the image lane's plain integer: throwing away a badly
    # cut dataset must cost the encode, not the triage. The clips stay, they just
    # stop claiming to have been promoted.
    promoted_dataset_id = db.Column(
        Integer, db.ForeignKey('video_dataset.id', ondelete='SET NULL'),
        nullable=True, index=True)
    created_at = db.Column(DateTime, default=db.func.current_timestamp())

    def __repr__(self):
        return f'<VideoClip {self.id} src={self.source_id} {self.start_s}-{self.end_s}>'


class VideoDataset(db.Model):
    """A built video training set: a flat folder of .mp4 files with homonym .txt
    captions, plus the memory of HOW it was built.

    `target_profile` is not decoration. It is the key into the target catalogue
    (services/video_targets.py) that says at which fps and at which length these
    clips were encoded, and it is stored here rather than recomputed because the
    files on disk are already committed to it. `fps` and `frames` are DENORMALISED
    copies of what the profile said at build time — deliberately, so a dataset
    built last month still reports what it actually contains even if the profile's
    defaults move."""
    __tablename__ = 'video_dataset'
    id = db.Column(Integer, primary_key=True)
    user_id = db.Column(String(36), nullable=False, index=True, default='local')
    name = db.Column(String(100), nullable=False)
    target_profile = db.Column(String(24), nullable=False)
    fps = db.Column(Integer, nullable=True)            # as encoded, not as configured now
    frames = db.Column(Integer, nullable=True)         # clip length actually used
    width = db.Column(Integer, nullable=True)
    height = db.Column(Integer, nullable=True)
    output_dir = db.Column(Text, nullable=False)
    created_at = db.Column(DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(DateTime, default=db.func.current_timestamp(),
                           onupdate=db.func.current_timestamp())

    def __repr__(self):
        return f'<VideoDataset {self.id} {self.name} {self.target_profile}>'


class VideoDatasetClip(db.Model):
    """One encoded .mp4 of a video dataset, and the memory of where it came from.

    The provenance columns are what make this more than a directory listing. The
    dataset holds finished clips, but each one still names its source file and the
    bounds it was cut at — so a later re-export to a different target is a re-encode
    from the original, not a re-scan from scratch. `source_clip_id` is intentionally
    NOT a foreign key: the bank it points into is a scratch container the user is
    free to delete, and the provenance must outlive it."""
    __tablename__ = 'video_dataset_clip'
    id = db.Column(Integer, primary_key=True)
    dataset_id = db.Column(
        Integer, db.ForeignKey('video_dataset.id', ondelete='CASCADE'),
        nullable=False, index=True)
    filename = db.Column(String(255), nullable=False)   # clip_0001.mp4
    caption = db.Column(Text, nullable=True)            # written to the homonym .txt
    source_bank_id = db.Column(Integer, nullable=True)
    source_clip_id = db.Column(Integer, nullable=True)
    src_relpath = db.Column(Text, nullable=True)
    start_s = db.Column(Float, nullable=True)
    end_s = db.Column(Float, nullable=True)
    created_at = db.Column(DateTime, default=db.func.current_timestamp())

    def __repr__(self):
        return f'<VideoDatasetClip {self.id} ds={self.dataset_id} {self.filename}>'
