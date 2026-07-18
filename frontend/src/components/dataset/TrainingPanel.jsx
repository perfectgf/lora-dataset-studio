// react-frontend/src/components/dataset/TrainingPanel.jsx
import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Link } from 'react-router-dom';
import { getCsrfToken } from '../../api/fetchClient';
import { useCapabilities } from '../../context/CapabilitiesContext';
import { postJson } from '../../hooks/useDataset';
import {
  checkpointSelectionMatchesTraining,
  checkpointVariantLabel,
  checkpointVariantOptions,
  cloudTrainingLaunchPayload,
  defaultCheckpointBase,
  defaultCheckpointVariant,
  loraFolderLabel,
  normalizeCheckpointVariant,
  trainingRunSelection,
  trainFamilyLabel,
} from '../../utils/checkpointBrowser';
import {
  describeZImageRecipe,
  isLongZImageTurboRun,
  ZIMAGE_TURBO_LONG_RUN_STEPS,
} from '../../utils/zimageTrainingRecipe';
import {
  compatibleTrainingPresetSelection,
  filterTrainingPresets,
  trainingPresetApplyPayload,
  trainingPresetDatasetKind,
  trainingPresetSnapshotScope,
} from '../../utils/trainingPresets';
import { runConfirmableTrainingRequest } from '../../utils/trainingConfirmations';
import { HelpBadge } from '../../help/HelpMode';
import { useToast } from '../common/Toast';
import ContinueDialog from './ContinueDialog';
import TrainingProgress from './TrainingProgress';
import PreflightModal from './PreflightModal';
import { DatasetVersionChip, RunIdChip } from './RunIdentityBadges';
import {
  cloudGroupsFrom, localRunIdentity, runRowDomId,
} from '../../utils/runIdentity';

// Plancher dur / recommandé par famille — miroir de TRAIN_MIN_IMAGES côté serveur
// (le preflight reste l'autorité ; ceci ne sert qu'à désactiver le bouton tôt).
const TRAIN_MIN = { zimage: [12, 20], sdxl: [20, 30], krea: [15, 20], flux: [15, 20], flux2klein: [15, 20] };
// Slider mode: images are only a denoising substrate → mirror of
// TRAIN_MIN_IMAGES_SLIDER server-side (the preflight stays authoritative).
const TRAIN_MIN_SLIDER = [4, 12];

// Slider LoRA (Beta) — honest per-family status. Every family can ATTEMPT a
// slider run (ai-toolkit's concept_slider trainer has no per-arch gating), but
// only community results back some of them; the notes say exactly what is known.
const SLIDER_FAMILY_NOTES = {
  krea: 'Krea 2 — reference family for sliders: the strongest-rated community sliders on Civitai are Krea-based, and the Turbo de-distillation adapter is already wired here. Still Beta: our own pipeline is untested.',
  zimage: 'Z-Image — experimental: ai-toolkit has known upstream issues on this slider path (issue #554). The community workaround (batch 1, no embedding cache) is applied automatically, but the run may still fail.',
  flux2klein: 'FLUX.2 Klein — experimental: BFL ships an undistilled Base checkpoint made for training, which is structurally promising for sliders, but nobody has verified this trainer path yet.',
  flux: 'FLUX.1 — experimental: the slider method\'s own authors describe FLUX support as experimental.',
  sdxl: 'SDXL — experimental, ironically: sliders were born on SDXL, but ai-toolkit\'s modern slider trainer routes SDXL through its legacy model class — signature-compatible, unproven.',
};

// « Custom weights… » : valeur-sentinelle de l'entrée du sélecteur de base qui
// révèle le champ chemin. Les familles qui l'exposent + celles honorant VAE/TE
// (miroir de CUSTOM_WEIGHTS_FAMILIES / VAE_TE_OVERRIDE_FAMILIES côté serveur ;
// base-info les renvoie, ces défauts ne servent qu'avant son chargement).
const CUSTOM_BASE_SENTINEL = '__custom_weights__';
const DEFAULT_CUSTOM_FAMILIES = ['sdxl', 'krea', 'flux', 'flux2klein'];
const defaultTrainingVariant = (family) => (
  family === 'krea' ? 'base' : family === 'flux2klein' ? '4b' : 'turbo'
);
// Absolute path = the persisted custom-weights path (never a ComfyUI-relative
// base name): Windows drive (C:\), UNC (\\), or POSIX (/…).
const looksAbsolute = (p) => /^(?:[A-Za-z]:[\\/]|\\\\|\/)/.test(String(p || ''));
const baseName = (p) => String(p || '').replace(/[\\/]+$/, '').split(/[\\/]/).pop() || String(p || '');

const fmtBytes = (b) => {
  if (b == null) return '';
  if (b >= 1e9) return `${(b / 1e9).toFixed(1)} GB`;
  if (b >= 1e6) return `${Math.round(b / 1e6)} MB`;
  return `${Math.max(1, Math.round(b / 1e3))} KB`;
};

function CheckpointPortal({ host, children }) {
  return host ? createPortal(children, host) : children;
}

// Relative "15m ago" from a naive-UTC backend timestamp — mirrors CloudRunsPage
// so a checkpoint group's header reads exactly like its Runs row.
function timeAgo(iso) {
  if (!iso) return '';
  const t = new Date(/[Z+]/.test(iso) ? iso : `${iso}Z`).getTime();
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

// Family label for a checkpoint group header — mirrors CloudRunsPage's FAMILY_LABEL.
const GROUP_FAMILY_LABEL = { zimage: 'Z-Image', krea: 'Krea 2', sdxl: 'SDXL', flux: 'FLUX.1', flux2klein: 'FLUX.2 Klein' };
const groupFamLabel = (f) => GROUP_FAMILY_LABEL[f] || f || 'LoRA';

/** Panneau d'entraînement LoRA : lance l'UI ai-toolkit (pause ComfyUI),
 * affiche l'état, liste les checkpoints et importe celui choisi.
 * Poll régulier : c'est ce poll qui fait avancer la file (fin du courant → suivant). */
export default function TrainingPanel({ ds, keptCount, kind, onCheckpointsChange,
                                        checkpointHost = null,
                                        navigationPanel = null,
                                        onNavigationStateChange,
                                        onPanelOpenChange }) {
  const isConcept = kind === 'concept';
  const isStyle = kind === 'style';
  const isConceptual = isConcept || isStyle;
  const { caps } = useCapabilities();
  const toast = useToast();
  const [status, setStatus] = useState({ in_progress: false, installed: true, queue: [], current: null });
  const [statusLoaded, setStatusLoaded] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [checkpointsOpen, setCheckpointsOpen] = useState(false);
  const [checkpoints, setCheckpoints] = useState([]);
  const [ckLoaded, setCkLoaded] = useState(false);
  // {registered, version, changed, diff} — provenance du dataset (registre).
  const [datasetState, setDatasetState] = useState(null);
  // Saves cloud synchronisés en local (y compris ceux d'un run EN COURS) —
  // liste séparée : le prompt Resume-or-Fresh ne raisonne que sur le local.
  const [cloudCkpts, setCloudCkpts] = useState([]);
  // Same saves GROUPED by source run (id/status/gpu/cost/timing) — the panel
  // renders one identity header per run so look-alike epoch sets are no longer
  // ambiguous. Falls back to a single synthetic group if the server is older.
  const [cloudGroups, setCloudGroups] = useState([]);
  // {run_dir_bytes, cloud_staging_bytes, deployed_bytes, total_bytes}
  const [diskUsage, setDiskUsage] = useState(null);
  // {steps, kind, n_images, rationale} renvoyé par /train/checkpoints — le POURQUOI
  // du barème adaptatif, affiché avec le champ Steps (pédagogie, pas boîte noire).
  const [stepsInfo, setStepsInfo] = useState(null);
  const [imported, setImported] = useState([]);
  const [enqErr, setEnqErr] = useState(null);
  // Base d'entraînement (officielle ou merge custom) + variante + conversion.
  const [baseInfo, setBaseInfo] = useState(null);
  const [base, setBase] = useState('');
  // « Custom weights… » (local-only) : quand actif, `base` porte un chemin ABSOLU
  // vers un .safetensors de la même architecture (krea/flux/flux2klein/sdxl).
  const [customBase, setCustomBase] = useState(false);
  // Overrides SDXL UNIQUEMENT : chemin VAE + chemin/te repo-id du text-encoder.
  const [vaePath, setVaePath] = useState('');
  const [tePath, setTePath] = useState('');
  const [variant, setVariant] = useState('turbo');
  // Type de LoRA : 'zimage' (défaut, encodeur Qwen3-4B) ou 'sdxl' (checkpoints ComfyUI).
  const [trainType, setTrainType] = useState('zimage');
  // Navigateur de résultats indépendant : changer la configuration du PROCHAIN
  // entraînement ne doit jamais faire disparaître les checkpoints que l'utilisateur
  // est en train de consulter dans la section dédiée.
  const [checkpointTrainType, setCheckpointTrainType] = useState('zimage');
  const [checkpointBase, setCheckpointBase] = useState('');
  const [checkpointVariant, setCheckpointVariant] = useState('turbo');
  const checkpointSelectionDataset = useRef(null);
  const checkpointRequest = useRef(0);
  // Réglages ai-toolkit avancés éditables (rank / resolution / save_every /
  // sample_every / sample_prompts), chargés depuis base-info ; persistés par POST
  // /train/settings via ds.setTrainSettings.
  const [adv, setAdv] = useState(null);
  // Slider LoRA mode (Beta) : état serveur (colonne dédiée train_slider) + brouillon
  // local des champs texte (édition libre, sauvés au blur comme les sample prompts).
  const [slider, setSlider] = useState(null);
  const [sliderBusy, setSliderBusy] = useState(false);
  const [sliderDraft, setSliderDraft] = useState({ positive: '', negative: '', target_class: '', anchor: '' });
  // Textarea des prompts de preview : état local (édition libre), sauvé au blur —
  // resynchronisé sur la valeur stockée canonique chaque fois que `adv` arrive/change.
  const [samplePromptsText, setSamplePromptsText] = useState('');
  // Presets de réglages avancés : snapshots nommés, partageables (fichier JSON).
  // Stockés bruts côté serveur ; la validation se fait à l'APPLICATION (clés
  // inconnues ignorées, valeurs invalides signalées) → tolérant aux versions.
  const [presets, setPresets] = useState([]);
  const [presetSel, setPresetSel] = useState('');
  const [presetBusy, setPresetBusy] = useState(false);
  const [trainTypeBusy, setTrainTypeBusy] = useState(false);
  const presetFileRef = useRef(null);

  const refreshStatus = async () => {
    try {
      const r = await fetch('/api/dataset/train/status', { credentials: 'include' });
      if (!r.ok) return;
      const d = await r.json();
      // {'available': false}: ai-toolkit went unconfigured/invalid after this
      // panel was already shown (stale client-side caps.training_visible, or
      // the server-side 30s capability cache just expired) — degrade to safe
      // defaults instead of storing a payload with none of the fields below.
      setStatus(d && d.available === false
        ? { in_progress: false, installed: false, queue: [], current: null }
        : d);
      setStatusLoaded(true);
    } catch { /* keep the last truthful status */ }
  };
  // Poll toutes les 10 s : avance la file côté serveur + maj de l'UI. Skipped
  // entirely while training is hidden (ai-toolkit not configured) — no point
  // hitting endpoints the backend doesn't expose in that state.
  useEffect(() => {
    setStatusLoaded(false);
    if (!caps.training_visible) return undefined;
    refreshStatus();
    const id = setInterval(refreshStatus, 10000);
    return () => clearInterval(id);
  }, [caps.training_visible]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    onNavigationStateChange?.({
      ready: !caps.training_visible || statusLoaded,
      queueCount: Array.isArray(status.queue) ? status.queue.length : 0,
    });
  }, [caps.training_visible, statusLoaded, status.queue, onNavigationStateChange]);

  useEffect(() => {
    if (navigationPanel === 'advanced') setAdvancedOpen(true);
    if (navigationPanel === 'checkpoints') setCheckpointsOpen(true);
  }, [navigationPanel]);

  const togglePanel = (panelId, current, setter) => (event) => {
    event.preventDefault();
    const next = !current;
    setter(next);
    onPanelOpenChange?.(panelId, next);
  };

  // Charge les bases + la base/variante du dataset au montage.
  useEffect(() => {
    if (!caps.training_visible) return undefined;
    let alive = true;
    ds.trainBaseInfo?.().then((info) => {
      if (alive && info) {
        setBaseInfo(info); setBase(info.base || '');
        // A persisted ABSOLUTE base is the « Custom weights… » path → reopen that mode.
        setCustomBase(looksAbsolute(info.base || ''));
        setVaePath(info.vae_path || '');
        setTePath(info.te_path || '');
        // Défaut family-aware : Krea sans variante persistée → Raw (reco officielle
        // « train on Raw, validate on Turbo ») ; FLUX.2 Klein → 4B (voie locale) —
        // y compris quand la variante PERSISTÉE vient d'une autre famille (un
        // dataset ex-Krea porte 'base', qui n'est pas une taille Klein valide) ;
        // les autres familles → Turbo.
        const fam = info.train_type || 'zimage';
        const v = info.variant
          || (fam === 'krea' ? 'base' : fam === 'flux2klein' ? '4b' : 'turbo');
        const safeVariant = normalizeCheckpointVariant(fam, v);
        setVariant(safeVariant);
        setTrainType(info.train_type || 'zimage');
        // Initialiser le navigateur une seule fois par dataset. Les refreshs de
        // base-info (conversion, réglages) ne doivent pas écraser son filtre.
        if (checkpointSelectionDataset.current !== ds.currentId) {
          checkpointSelectionDataset.current = ds.currentId;
          setCheckpointTrainType(fam);
          setCheckpointBase(info.base || '');
          setCheckpointVariant(safeVariant);
        }
        setAdv(info.train_settings || null);
        setSlider(info.slider || null);
      }
    });
    return () => { alive = false; };
  }, [ds.currentId, caps.training_visible]); // eslint-disable-line react-hooks/exhaustive-deps

  // Re-seed the slider text drafts from the canonical stored values whenever the
  // server state (re)loads — saves happen on blur, so no mid-typing overwrite.
  useEffect(() => {
    setSliderDraft({
      positive: slider?.positive ?? '',
      negative: slider?.negative ?? '',
      target_class: slider?.target_class ?? '',
      anchor: slider?.anchor ?? '',
    });
  }, [slider?.positive, slider?.negative, slider?.target_class, slider?.anchor]);

  // Pendant une conversion, poll le statut toutes les 4 s. Dépend de la fonction
  // STABLE (useCallback sur currentId), pas de l'objet `ds` entier — sinon
  // l'interval était recréé à chaque render et le timer 4 s n'aboutissait jamais.
  const getBaseInfo = ds.trainBaseInfo;
  useEffect(() => {
    if (!caps.training_visible || baseInfo?.convert?.status !== 'running') return undefined;
    const id = setInterval(async () => {
      const info = await getBaseInfo?.();
      if (info) setBaseInfo(info);
    }, 4000);
    return () => clearInterval(id);
  }, [baseInfo?.convert?.status, getBaseInfo, caps.training_visible]);

  // Bases selon le type choisi (zimage : officiel + merges ; sdxl : checkpoints ComfyUI).
  const currentBases = baseInfo?.bases_by_type?.[trainType] || baseInfo?.bases || [];
  // base_dir non configuré → les listers renvoient [] : distinguer « aucun modèle de
  // cette famille » de « ComfyUI pas encore pointé » (le vrai motif sur un clone neuf).
  // Défaut true tant que baseInfo n'est pas chargé, pour ne pas flasher la CTA au montage.
  const comfyConfigured = baseInfo?.comfyui_configured !== false;
  const isCustomBase = !!base;
  // « Custom weights… » (local-only) : familles qui l'exposent + celles honorant
  // VAE/TE (SDXL). base-info fait foi ; défauts avant chargement.
  const customFamilies = baseInfo?.custom_weights_families || DEFAULT_CUSTOM_FAMILIES;
  const customSupported = customFamilies.includes(trainType);
  const vaeTeFamilies = baseInfo?.vae_te_families || ['sdxl'];
  const vaeTeSupported = vaeTeFamilies.includes(trainType);
  // Mode custom actif mais chemin vide → rien à entraîner (bloque le bouton).
  const customWeightsEmpty = customBase && customSupported && !String(base).trim();
  // La conversion diffusers ne concerne QUE Z-Image (SDXL = single-file direct) ;
  // le mode « Custom weights… » (chemin absolu direct) ne convertit jamais.
  const needsConversion = trainType === 'zimage' && isCustomBase && !customBase;
  const baseConverted = needsConversion && !!(baseInfo?.converted?.[base]);
  const convertRunning = needsConversion && baseInfo?.convert?.status === 'running' && baseInfo?.convert?.z_model === base;
  const convertError = (needsConversion && baseInfo?.convert?.status === 'error' && baseInfo?.convert?.z_model === base)
    ? baseInfo.convert.error : null;
  // Bloque l'entraînement si la base custom Z-Image n'est pas encore convertie,
  // ou si SDXL sans base choisie (SDXL exige un checkpoint).
  const baseBlocksTrain = needsConversion && !baseConverted;
  const sdxlNeedsBase = trainType === 'sdxl' && !base;
  // Changement de type : réinitialise la base (les listes diffèrent ; SDXL → 1ère base réelle)
  // et PERSISTE la famille (choisie à la création, modifiable ici) pour que le menu
  // regroupé se ré-trie et que le format de caption suive.
  const onTypeChange = async (t) => {
    if (!t || t === trainType || trainTypeBusy) return;
    const previous = { trainType, base, variant, customBase };
    const nextVariant = defaultTrainingVariant(t);
    const list = baseInfo?.bases_by_type?.[t] || [];
    const nextBase = t === 'sdxl' ? (list[0]?.value || '') : '';
    setTrainTypeBusy(true);
    setPresetSel('');
    setStepsInfo(null);
    setTrainType(t);
    // Switching family leaves custom-weights mode (the path is arch-specific).
    setCustomBase(false);
    setBase(nextBase);
    // Z-Image → Turbo est le chemin sûr : base Turbo + adaptateur d'entraînement v2.
    // Cela empêche une variante Krea/Klein persistée de survivre en silence au switch.
    setVariant(nextVariant);
    let saved;
    try {
      saved = await ds.setDatasetTrainType?.(t);
    } catch {
      saved = { ok: false, error: 'Network error' };
    }
    if (saved?.ok === false) {
      setTrainType(previous.trainType);
      setBase(previous.base);
      setVariant(previous.variant);
      setCustomBase(previous.customBase);
      toastTrainError(saved, 'Could not change the training family');
      setTrainTypeBusy(false);
      return;
    }
    try {
      // Family persistence changes base defaults and effective advanced values.
      // Wait for both before re-enabling Apply so an old-family preset/settings
      // cannot race the new selection.
      const info = await ds.trainBaseInfo?.();
      if (info) {
        setBaseInfo(info);
        setAdv(info.train_settings || null);
      }
      const checkpointData = await ds.listCheckpoints?.(nextBase, t, nextVariant);
      setStepsInfo(checkpointData?.recommended_steps_info || null);
    } catch {
      // Persistence succeeded: keep the new family truthful and let the normal
      // effects retry base/steps instead of rolling the UI back to stale state.
      toast.warning('Training family saved, but its base/step details could not be refreshed yet.');
    } finally {
      setTrainTypeBusy(false);
    }
  };

  // Réglages avancés effectifs (client-side pour que le défaut family-aware du rank
  // suive un changement de type SANS re-fetch). `adv.rank` null = Auto.
  const advRankChoice = adv?.rank ?? 'auto';
  const advDefaultRank = adv?.default_rank
    ?? ((trainType === 'zimage' || trainType === 'flux' || trainType === 'flux2klein') ? 16 : 32);
  const advEffRank = advRankChoice === 'auto' ? advDefaultRank : advRankChoice;
  // Expert levers (all default to current behaviour when absent):
  const advAlphaChoice = adv?.alpha_setting ?? 'auto';
  const advDefaultAlpha = adv?.default_alpha ?? (trainType === 'sdxl' ? Math.max(1, Math.floor(advEffRank / 2)) : advEffRank);
  const advEffAlpha = advAlphaChoice !== 'auto' ? advAlphaChoice : advDefaultAlpha;
  const advAlphaChoices = adv?.alpha_choices ?? [1, 2, 4, 8, 16, 24, 32, 48, 64];
  const advDropout = adv?.dropout ?? 0;
  const advDropoutChoices = adv?.dropout_choices ?? [0.05, 0.1, 0.15, 0.2, 0.3];
  const advTimestep = adv?.timestep_type ?? 'auto';
  const advTimestepDefault = adv?.default_timestep_type ?? (trainType === 'krea' ? 'linear' : trainType === 'flux2klein' ? 'weighted' : (trainType === 'zimage' || trainType === 'flux') ? 'sigmoid' : null);   // miroir de _DEFAULT_TIMESTEP
  const advTimestepSupported = adv ? adv.timestep_type_supported !== false : trainType !== 'sdxl';
  const advTimestepChoices = adv?.timestep_type_choices ?? ['sigmoid', 'linear', 'weighted', 'shift'];
  const advOptimizer = adv?.optimizer ?? 'adamw8bit';
  const advOptimizerChoices = adv?.optimizer_choices ?? ['adamw8bit', 'adafactor', 'automagic', 'prodigy'];
  const advLrSched = adv?.lr_scheduler ?? 'constant';
  const advLrSchedChoices = adv?.lr_scheduler_choices ?? ['constant', 'linear', 'cosine', 'cosine_with_restarts', 'constant_with_warmup'];
  const advWarmup = adv?.warmup ?? 0;
  const advWarmupChoices = adv?.warmup_choices ?? [50, 100, 200, 500];
  const advGradAccum = adv?.grad_accum ?? 1;
  const advGradAccumChoices = adv?.grad_accum_choices ?? [1, 2, 4];
  // Recipe levers — network variant (LoKr) + EMA. LoKr is arch-generic in ai-toolkit,
  // so network_type_supported is always true today; the flag mirrors the timestep
  // pattern so a future family could be gated with one server-side flip.
  const advNetworkType = adv?.network_type ?? 'lora';
  const advNetworkChoices = adv?.network_type_choices ?? ['lora', 'lokr'];
  const advNetworkSupported = adv ? adv.network_type_supported !== false : true;
  const advEma = adv?.ema ?? 0;
  const advEmaChoices = adv?.ema_choices ?? [0.99, 0.999];
  const advDualCaptions = Boolean(adv?.dual_captions);
  const LR_SCHED_LABELS = { constant: 'Constant (default)', constant_with_warmup: 'Warmup → constant', linear: 'Linear decay', cosine: 'Cosine decay', cosine_with_restarts: 'Cosine + restarts' };
  // The resolution the next run will actually train at. Slider mode defaults to
  // 768 only (the slider loss makes several prediction passes per step — much
  // higher VRAM peak; multi-scale 768+1024 OOMs on 24 GB) unless the user picked
  // one explicitly. Reflected in both the control and the panel summary so they
  // never claim 768+1024 for a run that will emit 768.
  const advResStored = adv?.resolution ?? '768,1024';
  const advRes = (!!slider?.enabled && !adv?.resolution_explicit) ? '768' : advResStored;
  const advResLabel = { '768': '768px', '1024': '1024px', '768,1024': '768+1024px' }[advRes] || advRes;
  const advSave = adv?.save_every ?? 250;
  const advSampleEvery = adv?.sample_every ?? 250;
  const advSampleEveryChoices = adv?.sample_every_choices ?? [100, 250, 500, 1000];
  const advSampleDefault = adv?.sample_prompts_default ?? [];
  const advMaxPrompts = adv?.max_sample_prompts ?? 8;
  const saveAdv = async (patch) => {
    const eff = await ds.setTrainSettings?.(patch);
    if (eff) setAdv(eff);
  };
  // Seed / re-sync the preview-prompts textarea from the stored value whenever
  // base-info (re)loads. Save is on blur, so the user is never mid-typing here.
  useEffect(() => {
    setSamplePromptsText((adv?.sample_prompts ?? []).join('\n'));
  }, [adv?.sample_prompts]);
  const saveSamplePrompts = () => {
    const stored = (adv?.sample_prompts ?? []).join('\n');
    if (samplePromptsText === stored) return;      // no-op → skip the round-trip
    saveAdv({ sample_prompts: samplePromptsText }); // server splits on newlines + trims
  };

  // --- Slider LoRA mode (Beta) ------------------------------------------------
  const sliderOn = !!slider?.enabled;
  const sliderPromptsMissing = sliderOn
    && (!(slider?.positive || '').trim() || !(slider?.negative || '').trim());
  const saveSlider = async (patch) => {
    setSliderBusy(true);
    try {
      const d = await postTrain(`/api/dataset/${ds.currentId}/train/slider`, patch);
      if (d.ok === false) { toastTrainError(d, 'Slider settings save failed'); return null; }
      setSlider(d.slider);
      return d.slider;
    } finally {
      setSliderBusy(false);
    }
  };
  const toggleSliderMode = async () => {
    const next = !sliderOn;
    const saved = await saveSlider({ enabled: next });
    if (!saved) return;
    // Rank default (8 in slider mode) and the step policy both live server-side —
    // refresh base-info + the checkpoint/steps panel so labels stay truthful.
    try {
      const info = await ds.trainBaseInfo?.();
      if (info) { setBaseInfo(info); setAdv(info.train_settings || null); }
      const checkpointData = await ds.listCheckpoints?.(base, trainType, variant);
      if (checkpointData) setStepsInfo(checkpointData.recommended_steps_info || null);
    } catch { /* labels refresh is best-effort */ }
  };
  const saveSliderField = (key) => () => {
    const stored = slider?.[key] ?? '';
    if ((sliderDraft[key] ?? '') === stored) return;   // no-op → skip round-trip
    saveSlider({ [key]: sliderDraft[key] });
  };

  // --- Presets (save / apply / import / export / delete) ---------------------
  const presetContext = { trainType, datasetKind: kind, variant };
  const loadPresets = async (preferredSelection) => {
    try {
      const r = await fetch('/api/train/presets', { credentials: 'include' });
      if (r.ok) {
        const list = (await r.json()).presets || [];
        setPresets(list);
        setPresetSel((current) => compatibleTrainingPresetSelection(
          preferredSelection === undefined ? current : preferredSelection,
          list,
          presetContext,
        ));
        return list;
      }
    } catch { /* list is best-effort */ }
    return [];
  };
  useEffect(() => { loadPresets(); }, []);
  const visiblePresets = filterTrainingPresets(presets, presetContext);
  const selPreset = visiblePresets.find((p) => String(p.id) === presetSel) || null;
  useEffect(() => {
    setPresetSel((current) => compatibleTrainingPresetSelection(
      current, presets, { trainType, datasetKind: kind, variant },
    ));
  }, [presets, trainType, kind, variant]);
  const savePreset = async () => {
    const name = window.prompt('Preset name (an existing name is overwritten):');
    if (!name || !name.trim()) return;
    setPresetBusy(true);
    try {
      const d = await postTrain('/api/train/presets',
        { name: name.trim(), dataset_id: ds.currentId,
          ...trainingPresetSnapshotScope(presetContext) });
      if (d.ok === false) return toastTrainError(d, 'Preset save failed');
      toast.success(`Preset “${name.trim()}” saved.`);
      await loadPresets(d.id);
    } finally {
      setPresetBusy(false);
    }
  };
  const applyPreset = async () => {
    if (!selPreset || presetBusy || trainTypeBusy) return;
    // Every preset — built-in or user-created — is resolved by id on the server.
    // A null plan means the selection became incompatible between render/click;
    // importantly, no request is sent in that case.
    const payload = trainingPresetApplyPayload(selPreset, presetContext);
    if (!payload) {
      setPresetSel('');
      toast.error('This preset does not match the current model family or dataset kind.');
      return;
    }
    setPresetBusy(true);
    try {
      const d = await postTrain(`/api/dataset/${ds.currentId}/train/presets/apply`, payload);
      if (d.ok === false) return toastTrainError(d, 'Preset apply failed');
      setAdv(d.train_settings);
      if (payload.variant && payload.variant !== variant) setVariant(payload.variant);
      // Quick Style recipes own their researched step policy. Do not let a
      // temporary cap from a previous run silently override that policy.
      if (selPreset.builtin && trainingPresetDatasetKind(selPreset) === 'style') {
        setStepsOverride('');
      }
      const checkpointData = await ds.listCheckpoints?.(base, trainType, payload.variant);
      setStepsInfo(checkpointData?.recommended_steps_info || null);
      const notes = [];
      if (d.ignored?.length) notes.push(`unknown here, ignored: ${d.ignored.join(', ')}`);
      if (d.rejected?.length) notes.push(`rejected: ${d.rejected.map((r) => r.key).join(', ')}`);
      if (notes.length) toast.warning(`Preset applied — ${notes.join(' · ')}`);
      else toast.success(`Preset “${selPreset.name}” applied.`);
    } finally {
      setPresetBusy(false);
    }
  };
  const exportPreset = () => {
    if (!selPreset) return;
    const blob = new Blob([JSON.stringify({
      app: 'lora-dataset-studio', kind: 'training-preset', version: 1,
      name: selPreset.name, train_type: selPreset.train_type,
      dataset_kind: trainingPresetDatasetKind(selPreset) || kind,
      variants: Array.isArray(selPreset.variants) ? selPreset.variants : [],
      settings: selPreset.settings,
    }, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `lds-training-preset-${selPreset.name.replace(/[^\w.-]+/g, '_')}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  const importPreset = async (file) => {
    try {
      const j = JSON.parse(await file.text());
      if (j?.kind !== 'training-preset' || !j.name || typeof j.settings !== 'object' || !j.settings) {
        toast.error('Not a training-preset file (expected kind: "training-preset").');
        return;
      }
      setPresetBusy(true);
      const importedMeta = {
        ...j,
        train_type: j.train_type || trainType,
        dataset_kind: j.dataset_kind || kind,
      };
      const compatibleHere = filterTrainingPresets([importedMeta], presetContext).length === 1;
      const d = await postTrain('/api/train/presets',
        { name: String(j.name), train_type: importedMeta.train_type,
          dataset_kind: importedMeta.dataset_kind,
          variants: Array.isArray(importedMeta.variants) ? importedMeta.variants : [],
          settings: j.settings });
      if (d.ok === false) return toastTrainError(d, 'Preset import failed');
      await loadPresets(compatibleHere ? d.id : '');
      if (compatibleHere) toast.success(`Preset “${j.name}” imported and selected — review, then Apply.`);
      else toast.warning(`Preset “${j.name}” imported for ${importedMeta.train_type}/${importedMeta.dataset_kind}; it is hidden here because the current dataset is ${trainType}/${kind}.`);
    } catch {
      toast.error('Unreadable preset file.');
    } finally {
      setPresetBusy(false);
    }
  };
  const deletePreset = async () => {
    if (!selPreset || selPreset.builtin) return;   // built-ins ship with the app
    if (!window.confirm(`Delete the preset “${selPreset.name}”?`)) return;
    setPresetBusy(true);
    try {
      const r = await fetch(`/api/train/presets/${selPreset.id}`, {
        method: 'DELETE', headers: { 'X-CSRFToken': getCsrfToken() }, credentials: 'include',
      });
      if (!r.ok) toast.error('Could not delete the preset.');
      setPresetSel('');
      await loadPresets('');
    } catch { toast.error('Could not delete the preset.'); }
    finally { setPresetBusy(false); }
  };

  // Normalizes like useDataset's own postJson: a non-2xx response (e.g. the
  // 409 {'error','hint'} the training routes return when ai-toolkit isn't
  // configured, or a 400 for a refused enqueue) must surface as `ok: false`
  // — previously this just returned the raw body, so callers checking
  // `d.ok === false` never saw the error (d.ok stayed undefined) and it was
  // silently dropped instead of reaching the confirm/toast below.
  const postTrain = async (path, body) => {
    try {
      const r = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCsrfToken() },
        credentials: 'include',
        body: body ? JSON.stringify(body) : undefined,
      });
      let d = null;
      try { d = await r.json(); } catch { /* non-JSON body */ }
      if (!r.ok) return { ok: false, error: (d && d.error) || `Server error (${r.status})`, hint: d && d.hint };
      return d || { ok: true };
    } catch { return { ok: false, error: 'Network error' }; }
  };
  // 409 {'error','hint'} (or any other refusal) → toast, hint appended when present.
  const toastTrainError = (d, fallback) => {
    const msg = (d && d.error) || fallback;
    toast.error(d && d.hint ? `${msg} — ${d.hint}` : msg);
  };
  // Confirmable launch refusals: the server prefixes the error with a marker;
  // the window.confirm IS the user's answer, the retry carries the matching
  // force flag. Both can fire in sequence (uncaptioned first, then mismatch) —
  // call sites loop until launched, declined, or a non-confirmable error.
  const CONFIRMABLE_REFUSALS = [
    ['MISMATCH_CAPTION: ', 'allow_caption_mismatch'],
    ['UNCAPTIONED: ', 'allow_uncaptioned'],
    ['CAPTION_QUALITY: ', 'allow_caption_quality'],
    // Custom-weights arch sniff couldn't positively verify the file → the
    // window.confirm IS the answer, retry carries allow_unverified_weights.
    ['CUSTOM_WEIGHTS_UNVERIFIED: ', 'allow_unverified_weights'],
  ];
  const confirmableRetryFlag = (error, actionLabel) => {
    const s = String(error || '');
    for (const [marker, flag] of CONFIRMABLE_REFUSALS) {
      if (s.includes(marker)) {
        return window.confirm(s.replace(marker, '') + `\n\n${actionLabel}?`) ? flag : 'declined';
      }
    }
    return null;
  };

  // Pre-launch sanity gate (server preflight): blockers stop with a toast,
  // warnings open the interactive PreflightModal (lists WHICH captions leak /
  // WHICH pairs duplicate, editable/rejectable in place) and await the user's
  // Start-anyway / Cancel. Unreachable preflight never blocks.
  const [preflightReport, setPreflightReport] = useState(null);
  const preflightResolver = useRef(null);
  const resolvePreflight = (ok) => {
    setPreflightReport(null);
    preflightResolver.current?.(ok);
    preflightResolver.current = null;
  };
  const preflightOk = async () => {
    try {
      const r = await fetch(
        `/api/dataset/${ds.currentId}/train/preflight?train_type=${encodeURIComponent(trainType)}&variant=${encodeURIComponent(variant)}`,
        { credentials: 'include' });
      if (!r.ok) return true;
      const d = await r.json();
      if (d.blockers?.length) { toast.error(d.blockers.join('\n')); return false; }
      if (d.warnings?.length) {
        return await new Promise((resolve) => {
          preflightResolver.current = resolve;
          setPreflightReport(d);
        });
      }
      return true;
    } catch { return true; }
  };
  // Des checkpoints existent déjà → cliquer Train demande Resume ou Fresh :
  // ai-toolkit REPREND silencieusement le dernier checkpoint du run (les images
  // supprimées du dataset restent apprises dans ses poids) — après un remaniement
  // du dataset, l'utilisateur veut presque toujours repartir de zéro. Le choix
  // résout une promesse : 'fresh' | 'resume' | null (annuler).
  const [resumeAsk, setResumeAsk] = useState(null);   // {latest, final} | null
  const resumeResolver = useRef(null);
  const resolveResume = (v) => {
    setResumeAsk(null);
    resumeResolver.current?.(v);
    resumeResolver.current = null;
  };
  // ▶ Continue dialog (flexible resume): pick extra steps, the checkpoint to
  // resume from, and the safe settings to adjust. Open on the checkpoint list's
  // Continue button; resolves to a payload or null (cancel).
  const [continueOpen, setContinueOpen] = useState(false);
  const runContinue = async (payload) => {
    setContinueOpen(false);
    if (!payload) return;
    await runConfirmableTrainingRequest(
      (continueOpts) => ds.continueTraining(
        payload.extraSteps, checkpointBase, checkpointVariant, checkpointTrainType,
        { ...continueOpts, fromStep: payload.fromStep, overrides: payload.overrides }),
      { masked },
      (error) => confirmableRetryFlag(error, 'Continue anyway (force)'),
    );
    refreshStatus();
    loadCheckpoints(checkpointBase, checkpointTrainType, checkpointVariant);
  };
  const askResumeOrFresh = async () => {
    // Ne pas lire la liste affichée dans le navigateur de résultats : elle peut
    // volontairement pointer vers une autre famille/base. Le backend fait foi
    // pour la configuration d'entraînement actuellement sélectionnée.
    let existing = [];
    try {
      const data = await ds.listCheckpoints(base, trainType, variant);
      existing = Array.isArray(data?.checkpoints) ? data.checkpoints : [];
    } catch { /* le lancement normal garde le preflight serveur comme autorité */ }
    if (!existing.length) return 'resume';                     // pas de run → lancement normal
    const latest = Math.max(...existing.map((c) => c.step));
    const final = existing.some((c) => c.final);
    return new Promise((resolve) => {
      resumeResolver.current = resolve;
      setResumeAsk({ latest, final });
    });
  };

  // Masked training (fond 10 %) — défaut ON, persisté (partagé lancement/file/programmation).
  const [masked, setMaskedS] = useState(() => {
    try { return localStorage.getItem('trainMasked_v1') !== '0'; } catch { return true; }
  });
  const setMasked = (v) => {
    setMaskedS(v);
    try { localStorage.setItem('trainMasked_v1', v ? '1' : '0'); } catch { /* ignore */ }
  };
  // Concept/style : masked OFF. A person mask erases a concept and prevents an
  // always-on style from learning the full frame/background. Do not overwrite
  // the user's character preference while applying that safety default.
  useEffect(() => {
    if (isConceptual) setMaskedS(false);
    else { try { setMaskedS(localStorage.getItem('trainMasked_v1') !== '0'); } catch { setMaskedS(true); } }
  }, [ds.currentId, isConceptual]); // eslint-disable-line react-hooks/exhaustive-deps
  // Masked ON but rembg (person-mask backend) unavailable → the export silently
  // drops the masks and trains UNMASKED. Surface that instead of lying about it.
  // `=== false` (not `!caps.masks`) so we don't warn before caps have loaded.
  const maskedRembgMissing = masked && !isConceptual && !sliderOn && caps.masks === false;
  // Plafond de steps CHOISI (vide → adaptatif). NON persisté à dessein : un cap
  // oublié (ex. 2000) ne doit pas s'appliquer en douce au prochain dataset.
  const [stepsOverride, setStepsOverride] = useState('');
  // Cible envoyée au backend (Train / Add to queue / Schedule) : null = adaptatif ;
  // sinon plancher à 500 (le backend re-clampe pareil). Non numérique → 500.
  const stepsN = stepsOverride.trim()
    ? Math.max(500, parseInt(stepsOverride, 10) || 500)
    : null;

  const enqueue = async () => {
    if (!(await preflightOk())) return;
    // Mise en file AVEC la base/variante choisie (sinon le job reprend la base persistée).
    let body = { base_model: base, variant, train_type: trainType, masked, steps: stepsN,
                 ...(trainType === 'sdxl' ? { vae_path: vaePath, te_path: tePath } : {}) };
    let d = await postTrain(`/api/dataset/${ds.currentId}/train/enqueue`, body);
    for (let flag; d && d.ok === false && (flag = confirmableRetryFlag(d.error, 'Queue anyway (force)')); ) {
      if (flag === 'declined') { d = null; break; }  // the confirm WAS the answer
      body = { ...body, [flag]: true };
      d = await postTrain(`/api/dataset/${ds.currentId}/train/enqueue`, body);
    }
    if (d && d.ok === false) { setEnqErr(d.error || 'enqueue refused'); toastTrainError(d, 'enqueue refused'); }
    else setEnqErr(null);
    refreshStatus();
  };
  const dequeue = async (id) => {
    const d = await postTrain(`/api/dataset/${id}/train/dequeue`);
    if (d && d.ok === false) toastTrainError(d, 'dequeue failed');
    refreshStatus();
  };
  const queued = (status.queue || []).some((q) => q.dataset_id === ds.currentId);

  // --- Entraînement PROGRAMMÉ (jour + heure) : entre en file avec une échéance ;
  // à l'heure dite le ticker serveur le lance, ou le met en attente si un autre
  // entraînement occupe déjà le GPU (jamais d'erreur). ---
  const [showSched, setShowSched] = useState(false);
  const [schedAt, setSchedAt] = useState('');
  const openSched = () => {
    if (!schedAt) {
      // Défaut : dans 1 h, arrondi au quart d'heure (format datetime-local, heure locale).
      const t = new Date(Date.now() + 3600e3);
      t.setMinutes(Math.ceil(t.getMinutes() / 15) * 15, 0, 0);
      const p = (n) => String(n).padStart(2, '0');
      setSchedAt(`${t.getFullYear()}-${p(t.getMonth() + 1)}-${p(t.getDate())}T${p(t.getHours())}:${p(t.getMinutes())}`);
    }
    setShowSched((v) => !v);
  };
  const schedule = async () => {
    if (!schedAt) return;
    if (!(await preflightOk())) return;
    let body = { at: schedAt, base_model: base, variant, train_type: trainType, masked, steps: stepsN,
                 ...(trainType === 'sdxl' ? { vae_path: vaePath, te_path: tePath } : {}) };
    let d = await postTrain(`/api/dataset/${ds.currentId}/train/schedule`, body);
    for (let flag; d && d.ok === false && (flag = confirmableRetryFlag(d.error, 'Schedule anyway (force)')); ) {
      if (flag === 'declined') { d = null; break; }  // the confirm WAS the answer
      body = { ...body, [flag]: true };
      d = await postTrain(`/api/dataset/${ds.currentId}/train/schedule`, body);
    }
    if (d && d.ok === false) { setEnqErr(d.error || 'schedule refused'); toastTrainError(d, 'schedule refused'); }
    else { setEnqErr(null); setShowSched(false); }
    refreshStatus();
  };

  // Les checkpoints sont propres au filtre du NAVIGATEUR DE RÉSULTATS
  // (un run = dataset+famille+base+variante), indépendant du prochain entraînement.
  // Garde-fou : si appelé avec autre chose qu'une string (ex. onClick passe un
  // event), on retombe sur `base` au lieu d'envoyer [object Object] à l'API.
  const loadCheckpoints = async (forBase, forType, forVariant) => {
    const b = (typeof forBase === 'string') ? forBase : checkpointBase;
    const t = (typeof forType === 'string') ? forType : checkpointTrainType;
    const v = (typeof forVariant === 'string') ? forVariant : checkpointVariant;
    const requestId = ++checkpointRequest.current;
    const data = await ds.listCheckpoints(b, t, v);
    if (requestId !== checkpointRequest.current) return;
    setCheckpoints(data.checkpoints || []);
    setImported(data.imported || []);
    // Provenance : dernière version enregistrée du dataset vs son état ACTUEL
    // (alerte « le dataset a changé depuis vN » + numéro de la prochaine version).
    setDatasetState(data.dataset_state || null);
    setCloudCkpts(data.cloud_checkpoints || []);
    // Prefer the per-run grouped payload; fall back to grouping the flat list
    // by run_id so an older server still renders (single group per run).
    setCloudGroups(cloudGroupsFrom(data));
    setDiskUsage(data.disk_usage || null);
    setCkLoaded(true);
    onCheckpointsChange?.(
      (Array.isArray(data.checkpoints) ? data.checkpoints.length : 0)
      + (Array.isArray(data.imported) ? data.imported.length : 0),
    );
  };
  // Recharge dès que le filtre de résultats change. On
  // attend baseInfo pour charger directement la BONNE base persistée (pas de flash
  // « Officiel » avant que la base du dataset soit appliquée).
  useEffect(() => {
    if (!caps.training_visible || !ds.currentId || !baseInfo) return;
    setCkLoaded(false);
    loadCheckpoints(checkpointBase, checkpointTrainType, checkpointVariant);
  }, [checkpointBase, checkpointTrainType, checkpointVariant, ds.currentId, baseInfo, caps.training_visible]); // eslint-disable-line react-hooks/exhaustive-deps

  // Le barème affiché dans Training suit uniquement la configuration Training,
  // jamais le filtre indépendant du navigateur de résultats.
  useEffect(() => {
    if (!caps.training_visible || !ds.currentId || !baseInfo) return undefined;
    let alive = true;
    ds.listCheckpoints(base, trainType, variant).then((data) => {
      if (alive) setStepsInfo(data?.recommended_steps_info || null);
    }).catch(() => { /* keep the last truthful rationale */ });
    return () => { alive = false; };
  }, [base, trainType, variant, ds.currentId, baseInfo, caps.training_visible]); // eslint-disable-line react-hooks/exhaustive-deps
  const removeImported = async (filename, label) => {
    // Guard-rail: this LoRA may be the one the Studio's ★ best settings point to —
    // deleting it silently breaks the saved winning combo.
    const best = ds.data?.best_settings;
    const isBest = best?.lora_filename
      && String(best.lora_filename).split(/[\\/]/).pop() === String(filename).split(/[\\/]/).pop();
    const msg = isBest
      ? `⚠ « ${label} » is the LoRA saved as this dataset's ★ BEST SETTINGS in the Test Studio.\n\nDelete it anyway? The saved combo will stop working.`
      : `Permanently delete « ${label} » from ComfyUI's ${checkpointLorasLabel} folder?`;
    if (!window.confirm(msg)) return;
    await ds.deleteCheckpoint(filename, checkpointTrainType, checkpointVariant);
    loadCheckpoints();
  };
  const doPrepareBase = async () => {
    await ds.prepareBase(base);
    const info = await ds.trainBaseInfo();
    if (info) setBaseInfo(info);
  };

  // Best-epoch (jandordoe): score the run's samples vs the reference, recommend
  // the checkpoint closest to the best-scoring step. Result cleared on base change.
  const [bestEpoch, setBestEpoch] = useState(null);
  const [bestEpochBusy, setBestEpochBusy] = useState(false);
  useEffect(() => { setBestEpoch(null); }, [checkpointBase, checkpointTrainType, checkpointVariant, ds.currentId]);
  const findBestEpoch = async () => {
    setBestEpochBusy(true);
    try {
      const d = await postTrain(`/api/dataset/${ds.currentId}/train/best-epoch`,
        trainingRunSelection(checkpointBase, checkpointTrainType, checkpointVariant));
      if (d && d.ok === false) { toastTrainError(d, 'best-epoch scoring failed'); return; }
      setBestEpoch(d);
    } finally {
      setBestEpochBusy(false);
    }
  };

  // Steps are family + variant recipes owned by the backend. Never duplicate a
  // formula here: doing so previously showed a Z-Image estimate while a Krea or
  // Klein run used a different authoritative target.
  // Libellé lisible de la base sélectionnée (pour étiqueter les checkpoints de CE run).
  // Custom weights → basename du fichier (jamais le chemin complet dans le résumé).
  const baseLabel = customBase && base
    ? `custom: ${baseName(base)}`
    : (currentBases.find((b) => b.value === base)?.label || (base || 'Official'));
  const zimageRecipe = trainType === 'zimage'
    ? describeZImageRecipe({ variant, base, baseLabel, customBase })
    : null;
  const effectiveTargetSteps = stepsN ?? stepsInfo?.steps ?? null;
  const zimageTurboLongRun = trainType === 'zimage'
    && isLongZImageTurboRun({ variant, steps: effectiveTargetSteps });
  const typeLabel = trainFamilyLabel(trainType);
  const stepsRecipeType = stepsInfo?.train_type || trainType;
  const stepsRecipeFamily = stepsInfo?.family_label || trainFamilyLabel(stepsRecipeType);
  const stepsRecipeVariant = stepsInfo?.variant_label
    || checkpointVariantLabel(stepsRecipeType, stepsInfo?.variant || variant);
  const checkpointBasesRaw = baseInfo?.bases_by_type?.[checkpointTrainType] || baseInfo?.bases || [];
  const checkpointBaseOptions = checkpointBase && !checkpointBasesRaw.some((item) => item.value === checkpointBase)
    ? [{ value: checkpointBase, label: `custom: ${baseName(checkpointBase)}` }, ...checkpointBasesRaw]
    : checkpointBasesRaw;
  const checkpointBaseLabel = checkpointBaseOptions.find((item) => item.value === checkpointBase)?.label
    || (checkpointBase ? baseName(checkpointBase) : 'Official');
  const checkpointTypeLabel = trainFamilyLabel(checkpointTrainType);
  const checkpointVariants = checkpointVariantOptions(checkpointTrainType);
  const checkpointVariantDisplay = checkpointVariantLabel(checkpointTrainType, checkpointVariant);
  const checkpointLorasLabel = loraFolderLabel(checkpointTrainType);
  const checkpointMatchesTraining = checkpointSelectionMatchesTraining(
    checkpointTrainType, checkpointBase, checkpointVariant,
    trainType, base, variant);
  const onCheckpointTypeChange = (nextType) => {
    const choices = baseInfo?.bases_by_type?.[nextType] || [];
    setCheckpointTrainType(nextType);
    setCheckpointBase(defaultCheckpointBase(choices));
    setCheckpointVariant(defaultCheckpointVariant(nextType));
  };

  // Panel gated off (ai-toolkit not configured): the workspace's checkpoint
  // count must not keep a stale value from a previous dataset/session.
  useEffect(() => {
    if (!caps.training_visible) onCheckpointsChange?.(0);
  }, [caps.training_visible]); // eslint-disable-line react-hooks/exhaustive-deps

  // Cloud run status (global — several cloud runs may be active at once,
  // across different datasets, up to cloudStatus.limit). Polled independently
  // of the local `status` poll above, and only while a vast.ai key is
  // actually configured.
  const [cloudStatus, setCloudStatus] = useState({
    configured: false, limit: 1, actives: [], active: null, total_price_per_hour: 0, last: null,
  });
  useEffect(() => {
    if (!caps.cloud_training) return undefined;
    let alive = true;
    let t;
    const tick = async () => {
      try {
        const r = await fetch('/api/dataset/train/cloud/status', { credentials: 'include' });
        if (r.ok && alive) setCloudStatus(await r.json());
      } catch { /* transient */ }
      if (alive) t = setTimeout(tick, 5000);
    };
    t = setTimeout(tick, 0);
    return () => { alive = false; clearTimeout(t); };
  }, [caps.cloud_training]);
  // Compat: older servers (or a stale poll) may still answer with only the
  // single `active` field — fall back to a 1-element list built from it.
  const actives = cloudStatus.actives || (cloudStatus.active ? [cloudStatus.active] : []);
  // Per-(dataset, family): switching the LoRA-type selector shows THAT
  // family's run. A run without train_type (older server payload) matches
  // any family, preserving the previous behavior.
  const cloudActiveHere = actives.find((a) => a.dataset_id === ds.currentId
    && (!a.train_type || a.train_type === trainType));
  // Multi-family parallelism is safe again: each cloud run's monitor builds its
  // job config from its OWN stamped family/variant, not the shared dataset row
  // (backend _run_config_dataset — fix for the 2026-07-14 incident). So a Krea
  // run and a Z-Image run may train the same dataset at once; the button is
  // blocked only when a run of the SAME family is already active here.
  // Single source of truth for WHY « ☁ Train in cloud » is disabled — most
  // fundamental cause first (family unsupported > custom weights > too few
  // images > a run already active here > global limit). Drives BOTH the tooltip
  // AND the always-visible reason line below: a disabled button must state its
  // reason without a hover (the owner lost time guessing on a greyed SDXL button
  // whose only explanation lived in a title attribute).
  // Slider mode floors the image requirement at the substrate minimum (the
  // preflight/assert_trainable stay authoritative server-side).
  const trainMinFloor = sliderOn ? TRAIN_MIN_SLIDER[0] : (TRAIN_MIN[trainType]?.[0] ?? 12);
  const cloudTooFewImages = keptCount < trainMinFloor;
  const cloudLimitReached = actives.length >= (cloudStatus.limit || 1);
  const cloudDisabledReason =
    trainType === 'sdxl'
      ? 'SDXL trains locally only — the cloud lane covers Z-Image, Krea 2 and FLUX.2 Klein'
    : trainType === 'flux'
      ? 'FLUX.1 trains locally only — the cloud lane covers Z-Image, Krea 2 and FLUX.2 Klein'
    : (vaePath || tePath)
      ? 'Custom VAE/text-encoder overrides are local-only — clear them in Advanced options to train in the cloud'
    : customWeightsEmpty
      ? 'Enter the path to your custom weights .safetensors first'
    : baseBlocksTrain
      ? 'Convert the custom base first — the cloud lane pushes the converted copy to your Hugging Face account'
    : cloudTooFewImages
      ? `Only ${keptCount} image(s) kept — the cloud minimum for ${sliderOn ? 'a slider' : typeLabel} is ${trainMinFloor}`
    : cloudActiveHere
      ? `A ${typeLabel} cloud run is already active on this dataset`
    : cloudLimitReached
      ? `Cloud run limit reached (${actives.length}/${cloudStatus.limit || 1}) — stop one or raise the limit in Settings`
    : null;

  // Launch-time GPU speed picker: the ☁️ button opens a dialog that lists live
  // vast.ai offers by speed (price/h + approx time + cost); the chosen class is
  // forwarded as gpu_name. launchCloud carries the POST + the MISMATCH_CAPTION
  // retry that used to live inline in the button handler.
  const [cloudDialog, setCloudDialog] = useState(false);
  const launchCloud = async (gpuName) => {
    let body = cloudTrainingLaunchPayload({
      baseModel: base, variant, trainType, masked, steps: stepsN, gpuName,
    });
    let d = await postJson(`/api/dataset/${ds.currentId}/train/cloud`, body);
    for (let flag; d && d.ok === false && (flag = confirmableRetryFlag(d.error, 'Train anyway (force)')); ) {
      if (flag === 'declined') { d = null; break; }  // the confirm WAS the answer
      body = { ...body, [flag]: true };
      d = await postJson(`/api/dataset/${ds.currentId}/train/cloud`, body);
    }
    if (d && d.ok === false) {
      toastTrainError(d, 'Cloud training failed');
    }
    // Success needs no toast — the 5s cloud-status poll picks it up.
  };

  if (!caps.training_visible) {
    return (
      <div className="flex items-center gap-2 rounded-lg border border-border bg-surface p-3 text-content-muted text-sm">
        <span aria-hidden>🎓</span>
        Training needs ai-toolkit (local GPU) or a vast.ai API key (cloud) — set either in Settings.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-indigo-500/30 bg-indigo-500/5 p-3">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-content font-semibold text-sm"><span aria-hidden>🎓</span> LoRA Training ({typeLabel})</span>
        {!status.installed && (
          <span className="text-amber-300 text-[0.6875rem]">ai-toolkit not installed — run setup-aitoolkit.ps1</span>
        )}
        {status.in_progress
          ? <span className="ml-auto flex items-center gap-2">
              <span aria-live="polite" className="text-indigo-300 text-[0.6875rem]">
                <span aria-hidden>⏳</span> {status.current?.name ? `« ${status.current.name} » running` : 'running'} — ComfyUI paused
              </span>
              {/* Full progress bar, loss curve and samples live on the Runs hub —
                  this panel's own TrainingProgress only covers THIS dataset. */}
              <Link to="/cloud" title="Open the Runs page — full progress, loss curve and samples"
                className="px-1 py-0.5 text-indigo-300 hover:text-indigo-200 text-[0.6875rem] font-medium underline decoration-indigo-300/40">
                View in Runs ↗
              </Link>
            </span>
          : <span aria-live="polite" className="ml-auto text-content-subtle text-[0.6875rem]">{keptCount} image(s) kept</span>}
      </div>

      {/* A cloud run left its pod alive for manual recovery (any dataset) — it
          keeps billing until reaped, so this must stay visible regardless of
          which dataset's panel happens to be open. No action button: the
          recovery is manual (outside the app) and expiry-reaping is automatic. */}
      {cloudStatus.last?.status === 'error_pod_kept' && (
        <p className="m-0 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-1.5 text-amber-300 text-[0.6875rem]">
          ⚠ A previous cloud run kept its pod for manual recovery — it is still billing until reaped. {cloudStatus.last.error}
        </p>
      )}

      {/* Local training CRASHED (ai-toolkit run.py exited non-zero): the watcher
          captured the reason into training_error. Without surfacing it, a run that
          starts then dies just flips back to idle after the green "Training started"
          toast — the exact "shows confirmation but nothing happens" report (GH #3).
          Cleared automatically on the next launch (server resets training_error). */}
      {status.error && (!status.error.dataset_id || status.error.dataset_id === ds.currentId) && (
        <div className="rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-red-200 text-[0.6875rem]">
          <div className="font-semibold">
            ⚠ The last training run failed{status.error.rc != null ? ` (ai-toolkit exited ${status.error.rc})` : ''} — nothing is training now.
          </div>
          {status.error.log_tail && (
            <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded bg-black/30 p-1.5 font-mono text-[0.625rem] text-red-300/90">
              {status.error.log_tail}
            </pre>
          )}
          <div className="mt-1 text-red-300/80">
            Common first-run causes: ai-toolkit’s Python venv is missing packages
            (re-run its install), or the base model is still downloading / needs a
            Hugging Face token (gated models like Krea 2, FLUX.1 and FLUX.2 Klein). Fix the cause above, then Train again.
          </div>
        </div>
      )}

      {/* Live progress of THIS dataset's run: bar + loss sparkline + sample
          previews. Only while it is the one training (queued/other runs: no poll). */}
      {status.in_progress && status.current?.dataset_id === ds.currentId && (
        <TrainingProgress datasetId={ds.currentId}
          base={status.current?.base_model ?? base}
          trainType={status.current?.train_type || trainType}
          variant={status.current?.variant || variant} />
      )}

      {/* Cloud run progress + stop (this dataset only) — separate from the local
          poll above; runs entirely on the vast.ai pod. */}
      {cloudActiveHere && (
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2 text-[0.6875rem] text-sky-200 flex-wrap">
            <span aria-hidden>☁️</span>
            <span className="font-semibold">Cloud run — {cloudActiveHere.status}</span>
            {cloudActiveHere.gpu && <span>{cloudActiveHere.gpu}</span>}
            {cloudActiveHere.price_per_hour != null && (
              <span className="tabular-nums">${cloudActiveHere.price_per_hour}/h · ~${cloudActiveHere.cost_estimate} so far</span>
            )}
            {/* Full progress bar, loss curve and samples live on the Runs hub. */}
            <Link to="/cloud" title="Open the Runs page — full progress, loss curve and samples"
              className="ml-auto px-1 py-0.5 text-sky-300 hover:text-sky-200 font-medium underline decoration-sky-300/40">
              View in Runs ↗
            </Link>
            <button type="button" className="px-2 py-0.5 rounded bg-red-600/80 text-white text-[0.6875rem] font-semibold"
              onClick={async () => { await postJson('/api/dataset/train/cloud/stop', { run_id: cloudActiveHere.run_id }); }}>
              Stop cloud run
            </button>
          </div>
          <TrainingProgress datasetId={ds.currentId}
            base={cloudActiveHere.base_model ?? ''}
            trainType={cloudActiveHere.train_type || trainType}
            variant={cloudActiveHere.variant || variant} cloud />
        </div>
      )}
      {/* Download link only when the LAST run matches the selected family
          (a legacy payload without train_type matches any family). Keeping it
          keyed on cloudStatus.last stays simple — per-family history is
          served by ?train_type= on the checkpoint route itself. */}
      {caps.cloud_training && !cloudActiveHere && cloudStatus.last
        && cloudStatus.last.dataset_id === ds.currentId
        && (!cloudStatus.last.train_type || cloudStatus.last.train_type === trainType)
        && cloudStatus.last.checkpoint_ready && cloudStatus.last.status === 'done' && (
        <a href={`/api/dataset/${ds.currentId}/train/cloud/checkpoint?train_type=${encodeURIComponent(trainType)}`}
          className="text-sky-300 text-[0.6875rem] underline w-fit">
          ⬇ Download the cloud-trained LoRA (.safetensors)
        </a>
      )}

      {/* --- Chemin essentiel : choisir le type de LoRA et lancer. Le reste
           (base/variante, masked, plafond de steps, programmation) vit dans
           « Advanced options » ci-dessous — replié par défaut, tout y reste
           accessible en un clic. --- */}
      <div className="flex items-center gap-2 flex-wrap rounded-lg border border-border bg-surface px-3 py-2">
        <span className="text-content-muted text-[0.625rem] uppercase">LoRA type</span>
        <select value={trainType} onChange={(e) => onTypeChange(e.target.value)}
          disabled={trainTypeBusy || presetBusy}
          aria-label="Type of LoRA to train"
          title="Z-Image (prose, Qwen3 encoder) ~20 img · SDXL (ComfyUI checkpoints) ~30 img · Krea 2 (prose, base fixe Turbo) ~20 img · FLUX.1-dev (prose, gated HF, local-only) ~20 img · FLUX.2 Klein (prose, gated HF, 4B local / 9B cloud) ~20 img"
          className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem] disabled:opacity-50">
          <option value="zimage">Z-Image (~20 img)</option>
          <option value="sdxl">SDXL (~30 img)</option>
          <option value="krea">Krea 2 (~20 img)</option>
          <option value="flux">FLUX.1 (~20 img)</option>
          <option value="flux2klein">FLUX.2 Klein (~20 img)</option>
        </select>
        <button type="button" disabled={!status.installed || keptCount < trainMinFloor || status.in_progress || baseBlocksTrain || sdxlNeedsBase || customWeightsEmpty || sliderPromptsMissing}
          title={baseBlocksTrain ? 'Convert the custom base first'
            : customWeightsEmpty ? 'Enter the path to your custom weights .safetensors'
            : sdxlNeedsBase ? 'Choose a base SDXL checkpoint'
            : sliderPromptsMissing ? 'Slider mode needs both a positive and a negative prompt'
            : keptCount < trainMinFloor
              ? (sliderOn
                ? `${keptCount} kept image(s) — slider training still needs ${trainMinFloor}+ images as a denoising substrate`
                : `${keptCount} kept image(s) — the minimum for ${typeLabel} is ${trainMinFloor}`)
              : undefined}
          onClick={async () => {
            if (!(await preflightOk())) return;
            // Run existant → Resume (continue le LoRA) ou Fresh (archive le run,
            // repart de zéro). Le mismatch-retry re-passe fresh : le 1er appel a
            // échoué AVANT l'archivage (assert_trainable), rien n'a été écarté.
            const mode = await askResumeOrFresh();
            if (!mode) return;
            const fresh = mode === 'fresh';
            // ds.train takes camelCase opts — map the confirmable force flags.
            const OPT_FOR_FLAG = { allow_caption_mismatch: 'allowCaptionMismatch',
                                   allow_uncaptioned: 'allowUncaptioned',
                                   allow_caption_quality: 'allowCaptionQuality',
                                   allow_unverified_weights: 'allowUnverifiedWeights' };
            let opts = { baseModel: base, variant, trainType, masked, steps: stepsN, fresh,
                         vaePath, tePath };
            let d = await ds.train(opts);
            for (let flag; d && d.ok === false && (flag = confirmableRetryFlag(d.error, 'Train anyway (force)')); ) {
              if (flag === 'declined') break;        // the confirm WAS the answer
              opts = { ...opts, [OPT_FOR_FLAG[flag]]: true };
              d = await ds.train(opts);
            }
            refreshStatus();
          }}
          className="px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
          <span aria-hidden>🚀</span> Train the LoRA
        </button>
        <HelpBadge topic="action-training-launch" />
        {caps.cloud_training && (
          <button type="button"
            disabled={!!cloudDisabledReason}
            title={cloudDisabledReason
              || 'Rents a vast.ai GPU for this run (~$1-2), auto-terminated'}
            onClick={() => setCloudDialog(true)}
            className="px-3 py-1.5 rounded-lg border border-sky-500/50 bg-sky-500/10 text-sky-200 text-sm font-semibold disabled:opacity-40">
            <span aria-hidden>☁️</span> Train in cloud
          </button>
        )}
        {status.in_progress && (
          <button type="button" onClick={async () => { await ds.stopTraining(); refreshStatus(); }}
            className="px-3 py-1.5 rounded-lg bg-red-600/80 text-white text-sm font-semibold">
            Finish / re-enable ComfyUI
          </button>
        )}
        {status.in_progress && status.installed && keptCount >= trainMinFloor && !sliderPromptsMissing && (
          <button type="button" disabled={queued || baseBlocksTrain} onClick={enqueue}
            title={baseBlocksTrain
              ? 'Convert the selected custom base first'
              : `Train THIS dataset on « ${baseLabel} » once the current training finishes`}
            className="px-3 py-1.5 rounded-lg bg-indigo-500/20 border border-indigo-400/40 text-indigo-200 text-sm font-semibold disabled:opacity-40">
            {queued ? '✓ Queued' : `➕ Add to queue (${baseLabel})`}
          </button>
        )}
        {/* Résumé lisible de la config que le prochain run utilisera — les
            réglages eux-mêmes vivent dans « Advanced options ». */}
        <span className="ml-auto text-content-subtle text-[0.625rem]"
          title="The configuration the next run will use — change it in Advanced options below">
          {sliderOn ? '🎚 slider (Beta) · ' : ''}base « {zimageRecipe?.baseLabel || baseLabel} »{zimageRecipe ? ` · ${zimageRecipe.adapterActive ? 'Turbo adapter v2 ON' : 'no training adapter'}` : ''} · {sliderOn ? 'unmasked (slider)' : maskedRembgMissing ? 'unmasked (rembg missing)' : masked ? 'masked' : 'unmasked'} · {advResLabel} · {stepsOverride.trim() ? `${stepsN} steps` : sliderOn ? `${stepsInfo?.steps ?? 1000} steps (slider policy)` : 'adaptive steps'}{advNetworkType === 'lokr' ? ' · LoKr' : ''}{advEma ? ` · EMA ${advEma}` : ''}
        </span>
      </div>

      {/* --- Slider LoRA (Beta) : entraîne un LoRA BIPOLAIRE (±strength) depuis une
           paire de prompts via le trainer `concept_slider` d'ai-toolkit. Les images
           du dataset ne servent que de substrat de débruitage (captions ignorées).
           Feature expérimentale assumée — le badge Beta et les notes par famille
           disent exactement ce qui est prouvé et ce qui ne l'est pas. --- */}
      <div id="ds-training-slider" className={`rounded-lg border px-3 py-2 flex flex-col gap-2 ${
        sliderOn ? 'border-purple-400/50 bg-purple-500/5' : 'border-border bg-surface'}`}>
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-sm font-semibold text-content"><span aria-hidden>🎚</span> Slider LoRA</span>
          <span className="px-1.5 py-0.5 rounded border border-amber-400/50 bg-amber-500/10 text-amber-300 text-[0.625rem] font-semibold uppercase tracking-wide">Beta</span>
          <button type="button" role="switch" aria-checked={sliderOn}
            disabled={sliderBusy || status.in_progress}
            onClick={toggleSliderMode}
            title={status.in_progress ? 'Wait for the current training to finish'
              : sliderOn ? 'Turn slider mode off — back to normal LoRA training'
                : 'Turn slider mode on for this dataset'}
            className={`ml-auto px-2.5 py-1 rounded-lg border text-[0.75rem] font-semibold transition-colors disabled:opacity-50 ${
              sliderOn ? 'border-purple-400/60 bg-purple-500/20 text-purple-200'
                : 'border-border bg-surface text-content-muted'}`}>
            {sliderOn ? 'ON' : 'OFF'}
          </button>
        </div>
        <p className="m-0 text-content-subtle text-[0.6875rem]">
          Trains a <b>bipolar concept slider</b> from a prompt pair (no captions, no masks —
          the kept images are only a denoising substrate). Test it at negative and positive
          strengths in the Test Studio. Experimental: expect to iterate.
        </p>
        {sliderOn && (
          <>
            <div className="grid gap-2 sm:grid-cols-2">
              <label className="flex flex-col gap-1">
                <span className="text-content-muted text-[0.625rem] uppercase">Positive prompt *</span>
                <input type="text" value={sliderDraft.positive}
                  onChange={(e) => setSliderDraft((d) => ({ ...d, positive: e.target.value }))}
                  onBlur={saveSliderField('positive')}
                  placeholder="e.g. very muscular body, defined muscles"
                  title="What +strength amplifies (and −strength removes). Describe the EXTREME of the trait."
                  className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]" />
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-content-muted text-[0.625rem] uppercase">Negative prompt *</span>
                <input type="text" value={sliderDraft.negative}
                  onChange={(e) => setSliderDraft((d) => ({ ...d, negative: e.target.value }))}
                  onBlur={saveSliderField('negative')}
                  placeholder="e.g. skinny, frail body, thin arms"
                  title="The polar opposite of the positive prompt — what −strength amplifies."
                  className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]" />
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-content-muted text-[0.625rem] uppercase">Target class</span>
                <input type="text" value={sliderDraft.target_class}
                  onChange={(e) => setSliderDraft((d) => ({ ...d, target_class: e.target.value }))}
                  onBlur={saveSliderField('target_class')}
                  placeholder="e.g. person — empty affects everything"
                  title="The base concept whose representation slides (e.g. 'person'). Leave empty for a global slider (detail, lighting…)."
                  className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]" />
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-content-muted text-[0.625rem] uppercase">Anchor prompt</span>
                <input type="text" value={sliderDraft.anchor}
                  onChange={(e) => setSliderDraft((d) => ({ ...d, anchor: e.target.value }))}
                  onBlur={saveSliderField('anchor')}
                  placeholder="optional — e.g. a photo of a person"
                  title="A nearby concept held in place while training — the paper's fix against the slider bleeding into everything. Empty = no anchor (faster, less protected)."
                  className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]" />
              </label>
            </div>
            <div className="flex items-center gap-3 flex-wrap">
              <label className="flex items-center gap-1.5 text-[0.6875rem] text-content-muted"
                title="How hard the training pushes along the positive↔negative direction (trainer default 3). Higher = stronger effect per strength unit, higher collapse risk.">
                Guidance strength
                <select value={String(slider?.guidance ?? 3)} disabled={sliderBusy}
                  onChange={(e) => saveSlider({ guidance: Number(e.target.value) })}
                  className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
                  {[1, 2, 3, 4, 5, 6, 8].map((v) => (
                    <option key={v} value={String(v)}>{v}{v === 3 ? ' (default)' : ''}</option>
                  ))}
                </select>
              </label>
              <label className="flex items-center gap-1.5 text-[0.6875rem] text-content-muted"
                title="Weight of the anchor loss (trainer default 1). Only used when an anchor prompt is set.">
                Anchor strength
                <select value={String(slider?.anchor_strength ?? 1)} disabled={sliderBusy || !(slider?.anchor || '').trim()}
                  onChange={(e) => saveSlider({ anchor_strength: Number(e.target.value) })}
                  className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem] disabled:opacity-50">
                  {[0.25, 0.5, 1, 2, 4].map((v) => (
                    <option key={v} value={String(v)}>{v}{v === 1 ? ' (default)' : ''}</option>
                  ))}
                </select>
              </label>
              <span className="text-content-subtle text-[0.625rem]"
                title="Slider defaults: low rank (public sliders ship at rank 4-8) and a fixed step target — both editable in Advanced options.">
                rank {advEffRank} · ~{stepsInfo?.steps ?? 1000} steps · previews at −2/−1/+1/+2
              </span>
            </div>
            {sliderPromptsMissing && (
              <p className="m-0 text-amber-300 text-[0.6875rem]">
                ⚠ Both prompts are required — they define the two ends of the slider.
              </p>
            )}
            <p className="m-0 text-content-subtle text-[0.625rem]">
              {SLIDER_FAMILY_NOTES[trainType] || ''}
              {trainType !== 'krea' ? ' Krea 2 is the reference family for sliders — switch the LoRA type above to start there.' : ''}
            </p>
          </>
        )}
      </div>

      {/* A disabled ☁ Train-in-cloud button always states WHY, right under the
          button row — the tooltip alone was invisible until hovered, so a greyed
          SDXL cloud button read as an unexplained limit (owner-reported). */}
      {caps.cloud_training && cloudDisabledReason && (
        <p className="m-0 text-sky-300/90 text-[0.6875rem]">
          ☁ Cloud training unavailable — {cloudDisabledReason}
        </p>
      )}

      {actives.length > 0 && (
        <p className="m-0 text-content-subtle text-[0.625rem]">
          ☁ {actives.length}/{cloudStatus.limit || 1} cloud runs — ${cloudStatus.total_price_per_hour || 0}/h total
        </p>
      )}

      {/* Pointeur visible quand le bouton Train est bloqué par un réglage qui
          vit dans la section repliée — sinon la cause resterait cachée. */}
      {(baseBlocksTrain || sdxlNeedsBase) && (
        <p className="m-0 text-amber-300 text-[0.6875rem]">
          ⚠ {sdxlNeedsBase
            ? 'SDXL needs a base checkpoint — pick one in Advanced options below.'
            : convertRunning
              ? 'The selected base is being converted — training unlocks when it finishes (details in Advanced options).'
              : 'The selected custom base must be converted once before training — open Advanced options below.'}
        </p>
      )}

      <details id="ds-training-advanced" open={advancedOpen}
        className="rounded-lg border border-border bg-surface open:pb-2.5 scroll-mt-20">
        <summary data-workspace-focus
          onClick={togglePanel('advanced', advancedOpen, setAdvancedOpen)}
          className="cursor-pointer select-none px-3 py-2 text-sm text-content font-semibold">
          ⚙️ Advanced options
          <span className="ml-2 font-normal text-content-subtle text-[0.6875rem]">
            base &amp; variant · rank · resolution · masked · steps · scheduling · presets
          </span>
        </summary>
        <div className="px-3 pt-1 flex flex-col gap-2">
          {/* --- Presets : réglages nommés, ré-applicables et partageables en JSON.
               Appliquer REMPLACE les réglages explicites du dataset ; les clés
               inconnues d'un fichier importé sont ignorées (tolérance de version). --- */}
          <div className="flex items-center gap-1.5 flex-wrap rounded-lg border border-border bg-app/40 px-2 py-1.5">
            <span className="text-content-muted text-[0.625rem] uppercase">Presets</span>
            <select value={presetSel} onChange={(e) => setPresetSel(e.target.value)}
              aria-label="Training preset"
              className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem] max-w-[220px]">
              <option value="">— pick a preset —</option>
              {/* Built-ins first, as their own group: the shipped, researched
                  recipes for this family × kind (read-only, versioned with
                  the app). User snapshots follow. */}
              {visiblePresets.some((p) => p.builtin) && (
                <optgroup label="Built-in (researched)">
                  {visiblePresets.filter((p) => p.builtin).map((p) => (
                    <option key={p.id} value={p.id} title={p.description || undefined}>
                      ★ {p.name}
                    </option>
                  ))}
                </optgroup>
              )}
              {visiblePresets.some((p) => !p.builtin) && (
                <optgroup label="My presets">
                  {visiblePresets.filter((p) => !p.builtin).map((p) => (
                    <option key={p.id} value={p.id}>{p.name}</option>
                  ))}
                </optgroup>
              )}
            </select>
            <button type="button" onClick={applyPreset}
              disabled={!selPreset || presetBusy || trainTypeBusy}
              title="Replace this dataset's advanced settings with the selected preset"
              className="px-2.5 py-1 rounded-lg bg-primary/20 border border-primary/40 text-white text-[0.75rem] font-semibold disabled:opacity-40">
              Apply
            </button>
            <span className="mx-0.5 text-content-subtle" aria-hidden>·</span>
            <button type="button" onClick={savePreset} disabled={presetBusy || trainTypeBusy}
              title="Save this dataset's current advanced settings as a named preset"
              className="px-2.5 py-1 rounded-lg bg-surface-raised border border-border text-content text-[0.75rem] disabled:opacity-40">
              💾 Save current…
            </button>
            <button type="button" onClick={() => presetFileRef.current?.click()}
              disabled={presetBusy || trainTypeBusy}
              title="Import a preset from a JSON file (exported from any app version — unknown options are ignored at apply time)"
              className="px-2.5 py-1 rounded-lg bg-surface-raised border border-border text-content text-[0.75rem] disabled:opacity-40">
              ⬆ Import
            </button>
            <button type="button" onClick={exportPreset} disabled={!selPreset || presetBusy}
              title="Download the selected preset as a shareable JSON file"
              className="px-2.5 py-1 rounded-lg bg-surface-raised border border-border text-content text-[0.75rem] disabled:opacity-40">
              ⬇ Export
            </button>
            <button type="button" onClick={deletePreset} disabled={!selPreset || selPreset.builtin || presetBusy}
              title={selPreset?.builtin ? 'Built-in presets ship with the app and cannot be deleted' : 'Delete the selected preset'}
              className="px-2 py-1 rounded-lg bg-red-500/15 border border-red-500/40 text-red-300 text-[0.75rem] disabled:opacity-40">
              🗑
            </button>
            <input ref={presetFileRef} type="file" accept=".json,application/json" className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) importPreset(f);
                e.target.value = '';
              }} />
            {selPreset && (
              <span role="status" className="basis-full text-content-subtle text-[0.625rem] leading-relaxed">
                {selPreset.builtin ? '★ researched recipe' : 'user preset'} · {typeLabel} · {kind}
                {Array.isArray(selPreset.variants) && selPreset.variants.length
                  ? ` · recipe ${selPreset.variants.join(' / ')}` : ''}
                {selPreset.builtin && trainingPresetDatasetKind(selPreset) === 'style'
                  ? ' · applying also restores adaptive Style steps' : ''}
                {selPreset.description ? <>{' — '}{selPreset.description}</> : ''}
              </span>
            )}
          </div>
          {/* --- Base d'entraînement : officielle (recommandé) ou merge ComfyUI custom.
               Affichée MÊME pendant un training en cours → choisir la base du job mis
               en file (sinon « Mettre en file » réutilisait silencieusement la base persistée). --- */}
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-content-muted text-[0.625rem] uppercase">
                Base{status.in_progress ? ' (next queued job)' : ''}
              </span>
              <select value={customBase ? CUSTOM_BASE_SENTINEL : base}
                onChange={(e) => {
                  const v = e.target.value;
                  if (v === CUSTOM_BASE_SENTINEL) { setCustomBase(true); setBase(''); }
                  else { setCustomBase(false); setBase(v); }
                }}
                aria-label="Base model"
                className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem] max-w-[230px]">
                {(currentBases.length ? currentBases
                  : [{ value: '', label: trainType === 'sdxl' ? (comfyConfigured ? 'No SDXL checkpoint found' : 'ComfyUI not configured') : trainType === 'krea' ? 'Official — Krea 2' : trainType === 'flux' ? 'Official — FLUX.1-dev' : trainType === 'flux2klein' ? 'Official — FLUX.2 Klein' : 'Official — Z-Image-Turbo' }]).map((b) => (
                  <option key={b.value} value={b.value}>
                    {trainType === 'zimage' && !b.value ? 'Official recipe — selected by variant' : b.label}{b.value && baseInfo?.converted?.[b.value] ? ' ✓' : ''}
                  </option>
                ))}
                {/* Local-only: a free path to a .safetensors of the SAME architecture. */}
                {customSupported && (
                  <option value={CUSTOM_BASE_SENTINEL}>Custom weights… (local file)</option>
                )}
              </select>
              {/* Z-Image variants are distinct training recipes, even with the
                  official base. Never hide this selector: a persisted De-Turbo
                  value must be visible before the next local/cloud launch. */}
              {trainType === 'zimage' && (
                <select value={variant} onChange={(e) => setVariant(e.target.value)}
                  aria-label="Z-Image training recipe"
                  title="Z-Image training recipe — Turbo requires the v2 training adapter; Base and De-Turbo use separate non-distilled repositories without that adapter."
                  className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
                  <option value="turbo">Turbo · adapter v2</option>
                  <option value="base">Base · non-distilled</option>
                  <option value="deturbo">De-Turbo · no adapter</option>
                </select>
              )}
              {/* Krea 2 : reco officielle « train on Raw, validate on Turbo ». Le RAW
                  (non distillé) est le checkpoint prévu pour le fine-tuning ; sa LoRA
                  transfère vers Turbo à l'inférence. Turbo+adapter = alternative VRAM. */}
              {trainType === 'krea' && (
                <select value={variant} onChange={(e) => setVariant(e.target.value)}
                  aria-label="Krea 2 training base"
                  title="Krea 2 training base — Raw is the official recommendation (best quality; the LoRA transfers to Turbo at inference). Turbo+adapter is the VRAM-friendly alternative. First Raw training downloads the Raw weights (~24 GB) and runs longer."
                  className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
                  <option value="base">Raw (recommended)</option>
                  <option value="turbo">Turbo (w/ adapter)</option>
                </select>
              )}
              {/* FLUX.2 Klein : deux TAILLES de base (pas une histoire de distillation
                  comme Krea) — 4B = la voie locale 16-24 GB, 9B = 32-48 GB, pensé
                  pour ☁️ Train in cloud. Les deux sont gated sur Hugging Face. */}
              {trainType === 'flux2klein' && (
                <select value={variant} onChange={(e) => setVariant(e.target.value)}
                  aria-label="FLUX.2 Klein model size"
                  title="FLUX.2 Klein model size — 4B fits a 16-24 GB local GPU (recommended locally); 9B needs 32-48 GB VRAM, best trained via ☁️ Train in cloud. Both bases are gated on Hugging Face: accept the license and set a HF token before the first run."
                  className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
                  <option value="4b">4B (local, 16-24 GB)</option>
                  <option value="9b">9B (cloud, 32-48 GB)</option>
                </select>
              )}
            </div>
            {zimageRecipe && (
              <div className="flex flex-col gap-1 rounded-md border border-sky-400/30 bg-sky-500/[0.08] px-2.5 py-2 text-[0.6875rem]"
                aria-label="Effective Z-Image training recipe">
                <div className="flex items-center gap-1.5 flex-wrap">
                  <span className="text-sky-200 font-semibold">Effective Z-Image recipe</span>
                  <span className="text-content-subtle">·</span>
                  <span className="text-content">{zimageRecipe.variantLabel}</span>
                  <span className={`px-1.5 py-px rounded border ${zimageRecipe.adapterActive
                    ? 'border-green-400/40 bg-green-400/10 text-green-300'
                    : 'border-border bg-app/40 text-content-muted'}`}>
                    {zimageRecipe.adapterActive ? 'Turbo adapter v2: ON' : 'Training adapter: OFF'}
                  </span>
                </div>
                <span className="text-content-subtle leading-relaxed">
                  Base: <b className="text-content font-mono font-medium break-all">{zimageRecipe.baseLabel}</b>
                  {' '}· inference: {zimageRecipe.inferenceHint}.
                  {zimageRecipe.customVerificationRequired
                    ? ' This custom/converted base keeps its own weights. Its Turbo/Base/De-Turbo profile is declared by the selected variant; server preflight checks conversion and architecture only, and the adapter follows that declaration.'
                    : ' The server locks this official base/adapter pair before launch.'}
                </span>
                {zimageRecipe.customVerificationRequired && variant !== 'turbo' && (
                  <span className="text-amber-300">
                    ⚠ Confirm that this custom base is really {variant === 'deturbo' ? 'De-Turbo' : 'non-distilled Base'}; this cannot be detected automatically.
                  </span>
                )}
              </div>
            )}
            {/* « Custom weights… » : chemin local vers un .safetensors de la MÊME
                architecture. Local-only (le cloud refuse), TE/VAE restent officiels
                (sauf les overrides SDXL séparés plus bas). Vérifié au lancement. */}
            {customBase && customSupported && (
              <div className="flex flex-col gap-1">
                <input type="text" value={base} onChange={(e) => setBase(e.target.value)}
                  spellCheck={false}
                  placeholder={trainType === 'sdxl'
                    ? 'C:\\path\\to\\your-sdxl-checkpoint.safetensors'
                    : `C:\\path\\to\\your-${typeLabel.toLowerCase().replace(/[^a-z0-9]+/g, '')}-model.safetensors`}
                  aria-label="Custom weights path"
                  className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem] font-mono w-full max-w-[520px]" />
                <span className="text-content-subtle text-[0.625rem] leading-relaxed">
                  Local path to a <b className="text-content-muted font-medium">{typeLabel}</b> .safetensors
                  (same architecture). The file is checked at launch (exists, valid, arch signature);
                  an unrecognized file asks for confirmation. Local-only — cloud training refuses it.
                </span>
              </div>
            )}
            {/* krea et flux2klein n'ont QUE des bases officielles fixes (rien à
                lister depuis ComfyUI) → le warning « bases can't be listed » n'y
                apporte que du bruit. */}
            {!comfyConfigured && trainType !== 'krea' && trainType !== 'flux2klein' && (
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-amber-300 text-[0.625rem]">
                  ⚠️ ComfyUI folder not set — training bases can't be listed{trainType === 'sdxl' ? '' : ' (the official Z-Image base still works)'}.
                </span>
                <a href="#/setup"
                  className="px-2.5 py-1 rounded-lg bg-indigo-500/20 border border-indigo-400/40 text-indigo-200 text-[0.6875rem] font-semibold">
                  Point the app at ComfyUI →
                </a>
              </div>
            )}
            {needsConversion && !baseConverted && !convertRunning && (
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-amber-300 text-[0.625rem]">⚠️ Base must be converted before training (~12 GB, a few min, one time only).</span>
                <button type="button" onClick={doPrepareBase}
                  className="px-2.5 py-1 rounded-lg bg-indigo-500/20 border border-indigo-400/40 text-indigo-200 text-[0.6875rem] font-semibold">
                  ⚙️ Convert the base
                </button>
              </div>
            )}
            {convertRunning && (
              <span className="text-indigo-300 text-[0.625rem] flex items-center gap-1.5">
                <span className="inline-block w-3 h-3 border-2 border-indigo-400/40 border-t-indigo-400 rounded-full animate-spin" aria-hidden />
                Converting the base… (~a few minutes)
              </span>
            )}
            {baseConverted && (
              <span className="text-green-400/80 text-[0.625rem]">✓ Base ready — training will produce a LoRA native to this model.</span>
            )}
            {convertError && (
              <span className="text-red-300 text-[0.625rem] break-words">❌ Conversion failed: {convertError}</span>
            )}
            {/* SDXL-only: separate VAE / text-encoder overrides. SDXL is the one
                family where ai-toolkit honours these top-level (every other family
                bundles its TE/VAE) — the server refuses them elsewhere. Optional. */}
            {vaeTeSupported && (
              <div className="flex flex-col gap-1.5 mt-1 pt-2 border-t border-white/[0.07]">
                <span className="text-content-muted text-[0.625rem] uppercase">
                  SDXL overrides (optional)
                </span>
                <label className="flex flex-col gap-0.5">
                  <span className="text-content text-[0.6875rem]">VAE path</span>
                  <input type="text" value={vaePath} onChange={(e) => setVaePath(e.target.value)}
                    spellCheck={false} placeholder="leave empty to use the checkpoint's own VAE"
                    aria-label="SDXL VAE path"
                    className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem] font-mono w-full max-w-[520px]" />
                </label>
                <label className="flex flex-col gap-0.5">
                  <span className="text-content text-[0.6875rem]">Text encoder path or repo</span>
                  <input type="text" value={tePath} onChange={(e) => setTePath(e.target.value)}
                    spellCheck={false} placeholder="leave empty to use the checkpoint's own text encoders"
                    aria-label="SDXL text encoder path or HF repo"
                    className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem] font-mono w-full max-w-[520px]" />
                </label>
                <span className="text-content-subtle text-[0.625rem] leading-relaxed">
                  Leave both empty to use the checkpoint's own VAE/text encoders. A VAE is a local
                  .safetensors; the text encoder may be a local folder or a Hugging Face repo id.
                  Checked at launch. These are SDXL-only and local-only (cloud training refuses them).
                </span>
              </div>
            )}
          </div>

          {/* Model & training knobs — researched defaults (see the Research note),
              editable per dataset. Each carries a plain-English "why / how". */}
          <div className="flex flex-col rounded-lg border border-border bg-app/30 p-2.5 divide-y divide-white/[0.07] [&>*]:py-2.5 [&>*:first-child]:pt-0 [&>*:last-child]:pb-0">
            <div className="flex items-center gap-1.5 text-indigo-300/80 text-[0.625rem] font-semibold uppercase tracking-wider">
              <span aria-hidden className="h-1.5 w-1.5 rounded-full bg-indigo-400/60" /> Model &amp; training
            </div>

            <div className="flex flex-col gap-0.5">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-content text-[0.75rem] w-28 shrink-0">LoRA rank</span>
                <select value={String(advRankChoice)}
                  onChange={(e) => saveAdv({ rank: e.target.value === 'auto' ? 'auto' : Number(e.target.value) })}
                  aria-label="LoRA rank"
                  className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
                  <option value="auto">Auto ({advDefaultRank})</option>
                  <option value="8">8</option><option value="16">16</option><option value="24">24</option>
                  <option value="32">32</option><option value="48">48</option><option value="64">64</option>
                </select>
                <span className="text-content-subtle text-[0.625rem] tabular-nums">→ rank {advEffRank} / alpha {advEffAlpha}</span>
              </div>
              <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
                <b className="text-content-muted font-medium">Why:</b> how much capacity the LoRA has to learn the
                target — identity, concept, or visual style. <b className="text-content-muted font-medium">How:</b> use
                Auto or the researched preset for this family; higher ranks can capture broader, more complex variation
                but make a larger adapter and can overfit a small repetitive set. The effective rank/alpha is shown above.
              </span>
            </div>

            <div className="flex flex-col gap-0.5">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-content text-[0.75rem] w-28 shrink-0">Resolution</span>
                <select value={advRes} onChange={(e) => saveAdv({ resolution: e.target.value })}
                  aria-label="Training resolution"
                  className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
                  <option value="768,1024">768 + 1024 (multi-scale)</option>
                  <option value="1024">1024 only</option>
                  <option value="768">768 only (low VRAM)</option>
                </select>
              </div>
              <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
                <b className="text-content-muted font-medium">Why:</b> the size(s) images are trained at — and the #1
                VRAM lever. <b className="text-content-muted font-medium">How:</b> multi-scale trains at two sizes so
                the LoRA holds up from a close-up face to a full-body shot; single 1024 is a bit faster.
                <b className="text-content-muted font-medium"> 768 only</b> cuts memory use sharply and trains much
                faster — your best shot at Krea 2 on a GPU under 24 GB, at some cost in fine detail.
                {sliderOn && (
                  <span className="block mt-1 text-purple-200/90">
                    <b className="font-medium">Slider default: 768 only</b> — the slider loss makes several passes per
                    step, so its VRAM peak is much higher than a normal run. Pick a resolution here to override.
                  </span>
                )}
              </span>
            </div>

            <div className="flex flex-col gap-0.5">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-content text-[0.75rem] w-28 shrink-0">Save checkpoint</span>
                <select value={String(advSave)} onChange={(e) => saveAdv({ save_every: Number(e.target.value) })}
                  aria-label="Checkpoint frequency"
                  className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
                  <option value="250">every 250 steps</option>
                  <option value="500">every 500 steps</option>
                  <option value="1000">every 1000 steps</option>
                </select>
              </div>
              <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
                <b className="text-content-muted font-medium">Why:</b> how often a checkpoint is written.
                <b className="text-content-muted font-medium"> How:</b> finer (250) gives more epochs to pick the
                least-overfit one in the Test Studio; coarser saves disk.
              </span>
            </div>

            <div className="flex flex-col gap-0.5">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-content text-[0.75rem] w-28 shrink-0">Saves kept</span>
                <select value={String(adv?.max_step_saves ?? 4)}
                  onChange={(e) => saveAdv({ max_step_saves: Number(e.target.value) })}
                  aria-label="Maximum intermediate saves kept"
                  className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
                  <option value="2">last 2</option>
                  <option value="3">last 3</option>
                  <option value="4">last 4</option>
                  <option value="6">last 6</option>
                  <option value="10">last 10</option>
                </select>
              </div>
              <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
                <b className="text-content-muted font-medium">Why:</b> older intermediate saves are deleted by
                ai-toolkit itself (local and cloud) past this count — the old default of 10 piled up ~10 GB per
                Krea run. <b className="text-content-muted font-medium">How:</b> 4 is plenty to pick the best
                epoch; raise it only for long runs you want to comb through finely.
              </span>
            </div>

            <div className="flex flex-col gap-0.5">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-content text-[0.75rem] w-28 shrink-0">Preview every</span>
                <select value={String(advSampleEvery)} onChange={(e) => saveAdv({ sample_every: Number(e.target.value) })}
                  aria-label="Preview sample frequency"
                  className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
                  {advSampleEveryChoices.map((n) => (
                    <option key={n} value={String(n)}>every {n} steps</option>
                  ))}
                </select>
              </div>
              <label className="flex flex-col gap-1 mt-1">
                <span className="text-content text-[0.75rem]">Preview prompts</span>
                <textarea value={samplePromptsText}
                  onChange={(e) => setSamplePromptsText(e.target.value)}
                  onBlur={saveSamplePrompts}
                  rows={4}
                  placeholder={advSampleDefault.length ? advSampleDefault.join('\n') : 'one prompt per line'}
                  aria-label="Preview sample prompts, one per line"
                  className="px-2 py-1.5 rounded-lg border border-border bg-surface text-content text-[0.75rem] font-mono leading-relaxed resize-y placeholder:text-content-subtle" />
              </label>
              <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
                <b className="text-content-muted font-medium">Why:</b> these are the test images ai-toolkit renders
                during the run so you can watch the LoRA learn (and later pick the best epoch).
                <b className="text-content-muted font-medium"> How:</b> one prompt per line, up to {advMaxPrompts}. {isStyle
                  ? 'Style is always-on: prompts stay content-only and no activation trigger is added. Test varied subjects and lighting to reveal content bias.'
                  : isConcept
                    ? 'Your concept trigger is added automatically when absent. Leave empty for the concept defaults shown greyed.'
                    : 'Your character trigger is added automatically when absent. Leave empty for the portrait defaults shown greyed.'}
              </span>
            </div>
          </div>

          {/* Expert — last-mile levers. Collapsed by default; every control defaults
              to the current behaviour, so a newcomer who never opens this is unaffected. */}
          <details className="group rounded-lg border border-indigo-400/40 border-l-[3px] border-l-indigo-400 bg-indigo-500/[0.14] transition-colors hover:bg-indigo-500/20">
            <summary className="flex items-center gap-2 cursor-pointer select-none list-none [&::-webkit-details-marker]:hidden px-2.5 py-2.5 text-[0.6875rem] font-semibold uppercase tracking-wider text-indigo-100 hover:text-white">
              <span aria-hidden className="text-indigo-300 transition-transform group-open:rotate-90">▸</span>
              <span aria-hidden>🔬</span>
              <span>Expert — last-mile levers</span>
              <span className="ml-auto hidden sm:inline normal-case font-normal tracking-normal text-indigo-300/50">network · alpha · dropout{advTimestepSupported ? ' · timestep' : ''} · optimizer · schedule · EMA</span>
            </summary>
            <div className="flex flex-col px-2.5 pb-2.5 divide-y divide-indigo-400/10 [&>div]:py-2.5 [&>div:first-child]:pt-1 [&>div:last-child]:pb-0">
              {/* Network variant — LoRA (default) or LoKr. LoKr is arch-generic in
                  ai-toolkit, so it's offered on every family; the *_supported guard
                  mirrors the timestep pattern for a future family that can't run it. */}
              <div className="flex flex-col gap-0.5">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-content text-[0.75rem] w-28 shrink-0">Network</span>
                  <select value={advNetworkType} onChange={(e) => saveAdv({ network_type: e.target.value })}
                    aria-label="Network type"
                    className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
                    {advNetworkChoices.map((n) => <option key={n} value={n}>{n === 'lora' ? 'LoRA (default)' : 'LoKr'}</option>)}
                  </select>
                  {advNetworkType === 'lokr' && !advNetworkSupported && (
                    <span className="text-amber-300 text-[0.625rem]" title={`LoKr isn't supported for ${trainType} — this run would fall back to LoRA.`}>⚠ not supported for {trainType}</span>
                  )}
                </div>
                <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
                  <b className="text-content-muted font-medium">Why:</b> LoKr often locks likeness earlier at small
                  rank — community recipe: LoKr + low rank + EMA. <b className="text-content-muted font-medium">How:</b> LoRA
                  (default) is the standard adapter; LoKr factorises the update differently and can capture identity in
                  fewer steps on a tiny set. Pair it with a low rank and EMA below.
                </span>
              </div>
              {/* EMA — exponential moving average of the weights */}
              <div className="flex flex-col gap-0.5">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-content text-[0.75rem] w-28 shrink-0">EMA</span>
                  <select value={String(advEma)}
                    onChange={(e) => saveAdv({ ema: e.target.value === '0' ? 'off' : Number(e.target.value) })}
                    aria-label="EMA (exponential moving average)"
                    className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
                    <option value="0">Off (default)</option>
                    {advEmaChoices.map((d) => <option key={d} value={String(d)}>{d}</option>)}
                  </select>
                </div>
                <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
                  <b className="text-content-muted font-medium">Why:</b> exponential moving average of the weights —
                  smoother, often better checkpoints. <b className="text-content-muted font-medium">How:</b> Off by
                  default; 0.99 averages faster (the recommended pairing with LoKr on small sets), 0.999 is slower and
                  steadier.
                </span>
              </div>
              {/* Dual captions — train each image with a long AND a short caption */}
              <div className="flex flex-col gap-0.5">
                <label className="flex items-center gap-2 flex-wrap cursor-pointer">
                  <span className="text-content text-[0.75rem] w-28 shrink-0">Dual captions</span>
                  <input type="checkbox" checked={advDualCaptions}
                    onChange={(e) => saveAdv({ dual_captions: e.target.checked })}
                    aria-label="Dual long + short captions"
                    className="h-4 w-4 rounded border-border bg-surface accent-indigo-500" />
                  <span className="text-content-muted text-[0.75rem]">long + short (local training only)</span>
                </label>
                <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
                  <b className="text-content-muted font-medium">Why:</b> trains each image with both a full and a brief
                  caption (text-side augmentation) so the LoRA leans less on any single wording.
                  <b className="text-content-muted font-medium"> How:</b> the short variant is derived from the long one
                  when you (re-)caption — same rules (no trigger, identity/concept/aesthetic kept out); edit it per image in
                  the ⛶ caption editor. Cloud runs ignore this and train on the long caption only for now.
                </span>
              </div>
              {/* Decoupled alpha */}
              <div className="flex flex-col gap-0.5">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-content text-[0.75rem] w-28 shrink-0">Alpha</span>
                  <select value={String(advAlphaChoice)}
                    onChange={(e) => saveAdv({ alpha: e.target.value === 'auto' ? 'auto' : Number(e.target.value) })}
                    aria-label="LoRA alpha"
                    className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
                    <option value="auto">Auto (= {advDefaultAlpha})</option>
                    {advAlphaChoices.map((a) => <option key={a} value={String(a)}>{a}</option>)}
                  </select>
                  <span className="text-content-subtle text-[0.625rem] tabular-nums">→ scale {(advEffAlpha / Math.max(1, advEffRank)).toFixed(2)}×</span>
                </div>
                <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
                  <b className="text-content-muted font-medium">Why:</b> alpha ÷ rank is the LoRA&apos;s effective strength while
                  training — a soft learning-rate lever that isn&apos;t the LR. <b className="text-content-muted font-medium">How:</b> Auto
                  ties alpha to rank (scale 1.0); a lower alpha (e.g. ½ rank) softens the fit — a clean way to stop a tiny
                  (≤20-image) set from memorising without touching LR or rank.
                </span>
              </div>
              {/* Network dropout */}
              <div className="flex flex-col gap-0.5">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-content text-[0.75rem] w-28 shrink-0">Network dropout</span>
                  <select value={String(advDropout)}
                    onChange={(e) => saveAdv({ dropout: e.target.value === '0' ? 'off' : Number(e.target.value) })}
                    aria-label="Network dropout"
                    className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
                    <option value="0">Off</option>
                    {advDropoutChoices.map((d) => <option key={d} value={String(d)}>{d}</option>)}
                  </select>
                </div>
                <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
                  <b className="text-content-muted font-medium">Why:</b> this is <i>network</i> dropout: it randomly drops
                  adapter updates to reduce memorisation. It is separate from caption dropout. <b className="text-content-muted font-medium">How:</b> follow
                  the preset; 0.05 is gentle, while larger values can underfit. Krea&apos;s text-embedding cache affects caption
                  dropout, not this network control.
                </span>
              </div>
              {/* Timestep weighting — flowmatch families only (SDXL disables it) */}
              {advTimestepSupported && (
                <div className="flex flex-col gap-0.5">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-content text-[0.75rem] w-28 shrink-0">Timestep weighting</span>
                    <select value={advTimestep} onChange={(e) => saveAdv({ timestep_type: e.target.value })}
                      aria-label="Timestep weighting"
                      className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
                      <option value="auto">Auto ({advTimestepDefault})</option>
                      {advTimestepChoices.map((t) => <option key={t} value={t}>{t}</option>)}
                    </select>
                  </div>
                  <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
                    <b className="text-content-muted font-medium">Why:</b> which noise levels the loss emphasises — the
                    detail-versus-global-structure balance for flow-matching models. <b className="text-content-muted font-medium">How:</b> Auto
                    uses the family recipe ({advTimestepDefault}); use the researched Style preset unless you are deliberately
                    testing texture/detail versus composition/structure emphasis.
                  </span>
                </div>
              )}
              {/* Optimizer */}
              <div className="flex flex-col gap-0.5">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-content text-[0.75rem] w-28 shrink-0">Optimizer</span>
                  <select value={advOptimizer} onChange={(e) => saveAdv({ optimizer: e.target.value })}
                    aria-label="Optimizer"
                    className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
                    {advOptimizerChoices.map((o) => <option key={o} value={o}>{o}{o === 'adamw8bit' ? ' (default)' : ''}</option>)}
                  </select>
                </div>
                <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
                  <b className="text-content-muted font-medium">Why:</b> how the weights are updated — the biggest training
                  lever after the dataset. <b className="text-content-muted font-medium">How:</b> <i>adamw8bit</i> (default)
                  is fast and VRAM-light; <i>adafactor</i> uses less memory and auto-scales; <i>automagic</i> sets the
                  learning rate itself (no LR to tune, no extra install); <i>prodigy</i> also auto-tunes the LR and is
                  popular for tiny sets — but may need <code className="text-content-muted">pip install prodigyopt</code> in
                  the ai-toolkit venv. Picking an auto-LR optimiser is the &quot;push further without cranking the LR&quot; move.
                </span>
              </div>
              {/* LR schedule (+ warmup, only for the warmup schedule) */}
              <div className="flex flex-col gap-0.5">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-content text-[0.75rem] w-28 shrink-0">LR schedule</span>
                  <select value={advLrSched} onChange={(e) => saveAdv({ lr_scheduler: e.target.value })}
                    aria-label="Learning-rate schedule"
                    className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
                    {advLrSchedChoices.map((s) => <option key={s} value={s}>{LR_SCHED_LABELS[s] || s}</option>)}
                  </select>
                  {advLrSched === 'constant_with_warmup' && (
                    <select value={String(advWarmup || 100)} onChange={(e) => saveAdv({ warmup: Number(e.target.value) })}
                      aria-label="Warmup steps"
                      className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
                      {advWarmupChoices.map((w) => <option key={w} value={String(w)}>{w} warmup</option>)}
                    </select>
                  )}
                </div>
                <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
                  <b className="text-content-muted font-medium">Why:</b> how the learning rate moves over the run.
                  <b className="text-content-muted font-medium"> How:</b> <i>Constant</i> (default) holds it flat;
                  <i> Warmup → constant</i> ramps it up over the first N steps (a gentler start that avoids early
                  over-commitment on a small set) then holds; <i>Linear</i>/<i>Cosine</i> decay it toward 0 by the end for
                  cleaner convergence. The warmup-steps box only applies to the warmup schedule.
                </span>
              </div>
              {/* Gradient accumulation (effective batch) */}
              <div className="flex flex-col gap-0.5">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-content text-[0.75rem] w-28 shrink-0">Effective batch</span>
                  <select value={String(advGradAccum)} onChange={(e) => saveAdv({ grad_accum: Number(e.target.value) })}
                    aria-label="Gradient accumulation"
                    className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
                    {advGradAccumChoices.map((g) => <option key={g} value={String(g)}>{g === 1 ? '1 (default)' : `${g} × accum`}</option>)}
                  </select>
                </div>
                <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
                  <b className="text-content-muted font-medium">Why:</b> averages the gradient over N micro-batches before
                  each update — a larger <i>effective</i> batch with no extra VRAM. <b className="text-content-muted font-medium">How:</b> 1
                  (default); 2–4 smooths the noisy gradients a tiny dataset produces (steadier training), at the cost of a
                  bit more time per update. A cheap stabiliser for small sets.
                </span>
              </div>
            </div>
          </details>

          <label className="flex items-center gap-1.5 text-[0.6875rem] text-content-muted cursor-pointer"
            title={sliderOn
              ? 'Slider mode ignores masks: the slider loss never reads the masked-loss path, so the server trains unmasked regardless of this toggle.'
              : isStyle
              ? 'Keep this OFF for Style: the aesthetic must be learned across the whole frame, including backgrounds. A person mask would discard much of the style signal.'
              : isConcept
                ? 'Keep this OFF for a Concept dataset — a person mask could erase the recurring concept you are training.'
              : 'Masked training: a person mask is generated for every image (rembg, CPU) and the background only weighs 10% of the loss — identity binds to the face, not the room. Uncheck to train the old way.'}>
            <input type="checkbox" checked={masked && !sliderOn} disabled={sliderOn}
              onChange={(e) => setMasked(e.target.checked)}
              aria-label="Masked training (background at 10%)"
              className="accent-primary w-3.5 h-3.5 disabled:opacity-50" />
            <span className={masked && !sliderOn && !maskedRembgMissing ? 'text-emerald-300' : ''}>🎭 Masked (bg 10%)</span>
            {sliderOn && (
              <span className="text-content-subtle" title="The slider loss ignores masks — the server forces unmasked training in slider mode.">
                off in slider mode
              </span>
            )}
            {isConceptual && masked && !sliderOn && (
              <span className="text-amber-300" title={isStyle ? 'A person mask discards full-frame style information.' : 'A person mask can erase the concept.'}>
                ⚠️ off required for {isStyle ? 'styles' : 'concepts'}
              </span>
            )}
            {maskedRembgMissing && (
              <span className="text-amber-300"
                title="rembg isn't installed, so no person masks can be generated — this run will train UNMASKED (background at full weight), not masked. Install the ML extras from the Setup tab (requirements-ml.txt, Python 3.10–3.12) to enable masked training.">
                ⚠️ rembg missing — will train unmasked
              </span>
            )}
          </label>

          {!status.in_progress && keptCount >= 10 && (
            <label className="flex items-center gap-1.5 text-content-subtle text-[0.6875rem]"
              title={stepsInfo?.rationale
                ? `${stepsInfo.rationale} Leave empty to use it; applies to Train, Add to queue and Schedule.`
                : `Target training steps. Leave empty for the backend's ${typeLabel} / ${checkpointVariantLabel(trainType, variant)} adaptive recipe. Applies to Train, Add to queue and Schedule.`}>
              <span className="uppercase text-content-muted text-[0.625rem]">Steps</span>
              <input type="number" min={500} step={100}
                value={stepsOverride}
                onChange={(e) => setStepsOverride(e.target.value)}
                placeholder={stepsInfo?.steps != null ? String(stepsInfo.steps) : 'adaptive'}
                aria-label="Target training steps (leave empty for adaptive)"
                className="w-[4.5rem] rounded border border-border bg-app/60 px-1.5 py-0.5 text-content tabular-nums text-[0.75rem]" />
              <span>{stepsOverride.trim()
                ? 'target'
                : stepsInfo?.steps != null
                  ? `≈ ${stepsInfo.steps} · ${stepsRecipeFamily} / ${stepsRecipeVariant} (${keptCount} img)`
                  : `adaptive · ${typeLabel} / ${checkpointVariantLabel(trainType, variant)} (${keptCount} img)`}</span>
            </label>
          )}
          {zimageTurboLongRun && (
            <p role="status" className="m-0 rounded-md border border-amber-400/35 bg-amber-400/[0.08] px-2 py-1.5 text-amber-200 text-[0.6875rem] leading-relaxed">
              ⚠ Turbo long-run warning — the effective target is {effectiveTargetSteps} steps
              (over {ZIMAGE_TURBO_LONG_RUN_STEPS}). Turbo is distilled and can degrade or overfit on a long run;
              this does not block launch, but a lower cap plus best-checkpoint selection is safer.
            </p>
          )}

          {status.installed && keptCount >= (TRAIN_MIN[trainType]?.[0] ?? 12) && (
            <div className="flex items-center gap-2 flex-wrap">
              <button type="button" disabled={queued || baseBlocksTrain} onClick={openSched}
                aria-expanded={showSched}
                title={baseBlocksTrain
                  ? 'Convert the selected custom base first'
                  : 'Schedule this training for a specific day and time — it will queue up if another training is running then'}
                className="px-3 py-1.5 rounded-lg bg-amber-500/15 border border-amber-400/40 text-amber-200 text-sm font-semibold disabled:opacity-40">
                {queued ? '✓ Queued' : '⏰ Schedule'}
              </button>
              <span className="text-content-subtle text-[0.625rem]">
                run this training later, at a day &amp; time you pick
              </span>
            </div>
          )}

          {showSched && !queued && (
            <div className="flex items-center gap-2 flex-wrap rounded-lg border border-amber-400/30 bg-amber-500/5 px-3 py-2">
              <label className="flex items-center gap-2 text-content-muted text-[0.6875rem]">
                <span className="uppercase">Start at</span>
                <input type="datetime-local" value={schedAt}
                  onChange={(e) => setSchedAt(e.target.value)}
                  aria-label="Scheduled training date and time"
                  className="rounded border border-border bg-app/60 px-2 py-1 text-content text-[0.8125rem]" />
              </label>
              <span className="text-content-subtle text-[0.625rem]">
                Base « {baseLabel} » — if another training is running at that time, it waits in the queue.
              </span>
              <button type="button" onClick={schedule} disabled={!schedAt}
                className="ml-auto px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
                Schedule
              </button>
            </div>
          )}
        </div>
      </details>

      {Array.isArray(status.queue) && status.queue.length > 0 && (
        <div id="ds-training-queue" tabIndex={-1} data-workspace-focus
          className="flex flex-col gap-1 rounded-lg border border-indigo-400/30 bg-indigo-500/5 px-3 py-2 scroll-mt-20">
          <span className="text-content-muted text-[0.625rem] uppercase">Training queue ({status.queue.length})</span>
          {status.queue.map((q, i) => (
            <div key={q.dataset_id} className="flex items-center gap-2 text-[0.6875rem]">
              <span className="text-content-subtle tabular-nums">{i + 1}.</span>
              <span className="text-content">{q.name}</span>
              {q.base_label ? <span className="text-indigo-300/80">· {q.base_label}</span> : null}
              {q.extra_steps ? <span className="text-content-subtle">(+{q.extra_steps} steps)</span> : null}
              {q.steps ? <span className="text-content-subtle">→ {q.steps} steps</span> : null}
              {q.not_before ? (
                <span className="px-1.5 py-px rounded border border-amber-400/40 bg-amber-400/10 text-amber-300"
                  title="Scheduled — starts at this time (or right after the training running then)">
                  ⏰ {String(q.not_before).replace('T', ' ')}
                </span>
              ) : null}
              <button type="button" onClick={() => dequeue(q.dataset_id)}
                className="ml-auto px-2 py-0.5 rounded bg-red-500/15 border border-red-500/40 text-red-300">
                Remove
              </button>
            </div>
          ))}
        </div>
      )}

      {enqErr && (
        <p className="m-0 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-1.5 text-red-300 text-[0.6875rem]">
          ⚠️ Enqueue refused: {enqErr}
        </p>
      )}

      {sliderOn
        ? keptCount < TRAIN_MIN_SLIDER[1] && (
          <p className="m-0 text-content-subtle text-[0.625rem]">
            Slider mode: minimum {TRAIN_MIN_SLIDER[0]} substrate images,{' '}
            {TRAIN_MIN_SLIDER[1]}+ varied ones recommended — you have {keptCount}.
          </p>
        )
        : keptCount < (TRAIN_MIN[trainType]?.[1] ?? 20) && (
          <p className="m-0 text-content-subtle text-[0.625rem]">
            {typeLabel}: minimum {TRAIN_MIN[trainType]?.[0] ?? 12} kept images,{' '}
            {TRAIN_MIN[trainType]?.[1] ?? 20} recommended — you have {keptCount}.
          </p>
        )}

      {/* --- Résultats : checkpoints du run + LoRA déjà importés dans ComfyUI.
           Repliés par défaut ; le résumé du summary donne les comptes sans ouvrir. */}
      <CheckpointPortal host={checkpointHost}>
      <details id="ds-training-checkpoints" open={Boolean(checkpointHost) || checkpointsOpen}
        className="rounded-lg border border-border bg-surface open:pb-2.5 scroll-mt-20">
        <summary data-workspace-focus
          onClick={checkpointHost
            ? (event) => event.preventDefault()
            : togglePanel('checkpoints', checkpointsOpen, setCheckpointsOpen)}
          className="cursor-pointer select-none px-3 py-2 text-sm text-content font-semibold">
          📦 Checkpoints &amp; trained LoRAs
          <span className="ml-2 font-normal text-content-subtle text-[0.6875rem]">
            {ckLoaded
              ? `${checkpoints.length} checkpoint(s) · ${imported.length} in ComfyUI${diskUsage?.total_bytes ? ` · ${fmtBytes(diskUsage.total_bytes)} on disk` : ''}`
              : 'the files your training runs produce'}
          </span>
        </summary>
        <div className="px-3 pt-1 flex flex-col gap-2">
          <div className="flex items-center gap-2 rounded-lg border border-border bg-app px-3 py-2 flex-wrap">
            <span className="text-content-muted text-[0.625rem] uppercase">Browse results</span>
            <select value={checkpointTrainType} onChange={(event) => onCheckpointTypeChange(event.target.value)}
              aria-label="LoRA family to browse"
              className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
              <option value="zimage">Z-Image</option>
              <option value="sdxl">SDXL</option>
              <option value="krea">Krea 2</option>
              <option value="flux">FLUX.1</option>
              <option value="flux2klein">FLUX.2 Klein</option>
            </select>
            {checkpointBaseOptions.length > 0 ? (
              <select value={checkpointBase} onChange={(event) => setCheckpointBase(event.target.value)}
                aria-label="Training base to browse"
                className="min-w-0 max-w-full px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
                {checkpointBaseOptions.map((item) => (
                  <option key={`${checkpointTrainType}-${item.value}`} value={item.value}>{item.label}</option>
                ))}
              </select>
            ) : (
              <span className="text-content text-xs">{checkpointBaseLabel}</span>
            )}
            {checkpointVariants.length > 1 && (
              <select value={checkpointVariant}
                onChange={(event) => setCheckpointVariant(event.target.value)}
                aria-label="Training variant to browse"
                className="min-w-0 max-w-full px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
                {checkpointVariants.map((item) => (
                  <option key={`${checkpointTrainType}-${item.value}`} value={item.value}>{item.label}</option>
                ))}
              </select>
            )}
            <span className="ml-auto text-content-subtle text-[0.625rem]">
              Independent from the next Training configuration
            </span>
          </div>
          {/* Provenance du dataset : version courante + alerte si le dataset a
              changé depuis le dernier entraînement (les checkpoints listés ne
              reflètent alors PLUS l'état actuel). */}
          {datasetState?.registered && (datasetState.changed ? (
            <p className="m-0 rounded-md border border-amber-400/40 bg-amber-500/10 px-2 py-1 text-amber-200 text-[0.6875rem]">
              ⚠ The dataset has <b>changed since v{datasetState.version}</b>
              {datasetState.diff && (
                <>
                  {' '}(
                  {[
                    datasetState.diff.images_added ? `+${datasetState.diff.images_added} image${datasetState.diff.images_added > 1 ? 's' : ''}` : null,
                    datasetState.diff.images_removed ? `−${datasetState.diff.images_removed} image${datasetState.diff.images_removed > 1 ? 's' : ''}` : null,
                    datasetState.diff.captions_changed ? `${datasetState.diff.captions_changed} caption${datasetState.diff.captions_changed > 1 ? 's' : ''} edited` : null,
                    datasetState.diff.images_edited ? `${datasetState.diff.images_edited} image${datasetState.diff.images_edited > 1 ? 's' : ''} edited` : null,
                  ].filter(Boolean).join(', ')}
                  )
                </>
              )}
              {' '}— these checkpoints reflect the old state; the next training becomes <b>v{datasetState.version + 1}</b>.
            </p>
          ) : (
            <p className="m-0 text-content-subtle text-[0.625rem]">
              Dataset version: <span className="text-content font-semibold">v{datasetState.version}</span> — unchanged since the last training.
            </p>
          ))}
          <div className="flex items-center gap-2 flex-wrap">
            {/* () => … sinon React passe l'event en 1er arg → forBase = PointerEvent
                → base_model=[object Object] → run inexistant → liste vide. */}
            <button type="button" onClick={() => loadCheckpoints(checkpointBase, checkpointTrainType, checkpointVariant)}
              title="Reload the checkpoint list for this results filter"
              className="px-3 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-xs font-semibold">
              ↻ Refresh checkpoints
            </button>
            {/* Ouvre les dossiers dans l'explorateur du poste (app locale) :
                loras = imports ComfyUI de la famille ; run = checkpoints bruts. */}
            <button type="button"
              onClick={() => postTrain(`/api/dataset/${ds.currentId}/train/open-folder`,
                { target: 'loras', ...trainingRunSelection(undefined, checkpointTrainType, checkpointVariant) })}
              title={`Open the ComfyUI folder where imported ${checkpointTypeLabel} LoRAs live`}
              className="px-3 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-xs font-semibold">
              📂 LoRA folder
            </button>
            <button type="button"
              onClick={() => postTrain(`/api/dataset/${ds.currentId}/train/open-folder`,
                { target: 'run', ...trainingRunSelection(checkpointBase, checkpointTrainType, checkpointVariant) })}
              title="Open this run's output folder (raw checkpoints, samples, training log)"
              className="px-3 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-xs font-semibold">
              📂 Run folder
            </button>
            <span className="text-content-subtle text-[0.625rem]">
              import the checkpoint you like into ComfyUI to use (and test) the LoRA
            </span>
          </div>

          {checkpoints.length > 0 && (
            <div className="flex flex-col gap-1">
              {/* Identity header for the LOCAL active set: what this group IS +
                  which run produced it + a jump back to its Runs row. */}
              {(() => {
                const li = localRunIdentity(checkpoints);
                return (
                  <div className="flex items-center gap-2 flex-wrap rounded-md border border-violet-500/25 bg-violet-500/5 px-2 py-1">
                    {li && <RunIdChip source={li.source} id={li.id} />}
                    <span className="text-content-muted text-[0.6875rem]">
                      <b className="text-content">Active set</b> — used by Studio / Continue / Import; cloud epochs are mirrored here.
                    </span>
                    {li && (
                      <Link to={`/cloud#${runRowDomId(li.source, li.id)}`}
                        title="Jump to this run on the Runs page"
                        className="ml-auto px-1 py-0.5 text-violet-300 hover:text-violet-200 text-[0.6875rem] font-medium underline decoration-violet-300/40">
                        View in Runs ↗
                      </Link>
                    )}
                  </div>
                );
              })()}
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-content-muted text-[0.625rem] uppercase">
                  {checkpointTypeLabel} checkpoints — base « {checkpointBaseLabel} » · {checkpointVariantDisplay} (pick the earliest one that holds the identity)
                </span>
                <button type="button" disabled={bestEpochBusy}
                  onClick={findBestEpoch}
                  title="Scores every training sample vs the reference photo (face similarity, CPU) and recommends the checkpoint that holds the identity best — needs the Quality tools (ML extras)."
                  className="px-2.5 py-1 rounded-lg bg-amber-500/15 border border-amber-400/40 text-amber-200 text-[0.6875rem] font-semibold disabled:opacity-40">
                  {bestEpochBusy ? '🏆 Scoring samples…' : '🏆 Find best epoch'}
                </button>
                <button type="button" disabled={status.in_progress || !checkpointMatchesTraining}
                  onClick={() => setContinueOpen(true)}
                  title={!checkpointMatchesTraining
                    ? 'To continue this run, select the same LoRA family, base and variant in Training first'
                    : 'Resume from any of this run’s checkpoints — pick the step count, the checkpoint, and the safe settings'}
                  className="ml-auto px-2.5 py-1 rounded-lg bg-indigo-500/20 border border-indigo-400/40 text-indigo-200 text-[0.6875rem] font-semibold disabled:opacity-40">
                  ▶ Continue training…
                </button>
              </div>
              {bestEpoch && !bestEpoch.available && (
                <p className="m-0 text-amber-300 text-[0.625rem]">🏆 {bestEpoch.reason}</p>
              )}
              {bestEpoch?.available && (
                <p className="m-0 text-amber-200 text-[0.625rem]">
                  🏆 Best identity at <span className="font-semibold">step {bestEpoch.best_step}</span>
                  {' '}({(bestEpoch.steps.find((s) => s.step === bestEpoch.best_step)?.mean_sim ?? 0).toFixed(2)} mean similarity)
                  {' '}— per step: {bestEpoch.steps.map((s) => `${s.step}:${s.mean_sim.toFixed(2)}`).join(' · ')}
                </p>
              )}
              {checkpoints.map((c) => (
                <div key={c.filename} className="flex items-center gap-2 text-[0.6875rem]">
                  <span className={c.final ? 'text-green-400 font-semibold' : 'text-content'}>
                    {c.final ? '✓ final (training complete)' : `step ${c.step}`}
                  </span>
                  {c.version && (
                    <span className="px-1.5 py-px rounded border border-border bg-surface-raised text-content-subtle"
                      title={`Trained on dataset version v${c.version}${c.source ? ` (${c.source} run)` : ''}${datasetState?.changed ? ' — the dataset has changed since' : ''}`}>
                      v{c.version}{c.source === 'cloud' ? ' ☁' : ''}
                    </span>
                  )}
                  {bestEpoch?.available && bestEpoch.checkpoint === c.filename && (
                    <span className="px-1.5 py-px rounded border border-amber-400/50 bg-amber-400/15 text-amber-200 font-semibold"
                      title={`Closest checkpoint to the best-scoring step (${bestEpoch.best_step})`}>
                      🏆 recommended
                    </span>
                  )}
                  <button type="button"
                    onClick={async () => {
                      // await + refresh: the import must show up in "IN COMFYUI"
                      // without a manual Refresh click (user-observed). finally:
                      // the list refreshes even if the import failed (the error
                      // toast comes from the hook).
                      try { await ds.importCheckpoint(c.filename, checkpointBase, checkpointTrainType, checkpointVariant); }
                      finally { loadCheckpoints(checkpointBase, checkpointTrainType, checkpointVariant); }
                    }}
                    className="ml-auto px-2 py-0.5 rounded bg-primary/20 border border-primary/40 text-white">
                    Import → {checkpointLorasLabel}
                  </button>
                  <button type="button"
                    onClick={async () => {
                      if (!window.confirm(`Move « ${c.filename} » to the trash?\n\nRecoverable until you empty the trash in Settings.`)) return;
                      const d = await postTrain(`/api/dataset/${ds.currentId}/train/run-checkpoint/delete`,
                        { filename: c.filename, ...trainingRunSelection(checkpointBase, checkpointTrainType, checkpointVariant) });
                      if (d.ok === false) toastTrainError(d, 'Delete failed');
                      loadCheckpoints(checkpointBase, checkpointTrainType, checkpointVariant);
                    }}
                    title="Move this checkpoint to the trash (recoverable until the trash is emptied in Settings)"
                    className="px-2 py-0.5 rounded bg-red-500/15 border border-red-500/40 text-red-300">
                    🗑
                  </button>
                </div>
              ))}
              <div className="flex items-center gap-2">
                <button type="button"
                  onClick={async () => {
                    const finals = checkpoints.filter((c) => c.final).map((c) => c.filename);
                    const best = bestEpoch?.available ? [bestEpoch.checkpoint] : [];
                    const keep = [...new Set([...finals, ...best])];
                    if (!keep.length) {
                      // no final yet (unfinished run): keep the last step
                      const last = checkpoints[checkpoints.length - 1];
                      if (last) keep.push(last.filename);
                    }
                    const removed = checkpoints.filter((c) => !keep.includes(c.filename)).length;
                    if (!removed) return;
                    if (!window.confirm(`Clean up this run?\n\nKeeps ${keep.length} checkpoint(s) (${keep.join(', ')}) and moves ${removed} to the trash — recoverable until you empty the trash in Settings.`)) return;
                    const d = await postTrain(`/api/dataset/${ds.currentId}/train/checkpoints/cleanup`,
                      { keep_filenames: keep, ...trainingRunSelection(checkpointBase, checkpointTrainType, checkpointVariant) });
                    if (d.ok === false) toastTrainError(d, 'Cleanup failed');
                    loadCheckpoints(checkpointBase, checkpointTrainType, checkpointVariant);
                  }}
                  title="Keep the final (+ the 🏆 best-epoch pick if scored) and move every other checkpoint of this run to the trash"
                  className="px-2.5 py-1 rounded-lg bg-red-500/10 border border-red-500/30 text-red-200 text-[0.6875rem] font-semibold">
                  🧹 Clean up this run
                </button>
                <span className="text-content-subtle text-[0.625rem]">
                  keeps final{bestEpoch?.available ? ' + 🏆 best' : ''} — the rest goes to the trash
                </span>
              </div>
            </div>
          )}

          {cloudGroups.length > 0 && (
            <div className="flex flex-col gap-2">
              <span className="text-content-muted text-[0.625rem] uppercase">
                ☁ Cloud checkpoints (synced locally — every epoch harvested from the pod)
              </span>
              {cloudGroups.map((g) => (
                <div key={`crun${g.run_id ?? 'unknown'}`}
                  className="flex flex-col gap-1 rounded-md border border-sky-500/20 bg-sky-500/[0.04] px-2 py-1.5">
                  {/* Identity header: which run made these epochs — same facts as
                      its Runs row, so "this final" ties back to "that run". */}
                  <div className="flex items-center gap-2 flex-wrap">
                    {g.run_id != null
                      ? <RunIdChip source="cloud" id={g.run_id} />
                      : <span className="text-sky-200 text-[0.6875rem]" aria-hidden>☁ run unknown</span>}
                    {g.run_id != null && (
                      <span className="text-content-muted text-[0.6875rem] font-medium">Run #{g.run_id}</span>
                    )}
                    <span className="text-content-subtle text-[0.625rem] uppercase">{groupFamLabel(g.train_type)}</span>
                    <DatasetVersionChip version={g.version} />
                    {g.status && (
                      <span className={`rounded border px-1.5 py-0.5 text-[0.625rem] ${g.active
                        ? 'text-sky-300 border-sky-400/40 bg-sky-500/10'
                        : g.status === 'done' ? 'text-emerald-300 border-emerald-400/40 bg-emerald-500/10'
                        : 'text-content-muted border-border bg-surface'}`}>
                        {g.status}
                      </span>
                    )}
                    <span className="text-content-muted text-[0.625rem] tabular-nums">
                      {[timeAgo(g.finished_at || g.created_at), g.gpu,
                        g.cost_estimate != null ? `$${g.cost_estimate}` : null]
                        .filter(Boolean).join(' · ')}
                    </span>
                    {g.run_id != null && (
                      <Link to={`/cloud#${runRowDomId('cloud', g.run_id)}`}
                        title="Jump to this run on the Runs page"
                        className="ml-auto px-1 py-0.5 text-sky-300 hover:text-sky-200 text-[0.6875rem] font-medium underline decoration-sky-300/40">
                        View in Runs ↗
                      </Link>
                    )}
                  </div>
                  {g.checkpoints.map((c) => (
                    <div key={`cr${c.run_id}-${c.filename}`} className="flex items-center gap-2 text-[0.6875rem] pl-1">
                      <span className={c.final ? 'text-green-400 font-semibold' : 'text-content'}>
                        {c.final ? '✓ final (training complete)' : `step ${c.step}`}
                      </span>
                      {c.active && (
                        <span className="text-sky-300/80 text-[0.625rem]">· run in progress</span>
                      )}
                      <button type="button"
                        onClick={async () => {
                          const d = await postTrain(`/api/dataset/${ds.currentId}/train/import`,
                            { filename: c.filename, cloud_run_id: c.run_id,
                              ...trainingRunSelection(checkpointBase, checkpointTrainType, c.variant || checkpointVariant) });
                          // Success must be VISIBLE: without the toast a working
                          // import looked like a dead button (user-observed).
                          if (d.ok === false) toastTrainError(d, 'Import failed');
                          else toast.success(d.note || `LoRA imported: ${d.dest || c.filename}`);
                          loadCheckpoints(checkpointBase, checkpointTrainType, checkpointVariant);
                        }}
                        title={c.active ? 'Import the latest synced save — the run keeps training' : 'Import this cloud checkpoint into ComfyUI'}
                        className="ml-auto px-2 py-0.5 rounded bg-primary/20 border border-primary/40 text-white">
                        Import → {checkpointLorasLabel}
                      </button>
                      {!c.active && (
                        <button type="button"
                          onClick={async () => {
                            if (!window.confirm(`Move « ${c.filename} » to the trash?\n\nRecoverable until you empty the trash in Settings.`)) return;
                            const d = await postTrain(`/api/dataset/${ds.currentId}/train/run-checkpoint/delete`,
                              { filename: c.filename, cloud_run_id: c.run_id,
                                ...trainingRunSelection(checkpointBase, checkpointTrainType, c.variant || checkpointVariant) });
                            if (d.ok === false) toastTrainError(d, 'Delete failed');
                            loadCheckpoints(checkpointBase, checkpointTrainType, checkpointVariant);
                          }}
                          title="Move this cloud save to the trash"
                          className="px-2 py-0.5 rounded bg-red-500/15 border border-red-500/40 text-red-300">
                          🗑
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}

          {ckLoaded && checkpoints.length === 0 && cloudCkpts.length === 0 && !status.in_progress && (
            <p className="m-0 text-content-subtle text-[0.625rem]">
              No {checkpointTypeLabel} checkpoint for base « {checkpointBaseLabel} » · {checkpointVariantDisplay} — run this exact recipe first.
            </p>
          )}

          {imported.length > 0 && (
            <div className="flex flex-col gap-1">
              <span className="text-content-muted text-[0.625rem] uppercase">
                In ComfyUI ({checkpointLorasLabel}) — delete the ones you no longer need
              </span>
              {imported.map((c) => (
                <div key={c.filename} className="flex items-center gap-2 text-[0.6875rem]">
                  {/* Source run of this deployed file: two look-alike LoRAs from
                      different runs are now distinguishable at a glance. Files
                      imported before run tagging carry no id → "run unknown". */}
                  {c.run_id != null
                    ? <RunIdChip source={c.run_source} id={c.run_id} />
                    : <span className="text-content-subtle text-[0.625rem]"
                        title="Imported before run tagging — its source run is unknown">run ?</span>}
                  <span className="text-content break-all">{c.label}</span>
                  {/* Retrofit signal: the file's REAL arch (read from its header)
                      contradicts this folder's family — a mislabelled deploy that
                      would test as a silent no-op. No auto-move; just flag it. */}
                  {c.arch_mismatch && (
                    <span
                      title={`This file is a ${c.arch_label || c.arch_mismatch} LoRA, not ${checkpointLorasLabel} — testing it here has NO effect (ComfyUI silently drops it). Delete it and re-import under the ${c.arch_label || c.arch_mismatch} family.`}
                      className="px-1.5 py-0.5 rounded bg-amber-500/15 border border-amber-500/40 text-amber-300 whitespace-nowrap">
                      ⚠ {c.arch_label || c.arch_mismatch} LoRA
                    </span>
                  )}
                  <button type="button" onClick={() => removeImported(c.filename, c.label)}
                    title={`Delete this LoRA from ComfyUI's ${checkpointLorasLabel} folder`}
                    className="ml-auto px-2 py-0.5 rounded bg-red-500/15 border border-red-500/40 text-red-300">
                    🗑 Delete
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </details>
      </CheckpointPortal>

      {preflightReport && (
        <PreflightModal report={preflightReport} datasetId={ds.currentId} ds={ds}
          onResolve={resolvePreflight} />
      )}

      {cloudDialog && (
        <CloudLaunchDialog
          datasetId={ds.currentId} trainType={trainType} variant={variant}
          base={base} steps={stepsN}
          keptCount={keptCount} cloudStatus={cloudStatus}
          onClose={() => setCloudDialog(false)} onLaunch={launchCloud} />
      )}

      {/* Resume ou Fresh : un run existe déjà pour ce (trigger, base). ai-toolkit
          reprendrait silencieusement son dernier checkpoint — on demande. */}
      {resumeAsk && (
        <div role="dialog" aria-modal="true" aria-label="Previous training run found"
          className="fixed inset-0 z-50 grid place-items-center bg-black/60 p-4"
          onKeyDown={(e) => { if (e.key === 'Escape') resolveResume(null); }}>
          <div className="w-full max-w-md rounded-xl border border-border bg-surface-overlay p-4 flex flex-col gap-3">
            <h3 className="m-0 text-content font-bold text-sm">
              ⚠ Previous run found ({resumeAsk.final ? 'complete' : 'stopped'} · step {resumeAsk.latest})
            </h3>
            <p className="m-0 text-content-muted text-[0.8125rem] leading-relaxed">
              Training will <b className="text-content">resume that LoRA</b> from its last
              checkpoint — anything it learned from images you have since removed stays in
              its weights. If the dataset changed, start fresh instead: the old run is
              archived (not deleted) and checkpoints already imported into ComfyUI are kept.
            </p>
            <div className="flex items-center gap-2 flex-wrap">
              <button type="button" onClick={() => resolveResume('fresh')}
                className="px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold">
                ↺ Start fresh
              </button>
              <button type="button" onClick={() => resolveResume('resume')}
                title="Continue the existing LoRA from its last checkpoint (only useful with a HIGHER step target)."
                className="px-3 py-1.5 rounded-lg border border-border bg-surface text-content text-sm hover:bg-surface-raised">
                ▶ Continue from step {resumeAsk.latest}
              </button>
              <button type="button" onClick={() => resolveResume(null)}
                className="ml-auto px-3 py-1.5 rounded-lg text-content-muted hover:text-content text-sm">
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {continueOpen && (
        <ContinueDialog
          context={`${checkpointBaseLabel} · ${checkpointVariantDisplay}`}
          where="local"
          checkpoints={checkpoints}
          bestStep={bestEpoch?.available ? bestEpoch.best_step : null}
          settings={{ save_every: advSave, sample_every: advSampleEvery,
            sample_prompts: adv?.sample_prompts }}
          busy={status.in_progress}
          onResolve={runContinue} />
      )}
    </div>
  );
}

const _FAMILY_LABEL = { zimage: 'Z-Image', krea: 'Krea 2', sdxl: 'SDXL', flux: 'FLUX.1', flux2klein: 'FLUX.2 Klein' };

function _fmtDuration(min) {
  if (min == null) return '—';
  if (min < 90) return `~${min} min`;
  const h = Math.floor(min / 60);
  const m = min % 60;
  return m ? `~${h} h ${m} min` : `~${h} h`;
}

/* Custom-base gate inside the cloud dialog: a custom base trains from a
   PRIVATE repo on the user's Hugging Face account (lds-base-<hash>). This
   section checks whether that repo already carries the base (cache-hit →
   launch straight away) and otherwise offers the ONE-TIME push — uploaded
   once, reused by every future cloud run, never public. */
function CustomBasePushSection({ datasetId, trainType, variant, base, onReadyChange }) {
  const [state, setState] = useState(null);      // last GET /custom-base payload
  const [checkError, setCheckError] = useState(null);
  const [pushBusy, setPushBusy] = useState(false);
  const [pushError, setPushError] = useState(null);
  const [pollNonce, setPollNonce] = useState(0);

  useEffect(() => {
    let alive = true;
    let timer;
    const tick = async () => {
      let d = null;
      try {
        const qs = new URLSearchParams({ train_type: trainType, base_model: base });
        if (variant) qs.set('variant', variant);
        const r = await fetch(`/api/dataset/${datasetId}/train/cloud/custom-base?${qs.toString()}`,
          { credentials: 'include' });
        d = await r.json().catch(() => ({}));
        if (!alive) return;
        if (!r.ok || d.ok === false) {
          setCheckError(d.error || `Could not check the custom base (HTTP ${r.status})`);
          d = null;
        } else {
          setCheckError(null);
          setState(d);
        }
      } catch {
        if (alive) setCheckError('Network error while checking the custom base');
      }
      // Keep polling while the background push is running (multi-GB upload).
      if (alive && d?.job?.state === 'running') timer = setTimeout(tick, 3000);
    };
    tick();
    return () => { alive = false; clearTimeout(timer); };
  }, [datasetId, trainType, variant, base, pollNonce]);

  const ready = !!state?.ready;
  useEffect(() => { onReadyChange(ready); }, [ready]); // eslint-disable-line react-hooks/exhaustive-deps

  const startPush = async (allowUnverified = false) => {
    setPushBusy(true);
    setPushError(null);
    try {
      const d = await postJson(`/api/dataset/${datasetId}/train/cloud/custom-base/push`, {
        train_type: trainType, variant, base_model: base,
        ...(allowUnverified ? { allow_unverified_weights: true } : {}),
      });
      if (d && d.ok === false) {
        const msg = String(d.error || 'Push failed');
        const marker = 'CUSTOM_WEIGHTS_UNVERIFIED: ';
        if (!allowUnverified && msg.includes(marker)) {
          const detail = msg.slice(msg.indexOf(marker) + marker.length);
          if (window.confirm(`${detail}\n\nPush anyway (force)?`)) return startPush(true);
        } else {
          setPushError(msg);
        }
        return;
      }
      setPollNonce((n) => n + 1);        // job started — begin polling its state
    } finally {
      setPushBusy(false);
    }
  };

  const job = state?.job || {};
  const pushing = pushBusy || job.state === 'running';
  const sizeLabel = state?.local_size_bytes != null ? ` (~${fmtBytes(state.local_size_bytes)})` : '';
  let body;
  if (checkError) {
    body = <p className="m-0 text-red-300 text-[0.75rem]">⚠ {checkError}</p>;
  } else if (!state) {
    body = <p className="m-0 text-content-muted text-[0.75rem]">Checking your custom base on Hugging Face…</p>;
  } else if (ready) {
    body = (
      <p className="m-0 text-emerald-300 text-[0.75rem]">
        ✓ Custom base found in your private repo <span className="font-mono">{state.repo_id}</span> —
        the pod downloads it with your HF token. Nothing to upload again.
      </p>
    );
  } else if (state.reason === 'no_token') {
    body = (
      <p className="m-0 text-amber-300 text-[0.75rem]">
        ⚠ Add your Hugging Face token (HF_TOKEN) in Settings ▸ API keys first — your custom
        base rides in a private repo on your account, and the pod needs the token to read it.
      </p>
    );
  } else if (state.reason === 'token_invalid') {
    body = (
      <p className="m-0 text-amber-300 text-[0.75rem]">
        ⚠ Your Hugging Face token was rejected — paste a valid HF_TOKEN in Settings ▸ API keys.
      </p>
    );
  } else if (pushing) {
    body = (
      <p className="m-0 text-sky-200 text-[0.75rem]">
        ⬆ Uploading your custom base{sizeLabel} to the private repo
        {state.repo_id ? <> <span className="font-mono">{state.repo_id}</span></> : null}…
        One-time upload — every future cloud run reuses it. Keep the app running.
      </p>
    );
  } else {
    const why = state.reason === 'size_mismatch'
      ? 'Your local custom base changed since it was pushed — push it again to update the private copy.'
      : state.reason === 'file_missing'
        ? 'The private repo exists but is missing the file this variant needs — push again to add it.'
        : 'This run uses custom weights the pod cannot download yet.';
    body = (
      <div className="flex flex-col gap-1.5">
        <p className="m-0 text-content-muted text-[0.75rem]">
          {why} Pushing uploads your custom base{sizeLabel} to a <b className="text-content">PRIVATE</b> repo
          on your Hugging Face account — one time; future cloud runs reuse it. It is never made public.
        </p>
        {!state.local_available && (
          <p className="m-0 text-amber-300 text-[0.75rem]">
            ⚠ The local file is unavailable ({state.local_reason || 'missing'}) — restore it to push.
          </p>
        )}
        {(pushError || job.state === 'error') && (
          <p className="m-0 text-red-300 text-[0.75rem]">⚠ {pushError || job.error}</p>
        )}
        <button type="button" onClick={() => startPush(false)}
          disabled={!state.local_available || pushBusy}
          className="w-fit px-3 py-1.5 rounded-lg border border-sky-500/50 bg-sky-500/10 text-sky-200 text-sm font-semibold disabled:opacity-40">
          ⬆ Push custom base to Hugging Face (one-time)
        </button>
      </div>
    );
  }
  return (
    <div className="rounded-lg border border-border bg-surface px-3 py-2">
      <p className="m-0 mb-1 text-content text-[0.75rem] font-semibold">
        Custom base: <span className="font-mono font-normal">{baseName(base)}</span>
      </p>
      {body}
    </div>
  );
}

/* Launch-time GPU speed picker. Fetches live vast.ai offers grouped by GPU
   class (slowest→fastest), each with price/h and an APPROXIMATE training time
   and total run cost for this dataset+family. Picking a tier rents the cheapest
   live offer of that class; the price cap in Settings still bounds what's shown.
   A custom base adds the push gate above the tiers (see CustomBasePushSection). */
function CloudLaunchDialog({ datasetId, trainType, variant, base, steps, keptCount, cloudStatus, onClose, onLaunch }) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);     // {tiers, steps, family, max_price_per_hour}
  const [selected, setSelected] = useState(null);
  const [launching, setLaunching] = useState(false);
  // Custom base ('' = official): the launch stays blocked until the private
  // repo on the user's HF account carries the base (pushed once, reused).
  const isCustomBase = !!String(base || '').trim();
  const [customBaseReady, setCustomBaseReady] = useState(!isCustomBase);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const qs = new URLSearchParams({ train_type: trainType });
        if (steps) qs.set('steps', String(steps));
        const r = await fetch(`/api/dataset/${datasetId}/train/cloud/offers?${qs.toString()}`,
          { credentials: 'include' });
        const body = await r.json().catch(() => ({}));
        if (!alive) return;
        if (!r.ok || body.ok === false) {
          setError(body.error || body.hint || `Could not load offers (HTTP ${r.status})`);
        } else {
          setData(body);
          if (body.tiers && body.tiers.length) setSelected(body.tiers[0].gpu_name);
        }
      } catch {
        if (alive) setError('Network error while loading GPU offers');
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, [datasetId, trainType, steps]);

  const go = async () => {
    if (!selected) return;
    setLaunching(true);
    try {
      await onLaunch(selected);      // owns its own error toasts
      onClose();
    } finally {
      setLaunching(false);
    }
  };

  const tiers = data?.tiers || [];
  const budget = cloudStatus?.monthly_budget || 0;
  const spent = cloudStatus?.month_spend || 0;

  return (
    <div role="dialog" aria-modal="true" aria-label="Choose cloud GPU speed"
      className="fixed inset-0 z-50 grid place-items-center bg-black/60 p-4"
      onKeyDown={(e) => { if (e.key === 'Escape') onClose(); }}>
      <div className="w-full max-w-lg rounded-xl border border-border bg-surface-overlay p-4 flex flex-col gap-3">
        <h3 className="m-0 text-content font-bold text-sm">
          <span aria-hidden>☁️</span> Choose GPU speed for this run
        </h3>

        {isCustomBase && (
          <CustomBasePushSection
            datasetId={datasetId} trainType={trainType} variant={variant}
            base={base} onReadyChange={setCustomBaseReady} />
        )}

        {loading && <p className="m-0 text-content-muted text-sm">Loading live GPU offers…</p>}
        {error && <p className="m-0 text-red-300 text-sm">⚠ {error}</p>}
        {!loading && !error && tiers.length === 0 && (
          <p className="m-0 text-content-muted text-sm">
            No GPU available under ${data?.max_price_per_hour}/h right now — raise the
            price cap in Settings, or try again shortly.
          </p>
        )}

        {tiers.length > 0 && (
          <div className="flex flex-col gap-1.5 max-h-[50vh] overflow-y-auto">
            {tiers.map((t) => (
              <label key={t.gpu_name}
                className={`flex items-center gap-3 rounded-lg border px-3 py-2 cursor-pointer transition-colors ${
                  selected === t.gpu_name
                    ? 'border-sky-400/70 bg-sky-500/10'
                    : 'border-border bg-surface hover:bg-surface-raised'}`}>
                <input type="radio" name="gpu-tier" className="accent-sky-400"
                  checked={selected === t.gpu_name}
                  onChange={() => setSelected(t.gpu_name)} />
                <span className="flex-1 min-w-0">
                  <span className="block text-content text-sm font-semibold truncate">
                    {t.gpu_name}
                    {t.gpu_ram_gb ? <span className="text-content-subtle font-normal"> · {t.gpu_ram_gb} GB</span> : null}
                  </span>
                  <span className="block text-content-subtle text-[0.75rem] tabular-nums">
                    {t.dph_total != null ? `$${t.dph_total.toFixed(3)}/h` : 'price n/a'}
                    {' · '}{_fmtDuration(t.est_minutes)}
                    {t.est_cost != null ? ` · ≈ $${t.est_cost.toFixed(2)} total` : ''}
                  </span>
                  {t.exceeds_cap && (
                    <span className="block text-amber-300 text-[0.6875rem]">
                      ⚠ Longer than the {Math.round((data?.max_runtime_minutes || 480) / 60)} h runtime cap — the run would be cut short (checkpoint rescued). Pick a faster GPU or raise the cap in Settings.
                    </span>
                  )}
                </span>
              </label>
            ))}
          </div>
        )}

        <p className="m-0 text-content-subtle text-[0.6875rem]">
          {(data?.steps ?? steps ?? '—')} steps · {_FAMILY_LABEL[data?.family || trainType] || (data?.family || trainType)}
          {keptCount != null ? ` · ${keptCount} img` : ''}
          {budget > 0 ? ` · this month: $${spent.toFixed(2)} of $${budget.toFixed(2)}` : ''}
          {'. '}Time & cost are approximate; the pod is auto-terminated when done.
        </p>

        <div className="flex items-center gap-2">
          <button type="button" onClick={go} disabled={!selected || launching || !customBaseReady}
            title={!customBaseReady ? 'Push the custom base to your Hugging Face account first' : undefined}
            className="px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
            {launching ? 'Launching…' : '☁️ Rent & train'}
          </button>
          <button type="button" onClick={onClose} disabled={launching}
            className="ml-auto px-3 py-1.5 rounded-lg text-content-muted hover:text-content text-sm disabled:opacity-40">
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
