import VastLink from '../common/VastLink';
// react-frontend/src/components/dataset/TrainingPanel.jsx
import { useEffect, useRef, useState } from 'react';
import { Dna, Drama, Eraser, FolderOpen, GraduationCap, Microscope, Package, Rocket, Save, SlidersHorizontal, Trash2, Trophy } from 'lucide-react';
import { createPortal } from 'react-dom';
import { Link } from 'react-router';
import { apiFetch, fetchWithCsrfRetry, getCsrfToken } from '../../api/fetchClient';
import { useCapabilities } from '../../context/CapabilitiesContext';
import { postJson } from '../../hooks/useDataset';
import { animeFamilyNote } from './animeFamilyNote.js';
import { dualCaptionsSupport } from './dualCaptions.js';
import { loadMergeOpen, saveMergeOpen } from './loraMerge.js';
import { maskedCarryOverAction, clearLegacyMasked } from './maskedMigration.js';
import ConceptFaceMaskField from './ConceptFaceMaskField';
import DenseModelsPanel from './DenseModelsPanel';
import LoraMergeTool from './LoraMergeTool';
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
import { stepsRecipeRefreshDelay } from '../../utils/stepsRecipeRefresh';
import { confirmableRetryFlag, postWithConfirmations } from '../../utils/trainingRefusals';
import {
  deployedFilenamesOf,
  orphanImportedCheckpoints,
  panelCheckpointDeployment,
} from '../../utils/checkpointDeployment';
import {
  describeZImageRecipe,
  isLongZImageTurboRun,
  ZIMAGE_TURBO_LONG_RUN_STEPS,
} from '../../utils/zimageTrainingRecipe';
import {
  trainingPresetDatasetKind,
  trainingPresetOwnsSteps,
} from '../../utils/trainingPresets';
import { useTrainingPresets } from './useTrainingPresets';
import { useSliderTraining } from './useSliderTraining';
import { useBestEpoch } from './useBestEpoch';
import { runConfirmableTrainingRequest } from '../../utils/trainingConfirmations';
import { continueAttemptOutcome } from '../../utils/continueOutcome';
import { HelpBadge } from '../../help/HelpMode';
import { requestHelpTip } from '../../help/helpTips';
import { useToast } from '../common/Toast';
import ContinueDialog from './ContinueDialog';
import { graphContinueRefusal } from './lineageContinue.js';
import RunLineageGraph from './RunLineageGraph';
import LineageDetailPanel from './LineageDetailPanel';
import LineageDiffPanel from './LineageDiffPanel';
import { toggleDiffSelection } from './lineageDetail.js';
import { UseDatasetCaptionsButton } from './UseDatasetCaptionsButton';
import TrainingProgress from './TrainingProgress';
import PreflightModal from './PreflightModal';
import { laneOfPayload, preflightUrl } from './preflightLane.js';
import {
  baseOptionSuffix, baseSelectionNote, basesForFamily,
  cloudUnsupportedFamilyReason, isCustomWeightsBase, looksAbsoluteBase,
} from './trainingFamilyScope.js';
import { failureView } from './trainingFailure';
import {
  MEMORY_KEYS, MEMORY_LABELS, memoryAdviceText, memoryIsOverridden, memoryPatchFor,
  memoryRiskLine, memoryStateLabel,
} from './memorySavingAdvice';
import { stopOutcomeMessage } from '../../utils/runSilence';
import SettingsLink from '../common/SettingsLink';
import { DatasetVersionChip, RunIdChip } from './RunIdentityBadges';
import {
  cloudGroupsFrom, localRunIdentity, runRowDomId,
} from '../../utils/runIdentity';
import {
  TRAINING_MODE_FULL_TRANSFORMER,
  TRAINING_MODE_LORA,
  denseQuantizeTarget,
  denseTurboWarning,
  fullTransformerBaseLabel,
  fullTransformerUnavailableReason,
  hfCloudTokenReadiness,
  isFullTransformerEligible,
  isFullTransformerRun,
  normalizeTrainingMode,
} from '../../utils/trainingMode.js';
import CloudLaunchDialog from './CloudLaunchDialog';
import CloudStopDialog from '../shared/CloudStopDialog';
import {
  CUSTOM_BASE_SENTINEL, DEFAULT_CUSTOM_FAMILIES, DenseBasePicker,
  FullTransformerAdvancedRecipe, FullTransformerArtifactNotice,
} from './FullTransformerRecipe';
// Compat re-export: tests and callers keep importing these from the panel.
export { DenseBasePicker, FullTransformerAdvancedRecipe } from './FullTransformerRecipe';

// Plancher dur / recommandé par famille — miroir de TRAIN_MIN_IMAGES côté serveur
// (le preflight reste l'autorité ; ceci ne sert qu'à désactiver le bouton tôt).
const TRAIN_MIN = { zimage: [12, 20], sdxl: [20, 30], krea: [15, 20], flux: [15, 20], flux2klein: [15, 20], anima: [15, 20] };
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

const defaultTrainingVariant = (family) => (
  family === 'krea' ? 'base' : family === 'flux2klein' ? '4b' : 'turbo'
);
// Absolute path (Windows drive, UNC, or POSIX) no longer MEANS « Custom
// weights… » on its own: the Krea 2
// selector lists the checkpoints installed on this machine, and the trainer
// addresses those by absolute path (a relative name on Krea is read as another
// family's base and ignored). The catalog decides — see trainingFamilyScope.js.
const customBaseMode = (value, info, family) => (
  isCustomWeightsBase(value, basesForFamily(info, family))
);
import { baseName, fmtBytes } from './panelFormatters';

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
const GROUP_FAMILY_LABEL = { zimage: 'Z-Image', krea: 'Krea 2', sdxl: 'SDXL', flux: 'FLUX.1', flux2klein: 'FLUX.2 Klein', anima: 'Anima' };
const groupFamLabel = (f) => GROUP_FAMILY_LABEL[f] || f || 'LoRA';

/** Panneau d'entraînement LoRA : lance l'UI ai-toolkit (pause ComfyUI),
 * affiche l'état, liste les checkpoints et importe celui choisi.
 * Poll régulier : c'est ce poll qui fait avancer la file (fin du courant → suivant). */
export default function TrainingPanel({ ds, keptCount, kind, onCheckpointsChange,
                                        checkpointHost = null,
                                        navigationPanel = null,
                                        onNavigationStateChange,
                                        onPanelOpenChange,
                                        // La section Training est-elle à l'écran ? Le panneau reste
                                        // MONTÉ quand une autre section s'affiche (le poll de la file
                                        // doit continuer), donc rien d'autre ne dit qu'on le regarde —
                                        // et le barème de steps ne se rafraîchit que quand on le regarde.
                                        sectionVisible = true,
                                        // « Continue anyway » ack from the readiness pastille: a
                                        // bypassable quality blocker was acknowledged → relax the
                                        // image-floor gate and carry allow_not_ready into the launch.
                                        allowNotReady = false }) {
  const isConcept = kind === 'concept';
  const isStyle = kind === 'style';
  const isConceptual = isConcept || isStyle;
  const { caps } = useCapabilities();
  const toast = useToast();
  // Normalizes like useDataset's own postJson: a non-2xx response (e.g. the
  // 409 {'error','hint'} the training routes return when ai-toolkit isn't
  // configured, or a 400 for a refused enqueue) must surface as `ok: false`
  // — previously this just returned the raw body, so callers checking
  // `d.ok === false` never saw the error (d.ok stayed undefined) and it was
  // silently dropped instead of reaching the confirm/toast below.
  const postTrain = async (path, body) => {
    try {
      // The shared transport, not bare fetch: a stale CSRF token replays
      // once with a fresh one instead of failing the launch — the exact
      // recovery every other mutation already had. The X-CSRFToken header
      // below is what arms that retry.
      const r = await fetchWithCsrfRetry(path, {
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
  const [status, setStatus] = useState({ in_progress: false, installed: true, queue: [], current: null });
  const [statusLoaded, setStatusLoaded] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [checkpointsOpen, setCheckpointsOpen] = useState(false);
  // ◉ Graph of this dataset's runs + checkpoints — now the DEFAULT surface of the
  // Checkpoints & LoRAs manager (a flat ☰ List stays one click away). Lazy: the
  // lineage is fetched when the graph view is showing and the browse filter/
  // dataset changes. {tree|loading|error} | null.
  const [datasetGraph, setDatasetGraph] = useState(null);
  // ⚙ Run details / ⇄ compare, opened straight FROM a checkpoint card. The
  // recipe panel and the two-run diff have always existed — one screen away,
  // in the Lineage graph — and "one screen away" is exactly where nobody
  // found them (user-reported: "you never see this simply from the dataset's
  // checkpoints"). Same panels, same data; these two states only give the
  // cards a handle on them. Stored as record IDS (the diff reducer's own
  // contract), resolved to nodes at render time from the lineage tree.
  const [ckptDetailId, setCkptDetailId] = useState(null);
  const [ckptDiffIds, setCkptDiffIds] = useState([]);
  // Which view the manager opens on. Persisted; defaults to the graph — the
  // showcase surface — so a checkpoint's whole genealogy is the first thing seen.
  const [checkpointsView, setCheckpointsView] = useState(() => {
    try { return localStorage.getItem('lds.checkpointsView') === 'list' ? 'list' : 'graph'; }
    catch { return 'graph'; }
  });
  const setCkView = (v) => {
    setCheckpointsView(v);
    try { localStorage.setItem('lds.checkpointsView', v); } catch { /* ignore */ }
  };
  // The merge disclosure is CONTROLLED, and persisted. `open` on a <details> is
  // DOM state, and the block lives inside CheckpointPortal below — every swap
  // between the portal host and the inline place unmounts it, which closed the
  // tool on top of emptying it. Held here, above the portal, it also survives
  // the remount itself and not only a reload.
  const [mergeOpen, setMergeOpen] = useState(loadMergeOpen);
  const toggleMerge = (event) => {
    event.preventDefault();
    const next = !mergeOpen;
    setMergeOpen(next);
    saveMergeOpen(next);
  };
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
  // FULL MODELS of this dataset — a separate lane on purpose (see
  // DenseModelsPanel): those are 26 GB transformers, not LoRA adapters, and
  // they must never inherit the adapter verbs the two lists above carry.
  // Empty on an older server and for every LoRA-only dataset → renders nothing.
  const [denseModels, setDenseModels] = useState([]);
  // {run_dir_bytes, cloud_staging_bytes, deployed_bytes, total_bytes}
  const [diskUsage, setDiskUsage] = useState(null);
  // {steps, kind, n_images, rationale} renvoyé par /train/checkpoints — le POURQUOI
  // du barème adaptatif, affiché avec le champ Steps (pédagogie, pas boîte noire).
  const [stepsInfo, setStepsInfo] = useState(null);
  // Miroir en ref : le calcul du délai de refetch lit le barème AFFICHÉ sans en
  // faire une dépendance de l'effet — sinon chaque réponse relancerait l'effet.
  const stepsInfoRef = useRef(stepsInfo);
  stepsInfoRef.current = stepsInfo;
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
  const [trainingMode, setTrainingMode] = useState(TRAINING_MODE_LORA);
  const [trainingModeBusy, setTrainingModeBusy] = useState(false);
  const [trainingModeError, setTrainingModeError] = useState('');
  const trainingModeRadioRefs = useRef({});
  const incompatibleModeFallbackRef = useRef('');
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
  const {
    slider, setSlider, sliderBusy, sliderDraft, setSliderDraft, sliderOn,
    sliderPromptsMissing, saveSlider, toggleSliderMode, saveSliderField,
  } = useSliderTraining({
    ds, postTrain, toastTrainError, base, trainType, variant,
    setBaseInfo, setAdv, setStepsInfo,
  });
  // Textarea des prompts de preview : état local (édition libre), sauvé au blur —
  // resynchronisé sur la valeur stockée canonique chaque fois que `adv` arrive/change.
  const [samplePromptsText, setSamplePromptsText] = useState('');
  // Same protection for the Krea Differential Guidance decimal: a controlled
  // number input cannot be edited naturally if every partial keystroke writes
  // through to the server ("" / "3." are not valid saved scales).
  const [differentialGuidanceScaleDraft, setDifferentialGuidanceScaleDraft] = useState('3');
  // Preview steps / CFG (#46). Drafts for the same reason as the scale above,
  // plus one of their own: EMPTY is a meaningful value here — it means "follow
  // the family default" — so the box must be allowed to hold '' while the
  // placeholder shows the number that default resolves to.
  const [sampleStepsDraft, setSampleStepsDraft] = useState('');
  const [sampleGuidanceDraft, setSampleGuidanceDraft] = useState('');
  const [trainTypeBusy, setTrainTypeBusy] = useState(false);

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
  }, [caps.training_visible]);

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

  // Surface the dual-captions option once, the first time Advanced opens — it's
  // a checkbox deep in the panel that's easy to never notice.
  useEffect(() => {
    if (advancedOpen && trainingMode !== TRAINING_MODE_FULL_TRANSFORMER) {
      requestHelpTip('dual-captions-advanced');
    }
  }, [advancedOpen, trainingMode]);

  // Same problem for 🎲 Use dataset captions: it sits below the Preview-prompts
  // textarea, in BOTH recipes, and nothing points at it. The TipHost shows one
  // card at a time and does not mark the loser seen, so on the very first LoRA
  // open dual-captions wins and this one arrives on the next open — in
  // full-model mode, where dual captions does not apply, it shows right away.
  useEffect(() => {
    if (advancedOpen) requestHelpTip('sample-prompts-from-dataset');
  }, [advancedOpen]);

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
        setTrainingModeError('');
        incompatibleModeFallbackRef.current = '';
        setBaseInfo(info); setBase(info.base || '');
        setVaePath(info.vae_path || '');
        setTePath(info.te_path || '');
        // Défaut family-aware : Krea sans variante persistée → Raw (reco officielle
        // « train on Raw, validate on Turbo ») ; FLUX.2 Klein → 4B (voie locale) —
        // y compris quand la variante PERSISTÉE vient d'une autre famille (un
        // dataset ex-Krea porte 'base', qui n'est pas une taille Klein valide) ;
        // les autres familles → Turbo.
        const fam = info.train_type || 'zimage';
        // A persisted absolute base the catalog does NOT offer is the
        // « Custom weights… » path → reopen that mode. One the catalog offers is
        // a normal dropdown pick and must stay in the dropdown.
        setCustomBase(customBaseMode(info.base || '', info, fam));
        const v = info.variant
          || (fam === 'krea' ? 'base' : fam === 'flux2klein' ? '4b' : 'turbo');
        const safeVariant = normalizeCheckpointVariant(fam, v);
        setVariant(safeVariant);
        setTrainType(info.train_type || 'zimage');
        setTrainingMode(normalizeTrainingMode(info.training_mode));
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
  // Une famille non énumérée par le serveur repart VIDE (→ le placeholder
  // family-aware plus bas), jamais avec le catalogue Z-Image.
  const currentBases = basesForFamily(baseInfo, trainType);
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
  // What the server says about the SELECTED catalog entry: a packed inference
  // export the trainer cannot load ('error'), or a low-precision cast that
  // trains from degraded weights ('warning'). Computed from the list the server
  // already annotated, so switching entries answers with no round trip.
  //
  // A path TYPED into « Custom weights… » is in no list, so it had no verdict
  // until the save or the launch read it — the one base you cannot pick from a
  // dropdown was also the only one whose refusal arrived after the dataset had
  // been exported and, on the cloud lane, after a GPU had been rented. It is
  // asked for here instead, once the typing settles.
  const [typedBaseAdvisory, setTypedBaseAdvisory] = useState(null);
  useEffect(() => {
    const value = String(base || '');
    if (!customBase || !looksAbsoluteBase(value)) { setTypedBaseAdvisory(null); return undefined; }
    let alive = true;
    // 500 ms: long enough that a typed path is not one request per character,
    // short enough to land before the hand reaches the Train button.
    const timer = setTimeout(() => {
      apiFetch(`/api/train/base-file-advisory?path=${encodeURIComponent(value)}`)
        // `for` is what makes a late answer harmless — see typedBaseNote.
        .then((d) => { if (alive) setTypedBaseAdvisory({ ...d, for: value }); })
        .catch(() => { if (alive) setTypedBaseAdvisory(null); });
    }, 500);
    return () => { alive = false; clearTimeout(timer); };
  }, [base, customBase]);
  const baseNote = baseSelectionNote(currentBases, base, typedBaseAdvisory);
  const baseNotTrainable = baseNote?.level === 'error';
  // Bloque l'entraînement si la base custom Z-Image n'est pas encore convertie,
  // si la base choisie ne peut pas être chargée du tout, ou si SDXL sans base
  // choisie (SDXL exige un checkpoint).
  const baseBlocksTrain = (needsConversion && !baseConverted) || baseNotTrainable;
  // The button tooltips used to state one cause only ("convert the base"), which
  // would now be wrong half the time.
  const baseBlockTitle = baseNotTrainable
    ? 'This checkpoint cannot be loaded for training — pick the bf16/fp16 version of it'
    : 'Convert the custom base first';
  const sdxlNeedsBase = trainType === 'sdxl' && !base;
  // Changement de type : réinitialise la base (les listes diffèrent ; SDXL → 1ère base réelle)
  // et PERSISTE la famille (choisie à la création, modifiable ici) pour que le menu
  // regroupé se ré-trie et que le format de caption suive.
  const onTypeChange = async (t) => {
    if (!t || t === trainType || trainTypeBusy) return;
    setTrainingModeError('');
    const previous = { trainType, base, variant, customBase };
    const nextVariant = defaultTrainingVariant(t);
    const list = baseInfo?.bases_by_type?.[t] || [];
    const nextBase = t === 'sdxl' ? (list[0]?.value || '') : '';
    const nextSelection = { trainType: t, baseModel: nextBase, variant: nextVariant };

    // Leaving Krea Raw while dense mode is selected is one recipe transition,
    // not two unrelated writes. Persist LoRA + the destination selection in the
    // backend's single transaction before the UI shows either change.
    if (trainingMode === TRAINING_MODE_FULL_TRANSFORMER
        && !isFullTransformerEligible(nextSelection)) {
      setTrainTypeBusy(true);
      setTrainingModeBusy(true);
      setTrainingModeError('');
      try {
        const saved = await ds.setDatasetTrainingMode?.(TRAINING_MODE_LORA, nextSelection);
        if (!saved) {
          setTrainingModeError('The new recipe was not saved. The full-model recipe and previous selection are unchanged.');
          return;
        }
        const savedType = saved.trainType ?? t;
        const savedBase = saved.baseModel ?? nextBase;
        const savedVariant = normalizeCheckpointVariant(savedType, saved.variant ?? nextVariant);
        setTrainingMode(normalizeTrainingMode(saved.trainingMode));
        setTrainType(savedType);
        setBase(savedBase);
        setCustomBase(customBaseMode(savedBase, baseInfo, savedType));
        setVariant(savedVariant);
        setPresetSel('');
        setStepsInfo(null);
        setBaseInfo((current) => current ? {
          ...current,
          training_mode: TRAINING_MODE_LORA,
          train_type: savedType,
          base: savedBase,
          variant: savedVariant,
        } : current);
        toast.info('This recipe is not compatible with full-model training. LoRA mode was saved.');
        try {
          const info = await ds.trainBaseInfo?.();
          if (info) {
            setBaseInfo(info);
            setAdv(info.train_settings || null);
            setBase(info.base || savedBase);
            setCustomBase(customBaseMode(info.base || savedBase, info, savedType));
            setVariant(normalizeCheckpointVariant(savedType, info.variant || savedVariant));
          }
          const checkpointData = await ds.listCheckpoints?.(savedBase, savedType, savedVariant);
          setStepsInfo(checkpointData?.recommended_steps_info || null);
        } catch {
          toast.warning('LoRA mode was saved, but the new family details will refresh later.');
        }
      } catch (error) {
        setTrainingModeError(error?.message || 'Could not save the new training recipe.');
        toast.error(error?.message || 'Could not change the training recipe');
      } finally {
        setTrainingModeBusy(false);
        setTrainTypeBusy(false);
      }
      return;
    }
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
      // Re-seed base/variant from the SERVER, which now remembers them per
      // family. The optimistic reset above only cleared this browser's state:
      // it looked fixed until the next reload re-read the shared column and
      // handed the other family's base straight back — the "change it and come
      // back" dance. The server is the truth; adopt what it says. `nextBase`
      // stays the fallback so SDXL — the one family that REQUIRES a checkpoint
      // — still lands on a usable one the first time it is picked.
      let seededBase = nextBase;
      let seededVariant = nextVariant;
      if (info) {
        setBaseInfo(info);
        setAdv(info.train_settings || null);
        seededBase = info.base || nextBase;
        seededVariant = normalizeCheckpointVariant(t, info.variant || nextVariant);
        setBase(seededBase);
        setCustomBase(customBaseMode(seededBase, info, t));
        setVariant(seededVariant);
      }
      const checkpointData = await ds.listCheckpoints?.(seededBase, t, seededVariant);
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
  const advTimestepDefault = adv?.default_timestep_type ?? (trainType === 'krea' ? 'linear' : (trainType === 'flux2klein' || trainType === 'anima') ? 'weighted' : (trainType === 'zimage' || trainType === 'flux') ? 'sigmoid' : null);   // miroir de _DEFAULT_TIMESTEP
  const advTimestepSupported = adv ? adv.timestep_type_supported !== false : trainType !== 'sdxl';
  const advTimestepChoices = adv?.timestep_type_choices ?? ['sigmoid', 'linear', 'weighted', 'shift'];
  const advOptimizer = adv?.optimizer ?? 'adamw8bit';
  const advOptimizerChoices = adv?.optimizer_choices ?? ['adamw8bit', 'adafactor', 'automagic', 'automagic2', 'prodigy'];
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
  // LoKr's decomposition factor is meaningful only once the network is LoKr.
  // null deliberately means ai-toolkit's automatic factor, not factor zero.
  const advLokrFactor = adv?.lokr_factor ?? null;
  const advLokrFactorChoices = adv?.lokr_factor_choices ?? [4, 8, 16, 32];
  // These four knobs are intentionally exposed only for Krea. They make the
  // reported Krea Raw LoKr community recipe inspectable, without pretending an
  // anecdotal setting is a universal control for every model family.
  const advKreaRecipeSupported = Boolean(adv?.krea_recipe_supported);
  const advContentOrStyle = adv?.content_or_style ?? null;
  const advContentOrStyleChoices = adv?.content_or_style_choices ?? ['balanced', 'style', 'content'];
  const advContentOrStyleDefault = adv?.content_or_style_default ?? adv?.default_content_or_style ?? 'balanced';
  const advDifferentialGuidance = Boolean(adv?.do_differential_guidance);
  const advEma = adv?.ema ?? 0;
  const advEmaChoices = adv?.ema_choices ?? [0.99, 0.999];
  const advDualCaptions = Boolean(adv?.dual_captions);
  // Concept face masking (issue #15). `supported` is the SERVER's answer (concept
  // only) rather than a second `kind === 'concept'` in JSX that could drift from it.
  const advMaskFaces = Boolean(adv?.mask_faces);
  const advMaskFacesSupported = Boolean(adv?.mask_faces_supported);
  const advMaskFacesConflict = Boolean(adv?.mask_faces_concept_conflict);
  // Memory strategy (issue #14). Tri-state per key: null = "Auto" (the family's
  // calibrated default), true/false = an explicit choice. `advMemEff` is what the
  // run will actually send, so the checkboxes show the truth whether or not the
  // user has touched anything.
  const advMemDefault = adv?.memory_saving_default ?? { quantize: true, quantize_te: true, low_vram: true };
  const advMemEff = adv?.memory_saving_effective ?? advMemDefault;
  const advMemTouched = memoryIsOverridden(adv?.memory_saving);
  const advMemStateLabel = memoryStateLabel(advMemEff);
  const advMemAdviceText = memoryAdviceText(adv?.memory_advice);
  // Non-null only when a saver THIS family's recipe relies on is switched off.
  // The server decides (one rule, shared with the preflight); the panel renders.
  const advMemRiskLine = memoryRiskLine(adv?.memory_risk, adv?.family_label);
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
  // Preview steps / CFG (#46). The DEFAULTS come from the server because they
  // depend on the family AND the variant (8/CFG 1 on a distilled base, 25/CFG 4
  // on an undistilled one) — hardcoding them here is how a placeholder starts
  // announcing a number the job never sends.
  const advSampleStepsDefault = adv?.sample_steps_default ?? 25;
  const advSampleGuidanceDefault = adv?.sample_guidance_default ?? 4;
  const advSampleStepsRange = adv?.sample_steps_range ?? [1, 60];
  const advSampleGuidanceRange = adv?.sample_guidance_range ?? [1, 20];
  const advSampleQualityOverridden = (adv?.sample_steps_stored != null
                                      || adv?.sample_guidance_stored != null);
  const advFamilyLabel = adv?.family_label ?? 'this model';
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
  useEffect(() => {
    setDifferentialGuidanceScaleDraft(String(adv?.differential_guidance_scale ?? 3));
  }, [adv?.differential_guidance_scale]);
  // Seed from the STORED override, never from the effective value: seeding with
  // the effective one would fill the box with the family default and turn
  // "following the default" into a frozen copy of today's number.
  useEffect(() => {
    setSampleStepsDraft(adv?.sample_steps_stored == null ? '' : String(adv.sample_steps_stored));
  }, [adv?.sample_steps_stored]);
  useEffect(() => {
    setSampleGuidanceDraft(
      adv?.sample_guidance_stored == null ? '' : String(adv.sample_guidance_stored));
  }, [adv?.sample_guidance_stored]);
  // Persist an EXPLICIT text. The blur handler below reads the state; the 🎲
  // draw cannot — it has just called setSamplePromptsText, and the state it
  // would read back is the previous render's, so it would save the old lines.
  const persistSamplePrompts = (text) => {
    const stored = (adv?.sample_prompts ?? []).join('\n');
    if (text === stored) return;                  // no-op → skip the round-trip
    saveAdv({ sample_prompts: text });            // server splits on newlines + trims
  };
  const saveSamplePrompts = () => persistSamplePrompts(samplePromptsText);
  const applySamplePrompts = (text) => {
    setSamplePromptsText(text);
    persistSamplePrompts(text);
  };
  // Kept images carry the captions the 🎲 draw samples; the payload the panel
  // already has is the only source, so the button costs no request.
  const datasetImages = ds.data?.images || [];
  const saveDifferentialGuidanceScale = () => {
    const stored = String(adv?.differential_guidance_scale ?? 3);
    if (differentialGuidanceScaleDraft === stored) return;
    const value = Number(differentialGuidanceScaleDraft);
    if (!Number.isFinite(value) || value < 0.1 || value > 10) {
      setDifferentialGuidanceScaleDraft(stored);
      toast.warning('Differential guidance scale must be between 0.1 and 10.');
      return;
    }
    saveAdv({ differential_guidance_scale: value });
  };

  // Preview steps / CFG. One helper for both: same contract (blank = auto),
  // same bounds source (the server ships them in the payload, so the box and
  // the validator can never disagree about what is accepted).
  const savePreviewSampleValue = (key, draft, setDraft, { integer }) => {
    const stored = adv?.[`${key}_stored`];
    const storedText = stored == null ? '' : String(stored);
    if (draft.trim() === storedText) return;               // no-op → no round-trip
    if (!draft.trim()) {
      saveAdv({ [key]: null });                            // blank = back to the family default
      return;
    }
    const [lo, hi] = adv?.[`${key}_range`] ?? (integer ? [1, 60] : [1, 20]);
    const value = Number(draft);
    if (!Number.isFinite(value) || value < lo || value > hi
        || (integer && !Number.isInteger(value))) {
      setDraft(storedText);
      toast.warning(integer
        ? `Preview steps must be a whole number between ${lo} and ${hi}.`
        : `Preview CFG must be between ${lo} and ${hi}.`);
      return;
    }
    saveAdv({ [key]: value });
  };
  const saveSampleSteps = () => savePreviewSampleValue(
    'sample_steps', sampleStepsDraft, setSampleStepsDraft, { integer: true });
  const saveSampleGuidance = () => savePreviewSampleValue(
    'sample_guidance', sampleGuidanceDraft, setSampleGuidanceDraft, { integer: false });

  const fullTransformerSelection = { trainType, variant, baseModel: base, customBase };
  const fullTransformerEligible = isFullTransformerEligible(fullTransformerSelection);
  const fullTransformerReason = fullTransformerUnavailableReason(fullTransformerSelection);
  const fullMode = trainingMode === TRAINING_MODE_FULL_TRANSFORMER;

  const onTrainingModeChange = async (nextValue) => {
    const nextMode = normalizeTrainingMode(nextValue);
    if (nextMode === trainingMode || trainingModeBusy) return false;
    if (nextMode === TRAINING_MODE_FULL_TRANSFORMER && !fullTransformerEligible) {
      toast.warning(fullTransformerReason || 'Full-model training is not available for this recipe.');
      return false;
    }
    setTrainingModeBusy(true);
    setTrainingModeError('');
    try {
      const saved = ds.setDatasetTrainingMode
        ? await ds.setDatasetTrainingMode(nextMode, {
            ...fullTransformerSelection,
            // Slider + recipe are one transaction. A separate /train/slider call
            // could leave the dataset half-switched when the second request failed.
            disableSliderForFullTransformer: nextMode === TRAINING_MODE_FULL_TRANSFORMER,
          })
        : null;
      if (!saved) {
        setTrainingModeError('The training mode was not saved. The previous selection is unchanged.');
        return false;
      }
      if (nextMode === TRAINING_MODE_FULL_TRANSFORMER && saved.slider?.enabled !== false) {
        setTrainingModeError('The server did not confirm that Slider was disabled atomically. The displayed selection is unchanged.');
        toast.error('The switch to full-model training was not confirmed with Slider disabled. Refresh and try again.');
        return false;
      }
      const canonicalMode = normalizeTrainingMode(
        typeof saved === 'string' ? saved : saved.trainingMode,
      );
      const canonicalType = saved.trainType ?? trainType;
      const canonicalBase = saved.baseModel ?? base;
      const canonicalVariant = normalizeCheckpointVariant(
        canonicalType,
        saved.variant ?? variant,
      );
      setTrainingMode(canonicalMode);
      setTrainType(canonicalType);
      setBase(canonicalBase);
      setCustomBase(customBaseMode(canonicalBase, baseInfo, canonicalType));
      setVariant(canonicalVariant);
      if (saved.slider) setSlider(saved.slider);
      setBaseInfo((current) => current ? {
        ...current,
        training_mode: canonicalMode,
        train_type: canonicalType,
        base: canonicalBase,
        variant: canonicalVariant,
        ...(saved.slider ? { slider: saved.slider } : {}),
      } : current);
      return true;
    } catch (error) {
      setTrainingModeError(error?.message || 'The training mode was not saved. The previous selection is unchanged.');
      toast.error(error?.message || 'Could not save the training mode');
      return false;
    } finally {
      setTrainingModeBusy(false);
    }
  };

  const onTrainingModeKeyDown = async (event) => {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
    event.preventDefault();
    const nextMode = fullMode ? TRAINING_MODE_LORA : TRAINING_MODE_FULL_TRANSFORMER;
    if (nextMode === TRAINING_MODE_FULL_TRANSFORMER && !fullTransformerEligible) {
      trainingModeRadioRefs.current[TRAINING_MODE_LORA]?.focus();
      return;
    }
    const changed = await onTrainingModeChange(nextMode);
    const focusMode = changed ? nextMode : trainingMode;
    trainingModeRadioRefs.current[focusMode]?.focus();
  };

  // Family, Krea variant and official/custom base can all change independently.
  // Never leave an incompatible dense mode hidden in persisted dataset state.
  useEffect(() => {
    if (!baseInfo || !fullMode || fullTransformerEligible) {
      incompatibleModeFallbackRef.current = '';
      return undefined;
    }
    const fallbackKey = `${trainType}\u0000${variant}\u0000${base}`;
    if (trainingModeBusy || incompatibleModeFallbackRef.current === fallbackKey) return undefined;
    incompatibleModeFallbackRef.current = fallbackKey;
    let alive = true;
    setTrainingModeBusy(true);
    setTrainingModeError('');
    (async () => {
      let saved = null;
      try {
        saved = await ds.setDatasetTrainingMode?.(
          TRAINING_MODE_LORA,
          fullTransformerSelection,
        );
      } catch (error) {
        if (alive) toast.error(error?.message || 'Could not save the fallback LoRA recipe');
      }
      if (!alive) return;
      if (saved) {
        setTrainingMode(TRAINING_MODE_LORA);
        setBaseInfo((current) => current ? {
          ...current,
          training_mode: TRAINING_MODE_LORA,
          train_type: saved.trainType ?? trainType,
          base: saved.baseModel ?? base,
          variant: saved.variant ?? variant,
        } : current);
        toast.info(`${fullTransformerReason || 'Recipe incompatible with full-model training'} LoRA mode was saved.`);
        return;
      }

      // Do not claim LoRA when persistence failed. Re-read the canonical recipe
      // and put every selector back on the server state instead.
      setTrainingModeError('Automatic fallback to LoRA failed. The saved recipe has been restored.');
      try {
        const info = await ds.trainBaseInfo?.();
        if (alive && info) {
          const serverType = info.train_type || 'zimage';
          const serverBase = info.base || '';
          const serverVariant = normalizeCheckpointVariant(
            serverType,
            info.variant || defaultTrainingVariant(serverType),
          );
          setBaseInfo(info);
          setTrainType(serverType);
          setBase(serverBase);
          setCustomBase(customBaseMode(serverBase, info, serverType));
          setVariant(serverVariant);
          setTrainingMode(normalizeTrainingMode(info.training_mode));
        }
      } catch { /* keep the visible error and current state if refresh also fails */ }
    })().finally(() => { if (alive) setTrainingModeBusy(false); });
    return () => { alive = false; };
    // Deps = les ENTREES de la synchro. Lister trainingModeBusy (que l'effet
    // pose) le ferait s'auto-relancer ; ds/toast sont stables en pratique.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseInfo, fullMode, fullTransformerEligible, fullTransformerReason,
    trainType, variant, base, ds.setDatasetTrainingMode]);



  const {
    presetSel, setPresetSel, presetBusy, presetFileRef, visiblePresets,
    researchedBuiltins, communityBuiltins, selPreset, savePreset,
    applyPreset, exportPreset, importPreset, deletePreset,
  } = useTrainingPresets({
    ds, kind, trainType, variant, trainTypeBusy, toast, postTrain,
    toastTrainError,
    // Panel-state effects of a successful apply. An arrow on purpose:
    // several of these setters are declared further down the body and
    // resolve at call time - passing them as values would be the TDZ trap
    // (dep arrays and arguments evaluate at render, closures do not).
    onApplied: async (d, payload, applied) => {
      setAdv(d.train_settings);
      if (payload.variant && payload.variant !== variant) setVariant(payload.variant);
      // Any recipe with a researched step policy owns that target. Do not let a
      // temporary cap from a previous run silently override it.
      if (trainingPresetOwnsSteps(applied)) {
        setStepsOverride('');
      }
      const checkpointData = await ds.listCheckpoints?.(base, trainType, payload.variant);
      setStepsInfo(checkpointData?.recommended_steps_info || null);
    },
  });
  // Confirmable launch refusals live in utils/trainingRefusals.js — the Runs hub
  // needs the SAME markers now that its ▶ Continue can resume on this machine.

  // Pre-launch sanity gate (server preflight): blockers stop with a toast,
  // warnings open the interactive PreflightModal (lists WHICH captions leak /
  // WHICH pairs duplicate, editable/rejectable in place) and await the user's
  // Start-anyway / Cancel. Unreachable preflight never blocks.
  const [preflightReport, setPreflightReport] = useState(null);
  const [hfCloudTokenIssue, setHfCloudTokenIssue] = useState(null);
  const preflightResolver = useRef(null);
  const resolvePreflight = (ok) => {
    setPreflightReport(null);
    preflightResolver.current?.(ok);
    preflightResolver.current = null;
  };
  // `lane` ('local' | 'cloud') decides which rows are relevant: a cloud run is
  // not going to care about this box's VRAM or torch build. trainType/variant
  // default to the panel's selection, but ▶ Continue overrides them with the
  // checkpoint's own family so the image floor is checked against the right one.
  // `onRefused(message)` — WHERE the blockers are said. Default: a toast, which
  // is right for the launch buttons (nothing is open to say it in). ▶ Continue
  // passes its own so the reason lands inside the still-open dialog, next to the
  // choices the user would otherwise have had to retype.
  const preflightOk = async ({
    lane, trainType: tt, variant: va, baseModel: bm, onRefused,
  } = {}) => {
    try {
      const r = await fetch(
        // `masked` rides along so the modal can say "rembg is missing — this run
        // trains unmasked" BEFORE the GPU (or the rented pod) is paid for, instead
        // of the fallback showing up as a flag on a progress view that disappears
        // when the run ends (issue #24, 1Tomber).
        preflightUrl(ds.currentId, {
          trainType: tt ?? trainType,
          variant: va ?? variant,
          baseModel: bm ?? base,
          trainingMode,
          lane,
          masked,
        }),
        { credentials: 'include' });
      const d = await r.json().catch(() => ({}));
      const tokenReadiness = hfCloudTokenReadiness(d);
      const checksDenseCloudToken = lane === 'cloud'
        && normalizeTrainingMode(trainingMode) === TRAINING_MODE_FULL_TRANSFORMER;
      if (checksDenseCloudToken && tokenReadiness.blocked) {
        const detail = tokenReadiness.detail
          || 'HF_CLOUD_TOKEN is missing, invalid, or does not have the required permissions.';
        setHfCloudTokenIssue(detail);
        if (onRefused) onRefused(detail);
        return false;
      } else if (checksDenseCloudToken && tokenReadiness.signaled) {
        setHfCloudTokenIssue(null);
      }
      if (!r.ok) return true;
      if (d.blockers?.length) {
        // A blocker set that is entirely bypassable quality guard-rails
        // (d.can_override) may proceed when the user ticked « Continue anyway »;
        // the launch carries allow_not_ready and the server re-checks. A physical
        // impossibility (can_override false) always stops here.
        if (!(d.can_override && allowNotReady)) {
          const msg = d.blockers.join('\n');
          if (onRefused) onRefused(msg); else toast.error(msg);
          return false;
        }
      }
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
  // The checkpoint the dialog opens ON, when it was opened from a ◉ Graph pill
  // (null = the plain Continue button → the dialog's historical "latest" default).
  const [continueInitialStep, setContinueInitialStep] = useState(null);
  // The LAST refusal, rendered INSIDE the dialog (utils/continueOutcome.js), and
  // the in-flight flag that keeps it from being dismissed mid-request.
  const [continueError, setContinueError] = useState(null);
  const [continueSubmitting, setContinueSubmitting] = useState(false);
  const runContinue = async (payload) => {
    // POST WITH THE DIALOG STILL OPEN. Closing first was a workaround for a toast
    // container that sat UNDER every modal (fixed: Toast.jsx is z-[10000]), and
    // it cost the user the whole form on every refusal — including the preflight
    // one, which is precisely the case where the answer is "fix this, then retry
    // with the same choices".
    if (!payload) { setContinueOpen(false); setContinueInitialStep(null); setContinueError(null); return; }
    // ONE dialog, two lanes: the chosen checkpoint either resumes on this machine
    // or is seeded onto a fresh cloud pod. Same payload, same guarded+confirmable
    // request helper — only the hook call differs.
    const lane = laneOfPayload(payload);
    const inCloud = lane === 'cloud';
    setContinueSubmitting(true);
    setContinueError(null);
    try {
      // Continuing trains on the LIVE dataset, which can have drifted since the run
      // started (images added, captions edited, triage left half-done) — so it gets
      // the same sanity gate as a fresh launch, on whichever lane it resumes. The
      // checkpoint's own family/variant, not the panel's current selection.
      // Its blockers are a REFUSAL about the dataset: they belong in the dialog,
      // not only in a toast that outlives the choices it was about.
      // Cancelling the warning report says nothing further (it IS the answer);
      // a hard blocker is a message the dialog shows.
      if (!(await preflightOk({ lane, trainType: checkpointTrainType,
        variant: checkpointVariant, baseModel: checkpointBase,
        onRefused: setContinueError }))) return;
      const { response, declined } = await runConfirmableTrainingRequest(
        (continueOpts) => (inCloud ? ds.continueTrainingInCloud : ds.continueTraining)(
          payload.extraSteps, checkpointBase, checkpointVariant, checkpointTrainType,
          { ...continueOpts, fromStep: payload.fromStep, overrides: payload.overrides,
            resumeMode: payload.resumeMode, stateBundleId: payload.stateBundleId,
            // The hook toasts refusals for its other callers; here the dialog
            // shows them, and two copies of one sentence a centimetre apart read
            // as a bug. Its SUCCESS toast is kept — the dialog is gone by then.
            quiet: true }),
        // maskedOpt, pas masked : tant que les reglages du dataset ne sont pas
        // charges, le panneau n'envoie RIEN et le serveur lit la valeur stockee.
        // Envoyer un defaut ici ecraserait un masquage desactive, sur un run paye.
        { masked: maskedOpt },
        (error) => confirmableRetryFlag(error, 'Continue anyway (force)'),
      );
      // The hooks never throw (they return {ok:false,error}); a decline is the
      // user's own answer and says nothing further.
      const outcome = continueAttemptOutcome({ response, declined });
      if (!outcome.close) { setContinueError(outcome.error); return; }
      setContinueOpen(false);
      setContinueInitialStep(null);
      setContinueError(null);
    } finally {
      setContinueSubmitting(false);
    }
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
    if (!existing.length) return 'new';                        // pas de run → lancement normal
    const latest = Math.max(...existing.map((c) => c.step));
    const final = existing.some((c) => c.final);
    return new Promise((resolve) => {
      resumeResolver.current = resolve;
      setResumeAsk({ latest, final });
    });
  };

  // Masked training (background at 10 %) — a PERSISTED DATASET setting, resolved
  // server-side (default ON; forced OFF for concept/style and slider mode, which
  // is why no client-side reset effect is needed here any more). It used to be a
  // per-browser localStorage preference the server only saw at launch: the
  // readiness badge could not warn about it, a phone reverted to the default, and
  // no run snapshot recorded it. `maskedMigration` handles the one-time carry-over
  // of the legacy browser key.
  const masked = adv?.masked !== false;
  const setMasked = (v) => saveAdv({ masked: !!v });
  // Sent on a launch only once the settings have LOADED: an unloaded panel must
  // never overwrite the dataset's stored value with an optimistic default.
  const maskedOpt = adv ? masked : undefined;
  const [maskedCarryOver, setMaskedCarryOver] = useState(false);
  useEffect(() => {
    const store = (() => { try { return globalThis.localStorage || null; } catch { return null; } })();
    const action = maskedCarryOverAction(store, adv);
    if (action === 'clear') clearLegacyMasked(store);
    setMaskedCarryOver(action === 'prompt');
  }, [adv]);
  const answerMaskedCarryOver = async (keep) => {
    const store = (() => { try { return globalThis.localStorage || null; } catch { return null; } })();
    await saveAdv({ masked: keep });
    clearLegacyMasked(store);
    setMaskedCarryOver(false);
  };
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
    let body = { base_model: base, variant, train_type: trainType, training_mode: trainingMode, masked: maskedOpt,
                 steps: stepsN,
                 ...(allowNotReady ? { allow_not_ready: true } : {}),
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
    let body = { at: schedAt, base_model: base, variant, train_type: trainType,
                 training_mode: trainingMode,
                 masked: maskedOpt, steps: stepsN,
                 ...(allowNotReady ? { allow_not_ready: true } : {}),
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
    setDenseModels(Array.isArray(data.dense_models) ? data.dense_models : []);
    setDiskUsage(data.disk_usage || null);
    setCkLoaded(true);
    onCheckpointsChange?.(
      (Array.isArray(data.checkpoints) ? data.checkpoints.length : 0)
      + (Array.isArray(data.imported) ? data.imported.length : 0),
    );
  };
  // ◉ Graph: the whole dataset's runs + their checkpoints as one genealogy
  // forest, scoped to the browse filter's family/variant so it matches the list.
  // Reuses the exact component the Runs hub draws (RunLineageGraph) — no copy.
  const loadDatasetGraph = async () => {
    setDatasetGraph({ loading: true });
    try {
      setDatasetGraph({ tree: await fetchDatasetLineage() });
    } catch {
      setDatasetGraph({ error: 'Could not load this dataset’s run graph.' });
    }
  };
  // Bare fetch of the lineage tree (used by the initial load AND the child's
  // refetch after an inline import, so a freshly-deployed pill flips to testable).
  const fetchDatasetLineage = async () => {
    const qs = new URLSearchParams();
    if (checkpointTrainType) qs.set('train_type', checkpointTrainType);
    if (checkpointVariant) qs.set('variant', checkpointVariant);
    const r = await fetch(`/api/dataset/${ds.currentId}/train/lineage?${qs.toString()}`,
      { credentials: 'include' });
    if (!r.ok) throw new Error('unavailable');
    return r.json();
  };
  // The lineage tree, fetched ON DEMAND when a card asks for a panel before
  // the Graph view ever loaded it (the auto-load below only fires for the
  // graph). Cached into the same state the graph uses — one tree, two readers.
  const ensureLineageTree = async () => {
    if (datasetGraph?.tree?.nodes) return datasetGraph.tree;
    try {
      const tree = await fetchDatasetLineage();
      setDatasetGraph({ tree });
      return tree;
    } catch {
      toast.error('Could not load this dataset’s run lineage.');
      return null;
    }
  };
  const lineageNodeById = (tree, recordId) =>
    (tree?.nodes || []).find((n) => n.record_id === recordId) || null;
  const openRunDetails = async (recordId) => {
    const tree = await ensureLineageTree();
    if (!tree) return;
    if (!lineageNodeById(tree, recordId)) {
      toast.error('This run is not in the lineage tree (filtered out by the current base/family).');
      return;
    }
    setCkptDetailId(recordId);
  };
  const toggleRunCompare = async (recordId) => {
    const tree = await ensureLineageTree();
    if (!tree) return;
    if (!lineageNodeById(tree, recordId)) {
      toast.error('This run is not in the lineage tree (filtered out by the current base/family).');
      return;
    }
    setCkptDiffIds((cur) => toggleDiffSelection(cur, recordId));
  };

  // The manager is "open" when portaled to its sidebar host, or expanded inline.
  const checkpointManagerOpen = Boolean(checkpointHost) || checkpointsOpen;
  // Auto-load the lineage the moment the graph view is the one showing (default),
  // and reload it when the browse filter/dataset changes — so the graph is ready
  // without a manual click. The list view keeps its own loadCheckpoints effect.
  useEffect(() => {
    if (!caps.training_visible || !ds.currentId || !baseInfo) return;
    if (checkpointsView !== 'graph' || !checkpointManagerOpen) return;
    loadDatasetGraph();
  }, [checkpointsView, checkpointManagerOpen, checkpointBase, checkpointTrainType, // eslint-disable-line react-hooks/exhaustive-deps
      checkpointVariant, ds.currentId, baseInfo, caps.training_visible]);

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
  // keptCount EST une dépendance : le barème est calculé serveur à partir du
  // nombre d'images gardées, et ce panneau ne se démonte jamais (sections
  // masquées, pas démontées) — sans lui, curer puis revenir sur Train affichait
  // encore la recette calculée pour l'ancien compte, jusqu'à ce qu'un réglage
  // d'entraînement soit touché. Voir utils/stepsRecipeRefresh.js pour le quand.
  useEffect(() => {
    if (!caps.training_visible || !ds.currentId || !baseInfo) return undefined;
    const delay = stepsRecipeRefreshDelay(sectionVisible, stepsInfoRef.current, keptCount);
    if (delay == null) return undefined;
    let alive = true;
    const timer = setTimeout(() => {
      ds.listCheckpoints(base, trainType, variant).then((data) => {
        if (alive) setStepsInfo(data?.recommended_steps_info || null);
      }).catch(() => { /* keep the last truthful rationale */ });
    }, delay);
    return () => { alive = false; clearTimeout(timer); };
  }, [base, trainType, variant, keptCount, sectionVisible, ds.currentId, baseInfo, caps.training_visible]); // eslint-disable-line react-hooks/exhaustive-deps
  const removeImported = async (filename, label) => {
    // Guard-rail: this LoRA may be the one the Studio's ★ best settings point to —
    // deleting it silently breaks the saved winning combo.
    // The pin is stored PER FAMILY, so the payload flattens it to a list — reading
    // `best_settings.lora_filename` only ever matched the legacy flat shape, which
    // meant this warning had quietly stopped firing on modern pins.
    const tail = (s) => String(s).split(/[\\/]/).pop();
    const isBest = (ds.data?.best_settings_loras || [])
      .some((p) => tail(p) === tail(filename));
    // It goes to the TRASH (delete_imported_checkpoint), so the confirmation must
    // not claim a permanent deletion — and must say what it does NOT touch.
    const msg = isBest
      ? `⚠ “${label}” is the LoRA saved as this dataset's ★ BEST SETTINGS in the Test Studio.\n\nUndeploy it anyway? The saved combo will stop working.`
      : `Undeploy “${label}” from ComfyUI's ${checkpointLorasLabel} folder?\n\nOnly the ComfyUI copy goes to the trash (recoverable until you empty it in Settings). Any training save it came from is kept.`;
    if (!window.confirm(msg)) return;
    await ds.deleteCheckpoint(filename, checkpointTrainType, checkpointVariant);
    loadCheckpoints();
  };
  // ⏏ Undeploy, in place, next to the checkpoint it came from — the symmetric
  // counterpart of 📦 Import and the SAME operation the ◉ Graph offers: the
  // route, body and target file all come from the shared helper (which derives
  // them from checkpointDeleteTarget), so no second undeploy logic exists here.
  // Only the ComfyUI copy goes to the trash; the training save stays, which is
  // why the confirmation says so and the button is not painted as destruction.
  const [undeploying, setUndeploying] = useState(null);   // deployed filename
  const undeployCheckpoint = async (dep) => {
    if (!dep?.undeploy) return;
    const msg = dep.confirmMessage();
    if (msg && !window.confirm(msg)) return;
    setUndeploying(dep.undeploy.filename);
    try {
      const d = await postTrain(`/api/dataset/${ds.currentId}/${dep.undeploy.path}`,
                                dep.undeploy.body);
      if (d.ok === false) toastTrainError(d, 'Undeploy failed');
      else toast.success(`Undeployed from ComfyUI — the training save is kept, you can deploy it again: ${dep.undeploy.filename}`);
    } finally {
      setUndeploying(null);
      loadCheckpoints(checkpointBase, checkpointTrainType, checkpointVariant);
    }
  };
  /* The ONE control that says where a checkpoint stands with ComfyUI: 📦 Import
     when it isn't deployed, "✓ Deployed" + ⏏ Undeploy when it is. Rendered
     identically at both call sites (this run's saves and the cloud runs' saves)
     so the two lists cannot drift apart — and identically to the ◉ Graph pills,
     which is the point of this alignment. A deployed checkpoint no longer offers
     a second "Import →" that would just overwrite itself. */
  const renderDeployControl = (c, { onImport, importTitle } = {}) => {
    const dep = panelCheckpointDeployment(c, {
      trainType: checkpointTrainType,
      variant: c.variant || checkpointVariant,
      baseModel: checkpointBase,
    });
    if (!dep.deployed) {
      return (
        <button type="button" onClick={onImport} title={importTitle}
          className="ml-auto px-2 py-0.5 rounded bg-primary/20 border border-primary/40 text-white">
          Import → {checkpointLorasLabel}
        </button>
      );
    }
    const deployedName = dep.undeploy?.filename || dep.pill.deployed_filename || '';
    return (
      <span className="ml-auto flex items-center gap-1">
        <span title={`Already in ComfyUI's ${checkpointLorasLabel} folder${deployedName ? ` as “${deployedName}”` : ''}`}
          className="px-2 py-0.5 rounded border border-emerald-500/40 bg-emerald-600/10 text-emerald-200">
          ✓ Deployed
        </span>
        {dep.undeploy && (
          <button type="button" onClick={() => undeployCheckpoint(dep)}
            disabled={undeploying === dep.undeploy.filename}
            title={dep.undeploy.title}
            className="px-2 py-0.5 rounded border border-emerald-500/40 bg-emerald-600/5 text-emerald-200/90 hover:bg-emerald-600/20 disabled:opacity-50">
            {undeploying === dep.undeploy.filename ? '⏏ Undeploying…' : `⏏ ${dep.undeploy.label}`}
          </button>
        )}
      </span>
    );
  };
  // The deployed LoRAs that no checkpoint row above accounts for — all that is
  // left for the "Also in ComfyUI" list once every claimed file is actionable in
  // place. Computed from the rows the page is actually showing, so a failed join
  // leaves the file listed instead of unreachable.
  const otherImported = orphanImportedCheckpoints(
    imported,
    deployedFilenamesOf(checkpoints, cloudCkpts,
                        ...cloudGroups.map((g) => g.checkpoints || [])));
  const doPrepareBase = async () => {
    await ds.prepareBase(base);
    const info = await ds.trainBaseInfo();
    if (info) setBaseInfo(info);
  };

  const { bestEpoch, bestEpochBusy, findBestEpoch } = useBestEpoch({
    ds, postTrain, toastTrainError, checkpointBase, checkpointTrainType,
    checkpointVariant,
  });

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
  // The dense recipe's base, computed from the SAME selection the backend
  // resolves from — and the untested-Turbo notice that rides with it, shown
  // before the GPU is rented rather than in the run log afterwards.
  const denseBaseSummary = fullTransformerBaseLabel({
    baseModel: base, baseLabel, variant });
  const denseTurboNotice = denseTurboWarning({ baseModel: base, variant });
  const effectiveTargetSteps = stepsN ?? stepsInfo?.steps ?? null;
  const zimageTurboLongRun = trainType === 'zimage'
    && isLongZImageTurboRun({ variant, steps: effectiveTargetSteps });
  const typeLabel = trainFamilyLabel(trainType);
  const stepsRecipeType = stepsInfo?.train_type || trainType;
  const stepsRecipeFamily = stepsInfo?.family_label || trainFamilyLabel(stepsRecipeType);
  const stepsRecipeVariant = stepsInfo?.variant_label
    || checkpointVariantLabel(stepsRecipeType, stepsInfo?.variant || variant);
  const checkpointBasesRaw = basesForFamily(baseInfo, checkpointTrainType);
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
  // ▶ Continue from a ◉ Graph checkpoint pill — the SAME gesture as the Runs hub,
  // routed through THIS panel's local resume flow (no second call path): remember
  // the clicked step and open the shared dialog on it. The dialog resumes the
  // active checkpoint set, so a pill from another family/base has no step to
  // resume here: say so instead of silently falling back to the latest save.
  const continueFromGraphCheckpoint = (node, pill) => {
    const step = pill?.step ?? null;
    // The refusal must state the reason that is TRUE for this pill (a foreign
    // run identity vs a save this machine simply doesn't hold) — the rule lives
    // in lineageContinue.js, JSX-free and unit-tested.
    const refusal = graphContinueRefusal(node, pill, {
      steps: checkpoints.map((c) => c.step),
      trainType: checkpointTrainType, variant: checkpointVariant, base: checkpointBase,
      familyLabel: checkpointTypeLabel, variantLabel: checkpointVariantDisplay,
    });
    if (refusal) { toast.warning(refusal); return; }
    setContinueInitialStep(step);
    setContinueError(null);
    setContinueOpen(true);
  };
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
  // Per-(dataset, family) as before — but PLURAL now: two same-family runs may
  // train this dataset at once (settings A/B). A run without train_type (older
  // server payload) matches any family, preserving the previous behavior.
  const cloudActivesHere = actives.filter((a) => a.dataset_id === ds.currentId
    && (!a.train_type || a.train_type === trainType));
  // Which of them the card + progress + checkpoint link follow. View state
  // only (never server state): defaults to the newest, snaps back when the
  // selected run leaves the active list.
  const [selectedCloudRunId, setSelectedCloudRunId] = useState(null);
  const [cloudStopTarget, setCloudStopTarget] = useState(null);
  const cloudActiveHere = cloudActivesHere.find((a) => a.run_id === selectedCloudRunId)
    || cloudActivesHere[cloudActivesHere.length - 1] || null;
  // Same dialog the Runs hub uses: stopping from THIS panel used to skip the
  // "Do not rent this machine again" tick, which is the option you want when
  // a pod is stuck on boot and you are watching it here rather than on Runs.
  const doStopCloud = async (run, { banHost = false } = {}) => {
    if (!run) return;
    setCloudStopTarget(null);
    const d = await postJson('/api/dataset/train/cloud/stop',
      banHost ? { run_id: run.run_id, ban_host: true } : { run_id: run.run_id });
    // Never leave a stop unanswered: a pod that could not be terminated keeps
    // billing, and the message names the instance.
    const m = stopOutcomeMessage(d);
    if (m.level === 'error') toast.error(m.text, 20000);
    else toast.info(m.text, m.level === 'warn' ? 12000 : undefined);
  };
  const cloudLastHere = cloudStatus.last
    && cloudStatus.last.dataset_id === ds.currentId
    && (!cloudStatus.last.train_type || cloudStatus.last.train_type === trainType)
    ? cloudStatus.last
    : null;

  // A finished run writes its checkpoints to disk, but nothing re-read the list:
  // both polls only refreshed their own status, so a LoRA that had just finished
  // training stayed invisible until the browse filter changed or the page was
  // reloaded — reported as "sometimes I have to refresh the page to see the LoRAs".
  // Watch the run concerning THIS dataset (local OR cloud) end, then re-read what
  // it produced. Kept as a BOOLEAN dependency on purpose: the cloud poll returns a
  // fresh object every 5s, so depending on it directly would re-run this constantly
  // instead of only on the transition that actually means "new files exist".
  const runActiveHere = Boolean(
    (status.in_progress && status.current?.dataset_id === ds.currentId) || cloudActiveHere,
  );
  const runWasActiveHere = useRef(false);
  useEffect(() => {
    if (runWasActiveHere.current && !runActiveHere) {
      loadCheckpoints();
      // The graph shows the same checkpoints as pills; refresh it too when it is
      // the visible view, otherwise the two views would disagree.
      if (checkpointsView === 'graph' && checkpointManagerOpen) loadDatasetGraph();
    }
    runWasActiveHere.current = runActiveHere;
  }, [runActiveHere]); // eslint-disable-line react-hooks/exhaustive-deps
  // Multi-family parallelism is safe again: each cloud run's monitor builds its
  // job config from its OWN stamped family/variant, not the shared dataset row
  // (backend _run_config_dataset — fix for the 2026-07-14 incident). A run of
  // the SAME family no longer disables the button either: the server's
  // PARALLEL_RUN: refusal is a confirmable question ("second pod, billed
  // separately — launch anyway?") that postWithConfirmations relays, and a
  // disable here made it unreachable — reported as "I still cannot run two
  // trainings on the same dataset".
  // Single source of truth for WHY « ☁ Train in cloud » is disabled — most
  // fundamental cause first (family unsupported > custom weights > too few
  // images > global limit). Drives BOTH the tooltip
  // AND the always-visible reason line below: a disabled button must state its
  // reason without a hover (the owner lost time guessing on a greyed SDXL button
  // whose only explanation lived in a title attribute).
  // Slider mode floors the image requirement at the substrate minimum (the
  // preflight/assert_trainable stay authoritative server-side).
  const trainMinFloor = sliderOn ? TRAIN_MIN_SLIDER[0] : (TRAIN_MIN[trainType]?.[0] ?? 12);
  // « Continue anyway » relaxes the image-floor gate (a bypassable quality blocker
  // the user acknowledged); a physical impossibility never yields a truthy ack.
  const belowFloor = keptCount < trainMinFloor && !allowNotReady;
  const cloudTooFewImages = belowFloor;
  const cloudLimitReached = actives.length >= (cloudStatus.limit || 1);
  // Familles que la voie cloud ne sert pas — miroir des refus AVANT réservation
  // côté serveur. Anima y manquait : au-dessus du plancher d'images le bouton
  // s'activait et n'était refusé qu'après le clic.
  const cloudFamilyBlock = cloudUnsupportedFamilyReason(trainType);
  const cloudDisabledReason =
    !caps.cloud_training
      ? 'Cloud training needs a vast.ai API key — add it in Settings'
    : fullMode && !fullTransformerEligible
      ? (fullTransformerReason || 'Full-model training is a Krea 2 recipe')
    : cloudFamilyBlock
      ? cloudFamilyBlock
    : (vaePath || tePath)
      ? 'Custom VAE/text-encoder overrides are local-only — clear them in Advanced options to train in the cloud'
    : customWeightsEmpty
      ? 'Enter the path to your custom weights .safetensors first'
    : baseNotTrainable
      ? 'This checkpoint cannot be loaded for training — pick the bf16/fp16 version of it'
    : baseBlocksTrain
      ? 'Convert the custom base first — the cloud lane pushes the converted copy to your Hugging Face account'
    : cloudTooFewImages
      ? `Only ${keptCount} image(s) kept — the cloud minimum for ${sliderOn ? 'a slider' : typeLabel} is ${trainMinFloor}`
    : cloudLimitReached
      ? `Cloud run limit reached (${actives.length}/${cloudStatus.limit || 1}) — stop one or raise the limit in Settings`
    : null;

  // Informational pointer for anime datasets (see animeFamilyNote.js). Reads the
  // subject type straight off the dataset payload and Anima's real availability off
  // base-info — no local rule, no forced selection, no launch blocked.
  const animeNote = animeFamilyNote({
    subjectType: ds.data?.subject_type,
    trainType,
    animaSupported: baseInfo?.anima_supported,
  });

  // ▶ Continue — WHERE it runs. A checkpoint is just a file: a run trained on this
  // machine can be finished on a rented GPU (the file is seeded onto a fresh pod)
  // and a cloud epoch mirrored here can be finished locally. Each lane states its
  // own reason when it can't be used, so the dialog never shows a dead option.
  // The cloud reason reuses cloudDisabledReason — the app's single source of truth
  // for "why the cloud lane is closed" (family, custom weights, limits, budget).
  const continueLanes = {
    local: fullMode
      ? { available: false,
          reason: 'Full-model training is cloud-only. Switch back to LoRA to continue this local checkpoint.' }
      : caps.aitoolkit?.valid === false
      ? { available: false,
          reason: 'Local training needs ai-toolkit — set it up in Settings, or continue in the cloud.' }
      : status.in_progress
        ? { available: false,
            reason: 'A training is already running on this machine — continue in the cloud, or wait for it to finish.' }
        : { available: true },
    cloud: fullMode
      ? { available: false,
          reason: 'Continuing a full model is not available in this MVP. Switch back to LoRA to continue a LoRA.' }
      : !caps.cloud_training
      ? { available: false,
          reason: 'Cloud training needs a vast.ai API key — add it in Settings.' }
      : cloudDisabledReason
        ? { available: false, reason: cloudDisabledReason }
        : { available: true },
  };
  // The lane the dialog OPENS on = the lane the source checkpoint was trained in
  // (a cloud epoch mirrored into the run folder defaults to Cloud). Unknown step
  // → the newest save's provenance, the run the plain Continue button resumes.
  const laneOfStep = (step) => {
    const c = (step != null && checkpoints.find((k) => k.step === step))
      || checkpoints[checkpoints.length - 1];
    return c?.source === 'cloud' ? 'cloud' : 'local';
  };

  // Launch-time GPU speed picker: the ☁️ button opens a dialog that lists live
  // vast.ai offers by speed (price/h + approx time + cost); the chosen class is
  // forwarded as gpu_name. launchCloud carries the POST + the MISMATCH_CAPTION
  // retry that used to live inline in the button handler.
  const [cloudDialog, setCloudDialog] = useState(false);
  const launchCloud = async (gpuName) => {
    // A cloud launch spends real money, so it gets the SAME sanity gate as a local
    // one — minus the rows about this machine's GPU, which no rented pod will use.
    if (!(await preflightOk({ lane: 'cloud' }))) return false;
    let body = {
      ...cloudTrainingLaunchPayload({
        baseModel: base, variant, trainType, trainingMode,
        masked: maskedOpt, steps: stepsN, gpuName,
      }),
      ...(allowNotReady ? { allow_not_ready: true } : {}),
    };
    // postWithConfirmations, NOT a hand-rolled loop on `d.ok === false`.
    //
    // That loop could never run. Every confirmable refusal leaves the service as a
    // RuntimeError, which the route maps to HTTP 409 with `{error: "..."}` -- a
    // body carrying no `ok` key at all -- and postJson THROWS on any non-2xx
    // rather than returning it. So the await raised, everything below it was dead
    // code, and the user got the raw refusal text as an error toast with no way to
    // answer the question it was asking. Reported on the parallel-run one: "I
    // still cannot run two trainings on the same dataset", on an install whose
    // backend supported it perfectly.
    //
    // This helper is what the Runs hub and the canvas already use for these same
    // refusals: it catches, asks, and retries carrying the flag -- and refuses to
    // ask twice for a flag it already sent, so a refusal that survives its own
    // confirmation ends instead of looping.
    let d;
    try {
      d = await postWithConfirmations(
        (b) => postJson(`/api/dataset/${ds.currentId}/train/cloud`, b),
        body, 'Train anyway (force)');
    } catch (e) {
      toastTrainError({ ...(e?.body || {}), error: e?.message }, 'Cloud training failed');
      return false;
    }
    // The dialog closes on success and the panel's own launch view only appears
    // on the next 5 s poll — a silent close read as "nothing happened". Name the
    // run that now exists and where its launch steps are visible.
    if (d) {
      toast.info(d.run_id != null
        ? `Cloud run #${d.run_id} created — follow its launch on the Runs page.`
        : 'Cloud run created — follow its launch on the Runs page.');
    }
    return !!d;
  };

  if (!caps.training_visible) {
    return (
      <div className="flex items-center gap-2 rounded-lg border border-border bg-surface p-3 text-content-muted text-sm">
        <GraduationCap aria-hidden="true" className="h-4 w-4 shrink-0" />
        Training needs ai-toolkit (local GPU) or a <VastLink className="underline" /> API key (cloud) — set either in Settings.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-indigo-500/30 bg-indigo-500/5 p-3">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-content font-semibold text-sm">
          <GraduationCap aria-hidden="true" className="h-4 w-4" /> {fullMode ? 'Full-model training' : 'LoRA Training'} ({typeLabel})
        </span>
        {!status.installed && (
          <span className="text-amber-300 text-[0.6875rem]">ai-toolkit not ready — point to its Python (its venv/Scripts/python.exe) in Settings › Local tools</span>
        )}
        {status.in_progress
          ? <span className="ml-auto flex items-center gap-2">
              <span aria-live="polite" className="text-indigo-300 text-[0.6875rem]">
                <span aria-hidden>⏳</span> {status.current?.name ? `“${status.current.name}” running` : 'running'} — ComfyUI paused
              </span>
              {/* Full progress bar, loss curve and samples live on the Runs hub —
                  this panel's own TrainingProgress only covers THIS dataset. */}
              <Link to="/cloud" title="Open the Runs page — full progress, loss curve and samples"
                className="min-h-10 lg:min-h-0 inline-flex items-center px-1 py-0.5 text-indigo-300 hover:text-indigo-200 text-[0.6875rem] font-medium underline decoration-indigo-300/40">
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
      {status.error && (!status.error.dataset_id || status.error.dataset_id === ds.currentId) && (() => {
        const view = failureView(status.error);
        return (
        <div className="rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-red-200 text-[0.6875rem]">
          <div className="font-semibold">⚠ {view.title}</div>
          {/* WHICH Python ai-toolkit was run with. A path that exists and runs but
              has no torch passes every folder check, so the run dies on
              "No module named 'torch'" with nothing pointing at the setting at
              fault — hours lost, reported by strouder (GitHub #19). Named first,
              above every other verdict: once the path is on screen the mistake is
              obvious in one second. `break-all` because these paths are long and
              the panel has to survive a 400 px phone. */}
          {view.interpreter && (
            <div className="mt-1.5 rounded border border-amber-400/40 bg-amber-500/10 p-2 text-amber-100">
              <div className="font-semibold">{view.interpreter.title}</div>
              <p className="m-0 mt-0.5 break-words text-amber-200/90">{view.interpreter.message}</p>
              <div className="mt-1 break-all font-mono text-[0.625rem] text-amber-100">
                {view.interpreter.python}
              </div>
              <a href="#/settings/local-tools?focus=aitoolkit-python"
                className="mt-1 inline-block text-amber-300 underline decoration-amber-300/50">
                Settings ▸ Local tools ▸ Python interpreter →
              </a>
            </div>
          )}
          {/* The optional Hugging Face fast-download accelerator, dead. It reads
              like a network fault and is not one, and the app never sets that
              variable — reported by bobba84 (GitHub #18). */}
          {view.hfTransfer && (
            <div className="mt-1.5 rounded border border-amber-400/40 bg-amber-500/10 p-2 text-amber-100">
              <div className="font-semibold">{view.hfTransfer.title}</div>
              <p className="m-0 mt-0.5 break-words text-amber-200/90">{view.hfTransfer.message}</p>
            </div>
          )}
          {/* The GPU-architecture verdict is a PROVEN cause read from the venv
              that trains — it goes first, above the log, with its remedy. */}
          {view.gpuArch && (
            <div className="mt-1.5 rounded border border-amber-400/40 bg-amber-500/10 p-2 text-amber-100">
              <div className="font-semibold">This GPU needs a different PyTorch build</div>
              <p className="m-0 mt-0.5 text-amber-200/90">{view.gpuArch.message}</p>
              {view.gpuArch.command && (
                <>
                  <div className="mt-1 text-amber-200/80">Run this once, then Train again:</div>
                  <pre className="mt-0.5 overflow-x-auto whitespace-pre-wrap break-all rounded bg-black/40 p-1.5 font-mono text-[0.625rem] text-amber-100">
                    {view.gpuArch.command}
                  </pre>
                </>
              )}
            </div>
          )}
          {/* Gated Hugging Face base: 401 (no valid token) and 403 (licence not
              accepted) look identical in the raw HF sentence and have OPPOSITE
              fixes — reported by SurpassHR (GitHub) on Krea 2. Shown above the
              log with the remedy that matches the status code. */}
          {view.hfGated && (
            <div className="mt-1.5 rounded border border-amber-400/40 bg-amber-500/10 p-2 text-amber-100">
              <div className="font-semibold">{view.hfGated.title}</div>
              <p className="m-0 mt-0.5 break-words text-amber-200/90">{view.hfGated.message}</p>
              <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1">
                {view.hfGated.status === 401 && (
                  <a href="#/settings/local-tools"
                    className="text-amber-300 underline decoration-amber-300/50">
                    Settings ▸ API keys →
                  </a>
                )}
                {view.hfGated.repo && (
                  <a href={view.hfGated.url} target="_blank" rel="noreferrer"
                    className="min-w-0 break-all text-amber-300 underline decoration-amber-300/50">
                    {view.hfGated.repo} on Hugging Face →
                  </a>
                )}
              </div>
            </div>
          )}
          {view.excerpt && (
            <pre className={`mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded bg-black/30 p-1.5 font-mono text-[0.625rem] ${
              view.tone === 'error' ? 'text-red-300/90' : 'text-content-muted'}`}>
              {view.excerpt}
            </pre>
          )}
          {/* One click to the log, right here. The other "<FolderOpen aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Run folder" button
              lives inside a collapsed disclosure further down — nobody looks for
              a log under "checkpoints" (reported by wannadecryptor on Discord).
              NO run selection is sent on purpose: the persisted base/family/
              variant are those of the run that just died, whatever the checkpoint
              browser happens to show. */}
          <div className="mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-1">
            <button type="button"
              onClick={() => postTrain(`/api/dataset/${ds.currentId}/train/open-folder`,
                { target: 'run' })}
              title="Open the folder of the run that just failed — training.log is in it"
              className="shrink-0 px-2 py-1 rounded-lg bg-red-500/20 border border-red-400/40 text-red-100 text-[0.6875rem] font-semibold">
              <FolderOpen aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Open run folder
            </button>
            <span className="min-w-0 text-red-300/80">{view.note}</span>
          </div>
          {view.causes && <div className="mt-1 text-red-300/80">{view.causes}</div>}
        </div>
        );
      })()}

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
      {cloudActivesHere.length > 1 && (() => {
        // Two runs whose frozen dataset generations differ are NOT a settings
        // A/B any more — mark the odd ones out against the newest run.
        const newestFp = cloudActivesHere[cloudActivesHere.length - 1]?.dataset_fingerprint;
        return (
          <div className="flex items-center gap-1 flex-wrap" role="tablist"
            aria-label="Active cloud runs on this dataset">
            {cloudActivesHere.map((a) => {
              const selected = a.run_id === cloudActiveHere?.run_id;
              const diverges = Boolean(a.dataset_fingerprint && newestFp
                && a.dataset_fingerprint !== newestFp);
              return (
                <button key={a.run_id} type="button" role="tab" aria-selected={selected}
                  onClick={() => setSelectedCloudRunId(a.run_id)}
                  title={diverges
                    ? 'This run trains a different dataset generation than its sibling — the runs are not comparing settings on the same data.'
                    : `Cloud run #${a.run_id} — ${a.status}`}
                  className={`rounded-full border px-2 py-0.5 text-[0.6875rem] tabular-nums ${
                    selected
                      ? 'border-sky-400/60 bg-sky-500/15 text-sky-200'
                      : 'border-border bg-surface text-content-muted hover:text-content'}`}>
                  ☁ #{a.run_id} · {a.status}{diverges ? ' ⚠' : ''}
                </button>
              );
            })}
          </div>
        );
      })()}
      {cloudActiveHere && (
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2 text-[0.6875rem] text-sky-200 flex-wrap">
            <span aria-hidden>☁️</span>
            <span className="font-semibold">Cloud run — {cloudActiveHere.status}</span>
            {cloudActivesHere.length > 1 && (() => {
              const fps = new Set(cloudActivesHere.map((a) => a.dataset_fingerprint).filter(Boolean));
              return fps.size > 1 ? (
                <span className="text-amber-300">⚠ different dataset generation</span>
              ) : null;
            })()}
            {cloudActiveHere.gpu && <span>{cloudActiveHere.gpu}</span>}
            {cloudActiveHere.price_per_hour != null && (
              <span className="tabular-nums">${cloudActiveHere.price_per_hour}/h · ~${cloudActiveHere.cost_estimate} so far</span>
            )}
            {/* Full progress bar, loss curve and samples live on the Runs hub. */}
            <Link to="/cloud" title="Open the Runs page — full progress, loss curve and samples"
              className="ml-auto min-h-10 lg:min-h-0 inline-flex items-center px-1 py-0.5 text-sky-300 hover:text-sky-200 font-medium underline decoration-sky-300/40">
              View in Runs ↗
            </Link>
            <button type="button" className="px-2 py-0.5 rounded bg-red-600/80 text-white text-[0.6875rem] font-semibold"
              onClick={() => setCloudStopTarget(cloudActiveHere)}>
              Stop cloud run
            </button>
          </div>
          <TrainingProgress datasetId={ds.currentId}
            base={cloudActiveHere.base_model ?? ''}
            trainType={cloudActiveHere.train_type || trainType}
            variant={cloudActiveHere.variant || variant} cloud
            runId={cloudActiveHere.run_id} />
          {normalizeTrainingMode(cloudActiveHere.training_mode) === TRAINING_MODE_FULL_TRANSFORMER && (
            <FullTransformerArtifactNotice run={cloudActiveHere} />
          )}
        </div>
      )}
      {/* Download link only when the LAST run matches the selected family
          (a legacy payload without train_type matches any family). Keeping it
          keyed on cloudStatus.last stays simple — per-family history is
          served by ?train_type= on the checkpoint route itself. */}
      {caps.cloud_training && !fullMode && !cloudActiveHere && cloudLastHere
        && normalizeTrainingMode(cloudLastHere.training_mode) === TRAINING_MODE_LORA
        && cloudLastHere.checkpoint_ready && cloudLastHere.status === 'done' && (
        <a href={`/api/dataset/${ds.currentId}/train/cloud/checkpoint?train_type=${encodeURIComponent(trainType)}`}
          className="text-sky-300 text-[0.6875rem] underline w-fit">
          ⬇ Download the cloud-trained LoRA (.safetensors)
        </a>
      )}
      {!cloudActiveHere && cloudLastHere
        && normalizeTrainingMode(cloudLastHere.training_mode) === TRAINING_MODE_FULL_TRANSFORMER && (
        <FullTransformerArtifactNotice run={cloudLastHere} />
      )}

      {/* --- Chemin essentiel : choisir le type de LoRA et lancer. Le reste
           (base/variante, masked, plafond de steps, programmation) vit dans
           « Advanced options » ci-dessous — replié par défaut, tout y reste
           accessible en un clic. --- */}
      <div className="flex items-center gap-2 flex-wrap rounded-lg border border-border bg-surface px-3 py-2">
        <span className="text-content-muted text-[0.625rem] uppercase">Model family</span>
        {/* Which family is preselected, and the cloud GPU guard-rails (budget,
            price ceiling, stall timeout), are settings — say so where the choice
            is actually being made.
            No `focus` on purpose: this label names TWO things that live in two
            different cards (training-default-family, and the cloud limits card
            above it). Focusing either would ring the wrong half for half the
            readers, which is worse than landing on the section that holds both. */}
        <SettingsLink section="training" className="order-last ml-auto">
          Defaults &amp; cloud limits
        </SettingsLink>
        <select value={trainType} onChange={(e) => onTypeChange(e.target.value)}
          disabled={trainTypeBusy || presetBusy || trainingModeBusy}
          aria-label="Training model family"
          title="Z-Image (prose, Qwen3 encoder) ~20 img · SDXL (ComfyUI checkpoints) ~30 img · Krea 2 (prose, base fixe Turbo) ~20 img · FLUX.1-dev (prose, gated HF, local-only) ~20 img · FLUX.2 Klein (prose, gated HF, 4B local / 9B cloud) ~20 img · Anima (prose OR booru tags — both native, Qwen LLM encoder, anime, public base, local-only) ~20 img"
          className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem] disabled:opacity-50">
          <option value="zimage">Z-Image (~20 img)</option>
          <option value="sdxl">SDXL (~30 img)</option>
          <option value="krea">Krea 2 (~20 img)</option>
          <option value="flux">FLUX.1 (~20 img)</option>
          <option value="flux2klein">FLUX.2 Klein (~20 img)</option>
          <option value="anima">Anima (~20 img)</option>
        </select>
        <div role="radiogroup" aria-label="Training mode"
          aria-describedby={[
            !fullTransformerEligible ? 'full-transformer-disabled-reason' : '',
            trainingModeError ? 'training-mode-save-error' : '',
          ].filter(Boolean).join(' ') || undefined}
          className="inline-flex max-w-full rounded-lg border border-border bg-app/50 p-0.5">
          <button type="button" role="radio" aria-checked={!fullMode}
            ref={(node) => { trainingModeRadioRefs.current[TRAINING_MODE_LORA] = node; }}
            tabIndex={!fullMode || !fullTransformerEligible ? 0 : -1}
            disabled={trainingModeBusy}
            aria-describedby={trainingModeError ? 'training-mode-save-error' : undefined}
            onKeyDown={onTrainingModeKeyDown}
            onClick={() => onTrainingModeChange(TRAINING_MODE_LORA)}
            className={`px-2.5 py-1 rounded-md text-[0.75rem] font-semibold transition-colors disabled:opacity-50 ${
              !fullMode ? 'bg-indigo-500/25 text-indigo-100' : 'text-content-muted hover:text-content'}`}>
            LoRA
          </button>
          <button type="button" role="radio" aria-checked={fullMode}
            ref={(node) => { trainingModeRadioRefs.current[TRAINING_MODE_FULL_TRANSFORMER] = node; }}
            tabIndex={fullMode && fullTransformerEligible ? 0 : -1}
            disabled={trainingModeBusy || !fullTransformerEligible}
            aria-describedby={[
              !fullTransformerEligible ? 'full-transformer-disabled-reason' : '',
              trainingModeError ? 'training-mode-save-error' : '',
            ].filter(Boolean).join(' ') || undefined}
            onKeyDown={onTrainingModeKeyDown}
            onClick={() => onTrainingModeChange(TRAINING_MODE_FULL_TRANSFORMER)}
            title={fullTransformerReason || `Experimental · ${denseBaseSummary} · cloud-only`}
            className={`px-2.5 py-1 rounded-md text-[0.75rem] font-semibold transition-colors disabled:opacity-40 ${
              fullMode ? 'bg-sky-500/25 text-sky-100' : 'text-content-muted hover:text-content'}`}>
            Full model
          </button>
        </div>
        {!fullTransformerEligible && (
          <span id="full-transformer-disabled-reason"
            className="basis-full text-amber-300 text-[0.6875rem] leading-relaxed">
            Full model unavailable: {fullTransformerReason}
          </span>
        )}
        {trainingModeError && (
          <span id="training-mode-save-error" role="alert"
            className="basis-full text-rose-300 text-[0.6875rem] leading-relaxed">
            {trainingModeError}
          </span>
        )}
        {!fullMode && <button type="button" disabled={!status.installed || belowFloor || status.in_progress || baseBlocksTrain || sdxlNeedsBase || customWeightsEmpty || sliderPromptsMissing}
          title={baseBlocksTrain ? baseBlockTitle
            : customWeightsEmpty ? 'Enter the path to your custom weights .safetensors'
            : sdxlNeedsBase ? 'Choose a base SDXL checkpoint'
            : sliderPromptsMissing ? 'Slider mode needs both a positive and a negative prompt'
            : belowFloor
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
            // The old "Resume" button called /train and let ai-toolkit infer
            // what it would restore. Route every existing run through the
            // shared dialog instead, where Full state vs Weights only is
            // explicit and checkpoint-specific.
            if (mode === 'continue') {
              setContinueInitialStep(null); // null deliberately selects the latest save
              setContinueOpen(true);
              return;
            }
            const fresh = mode === 'fresh';
            // ds.train takes camelCase opts — map the confirmable force flags.
            const OPT_FOR_FLAG = { allow_caption_mismatch: 'allowCaptionMismatch',
                                   allow_uncaptioned: 'allowUncaptioned',
                                   allow_caption_quality: 'allowCaptionQuality',
                                   allow_unverified_weights: 'allowUnverifiedWeights' };
            let opts = { baseModel: base, variant, trainType, trainingMode, masked: maskedOpt,
                         steps: stepsN, fresh,
                         vaePath, tePath, allowNotReady };
            let d = await ds.train(opts);
            for (let flag; d && d.ok === false && (flag = confirmableRetryFlag(d.error, 'Train anyway (force)')); ) {
              if (flag === 'declined') break;        // the confirm WAS the answer
              opts = { ...opts, [OPT_FOR_FLAG[flag]]: true };
              d = await ds.train(opts);
            }
            refreshStatus();
          }}
          className="px-3 py-1.5 rounded-lg bg-gradient-primary text-gray-950 text-sm font-semibold disabled:opacity-40">
          <Rocket aria-hidden="true" className="h-4 w-4" /> Train the LoRA
        </button>}
        {!fullMode && <HelpBadge topic="action-training-launch" />}
        {(caps.cloud_training || fullMode) && (
          <button type="button"
            disabled={!!cloudDisabledReason}
            title={cloudDisabledReason
              || 'Rents a vast.ai GPU for this run (~$1-2), auto-terminated'}
            onClick={() => {
              setHfCloudTokenIssue(null);
              setCloudDialog(true);
            }}
            className="px-3 py-1.5 rounded-lg border border-sky-500/50 bg-sky-500/10 text-sky-200 text-sm font-semibold disabled:opacity-40">
            <span aria-hidden>☁️</span> {fullMode ? 'Start full-model training' : 'Train in cloud'}
          </button>
        )}
        {/* ⏹ Stop = the DESTRUCTIVE action, named after what it does. Its old
            label named only the SIDE EFFECT (the GPU going back to ComfyUI) and
            read like housekeeping — users lost hours-long runs to one click, and
            one reported avoiding the button for days rather than finding out.
            Verb first, consequence in the title, and the same confirm wording as
            the Runs hub's "Stop run" for the very same endpoint.
            Reported by wannadecryptor (Discord). */}
        {status.in_progress && (
          <>
            <button type="button"
              title="Stops this training run. Checkpoints already saved are kept — ComfyUI gets the GPU back."
              onClick={async () => {
                const who = status.current?.name ? `“${status.current.name}”` : 'this dataset';
                if (!window.confirm(`Stop the training run for ${who}?\n\n`
                  + 'The training process is terminated and the pending local training queue is cleared. '
                  + 'Checkpoints already saved remain available, and ComfyUI gets the GPU back.')) return;
                await ds.stopTraining();
                refreshStatus();
              }}
              className="px-3 py-1.5 rounded-lg bg-red-600/80 text-white text-sm font-semibold">
              ⏹ Stop training
            </button>
            <HelpBadge topic="action-training-stop" />
          </>
        )}
        {!fullMode && status.in_progress && status.installed && (keptCount >= trainMinFloor || allowNotReady) && !sliderPromptsMissing && (
          <button type="button" disabled={queued || baseBlocksTrain} onClick={enqueue}
            title={baseBlocksTrain
              ? baseBlockTitle
              : `Train THIS dataset on “${baseLabel}” once the current training finishes`}
            className="px-3 py-1.5 rounded-lg bg-indigo-500/20 border border-indigo-400/40 text-indigo-200 text-sm font-semibold disabled:opacity-40">
            {queued ? '✓ Queued' : `➕ Add to queue (${baseLabel})`}
          </button>
        )}
        {/* Résumé lisible de la config que le prochain run utilisera — les
            réglages eux-mêmes vivent dans « Advanced options ». */}
        <span className="ml-auto text-content-subtle text-[0.625rem]"
          title="The configuration the next run will use — change it in Advanced options below">
          {fullMode ? (
            <>Full model · {denseBaseSummary} · cloud · {stepsOverride.trim() ? `${stepsN} steps` : 'adaptive steps'}</>
          ) : (<>
          {sliderOn ? 'slider (Beta) · ' : ''}base “{zimageRecipe?.baseLabel || baseLabel}”{zimageRecipe ? ` · ${zimageRecipe.adapterActive ? 'Turbo adapter v2 ON' : 'no training adapter'}` : ''} · {sliderOn ? 'unmasked (slider)' : maskedRembgMissing ? 'unmasked (rembg missing)' : masked ? 'masked' : 'unmasked'} · {advResLabel} · {stepsOverride.trim() ? `${stepsN} steps` : sliderOn ? `${stepsInfo?.steps ?? 1000} steps (slider policy)` : 'adaptive steps'}{advNetworkType === 'lokr' ? ` · LoKr${advLokrFactor ? ` factor ${advLokrFactor}` : ''}` : ''}{advEma ? ` · EMA ${advEma}` : ''}
          </>)}
        </span>
      </div>

      {fullMode && (
        <div role="status" className="rounded-lg border border-sky-400/45 bg-sky-500/10 px-3 py-2 text-[0.75rem] leading-relaxed">
          <div className="font-semibold text-sky-100">Experimental · advanced cloud mode · not recommended by default</div>
          <p className="m-0 mt-1 text-sky-200/90">
            Krea officially recommends training a LoRA on Raw, then applying it to Turbo.
            Full-model training requires a much larger, more diverse dataset, an 80 GB GPU,
            and at least 200 GB of disk.
          </p>
          <p className="m-0 mt-1 text-amber-100/95">
            The ~26 GB model is uploaded using a dedicated <code>HF_CLOUD_TOKEN</code>. A tightly scoped
            fine-grained token is recommended: read access only to the Krea 2 base this run uses
            ({denseBaseSummary}) and write access to a single dedicated Hugging Face namespace
            containing only LDS deliveries. A global write token is also
            accepted with a warning. Configure it in{' '}
            <SettingsLink section="local-tools" focus="HF_CLOUD_TOKEN" tone="warning">Settings ▸ Local tools</SettingsLink>.
            {' '}Stopping the run or hitting the runtime cap may lose the latest full-model checkpoint if it has not been uploaded.
          </p>
        </div>
      )}

      {/* Turbo in dense mode: allowed, and UNMEASURED. Said here — above the
          launch buttons, before a GPU is rented — and phrased as what is
          unknown. It promises no result and predicts no failure, and it never
          blocks: the product decision is warn and let through. */}
      {fullMode && denseTurboNotice && (
        <div role="status" id="dense-turbo-warning"
          className="rounded-lg border border-amber-400/45 bg-amber-500/[0.09] px-3 py-2 text-[0.75rem] leading-relaxed">
          <div className="font-semibold text-amber-100">⚠ {denseTurboNotice.title}</div>
          <p className="m-0 mt-1 text-amber-100/90">{denseTurboNotice.body}</p>
        </div>
      )}

      {/* A custom base picked on ANOTHER family was still attached to this
          dataset (one shared column). The run ignores it — say so, once, rather
          than let the summary above advertise weights nobody will load. */}
      {baseInfo?.base_family_mismatch && (
        <p className="m-0 mt-1 text-amber-300 text-[0.6875rem]">
          ⚠ {baseInfo.base_family_mismatch}
        </p>
      )}

      {/* --- Slider LoRA (Beta) : entraîne un LoRA BIPOLAIRE (±strength) depuis une
           paire de prompts via le trainer `concept_slider` d'ai-toolkit. Les images
           du dataset ne servent que de substrat de débruitage (captions ignorées).
           Feature expérimentale assumée — le badge Beta et les notes par famille
           disent exactement ce qui est prouvé et ce qui ne l'est pas. --- */}
      {!fullMode && (<div id="ds-training-slider" className={`rounded-lg border px-3 py-2 flex flex-col gap-2 ${
        sliderOn ? 'border-purple-400/50 bg-purple-500/5' : 'border-border bg-surface'}`}>
        <div className="flex items-center gap-2 flex-wrap">
          <span className="inline-flex items-center gap-1.5 text-sm font-semibold text-content"><SlidersHorizontal aria-hidden="true" className="h-4 w-4" /> Slider LoRA</span>
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
      </div>)}

      {/* Anime dataset on a non-Anima family: a pointer, not a rule. Deliberately
          NOT a warning (nothing is wrong — SDXL trains an anime character fine) and
          deliberately not a preselection: Anima is local-only and needs an up-to-date
          ai-toolkit, so forcing it would turn a description of the subject into a
          launch failure. See animeFamilyNote.js — it also keeps quiet when this
          machine cannot run Anima at all. */}
      {animeNote && (
        <p className="m-0 text-sky-300/90 text-[0.6875rem]">
          ℹ {animeNote}
        </p>
      )}

      {/* A disabled ☁ Train-in-cloud button always states WHY, right under the
          button row — the tooltip alone was invisible until hovered, so a greyed
          SDXL cloud button read as an unexplained limit (owner-reported). */}
      {(caps.cloud_training || fullMode) && cloudDisabledReason && (
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
          ⚠ {baseNotTrainable
            ? 'The selected base cannot be loaded for training — open Advanced options below to pick another one.'
            : sdxlNeedsBase
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
          {fullMode ? (
            <>
              ⚙️ Full-model recipe · steps · prompts · LR · resolution · checkpoints
              <span className="ml-2 font-normal text-sky-200/80 text-[0.6875rem]">
                {denseBaseSummary} · cloud
              </span>
            </>
          ) : (
            <>
              ⚙️ Advanced options
              <span className="ml-2 font-normal text-content-subtle text-[0.6875rem]">
                base &amp; variant · rank · resolution · masked · steps · scheduling · presets
              </span>
            </>
          )}
        </summary>
        <div className="px-3 pt-1 flex flex-col gap-2">
          {/* FULL_TRANSFORMER_ADVANCED_BRANCH_START — no LoRA control may escape
              the false branch below: the backend ignores all of them in dense. */}
          {fullMode ? (<>
            {/* DENSE_BASE_PICKER_START — the control the owner could not find.
                Its markup lives in the DenseBasePicker component above so a
                test can execute it; what has to stay HERE is the fact that the
                full-model arm renders it at all. */}
            <DenseBasePicker
              variant={variant} setVariant={setVariant}
              base={base} setBase={setBase}
              customBase={customBase} setCustomBase={setCustomBase}
              currentBases={currentBases} customSupported={customSupported}
              baseNote={baseNote} baseSummary={denseBaseSummary}
              busy={trainingModeBusy} />
            {/* DENSE_BASE_PICKER_END */}
            <FullTransformerAdvancedRecipe
              stepsOverride={stepsOverride}
              setStepsOverride={setStepsOverride}
              adv={adv} saveAdv={saveAdv}
              samplePromptsText={samplePromptsText}
              setSamplePromptsText={setSamplePromptsText}
              saveSamplePrompts={saveSamplePrompts}
              samplePromptsDefault={advSampleDefault}
              maxSamplePrompts={advMaxPrompts}
              datasetImages={datasetImages}
              applySamplePrompts={applySamplePrompts}
              quantizeTarget={denseQuantizeTarget(cloudLastHere || {})}
              suggestedQuantizePath={looksAbsoluteBase(base) ? String(base).trim() : ''}
              baseSummary={denseBaseSummary}
              disabled={trainingModeBusy || cloudActiveHere} />
          </>) : (<>
          {/* LORA_ADVANCED_CONTROLS_START */}
          {/* --- Presets : réglages nommés, ré-applicables et partageables en JSON.
               Appliquer REMPLACE les réglages explicites du dataset ; les clés
               inconnues d'un fichier importé sont ignorées (tolérance de version). --- */}
          <div className="flex items-center gap-1.5 flex-wrap rounded-lg border border-border bg-app/40 px-2 py-1.5">
            <span className="text-content-muted text-[0.625rem] uppercase">Presets</span>
            <select value={presetSel} onChange={(e) => setPresetSel(e.target.value)}
              aria-label="Training preset"
              className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem] max-w-[220px]">
              <option value="">— pick a preset —</option>
              {/* Built-ins are read-only and versioned with the app. Evidence-backed
                  recipes and source-labelled community starters stay visibly separate;
                  user snapshots follow. */}
              {researchedBuiltins.length > 0 && (
                <optgroup label="Built-in (researched)">
                  {researchedBuiltins.map((p) => (
                    <option key={p.id} value={p.id} title={p.description || undefined}>
                      ★ {p.name}
                    </option>
                  ))}
                </optgroup>
              )}
              {communityBuiltins.length > 0 && (
                <optgroup label="Built-in (community starting points)">
                  {communityBuiltins.map((p) => (
                    <option key={p.id} value={p.id} title={p.description || undefined}>
                      ◌ {p.name}
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
              <Save aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Save current…
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
              <Trash2 aria-hidden="true" className="h-3.5 w-3.5" />
            </button>
            <input ref={presetFileRef} type="file" accept=".json,application/json" className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) importPreset(f);
                e.target.value = '';
              }} />
            {selPreset && (
              <span role="status" className="basis-full text-content-subtle text-[0.625rem] leading-relaxed">
                {selPreset.builtin
                  ? selPreset.community
                    ? '◌ reported community starting point — not a result guarantee'
                    : '★ researched recipe'
                  : 'user preset'} · {typeLabel} · {kind}
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
                disabled={trainingModeBusy}
                aria-label="Base model"
                className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem] max-w-[230px]">
                {(currentBases.length ? currentBases
                  : [{ value: '', label: trainType === 'sdxl' ? (comfyConfigured ? 'No SDXL checkpoint found' : 'ComfyUI not configured') : trainType === 'krea' ? 'Official — Krea 2' : trainType === 'flux' ? 'Official — FLUX.1-dev' : trainType === 'flux2klein' ? 'Official — FLUX.2 Klein' : trainType === 'anima' ? 'Official — Anima' : 'Official — Z-Image-Turbo' }]).map((b) => (
                  <option key={b.value} value={b.value}>
                    {trainType === 'zimage' && !b.value ? 'Official recipe — selected by variant' : b.label}{b.value && baseInfo?.converted?.[b.value] ? ' ✓' : ''}{baseOptionSuffix(b)}
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
                  disabled={trainingModeBusy}
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
                  disabled={trainingModeBusy}
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
                  disabled={trainingModeBusy}
                  aria-label="FLUX.2 Klein model size"
                  title="FLUX.2 Klein model size — 4B fits a 16-24 GB local GPU (recommended locally); 9B needs 32-48 GB VRAM, best trained via ☁️ Train in cloud. Both bases are gated on Hugging Face: accept the license and set a HF token before the first run."
                  className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
                  <option value="4b">4B (local, 16-24 GB)</option>
                  <option value="9b">9B (cloud, 32-48 GB)</option>
                </select>
              )}
            </div>
            {/* Quantization verdict for the SELECTED base. Two different facts,
                so two different colours: rose = the trainer cannot load this
                file at all (the run is blocked here, before the dataset export
                and before a GPU is rented); amber = it loads and trains, from
                weights a cast already degraded. Stacked and wrapping, because
                this panel is read on a phone. */}
            {baseNote && (
              <div
                aria-label="Selected base precision"
                className={`flex flex-col gap-1 rounded-md border px-2.5 py-2 text-[0.6875rem] leading-relaxed ${
                  baseNote.level === 'error'
                    ? 'border-rose-400/40 bg-rose-500/[0.08] text-rose-200'
                    : 'border-amber-400/40 bg-amber-500/[0.08] text-amber-200'}`}>
                <span className="font-semibold">
                  {baseNote.level === 'error'
                    ? '⛔ This base cannot be used for training'
                    : '⚠ This base trains, from already-degraded weights'}
                </span>
                <span className="text-content-subtle break-words">{baseNote.text}</span>
              </div>
            )}
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
                  disabled={trainingModeBusy}
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
            {/* flux2klein et anima n'ont QUE des bases officielles fixes (rien à
                lister depuis ComfyUI) → le warning « bases can't be listed » n'y
                apporte que du bruit. Krea EN A une depuis que les checkpoints
                installés sont proposés : sans ComfyUI pointé, la liste se réduit
                à la base officielle et le dire vaut mieux que la laisser vide. */}
            {!comfyConfigured && trainType !== 'flux2klein' && trainType !== 'anima' && (
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-amber-300 text-[0.625rem]">
                  ⚠️ ComfyUI folder not set — training bases can't be listed{trainType === 'sdxl' ? '' : trainType === 'krea' ? ' (the official Krea 2 base still works)' : ' (the official Z-Image base still works)'}.
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

            {!fullMode && (<div className="flex flex-col gap-0.5">
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
            </div>)}

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
                faster — on the 12B families (Krea 2, FLUX) it is your best shot on a GPU under
                24 GB, at some cost in fine detail.
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

              <div className="flex flex-col gap-0.5 mt-1">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-content text-[0.75rem] w-28 shrink-0">Preview quality</span>
                  <label className="flex items-center gap-1.5">
                    <input type="number" min={advSampleStepsRange[0]} max={advSampleStepsRange[1]} step="1"
                      value={sampleStepsDraft}
                      onChange={(e) => setSampleStepsDraft(e.target.value)}
                      onBlur={saveSampleSteps}
                      onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur(); }}
                      placeholder={String(advSampleStepsDefault)}
                      aria-label="Preview steps"
                      className="w-16 px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]" />
                    <span className="text-content-muted text-[0.6875rem]">steps</span>
                  </label>
                  <label className="flex items-center gap-1.5">
                    <input type="number" min={advSampleGuidanceRange[0]} max={advSampleGuidanceRange[1]} step="0.5"
                      value={sampleGuidanceDraft}
                      onChange={(e) => setSampleGuidanceDraft(e.target.value)}
                      onBlur={saveSampleGuidance}
                      onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur(); }}
                      placeholder={String(advSampleGuidanceDefault)}
                      aria-label="Preview guidance scale"
                      className="w-16 px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]" />
                    <span className="text-content-muted text-[0.6875rem]">CFG</span>
                  </label>
                  {advSampleQualityOverridden && (
                    <button type="button"
                      onClick={() => saveAdv({ sample_steps: null, sample_guidance: null })}
                      className="px-2 py-1 rounded-lg border border-border bg-surface text-content-muted text-[0.6875rem] hover:text-content">
                      Auto
                    </button>
                  )}
                </div>
                <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
                  <b className="text-content-muted font-medium">Why:</b> previews only — this never touches the
                  weights. Leave both empty to follow {advFamilyLabel} ({advSampleStepsDefault} steps, CFG{' '}
                  {advSampleGuidanceDefault}); the default follows the base you picked, and a distilled one wants
                  far fewer steps than an undistilled one.
                  <b className="text-content-muted font-medium"> How:</b> raise the steps if your previews look
                  like unfinished sketches, lower them if a preview costs more time than the training it pauses.
                </span>
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
              <div className="flex items-center gap-1.5 mt-1">
                <UseDatasetCaptionsButton images={datasetImages} max={advMaxPrompts}
                  onPick={applySamplePrompts} />
                <HelpBadge topic="training.sample_prompts_from_dataset" />
              </div>
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
          {!fullMode && (<details className="group rounded-lg border border-indigo-400/40 border-l-[3px] border-l-indigo-400 bg-indigo-500/[0.14] transition-colors hover:bg-indigo-500/20">
            <summary className="flex items-center gap-2 cursor-pointer select-none list-none [&::-webkit-details-marker]:hidden px-2.5 py-2.5 text-[0.6875rem] font-semibold uppercase tracking-wider text-indigo-100 hover:text-white">
              <span aria-hidden className="text-indigo-300 transition-transform group-open:rotate-90">▸</span>
              <Microscope aria-hidden="true" className="h-3.5 w-3.5" />
              <span>Expert — last-mile levers</span>
              <span className="ml-auto hidden sm:inline normal-case font-normal tracking-normal text-indigo-300/50">network · alpha · memory{advTimestepSupported ? ' · timestep' : ''} · optimizer · schedule · EMA</span>
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
                {advNetworkType === 'lokr' && (
                  <div className="flex items-center gap-2 flex-wrap mt-1">
                    <span className="text-content text-[0.75rem] w-28 shrink-0 inline-flex items-center gap-1">
                      LoKr factor<HelpBadge topic="training.lokr_factor" />
                    </span>
                    <select value={advLokrFactor == null ? 'auto' : String(advLokrFactor)}
                      onChange={(e) => saveAdv({ lokr_factor: e.target.value === 'auto' ? 'auto' : Number(e.target.value) })}
                      aria-label="LoKr decomposition factor"
                      className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
                      <option value="auto">Auto (ai-toolkit)</option>
                      {advLokrFactorChoices.map((factor) => <option key={factor} value={String(factor)}>{factor}</option>)}
                    </select>
                  </div>
                )}
                <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
                  <b className="text-content-muted font-medium">Why:</b> LoRA is the standard adapter; LoKr factorises
                  the update differently. <b className="text-content-muted font-medium">How:</b> keep LoRA unless you are
                  deliberately comparing it. The Krea Raw community starter below pins LoKr factor 16, but a network type
                  alone cannot make up for the wrong images, captions or total steps.
                </span>
              </div>
              {/* Krea-only fields from the reported Krea Raw LoKr recipe. Kept out of
                  every other family: this is a transparent, testable starting point,
                  not an assertion that one anecdote transfers to another architecture. */}
              {advKreaRecipeSupported && (
                <div className="flex flex-col gap-2">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-content text-[0.75rem] w-28 shrink-0 inline-flex items-center gap-1">
                      Krea community recipe<HelpBadge topic="training.krea_community_recipe" />
                    </span>
                    <span className="text-amber-200 text-[0.6875rem] leading-relaxed">
                      Community starting point, not a likeness promise — compare your own checkpoints.
                    </span>
                  </div>
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-content text-[0.75rem] w-28 shrink-0">Content / style</span>
                    <select value={advContentOrStyle ?? 'auto'}
                      onChange={(e) => saveAdv({ content_or_style: e.target.value === 'auto' ? 'auto' : e.target.value })}
                      aria-label="Krea content or style balance"
                      className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
                      <option value="auto">Auto ({advContentOrStyleDefault})</option>
                      {advContentOrStyleChoices.map((mode) => <option key={mode} value={mode}>{mode}</option>)}
                    </select>
                  </div>
                  <div className="flex items-center gap-2 flex-wrap">
                    <label className="flex items-center gap-2 flex-wrap cursor-pointer">
                      <span className="text-content text-[0.75rem] w-28 shrink-0 inline-flex items-center gap-1">
                        Differential guidance<HelpBadge topic="training.krea_community_recipe" />
                      </span>
                      <input type="checkbox" checked={advDifferentialGuidance}
                        onChange={(e) => saveAdv({ do_differential_guidance: e.target.checked })}
                        aria-label="Enable Krea differential guidance"
                        className="h-4 w-4 rounded border-border bg-surface accent-indigo-500" />
                      <span className="text-content-muted text-[0.75rem]">enable</span>
                    </label>
                    <label className="flex items-center gap-2">
                      <span className="text-content-muted text-[0.6875rem]">scale</span>
                      <input type="number" min="0.1" max="10" step="0.1"
                        value={differentialGuidanceScaleDraft}
                        disabled={!advDifferentialGuidance}
                        onChange={(e) => setDifferentialGuidanceScaleDraft(e.target.value)}
                        onBlur={saveDifferentialGuidanceScale}
                        onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur(); }}
                        aria-label="Krea differential guidance scale"
                        className="w-16 px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem] disabled:cursor-not-allowed disabled:opacity-50" />
                    </label>
                  </div>
                  <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
                    <b className="text-content-muted font-medium">Reported starter:</b> Balanced plus differential
                    guidance at scale 3 is what the Krea 2 Raw · LoKr likeness preset applies. Change one variable at a
                    time and keep the intermediate saves that actually work for your dataset.
                  </span>
                </div>
              )}
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
                  default; 0.99 averages faster, 0.999 is slower and steadier. Test it as a separate variable: it is
                  not part of the Krea Raw LoKr likeness starter.
                </span>
              </div>
              {/* Dual captions — train each image with a long AND a short caption */}
              <div className="flex flex-col gap-0.5">
                <label className="flex items-center gap-2 flex-wrap cursor-pointer">
                  <span className="text-content text-[0.75rem] w-28 shrink-0 inline-flex items-center gap-1">
                    Dual captions<HelpBadge topic="training.dual_captions" />
                  </span>
                  <input type="checkbox" checked={advDualCaptions}
                    onChange={(e) => saveAdv({ dual_captions: e.target.checked })}
                    aria-label="Dual long + short captions"
                    className="h-4 w-4 rounded border-border bg-surface accent-indigo-500" />
                  <span className="text-content-muted text-[0.75rem]">long + short (local training only)</span>
                </label>
                {/* Issue #22 (1Tomber): Krea 2 / Anima cache their text embeddings, so the
                    short caption can never be encoded. Say it here rather than let the user
                    believe two wordings are training. Wraps at 400 px — no fixed width. */}
                {advDualCaptions && !dualCaptionsSupport(trainType).supported && (
                  <span className="text-amber-400 text-[0.6875rem] leading-relaxed">
                    Ignored here: {dualCaptionsSupport(trainType).note}
                  </span>
                )}
                <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
                  <b className="text-content-muted font-medium">Why:</b> trains each image with both a full and a brief
                  caption (text-side augmentation) so the LoRA leans less on any single wording.
                  <b className="text-content-muted font-medium"> How:</b> the short variant is derived from the long one
                  when you (re-)caption — same rules (no trigger, identity/concept/aesthetic kept out); edit it per image in
                  the ⛶ caption editor. Cloud runs ignore this and train on the long caption only for now.
                </span>
              </div>
              {/* Concept face masking — teach the act, not the identities (issue #15,
                  reported by shivdbz2010). Renders nothing outside a concept dataset. */}
              <ConceptFaceMaskField
                datasetId={ds.currentId}
                enabled={advMaskFaces}
                supported={advMaskFacesSupported}
                conceptConflict={advMaskFacesConflict}
                faceCapability={caps.face_scoring}
                expandDefault={undefined}
                onToggle={(v) => saveAdv({ mask_faces: v })}
              />
              {/* Memory saving — quantisation + low-VRAM streaming (issue #14).
                  The recipes are calibrated for 24 GB; on a bigger card that is a
                  tax nobody asked for. Defaults are UNCHANGED — this only makes
                  them a choice. Sending 'auto' when a box returns to its family
                  default keeps an untouched-again dataset byte-identical. */}
              <div className="flex flex-col gap-1">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-content text-[0.75rem] w-28 shrink-0 inline-flex items-center gap-1">
                    Memory saving<HelpBadge topic="training.memory_saving" />
                  </span>
                  <span className="text-content-subtle text-[0.6875rem]">
                    {advMemStateLabel}
                  </span>
                  {advMemTouched && (
                    <button type="button"
                      onClick={() => saveAdv(Object.fromEntries(MEMORY_KEYS.map((k) => [k, 'auto'])))}
                      className="ml-auto text-[0.625rem] text-content-subtle hover:text-content underline underline-offset-2">
                      Reset to default
                    </button>
                  )}
                </div>
                <div className="flex flex-col gap-1 sm:flex-row sm:flex-wrap sm:gap-x-4">
                  {MEMORY_KEYS.map((k) => (
                    <label key={k} className="flex items-center gap-2 cursor-pointer min-w-0">
                      <input type="checkbox" checked={Boolean(advMemEff[k])}
                        onChange={(e) => saveAdv(memoryPatchFor(k, e.target.checked, advMemDefault))}
                        aria-label={MEMORY_LABELS[k]}
                        className="h-4 w-4 shrink-0 rounded border-border bg-surface accent-indigo-500" />
                      <span className="text-content-muted text-[0.75rem] truncate">{MEMORY_LABELS[k]}</span>
                    </label>
                  ))}
                </div>
                {/* A saver this family's recipe relies on is off — most often
                    because it was switched off on a 2B family (Anima/SDXL, where
                    off IS the default) and the family then changed. Wraps at
                    400 px: no fixed width, no truncation. */}
                {advMemRiskLine && (
                  <span className={`text-[0.6875rem] leading-relaxed ${
                    adv?.memory_risk?.verdict === 'can_disable' ? 'text-content-muted' : 'text-amber-300'}`}>
                    {adv?.memory_risk?.verdict === 'can_disable' ? '' : '⚠️ '}{advMemRiskLine}
                  </span>
                )}
                <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
                  <b className="text-content-muted font-medium">Why:</b> the recipes are tuned so a 12B model fits in
                  24 GB — quantisation costs precision and low-VRAM streaming costs a lot of speed. If your card is
                  bigger than the target, you are paying for nothing.
                  <b className="text-content-muted font-medium"> How:</b> {advMemAdviceText}
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
                  is fast and VRAM-light; <i>adafactor</i> uses less memory and auto-scales; <i>automagic</i>/<i>automagic2</i>
                  use ai-toolkit&apos;s adaptive update rule; the Krea community starter still records its reported initial
                  LR of 1e-4. <i>prodigy</i> also auto-tunes the LR and is popular for tiny sets — but may need
                  <code className="text-content-muted">pip install prodigyopt</code> in the ai-toolkit venv. Picking an
                  adaptive optimiser is the &quot;push further without cranking the LR&quot; move.
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
          </details>)}

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
            <span className={masked && !sliderOn && !maskedRembgMissing ? 'text-emerald-300' : ''}><Drama aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Masked (bg 10%)</span>
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

          {/* One-time carry-over of the legacy per-browser preference. Shown ONLY
              to the browsers that had explicitly turned masking off — adopting
              that silently would spread one browser's choice over every dataset,
              and dropping it silently would start paying for a mask pass nobody
              asked for. Neither happens: the user answers, once. */}
          {maskedCarryOver && (
            <div className="rounded-lg border border-amber-400/40 bg-amber-500/10 px-3 py-2 flex flex-col gap-2 text-[0.6875rem]">
              <span className="text-content leading-relaxed">
                <b>Masked training is now a dataset setting</b>, shared across your
                browsers and devices instead of living in this one. This browser had it
                turned <b>off</b>; datasets default to <b>on</b>. Which do you want here?
              </span>
              <div className="flex flex-wrap gap-2">
                <button type="button" onClick={() => answerMaskedCarryOver(false)}
                  className="px-2 py-1 rounded border border-border text-content-muted hover:text-content hover:bg-surface-raised">
                  Turn it off for this dataset
                </button>
                <button type="button" onClick={() => answerMaskedCarryOver(true)}
                  className="px-2 py-1 rounded border border-emerald-400/40 text-emerald-300 hover:bg-emerald-500/10">
                  Keep masking on
                </button>
              </div>
            </div>
          )}

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

          {!fullMode && status.installed && (keptCount >= (TRAIN_MIN[trainType]?.[0] ?? 12) || allowNotReady) && (
            <div className="flex items-center gap-2 flex-wrap">
              <button type="button" disabled={queued || baseBlocksTrain} onClick={openSched}
                aria-expanded={showSched}
                title={baseBlocksTrain
                  ? baseBlockTitle
                  : 'Schedule this training for a specific day and time — it will queue up if another training is running then'}
                className="px-3 py-1.5 rounded-lg bg-amber-500/15 border border-amber-400/40 text-amber-200 text-sm font-semibold disabled:opacity-40">
                {queued ? '✓ Queued' : '⏰ Schedule'}
              </button>
              <span className="text-content-subtle text-[0.625rem]">
                run this training later, at a day &amp; time you pick
              </span>
            </div>
          )}

          {!fullMode && showSched && !queued && (
            <div className="flex items-center gap-2 flex-wrap rounded-lg border border-amber-400/30 bg-amber-500/5 px-3 py-2">
              <label className="flex items-center gap-2 text-content-muted text-[0.6875rem]">
                <span className="uppercase">Start at</span>
                <input type="datetime-local" value={schedAt}
                  onChange={(e) => setSchedAt(e.target.value)}
                  aria-label="Scheduled training date and time"
                  className="rounded border border-border bg-app/60 px-2 py-1 text-content text-[0.8125rem]" />
              </label>
              <span className="text-content-subtle text-[0.625rem]">
                Base “{baseLabel}” — if another training is running at that time, it waits in the queue.
              </span>
              <button type="button" onClick={schedule} disabled={!schedAt}
                className="ml-auto px-3 py-1.5 rounded-lg bg-gradient-primary text-gray-950 text-sm font-semibold disabled:opacity-40">
                Schedule
              </button>
            </div>
          )}
          {/* LORA_ADVANCED_CONTROLS_END */}
          </>)}
          {/* FULL_TRANSFORMER_ADVANCED_BRANCH_END */}
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

      {fullMode
        ? (
          <p className="m-0 text-content-subtle text-[0.625rem]">
            Full model: a small LoRA-sized dataset is only a smoke test; quality requires a much
            larger, more varied, deduplicated collection.
          </p>
        )
        : sliderOn
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
          <Package aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Checkpoints &amp; trained LoRAs
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
              <option value="anima">Anima</option>
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
          {/* FULL MODELS first, above the LoRA lanes: when a dataset has one it
              is the biggest thing it produced, and until now it appeared in this
              panel not at all — the only way to try it was to open ComfyUI by
              hand and find the file. Renders nothing when there is none. */}
          <DenseModelsPanel datasetId={ds.currentId} models={denseModels}
            onChanged={() => loadCheckpoints(checkpointBase, checkpointTrainType, checkpointVariant)} />
          {/* Merging needs no dense run of its own — the usual case is a LoRA
              trained here folded into a base downloaded from anywhere — and
              DenseModelsPanel renders NOTHING when this dataset has no full
              model. So the tool also lives here, where it is always reachable,
              collapsed because most visits to this panel are not about it.
              (Inside a dense card it appears again with the base pre-filled.) */}
          <details open={mergeOpen}
            className="rounded-lg border border-border bg-surface-raised px-3 py-2">
            <summary onClick={toggleMerge}
              className="cursor-pointer text-content text-xs font-semibold">
              <Dna aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Merge a LoRA into a base checkpoint
            </summary>
            <p className="m-0 mt-1 text-content-subtle text-[0.625rem] leading-relaxed">
              Folds one or more LoRAs into a full-precision checkpoint and writes a new
              full model — the step between “I trained a LoRA” and “I have a model to
              publish”. Nothing is overwritten, and the result says in its own metadata
              that it is a merge and not a training run.
            </p>
            <div className="mt-1.5">
              <LoraMergeTool framed={false} family={checkpointTrainType} />
            </div>
          </details>
          <div className="flex items-center gap-2 flex-wrap">
            {/* () => … sinon React passe l'event en 1er arg → forBase = PointerEvent
                → base_model=[object Object] → run inexistant → liste vide. */}
            <button type="button" onClick={() => loadCheckpoints(checkpointBase, checkpointTrainType, checkpointVariant)}
              title="Reload the checkpoint list for this results filter"
              className="px-3 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-xs font-semibold">
              ↻ Refresh checkpoints
            </button>
            {/* Graph ↔ List. The graph is the flagship surface, so ◉ Graph wears
                the accent style when active (not a bare grey button) to signal
                it's THE view; ☰ List stays available for the flat step list. */}
            <div role="group" aria-label="Checkpoints view"
              className="inline-flex items-center gap-0.5 rounded-lg border border-border bg-app p-0.5">
              <button type="button" onClick={() => setCkView('graph')}
                aria-pressed={checkpointsView === 'graph'}
                title="See this dataset's runs and their checkpoints as a graph — continuations shown, import / generate / download / continue from any checkpoint"
                className={'px-3 py-1 rounded-md text-xs font-semibold transition-colors '
                  + (checkpointsView === 'graph'
                    ? 'bg-indigo-500 text-gray-950 shadow-sm '
                    : 'text-content-muted hover:text-content ')}>
                ◉ Graph
              </button>
              <button type="button" onClick={() => setCkView('list')}
                aria-pressed={checkpointsView === 'list'}
                title="Flat list of this run's steps"
                className={'px-3 py-1 rounded-md text-xs font-semibold transition-colors '
                  + (checkpointsView === 'list'
                    ? 'bg-surface-raised text-content shadow-sm '
                    : 'text-content-muted hover:text-content ')}>
                ☰ List
              </button>
            </div>
            {/* Ouvre les dossiers dans l'explorateur du poste (app locale) :
                loras = imports ComfyUI de la famille ; run = checkpoints bruts. */}
            <button type="button"
              onClick={() => postTrain(`/api/dataset/${ds.currentId}/train/open-folder`,
                { target: 'loras', ...trainingRunSelection(undefined, checkpointTrainType, checkpointVariant) })}
              title={`Open the ComfyUI folder where imported ${checkpointTypeLabel} LoRAs live`}
              className="px-3 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-xs font-semibold">
              <FolderOpen aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />LoRA folder
            </button>
            <button type="button"
              onClick={() => postTrain(`/api/dataset/${ds.currentId}/train/open-folder`,
                { target: 'run', ...trainingRunSelection(checkpointBase, checkpointTrainType, checkpointVariant) })}
              title="Open this run's output folder (raw checkpoints, samples, training log)"
              className="px-3 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-xs font-semibold">
              <FolderOpen aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Run folder
            </button>
            <span className="text-content-subtle text-[0.625rem]">
              import the checkpoint you like into ComfyUI to use (and test) the LoRA
            </span>
          </div>

          {/* ◉ Graph — the DEFAULT view: the dataset's runs + their checkpoints as
              one genealogy, where each pill can be imported / generated / downloaded
              on the spot. Same component the Runs hub draws (no copy). */}
          {checkpointsView === 'graph' && (
            <div className="flex flex-col gap-2">
              {datasetGraph?.loading && (
                <div className="flex items-center gap-2 text-content-subtle text-[0.75rem]">
                  <span aria-hidden className="h-3 w-3 animate-spin rounded-full border-2 border-border-strong border-t-indigo-400" />
                  Building the graph…
                </div>
              )}
              {datasetGraph?.error && <p className="m-0 text-rose-300/80 text-[0.75rem]">{datasetGraph.error}</p>}
              {datasetGraph?.tree && (
                Array.isArray(datasetGraph.tree.nodes) && datasetGraph.tree.nodes.length
                  ? <RunLineageGraph tree={datasetGraph.tree}
                      // Same ★ best-settings guard-rail as the flat list's 🗑:
                      // the pill's delete warns loudly when it would unpin the
                      // Studio's saved winning combo.
                      bestSettingsLora={ds.data?.best_settings_loras || null}
                      // Same pill gesture as the Runs hub, served by this panel's
                      // Continue dialog: 'any' lets a local run's save offer it too.
                      // Gated on the checkpoint selection matching Training (the
                      // dialog resumes THAT lane); whether the continuation can run
                      // locally, in the cloud or neither is the dialog's own,
                      // per-lane answer — so a local training in flight no longer
                      // hides the action, it just closes the Local lane.
                      continueSource="any"
                      onContinueCheckpoint={checkpointMatchesTraining
                        ? continueFromGraphCheckpoint : undefined}
                      refetchTree={async () => {
                        const tree = await fetchDatasetLineage();
                        setDatasetGraph({ tree });
                        // Mirror deploy state into the flat list too, so switching
                        // to ☰ List after an inline import shows it right away.
                        loadCheckpoints(checkpointBase, checkpointTrainType, checkpointVariant);
                        return tree;
                      }} />
                  : <p className="m-0 text-content-subtle text-[0.75rem]">
                      No training runs recorded for this family yet — train once and its runs will appear here.
                    </p>
              )}
            </div>
          )}

          {checkpointsView === 'list' && (<>

          {checkpoints.length > 0 && (
            <div className="flex flex-col gap-1">
              {/* Identity header for the LOCAL active set: what this group IS +
                  which run produced it + a jump back to its Runs row. */}
              {(() => {
                const li = localRunIdentity(checkpoints);
                return (
                  <div className="flex items-center gap-2 flex-wrap rounded-md border border-violet-500/25 bg-violet-500/5 px-2 py-1">
                    {li && <RunIdChip source={li.source} recordId={li.recordId} cloudId={li.cloudId} />}
                    <span className="text-content-muted text-[0.6875rem]">
                      <b className="text-content">Active set</b> — used by Studio / Continue / Import; cloud epochs are mirrored here.
                    </span>
                    {li && (
                      <Link to={`/cloud#${runRowDomId(li.source, li.id)}`}
                        title="Jump to this run on the Runs page"
                        className="ml-auto min-h-10 lg:min-h-0 inline-flex items-center px-1 py-0.5 text-violet-300 hover:text-violet-200 text-[0.6875rem] font-medium underline decoration-violet-300/40">
                        View in Runs ↗
                      </Link>
                    )}
                  </div>
                );
              })()}
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-content-muted text-[0.625rem] uppercase">
                  {checkpointTypeLabel} checkpoints — base “{checkpointBaseLabel}” · {checkpointVariantDisplay} (pick the earliest one that holds the identity)
                </span>
                <button type="button" disabled={bestEpochBusy}
                  onClick={findBestEpoch}
                  title="Scores every training sample vs the reference photo (face similarity, CPU) and recommends the checkpoint that holds the identity best — needs the Quality tools (ML extras)."
                  className="px-2.5 py-1 rounded-lg bg-amber-500/15 border border-amber-400/40 text-amber-200 text-[0.6875rem] font-semibold disabled:opacity-40">
                  <Trophy aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />{bestEpochBusy ? 'Scoring samples…' : 'Find best epoch'}
                </button>
                {/* A local training in flight no longer locks this button: the
                    dialog offers the cloud lane, and closes the local one with its
                    reason. It stays disabled when NEITHER lane could run. */}
                <button type="button"
                  disabled={!checkpointMatchesTraining
                    || (status.in_progress && !continueLanes.cloud.available)}
                  onClick={() => { setContinueInitialStep(null); setContinueError(null); setContinueOpen(true); }}
                  title={!checkpointMatchesTraining
                    ? 'To continue this run, select the same LoRA family, base and variant in Training first'
                    : status.in_progress
                      ? 'A training is running here — continue this run in the cloud instead'
                      : 'Resume from any of this run’s checkpoints — pick where it runs, the step count, the checkpoint, and the safe settings'}
                  className="ml-auto px-2.5 py-1 rounded-lg bg-indigo-500/20 border border-indigo-400/40 text-indigo-200 text-[0.6875rem] font-semibold disabled:opacity-40">
                  ▶ Continue training…
                </button>
              </div>
              {/* A disabled button whose only explanation is a title= reads as a DEAD
                  button: no hover, no reason, and the report is "it does nothing".
                  State it in the panel, like the Find-best-epoch lines just below. */}
              {!checkpointMatchesTraining && (
                <p className="m-0 text-amber-300 text-[0.625rem]">
                  ▶ Continue is off: these checkpoints come from a different LoRA family,
                  base or variant than the one selected in Training above. Match them to
                  continue this run.
                </p>
              )}
              {checkpointMatchesTraining && status.in_progress
                && !continueLanes.cloud.available && (
                <p className="m-0 text-amber-300 text-[0.625rem]">
                  ▶ Continue is off while a training runs on this machine.
                  {' '}{continueLanes.cloud.reason}
                </p>
              )}
              {bestEpoch && !bestEpoch.available && (
                <p className="m-0 text-amber-300 text-[0.625rem]"><Trophy aria-hidden="true" className="mr-1 inline h-3 w-3 align-[-1px]" />{bestEpoch.reason}</p>
              )}
              {bestEpoch?.available && (
                <p className="m-0 text-amber-200 text-[0.625rem]">
                  <Trophy aria-hidden="true" className="mr-1 inline h-3 w-3 align-[-1px]" />Best identity at <span className="font-semibold">step {bestEpoch.best_step}</span>
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
                      <Trophy aria-hidden="true" className="mr-0.5 inline h-3 w-3 align-[-1px]" />recommended
                    </span>
                  )}
                  {renderDeployControl(c, {
                    // await + refresh: the import must show up as deployed
                    // without a manual Refresh click (user-observed). finally:
                    // the list refreshes even if the import failed (the error
                    // toast comes from the hook).
                    onImport: async () => {
                      try { await ds.importCheckpoint(c.filename, checkpointBase, checkpointTrainType, checkpointVariant); }
                      finally { loadCheckpoints(checkpointBase, checkpointTrainType, checkpointVariant); }
                    },
                    importTitle: `Deploy this checkpoint into ComfyUI's ${checkpointLorasLabel} folder so you can test and generate with it`,
                  })}
                  <button type="button"
                    onClick={async () => {
                      if (!window.confirm(`Move “${c.filename}” to the trash?\n\nRecoverable until you empty the trash in Settings.`)) return;
                      const d = await postTrain(`/api/dataset/${ds.currentId}/train/run-checkpoint/delete`,
                        { filename: c.filename, ...trainingRunSelection(checkpointBase, checkpointTrainType, checkpointVariant) });
                      if (d.ok === false) toastTrainError(d, 'Delete failed');
                      loadCheckpoints(checkpointBase, checkpointTrainType, checkpointVariant);
                    }}
                    title="Move this checkpoint to the trash (recoverable until the trash is emptied in Settings)"
                    className="px-2 py-0.5 rounded bg-red-500/15 border border-red-500/40 text-red-300">
                    <Trash2 aria-hidden="true" className="h-3.5 w-3.5" />
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
                  title="Keep the final (+ the best-epoch pick if scored) and move every other checkpoint of this run to the trash"
                  className="px-2.5 py-1 rounded-lg bg-red-500/10 border border-red-500/30 text-red-200 text-[0.6875rem] font-semibold">
                  <Eraser aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Clean up this run
                </button>
                <span className="text-content-subtle text-[0.625rem]">
                  keeps final{bestEpoch?.available ? ' + best' : ''} — the rest goes to the trash
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
                    {/* ONE number, once: the chip prints the record id (the
                        same #N as the lineage card and the inspector); the
                        cloud id lives in its tooltip. The old header printed
                        the cloud id twice — as the chip AND as "Run #N" text
                        — while every other surface named the run differently. */}
                    {g.run_id != null
                      ? <RunIdChip source="cloud" recordId={g.record_id} cloudId={g.run_id} />
                      : <span className="text-sky-200 text-[0.6875rem]" aria-hidden>☁ run unknown</span>}
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
                      <span className="ml-auto flex items-center gap-1.5">
                        {/* ⚙/⇄ address the lineage tree, whose nodes key on the
                            RECORD id — passing the cloud id here chased a node
                            that does not exist and answered with a misleading
                            "not in the lineage tree" error (user-reported,
                            2026-08-29). Gated on record_id: a pre-registry run
                            recorded no recipe, so there is nothing to open —
                            no button beats a dead-end toast after the click. */}
                        {g.record_id != null && (<>
                        {/* ⚙ The SAME recipe panel the Lineage graph opens —
                            rank, LR, optimizer, resolution, notes — one click
                            from the checkpoints it produced. */}
                        <button type="button" data-testid="ckpt-run-details"
                          onClick={() => openRunDetails(g.record_id)}
                          title="Open this run's full recipe — rank, learning rate, optimizer, resolution, notes"
                          className="min-h-10 lg:min-h-0 rounded border border-white/15 px-1.5 py-0.5 text-[0.6875rem] font-medium text-content hover:border-white/35">
                          ⚙ Details
                        </button>
                        {/* ⇄ Two picks open the SAME two-run diff as the graph:
                            recipe deltas, the frozen dataset (images added /
                            removed / re-captioned), and the machine. */}
                        <button type="button" data-testid="ckpt-run-compare"
                          aria-pressed={ckptDiffIds.includes(g.record_id)}
                          onClick={() => toggleRunCompare(g.record_id)}
                          title="Pick this run for comparison — pick two to see what changed between them (recipe, dataset images and captions, machine)"
                          className={`min-h-10 lg:min-h-0 rounded border px-1.5 py-0.5 text-[0.6875rem] font-medium ${ckptDiffIds.includes(g.record_id)
                            ? 'border-indigo-400/70 bg-indigo-500/25 text-indigo-100'
                            : 'border-white/15 text-content hover:border-white/35'}`}>
                          ⇄ Compare{ckptDiffIds.includes(g.record_id) ? ' ✓' : ''}
                        </button>
                        </>)}
                        <Link to={`/cloud#${runRowDomId('cloud', g.run_id)}`}
                          title="Jump to this run on the Runs page"
                          className="min-h-10 lg:min-h-0 inline-flex items-center px-1 py-0.5 text-sky-300 hover:text-sky-200 text-[0.6875rem] font-medium underline decoration-sky-300/40">
                          View in Runs ↗
                        </Link>
                      </span>
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
                      {renderDeployControl(c, {
                        onImport: async () => {
                          const d = await postTrain(`/api/dataset/${ds.currentId}/train/import`,
                            { filename: c.filename, cloud_run_id: c.run_id,
                              ...trainingRunSelection(checkpointBase, checkpointTrainType, c.variant || checkpointVariant) });
                          // Success must be VISIBLE: without the toast a working
                          // import looked like a dead button (user-observed).
                          if (d.ok === false) toastTrainError(d, 'Import failed');
                          else toast.success(d.note || `LoRA imported: ${d.dest || c.filename}`);
                          loadCheckpoints(checkpointBase, checkpointTrainType, checkpointVariant);
                        },
                        importTitle: c.active
                          ? 'Import the latest synced save — the run keeps training'
                          : 'Import this cloud checkpoint into ComfyUI',
                      })}
                      {!c.active && (
                        <button type="button"
                          onClick={async () => {
                            if (!window.confirm(`Move “${c.filename}” to the trash?\n\nRecoverable until you empty the trash in Settings.`)) return;
                            const d = await postTrain(`/api/dataset/${ds.currentId}/train/run-checkpoint/delete`,
                              { filename: c.filename, cloud_run_id: c.run_id,
                                ...trainingRunSelection(checkpointBase, checkpointTrainType, c.variant || checkpointVariant) });
                            if (d.ok === false) toastTrainError(d, 'Delete failed');
                            loadCheckpoints(checkpointBase, checkpointTrainType, checkpointVariant);
                          }}
                          title="Move this cloud save to the trash"
                          className="px-2 py-0.5 rounded bg-red-500/15 border border-red-500/40 text-red-300">
                          <Trash2 aria-hidden="true" className="h-3.5 w-3.5" />
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
              No {checkpointTypeLabel} checkpoint for base “{checkpointBaseLabel}” · {checkpointVariantDisplay} — run this exact recipe first.
            </p>
          )}

          {/* Every deployed file the lists above ALREADY account for is now
              actionable right next to its checkpoint (✓ Deployed / ⏏ Undeploy),
              so repeating it here — under a red 🗑 that reads as destruction —
              is exactly the contradiction this section used to create. What
              stays is what nothing above explains: LoRAs imported before run
              tagging ("run ?"), files dropped in the folder by hand, and any
              file carrying an arch-mismatch warning (never hidden). */}
          {otherImported.length > 0 && (
            <div className="flex flex-col gap-1">
              <span className="text-content-muted text-[0.625rem] uppercase"
                title="Deployed LoRAs that none of the checkpoints above accounts for — imported before run tagging, or copied into the folder by hand">
                Also in ComfyUI ({checkpointLorasLabel}) — not from a checkpoint listed above
              </span>
              {otherImported.map((c) => (
                <div key={c.filename} className="flex items-center gap-2 text-[0.6875rem]">
                  {/* Source run of this deployed file: two look-alike LoRAs from
                      different runs are now distinguishable at a glance. Files
                      imported before run tagging carry no id → "run unknown". */}
                  {c.run_id != null
                    ? <RunIdChip source={c.run_source} recordId={c.record_id}
                        cloudId={c.run_source === 'cloud' ? c.run_id : null} />
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
                  {/* Same operation as ⏏ Undeploy above, so it wears the same
                      name: it moves the ComfyUI copy to the trash, it does not
                      destroy anything permanently. */}
                  <button type="button" onClick={() => removeImported(c.filename, c.label)}
                    title={`Move this LoRA out of ComfyUI's ${checkpointLorasLabel} folder (to the trash — recoverable until you empty it in Settings)`}
                    className="ml-auto px-2 py-0.5 rounded border border-emerald-500/40 bg-emerald-600/5 text-emerald-200/90 hover:bg-emerald-600/20">
                    ⏏ Undeploy
                  </button>
                </div>
              ))}
            </div>
          )}

          </>)}
        </div>
      </details>
      </CheckpointPortal>

      {/* Portalled, like the ▶ Continue dialog above and for the same reason:
          this panel lives in a section the workspace hides with display:none,
          and the ▶ Continue buttons are clickable from ANOTHER section. Rendered
          in place, the report would open invisible and the launch would wait
          forever on an answer nobody could give. It is also z-[9992] — ABOVE the
          continue dialog that raises it, which now stays open behind it. */}
      {preflightReport && createPortal((
        <PreflightModal report={preflightReport} datasetId={ds.currentId} ds={ds}
          onResolve={resolvePreflight} />
      ), document.body)}

      {cloudDialog && (
        <CloudLaunchDialog
          datasetId={ds.currentId} trainType={trainType} variant={variant}
          trainingMode={trainingMode}
          base={base} steps={stepsN}
          keptCount={keptCount} cloudStatus={cloudStatus}
          preflightTokenIssue={hfCloudTokenIssue}
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
                className="px-3 py-1.5 rounded-lg bg-gradient-primary text-gray-950 text-sm font-semibold">
                ↺ Start fresh
              </button>
              <button type="button" onClick={() => resolveResume('continue')}
                title="Choose the checkpoint and explicitly restore either its full verified state or its weights only."
                className="px-3 py-1.5 rounded-lg border border-border bg-surface text-content text-sm hover:bg-surface-raised">
                ▶ Choose how to continue
              </button>
              <button type="button" onClick={() => resolveResume(null)}
                className="ml-auto px-3 py-1.5 rounded-lg text-content-muted hover:text-content text-sm">
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Rendered at the BODY level, not in the panel's own subtree: this panel
          is mounted once inside the Training section, which the workspace keeps
          `display:none` while another section is shown — but its checkpoint
          manager PORTALS into the Checkpoints section (CheckpointPortal above).
          So the ▶ Continue buttons and the ◉ Graph pills are visible and
          clickable from a section where everything this component renders on its
          own is invisible: the dialog opened inside the hidden container and the
          click looked completely dead (no dialog, no toast, no request — it only
          appeared once the user navigated to Training). A modal must live where
          it is seen, whichever section opened it. */}
      {continueOpen && createPortal((
        <ContinueDialog
          context={`${checkpointBaseLabel} · ${checkpointVariantDisplay}`}
          where={laneOfStep(continueInitialStep)}
          lanes={continueLanes}
          checkpoints={checkpoints}
          bestStep={bestEpoch?.available ? bestEpoch.best_step : null}
          initialFromStep={continueInitialStep}
          settings={{ save_every: advSave, sample_every: advSampleEvery,
            sample_prompts: adv?.sample_prompts,
            optimizer: adv?.optimizer, learning_rate: adv?.learning_rate }}
          // A local training in flight no longer freezes the whole dialog: the
          // Local lane is disabled with its reason, the Cloud lane stays usable.
          busy={continueSubmitting || (status.in_progress && !continueLanes.cloud.available)}
          error={continueError}
          onResolve={runContinue} />
      ), document.body)}

      {/* ⚙ / ⇄ opened from a checkpoint CARD — the same panels the Lineage
          graph mounts, portaled to body for the same reason as the Continue
          dialog above: a modal must live where it is seen, whichever section
          opened it. Nodes are resolved at render time from the shared tree;
          a node retired by a refetch simply closes its panel. */}
      {ckptDetailId != null && lineageNodeById(datasetGraph?.tree, ckptDetailId) && createPortal((
        <LineageDetailPanel
          node={lineageNodeById(datasetGraph?.tree, ckptDetailId)}
          onClose={() => setCkptDetailId(null)}
          onNodeChanged={() => loadDatasetGraph()}
          onNodeDeleted={() => { setCkptDetailId(null); loadDatasetGraph(); }} />
      ), document.body)}
      {ckptDiffIds.length === 2
        && lineageNodeById(datasetGraph?.tree, ckptDiffIds[0])
        && lineageNodeById(datasetGraph?.tree, ckptDiffIds[1]) && createPortal((
        <LineageDiffPanel
          a={lineageNodeById(datasetGraph?.tree, ckptDiffIds[0])}
          b={lineageNodeById(datasetGraph?.tree, ckptDiffIds[1])}
          onClose={() => setCkptDiffIds([])} />
      ), document.body)}

      {/* Same dialog as the Runs hub, portaled for the same reason as Continue
          above: a modal opened from Training must still be seen if the
          workspace has switched section. */}
      {createPortal((
        <CloudStopDialog run={cloudStopTarget}
          fullModel={!!cloudStopTarget && isFullTransformerRun(cloudStopTarget)}
          onCancel={() => setCloudStopTarget(null)}
          onConfirm={({ banHost }) => doStopCloud(cloudStopTarget, { banHost })} />
      ), document.body)}

    </div>
  );
}
