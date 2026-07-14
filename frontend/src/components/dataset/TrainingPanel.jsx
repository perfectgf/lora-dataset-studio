// react-frontend/src/components/dataset/TrainingPanel.jsx
import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Link } from 'react-router-dom';
import { getCsrfToken } from '../../api/fetchClient';
import { useCapabilities } from '../../context/CapabilitiesContext';
import { postJson } from '../../hooks/useDataset';
import { useToast } from '../common/Toast';
import TrainingProgress from './TrainingProgress';
import PreflightModal from './PreflightModal';

// Plancher dur / recommandé par famille — miroir de TRAIN_MIN_IMAGES côté serveur
// (le preflight reste l'autorité ; ceci ne sert qu'à désactiver le bouton tôt).
const TRAIN_MIN = { zimage: [12, 20], sdxl: [20, 30], krea: [15, 20], flux: [15, 20], flux2klein: [15, 20] };

// « Custom weights… » : valeur-sentinelle de l'entrée du sélecteur de base qui
// révèle le champ chemin. Les familles qui l'exposent + celles honorant VAE/TE
// (miroir de CUSTOM_WEIGHTS_FAMILIES / VAE_TE_OVERRIDE_FAMILIES côté serveur ;
// base-info les renvoie, ces défauts ne servent qu'avant son chargement).
const CUSTOM_BASE_SENTINEL = '__custom_weights__';
const DEFAULT_CUSTOM_FAMILIES = ['sdxl', 'krea', 'flux', 'flux2klein'];
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

/** Panneau d'entraînement LoRA : lance l'UI ai-toolkit (pause ComfyUI),
 * affiche l'état, liste les checkpoints et importe celui choisi.
 * Poll régulier : c'est ce poll qui fait avancer la file (fin du courant → suivant). */
export default function TrainingPanel({ ds, keptCount, kind, onCheckpointsChange,
                                        checkpointHost = null,
                                        navigationPanel = null,
                                        onNavigationStateChange,
                                        onPanelOpenChange }) {
  const concept = kind === 'concept' || kind === 'style';  // style: même chemin UI
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
  // Réglages ai-toolkit avancés éditables (rank / resolution / save_every /
  // sample_every / sample_prompts), chargés depuis base-info ; persistés par POST
  // /train/settings via ds.setTrainSettings.
  const [adv, setAdv] = useState(null);
  // Textarea des prompts de preview : état local (édition libre), sauvé au blur —
  // resynchronisé sur la valeur stockée canonique chaque fois que `adv` arrive/change.
  const [samplePromptsText, setSamplePromptsText] = useState('');
  // Presets de réglages avancés : snapshots nommés, partageables (fichier JSON).
  // Stockés bruts côté serveur ; la validation se fait à l'APPLICATION (clés
  // inconnues ignorées, valeurs invalides signalées) → tolérant aux versions.
  const [presets, setPresets] = useState([]);
  const [presetSel, setPresetSel] = useState('');
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
        setVariant(fam === 'flux2klein' && !['4b', '9b'].includes(v) ? '4b' : v);
        setTrainType(info.train_type || 'zimage');
        setAdv(info.train_settings || null);
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
  const onTypeChange = (t) => {
    setTrainType(t);
    // Switching family leaves custom-weights mode (the path is arch-specific).
    setCustomBase(false);
    const list = baseInfo?.bases_by_type?.[t] || [];
    setBase(t === 'sdxl' ? (list[0]?.value || '') : '');
    // Krea → Raw par défaut (reco officielle « train on Raw, validate on Turbo »).
    if (t === 'krea') setVariant('base');
    // FLUX.2 Klein → 4B par défaut (voie locale 16-24 GB ; le 9B est la voie cloud).
    if (t === 'flux2klein') setVariant('4b');
    ds.setDatasetTrainType?.(t);
  };

  // Réglages avancés effectifs (client-side pour que le défaut family-aware du rank
  // suive un changement de type SANS re-fetch). `adv.rank` null = Auto.
  const advRankChoice = adv?.rank ?? 'auto';
  const advDefaultRank = (trainType === 'zimage' || trainType === 'flux' || trainType === 'flux2klein') ? 16 : 32;   // miroir de _DEFAULT_RANK
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
  const LR_SCHED_LABELS = { constant: 'Constant (default)', constant_with_warmup: 'Warmup → constant', linear: 'Linear decay', cosine: 'Cosine decay', cosine_with_restarts: 'Cosine + restarts' };
  const advRes = adv?.resolution ?? '768,1024';
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

  // --- Presets (save / apply / import / export / delete) ---------------------
  const loadPresets = async () => {
    try {
      const r = await fetch('/api/train/presets', { credentials: 'include' });
      if (r.ok) setPresets((await r.json()).presets || []);
    } catch { /* list is best-effort */ }
  };
  useEffect(() => { loadPresets(); }, []);
  const selPreset = presets.find((p) => String(p.id) === presetSel) || null;
  const savePreset = async () => {
    const name = window.prompt('Preset name (an existing name is overwritten):');
    if (!name || !name.trim()) return;
    const d = await postTrain('/api/train/presets',
      { name: name.trim(), dataset_id: ds.currentId, train_type: trainType });
    if (d.ok === false) return toastTrainError(d, 'Preset save failed');
    toast.success(`Preset “${name.trim()}” saved.`);
    loadPresets();
  };
  const applyPreset = async () => {
    if (!selPreset) return;
    // Built-ins live in the code, not the DB — apply them by VALUE. Same
    // validated path server-side either way.
    const d = await postTrain(`/api/dataset/${ds.currentId}/train/presets/apply`,
      selPreset.builtin ? { settings: selPreset.settings } : { preset_id: selPreset.id });
    if (d.ok === false) return toastTrainError(d, 'Preset apply failed');
    setAdv(d.train_settings);
    const notes = [];
    if (d.ignored?.length) notes.push(`unknown here, ignored: ${d.ignored.join(', ')}`);
    if (d.rejected?.length) notes.push(`rejected: ${d.rejected.map((r) => r.key).join(', ')}`);
    if (notes.length) toast.warning(`Preset applied — ${notes.join(' · ')}`);
    else toast.success(`Preset “${selPreset.name}” applied.`);
  };
  const exportPreset = () => {
    if (!selPreset) return;
    const blob = new Blob([JSON.stringify({
      app: 'lora-dataset-studio', kind: 'training-preset', version: 1,
      name: selPreset.name, train_type: selPreset.train_type,
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
      const d = await postTrain('/api/train/presets',
        { name: String(j.name), train_type: j.train_type || trainType, settings: j.settings });
      if (d.ok === false) return toastTrainError(d, 'Preset import failed');
      toast.success(`Preset “${j.name}” imported — select it and Apply.`);
      loadPresets();
    } catch {
      toast.error('Unreadable preset file.');
    }
  };
  const deletePreset = async () => {
    if (!selPreset || selPreset.builtin) return;   // built-ins ship with the app
    if (!window.confirm(`Delete the preset “${selPreset.name}”?`)) return;
    try {
      await fetch(`/api/train/presets/${selPreset.id}`, {
        method: 'DELETE', headers: { 'X-CSRFToken': getCsrfToken() }, credentials: 'include',
      });
    } catch { /* the reload below shows the truth either way */ }
    setPresetSel('');
    loadPresets();
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
        `/api/dataset/${ds.currentId}/train/preflight?train_type=${encodeURIComponent(trainType)}`,
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
  const askResumeOrFresh = () => {
    if (!checkpoints.length) return Promise.resolve('resume');   // pas de run → lancement normal
    const latest = Math.max(...checkpoints.map((c) => c.step));
    const final = checkpoints.some((c) => c.final);
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
  // Dataset CONCEPT : masked OFF par défaut (un masque « personne » effacerait le
  // concept qu'on veut apprendre). On force l'état SANS écrire la préférence perso
  // (setMaskedS direct) → rouvrir un personnage retrouve ON. Rejoué au changement de
  // dataset ou de nature.
  useEffect(() => {
    if (concept) setMaskedS(false);
    else { try { setMaskedS(localStorage.getItem('trainMasked_v1') !== '0'); } catch { setMaskedS(true); } }
  }, [ds.currentId, concept]); // eslint-disable-line react-hooks/exhaustive-deps
  // Masked ON but rembg (person-mask backend) unavailable → the export silently
  // drops the masks and trains UNMASKED. Surface that instead of lying about it.
  // `=== false` (not `!caps.masks`) so we don't warn before caps have loaded.
  const maskedRembgMissing = masked && !concept && caps.masks === false;
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

  // Les checkpoints sont propres à la base SÉLECTIONNÉE (un run = dataset+base).
  // Garde-fou : si appelé avec autre chose qu'une string (ex. onClick passe un
  // event), on retombe sur `base` au lieu d'envoyer [object Object] à l'API.
  const loadCheckpoints = async (forBase) => {
    const b = (typeof forBase === 'string') ? forBase : base;
    const data = await ds.listCheckpoints(b, trainType);
    setCheckpoints(data.checkpoints || []);
    setImported(data.imported || []);
    // Provenance : dernière version enregistrée du dataset vs son état ACTUEL
    // (alerte « le dataset a changé depuis vN » + numéro de la prochaine version).
    setDatasetState(data.dataset_state || null);
    setCloudCkpts(data.cloud_checkpoints || []);
    setDiskUsage(data.disk_usage || null);
    // Rationale du barème adaptatif (backend = source de vérité) : affiché en tooltip
    // du champ Steps pour que l'app EXPLIQUE le nombre au lieu de le décréter.
    setStepsInfo(data.recommended_steps_info || null);
    setCkLoaded(true);
    onCheckpointsChange?.(
      (Array.isArray(data.checkpoints) ? data.checkpoints.length : 0)
      + (Array.isArray(data.imported) ? data.imported.length : 0),
    );
  };
  // Recharge dès que la base change (sinon le panneau montrait les checkpoints du
  // dernier run + un « Continuer » trompeur, quelle que soit la base choisie). On
  // attend baseInfo pour charger directement la BONNE base persistée (pas de flash
  // « Officiel » avant que la base du dataset soit appliquée).
  useEffect(() => {
    if (!caps.training_visible || !ds.currentId || !baseInfo) return;
    loadCheckpoints(base);
    // trainType dans les deps : changer de famille (Z-Image/SDXL/Krea) recharge la
    // liste « IN COMFYUI » + les checkpoints pour CETTE famille (sinon liste figée).
  }, [base, trainType, ds.currentId, baseInfo, caps.training_visible]); // eslint-disable-line react-hooks/exhaustive-deps
  const removeImported = async (filename, label) => {
    // Guard-rail: this LoRA may be the one the Studio's ★ best settings point to —
    // deleting it silently breaks the saved winning combo.
    const best = ds.data?.best_settings;
    const isBest = best?.lora_filename
      && String(best.lora_filename).split(/[\\/]/).pop() === String(filename).split(/[\\/]/).pop();
    const msg = isBest
      ? `⚠ « ${label} » is the LoRA saved as this dataset's ★ BEST SETTINGS in the Test Studio.\n\nDelete it anyway? The saved combo will stop working.`
      : `Permanently delete « ${label} » from ComfyUI's ${lorasLabel} folder?`;
    if (!window.confirm(msg)) return;
    await ds.deleteCheckpoint(filename, trainType);
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
  useEffect(() => { setBestEpoch(null); }, [base, trainType, ds.currentId]);
  const findBestEpoch = async () => {
    setBestEpochBusy(true);
    try {
      const d = await postTrain(`/api/dataset/${ds.currentId}/train/best-epoch`,
        { base_model: base, train_type: trainType });
      if (d && d.ok === false) { toastTrainError(d, 'best-epoch scoring failed'); return; }
      setBestEpoch(d);
    } finally {
      setBestEpochBusy(false);
    }
  };

  // Estimation des steps adaptatifs — purement indicative ; le backend recalcule la
  // valeur autoritaire au lancement (même barème). Character : ~120/image, bornés
  // [1500,3500]. Concept/style : SOUS-LINÉAIRE 475·√n, bornés [2000,12000] — un gros
  // set doit généraliser, pas mémoriser (à 400 img : ~9500 steps, pas 3500).
  const recoSteps = concept
    ? Math.max(2000, Math.min(12000, Math.round((475 * Math.sqrt(Math.max(keptCount, 1))) / 100) * 100))
    : Math.max(1500, Math.min(3500, Math.round((keptCount * 120) / 100) * 100));
  // Libellé lisible de la base sélectionnée (pour étiqueter les checkpoints de CE run).
  // Custom weights → basename du fichier (jamais le chemin complet dans le résumé).
  const baseLabel = customBase && base
    ? `custom: ${baseName(base)}`
    : (currentBases.find((b) => b.value === base)?.label || (base || 'Official'));
  const typeLabel = trainType === 'sdxl' ? 'SDXL' : trainType === 'krea' ? 'Krea 2' : trainType === 'flux' ? 'FLUX.1' : trainType === 'flux2klein' ? 'FLUX.2 Klein' : 'Z-Image';
  const lorasLabel = trainType === 'sdxl' ? 'loras/sdxl' : trainType === 'krea' ? 'loras/krea' : trainType === 'flux' ? 'loras/flux' : trainType === 'flux2klein' ? 'loras/flux2klein' : 'loras/z image';

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
  const cloudTooFewImages = keptCount < (TRAIN_MIN[trainType]?.[0] ?? 12);
  const cloudLimitReached = actives.length >= (cloudStatus.limit || 1);
  const cloudDisabledReason =
    trainType === 'sdxl'
      ? 'SDXL trains locally only — the cloud lane covers Z-Image, Krea 2 and FLUX.2 Klein'
    : trainType === 'flux'
      ? 'FLUX.1 trains locally only — the cloud lane covers Z-Image, Krea 2 and FLUX.2 Klein'
    : (customBase || vaePath || tePath)
      ? 'Custom weights are local-only — cloud training uses the official Hugging Face bases'
    : cloudTooFewImages
      ? `Only ${keptCount} image(s) kept — the cloud minimum for ${typeLabel} is ${TRAIN_MIN[trainType]?.[0] ?? 12}`
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
    let body = { variant, train_type: trainType, masked,
      ...(stepsN ? { steps: stepsN } : {}), ...(gpuName ? { gpu_name: gpuName } : {}) };
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
        <TrainingProgress datasetId={ds.currentId} base={base} trainType={trainType} />
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
          <TrainingProgress datasetId={ds.currentId} base={base} trainType={trainType} cloud />
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
          aria-label="Type of LoRA to train"
          title="Z-Image (prose, Qwen3 encoder) ~20 img · SDXL (ComfyUI checkpoints) ~30 img · Krea 2 (prose, base fixe Turbo) ~20 img · FLUX.1-dev (prose, gated HF, local-only) ~20 img · FLUX.2 Klein (prose, gated HF, 4B local / 9B cloud) ~20 img"
          className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
          <option value="zimage">Z-Image (~20 img)</option>
          <option value="sdxl">SDXL (~30 img)</option>
          <option value="krea">Krea 2 (~20 img)</option>
          <option value="flux">FLUX.1 (~20 img)</option>
          <option value="flux2klein">FLUX.2 Klein (~20 img)</option>
        </select>
        <button type="button" disabled={!status.installed || keptCount < (TRAIN_MIN[trainType]?.[0] ?? 12) || status.in_progress || baseBlocksTrain || sdxlNeedsBase || customWeightsEmpty}
          title={baseBlocksTrain ? 'Convert the custom base first'
            : customWeightsEmpty ? 'Enter the path to your custom weights .safetensors'
            : sdxlNeedsBase ? 'Choose a base SDXL checkpoint'
            : keptCount < (TRAIN_MIN[trainType]?.[0] ?? 12)
              ? `${keptCount} kept image(s) — the minimum for ${typeLabel} is ${TRAIN_MIN[trainType]?.[0] ?? 12}`
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
        {status.in_progress && status.installed && keptCount >= (TRAIN_MIN[trainType]?.[0] ?? 12) && (
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
          base « {baseLabel} » · {maskedRembgMissing ? 'unmasked (rembg missing)' : masked ? 'masked' : 'unmasked'} · {stepsOverride.trim() ? `${stepsN} steps` : 'adaptive steps'}{advNetworkType === 'lokr' ? ' · LoKr' : ''}{advEma ? ` · EMA ${advEma}` : ''}
        </span>
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
              {presets.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.builtin ? '★ ' : ''}{p.name} ({p.train_type})
                </option>
              ))}
            </select>
            <button type="button" onClick={applyPreset} disabled={!selPreset}
              title="Replace this dataset's advanced settings with the selected preset"
              className="px-2.5 py-1 rounded-lg bg-primary/20 border border-primary/40 text-white text-[0.75rem] font-semibold disabled:opacity-40">
              Apply
            </button>
            <span className="mx-0.5 text-content-subtle" aria-hidden>·</span>
            <button type="button" onClick={savePreset}
              title="Save this dataset's current advanced settings as a named preset"
              className="px-2.5 py-1 rounded-lg bg-surface-raised border border-border text-content text-[0.75rem]">
              💾 Save current…
            </button>
            <button type="button" onClick={() => presetFileRef.current?.click()}
              title="Import a preset from a JSON file (exported from any app version — unknown options are ignored at apply time)"
              className="px-2.5 py-1 rounded-lg bg-surface-raised border border-border text-content text-[0.75rem]">
              ⬆ Import
            </button>
            <button type="button" onClick={exportPreset} disabled={!selPreset}
              title="Download the selected preset as a shareable JSON file"
              className="px-2.5 py-1 rounded-lg bg-surface-raised border border-border text-content text-[0.75rem] disabled:opacity-40">
              ⬇ Export
            </button>
            <button type="button" onClick={deletePreset} disabled={!selPreset || selPreset.builtin}
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
                    {b.label}{b.value && baseInfo?.converted?.[b.value] ? ' ✓' : ''}
                  </option>
                ))}
                {/* Local-only: a free path to a .safetensors of the SAME architecture. */}
                {customSupported && (
                  <option value={CUSTOM_BASE_SENTINEL}>Custom weights… (local file)</option>
                )}
              </select>
              {trainType === 'zimage' && isCustomBase && (
                <select value={variant} onChange={(e) => setVariant(e.target.value)}
                  title="Base model variant (sets the de-distillation adapter + the sampler)"
                  className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
                  <option value="turbo">Turbo (distilled)</option>
                  <option value="base">Base (non-distilled)</option>
                  <option value="deturbo">De-Turbo</option>
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
                <b className="text-content-muted font-medium">Why:</b> how much capacity the LoRA has to memorize the
                identity. <b className="text-content-muted font-medium">How:</b> higher (32+) captures a hard face more
                faithfully but makes a bigger file and can overfit small sets; lower (16) is lighter and fine for clean
                frontal datasets. ai-toolkit ties alpha to rank (SDXL keeps alpha = rank ÷ 2).
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
                <b className="text-content-muted font-medium"> How:</b> one prompt per line, up to {advMaxPrompts}. Your
                trigger word is added automatically if you leave it out. {concept
                  ? 'Leave empty for concept-friendly defaults (the greyed text) — the portrait wording only fits a person LoRA.'
                  : 'Leave empty for the portrait defaults shown greyed.'}
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
                  <b className="text-content-muted font-medium">Why:</b> the anti-overfit regulariser for small sets — randomly
                  drops LoRA weights so it generalises instead of memorising. <b className="text-content-muted font-medium">How:</b> Off
                  by default; 0.05–0.1 is a gentle start for a tiny (≤20-image) dataset, higher = stronger regularisation.
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
                    &quot;character&quot; knob for flow-matching models (Z-Image / Krea). <b className="text-content-muted font-medium">How:</b> Auto
                    = the tuned default ({advTimestepDefault}); <i>sigmoid</i> favours the subject, <i>shift</i>/<i>weighted</i> shift
                    the detail-vs-structure balance.
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
            title={concept
              ? 'For a CONCEPT dataset keep this OFF — a person mask would erase the very concept you are training. Masking only makes sense for a person/face LoRA.'
              : 'Masked training: a person mask is generated for every image (rembg, CPU) and the background only weighs 10% of the loss — identity binds to the face, not the room. Uncheck to train the old way.'}>
            <input type="checkbox" checked={masked} onChange={(e) => setMasked(e.target.checked)}
              aria-label="Masked training (background at 10%)"
              className="accent-primary w-3.5 h-3.5" />
            <span className={masked && !maskedRembgMissing ? 'text-emerald-300' : ''}>🎭 Masked (bg 10%)</span>
            {concept && masked && (
              <span className="text-amber-300" title="A person mask would erase the concept.">⚠️ off recommended for concepts</span>
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
                : concept
                  ? 'Target training steps. Leave empty for the adaptive value (sublinear 475·√images, capped 2000–12000): the bigger the set, the fewer views per image, so the LoRA generalizes instead of memorizing shots. Applies to Train, Add to queue and Schedule.'
                  : 'Target training steps. Leave empty for the adaptive value (~120/image, capped 1500–3500). Set a lower cap (e.g. 2000) to stop earlier — it trains faster and lighter; then pick the best checkpoint in the Test Studio. Applies to Train, Add to queue and Schedule.'}>
              <span className="uppercase text-content-muted text-[0.625rem]">Steps</span>
              <input type="number" min={500} step={100}
                value={stepsOverride}
                onChange={(e) => setStepsOverride(e.target.value)}
                placeholder={String(stepsInfo?.steps ?? recoSteps)}
                aria-label="Target training steps (leave empty for adaptive)"
                className="w-[4.5rem] rounded border border-border bg-app/60 px-1.5 py-0.5 text-content tabular-nums text-[0.75rem]" />
              <span>{stepsOverride.trim() ? 'target' : `≈ adaptive (${keptCount} img)`}</span>
            </label>
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

      {keptCount < (TRAIN_MIN[trainType]?.[1] ?? 20) && (
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
            <button type="button" onClick={() => loadCheckpoints(base)}
              title="Reload the checkpoint list for the selected base"
              className="px-3 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-xs font-semibold">
              ↻ Refresh checkpoints
            </button>
            {/* Ouvre les dossiers dans l'explorateur du poste (app locale) :
                loras = imports ComfyUI de la famille ; run = checkpoints bruts. */}
            <button type="button"
              onClick={() => postTrain(`/api/dataset/${ds.currentId}/train/open-folder`,
                { target: 'loras', train_type: trainType })}
              title={`Open the ComfyUI folder where imported ${typeLabel} LoRAs live (loras/${trainType === 'zimage' ? 'z image' : trainType})`}
              className="px-3 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-xs font-semibold">
              📂 LoRA folder
            </button>
            <button type="button"
              onClick={() => postTrain(`/api/dataset/${ds.currentId}/train/open-folder`,
                { target: 'run', train_type: trainType, base_model: base })}
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
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-content-muted text-[0.625rem] uppercase">
                  Checkpoints — base « {baseLabel} » (pick the earliest one that holds the identity)
                </span>
                <button type="button" disabled={bestEpochBusy}
                  onClick={findBestEpoch}
                  title="Scores every training sample vs the reference photo (face similarity, CPU) and recommends the checkpoint that holds the identity best — needs the Quality tools (ML extras)."
                  className="px-2.5 py-1 rounded-lg bg-amber-500/15 border border-amber-400/40 text-amber-200 text-[0.6875rem] font-semibold disabled:opacity-40">
                  {bestEpochBusy ? '🏆 Scoring samples…' : '🏆 Find best epoch'}
                </button>
                <button type="button" disabled={status.in_progress || baseBlocksTrain}
                  onClick={async () => {
                    const last = Math.max(...checkpoints.map((c) => c.step));
                    if (window.confirm(`Resume training « ${baseLabel} » from step ${last} and continue for +1000 steps (→ ${last + 1000})?`)) {
                      await ds.continueTraining(1000, base, variant); refreshStatus(); loadCheckpoints(base);
                    }
                  }}
                  title={baseBlocksTrain ? 'Convert the custom base first' : 'Resumes from this base’s last checkpoint and trains 1000 more steps'}
                  className="ml-auto px-2.5 py-1 rounded-lg bg-indigo-500/20 border border-indigo-400/40 text-indigo-200 text-[0.6875rem] font-semibold disabled:opacity-40">
                  ▶ Continue training (+1000)
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
                      try { await ds.importCheckpoint(c.filename, base, trainType); }
                      finally { loadCheckpoints(base); }
                    }}
                    className="ml-auto px-2 py-0.5 rounded bg-primary/20 border border-primary/40 text-white">
                    Import → {lorasLabel}
                  </button>
                  <button type="button"
                    onClick={async () => {
                      if (!window.confirm(`Move « ${c.filename} » to the trash?\n\nRecoverable until you empty the trash in Settings.`)) return;
                      const d = await postTrain(`/api/dataset/${ds.currentId}/train/run-checkpoint/delete`,
                        { filename: c.filename, base_model: base, train_type: trainType });
                      if (d.ok === false) toastTrainError(d, 'Delete failed');
                      loadCheckpoints(base);
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
                      { keep_filenames: keep, base_model: base, train_type: trainType });
                    if (d.ok === false) toastTrainError(d, 'Cleanup failed');
                    loadCheckpoints(base);
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

          {cloudCkpts.length > 0 && (
            <div className="flex flex-col gap-1">
              <span className="text-content-muted text-[0.625rem] uppercase">
                ☁ Cloud checkpoints (synced locally — every epoch harvested from the pod)
              </span>
              {cloudCkpts.map((c) => (
                <div key={`cr${c.run_id}-${c.filename}`} className="flex items-center gap-2 text-[0.6875rem]">
                  <span className={c.final ? 'text-green-400 font-semibold' : 'text-content'}>
                    {c.final ? '✓ final (training complete)' : `step ${c.step}`}
                  </span>
                  <span className="px-1.5 py-px rounded border border-sky-500/40 bg-sky-500/10 text-sky-200"
                    title={`Cloud run #${c.run_id}${c.version ? ` · dataset v${c.version}` : ''}${c.trained_at ? ` · ${new Date(/[Z+]/.test(c.trained_at) ? c.trained_at : `${c.trained_at}Z`).toLocaleString()}` : ''}`}>
                    ☁{c.version ? ` v${c.version}` : ''}{c.active ? ' · run in progress' : ''}
                  </span>
                  <button type="button"
                    onClick={async () => {
                      const d = await postTrain(`/api/dataset/${ds.currentId}/train/import`,
                        { filename: c.filename, train_type: trainType, cloud_run_id: c.run_id });
                      // Success must be VISIBLE: without the toast a working
                      // import looked like a dead button (user-observed).
                      if (d.ok === false) toastTrainError(d, 'Import failed');
                      else toast.success(`LoRA imported: ${d.dest || c.filename}`);
                      loadCheckpoints(base);
                    }}
                    title={c.active ? 'Import the latest synced save — the run keeps training' : 'Import this cloud checkpoint into ComfyUI'}
                    className="ml-auto px-2 py-0.5 rounded bg-primary/20 border border-primary/40 text-white">
                    Import → {lorasLabel}
                  </button>
                  {!c.active && (
                    <button type="button"
                      onClick={async () => {
                        if (!window.confirm(`Move « ${c.filename} » to the trash?\n\nRecoverable until you empty the trash in Settings.`)) return;
                        const d = await postTrain(`/api/dataset/${ds.currentId}/train/run-checkpoint/delete`,
                          { filename: c.filename, cloud_run_id: c.run_id });
                        if (d.ok === false) toastTrainError(d, 'Delete failed');
                        loadCheckpoints(base);
                      }}
                      title="Move this cloud save to the trash"
                      className="px-2 py-0.5 rounded bg-red-500/15 border border-red-500/40 text-red-300">
                      🗑
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}

          {ckLoaded && checkpoints.length === 0 && cloudCkpts.length === 0 && !status.in_progress && (
            <p className="m-0 text-content-subtle text-[0.625rem]">
              No checkpoint for base « {baseLabel} » — run a training on this base first.
            </p>
          )}

          {imported.length > 0 && (
            <div className="flex flex-col gap-1">
              <span className="text-content-muted text-[0.625rem] uppercase">
                In ComfyUI ({lorasLabel}) — delete the ones you no longer need
              </span>
              {imported.map((c) => (
                <div key={c.filename} className="flex items-center gap-2 text-[0.6875rem]">
                  <span className="text-content break-all">{c.label}</span>
                  {/* Retrofit signal: the file's REAL arch (read from its header)
                      contradicts this folder's family — a mislabelled deploy that
                      would test as a silent no-op. No auto-move; just flag it. */}
                  {c.arch_mismatch && (
                    <span
                      title={`This file is a ${c.arch_label || c.arch_mismatch} LoRA, not ${lorasLabel} — testing it here has NO effect (ComfyUI silently drops it). Delete it and re-import under the ${c.arch_label || c.arch_mismatch} family.`}
                      className="px-1.5 py-0.5 rounded bg-amber-500/15 border border-amber-500/40 text-amber-300 whitespace-nowrap">
                      ⚠ {c.arch_label || c.arch_mismatch} LoRA
                    </span>
                  )}
                  <button type="button" onClick={() => removeImported(c.filename, c.label)}
                    title={`Delete this LoRA from ComfyUI's ${lorasLabel} folder`}
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
          datasetId={ds.currentId} trainType={trainType} steps={stepsN}
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

/* Launch-time GPU speed picker. Fetches live vast.ai offers grouped by GPU
   class (slowest→fastest), each with price/h and an APPROXIMATE training time
   and total run cost for this dataset+family. Picking a tier rents the cheapest
   live offer of that class; the price cap in Settings still bounds what's shown. */
function CloudLaunchDialog({ datasetId, trainType, steps, keptCount, cloudStatus, onClose, onLaunch }) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);     // {tiers, steps, family, max_price_per_hour}
  const [selected, setSelected] = useState(null);
  const [launching, setLaunching] = useState(false);

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
          <button type="button" onClick={go} disabled={!selected || launching}
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
