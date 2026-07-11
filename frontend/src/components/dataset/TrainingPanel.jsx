// react-frontend/src/components/dataset/TrainingPanel.jsx
import { useEffect, useRef, useState } from 'react';
import { getCsrfToken } from '../../api/fetchClient';
import { useCapabilities } from '../../context/CapabilitiesContext';
import { useToast } from '../common/Toast';
import TrainingProgress from './TrainingProgress';
import PreflightModal from './PreflightModal';

// Plancher dur / recommandé par famille — miroir de TRAIN_MIN_IMAGES côté serveur
// (le preflight reste l'autorité ; ceci ne sert qu'à désactiver le bouton tôt).
const TRAIN_MIN = { zimage: [12, 20], sdxl: [20, 30], krea: [15, 20] };

/** Panneau d'entraînement LoRA : lance l'UI ai-toolkit (pause ComfyUI),
 * affiche l'état, liste les checkpoints et importe celui choisi.
 * Poll régulier : c'est ce poll qui fait avancer la file (fin du courant → suivant). */
export default function TrainingPanel({ ds, keptCount, kind, onCheckpointsChange }) {
  const concept = kind === 'concept';
  const { caps } = useCapabilities();
  const toast = useToast();
  const [status, setStatus] = useState({ in_progress: false, installed: true, queue: [], current: null });
  const [checkpoints, setCheckpoints] = useState([]);
  const [ckLoaded, setCkLoaded] = useState(false);
  const [imported, setImported] = useState([]);
  const [enqErr, setEnqErr] = useState(null);
  // Base d'entraînement (officielle ou merge custom) + variante + conversion.
  const [baseInfo, setBaseInfo] = useState(null);
  const [base, setBase] = useState('');
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
    } catch { /* ignore */ }
  };
  // Poll toutes les 10 s : avance la file côté serveur + maj de l'UI. Skipped
  // entirely while training is hidden (ai-toolkit not configured) — no point
  // hitting endpoints the backend doesn't expose in that state.
  useEffect(() => {
    if (!caps.training_visible) return undefined;
    refreshStatus();
    const id = setInterval(refreshStatus, 10000);
    return () => clearInterval(id);
  }, [caps.training_visible]); // eslint-disable-line react-hooks/exhaustive-deps

  // Charge les bases + la base/variante du dataset au montage.
  useEffect(() => {
    if (!caps.training_visible) return undefined;
    let alive = true;
    ds.trainBaseInfo?.().then((info) => {
      if (alive && info) {
        setBaseInfo(info); setBase(info.base || '');
        // Défaut family-aware : Krea sans variante persistée → Raw (reco officielle
        // « train on Raw, validate on Turbo ») ; les autres familles → Turbo.
        setVariant(info.variant || ((info.train_type || 'zimage') === 'krea' ? 'base' : 'turbo'));
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
  // La conversion diffusers ne concerne QUE Z-Image (SDXL = single-file direct).
  const needsConversion = trainType === 'zimage' && isCustomBase;
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
    const list = baseInfo?.bases_by_type?.[t] || [];
    setBase(t === 'sdxl' ? (list[0]?.value || '') : '');
    // Krea → Raw par défaut (reco officielle « train on Raw, validate on Turbo »).
    if (t === 'krea') setVariant('base');
    ds.setDatasetTrainType?.(t);
  };

  // Réglages avancés effectifs (client-side pour que le défaut family-aware du rank
  // suive un changement de type SANS re-fetch). `adv.rank` null = Auto.
  const advRankChoice = adv?.rank ?? 'auto';
  const advDefaultRank = trainType === 'zimage' ? 16 : 32;   // miroir de _DEFAULT_RANK
  const advEffRank = advRankChoice === 'auto' ? advDefaultRank : advRankChoice;
  const advEffAlpha = trainType === 'sdxl' ? Math.max(1, Math.floor(advEffRank / 2)) : advEffRank;
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
    let d = await postTrain(`/api/dataset/${ds.currentId}/train/enqueue`, { base_model: base, variant, train_type: trainType, masked, steps: stepsN });
    if (d && d.ok === false && String(d.error || '').includes('MISMATCH_CAPTION')) {
      if (window.confirm(String(d.error).replace('MISMATCH_CAPTION: ', '') + '\n\nQueue anyway (force)?')) {
        d = await postTrain(`/api/dataset/${ds.currentId}/train/enqueue`,
          { base_model: base, variant, train_type: trainType, masked, steps: stepsN, allow_caption_mismatch: true });
      } else {
        d = null; // declined — matches ds.train(): no error surfaced, the confirm WAS the answer
      }
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
    let d = await postTrain(`/api/dataset/${ds.currentId}/train/schedule`,
      { at: schedAt, base_model: base, variant, train_type: trainType, masked, steps: stepsN });
    if (d && d.ok === false && String(d.error || '').includes('MISMATCH_CAPTION')) {
      if (window.confirm(String(d.error).replace('MISMATCH_CAPTION: ', '') + '\n\nSchedule anyway (force)?')) {
        d = await postTrain(`/api/dataset/${ds.currentId}/train/schedule`,
          { at: schedAt, base_model: base, variant, train_type: trainType, masked, steps: stepsN, allow_caption_mismatch: true });
      } else {
        d = null; // declined — matches ds.train(): no error surfaced, the confirm WAS the answer
      }
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
    setCkLoaded(true);
    onCheckpointsChange?.(Array.isArray(data.checkpoints) ? data.checkpoints.length : 0);
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

  // Estimation des steps adaptatifs (~120/image, bornés [1500,3500]) — purement
  // indicative ; le backend recalcule la valeur autoritaire au lancement.
  const recoSteps = Math.max(1500, Math.min(3500, Math.round((keptCount * 120) / 100) * 100));
  // Libellé lisible de la base sélectionnée (pour étiqueter les checkpoints de CE run).
  const baseLabel = currentBases.find((b) => b.value === base)?.label || (base || 'Official');
  const typeLabel = trainType === 'sdxl' ? 'SDXL' : trainType === 'krea' ? 'Krea 2' : 'Z-Image';
  const lorasLabel = trainType === 'sdxl' ? 'loras/sdxl' : trainType === 'krea' ? 'loras/krea' : 'loras/z image';

  // Panel gated off (ai-toolkit not configured): the workspace's checkpoint
  // count must not keep a stale value from a previous dataset/session.
  useEffect(() => {
    if (!caps.training_visible) onCheckpointsChange?.(0);
  }, [caps.training_visible]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!caps.training_visible) {
    return (
      <div className="flex items-center gap-2 rounded-lg border border-border bg-surface p-3 text-content-muted text-sm">
        <span aria-hidden>🎓</span>
        Training requires ai-toolkit — set its folder in Settings.
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
          ? <span aria-live="polite" className="ml-auto text-indigo-300 text-[0.6875rem]">
              <span aria-hidden>⏳</span> {status.current?.name ? `« ${status.current.name} » running` : 'running'} — ComfyUI paused
            </span>
          : <span aria-live="polite" className="ml-auto text-content-subtle text-[0.6875rem]">{keptCount} image(s) kept</span>}
      </div>

      {/* Live progress of THIS dataset's run: bar + loss sparkline + sample
          previews. Only while it is the one training (queued/other runs: no poll). */}
      {status.in_progress && status.current?.dataset_id === ds.currentId && (
        <TrainingProgress datasetId={ds.currentId} base={base} trainType={trainType} />
      )}

      {/* --- Chemin essentiel : choisir le type de LoRA et lancer. Le reste
           (base/variante, masked, plafond de steps, programmation) vit dans
           « Advanced options » ci-dessous — replié par défaut, tout y reste
           accessible en un clic. --- */}
      <div className="flex items-center gap-2 flex-wrap rounded-lg border border-border bg-surface px-3 py-2">
        <span className="text-content-muted text-[0.625rem] uppercase">LoRA type</span>
        <select value={trainType} onChange={(e) => onTypeChange(e.target.value)}
          aria-label="Type of LoRA to train"
          title="Z-Image (prose, Qwen3 encoder) ~20 img · SDXL (ComfyUI checkpoints) ~30 img · Krea 2 (prose, base fixe Turbo) ~20 img"
          className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
          <option value="zimage">Z-Image (~20 img)</option>
          <option value="sdxl">SDXL (~30 img)</option>
          <option value="krea">Krea 2 (~20 img)</option>
        </select>
        <button type="button" disabled={!status.installed || keptCount < (TRAIN_MIN[trainType]?.[0] ?? 12) || status.in_progress || baseBlocksTrain || sdxlNeedsBase}
          title={baseBlocksTrain ? 'Convert the custom base first'
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
            let d = await ds.train({ baseModel: base, variant, trainType, masked, steps: stepsN, fresh });
            if (d && d.ok === false && String(d.error || '').includes('MISMATCH_CAPTION')) {
              if (window.confirm(String(d.error).replace('MISMATCH_CAPTION: ', '') + '\n\nTrain anyway (force)?')) {
                await ds.train({ baseModel: base, variant, trainType, masked, steps: stepsN, allowCaptionMismatch: true, fresh });
              }
            }
            refreshStatus();
          }}
          className="px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
          <span aria-hidden>🚀</span> Train the LoRA
        </button>
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
          base « {baseLabel} » · {masked ? 'masked' : 'unmasked'} · {stepsOverride.trim() ? `${stepsN} steps` : 'adaptive steps'}
        </span>
      </div>

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

      <details className="rounded-lg border border-border bg-surface open:pb-2.5">
        <summary className="cursor-pointer select-none px-3 py-2 text-sm text-content font-semibold">
          ⚙️ Advanced options
          <span className="ml-2 font-normal text-content-subtle text-[0.6875rem]">
            base &amp; variant · rank · resolution · masked · steps · scheduling
          </span>
        </summary>
        <div className="px-3 pt-1 flex flex-col gap-2">
          {/* --- Base d'entraînement : officielle (recommandé) ou merge ComfyUI custom.
               Affichée MÊME pendant un training en cours → choisir la base du job mis
               en file (sinon « Mettre en file » réutilisait silencieusement la base persistée). --- */}
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-content-muted text-[0.625rem] uppercase">
                Base{status.in_progress ? ' (next queued job)' : ''}
              </span>
              <select value={base} onChange={(e) => setBase(e.target.value)}
                aria-label="Base model"
                className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem] max-w-[230px]">
                {(currentBases.length ? currentBases
                  : [{ value: '', label: trainType === 'sdxl' ? (comfyConfigured ? 'No SDXL checkpoint found' : 'ComfyUI not configured') : trainType === 'krea' ? 'Official — Krea 2' : 'Official — Z-Image-Turbo' }]).map((b) => (
                  <option key={b.value} value={b.value}>
                    {b.label}{b.value && baseInfo?.converted?.[b.value] ? ' ✓' : ''}
                  </option>
                ))}
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
            </div>
            {!comfyConfigured && trainType !== 'krea' && (
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
          </div>

          {/* Model & training knobs — researched defaults (see the Research note),
              editable per dataset. Each carries a plain-English "why / how". */}
          <div className="flex flex-col gap-2 rounded-lg border border-border bg-app/30 p-2.5">
            <span className="text-content-muted text-[0.625rem] uppercase tracking-wide">Model &amp; training</span>

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
                least-overfit one in the Test Studio; coarser saves disk. Only the last 10 are kept.
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

          <label className="flex items-center gap-1.5 text-[0.6875rem] text-content-muted cursor-pointer"
            title={concept
              ? 'For a CONCEPT dataset keep this OFF — a person mask would erase the very concept you are training. Masking only makes sense for a person/face LoRA.'
              : 'Masked training: a person mask is generated for every image (rembg, CPU) and the background only weighs 10% of the loss — identity binds to the face, not the room. Uncheck to train the old way.'}>
            <input type="checkbox" checked={masked} onChange={(e) => setMasked(e.target.checked)}
              aria-label="Masked training (background at 10%)"
              className="accent-primary w-3.5 h-3.5" />
            <span className={masked ? 'text-emerald-300' : ''}>🎭 Masked (bg 10%)</span>
            {concept && masked && (
              <span className="text-amber-300" title="A person mask would erase the concept.">⚠️ off recommended for concepts</span>
            )}
          </label>

          {!status.in_progress && keptCount >= 10 && (
            <label className="flex items-center gap-1.5 text-content-subtle text-[0.6875rem]"
              title="Target training steps. Leave empty for the adaptive value (~120/image, capped 1500–3500). Set a lower cap (e.g. 2000) to stop earlier — it trains faster and lighter; then pick the best checkpoint in the Test Studio. Applies to Train, Add to queue and Schedule.">
              <span className="uppercase text-content-muted text-[0.625rem]">Steps</span>
              <input type="number" min={500} step={100}
                value={stepsOverride}
                onChange={(e) => setStepsOverride(e.target.value)}
                placeholder={String(recoSteps)}
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
        <div className="flex flex-col gap-1 rounded-lg border border-indigo-400/30 bg-indigo-500/5 px-3 py-2">
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
      <details className="rounded-lg border border-border bg-surface open:pb-2.5">
        <summary className="cursor-pointer select-none px-3 py-2 text-sm text-content font-semibold">
          📦 Checkpoints &amp; trained LoRAs
          <span className="ml-2 font-normal text-content-subtle text-[0.6875rem]">
            {ckLoaded
              ? `${checkpoints.length} checkpoint(s) · ${imported.length} in ComfyUI`
              : 'the files your training runs produce'}
          </span>
        </summary>
        <div className="px-3 pt-1 flex flex-col gap-2">
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
                  {bestEpoch?.available && bestEpoch.checkpoint === c.filename && (
                    <span className="px-1.5 py-px rounded border border-amber-400/50 bg-amber-400/15 text-amber-200 font-semibold"
                      title={`Closest checkpoint to the best-scoring step (${bestEpoch.best_step})`}>
                      🏆 recommended
                    </span>
                  )}
                  <button type="button" onClick={() => ds.importCheckpoint(c.filename, base, trainType)}
                    className="ml-auto px-2 py-0.5 rounded bg-primary/20 border border-primary/40 text-white">
                    Import → {lorasLabel}
                  </button>
                </div>
              ))}
            </div>
          )}

          {ckLoaded && checkpoints.length === 0 && !status.in_progress && (
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

      {preflightReport && (
        <PreflightModal report={preflightReport} datasetId={ds.currentId} ds={ds}
          onResolve={resolvePreflight} />
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
