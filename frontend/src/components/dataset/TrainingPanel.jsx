// react-frontend/src/components/dataset/TrainingPanel.jsx
import { useEffect, useState } from 'react';
import { getCsrfToken } from '../../api/fetchClient';
import { useCapabilities } from '../../context/CapabilitiesContext';
import { useToast } from '../common/Toast';
import TrainingProgress from './TrainingProgress';

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
        setBaseInfo(info); setBase(info.base || ''); setVariant(info.variant || 'turbo');
        setTrainType(info.train_type || 'zimage');
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
    ds.setDatasetTrainType?.(t);
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
    if (!window.confirm(`Permanently delete « ${label} » from ComfyUI's ${lorasLabel} folder?`)) return;
    await ds.deleteCheckpoint(filename, trainType);
    loadCheckpoints();
  };
  const doPrepareBase = async () => {
    await ds.prepareBase(base);
    const info = await ds.trainBaseInfo();
    if (info) setBaseInfo(info);
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

      {/* --- Base d'entraînement : officielle (recommandé) ou merge ComfyUI custom.
           Affichée MÊME pendant un training en cours → choisir la base du job mis
           en file (sinon « Mettre en file » réutilisait silencieusement la base persistée). --- */}
      {(
        <div className="flex flex-col gap-1.5 rounded-lg border border-border bg-surface px-3 py-2">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-content-muted text-[0.625rem] uppercase">LoRA type</span>
            <select value={trainType} onChange={(e) => onTypeChange(e.target.value)}
              aria-label="Type of LoRA to train"
              title="Z-Image (prose, Qwen3 encoder) ~20 img · SDXL (ComfyUI checkpoints) ~30 img · Krea 2 (prose, base fixe Turbo) ~20 img"
              className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
              <option value="zimage">Z-Image (~20 img)</option>
              <option value="sdxl">SDXL (~30 img)</option>
              <option value="krea">Krea 2 (~20 img)</option>
            </select>
            <span className="text-content-muted text-[0.625rem] uppercase">
              Base{status.in_progress ? ' (next queued job)' : ''}
            </span>
            <select value={base} onChange={(e) => setBase(e.target.value)}
              aria-label="Base model"
              className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem] max-w-[230px]">
              {(currentBases.length ? currentBases
                : [{ value: '', label: trainType === 'sdxl' ? (comfyConfigured ? 'No SDXL checkpoint found' : 'ComfyUI not configured') : trainType === 'krea' ? 'Official — Krea 2 Turbo' : 'Official — Z-Image-Turbo' }]).map((b) => (
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
      )}

      <div className="flex items-center gap-2 flex-wrap">
        <button type="button" disabled={!status.installed || keptCount < 10 || status.in_progress || baseBlocksTrain || sdxlNeedsBase}
          title={baseBlocksTrain ? 'Convert the custom base first'
            : sdxlNeedsBase ? 'Choose a base SDXL checkpoint' : undefined}
          onClick={async () => {
            let d = await ds.train({ baseModel: base, variant, trainType, masked, steps: stepsN });
            if (d && d.ok === false && String(d.error || '').includes('MISMATCH_CAPTION')) {
              if (window.confirm(String(d.error).replace('MISMATCH_CAPTION: ', '') + '\n\nTrain anyway (force)?')) {
                await ds.train({ baseModel: base, variant, trainType, masked, steps: stepsN, allowCaptionMismatch: true });
              }
            }
            refreshStatus();
          }}
          className="px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
          <span aria-hidden>🚀</span> Train the LoRA
        </button>
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
        {status.in_progress && (
          <button type="button" onClick={async () => { await ds.stopTraining(); refreshStatus(); }}
            className="px-3 py-1.5 rounded-lg bg-red-600/80 text-white text-sm font-semibold">
            Finish / re-enable ComfyUI
          </button>
        )}
        {status.in_progress && status.installed && keptCount >= 10 && (
          <button type="button" disabled={queued || baseBlocksTrain} onClick={enqueue}
            title={baseBlocksTrain
              ? 'Convert the selected custom base first'
              : `Train THIS dataset on « ${baseLabel} » once the current training finishes`}
            className="px-3 py-1.5 rounded-lg bg-indigo-500/20 border border-indigo-400/40 text-indigo-200 text-sm font-semibold disabled:opacity-40">
            {queued ? '✓ Queued' : `➕ Add to queue (${baseLabel})`}
          </button>
        )}
        {status.installed && keptCount >= 10 && (
          <button type="button" disabled={queued || baseBlocksTrain} onClick={openSched}
            aria-expanded={showSched}
            title={baseBlocksTrain
              ? 'Convert the selected custom base first'
              : 'Schedule this training for a specific day and time — it will queue up if another training is running then'}
            className="px-3 py-1.5 rounded-lg bg-amber-500/15 border border-amber-400/40 text-amber-200 text-sm font-semibold disabled:opacity-40">
            {queued ? '✓ Queued' : '⏰ Schedule'}
          </button>
        )}
        {/* () => … sinon React passe l'event en 1er arg → forBase = PointerEvent
            → base_model=[object Object] → run inexistant → liste vide. */}
        <button type="button" onClick={() => loadCheckpoints(base)}
          className="px-3 py-1.5 rounded-lg bg-surface text-content text-sm">
          View checkpoints
        </button>
      </div>

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

      {keptCount < 10 && (
        <p className="m-0 text-content-subtle text-[0.625rem]">At least 10 kept images recommended before training.</p>
      )}

      {checkpoints.length > 0 && (
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-content-muted text-[0.625rem] uppercase">
              Checkpoints — base « {baseLabel} » (pick the earliest one that holds the identity)
            </span>
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
          {checkpoints.map((c) => (
            <div key={c.filename} className="flex items-center gap-2 text-[0.6875rem]">
              <span className={c.final ? 'text-green-400 font-semibold' : 'text-content'}>
                {c.final ? '✓ final (training complete)' : `step ${c.step}`}
              </span>
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
  );
}
