/** Variation catalog: presets + per-entry toggles + multiplier + Klein picker. */
import { useEffect, useMemo, useRef, useState } from 'react';
import KleinModelSetting from '../shared/KleinModelSetting';
import { useToast } from '../common/Toast';
import SettingsLink from '../common/SettingsLink';
import { useCapabilities } from '../../context/CapabilitiesContext';
import { apiFetch, putJson } from '../../api/fetchClient';
import ShotIllustration, { contextEmoji } from './ShotIllustration';
import { displayLabel } from '../../utils/labels';
import { generationLoraPresetPayload, sanitizeGenerationLoraPresets } from '../../utils/generationLoras';
import { kreaGenerationLoraPresetPayload, sanitizeKreaGenerationLoraPresets } from '../../utils/kreaGenerationLoras';
import { requestHelpTip } from '../../help/helpTips';
import { HelpBadge } from '../../help/HelpMode';
import {
  applyShotPreset,
  deleteShotPreset,
  loadShotPresets,
  persistShotPresets,
  renameShotPreset,
  saveShotPreset,
} from '../../utils/shotPresets';
import {
  applyShotImport, buildShotExport, parseShotImport, promoteCustomShot, MAX_IMPORT_BYTES,
} from '../../utils/shotImport';
import {
  ENGINE_ACCENTS, ENGINE_LABELS, billingEngines, canonicalEngines, engineBatches,
  estimateCost, generateBlockedReason, localQueuesBehindApi, localOnly, readEngines,
  readMode, totalImages, writeEngines, writeMode,
} from './engineSelection.js';
import { kreaUnavailableReason, groundingDescription } from '../../utils/kreaEngine.js';
import { kleinUnavailableReason } from '../../utils/localEngineReason.js';
import {
  SUBJECT_TYPES, SUBJECT_TYPE_LABELS, SUBJECT_TYPE_HINTS,
  normalizeSubjectType, framingLabel, defaultPresetKey,
} from './subjectTypes.js';

/** localStorage, or null when it can't be touched (private mode / SSR) — the
 *  engine helpers degrade to their defaults instead of throwing. */
const storage = () => (typeof localStorage === 'undefined' ? null : localStorage);

const FRAMING_LABEL = { face: 'Face', bust: 'Bust', body: 'Body', back: 'Back' };
// The framings a prompt suffix can target (same buckets the backend wraps by).
const SUFFIX_KEYS = ['face', 'bust', 'body', 'back'];
// Framing accent colors — shared by the section headers, the preset composition
// bars and the legend so the same hue always means the same framing.
const FRAMING_COLOR = {
  face: 'bg-indigo-400',
  bust: 'bg-violet-400',
  body: 'bg-sky-400',
  back: 'bg-slate-400',
};
// Training composition target (mirrors CompositionBar): used to highlight the
// variation cards of the framings that are still missing — a visual quota.
const TARGET = { face: 12, bust: 6, body: 6, back: 1 };

const PRESET_META = [
  { key: 'balanced_25', name: 'Balanced', hint: 'The all-round default: every framing covered in training proportions.' },
  { key: 'zimage_12', name: 'Z-Image 12', hint: 'Compact 12-shot set tuned for Z-Image LoRA training.' },
  { key: 'balanced_multiformat', name: 'Multi-format', hint: 'Balanced set with landscape / vertical / cinema frames mixed in.' },
  { key: 'face_focused', name: 'Face-focused', hint: 'Face only (close-ups + busts, varied formats, no body shots) — body stays generic.' },
  { key: 'fullbody_focused', name: 'Full-body', hint: 'Reliable full-body: ~50/50 identity (face+bust) and full-body + back, varied formats. For a character that must hold up full-length without losing the face.' },
  { key: 'body_emphasis', name: 'Body emphasis', hint: 'Body-fidelity pick: figure-revealing but API-safe outfits (fitted tops, swimwear at the beach/pool, sportswear, bodycon, backlit silhouette) so the body shape is actually visible in the training shots. For explicit content, generate with the local Klein engine instead.' },
];

/** Mini stacked bar showing a preset's framing mix (face/bust/body/back). */
function CompositionMiniBar({ counts, total }) {
  if (!total) return null;
  return (
    <span className="flex h-1.5 w-full rounded-full overflow-hidden bg-app/60" aria-hidden="true">
      {['face', 'bust', 'body', 'back'].map((fr) => counts[fr] ? (
        <span key={fr} className={FRAMING_COLOR[fr]} style={{ width: `${(counts[fr] / total) * 100}%` }} />
      ) : null)}
    </span>
  );
}

/** Minimal ChatGPT pictogram — hexagonal knot silhouette, currentColor. */
function ChatGptIcon({ className }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden="true" focusable="false">
      {[0, 60, 120, 180, 240, 300].map((a) => (
        <path key={a} transform={`rotate(${a} 16 16)`}
          d="M16 4.5 a 6.2 6.2 0 0 1 6.2 6.2 v 4 l -3.4 -2 v -2 a 2.8 2.8 0 0 0 -2.8 -2.8 z"
          fill="currentColor" />
      ))}
      <circle cx="16" cy="16" r="3.1" fill="none" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  );
}

/** Routing pictogram for the OpenRouter card: one input fanning out to several
 *  providers — which is exactly what the engine does (one key, many models). */
function RouterIcon({ className }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden="true" focusable="false">
      <g stroke="currentColor" strokeWidth="1.8" fill="none" strokeLinecap="round">
        <line x1="9" y1="16" x2="17" y2="16" />
        <path d="M17 16 C 21 16, 21 8, 25 8" />
        <path d="M17 16 C 21 16, 21 24, 25 24" />
      </g>
      <circle cx="7" cy="16" r="3" fill="currentColor" />
      <circle cx="25" cy="8" r="2.4" fill="currentColor" opacity="0.85" />
      <circle cx="25" cy="16" r="2.4" fill="currentColor" opacity="0.85" />
      <circle cx="25" cy="24" r="2.4" fill="currentColor" opacity="0.85" />
    </svg>
  );
}

/** Small inline GPU-chip pictogram for the local Klein engine card. */
function GpuIcon({ className }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden="true" focusable="false">
      <rect x="7" y="7" width="18" height="18" rx="3" fill="none" stroke="currentColor" strokeWidth="1.8" />
      <rect x="12" y="12" width="8" height="8" rx="1.5" fill="currentColor" opacity="0.85" />
      {[10, 16, 22].map((p) => (
        <g key={p} stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
          <line x1={p} y1="2.5" x2={p} y2="6" />
          <line x1={p} y1="26" x2={p} y2="29.5" />
          <line x1="2.5" y1={p} x2="6" y2={p} />
          <line x1="26" y1={p} x2="29.5" y2={p} />
        </g>
      ))}
    </svg>
  );
}

/** Krea 2 Identity Edit: a portrait frame with an identity anchor — it re-stages
 *  the SAME subject rather than generating a new one. Distinct silhouette from
 *  the GPU chip so the two local cards never read as the same engine. */
function IdentityFrameIcon({ className }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden="true" focusable="false">
      <rect x="4" y="4" width="24" height="24" rx="4" fill="none" stroke="currentColor" strokeWidth="1.8" />
      <circle cx="16" cy="13" r="4" fill="none" stroke="currentColor" strokeWidth="1.8" />
      <path d="M9 24c1.6-3.6 4.1-5.4 7-5.4s5.4 1.8 7 5.4" fill="none"
        stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      <circle cx="16" cy="13" r="1.2" fill="currentColor" />
    </svg>
  );
}

const MODE_CHOICES = [
  { id: 'split', name: 'Split across engines',
    desc: 'Each shot goes to ONE engine — same image count and cost as a single engine, but a more varied dataset.' },
  { id: 'all', name: 'All engines',
    desc: 'Every engine renders every shot — compare the results side by side, then keep the ones you like. Multiplies the cost.' },
];

/** One engine CHECKBOX card. A checkbox, not a radio: engines combine. Each
 *  carries its own accent (see ENGINE_ACCENTS) so a mixed run is readable —
 *  green is deliberately not one of them, it already means "kept / free". */
function EngineCard({ id, checked, available, generating, onToggle, icon, title, tags, hint }) {
  const accent = ENGINE_ACCENTS[id];
  return (
    <button type="button" role="checkbox" aria-checked={checked}
      aria-label={ENGINE_LABELS[id]}
      onClick={() => onToggle(id)}
      disabled={!available || !!generating}
      title={generating ? 'A generation batch is running — wait for it to finish before changing engines' : undefined}
      className={`relative flex items-start gap-3 rounded-xl border p-3 text-left transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${checked
        ? accent.card
        : 'border-border bg-app/40 hover:enabled:bg-surface-raised'}`}>
      <span aria-hidden="true"
        className={`absolute top-2 right-2 w-4 h-4 rounded border grid place-items-center text-[0.625rem] font-bold ${checked
          ? `${accent.pill} border-transparent` : 'border-border text-transparent'}`}>✓</span>
      {icon}
      <span className="flex flex-col gap-1 min-w-0">
        <span className={`text-[0.8125rem] font-semibold ${checked ? accent.title : 'text-content-muted'}`}>
          {title}
        </span>
        <span className="flex flex-wrap gap-1">{tags}</span>
        {hint}
      </span>
    </button>
  );
}

export default function VariationCatalog({ datasetId = null, onGenerate, busy, generating = null, hasRef, composition, images = [], bodyFidelity = false, promptSuffix = '', promptSuffixes = null, onSaveSuffixes = null, subjectType = 'human', onSaveSubjectType = null }) {
  const toast = useToast();
  const { caps } = useCapabilities();
  const [catalog, setCatalog] = useState([]);
  const [nsfwCatalog, setNsfwCatalog] = useState([]);
  const [presets, setPresets] = useState({});
  // Preset display metadata: the backend sends it for NON-human types (their
  // preset keys aren't known here); human returns none -> the hardcoded
  // PRESET_META is used, so the human panel is unchanged.
  const [presetMeta, setPresetMeta] = useState(null);
  // WHAT the subject is — steers which catalog/presets are fetched and the
  // identity lock the backend applies. Synced from the dataset; a change persists
  // to the dataset and refetches the catalog.
  const [subject, setSubject] = useState(() => normalizeSubjectType(subjectType));
  useEffect(() => { setSubject(normalizeSubjectType(subjectType)); }, [subjectType]);
  const changeSubject = (st) => {
    const n = normalizeSubjectType(st);
    if (n === subject) return;
    setSubject(n);
    if (onSaveSubjectType) onSaveSubjectType(n);
  };
  const frLabel = (fr) => framingLabel(subject, fr);
  const [selected, setSelected] = useState(new Set());
  const [multiplier, setMultiplier] = useState(1);
  const [klein, setKlein] = useState(null);
  // 🔞 NSFW mode — local Klein ONLY (the backend refuses NSFW on API engines).
  // Unlocks the uncensored body catalog + a free-prompt custom variation.
  const [nsfwMode, setNsfwMode] = useState(() => {
    try { return localStorage.getItem('datasetNsfwMode') === '1'; } catch { return false; }
  });
  useEffect(() => {
    try { localStorage.setItem('datasetNsfwMode', nsfwMode ? '1' : '0'); } catch { /* ignore */ }
  }, [nsfwMode]);
  const [customPrompt, setCustomPrompt] = useState('');
  const [customFraming, setCustomFraming] = useState('body');
  // User-authored shot cards ("Add" under the free prompt): they live in their
  // own Custom group after BACK, are selectable like catalog cards and are the
  // only DELETABLE ones (catalog cards stay fixed). Persisted across sessions.
  const [customShots, setCustomShots] = useState(() => {
    try { return JSON.parse(localStorage.getItem('datasetCustomShots') || '[]'); }
    catch { return []; }
  });
  useEffect(() => {
    try { localStorage.setItem('datasetCustomShots', JSON.stringify(customShots)); }
    catch { /* ignore */ }
  }, [customShots]);
  const [customPresets, setCustomPresets] = useState(() => loadShotPresets());
  useEffect(() => {
    try { persistShotPresets(localStorage, customPresets); }
    catch { /* private browsing / full storage: keep the in-memory preset usable */ }
  }, [customPresets]);

  const addCustomShot = () => {
    const p = customPrompt.trim();
    if (!p) return;
    const hot = nsfwMode && localOnlyRun;
    const shot = { id: `custom_${Date.now()}`, label: `${hot ? '🔞' : '✨'} ${p.slice(0, 40)}`,
                   prompt: p, framing: customFraming, nsfw: hot };
    setCustomShots((s) => [...s, shot]);
    setSelected((s) => new Set(s).add(shot.id));   // freshly added = selected
    setCustomPrompt('');
  };

  const removeCustomShot = (id) => {
    setCustomShots((s) => s.filter((c) => c.id !== id));
    setSelected((s) => { const n = new Set(s); n.delete(id); return n; });
  };

  // 📥 Imported shots (idea by ashish.sinha — Discord): a JSON catalog the user
  // had an LLM write, rather than typing 40 shots by hand. These live SERVER-side
  // (config `custom_shots`, per subject type), so they survive a browser wipe,
  // show up on a phone as well as the desktop and ride along in the full backup —
  // the ✨ cards above stay in localStorage, unchanged.
  const [importedShots, setImportedShots] = useState([]);
  // Every label the by-label resolvers already answer for (all catalogs + legacy
  // aliases). An imported label that shadows one of these resolves to the WRONG
  // entry on regenerate, so the importer refuses them — see shotImport.js.
  const [reservedLabels, setReservedLabels] = useState([]);
  const [importReview, setImportReview] = useState(null);   // {result, name} — nothing written yet
  const [importBusy, setImportBusy] = useState(false);
  const importFileRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    apiFetch(`/api/dataset/shot-catalog?subject_type=${encodeURIComponent(subject)}`)
      .then((d) => {
        if (cancelled) return;
        setImportedShots(d.shots || []);
        setReservedLabels(d.reserved_labels || []);
      })
      .catch(() => { /* no imported shots is a valid state — never block the panel */ });
    return () => { cancelled = true; };
  }, [subject]);

  /** Persist the WHOLE list for this subject (a removal is just a shorter list).
   *  The server re-validates and answers with what actually landed. */
  const persistImported = async (shots) => {
    const d = await putJson('/api/dataset/shot-catalog', { subject_type: subject, shots });
    setImportedShots(d.shots || []);
    return d;
  };

  const readImportFile = async (file) => {
    if (!file) return;
    setImportReview(null);
    let text = '';
    try { text = await file.text(); }
    catch { toast.error('Could not read that file.'); return; }
    const result = parseShotImport(text, {
      subjectType: subject,
      reservedLabels,
      // A new label may not collide with the built-ins NOR with a shot the user
      // already has — imported or hand-written.
      existingLabels: [...importedShots, ...customShots].map((s) => s.label),
      byteLength: file.size,
    });
    if (result.blocked) { toast.error(result.blocked.message); return; }
    setImportReview({ result, name: file.name });
  };

  /** Second stage: the user has SEEN the summary and says go. Until this runs,
   *  nothing has been written — a 40-shot file with a bad 37th entry can never
   *  leave 36 shots half-imported. */
  const confirmImport = async () => {
    const { result } = importReview || {};
    if (!result?.accepted.length) return;
    setImportBusy(true);
    try {
      const d = await persistImported(applyShotImport(importedShots, result.accepted));
      setImportReview(null);
      toast.success(d.dropped
        ? `${result.accepted.length - d.dropped} shots imported (${d.dropped} refused by the server)`
        : `${result.accepted.length} shot${result.accepted.length === 1 ? '' : 's'} imported`);
    } catch {
      toast.error('Could not save the imported shots.');
    } finally { setImportBusy(false); }
  };

  const removeImportedShot = async (shot) => {
    setSelected((s) => { const n = new Set(s); n.delete(shot.id); return n; });
    try { await persistImported(importedShots.filter((s) => s.id !== shot.id)); }
    catch { toast.error('Could not remove that shot.'); }
  };

  /** ⇪ Keep — promote a hand-written ✨ card into the durable catalog. Those cards
   *  live in localStorage and die with the browser cache; exporting and
   *  re-importing them can't rescue them (they collide with themselves), so this
   *  is the only path — one click, and the card visibly moves to the 📥 group.
   *  Saved first, removed from localStorage only once the server confirms it
   *  landed: a failure must never make the card disappear. */
  const keepCustomShot = async (shot) => {
    const res = promoteCustomShot({ shot, customShots, importedShots, reservedLabels });
    if (!res.ok) { toast.error(res.message); return; }
    try {
      const d = await persistImported(res.importedShots);
      if (!(d.shots || []).some((s) => s.id === res.promoted.id)) {
        toast.error(`“${shot.label}” was refused by the app — the card is still here.`);
        return;
      }
      setCustomShots(res.customShots);
      toast.success(`Kept “${shot.label}” — it now lives with the app, not in this browser`);
    } catch {
      toast.error('Could not keep that shot — the card is still here.');
    }
  };

  const removeAllImported = async () => {
    if (!window.confirm(`Remove all ${importedShots.length} imported shots for this subject type? The built-in shots are not affected.`)) return;
    const ids = new Set(importedShots.map((s) => s.id));
    setSelected((s) => new Set([...s].filter((id) => !ids.has(id))));
    try { await persistImported([]); }
    catch { toast.error('Could not clear the imported shots.'); }
  };

  /** The file the user hands to an LLM — and the backup of their own shots. */
  const exportShotCatalog = () => {
    const blob = new Blob([buildShotExport({
      subjectType: subject, shots: [...importedShots, ...customShots], catalog,
    })], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `lds-shots-${subject}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Shots the user owns, in ONE list: everything below (selection, the 🔞 purge,
  // the Generate payload, the preset composition bars) treats them alike.
  const userShots = useMemo(() => [...customShots, ...importedShots],
    [customShots, importedShots]);

  /** One user-shot card — selectable like a catalog card, plus the ✕ that only
   *  user shots have. Shared by the ✨ Custom and 📥 Imported groups so they can
   *  never drift apart. */
  const renderUserShot = (c, onRemove, removeTitle, onKeep = null) => {
    const on = selected.has(c.id);
    const done = doneByLabel.get(c.label) || 0;
    const blocked = c.nsfw && !localOnlyRun;   // 🔞 card while an API engine is in the run
    const cls = on
      ? 'bg-primary/20 border-primary/50 text-white ring-1 ring-primary/30'
      : done > 0
        ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-100/90 hover:bg-emerald-500/15'
        : 'border-border bg-app/40 text-content-muted hover:bg-surface-raised';
    return (
      <div key={c.id} className={`relative flex items-center gap-1.5 px-1.5 py-1 rounded-lg text-[0.625rem] border transition-colors ${cls} ${blocked ? 'opacity-40' : ''}`}>
        <button type="button" onClick={() => !blocked && toggle(c.id)} aria-pressed={on}
          disabled={blocked}
          title={blocked ? '🔞 shot — check Klein alone to generate it' : c.prompt}
          className="flex items-center gap-1.5 flex-1 min-w-0 text-left disabled:cursor-not-allowed">
          <ShotIllustration framing={c.framing} label={c.label} className="w-7 h-7 shrink-0" />
          {/* Wraps like a catalog card instead of truncating: an imported label is
              a real name the user chose, and "Shiba, zo…" identifies nothing. */}
          <span className="min-w-0 leading-tight break-words">{c.label}</span>
          <span className="ml-auto shrink-0 flex items-center gap-1">
            {done > 0 && <span className="text-emerald-300 font-semibold">✓×{done}</span>}
            {on && <span className="text-indigo-300" aria-hidden="true">✓</span>}
          </span>
        </button>
        <span className="shrink-0 flex flex-col items-stretch gap-0.5">
          {/* "Keep" in words, not a glyph: the point of this button is that the
              card stops living in the browser, and no icon says that. */}
          {onKeep && (
            <button type="button" onClick={onKeep}
              aria-label={`Keep the shot ${c.label}`}
              title="Keep this shot for good — it moves to Imported and is saved with the app, so it survives clearing your browser and shows up on your other devices"
              className="px-1 py-px rounded bg-black/40 text-content-subtle hover:text-emerald-300 text-[0.5625rem] leading-none">
              Keep
            </button>
          )}
          <button type="button" onClick={onRemove}
            aria-label={`${removeTitle} ${c.label}`} title={removeTitle}
            className="self-end w-4 h-4 grid place-items-center rounded bg-black/40 text-content-subtle hover:text-white text-[0.625rem] leading-none">
            ✕
          </button>
        </span>
      </div>
    );
  };
  // Identity LoRA strength (F1): higher = closer to the reference face,
  // lower = more variety in the generated variations.
  // dx8152 consistency LoRA: anchors STRUCTURE, its guide recommends ~0.5 and
  // warns 0.8-1.0 can stop edits from applying (0.9 made variations near-copies).
  const [loraStrength, setLoraStrength] = useState(0.5);
  // Optional generation-LoRA PRESETS (Idea by @waltm — Discord feature
  // request): the user's named combinations from Settings
  // (klein.generation_lora_presets). Per run the workspace just PICKS one —
  // "None" by default on every mount (deliberately not persisted, so no one
  // inherits yesterday's stack); the preset's config is authoritative (no
  // per-LoRA knobs here) and its chain applies to the whole run.
  const [loraPresets, setLoraPresets] = useState([]);   // [{name, loras:[{file, strength}]}]
  const [loraPresetName, setLoraPresetName] = useState('');   // '' = None
  const activeLoraPreset = loraPresets.find((p) => p.name === loraPresetName) || null;
  // Generator backends — a SET, not one card: Nano Banana Pro (Gemini API,
  // ~$0.15/image, zero GPU, best face fidelity — the historic default), ChatGPT
  // (API or subscription) and local Klein (GPU, free). Several at once either
  // SPLIT the shots between them (varied dataset, same cost) or run ALL of them
  // on every shot (compare the engines, then triage). engineSelection.js owns
  // the storage compatibility: the legacy single-string `datasetGenerator` key
  // is still written for older profiles and the ✎ modal. Tile Retry uses the
  // stored engine of its own row instead.
  const [engines, setEngines] = useState(() => readEngines(storage()));
  useEffect(() => { writeEngines(storage(), engines); }, [engines]);
  const [engineMode, setEngineMode] = useState(() => readMode(storage()));
  useEffect(() => { writeMode(storage(), engineMode); }, [engineMode]);
  const toggleEngine = (id) => setEngines((list) => (list.includes(id)
    ? list.filter((e) => e !== id) : canonicalEngines([...list, id])));

  const isNB = engines.includes('nanobanana');
  const isGPT = engines.includes('chatgpt');
  const isOR = engines.includes('openrouter');
  // Per-engine affordances (the Klein tuning panel, the Krea one) light up as
  // soon as that engine is part of the run — its shots really are rendered
  // locally.
  const isKlein = engines.includes('klein');
  const isKrea = engines.includes('krea');
  const multiEngine = engines.length > 1;
  // 🔞 shots stay exactly as strict as before, only the wording of "local"
  // widened: they exist when EVERY selected engine is local. A local engine
  // alongside an API one would either send them to that API (which the backend
  // refuses, failing the whole run) or need a per-shot routing rule the
  // cost/count display could not honestly show — so the uncensored catalog
  // stays locked until the run is local-only. Two local engines together are
  // fine: both accept 🔞.
  const localOnlyRun = localOnly(engines);

  /** Images THIS engine renders in the current run. For an engine that isn't
   *  checked, what it would render if it were picked alone — so the card's price
   *  answers "what would this cost me?" before the click, as it always did.
   *  Reuses the batch split so the card, the mode selector and the payload can
   *  never disagree on the share. */
  const engineShare = (id) => {
    const shots = Array.from({ length: selected.size }, (_, i) => i);
    if (!engines.includes(id)) return shots.length * multiplier;
    const batch = engineBatches(shots, engines, engineMode).find((b) => b.generator === id);
    return (batch ? batch.variations.length : 0) * multiplier;
  };

  // Which engines the user actually enabled in Settings (config.engines.enabled),
  // on top of the live reachability probe in `caps.engines`.
  const [enabledEngines, setEnabledEngines] = useState(['nanobanana', 'chatgpt', 'klein']);
  // ChatGPT auth lane (auto|api|subscription) — decides whether the card shows a
  // per-image API price or "uses your ChatGPT subscription quota".
  const [chatgptAuth, setChatgptAuth] = useState('auto');
  // Which OpenRouter model the run will actually bill: free text in Settings, so
  // the card names it rather than implying a fixed one.
  const [openrouterModel, setOpenrouterModel] = useState('');
  // Same for the ChatGPT API lane, whose model is free text in Settings too:
  // this card used to state "gpt-image-2" flatly, which became a lie the moment
  // someone changed it. Blank = the engine's own default, named here.
  const [chatgptImageModel, setChatgptImageModel] = useState('');
  // Krea's consistency <-> prompt-adherence dial, mirrored from Settings.
  const [kreaGrounding, setKreaGrounding] = useState(512);
  // Krea's own always-on presets (krea.generation_lora_presets) — a SEPARATE
  // pick from Klein's, because one run can dispatch to both engines and each
  // graph takes its own family of LoRAs. "None" on every visit, like Klein's.
  const [kreaLoraPresets, setKreaLoraPresets] = useState([]);
  const [kreaLoraPresetName, setKreaLoraPresetName] = useState('');
  const activeKreaLoraPreset = kreaLoraPresets.find((p) => p.name === kreaLoraPresetName) || null;
  useEffect(() => {
    let cancelled = false;
    apiFetch('/api/settings')
      .then((d) => {
        if (cancelled) return;
        setEnabledEngines(d.config?.engines?.enabled || []);
        setChatgptAuth(d.config?.engines?.chatgpt_auth || 'auto');
        setOpenrouterModel((d.config?.engines?.openrouter_model || '').trim());
        setChatgptImageModel((d.config?.engines?.chatgpt_image_model || '').trim());
        // Optional generation-LoRA presets: names + chains for the picker.
        setLoraPresets(sanitizeGenerationLoraPresets(d.config?.klein?.generation_lora_presets));
        setKreaLoraPresets(sanitizeKreaGenerationLoraPresets(d.config?.krea?.generation_lora_presets));
        // Krea's one dial. It lives in Settings (it changes the meaning of every
        // shot in the batch identically, so it is not a per-run argument), and is
        // MIRRORED here so the workspace can say what the run will actually do.
        setKreaGrounding(Number(d.config?.krea?.grounding_px) || 512);
      })
      .catch(() => { /* keep the permissive default on a transient failure */ });
    return () => { cancelled = true; };
  }, []);
  const nbAvailable = enabledEngines.includes('nanobanana') && caps.engines.nanobanana;
  const gptAvailable = enabledEngines.includes('chatgpt') && caps.engines.chatgpt;
  const orAvailable = enabledEngines.includes('openrouter') && caps.engines.openrouter;
  const klAvailable = enabledEngines.includes('klein') && caps.engines.klein;
  const krAvailable = enabledEngines.includes('krea') && caps.engines.krea;
  const available = { klein: klAvailable, krea: krAvailable, nanobanana: nbAvailable,
    chatgpt: gptAvailable, openrouter: orAvailable };

  // The persisted selection can name engines that have since been disabled in
  // Settings (or lost their key/backend): drop those instead of trying to
  // generate on a dead one, and fall back to the first usable card when that
  // empties the selection. This governs future batches only; tile Retry follows
  // its recorded engine.
  useEffect(() => {
    const usable = engines.filter((e) => available[e]);
    if (usable.length === engines.length) return;
    const first = nbAvailable ? 'nanobanana' : gptAvailable ? 'chatgpt'
      : orAvailable ? 'openrouter' : klAvailable ? 'klein'
      : krAvailable ? 'krea' : null;
    setEngines(usable.length ? usable : (first ? [first] : []));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [engines, nbAvailable, gptAvailable, orAvailable, klAvailable, krAvailable]);
  // Effective ChatGPT lane: the subscription (ChatGPT Plus/Pro image quota) vs the
  // pay-per-use API key. Mirrors the backend "auto = subscription when connected".
  const gptSub = caps.chatgpt_subscription || {};
  const gptViaSub = chatgptAuth === 'subscription'
    || (chatgptAuth === 'auto' && !!gptSub.connected);
  const gptPlanLabel = gptSub.plan && gptSub.plan !== 'free'
    ? gptSub.plan.charAt(0).toUpperCase() + gptSub.plan.slice(1)
    : 'Plus/Pro';
  // What this run costs and, when it can't run, why. `caps.max_fanout` is the
  // SERVER's per-batch cap, published by /api/capabilities — mirrored so the
  // limit is explained before the click, never hardcoded here. A server that
  // doesn't publish it (older build) simply keeps the check off.
  const runCost = estimateCost(selected.size, engines, engineMode, { multiplier, gptViaSub });
  const blockedReason = generateBlockedReason({
    engines, shotCount: selected.size, mode: engineMode, multiplier,
    maxFanout: Number(caps.max_fanout) || 0,
  });
  // Klein unavailable has FOUR distinct causes and the hint must name the right
  // one — a reachable ComfyUI with no Klein model used to show "Configure
  // ComfyUI", sending the user to re-check a step that was already green; and the
  // `beta57` scheduler (a RES4LYF node-pack value the shipped graph pinned,
  // reported by IndependentProcess0 on Reddit) took Klein out for everyone while
  // every other check went green. All four now live in
  // utils/localEngineReason.js, WITH their tests: the ✦ Edit-reference modal
  // offers Klein too, and one gap explained two different ways in two dialogs is
  // the same bug as no explanation at all.
  const kleinHint = klAvailable ? null
    : kleinUnavailableReason({
      enabledInSettings: enabledEngines.includes('klein'),
      comfyuiReachable: !!caps.comfyui?.reachable,
      comfyui: caps.comfyui,
      missingAssets: caps.comfyui?.klein_missing,
      unsupportedEnums: caps.comfyui?.klein_unsupported_enums,
    });
  // Krea has one more failure mode than Klein — a missing CUSTOM-NODE PACK — and
  // "install a node pack" is a different action from "place a weight file", so
  // the reason is computed (and unit-tested) rather than collapsed into one
  // "not available". See utils/kreaEngine.js.
  const kreaHint = krAvailable ? null : kreaUnavailableReason({
    enabledInSettings: enabledEngines.includes('krea'),
    comfyuiReachable: !!caps.comfyui?.reachable,
    comfyui: caps.comfyui,
    missingAssets: caps.comfyui?.krea_missing,
    missingNodes: caps.comfyui?.krea_nodes_missing,
    invalidAssets: caps.comfyui?.krea_invalid,
    nodePackInstalled: caps.comfyui?.krea_nodes_installed,
  });

  useEffect(() => {
    let cancelled = false;
    fetch(`/api/dataset/variations?subject_type=${encodeURIComponent(subject)}`,
      { credentials: 'include' })
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then((d) => {
        if (cancelled) return;
        const p = d.presets || {};
        setCatalog(d.catalog || []);
        setNsfwCatalog(d.nsfw_catalog || []);
        setPresets(p);
        setPresetMeta(d.preset_meta || null);
        // Auto-select the type's default preset (human: balanced / body-emphasis;
        // non-human: its single balanced spread). Re-runs on a subject switch.
        const key = defaultPresetKey(p, subject, { bodyFidelity });
        setSelected(new Set((key && p[key]) || []));
      })
      .catch(() => {
        // Loud failure (M6): an empty catalog otherwise looks like a UI bug.
        if (!cancelled) toast.error('Could not load the variation catalog');
      });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [toast, subject]);

  const byFraming = useMemo(() => {
    const g = { face: [], bust: [], body: [], back: [] };
    catalog.forEach((e) => g[e.framing]?.push(e));
    return g;
  }, [catalog]);

  // Switching to an API engine drops any selected NSFW shots (Klein-only) —
  // catalog nsfw_ entries AND 🔞 custom cards alike.
  useEffect(() => {
    if (localOnlyRun) return;
    const hotCustom = new Set(userShots.filter((c) => c.nsfw).map((c) => c.id));
    setSelected((s) => {
      const n = new Set([...s].filter((id) => !id.startsWith('nsfw_') && !hotCustom.has(id)));
      return n.size === s.size ? s : n;
    });
  }, [localOnlyRun, userShots]);

  // "Already in the dataset" per variation label: live images (kept, pending or
  // still generating — not failed/rejected) → the green ✓×N state on the cards.
  const doneByLabel = useMemo(() => {
    const m = new Map();
    for (const img of images) {
      if (!img.variation_label || img.status === 'failed' || img.status === 'reject') continue;
      m.set(img.variation_label, (m.get(img.variation_label) || 0) + 1);
    }
    return m;
  }, [images]);

  // Framing mix of each preset — feeds the mini composition bar on its card.
  const presetStats = useMemo(() => {
    const framingById = new Map(catalog.map((e) => [e.id, e.framing]));
    const stats = {};
    Object.entries(presets).forEach(([key, ids]) => {
      const counts = { face: 0, bust: 0, body: 0, back: 0 };
      (ids || []).forEach((id) => { const fr = framingById.get(id); if (fr) counts[fr] += 1; });
      stats[key] = { counts, total: (ids || []).length };
    });
    return stats;
  }, [catalog, presets]);

  // Which preset (if any) matches the current selection exactly → highlighted card.
  const activePreset = useMemo(() => {
    const entry = Object.entries(presets).find(([, ids]) =>
      ids && ids.length === selected.size && ids.every((id) => selected.has(id)));
    return entry ? entry[0] : null;
  }, [presets, selected]);

  const activeCustomPreset = useMemo(() => customPresets.find((preset) =>
    preset.selectedIds.length === selected.size
      && preset.selectedIds.every((id) => selected.has(id)))?.id || null,
  [customPresets, selected]);

  const customPresetStats = useMemo(() => {
    const framingById = new Map([
      ...catalog, ...nsfwCatalog, ...userShots,
      ...customPresets.flatMap((preset) => preset.customShots || []),
    ].map((shot) => [shot.id, shot.framing]));
    return Object.fromEntries(customPresets.map((preset) => {
      const counts = { face: 0, bust: 0, body: 0, back: 0 };
      preset.selectedIds.forEach((id) => { const fr = framingById.get(id); if (fr) counts[fr] += 1; });
      return [preset.id, { counts, total: preset.selectedIds.length }];
    }));
  }, [catalog, nsfwCatalog, customShots, customPresets]);

  const toggle = (id) => setSelected((s) => {
    const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n;
  });

  // Never wipe the current selection when the preset is unavailable (M6).
  // Toggle: re-clicking the ACTIVE preset (exact selection match) clears the
  // whole selection instead of re-applying it.
  const applyPreset = (key) => {
    const ids = presets[key];
    if (!ids?.length) return;
    setSelected(activePreset === key ? new Set() : new Set(ids));
  };

  const saveCurrentPreset = () => {
    const name = window.prompt('Name this shot preset:');
    if (name == null) return;
    try {
      const next = saveShotPreset(customPresets, name, selected, userShots);
      setCustomPresets(next);
      toast.success(`Preset saved: ${next.at(-1).name}`);
    } catch (error) { toast.error(error.message || 'Could not save preset'); }
  };

  const applyCustomPreset = (preset) => {
    if (activeCustomPreset === preset.id) {
      setSelected(new Set());
      return;
    }
    // A saved preset carries a COPY of the user shots it selected. Restore the
    // missing ones as ✨ cards, minus those the subject already has imported —
    // otherwise an imported shot would be duplicated into localStorage.
    const restored = applyShotPreset(preset, userShots);
    const importedIds = new Set(importedShots.map((s) => s.id));
    setCustomShots(restored.customShots.filter((s) => !importedIds.has(s.id)));
    setSelected(new Set(restored.selectedIds));
  };

  const renameCustomPreset = (preset) => {
    const name = window.prompt('Rename shot preset:', preset.name);
    if (name == null) return;
    try { setCustomPresets((items) => renameShotPreset(items, preset.id, name)); }
    catch (error) { toast.error(error.message || 'Could not rename preset'); }
  };

  const removeCustomPreset = (preset) => {
    if (!window.confirm(`Delete the preset “${preset.name}”?`)) return;
    setCustomPresets((items) => deleteShotPreset(items, preset.id));
  };

  // ✨ Prompt suffixes (Idea by waltm — Discord): the dataset's creative-
  // direction text (one global + one per framing), surfaced here so it can be
  // tuned PER BATCH without opening the Settings modal. These are the SAME
  // dataset fields the modal edits (one shared truth) — pre-filled from the
  // dataset and, on Generate, saved back to it. The backend then applies the
  // dataset's current suffix at wrap time for every engine.
  const baseSuffixes = promptSuffixes || {};
  const [gSuffix, setGSuffix] = useState(promptSuffix || '');
  const [fSuffix, setFSuffix] = useState(
    () => Object.fromEntries(SUFFIX_KEYS.map((k) => [k, baseSuffixes[k] || ''])));
  const [suffixOpen, setSuffixOpen] = useState(() => Boolean(
    (promptSuffix || '').trim() || SUFFIX_KEYS.some((k) => (baseSuffixes[k] || '').trim())));
  // Adopt an EXTERNAL change to the shared truth (the Settings modal saved new
  // suffixes) without clobbering an in-progress edit on an unrelated refresh:
  // re-sync the fields only when the dataset's stored value actually changed.
  const baselineKey = useMemo(
    () => JSON.stringify([(promptSuffix || '').trim(),
      SUFFIX_KEYS.map((k) => (baseSuffixes[k] || '').trim())]),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [promptSuffix, promptSuffixes]);
  const appliedSuffixKey = useRef(baselineKey);
  useEffect(() => {
    if (appliedSuffixKey.current === baselineKey) return;
    appliedSuffixKey.current = baselineKey;
    setGSuffix(promptSuffix || '');
    setFSuffix(Object.fromEntries(SUFFIX_KEYS.map((k) => [k, baseSuffixes[k] || ''])));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baselineKey]);
  const suffixDirty = (gSuffix || '').trim() !== (promptSuffix || '').trim()
    || SUFFIX_KEYS.some((k) => (fSuffix[k] || '').trim() !== (baseSuffixes[k] || '').trim());
  const suffixPayload = () => ({
    prompt_suffix: (gSuffix || '').trim(),
    prompt_suffixes: Object.fromEntries(
      SUFFIX_KEYS.map((k) => [k, (fSuffix[k] || '').trim()]).filter(([, v]) => v)),
  });

  const go = async () => {
    const variations = catalog.filter((e) => selected.has(e.id))
      .map((e) => ({ label: e.label, prompt: e.prompt, framing: e.framing }));
    // NSFW shots: local Klein only (the toggle is gated on the Klein engine,
    // and the backend refuses them on API engines).
    if (nsfwMode && localOnlyRun) {
      variations.push(...nsfwCatalog.filter((e) => selected.has(e.id))
        .map((e) => ({ label: e.label, prompt: e.prompt, framing: e.framing, nsfw: true })));
    }
    // Custom cards: selectable like catalog shots; 🔞 ones only ride with Klein
    // (the label prefix is what regenerate uses to re-pick the uncensored wrapper).
    variations.push(...userShots
      .filter((c) => selected.has(c.id) && (localOnlyRun || !c.nsfw))
      .map((c) => ({ label: c.label, prompt: c.prompt, framing: c.framing,
                     ...(c.nsfw ? { nsfw: true } : {}) })));
    if (!variations.length) return;
    // Guard-rail: the selection survives a previous Generate, so a re-click would
    // re-generate (and re-bill) shots that already exist. Ask — OK = duplicates
    // on purpose, Cancel = only the newly added shots.
    const dupes = variations.filter((v) => doneByLabel.get(v.label));
    let toGen = variations;
    if (dupes.length === variations.length) {
      if (!window.confirm(
        `All ${dupes.length} selected shot(s) already exist in the dataset (green ✓×N cards).\n\n`
        + 'Generate them AGAIN anyway (duplicates)?')) return;
    } else if (dupes.length > 0) {
      const fresh = variations.length - dupes.length;
      if (!window.confirm(
        `${dupes.length} of the ${variations.length} selected shot(s) already exist in the dataset.\n\n`
        + `OK — generate everything (including ${dupes.length} duplicate(s))\n`
        + `Cancel — only generate the ${fresh} new one(s)`)) {
        toGen = variations.filter((v) => !doneByLabel.get(v.label));
      }
    }
    if (!toGen.length) return;
    // Guard-rail: pay-per-use API engines bill per image — above $5 estimated,
    // confirm with the amount (silent for the free local Klein AND for the
    // ChatGPT subscription lane, which spends plan quota, not dollars). On a
    // multi-engine run the amount is the WHOLE run's, and only the lanes that
    // really charge are named.
    const cost = estimateCost(toGen.length, engines, engineMode, { multiplier, gptViaSub });
    const billing = billingEngines(engines, { gptViaSub });
    if (cost > 5 && !window.confirm(
      `This will launch ${totalImages(toGen.length, engines, engineMode, multiplier)} generation(s) `
      + `≈ $${cost.toFixed(2)} (${billing.map((e) => ENGINE_LABELS[e]).join(' + ')}).\n\nProceed?`)) return;
    // Persist any per-batch suffix edit BEFORE enqueueing: the backend applies
    // the dataset's CURRENT suffix at wrap time, so the save must land first or
    // the batch would generate with the old creative direction (Idea by waltm).
    if (suffixDirty && onSaveSuffixes) {
      const res = await onSaveSuffixes(suffixPayload());
      if (!res?.ok) return;   // save failed → don't generate with a stale suffix
    }
    // Optional generation-LoRA preset (Klein only): only the NAME rides — the
    // backend resolves the chain from its own config (fail-closed).
    // The shots are already shared between the engines here (API batches first,
    // the GPU-bound Klein one last); the server re-checks every batch.
    onGenerate(engineBatches(toGen, engines, engineMode), multiplier, klein, loraStrength,
      { ...generationLoraPresetPayload({ isKlein, presetName: loraPresetName, presets: loraPresets }),
        ...kreaGenerationLoraPresetPayload({ isKrea, presetName: kreaLoraPresetName, presets: kreaLoraPresets }) });
  };

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-border bg-surface p-3">
      <div className="flex items-center gap-2">
        <span aria-hidden="true">🎬</span>
        <h2 className="text-content font-semibold text-sm">Generate variations</h2>
        <span className="text-content-subtle text-[0.6875rem]">
          pick the shots to synthesize from the reference photo
        </span>
      </div>

      {/* Subject type — WHAT the reference is. Anything but Human switches the
          shot catalog AND the identity lock so the prompts stop assuming a person
          (a dog keeps its breed/markings, a product its shape/logo). Persisted per
          dataset; changing it reloads the shot list and its default preset. */}
      <div className="flex flex-col gap-1 rounded-lg border border-border bg-app/30 px-2.5 py-2">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-content-muted text-[0.6875rem] uppercase">Subject type</span>
          <div role="radiogroup" aria-label="Subject type" className="flex flex-wrap gap-1">
            {SUBJECT_TYPES.map((st) => (
              <button key={st} type="button" role="radio" aria-checked={subject === st}
                onClick={() => changeSubject(st)} disabled={!!generating}
                title={SUBJECT_TYPE_HINTS[st]}
                className={`px-2 py-0.5 rounded-full text-[0.6875rem] border transition-colors disabled:opacity-50 ${subject === st
                  ? 'border-primary/60 bg-primary/15 text-white ring-1 ring-primary/30'
                  : 'border-border bg-app/40 text-content-muted hover:bg-surface-raised'}`}>
                {SUBJECT_TYPE_LABELS[st]}
              </button>
            ))}
          </div>
          <HelpBadge topic="subject-type" />
        </div>
        <span className="text-content-subtle text-[0.625rem]">{SUBJECT_TYPE_HINTS[subject]}</span>
      </div>

      {/* Engine cards — Klein and Krea 2 Edit (local GPU), Nano Banana Pro,
          ChatGPT and OpenRouter (APIs).
          CHECKBOXES, not a radio group: several engines can run in one batch.
          Each card disables itself with an actionable hint when its engine
          isn't configured/reachable or was turned off in Settings, and carries
          its own accent colour so a mixed run stays readable at a glance. */}
      <div className="flex items-center gap-2">
        <span className="text-content-muted text-[0.6875rem] uppercase">Engines</span>
        <span className="text-content-subtle text-[0.625rem]">
          where the images are made — pick one or several · Klein and Krea 2 Edit run free on your GPU · APIs bill per image (or use your ChatGPT subscription)
        </span>
        <HelpBadge topic="dataset-engine-mode" />
      </div>
      {/* Discoverability: the generation prompt (identity/style directives) is
          editable, but users don't know where. Point them at it right where the
          "why is this coming out realistic?" question arises. */}
      <p className="text-content-subtle text-[0.625rem] -mt-1">
        Not the look you wanted (a stylized reference coming out realistic)? Edit the generation prompt in{' '}
        <a href="#/settings/engines" className="text-amber-300 underline decoration-amber-300/50">Settings › Image engines →</a>
      </p>
      {/* Two facts about the Gemini engine that belong next to the choice, not in
          a support thread after the fact. Both are stated flat, with no verdict
          attached: the filter has no setting to offer, and nobody — in either
          direction — has measured what SynthID does to trained weights, so this
          says it is present and stops there. Plain <p>: wraps freely at 400 px. */}
      <p className="text-content-subtle text-[0.625rem] -mt-1">
        Building with <span className="text-content-muted">Nano Banana</span>? Google
        screens every image it returns and refuses some of them — LDS names each
        refusal on the tile; the filter itself is not configurable. Its outputs also
        carry SynthID, Google&apos;s invisible provenance watermark.{' '}
        <HelpBadge topic="nanobanana-filter-and-synthid" />
      </p>
      {/* Five cards now, and the column stops at THREE. Tailwind breakpoints read
          the VIEWPORT, but these cards live in the workspace column next to the
          sidebar — a `2xl:grid-cols-5` measured on a 1600 px window put five cards
          in ~700 px (≈130 px each) and wrapped every hint to one word per line.
          Three-across is wider per card than the four-across this grid shipped
          with. One column on a phone (nothing is clipped at 400 px), two from sm. */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-2">
        <EngineCard id="klein" checked={isKlein} available={klAvailable} generating={generating}
          onToggle={toggleEngine} share={engineShare('klein')}
          icon={<GpuIcon className={`w-9 h-9 shrink-0 ${isKlein ? ENGINE_ACCENTS.klein.icon : 'text-content-subtle'}`} />}
          title={<>Klein <span className="font-normal text-content-subtle">· local</span></>}
          tags={[
            // Green stays a statement about the PRICE, never a selection state.
            <span key="free" className="px-1.5 py-px rounded-full bg-emerald-500/15 border border-emerald-400/40 text-emerald-300 text-[0.625rem]">Free</span>,
            <span key="gpu" className="px-1.5 py-px rounded-full bg-app/60 border border-border text-content-muted text-[0.625rem]">Your GPU</span>,
            <span key="nsfw" className="px-1.5 py-px rounded-full bg-app/60 border border-border text-content-muted text-[0.625rem]">NSFW OK</span>,
          ]}
          hint={klAvailable ? (
            <span className="text-content-subtle text-[0.625rem]">
              Runs on this machine — slower, tunable face fidelity.
              {localQueuesBehindApi(engines) && (
                <> <span className={ENGINE_ACCENTS.klein.text}>
                  Its {engineShare('klein')} shot(s) queue on your GPU, one at a time, after the API ones.
                </span></>
              )}
            </span>
          ) : (
            <a href="#/setup" onClick={(e) => e.stopPropagation()}
              className="text-amber-300 text-[0.625rem] underline decoration-amber-300/50">
              {kleinHint}
            </a>
          )} />
        {/* Krea 2 Identity Edit — the second LOCAL engine. It keeps the identity
            from the reference photo ALONE (no character LoRA needed), which is
            exactly the bootstrap case: a character that has no LoRA yet. */}
        <EngineCard id="krea" checked={isKrea} available={krAvailable} generating={generating}
          onToggle={toggleEngine} share={engineShare('krea')}
          icon={<IdentityFrameIcon className={`w-9 h-9 shrink-0 ${isKrea ? ENGINE_ACCENTS.krea.icon : 'text-content-subtle'}`} />}
          title={<>Krea 2 Edit <span className="font-normal text-content-subtle">· local</span></>}
          tags={[
            <span key="free" className="px-1.5 py-px rounded-full bg-emerald-500/15 border border-emerald-400/40 text-emerald-300 text-[0.625rem]">Free</span>,
            <span key="gpu" className="px-1.5 py-px rounded-full bg-app/60 border border-border text-content-muted text-[0.625rem]">Your GPU</span>,
            <span key="nsfw" className="px-1.5 py-px rounded-full bg-app/60 border border-border text-content-muted text-[0.625rem]">NSFW OK</span>,
          ]}
          hint={krAvailable ? (
            <span className="text-content-subtle text-[0.625rem]">
              Identity-preserving edit — strongest likeness from a single reference photo.
              Krea Fit v1.2 honors the selected shot card&rsquo;s framing and aspect ratio.
              {localQueuesBehindApi(engines) && (
                <> <span className={ENGINE_ACCENTS.krea.text}>
                  Its {engineShare('krea')} shot(s) queue on your GPU, one at a time, after the API ones.
                </span></>
              )}
            </span>
          ) : (
            <a href="#/setup" onClick={(e) => e.stopPropagation()}
              className="text-amber-300 text-[0.625rem] underline decoration-amber-300/50">
              {kreaHint}
            </a>
          )} />
        <EngineCard id="nanobanana" checked={isNB} available={nbAvailable} generating={generating}
          onToggle={toggleEngine} share={engineShare('nanobanana')}
          icon={<span className="w-9 h-9 shrink-0 grid place-items-center text-2xl" aria-hidden="true">🍌</span>}
          title={<>Nano Banana Pro <span className="font-normal text-content-subtle">· API</span></>}
          tags={[
            <span key="gpu" className="px-1.5 py-px rounded-full bg-app/60 border border-border text-content-muted text-[0.625rem]">No GPU</span>,
            <span key="price" className="px-1.5 py-px rounded-full bg-app/60 border border-border text-content-muted text-[0.625rem]">~$0.15/image</span>,
            <span key="sfw" className="px-1.5 py-px rounded-full bg-app/60 border border-border text-content-muted text-[0.625rem]">SFW</span>,
          ]}
          hint={nbAvailable ? (
            <span className={`text-[0.625rem] ${isNB ? ENGINE_ACCENTS.nanobanana.text : 'text-content-subtle'}`}>
              Best face fidelity · {engineShare('nanobanana')} image(s) ≈ ${(engineShare('nanobanana') * 0.15).toFixed(2)}
            </span>
          ) : (
            <span className="text-amber-300 text-[0.625rem]">⚠ Add GEMINI_API_KEY in Settings</span>
          )} />
        <EngineCard id="chatgpt" checked={isGPT} available={gptAvailable} generating={generating}
          onToggle={toggleEngine} share={engineShare('chatgpt')}
          icon={<ChatGptIcon className={`w-9 h-9 shrink-0 ${isGPT ? ENGINE_ACCENTS.chatgpt.icon : 'text-content-subtle'}`} />}
          title={<>ChatGPT <span className="font-normal text-content-subtle">{gptViaSub ? '· subscription' : '· API'}</span></>}
          tags={[
            <span key="gpu" className="px-1.5 py-px rounded-full bg-app/60 border border-border text-content-muted text-[0.625rem]">No GPU</span>,
            <span key="price" className="px-1.5 py-px rounded-full bg-app/60 border border-border text-content-muted text-[0.625rem]">{gptViaSub ? 'Plan quota' : '~$0.17/image'}</span>,
            <span key="sfw" className="px-1.5 py-px rounded-full bg-app/60 border border-border text-content-muted text-[0.625rem]">SFW</span>,
          ]}
          hint={gptAvailable ? (
            <span className={`text-[0.625rem] ${isGPT ? ENGINE_ACCENTS.chatgpt.text : 'text-content-subtle'}`}>
              {/* The subscription lane renders on the plan's own image model and
                  ignores the Settings field, so only the API lane names it. */}
              {gptViaSub
                ? `gpt-image-2 · uses your ChatGPT ${gptPlanLabel} quota`
                : <><span className="break-all">{chatgptImageModel || 'gpt-image-2'}</span>
                    {` · ${engineShare('chatgpt')} image(s) ≈ $${(engineShare('chatgpt') * 0.17).toFixed(2)}`}</>}
            </span>
          ) : (
            <span className="text-amber-300 text-[0.625rem]">⚠ Add an API key or connect a subscription in Settings</span>
          )} />
        <EngineCard id="openrouter" checked={isOR} available={orAvailable} generating={generating}
          onToggle={toggleEngine} share={engineShare('openrouter')}
          icon={<RouterIcon className={`w-9 h-9 shrink-0 ${isOR ? ENGINE_ACCENTS.openrouter.icon : 'text-content-subtle'}`} />}
          title={<>OpenRouter <span className="font-normal text-content-subtle">· API</span></>}
          tags={[
            <span key="gpu" className="px-1.5 py-px rounded-full bg-app/60 border border-border text-content-muted text-[0.625rem]">No GPU</span>,
            <span key="price" className="px-1.5 py-px rounded-full bg-app/60 border border-border text-content-muted text-[0.625rem]">Your credits</span>,
            <span key="sfw" className="px-1.5 py-px rounded-full bg-app/60 border border-border text-content-muted text-[0.625rem]">SFW</span>,
          ]}
          hint={orAvailable ? (
            <span className={`text-[0.625rem] ${isOR ? ENGINE_ACCENTS.openrouter.text : 'text-content-subtle'}`}>
              {/* The model is free text in Settings, so the price here is an
                  estimate for the DEFAULT one — say so rather than quote a
                  number the user may have changed under it. */}
              <span className="break-all">{openrouterModel || 'default model'}</span>
              {' · '}{engineShare('openrouter')} image(s), billed by OpenRouter at that model&rsquo;s rate
            </span>
          ) : (
            <span className="text-amber-300 text-[0.625rem]">⚠ Add OPENROUTER_API_KEY in Settings</span>
          )} />
      </div>

      {/* How several engines share the run. Only shown when it can change
          anything (2+ engines): with a single one both modes are identical, and
          an inert radio pair would just be noise. */}
      {multiEngine && (
        <fieldset className="rounded-lg border border-border bg-app/30 px-2.5 py-2 flex flex-col gap-1.5">
          <legend className="px-1 text-content-muted text-[0.6875rem] uppercase">
            {engines.length} engines selected
          </legend>
          {MODE_CHOICES.map(({ id, name, desc }) => {
            const count = totalImages(selected.size, engines, id, multiplier);
            const price = estimateCost(selected.size, engines, id, { multiplier, gptViaSub });
            return (
              <label key={id} className={`flex items-start gap-2 rounded-md px-2 py-1 cursor-pointer ${engineMode === id ? 'bg-surface-raised' : ''}`}>
                <input type="radio" name="engine-mode" value={id} checked={engineMode === id}
                  onChange={() => setEngineMode(id)} disabled={!!generating}
                  className="mt-0.5 accent-indigo-500" />
                <span className="flex flex-col min-w-0">
                  <span className="text-content text-[0.75rem] font-semibold">
                    {name}
                    <span className="ml-2 font-normal text-content-muted">
                      {count} image{count === 1 ? '' : 's'}
                      {price > 0 ? ` · ≈ $${price.toFixed(2)}` : ' · free'}
                    </span>
                  </span>
                  <span className="text-content-subtle text-[0.625rem]">{desc}</span>
                </span>
              </label>
            );
          })}
        </fieldset>
      )}

      {/* Klein-only tuning, grouped: model file + consistency-LoRA strength.
          A <details> so the defaults stay out of a newcomer's way — children
          remain mounted, so the model picker still reports its choice. */}
      {isKlein && klAvailable && (
        <details className="rounded-lg border border-border bg-app/30 open:pb-2"
          onToggle={(e) => { if (e.currentTarget.open) requestHelpTip('klein-tuning-open'); }}>
          <summary className="cursor-pointer select-none px-2.5 py-1.5 text-[0.75rem] text-content font-semibold">
            🖥️ Klein tuning
            <span className="ml-2 font-normal text-content-subtle text-[0.625rem]">
              model file · consistency LoRA {loraStrength <= 0 ? 'off' : loraStrength.toFixed(2)}
              {activeLoraPreset && activeLoraPreset.loras.length > 0
                ? ` · LoRA preset: ${activeLoraPreset.name}` : ''}
            </span>
          </summary>
          <div className="px-2.5 pt-1 flex flex-col gap-2">
            {/* The SAME dataset setting ✨ Upscale & improve uses: both run the
                same UNETLoader in the same graph, and two near-identical model
                dropdowns side by side is a confusion that never goes away.
                Replaces the old per-BROWSER pick (editPage_flux2KleinModel_v1),
                which described what a dataset contains but travelled with the
                browser — see kleinModelChoice.js for the carry-over. */}
            <div className="max-w-sm">
              <KleinModelSetting datasetId={datasetId} onChange={setKlein} />
            </div>
            <div className="flex flex-col gap-0.5">
              <label className="flex items-center gap-2 text-content-muted text-[0.6875rem]">
                <span className="whitespace-nowrap">
                  Consistency LoRA: {loraStrength <= 0 ? 'off' : loraStrength.toFixed(2)}
                </span>
                <input type="range" min={0} max={1.2} step={0.05} value={loraStrength}
                  onChange={(e) => setLoraStrength(Number(e.target.value))}
                  aria-label="Consistency LoRA strength"
                  className="flex-1 min-w-[120px] accent-indigo-500" />
              </label>
              <p className="text-content-subtle text-[0.625rem]">
                Anchors the COMPOSITION, not the face — high values suppress pose/framing changes.
                ~0.5 balanced · 0.2–0.4 for big restagings · 0 = off. Face identity comes from the
                reference photo(s); add extra references for a stronger identity lock.
              </p>
            </div>
            {/* Optional generation-LoRA preset (Idea by @waltm) — pick one of
                the named combinations from Settings; its chain (read-only
                here) applies to every variation of the run. "None" on each
                visit by default. */}
            <div className="flex flex-col gap-1">
              <label className="flex items-center gap-2 text-content-muted text-[0.6875rem]">
                <span className="whitespace-nowrap">LoRA preset</span>
                <select value={loraPresetName} aria-label="Generation LoRA preset"
                  onChange={(e) => setLoraPresetName(e.target.value)}
                  className="bg-app/60 border border-border rounded px-1 py-0.5 text-content text-[0.6875rem]">
                  <option value="">None</option>
                  {loraPresets.map((p) => (
                    <option key={p.name} value={p.name}>{p.name} ({p.loras.length})</option>
                  ))}
                </select>
                <span className="text-content-subtle text-[0.625rem]">
                  your own LoRA combos — applies to every shot of this run
                </span>
              </label>
              {loraPresets.length === 0 && (
                <p className="text-content-subtle text-[0.625rem]">
                  No presets yet — build combinations of your own LoRA files (texture, anatomy, style…) in{' '}
                  <a href="#/settings/engines" className="text-amber-300 underline decoration-amber-300/50">
                    Settings › Image engines
                  </a>.
                </p>
              )}
              {activeLoraPreset && (
                activeLoraPreset.loras.length === 0 ? (
                  <p className="text-content-subtle text-[0.625rem]">
                    This preset is empty — add LoRA files to it in Settings.
                  </p>
                ) : (
                  <ol className="flex flex-col gap-0.5 text-[0.625rem] text-content-subtle">
                    {activeLoraPreset.loras.map((row, i) => (
                      <li key={`${row.file}-${i}`} className="flex items-center gap-1.5" title={row.file}>
                        <span className="text-content-muted">{i + 1}.</span>
                        <span className="font-mono truncate max-w-[18rem]">{row.file.split(/[\\/]/).pop()}</span>
                        <span>@ {row.strength.toFixed(2)}</span>
                      </li>
                    ))}
                  </ol>
                )
              )}
            </div>
          </div>
        </details>
      )}

      {/* Krea tuning — deliberately a READ-OUT, not a second set of sliders.
          Krea has exactly one dial and it is a SETTING: `grounding_px` changes
          the meaning of every shot in the batch identically, so a per-run copy
          would be a second truth to keep in sync (and a value silently different
          from the one the Settings page shows). What belongs here is knowing
          what the run is about to do, and one click to change it. */}
      {isKrea && krAvailable && (
        <details className="rounded-lg border border-border bg-app/30 open:pb-2">
          <summary className="cursor-pointer select-none px-2.5 py-1.5 text-[0.75rem] text-content font-semibold">
            🧬 Krea 2 Edit tuning
            <span className="ml-2 font-normal text-content-subtle text-[0.625rem]">
              reference grounding {groundingDescription(kreaGrounding)}
              {activeKreaLoraPreset && activeKreaLoraPreset.loras.length > 0
                ? ` · LoRA preset: ${activeKreaLoraPreset.name}` : ''}
            </span>
          </summary>
          <div className="px-2.5 pt-1 flex flex-col gap-1.5">
            <p className="text-content-subtle text-[0.625rem]">
              <b className="text-content-muted font-semibold">Reference grounding</b> is the
              consistency ↔ prompt dial: the low end follows the shot description (more variety
              in pose, outfit and scene, looser likeness), while HIGH resembles the reference
              more closely — and can copy the very pose and outfit you asked it to change.
              512 px is the dataset-restaging balance: it keeps the prompt and selected shot
              card in charge while preserving identity. Raise it deliberately when reference
              likeness matters more.
            </p>
            <p className="text-content-subtle text-[0.625rem]">
              Identity comes from the reference photo alone — no character LoRA needed. Extra
              reference images are not used by this engine. Krea Fit v1.2 honors each selected
              card&rsquo;s framing and aspect ratio instead of forcing the source photo&rsquo;s shape.
            </p>
            <p className="text-content-subtle text-[0.625rem]">
              Change it in{' '}
              <SettingsLink section="engines" focus="krea-grounding" tone="warning" className="text-[0.625rem]">
                Settings › Image engines
              </SettingsLink>{' '}— it applies to every Krea run.
            </p>
            {/* The one live control in this panel. It does not restate a value —
                it NAMES a preset defined in Settings, which is why it belongs
                here while the grounding number stays a read-out. */}
            <div className="flex flex-col gap-1 pt-1 border-t border-white/10">
              <label className="flex items-center gap-2 text-content-muted text-[0.6875rem]">
                <span className="whitespace-nowrap">LoRA preset</span>
                <select value={kreaLoraPresetName} aria-label="Krea generation LoRA preset"
                  onChange={(e) => setKreaLoraPresetName(e.target.value)}
                  className="bg-app/60 border border-border rounded px-1 py-0.5 text-content text-[0.6875rem]">
                  <option value="">None</option>
                  {kreaLoraPresets.map((p) => (
                    <option key={p.name} value={p.name}>{p.name} ({p.loras.length})</option>
                  ))}
                </select>
                <span className="text-content-subtle text-[0.625rem]">
                  your own Krea LoRAs — applies to every shot of this run
                </span>
              </label>
              {kreaLoraPresets.length === 0 && (
                <p className="text-content-subtle text-[0.625rem]">
                  No presets yet — build combinations of your own Krea 2 LoRA files in{' '}
                  <a href="#/settings/engines" className="text-amber-300 underline decoration-amber-300/50">
                    Settings › Image engines
                  </a>.
                </p>
              )}
              {activeKreaLoraPreset && (
                activeKreaLoraPreset.loras.length === 0 ? (
                  <p className="text-content-subtle text-[0.625rem]">
                    This preset is empty — add LoRA files to it in Settings.
                  </p>
                ) : (
                  <ol className="flex flex-col gap-0.5 text-[0.625rem] text-content-subtle">
                    {activeKreaLoraPreset.loras.map((row, i) => (
                      <li key={`${row.file}-${i}`} className="flex items-center gap-1.5" title={row.file}>
                        <span className="text-content-muted">{i + 1}.</span>
                        <span className="font-mono truncate max-w-[18rem]">{row.file.split(/[\\/]/).pop()}</span>
                        <span>@ {row.strength.toFixed(2)}</span>
                      </li>
                    ))}
                  </ol>
                )
              )}
              <p className="text-content-subtle text-[0.625rem]">
                Not applied by the 🔄 single-image regenerate, and only the model side is
                patched (a LoRA's text-encoder weights are ignored here).
              </p>
            </div>
          </div>
        </details>
      )}

      {/* Preset cards with their framing-mix bar. */}
      <div>
        <div className="flex items-center gap-2 mb-1.5 flex-wrap">
          <span className="text-content-muted text-[0.6875rem] uppercase">Presets</span>
          <button type="button" onClick={saveCurrentPreset} disabled={!selected.size}
            aria-label="Save the current shot selection as a custom preset"
            className="rounded-md border border-primary/40 bg-primary/10 px-2 py-1 text-[0.625rem] font-semibold text-indigo-200 hover:bg-primary/20 disabled:opacity-40">
            ＋ Save preset
          </button>
          <span className="ml-auto flex items-center gap-2 flex-wrap text-[0.625rem] text-content-subtle" aria-hidden="true">
            {['face', 'bust', 'body', 'back'].map((fr) => (
              <span key={fr} className="flex items-center gap-1">
                <span className={`w-2 h-2 rounded-full ${FRAMING_COLOR[fr]}`} />{frLabel(fr)}
              </span>
            ))}
          </span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-1.5">
          {(presetMeta && presetMeta.length ? presetMeta : PRESET_META).map(({ key, name, hint }) => {
            const st = presetStats[key];
            const active = activePreset === key;
            return (
              <button key={key} type="button" onClick={() => applyPreset(key)} title={hint}
                aria-pressed={active} disabled={!st?.total}
                className={`flex flex-col gap-1.5 rounded-lg border p-2 text-left transition-colors disabled:opacity-40 ${active
                  ? 'border-primary/60 bg-primary/15 ring-1 ring-primary/40'
                  : 'border-border bg-app/40 hover:bg-surface-raised'}`}>
                <span className="flex items-baseline gap-1 min-w-0">
                  <span className={`text-[0.6875rem] font-semibold truncate ${active ? 'text-white' : 'text-content'}`}>{name}</span>
                  <span className="ml-auto text-content-subtle text-[0.625rem] shrink-0">{st?.total || 0}</span>
                </span>
                <CompositionMiniBar counts={st?.counts || {}} total={st?.total || 0} />
              </button>
            );
          })}
          {customPresets.map((preset) => {
            const active = activeCustomPreset === preset.id;
            const st = customPresetStats[preset.id];
            return (
              <div key={preset.id}
                className={`relative min-w-0 rounded-lg border transition-colors ${active
                  ? 'border-primary/60 bg-primary/15 ring-1 ring-primary/40'
                  : 'border-border bg-app/40 hover:bg-surface-raised'}`}>
                <button type="button" onClick={() => applyCustomPreset(preset)} aria-pressed={active}
                  aria-label={`Apply custom preset ${preset.name}`}
                  className="flex w-full min-w-0 flex-col gap-1.5 p-2 pr-12 text-left">
                  <span className="flex w-full min-w-0 items-baseline gap-1">
                    <span className={`truncate text-[0.6875rem] font-semibold ${active ? 'text-white' : 'text-content'}`}>
                      ✨ {preset.name}
                    </span>
                    <span className="ml-auto shrink-0 text-[0.625rem] text-content-subtle">{st?.total || 0}</span>
                  </span>
                  <CompositionMiniBar counts={st?.counts || {}} total={st?.total || 0} />
                </button>
                <div className="absolute right-1 top-1 flex gap-0.5">
                  <button type="button" onClick={() => renameCustomPreset(preset)}
                    aria-label={`Rename custom preset ${preset.name}`} title="Rename preset"
                    className="grid h-5 w-5 place-items-center rounded text-[0.625rem] text-content-subtle hover:bg-white/10 hover:text-content">✎</button>
                  <button type="button" onClick={() => removeCustomPreset(preset)}
                    aria-label={`Delete custom preset ${preset.name}`} title="Delete preset"
                    className="grid h-5 w-5 place-items-center rounded text-[0.625rem] text-content-subtle hover:bg-red-500/15 hover:text-red-300">✕</button>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Shot list header + card-state legend — three unambiguous states (the
          amber chips in the group headers are the composition quota, a
          separate concern). */}
      <div className="flex items-center gap-2 pt-1">
        <span className="text-content-muted text-[0.6875rem] uppercase">Shots</span>
        <span className="text-content-subtle text-[0.625rem]">
          a preset pre-selects a balanced mix — click any card to add or remove it
        </span>
      </div>
      <div className="flex items-center gap-3 flex-wrap text-[0.625rem] text-content-subtle" aria-hidden="true">
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded border border-primary/50 bg-primary/20 ring-1 ring-primary/30" />
          selected — will be generated
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded border border-emerald-500/40 bg-emerald-500/10" />
          <span className="text-emerald-300">✓×N</span> already in your dataset
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded border border-border bg-app/40" />
          not selected
        </span>
      </div>

      {/* Shot picker, grouped by framing with a quota progress bar per group. */}
      <div className="max-h-80 overflow-auto flex flex-col gap-2 pr-1">
        {['face', 'bust', 'body', 'back'].map((fr) => {
          const have = (composition && composition[fr]) || 0;
          const missing = Math.max(0, TARGET[fr] - have);
          const pct = Math.min(100, (have / TARGET[fr]) * 100);
          const selCount = byFraming[fr].filter((e) => selected.has(e.id)).length;
          return (
            <div key={fr}>
              <div className="flex items-center gap-2 mb-1"
                title={`Your dataset contains ${have} "${frLabel(fr)}" image(s). Target for balanced training: ${TARGET[fr]} (this quota does NOT affect the generation selection).`}>
                <ShotIllustration framing={fr} label=""
                  className={`w-5 h-5 ${missing ? 'text-amber-300' : 'text-content-subtle'}`} />
                <span className={`text-[0.6875rem] uppercase font-semibold ${missing ? 'text-amber-300' : 'text-content-muted'}`}>
                  {frLabel(fr)}
                </span>
                <span className="w-24 h-1.5 rounded-full bg-app/60 overflow-hidden" aria-hidden="true">
                  <span className={`block h-full rounded-full ${missing ? 'bg-amber-400' : 'bg-emerald-400'}`}
                    style={{ width: `${pct}%` }} />
                </span>
                {missing > 0 ? (
                  <span className="px-1.5 py-px rounded-full bg-amber-400/15 border border-amber-400/40 text-amber-300 text-[0.625rem]">
                    {have}/{TARGET[fr]} in the dataset · {missing} missing
                  </span>
                ) : (
                  <span className="text-emerald-400/90 text-[0.625rem]">✓ {have}/{TARGET[fr]}</span>
                )}
                {selCount > 0 && (
                  <span className="ml-auto text-content-subtle text-[0.625rem]">{selCount} selected</span>
                )}
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-1.5">
                {byFraming[fr].map((e) => {
                  const on = selected.has(e.id);
                  const done = doneByLabel.get(e.label) || 0;
                  const emoji = contextEmoji(e.label);
                  // Three unambiguous states (cf. legend above): indigo = selected,
                  // green = already generated in this dataset, neutral = neither.
                  // The old amber "deficit" glow on unselected cards read as a
                  // selection — the quota cue now lives only in the group header.
                  const cls = on
                    ? 'bg-primary/20 border-primary/50 text-white ring-1 ring-primary/30'
                    : done > 0
                      ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-100/90 hover:bg-emerald-500/15'
                      : 'border-border bg-app/40 text-content-muted hover:bg-surface-raised';
                  return (
                    <button key={e.id} type="button" onClick={() => toggle(e.id)}
                      aria-pressed={on}
                      title={done > 0 ? `${done} image(s) of this shot already in the dataset` : undefined}
                      className={`flex items-center gap-1.5 px-1.5 py-1 rounded-lg text-[0.625rem] border text-left transition-colors ${cls}`}>
                      <ShotIllustration framing={e.framing} label={e.label} className="w-7 h-7 shrink-0" />
                      <span className="min-w-0 leading-tight">
                        {emoji && <span className="mr-1" aria-hidden="true">{emoji}</span>}
                        {displayLabel(e.label)}
                      </span>
                      <span className="ml-auto shrink-0 flex items-center gap-1">
                        {done > 0 && (
                          <span className="text-emerald-300 font-semibold" aria-label={`${done} already in the dataset`}>
                            ✓×{done}
                          </span>
                        )}
                        {on && <span className="text-indigo-300" aria-hidden="true">✓</span>}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })}

        {/* Custom group — user-authored cards (the only deletable ones). */}
        {customShots.length > 0 && (
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span aria-hidden="true">✨</span>
              <span className="text-[0.6875rem] uppercase font-semibold text-content-muted">Custom</span>
              <span className="text-content-subtle text-[0.625rem]">
                your own shots, stored in this browser — Keep saves one for good, ✕ removes it
              </span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-1.5">
              {customShots.map((c) => renderUserShot(c, () => removeCustomShot(c.id),
                'Remove this custom shot', () => keepCustomShot(c)))}
            </div>
          </div>
        )}

        {/* Imported group — a JSON catalog (idea by ashish.sinha, Discord). Always
            AFTER the built-ins and never in their place: an import can be undone
            shot by shot, or all at once, and the shipped catalog is untouched. */}
        {importedShots.length > 0 && (
          <div>
            <div className="flex flex-wrap items-center gap-x-2 gap-y-1 mb-1">
              <span aria-hidden="true">📥</span>
              <span className="text-[0.6875rem] uppercase font-semibold text-content-muted">Imported</span>
              <span className="text-content-subtle text-[0.625rem]">
                {importedShots.length} shot{importedShots.length === 1 ? '' : 's'} from your JSON catalog — saved on this machine, not in the browser
              </span>
              <button type="button" onClick={removeAllImported}
                className="ml-auto px-1.5 py-px rounded border border-border text-content-subtle hover:text-white text-[0.625rem]">
                Remove all
              </button>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-1.5">
              {importedShots.map((c) => renderUserShot(c, () => removeImportedShot(c),
                'Remove this imported shot'))}
            </div>
          </div>
        )}
      </div>

      {/* 🔞 NSFW — local Klein only. Uncensored body catalog + free prompt.
          Never offered on the API engines (and the backend refuses them there). */}
      {/* `localOnlyRun` already means every selected engine is local, and the
          effect above prunes engines that aren't available — so no second
          availability test here (the old `&& klAvailable` would have hidden the
          🔞 catalog on a Krea-only run). */}
      {localOnlyRun && nsfwCatalog.length > 0 && (
        <div className={`rounded-lg border p-2 flex flex-col gap-2 ${nsfwMode
          ? 'border-rose-500/40 bg-rose-500/5' : 'border-border bg-app/30'}`}>
          <button type="button" onClick={() => setNsfwMode((v) => !v)} aria-pressed={nsfwMode}
            className="flex items-center gap-2 text-left">
            <span aria-hidden="true">🔞</span>
            <span className={`text-[0.75rem] font-semibold ${nsfwMode ? 'text-rose-300' : 'text-content-muted'}`}>
              NSFW mode {nsfwMode ? 'ON' : 'OFF'}
            </span>
            <span className="text-content-subtle text-[0.625rem]">
              uncensored body shots — generated locally by Klein, never sent to an API
            </span>
            <span className={`ml-auto w-8 h-4 rounded-full relative transition-colors ${nsfwMode ? 'bg-rose-500/70' : 'bg-app/80 border border-border'}`}
              aria-hidden="true">
              <span className={`absolute top-0.5 left-0.5 w-3 h-3 rounded-full bg-white transition-transform ${nsfwMode ? 'translate-x-4' : 'translate-x-0'}`} />
            </span>
          </button>
          {nsfwMode && (
            <>
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-1.5">
                {nsfwCatalog.map((e) => {
                  const on = selected.has(e.id);
                  const done = doneByLabel.get(e.label) || 0;
                  const cls = on
                    ? 'bg-rose-500/20 border-rose-400/60 text-white ring-1 ring-rose-400/30'
                    : done > 0
                      ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-100/90 hover:bg-emerald-500/15'
                      : 'border-border bg-app/40 text-content-muted hover:bg-surface-raised';
                  return (
                    <button key={e.id} type="button" onClick={() => toggle(e.id)} aria-pressed={on}
                      title={done > 0 ? `${done} image(s) of this shot already in the dataset` : e.prompt}
                      className={`flex items-center gap-1.5 px-1.5 py-1 rounded-lg text-[0.625rem] border text-left transition-colors ${cls}`}>
                      <ShotIllustration framing={e.framing} label={e.label} className="w-7 h-7 shrink-0" />
                      <span className="min-w-0 leading-tight">{displayLabel(e.label)}</span>
                      <span className="ml-auto shrink-0 flex items-center gap-1">
                        {done > 0 && <span className="text-emerald-300 font-semibold">✓×{done}</span>}
                        {on && <span className="text-rose-300" aria-hidden="true">✓</span>}
                      </span>
                    </button>
                  );
                })}
              </div>
              <p className="text-content-subtle text-[0.625rem]">
                Captions must keep describing the state (nude / lingerie…) so it stays
                promptable and does not bind to the trigger word — the captioner does this
                automatically. The Custom shot below follows this register while 🔞 is on.
              </p>
            </>
          )}
        </div>
      )}

      {/* Custom shot — free prompt, EVERY engine (rides the 🔞 register only when
          NSFW mode is on with Klein). Included in the next Generate alongside the
          selected catalog shots. Collapsed by default (power-user tool) — the
          <details> keeps its fields mounted, so drafts survive fold/unfold. */}
      <details className="rounded-lg border border-border bg-app/30 open:pb-2">
        <summary className="cursor-pointer select-none px-2.5 py-1.5 text-[0.75rem] text-content font-semibold">
          ✨ Custom shot
          <span className="ml-2 font-normal text-content-subtle text-[0.625rem]">
            write your own prompt — it becomes a reusable card in the Custom group above{nsfwMode && localOnlyRun ? ' — 🔞 register active' : ''}
          </span>
        </summary>
        <div className="px-2.5 pt-1 flex flex-col gap-1">
          <label className="text-content-muted text-[0.6875rem]" htmlFor="custom-shot-prompt">
            Describe outfit, pose and setting, pick a framing, then Add.
          </label>
          <div className="flex gap-1.5 items-start">
            <textarea id="custom-shot-prompt" value={customPrompt} rows={2}
              onChange={(e) => setCustomPrompt(e.target.value)}
              placeholder="e.g. full body shot, sitting on a vintage motorbike in a garage, leather jacket, warm light"
              className="flex-1 bg-app/60 border border-border rounded px-2 py-1 text-[0.6875rem] text-content resize-y" />
            <select value={customFraming} onChange={(e) => setCustomFraming(e.target.value)}
              aria-label="Custom shot framing"
              className="bg-app/60 border border-border rounded px-1 py-1 text-[0.6875rem] text-content">
              {['face', 'bust', 'body', 'back'].map((fr) => (
                <option key={fr} value={fr}>{FRAMING_LABEL[fr]}</option>
              ))}
            </select>
            <button type="button" onClick={addCustomShot} disabled={!customPrompt.trim()}
              className="px-2.5 py-1 rounded-lg bg-gradient-primary text-white text-[0.6875rem] font-semibold disabled:opacity-40">
              ＋ Add
            </button>
          </div>
        </div>
      </details>

      {/* 📥 Shot catalog JSON — idea by ashish.sinha (Discord): export the catalog,
          have an LLM write 40 more shots in the same shape, import the result.
          Export FIRST on purpose: nobody (and no LLM) can produce the right JSON
          without an example of it. Collapsed by default. */}
      <details className="rounded-lg border border-border bg-app/30 open:pb-2">
        <summary className="cursor-pointer select-none px-2.5 py-1.5 text-[0.75rem] text-content font-semibold">
          📥 Shot catalog (JSON)
          <span className="ml-2 font-normal text-content-subtle text-[0.625rem]">
            import your own shots — export first to get the format
          </span>
          <HelpBadge topic="shot-catalog-json" />
        </summary>
        <div className="px-2.5 pt-1 flex flex-col gap-1.5">
          <p className="text-content-muted text-[0.6875rem]">
            Export the {SUBJECT_TYPE_LABELS[subject]?.toLowerCase() || 'current'} catalog, ask an LLM
            for more shots in the same shape, then import the file. Imported shots are saved on this
            machine (not in the browser), so they follow you from one device to the next.
          </p>
          <div className="flex flex-wrap gap-1.5">
            <button type="button" onClick={() => importFileRef.current?.click()}
              title={`Import a JSON shot catalog (max ${Math.round(MAX_IMPORT_BYTES / 1024)} KB — nothing is added until you confirm the summary)`}
              className="px-2.5 py-1 rounded-lg border border-border text-content text-[0.6875rem] font-semibold hover:bg-surface-raised">
              ⬆ Import
            </button>
            <button type="button" onClick={exportShotCatalog}
              title="Download this subject's catalog as JSON — your own shots, plus a few built-in examples to show the format"
              className="px-2.5 py-1 rounded-lg border border-border text-content text-[0.6875rem] font-semibold hover:bg-surface-raised">
              ⬇ Export
            </button>
            <input ref={importFileRef} type="file" accept="application/json,.json" className="hidden"
              onChange={(e) => { readImportFile(e.target.files?.[0]); e.target.value = ''; }} />
          </div>
          {/* The review step. Nothing has been written yet — this is what makes a
              partly-bad file safe: the user sees exactly what would land and what
              was refused, and decides. */}
          {importReview && (
            <div className="rounded-lg border border-border bg-app/60 p-2 flex flex-col gap-1.5">
              <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                <span className="text-[0.6875rem] font-semibold text-content truncate max-w-full">
                  {importReview.name}
                </span>
                <span className="text-[0.625rem] text-emerald-300">
                  {importReview.result.accepted.length} ready
                </span>
                {importReview.result.rejected.length > 0 && (
                  <span className="text-[0.625rem] text-amber-300">
                    {importReview.result.rejected.length} rejected
                  </span>
                )}
              </div>
              {importReview.result.rejected.length > 0 && (
                <ul className="max-h-32 overflow-y-auto flex flex-col gap-0.5 text-[0.625rem] text-amber-200/90">
                  {importReview.result.rejected.map((r) => (
                    <li key={`${r.index}-${r.code}`}>• {r.message}</li>
                  ))}
                </ul>
              )}
              {(importReview.result.skippedExamples > 0 || importReview.result.ignoredFields.length > 0) && (
                <p className="text-[0.625rem] text-content-subtle">
                  {importReview.result.skippedExamples > 0
                    && `${importReview.result.skippedExamples} built-in example${importReview.result.skippedExamples === 1 ? '' : 's'} ignored. `}
                  {importReview.result.ignoredFields.length > 0
                    && `Ignored fields: ${importReview.result.ignoredFields.join(', ')} — an imported shot uses its framing's default aspect ratio.`}
                </p>
              )}
              <div className="flex flex-wrap gap-1.5">
                <button type="button" onClick={confirmImport}
                  disabled={importBusy || !importReview.result.accepted.length}
                  className="px-2.5 py-1 rounded-lg bg-gradient-primary text-white text-[0.6875rem] font-semibold disabled:opacity-40">
                  {importBusy ? 'Importing…' : `Import ${importReview.result.accepted.length} shot${importReview.result.accepted.length === 1 ? '' : 's'}`}
                </button>
                <button type="button" onClick={() => setImportReview(null)}
                  className="px-2.5 py-1 rounded-lg border border-border text-content-muted text-[0.6875rem] hover:bg-surface-raised">
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      </details>

      {/* ✨ Prompt suffixes — the dataset's creative-direction text, editable
          right here so it can be adjusted per batch (Idea by waltm — Discord).
          Applies to EVERY engine at generation time and shares the dataset
          fields with the ⚙️ Settings modal; persisted just before the batch is
          enqueued. Collapsed unless a suffix is already set. */}
      <details className="rounded-lg border border-border bg-app/30 open:pb-2"
        open={suffixOpen} onToggle={(e) => setSuffixOpen(e.currentTarget.open)}>
        <summary className="cursor-pointer select-none px-2.5 py-1.5 text-[0.75rem] text-content font-semibold flex items-center gap-1.5">
          ✨ Prompt suffixes
          <span className="font-normal text-content-subtle text-[0.625rem]">
            creative direction added to every generated shot{suffixDirty ? ' · applied when you generate' : ''}
          </span>
          <HelpBadge topic="prompt-suffixes" className="ml-1" />
        </summary>
        <div className="px-2.5 pt-1 flex flex-col gap-2">
          <label className="flex flex-col gap-1">
            <span className="text-content-muted text-[0.6875rem]">All shots</span>
            <input value={gSuffix} maxLength={300}
              onChange={(e) => setGSuffix(e.target.value)}
              placeholder="e.g. shot on 35mm film, warm tones"
              className="bg-app/60 border border-border rounded px-2 py-1 text-[0.6875rem] text-content" />
          </label>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {SUFFIX_KEYS.map((k) => (
              <label key={k} className="flex flex-col gap-1">
                <span className="text-content-muted text-[0.6875rem]">{FRAMING_LABEL[k]} shots</span>
                <input value={fSuffix[k]} maxLength={300}
                  onChange={(e) => setFSuffix((s) => ({ ...s, [k]: e.target.value }))}
                  aria-label={`${FRAMING_LABEL[k]} prompt suffix`}
                  className="bg-app/60 border border-border rounded px-2 py-1 text-[0.6875rem] text-content" />
              </label>
            ))}
          </div>
          <p className="text-content-subtle text-[0.625rem]">
            Free text appended to every <b>generated</b> variation — the identity lock is
            untouched. A framing suffix applies to that shot type first, then the global one.
            Saved to the dataset when you generate, and shared with ⚙️ Dataset settings.
          </p>
        </div>
      </details>
      <div className="flex items-center gap-2 flex-wrap border-t border-border pt-2">
        <span className="text-content-muted text-[0.6875rem]">{selected.size} selected</span>
        {selected.size > 0 && (
          <button type="button" onClick={() => setSelected(new Set())}
            className="text-content-subtle text-[0.6875rem] underline decoration-border hover:text-content"
            title="Clear the whole selection (presets and shots)">
            ✕ Deselect all
          </button>
        )}
        <label className="text-content-muted text-[0.6875rem] flex items-center"
          title="Generate each selected shot this many times">×
          <select value={multiplier} onChange={(e) => setMultiplier(+e.target.value)}
            aria-label="Variation multiplier"
            className="bg-app/60 border border-border rounded px-1 py-0.5 text-content ml-1">
            {[1, 2, 3].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>
        {!hasRef && (
          <span className="text-amber-300 text-[0.6875rem]">Set a reference photo first</span>
        )}
        {/* The run's real price, next to the button that spends it — switching to
            "All engines" multiplies the bill, and that must be visible where the
            decision is made, not only inside the mode selector. */}
        {engines.length > 0 && selected.size > 0 && (
          <span className="text-content-muted text-[0.6875rem]">
            {engines.map((e) => ENGINE_LABELS[e]).join(' + ')}
            {' · '}
            {runCost > 0 ? `≈ $${runCost.toFixed(2)}` : 'free'}
          </span>
        )}
        {/* Never a silently empty batch: an unchecked engine grid, an empty shot
            selection or a run over the server's per-batch cap all SAY why the
            button is dead instead of just greying it out. */}
        {blockedReason && hasRef && (
          <span className="text-amber-300 text-[0.6875rem]">{blockedReason}</span>
        )}
        {/* Disabled for the WHOLE batch, not just the launch request: `busy` is the
            hook's busyLive (local flag OR any server-side activity, restored on
            reload), so a generation already in flight — Nano Banana / ChatGPT /
            Klein alike — keeps this locked with a visible reason. */}
        <button type="button" onClick={go} disabled={busy || !hasRef || !!blockedReason}
          title={generating ? 'A generation batch is already running' : (blockedReason || undefined)}
          className="ml-auto px-4 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
          {busy
            ? (generating
                ? `Generating…${generating.total ? ` ${generating.done}/${generating.total}` : ''}`
                : '…')
            : `⚡ Generate (${totalImages(selected.size, engines, engineMode, multiplier)})`}
        </button>
      </div>
    </div>
  );
}
