/** Variation catalog: presets + per-entry toggles + multiplier + Klein picker. */
import { useEffect, useMemo, useState } from 'react';
import Flux2KleinModelPicker from '../shared/Flux2KleinModelPicker';
import { useToast } from '../common/Toast';
import { useCapabilities } from '../../context/CapabilitiesContext';
import { apiFetch } from '../../api/fetchClient';
import ShotIllustration, { contextEmoji } from './ShotIllustration';
import { displayLabel } from '../../utils/labels';

const FRAMING_LABEL = { face: 'Face', bust: 'Bust', body: 'Body', back: 'Back' };
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

export default function VariationCatalog({ onGenerate, busy, hasRef, composition, images = [], bodyFidelity = false }) {
  const toast = useToast();
  const { caps } = useCapabilities();
  const [catalog, setCatalog] = useState([]);
  const [nsfwCatalog, setNsfwCatalog] = useState([]);
  const [presets, setPresets] = useState({});
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

  const addCustomShot = () => {
    const p = customPrompt.trim();
    if (!p) return;
    const hot = nsfwMode && isKlein;
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
  // Identity LoRA strength (F1): higher = closer to the reference face,
  // lower = more variety in the generated variations.
  // dx8152 consistency LoRA: anchors STRUCTURE, its guide recommends ~0.5 and
  // warns 0.8-1.0 can stop edits from applying (0.9 made variations near-copies).
  const [loraStrength, setLoraStrength] = useState(0.5);
  // Generator backend: Nano Banana Pro (Gemini API, ~0,15 $/image, zero GPU,
  // best face fidelity — user-validated default) or local Klein (GPU, free).
  const [generator, setGenerator] = useState(() => {
    try { return localStorage.getItem('datasetGenerator') || 'nanobanana'; } catch { return 'nanobanana'; }
  });
  useEffect(() => {
    try { localStorage.setItem('datasetGenerator', generator); } catch { /* ignore */ }
  }, [generator]);
  const isNB = generator === 'nanobanana';
  const isGPT = generator === 'chatgpt';
  const isKlein = !isNB && !isGPT;

  // Which engines the user actually enabled in Settings (config.engines.enabled),
  // on top of the live reachability probe in `caps.engines`.
  const [enabledEngines, setEnabledEngines] = useState(['nanobanana', 'chatgpt', 'klein']);
  useEffect(() => {
    let cancelled = false;
    apiFetch('/api/settings')
      .then((d) => { if (!cancelled) setEnabledEngines(d.config?.engines?.enabled || []); })
      .catch(() => { /* keep the permissive default on a transient failure */ });
    return () => { cancelled = true; };
  }, []);
  const nbAvailable = enabledEngines.includes('nanobanana') && caps.engines.nanobanana;
  const gptAvailable = enabledEngines.includes('chatgpt') && caps.engines.chatgpt;
  const klAvailable = enabledEngines.includes('klein') && caps.engines.klein;
  const currentAvailable = isKlein ? klAvailable : isNB ? nbAvailable : gptAvailable;
  // Klein unavailable has THREE distinct causes — the hint must name the right
  // one (a reachable ComfyUI with no Klein model used to show "Configure
  // ComfyUI", sending the user to re-check a step that was already green).
  const kleinHint = klAvailable ? null
    : !enabledEngines.includes('klein') ? '⚠ Klein is disabled in Settings (engines)'
    : !caps.comfyui?.reachable ? '⚠ Configure ComfyUI in Settings'
    : '⚠ Klein model missing — download it in the Setup step (models/unet/klein/)';

  useEffect(() => {
    let cancelled = false;
    fetch('/api/dataset/variations', { credentials: 'include' })
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then((d) => {
        if (cancelled) return;
        setCatalog(d.catalog || []);
        setNsfwCatalog(d.nsfw_catalog || []);
        setPresets(d.presets || {});
        // Body-fidelity datasets start on the body-emphasis preset (figure-visible
        // outfits); everyone else keeps the balanced default.
        const def = bodyFidelity ? (d.presets?.body_emphasis || d.presets?.balanced_25)
          : d.presets?.balanced_25;
        setSelected(new Set(def || []));
      })
      .catch(() => {
        // Loud failure (M6): an empty catalog otherwise looks like a UI bug.
        if (!cancelled) toast.error('Could not load the variation catalog');
      });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [toast]);

  const byFraming = useMemo(() => {
    const g = { face: [], bust: [], body: [], back: [] };
    catalog.forEach((e) => g[e.framing]?.push(e));
    return g;
  }, [catalog]);

  // Switching to an API engine drops any selected NSFW shots (Klein-only) —
  // catalog nsfw_ entries AND 🔞 custom cards alike.
  useEffect(() => {
    if (isKlein) return;
    const hotCustom = new Set(customShots.filter((c) => c.nsfw).map((c) => c.id));
    setSelected((s) => {
      const n = new Set([...s].filter((id) => !id.startsWith('nsfw_') && !hotCustom.has(id)));
      return n.size === s.size ? s : n;
    });
  }, [isKlein, customShots]);

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

  const go = () => {
    const variations = catalog.filter((e) => selected.has(e.id))
      .map((e) => ({ label: e.label, prompt: e.prompt, framing: e.framing }));
    // NSFW shots: local Klein only (the toggle is gated on the Klein engine,
    // and the backend refuses them on API engines).
    if (nsfwMode && isKlein) {
      variations.push(...nsfwCatalog.filter((e) => selected.has(e.id))
        .map((e) => ({ label: e.label, prompt: e.prompt, framing: e.framing, nsfw: true })));
    }
    // Custom cards: selectable like catalog shots; 🔞 ones only ride with Klein
    // (the label prefix is what regenerate uses to re-pick the uncensored wrapper).
    variations.push(...customShots
      .filter((c) => selected.has(c.id) && (isKlein || !c.nsfw))
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
    // Guard-rail: API engines bill per image — above $5 estimated, confirm with
    // the amount (silent for the free local Klein).
    const rate = isNB ? 0.15 : isGPT ? 0.17 : 0;
    const cost = toGen.length * multiplier * rate;
    if (cost > 5 && !window.confirm(
      `This will launch ${toGen.length * multiplier} API generation(s) `
      + `≈ $${cost.toFixed(2)} (${isNB ? 'Nano Banana' : 'ChatGPT'}).\n\nProceed?`)) return;
    onGenerate(toGen, multiplier, klein, loraStrength, generator);
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

      {/* Engine cards — Klein (local GPU) vs Nano Banana Pro vs ChatGPT (APIs).
          Each card disables itself with an actionable hint when its engine
          isn't configured/reachable or was turned off in Settings. */}
      <div className="flex items-center gap-2">
        <span className="text-content-muted text-[0.6875rem] uppercase">Engine</span>
        <span className="text-content-subtle text-[0.625rem]">
          where the images are made — Klein runs free on your GPU, APIs bill per image
        </span>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
        <button type="button" onClick={() => setGenerator('klein')} aria-pressed={isKlein}
          disabled={!klAvailable}
          className={`flex items-start gap-3 rounded-xl border p-3 text-left transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${isKlein
            ? 'border-primary/60 bg-primary/15 ring-1 ring-primary/40'
            : 'border-border bg-app/40 hover:enabled:bg-surface-raised'}`}>
          <GpuIcon className={`w-9 h-9 shrink-0 ${isKlein ? 'text-indigo-300' : 'text-content-subtle'}`} />
          <span className="flex flex-col gap-1 min-w-0">
            <span className={`text-[0.8125rem] font-semibold ${isKlein ? 'text-white' : 'text-content-muted'}`}>
              Klein <span className="font-normal text-content-subtle">· local</span>
            </span>
            <span className="flex flex-wrap gap-1">
              <span className="px-1.5 py-px rounded-full bg-emerald-500/15 border border-emerald-400/40 text-emerald-300 text-[0.625rem]">Free</span>
              <span className="px-1.5 py-px rounded-full bg-app/60 border border-border text-content-muted text-[0.625rem]">Your GPU</span>
              <span className="px-1.5 py-px rounded-full bg-app/60 border border-border text-content-muted text-[0.625rem]">NSFW OK</span>
            </span>
            {klAvailable ? (
              <span className="text-content-subtle text-[0.625rem]">Runs on this machine — slower, tunable face fidelity.</span>
            ) : (
              <a href="#/setup" onClick={(e) => e.stopPropagation()}
                className="text-amber-300 text-[0.625rem] underline decoration-amber-300/50">
                {kleinHint}
              </a>
            )}
          </span>
        </button>
        <button type="button" onClick={() => setGenerator('nanobanana')} aria-pressed={isNB}
          disabled={!nbAvailable}
          className={`flex items-start gap-3 rounded-xl border p-3 text-left transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${isNB
            ? 'border-amber-400/60 bg-amber-500/15 ring-1 ring-amber-400/40'
            : 'border-border bg-app/40 hover:enabled:bg-surface-raised'}`}>
          <span className="w-9 h-9 shrink-0 grid place-items-center text-2xl" aria-hidden="true">🍌</span>
          <span className="flex flex-col gap-1 min-w-0">
            <span className={`text-[0.8125rem] font-semibold ${isNB ? 'text-amber-200' : 'text-content-muted'}`}>
              Nano Banana Pro <span className="font-normal text-content-subtle">· API</span>
            </span>
            <span className="flex flex-wrap gap-1">
              <span className="px-1.5 py-px rounded-full bg-app/60 border border-border text-content-muted text-[0.625rem]">No GPU</span>
              <span className="px-1.5 py-px rounded-full bg-app/60 border border-border text-content-muted text-[0.625rem]">~$0.15/image</span>
              <span className="px-1.5 py-px rounded-full bg-app/60 border border-border text-content-muted text-[0.625rem]">SFW</span>
            </span>
            {nbAvailable ? (
              <span className={`text-[0.625rem] ${isNB ? 'text-amber-300' : 'text-content-subtle'}`}>
                Best face fidelity · estimated cost ≈ ${(selected.size * multiplier * 0.15).toFixed(2)}
              </span>
            ) : (
              <span className="text-amber-300 text-[0.625rem]">⚠ Add GEMINI_API_KEY in Settings</span>
            )}
          </span>
        </button>
        <button type="button" onClick={() => setGenerator('chatgpt')} aria-pressed={isGPT}
          disabled={!gptAvailable}
          className={`flex items-start gap-3 rounded-xl border p-3 text-left transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${isGPT
            ? 'border-emerald-400/60 bg-emerald-500/15 ring-1 ring-emerald-400/40'
            : 'border-border bg-app/40 hover:enabled:bg-surface-raised'}`}>
          <ChatGptIcon className={`w-9 h-9 shrink-0 ${isGPT ? 'text-emerald-300' : 'text-content-subtle'}`} />
          <span className="flex flex-col gap-1 min-w-0">
            <span className={`text-[0.8125rem] font-semibold ${isGPT ? 'text-emerald-200' : 'text-content-muted'}`}>
              ChatGPT <span className="font-normal text-content-subtle">· API</span>
            </span>
            <span className="flex flex-wrap gap-1">
              <span className="px-1.5 py-px rounded-full bg-app/60 border border-border text-content-muted text-[0.625rem]">No GPU</span>
              <span className="px-1.5 py-px rounded-full bg-app/60 border border-border text-content-muted text-[0.625rem]">~$0.17/image</span>
              <span className="px-1.5 py-px rounded-full bg-app/60 border border-border text-content-muted text-[0.625rem]">SFW</span>
            </span>
            {gptAvailable ? (
              <span className={`text-[0.625rem] ${isGPT ? 'text-emerald-300' : 'text-content-subtle'}`}>
                gpt-image-2 · estimated cost ≈ ${(selected.size * multiplier * 0.17).toFixed(2)}
              </span>
            ) : (
              <span className="text-amber-300 text-[0.625rem]">⚠ Add OPENAI_API_KEY in Settings</span>
            )}
          </span>
        </button>
      </div>

      {/* Preset cards with their framing-mix bar. */}
      <div>
        <div className="flex items-center gap-2 mb-1.5">
          <span className="text-content-muted text-[0.6875rem] uppercase">Presets</span>
          <span className="ml-auto flex items-center gap-2 text-[0.625rem] text-content-subtle" aria-hidden="true">
            {['face', 'bust', 'body', 'back'].map((fr) => (
              <span key={fr} className="flex items-center gap-1">
                <span className={`w-2 h-2 rounded-full ${FRAMING_COLOR[fr]}`} />{FRAMING_LABEL[fr]}
              </span>
            ))}
          </span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-1.5">
          {PRESET_META.map(({ key, name, hint }) => {
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
                title={`Your dataset contains ${have} "${FRAMING_LABEL[fr]}" image(s). Target for balanced training: ${TARGET[fr]} (this quota does NOT affect the generation selection).`}>
                <ShotIllustration framing={fr} label=""
                  className={`w-5 h-5 ${missing ? 'text-amber-300' : 'text-content-subtle'}`} />
                <span className={`text-[0.6875rem] uppercase font-semibold ${missing ? 'text-amber-300' : 'text-content-muted'}`}>
                  {FRAMING_LABEL[fr]}
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
              <span className="text-content-subtle text-[0.625rem]">your own shots — remove with ✕</span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-1.5">
              {customShots.map((c) => {
                const on = selected.has(c.id);
                const done = doneByLabel.get(c.label) || 0;
                const blocked = c.nsfw && !isKlein;   // 🔞 card while an API engine is active
                const cls = on
                  ? 'bg-primary/20 border-primary/50 text-white ring-1 ring-primary/30'
                  : done > 0
                    ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-100/90 hover:bg-emerald-500/15'
                    : 'border-border bg-app/40 text-content-muted hover:bg-surface-raised';
                return (
                  <div key={c.id} className={`relative flex items-center gap-1.5 px-1.5 py-1 rounded-lg text-[0.625rem] border transition-colors ${cls} ${blocked ? 'opacity-40' : ''}`}>
                    <button type="button" onClick={() => !blocked && toggle(c.id)} aria-pressed={on}
                      disabled={blocked}
                      title={blocked ? '🔞 shot — switch the generator to Klein' : c.prompt}
                      className="flex items-center gap-1.5 flex-1 min-w-0 text-left disabled:cursor-not-allowed">
                      <ShotIllustration framing={c.framing} label={c.label} className="w-7 h-7 shrink-0" />
                      <span className="min-w-0 leading-tight truncate">{c.label}</span>
                      <span className="ml-auto shrink-0 flex items-center gap-1">
                        {done > 0 && <span className="text-emerald-300 font-semibold">✓×{done}</span>}
                        {on && <span className="text-indigo-300" aria-hidden="true">✓</span>}
                      </span>
                    </button>
                    <button type="button" onClick={() => removeCustomShot(c.id)}
                      aria-label={`Remove custom shot ${c.label}`} title="Remove this custom shot"
                      className="shrink-0 w-4 h-4 grid place-items-center rounded bg-black/40 text-content-subtle hover:text-white text-[0.625rem] leading-none">
                      ✕
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* 🔞 NSFW — local Klein only. Uncensored body catalog + free prompt.
          Never offered on the API engines (and the backend refuses them there). */}
      {isKlein && klAvailable && (
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
              <span className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-all ${nsfwMode ? 'left-4' : 'left-0.5'}`} />
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
            write your own prompt — it becomes a reusable card in the Custom group above{nsfwMode && isKlein ? ' — 🔞 register active' : ''}
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

      {/* Klein-only tuning, grouped: model file + consistency-LoRA strength.
          A <details> so the defaults stay out of a newcomer's way — children
          remain mounted, so the model picker still reports its choice. */}
      {isKlein && klAvailable && (
        <details className="rounded-lg border border-border bg-app/30 open:pb-2">
          <summary className="cursor-pointer select-none px-2.5 py-1.5 text-[0.75rem] text-content font-semibold">
            🖥️ Klein tuning
            <span className="ml-2 font-normal text-content-subtle text-[0.625rem]">
              model file · consistency LoRA {loraStrength <= 0 ? 'off' : loraStrength.toFixed(2)}
            </span>
          </summary>
          <div className="px-2.5 pt-1 flex flex-col gap-2">
            <div className="max-w-sm"><Flux2KleinModelPicker onChange={setKlein} /></div>
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
          </div>
        </details>
      )}
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
        <button type="button" onClick={go} disabled={busy || !selected.size || !hasRef || !currentAvailable}
          className="ml-auto px-4 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
          {busy ? '…' : `⚡ Generate (${selected.size * multiplier})`}
        </button>
      </div>
    </div>
  );
}
