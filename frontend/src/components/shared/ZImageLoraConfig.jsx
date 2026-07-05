/**
 * ZImageLoraConfig — character/style LoRA stack for the Z-Image (zturbo) mode.
 *
 * Mirrors the edit page's KleinLoraConfig pattern: self-persists the
 * {filename: {enabled, strength}} map in localStorage and reports the enabled
 * stack up via onChange([{filename, strength}]) so the generate handler can
 * forward it as `z_loras`. LoRA files live in ComfyUI/models/loras/z image/.
 */
import { useEffect, useMemo, useState } from 'react';

// Plage de strength par LoRA selon la famille :
//  - Krea : 0..6, étendue à 20 pour les LoRA « utility » (ex. filter-bypass, sans
//    effet marqué sous ~13) — alignée sur le clamp backend (inject_krea_loras ≤ 20).
//  - Hors Krea (Z-Image / SDXL) : -2..2 (demande user — le négatif inverse le
//    concept, au-delà de 2 ça dégrade) ; clamp backend élargi à [-2, 6].
const EXTENDED_MAX_PATTERNS = [/filterbypass/i];
const strengthRangeFor = (filename, krea) => {
  if (krea) {
    return EXTENDED_MAX_PATTERNS.some((re) => re.test(filename))
      ? { min: 0, max: 20 } : { min: 0, max: 6 };
  }
  return { min: -2, max: 2 };
};

// Libellé court d'un checkpoint dans un groupe déplié : seul le step distingue les
// frères (le nom du dataset est déjà porté par l'en-tête du groupe). Pas de step =
// checkpoint FINAL (entraînement terminé).
const stepLabel = (l) => (l.step != null ? `${l.step} steps` : 'final');

export default function ZImageLoraConfig({ loras = [], onChange, zModel = '', isFavorite, onToggleFavorite,
                                          storageKey = 'zimageLoras_v1', label = 'Character / style LoRA', emptyHint,
                                          krea = false, batchToggle = false }) {
  const [cfg, setCfg] = useState(() => {
    try { return JSON.parse(localStorage.getItem(storageKey)) || {}; } catch { return {}; }
  });
  // Quels groupes de checkpoints (dataset) sont dépliés — persisté à part du cfg
  // (le cfg reste indexé par filename : aucune migration d'état de sélection).
  const [openGroups, setOpenGroups] = useState(() => {
    try { return JSON.parse(localStorage.getItem(`${storageKey}_open`)) || {}; } catch { return {}; }
  });
  useEffect(() => {
    try { localStorage.setItem(`${storageKey}_open`, JSON.stringify(openGroups)); } catch { /* ignore */ }
  }, [openGroups, storageKey]);
  const toggleGroup = (key) => setOpenGroups((o) => ({ ...o, [key]: !o[key] }));

  useEffect(() => {
    try { localStorage.setItem(storageKey, JSON.stringify(cfg)); } catch { /* ignore */ }
    const enabled = loras
      .filter((l) => cfg[l.filename]?.enabled)
      .map((l) => ({
        filename: l.filename,
        strength: cfg[l.filename]?.strength ?? 1.0,
        // Studio uniquement (batchToggle) : ☑ batch = ce LoRA devient un AXE de
        // test (cellules avec/sans) au lieu d'être appliqué à toutes les cellules.
        ...(batchToggle ? { batch: !!cfg[l.filename]?.batch } : {}),
      }));
    onChange?.(enabled);
  }, [cfg, loras, onChange, storageKey, batchToggle]);

  // Favoris d'abord (pour le modèle courant) → repérage « d'un coup d'œil ».
  const favFirst = useMemo(() => {
    if (!zModel || !isFavorite) return loras;
    const fav = [], rest = [];
    for (const l of loras) (isFavorite(zModel, l.filename) ? fav : rest).push(l);
    return [...fav, ...rest];
  }, [loras, zModel, isFavorite]);

  // Séparation CHARACTER / STYLE (demande user) : un LoRA de personnage est
  // entraîné avec un trigger word (injecté auto au prompt) ; un LoRA sans
  // trigger = style/utilitaire partagé. Deux sections visuellement distinctes.
  const characterLoras = useMemo(() => favFirst.filter((l) => l.triggerWord), [favFirst]);
  const styleLoras = useMemo(() => favFirst.filter((l) => !l.triggerWord), [favFirst]);

  // Regroupement des checkpoints d'un même dataset (même trigger + base = `l.group`
  // fourni par le backend). Un LoRA sans `group` (non entraîné via l'app) reste seul
  // sous sa propre clé (= filename). Chaque groupe est trié par step croissant, le
  // checkpoint FINAL (step nul) en dernier. L'ordre des groupes suit favFirst (les
  // datasets avec un favori remontent). Un groupe à 1 checkpoint = ligne simple.
  const characterGroups = useMemo(() => {
    const map = new Map();
    for (const l of characterLoras) {
      const key = (l.group && l.group.trim()) ? l.group : l.filename;
      if (!map.has(key)) map.set(key, []);
      map.get(key).push(l);
    }
    return [...map.entries()].map(([key, items]) => ({
      key,
      items: items.slice().sort((a, b) => (a.step ?? Infinity) - (b.step ?? Infinity)),
    }));
  }, [characterLoras]);

  if (!loras.length) {
    return (
      <p className="normal-case tracking-normal text-[0.625rem] text-content-muted/70 m-0 text-left">
        {emptyHint || (<>No Z-Image LoRA — drop your .safetensors into{' '}
        <code className="text-content-muted">ComfyUI/models/loras/z image/</code></>)}
      </p>
    );
  }

  const toggle = (fn) => setCfg((c) => ({
    ...c, [fn]: { ...c[fn], enabled: !c[fn]?.enabled, strength: c[fn]?.strength ?? 1.0 },
  }));
  const setStrength = (fn, v) => setCfg((c) => {
    if (c[fn]?.locked) return c; // force verrouillée → ignore
    return { ...c, [fn]: { ...c[fn], enabled: c[fn]?.enabled ?? false, strength: v } };
  });
  const toggleLock = (fn) => setCfg((c) => ({
    ...c, [fn]: { ...c[fn], enabled: c[fn]?.enabled ?? false, strength: c[fn]?.strength ?? 1.0, locked: !c[fn]?.locked },
  }));

  return (
    <div className="flex flex-col gap-1.5 text-left normal-case tracking-normal">
      {/* label=null/'' → pas de titre interne (le parent Collapsible porte le titre). */}
      {label && (
        <span className="text-[0.6875rem] text-content-muted uppercase tracking-wide">
          {label}
        </span>
      )}
      {(() => {
        // visibleLabel : libellé affiché à la place de displayName (ex. step court
        // « 2000 steps » quand la ligne est un enfant d'un groupe déplié). Les
        // aria-label gardent le displayName COMPLET (contexte lecteur d'écran).
        const renderLora = (l, visibleLabel) => {
        const c = cfg[l.filename] || {};
        const fav = !!(zModel && isFavorite?.(zModel, l.filename));
        return (
          <div key={l.filename} className="flex flex-col gap-1 rounded-md border border-border bg-app/40 px-2 py-1.5">
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => onToggleFavorite?.(zModel, l.filename)}
                disabled={!zModel}
                aria-pressed={fav}
                aria-label={fav ? `Remove ${l.displayName} from this model's favorites` : `Mark ${l.displayName} as favorite for this model`}
                title={fav ? 'Favorite for this model — click to remove' : 'Mark as favorite for this model'}
                className={`shrink-0 leading-none text-[0.95rem] ${fav ? 'text-amber-300' : 'text-content-muted/40 hover:text-amber-300'} ${zModel ? 'cursor-pointer' : 'opacity-40 cursor-not-allowed'}`}
              >
                {fav ? '★' : '☆'}
              </button>
              <label className="flex items-center gap-2 cursor-pointer flex-1 min-w-0">
                <input type="checkbox" checked={!!c.enabled} onChange={() => toggle(l.filename)}
                  aria-label={`Enable ${l.displayName}`} />
                <span className="text-content text-[0.8125rem] truncate">{visibleLabel || l.displayName}</span>
              </label>
              {c.enabled && (
                <>
                  <span className="text-content-muted text-[0.6875rem] tabular-nums">
                    {(c.strength ?? 1.0).toFixed(2)}
                  </span>
                  <button type="button" onClick={() => toggleLock(l.filename)}
                    aria-pressed={!!c.locked}
                    title={c.locked ? 'Strength locked — click to unlock' : 'Lock the strength (prevents accidental changes)'}
                    className={`px-1 py-0.5 rounded text-[0.75rem] border leading-none ${c.locked ? 'border-amber-400/60 bg-amber-400/15 text-amber-300' : 'border-border bg-surface text-content-muted hover:text-content'}`}>
                    {c.locked ? '🔒' : '🔓'}
                  </button>
                </>
              )}
            </div>
            {c.enabled && (() => {
              const range = strengthRangeFor(l.filename, krea);
              return (
                <input type="range" min={range.min} max={range.max} step="0.05" value={c.strength ?? 1.0}
                  onChange={(e) => setStrength(l.filename, parseFloat(e.target.value))}
                  disabled={c.locked}
                  aria-label={`Strength of ${l.displayName}`}
                  className={`w-full accent-indigo-500 ${c.locked ? 'opacity-40 cursor-not-allowed' : ''}`} />
              );
            })()}
            {c.enabled && l.triggerWord && (
              <span className="text-content-subtle text-[0.625rem]">
                trigger: <code className="text-content-muted">{l.triggerWord}</code> (added automatically)
              </span>
            )}
            {c.enabled && batchToggle && (
              <label className="flex items-center gap-1.5 cursor-pointer text-[0.625rem] text-content-muted"
                title="Checked: this LoRA becomes a test AXIS — each config runs once WITHOUT it and once WITH it, instead of applying to every cell.">
                <input type="checkbox" checked={!!c.batch}
                  onChange={() => setCfg((cur) => ({
                    ...cur,
                    [l.filename]: { ...cur[l.filename], batch: !cur[l.filename]?.batch },
                  }))}
                  aria-label={`Test ${l.displayName} as a batch axis (with/without)`}
                  className="accent-amber-400 w-3.5 h-3.5" />
                <span className={c.batch ? 'text-amber-300 font-semibold' : ''}>
                  ⚖ Batch axis (compare with / without)
                </span>
              </label>
            )}
          </div>
        );
        };
        // Un groupe de checkpoints (dataset) : en-tête dépliable + enfants (steps).
        // Groupe à 1 checkpoint → ligne simple (pas de repli inutile). Replié, l'en-tête
        // montre le checkpoint ACTIF (+ sa strength) ou le nombre de checkpoints.
        const renderGroup = ({ key, items }) => {
          if (items.length === 1) return renderLora(items[0]);
          const open = !!openGroups[key];
          const enabledItems = items.filter((l) => cfg[l.filename]?.enabled);
          const active = enabledItems[0];
          const anyFav = !!(zModel && isFavorite && items.some((l) => isFavorite(zModel, l.filename)));
          return (
            <div key={key} className="rounded-md border border-border bg-app/40">
              <button type="button" onClick={() => toggleGroup(key)} aria-expanded={open}
                title={open ? 'Collapse this dataset' : 'Expand to pick a checkpoint'}
                className="flex items-center gap-2 w-full px-2 py-1.5 text-left">
                <span aria-hidden className="shrink-0 w-3 text-content-muted text-[0.7rem]">{open ? '▾' : '▸'}</span>
                {anyFav && <span aria-hidden className="shrink-0 text-amber-300 text-[0.85rem] leading-none">★</span>}
                <span className="flex-1 min-w-0 truncate text-content text-[0.8125rem]">{key}</span>
                {active ? (
                  <span className="shrink-0 whitespace-nowrap text-content-muted text-[0.6875rem]">
                    using {stepLabel(active)}{enabledItems.length > 1 ? ` (+${enabledItems.length - 1})` : ''}
                    {' · '}<span className="tabular-nums">{(cfg[active.filename]?.strength ?? 1.0).toFixed(2)}</span>
                  </span>
                ) : (
                  <span className="shrink-0 whitespace-nowrap text-content-subtle text-[0.6875rem]">{items.length} checkpoints</span>
                )}
              </button>
              {open && (
                <div className="flex flex-col gap-1 px-2 pb-2 pt-0.5">
                  {items.map((l) => renderLora(l, stepLabel(l)))}
                </div>
              )}
            </div>
          );
        };
        // Sous-titres seulement quand les DEUX catégories existent (sinon bruit).
        const both = characterLoras.length > 0 && styleLoras.length > 0;
        return (
          <>
            {both && (
              <span className="text-[0.625rem] text-content-subtle uppercase tracking-wide mt-0.5">
                Character LoRAs
              </span>
            )}
            {characterGroups.map(renderGroup)}
            {both && (
              <span className="text-[0.625rem] text-content-subtle uppercase tracking-wide mt-1.5">
                Style / utility LoRAs
              </span>
            )}
            {/* arrow-wrap : renderLora prend (l, visibleLabel) — .map passerait l'index
                comme label. Le style/utility reste PLAT (pas de regroupement). */}
            {styleLoras.map((l) => renderLora(l))}
          </>
        );
      })()}
    </div>
  );
}
