// react-frontend/src/components/dataset/studio/LoraPicker.jsx
/**
 * Sélecteur de LoRA(s) à tester (Studio autonome). Fetch /api/studio/checkpoints,
 * affiche une carte cochable par (dataset × FAMILLE) ; chaque LoRA coché ouvre un
 * <select> de checkpoint (défaut = le 1er = le final). Pré-coche `preselectDataset`.
 *
 * Un dataset entraîné en PLUSIEURS familles (ex. « Lola » en Z-Image ET Krea) apparaît
 * en PLUSIEURS lignes (une par famille) — le backend émet une entrée par (dataset,
 * famille). D'où la clé COMPOSITE `${dataset_id}:${family}` : sans elle, deux lignes du
 * même dataset auraient la même clé React et un état `picked` ambigu.
 *
 * Émet `onSelectionChange([{dataset_id, checkpoint, lora_label, train_type, family}])` à
 * chaque changement (coche/décoche/choix de checkpoint). Affiche un badge
 * « Comparaison » dès que ≥2 LoRA sont cochés.
 *
 * Verrou de type (Task 5) : un run = une seule famille. Dès qu'un LoRA est coché, les
 * LoRA d'une autre famille sont désactivés (grisés + infobulle). Désélectionner tout
 * réinitialise le verrou.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { useToast } from '../../common/Toast';
import { useI18n } from '../../../i18n/I18nContext';

// Famille de l'entrée (le backend la fournit ; `train_type` = alias rétro-compat).
const famOf = (l) => l.family || l.train_type || 'zimage';
// Clé composite d'une ligne (dataset × famille) — identité stable dans `picked`.
const keyOf = (l) => `${l.dataset_id}:${famOf(l)}`;
// Badge de famille : libellé + couleur DISTINCTE par pipeline (toutes taguées, y
// compris Z-Image — sinon une ligne zimage d'un dataset multi-famille reste ambiguë).
const FAMILY_BADGE_LABEL = { zimage: 'Z-Image', sdxl: 'SDXL', krea: 'Krea' };
const familyBadgeClass = (fam) => ({
  zimage: 'border-sky-400/40 bg-sky-500/10 text-sky-300',
  sdxl: 'border-violet-400/40 bg-violet-500/10 text-violet-300',
  krea: 'border-amber-400/40 bg-amber-500/10 text-amber-300',
}[fam] || 'border-border-strong bg-white/5 text-content-muted');

export default function LoraPicker({ preselectDataset, onSelectionChange }) {
  const toast = useToast();
  const { t } = useI18n();
  const [loras, setLoras] = useState([]);
  const [loading, setLoading] = useState(true);
  // Map "datasetId:family" -> checkpoint filename choisi (présence de la clé = coché).
  const [picked, setPicked] = useState({});
  // Pré-cocher une seule fois (sinon on re-coche à chaque re-fetch).
  const preselectedRef = useRef(false);
  // Restauration de la sélection persistée : une seule fois, après le 1er fetch.
  const restoredRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    fetch('/api/studio/checkpoints', { credentials: 'include' })
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then((d) => {
        if (cancelled) return;
        setLoras(d.loras || []);
        setLoading(false);
      })
      .catch(() => {
        if (cancelled) return;
        setLoading(false);
        toast.error(t('studio.loraPicker.loadFailed'));
      });
    return () => { cancelled = true; };
  }, [t, toast]);

  // Pré-coche la 1re ligne du dataset pré-sélectionné (depuis l'URL) une fois la liste
  // chargée : checkpoint par défaut = le 1er (= le final côté backend).
  useEffect(() => {
    if (preselectedRef.current || !preselectDataset || !loras.length) return;
    const target = loras.find((l) => String(l.dataset_id) === String(preselectDataset));
    if (target && target.checkpoints?.length) {
      preselectedRef.current = true;
      setPicked({ [keyOf(target)]: target.checkpoints[0].filename });
    }
  }, [preselectDataset, loras]);

  // Restaure la sélection du DERNIER passage (recharger la page ne perd plus les
  // LoRA cochés ni leurs checkpoints — demande user 2026-07-03). L'URL
  // `?dataset=` (préselection explicite) garde la priorité. On ne restaure que
  // les entrées encore présentes dans la liste fraîche (LoRA supprimé = ignoré),
  // et une seule famille (règle « un run = une famille »).
  useEffect(() => {
    if (restoredRef.current || !loras.length) return;
    restoredRef.current = true;
    if (preselectDataset) return;
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem('studioPicked_v1') || '{}') || {}; } catch { /* ignore */ }
    const valid = {};
    let fam = null;
    for (const l of loras) {
      const k = keyOf(l);
      if (saved[k] == null) continue;
      if (fam === null) fam = famOf(l);
      if (famOf(l) !== fam) continue;
      const cps = (l.checkpoints || []).map((c) => c.filename);
      valid[k] = cps.includes(saved[k]) ? saved[k] : (cps[0] || '');
    }
    if (Object.keys(valid).length) setPicked(valid);
  }, [loras, preselectDataset]);

  // Persiste la sélection à chaque changement (après restauration seulement,
  // sinon le {} initial écraserait la sauvegarde avant qu'on l'ait relue).
  useEffect(() => {
    if (!restoredRef.current && !preselectedRef.current) return;
    try { localStorage.setItem('studioPicked_v1', JSON.stringify(picked)); } catch { /* ignore */ }
  }, [picked]);

  // Remonte la sélection normalisée au parent à chaque changement. Chaque entrée inclut
  // train_type ET family (= la famille de la LIGNE, pas le train_type du dataset) pour
  // que StudioShell fetch les bonnes bases et que le backend valide la famille unique.
  const selection = useMemo(() => {
    const out = [];
    for (const l of loras) {
      const cp = picked[keyOf(l)];
      if (cp) out.push({
        dataset_id: l.dataset_id,
        checkpoint: cp,
        lora_label: l.lora_label,
        train_type: famOf(l),
        family: famOf(l),
      });
    }
    return out;
  }, [loras, picked]);

  // Famille du run = celle du 1er LoRA coché (null si rien coché).
  const runType = selection.length > 0 ? selection[0].family : null;

  useEffect(() => { onSelectionChange?.(selection); }, [selection, onSelectionChange]);

  const toggle = (l) => {
    // Ne pas permettre de cocher un LoRA d'une famille différente du run en cours.
    const k = keyOf(l);
    if (runType !== null && famOf(l) !== runType && picked[k] == null) return;
    setPicked((cur) => {
      const next = { ...cur };
      if (next[k] != null) delete next[k];
      else next[k] = l.checkpoints?.[0]?.filename || '';
      return next;
    });
  };
  const setCheckpoint = (key, filename) =>
    setPicked((cur) => ({ ...cur, [key]: filename }));

  const count = selection.length;

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-border bg-surface p-3">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-content-muted text-[0.6875rem] uppercase">{t('studio.loraPicker.title')}</span>
        {count >= 2 && (
          <span className="px-2 py-0.5 rounded-full text-[0.625rem] font-semibold bg-amber-400/15 border border-amber-400/40 text-amber-200">
            ⚖ {t('studio.loraPicker.comparison', { count })}
          </span>
        )}
        <span className="ml-auto text-content-subtle text-[0.6875rem]">
          {t('studio.loraPicker.checked', { count })}
        </span>
      </div>

      {loading ? (
        <p className="text-content-subtle text-sm">{t('studio.loraPicker.loading')}</p>
      ) : loras.length === 0 ? (
        <p className="text-content-subtle text-sm">
          {t('studio.loraPicker.empty')}
        </p>
      ) : (
        <div className="max-h-72 overflow-auto flex flex-col gap-1.5">
          {loras.map((l) => {
            const k = keyOf(l);
            const on = picked[k] != null;
            // Verrou de famille : grisé si une autre famille est déjà sélectionnée.
            const lType = famOf(l);
            const locked = runType !== null && !on && lType !== runType;
            return (
              <div key={k}
                className={`flex flex-col gap-1 rounded-lg border px-2.5 py-2 ${
                  on ? 'border-primary/40 bg-primary/10'
                  : locked ? 'border-border bg-surface-raised opacity-50'
                  : 'border-border bg-surface-raised'
                }`}>
                <button type="button" onClick={() => toggle(l)} aria-pressed={on}
                  disabled={locked}
                  title={locked ? t('studio.loraPicker.familyLocked') : undefined}
                  className={`flex items-center gap-2 text-left ${locked ? 'cursor-not-allowed' : ''}`}>
                  <span aria-hidden className={`inline-flex w-4 h-4 shrink-0 items-center justify-center rounded border text-[0.625rem] ${on ? 'border-primary bg-primary/30 text-white' : 'border-border text-transparent'}`}>
                    ✓
                  </span>
                  <span className="text-content font-medium text-sm truncate" title={l.lora_label}>
                    {l.lora_label}
                  </span>
                  {l.trigger_word && (
                    <code className="px-1.5 py-0.5 rounded border border-indigo-400/40 bg-indigo-500/10 text-indigo-300 text-[0.625rem] font-semibold">
                      {l.trigger_word}
                    </code>
                  )}
                  {/* Badge de famille — TOUTES les familles taguées (Z-Image incluse) :
                      un dataset multi-famille a une ligne par pipeline, une ligne sans
                      badge serait ambiguë. Couleur distincte par famille. */}
                  <span className={`px-1.5 py-0.5 rounded border text-[0.5625rem] font-semibold uppercase ${familyBadgeClass(lType)}`}>
                    {FAMILY_BADGE_LABEL[lType] || lType}
                  </span>
                  <span className="ml-auto text-content-subtle text-[0.625rem] truncate max-w-[120px]" title={l.dataset_name}>
                    {l.dataset_name}
                  </span>
                </button>
                {on && l.checkpoints?.length > 1 && (
                  <label className="flex items-center gap-2 text-content-muted text-[0.6875rem] pl-6">
                    <span className="whitespace-nowrap">{t('studio.common.checkpoint')}:</span>
                    <select value={picked[k] || ''}
                      onChange={(e) => setCheckpoint(k, e.target.value)}
                      aria-label={t('studio.loraPicker.checkpointFor', { name: l.lora_label })}
                      className="flex-1 min-w-0 rounded border border-border bg-app/60 px-1.5 py-0.5 text-content">
                      {l.checkpoints.map((c) => (
                        <option key={c.filename} value={c.filename}>{c.label}</option>
                      ))}
                    </select>
                  </label>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
