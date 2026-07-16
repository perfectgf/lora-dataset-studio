// react-frontend/src/components/dataset/studio/ResultsArea.jsx
/**
 * Zone « 📊 Résultats » du Studio de test LoRA. Possède l'état d'affichage
 * (repli `showResults`) et le run sélectionné (`selRun`), recalcule tout le
 * regroupement par run / config / variante à partir de `d.cells` et `d.scores`
 * (extraction behavior-preserving depuis l'ancien LoraTestStudio.jsx), puis rend
 * le sélecteur de run + une grille par variante (format × cfg × steps).
 */
import { useMemo, useState } from 'react';
import { fmt } from '../../../utils/studioFormat';
import RunSelector from './RunSelector';
import ResultsGrid from './ResultsGrid';
import ExportGridModal from './ExportGridModal';

export default function ResultsArea({ datasetId, d, studio, vote, onOpen }) {
  // Repli des grilles de résultats (pour ne pas encombrer la page).
  const [showResults, setShowResults] = useState(true);
  // Run sélectionné (null = run le plus récent par défaut).
  const [selRun, setSelRun] = useState(null);
  // Modale « Export grid » (compose le run affiché en UNE image partageable).
  const [exportOpen, setExportOpen] = useState(false);

  // --- Regroupement par RUN (un lancement = même seed + prompt + modèle). On
  // n'affiche que le run sélectionné (le plus récent par défaut) pour ne pas
  // mélanger d'anciens tests déjà votés avec un nouveau run.
  const runs = useMemo(() => {
    const groups = new Map();
    for (const c of d?.cells || []) {
      // Un lancement = un run_seed (regroupe les N seeds d'un batch). Fallback sur
      // `seed` pour les anciens runs (avant la colonne run_seed).
      const runSeed = c.run_seed ?? c.seed;
      // Un lancement = un run_seed (N seeds d'un batch + TOUS les modèles de base
      // balayés). Le modèle est un axe de VARIANTE, pas un run distinct.
      const key = `${runSeed}|${c.prompt || ''}`;
      let g = groups.get(key);
      if (!g) {
        g = { key, seed: runSeed, prompt: c.prompt || '', models: new Set(),
              cells: [], latestId: 0, likes: 0, dislikes: 0 };
        groups.set(key, g);
      }
      g.cells.push(c);
      if (c.z_model_label) g.models.add(c.z_model_label);
      if (c.id > g.latestId) g.latestId = c.id;
      if (c.rating === 1) g.likes += 1; else if (c.rating === -1) g.dislikes += 1;
    }
    return [...groups.values()].map((g) => ({
      ...g, modelLabel: g.models.size > 1 ? `${g.models.size} models` : ([...g.models][0] || ''),
    })).sort((a, b) => b.latestId - a.latestId);
  }, [d]);
  const activeRunKey = (runs.find((r) => r.key === selRun) ? selRun : runs[0]?.key) || null;
  const displayedCells = useMemo(() => {
    const r = runs.find((x) => x.key === activeRunKey);
    return r ? r.cells : [];
  }, [runs, activeRunKey]);

  // Dernière cellule par config dans le run affiché.
  // Clé d'une cellule = checkpoint|strength|format|cfg|steps|steps2 (steps2 = pass 2
  // SDXL ; vide pour Z-Image → clé inchangée).
  const ckey = (c) => `${c.checkpoint}|${c.strength}|${c.z_model || ''}|${c.aspect || ''}|${c.cfg ?? ''}|${c.steps ?? ''}|${c.steps2 ?? ''}`;
  // Batch : TOUTES les cellules par config (les N seeds), triées par seed → bande.
  const cellList = useMemo(() => {
    const m = new Map();
    for (const c of displayedCells) {
      const k = ckey(c);
      if (!m.has(k)) m.set(k, []);
      m.get(k).push(c);
    }
    for (const arr of m.values()) arr.sort((a, b) => (a.seed || 0) - (b.seed || 0));
    return m;
  }, [displayedCells]);

  // Score cross-runs PAR CONFIG (modèle + cfg + steps inclus) — aligné backend.
  const scoreMap = useMemo(() => {
    const m = new Map();
    for (const s of d?.scores || []) {
      m.set(`${s.checkpoint}|${s.strength}|${s.aspect || ''}|${s.z_model || ''}|${s.cfg ?? ''}|${s.steps ?? ''}|${s.steps2 ?? ''}`, s);
    }
    return m;
  }, [d]);

  // Variantes présentes dans le run affiché (format × cfg × steps) → une grille par variante.
  const variantsInData = useMemo(() => {
    const m = new Map();
    for (const c of displayedCells) {
      const k = `${c.z_model || ''}|${c.aspect || ''}|${c.cfg ?? ''}|${c.steps ?? ''}|${c.steps2 ?? ''}`;
      if (!m.has(k)) m.set(k, { key: k, zModel: c.z_model || '', zModelLabel: c.z_model_label || '',
                                aspect: c.aspect || '', cfg: c.cfg, steps: c.steps, steps2: c.steps2 });
    }
    return [...m.values()].sort((a, b) =>
      (a.zModelLabel || '').localeCompare(b.zModelLabel || '')
      || a.aspect.localeCompare(b.aspect) || ((a.cfg ?? 0) - (b.cfg ?? 0))
      || ((a.steps ?? 0) - (b.steps ?? 0)) || ((a.steps2 ?? 0) - (b.steps2 ?? 0)));
  }, [displayedCells]);

  const gridRows = useMemo(() => {
    const seen = new Map();
    for (const c of displayedCells) if (!seen.has(c.checkpoint)) seen.set(c.checkpoint, c.label);
    return [...seen.entries()].map(([filename, label]) => ({ filename, label }))
      .sort((a, b) => a.label.localeCompare(b.label, undefined, { numeric: true }));
  }, [displayedCells]);

  const gridCols = useMemo(() => {
    const set = new Set(displayedCells.map((c) => c.strength));
    return [...set].sort((a, b) => a - b);
  }, [displayedCells]);  // dépend des cellules affichées (pas de d) — sinon colonnes figées au changement de run

  // Run actif (objet) + axes présents pour la modale d'export.
  const activeRun = useMemo(() => runs.find((r) => r.key === activeRunKey) || null, [runs, activeRunKey]);
  const exportAspects = useMemo(
    () => [...new Set(displayedCells.map((c) => c.aspect).filter(Boolean))].sort(),
    [displayedCells]);
  const canExport = displayedCells.some((c) => c.status === 'done' && c.filename);

  // --- Mode vote rapide : enchaîne les images non votées (swipe / 👍 / 👎) ----
  const unvoted = displayedCells.filter((c) => c.status === 'done' && c.filename && !c.rating);
  // 2e passe : revoter UNIQUEMENT les 👍 pour resserrer (un 👎 les bascule rouge,
  // un 👍 les reconfirme, passer les laisse vertes).
  const greens = displayedCells.filter((c) => c.status === 'done' && c.filename && c.rating === 1);

  if (gridRows.length === 0) return null;

  return (
    <div className="flex flex-col gap-1">
      <RunSelector
        runs={runs}
        activeRunKey={activeRunKey}
        onSelect={(key) => setSelRun(key)}
        unvotedCount={unvoted.length}
        onStartVote={() => vote.startVoting(unvoted)}
        greenCount={greens.length}
        onStartReVote={() => vote.startVoting(greens, '♻️ Reconfirm the 👍')}
        displayedCount={displayedCells.length}
        showResults={showResults}
        onToggleResults={() => setShowResults((v) => !v)}
        canExport={canExport}
        onExport={() => setExportOpen(true)}
      />
      {showResults && (
        <ResultsGrid
          gridRows={gridRows}
          gridCols={gridCols}
          variantsInData={variantsInData}
          cellList={cellList}
          scoreMap={scoreMap}
          best={d.best_cell}
          datasetId={datasetId}
          onRate={studio.rate}
          onOpen={onOpen}
          fmt={fmt}
        />
      )}
      <ExportGridModal
        open={exportOpen}
        onClose={() => setExportOpen(false)}
        datasetId={datasetId}
        family={d.family}
        run={activeRun}
        aspects={exportAspects}
        rows={gridRows.length}
        cols={gridCols.length}
      />
    </div>
  );
}
