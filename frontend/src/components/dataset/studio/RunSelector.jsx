// react-frontend/src/components/dataset/studio/RunSelector.jsx
/**
 * En-tête de la zone résultats : toggle « 📊 Résultats » (repli), bouton
 * « 🗳 Voter (N) » (file de vote rapide) et sélecteur du run à afficher.
 * Extraction behavior-preserving du bloc d'en-tête de l'ancien LoraTestStudio.jsx.
 */
export default function RunSelector({
  runs,
  activeRunKey,
  onSelect,
  unvotedCount,
  onStartVote,
  greenCount,
  onStartReVote,
  displayedCount,
  showResults,
  onToggleResults,
  canExport,
  onExport,
}) {
  return (
    <div className="flex items-center gap-2 flex-wrap">
      <button type="button" onClick={onToggleResults} aria-expanded={showResults}
        className="flex items-center gap-1.5 text-left text-content-muted text-[0.625rem] uppercase">
        <span aria-hidden>📊</span> Results
        <span className="text-content-subtle normal-case">{displayedCount} img</span>
        <span aria-hidden>{showResults ? '▾' : '▸'}</span>
      </button>
      {unvotedCount > 0 && (
        <button type="button" onClick={onStartVote}
          title="Quickly vote on all unrated images (swipe or 👍/👎)"
          className="px-2.5 py-1 rounded-lg bg-gradient-primary text-white text-[0.6875rem] font-semibold">
          🗳 Vote ({unvotedCount})
        </button>
      )}
      {greenCount > 0 && (
        <button type="button" onClick={onStartReVote}
          title="2nd pass: re-vote ONLY the 👍 to narrow down (👎 = remove, 👍 = reconfirm, skip = unchanged)"
          className="px-2.5 py-1 rounded-lg border border-green-400/60 bg-green-500/15 text-green-200 text-[0.6875rem] font-semibold">
          ♻️ Re-vote 👍 ({greenCount})
        </button>
      )}
      <button type="button" onClick={onExport} disabled={!canExport}
        title={canExport ? 'Compose this run into one shareable image (checkpoints × strengths)' : 'No finished image to export yet'}
        className="px-2.5 py-1 rounded-lg border border-border bg-surface text-content-muted text-[0.6875rem] font-semibold hover:text-content disabled:opacity-40 disabled:cursor-not-allowed">
        🖼 Export grid
      </button>
      {runs.length > 1 && (
        <select value={activeRunKey || ''} onChange={(e) => onSelect(e.target.value)}
          aria-label="Choose the test run to display"
          className="ml-auto rounded-lg border border-border bg-surface px-2 py-1 text-[0.6875rem] text-content max-w-[280px]">
          {runs.map((r, i) => {
            // Taux de 👍 parmi les votes du run (likes / votés), comme le « % 👍 »
            // affiché par cellule. Caché si aucun vote (division par zéro).
            const voted = r.likes + r.dislikes;
            const pct = voted ? Math.round((r.likes / voted) * 100) : null;
            return (
              <option key={r.key} value={r.key}>
                {i === 0 ? '● Current run' : `Run #${runs.length - i}`} — {r.modelLabel || '?'} · 👍{r.likes} 👎{r.dislikes}{pct !== null ? ` · ${pct}% 👍` : ''} · «{(r.prompt || '').slice(0, 22)}»
              </option>
            );
          })}
        </select>
      )}
    </div>
  );
}
