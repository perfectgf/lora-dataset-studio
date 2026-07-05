// react-frontend/src/components/dataset/studio/ResultsGrid.jsx
/**
 * Grille(s) de résultats : une `<table>` par variante (format × cfg × steps).
 * Lignes = checkpoint, colonnes = strength ; chaque case = `<ResultCell>` (bande de
 * tuiles + score + ★). Extrait 1:1 du bloc `<table>` de l'ancien LoraTestStudio
 * (behavior-preserving) : mêmes classes Tailwind, même en-tête « ckpt \ strength ».
 */
import ResultCell from './ResultCell';

export default function ResultsGrid({ gridRows, gridCols, variantsInData, cellList, scoreMap, best, datasetId, onRate, onOpen, fmt }) {
  return variantsInData.map((variant) => (
    <div key={variant.key} className="flex flex-col gap-1">
      {variantsInData.length > 1 && (
        <span className="text-content-muted text-[0.625rem] uppercase">
          {variant.zModelLabel ? `${variant.zModelLabel} · ` : ''}Format {variant.aspect || '—'}{variant.cfg != null ? ` · CFG ${fmt(variant.cfg)}` : ''}{variant.steps != null ? ` · ${variant.steps}${variant.steps2 != null ? '/' + variant.steps2 : ''} steps` : ''}
        </span>
      )}
      <div className="overflow-x-auto">
        <table className="border-separate border-spacing-1">
          <caption className="sr-only">Test grid {variant.key}: rows = checkpoint, columns = strength</caption>
          <thead>
            <tr>
              <th scope="col" className="text-content-subtle text-[0.625rem] font-normal text-left px-1">ckpt \ strength</th>
              {gridCols.map((s) => (
                <th key={s} scope="col" className="text-content-muted text-[0.6875rem] tabular-nums px-1">{fmt(s)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {gridRows.map((row) => (
              <tr key={row.filename}>
                <th scope="row" className="text-content text-[0.6875rem] font-medium text-left px-1 whitespace-nowrap">{row.label}</th>
                {gridCols.map((s) => (
                  <ResultCell key={s} row={row} strength={s} variant={variant}
                    cellList={cellList} scoreMap={scoreMap} best={best} datasetId={datasetId}
                    onRate={onRate} onOpen={onOpen} fmt={fmt} />
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  ));
}
