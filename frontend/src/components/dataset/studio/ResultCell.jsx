// react-frontend/src/components/dataset/studio/ResultCell.jsx
/**
 * Une cellule de grille (checkpoint × strength) pour une variante (format/cfg/steps).
 * Bande de N tuiles (les seeds du batch) + score agrégé PAR CONFIG affiché UNE fois
 * + ★ sur la meilleure config. Extrait 1:1 du `renderCell` de l'ancien LoraTestStudio
 * (behavior-preserving) : même logique de clé `ckey`, même calcul `score`/`isBest`.
 *
 * Contrat souple (le calcul `list`/`score`/`isBest` est fait ICI, comme `renderCell`).
 */
import ResultTile from './ResultTile';

export default function ResultCell({ row, strength, variant, cellList, scoreMap, best, datasetId, onRate, onOpen, fmt }) {
  const key = `${row.filename}|${strength}|${variant.zModel || ''}|${variant.aspect || ''}|${variant.cfg ?? ''}|${variant.steps ?? ''}|${variant.steps2 ?? ''}`;
  const list = cellList.get(key);
  if (!list || !list.length) {
    return <td className="px-1 text-content-subtle text-[0.625rem] text-center">—</td>;
  }
  const zk = variant.zModel || '';
  const score = scoreMap.get(`${row.filename}|${strength}|${variant.aspect || ''}|${zk}|${variant.cfg ?? ''}|${variant.steps ?? ''}|${variant.steps2 ?? ''}`);
  const isBest = best && best.checkpoint === row.filename && best.strength === strength
    && (best.aspect || '') === (variant.aspect || '') && (best.z_model || '') === zk
    && (best.cfg ?? '') === (variant.cfg ?? '') && (best.steps ?? '') === (variant.steps ?? '')
    && (best.steps2 ?? '') === (variant.steps2 ?? '');
  return (
    <td className={`align-top rounded-lg p-1 ${isBest ? 'bg-amber-400/10 outline outline-1 outline-amber-400/50' : ''}`}>
      <div className="flex items-start gap-1">
        {list.map((c) => (
          <ResultTile key={c.id} cell={c} row={row} strength={strength} variant={variant}
            datasetId={datasetId} onRate={onRate} onOpen={onOpen} fmt={fmt} />
        ))}
      </div>
      {/* Score agrégé PAR CONFIG (toutes seeds/runs confondus) + confiance. */}
      <div className="flex items-center justify-end gap-1 mt-0.5">
        <span className="text-content-muted text-[0.6875rem] tabular-nums"
          title={score ? `+${score.likes} / −${score.dislikes} on ${score.images} image(s)` : ''}>
          {score && score.score !== 0 ? (score.score > 0 ? `+${score.score}` : score.score) : '·'}
          {score && score.voted > 0 && (
            <span className="text-content-subtle"> · {score.voted}/{score.images}
              {score.like_rate != null ? ` · ${Math.round(score.like_rate * 100)}%👍` : ''}</span>
          )}
          {score && score.low_confidence && score.voted > 0 && (
            <span className="text-amber-400" title="Few votes — low reliability"> ⚠</span>
          )}
          {isBest && <span aria-label="best config" title="Best config"> ★</span>}
        </span>
      </div>
    </td>
  );
}
