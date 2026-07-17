// react-frontend/src/components/dataset/studio/flipOrder.js
/**
 * Ordre de navigation de la lightbox de résultats. But utilisateur explicite :
 * feuilleter d'un strength à l'autre POUR LE MÊME RENDU (même image, seed identique)
 * sans fermer la lightbox → les variantes de strength d'un même rendu doivent être
 * ADJACENTES dans la navigation.
 *
 * `keyOf(cell)` renvoie le tuple de tri avec la STRENGTH EN DERNIER : on regroupe
 * donc par « identité du rendu » (tout sauf la strength), et à identité égale on
 * ordonne par strength croissante. Deux images voisines ne diffèrent alors que par
 * la strength — exactement le balayage que l'utilisateur veut. On ne garde que les
 * cellules affichables (générées + fichier présent), les seules que la lightbox
 * sait ouvrir.
 */
function cmpTuple(a, b) {
  const n = Math.max(a.length, b.length);
  for (let i = 0; i < n; i += 1) {
    const x = a[i];
    const y = b[i];
    if (x === y) continue;
    if (typeof x === 'number' && typeof y === 'number') return x - y;
    return String(x ?? '').localeCompare(String(y ?? ''), undefined, { numeric: true });
  }
  return 0;
}

export function flipOrder(cells, keyOf) {
  return (cells || [])
    .filter((c) => c.status === 'done' && c.filename)
    .map((c) => ({ c, k: keyOf(c) }))
    .sort((a, b) => cmpTuple(a.k, b.k))
    .map((x) => x.c);
}
