export const CANVAS_FAMILY_SELECTION_KEY = 'lds.canvasModelFamilies';
const CANVAS_EXTRA_FILTERS_KEY = 'lds.canvasExtraFilters';
const CANVAS_STATUS_ORDER = ['active', 'completed', 'error', 'unknown'];

const ACTIVE_STATUSES = new Set([
  'preparing', 'provisioning', 'uploading', 'training', 'downloading', 'terminating',
]);

const CANVAS_FAMILY_LABELS = {
  zimage: 'Z-Image', krea: 'Krea 2', sdxl: 'SDXL',
  flux: 'FLUX.1', flux2klein: 'FLUX.2 Klein', anima: 'Anima',
};

const FAMILY_ORDER = Object.keys(CANVAS_FAMILY_LABELS);
const asFamilies = (value) => (Array.isArray(value) ? value : [])
  .filter((family) => typeof family === 'string')
  .map((family) => family.trim())
  .filter(Boolean)
  .filter((family, index, all) => all.indexOf(family) === index);

export const familyLabel = (family) => CANVAS_FAMILY_LABELS[family] || family;

export function runStatusCategory(node) {
  const status = String(node?.status || '').trim().toLowerCase();
  if (ACTIVE_STATUSES.has(status)) return 'active';
  if (status === 'done') return 'completed';
  if (status.includes('error') || ['failed', 'cancelled', 'stopped'].includes(status)) return 'error';
  // Successful local runs have no CloudTrainingRun row, hence no status. The
  // backend stamps the one failed local record as `error`; every other local
  // provenance record is a completed run even after its saves were cleaned up.
  if (!status && node?.source === 'local') return 'completed';
  if (!status && (node?.checkpoint_ready === true || (node?.checkpoints || []).length > 0)) {
    return 'completed';
  }
  return 'unknown';
}

export function availableStatusCategories(trees) {
  const found = new Set();
  for (const state of Object.values(trees || {})) {
    for (const node of (state?.tree?.nodes || [])) found.add(runStatusCategory(node));
  }
  return CANVAS_STATUS_ORDER.filter((status) => found.has(status));
}

export function readCanvasExtraFilters(store, key = CANVAS_EXTRA_FILTERS_KEY) {
  try {
    const value = JSON.parse(store?.getItem(key));
    return {
      statuses: Array.isArray(value?.statuses) ? value.statuses.filter((s) => CANVAS_STATUS_ORDER.includes(s)) : null,
      showPinned: value?.showPinned !== false,
    };
  } catch {
    return { statuses: null, showPinned: true };
  }
}

export function writeCanvasExtraFilters(store, value, key = CANVAS_EXTRA_FILTERS_KEY) {
  try {
    store?.setItem(key, JSON.stringify({
      statuses: Array.isArray(value?.statuses) ? value.statuses.filter((s) => CANVAS_STATUS_ORDER.includes(s)) : null,
      showPinned: value?.showPinned !== false,
    }));
    return true;
  } catch {
    return false;
  }
}

/* Le panneau « Datasets » est-il DÉPLIÉ ?
   Replié par défaut, à toutes les largeurs : il s'ouvrait grand à chaque
   chargement sur écran large, et sa liste de cases poussait le board — la chose
   qu'on vient regarder — sous la ligne de flottaison. C'est un filtre, on ne le
   consulte pas à chaque visite ; le bouton dit déjà ce qui est affiché (« 3 of
   7 »), donc le replier ne cache aucune information.
   Le choix de l'utilisateur, lui, SURVIT au rechargement (même contrat que les
   autres préférences d'affichage du canvas, juste au-dessus). */
export const CANVAS_FILTER_OPEN_KEY = 'lds.canvasFilterOpen';

export function readCanvasFilterOpen(store, key = CANVAS_FILTER_OPEN_KEY) {
  try {
    return store?.getItem(key) === '1';
  } catch {
    return false;
  }
}

export function writeCanvasFilterOpen(store, open, key = CANVAS_FILTER_OPEN_KEY) {
  try {
    store?.setItem(key, open ? '1' : '0');
    return true;
  } catch {
    return false;
  }
}

export function availableModelFamilies(datasets) {
  const found = asFamilies((datasets || []).flatMap((d) => d?.families || []));
  return found.sort((a, b) => {
    const ai = FAMILY_ORDER.indexOf(a);
    const bi = FAMILY_ORDER.indexOf(b);
    if (ai < 0 && bi < 0) return a.localeCompare(b);
    if (ai < 0) return 1;
    if (bi < 0) return -1;
    return ai - bi;
  });
}

export function readFamilySelection(store, key = CANVAS_FAMILY_SELECTION_KEY) {
  try {
    const raw = store?.getItem(key);
    if (raw == null) return null;
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? asFamilies(parsed) : null;
  } catch {
    return null;
  }
}

export function writeFamilySelection(store, families, key = CANVAS_FAMILY_SELECTION_KEY) {
  try {
    store?.setItem(key, JSON.stringify(asFamilies(families)));
    return true;
  } catch {
    return false;
  }
}

export function resolveFamilySelection(available, stored) {
  const all = asFamilies(available);
  if (stored == null) return all;
  const wanted = new Set(asFamilies(stored));
  return all.filter((family) => wanted.has(family));
}

export function toggleFamilySelection(selected, family, available) {
  const all = asFamilies(available);
  const current = new Set(asFamilies(selected));
  if (current.has(family)) current.delete(family); else current.add(family);
  return all.filter((item) => current.has(item));
}

export function filterDatasetIdsByFamilies(datasets, selectedIds, selectedFamilies) {
  const wantedIds = new Set((selectedIds || []).map(Number));
  const wantedFamilies = new Set(asFamilies(selectedFamilies));
  return (datasets || [])
    .filter((d) => wantedIds.has(Number(d?.id))
      && asFamilies(d?.families).some((family) => wantedFamilies.has(family)))
    .map((d) => Number(d.id));
}

/** Remove runs from unselected model families while keeping the lineage shape
 * valid: edges to hidden runs disappear and a retained orphan becomes a root. */
export function filterLineageTreeByFamilies(tree, selectedFamilies) {
  return filterLineageTree(tree, { families: selectedFamilies });
}

export function filterLineageTree(tree, {
  families = null, statuses = null, query = '', datasetName = '',
} = {}) {
  if (!tree || !Array.isArray(tree.nodes)) return tree;
  const wantedFamilies = families == null ? null : new Set(asFamilies(families));
  const wantedStatuses = statuses == null ? null : new Set(statuses);
  const needle = String(query || '').trim().toLowerCase();
  const nodes = tree.nodes.filter((node) => {
    if (wantedFamilies && !wantedFamilies.has(node?.train_type)) return false;
    if (wantedStatuses && !wantedStatuses.has(runStatusCategory(node))) return false;
    if (!needle) return true;
    return [datasetName, node?.dataset_name, node?.record_id, node?.run_id,
      node?.train_type, node?.variant, node?.base_model, node?.note]
      .some((value) => String(value ?? '').toLowerCase().includes(needle));
  });
  const ids = new Set(nodes.map((node) => node.record_id));
  const fallbackCurrent = nodes[nodes.length - 1]?.record_id ?? null;
  return {
    ...tree,
    nodes,
    edges: (Array.isArray(tree.edges) ? tree.edges : [])
      .filter((edge) => ids.has(edge?.parent) && ids.has(edge?.child)),
    root_id: ids.has(tree.root_id) ? tree.root_id : (nodes[0]?.record_id ?? null),
    current_id: ids.has(tree.current_id) ? tree.current_id : fallbackCurrent,
  };
}
