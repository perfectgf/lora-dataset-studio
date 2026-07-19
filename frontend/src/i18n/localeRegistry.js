const DEFAULT_TITLE = 'LoRA Dataset Studio';
const DATASET_WORKSPACE_GROUPS = [
  'captionTools', 'imageBulk', 'captions', 'scrape', 'datasetSettings',
];
const SETTINGS_DETAIL_GROUPS = [
  'scraping', 'localTools', 'captioning', 'training', 'server', 'maintenance',
];

function catalogFromModule(module) {
  return module?.default || module || {};
}

function localeCodeFromPath(path) {
  return String(path || '').split('/').pop()?.replace(/\.json$/i, '') || '';
}

// Keep older catalog group locations compatible with the namespaces used by
// current components. This can be removed after the locale files are reorganized.
export function withCatalogAliases(catalog = {}) {
  const datasetGroups = Object.fromEntries(
    DATASET_WORKSPACE_GROUPS
      .map((key) => [key, catalog.datasets?.[key]])
      .filter(([, value]) => value !== undefined),
  );
  const settingsGroups = Object.fromEntries(
    SETTINGS_DETAIL_GROUPS
      .map((key) => [key, catalog.workspace?.[key]])
      .filter(([, value]) => value !== undefined),
  );
  return {
    ...catalog,
    workspace: { ...(catalog.workspace || {}), ...datasetGroups },
    settings: { ...(catalog.settings || {}), ...settingsGroups },
  };
}

/**
 * Build the runtime registry from Vite's import.meta.glob result.
 *
 * A new locale is therefore data-only: add locales/<code>.json and optionally
 * describe its native label/title/direction under `_meta`.
 */
export function buildLocaleRegistry(modules) {
  const seen = new Set();
  const records = Object.entries(modules || {}).map(([path, module]) => {
    const rawCatalog = catalogFromModule(module);
    const meta = rawCatalog._meta || {};
    const code = String(meta.locale || localeCodeFromPath(path)).trim();
    if (!code) throw new Error(`Locale file has no locale code: ${path}`);
    if (seen.has(code)) throw new Error(`Duplicate locale code: ${code}`);
    seen.add(code);
    return {
      code,
      label: String(meta.label || code),
      shortLabel: String(meta.shortLabel || meta.label || code),
      documentTitle: String(meta.documentTitle || DEFAULT_TITLE),
      direction: meta.direction === 'rtl' ? 'rtl' : 'ltr',
      order: Number.isFinite(Number(meta.order)) ? Number(meta.order) : 999,
      defaultForLanguage: Boolean(meta.defaultForLanguage),
      catalog: withCatalogAliases(rawCatalog),
    };
  });

  records.sort((a, b) => a.order - b.order || a.label.localeCompare(b.label));
  return {
    catalogs: Object.fromEntries(records.map(({ code, catalog }) => [code, catalog])),
    locales: records.map(({ catalog, order, defaultForLanguage, ...locale }) => ({
      ...locale,
      defaultForLanguage,
    })),
  };
}

function canonicalLocale(code, locales) {
  const wanted = String(code || '').toLowerCase();
  return locales.find((locale) => locale.code.toLowerCase() === wanted)?.code || null;
}

export function detectLocale({
  saved,
  browserLanguages = [],
  locales = [],
  defaultLocale = 'en',
}) {
  const savedLocale = canonicalLocale(saved, locales);
  if (savedLocale) return savedLocale;

  for (const browserLanguage of browserLanguages) {
    const normalized = String(browserLanguage || '').toLowerCase();
    if (!normalized) continue;

    const exact = canonicalLocale(normalized, locales);
    if (exact) return exact;

    // A base pack such as `fr.json` covers browsers reporting `fr-FR`.
    const prefix = locales.find(
      (locale) => normalized.startsWith(`${locale.code.toLowerCase()}-`),
    );
    if (prefix) return prefix.code;

    // If only one regional pack exists for a language, `zh` can select
    // `zh-CN`. With several variants, `_meta.defaultForLanguage` breaks the tie.
    const base = normalized.split('-')[0];
    const regional = locales.filter(
      (locale) => locale.code.toLowerCase().split('-')[0] === base,
    );
    if (regional.length === 1) return regional[0].code;
    const preferred = regional.find((locale) => locale.defaultForLanguage);
    if (preferred) return preferred.code;
  }

  return canonicalLocale(defaultLocale, locales) || locales[0]?.code || defaultLocale;
}
