/**
 * Pure helpers for the Klein generation-LoRA picker combobox.
 *
 * The backend (GET /api/loras/list) returns the LoRAs actually on
 * disk as [{ name, arch, label, compatible }], already sorted Klein-compatible
 * first. These helpers turn that list into what the combobox renders — grouped,
 * filtered, and reconciled against a preset row's stored value — with zero React
 * so they can be unit-tested under `node --test` (the repo's frontend test style).
 */

/**
 * Separator- and case-insensitive key for comparing a stored preset value to a
 * scanned name. ComfyUI's relative names use the OS separator (backslash on
 * Windows), while a hand-typed value or a legacy config may use a forward slash;
 * the resolver treats them alike, so the picker's "is this on disk?" check must
 * too. Trims surrounding whitespace; collapses every `\` to `/`.
 */
export function normalizeLoraName(name) {
  return String(name ?? '').trim().replace(/\\/g, '/').toLowerCase();
}

/**
 * The scanned entry whose name matches `value` (separator/case-insensitive), or
 * null. Blank value → null. Used to badge the current selection and to decide
 * whether a saved preset value is "not found" on disk.
 */
export function findLora(value, loras) {
  const key = normalizeLoraName(value);
  if (!key) return null;
  return (loras || []).find((e) => normalizeLoraName(e.name) === key) || null;
}

/** Whether `value` names a LoRA present in the scan (so it will resolve). */
export function isKnownLora(value, loras) {
  return findLora(value, loras) != null;
}

/**
 * Split the scan into the two rendered groups, preserving the backend's order
 * (compatible-first, then alphabetical) within each:
 *   - `compatible`: Klein-compatible (flux2klein, or FLUX.1 sharing the namespace)
 *   - `other`: a different arch (SDXL/Krea/Z-Image — a silent no-op in the Klein
 *     graph) or an undetectable header.
 */
export function groupLoras(loras) {
  const compatible = [];
  const other = [];
  for (const e of loras || []) {
    (e.compatible === 'yes' ? compatible : other).push(e);
  }
  return { compatible, other };
}

/**
 * Case-insensitive SUBSTRING filter over the LoRA name (subfolders included) and
 * its arch label — a `<datalist>` would only match a prefix, so typing "klein"
 * must still find "lora1_klein1.2_detail.safetensors" (vvilams). An empty/blank
 * query returns the list unchanged.
 */
export function filterLoras(loras, query) {
  const q = String(query ?? '').trim().toLowerCase();
  if (!q) return loras || [];
  return (loras || []).filter((e) => {
    const hay = `${e.name || ''} ${e.label || ''}`.toLowerCase();
    return hay.includes(q);
  });
}

// Cap on rendered options — the dropdown never floods the viewport, and the
// keyboard-navigation index stays bounded (vvilams: ≤20 shown). Filtering narrows
// further; a footer tells the user to refine when matches exceed the cap.
export const MAX_VISIBLE_OPTIONS = 20;

/**
 * The options actually rendered for a query: filtered, capped to `max`, then split
 * into the two groups. `options` is the FLAT list in exact render order (all
 * compatible, then all other) so a keyboard highlight index maps 1:1 to what's on
 * screen; `hiddenCount` is how many matches the cap dropped. The backend already
 * sorts compatible-first, so the cap keeps Klein-compatible LoRAs preferentially.
 */
export function buildVisibleOptions(loras, query, max = MAX_VISIBLE_OPTIONS) {
  const filtered = filterLoras(loras, query);
  const shown = filtered.slice(0, Math.max(0, max));
  const { compatible, other } = groupLoras(shown);
  return {
    options: [...compatible, ...other],
    compatible,
    other,
    hiddenCount: Math.max(0, filtered.length - shown.length),
  };
}

/**
 * Semantic badge descriptor for a compatibility verdict — { tone, text, title } —
 * mapped to concrete colors by the component. `label` is the arch name when known
 * (e.g. "SDXL"), used in the incompatible/other wording. Pure presentation logic,
 * no arch DECISION here (the backend already decided via lora_arch_conflicts).
 */
export function compatBadge(compatible, label) {
  if (compatible === 'yes') {
    return { tone: 'compatible', text: label || 'Klein',
      title: `${label || 'Klein'} — compatible with the Klein graph` };
  }
  if (compatible === 'no') {
    const arch = label || 'Other arch';
    return { tone: 'incompatible', text: arch,
      title: `${arch} LoRA — a different architecture; ComfyUI would load it as a no-op in the Klein graph` };
  }
  return { tone: 'unknown', text: label || 'Unknown arch',
    title: "Architecture couldn't be read from the file header — use only if you know it's Klein-compatible" };
}
