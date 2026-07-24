// Subject-type selector data + helpers (PURE JS, JSX-free so node --test can
// import it). The backend (face_variations.SUBJECT_TYPES) is the source of truth
// for the list and the catalogs; this mirrors the list for the UI and owns the
// per-type GROUP-HEADER relabels. The internal framing enum (face/bust/body/back)
// is shared across every subject type — composition, aspect and the stored column
// never change — only the header WORDING adapts so "Bust" never shows for a dog
// or a car.

export const SUBJECT_TYPES = ['human', 'animal', 'creature', 'object', 'other'];

export const SUBJECT_TYPE_LABELS = {
  human: 'Human', animal: 'Animal', creature: 'Creature', object: 'Object', other: 'Other',
};

// One-line hint shown under the selector, per type.
export const SUBJECT_TYPE_HINTS = {
  human: 'A person — the default. Face / bust / body / back shots.',
  animal: 'A pet or animal — head, half-body, full-body and rear shots.',
  creature: 'A fictional being or character — face, bust, full-body, rear.',
  object: 'A product or object — front, angle, detail and rear views.',
  other: 'Anything else — angles, framings and detail shots.',
};

const FRAMING_HEADERS = {
  human: { face: 'Face', bust: 'Bust', body: 'Body', back: 'Back' },
  animal: { face: 'Head', bust: 'Half-body', body: 'Full body', back: 'Rear' },
  creature: { face: 'Face', bust: 'Bust', body: 'Full body', back: 'Rear' },
  object: { face: 'Detail', bust: 'Angle', body: 'Full', back: 'Rear' },
  other: { face: 'Detail', bust: 'Medium', body: 'Full', back: 'Rear' },
};

export function normalizeSubjectType(v) {
  return SUBJECT_TYPES.includes(v) ? v : 'human';
}

export function framingHeaders(subjectType) {
  return FRAMING_HEADERS[normalizeSubjectType(subjectType)];
}

export function framingLabel(subjectType, framing) {
  return framingHeaders(subjectType)[framing] || framing;
}

// The preset to auto-select when the catalog for a subject type loads: human keeps
// 'balanced_25' (or 'body_emphasis' for a body-fidelity dataset); every non-human
// type ships a single balanced preset, so pick the first available key.
export function defaultPresetKey(presets, subjectType, { bodyFidelity = false } = {}) {
  const keys = Object.keys(presets || {});
  if (normalizeSubjectType(subjectType) === 'human') {
    if (bodyFidelity && keys.includes('body_emphasis')) return 'body_emphasis';
    return keys.includes('balanced_25') ? 'balanced_25' : (keys[0] || null);
  }
  return keys[0] || null;
}
