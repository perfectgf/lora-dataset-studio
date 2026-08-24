// Face detection (InsightFace) as an OPTIONAL, contextually-offered extra.
// PURE JS (JSX-free) so node --test can import it.
//
// WHY THIS FILE EXISTS
// --------------------
// The capability is filed under the key `face_scoring` everywhere — the probe
// (capabilities.probe_face_scoring), the installer action, the config overrides
// (`face_scoring.python`, `face_scoring.models_root`). That key is STORED (config
// JSON on every install out there), so it is never renamed. What was wrong is
// what the user SEES: somebody ticking "Mask faces" has no reason to go hunting
// for a "face-similarity scoring" extra. So the key stays, the wording changes,
// and this module is the single place that maps one to the other.
//
// It is also deliberately a PROMPT, not a requirement. InsightFace is a few
// hundred megabytes; nothing here installs on its own, at startup or "on first
// need". The app stays fully usable without it — the option is simply disabled
// AND explained, and never nags.

/** The installer action. STORED KEY — never rename without an alias path
 *  (backend `_CAPABILITY_PACKAGES`, `_INSTALL_ALL_ORDER`, config `face_scoring.*`). */
export const FACE_DETECTION_ACTION = 'face_scoring';

/** What the user reads. Names the tool, not the first historical use of it. */
export const FACE_DETECTION_LABEL = 'Face detection (InsightFace)';

/** Announced BEFORE the click, on purpose: someone who accepts a 400 MB download
 *  knowingly does not feel trapped; someone who discovers it mid-flight does.
 *  Order of magnitude from requirements-ml.txt (insightface + onnxruntime + numpy
 *  + headless OpenCV) plus the antelopev2 model the first run fetches. */
const FACE_DETECTION_COST = '~400 MB, a few minutes';

const READY_DETAIL = 'InsightFace is installed — face masking is available.';

/** Decide what to show where face detection is USED (the Mask faces option, its
 *  preview), from the live capabilities.
 *
 *  @param capable      `caps.face_scoring` — true once the probe imports OK.
 *  @param capsLoading  capabilities still in flight -> stay quiet rather than
 *                      flashing "not installed" for a frame.
 *  @param python       `caps.python` = {version, ml_supported, ml_range}. ABSENT
 *                      or shapeless -> treated as supported: an unknown probe
 *                      must never hide the install button (designing for every
 *                      install, not for this machine).
 *  @returns {status, canInstall, action, label, headline, detail}
 *           status: 'loading' | 'ready' | 'installable' | 'unsupported_python'
 */
export function faceDetectionInstallState({ capable, capsLoading, python } = {}) {
  const base = { action: FACE_DETECTION_ACTION, label: FACE_DETECTION_LABEL,
                 canInstall: false };
  if (capsLoading) {
    return { ...base, status: 'loading', headline: '', detail: '' };
  }
  if (capable === true) {
    return { ...base, status: 'ready',
             headline: `${FACE_DETECTION_LABEL} is ready`, detail: READY_DETAIL };
  }
  // Python outside the insightface wheel range (3.10–3.12) is the single most
  // likely state of a brand-new install on a current Python. Offering a button
  // that can only end in a source build and a 200-line pip traceback would be a
  // lie — say what is wrong and what the way out is instead.
  if (python && python.ml_supported === false) {
    const version = python.version || 'this Python';
    const range = python.ml_range || '3.10–3.12';
    return {
      ...base,
      status: 'unsupported_python',
      headline: `${FACE_DETECTION_LABEL} can't be installed on Python ${version}`,
      detail: `This app runs on Python ${version}, and InsightFace publishes no wheels `
        + `outside Python ${range}, so installing it here would try to build from `
        + `source and fail. Install it into a separate `
        + `Python ${range} environment, then point face_scoring.python at that `
        + `interpreter in Settings ▸ Local tools.`,
    };
  }
  return {
    ...base,
    status: 'installable',
    canInstall: true,
    headline: `${FACE_DETECTION_LABEL} isn't installed`,
    detail: `Optional download (${FACE_DETECTION_COST}). Without it the app works `
      + 'exactly as it does now — this one option just stays off.',
  };
}
