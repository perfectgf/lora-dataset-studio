/**
 * Centralized API fetch client with global error interception via toast.
 */
/* Extension spelled out: this module is imported directly by `node --test`
   (api/fetchClientBackground.test.js), and Node's ESM resolver does not guess. */
import { reportRequestFailure, reportRequestSuccess } from '../utils/connectionStatus.js';

let toastRef = null;

export function setToastRef(toast) {
  toastRef = toast;
}

export function getCsrfToken() {
  const match = document.cookie.match(/csrf_token=([^;]+)/);
  if (match) return decodeURIComponent(match[1]);
  return document.querySelector('meta[name="csrf-token"]')?.content || '';
}

/* The single human, actionable message shown when a CSRF 400 survives the
   automatic retry — never the cryptic "HTTP 400". Shared so apiFetch and
   useDataset's local postJson word it identically. */
export const CSRF_EXPIRED_MESSAGE =
  'Session token expired — refresh the page (Ctrl+Shift+R) and try again.';

/* Flask-WTF rejects a stale/missing CSRF token with a 400 whose body is an HTML
   page, NOT one of our JSON error envelopes (which are always application/json).
   That content-type mismatch is the honest, body-safe signal to refresh + retry;
   a genuine 400 from our own handlers is application/json and is left untouched. */
function isCsrfRejection(res) {
  if (res.status !== 400) return false;
  const ct = res.headers.get('content-type') || '';
  return !ct.includes('application/json');
}

// A request carries an X-CSRFToken header only when it mutates state (our
// post/put/del/postForm helpers). Only those can be rejected for a stale token
// and only those are meaningful to replay — a bare GET never enters the retry.
function csrfHeaderName(headers) {
  return Object.keys(headers || {}).find((k) => k.toLowerCase() === 'x-csrftoken');
}

// Rebuild request options with a freshly-read CSRF token: the header for JSON
// bodies, plus the csrf_token field for FormData bodies (the generation path
// sends the token both ways). FormData is mutable, so it is reused in place.
function withFreshCsrf(options) {
  const token = getCsrfToken();
  const name = csrfHeaderName(options.headers) || 'X-CSRFToken';
  if (typeof FormData !== 'undefined' && options.body instanceof FormData) {
    options.body.set?.('csrf_token', token);
  }
  return { ...options, headers: { ...(options.headers || {}), [name]: token } };
}

/**
 * fetch() with ONE automatic CSRF recovery, shared by every JSON and FormData
 * caller. When a state-changing request comes back as a CSRF rejection (see
 * isCsrfRejection — the classic "SPA left open past WTF_CSRF_TIME_LIMIT" case),
 * refresh the token and replay the request exactly once with the fresh token.
 * The backend re-plants a fresh cookie on every response (including the 400
 * itself), and the light GET below is a belt-and-suspenders for any path that
 * somehow didn't. Returns the raw Response so both parsed-JSON and raw-Response
 * callers reuse the same recovery. Network errors propagate to the caller.
 */
export async function fetchWithCsrfRetry(url, options = {}) {
  const opts = { credentials: 'include', ...options };
  let res = await fetch(url, opts);
  if (isCsrfRejection(res) && csrfHeaderName(opts.headers)) {
    await refreshCsrfToken();
    res = await fetch(url, { credentials: 'include', ...withFreshCsrf(opts) });
  }
  return res;
}

/* The two sentences the network layer is allowed to say out loud. Exported so
   the tests and the offline indicator quote the same strings. */
export const CONNECTION_LOST_MESSAGE = 'Connection lost. Please check your network.';
export const CONNECTION_BACK_MESSAGE = 'Back online.';

/**
 * @param {string} url
 * @param {RequestInit & { background?: boolean }} options
 *   `background: true` marks an AUTOMATIC, periodic request — a progress poll,
 *   a live indicator. Its failure is expected weather, not news: it updates the
 *   shared connection state (which drives the persistent "Offline —
 *   reconnecting…" indicator) and says nothing. That silence covers BOTH ways a
 *   poll fails — the request never landing, and the server answering 401/429/5xx
 *   — because a poll repeating on a timer would repeat the toast on the timer
 *   too. The rejection still reaches the caller either way.
 *
 *   Recovery is the one thing a background call is still allowed to announce:
 *   during an outage nobody is clicking, so the poll IS what notices the server
 *   came back, and reportRequestSuccess() already fires only on that one edge.
 *
 *   Everything else keeps today's behaviour, so no existing call site changes
 *   meaning by staying silent about the flag; only pollers opt in.
 */
export async function apiFetch(url, options = {}) {
  const { background = false, ...init } = options;
  let res;
  try {
    res = await fetchWithCsrfRetry(url, init);
  } catch {
    // One banner per outage, from the foreground only. Ten failed polls used to
    // mean ten stacked banners covering the app on a phone.
    if (reportRequestFailure({ background })) toastRef?.error(CONNECTION_LOST_MESSAGE);
    throw new Error('Network error');
  }

  // Any response proves the server is reachable — a 500 closes the outage just
  // as well as a 200. Announced exactly once, then quiet.
  if (reportRequestSuccess()) toastRef?.success(CONNECTION_BACK_MESSAGE);

  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    let body = null;
    let parsed = false;
    try {
      body = await res.json();
      parsed = true;
      msg = (body && (body.error || body.detail || body.message)) || msg;
    } catch {}

    if (res.status === 400 && !parsed) {
      // A 400 whose body still isn't our JSON envelope after the retry above is
      // an unrecoverable CSRF rejection — surface the actionable message, not
      // the raw "HTTP 400".
      msg = CSRF_EXPIRED_MESSAGE;
    } else if (background) {
      // The server ANSWERED — the connection store is right to call the outage
      // closed — but it answered badly, and a poll retrying every 2-3 s would
      // repeat that answer forever. A container still booting says 503 for a
      // minute; that used to be twenty "Server error" toasts. The caller still
      // gets the rejection and decides what to draw.
    } else if (res.status === 401) {
      toastRef?.error('Session expired. Please log in again.');
    } else if (res.status === 429) {
      toastRef?.warning('Too many requests. Please wait a moment.');
    } else if (res.status >= 500) {
      toastRef?.error('Server error. Please try again later.');
    }

    // A stalled ComfyUI job blocks generation app-wide, and this refusal is the
    // one moment we know the user is watching. Announce it so the shell's
    // recovery banner appears immediately, wherever the call came from —
    // otherwise the way out stays hidden until the next 20-second poll.
    if (body?.code === 'comfyui_recovery_required') {
      globalThis.dispatchEvent?.(new Event('lds:comfyui-recovery-required'));
    }

    const err = new Error(msg);
    err.status = res.status;
    // Carry the parsed error body so callers can read structured fields (e.g. a
    // 409's `studio_missing`) instead of just the flat message.
    err.body = body;
    throw err;
  }

  return res.json();
}

/* `opts` reaches apiFetch untouched — notably `{ background: true }`, which keeps
   an automatic POST from announcing a failure the user did not ask for. A POST is
   not always a user action: the threshold panel previews counts while a number is
   being typed, and without this a server that blinked would speak once per keystroke.
   It was silently dropped before (the signature took two arguments), so callers
   passing it were passing nothing. */
export function postJson(url, body, opts = {}) {
  return apiFetch(url, {
    ...opts,
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': getCsrfToken(),
    },
    body: JSON.stringify(body),
  });
}

export function putJson(url, body, opts = {}) {
  return apiFetch(url, {
    ...opts,
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': getCsrfToken(),
    },
    body: JSON.stringify(body),
  });
}

/* PATCH, for a route that edits SOME fields of a thing that already exists and is
   idempotent — a video clip's bounds being the first. PUT would claim the body is
   the whole resource, which for a clip carrying a status, a thumbnail state and a
   promotion link it very much is not. */
export function patchJson(url, body, opts = {}) {
  return apiFetch(url, {
    ...opts,
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': getCsrfToken(),
    },
    body: JSON.stringify(body),
  });
}

export function del(url) {
  return apiFetch(url, {
    method: 'DELETE',
    headers: { 'X-CSRFToken': getCsrfToken() },
  });
}

export function postForm(url, formData) {
  formData.append('csrf_token', getCsrfToken());
  return apiFetch(url, {
    method: 'POST',
    headers: { 'X-CSRFToken': getCsrfToken() },
    body: formData,
  });
}

/**
 * Refresh the CSRF token (server regenerates `session['csrf_token']` and
 * resets the matching cookie). Used as a recovery step when a POST fails
 * with a CSRF-mismatch 400 (typical after the session was regenerated
 * server-side — e.g. Flask-Login's session_protection='strong').
 */
export async function refreshCsrfToken() {
  try {
    await fetch('/api/csrf-token', { credentials: 'include' });
  } catch { /* network errors handled by caller */ }
}

/**
 * POST a FormData body to a Flask endpoint that returns JSON, with the same
 * refresh-and-retry-once CSRF recovery as every other mutating call. Returns the
 * raw Response so callers can do their own status-based handling (the /generate /
 * /generate_edit call sites need the Response, not just JSON). Thin wrapper over
 * the shared fetchWithCsrfRetry — kept as a named export for those call sites.
 */
export async function postFormWithCsrfRetry(url, formData, { signal } = {}) {
  formData.set?.('csrf_token', getCsrfToken());
  return fetchWithCsrfRetry(url, {
    method: 'POST',
    headers: { 'X-CSRFToken': getCsrfToken() },
    body: formData,
    signal,
  });
}
