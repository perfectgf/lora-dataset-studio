/**
 * Centralized API fetch client with global error interception via toast.
 */

let toastRef = null;

export function setToastRef(toast) {
  toastRef = toast;
}

export function getCsrfToken() {
  const match = document.cookie.match(/csrf_token=([^;]+)/);
  if (match) return decodeURIComponent(match[1]);
  return document.querySelector('meta[name="csrf-token"]')?.content || '';
}

export async function apiFetch(url, options = {}) {
  let res;
  try {
    res = await fetch(url, { credentials: 'include', ...options });
  } catch {
    toastRef?.error('Connection lost. Please check your network.');
    throw new Error('Network error');
  }

  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    let body = null;
    try {
      body = await res.json();
      msg = body.error || body.detail || body.message || msg;
    } catch {}

    if (res.status === 401) {
      toastRef?.error('Session expired. Please log in again.');
    } else if (res.status === 429) {
      toastRef?.warning('Too many requests. Please wait a moment.');
    } else if (res.status >= 500) {
      toastRef?.error('Server error. Please try again later.');
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

export function postJson(url, body) {
  return apiFetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': getCsrfToken(),
    },
    body: JSON.stringify(body),
  });
}

export function putJson(url, body) {
  return apiFetch(url, {
    method: 'PUT',
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
 * POST a FormData body to a Flask endpoint that returns JSON, and treat a
 * 400 with an HTML body (Flask-WTF's default CSRF rejection page) as a
 * recoverable CSRF mismatch: refresh the token and retry once before
 * surfacing the failure. Returns the raw Response so callers can do their
 * own status-based handling (the existing /generate / /generate_edit
 * call sites need the Response, not just JSON).
 */
export async function postFormWithCsrfRetry(url, formData, { signal } = {}) {
  const submit = async () => {
    // Rebuild csrf_token on the body each attempt — `getCsrfToken()` reads
    // the live cookie, so a refresh between attempts is automatically
    // reflected here.
    formData.set?.('csrf_token', getCsrfToken());
    return fetch(url, {
      method: 'POST',
      headers: { 'X-CSRFToken': getCsrfToken() },
      credentials: 'include',
      body: formData,
      signal,
    });
  };

  let res = await submit();
  if (res.status === 400) {
    const contentType = res.headers.get('content-type') || '';
    if (contentType.includes('text/html')) {
      await refreshCsrfToken();
      res = await submit();
    }
  }
  return res;
}
