import { useCallback, useEffect, useMemo, useState } from 'react';
import { ExternalLink, RefreshCw } from 'lucide-react';
import { apiFetch, postJson } from '../../api/fetchClient.js';
import {
  ACCESS_NOTICE_DISMISSED_KEY, ADMIN_HINT, CODE_PROMPT, JUST_JOINED_HINT, PATREON_ACCESS_SUMMARY,
  membershipLabel, patreonActions, safeAuthorizeUrl, safeJoinUrl, safeRequestUrl,
} from './patreonModel.js';

const ENDPOINT = '/api/plugins/store/patreon';
const BTN = 'min-h-10 rounded-md border border-border px-3 py-2 text-sm font-medium hover:bg-surface-raised disabled:opacity-50';
const LINK = BTN + ' inline-flex items-center gap-2 no-underline';
const PRIMARY = 'min-h-10 inline-flex items-center gap-2 rounded-md border border-primary bg-primary px-3 py-2 text-sm font-semibold text-primary-foreground no-underline hover:bg-primary/90';

/* Membership state and the two Patreon mutations. `adminToken` is the optional
   remote plugin-admin token (the same one the Store uses). */
export function usePatreonAccess(adminToken = '', initial = null) {
  const headers = useMemo(() => (adminToken ? { 'X-LDS-Plugin-Admin': adminToken } : {}), [adminToken]);
  const [access, setAccess] = useState(initial);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [authorizeUrl, setAuthorizeUrl] = useState('');

  const load = useCallback(async (signal) => {
    try {
      const value = await apiFetch(ENDPOINT, { headers, signal, cache: 'no-store', background: true });
      if (!signal?.aborted) {
        setAccess(value); setError('');
        // A member needs neither link of the Dataset page announcement.
        if (value?.patreon?.status === 'active') {
          try { localStorage.setItem(ACCESS_NOTICE_DISMISSED_KEY, '1'); }
          catch { /* It stays dismissible by hand. */ }
        }
      }
    } catch (failure) {
      if (signal?.aborted || failure?.name === 'AbortError') return;
      setAccess({ status: failure?.status === 403 ? 'not_permitted' : 'unavailable', patreon: null });
      setError('');
    }
  }, [headers]);

  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal);
    return () => controller.abort();
  }, [load]);

  const post = async (path, body = {}) => {
    setBusy(true); setError('');
    try {
      return await postJson(`${ENDPOINT}/${path}`, body, { headers });
    } catch (failure) {
      setError(failure?.message || 'Patreon could not be reached. Installed plugins and this version of LDS keep working.');
      return null;
    } finally { setBusy(false); }
  };

  const connect = async () => {
    setAuthorizeUrl('');
    const result = await post('link');
    if (!result) return;
    const url = safeAuthorizeUrl(result.authorize_url);
    if (!url) setError('The Patreon authorization link is invalid. Nothing was changed.');
    setAuthorizeUrl(url);
  };

  const disconnect = async () => {
    if (!window.confirm('Disconnect Patreon from this installation? Installed plugins keep working; new downloads and updates will need Patreon again.')) return;
    if (await post('unlink')) { setAuthorizeUrl(''); await load(); }
  };

  // The server names each refusal (wrong or expired code, too many attempts…).
  const confirm = async (code) => {
    if (!(await post('confirm', { code }))) return false;
    setAuthorizeUrl('');
    await load();
    return true;
  };

  return { access, busy, error, authorizeUrl, load, connect, disconnect, confirm };
}

/* Approving on Patreon links nothing by itself (protocol v1.1): the Patreon
   page shows a code, typed here, so only this installation is linked. */
function ConfirmCode({ busy, disabled, onConfirm }) {
  const [code, setCode] = useState('');
  const submit = async (event) => {
    event.preventDefault();
    if (await onConfirm?.(code)) setCode('');
  };
  return <form className="flex flex-wrap items-end gap-2" onSubmit={submit} data-patreon-code>
    <label className="flex min-w-0 flex-col gap-1 text-sm">
      <span>{CODE_PROMPT}</span>
      <input type="text" value={code} onChange={(event) => setCode(event.target.value)} maxLength={32}
        autoComplete="off" autoCapitalize="characters" spellCheck={false} placeholder="XXXX-XXXX"
        className="min-h-10 w-44 rounded-md border border-border bg-surface px-3 py-2 font-mono uppercase tracking-widest" />
    </label>
    <button type="submit" className={BTN} disabled={busy || disabled || !code.trim()}>
      {busy ? 'Confirming…' : 'Confirm'}
    </button>
  </form>;
}

/* Connect / Join / Disconnect / Request a custom plugin (only for a `pro` tier,
   so never without an operator-configured one), plus the one-time
   "Continue on Patreon" link and the code it leads to. No request leaves the
   app on mount, and opening Patreon is always a plain link the user clicks. */
export function PatreonAccessActions({ state, compact = false }) {
  const { access, busy, error, authorizeUrl, load, connect, disconnect, confirm } = state;
  const actions = patreonActions(access);
  const patreon = access?.patreon;
  const canManage = access?.can_manage !== false;
  const askCode = Boolean(authorizeUrl) || actions.confirm;
  return <div className="space-y-2" data-patreon-actions>
    <div className="flex flex-wrap items-center gap-2">
      {actions.connect && <button type="button" className={BTN} disabled={busy || !canManage} onClick={connect}>
        {busy ? 'Connecting…' : actions.reconnect ? 'Connect Patreon again' : 'Connect Patreon'}
      </button>}
      {actions.join && <a href={safeJoinUrl(patreon.join_url)} target="_blank" rel="noopener noreferrer" className={LINK}>
        Join on Patreon<ExternalLink className="h-4 w-4" aria-hidden="true" />
      </a>}
      {actions.request && <a href={safeRequestUrl(patreon.request_url)} target="_blank" rel="noopener noreferrer" className={LINK}>
        Request a custom plugin<ExternalLink className="h-4 w-4" aria-hidden="true" />
      </a>}
      <button type="button" className={BTN + ' inline-flex items-center gap-2'} disabled={busy} onClick={() => load()}>
        <RefreshCw className="h-4 w-4" aria-hidden="true" />Refresh
      </button>
      {!compact && actions.disconnect && <button type="button" className={BTN} disabled={busy || !canManage} onClick={disconnect}>
        Disconnect
      </button>}
    </div>
    {actions.join && !authorizeUrl && <p className="text-xs text-content-muted">{JUST_JOINED_HINT}</p>}
    {authorizeUrl && <div className="space-y-1 rounded-lg border border-primary/40 p-3" role="status">
      <a href={authorizeUrl} target="_blank" rel="noopener noreferrer" className={PRIMARY}>
        Continue on Patreon<ExternalLink className="h-4 w-4" aria-hidden="true" />
      </a>
      <p className="text-xs text-content-muted">Approve LDS on Patreon, then type the code shown there below. This link expires after a few minutes.</p>
    </div>}
    {askCode && <ConfirmCode busy={busy} disabled={!canManage} onConfirm={confirm} />}
    {!canManage && (actions.connect || askCode) && <p className="text-xs text-content-muted">{ADMIN_HINT}</p>}
    {error && <p role="alert" className="text-sm text-amber-500">{error}</p>}
  </div>;
}

/* Plugins → Purchases: the membership card. */
export default function PatreonCard({ adminToken = '' }) {
  const state = usePatreonAccess(adminToken);
  const { access } = state;
  const patreon = access?.patreon;
  if (access?.status === 'not_configured') return null;
  return <section aria-label="Patreon membership" className="space-y-3 rounded-xl border border-border bg-surface p-5" data-patreon-card>
    <h2 className="text-lg font-semibold">Patreon membership</h2>
    <p className="text-sm text-content-muted">{PATREON_ACCESS_SUMMARY}</p>
    <p className="text-sm font-medium" role="status">{access ? membershipLabel(access) : 'Checking Patreon…'}</p>
    {patreon?.status === 'active' && patreon.valid_until && <p className="text-sm text-content-muted">
      Next renewal on Patreon: {new Date(patreon.valid_until).toLocaleDateString()} (as reported by Patreon).
    </p>}
    {access && <PatreonAccessActions state={state} />}
  </section>;
}
