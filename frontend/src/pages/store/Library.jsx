import { useCallback, useEffect, useMemo, useState } from 'react';
import { KeyRound, RefreshCw, ShoppingBag } from 'lucide-react';
import { apiFetch, getCsrfToken } from '../../api/fetchClient.js';
import PatreonCard from '../../components/patreon/PatreonAccess.jsx';

const ENDPOINT = '/api/plugins/store/commerce';
const BTN = 'min-h-10 rounded-md border border-border px-3 py-2 text-sm font-medium hover:bg-surface-raised disabled:opacity-50';
const STATUS = { active: 'Acquired', revoked: 'Acquisition unavailable', verification_required: 'Verification needed' };

export function policyText(item) {
  if (item.updates === 'included') return 'Published updates are included.';
  if (item.updates === 'while_active') return 'New downloads and updates require an active subscription.';
  if (item.acquisition === 'through_purchase_date') return 'Includes releases published through your purchase date.';
  return 'Check the purchase terms before acquiring another release.';
}

export function LibraryContents({ library, products = [], busy, onPlan }) {
  return <div className="space-y-3">
    {(library?.items || []).map((item) => {
      const product = products.find((candidate) => candidate.id === item.plugin_id);
      const release = product?.recommended || product?.releases?.[0];
      return <article key={item.plugin_id} className="rounded-lg border border-border p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="font-medium">{release?.manifest?.name || item.plugin_id}</h3>
          <span className="text-sm text-content-muted">{STATUS[item.status] || 'Verification needed'}</span>
        </div>
        <p className="mt-2 text-sm text-content-muted">{policyText(item)}</p>
        {item.valid_until && <p className="mt-1 text-sm text-content-muted">Valid until {new Date(item.valid_until).toLocaleDateString()}.</p>}
        {onPlan && release && <button type="button" className={BTN + ' mt-3'}
          disabled={busy || item.status !== 'active' || release.compatibility_issues?.length > 0}
          onClick={() => onPlan(item.plugin_id, release.manifest.version)}>Review installation</button>}
      </article>;
    })}
    {library?.status === 'ready' && !library.items?.length &&
      <p className="text-sm text-content-muted">No acquired plugin is linked to this installation yet.</p>}
  </div>;
}

export default function Library({ catalog, refresh = 0, onChanges, onPlan, adminToken = '', initialPluginId = '' }) {
  const [library, setLibrary] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [pluginId, setPluginId] = useState(initialPluginId);
  const [licenseKey, setLicenseKey] = useState('');
  const [checkout, setCheckout] = useState(null);
  const options = useMemo(() => ({ headers: adminToken ? { 'X-LDS-Plugin-Admin': adminToken } : {} }), [adminToken]);
  const products = useMemo(() => (catalog?.products || []).filter((product) =>
    (product.recommended || product.releases?.[0])?.price?.kind === 'paid'), [catalog]);
  const selected = products.some((product) => product.id === pluginId) ? pluginId : products[0]?.id || '';

  const load = useCallback(async (signal) => {
    try {
      const value = await apiFetch(`${ENDPOINT}/library`, { ...options, signal });
      if (!signal?.aborted) { setLibrary(value); setError(''); }
    } catch (failure) {
      if (!signal?.aborted) { setLibrary({ status: 'unavailable', items: [] }); setError(failure.message || 'The library could not be refreshed.'); }
    }
  }, [options]);
  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal);
    return () => controller.abort();
  }, [load, refresh]);

  const act = async (path, payload) => {
    setBusy(true); setError(''); setCheckout(null);
    try {
      const result = await apiFetch(`${ENDPOINT}/${path}`, {
        method: 'POST', headers: { ...options.headers, 'Content-Type': 'application/json', 'X-CSRFToken': getCsrfToken() },
        body: JSON.stringify(payload),
      });
      if (result.status === 'not_configured') { setLibrary(result); return; }
      if (path === 'checkout') {
        // The backend pins this URL to the operator's configured payment origin.
        // A normal link click opens it; no request or purchase is started on mount.
        const url = new URL(result.hosted_checkout);
        if (!['https:', 'http:'].includes(url.protocol) || url.username || url.password) throw new Error('The checkout link is invalid.');
        setCheckout({ url: url.href, name: products.find((product) => product.id === selected)?.recommended?.manifest?.name || selected });
      } else {
        await load();
        onChanges?.();
      }
    } catch (failure) { setError(failure.message || 'The request could not be completed.'); }
    finally { setBusy(false); }
  };
  const activate = (event) => {
    event.preventDefault();
    const key = licenseKey.trim();
    setLicenseKey('');
    act('activate', { plugin_id: selected, license_key: key });
  };
  const unavailable = !library || ['not_configured', 'unavailable'].includes(library.status);
  const canManage = library?.can_manage !== false;
  // The license-key library exists only where an operator configured it. Without
  // it, the card would say "not connected" right under an active Patreon membership.
  const licenseLibrary = library && library.status !== 'not_configured';
  return <div className="space-y-4">
  <PatreonCard adminToken={adminToken} />
  {licenseLibrary && <section aria-label="Plugin purchase library" className="space-y-4 rounded-xl border border-border bg-surface p-5">
    <div className="flex flex-wrap items-center justify-between gap-2">
      <h2 className="text-lg font-semibold">Your plugin library</h2>
      <button type="button" className={BTN + ' inline-flex items-center gap-2'} disabled={busy} onClick={() => load()}>
        <RefreshCw className="h-4 w-4" aria-hidden="true" />Refresh library
      </button>
    </div>
    <p className="text-sm text-content-muted">Purchased plugins keep working offline. Members-only plugins require an active Patreon membership. Your installed files are kept.</p>
    {error && <p role="alert" className="text-sm text-amber-500">{error}</p>}
    {unavailable && <p role="status" className="text-sm text-content-muted">{library?.message || (!library ? 'Loading your library…' : 'The purchase service is unavailable. Existing installations are unaffected.')}</p>}
    {library?.status === 'not_connected' && <div className="space-y-2">
      <p className="text-sm text-content-muted">Connect this installation to view purchases or activate an existing license.</p>
      <button type="button" className={BTN} disabled={busy || !canManage} onClick={() => act('connect', {})}>Connect this installation</button>
    </div>}
    <LibraryContents library={library} products={catalog?.products} busy={busy} onPlan={onPlan} />
    {!unavailable && products.length > 0 && <form onSubmit={activate} className="space-y-3 border-t border-border pt-4">
      <label className="block text-sm font-medium">Plugin
        <select value={selected} onChange={(event) => { setPluginId(event.target.value); setCheckout(null); }}
          className="mt-1 block min-h-10 w-full rounded-md border border-border bg-surface px-3 py-2">
          {products.map((product) => <option key={product.id} value={product.id}>
            {(product.recommended || product.releases?.[0])?.manifest?.name || product.id}
          </option>)}
        </select>
      </label>
      <label className="block text-sm font-medium">License key
        <input type="password" autoComplete="off" value={licenseKey} maxLength={2048}
          onChange={(event) => setLicenseKey(event.target.value)}
          className="mt-1 block min-h-10 w-full rounded-md border border-border bg-surface px-3 py-2" />
      </label>
      <div className="flex flex-wrap gap-2">
        <button type="submit" disabled={busy || !selected || !licenseKey.trim() || !canManage} className={BTN + ' inline-flex items-center gap-2'}>
          <KeyRound className="h-4 w-4" aria-hidden="true" />Activate license
        </button>
        <button type="button" disabled={busy || !selected || !canManage} className={BTN + ' inline-flex items-center gap-2'}
          onClick={() => act('checkout', { plugin_id: selected })}>
          <ShoppingBag className="h-4 w-4" aria-hidden="true" />Get purchase link
        </button>
      </div>
    </form>}
    {checkout && <div className="space-y-2 rounded-lg border border-primary/40 p-4" role="status">
      <p className="text-sm">Review the price and terms for {checkout.name} on the payment page.</p>
      <a href={checkout.url} target="_blank" rel="noopener noreferrer" className={BTN + ' inline-block'}>Open secure checkout</a>
      <p className="text-xs text-content-muted">After payment, refresh your library to check acquisition. Opening checkout does not confirm a purchase.</p>
    </div>}
  </section>}
  </div>;
}
