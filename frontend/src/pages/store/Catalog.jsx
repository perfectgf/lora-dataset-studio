import { useEffect, useMemo, useRef, useState } from 'react';
import { Download, LockKeyhole, Search, ShoppingBag } from 'lucide-react';
import PluginCard from './PluginCard.jsx';
import { availableUpdateIds, canSelectEntry, catalogEntries, matchesFilter } from './catalogModel.js';
import { PatreonAccessActions, usePatreonAccess } from '../../components/patreon/PatreonAccess.jsx';
import { isPatreonRelease, membershipLabel } from '../../components/patreon/patreonModel.js';

const BTN = 'min-h-10 rounded-md border border-border px-3 py-2 text-sm font-medium hover:bg-surface-raised disabled:opacity-50';
const UPDATE_BTN = 'min-h-10 rounded-md border border-primary bg-primary px-3 py-2 text-sm font-semibold text-gray-950 hover:bg-primary-dark disabled:opacity-50';

export default function Catalog({ catalog, installed, filter = 'all', onFilterChange, busy, onPlan, onUnlock,
  selectedId = '', onClearSelection, loading = false, onRetry, onToggle, onRemove, onInstalled, caps, capsKnown,
  onUpdateAll, pendingRestart = false, adminToken = '' }) {
  // One membership read words every card's membership features.
  const patreon = usePatreonAccess(adminToken);
  const [search, setSearch] = useState('');
  const [selection, setSelection] = useState([]);
  const entries = useMemo(() => catalogEntries(catalog?.products, installed), [catalog, installed]);
  const selectable = useMemo(() => entries.filter(canSelectEntry).map(entry => entry.id), [entries]);
  const chosen = selection.filter(id => selectable.includes(id));
  const canSelect = !busy && !loading && catalog?.status === 'ready' && catalog?.can_manage;
  const updates = availableUpdateIds(entries);
  const products = entries.filter(entry => {
    if (selectedId) return entry.id === selectedId;
    if (!matchesFilter(entry, filter)) return false;
    const manifest = entry.plugin || entry.release?.manifest || {};
    return `${manifest.name || ''} ${manifest.description || ''}`.toLowerCase().includes(search.toLowerCase());
  });

  return <div className="space-y-4" data-store-catalog>
    <div role="group" aria-label="Filter plugins" className="flex flex-wrap items-center gap-2">
      {[['all', 'All'], ['installed', 'Installed'], ['updates', 'Updates']].map(([key, label]) =>
        <button key={key} type="button" aria-pressed={!selectedId && filter === key} onClick={() => onFilterChange(key)}
          className={BTN + (!selectedId && filter === key ? ' border-primary bg-primary/10 text-primary' : '')}>
          {label} <span className="ml-1 text-xs">{entries.filter(entry => matchesFilter(entry, key)).length}</span>
        </button>)}
      {updates.length > 0 && onUpdateAll && <button type="button" className={UPDATE_BTN + ' sm:ml-auto'}
        disabled={!canSelect || pendingRestart} onClick={() => onUpdateAll(updates)}>
        Update all ({updates.length})
      </button>}
    </div>
    {updates.length > 0 && onUpdateAll && <p className="text-xs text-content-muted">
      {pendingRestart ? 'Apply the pending changes before preparing more updates.'
        : 'Update all installed plugins with an available compatible release, then restart LDS once. Disabled plugins stay disabled.'}
    </p>}
    {selectedId && <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-primary/40 p-3 text-sm">
      <p>Selected plugin</p>
      <button type="button" onClick={() => { setSearch(''); onClearSelection(); }} className={BTN}>Show all plugins</button>
    </div>}
    <label className="flex min-h-11 items-center gap-2 rounded-lg border border-border bg-surface px-3">
      <Search className="h-4 w-4 text-content-muted" aria-hidden="true" />
      <input type="search" className="min-h-10 min-w-0 flex-1 bg-transparent py-2 text-sm outline-none"
        aria-label="Search plugins" placeholder="Find a plugin…" value={search} onChange={(event) => {
          setSearch(event.target.value); if (selectedId) onClearSelection();
        }} />
    </label>
    {(catalog?.status !== 'ready' || loading) && <div className="space-y-3 rounded-lg border border-border p-4 text-sm text-content-muted">
      <p role="status">{loading ? 'Connecting to the plugin catalog…' : catalog?.message || 'Connecting to the plugin catalog…'}</p>
      {catalog && onRetry && <button type="button" className={BTN} disabled={busy || loading} onClick={onRetry}>
        {loading ? 'Retrying catalog…' : 'Retry catalog'}
      </button>}
    </div>}
    {selectable.length > 0 && <div className="flex flex-wrap items-center gap-3 rounded-lg border border-border p-3" data-store-selection>
      <p className="min-w-0 flex-1 basis-60 text-sm text-content-muted">Select several plugins, review them together, then restart LDS once.</p>
      <button type="button" className={BTN} disabled={!canSelect || !chosen.length}
        onClick={() => onPlan(chosen)}>Review selected ({chosen.length})</button>
      {chosen.length > 0 && <button type="button" className={BTN} disabled={busy} onClick={() => setSelection([])}>Clear selection</button>}
    </div>}
    <div className="grid grid-cols-1 items-start gap-4 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
      {products.map(({ id, product, release, plugin }) => {
        if (!plugin && !release) return null;
        const manifest = release?.manifest;
        const incompatible = Boolean(release?.compatibility_issues?.length);
        const pending = Boolean(plugin?.pending_action);
        const paid = release?.price?.kind === 'paid';
        // Patreon products install through the ordinary review; it asks for
        // the membership when this installation does not have one.
        const patreonProduct = isPatreonRelease(release);
        const needsAdmin = catalog?.can_manage === false && !pending && !incompatible;
        const action = (reinstall = false) => <button type="button" role={reinstall ? 'menuitem' : undefined}
          className={(product?.update_available && !reinstall ? UPDATE_BTN : BTN) + ' inline-flex items-center gap-2'}
          disabled={busy || loading || pending || incompatible || catalog?.status !== 'ready' || (!catalog?.can_manage && !(needsAdmin && onUnlock))}
          onClick={() => needsAdmin ? onUnlock?.() : onPlan(id, manifest.version)}>
          {needsAdmin ? <LockKeyhole className="h-4 w-4" aria-hidden="true" />
            : paid && !plugin && !patreonProduct ? <ShoppingBag className="h-4 w-4" aria-hidden="true" /> : <Download className="h-4 w-4" aria-hidden="true" />}
          {needsAdmin ? 'Unlock installation' : pending ? 'Change pending' : reinstall ? 'Reinstall' : product?.update_available ? 'Update' : paid && !patreonProduct ? 'Review purchase' : 'Install'}
        </button>;
        return <PluginCard key={id} plugin={plugin} release={release}
          mediaKey={`${manifest?.version}:${catalog?.checked_at || ''}`} updateAvailable={product?.update_available}
          busy={busy} onToggle={onToggle} onRemove={onRemove} onInstalled={onInstalled} caps={caps} capsKnown={capsKnown}
          membersReleases={product?.members_releases} patreonAccess={patreon.access}
          onPatreonChanged={() => patreon.load()} adminToken={adminToken}
          reinstallAction={plugin && release && !product?.update_available ? action(true) : null}
          storeActions={release && (!plugin || product?.update_available) && <>
            {selectable.includes(id) && <label className="flex min-h-10 cursor-pointer items-center gap-2 text-sm">
              <input type="checkbox" checked={chosen.includes(id)} disabled={!canSelect}
                aria-label={`Select ${manifest.name} for installation`}
                onChange={event => setSelection(current => event.target.checked
                  ? [...new Set([...current, id])] : current.filter(item => item !== id))} />
              Select
            </label>}
            {action()}
          </>} />;
      })}
    </div>
    {products.length === 0 && <p className="py-6 text-sm text-content-muted">
      {selectedId ? 'This plugin is not installed or available in the current catalog.'
        : search ? 'No plugin matches your search.' : filter === 'updates' ? 'No compatible update is available in this catalog.'
          : filter === 'installed' ? 'No plugin installed yet.' : 'No plugins are available in the current catalog.'}
    </p>}
  </div>;
}

function PatreonPlanAccess({ names, adminToken, initial, busy, onRecheck }) {
  const state = usePatreonAccess(adminToken, initial);
  return <div className="space-y-2 rounded-lg border border-primary/40 p-3" role="status" data-patreon-required>
    <p className="text-sm">{names.join(', ')} {names.length === 1 ? 'is' : 'are'} available to Patreon members (LDS Member).
      Installed plugins keep working; new downloads and updates need an active membership.</p>
    <p className="text-sm text-content-muted">{state.access ? membershipLabel(state.access) : 'Checking Patreon…'}</p>
    <PatreonAccessActions state={state} compact />
    {onRecheck && <button type="button" className={BTN} disabled={busy} onClick={onRecheck}>Check again</button>}
  </div>;
}

export function InstallPlan({ plan, busy, onConfirm, onCancel, onAcquire, restart, adminToken = '', onRecheck }) {
  const panelRef = useRef(null);
  useEffect(() => {
    // The review sits above the catalog, often outside the viewport of the
    // clicked card. Move to each new plan, without stealing focus on busy updates.
    const panel = panelRef.current;
    if (!panel) return;
    panel.scrollIntoView({ block: 'center' });
    panel.focus({ preventScroll: true });
  }, [plan]);
  const purchase = plan.purchase_required?.length > 0;
  const patreonRequired = plan.patreon_required || [];
  const otherPurchase = (plan.purchase_required || []).filter(id => !patreonRequired.includes(id));
  const nameOf = id => plan.packages.find(item => item.manifest.id === id)?.manifest?.name || id;
  return <section ref={panelRef} role="region" aria-label="Review plugin installation" tabIndex={-1} className="space-y-4 rounded-xl border border-primary/40 bg-surface p-5" data-store-plan>
    <h2 className="text-lg font-semibold">{purchase ? 'Review your plugins' : plan.update_only ? 'Ready to update all' : 'Ready to install'}</h2>
    <ul className="divide-y divide-border">
      {plan.packages.map(({ manifest, action, will_enable: willEnable, previous_version: previousVersion, reason, remains_disabled: remainsDisabled }) => <li key={manifest.id} className="py-3">
        <p className="font-medium">{manifest.name} <span className="text-sm text-content-muted">{manifest.version}</span></p>
        <p className="mt-1 text-sm text-content-muted">{action === 'enable' ? 'Enable the installed plugin' : previousVersion && previousVersion !== manifest.version ? `Replace installed version ${previousVersion}` : 'Download this complete plugin'}{willEnable ? ' and activate it after restarting.' : '.'}</p>
        {reason === 'compatibility_update' && <p className="mt-1 text-sm text-content-muted">This update is needed for the selected plugins to work together. Its activation setting is kept.</p>}
        {remainsDisabled && <p className="mt-1 text-sm text-content-muted">This plugin will stay disabled.</p>}
        {manifest.permissions?.length > 0 && <p className="mt-1 text-xs text-content-muted">Access: {manifest.permissions.join(', ')}</p>}
      </li>)}
    </ul>
    <p className="text-sm text-content-muted">{plan.update_only
      ? `Your data and activation settings are kept. All updates are verified and prepared together. ${restart?.can_apply ? 'LDS will restart automatically once, then reload this page.' : restart?.how || 'Restart LDS once after preparation to apply all updates.'}`
      : 'Your data is kept. All listed plugins are prepared together and applied in one restart. The current features stay available until you apply the changes and restart LDS.'}</p>
    {plan.excluded?.length > 0 && <p role="status" className="text-sm text-content-muted" data-store-excluded>
      Not included: {plan.excluded.map(item => item.name || item.id).join(', ')}. {plan.excluded.length === 1 ? 'This update needs' : 'These updates need'} an
      active Patreon membership; the installed {plan.excluded.length === 1 ? 'version keeps' : 'versions keep'} working. Connect Patreon under Purchases to include {plan.excluded.length === 1 ? 'it' : 'them'}.
    </p>}
    {otherPurchase.length > 0 && <p role="status" className="text-sm text-content-muted">Open Purchases to connect this installation, activate a license or review the purchase terms for: {otherPurchase.join(', ')}.</p>}
    {patreonRequired.length > 0 && <PatreonPlanAccess names={patreonRequired.map(nameOf)} adminToken={adminToken}
      initial={plan.patreon_access} busy={busy} onRecheck={onRecheck} />}
    {plan.paid_packages?.length > 0 && !purchase && <p className="text-sm text-content-muted">Your acquisition is linked. Download eligibility for these versions will be verified before preparing the installation.</p>}
    <div className="flex flex-wrap gap-2">
      <button type="button" disabled={busy || purchase || !plan.packages.length} className={BTN + ' border-primary bg-primary text-primary-foreground'} onClick={onConfirm}>{busy ? 'Downloading and preparing…' : plan.update_only ? restart?.can_apply ? 'Update all and restart' : 'Update all' : 'Confirm installation'}</button>
      {otherPurchase.length > 0 && onAcquire && <button type="button" disabled={busy} className={BTN} onClick={() => onAcquire(otherPurchase[0])}>Open purchases</button>}
      <button type="button" disabled={busy} className={BTN} onClick={onCancel}>Cancel</button>
    </div>
  </section>;
}
