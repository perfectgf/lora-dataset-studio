import { useCallback, useEffect, useRef, useState } from 'react';
import { Puzzle, RefreshCw, Upload } from 'lucide-react';
import { apiFetch, fetchWithCsrfRetry, postJson, postForm, del } from '../api/fetchClient';
import { useToast } from '../components/common/Toast';
import { Card, SectionHeader } from '../components/settings/primitives';
import { waitForPluginBoot } from '../plugins/lifecycle.js';
import { useSearchParams } from 'react-router';
import Catalog, { InstallPlan } from './store/Catalog.jsx';
import { useCapabilities } from '../context/CapabilitiesContext';
import Library from './store/Library.jsx';
import Administration from './store/Administration.jsx';
import { catalogView } from './store/catalogModel.js';

// Store packages bring their screens, services and help together. The running
// registry remains authoritative until a prepared transaction restarts LDS.

const BTN = 'min-h-10 lg:min-h-0 rounded-md border border-border px-3 py-1.5 text-sm font-medium transition-colors hover:bg-surface-raised disabled:opacity-50';
const BTN_PRIMARY = BTN + ' border-primary bg-primary text-primary-foreground hover:bg-primary/90';

export default function PluginsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const toast = useToast();
  const { caps, known: capsKnown, refresh: refreshCaps } = useCapabilities();
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [applying, setApplying] = useState(false);
  const [applyError, setApplyError] = useState('');
  const [applyWarning, setApplyWarning] = useState('');
  const [consent, setConsent] = useState(null);
  const { tab, filter } = catalogView(searchParams);
  const setTab = (nextTab) => setSearchParams(current => {
    const next = new URLSearchParams(current);
    next.set('tab', nextTab === 'purchases' ? 'purchases' : 'plugins');
    if (nextTab === 'installed' || nextTab === 'updates') next.set('filter', nextTab);
    return next;
  });
  const setFilter = (nextFilter) => setSearchParams(current => {
    const next = new URLSearchParams(current);
    next.set('tab', 'plugins'); next.set('filter', nextFilter); next.delete('plugin');
    return next;
  });
  const [catalog, setCatalog] = useState(null);
  const [catalogLoading, setCatalogLoading] = useState(false);
  const [storePlan, setStorePlan] = useState(null);
  const [purchaseId, setPurchaseId] = useState('');
  const [adminToken, setAdminToken] = useState('');
  const [adminDraft, setAdminDraft] = useState('');
  const fileRef = useRef(null);
  const restartAbort = useRef(null);
  const scrolledPlugin = useRef('');
  const catalogRequest = useRef(null);
  const catalogAdminToken = useRef('');
  const requestedPlugin = searchParams.get('plugin') || '';
  useEffect(() => {
    if (tab === 'plugins' && requestedPlugin && data && scrolledPlugin.current !== requestedPlugin) {
      const element = document.getElementById('plugin-row-' + requestedPlugin);
      if (element) { element.scrollIntoView({ block: 'nearest' }); scrolledPlugin.current = requestedPlugin; }
    }
  }, [tab, requestedPlugin, data, catalog]);

  const load = useCallback(async () => {
    try {
      const next = await apiFetch('/api/plugins/', { cache: 'no-store' });
      if (next.boot_id && window.lds?.bootId && next.boot_id !== window.lds.bootId) {
        window.location.reload();
        return;
      }
      setData(next);
      setError('');
    } catch (e) {
      setError(e?.message || 'Could not read the plugin list.');
    }
  }, []);

  const loadCatalog = useCallback(async () => {
    // A completed mutation may still hold the callback for an older token.
    if (adminToken !== catalogAdminToken.current) return;
    const request = {};
    catalogRequest.current = request;
    setCatalogLoading(true);
    try {
      // A catalog outage belongs to this panel, not LDS's shared connection
      // indicator. Keep the usual API error handling for every other request.
      const response = await fetchWithCsrfRetry('/api/plugins/store/catalog', { cache: 'no-store',
        headers: adminToken ? { 'X-LDS-Plugin-Admin': adminToken } : {} });
      if (!response.ok) throw new Error('Catalog request failed');
      const next = await response.json();
      if (!next || !Array.isArray(next.products) || typeof next.status !== 'string') throw new Error('Invalid catalog response');
      if (request === catalogRequest.current) setCatalog(next);
    } catch {
      if (request === catalogRequest.current) setCatalog(previous => ({ status: 'unavailable',
        products: previous?.products || [],
        message: 'The plugin catalog could not be loaded. Check your connection and retry. You can still manage your installed plugins below.' }));
    } finally {
      if (request === catalogRequest.current) setCatalogLoading(false);
    }
  }, [adminToken]);
  const unlockAdministration = (event) => {
    event.preventDefault();
    catalogAdminToken.current = adminDraft;
    catalogRequest.current = null;
    // Old permissions must not survive a change of credentials, even while the
    // new request is pending. Product descriptions remain safe to browse.
    setCatalog(previous => previous ? { ...previous, status: 'unavailable', can_manage: false } : null);
    if (adminToken === adminDraft) loadCatalog();
    else setAdminToken(adminDraft);
  };
  const adminOptions = { headers: adminToken ? { 'X-LDS-Plugin-Admin': adminToken } : {} };
  const mutation = (url, body) => postJson(url, body, adminOptions);
  const upload = (form) => postForm('/api/plugins/install', form, adminOptions);

  const planInstall = async (id, version, updateOnly = false) => {
    setBusy(true);
    try {
      setStorePlan(await mutation('/api/plugins/store/plan', {
        ...(Array.isArray(id) ? { ids: id } : { id, version }), update_only: updateOnly,
      }));
    } catch (e) {
      toast.error(e?.message || 'Could not prepare this installation.');
    } finally { setBusy(false); }
  };
  const installFromStore = async () => {
    setBusy(true);
    try {
      const selection = Array.isArray(storePlan.requested) ? { ids: storePlan.requested }
        : { id: storePlan.requested, version: storePlan.version };
      const prepared = await mutation('/api/plugins/store/install', {
        ...selection, plan_id: storePlan.plan_id, update_only: Boolean(storePlan.update_only),
      });
      const autoRestart = storePlan.update_only && prepared.restart?.can_apply;
      setStorePlan(null);
      setTab('installed');
      await load();
      if (autoRestart) {
        await apply();
        return;
      }
      await loadCatalog();
      toast.success('Plugins downloaded. Apply the changes to use them.');
    } catch (e) { toast.error(e?.message || 'Installation could not be prepared.'); }
    finally { setBusy(false); }
  };

  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    loadCatalog();
    return () => { catalogRequest.current = null; };
  }, [loadCatalog]);
  useEffect(() => () => restartAbort.current?.abort(), []);

  const apply = async () => {
    setBusy(true);
    setApplying(true);
    setApplyError('');
    setApplyWarning('');
    const controller = new AbortController();
    restartAbort.current = controller;
    try {
      const result = await mutation('/api/plugins/apply', {});
      if (!result.ok || !result.restarting) throw new Error(result.error || 'The restart could not be started.');
      setApplyWarning((result.warnings || []).join(' '));
      await waitForPluginBoot(result.boot_id || data?.boot_id, { signal: controller.signal });
      window.location.reload();
    } catch (e) {
      if (!controller.signal.aborted) setApplyError(e?.message || 'Could not restart LDS.');
    } finally {
      if (!controller.signal.aborted) {
        setBusy(false);
        setApplying(false);
      }
    }
  };

  const toggle = async (plugin, on) => {
    setBusy(true);
    try {
      await mutation(`/api/plugins/${encodeURIComponent(plugin.id)}/${on ? 'enable' : 'disable'}`, {});
      setApplyError('');
      await load();
    } catch (e) {
      toast.error(e?.message || 'Could not change the plugin.');
    } finally {
      setBusy(false);
    }
  };

  const remove = async (plugin) => {
    if (!window.confirm(`Remove ${plugin.name} after restarting? Its data will be kept and reused when you reinstall it.`)) return;
    setBusy(true);
    try {
      await del(`/api/plugins/${encodeURIComponent(plugin.id)}`, adminOptions);
      setApplyError('');
      await load();
    } catch (e) {
      toast.error(e?.message || 'Could not remove the plugin.');
    } finally {
      setBusy(false);
    }
  };

  const inspect = async (file) => {
    if (!file) return;
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await upload(fd);
      setConsent({ file, manifest: res.manifest, install_dir: res.install_dir,
        can_install: res.can_install === true,
        compatibility_issues: Array.isArray(res.compatibility_issues) ? res.compatibility_issues : [] });
    } catch (e) {
      toast.error(e?.message || 'That archive is not a plugin.');
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = '';
    }
  };

  const confirmInstall = async (replace = false) => {
    if (!consent?.can_install) return;
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append('file', consent.file);
      fd.append('confirm', '1');
      if (replace) fd.append('replace', '1');
      await upload(fd);
      toast.success(`${consent.manifest.name} is ready to apply. Restart LDS to use it.`);
      setConsent(null);
      setApplyError('');
      await load();
    } catch (e) {
      if (/already installed/i.test(e?.message || '') && !replace) {
        if (window.confirm(`${consent.manifest.name} is already installed. Replace it?`)) return confirmInstall(true);
        return;
      }
      toast.error(e?.message || 'Install failed.');
    } finally {
      setBusy(false);
    }
  };

  const plugins = data?.plugins || [];
  const restart = data?.restart;
  const pendingRestart = Boolean(data?.pending_restart);

  return (
    <div className="mx-auto w-full space-y-6" data-probe-panel="plugins" data-probe-content="plugin-store">
      <div className="space-y-2">
        <SectionHeader eyebrow="Workspace" title="Plugins"
          description="Find new tools and manage your installed plugins in one place." />
        <p className="flex flex-wrap items-center gap-2 text-xs text-content-muted">
          <Puzzle aria-hidden="true" className="h-3.5 w-3.5" />
          Your data stays with LDS when a plugin is updated or removed.
        </p>
      </div>

      {tab === 'plugins' && <div className="flex flex-wrap items-center gap-2">
        <label className={BTN + ' inline-flex cursor-pointer items-center gap-1'}>
          <Upload aria-hidden="true" className="h-3.5 w-3.5" /> Install from a ZIP
          <input ref={fileRef} type="file" accept=".ldsplugin,.zip,application/zip" className="sr-only"
            onChange={(e) => inspect(e.target.files && e.target.files[0])} disabled={busy} />
        </label>
        <span className="text-xs text-content-muted">The archive is inspected first; nothing is written until you confirm.</span>
      </div>}

      <div role="tablist" aria-label="Plugin store sections" className="flex flex-wrap gap-2 border-b border-border pb-3"
        onKeyDown={(event) => {
          const keys = ['plugins', 'purchases'];
          let next;
          if (event.key === 'ArrowRight') next = (keys.indexOf(tab) + 1) % keys.length;
          if (event.key === 'ArrowLeft') next = (keys.indexOf(tab) + keys.length - 1) % keys.length;
          if (event.key === 'Home') next = 0;
          if (event.key === 'End') next = keys.length - 1;
          if (next === undefined) return;
          event.preventDefault();
          setTab(keys[next]);
          document.getElementById(`store-tab-${keys[next]}`)?.focus();
        }}>
        {[['plugins', 'Plugins'], ['purchases', 'Purchases']].map(([key, label]) =>
          <button key={key} type="button" role="tab" tabIndex={tab === key ? 0 : -1} aria-selected={tab === key} aria-controls={`store-${key}`} id={`store-tab-${key}`}
            className={tab === key ? BTN_PRIMARY : BTN} onClick={() => setTab(key)}>{label}</button>)}
      </div>

      {catalog?.can_manage === false && <Administration value={adminDraft} onChange={setAdminDraft}
        onUnlock={unlockAdministration} checking={catalogLoading}
        rejected={Boolean(adminToken) && adminToken === adminDraft && !catalogLoading && catalog.status === 'ready'} />}

      {storePlan && <InstallPlan plan={storePlan} busy={busy} onConfirm={installFromStore} onCancel={() => setStorePlan(null)}
        restart={data?.restart} adminToken={adminToken}
        onRecheck={() => planInstall(storePlan.requested, storePlan.version, Boolean(storePlan.update_only))}
        onAcquire={id => { setPurchaseId(id); setStorePlan(null); setTab('purchases'); }} />}

      {error && <p className="rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm">{error}</p>}
      {data?.lifecycle_errors?.length > 0 && (
        <div role="alert" className="space-y-1 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm">
          <p className="font-medium">Some plugin changes could not be applied.</p>
          {data.lifecycle_errors.map((item, index) => <p key={index}>{typeof item === 'string' ? item : `${item.id}: ${item.error || item.reason}`}</p>)}
        </div>
      )}

      {(pendingRestart || applying) && restart && (
        <div className="space-y-3 rounded-md border border-primary/40 bg-primary/10 p-4 text-sm" data-probe-panel="plugin-apply">
          <div className="flex flex-wrap items-center gap-3">
            <div role="status" aria-live="polite" className="flex min-w-0 flex-1 basis-64 items-start gap-2">
              <RefreshCw aria-hidden="true" className={'mt-0.5 h-4 w-4 shrink-0' + (applying ? ' animate-spin' : '')} />
              <div>
                <p className="font-medium">{applying ? 'Restarting LDS…' : 'Changes ready to apply'}</p>
                <p className="mt-1 text-content-muted">{applying ? 'Waiting for the new server. This page will reload when it is ready.' : 'Your choices are saved. The current features stay available until LDS restarts.'}</p>
              </div>
            </div>
            {restart.can_apply && <button type="button" className={BTN_PRIMARY + ' shrink-0'} disabled={busy} onClick={apply}>Apply and restart</button>}
          </div>
          {!restart.can_apply && <p className="text-content-muted">{restart.how}</p>}
          {applyWarning && <p role="status" className="text-amber-500">{applyWarning}</p>}
          {applyError && <p role="alert" className="text-amber-500">{applyError}</p>}
        </div>
      )}

      {tab === 'plugins' && <div role="tabpanel" id="store-plugins" aria-labelledby="store-tab-plugins">
        <Catalog catalog={catalog} installed={plugins} filter={filter} onFilterChange={setFilter} busy={busy} onPlan={planInstall}
          onUpdateAll={ids => planInstall(ids, undefined, true)} pendingRestart={pendingRestart}
          loading={catalogLoading} onRetry={loadCatalog}
          onToggle={toggle} onRemove={remove} onInstalled={() => { load(); refreshCaps(true); }} caps={caps} capsKnown={capsKnown}
          onUnlock={() => document.getElementById('plugin-admin-token')?.focus()} adminToken={adminToken}
          selectedId={requestedPlugin} onClearSelection={() => {
            const next = new URLSearchParams(searchParams); next.delete('plugin'); next.set('tab', 'plugins'); next.set('filter', 'all'); setSearchParams(next, { replace: true });
          }} />
      </div>}
      {tab === 'purchases' && <div role="tabpanel" id="store-purchases" aria-labelledby="store-tab-purchases">
        <Library key={purchaseId} catalog={catalog} initialPluginId={purchaseId} adminToken={adminToken}
          onPlan={planInstall} onChanges={() => { load(); loadCatalog(); }} />
      </div>}

      {tab === 'plugins' && <div className="space-y-5">
      {data?.transactions?.length > 0 && <details className="rounded-lg border border-border p-4 text-sm"
        open={data.transactions[0].phase === 'rolled_back'}>
        <summary className="min-h-10 cursor-pointer py-2 font-medium">Installation history</summary>
        <ul className="divide-y divide-border">{data.transactions.map(item => <li key={item.id} className="py-3">
          <p className="font-medium">{item.plugins.map(plugin => `${plugin.id}${plugin.version ? ` ${plugin.version}` : ''}`).join(', ')}</p>
          <p className="text-content-muted">{item.phase === 'committed' ? 'Applied successfully.' : item.phase === 'rolled_back'
            ? 'Installation did not complete. Your previous plugins and data were restored.' : 'Waiting for the installation to finish.'}</p>
          {item.reason && <p className="mt-1 text-content-muted">{item.reason}</p>}
        </li>)}</ul>
      </details>}
      {window.lds?.loadProblems?.length > 0 && <div role="alert" className="rounded-lg border border-amber-500/40 p-4 text-sm">
        <p className="font-medium">Some plugin interfaces did not load.</p>
        {window.lds.loadProblems.map((item) => <p key={item.plugin}>{item.plugin}: {item.reason}</p>)}
        <button type="button" className={BTN + ' mt-3'} onClick={() => window.location.reload()}>Reload interfaces</button>
      </div>}
      </div>}

      {consent && (
        <Card title={`Install ${consent.manifest.name}?`} id="plugins-consent">
          <dl className="grid grid-cols-1 gap-x-4 gap-y-1 text-sm sm:grid-cols-[max-content_1fr]">
            <dt className="text-content-muted">Id</dt><dd>{consent.manifest.id} · v{consent.manifest.version}</dd>
            <dt className="text-content-muted">Author</dt><dd>{consent.manifest.author || '—'}</dd>
            <dt className="text-content-muted">License</dt><dd>{consent.manifest.license || '—'}</dd>
            <dt className="text-content-muted">Description</dt><dd>{consent.manifest.description || '—'}</dd>
            <dt className="text-content-muted">Requires</dt><dd>{(consent.manifest.requires || []).join(', ') || 'nothing'}</dd>
            <dt className="text-content-muted">Python packages</dt>
            <dd>{consent.manifest.requirements ? `a requirements file, installed into the plugin's own environment` : 'none'}</dd>
            <dt className="text-content-muted">Model files</dt>
            <dd>{consent.manifest.models?.length ? consent.manifest.models.map((m) => m.key).join(', ') : 'none'}</dd>
            <dt className="text-content-muted">ComfyUI node packs</dt>
            <dd>{consent.manifest.node_packs?.length ? consent.manifest.node_packs.map((pack) =>
              `${pack.pack} (${pack.installation ? 'installation available in plugin setup' : 'separate setup'})`).join(', ') : 'none'}</dd>
            <dt className="text-content-muted">Declares</dt><dd>{(consent.manifest.permissions || []).join(', ') || 'nothing'}</dd>
            <dt className="text-content-muted">Installs to</dt><dd className="break-all">{consent.install_dir}</dd>
          </dl>
          {!consent.can_install && <div role="alert" className="mt-3 rounded-lg border border-amber-500/40 p-3 text-sm">
            <p className="font-medium">This plugin cannot be installed in the current app configuration.</p>
            {consent.compatibility_issues.length > 0
              ? <ul className="mt-2 list-disc space-y-1 break-words pl-5">
                {consent.compatibility_issues.map((issue, index) => <li key={`${issue.field}-${index}`}>{issue.message}</li>)}
              </ul>
              : <p className="mt-2 text-content-muted">Compatibility could not be checked. Reload this page after updating LDS.</p>}
          </div>}
          <p className="mt-3 text-xs text-content-muted">
            A plugin runs inside the app with the app's own trust — it can read your datasets, your keys and use your GPU.
            Install only what you trust.
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            <button type="button" className={BTN_PRIMARY} disabled={busy || !consent.can_install} onClick={() => confirmInstall(false)}>Install</button>
            <button type="button" className={BTN} disabled={busy} onClick={() => setConsent(null)}>Cancel</button>
          </div>
        </Card>
      )}

      {(data?.misplaced?.length > 0 || data?.invalid?.length > 0 || data?.stale?.length > 0) && (
        <Card title="Needs attention" id="plugins-attention">
          <ul className="space-y-1 text-sm">
            {(data.misplaced || []).map((m) => (
              <li key={'m-' + m.dir}><span className="font-medium">{m.dir}</span>: {m.reason}</li>
            ))}
            {(data.invalid || []).map((m) => (
              <li key={'i-' + m.dir}><span className="font-medium">{m.dir}</span>: {m.reason}</li>
            ))}
            {(data.stale || []).map((id) => (
              <li key={'s-' + id}><span className="font-medium">{id}</span>: switched off in the settings, but no plugin of that name is installed.</li>
            ))}
          </ul>
        </Card>
      )}

    </div>
  );
}
