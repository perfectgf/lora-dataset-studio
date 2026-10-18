import { MoreHorizontal, Trash2 } from 'lucide-react';
import { Link } from 'react-router';
import { pluginWhatsNew } from '../../plugins/registry.js';
import { sortedEntries } from '../../whatsNew.js';
import InstallRunner from '../../components/setup/InstallRunner';
import HeaderMenu from '../../components/common/HeaderMenu';
import { pendingLabel, pluginActive, pluginDesired } from '../../plugins/lifecycle.js';
import { productReadiness } from '../../plugins/readiness.js';
import { pluginSettingsPath } from '../pluginSettings.js';
import PluginAvatar from './PluginAvatar.jsx';
import Presentation, { presentationImages } from './Presentation.jsx';
import { isPatreonRelease, membershipFeatures } from '../../components/patreon/patreonModel.js';
import { PatreonJoin } from '../../components/patreon/PatreonUpdates.jsx';

const BTN = 'min-h-10 rounded-md border border-border px-3 py-1.5 text-sm font-medium transition-colors hover:bg-surface-raised disabled:opacity-50';

const STATE_LABEL = {
  loaded: 'Active now', disabled: 'Inactive', pending: 'Not active yet', error: 'Failed to load', incompatible: 'Incompatible', misplaced: 'Misplaced',
};

function stateClass(state) {
  if (state === 'loaded') return 'border-emerald-400/25 bg-emerald-400/10 text-emerald-300';
  if (state === 'disabled') return 'border-border bg-surface text-content-muted';
  return 'border-amber-400/25 bg-amber-400/10 text-amber-300';
}

/* What Patreon members get from this plugin and free users do not: what this
   installation has, or what it would get, next to Connect / Join. */
function MembershipFeatures({ name, membership, access, adminToken, onChanged }) {
  const { whole, version, items, owned, needsCore } = membership;
  const active = access?.patreon?.status === 'active';
  const lead = whole
    ? owned ? 'Included with your Patreon membership.'
      : `This plugin is available to Patreon members (LDS Member)${needsCore ? ', with the members version of LDS' : ''}.`
    : owned ? `Version ${version} is included with your members version of LDS.`
      : `Version ${version} is available to Patreon members (LDS Member) first, with the members version of LDS.`;
  return <section aria-label={`Membership features of ${name}`} className="space-y-2 rounded-lg border border-primary/40 p-3 text-sm" data-membership-features>
    <p className="font-medium text-primary">Membership features</p>
    <p className="text-content-muted">{lead}</p>
    {items.length > 0 && <ul className="list-disc space-y-1 pl-5">{items.map((item, index) => <li key={index}>{item}</li>)}</ul>}
    {active && needsCore && <p className="text-xs text-content-muted">Install the members version of LDS from Settings › Maintenance › Updates.</p>}
    {access && !owned && !active && <PatreonJoin access={access} onChanged={onChanged} adminToken={adminToken} />}
  </section>;
}

export default function PluginCard({ plugin, release, mediaKey, updateAvailable, storeActions, reinstallAction,
  onToggle, onRemove, onInstalled, busy, caps, capsKnown = false, membersReleases, patreonAccess = null,
  onPatreonChanged, adminToken = '' }) {
  const installed = plugin;
  plugin = plugin || release.manifest;
  const requires = plugin.requires || [];
  const news = installed ? sortedEntries(pluginWhatsNew(plugin.id)) : [];
  const desired = installed && pluginDesired(plugin);
  const pending = installed && pendingLabel(plugin);
  const removing = plugin.pending_action === 'remove';
  const packagePending = ['install', 'update', 'remove'].includes(plugin.pending_action);
  const readiness = installed && pluginActive(plugin) && capsKnown ? productReadiness(plugin.id, caps) : [];
  const settingsPath = pluginSettingsPath(plugin.id);
  const experience = installed && pluginActive(plugin) ? plugin.package_contract?.experience : null;
  const details = installed ? plugin.package_contract?.experience : plugin.experience;
  const patreonProduct = isPatreonRelease(release);
  const membership = membershipFeatures({ release, membersReleases, access: patreonAccess });
  return (
    <article data-store-product={plugin.id} id={'plugin-row-' + plugin.id} data-plugin-id={installed ? plugin.id : undefined} aria-labelledby={'plugin-title-' + plugin.id}
      className="flex min-w-0 scroll-mt-6 flex-col rounded-xl border border-border-strong bg-surface shadow-sm">
      <header className="flex flex-wrap items-start justify-between gap-4 rounded-t-xl border-b border-border bg-surface-raised px-4 py-4 sm:px-5">
        <div className="flex min-w-0 flex-1 basis-48 items-start gap-3">
          <PluginAvatar id={plugin.id} name={plugin.name} />
          <div className="min-w-0">
            <h3 id={'plugin-title-' + plugin.id} className="break-words text-lg font-semibold leading-snug text-content">{plugin.name}</h3>
            <p className="mt-1 text-xs text-content-muted">{installed ? 'Installed' : 'Version'} {plugin.installed_version || plugin.version} · {plugin.official ? 'LDS' : plugin.bundled ? 'Included' : plugin.publisher?.name || plugin.author || 'External'}</p>
            {patreonProduct && <span className="mt-1.5 inline-flex items-center rounded-full border border-primary/30 bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary" data-patreon-badge>
              For Patreon members
            </span>}
          </div>
        </div>
        <span className={'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ' + stateClass(installed && pluginActive(plugin) ? 'loaded' : plugin.state || 'disabled')}>
          <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-current" />
          {installed ? pluginActive(plugin) ? 'Active now' : STATE_LABEL[plugin.state] || plugin.state : patreonProduct ? 'Patreon' : release.price?.kind === 'paid' ? 'Paid plugin' : 'Free'}
        </span>
      </header>
      <div className="flex flex-1 flex-col gap-3 px-4 py-4 sm:px-5">
        {updateAvailable && release && <div className="rounded-lg border border-primary/40 bg-primary/10 p-3">
          <p className="text-sm font-medium text-primary">Update available · {release.manifest.version}</p>
          <div className="mt-2 flex flex-wrap items-center justify-between gap-2">{storeActions}</div>
        </div>}
        {/* The Patreon catalog serves no presentation images: no "coming soon" there. */}
        {release && !(patreonProduct && !presentationImages(release).length) && <Presentation key={mediaKey} release={release} />}
        {pending && <p className="rounded-md border border-primary/30 bg-primary/10 px-3 py-2 text-sm text-content" data-plugin-pending>{pending}</p>}
        {release?.compatibility_issues?.length > 0 && <p className="text-sm text-amber-500">{release.compatibility_issues[0].message}</p>}
        {plugin.description && <p className="text-sm leading-relaxed text-content-muted">{plugin.description}</p>}
        {membership && <MembershipFeatures name={plugin.name} membership={membership} access={patreonAccess}
          adminToken={adminToken} onChanged={onPatreonChanged} />}
        <div className="flex flex-wrap items-center gap-2">
          {experience?.entrypoints?.length > 0 && <div className="flex flex-wrap gap-2">
            {experience.entrypoints.map((entry, index) => <Link key={entry.path} to={entry.path}
              className={index === 0 ? BTN + ' border-primary/40 text-primary hover:bg-primary/10' : BTN}>{entry.label}</Link>)}
          </div>}
          {installed && <div className="flex flex-wrap items-center gap-2">
            <Link to={settingsPath} className={BTN} aria-label={`Settings for ${plugin.name}`}>Settings</Link>
            {experience?.setup && experience.setup.path !== settingsPath && <Link to={experience.setup.path} className={BTN}>
              {experience.setup.label}
            </Link>}
            {plugin.state !== 'misplaced' && plugin.state !== 'incompatible' && (
              <button type="button" className={BTN} disabled={busy || removing}
                onClick={() => onToggle(plugin, !desired)} aria-pressed={desired}
                aria-label={`${plugin.pending_action === 'enable' || plugin.pending_action === 'disable' ? 'Undo change for' : desired ? 'Turn off' : 'Turn on'} ${plugin.name}`}>
                {plugin.pending_action === 'enable' || plugin.pending_action === 'disable' ? 'Undo change' : desired ? 'Turn off' : 'Turn on'}
              </button>
            )}
            {(!plugin.bundled || reinstallAction) && <HeaderMenu triggerLabel={<MoreHorizontal aria-hidden="true" className="h-7 w-5" />} triggerTitle={`More actions for ${plugin.name}`}>
              {(close) => <>
                {reinstallAction && <div onClick={close}>{reinstallAction}</div>}
                {!plugin.bundled && <button type="button" role="menuitem" className={BTN + ' flex items-center gap-2 text-left hover:text-rose-300'}
                  disabled={busy || packagePending} aria-label={`Remove ${plugin.name}`}
                  title={packagePending ? 'Apply the pending package change before removing this plugin' : 'Remove this plugin (its data is kept for reinstalling)'}
                  onClick={() => { close(); onRemove(plugin); }}><Trash2 aria-hidden="true" className="h-4 w-4" />Remove plugin</button>}
              </>}
            </HeaderMenu>}
          </div>}
          {!(updateAvailable && release) && storeActions}
        </div>
        {plugin.error && <p className="mt-1 text-xs text-amber-500">{plugin.error}</p>}
        {installed && pluginActive(plugin) && !capsKnown && <p className="text-xs text-content-muted">Checking configured components…</p>}
        {readiness.length > 0 && <details className="mt-2 text-sm" data-plugin-readiness>
          <summary className="min-h-10 cursor-pointer py-2 font-medium">Configured components · {readiness.filter(row => row.state === 'ready').length}/{readiness.length} ready</summary>
          <ul className="space-y-2 rounded-md border border-border p-3">
            {readiness.map((row, index) => <li key={index}>
              <span className="font-medium">{row.label}</span>{' · '}
              <span className={row.state === 'ready' ? 'text-emerald-500' : 'text-content-muted'}>
                {{ ready: 'Ready', pending: 'Waiting for the local service', setup: 'Needs setup', unknown: 'Check unavailable' }[row.state]}
              </span>
              {(row.note || row.what) && <p className="mt-1 text-xs text-content-muted">{row.note || row.what}</p>}
            </li>)}
          </ul>
        </details>}
        {plugin.disabled_by && plugin.disabled_by.length > 0 && (
          <p className="mt-1 text-xs text-content-muted">Off because it needs: {plugin.disabled_by.join(', ')}</p>
        )}
        {(details?.setup_hint || requires.length > 0 || plugin.permissions?.length > 0 || release) && <details className="text-sm">
          <summary className="min-h-10 cursor-pointer py-2 font-medium">Details and requirements</summary>
          <div className="space-y-2 rounded-lg border border-border p-3 text-content-muted">
            {details?.setup_hint && <p>{details.setup_hint}</p>}
            {details?.surfaces?.length > 0 && <p>Included in: {details.surfaces.join(', ')}.</p>}
            {requires.length > 0 && <p>Requires: {requires.join(', ')}</p>}
            {plugin.permissions?.length > 0 && <p className="break-words text-xs">Access: {plugin.permissions.join(', ')}</p>}
            {(plugin.models?.length > 0 || plugin.node_packs?.length > 0 || plugin.requirements) &&
              <p>Some functions need additional models or tools. Configure these from the plugin settings after installation.</p>}
            {plugin.license && <p>License: {plugin.license}</p>}
            {release?.changelog?.length > 0 && <ul className="list-disc space-y-1 pl-5">{release.changelog.map((entry, i) => <li key={i}>{entry}</li>)}</ul>}
          </div>
        </details>}
        {plugin.environment && (
          <div className="mt-3 space-y-2 rounded-md border border-border p-3 [&_button]:min-h-10 lg:[&_button]:min-h-0">
            <p className="text-sm font-medium">Python environment · {plugin.environment.ready ? 'Ready' : 'Needs installation'}</p>
            <p className="text-xs text-content-muted">Installs this plugin’s CPU dependencies in its own environment. Installation logs appear here; keep LDS open until it finishes.</p>
            {plugin.environment.reason && <p className="text-xs text-content-muted">{plugin.environment.reason}</p>}
            {plugin.environment.can_install && (
              <InstallRunner action={plugin.environment.action}
                buttonLabel={plugin.environment.ready ? 'Repair Python environment' : 'Install Python environment'}
                onDone={onInstalled} />
            )}
          </div>
        )}
        {news.length > 0 && (
          <details className="mt-2" data-probe-reading>
            <summary className="min-h-10 cursor-pointer py-2 text-sm font-medium lg:min-h-0">
              What’s new in {plugin.name}
            </summary>
            <ol className="max-h-96 space-y-4 overflow-y-auto rounded-md border border-border p-3">
              {news.map((entry) => (
                <li key={entry.id} className="break-words text-sm">
                  <time className="text-xs text-content-muted" dateTime={entry.date}>{entry.date}</time>
                  <p className="font-medium">{entry.title}</p>
                  <p className="mt-1 text-content-muted">{entry.blurb}</p>
                </li>
              ))}
            </ol>
          </details>
        )}
      </div>
    </article>
  );
}
