import { useEffect, useId, useRef, useState } from 'react';
import { apiFetch } from '../../api/fetchClient';
import { HelpBadge } from '../../help/HelpMode';
import InstallRunner from './InstallRunner';
import SettingsLink from '../common/SettingsLink';

const UPDATE_GUIDE = 'https://docs.comfy.org/installation/update_comfyui';
const PORTABLE_RELEASES = 'https://github.com/Comfy-Org/ComfyUI/releases/latest';
const MODEL_PATHS = 'https://docs.comfy.org/installation/comfyui_portable_windows#adding-extra-model-paths';
const buttonClass = 'min-h-10 rounded-md border border-border px-3 py-1.5 text-xs text-content hover:bg-surface-raised disabled:opacity-50';
const linkClass = 'break-words text-xs text-content underline';

export function safeComfyLink(value) {
  if (typeof value !== 'string' || !/^https?:\/\//i.test(value.trim())
      || /[\u0000-\u001f\u007f]/.test(value)) return '';
  try {
    const url = new URL(value);
    return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password
      ? url.href : '';
  } catch {
    return '';
  }
}

/** Guidance and a fresh node probe; every install/restart remains user-operated. */
export default function ComfyNodeRepair({ nodes = [], nodePacks = [], onRefresh }) {
  const [expanded, setExpanded] = useState(false);
  const [method, setMethod] = useState('portable');
  const [comfyUrl, setComfyUrl] = useState('');
  const [addressError, setAddressError] = useState('');
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const checkRequest = useRef(null);
  const helpId = useId();
  const names = [...new Set((Array.isArray(nodes) ? nodes : [])
    .filter((name) => typeof name === 'string' && name))];
  const namesKey = JSON.stringify(names);
  const packs = Array.isArray(nodePacks) ? nodePacks : [];
  const custom = names.map((name) => packs.find((pack) => pack.class_type === name && !pack.core))
    .filter(Boolean);
  const installGroups = custom.filter((pack, index) => !pack.install_action
    || custom.findIndex((other) => other.install_action === pack.install_action) === index);
  const hasCore = names.some((name) => !custom.some((pack) => pack.class_type === name));
  const currentResult = result?.key === namesKey ? result : null;
  const resolved = currentResult?.missing.length === 0;

  useEffect(() => () => checkRequest.current?.abort(), []);

  useEffect(() => {
    if (!expanded) return undefined;
    const controller = new AbortController();
    setComfyUrl('');
    setAddressError('');
    apiFetch('/api/settings', { signal: controller.signal }).then((data) => {
      if (controller.signal.aborted) return;
      const url = safeComfyLink(data.config?.comfyui?.api_url);
      if (url) setComfyUrl(url);
      else setAddressError('Set a valid HTTP or HTTPS ComfyUI address without embedded credentials in Settings.');
    }).catch((e) => {
      if (!controller.signal.aborted) setAddressError(e.message || 'Could not read the configured ComfyUI address.');
    });
    return () => controller.abort();
  }, [expanded]);

  if (!names.length) return null;

  const recheck = async () => {
    if (checkRequest.current) return;
    const controller = new AbortController();
    checkRequest.current = controller;
    setChecking(true);
    setError('');
    setResult(null);
    try {
      const query = new URLSearchParams();
      names.forEach((name) => query.append('nodes', name));
      const data = await apiFetch(`/api/comfy/node-check?${query}`, { signal: controller.signal });
      if (controller.signal.aborted) return;
      if (data.nodes_checked !== true || !Array.isArray(data.missing_nodes)) {
        throw new Error('ComfyUI could not verify these nodes. Start it and check again.');
      }
      setResult({ key: namesKey, missing: data.missing_nodes });
      if (!data.missing_nodes.length) await onRefresh?.();
    } catch (e) {
      if (!controller.signal.aborted) setError(e.message || 'Could not check ComfyUI. Start it and try again.');
    } finally {
      if (!controller.signal.aborted) setChecking(false);
      if (checkRequest.current === controller) checkRequest.current = null;
    }
  };

  return (
    <section aria-label="ComfyUI node repair"
      className="min-w-0 space-y-3 rounded-lg border border-amber-400/40 bg-amber-400/5 p-3 text-xs text-content-muted">
      <div className="space-y-1">
        <p className="font-semibold text-content">
          {resolved ? 'Required ComfyUI nodes are available' : 'Missing ComfyUI nodes'}
          <HelpBadge topic="studio_models" />
        </p>
        <ul className="space-y-1">
          {names.map((name) => <li key={name}><code className="break-all text-content">{name}</code></li>)}
        </ul>
      </div>
      <div className="flex flex-wrap gap-2">
        <button type="button" className={buttonClass} aria-expanded={expanded} aria-controls={helpId}
          onClick={() => setExpanded((value) => !value)}>Fix missing ComfyUI nodes</button>
        <button type="button" className={buttonClass} disabled={checking} onClick={recheck}>
          {checking ? 'Checking ComfyUI…' : 'Check nodes again'}
        </button>
      </div>
      <div aria-live="polite" className="space-y-1">
        {resolved && <p>All listed nodes were detected. Return to the test setup to continue.</p>}
        {currentResult?.missing.length > 0 && (
          <p className="break-words">Still missing: <span className="break-all">{currentResult.missing.join(', ')}</span>.
            Follow the steps below, restart ComfyUI, then check again.</p>
        )}
      </div>
      {error && <p role="alert" className="break-words text-red-400">{error}</p>}
      {expanded && (
        <div id={helpId} className="min-w-0 space-y-3 border-t border-border pt-3">
          {comfyUrl && <a href={comfyUrl} target="_blank" rel="noopener noreferrer" className={linkClass}>Open ComfyUI</a>}
          {addressError && <p role="alert" className="break-words text-red-400">
            {addressError}{' '}
            <SettingsLink section="local-tools" focus="comfyui-api-url" tone="warning">Check ComfyUI address</SettingsLink>
          </p>}
          <p>Let running and queued jobs finish before closing ComfyUI.</p>
          {installGroups.map((pack) => (
            <div key={pack.class_type} className="space-y-1">
              <p><code className="break-all">{pack.class_type}</code> needs <b>{pack.pack}</b>.</p>
              {pack.install_action === 'moss_nodes' ? (
                <>
                  <InstallRunner action="moss_nodes" buttonLabel={`Install ${pack.pack}`} onDone={onRefresh} />
                  <p>Let running and queued jobs finish, restart ComfyUI, then use “Check nodes again”.
                    {' MOSS uses the bundled LDS loaders and requires ComfyUI with Qwen3.5 support.'}</p>
                </>
              ) : safeComfyLink(pack.url) ? (
                <>
                  <p>In ComfyUI-Manager, search for “{pack.search || pack.pack}” and install that package.
                    Restart ComfyUI when installation finishes.</p>
                  <a href={safeComfyLink(pack.url)} target="_blank" rel="noopener noreferrer" className={linkClass}>
                    {pack.pack} installation instructions
                  </a>
                </>
              ) : <p>Install the package from {pack.setup || 'Setup'}, then restart ComfyUI.</p>}
            </div>
          ))}
          {hasCore && (
            <>
              <p>Update ComfyUI for newer built-in nodes. If an unlisted custom package supplies a missing node,
                check its installation and ComfyUI’s startup log. Choose your installation method below.</p>
              <label className="block space-y-1">
                <span className="text-content">Installation method</span>
                <select value={method} onChange={(e) => setMethod(e.target.value)}
                  className="min-h-10 w-full min-w-0 rounded-md border border-border bg-surface px-2 text-content">
                  <option value="portable">Portable Windows</option>
                  <option value="desktop">Desktop</option>
                  <option value="manual">Manual or server</option>
                </select>
              </label>
              <p>If you have edited ComfyUI or added local commits, back up and preserve those changes first.
                A separate updated installation can keep your current setup intact.</p>
              {method === 'portable' && (
                <ol className="list-decimal space-y-1 pl-5">
                  <li>Close ComfyUI after its jobs finish.</li>
                  <li>Open the <code>update</code> folder beside your portable launcher.</li>
                  <li>Run <code className="break-all">update_comfyui.bat</code> to get the latest development version.
                    The stable script can lag behind new nodes. Use the ordinary updater first; the Python-dependencies
                    updater is for environment repairs.</li>
                  <li>When it finishes successfully, relaunch ComfyUI with your usual launcher, then check nodes again.</li>
                </ol>
              )}
              {method === 'desktop' && (
                <p>Open Desktop’s <b>Manage → Update</b> controls. If the Stable channel still lacks these nodes,
                  choose <b>Latest on GitHub</b> where available. Older Desktop versions use
                  <b> Help → Check for Updates</b>. Follow the official guide below, restart, then check again.</p>
              )}
              {method === 'manual' && (
                <div className="space-y-2">
                  <p>On the ComfyUI host, stop its service after jobs finish. Open its installation folder
                    and activate its own Python environment. Check changes first:</p>
                  <pre className="whitespace-pre-wrap break-all rounded bg-surface p-2 text-content">git status --short</pre>
                  <p>Continue only with a clean checkout and no local commits to preserve. Run each command
                    separately; stop if either fails:</p>
                  <pre className="whitespace-pre-wrap break-all rounded bg-surface p-2 text-content">{'git pull --ff-only\npython -m pip install -r requirements.txt'}</pre>
                  <p>Restart using your usual launcher or service, then check nodes again.
                    For a managed server, ask its owner to perform this update.</p>
                </div>
              )}
              <div className="flex flex-col items-start gap-2">
                <a href={UPDATE_GUIDE} target="_blank" rel="noopener noreferrer" className={linkClass}>Official ComfyUI update guide</a>
                <a href={PORTABLE_RELEASES} target="_blank" rel="noopener noreferrer" className={linkClass}>Download the latest portable release</a>
                <a href={MODEL_PATHS} target="_blank" rel="noopener noreferrer" className={linkClass}>Share existing models with a separate installation</a>
              </div>
              <p>For a separate installation, select its folder and API address in LDS Settings before checking again.</p>
              <SettingsLink section="local-tools" focus="comfyui-base-dir">Select ComfyUI folder</SettingsLink>
            </>
          )}
        </div>
      )}
    </section>
  );
}
