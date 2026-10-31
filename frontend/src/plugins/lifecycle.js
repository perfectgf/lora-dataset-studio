import { pluginMembership } from './registry.js';

// Registration survives a membership change, so reconnection needs no restart.
export function pluginLoaded(plugin) {
  return typeof plugin.active === 'boolean'
    ? plugin.active : Boolean(plugin.enabled && plugin.state === 'loaded');
}

export function pluginActive(plugin) {
  return pluginLoaded(plugin) && pluginMembership(plugin.id, plugin.membership)?.allowed !== false;
}

export function pluginDesired(plugin) {
  return plugin.desired_enabled ?? plugin.enabled ?? false;
}

export function pendingLabel(plugin) {
  switch (plugin.pending_action) {
    case 'enable': return 'Will turn on after restart';
    case 'disable': return 'Will turn off after restart';
    case 'install': return 'Installation ready to apply';
    case 'update': return `Update${plugin.pending_version ? ` to ${plugin.pending_version}` : ''} ready to apply`;
    case 'remove': return 'Will be removed after restart · data kept';
    default: return '';
  }
}

/** Wait for a different process, not the old process's last healthy response. */
export async function waitForPluginBoot(previousBootId, {
  fetchImpl = globalThis.fetch, timeoutMs = 90000, intervalMs = 1000,
  now = Date.now, sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
  signal,
} = {}) {
  if (!previousBootId) throw new Error('Refresh the plugin list before restarting.');
  const deadline = now() + timeoutMs;
  while (now() < deadline) {
    signal?.throwIfAborted();
    try {
      const timeout = AbortSignal.timeout(Math.max(1, Math.min(5000, deadline - now())));
      const response = await fetchImpl('/api/plugins/', {
        cache: 'no-store', credentials: 'include',
        signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
      });
      if (response.ok) {
        const payload = await response.json();
        if (payload.boot_id && payload.boot_id !== previousBootId) return payload;
      }
    } catch { /* A connection failure is expected while the server restarts. */ }
    signal?.throwIfAborted();
    await sleep(Math.min(intervalMs, Math.max(0, deadline - now())));
  }
  throw new Error('LDS has not restarted yet. Check your launcher, then refresh this page. Your requested changes are saved.');
}
