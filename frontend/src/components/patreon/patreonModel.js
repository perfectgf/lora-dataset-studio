// Pure helpers for the Patreon membership surfaces (Library card, install
// review, Updates card). The backend already pins every address; these checks
// repeat the pin before an address becomes a clickable link.

export const PATREON_AUTHORIZE_PREFIX = 'https://www.patreon.com/oauth2/authorize?';

export const PATREON_ACCESS_SUMMARY = 'New LDS versions reach Patreon members (LDS Member) first, then everyone, '
  + 'and exclusive plugins stay members-only. The version you have stays free and keeps working.';

// Dismissal of the Dataset page announcement (SupportProjectNotice). A new key
// on purpose: people who dismissed the earlier support appeal
// ('lds:support-notice-dismissed') see the announcement once. An active
// membership sets it too: a member needs neither Join nor Connect.
export const ACCESS_NOTICE_DISMISSED_KEY = 'lds:patreon-access-notice-dismissed';

// Patreon offers no lookup by user: a linked non-member becomes active through
// the creator's webhook, or by connecting again.
export const JUST_JOINED_HINT = 'Just joined? Press Connect Patreon again if your membership does not appear.';

// Protocol v1.1: approving on Patreon links nothing until the code shown on the
// Patreon page is typed here, so the person who approved cannot link someone
// else's installation.
export const CODE_PROMPT = 'Enter the code shown on the Patreon page';

export const FREE_TRIAL_TEXT = 'Patreon free trials do not include LDS access. '
  + 'Access starts with your first payment on Patreon.';

// Reading the membership needs no administration; changing it does (Docker's
// bridge and LAN browsers are not local administration).
export const ADMIN_HINT = 'Connecting Patreon needs local administration: use the computer that runs LDS, '
  + 'or unlock installation with the plugin admin token under Plugins.';

// One offered tier. The service reports `pro` only for an operator-configured
// pro tier, none since 2026-10-11; it keeps its request link but no name.
const TIERS = Object.freeze({ member: 'LDS Member' });

export function tierLabel(tier) {
  return TIERS[tier] || '';
}

function httpsUrl(value) {
  if (typeof value !== 'string' || !value || /[\s\\]/.test(value)) return null;
  try {
    const url = new URL(value);
    return url.protocol === 'https:' && !url.username && !url.password && !url.port && !url.hash ? url : null;
  } catch {
    return null;
  }
}

export function safeAuthorizeUrl(value) {
  const url = httpsUrl(value);
  return url && value.startsWith(PATREON_AUTHORIZE_PREFIX) && url.host === 'www.patreon.com'
    && url.pathname === '/oauth2/authorize' ? value : '';
}

export function safeJoinUrl(value) {
  const url = httpsUrl(value);
  return url && url.host === 'www.patreon.com' ? value : '';
}

export function safeRequestUrl(value) {
  return httpsUrl(value) ? value : '';
}

/* One line for the membership state; `access` is the backend's
   {status, patreon} summary (status: ready | not_connected | not_configured |
   unavailable, or not_permitted when the request lacked administration). */
export function membershipLabel(access) {
  const patreon = access?.patreon;
  if (access?.status === 'not_configured') return 'Patreon access is not available on this installation yet.';
  if (access?.status === 'not_permitted') {
    return 'Patreon membership is managed from this computer, or with the plugin admin token.';
  }
  if (!patreon) return 'Patreon could not be reached. Installed plugins and this version of LDS keep working.';
  if (patreon.status === 'active') return `Active Patreon membership${tierLabel(patreon.tier) ? ` · ${tierLabel(patreon.tier)}` : ''}`;
  if (patreon.reason === 'free_trial') return FREE_TRIAL_TEXT;
  if (patreon.pending === true) return `Approved on Patreon. ${CODE_PROMPT} to finish connecting.`;
  if (!patreon.linked) return 'Patreon is not connected to this installation.';
  if (patreon.status === 'verification_required') {
    return 'Patreon could not confirm your membership recently. Press Refresh; reconnect Patreon if this persists.';
  }
  return 'Patreon is connected, but no active LDS membership was found.';
}

/* Which actions to offer. Connect (again) and Join while not an active member,
   Disconnect while linked, the request link only for an active `pro` tier (so
   never without an operator-configured one), and the code field while an
   approval on Patreon waits for it. */
export function patreonActions(access) {
  const patreon = access?.patreon;
  const ready = Boolean(patreon) && access?.status !== 'not_configured' && access?.status !== 'unavailable';
  const linked = Boolean(ready && patreon.linked);
  const active = linked && patreon.status === 'active';
  return {
    connect: ready && !active,
    reconnect: linked && !active,
    join: ready && !active && Boolean(safeJoinUrl(patreon.join_url)),
    disconnect: linked,
    request: active && patreon.tier === 'pro' && Boolean(safeRequestUrl(patreon.request_url)),
    confirm: ready && patreon.pending === true,
  };
}

export function isPatreonRelease(release) {
  return release?.price?.provider === 'patreon';
}

/* What a plugin card lists under "Membership features", or null. A members-only
   plugin (Patreon catalog) is reserved as a whole: its release changelog. An
   official plugin lists its releases that need the members version of LDS
   (`members_releases` of the catalog, decided by the backend from the signed
   catalogs). `owned`: this installation has them, through an active membership
   for a members-only plugin, through the members version of LDS for an
   official one. `needsCore`: the release still needs that version here. */
export function membershipFeatures({ release, membersReleases, access }) {
  if (isPatreonRelease(release)) {
    return { whole: true, version: release.manifest?.version || '', items: release.changelog || [],
      owned: access?.patreon?.status === 'active', needsCore: Boolean(release.compatibility_issues?.length) };
  }
  if (!Array.isArray(membersReleases) || !membersReleases.length) return null;
  const owned = membersReleases[0].available === true;
  return { whole: false, version: membersReleases[0].version, owned, needsCore: !owned,
    items: [...new Set(membersReleases.flatMap(item => item.changelog || []))] };
}
