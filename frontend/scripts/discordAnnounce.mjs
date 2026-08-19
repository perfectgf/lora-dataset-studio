// =====================================================================//
//  Discord announcement generator — the same 🎁 What's new, condensed
// =====================================================================//
//  WHY THIS FILE EXISTS
//  --------------------
//  Announcements were written by hand from memory, at the end of a long day,
//  from a conversation that had already scrolled past half the work. That is
//  how a wave gets announced with nine of its thirteen items — and the four
//  that fall off are always the ones nobody happened to mention twice.
//
//  The changelog is already written, once, at ship time, in user-facing
//  English with the contributor credits: frontend/src/whatsNew.js. The release
//  notes read it (releaseNotes.mjs). This reads the SAME source, through the
//  SAME functions, so the announcement cannot drift from the release.
//
//  ── AN ANNOUNCEMENT IS NOT A CHANGELOG ──────────────────────────────────────
//  It is the only piece of the three that has to be READ. The panel is browsed
//  by someone already in the app; the release notes are consulted. A Discord
//  post is scrolled past. So this deliberately drops the blurbs and keeps the
//  titles: they are benefit-first by convention (see whatsNew.js), which makes
//  them exactly the one-line summaries an announcement needs. The detail is one
//  click away and it is already written twice.
//
//  Practical corollary: 13 entries carry ~7 600 characters of blurb. Discord's
//  hard limit is 2 000 per message. Pasting the notes was never an option —
//  either it is condensed, or it is silently truncated by the client, and a
//  truncated announcement is worse than a short one because nobody can tell.
//
//  ── SPLITTING ───────────────────────────────────────────────────────────────
//  When even the condensed form exceeds the limit, it is split on ENTRY
//  boundaries and each part is numbered. Never mid-sentence: a reader must be
//  able to see that part 1 of 2 is incomplete, which a hard cut does not show.
//
//  Usage:
//    node scripts/discordAnnounce.mjs --tag v2026.07.28.9 [--previous v…] [--json]
//  Prints the message(s) to stdout. It NEVER posts: posting is a separate,
//  reviewed act (see the lds-announce skill). A generator that could post is a
//  generator that will post something nobody read.
// =====================================================================//
import { extractCredits, newEntries, idsAtTag, previousTagOf, REPO_URL } from './releaseNotes.mjs';

/** Discord's per-message ceiling. Not a style choice — the API rejects beyond. */
export const DISCORD_LIMIT = 2000;

/**
 * Where a change lives, from the route the entry already carries.
 *
 * A title says WHAT changed and never WHERE — and a reader of #announcements
 * does not know which screen "Spot the footage that has been through the mill"
 * belongs to. They then have to open the app and hunt, or skip the line. The
 * entry already knows: `to:` is the deep link the What's-new panel uses, so the
 * surface is derived, never re-typed — a route that changes cannot leave a
 * stale label behind.
 *
 * Unknown or missing route → '' and the line renders exactly as before, rather
 * than inventing a place.
 */
const SURFACE_BY_ROUTE = [
  ['/video-bank', 'Video bank'],
  ['/bank', 'Bank'],
  ['/datasets', 'Datasets'],
  ['/canvas', 'Canvas'],
  ['/studio', 'Test Studio'],
  ['/cloud', 'Cloud'],
  ['/setup', 'Setup'],
  ['/settings', 'Settings'],
];

export function surfaceOf(entry) {
  const to = typeof entry?.to === 'string' ? entry.to.split('?')[0] : '';
  if (!to) return '';
  // Longest prefix wins: '/video-bank' must not be read as '/bank'.
  const hit = SURFACE_BY_ROUTE
    .filter(([route]) => to === route || to.startsWith(`${route}/`))
    .sort((a, b) => b[0].length - a[0].length)[0];
  return hit ? hit[1] : '';
}

/**
 * One line per entry. The title carries the news; the id never appears (it is
 * a storage key, not prose). The surface is prefixed in bold so a reader can
 * find the change without opening the app and hunting for it.
 */
export function renderLines(entries) {
  return entries.map((e) => {
    const where = surfaceOf(e);
    return `• ${where ? `**${where}** — ` : ''}${String(e.title).trim()}`;
  });
}

/**
 * The announcement, as an array of messages that each fit the limit.
 *
 * `head` and `tail` are kept WHOLE: the greeting must open part 1 and the
 * credits must close the last part, whatever the split. Splitting those would
 * produce a thank-you addressed to nobody.
 */
export function renderAnnouncement({ tag, entries, previousTag, limit = DISCORD_LIMIT }) {
  if (!entries.length) {
    throw new Error(`nothing to announce for ${tag}: no What's-new entry since `
      + `${previousTag || 'the previous release'}. An announcement that lists nothing `
      + 'is how a wave gets skipped — fix the changelog, do not post an empty message.');
  }
  const head = `## 🎁 ${tag} is out — ${entries.length} change${entries.length > 1 ? 's' : ''}\n`;
  const credits = extractCredits(entries);
  const tail = [
    '',
    credits.length
      ? `Thanks to **${credits.join('**, **')}** — these came from your reports. 🙏`
      : '',
    `Update from **Settings ▸ Maintenance ▸ Update & restart**, or grab the ZIP: <${REPO_URL}/releases/tag/${tag}>`,
  ].filter(Boolean).join('\n');

  const lines = renderLines(entries);
  const parts = [];
  let cur = [];

  // Reserve room for the numbering suffix that a multi-part message will need.
  // Measuring after the fact and re-splitting would be the classic off-by-one
  // that ships a 2 003-character message.
  const suffix = '\n\n*(part 99/99)*';
  const budgetFirst = limit - head.length - suffix.length;
  const budgetLast = limit - tail.length - suffix.length;

  for (const line of lines) {
    const budget = parts.length === 0 ? budgetFirst : budgetLast;
    const size = [...cur, line].join('\n').length;
    if (cur.length && size > budget) { parts.push(cur); cur = []; }
    cur.push(line);
  }
  if (cur.length) parts.push(cur);

  const multi = parts.length > 1;
  return parts.map((group, i) => {
    const body = group.join('\n');
    const open = i === 0 ? head : '';
    const close = i === parts.length - 1 ? `\n${tail}` : '';
    const mark = multi ? `\n\n*(part ${i + 1}/${parts.length})*` : '';
    return `${open}${body}${close}${mark}`;
  });
}

function parseArgv(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === '--json') out.json = true;
    else if (a.startsWith('--')) { out[a.slice(2)] = argv[i + 1]; i += 1; }
  }
  return out;
}

async function main() {
  const opts = parseArgv(process.argv.slice(2));
  const tag = opts.tag;
  if (!tag) {
    console.error('usage: node scripts/discordAnnounce.mjs --tag vX [--previous vY] [--json]');
    process.exit(2);
  }
  const repoRoot = new URL('../..', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1');
  const previousTag = opts.previous || previousTagOf(tag, repoRoot);
  const current = (await import('../src/whatsNew.js')).WHATS_NEW;
  const entries = newEntries(current, idsAtTag(previousTag, repoRoot));

  let messages;
  try {
    messages = renderAnnouncement({ tag, entries, previousTag });
  } catch (err) {
    console.error(String(err.message));
    process.exit(2);
    return;
  }
  if (opts.json) { console.log(JSON.stringify({ tag, previousTag, messages }, null, 2)); return; }
  messages.forEach((m, i) => {
    if (i) console.log('\n----------------------------- 8< -----------------------------\n');
    console.log(m);
  });
  console.error(`\n[${messages.length} message(s), ${messages.map((m) => m.length).join(' + ')} chars, `
    + `limit ${DISCORD_LIMIT}]`);
}

if (process.argv[1] && import.meta.url.endsWith(process.argv[1].replace(/\\/g, '/').split('/').pop())) {
  main();
}
