// =====================================================================//
//  Release notes generator — turns 🎁 What's new into a release body
// =====================================================================//
//  WHY THIS FILE EXISTS
//  --------------------
//  Release bodies used to be `--notes "<preamble>" --generate-notes`. gh builds
//  "What's Changed" from MERGED PULL REQUESTS, and this repo has none — every
//  change lands straight on main. So the generated half was always empty and the
//  body was the 747-character preamble and nothing else. Three releases shipped
//  that way in one day before anyone noticed, because nothing checked.
//
//  The product changelog already exists and is written by hand at ship time, in
//  user-facing English, with the contributor credits: frontend/src/whatsNew.js.
//  This module wires it to the release.
//
//  ── WHICH ENTRIES BELONG TO A RELEASE ───────────────────────────────────────
//  By git diff of whatsNew.js between the two tags, NOT by entry `date`.
//  Three releases were cut on 2026-07-27 (v2026.07.28, .1, .2) and every entry
//  they shipped is dated 2026-07-27 — a date filter cannot split them, and would
//  give all three the same body. It also mis-files the routine case of an entry
//  written yesterday and shipped today. The diff answers the only question that
//  matters: what is in THIS tag that was not in the previous one.
//
//  ── HOW THE TWO SIDES ARE READ (deliberately asymmetric) ────────────────────
//  • CURRENT tag: the file is IMPORTED as the JS module it is. Titles, blurbs
//    and ids come from the real array — no regex can drift away from it.
//  • PREVIOUS tag: only the ID SET is needed, and importing a historical file
//    would mean resolving ITS imports against TODAY's registries (renamed export
//    => crash while cutting a release). So ids are read from the old source text
//    with the same `^ {4}id: '` scan that whatsNew.test.js already asserts is
//    exhaustive ("no entry is swallowed by a merge"). Tolerant where it must be,
//    exact where it matters.
//
//  ── WHY --generate-notes IS GONE FROM release.yml ───────────────────────────
//  With no PRs it contributes exactly one useful line, the Full Changelog compare
//  link, wrapped in an empty "## What's Changed" heading that reads like a bug
//  next to a real changelog. We emit the compare link ourselves instead.
//
//  ── SILENCE IS THE DEFECT ───────────────────────────────────────────────────
//  A release with zero new entries FAILS here, loudly, before anything is built.
//  A pure-plumbing release is legitimate — say so on purpose by putting
//  [no-notes] in the annotated tag message. "Nothing to announce, on purpose"
//  and "the generator found nothing" must never look the same.
// =====================================================================//
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync, rmSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';

export const REPO_URL = 'https://github.com/perfectgf/lora-dataset-studio';

// Path of the changelog module inside the repo, as git wants it.
export const WHATS_NEW_REPO_PATH = 'frontend/src/whatsNew.js';

/**
 * Ids present in a historical whatsNew.js SOURCE TEXT.
 * Mirrors the scan whatsNew.test.js pins as exhaustive against the parsed array.
 */
export function extractIds(source) {
  const start = source.indexOf('export const WHATS_NEW = [');
  const body = start === -1 ? source : source.slice(start);
  return new Set((body.match(/^ {4}id: '([^']+)'/gm) || []).map((line) => line.slice("    id: '".length, -1)));
}

/** Entries of `current` whose id was not already shipped. Feed order is kept. */
export function newEntries(current, previousIds) {
  return current.filter((e) => !previousIds.has(e.id));
}

// "Reported by nofaceman (Discord)." — the credit lives inside the blurb prose,
// which is where it should stay. This only LIFTS it into a Thanks line as well;
// missing one costs visibility, never the credit itself.
// Anchored on the SHAPE of a credit — "<by|to> <handle> (<source>)" — and not
// on a list of verbs, because every one of this wave's five credits escaped the
// verb-list version and shipped a release with no Thanks line at all:
//   • "Found and diagnosed by 1Tomber (…)" — two words between verb and "by";
//   • "…and fixed by …"                    — a verb the list did not have;
//   • "Thanks to j_o_e_l. (Discord)"       — a handle containing a dot, which
//     the old `[^.,(]+?` could not cross;
//   • "Reported by 1Tomber (GitHub #23)"   — an issue number inside the parens.
// Prose is kept out by the HANDLE charset (no spaces) rather than by grammar:
// "…thanks to the work of everyone (Discord)" cannot match, "by nofaceman
// (Discord)" must. An issue number is dropped from the displayed credit — the
// person is the credit, the number is a coordinate.
const CREDIT_RE = /\b(?:by|to) ([A-Za-z0-9][A-Za-z0-9._-]{1,30}?)\.? \((Reddit|Discord|GitHub|Civitai)(?:[^)]*)?\)/g;

export function extractCredits(entries) {
  const seen = new Set();
  for (const e of entries) {
    for (const m of String(e.blurb || '').matchAll(CREDIT_RE)) {
      seen.add(`${m[1].trim()} (${m[2]})`);
    }
  }
  return [...seen];
}

/**
 * A screenshot's absolute URL, pinned to the TAG being released.
 *
 * Two things a relative path cannot do. A release body is rendered on a page
 * that has no repository root, so `docs/screenshots/x.png` resolves to nothing
 * and shows a broken image. And pinning to `main` would mean the picture in a
 * six-month-old release silently becomes the CURRENT screen — a note claiming
 * to show what shipped, showing something else. The tag ref is immutable, so
 * the image stays the one that was true when it was written.
 */
export function screenshotUrl(tag, relPath) {
  const clean = String(relPath).replace(/^\/+/, '');
  return `${REPO_URL}/raw/${encodeURIComponent(tag)}/${clean.split('/').map(encodeURIComponent).join('/')}`;
}

/**
 * The release body. `to:` targets are dropped on purpose: they are in-app router
 * paths ('/settings/engines'), which mean nothing on a GitHub page and would
 * render as dead links. `image:` is the opposite case — it exists precisely to
 * be seen here, and is rendered AFTER the prose: the text explains what changed,
 * the picture shows it.
 */
export function renderNotes({ preamble = '', tag, previousTag, entries }) {
  const out = [];
  if (preamble.trim()) out.push(preamble.trim(), '');
  out.push(`## 🎁 What's new in ${tag}`, '');

  for (const e of entries) {
    out.push(`### ${e.title}`, '', String(e.blurb).trim(), '');
    if (e.image) {
      // The alt text is the entry's own title: a reader on a screen reader, or
      // on a connection that never loads it, still learns what the picture was
      // meant to show.
      out.push(`![${e.title}](${screenshotUrl(tag, e.image)})`, '');
    }
  }

  const credits = extractCredits(entries);
  if (credits.length) {
    out.push('---', '', `**Thanks to ${credits.join(', ')}** — these came from you.`, '');
  }

  if (previousTag) {
    out.push(`**Full changelog**: ${REPO_URL}/compare/${previousTag}...${tag}`);
  }
  return `${out.join('\n').replace(/\n{3,}/g, '\n\n').trimEnd()}\n`;
}

/**
 * What a release with no new entry should do. Pure on purpose: the whole point
 * of this change is that "announced nothing" is never silent, so the decision
 * has to be testable without a git repo or a workflow run.
 * Returns null when there IS something to announce.
 */
export function emptySignal({ entries, allowEmpty = false, tag, previousTag }) {
  if (entries.length > 0) return null;
  const why = `No new What's-new entry between ${previousTag} and ${tag}: this release would `
    + 'announce nothing. Add one to frontend/src/whatsNew.js, or, if this really is a '
    + 'plumbing-only release, say so on purpose by putting [no-notes] in the tag message.';
  return allowEmpty
    ? { severity: 'warning', exitCode: 0, message: `Accepted via [no-notes]. ${why}`,
        annotation: `::warning title=Release with no user-facing notes::Accepted via [no-notes]. ${why}` }
    : { severity: 'error', exitCode: 2, message: why,
        annotation: `::error title=Release notes are empty::${why}` };
}

// ── git plumbing (CLI only; the pure functions above are what tests exercise) ─

function git(args, cwd) {
  return execFileSync('git', args, { cwd, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
}

export function previousTagOf(tag, cwd) {
  try {
    return git(['describe', '--tags', '--abbrev=0', `${tag}^`], cwd).trim();
  } catch {
    return null;
  }
}

export function idsAtTag(tag, cwd) {
  return extractIds(git(['show', `${tag}:${WHATS_NEW_REPO_PATH}`], cwd));
}

// Historical changelog, imported as a real module. It is dropped NEXT TO the
// current one so its relative imports ('./components/...') still resolve —
// against today's registries, which is why this is opt-in and not the CI path.
async function importWhatsNewAt(tag, repoRoot) {
  const tmp = path.join(repoRoot, 'frontend', 'src', `whatsNew.__at-${tag.replace(/[^\w.-]/g, '_')}.mjs`);
  writeFileSync(tmp, git(['show', `${tag}:${WHATS_NEW_REPO_PATH}`], repoRoot), 'utf8');
  try {
    return await import(pathToFileURL(tmp).href);
  } finally {
    rmSync(tmp, { force: true });
  }
}

function parseArgv(argv) {
  const opts = { allowEmpty: false };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === '--allow-empty') opts.allowEmpty = true;
    else if (a.startsWith('--')) { opts[a.slice(2).replace(/-(\w)/g, (_, c) => c.toUpperCase())] = argv[++i]; }
  }
  return opts;
}

async function main() {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const repoRoot = path.resolve(here, '..', '..');
  const opts = parseArgv(process.argv.slice(2));

  const tag = opts.tag;
  if (!tag) throw new Error('usage: node scripts/releaseNotes.mjs --tag <vX> [--prev <vY>] [--out <file>] [--allow-empty]');

  const previousTag = opts.prev || previousTagOf(tag, repoRoot);
  if (!previousTag) {
    throw new Error(
      `cannot resolve the tag before ${tag}. Pass --prev <tag> explicitly; `
      + 'guessing would dump the entire changelog into one release.',
    );
  }

  // --at <tag> regenerates the notes of a PAST release from the changelog as it
  // stood at that tag (CI never uses it: the checkout already is the tag).
  const { WHATS_NEW } = opts.at ? await importWhatsNewAt(opts.at, repoRoot) : await import('../src/whatsNew.js');
  const entries = newEntries(WHATS_NEW, idsAtTag(previousTag, repoRoot));

  const preamblePath = opts.preamble || path.join(repoRoot, '.github', 'RELEASE_PREAMBLE.md');
  const preamble = readFileSync(preamblePath, 'utf8');
  const body = renderNotes({ preamble, tag, previousTag, entries });

  const signal = emptySignal({ entries, allowEmpty: opts.allowEmpty, tag, previousTag });
  if (signal) {
    process.stdout.write(`${signal.annotation}\n`);
    if (signal.exitCode !== 0) {
      process.exitCode = signal.exitCode;
      return;
    }
  }

  if (opts.out) writeFileSync(opts.out, body, 'utf8');
  else process.stdout.write(body);
  process.stderr.write(`release notes: ${entries.length} new entr${entries.length === 1 ? 'y' : 'ies'} `
    + `between ${previousTag} and ${tag}\n`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((err) => {
    process.stdout.write(`::error title=Release notes generator failed::${err.message}\n`);
    process.exitCode = 1;
  });
}
