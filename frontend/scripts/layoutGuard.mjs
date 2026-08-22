#!/usr/bin/env node
/* 🚧 The thing that makes the responsive probe a GUARD RAIL rather than a
 * command somebody has to remember.
 *
 * ── The problem it solves ───────────────────────────────────────────────────
 *
 * `scripts/responsiveProbe.mjs` can see what the source-text tests cannot: a
 * panel full of air, a box past the right edge, a bar eating the fold. It found
 * 21 violations the first time it ran on a screen everyone believed was fine.
 *
 * But a check that has to be REMEMBERED is not a check. Three responsive
 * regressions shipped in one week with a full green suite behind them, and the
 * missing piece was never knowledge — it was that nothing fired at the moment
 * of the mistake. So this hooks the probe onto the edit itself:
 *
 *   mark    ← PostToolUse. An edit that touches layout stamps the tree dirty.
 *   check   ← Stop. The turn cannot end while the tree is dirtier than the last
 *             green probe run. Exit 2 hands the reason back to the assistant.
 *   waive   ← the escape hatch, which LEAVES A TRACE. Sometimes the app cannot
 *             be booted; that has to be sayable, and it has to be visible after
 *             the fact, because an escape hatch nobody can audit becomes the
 *             default route within a week.
 *
 * ── Why the marks live in .git/ ─────────────────────────────────────────────
 *
 * They are per-checkout state, never content: `.git/` is the one directory that
 * is already private to this working tree, is never committed by construction,
 * and travels with a worktree the way the marks need to. No .gitignore entry to
 * add, and no chance of a stamp reaching a contributor's clone.
 */

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const GITDIR = (() => {
  const dotGit = path.join(REPO, '.git');
  try {
    const st = fs.statSync(dotGit);
    if (st.isDirectory()) return dotGit;
    // A worktree's .git is a FILE pointing at the real directory. Following it
    // is what keeps the marks per-worktree instead of shared with the parent.
    const m = /^gitdir:\s*(.+)$/m.exec(fs.readFileSync(dotGit, 'utf8'));
    if (m) return m[1].trim();
  } catch { /* not a repo — handled by the callers below */ }
  return null;
})();

const DIRTY = GITDIR && path.join(GITDIR, 'lds-layout-dirty');
const GREEN = GITDIR && path.join(GITDIR, 'lds-probe-green');
const WAIVERS = GITDIR && path.join(GITDIR, 'lds-probe-waivers.log');

const mtime = (p) => { try { return fs.statSync(p).mtimeMs; } catch { return 0; } };

/** Does this edit plausibly move pixels?
 *
 *  Path alone is too blunt: every component in the app is .jsx, and stamping
 *  the tree dirty for a change to a fetch call would have the assistant running
 *  a browser probe after edits that cannot possibly have moved anything — and a
 *  check that cries wolf is a check that gets switched off. So the CONTENT
 *  decides for .jsx, and the path decides for stylesheets, where every line is
 *  layout by definition. */
function touchesLayout(filePath, written) {
  if (!filePath) return false;
  const rel = path.relative(REPO, path.resolve(filePath)).split(path.sep).join('/');
  if (!rel.startsWith('frontend/')) return false;
  if (/\.(css|scss)$/.test(rel)) return true;
  if (/tailwind\.config\.|postcss\.config\.|index\.html$/.test(rel)) return true;
  if (!/\.(jsx|tsx)$/.test(rel)) return false;
  // `.test.js` files never render anything.
  if (/\.test\./.test(rel)) return false;
  return /className|class=|style=\{|styled\.|<style/.test(written || '');
}

function readStdin() {
  try { return fs.readFileSync(0, 'utf8'); } catch { return ''; }
}

// ── mark ────────────────────────────────────────────────────────────────────
function mark() {
  if (!DIRTY) process.exit(0);
  let payload = {};
  try { payload = JSON.parse(readStdin() || '{}'); } catch { /* not our hook shape */ }
  const input = payload.tool_input || {};
  const filePath = input.file_path || input.notebook_path;
  // Everything the edit could have written, whichever tool wrote it.
  const written = [
    input.new_string, input.content,
    ...(Array.isArray(input.edits) ? input.edits.map((e) => e && e.new_string) : []),
  ].filter(Boolean).join('\n');
  if (!touchesLayout(filePath, written)) process.exit(0);
  fs.writeFileSync(DIRTY, `${path.relative(REPO, filePath)}\n`, { flag: 'a' });
  process.exit(0);
}

// ── check ───────────────────────────────────────────────────────────────────
function check() {
  if (!DIRTY) process.exit(0);
  const dirty = mtime(DIRTY);
  if (!dirty || dirty <= mtime(GREEN)) process.exit(0);
  let files = [];
  try {
    files = [...new Set(fs.readFileSync(DIRTY, 'utf8').split('\n').filter(Boolean))];
  } catch { /* the list is a nicety, the block is not */ }
  process.stderr.write(
    'This turn changed layout and the responsive probe has not run green on this tree.\n\n'
    + `Touched: ${files.slice(0, 6).join(', ')}${files.length > 6 ? `, +${files.length - 6} more` : ''}\n\n`
    + 'The frontend test suite CANNOT catch this — it reads the JSX as text and\n'
    + 'matches class names, so it stays green while the screen is unusable. Run:\n\n'
    + '  cd frontend && npm run probe:responsive -- --url http://127.0.0.1:<port>/#/canvas\n\n'
    + '(start a dev server first: LDS_DEV_API_TARGET=http://127.0.0.1:5050 npx vite --port 5175)\n\n'
    + 'If this change genuinely cannot be rendered, say so on the record:\n'
    + '  node frontend/scripts/layoutGuard.mjs waive "<why>"\n');
  process.exit(2);
}

// ── green / waive ───────────────────────────────────────────────────────────
function green() {
  if (!GREEN) return;
  fs.writeFileSync(GREEN, new Date().toISOString());
  try { fs.rmSync(DIRTY, { force: true }); } catch { /* already gone */ }
}

function waive() {
  if (!GREEN) process.exit(0);
  const why = process.argv.slice(3).join(' ').trim();
  if (!why) {
    process.stderr.write('waive needs a reason: layoutGuard.mjs waive "<why>"\n');
    process.exit(1);
  }
  // Appended, never overwritten. A waiver that leaves no trace is a silent
  // skip wearing a different hat, and the silent skip is the failure this
  // whole file exists to prevent.
  fs.appendFileSync(WAIVERS, `${new Date().toISOString()}  ${why}\n`);
  green();
  process.stdout.write(`waived and logged: ${why}\n`);
}

/* ⚠️ The CLI runs only when this file IS the entry point. Importing it — which
   its own test does — must not print usage or, worse, stamp the tree. A module
   with side effects on import is a module that cannot be tested. */
const cmd = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
  ? process.argv[2] : null;
if (cmd === null) { /* imported, not run */ }
else if (cmd === 'mark') mark();
else if (cmd === 'check') check();
else if (cmd === 'green') { green(); }
else if (cmd === 'waive') waive();
else {
  process.stdout.write('usage: layoutGuard.mjs mark|check|green|waive "<why>"\n');
}

export { touchesLayout };
