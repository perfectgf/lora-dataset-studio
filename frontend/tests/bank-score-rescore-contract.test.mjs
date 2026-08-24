/**
 * Contract for ✨ Score's two intents.
 *
 * The trap this pins is a repeat of one already made on the neighbouring pass:
 * "make the pass resume" reads like "hand it only the unscored images", and the
 * button that used to mean "do the whole bank" then becomes a no-op that still
 * reports a success. Here it would be worse than a no-op, because the style
 * cluster ids are ONE numbering of the whole bank — a pass over a subset would
 * restart them at 1 and land them on unrelated groups already stored.
 *
 * So three things are pinned, across both sides of the wire:
 *   • ✨ Score keeps posting an EMPTY body — it never silently re-scores;
 *   • the recompute-everything intent is a SEPARATE, explicit button, offered
 *     only when there is something to recompute (same shape as "Rescan all");
 *   • the server's pool stays "every non-rejected image", with no
 *     "already scored" filter anywhere near it.
 */
import test from 'node:test'
import { readSource } from './support/readSource.mjs'
import assert from 'node:assert/strict'
import { BANK_PASSES } from '../src/components/bank/bankPasses.js'

// CRLF-normalised: these files are checked out with native line endings on
// Windows and every pattern below spans lines.
const read = readSource
const workspace = read('src/components/bank/BankWorkspace.jsx')
// ✨ Score is a button of the PASSES PANEL since the Encre redesign; the
// workspace still owns the state the panel is handed.
const passPanel = read('src/components/bank/BankPassesPanel.jsx')
const route = read('../backend/app/routes/bank.py')
const service = read('../backend/app/services/image_bank_service.py')
const passes = read('src/components/bank/bankPasses.js')

test('the ✨ Score button posts nothing extra — its meaning is unchanged', () => {
  // The body is built by the SHARED builder now: `rescore` is spread in only when
  // the window's redo line is ticked, so an untouched run posts `{}` exactly as it
  // always did. The key itself is named in ✨ Score's spec, not at the call site.
  const body = workspace.slice(workspace.indexOf('const passBody'),
    workspace.indexOf('const runPass'))
  assert.match(body, /spec\?\.redo && redo \? \{ \[spec\.redo\.key\]: true \} : \{\}/)
  assert.match(passes, /key: 'rescore'/)
  assert.match(passPanel, /onClick=\{\(\) => onPassOpen\('score'\)\}/)
})

test('"Rescore all" is a separate intent, and it is now a line in ✨ Score\'s window', () => {
  // WHERE IT WENT. It was a second button on the pass row, shown only when there
  // was something to redo. It is the same intent, in the same place as every other
  // scope decision: the last line of the window's THIS RUN block, unticked, next to
  // the pool it re-runs. What must not change is that it stays EXPLICIT and priced.
  const spec = BANK_PASSES.score
  assert.ok(spec, 'the ✨ Score spec is missing')
  assert.equal(spec.redo?.key, 'rescore')
  assert.match(spec.redo.label, /Throw the cached embeddings away and recompute everything/)
  assert.match(spec.redo.note, /Costs a full pass/)
  // …and it is never pre-ticked: the workspace boots every redo flag off.
  assert.match(workspace, /const \[passRedo, setPassRedo\] = useState\(\{\}\)/)
  assert.match(workspace, /redo=\{!!passRedo\[passOpen\]\}/)
})

test('✨ Score refuses a partial scope, visibly, with the reason', () => {
  // The style ids are one numbering of the whole bank. Offering "kept only" would
  // renumber a sub-population onto groups already stored — so the option is shown
  // DISABLED with its objection, never removed and never obeyed.
  //
  // Read off the VALUES, not the source: the earlier version matched across the
  // `+` of a wrapped string literal, so re-flowing the sentence turned it red
  // while every user-visible word was unchanged. What is pinned is the objection
  // the user READS, and that both refusals say the same thing — a scope refused
  // for one reason and a selection refused for another would read as two bugs.
  const spec = BANK_PASSES.score
  assert.equal(typeof spec.scopes, 'string', 'the scope must be refused, not offered')
  assert.equal(spec.selection, spec.scopes)
  assert.match(spec.scopes, /^The style grouping is one numbering of the WHOLE bank/)
  assert.match(spec.scopes,
    /number that part from 1 and land those ids on top of unrelated groups already saved/)
})

test('the plain ✨ Score button tells the user a relaunch is cheap', () => {
  // The whole point of the resume is invisible unless it is said: without this
  // sentence people stop a long pass expecting to lose everything.
  assert.match(passPanel,
    /Already-scored images are reused, so stopping and relaunching costs only what is left/)
})

test('the server reads the intent from the body and nowhere else', () => {
  assert.match(route,
    /def bank_score\(bank_id\):[\s\S]{0,600}?rescore=bool\(data\.get\('rescore'\)\)/)
  assert.match(service, /def start_score\(app, user_id, bank_id, rescore=False\):/)
})

test('the scoring pool is never narrowed to "not scored yet"', () => {
  const job = service.slice(service.indexOf('def _score_job(bank_id'),
    service.indexOf("'rescore': bool(rescore)"))
  assert.ok(job.length > 0)
  assert.match(job, /\.filter\(BankImage\.status != 'reject'\)/)
  // A per-image score column appearing in this query is exactly the regression:
  // it would shrink the payload the style clustering is computed from.
  assert.doesNotMatch(job, /aesthetic_score/)
  assert.doesNotMatch(job, /nsfw_score/)
  assert.doesNotMatch(job, /style_cluster\.is_\(None\)/)
})

test('a stopped pass writes scores but never a half style partition', () => {
  // Both halves matter and they pull in opposite directions, so both are pinned
  // here: the salvage write must NOT honour the (already set) cancel flag, and
  // the cluster write must NOT run when the write-back was interrupted.  The
  // independently-proven SigLIP2 lane must also travel through this salvage
  // write: a stopped Score pass is not allowed to erase semantic work.
  assert.match(service,
    /_apply_score_results\(\s*job, by_path, score_results, interruptible=False,\s*preserved_siglip2_groups=preserved_siglip2_groups,\s*selected_engine=selected_engine\)/)
  // The gap after the `return` is deliberately loose: the pass now publishes a
  // phase line before the partition write (that step is minutes long and used
  // to run mute behind a full progress bar). What is pinned is the ORDER — the
  // cluster write stays behind the stopped-return, whatever is said in between —
  // the widths are a tolerance, not the property. They grew again when the
  // stopped line gained the clause naming the images this pass did not write
  // (_skipped_note): a sentence, not a reordering.
  assert.match(service,
    /if stopped:[\s\S]{0,600}?Stopped while saving[\s\S]{0,600}?return\n[\s\S]{0,700}?_write_style_clusters/)
})
