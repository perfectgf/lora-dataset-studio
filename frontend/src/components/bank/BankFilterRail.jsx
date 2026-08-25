/**
 * The Bank's filter rail — all of triage, beside the grid it filters.
 *
 * THE PROBLEM IT SOLVES. The workspace used to stack four zones vertically, so
 * adjusting a filter meant scrolling up, clicking, and scrolling back down to
 * see what changed. On a bank of 20 000+ images that round trip is the cost, not
 * the click count. Putting the controls next to their result removes it.
 *
 * THE ORDER is the design note's, and it is by how often a facet is reached, not
 * by which pass produced it: search, then Status / Quality / Groups and the
 * person + style strips, then Score / Framing / Medium / Angle / Resolution /
 * Origin folded into one disclosure. Six measured axes expanded at once is what
 * made the old filter block a wall.
 *
 * ⚠️ AT 400 px IT FOLDS RATHER THAN OVERFLOWS. Below `sm` the rail cannot share
 * the row with the grid, so it becomes a drawer over it, with its own ✕ and a
 * backdrop. The decision of which mode applies lives in bankLayout.js, where
 * `node --test` can reach it.
 *
 * TAB ORDER. This component is the SECOND child of the workspace, after the top
 * bar and before the grid, so the DOM order is the reading order and no
 * tabindex is needed to keep them agreeing.
 *
 * ⚠️ Every prop is named exactly as the workspace's own local, so this JSX moved
 * here byte-identical. That is deliberate: the surface contract catches a
 * DELETED button, but nothing catches a filter silently wired to the wrong
 * prop — so the safest rewrite is the one that rewrites nothing.
 */
import BankThresholdsPanel from './BankThresholdsPanel.jsx'
import { Ban, Palette, Search, SlidersHorizontal, SlidersVertical, UserX } from 'lucide-react';
import DescribeFilterBar from './DescribeFilterBar.jsx'
import SelectionTagsPanel from './SelectionTagsPanel.jsx'
import SubfolderPersonPanel from './SubfolderPersonPanel'
import { Chip, FilterGroup, GroupLabel } from './BankAtoms.jsx'
import { FLAG_HINT, FLAG_LABEL, ORIGIN_BUCKETS } from './bankFacets.js'
import { foldedCount } from './bankLayout.js'
import { angleTitle, mediumTitle } from './bankMedium.js'
import { assertionFor, folderMarker, scanOffer, suggestionFor } from './folderPerson.js'

export default function BankFilterRail({
  bankId, filter, setF,
  clusters, styleClusters,
  searchText, setSearchText, excludeText, setExcludeText,
  subfolders, folderPersonInfo, folderPersons, folderPersonBusy,
  assertFolderPerson, revokeFolderPerson, checkFolderPerson, scanFolderPersons,
  tagRow, tagPicked, toggleTag, clearTags,
  chipsFiltered, flags, statusCounts, availableScoreFlags, payload,
  shownResBuckets, resBuckets, originMeasured, originCounts,
  shownFramings, framingCounts, shownMediums, mediumCounts, mediumNote,
  shownAngles, angleCounts, angleState,
  live, setPassOpen,
  thresholdsOpen, setThresholdsOpen, connection, refreshPayload, refreshImages, onRunPass,
  sortGroups, setSort, tileSize, setTileSize,
  moreOpen, setMoreOpen, isDrawer, onClose,
}) {
  // Which measured axes have data. The conditions are the ones the workspace
  // already used — moving a facet into the rail must not change WHEN it appears.
  const available = {
    score: availableScoreFlags.length > 0,
    framing: shownFramings.length > 0,
    medium: shownMediums.length > 0,
    angle: shownAngles.length > 0 || !!angleState.offer,
    resolution: shownResBuckets.length > 0,
    origin: originMeasured > 0,
  }
  const hiddenAxes = foldedCount(available)

  return (
    <aside aria-label="Bank filters"
      data-probe-chrome={isDrawer ? 'rail' : undefined} data-probe-layer={isDrawer ? '' : undefined}
      data-probe-panel="rail"
      className={isDrawer
        /* ⚠️ `bg-surface-overlay`, NOT `bg-surface`: the tint is 4 %-alpha white
           for cards sitting ON the opaque page — painted with it, this drawer
           is a sheet of glass over the grid. Pinned in bankLayout.test.js. */
        ? 'shadow-2xl fixed inset-y-0 left-0 z-50 w-[19rem] max-w-[88vw] overflow-y-auto border-r border-border bg-surface-overlay p-3 space-y-3'
        : 'space-y-3 self-start rounded-xl border border-border bg-surface p-3 sm:sticky sm:top-3 sm:max-h-[calc(100vh-1.5rem)] sm:overflow-y-auto'}>
      <div className="flex items-center gap-2">
        <GroupLabel>Filters</GroupLabel>
        {isDrawer && (
          <button type="button" onClick={onClose} aria-label="Close the filters"
            className="min-h-10 lg:min-h-0 ml-auto rounded-md border border-border px-2 py-0.5 text-xs text-content-muted hover:text-content">
            ✕
          </button>
        )}
      </div>

      {/* ── Status — FIRST, and dressed as what it is: the triage's main
          gesture. These four were the same gray chips as every other facet,
          lost mid-rail; asked for from live use ("plus visibles, mieux
          placés"). Each carries its live count and its status colour, so the
          split of the bank is readable before anything is clicked. */}
      <div className="grid grid-cols-2 gap-1.5" role="group" aria-label="Filter by status">
        {[
          { id: null, label: 'All', n: statusCounts?.total,
            on: 'border-indigo-400/70 bg-indigo-500/25 text-indigo-100',
            active: !filter.status && !filter.flag && filter.cluster == null && filter.style == null,
            set: () => setF({ status: null, flag: null, cluster: null, style: null }) },
          { id: 'pending', label: 'Undecided', n: statusCounts?.pending,
            on: 'border-amber-400/70 bg-amber-500/20 text-amber-100' },
          { id: 'keep', label: '✓ Kept', n: statusCounts?.keep,
            on: 'border-emerald-400/70 bg-emerald-500/20 text-emerald-100' },
          { id: 'reject', label: '✕ Rejected', n: statusCounts?.reject,
            on: 'border-rose-400/70 bg-rose-500/20 text-rose-100' },
        ].map((s) => {
          const active = s.active ?? filter.status === s.id
          return (
            <button key={s.label} type="button"
              onClick={s.set || (() => setF({ status: filter.status === s.id ? null : s.id }))}
              aria-pressed={active}
              className={`min-h-10 lg:min-h-0 rounded-md border px-2.5 py-1.5 text-sm font-semibold transition-colors ${active
                ? s.on
                : 'border-border bg-surface-raised text-content-muted hover:text-content hover:bg-surface'}`}>
              {s.label}
              {typeof s.n === 'number' && (
                <span className="ml-1.5 text-xs font-normal tabular-nums opacity-80">
                  {s.n.toLocaleString()}
                </span>
              )}
            </button>
          )
        })}
      </div>

      {/* Say it in words; the app sets its own chips and the counters below —
          measured, not the model — say what that lands on. At the TOP of the
          rail because it is a shortcut to the controls underneath, not a
          separate way of selecting. */}
      <DescribeFilterBar bankId={bankId} onApply={setF} />

      <div className="relative">
        <Search aria-hidden="true" className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-content-subtle" />
        <input type="search" value={searchText} onChange={(e) => setSearchText(e.target.value)}
          placeholder="Search captions and file names… (e.g. red dress)"
          aria-label="Search the bank by caption or file name"
          className="w-full rounded-md border border-border bg-surface py-1.5 pl-8 pr-8 text-sm text-content placeholder:text-content-subtle" />
        {searchText && (
          <button type="button" onClick={() => setSearchText('')} aria-label="Clear search"
            className="absolute right-2 top-1/2 -translate-y-1/2 text-content-subtle hover:text-content">✕</button>
        )}
      </div>
      {/* 🚫 Exclude — the search bar read backwards. Captioning a big bank
          turns it into a checklist ("which ones have I not tagged with X
          yet?"), and that question has no answer while the only text tool
          can just narrow TO a word. Same fields as the search (caption +
          file name), same debounce, composes with every other facet.
          Comma-separated: hiding 'logo, watermark' in one pass is the
          normal case. */}
      <div className="relative">
        <Ban aria-hidden="true" className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-content-subtle" />
        <input type="search" value={excludeText} onChange={(e) => setExcludeText(e.target.value)}
          placeholder="Exclude words… (e.g. logo, watermark)"
          aria-label="Hide images whose caption or file name contains these words"
          title="Hides every image whose caption or file name contains one of these words (comma-separated). Matches anywhere in the text, so 'car' also hides 'scarf'. Images with no caption are never hidden."
          className="w-full rounded-md border border-border bg-surface py-1.5 pl-8 pr-8 text-sm text-content placeholder:text-content-subtle" />
        {excludeText && (
          <button type="button" onClick={() => setExcludeText('')} aria-label="Clear the exclude filter"
            className="absolute right-2 top-1/2 -translate-y-1/2 text-content-subtle hover:text-content">✕</button>
        )}
      </div>

      {/* Subfolder scoping (a Telegram export nests one folder per chat/date).
          Scoping to ONE folder is also the moment the user knows whose it is,
          so the 👤 "Single person here" assertion lives right under it. */}
      {subfolders.length > 1 && (
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center gap-1.5">
            <GroupLabel>Subfolder</GroupLabel>
            <select value={filter.subfolder ?? '__all__'}
              onChange={(e) => setF({ subfolder: e.target.value === '__all__' ? null : e.target.value })}
              className="min-w-0 flex-1 rounded-md border border-border bg-surface px-2 py-1 text-xs text-content">
              <option value="__all__">All subfolders</option>
              {subfolders.map((s) => (
                <option key={s.name || '__root__'} value={s.name}>
                  {s.name === '' ? '(bank root)' : s.name} · {s.count}
                  {/* '👤?' = the app sampled this folder and it looked like
                      one person. A hint that something is worth opening,
                      never a claim — the panel below says the numbers. */}
                  {folderMarker(folderPersonInfo?.suggestions, s.name)}
                </option>
              ))}
            </select>
          </div>
          <SubfolderPersonPanel
            subfolder={filter.subfolder}
            entry={assertionFor(folderPersons, filter.subfolder)}
            suggestion={suggestionFor(folderPersonInfo?.suggestions, filter.subfolder)}
            offer={scanOffer(folderPersonInfo)}
            busy={folderPersonBusy}
            onAssert={assertFolderPerson}
            onRevoke={revokeFolderPerson}
            onCheck={checkFolderPerson}
            onScan={scanFolderPersons} />
        </div>
      )}

      {/* Person clusters (after the face pass) */}
      {clusters.length > 0 && (
        <div className="space-y-1">
          <GroupLabel>
            People ({clusters.length} cluster{clusters.length > 1 ? 's' : ''} — biggest first)
          </GroupLabel>
          <ul className="flex gap-2 overflow-x-auto pb-1">
            {clusters.map((c) => (
              <li key={c.id} className="shrink-0">
                <button type="button" onClick={() => setF({ cluster: filter.cluster === c.id ? null : c.id, flag: null })}
                  title={`Show person #${c.id} (${c.size} image(s))`}
                  className={`relative block overflow-hidden rounded-lg border ${filter.cluster === c.id
                    ? 'border-indigo-400 ring-2 ring-indigo-400' : 'border-border'}`}>
                  {c.cover_image_id != null && (
                    <img src={`/api/bank/${bankId}/thumb/${c.cover_image_id}`} alt={`Person ${c.id}`}
                      loading="lazy" className="h-16 w-16 object-cover" />
                  )}
                  <span className="absolute bottom-0 inset-x-0 bg-black/60 text-center text-[10px] font-semibold text-white">
                    #{c.id} · {c.size}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Style clusters (after the scoring pass) — group screenshots/memes vs photoreal */}
      {styleClusters.length > 0 && (
        <div className="space-y-1">
          <GroupLabel>
            Styles ({styleClusters.length} group{styleClusters.length > 1 ? 's' : ''} — biggest first)
          </GroupLabel>
          <ul className="flex gap-2 overflow-x-auto pb-1">
            {styleClusters.map((c) => (
              <li key={c.id} className="shrink-0">
                <button type="button" onClick={() => setF({ style: filter.style === c.id ? null : c.id, flag: null, cluster: null })}
                  title={`Show style group #${c.id} (${c.size} image(s))`}
                  className={`relative block overflow-hidden rounded-lg border ${filter.style === c.id
                    ? 'border-fuchsia-400 ring-2 ring-fuchsia-400' : 'border-border'}`}>
                  {c.cover_image_id != null && (
                    <img src={`/api/bank/${bankId}/thumb/${c.cover_image_id}`} alt={`Style ${c.id}`}
                      loading="lazy" className="h-16 w-16 object-cover" />
                  )}
                  <span className="absolute bottom-0 inset-x-0 bg-black/60 text-center text-[10px] font-semibold text-white">
                    <Palette aria-hidden="true" className="mr-0.5 inline h-3 w-3 align-[-1px]" />{c.id} · {c.size}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* While anything is filtered the numbers describe the FILTERED bank, and
          the row says so: a count that silently changed meaning would be worse
          than the bank-wide one it replaces. */}
      {chipsFiltered && (
        <p className="m-0 text-[11px] leading-snug text-content-subtle">
          Counts below follow the active filters. Each chip is counted with the
          others applied and its own value lifted, so you can always switch to
          a neighbour.
        </p>
      )}

      <div className="flex flex-col gap-3 border-t border-border pt-2">
        <div className="flex flex-col gap-1.5">
          <FilterGroup label="Quality">
            {['blur', 'noise', 'uniform', 'small', 'soft_detail', 'bars', 'unreadable'].map((f) => (
              <Chip key={f} active={filter.flag === f}
                onClick={() => setF({ flag: filter.flag === f ? null : f })}
                title={FLAG_HINT[f] || 'Sorted worst-first'}>
                {FLAG_LABEL[f]} {flags[f] ?? 0}
              </Chip>
            ))}
            {/* Counted like its six neighbours (the backend shares the chip's
                own criterion): a chip with no number read as "not a filter". */}
            <Chip active={filter.flag === 'clean'}
              onClick={() => setF({ flag: filter.flag === 'clean' ? null : 'clean' })}
              title="Scanned and carrying none of the quality flags">
              Clean {flags.clean ?? 0}
            </Chip>
          </FilterGroup>
          {/* Coverage, stated where the zeros are. Measured on a real bank:
              1 107 of 36 921 scanned showed a row of honest zeros that read as
              "nothing is blurry" — the flags only see the scanned slice, and
              only this row knows that. */}
          {(payload?.counts?.unscanned_scannable ?? 0) > 0 && (
            <p className="m-0 pl-1 text-[11px] leading-snug text-amber-300/90">
              Only {(payload.counts.scanned ?? 0).toLocaleString()} of{' '}
              {(payload.counts.total ?? 0).toLocaleString()} images are scanned —
              these flags only see that part.{' '}
              <button type="button" onClick={() => setPassOpen('scan')}
                className="underline underline-offset-2 hover:text-amber-200">
                <Search aria-hidden="true" className="mr-1 inline h-3 w-3 align-[-1px]" />Scan the rest
              </button>
            </p>
          )}
        </div>

        <FilterGroup label="Groups">
          <Chip active={filter.flag === 'dups'} onClick={() => setF({ flag: filter.flag === 'dups' ? null : 'dups' })}
            title="Exact / resized duplicate groups (perceptual hash) with their resolution panel">
            ≈ Duplicates {payload?.dup?.unresolved ?? 0}
          </Chip>
          {(payload?.semantic_dup?.groups ?? 0) > 0 && (
            <Chip active={filter.flag === 'semantic_dups'}
              onClick={() => setF({ flag: filter.flag === 'semantic_dups' ? null : 'semantic_dups' })}
              title="Semantic near-duplicates — same shot, different crop/compression — with their resolution panel">
              ✂ Same shot {payload?.semantic_dup?.unresolved ?? 0}
            </Chip>
          )}
          {payload?.faces_scanned > 0 && (
            <Chip active={filter.flag === 'no_face'} onClick={() => setF({ flag: filter.flag === 'no_face' ? null : 'no_face' })}>
              <UserX aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />No face
            </Chip>
          )}
        </FilterGroup>
      </div>

      {/* The six MEASURED axes, folded into one disclosure. Expanded they were a
          wall; hidden without a count they would be a disclosure nobody opens,
          so the button says how many it holds. */}
      {hiddenAxes > 0 && (
        <div className="border-t border-border pt-2">
          <button type="button" onClick={() => setMoreOpen((v) => !v)}
            aria-expanded={moreOpen} aria-controls="bank-rail-more"
            className="min-h-10 lg:min-h-0 flex w-full items-center gap-1.5 text-left text-xs text-content-muted hover:text-content">
            <SlidersHorizontal aria-hidden="true" className="h-3.5 w-3.5" />
            <span className="font-medium">More filters</span>
            <span className="text-content-subtle">({hiddenAxes})</span>
            <span aria-hidden className="ml-auto text-content-subtle">{moreOpen ? '▲' : '▼'}</span>
          </button>
          {moreOpen && (
            <div id="bank-rail-more" className="mt-2 flex flex-col gap-3">
              {/* Score-derived flags — only surfaced once their pass has produced data.
                  These used to clear the person/style cluster on click, and the ≈
                  Duplicates chips cleared the cluster too. That silently undid a
                  filter the user had set — and now that each chip PRINTS the size of
                  the page it opens, it would also have made that number wrong.
                  A chip toggles its own facet and nothing else. */}
              {available.score && (
                <FilterGroup label="Score">
                  {availableScoreFlags.map((f) => (
                    <Chip key={f} active={filter.flag === f}
                      onClick={() => setF({ flag: filter.flag === f ? null : f })}
                      title={f === 'watermark' ? 'Overlaid watermark detected' : 'Sorted worst-first'}>
                      {FLAG_LABEL[f]} {flags[f] ?? 0}
                    </Chip>
                  ))}
                </FilterGroup>
              )}

              {/* Framing tiers — face/bust/body/back (+ unknown), from the 📐 Framing
                  pass. One active at a time; re-click clears. Composes with everything. */}
              {available.framing && (
                <FilterGroup label="Framing">
                  {shownFramings.map((b) => (
                    <Chip key={b.id} active={filter.framing === b.id}
                      onClick={() => setF({ framing: filter.framing === b.id ? null : b.id })}
                      title="Filter the grid to this shot type (from the Framing pass)">
                      {b.label} {framingCounts[b.id] ?? 0}
                    </Chip>
                  ))}
                </FilterGroup>
              )}

              {/* 🎨 Medium — what the picture is MADE of, from the ✨ Score
                  embeddings. A DIFFERENT question from 🔎 Origin below, which reads
                  the file's metadata: a photorealistic AI render is 🤖 AI and 📷
                  Photo at once. The limits sentence is part of the row, not a
                  tooltip, because "unsure" being the second-biggest pile is the
                  single most important thing to know about this measurement. */}
              {available.medium && (
                <div className="flex flex-col gap-1">
                  <FilterGroup label="Medium">
                    {shownMediums.map((b) => (
                      <Chip key={b.id} active={filter.medium === b.id}
                        onClick={() => setF({ medium: filter.medium === b.id ? null : b.id })}
                        title={mediumTitle(b.id)}>
                        {b.label} {mediumCounts[b.id] ?? 0}
                      </Chip>
                    ))}
                  </FilterGroup>
                  {mediumNote && (
                    <p className="m-0 pl-1 text-[11px] leading-snug text-content-subtle">{mediumNote}</p>
                  )}
                </div>
              )}

              {/* ⤢ Angle — measured in the pixels by the 🎭 Faces pass. The backfill
                  offer lives HERE rather than with the other passes: it only makes
                  sense next to the counts that explain why it exists, and it must be
                  priced before it is clicked. */}
              {available.angle && (
                <div className="flex flex-col gap-1">
                  <FilterGroup label="⤢ Angle">
                    {shownAngles.map((b) => (
                      <Chip key={b.id} active={filter.angle === b.id}
                        onClick={() => setF({ angle: filter.angle === b.id ? null : b.id })}
                        title={angleTitle(b.id)}>
                        {b.label} {angleCounts[b.id] ?? 0}
                      </Chip>
                    ))}
                  </FilterGroup>
                  {angleState.note && (
                    <p className="m-0 pl-1 text-[11px] leading-snug text-content-subtle">{angleState.note}</p>
                  )}
                  {angleState.offer && (
                    <p className="m-0 flex flex-wrap items-center gap-2 pl-1 text-[11px] leading-snug text-content-subtle">
                      <span>{angleState.offer.why}</span>
                      <button type="button" onClick={() => setPassOpen('angles')} disabled={!!live}
                        title={angleState.offer.why}
                        className="min-h-10 lg:min-h-0 rounded-md border border-border bg-surface-raised px-2 py-0.5 text-[11px] text-content transition-colors hover:bg-surface disabled:opacity-50">
                        ⤢ {angleState.offer.label}
                      </button>
                    </p>
                  )}
                </div>
              )}

              {/* Resolution tiers — one active at a time; re-click clears.
                  Composes with every filter and with the Sort menu below. */}
              {available.resolution && (
                <FilterGroup label="Resolution">
                  {shownResBuckets.map((b) => (
                    <Chip key={b.id} active={filter.resBucket === b.id}
                      onClick={() => setF({ resBucket: filter.resBucket === b.id ? null : b.id })}
                      title="Filter the grid to this resolution tier (megapixels = width×height)">
                      {b.label} {resBuckets[b.id] ?? 0}
                    </Chip>
                  ))}
                </FilterGroup>
              )}

              {/* Origin — ai / camera / unknown, read off the file's own metadata by
                  the quality scan. 'Unknown' is the usual answer and is deliberately
                  shown next to the other two: silence is not evidence of anything. */}
              {available.origin && (
                <FilterGroup label="Origin">
                  {ORIGIN_BUCKETS.map((b) => (
                    <Chip key={b.id} active={filter.origin === b.id}
                      onClick={() => setF({ origin: filter.origin === b.id ? null : b.id })}
                      title={b.id === 'unknown'
                        ? 'No metadata left in the file — the normal case for anything '
                          + 'scraped or sent through a chat app. NOT evidence that it is '
                          + 'a real photo.'
                        : (b.id === 'ai'
                          ? 'The file still carries generation metadata (ComfyUI workflow, '
                            + 'A1111 parameters, or a C2PA "generated" marker).'
                          : 'The file still carries camera EXIF (make, model or exposure).')}>
                      {b.label} {originCounts[b.id] ?? 0}
                    </Chip>
                  ))}
                </FilterGroup>
              )}
            </div>
          )}
        </div>
      )}

      {/* 🎚 The numbers BEHIND those chips. Folded by default — the chips are
          the everyday gesture and the thresholds are the occasional one — but
          present, because tuning them used to mean leaving the bank for
          Settings and coming back blind. Same config keys, same save. */}
      <div className="border-t border-border pt-2">
        <button type="button" onClick={() => setThresholdsOpen((v) => !v)}
          aria-expanded={thresholdsOpen} aria-controls="bank-thresholds-panel"
          className="min-h-10 lg:min-h-0 flex w-full items-center gap-1.5 text-left text-xs text-content-muted hover:text-content">
          <SlidersVertical aria-hidden="true" className="h-3.5 w-3.5" />
          <span className="font-medium">Filter thresholds</span>
          {/* The gloss is the first thing to go on a phone: the label already
              says what it opens, and a wrapped subtitle costs two lines. */}
          <span className="hidden text-content-subtle sm:inline">
            — what the chips above count as blurry, small, duplicate…
          </span>
          <span aria-hidden className="ml-auto text-content-subtle">{thresholdsOpen ? '▲' : '▼'}</span>
        </button>
        {thresholdsOpen && (
          <div id="bank-thresholds-panel" className="mt-2">
            {/* The panel's ↻ re-run buttons start the SAME one-job-per-bank
                passes as the toolbar above, so they need the same two facts
                the progress bar reads: whether a job holds the bank (to refuse
                the click before it is made, saying which pass and where it is)
                and the duplicate counts (to report what the pass produced).
                Both are already in this payload — no extra poll, no second
                progress mechanism. */}
            <BankThresholdsPanel bankId={bankId}
              activity={payload?.activity} offline={!connection.online}
              dupSummary={payload?.dup} semanticDupSummary={payload?.semantic_dup}
              onSaved={() => { refreshPayload(); refreshImages() }}
              onRunPass={onRunPass} />
          </div>
        )}
      </div>

      {/* View controls — order and tile size. */}
      <div className="flex flex-wrap items-center gap-2 border-t border-border pt-2">
        <GroupLabel>View</GroupLabel>
        <label className="flex min-w-0 flex-1 items-center gap-1 text-xs text-content-muted">
          Sort
          {/* Order the grid on ANY quantity the passes measured — resolution,
              file size, aesthetic rating, NSFW likelihood, sharpness, noise,
              contrast, detail, bars, JPEG quality, face confidence — each way,
              so a review opens on what it is looking for. Grouped by the pass
              that produces them: eleven measures is a long flat list, and the
              grouping doubles as "run THIS pass to unlock these".
              Images the matching pass never reached sink to the end (never the
              top), and an entry whose pass has produced nothing yet is greyed
              out saying which pass to run. The value rides to the server, which
              sorts in SQL: it applies to the WHOLE filter, not this page, so
              "Select all in filter" and ▶ Review walk the same order — and it
              is remembered per bank, so a dump you review by sharpness opens
              that way tomorrow. */}
          <select value={filter.sort} onChange={(e) => setSort(e.target.value)}
            title="Order the grid by anything the passes measured — resolution, size, aesthetic, NSFW, sharpness, noise, contrast, detail, bars, JPEG quality, face confidence. Images a pass never reached sink to the end. Remembered for this bank."
            aria-label="Sort the grid"
            className="min-w-0 flex-1 rounded-md border border-border bg-surface px-2 py-0.5 text-xs text-content">
            {sortGroups.map((g) => (g.group ? (
              <optgroup key={g.group} label={g.group}>
                {g.options.map((o) => (
                  <option key={o.id} value={o.id} disabled={o.disabled} title={o.title}>
                    {o.label}
                  </option>
                ))}
              </optgroup>
            ) : g.options.map((o) => (
              <option key={o.id} value={o.id} disabled={o.disabled} title={o.title}>
                {o.label}
              </option>
            ))))}
          </select>
        </label>
        <button type="button" onClick={() => setTileSize((s) => (s === 'M' ? 'S' : 'M'))}
          className="min-h-10 lg:min-h-0 rounded-md border border-border px-2 py-0.5 text-xs text-content-muted hover:text-content">
          {tileSize === 'M' ? 'Small tiles' : 'Medium tiles'}
        </button>
      </div>

      {/* 🏷️ The selection's caption tags — ONE mount now. The old screen carried
          two (a phone copy in the filter zone, a sticky desktop inspector) and
          hid one with `xl:hidden`; the rail answers that responsive problem for
          every control at once, so the duplicate is gone rather than moved. */}
      {/* Still a NAMED landmark. Folding the two mounts into one nearly cost the
          `aria-label="Image tags"` the desktop inspector carried — the surface
          contract caught it, which is exactly the deletion it exists to make
          loud. The panel moved; a screen reader must still be able to jump to
          it by name. */}
      {tagRow && (
        <section aria-label="Image tags" className="border-t border-border pt-2">
          <SelectionTagsPanel tagRow={tagRow} tagPicked={tagPicked}
            onToggle={(tag) => toggleTag(tag, tagRow)} onClear={clearTags} />
        </section>
      )}
    </aside>
  )
}
