import { useCallback, useMemo, useRef, useState } from 'react'
import { ArrowLeft, Clapperboard, Copy, Folder } from 'lucide-react'
import { Link, useSearchParams } from 'react-router'
import { postForm, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import { HelpBadge } from '../../help/HelpMode'
import { captionFrequencyEntries } from '../dataset/captionCategory'
import TrainingReadiness from '../dataset/TrainingReadiness'
import VideoTrainingBlock from './VideoTrainingBlock'
import VideoCheckpointManager from './VideoCheckpointManager'
import { videoPreflightUrl } from './videoCloudLaunch'
import VideoDatasetGrid from './VideoDatasetGrid'
import VideoDatasetLightbox from './VideoDatasetLightbox'
import {
  videoDatasetClipCaptionUrl, videoDatasetReferencesUrl, videoDatasetRemoveClipsUrl,
} from './videoBankApi'
import { toggleSelection, selectRange } from './videoTriage'
import { VIDEO_DATASET_SECTIONS } from './videoDatasetSections'
import {
  getVideoDatasetPanels, resolveVideoDatasetLocation, visibleVideoDatasetSections,
  withVideoDatasetLocation,
} from './videoDatasetNavigation'
import {
  CLIP_FILTERS, CLIP_SORTS, captionCoverageNote, clipCounts, clipFilterCount,
  hasCaption, lightboxTargets, purgeDraft, removeClipsConfirmation, removeClipsReport,
  visibleClips,
} from './videoDatasetClips'
import {
  captionEditConfirmation, captionEditPlan, captionEditProgressLabel, captionEditReport,
} from './videoDatasetCaptionTools'

/** 🎬 ONE video dataset, worked on — the surface this lane did not have.
 *
 * WHAT WAS MISSING AND WHY IT MATTERED. A video dataset used to be a CARD at the
 * bottom of the library: an accordion listing its clips, one textarea each, and
 * the training block. Everything you would actually work a set with — a grid, a
 * player, a search, bulk edits — existed only on the BANK, which triages SHOTS,
 * before any encode. So the object you are about to spend a night (or a pod
 * bill) training on was the one object in the app you could not look at
 * properly. The image lane has never worked that way, and CLAUDE.md's standing
 * rule is that a difference between the two surfaces is the maintainer's call,
 * not an accident of what got built first.
 *
 * WHAT IS DELIBERATELY NOT HERE, so nobody reads its absence as an oversight:
 *  · the quality passes (dedup, watermark, safe zone, defects) run on the bank's
 *    shots, on the SOURCE files, before an encode exists. Re-running them on
 *    encoded clips is a different pipeline, not a port;
 *  · trimming. A bank clip is a pair of bounds and a trim is free; a dataset clip
 *    is an encoded file at a fixed frame count, and shortening it means encoding
 *    it again. The honest gesture is to re-cut in the bank, which is why the
 *    lightbox names the source rush and the timecode;
 *  · an export section. A video dataset IS its output folder — a flat directory
 *    of .mp4 and homonym .txt, which is exactly what every trainer reads.
 */
export default function VideoDatasetWorkspace({ ds, items, refresh, onBack }) {
  const toast = useToast()
  const [searchParams, setSearchParams] = useSearchParams()

  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState('all')
  const [sort, setSort] = useState('filename')
  const [selected, setSelected] = useState([])
  const anchor = useRef(null)
  const [openId, setOpenId] = useState(null)
  // Caption drafts, keyed by clip id. Held here rather than in the lightbox so
  // stepping to the next clip and back does not lose text that was never blurred.
  const [drafts, setDrafts] = useState({})
  const [savingId, setSavingId] = useState(null)
  const [bulk, setBulk] = useState(null)        // {done,total} while a plan replays
  // Only ever consulted on a fold under 500 px (a phone held sideways), where
  // the clip toolbar is folded away to give the grid the screen back.
  const [toolsOpen, setToolsOpen] = useState(false)
  // The training block's polls report how many saves exist; the Checkpoints &
  // LoRAs section re-reads on that number, so a harvest shows up there without
  // a poll of its own. The other way round, a delete in that section bumps
  // `trainingRefresh` so the block's "Train further" stops offering a run that
  // is gone.
  const [saveCount, setSaveCount] = useState(0)
  const [trainingRefresh, setTrainingRefresh] = useState(0)

  const counts = useMemo(() => clipCounts(items), [items])
  const shown = useMemo(() => visibleClips(items, { query, filter, sort }),
    [items, query, filter, sort])
  const shownIds = useMemo(() => shown.map((c) => c.id), [shown])

  const navContext = {
    selected: selected.length,
    clips: counts.total,
    requiresReferences: !!ds.requires_references,
  }
  const sections = visibleVideoDatasetSections(navContext)
  const location = resolveVideoDatasetLocation(searchParams, navContext)
  const section = location.section

  const setSection = useCallback((id, panelId = null) => {
    setSearchParams(withVideoDatasetLocation(searchParams, id, panelId),
      { replace: true })
    if (panelId) {
      const target = VIDEO_DATASET_SECTIONS.find((s) => s.id === id)
        ?.panels.find((p) => p.id === panelId)
      // After the section swap, not before: the anchor is inside a section that
      // is `hidden` until this render lands, and scrollIntoView on a display:none
      // element silently does nothing.
      if (target) {
        requestAnimationFrame(() => {
          document.getElementById(target.targetId)
            ?.scrollIntoView({ block: 'start', behavior: 'smooth' })
        })
      }
    }
  }, [searchParams, setSearchParams])

  const draftOf = (clip) => drafts[clip.id] ?? clip.caption ?? ''

  const saveCaption = useCallback(async (clip) => {
    const caption = drafts[clip.id]
    if (caption === undefined || caption === (clip.caption ?? '')) return
    setSavingId(clip.id)
    try {
      const d = await postJson(videoDatasetClipCaptionUrl(ds.id, clip.id), { caption })
      // Not a detail, and not a toast we can skip: the trainer reads the FILE. A
      // row saved without its sidecar trains the previous text with nothing on
      // screen to reveal it.
      if (!d.sidecar_written) {
        toast.warning('Caption saved in the app, but its .txt file could not be written — the trainer reads the file.')
      }
      await refresh()
      // Only if the draft is still what was posted: anything typed during the
      // two awaits above lives in the draft alone, and dropping it would throw
      // the user's text away in silence (purgeDraft says why in full).
      setDrafts((m) => purgeDraft(m, clip.id, caption))
    } catch (e) {
      toast.error(e?.message || 'Could not save that caption.')
    } finally {
      setSavingId(null)
    }
  }, [drafts, ds.id, refresh, toast])

  const removeClips = useCallback(async (ids) => {
    const doomed = items.filter((c) => ids.includes(c.id))
    // A stills set has no bank behind it at all (its rows are written straight
    // from an image dataset, with no source_bank_id), so the promise that makes
    // this a safe click is not true there and must not be printed there.
    const text = removeClipsConfirmation(doomed.map((c) => c.filename),
      { fromBank: doomed.some((c) => c.source_clip_id), mode: ds.delete_mode })
    if (!text || !window.confirm(text)) return
    try {
      const d = await postJson(videoDatasetRemoveClipsUrl(ds.id), { ids })
      // files_kept is the answer nobody expects: the row is still there because
      // the FILE would not move, and the folder is what the trainer reads. A
      // plain "removed" toast over that is the same lie as a caption saved
      // without its sidecar — so it gets the same warning treatment.
      toast[d.files_kept ? 'warning' : 'success'](removeClipsReport(d))
      // The server does not say WHICH clips it kept, so when it kept any the
      // selection stays as it was: the toast says "try again", and clearing the
      // selection would take away the very thing to try again with.
      if (!d.files_kept) {
        setSelected((list) => list.filter((id) => !ids.includes(id)))
        setOpenId((id) => (ids.includes(id) ? null : id))
      }
      await refresh()
    } catch (e) {
      toast.error(e?.message || 'Could not remove those clips.')
    }
  }, [items, ds.id, ds.delete_mode, refresh, toast])

  /** Replay a caption plan over the existing per-clip route. One request per
   * clip is the price of not inventing a bulk endpoint this wave, and it buys
   * something real: each clip's .txt is rewritten by the same code path that
   * writes it when you edit one by hand, so the two cannot drift. */
  const applyCaptionOp = useCallback(async (op) => {
    const scope = selected.length ? items.filter((c) => selected.includes(c.id)) : items
    const plan = captionEditPlan(scope, op)
    const confirmText = captionEditConfirmation(plan, op)
    if (!confirmText) { toast.info('Nothing to change — no caption matches.'); return }
    if (!window.confirm(confirmText)) return
    let changed = 0
    let sidecarFailed = 0
    let failed = 0
    const written = []
    setBulk({ done: 0, total: plan.length })
    for (const entry of plan) {
      try {
        const d = await postJson(videoDatasetClipCaptionUrl(ds.id, entry.id),
          { caption: entry.after })
        // THREE outcomes, not two. The server commits the row BEFORE it writes
        // the sidecar, so `sidecar_written: false` means "the app has the new
        // text, the .txt still has the old one" — the opposite of a request that
        // threw, where nothing moved at all. Counting them together made the
        // report say "the failed ones still hold their previous text" about
        // clips whose row had already changed.
        written.push({ id: entry.id, after: entry.after })
        if (d.sidecar_written) changed += 1
        else sidecarFailed += 1
      } catch { failed += 1 }
      setBulk((b) => (b ? { ...b, done: b.done + 1 } : b))
    }
    setBulk(null)
    // Drop the drafts of every clip this pass rewrote. Without it a draft left
    // behind by an earlier edit keeps masking the server's value, and one click
    // in and out of that box reposts the stale text — silently undoing the bulk
    // rewrite on disk. Measured.
    if (written.length) {
      // Same rule as a single save: a draft is dropped only if it still holds
      // the value that was posted for it — the user may have typed into a box
      // while the loop ran (the per-clip textareas are never disabled).
      setDrafts((m) => written.reduce((acc, { id, after }) => purgeDraft(acc, id, after), m))
    }
    const report = captionEditReport({ changed, sidecarFailed, failed })
    if (failed || sidecarFailed) toast.warning(report)
    else toast.success(report)
    await refresh()
  }, [selected, items, ds.id, refresh, toast])

  // Two lists, one rule, stated once in videoDatasetClips.lightboxTargets: the
  // clip comes from the FULL set, the stepping comes from the filtered one. The
  // ref remembers the slot the open clip held in that filtered list, so that
  // once it leaves the list (captioned, under the "No caption" filter) the
  // arrows resume from where it was instead of going dead.
  const lastIndex = useRef(-1)
  const player = lightboxTargets(items, shown, openId, lastIndex.current)
  if (player.index >= 0) lastIndex.current = player.index

  const toggle = (clip, event) => {
    if (event?.shiftKey && anchor.current != null) {
      setSelected((list) => selectRange(list, shownIds, anchor.current, clip.id))
    } else {
      setSelected((list) => toggleSelection(list, clip.id))
      anchor.current = clip.id
    }
  }

  const sectionCls = (id) => (section === id ? 'flex flex-col gap-3' : 'hidden')
  const sectionMeta = Object.fromEntries(VIDEO_DATASET_SECTIONS.map((s) => [s.id, s]))

  const navItem = (s, chip) => {
    const isActive = s.id === section
    const base = chip
      ? `flex min-h-10 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full border px-3 py-1.5 text-xs font-medium ${
        isActive ? 'border-border-strong bg-surface-raised text-content' : 'border-border text-content-muted hover:text-content'}`
      : `relative flex w-full items-center gap-2.5 rounded-md px-3 py-2 text-left text-sm font-medium ${
        isActive ? 'bg-surface-raised text-content' : 'text-content-muted hover:bg-surface hover:text-content'}`
    return (
      <button type="button" onClick={() => setSection(s.id)}
        aria-current={isActive ? 'page' : undefined}
        data-mobile-section={chip ? s.id : undefined}
        className={base}>
        {!chip && isActive && (
          <span aria-hidden className="absolute bottom-1.5 left-0 top-1.5 w-0.5 rounded bg-gradient-primary" />
        )}
        <s.icon aria-hidden="true" className="h-4 w-4 shrink-0" />
        <span>{s.title}</span>
      </button>
    )
  }

  const panelNavItem = (sectionId, destination, chip) => {
    const base = chip
      ? 'flex min-h-10 shrink-0 items-center rounded-full border border-border px-3 py-1.5 text-xs text-content-muted hover:text-content'
      : 'relative flex w-full items-center rounded-md px-4 py-1.5 text-left text-xs text-content-subtle hover:bg-surface hover:text-content-muted'
    return (
      <button type="button" onClick={() => setSection(sectionId, destination.id)}
        data-mobile-panel={chip ? destination.id : undefined}
        className={base}>
        {destination.title}
      </button>
    )
  }

  const activePanels = getVideoDatasetPanels(section, navContext)

  const heading = (id) => {
    const s = sectionMeta[id]
    return (
      <div className="flex flex-col gap-0.5 border-b border-border pb-2">
        <p className="m-0 font-mono text-[11px] uppercase tracking-[0.18em] text-content-subtle">
          {s.eyebrow}
        </p>
        <h2 className="flex items-center gap-2 text-base font-semibold text-content">
          {s.title}
          {/* Read off the section, not built from its id. The registry contract
              test can only see topic names written as literal strings, so an
              interpolated one would let a renamed section ship a ? badge
              pointing at nothing — green. Test (5) checks these by name. */}
          <HelpBadge topic={s.helpTopic} />
        </h2>
        <p className="m-0 max-w-3xl text-xs text-content-muted">{s.description}</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3">
      <div data-probe-chrome="header" className="relative z-30 flex flex-wrap items-center gap-x-2 gap-y-1">
        <button type="button" onClick={onBack}
          className="flex min-h-10 items-center gap-1 rounded-lg border border-border bg-surface px-3 py-1.5 text-sm text-content-muted transition-colors hover:bg-surface-raised hover:text-content lg:min-h-0">
          <ArrowLeft aria-hidden="true" className="h-4 w-4" /> Datasets
        </button>
        <h1 className="flex items-center gap-2 font-bold text-content">
          <Clapperboard aria-hidden="true" className="h-5 w-5" />{ds.name}
        </h1>
        <span className="rounded-full border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-amber-600 dark:text-amber-400">
          Beta
        </span>
        <HelpBadge topic="page-video-dataset" />
        {ds.trigger_word && (
          <button type="button"
            onClick={() => { try { navigator.clipboard.writeText(ds.trigger_word) } catch { /* denied */ } }}
            title="Copy the trigger word — it is prepended to every sidecar at write time"
            className="flex min-h-10 items-center gap-1 rounded-lg border border-indigo-400/40 bg-indigo-500/10 px-2 py-0.5 text-[0.6875rem] lg:min-h-0">
            <span className="text-content-subtle">trigger:</span>
            <code className="font-semibold text-indigo-300">{ds.trigger_word}</code>
            <Copy aria-hidden="true" className="h-3 w-3 text-content-subtle" />
          </button>
        )}
        {/* Desktop only, and measured: on a 360 px phone this line is a THIRD
            header row, and the fold has 28% to spend on everything fixed before
            the user has asked for anything (responsiveProbe's resting budget).
            Nothing is lost — the clip count is under the grid, and the target,
            frames and size are all on the library card and in Training. */}
        {/* 🔎 On a phone held sideways there are 390 px of fold, and a header
            plus a section rail already spend a quarter of it. So the clip
            toolbar folds away there and comes back from here — one tap, nothing
            lost, and the rest of the screen goes to the clips. Everywhere else
            this button does not exist and the toolbar is simply open. */}
        <button type="button" onClick={() => setToolsOpen((v) => !v)}
          aria-expanded={toolsOpen} aria-controls="vds-clips-tools"
          className="ml-auto hidden min-h-10 items-center gap-1 rounded-lg border border-border bg-surface px-3 py-1.5 text-xs text-content-muted hover:text-content [@media(max-height:500px)]:inline-flex">
          🔎 Filter & sort
        </button>
        <span className="ml-auto hidden text-xs text-content-muted sm:inline [@media(max-height:500px)]:hidden">
          {counts.total} clip{counts.total === 1 ? '' : 's'} · {ds.target_label}
          {ds.frames ? ` · ${ds.frames} frames` : ''}
          {ds.fps ? ` @ ${ds.fps} fps` : ''}
          {ds.width && ds.height ? ` · ${ds.width}×${ds.height}` : ' · source size'}
        </span>
      </div>

      {/* The two facts that decide whether this set is worth a night, kept at the
          top of the workspace and not only on the library card: you come BACK to
          a dataset weeks later, and "can I train this, and am I allowed to
          publish from it where I live" is exactly the question you have then. */}
      {!ds.training_verified && (
        <p className="rounded border border-amber-500/50 bg-amber-500/10 px-3 py-1.5 text-xs text-amber-100">
          ⚠ No LoRA trainer is known to exist for {ds.target_label} yet — the rail is
          wired from ai-toolkit’s own settings but has never been proven by a run.
        </p>
      )}
      {ds.licence_note && (
        <p className="rounded border border-rose-500/60 bg-rose-500/10 px-3 py-1.5 text-xs text-rose-100">
          ⚖ {ds.licence_note}
        </p>
      )}

      <ClipsFolderNote path={ds.output_dir} />

      <div className="lg:grid lg:grid-cols-[15rem_minmax(0,1fr)] lg:gap-4 lg:items-start">
        <aside>
          {/* `relative` is load-bearing on this rail, not decoration: an
              overflow-x-auto box only clips a descendant when it is also its
              containing block, and a static one never is. The bank workspace
              paid for that lesson with a page rendering at 73% on a phone. */}
          <nav aria-label="Video dataset sections" data-probe-chrome="sections"
            className="relative -mx-4 overflow-x-auto px-4 pb-1 lg:hidden">
            <ul className="m-0 flex list-none gap-2 p-0">
              {sections.map((s) => <li key={s.id}>{navItem(s, true)}</li>)}
            </ul>
          </nav>
          {/* TWO destinations or none: a rail holding a single chip is a 44 px
              band of fixed chrome that navigates to the section you are already
              looking at. At rest that is exactly the case here — "Bulk actions"
              needs a selection, "Caption tools" needs a caption — so the rail
              would appear on every phone, empty of purpose, before the user has
              done anything. */}
          {activePanels.length > 1 && (
            <nav aria-label={`${sectionMeta[section].title} destinations`}
              data-probe-chrome="destinations"
              className="relative -mx-4 overflow-x-auto px-4 pb-1 lg:hidden [@media(max-height:500px)]:hidden">
              <ul className="m-0 flex list-none gap-2 p-0">
                {activePanels.map((d) => <li key={d.id}>{panelNavItem(section, d, true)}</li>)}
              </ul>
            </nav>
          )}
          <div data-probe-panel="sections-rail"
            className="hidden lg:sticky lg:top-20 lg:flex lg:flex-col lg:gap-3">
            <nav aria-label="Video dataset sections">
              <p className="m-0 px-3 pb-2 font-mono text-[11px] uppercase tracking-[0.18em] text-content-subtle">
                Video dataset
              </p>
              <ul className="m-0 flex list-none flex-col gap-0.5 p-0">
                {sections.map((s) => (
                  <li key={s.id}>
                    {navItem(s, false)}
                    {s.id === section && activePanels.length > 1 && (
                      <ul className="m-0 ml-4 flex list-none flex-col gap-0.5 border-l border-border p-0 py-1 pl-1">
                        {activePanels.map((d) => (
                          <li key={d.id}>{panelNavItem(s.id, d, false)}</li>
                        ))}
                      </ul>
                    )}
                  </li>
                ))}
              </ul>
            </nav>
          </div>
        </aside>

        <div className="mt-1 flex min-w-0 flex-col gap-3 lg:mt-0">
          {/* Inactive sections stay MOUNTED and hidden, exactly as the image
              workspace does it: the training block's poll advances a real run
              server-side, and it must not stop because someone looked at the
              clip grid. */}
          <section className={sectionCls('clips')} aria-hidden={section !== 'clips'}>
            {heading('clips')}
            <div id="vds-clips-review" className="flex flex-col gap-2">
              <div id="vds-clips-tools"
                className={`flex flex-col gap-2 ${toolsOpen ? '' : '[@media(max-height:500px)]:hidden'}`}>
              {/* ONE row, search and sort together. They were two, which on a
                  phone is two 40 px rows of fixed chrome for one question. The
                  sort keeps the FORM its neighbour on the image side uses — a
                  labelled <select> next to the search box (GridSortSelect in
                  DatasetWorkspace) — because the same kind of control should
                  have the same look across the two surfaces. */}
              <div data-probe-chrome="grid-toolbar" className="flex flex-wrap items-center gap-2">
                {/* `basis-0` is the whole reason this is ONE row: flex wraps on
                    BASE sizes and only then grows what is left, so with the
                    input's base at `auto` (~170 px for a search field) the sort
                    could not share the line at 360 px and this toolbar doubled
                    to 88 px of fixed chrome. Measured, not reasoned. */}
                <input type="search" value={query} onChange={(e) => setQuery(e.target.value)}
                  placeholder="Filter by file name, caption or source rush"
                  aria-label="Filter the clips"
                  className="min-h-10 min-w-32 grow basis-0 rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content lg:min-h-0" />
                <label className="flex shrink-0 items-center gap-1 text-xs text-content-subtle">
                  {/* The word costs 30 px and a phone cannot afford it: with it,
                      this row wraps and the toolbar doubles to 88 px of fixed
                      chrome. The select keeps its accessible name either way. */}
                  <span className="hidden sm:inline">Sort</span>
                  <select value={sort} onChange={(e) => setSort(e.target.value)}
                    aria-label="Sort the clips"
                    className="min-h-10 max-w-36 rounded-md border border-border bg-surface-raised px-2 py-1 text-xs text-content lg:min-h-0">
                    {CLIP_SORTS.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
                  </select>
                </label>
              </div>
              <div data-probe-chrome="filter-bar" className="flex flex-wrap items-center gap-1.5">
                {CLIP_FILTERS.map((f) => (
                  <button key={f.id} type="button" onClick={() => setFilter(f.id)}
                    aria-pressed={filter === f.id}
                    className={`min-h-10 rounded-full border px-3 py-0.5 text-[0.6875rem] font-semibold tabular-nums lg:min-h-0 ${
                      filter === f.id
                        ? 'border-border-strong bg-surface-raised text-content'
                        : 'border-border text-content-muted hover:text-content'}`}>
                    {f.label} ({clipFilterCount(items, f.id)})
                  </button>
                ))}
              </div>
              </div>
              {/* The counts and the coverage in ONE line of CONTENT, below the
                  chrome. They used to be two, one of them wrapping the filter
                  rail onto a second row on a phone — and the second row said
                  what this one says. */}
              <p className="text-xs text-content-muted">
                {shown.length === items.length
                  ? `${counts.total} clips · ${counts.seconds}s of footage. `
                  : `${shown.length} of ${counts.total} shown. `}
                {captionCoverageNote(counts, ds.trigger_word)}
              </p>
              <VideoDatasetGrid datasetId={ds.id} clips={shown} selected={selected}
                onToggle={toggle} onOpen={(clip) => setOpenId(clip.id)}
                emptyMessage={items.length
                  ? 'No clip matches this filter.'
                  : 'This dataset has no clip — promote shots from a video bank, or build a stills set from an image dataset.'} />
            </div>
            {selected.length > 0 && (
              <div id="vds-clips-bulk"
                className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2">
                <span className="text-xs font-semibold text-content">
                  {selected.length} selected
                </span>
                <button type="button" onClick={() => setSelected(shownIds)}
                  className="min-h-10 rounded border border-border bg-surface-raised px-2 py-1 text-[0.6875rem] text-content-muted hover:bg-surface lg:min-h-0">
                  Select all shown ({shown.length})
                </button>
                <button type="button" onClick={() => setSelected([])}
                  className="min-h-10 rounded border border-border bg-surface-raised px-2 py-1 text-[0.6875rem] text-content-muted hover:bg-surface lg:min-h-0">
                  Clear
                </button>
                <button type="button" onClick={() => removeClips(selected)}
                  className="ml-auto min-h-10 rounded border border-border bg-surface-raised px-2 py-1 text-[0.6875rem] font-semibold text-content hover:border-rose-500/60 hover:text-rose-300 lg:min-h-0">
                  🗑 Remove from dataset
                </button>
              </div>
            )}
          </section>

          <section className={sectionCls('captions')} aria-hidden={section !== 'captions'}>
            {heading('captions')}
            <div id="vds-captions-list" className="flex flex-col gap-2">
              <p className="text-xs text-content-muted">
                {captionCoverageNote(counts, ds.trigger_word)}
              </p>
              <ul className="flex flex-col gap-2">
                {shown.map((clip) => (
                  <li key={clip.id} className="min-w-0 rounded-lg border border-border bg-surface p-2">
                    <div className="flex min-w-0 flex-wrap items-center gap-1.5">
                      <button type="button" onClick={() => setOpenId(clip.id)}
                        className="min-h-10 rounded border border-border bg-surface-raised px-1.5 py-0.5 text-[0.625rem] font-semibold text-content hover:bg-surface lg:min-h-0">
                        ▶ Open
                      </button>
                      <span className="min-w-0 truncate font-mono text-[0.625rem] text-content-subtle"
                        title={clip.src_relpath || clip.filename}>
                        {clip.filename}
                      </span>
                      {!hasCaption(clip) && (
                        <span className="rounded bg-amber-600/80 px-1 text-[0.625rem] font-bold text-white">
                          no caption
                        </span>
                      )}
                    </div>
                    <textarea rows={2} value={draftOf(clip)}
                      onChange={(e) => setDrafts((m) => ({ ...m, [clip.id]: e.target.value }))}
                      onBlur={() => saveCaption(clip)}
                      aria-label={`Caption for ${clip.filename}`}
                      placeholder="Describe the clip — this is written to the .txt next to it."
                      className="mt-1 w-full rounded border border-border bg-surface-raised px-2 py-1 text-[0.6875rem] text-content" />
                    {/* The space is RESERVED, not inserted. Blur is a discrete
                        event, so React flushes this before the browser delivers
                        the mouseup: a line appearing here pushed everything
                        below down by its own height mid-click, and the click
                        landed beside the Caption tools button it was aimed at
                        (measured: +15 px between mousedown and mouseup). */}
                    <p aria-live="polite"
                      className="min-h-4 text-[0.625rem] text-content-subtle">
                      {savingId === clip.id ? 'Saving…' : ''}
                    </p>
                  </li>
                ))}
              </ul>
            </div>
            {/* Gated on CLIPS, not on captions. The prefix operation is
                written for the silent ones ("a trigger-style prefix belongs
                on the silent ones too"), so hiding the whole panel until
                something already had a caption hid it from the one set it
                was designed for: a freshly promoted, entirely uncaptioned
                one. */}
            {counts.total > 0 && (
              <CaptionTools id="vds-captions-tools" clips={items} selected={selected}
                busy={bulk} onApply={applyCaptionOp} />
            )}
          </section>

          {ds.requires_references && (
            <section className={sectionCls('references')} aria-hidden={section !== 'references'}>
              {heading('references')}
              <div id="vds-references-attach">
                <ReferenceAttach ds={ds} onChanged={refresh} />
              </div>
            </section>
          )}

          <section className={sectionCls('training')} aria-hidden={section !== 'training'}>
            {heading('training')}
            <div id="vds-training-launch" className="flex flex-col gap-2">
              {/* The image lane's readiness card, reading the VIDEO preflight —
                  one card, two lanes, the parity rule made literal. Local lane
                  here (this machine's toolkit and weights); the cloud lane's
                  report is asked by the launch window, right before the money.
                  Fix → jumps to the section a row names. */}
              <TrainingReadiness datasetId={ds.id}
                endpoint={videoPreflightUrl(ds.id, 'local')}
                refreshKey={`${counts.total}:${ds.references}:${ds.target_profile}`}
                onJump={(target) => setSection(target)} />
              <VideoTrainingBlock ds={{ ...ds, clips: counts.total }}
                onSaveCount={setSaveCount} refreshKey={trainingRefresh} />
            </div>
          </section>

          <section className={sectionCls('checkpoints')} aria-hidden={section !== 'checkpoints'}>
            {heading('checkpoints')}
            <div id="vds-checkpoints-manager" className="flex flex-col gap-2">
              <VideoCheckpointManager ds={ds} refreshKey={saveCount}
                onSavesChange={() => setTrainingRefresh((n) => n + 1)} />
            </div>
          </section>

          <section className={sectionCls('studio')} aria-hidden={section !== 'studio'}>
            {heading('studio')}
            <div id="vds-studio-launcher"
              className="flex flex-col gap-2 rounded-lg border border-border bg-surface-raised p-3">
              {/* A launcher and not the Studio itself — the image workspace makes
                  the same choice: the Studio is a page (queues, a picker across
                  every dataset's LoRAs), and it opens on its Video tab here. */}
              <p className="m-0 max-w-3xl text-xs text-content-muted">
                Deploy a save from Checkpoints &amp; LoRAs first, then pick it in the Studio’s
                Video tab: one clip per setting, the same prompt, so the LoRA is judged on what
                it renders.
              </p>
              <Link to="/studio?lane=video"
                className="inline-flex w-fit min-h-10 items-center gap-1.5 rounded-md border border-primary/40 bg-primary/20 px-3 py-1.5 text-xs font-medium text-white no-underline hover:bg-primary/30 lg:min-h-0">
                ⤢ Open Studio
              </Link>
            </div>
          </section>
        </div>
      </div>

      {player.clip && (
        <VideoDatasetLightbox datasetId={ds.id} clip={player.clip}
          caption={draftOf(player.clip)}
          onCaptionChange={(text) => setDrafts((m) => ({ ...m, [player.clip.id]: text }))}
          onSave={() => saveCaption(player.clip)}
          saving={savingId === player.clip.id}
          onClose={() => setOpenId(null)}
          onPrev={() => setOpenId(player.prevId ?? openId)}
          onNext={() => setOpenId(player.nextId ?? openId)}
          onRemove={(clip) => removeClips([clip.id])}
          hasPrev={player.prevId != null}
          hasNext={player.nextId != null} />
      )}
    </div>
  )
}

/** Where the clips are, and the one warning that is TRUE of this folder.
 *
 * Not DatasetFolderNote: its second paragraph is about image banks and "Import
 * to bank below", neither of which exists here. Identical wording for different
 * behaviour is the failure mode CLAUDE.md names — so this says the thing that is
 * actually load-bearing for a video set, which the routes' own docstring
 * states: the trainer walks this directory recursively, so anything you leave in
 * it is trained on. */
function ClipsFolderNote({ path }) {
  const [copied, setCopied] = useState(false)
  if (!path) return null
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(path)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch { /* clipboard denied — the path is still readable on screen */ }
  }
  return (
    <div className="min-w-0 rounded-lg border border-border bg-surface px-3 py-2 text-xs">
      <div className="flex min-w-0 items-center gap-2">
        <span className="inline-flex shrink-0 items-center gap-1.5 text-content-subtle">
          <Folder aria-hidden="true" className="h-3.5 w-3.5" /> Clips folder
        </span>
        <code className="min-w-0 grow truncate font-mono text-content-muted" title={path}>
          {path}
        </code>
        <button type="button" onClick={copy}
          aria-label="Copy the dataset's clips folder path"
          className="shrink-0 rounded border border-border px-2 py-0.5 text-content-muted hover:bg-surface-raised hover:text-content">
          {copied ? '✓ Copied' : '⧉ Copy'}
        </button>
      </div>
      <p className="mt-1 text-content-subtle">
        This folder IS the dataset: a flat directory of .mp4 files with homonym .txt
        captions. Trainers scan it recursively, so anything else you leave in here is
        trained on without a word.
      </p>
    </div>
  )
}

/** 📝 Bulk caption edits. The plan is computed and CONFIRMED before anything is
 * written, because every entry in it is a file rewritten on disk. */
function CaptionTools({ id, clips, selected, busy, onApply }) {
  const [find, setFind] = useState('')
  const [replace, setReplace] = useState('')
  const [wholeWord, setWholeWord] = useState(true)
  const [affix, setAffix] = useState('')
  const scope = selected.length ? clips.filter((c) => selected.includes(c.id)) : clips
  const freq = useMemo(
    () => captionFrequencyEntries(scope.map((c) => c.caption).filter(Boolean), 'prose'),
    [scope])

  return (
    <div id={id} className="flex flex-col gap-2 rounded-lg border border-border bg-surface px-3 py-2">
      <p className="text-xs font-semibold text-content">
        Caption tools
        <span className="ml-1.5 font-normal text-content-subtle">
          {selected.length
            ? `— applied to the ${selected.length} selected clip${selected.length === 1 ? '' : 's'}`
            : `— applied to all ${clips.length} clips`}
        </span>
      </p>
      <div className="flex flex-wrap items-end gap-1.5">
        <label className="flex flex-col text-[0.625rem] text-content-subtle">
          Find
          <input value={find} onChange={(e) => setFind(e.target.value)}
            className="min-h-10 rounded border border-border bg-surface-raised px-2 py-1 text-[0.6875rem] text-content lg:min-h-0" />
        </label>
        <label className="flex flex-col text-[0.625rem] text-content-subtle">
          Replace with (empty removes it)
          <input value={replace} onChange={(e) => setReplace(e.target.value)}
            className="min-h-10 rounded border border-border bg-surface-raised px-2 py-1 text-[0.6875rem] text-content lg:min-h-0" />
        </label>
        <label className="flex items-center gap-1 text-[0.625rem] text-content-muted">
          <input type="checkbox" checked={wholeWord}
            onChange={(e) => setWholeWord(e.target.checked)}
            className="h-3.5 w-3.5 accent-indigo-500" />
          whole word
        </label>
        <button type="button" disabled={!find.trim() || !!busy}
          onClick={() => onApply({ kind: 'replace', find, replace, wholeWord })}
          className="min-h-10 rounded border border-border bg-surface-raised px-2 py-1 text-[0.6875rem] font-semibold text-content hover:bg-surface disabled:opacity-40 lg:min-h-0">
          Replace
        </button>
      </div>
      <div className="flex flex-wrap items-end gap-1.5">
        <label className="flex flex-col text-[0.625rem] text-content-subtle">
          Text to add
          <input value={affix} onChange={(e) => setAffix(e.target.value)}
            className="min-h-10 rounded border border-border bg-surface-raised px-2 py-1 text-[0.6875rem] text-content lg:min-h-0" />
        </label>
        <button type="button" disabled={!affix.trim() || !!busy}
          onClick={() => onApply({ kind: 'prefix', text: affix })}
          title="Added in front of every caption — including the clips that have none"
          className="min-h-10 rounded border border-border bg-surface-raised px-2 py-1 text-[0.6875rem] text-content hover:bg-surface disabled:opacity-40 lg:min-h-0">
          Add as prefix
        </button>
        <button type="button" disabled={!affix.trim() || !!busy}
          onClick={() => onApply({ kind: 'suffix', text: affix })}
          title="Appended to captions that already say something — never to an empty one"
          className="min-h-10 rounded border border-border bg-surface-raised px-2 py-1 text-[0.6875rem] text-content hover:bg-surface disabled:opacity-40 lg:min-h-0">
          Add as suffix
        </button>
      </div>
      {busy && (
        <p className="text-[0.6875rem] text-content-muted">
          {captionEditProgressLabel(busy.done, busy.total)}
        </p>
      )}
      {freq.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {/* The same frequency read the image lane offers, over the same helper —
              a term that shows up in every caption is a term the LoRA will bind
              to the trigger whether or not you meant it to. */}
          <span className="text-[0.625rem] text-content-subtle">Most repeated words:</span>
          {freq.slice(0, 12).map(([term, n]) => (
            <button key={term} type="button" onClick={() => setFind(term)}
              title={`In ${n} caption${n === 1 ? '' : 's'} — click to put it in the Find field`}
              className="rounded border border-border bg-surface-raised px-1.5 py-0.5 text-[0.625rem] text-content-muted hover:text-content">
              {term} <span className="tabular-nums text-content-subtle">{n}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

/** 📎 Identity references for a ref2va dataset — the launch precondition.
 *
 * The trainer reads these as control images; without them it trains
 * unconditioned in silence, so the server refuses a reference-less launch and
 * this is how the user satisfies it. Replacing is whole-set: refs are one
 * identity, not an album. */
function ReferenceAttach({ ds, onChanged }) {
  const toast = useToast()
  const [busy, setBusy] = useState(false)

  const upload = async (fileList) => {
    const files = Array.from(fileList || [])
    if (!files.length) return
    setBusy(true)
    try {
      const form = new FormData()
      files.forEach((f) => form.append('files', f))
      // postForm, NEVER a bare apiFetch with a FormData body. CSRFProtect is on
      // for the whole app and exempts no blueprint, so a multipart POST without
      // the token is refused with a 400 the view never sees — and the retry in
      // fetchClient cannot rescue it either (it is keyed on an X-CSRFToken
      // header that is not there), so the user reads "refresh the page" in a
      // loop. Measured on a live app with WTF_CSRF_ENABLED=True: 400 text/html
      // without the token, 404 application/json with it. The backend suite is
      // blind to this — conftest builds the app with CSRF disabled.
      const r = await postForm(videoDatasetReferencesUrl(ds.id), form)
      toast.success(`${r.references} reference(s) attached — every clip covered.`)
      await onChanged?.()
    } catch (e) {
      toast.error(e?.message || 'Could not attach the references.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2">
      <span className={`text-xs ${ds.references > 0 ? 'text-content-muted' : 'text-amber-300'}`}>
        📎 References: {ds.references || 0}{ds.references > 0 ? '' : ' — required, the launch is refused without them'}
      </span>
      <label className="min-h-10 cursor-pointer rounded border border-border bg-surface-raised px-2 py-1 text-[0.6875rem] font-semibold text-content hover:bg-surface lg:min-h-0">
        <input type="file" multiple accept="image/*" hidden disabled={busy}
          onChange={(e) => { upload(e.target.files); e.target.value = '' }} />
        {busy ? 'Attaching…' : ds.references > 0 ? 'Replace them' : 'Attach 1-4 images'}
      </label>
    </div>
  )
}
