import { useCallback, useEffect, useMemo, useState } from 'react';
import { FolderOpen, Images } from 'lucide-react'
import { apiFetch, postJson } from '../../api/fetchClient';
import {
  allGalleryImageIds, galleryActionBar, galleryDeleteConfirmation, galleryDeleteSummary,
  galleryTilePin, pruneGallerySelection, toggleGalleryImage,
} from '../../utils/gallerySelection';
import {
  checkpointNotes, defaultOpenGroups, galleryEndpoints, galleryHeading,
  galleryScope, galleryTargetKey, runGalleryGroups, runGallerySummary,
  stepGroupLabel, unlinkedNote, visibleGalleryImages,
} from '../../utils/runGallery';
import { imageFactsLine } from '../../utils/generatedImageFacts';
import {
  galleryZipPlanUrl, galleryZipUrl, planNotice, zipButtonState,
} from '../../utils/galleryDownload';
import { canImproveCanvasImage } from '../../utils/canvasImprove';
import { useCanvasImageImprove } from '../../hooks/useCanvasImageImprove';
import { useRestoreImproveSettings } from '../../hooks/useRestoreImproveSettings';
import { configRows } from '../dataset/lineageDetail.js';
import RunDeleteSection from './RunDeleteSection';
import GeneratedImageLightbox from './GeneratedImageLightbox';
import CheckpointTimelinePanel from './CheckpointTimelinePanel';

/* 🖼 Everything one checkpoint — or one whole RUN — ever produced.

   Images used to be attached to a checkpoint by PARSING the LoRA's filename on
   every render, and a checkpoint could hold exactly one preview — regenerating
   replaced it. Both are gone: the link is a pair of columns written when the
   image is generated, and previews accumulate. So a pill can now open a real
   history, whatever produced it — an inline canvas preview, a Test-Studio grid
   cell, a comparison run.

   ONE panel, two scopes. A click on a run CARD opens the same component with a
   `{kind:'run'}` target: every step of that run at once, grouped, plus the run's
   notes and the settings it trained with. A second panel for the run scope would
   have been two grids over the same rows, two Select modes and two places to
   keep the delete's promises true — agreeing on the day they shipped and
   drifting on the first change. Which endpoints, which title and how the groups
   are laid out all come from utils/runGallery (pure, unit-tested).

   Volume is handled, not hoped for. A run with fourteen checkpoints answers a
   payload capped per step AND overall, the panel opens the three most-trained
   groups and folds the rest behind their counts, and every cut is stated: a
   capped gallery that looks complete lies about the run.

   `unlinked` is stated, not hidden. Images generated before the columns existed,
   whose filename carries no run tag, cannot be attributed without guessing — so
   they are not shown under one, and the panel says how many there are. An
   incomplete history that says so beats a tidy one that lies. Images whose name
   names a RUN but no step are no longer part of that number: they are a real
   "Step unknown" group, because the run genuinely is known.

   🗑 And it deletes. A checkpoint accumulates dozens of renders and most are
   misses; a gallery that can only show them makes the user leave the board to
   clean up. The delete is REAL (the row is the Test Studio's own cell — see
   galleryDeleteConfirmation for the sentences that say so before the click) and
   the file goes to the recycle bin / the app Trash rather than being destroyed.
   The run scope deletes through the SAME backend function with a wider filter,
   so nothing about that promise is maintained twice.

   Deletion is deliberately UNREACHABLE by accident: it needs Select mode, then a
   pick, then a confirmation. Tapping a tile while scrolling a phone grid can
   never delete anything — outside Select mode a tap only zooms. Select now opens
   that mode from the same pinned bar the rest of it lives in (it used to sit in
   the header, a reach away from everything it leads to) — from the OPPOSITE end
   of that bar, and Delete stays disabled until a tile is picked, so sharing a
   row with the destructive button costs the guard nothing. */
export default function CheckpointGalleryPanel({ target, onClose, onDeleted, onDetails, onPin }) {
  const [state, setState] = useState({ status: 'loading', data: null, error: null });
  const [zoom, setZoom] = useState(null);
  const [picking, setPicking] = useState(false);
  const [selected, setSelected] = useState(() => new Set());
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState(null);
  const [openGroups, setOpenGroups] = useState(() => new Set());
  const [zipping, setZipping] = useState(false);
  const [timelineOpen, setTimelineOpen] = useState(false);
  /* ✨ The SAME handler the ◉ Canvas lightbox uses, and deliberately not a second
     one: one route, one toast, one place for the id space to be right (see
     hooks/useCanvasImageImprove). This gallery is where the result of the pass
     actually lands, which is what makes it the surface where the gesture is most
     natural — you are already looking at the checkpoint's renders. */
  const improveImage = useCanvasImageImprove();
  const restoreImproveSettings = useRestoreImproveSettings();

  const key = galleryTargetKey(target);
  const scope = galleryScope(target);
  const isRun = scope === 'run';
  const endpoints = galleryEndpoints(target);
  const heading = galleryHeading(target);

  const load = useCallback(() => {
    const ep = galleryEndpoints(target);
    if (!ep) return Promise.resolve();
    return apiFetch(ep.list)
      .then((d) => {
        setState({ status: 'ready', data: d, error: null });
        if (galleryScope(target) === 'run') {
          // Only on the FIRST read of a target: a refresh after a delete must not
          // fold back the groups the user opened.
          setOpenGroups((cur) => (cur.size ? cur : defaultOpenGroups(runGalleryGroups(d))));
        }
        // A refresh that no longer lists an image must not leave it armed.
        setSelected((cur) => pruneGallerySelection(
          cur, galleryScope(target) === 'run'
            ? visibleGalleryImages(runGalleryGroups(d), null)
            : d.images));
      })
      .catch((e) => {
        setState({
          status: 'error', data: null,
          error: e?.message || 'Could not load these images',
        });
      });
  }, [key]);   // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!target) return undefined;
    setState({ status: 'loading', data: null, error: null });
    setPicking(false);
    setSelected(new Set());
    setConfirming(false);
    setNotice(null);
    setOpenGroups(new Set());
    setTimelineOpen(false);
    load();
    return undefined;
  }, [key]);   // eslint-disable-line react-hooks/exhaustive-deps

  const d = state.data;
  const groups = useMemo(() => (isRun ? runGalleryGroups(d) : []), [isRun, d]);
  // What Select mode operates on: only what is LAID OUT. Arming a delete for an
  // image folded away — or never fetched, past the cap — would delete something
  // the user cannot see.
  const images = isRun
    ? visibleGalleryImages(groups, openGroups)
    : (d?.images || []);

  const runDelete = useCallback(async () => {
    if (!endpoints || busy) return;
    const ids = [...selected];
    setBusy(true);
    try {
      const res = await postJson(endpoints.remove, { image_ids: ids });
      setNotice({ kind: 'ok', text: galleryDeleteSummary(res) });
      setConfirming(false);
      setSelected(new Set());
      setPicking(false);
      await load();
      // The pills outside carry a results COUNT and a thumbnail: without this
      // the board keeps advertising images that no longer exist.
      onDeleted?.(res.dataset_ids || []);
    } catch (e) {
      setNotice({ kind: 'error', text: e?.message || 'Could not delete these images' });
    } finally {
      setBusy(false);
    }
  }, [endpoints?.remove, selected, busy, load, onDeleted]);  // eslint-disable-line react-hooks/exhaustive-deps

  /* ⬇ The gallery, as one ZIP.
   *
   * PREFLIGHT FIRST, and that is the whole design. A ZIP is handed to the
   * browser as a file: once it is downloading there is no place left to say
   * "three of these were missing" or "you got the newest 500 of 812". So the
   * counts are asked for BEFORE any byte is built (a cheap rows + exists()
   * read), the sentence lands in the panel's own notice, and a scope with
   * nothing left on disk is refused there instead of arriving as an archive
   * that looks complete and is not.
   *
   * The download itself is an ANCHOR, not fetch+blob: a 500-image archive read
   * into a Blob is a few hundred megabytes held in the tab for no reason, and
   * the name is already decided by Content-Disposition.
   */
  const runZip = useCallback(async () => {
    if (!target || zipping) return;
    // Select mode narrows the SAME button to the picks — the mode is already on
    // screen and already means "these ones"; a second download button that
    // differed from the first by one word would be unreadable.
    const ids = picking ? [...selected] : null;
    setZipping(true);
    setNotice(null);
    try {
      let plan = null;
      try {
        plan = await apiFetch(galleryZipPlanUrl(target, ids));
      } catch { plan = null; }
      const said = planNotice(plan);
      if (said) setNotice({ kind: said.kind, text: said.text });
      if (said?.blocked) return;
      const a = document.createElement('a');
      a.href = galleryZipUrl(target, ids);
      document.body.appendChild(a);
      a.click();
      a.remove();
    } finally {
      setZipping(false);
    }
  }, [target, key, picking, selected, zipping]);  // eslint-disable-line react-hooks/exhaustive-deps

  const toggleGroup = useCallback((groupKey) => {
    setOpenGroups((cur) => {
      const next = new Set(cur);
      if (next.has(groupKey)) next.delete(groupKey);
      else next.add(groupKey);
      return next;
    });
  }, []);

  if (!target) return null;
  // Everything the pinned bottom bar shows — including whether it exists at all.
  const bar = galleryActionBar({
    status: state.status, picking, imageCount: images.length,
    selectedCount: selected.size, busy,
  });
  const confirmation = galleryDeleteConfirmation(selected.size, d?.delete_mode);
  // The count is the SCOPE's, not the grid's: a run gallery lays out only the
  // steps that are open, but the archive holds the run. Saying "12" over a
  // button that fetches 240 would be the same lie as a silently capped ZIP.
  const zipBtn = zipButtonState({
    picking, selectedCount: selected.size,
    totalCount: Number(d?.count) || 0, busy: zipping,
  });
  const node = target.node || null;
  const paramRows = isRun ? configRows(node?.config) : [];
  const ckNotes = isRun ? checkpointNotes(d, node) : [];
  // Where these images LIVE: generated cells are moved into the dataset's own
  // folder on completion (data/datasets/<id> — see lora_test_studio's move on
  // completion), so "open the folder" is the dataset folder, resolved from the
  // run's node when the panel has one, else from any listed image (the images
  // of a checkpoint scope all carry their dataset_id).
  const folderDatasetId = node?.dataset_id ?? images.find((i) => i?.dataset_id != null)?.dataset_id ?? null;

  const tile = (img, altLabel) => {
    const isPicked = selected.has(img.id);
    // 📌 straight from the grid. It used to exist ONLY inside the viewer, so the
    // board's headline feature was behind "open an image and hope" — the person
    // who asked for it never found it. The rule (never in Select mode, never
    // without a board to pin onto) lives in gallerySelection so it is testable.
    const showPin = galleryTilePin({ picking, canPin: typeof onPin === 'function' });
    return (
      // A wrapper, because the tile itself is a <button> and a button cannot
      // contain one. The wrapper carries no handler of its own: the image target
      // stays exactly as big as it was, and the pin is the only new hit area.
      <div key={img.id} className="relative">
        <button type="button"
          data-testid={picking ? 'gallery-pick' : 'gallery-zoom'}
          onClick={() => (picking
            ? setSelected((cur) => toggleGalleryImage(cur, img.id))
            : setZoom(img))}
          aria-pressed={picking ? isPicked : undefined}
          title={picking
            ? (isPicked ? 'Selected — tap to unselect' : 'Tap to select')
            : imageFactsLine(img)}
          className={`block aspect-square w-full overflow-hidden rounded-md border ${isPicked
            ? 'border-rose-400 ring-2 ring-rose-400/70'
            : 'border-border hover:border-indigo-400/60'}`}>
          <img src={img.url} alt={altLabel} loading="lazy"
            className={`h-full w-full object-cover ${isPicked ? 'opacity-60' : ''}`} />
        </button>
        {picking && (
          // A 24-px tick, not a hairline checkbox: the target has to be hittable
          // with a thumb on a 400-px grid.
          <span aria-hidden
            className={`pointer-events-none absolute left-1 top-1 flex h-6 w-6 items-center justify-center rounded-full border text-[0.75rem] ${isPicked
              ? 'border-rose-300 bg-rose-500 text-white'
              : 'border-white/60 bg-black/50 text-transparent'}`}>✓</span>
        )}
        {img.rating === 1 && (
          <span aria-hidden
            className="pointer-events-none absolute right-0.5 top-0.5 text-[0.625rem]">👍</span>
        )}
        {img.rating === -1 && (
          <span aria-hidden
            className="pointer-events-none absolute right-0.5 top-0.5 text-[0.625rem]">👎</span>
        )}
        {showPin && (
          // BOTTOM-right, because top-right is where the 👍/👎 verdict sits and a
          // thumb aiming for one must not find the other.
          //
          // The size is INVERTED from the usual responsive instinct, because the
          // layout is: the phone bottom-sheet lays two columns across the whole
          // screen (~170 px tiles) and is driven by a thumb, so 28 px there; from
          // `sm` the panel becomes a 22-rem side drawer with THREE columns
          // (~78 px tiles) driven by a mouse, where the same badge would cover a
          // third of the picture it is meant to let you judge. Measured on both.
          <button type="button" data-testid="gallery-tile-pin"
            onClick={() => onPin(img)}
            aria-label="Pin this image to the canvas"
            title="📌 Pin to canvas — put this image on the board, beside the checkpoint that made it"
            className="absolute bottom-1 right-1 flex h-7 w-7 items-center justify-center rounded-full
                       border border-indigo-300/70 bg-black/60 text-[0.8125rem] text-indigo-100
                       hover:bg-indigo-500/50 focus-visible:bg-indigo-500/50
                       sm:bottom-0.5 sm:right-0.5 sm:h-5 sm:w-5 sm:text-[0.625rem]">
            <span aria-hidden>📌</span>
          </button>
        )}
      </div>
    );
  };

  // Two columns at 400 px, three from `sm` — thumbnails that stay big enough to
  // compare on a phone.
  const GRID = 'grid grid-cols-2 gap-1.5 sm:grid-cols-3';

  return (
    <>
      {/* Bottom sheet up to `lg`, side drawer from there. The switch used to be
          at `sm` (640 px), which handed a 768-px tablet a 352-px drawer over a
          416-px sliver of board — the exact shape the sheet exists to avoid. */}
      <aside
        data-testid={isRun ? 'run-gallery-panel' : 'checkpoint-gallery-panel'}
        aria-label={heading.aria}
        className="fixed inset-x-0 bottom-0 z-50 flex max-h-[70vh] flex-col overflow-hidden border-t border-border bg-surface-overlay shadow-xl
                   lg:inset-x-auto lg:left-0 lg:top-0 lg:h-full lg:max-h-none lg:w-[22rem] lg:border-r lg:border-t-0">
        <header className="flex shrink-0 items-center gap-2 border-b border-border px-3 py-2">
          <h3 className="min-w-0 flex-1 truncate text-sm font-semibold text-content">
            <Images aria-hidden="true" className="h-4 w-4" /> {heading.title}
            <span className="font-normal text-content-muted"> · {heading.subtitle}</span>
          </h3>
          {/* The way back to the board — 44 px of thumb below `lg`. */}
          <button type="button" onClick={onClose} aria-label="Close"
            className="-my-1 -mr-1 flex h-11 w-11 shrink-0 items-center justify-center text-content-subtle hover:text-content lg:h-8 lg:w-8">✕</button>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto p-3">
          {state.status === 'loading' && (
            <p className="m-0 text-content-subtle text-[0.75rem]">Loading…</p>
          )}
          {state.status === 'error' && (
            <p className="m-0 rounded-lg border border-amber-400/40 bg-amber-500/10 px-2 py-1.5 text-amber-100 text-[0.75rem]">
              {state.error}
            </p>
          )}
          {notice && (
            /* 'warn' is its own tone on purpose: "the ZIP you just started is
               shorter than this gallery" is neither a failure nor a success,
               and dressing it in the green of a finished delete would bury the
               one sentence that has to be read. */
            <p role={notice.kind === 'ok' ? undefined : 'alert'}
              className={`m-0 mb-2 rounded-lg border px-2 py-1.5 text-[0.6875rem] ${
                notice.kind === 'error'
                  ? 'border-rose-400/50 bg-rose-500/10 text-rose-100'
                  : notice.kind === 'warn'
                    ? 'border-amber-400/40 bg-amber-500/10 text-amber-100'
                    : 'border-emerald-400/40 bg-emerald-500/10 text-emerald-100'}`}>
              {notice.text}
            </p>
          )}

          {state.status === 'ready' && !isRun && (
            <>
              <p className="m-0 mb-2 text-content-muted text-[0.6875rem]">
                {d.count === 0
                  ? 'Nothing generated from this checkpoint yet — tick it and run from the board.'
                  : picking
                    ? `Tap the misses, then 🗑 Delete. ${d.count} image${d.count > 1 ? 's' : ''} here.`
                    : `${d.count} image${d.count > 1 ? 's' : ''}, newest first.`}
              </p>
              <div className={GRID}>
                {images.map((img) => tile(img, `Generated at step ${target.step}`))}
              </div>
            </>
          )}

          {state.status === 'ready' && isRun && (
            <>
              <div className="mb-2 flex items-start gap-2">
                <p className="m-0 min-w-0 flex-1 text-content-muted text-[0.6875rem]">
                  {picking
                    ? `Tap the misses, then 🗑 Delete. ${images.length} shown in the open steps.`
                    : runGallerySummary(d)}
                </p>
                <button type="button" data-testid="run-gallery-timeline"
                  onClick={() => setTimelineOpen(true)} aria-haspopup="dialog"
                  className="shrink-0 rounded-md border border-indigo-400/60 bg-indigo-500/15 px-2 py-1.5 text-[0.6875rem] font-semibold text-indigo-100 hover:bg-indigo-500/25">
                  <span aria-hidden>🎞</span> Timeline
                </button>
              </div>

              {/* One foldable section per checkpoint, most-trained first: the end
                  of training is where over- and under-fitting is judged, so the
                  panel opens on it. Folded groups still state their count — the
                  fold hides thumbnails, never information. */}
              {groups.map((g) => {
                const open = openGroups.has(g.key);
                return (
                  <section key={g.key} data-testid="run-gallery-group"
                    data-step={g.step == null ? 'unknown' : g.step}
                    className="mb-2 rounded-lg border border-border">
                    <button type="button" data-testid="run-gallery-group-toggle"
                      onClick={() => toggleGroup(g.key)} aria-expanded={open}
                      className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left hover:bg-app/40">
                      <span aria-hidden className="shrink-0 text-content-subtle text-[0.625rem]">
                        {open ? '▾' : '▸'}
                      </span>
                      <span className="min-w-0 flex-1 truncate text-content text-[0.75rem] font-semibold tabular-nums">
                        {g.label}
                      </span>
                      <span className="shrink-0 rounded-full border border-border px-1.5 text-content-muted text-[0.625rem] tabular-nums">
                        {g.count}
                      </span>
                    </button>
                    {open && (
                      <div className="px-2 pb-2">
                        {g.step == null && (
                          <p className="m-0 mb-1.5 text-content-subtle text-[0.625rem]">
                            These came from this run — the file name named the run but
                            not the step, so they sit under no checkpoint.
                          </p>
                        )}
                        <div className={GRID}>
                          {g.images.map((img) => tile(img, `Generated by run ${target.recordId}`))}
                        </div>
                        {g.truncated && (
                          <p className="m-0 mt-1.5 text-content-subtle text-[0.625rem]">
                            Newest {g.images.length} of {g.count} — open this checkpoint’s
                            own gallery from its pill for the rest.
                          </p>
                        )}
                      </div>
                    )}
                  </section>
                );
              })}

              {/* 📝 The notes, where the images they describe are. They stay
                  READ-ONLY here: editing lives in the details drawer, which
                  already owns saving, and two editors over one field is one too
                  many. A run with no note simply has no section. */}
              {(node?.note || ckNotes.length > 0) && (
                <section data-testid="run-gallery-notes"
                  className="mt-3 border-t border-border pt-2">
                  <h4 className="m-0 mb-1 text-content text-[0.6875rem] font-semibold">
                    <span aria-hidden>📝</span> Notes
                  </h4>
                  {node?.note && (
                    <p className="m-0 mb-1.5 whitespace-pre-wrap break-words text-content-muted text-[0.6875rem]">
                      {node.note}
                    </p>
                  )}
                  {ckNotes.map((c) => (
                    <p key={c.step}
                      className="m-0 mb-1 whitespace-pre-wrap break-words text-content-muted text-[0.625rem]">
                      <span className="font-semibold text-content-subtle tabular-nums">
                        {stepGroupLabel(c.step)}:
                      </span>{' '}{c.note}
                    </p>
                  ))}
                </section>
              )}

              {/* ⚙ What this run trained with — the persisted snapshot the detail
                  drawer reads. A legacy run that never recorded its settings says
                  so rather than showing an empty table. */}
              <section data-testid="run-gallery-settings"
                className="mt-3 border-t border-border pt-2">
                <h4 className="m-0 mb-1 text-content text-[0.6875rem] font-semibold">
                  <span aria-hidden>⚙</span> Training settings
                </h4>
                {paramRows.length === 0 ? (
                  <p className="m-0 text-content-subtle text-[0.625rem]">
                    This run did not record its settings (it predates the snapshot).
                  </p>
                ) : (
                  <dl className="m-0 grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)] gap-x-2 gap-y-0.5">
                    {paramRows.map((r) => (
                      <div key={r.label} className="contents">
                        <dt className="m-0 truncate text-content-subtle text-[0.625rem]">{r.label}</dt>
                        <dd className="m-0 break-words text-content-muted text-[0.625rem]">{r.value}</dd>
                      </div>
                    ))}
                  </dl>
                )}
                {typeof onDetails === 'function' && node && (
                  <button type="button" data-testid="run-gallery-details"
                    onClick={() => onDetails(node)}
                    title="Open the full run details, where the notes can be edited"
                    className="mt-2 rounded-md border border-border px-2 py-1.5 text-content text-[0.625rem] hover:border-indigo-400/50">
                    <span aria-hidden>ⓘ</span> Full details &amp; edit notes
                  </button>
                )}
              </section>
            </>
          )}

          {state.status === 'ready' && d.unlinked > 0 && (
            <p className="m-0 mt-3 border-t border-border pt-2 text-content-subtle text-[0.625rem]">
              {unlinkedNote(d.unlinked, scope)}
            </p>
          )}

          {/* ⚠ Deleting the RUN itself — only in the run scope, and only at the
              very bottom. In the checkpoint scope the panel is looking at one
              step, not a run: offering "delete the whole run" there would be a
              destructive action aimed at something other than what the title
              says. Its own file (RunDeleteSection) owns the confirmation, the
              counts and the keyboard behaviour; deleting reuses `onDeleted`,
              which both hosts already answer by refetching the lane — so the
              card disappears from the board with no new prop. */}
          {state.status === 'ready' && isRun && node && (
            <RunDeleteSection recordId={target.recordId} datasetId={node.dataset_id}
              onDeleted={onDeleted} onClose={onClose} />
          )}
        </div>

        {/* 📌 The action bar, pinned so it stays reachable with a thumb however far
            the grid has been scrolled — and PERMANENT: `Select` used to sit up in
            the header, so entering the mode meant reaching the top of a panel
            whose every other control was already down here. It also means the
            gate does not move when the mode turns on; it just gets company.

            Everything destructive still hangs off `picking`, and the bar as a
            whole off having images — an empty gallery carries no bar at all.
            Both decisions are taken in galleryActionBar (unit-tested), not in an
            inline `&&` that a rewrite could quietly loosen. */}
        {bar.shown && (
          <div data-testid="gallery-action-bar"
            className="flex shrink-0 flex-wrap items-center gap-2 border-t border-border bg-surface-overlay px-3 py-2">
            {/* Indigo — the app's accent, and already what this button turned when
                it was on. The state is NOT the colour though: the label flips
                Select/Done, the glyph with it, and aria-pressed says it out loud. */}
            <button type="button" data-testid="gallery-select-toggle"
              onClick={() => { setPicking((v) => !v); setSelected(new Set()); setNotice(null); }}
              aria-pressed={bar.togglePressed}
              aria-label={picking ? 'Leave selection mode' : 'Select images to delete'}
              title={picking ? 'Leave selection mode' : 'Select images to delete'}
              className={`shrink-0 rounded-md border px-3 py-1.5 text-[0.75rem] font-semibold ${picking
                ? 'border-indigo-300 bg-indigo-500/40 text-white'
                : 'border-indigo-400/70 bg-indigo-500/15 text-indigo-200 hover:bg-indigo-500/25'}`}>
              <span aria-hidden>{picking ? '✓' : '☑'}</span> {bar.toggleLabel}
            </button>
            {/* ⬇ ONE button, two meanings, taken from the mode already on
                screen: the whole gallery normally, the picks in Select mode.
                It carries its own count — and when the gallery is bigger than
                one archive it carries the CAP, on its face and in its tooltip,
                so a short ZIP is never a discovery. */}
            {zipBtn.shown && (
              <button type="button" data-testid="gallery-download-zip"
                onClick={runZip} disabled={zipBtn.disabled} title={zipBtn.title}
                className="shrink-0 rounded-md border border-border px-2.5 py-1.5 text-content-muted text-[0.75rem] hover:border-indigo-400/50 hover:text-content disabled:opacity-40">
                {zipBtn.label}
              </button>
            )}
            {/* 📂 The same folder the completed cells are moved into (the
                dataset's own directory) — for the person who wants the files
                themselves, not an archive. Server-resolved target, no path in
                the request; hidden when no row told us which dataset this is
                (a legacy gallery of unlinked images). */}
            {folderDatasetId != null && (
              <button type="button" data-testid="gallery-open-folder"
                onClick={() => {
                  postJson(`/api/dataset/${folderDatasetId}/train/open-folder`, { target: 'dataset' })
                    .catch((e) => setNotice({ kind: 'error', text: e?.message || 'Could not open the folder' }));
                }}
                title="Open the folder these images are saved in (the dataset's folder, on the machine running the app)"
                className="shrink-0 rounded-md border border-border px-2.5 py-1.5 text-content-muted text-[0.75rem] hover:border-indigo-400/50 hover:text-content">
                <FolderOpen aria-hidden="true" className="h-3.5 w-3.5" /> Open folder
              </button>
            )}
            {bar.showsDelete && (
              <>
                <span className="text-content-muted text-[0.6875rem] tabular-nums">
                  {selected.size} selected
                </span>
                <button type="button"
                  onClick={() => setSelected(selected.size === images.length
                    ? new Set() : allGalleryImageIds(images))}
                  className="rounded-md border border-border px-2 py-1.5 text-content-muted text-[0.6875rem] hover:text-content">
                  {bar.selectAllLabel}
                </button>
                {/* Last, and pushed to the far edge: the gate that opens this mode
                    sits at the opposite end of the row, and this stays disabled
                    until a tile has been picked — so the two taps that arm a
                    delete can never be one thumb slide apart. */}
                <button type="button" data-testid="gallery-delete"
                  disabled={bar.deleteDisabled}
                  onClick={() => setConfirming(true)}
                  className="ml-auto rounded-md border border-rose-500/50 px-3 py-1.5 text-[0.75rem] text-rose-300 disabled:opacity-40 hover:bg-rose-500/10">
                  🗑 Delete{selected.size ? ` (${selected.size})` : ''}
                </button>
              </>
            )}
          </div>
        )}
      </aside>

      {/* The confirmation. Both consequences are stated BEFORE the button arms:
          the images leave the Test Studio too, and where the files land. */}
      {confirming && (
        <div role="dialog" aria-modal="true" aria-label="Confirm deletion"
          data-testid="gallery-confirm"
          className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 p-3">
          <div className="w-full max-w-sm rounded-xl border border-border bg-surface-overlay p-4 shadow-xl">
            <h4 className="m-0 mb-2 text-sm font-semibold text-content">{confirmation.title}</h4>
            <ul className="m-0 mb-3 list-disc space-y-1 pl-4 text-content-muted text-[0.75rem]">
              {confirmation.lines.map((line) => <li key={line}>{line}</li>)}
            </ul>
            {confirmation.destructive && (
              <p className="m-0 mb-3 rounded-lg border border-rose-400/40 bg-rose-500/10 px-2 py-1.5 text-rose-100 text-[0.6875rem]">
                This cannot be undone.
              </p>
            )}
            <div className="flex flex-wrap justify-end gap-2">
              <button type="button" autoFocus onClick={() => setConfirming(false)}
                className="rounded-md border border-border px-3 py-2 text-content-muted text-[0.75rem] hover:text-content">
                Cancel
              </button>
              <button type="button" data-testid="gallery-confirm-delete"
                disabled={busy} onClick={runDelete}
                className="rounded-md border border-rose-500/60 bg-rose-500/15 px-3 py-2 text-[0.75rem] text-rose-100 disabled:opacity-40 hover:bg-rose-500/25">
                {busy ? 'Deleting…' : `Delete ${selected.size}`}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 🔍 The image, large, with its facts laid out instead of run together in
          one paragraph across the whole screen — see GeneratedImageLightbox.
          `onPin` is the canvas's: it drops this render onto the board as a node
          of its own, which is where two checkpoints actually get compared.

          ✨ And it improves, from HERE — the screen the improvement is delivered
          to. The pass writes its result as a row of this very gallery, next to
          the original, so this is where the button costs the fewest gestures:
          you are already comparing renders when you decide one deserves a 2 MP
          pass. Exactly like the board, the action is the explicit `onImprove`
          opt-in and it is withheld for a picture that cannot take it (a row
          that IS an improvement — canvasImprove.js states the reasons), so the
          refusal is read before the click instead of arriving as a 400 after
          it. `dataset_id` travels with it: Klein's amber note needs it. */}
      <GeneratedImageLightbox
        img={zoom} alt={`Generated by run ${target.recordId}`}
        onClose={() => setZoom(null)}
        onImprove={canImproveCanvasImage(zoom) ? improveImage : undefined}
        onUseImproveSettings={restoreImproveSettings}
        datasetId={zoom?.dataset_id ?? null}
        /* ✦ and 📷 arrive with the viewer now; this gallery only refreshes
           its own grid after a repair rewrote a file. 📌 Pin stays in the
           `actions` slot — it is genuinely THIS host's verb, not a shared one:
           it needs the board the panel sits beside. */
        onRowChanged={() => load()}
        actions={typeof onPin === 'function' && zoom ? (
          <button type="button" data-testid="gallery-pin-image"
            onClick={() => { onPin(zoom); setZoom(null); }}
            title="Put this image on the board, beside the checkpoint that made it"
            className="rounded-md border border-indigo-400/60 bg-indigo-500/20 px-3 py-1.5 text-[0.75rem] font-semibold text-indigo-100 hover:bg-indigo-500/30">
            <span aria-hidden>📌</span> Pin to canvas
          </button>
        ) : null} />
      {isRun && timelineOpen && (
        <CheckpointTimelinePanel recordId={target.recordId}
          onClose={() => setTimelineOpen(false)} />
      )}
    </>
  );
}
