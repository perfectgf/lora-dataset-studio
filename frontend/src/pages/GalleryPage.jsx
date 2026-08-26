import { useCallback, useEffect, useRef, useState } from 'react';
import { Camera, Images, ThumbsDown, ThumbsUp, Trash2 } from 'lucide-react';
import { apiFetch, postJson } from '../api/fetchClient';
import CameraAnglePicker from '../components/shared/CameraAnglePicker';
import GeneratedImageLightbox from '../components/shared/GeneratedImageLightbox';
import { useCameraAngles } from '../hooks/useCameraAngles';
import { cameraRefusal, isCameraView, poseLabel } from '../utils/cameraAngles';
import { useCanvasImageImprove } from '../hooks/useCanvasImageImprove';
import { useRestoreImproveSettings } from '../hooks/useRestoreImproveSettings';
import { canImproveCanvasImage } from '../utils/canvasImprove';
import { imageFactsLine } from '../utils/generatedImageFacts';
import {
  allGalleryImageIds, galleryActionBar, galleryDeleteConfirmation,
  galleryDeleteSummary, pruneGallerySelection, toggleGalleryImage,
} from '../utils/gallerySelection';
import {
  galleryZipPlanUrl, galleryZipUrl, planNotice, zipButtonState,
} from '../utils/galleryDownload';
import {
  GALLERY_KINDS, datasetFilterOptions, galleryEmptyMessage, galleryFeedUrl,
  galleryImproveLaunchMessage, gallerySummaryLine, mergeGalleryPage,
} from '../utils/appGallery';

/* 🖼 THE GALLERY — every image the app ever generated, one feed.

   The checkpoint and run galleries answer "what did THIS training produce";
   this page answers "what did I make", across every dataset and surface at
   once. Same rows, same lightbox, same improve handler as the ◉ Canvas — the
   page adds no second implementation of anything, which is the whole design:
   the feed URL, the merge rule and every sentence live in utils/appGallery.js
   where `node --test` reads them, and the verbs (✨ improve, ⬇ download,
   🗑 delete, ZIP) are the ones the other galleries already ship.

   Browsing is the loop this page owns, so the viewer gains ‹ › navigation
   (GeneratedImageLightbox's onPrev/onNext — buttons AND arrow keys) instead of
   the open-close-open-next dance the per-checkpoint galleries get away with.

   🗑 Deletion follows the panel's own guard rails: Select mode, then a pick,
   then a confirmation that names where the files land — and the rows leave the
   Test Studio too, which the confirmation says before the click. The ZIP is a
   SELECTION: this feed can span thousands of images, and "download everything"
   is a request the backend deliberately refuses to infer from a missing
   parameter. */

const FILTER_BTN =
  'min-h-10 lg:min-h-0 rounded-md border px-2.5 py-1 text-[0.75rem] font-medium transition-colors';

export default function GalleryPage() {
  const [filters, setFilters] = useState({ datasetId: '', kind: '', liked: false });
  const [feed, setFeed] = useState({
    images: [], count: 0, hasMore: false, nextBeforeId: null, datasets: [],
    deleteMode: null,
  });
  const [status, setStatus] = useState('loading');   // loading | ready | error
  const [error, setError] = useState(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const [zoomIndex, setZoomIndex] = useState(null);
  const [picking, setPicking] = useState(false);
  const [selected, setSelected] = useState(() => new Set());
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState(null);
  const [zipping, setZipping] = useState(false);
  // 📷 The picture the camera picker is open for, or null. Held as the ROW and
  // not a boolean: the picker outlives a ‹ › step through the feed, and a
  // flag would silently re-shoot whatever the viewer landed on instead.
  const [cameraFor, setCameraFor] = useState(null);
  const shootCameraViews = useCameraAngles();
  const alive = useRef(true);
  // Set true INSIDE the effect, not only at ref creation: StrictMode runs the
  // cleanup once at mount (false) and re-runs the effect — a ref left false
  // there silently discards every response for the life of the page. Same
  // shape KleinImproveNote uses, for the same reason.
  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; };
  }, []);

  const improveImage = useCanvasImageImprove({
    // Its own address: the result lands at the head of THIS feed, and the
    // shared wording would send the user to a checkpoint gallery to find it.
    launchMessage: galleryImproveLaunchMessage,
  });
  const restoreImproveSettings = useRestoreImproveSettings();

  const applyPage = useCallback((d, { append }) => {
    setFeed((cur) => ({
      images: append ? mergeGalleryPage(cur.images, d.images) : (d.images || []),
      count: d.count || 0,
      hasMore: !!d.has_more,
      nextBeforeId: d.next_before_id ?? null,
      datasets: d.datasets || [],
      deleteMode: d.delete_mode || null,
    }));
    // A refresh that no longer lists an image must not leave it armed. A page
    // APPENDED to the feed retires nothing, so the selection stands.
    if (!append) {
      setSelected((cur) => pruneGallerySelection(cur, d.images || []));
    }
  }, []);

  const load = useCallback((f) => {
    setStatus('loading');
    setError(null);
    setZoomIndex(null);
    return apiFetch(galleryFeedUrl(f))
      .then((d) => {
        if (!alive.current) return;
        applyPage(d, { append: false });
        setStatus('ready');
      })
      .catch((e) => {
        if (!alive.current) return;
        setStatus('error');
        setError(e?.message || 'Could not load the gallery');
      });
  }, [applyPage]);

  useEffect(() => { load(filters); }, [filters, load]);

  const loadMore = useCallback(() => {
    if (status !== 'ready' || loadingMore || !feed.hasMore
      || feed.nextBeforeId == null) return;
    setLoadingMore(true);
    apiFetch(galleryFeedUrl(filters, { beforeId: feed.nextBeforeId }))
      .then((d) => { if (alive.current) applyPage(d, { append: true }); })
      .catch((e) => {
        if (alive.current) {
          setNotice({ kind: 'error', text: e?.message || 'Could not load more images' });
        }
      })
      .finally(() => { if (alive.current) setLoadingMore(false); });
  }, [filters, feed.hasMore, feed.nextBeforeId, loadingMore, applyPage, status]);

  /* ♾ The feed loads ITSELF as the reader approaches its end — a browsing
     page whose next screen is behind a button is a page that stalls every 60
     images. The sentinel is the load-more ROW (small on purpose: an
     IntersectionObserver threshold is a % of the ELEMENT, unreachable on a
     tall one), threshold 0 with a 600 px rootMargin so the next page is
     usually there before the reader is. The button STAYS — it is the visible
     state ("Loading…", how many are left) and the fallback wherever
     IntersectionObserver is not (very old WebViews).

     Re-created whenever `loadMore` changes identity (each appended page):
     that re-fires an immediately-visible sentinel, which is exactly what
     fills a viewport taller than one page without a single scroll event. */
  const loadMoreRef = useRef(null);
  useEffect(() => {
    const el = loadMoreRef.current;
    if (!el || typeof IntersectionObserver === 'undefined') return undefined;
    const io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) loadMore();
    }, { rootMargin: '600px 0px' });
    io.observe(el);
    return () => io.disconnect();
  }, [loadMore]);

  const setFilter = (patch) => {
    setPicking(false);
    setSelected(new Set());
    setNotice(null);
    setFilters((f) => ({ ...f, ...patch }));
  };

  const images = feed.images;
  const zoom = zoomIndex != null ? images[zoomIndex] ?? null : null;

  const runDelete = useCallback(async () => {
    if (busy) return;
    const ids = [...selected];
    setBusy(true);
    try {
      const res = await postJson('/api/gallery/images/delete', { image_ids: ids });
      setNotice({ kind: 'ok', text: galleryDeleteSummary(res) });
      setConfirming(false);
      setPicking(false);
      setSelected(new Set());
      // The feed is a cursor over ids — removing the deleted rows locally keeps
      // every loaded page and the scroll position, where a full reload would
      // throw both away to learn what this answer already says.
      const gone = new Set(ids.filter((i) => !(res.skipped || [])
        .some((s) => s.id === i)));
      setFeed((cur) => ({
        ...cur,
        images: cur.images.filter((img) => !gone.has(img.id)),
        count: Math.max(0, cur.count - (res.rows_removed || 0)),
      }));
    } catch (e) {
      setNotice({ kind: 'error', text: e?.message || 'Could not delete these images' });
    } finally {
      setBusy(false);
    }
  }, [selected, busy]);

  const runZip = useCallback(async () => {
    if (zipping || selected.size === 0) return;
    const ids = [...selected];
    setZipping(true);
    setNotice(null);
    try {
      let plan = null;
      try {
        plan = await apiFetch(galleryZipPlanUrl({ kind: 'app' }, ids));
      } catch { plan = null; }
      const said = planNotice(plan);
      if (said) setNotice({ kind: said.kind, text: said.text });
      if (said?.blocked) return;
      const a = document.createElement('a');
      a.href = galleryZipUrl({ kind: 'app' }, ids);
      document.body.appendChild(a);
      a.click();
      a.remove();
    } finally {
      setZipping(false);
    }
  }, [selected, zipping]);

  const bar = galleryActionBar({
    status, picking, imageCount: images.length,
    selectedCount: selected.size, busy,
  });
  const confirmation = galleryDeleteConfirmation(selected.size, feed.deleteMode);
  // Selection-only on this page (see the header): outside Select mode there is
  // no ZIP button at all, so `totalCount` is the picks and never the feed.
  const zipBtn = zipButtonState({
    picking, selectedCount: selected.size,
    totalCount: picking ? selected.size : 0, busy: zipping,
  });

  const kindBtnClass = (active) => `${FILTER_BTN} ${active
    ? 'border-indigo-300 bg-indigo-500/30 text-white'
    : 'border-border text-content-muted hover:text-content hover:border-indigo-400/50'}`;

  return (
    <div className="space-y-3">
      <header className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        {/* The Beta chip marks what the page CAN DO, not how old it is: the feed
            itself has been stable for weeks, but 📷 Camera angles ships today and
            brings a second 20 GB engine with it. Same amber chip as the ◉ Canvas
            and the Slider trainer, so "beta" means one thing across the app. */}
        <h1 className="m-0 flex items-center gap-2 text-lg font-bold text-content">
          <Images aria-hidden="true" className="h-4 w-4" /> Gallery
          <span className="rounded border border-amber-400/50 bg-amber-500/10 px-1.5 py-0.5 text-[0.625rem] font-semibold uppercase tracking-wide text-amber-300">Beta</span>
        </h1>
        <p className="m-0 text-content-muted text-[0.75rem]">
          {status === 'ready'
            ? gallerySummaryLine({ count: feed.count, shown: images.length })
            : ''}
        </p>
      </header>

      {/* The filter rail. One row that wraps: on a phone the selects stack,
          nothing overflows sideways. Deliberately NOT data-probe-panel: the
          fill check reads a panel's DIRECT CHILDREN as rows against the full
          panel width, which is the right question for a vertical shelf and a
          structurally failing one for a one-line toolbar on the 1800-px
          measure. It IS chrome: it costs the fold at rest, like the dataset
          grid-toolbar, and it is what proves to the probe the page painted. */}
      <div data-probe-chrome="gallery-filters"
        className="flex flex-wrap items-center gap-2">
        <label className="flex min-w-0 items-center gap-1.5 text-[0.75rem] text-content-muted">
          <span className="sr-only">Dataset</span>
          <select value={filters.datasetId}
            onChange={(e) => setFilter({ datasetId: e.target.value })}
            aria-label="Show one dataset's images"
            className="min-h-10 lg:min-h-0 min-w-0 max-w-full rounded-md border border-border bg-surface-raised px-2 py-1 text-[0.75rem] text-content focus:outline-none focus:border-primary/60">
            {datasetFilterOptions(feed.datasets,
              feed.datasets.reduce((n, d) => n + (d.count || 0), 0))
              .map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
          </select>
        </label>
        <div role="group" aria-label="Filter by kind" className="flex flex-wrap gap-1">
          {GALLERY_KINDS.map((k) => (
            <button key={k.id} type="button"
              aria-pressed={filters.kind === k.id}
              onClick={() => setFilter({ kind: k.id })}
              className={kindBtnClass(filters.kind === k.id)}>
              {k.label}
            </button>
          ))}
        </div>
        <button type="button" aria-pressed={filters.liked}
          onClick={() => setFilter({ liked: !filters.liked })}
          title="Only the images you liked"
          className={kindBtnClass(filters.liked)}>
          <ThumbsUp aria-hidden="true" className="h-3.5 w-3.5" /> Liked
        </button>
      </div>

      {notice && (
        <p role={notice.kind === 'ok' ? undefined : 'alert'}
          className={`m-0 rounded-lg border px-2 py-1.5 text-[0.6875rem] ${
            notice.kind === 'error'
              ? 'border-rose-400/50 bg-rose-500/10 text-rose-100'
              : notice.kind === 'warn'
                ? 'border-amber-400/40 bg-amber-500/10 text-amber-100'
                : 'border-emerald-400/40 bg-emerald-500/10 text-emerald-100'}`}>
          {notice.text}
        </p>
      )}

      {status === 'loading' && (
        <p className="m-0 text-content-subtle text-[0.75rem]">Loading…</p>
      )}
      {status === 'error' && (
        <p className="m-0 rounded-lg border border-amber-400/40 bg-amber-500/10 px-2 py-1.5 text-amber-100 text-[0.75rem]">
          {error}
        </p>
      )}
      {status === 'ready' && images.length === 0 && (
        <p className="m-0 text-content-muted text-[0.8125rem]">
          {galleryEmptyMessage(filters)}
        </p>
      )}

      {images.length > 0 && (
        <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6">
          {images.map((img, idx) => {
            const isPicked = selected.has(img.id);
            return (
              <div key={img.id} className="relative">
                <button type="button"
                  data-testid={picking ? 'gallery-pick' : 'gallery-zoom'}
                  onClick={() => (picking
                    ? setSelected((cur) => toggleGalleryImage(cur, img.id))
                    : setZoomIndex(idx))}
                  aria-pressed={picking ? isPicked : undefined}
                  title={picking
                    ? (isPicked ? 'Selected — tap to unselect' : 'Tap to select')
                    : imageFactsLine(img)}
                  className={`block aspect-square w-full overflow-hidden rounded-md border bg-black/40 ${isPicked
                    ? 'border-rose-400 ring-2 ring-rose-400/70'
                    : 'border-border hover:border-indigo-400/60'}`}>
                  {/* object-CONTAIN, not cover: the cell stays square (row
                      order, ‹ › navigation and the selection grid all read
                      row-major), but the picture inside is never cropped — a
                      9:16 render used to lose its head and feet to the cell.
                      The dark cell background is what makes the letterbox
                      read as a mat, not a broken thumbnail. */}
                  <img src={img.url} alt={imageFactsLine(img) || 'Generated image'}
                    loading="lazy"
                    className={`h-full w-full object-contain ${isPicked ? 'opacity-60' : ''}`} />
                </button>
                {picking && (
                  <span aria-hidden
                    className={`pointer-events-none absolute left-1 top-1 flex h-6 w-6 items-center justify-center rounded-full border text-[0.75rem] ${isPicked
                      ? 'border-rose-300 bg-rose-500 text-white'
                      : 'border-white/60 bg-black/50 text-transparent'}`}>✓</span>
                )}
                {img.rating === 1 && (
                  <ThumbsUp aria-hidden="true" className="pointer-events-none absolute right-0.5 top-0.5 h-3 w-3 text-emerald-300" />
                )}
                {img.rating === -1 && (
                  <ThumbsDown aria-hidden="true" className="pointer-events-none absolute right-0.5 top-0.5 h-3 w-3 text-rose-300" />
                )}
                {img.derivation_kind && (
                  /* BOTTOM-left: top-right is the verdict's corner and the
                     selection tick owns top-left while picking. Two derivations
                     reach this grid and they are NOT the same picture: ✨ is an
                     upscale of what you see, 📷 is the same scene from another
                     camera position. One badge for both would make a tile lie
                     about what produced it — and the camera view carries its
                     pose, because eight of them side by side are unreadable
                     otherwise. */
                  <span aria-hidden
                    title={isCameraView(img)
                      ? `Camera view — ${poseLabel(img.camera_pose) || 'another angle'}`
                      : 'Upscale & improve result'}
                    className="pointer-events-none absolute bottom-0.5 left-1 text-[0.625rem]">
                    {isCameraView(img) ? '📷' : '✨'}
                  </span>
                )}
                {isCameraView(img) && poseLabel(img.camera_pose) && (
                  <span aria-hidden
                    className="pointer-events-none absolute inset-x-0 bottom-0 truncate bg-gradient-to-t from-black/80 to-transparent px-1 pb-0.5 pt-2 text-center text-[0.55rem] leading-tight text-white/85">
                    {poseLabel(img.camera_pose)}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      )}

      {feed.hasMore && status === 'ready' && (
        <div ref={loadMoreRef} className="flex justify-center">
          <button type="button" data-testid="gallery-load-more"
            onClick={loadMore} disabled={loadingMore}
            className="min-h-10 rounded-md border border-border px-4 py-1.5 text-[0.8125rem] text-content-muted hover:border-indigo-400/50 hover:text-content disabled:opacity-50">
            {loadingMore ? 'Loading…' : `Load more (${feed.count - images.length} left)`}
          </button>
        </div>
      )}

      {/* The action bar — sticky, so Select/Delete stay reachable however far
          the grid has been scrolled; same guard rails as the checkpoint panel
          (galleryActionBar decides when it exists at all). */}
      {bar.shown && (
        <div data-testid="gallery-action-bar" data-probe-chrome="gallery-bar"
          className="sticky bottom-0 z-30 -mx-3 flex flex-wrap items-center gap-2 border-t border-border bg-surface-overlay px-3 py-2 sm:-mx-4 sm:px-4">
          <button type="button" data-testid="gallery-select-toggle"
            onClick={() => { setPicking((v) => !v); setSelected(new Set()); setNotice(null); }}
            aria-pressed={bar.togglePressed}
            aria-label={picking ? 'Leave selection mode' : 'Select images to delete or download'}
            title={picking ? 'Leave selection mode' : 'Select images to delete or download'}
            className={`min-h-10 lg:min-h-0 shrink-0 rounded-md border px-3 py-1.5 text-[0.75rem] font-semibold ${picking
              ? 'border-indigo-300 bg-indigo-500/40 text-white'
              : 'border-indigo-400/70 bg-indigo-500/15 text-indigo-200 hover:bg-indigo-500/25'}`}>
            <span aria-hidden>{picking ? '✓' : '☑'}</span> {bar.toggleLabel}
          </button>
          {picking && zipBtn.shown && (
            <button type="button" data-testid="gallery-download-zip"
              onClick={runZip} disabled={zipBtn.disabled} title={zipBtn.title}
              className="min-h-10 lg:min-h-0 shrink-0 rounded-md border border-border px-2.5 py-1.5 text-content-muted text-[0.75rem] hover:border-indigo-400/50 hover:text-content disabled:opacity-40">
              {zipBtn.label}
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
                className="min-h-10 lg:min-h-0 rounded-md border border-border px-2 py-1.5 text-content-muted text-[0.6875rem] hover:text-content">
                {bar.selectAllLabel}
              </button>
              <button type="button" data-testid="gallery-delete"
                disabled={bar.deleteDisabled}
                onClick={() => setConfirming(true)}
                className="ml-auto min-h-10 lg:min-h-0 rounded-md border border-rose-500/50 px-3 py-1.5 text-[0.75rem] text-rose-300 disabled:opacity-40 hover:bg-rose-500/10">
                <Trash2 aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Delete{selected.size ? ` (${selected.size})` : ''}
              </button>
            </>
          )}
        </div>
      )}

      {confirming && (
        <div role="dialog" aria-modal="true" aria-label="Confirm deletion"
          data-testid="gallery-confirm" data-probe-layer
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

      {/* 🔍 The shared viewer, with ‹ › because browsing is this page's loop.
          Handlers are null AT the ends, so the chevron that cannot go anywhere
          is not drawn. ✨ improve is the canvas handler with the Gallery's own
          toast address; withheld for a row that cannot take it, so the refusal
          is read before the click. `dataset_id` travels for Klein's note. */}
      <GeneratedImageLightbox
        img={zoom} alt={zoom ? imageFactsLine(zoom) || 'Generated image' : undefined}
        onClose={() => setZoomIndex(null)}
        onImprove={canImproveCanvasImage(zoom) ? improveImage : undefined}
        onUseImproveSettings={restoreImproveSettings}
        datasetId={zoom?.dataset_id ?? null}
        /* 📷 In the viewer's footer, beside the other verbs. Shown DISABLED
           with its reason rather than hidden when the row cannot take it: a
           button that vanishes teaches nothing, and "why can't I?" is the
           question this panel exists to answer. */
        actions={zoom ? (
          <button type="button" data-testid="lightbox-camera-angles"
            onClick={() => setCameraFor(zoom)}
            disabled={!!cameraRefusal(zoom)}
            title={cameraRefusal(zoom) || 'Re-shoot this scene from another camera position'}
            /* Auto width, NOT `w-full sm:w-auto` like the improve engines: two
               engine buttons side by side would each be a stub on a 400 px
               phone, but ONE verb beside ⬇ Download fits and costs no row. It
               was full-width first and the measurement said no — at 360×800
               the fourth stacked row put this button at y=926 in an 800 px
               window, i.e. reachable only by scrolling the details column. */
            className="min-h-10 lg:min-h-0 inline-flex items-center gap-2 rounded-lg border border-indigo-400/50 bg-indigo-500/20 px-3 py-1.5 text-[0.75rem] font-semibold text-indigo-100 hover:bg-indigo-500/30 disabled:cursor-not-allowed disabled:opacity-45">
            <Camera className="size-3.5" aria-hidden />
            Camera angles
          </button>
        ) : null}
        onPrev={zoomIndex > 0 ? () => setZoomIndex(zoomIndex - 1) : null}
        onNext={zoomIndex != null && zoomIndex < images.length - 1
          ? () => setZoomIndex(zoomIndex + 1) : null} />

      {/* The picker sits ABOVE the viewer (both are fixed layers) so the
          reference picture stays visible behind it while positions are chosen —
          picking an angle of something you cannot see is guesswork. */}
      {cameraFor && (
        <CameraAnglePicker
          onClose={() => setCameraFor(null)}
          onShoot={async (poses) => {
            const ok = await shootCameraViews(cameraFor.id, poses);
            if (ok) setCameraFor(null);
          }} />
      )}
    </div>
  );
}
