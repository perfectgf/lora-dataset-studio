/**
 * The first frame — where the clip starts from, or nothing at all — and the
 * LAST frame, the picture the clip ends on (H3 is first-last-to-video).
 *
 * Five ways in, because a video LoRA gets judged against five different kinds
 * of picture and exporting to disk first would be busywork:
 *
 *   • a file from the machine — the general case;
 *   • an image from the Bank — animating the very portrait the LoRA was
 *     trained from;
 *   • an image from the GALLERY — what this app just generated, which is where
 *     the picture someone wants to animate usually already is (asked for from
 *     live use: it was the one source that sent people back through disk);
 *   • the first frame of a clip in a video training set — the honest baseline,
 *     since that frame is material the LoRA actually saw;
 *   • the last frame of a clip this Studio rendered (the Rendered clip tab) —
 *     the next shot's start, or the picture to end on (the joined render
 *     stays the card's ⏭ Continue).
 *
 * Four end at the same server route, which stages the picture into ComfyUI's
 * input folder with EXIF stripped; the last frame goes through the clip's own
 * last-frame route — the same staging, on both. The component never holds a path
 * from the user's disk: what comes back is the staged NAME the graph will use.
 *
 * SEVERAL AT ONCE (2026-09-02). A pick APPENDS to a strip rather than
 * replacing the frame, and Generate queues one clip per frame on one seed —
 * asked for from the picker: judging a motion LoRA on one portrait is judging
 * it on one portrait. The strip is the parent's list (it is what Generate
 * walks); this component stages, hands the frames up, and draws the strip.
 * A picture already in the strip is not staged twice — its ORIGIN is the
 * key, since the server stages every pick under a fresh name.
 *
 * TWO TARGETS (2026-09-04). "Stage as" above the containers says which frame
 * a pick becomes: the first frame (into the strip, several) or the LAST
 * frame — ONE per launch, shared by every clip of a batch, kept apart from
 * the strip under an `end:` key so the strip's dedupe never applies to it (a
 * clip may loop back onto its own start). Text-only has no first frame to
 * stage, so there the containers serve the last frame alone and the clip
 * resolves from the prompt onto that picture — the row under the containers
 * shows it, and removes it, in both modes.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Image as ImageIcon, Upload, Film, Type, Sparkles, SkipForward, ZoomIn } from 'lucide-react';
import { apiFetch, postJson, postForm } from '@lds/plugin-sdk';
import { useToast } from '@lds/plugin-sdk';
import { HelpBadge } from '@lds/plugin-sdk';
import { datasetThumbUrl } from '@lds/plugin-sdk/canvas';
import VideoPickerGrid from './VideoPickerGrid';
import { datasetClipPoster } from '../../videobank/videoDatasetClips';
import { appendImages, datasetClips, galleryPage } from './videoPickerFeeds';
import { clampTile, gridBoxHeight, readTile, writeTile, TILE_MAX, TILE_MIN, TILE_STEP } from './videoPickerTile';
import { releasePreview, uploadKey } from './videoStartFrames';
import { clipLastFramePngUrl, clipLastFrameUrl, sourceUrl } from './videoStudioApi';

const TABS = [
  { id: 'upload', label: 'Upload', icon: Upload },
  { id: 'bank', label: 'Bank', icon: ImageIcon },
  { id: 'gallery', label: 'Gallery', icon: Sparkles },
  { id: 'clip', label: 'Dataset clip', icon: Film },
  // 🎞 Every clip this Studio rendered offers its last frame as a picture to
  // pick — the first frame of the next shot, or the last frame of this one.
  // Joining the next clip behind it stays the card's ⏭ Continue.
  { id: 'last', label: 'Rendered clip', icon: SkipForward },
];

/* One page of the feed, and the picker walks the same cursor the Gallery page
   walks — 60 thumbnails is what fits without a wait, not how many pictures
   somebody owns. "Show more" fetches the next page; the count says how far in
   the feed you are, so a picture further back is reachable rather than absent. */
const GALLERY_PAGE = 60;
/* The most last-frame tiles the tab lists — the history's own page. */
const LAST_FRAME_TILES = 24;

/** An <img> that becomes its placeholder when the picture cannot load — not a
 * broken-image glyph, not a blank. A bank thumbnail 404s in the ordinary course
 * of things (the bank was deleted, the thumbnails pass never ran), and the tile
 * must still be a tile. The verdict resets with the source, so a tile reused
 * for another clip does not inherit a stale "broken". */
function Poster({ src, className, fallback }) {
  const [broken, setBroken] = useState(false);
  useEffect(() => { setBroken(false); }, [src]);
  if (!src || broken) return fallback;
  return <img src={src} alt="" loading="lazy" decoding="async" onError={() => setBroken(true)} className={className} />;
}

export default function VideoSourcePicker({ mode, onMode, frames = [], onAdd, onRemove, onClear, aspect, onAspect,
  history = [], onOpen = null,
  endFrame = null, onSetEnd = null, onClearEnd = null, onOpenEnd = null, libraryOnly = false,
  allowReferences = true, singleFrame = false, identityReferences = false }) {
  const toast = useToast();
  // The clips whose last frame can be staged: rendered, with a file. The
  // history is the studio's own list — the same one the cards show.
  // Capped: every tile asks the server for a last-frame PNG it extracts on
  // first request, so the tab lists the newest clips rather than the whole
  // history at once.
  const finished = (history || []).filter((c) => c && c.status === 'done').slice(0, LAST_FRAME_TILES);
  // Whether a picture is already in the strip — by where it came from. A
  // tile the strip holds clicks OUT again (a pressed tile, the way a
  // multi-pick grid reads), rather than staging the same portrait under a
  // second name.
  // …or by the library row a staged frame was given when it was opened: the
  // Gallery tab must not offer a picture the strip already holds.
  const selectedKeys = useMemo(() => new Set(frames.flatMap((f) =>
    f.galleryImageId ? [f.key, `gallery:${f.galleryImageId}`] : [f.key])), [frames]);
  const held = (key) => (effTarget === 'end'
    ? !!(endFrame && (endFrame.key === `end:${key}`
      || (endFrame.galleryImageId && key === `gallery:${endFrame.galleryImageId}`)))
    : selectedKeys.has(key));
  // The picks whose staging is in flight: a tile is not in the strip until
  // its POST answers, so a second click in that gap would stage it again.
  const inFlight = useRef(new Set());
  const [tab, setTab] = useState(libraryOnly ? 'bank' : 'upload');
  const [banks, setBanks] = useState([]);
  const [bankId, setBankId] = useState(null);
  const [images, setImages] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [datasetId, setDatasetId] = useState(null);
  const [clips, setClips] = useState([]);
  const [gallery, setGallery] = useState([]);
  const [more, setMore] = useState(null);   // {before, more} — the feed's cursor
  const [paging, setPaging] = useState(false);
  const [busy, setBusy] = useState(false);
  // 🎞 WHICH frame a pick becomes. H3 renders from a first frame TO a last one
  // (first-last-to-video); the same containers serve both, so the choice is a
  // target above the tabs, not a second picker. One last frame per launch.
  const [target, setTarget] = useState('start');
  // Text-only has no first frame to stage: every pick is the last frame there.
  const effTarget = mode === 't2v' ? 'end' : target;
  // One preview size for the three grids, kept per browser. Read lazily: the
  // store is touched once, on mount, not on every render — and never named
  // here: the helper owns it, so a browser that blocks site data cannot throw
  // inside this render. The state follows the value, the store follows the
  // state: a write the store refuses still moves the dial for the session.
  const [tile, setTile] = useState(() => readTile());
  const changeTile = (value) => {
    const next = clampTile(value);
    setTile(next);
    writeTile(next);
  };
  // The one column count for every grid, from the dial: as many tiles of at
  // least `tile` px as the row holds, stretched to fill it — so the phone and
  // the desktop differ by how many columns they get, not by a class each.
  const gridStyle = {
    gridTemplateColumns: `repeat(auto-fill, minmax(${tile}px, 1fr))`,
    maxHeight: `min(${gridBoxHeight(tile)}px, 70vh)`,
  };
  const gridShown = (tab === 'bank' && bankId && images.length > 0)
    || (tab === 'gallery' && gallery.length > 0)
    || (tab === 'clip' && datasetId && clips.length > 0)
    || (tab === 'last' && finished.length > 0);
  const [clipsLoading, setClipsLoading] = useState(false);

  /* Lists are fetched when their tab is opened, never on mount: a bank walk is
     the most expensive GET in the app and this panel is not the bank. */
  useEffect(() => {
    if (tab !== 'bank' || banks.length) return;
    apiFetch('/api/banks').then((d) => setBanks(d.banks || [])).catch(() => setBanks([]));
  }, [tab, banks.length]);
  useEffect(() => {
    if (tab !== 'clip' || datasets.length) return;
    apiFetch('/api/video-datasets').then((d) => setDatasets(d.datasets || []))
      .catch(() => setDatasets([]));
  }, [tab, datasets.length]);
  useEffect(() => {
    if (tab !== 'gallery' || gallery.length) return;
    apiFetch(`/api/gallery/images?limit=${GALLERY_PAGE}`)
      .then((d) => {
        const page = galleryPage(d);
        setGallery(page.images);
        setMore({ before: page.before, more: page.more });
      })
      .catch(() => { setGallery([]); setMore(null); });
  }, [tab, gallery.length]);
  // A list empties the moment its source changes, and a reply that lands
  // after the source moved on is dropped. Without both, the previous bank's or
  // set's tiles stayed up under the new name until the new reply arrived — and
  // a slow first reply could overwrite a fast second one for good.
  useEffect(() => {
    if (!bankId) return undefined;
    let stale = false;
    setImages([]);
    apiFetch(`/api/bank/${bankId}/images?limit=60`)
      .then((d) => { if (!stale) setImages(d.images || []); })
      .catch(() => { if (!stale) setImages([]); });
    return () => { stale = true; };
  }, [bankId]);
  useEffect(() => {
    if (!datasetId) return undefined;
    let stale = false;
    setClips([]);
    setClipsLoading(true);
    // `items`, never `clips` — see videoPickerFeeds: on that payload `clips` is
    // the count, and rendering a number took the whole page down.
    apiFetch(`/api/video-dataset/${datasetId}`)
      .then((d) => { if (!stale) setClips(datasetClips(d)); })
      .catch(() => { if (!stale) setClips([]); })
      .finally(() => { if (!stale) setClipsLoading(false); });
    return () => { stale = true; };
  }, [datasetId]);

  /* The next page of the Gallery feed, appended. Failure leaves what is
     already on screen and simply stops offering more: half a feed still lets
     somebody pick a picture. */
  const showMore = useCallback(async () => {
    if (!more?.more || paging) return;
    setPaging(true);
    try {
      const page = galleryPage(await apiFetch(
        `/api/gallery/images?limit=${GALLERY_PAGE}&before_id=${more.before}`));
      setGallery((list) => appendImages(list, page.images));
      setMore({ before: page.before, more: page.more });
    } catch {
      setMore((m) => (m ? { ...m, more: false } : null));
    } finally {
      setPaging(false);
    }
  }, [more, paging]);

  /* Stage a list of picks — `{ key, preview, send }` each — in order, and hand
     up what staged ONCE, so the parent appends a batch, not a frame at a time.
     A pick the strip already holds is skipped before any request (the same
     file chosen twice from the dialog; a tile is caught earlier, it toggles),
     and so is one whose staging is still in flight (a double click). A
     refusal mid-way refuses that pick alone: the walk goes on, so five
     pictures with one bad file are four start frames and one message that
     counts ("Staged 4 of 5 — …"), not one frame and silence over the rest. */
  const stage = useCallback(async (picks) => {
    // The last frame may be a picture the strip already holds (a clip that
    // loops back onto its start), so the strip's dedupe does not apply to it.
    const seen = new Set(effTarget === 'end' ? [] : frames.map((f) => f.key));
    if (effTarget === 'end' && picks.length > 1) {
      // One last frame per launch: the rest are let go of — their previews
      // too — and SAID, like every other pick this walk drops.
      picks.slice(1).forEach(releasePreview);
      toast.info(`One last frame per launch — kept the first of ${picks.length}.`);
      picks = picks.slice(0, 1);
    }
    const fresh = [];
    let dropped = 0;
    for (const pick of picks) {
      if (inFlight.current.has(pick.key)) { releasePreview(pick); continue; }
      if (seen.has(pick.key)) { releasePreview(pick); dropped += 1; continue; }
      seen.add(pick.key);
      fresh.push(pick);
    }
    if (dropped) {
      toast.info(picks.length === 1
        ? 'Already in the batch — remove it from the strip to pick it again.'
        : `${dropped} already in the batch — skipped.`);
    }
    if (!fresh.length) return;
    setBusy(true);
    fresh.forEach((pick) => inFlight.current.add(pick.key));
    const staged = [];
    let refusal = null;
    try {
      for (const pick of fresh) {
        try {
          const r = await pick.send();
          staged.push({ key: pick.key, image: r.image, ratio: r.ratio, preview: pick.preview || null });
        } catch (e) {
          releasePreview(pick);
          if (!refusal) refusal = e?.message || `That image could not be used as a ${effTarget === 'end' ? 'last' : 'start'} frame.`;
        }
      }
    } finally {
      fresh.forEach((pick) => inFlight.current.delete(pick.key));
      if (staged.length) {
        if (effTarget === 'end') {
          releasePreview(endFrame);
          onSetEnd?.({ ...staged[0], key: `end:${staged[0].key}` });
        } else onAdd(staged);
      }
      if (refusal) {
        toast.error(staged.length ? `Staged ${staged.length} of ${fresh.length} — ${refusal}` : refusal);
      }
      setBusy(false);
    }
  }, [effTarget, endFrame, frames, onAdd, onSetEnd, toast]);

  /* A tile's click: into the strip, or out of it if it is there — and the
     frame that goes out lets go of its upload preview first, like the ✕ does. */
  const toggle = (pick) => {
    if (!held(pick.key)) return stage([pick]);
    if (effTarget === 'end') {
      releasePreview(endFrame);
      return onClearEnd?.();
    }
    releasePreview(frames.find((f) => f.key === pick.key
      || (f.galleryImageId && pick.key === `gallery:${f.galleryImageId}`)));
    return onRemove(pick.key);
  };

  const onFiles = (files) => {
    const pickedFiles = Array.from(files || [])
    const picks = (singleFrame ? pickedFiles.slice(0, 1) : pickedFiles).map((file) => {
      const fd = new FormData();
      fd.append('image', file);
      return { key: uploadKey(file), preview: URL.createObjectURL(file), send: () => postForm(sourceUrl(), fd) };
    });
    if (picks.length) stage(picks);
  };

  return (
    <section data-probe-panel="video-studio-source"
      className="flex flex-col gap-1.5 rounded-xl border border-border bg-surface p-2">
      {!libraryOnly && <header className="flex flex-wrap items-center gap-1.5">
        <h2 className="flex items-center gap-1.5 text-sm font-semibold text-content">
          {libraryOnly ? 'Library images' : mode === 't2v' ? 'Last frame' : 'Start frame'}
          <HelpBadge topic={mode === 't2v' ? 'video-last-frame-tab' : 'video-studio-start-frame'} />
        </h2>
        {!libraryOnly && <div className="ml-auto flex flex-wrap rounded-lg border border-border p-0.5">
          {[['i2v', 'From an image'], ['t2v', 'Text only'], ['ref2va', 'References']].filter(([id]) => allowReferences || id !== 'ref2va').map(([id, label]) => (
            <button key={id} type="button" onClick={() => onMode(id)}
              aria-pressed={mode === id}
              className={`rounded-md px-2 py-1 text-xs min-h-10 lg:min-h-0 ${
                mode === id ? 'bg-primary text-white' : 'text-content-muted hover:text-content'}`}>
              {id === 't2v' && <Type aria-hidden="true" className="mr-1 inline h-3 w-3" />}
              {id === 't2v' && identityReferences ? 'No start frame' : label}
            </button>
          ))}
        </div>}
      </header>}

      {!libraryOnly && (mode === 't2v' ? (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-content-muted">Shape</span>
          {[['landscape', '16:9'], ['portrait', '9:16'], ['square', '1:1']].map(([id, label]) => (
            <button key={id} type="button" onClick={() => onAspect(id)}
              className={`rounded-lg border px-2 py-1 text-xs min-h-10 lg:min-h-0 ${
                (aspect === 'auto' ? 'landscape' : aspect) === id ? 'border-primary bg-primary/10 text-content' : 'border-border text-content-muted'}`}>
              {label}
            </button>
          ))}
          <p className="w-full text-[0.6875rem] text-content-subtle">
            {identityReferences ? 'The model composes the shot from the prompt and identity references. A last frame is optional.'
              : 'No start frame: the model composes the shot from the prompt alone — pick a last frame below and the clip resolves onto that picture.'}
          </p>
        </div>
      ) : (
        <>
          {/* 🎞 The target: which frame a pick becomes. Two choices, so a
              segmented pair — and it sits above the containers because it
              changes what every tile below does. */}
          <div className="flex w-full items-center gap-2">
          <span className="shrink-0 text-xs font-semibold text-content">Stage as</span>
          <div data-testid="video-frame-target" role="radiogroup" aria-label="Stage as"
            className="flex flex-1 gap-1 rounded-lg border border-border bg-app p-0.5">
            {[['start', 'First frame'], ['end', 'Last frame']].map(([id, text]) => (
              <button key={id} type="button" role="radio" aria-checked={target === id}
                onClick={() => setTarget(id)}
                className={`min-h-10 flex-1 rounded-md px-2 py-1 text-xs font-semibold lg:min-h-0 ${
                  target === id ? 'bg-primary text-white' : 'text-content-muted hover:text-content'}`}>
                {text}
              </button>
            ))}
          </div>
          </div>
          <p className="w-full text-[0.6875rem] text-content-subtle">
            {target === 'end'
              ? 'The picture the clip ENDS on — H3 renders from the first frame to it. One per launch; a pick replaces it.'
              : singleFrame ? 'One start frame shared by all selected checkpoints. A new pick replaces it.'
                : 'The picture the clip starts from. Pick several and each queues its own clip.'}
          </p>
        </>
      ))}
      {/* flex-1, not a left-parked group: three chips against a 976 px row
            read as a panel that forgot its content (the responsive probe
            measures exactly that, and flagged this row at 25 %). */}
        <div className="flex w-full flex-wrap gap-1">
          {TABS.filter((t) => !libraryOnly || t.id !== 'upload').map(({ id, label, icon: Icon }) => (
            <button key={id} type="button" onClick={() => setTab(id)}
              data-testid={`video-source-${id}`}
              className={`flex flex-1 items-center justify-center gap-1 rounded-lg border px-2 py-1 text-xs min-h-10 sm:whitespace-nowrap lg:min-h-0 ${
                tab === id ? 'border-primary bg-primary/10 text-content' : 'border-border text-content-muted'}`}>
              <Icon aria-hidden="true" className="h-3.5 w-3.5" />{label}
            </button>
          ))}
          {/* The dial that sizes the tiles, shown only over a grid that has
              some: at the default a face is a smudge, and the frame is chosen
              by eye. It ends the tab strip's row rather than taking one of
              its own (alone, it filled 35 % of a landscape phone's row — a
              row that forgot its content, to the probe) and wraps under the
              tabs on a phone. Above a phone the tab labels stay on one line
              (sm:whitespace-nowrap): a flex row breaks by the longest WORD,
              so without it the dial stayed on the row and "Dataset clip"
              folded in two. Not padlocked like the render dials — a drift
              here shows itself at once and changes nothing about the clip. */}
          {gridShown && (
            <label className="ml-auto flex items-center gap-1.5 text-[0.6875rem] text-content-muted"
              title="Preview size — enlarge the tiles to judge a frame before you pick it">
              <ZoomIn aria-hidden="true" className="h-3.5 w-3.5 shrink-0" />
              <span className="shrink-0">Preview size</span>
              <input type="range" min={TILE_MIN} max={TILE_MAX} step={TILE_STEP} value={tile}
                onChange={(e) => changeTile(e.target.value)}
                aria-label="Preview size" aria-valuetext={`${tile} px`}
                className="w-32 cursor-pointer accent-primary min-h-10 lg:min-h-0" />
            </label>
          )}
        </div>

        {/* The ink spans the whole dropzone rather than huddling in the
            middle: a centred icon plus a centred sentence measured 2 % of the
            row, which the probe reads as an empty box — correctly. */}
        {/* A drop lands here too — the label always said so, and until the
            batch nothing listened: a label around a file input takes no
            drop by itself. */}
        {tab === 'upload' && (
          <label className="flex w-full cursor-pointer items-center gap-2 rounded-lg border border-dashed border-border px-3 py-4 text-xs text-content-muted hover:border-primary/60"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => { e.preventDefault(); if (!busy) onFiles(e.dataTransfer.files); }}>
            <Upload aria-hidden="true" className="h-4 w-4 shrink-0" />
            <span className="flex-1">
              {busy ? 'Preparing…' : (effTarget === 'end'
                ? 'Drop an image here, or choose one from this machine — one last frame per launch'
                : singleFrame ? 'Drop an image here, or choose one shared start frame'
                  : 'Drop images here, or choose them from this machine — several at once queue one clip each')}
            </span>
            <span className="shrink-0 rounded-md border border-border px-2 py-1">
              {busy ? '…' : 'Browse'}
            </span>
            {/* The value is cleared after the pick: a file removed from the
                strip and chosen again is a change the input would otherwise
                not report, the same file being "still" selected. */}
            <input type="file" accept="image/*" multiple={!singleFrame} className="hidden"
              onChange={(e) => { onFiles(e.target.files); e.target.value = ''; }} />
          </label>
        )}

        {tab === 'bank' && (
          <div className="flex flex-col gap-1.5">
            <select value={bankId || ''} onChange={(e) => setBankId(Number(e.target.value) || null)}
              className="w-full rounded-lg border border-border bg-app px-2 py-1.5 text-xs text-content min-h-10 lg:min-h-0">
              <option value="">Pick a bank…</option>
              {banks.map((b) => <option key={b.id} value={b.id}>{b.name}</option>)}
            </select>
            {bankId && (
              <VideoPickerGrid key={bankId} count={images.length} tile={tile} label="Bank images"
                itemKey={(index) => images[index].id}>
                {(index) => {
                  const im = images[index];
                  const key = `bank:${bankId}:${im.id}`;
                  return (
                    <button key={im.id} type="button" title={im.filename} aria-pressed={held(key)}
                      onClick={() => toggle({
                        key,
                        preview: `/api/bank/${bankId}/thumb/${im.id}`,
                        send: () => postJson(sourceUrl(), { bank_id: bankId, image_id: im.id }),
                      })}
                      className={`h-full w-full overflow-hidden rounded-md border hover:border-primary ${
                        held(key) ? 'border-primary ring-2 ring-primary' : 'border-border'}`}>
                      <img src={`/api/bank/${bankId}/thumb/${im.id}`} alt=""
                        loading="lazy" decoding="async" className="h-full w-full object-cover" />
                    </button>
                  );
                }}
              </VideoPickerGrid>
            )}
          </div>
        )}

        {tab === 'gallery' && (
          <div className="flex flex-col gap-1.5">
            {gallery.length === 0 ? (
              <p className="rounded-lg border border-dashed border-border px-3 py-4 text-xs text-content-muted">
                Nothing generated yet — images made in the Studio or on a
                checkpoint show up here, newest first.
              </p>
            ) : (
              <VideoPickerGrid count={gallery.length} tile={tile} label="Gallery images"
                itemKey={(index) => gallery[index].id}>
                {(index) => {
                  const g = gallery[index];
                  const key = `gallery:${g.id}`;
                  return (
                    <button key={g.id} type="button" aria-pressed={held(key)}
                      title={g.prompt || 'Generated image'}
                      onClick={() => toggle({
                        key,
                        preview: g.url,
                        send: () => postJson(sourceUrl(), { gallery_image_id: g.id }),
                      })}
                      className={`h-full w-full overflow-hidden rounded-md border hover:border-primary ${
                        held(key) ? 'border-primary ring-2 ring-primary' : 'border-border'}`}>
                      <img src={datasetThumbUrl(g.url)} alt="" loading="lazy" decoding="async"
                        className="h-full w-full object-cover" />
                    </button>
                  );
                }}
              </VideoPickerGrid>
            )}
            {/* How much of the feed is on screen, and the way to the rest.
                Without this the newest 60 read as the whole Gallery. */}
            <div className="flex flex-wrap items-center gap-2">
              <p className="min-w-0 flex-1 text-[0.6875rem] text-content-subtle">
                Animates the picture at full size, not its thumbnail.
                {gallery.length > 0 && (
                  <span className="ml-1">
                    {more?.more ? `Newest ${gallery.length}.` : `All ${gallery.length}.`}
                  </span>
                )}
              </p>
              {more?.more && (
                <button type="button" onClick={showMore} disabled={paging}
                  className="shrink-0 rounded-lg border border-border px-2 py-1 text-[0.6875rem] text-content-muted hover:border-primary hover:text-content disabled:opacity-50 min-h-10 lg:min-h-0">
                  {paging ? 'Loading…' : 'Show older'}
                </button>
              )}
            </div>
          </div>
        )}

        {tab === 'clip' && (
          <div className="flex flex-col gap-1.5">
            <select value={datasetId || ''} onChange={(e) => setDatasetId(Number(e.target.value) || null)}
              className="w-full rounded-lg border border-border bg-app px-2 py-1.5 text-xs text-content min-h-10 lg:min-h-0">
              <option value="">Pick a video training set…</option>
              {datasets.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
            {datasetId && !clipsLoading && clips.length === 0 && (
              <p className="rounded-lg border border-dashed border-border px-3 py-4 text-xs text-content-muted">
                No clip in that training set — or it could not be read.
              </p>
            )}
            {/* A GRID, like the Bank and Gallery tabs — not a flex column: capped
                at max-h-72, a flex column shrinks its rows to fit instead of
                scrolling (truncate's overflow:hidden zeroes their min-height),
                and 21 clips arrived as 21 unreadable 12 px slivers. Grid rows
                keep their size; the box scrolls. And a picture per tile, the
                training set's own poster, so the clip is chosen by eye. */}
            {datasetId && clips.length > 0 && (
              <div className="grid gap-1 overflow-y-auto" style={gridStyle}>
                {clips.map((c) => {
                  const poster = datasetClipPoster(datasetId, c);
                  const key = `clip:${datasetId}:${c.filename}`;
                  return (
                    <button key={c.id} type="button" title={c.filename} aria-pressed={held(key)}
                      onClick={() => toggle({
                        key,
                        preview: poster,
                        send: () => postJson(sourceUrl(), { dataset_id: datasetId, filename: c.filename }),
                      })}
                      className={`flex min-w-0 flex-col overflow-hidden rounded-md border hover:border-primary ${
                        held(key) ? 'border-primary ring-2 ring-primary' : 'border-border'}`}>
                      <Poster src={poster} className="aspect-square w-full object-cover"
                        fallback={(
                          <span aria-hidden="true"
                            className="grid aspect-square w-full place-items-center bg-app text-xl text-content-subtle">
                            🎞
                          </span>
                        )} />
                      <span className="w-full truncate px-1 text-left text-[0.625rem] text-content-muted">
                        {c.filename}
                      </span>
                    </button>
                  );
                })}
              </div>
            )}
            <p className="text-[0.6875rem] text-content-subtle">
              Uses the clip’s first frame, at full size — the same material the
              LoRA trained on.
            </p>
          </div>
        )}

        {tab === 'last' && (
          <div className="flex flex-col gap-1.5">
            {finished.length === 0 ? (
              <p className="rounded-lg border border-dashed border-border px-3 py-4 text-xs text-content-muted">
                No finished clip yet — every clip this Studio renders offers its
                last frame here.
              </p>
            ) : (
              <div className="grid gap-1 overflow-y-auto" style={gridStyle}>
                {finished.map((c) => {
                  const key = `last:${c.id}`;
                  return (
                    <button key={c.id} type="button" aria-pressed={held(key)}
                      title={`Last frame of clip #${c.id}`}
                      onClick={() => toggle({
                        key,
                        preview: clipLastFramePngUrl(c.id),
                        send: () => postJson(clipLastFrameUrl(c.id), {}),
                      })}
                      className={`relative aspect-square overflow-hidden rounded-md border hover:border-primary ${
                        held(key) ? 'border-primary ring-2 ring-primary' : 'border-border'}`}>
                      <Poster src={clipLastFramePngUrl(c.id)} className="h-full w-full object-cover"
                        fallback={(
                          <span className="flex h-full w-full items-center justify-center bg-surface text-content-subtle">
                            <Film aria-hidden="true" className="h-5 w-5" />
                          </span>
                        )} />
                      <span className="absolute bottom-0 left-0 rounded-tr bg-black/70 px-1 text-[0.625rem] text-white">#{c.id}</span>
                    </button>
                  );
                })}
              </div>
            )}
            <p className="text-[0.6875rem] text-content-subtle">
              A finished clip’s last frame, at full size — the first frame of the
              next shot, or the last frame of this one. To join the next clip
              behind it instead, use ⏭ Continue on the clip’s card.
              {(history || []).filter((c) => c && c.status === 'done').length > LAST_FRAME_TILES
                && ` Newest ${LAST_FRAME_TILES} clips.`}
            </p>
          </div>
        )}

        {/* The strip: what the next launch walks, in pick order, each frame
            with its ✕ (the same corner the reference panel uses). One frame
            reads as it always did; several say what a click will do. */}
        {!libraryOnly && mode !== 't2v' && frames.length > 0 && (
          <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-app p-1.5">
            {frames.map((f, i) => (
              <div key={f.key} className="relative shrink-0" title={f.image}>
                {/* 🔍 The tile opens the SHARED viewer (the host wires it): the
                    same picture at full size with the same verbs as the Gallery.
                    Without a host handler the tile stays a picture, as before. */}
                {onOpen ? (
                  <button type="button" onClick={() => onOpen(f, i)}
                    aria-label={`Open start frame ${i + 1}`}
                    title="Open at full size — the same actions as the Gallery"
                    className="block rounded-md focus:outline-none focus:ring-2 focus:ring-primary">
                    <Poster src={f.preview} className="h-14 w-14 rounded-md object-cover"
                      fallback={(
                        <span className="flex h-14 w-14 items-center justify-center rounded-md bg-surface text-content-subtle">
                          <ImageIcon aria-hidden="true" className="h-5 w-5" />
                        </span>
                      )} />
                  </button>
                ) : (
                  <Poster src={f.preview} className="h-14 w-14 rounded-md object-cover"
                    fallback={(
                      <span className="flex h-14 w-14 items-center justify-center rounded-md bg-surface text-content-subtle">
                        <ImageIcon aria-hidden="true" className="h-5 w-5" />
                      </span>
                    )} />
                )}
                <button type="button" onClick={() => { releasePreview(f); onRemove(f.key); }} disabled={busy}
                  aria-label={`Remove start frame ${i + 1}`} title="Remove this start frame"
                  className="absolute top-0 right-0 flex h-4 w-4 items-center justify-center rounded-bl bg-black/70 text-[0.625rem] leading-none text-white disabled:opacity-40">
                  ✕
                </button>
              </div>
            ))}
            <span className="min-w-[10rem] flex-1 text-[0.6875rem] text-content-muted">
              {frames.length === 1 ? (
                <>
                  Ready — staged into ComfyUI as
                  <code className="ml-1 break-all">{frames[0].image}</code>
                </>
              ) : (
                <>{frames.length} start frames — one clip each, on one seed.</>
              )}
            </span>
            {frames.length > 1 && (
              <button type="button" onClick={() => { frames.forEach(releasePreview); onClear(); }} disabled={busy}
                className="shrink-0 rounded-lg border border-border px-2 py-1 text-[0.6875rem] text-content-muted hover:border-primary hover:text-content disabled:opacity-50 min-h-10 lg:min-h-0">
                Clear all
              </button>
            )}
          </div>
        )}
        {/* 🎞 The last frame — ONE per launch, every clip of the batch ends on
            it. Its own row under the strip, with the same ✕ and the same
            viewer as a start frame. */}
        {endFrame && (
          <div data-testid="video-end-frame" className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-app p-1.5">
            <div className="relative shrink-0" title={endFrame.image}>
              {onOpenEnd ? (
                <button type="button" onClick={() => onOpenEnd(endFrame)} aria-label="Open the last frame"
                  title="Open at full size — the same actions as the Gallery"
                  className="block rounded-md focus:outline-none focus:ring-2 focus:ring-primary">
                  <Poster src={endFrame.preview} className="h-14 w-14 rounded-md object-cover"
                    fallback={(
                      <span className="flex h-14 w-14 items-center justify-center rounded-md bg-surface text-content-subtle">
                        <ImageIcon aria-hidden="true" className="h-5 w-5" />
                      </span>
                    )} />
                </button>
              ) : (
                <Poster src={endFrame.preview} className="h-14 w-14 rounded-md object-cover"
                  fallback={(
                    <span className="flex h-14 w-14 items-center justify-center rounded-md bg-surface text-content-subtle">
                      <ImageIcon aria-hidden="true" className="h-5 w-5" />
                    </span>
                  )} />
              )}
              <button type="button" onClick={() => { releasePreview(endFrame); onClearEnd?.(); }} disabled={busy}
                aria-label="Remove the last frame" title="Remove the last frame"
                className="absolute top-0 right-0 flex h-4 w-4 items-center justify-center rounded-bl bg-black/70 text-[0.625rem] leading-none text-white disabled:opacity-40">
                ✕
              </button>
            </div>
            <span className="min-w-[10rem] flex-1 text-[0.6875rem] text-content-muted">
              Last frame — the clip ends on this picture, staged as
              <code className="ml-1 break-all">{endFrame.image}</code>
            </span>
          </div>
        )}
    </section>
  );
}
