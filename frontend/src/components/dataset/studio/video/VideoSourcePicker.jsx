/**
 * The first frame — where the clip starts from, or nothing at all.
 *
 * Four ways in, because a video LoRA gets judged against four different kinds
 * of picture and exporting to disk first would be busywork:
 *
 *   • a file from the machine — the general case;
 *   • an image from the Bank — animating the very portrait the LoRA was
 *     trained from;
 *   • an image from the GALLERY — what this app just generated, which is where
 *     the picture someone wants to animate usually already is (asked for from
 *     live use: it was the one source that sent people back through disk);
 *   • the first frame of a clip in a video training set — the honest baseline,
 *     since that frame is material the LoRA actually saw.
 *
 * All four end at the same server route, which stages the picture into
 * ComfyUI's input folder with EXIF stripped. The component never holds a path
 * from the user's disk: what comes back is the staged NAME the graph will use.
 *
 * SEVERAL AT ONCE (2026-09-02). A pick APPENDS to a strip rather than
 * replacing the frame, and Generate queues one clip per frame on one seed —
 * asked for from the picker: judging a motion LoRA on one portrait is judging
 * it on one portrait. The strip is the parent's list (it is what Generate
 * walks); this component stages, hands the frames up, and draws the strip.
 * A picture already in the strip is not staged twice — its ORIGIN is the
 * key, since the server stages every pick under a fresh name.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { Image as ImageIcon, Upload, Film, Type, Sparkles, ZoomIn } from 'lucide-react';
import { apiFetch, postJson, postForm } from '../../../../api/fetchClient';
import { useToast } from '../../../common/Toast';
import { HelpBadge } from '../../../../help/HelpMode';
import { datasetClipPoster } from '../../../videobank/videoDatasetClips';
import { appendImages, datasetClips, galleryPage } from './videoPickerFeeds';
import { clampTile, gridBoxHeight, readTile, writeTile, TILE_MAX, TILE_MIN, TILE_STEP } from './videoPickerTile';
import { releasePreview, uploadKey } from './videoStartFrames';
import { sourceUrl } from './videoStudioApi';

const TABS = [
  { id: 'upload', label: 'Upload', icon: Upload },
  { id: 'bank', label: 'Bank', icon: ImageIcon },
  { id: 'gallery', label: 'Gallery', icon: Sparkles },
  { id: 'clip', label: 'Dataset clip', icon: Film },
];

/* One page of the feed, and the picker walks the same cursor the Gallery page
   walks — 60 thumbnails is what fits without a wait, not how many pictures
   somebody owns. "Show more" fetches the next page; the count says how far in
   the feed you are, so a picture further back is reachable rather than absent. */
const GALLERY_PAGE = 60;

/** An <img> that becomes its placeholder when the picture cannot load — not a
 * broken-image glyph, not a blank. A bank thumbnail 404s in the ordinary course
 * of things (the bank was deleted, the thumbnails pass never ran), and the tile
 * must still be a tile. The verdict resets with the source, so a tile reused
 * for another clip does not inherit a stale "broken". */
function Poster({ src, className, fallback }) {
  const [broken, setBroken] = useState(false);
  useEffect(() => { setBroken(false); }, [src]);
  if (!src || broken) return fallback;
  return <img src={src} alt="" loading="lazy" onError={() => setBroken(true)} className={className} />;
}

export default function VideoSourcePicker({ mode, onMode, frames = [], onAdd, onRemove, onClear, aspect, onAspect }) {
  const toast = useToast();
  // Whether a picture is already in the strip — by where it came from. A
  // tile the strip holds clicks OUT again (a pressed tile, the way a
  // multi-pick grid reads), rather than staging the same portrait under a
  // second name.
  const held = (key) => frames.some((f) => f.key === key);
  // The picks whose staging is in flight: a tile is not in the strip until
  // its POST answers, so a second click in that gap would stage it again.
  const inFlight = useRef(new Set());
  const [tab, setTab] = useState('upload');
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
    || (tab === 'clip' && datasetId && clips.length > 0);
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
    const seen = new Set(frames.map((f) => f.key));
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
          if (!refusal) refusal = e?.message || 'That image could not be used as a start frame.';
        }
      }
    } finally {
      fresh.forEach((pick) => inFlight.current.delete(pick.key));
      if (staged.length) onAdd(staged);
      if (refusal) {
        toast.error(staged.length ? `Staged ${staged.length} of ${fresh.length} — ${refusal}` : refusal);
      }
      setBusy(false);
    }
  }, [frames, onAdd, toast]);

  /* A tile's click: into the strip, or out of it if it is there. */
  const toggle = (pick) => (held(pick.key) ? onRemove(pick.key) : stage([pick]));

  const onFiles = (files) => {
    const picks = Array.from(files || []).map((file) => {
      const fd = new FormData();
      fd.append('image', file);
      return { key: uploadKey(file), preview: URL.createObjectURL(file), send: () => postForm(sourceUrl(), fd) };
    });
    if (picks.length) stage(picks);
  };

  return (
    <section data-probe-panel="video-studio-source"
      className="flex flex-col gap-1.5 rounded-xl border border-border bg-surface p-2">
      <header className="flex flex-wrap items-center gap-1.5">
        <h2 className="flex items-center gap-1.5 text-sm font-semibold text-content">
          Start frame
          <HelpBadge topic="video-studio-start-frame" />
        </h2>
        <div className="ml-auto flex rounded-lg border border-border p-0.5">
          {[['i2v', 'From an image'], ['t2v', 'Text only']].map(([id, label]) => (
            <button key={id} type="button" onClick={() => onMode(id)}
              className={`rounded-md px-2 py-1 text-xs min-h-10 lg:min-h-0 ${
                mode === id ? 'bg-primary text-white' : 'text-content-muted hover:text-content'}`}>
              {id === 't2v' && <Type aria-hidden="true" className="mr-1 inline h-3 w-3" />}
              {label}
            </button>
          ))}
        </div>
      </header>

      {mode === 't2v' ? (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-content-muted">Shape</span>
          {[['landscape', '16:9'], ['portrait', '9:16'], ['square', '1:1']].map(([id, label]) => (
            <button key={id} type="button" onClick={() => onAspect(id)}
              className={`rounded-lg border px-2 py-1 text-xs min-h-10 lg:min-h-0 ${
                aspect === id ? 'border-primary bg-primary/10 text-content' : 'border-border text-content-muted'}`}>
              {label}
            </button>
          ))}
          <p className="w-full text-[0.6875rem] text-content-subtle">
            No start frame: the model composes the shot from the prompt alone.
          </p>
        </div>
      ) : (
        <>
          {/* flex-1, not a left-parked group: three chips against a 976 px row
              read as a panel that forgot its content (the responsive probe
              measures exactly that, and flagged this row at 25 %). */}
          <div className="flex w-full flex-wrap gap-1">
            {TABS.map(({ id, label, icon: Icon }) => (
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
                {busy ? 'Preparing…' : 'Drop images here, or choose them from this machine — several at once queue one clip each'}
              </span>
              <span className="shrink-0 rounded-md border border-border px-2 py-1">
                {busy ? '…' : 'Browse'}
              </span>
              {/* The value is cleared after the pick: a file removed from the
                  strip and chosen again is a change the input would otherwise
                  not report, the same file being "still" selected. */}
              <input type="file" accept="image/*" multiple className="hidden"
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
                <div className="grid gap-1 overflow-y-auto" style={gridStyle}>
                  {images.map((im) => {
                    const key = `bank:${bankId}:${im.id}`;
                    return (
                      <button key={im.id} type="button" title={im.filename} aria-pressed={held(key)}
                        onClick={() => toggle({
                          key,
                          preview: `/api/bank/${bankId}/thumb/${im.id}`,
                          send: () => postJson(sourceUrl(), { bank_id: bankId, image_id: im.id }),
                        })}
                        className={`aspect-square overflow-hidden rounded-md border hover:border-primary ${
                          held(key) ? 'border-primary ring-2 ring-primary' : 'border-border'}`}>
                        <img src={`/api/bank/${bankId}/thumb/${im.id}`} alt=""
                          loading="lazy" className="h-full w-full object-cover" />
                      </button>
                    );
                  })}
                </div>
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
                <div className="grid gap-1 overflow-y-auto" style={gridStyle}>
                  {gallery.map((g) => {
                    const key = `gallery:${g.id}`;
                    return (
                      <button key={g.id} type="button" aria-pressed={held(key)}
                        title={g.prompt || 'Generated image'}
                        onClick={() => toggle({
                          key,
                          preview: g.url,
                          send: () => postJson(sourceUrl(), { gallery_image_id: g.id }),
                        })}
                        className={`aspect-square overflow-hidden rounded-md border hover:border-primary ${
                          held(key) ? 'border-primary ring-2 ring-primary' : 'border-border'}`}>
                        <img src={g.url} alt="" loading="lazy"
                          className="h-full w-full object-cover" />
                      </button>
                    );
                  })}
                </div>
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

          {/* The strip: what the next launch walks, in pick order, each frame
              with its ✕ (the same corner the reference panel uses). One frame
              reads as it always did; several say what a click will do. */}
          {frames.length > 0 && (
            <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-app p-1.5">
              {frames.map((f, i) => (
                <div key={f.key} className="relative shrink-0" title={f.image}>
                  <Poster src={f.preview} className="h-14 w-14 rounded-md object-cover"
                    fallback={(
                      <span className="flex h-14 w-14 items-center justify-center rounded-md bg-surface text-content-subtle">
                        <ImageIcon aria-hidden="true" className="h-5 w-5" />
                      </span>
                    )} />
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
                  <>{frames.length} start frames — one clip each, on one seed; ✨ reads the first.</>
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
        </>
      )}
    </section>
  );
}
