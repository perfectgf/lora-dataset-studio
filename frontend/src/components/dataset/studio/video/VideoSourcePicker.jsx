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
 */
import { useCallback, useEffect, useState } from 'react';
import { Image as ImageIcon, Upload, Film, Type, Sparkles } from 'lucide-react';
import { apiFetch, postJson, postForm } from '../../../../api/fetchClient';
import { useToast } from '../../../common/Toast';
import { appendImages, datasetClips, galleryPage } from './videoPickerFeeds';
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

export default function VideoSourcePicker({ mode, onMode, image, preview, onPicked, aspect, onAspect }) {
  const toast = useToast();
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
  useEffect(() => {
    if (!bankId) return;
    apiFetch(`/api/bank/${bankId}/images?limit=60`)
      .then((d) => setImages(d.images || [])).catch(() => setImages([]));
  }, [bankId]);
  useEffect(() => {
    if (!datasetId) return;
    // `items`, never `clips` — see videoPickerFeeds: on that payload `clips` is
    // the count, and rendering a number took the whole page down.
    apiFetch(`/api/video-dataset/${datasetId}`)
      .then((d) => setClips(datasetClips(d))).catch(() => setClips([]));
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

  const stage = useCallback(async (send) => {
    setBusy(true);
    try {
      const r = await send();
      onPicked({ image: r.image, ratio: r.ratio, preview: r.preview || null });
    } catch (e) {
      toast.error(e?.message || 'That image could not be used as a start frame.');
    } finally {
      setBusy(false);
    }
  }, [onPicked, toast]);

  const onFile = (file) => {
    if (!file) return;
    const fd = new FormData();
    fd.append('image', file);
    const localPreview = URL.createObjectURL(file);
    stage(async () => ({ ...(await postForm(sourceUrl(), fd)), preview: localPreview }));
  };

  return (
    <section data-probe-panel="video-studio-source"
      className="flex flex-col gap-1.5 rounded-xl border border-border bg-surface p-2">
      <header className="flex flex-wrap items-center gap-1.5">
        <h2 className="text-sm font-semibold text-content">Start frame</h2>
        <div className="ml-auto flex rounded-lg border border-border p-0.5">
          {[['i2v', 'From an image'], ['t2v', 'Text only']].map(([id, label]) => (
            <button key={id} type="button" onClick={() => onMode(id)}
              className={`rounded-md px-2 py-1 text-xs min-h-10 lg:min-h-0 ${
                mode === id ? 'bg-accent text-accent-contrast' : 'text-content-muted hover:text-content'}`}>
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
                aspect === id ? 'border-accent bg-accent/10 text-content' : 'border-border text-content-muted'}`}>
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
          <div className="flex w-full gap-1">
            {TABS.map(({ id, label, icon: Icon }) => (
              <button key={id} type="button" onClick={() => setTab(id)}
                data-testid={`video-source-${id}`}
                className={`flex flex-1 items-center justify-center gap-1 rounded-lg border px-2 py-1 text-xs min-h-10 lg:min-h-0 ${
                  tab === id ? 'border-accent bg-accent/10 text-content' : 'border-border text-content-muted'}`}>
                <Icon aria-hidden="true" className="h-3.5 w-3.5" />{label}
              </button>
            ))}
          </div>

          {/* The ink spans the whole dropzone rather than huddling in the
              middle: a centred icon plus a centred sentence measured 2 % of the
              row, which the probe reads as an empty box — correctly. */}
          {tab === 'upload' && (
            <label className="flex w-full cursor-pointer items-center gap-2 rounded-lg border border-dashed border-border px-3 py-4 text-xs text-content-muted hover:border-accent/60">
              <Upload aria-hidden="true" className="h-4 w-4 shrink-0" />
              <span className="flex-1">
                {busy ? 'Preparing…' : 'Drop an image here, or choose one from this machine'}
              </span>
              <span className="shrink-0 rounded-md border border-border px-2 py-1">
                {busy ? '…' : 'Browse'}
              </span>
              <input type="file" accept="image/*" className="hidden"
                onChange={(e) => onFile(e.target.files?.[0])} />
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
                <div className="grid max-h-56 grid-cols-4 gap-1 overflow-y-auto sm:grid-cols-6">
                  {images.map((im) => (
                    <button key={im.id} type="button" title={im.filename}
                      onClick={() => stage(async () => ({
                        ...(await postJson(sourceUrl(), { bank_id: bankId, image_id: im.id })),
                        preview: `/api/bank/${bankId}/thumb/${im.id}`,
                      }))}
                      className="aspect-square overflow-hidden rounded-md border border-border hover:border-accent">
                      <img src={`/api/bank/${bankId}/thumb/${im.id}`} alt=""
                        loading="lazy" className="h-full w-full object-cover" />
                    </button>
                  ))}
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
                <div className="grid max-h-72 grid-cols-4 gap-1 overflow-y-auto sm:grid-cols-6">
                  {gallery.map((g) => (
                    <button key={g.id} type="button"
                      title={g.prompt || 'Generated image'}
                      onClick={() => stage(async () => ({
                        ...(await postJson(sourceUrl(), { gallery_image_id: g.id })),
                        preview: g.url,
                      }))}
                      className="aspect-square overflow-hidden rounded-md border border-border hover:border-accent">
                      <img src={g.url} alt="" loading="lazy"
                        className="h-full w-full object-cover" />
                    </button>
                  ))}
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
                    className="shrink-0 rounded-lg border border-border px-2 py-1 text-[0.6875rem] text-content-muted hover:border-accent hover:text-content disabled:opacity-50 min-h-10 lg:min-h-0">
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
              {datasetId && clips.length === 0 && (
                <p className="rounded-lg border border-dashed border-border px-3 py-4 text-xs text-content-muted">
                  No clip in that training set — or it could not be read.
                </p>
              )}
              {datasetId && (
                <div className="flex max-h-72 flex-col gap-1 overflow-y-auto">
                  {clips.map((c) => (
                    <button key={c.id} type="button"
                      onClick={() => stage(async () => postJson(sourceUrl(), {
                        dataset_id: datasetId, filename: c.filename,
                      }))}
                      className="truncate rounded-lg border border-border px-2 py-1.5 text-left text-xs text-content-muted hover:border-accent min-h-10 lg:min-h-0">
                      {c.filename}
                    </button>
                  ))}
                </div>
              )}
              <p className="text-[0.6875rem] text-content-subtle">
                Uses the clip’s first frame, at full size — the same material the
                LoRA trained on.
              </p>
            </div>
          )}

          {image && (
            <div className="flex items-center gap-2 rounded-lg border border-border bg-app p-1.5">
              {preview
                ? <img src={preview} alt="" className="h-14 w-14 rounded-md object-cover" />
                : <span className="flex h-14 w-14 items-center justify-center rounded-md bg-surface text-content-subtle">
                    <ImageIcon aria-hidden="true" className="h-5 w-5" />
                  </span>}
              <span className="min-w-0 flex-1 text-[0.6875rem] text-content-muted">
                Ready — staged into ComfyUI as
                <code className="ml-1 break-all">{image}</code>
              </span>
            </div>
          )}
        </>
      )}
    </section>
  );
}
