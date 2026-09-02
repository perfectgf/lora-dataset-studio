import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { apiFetch, postJson } from '../../api/fetchClient';
import { useToast } from '../common/Toast';
import { useFocusTrap } from '../../hooks/useFocusTrap';
import { checkpointFileLabel } from '../../utils/generatedImageFacts';
import {
  CIVITAI_API, CIVITAI_POLL_MS, civitaiLinkLine, civitaiTarget, civitaiTargetKnown,
  draftFormFrom, draftFormRefusal, jobOutcome, jobPhaseLabel,
} from './civitaiPublish';

/* 📤 Publish to Civitai — ONE modal, two doors.

   From the checkpoint popover (◉ Graph, LoRA Canvas) it is about a CHECKPOINT:
   mark the Civitai model page it already has (paste the address), or create
   that page from here — model, version, the .safetensors uploaded, as a draft
   to finish on Civitai or published straight away.

   From the shared image viewer it is about a PICTURE: post it on the page its
   checkpoint is linked to, with the prompt, seed, sampler and LoRA weight it
   was made with. The link is what turns that into one press; a picture whose
   checkpoint has no page yet is walked to the marking step right here.

   Both doors are the same component on purpose — the "which page is this"
   question is answered once (civitaiPublish.js), the same words describe the
   same state everywhere, and a host that gains the modal gains both halves.

   The slow work (a checkpoint over a home uplink) runs server-side as a job;
   this dialog polls it and draws the phase. Closing mid-job abandons nothing:
   the upload keeps going and the toast says where to look. */

const FIELD =
  'w-full rounded-lg border border-border bg-surface-raised px-3 py-1.5 text-sm text-content '
  + 'placeholder:text-content-subtle focus:border-indigo-500 outline-none disabled:opacity-50';
const LABEL = 'text-content-muted text-xs';
const PRIMARY =
  'rounded-lg bg-gradient-primary px-3 py-1.5 text-sm font-semibold text-gray-950 '
  + 'disabled:cursor-not-allowed disabled:opacity-40 flex items-center gap-2';
const SECONDARY =
  'rounded-lg border border-border bg-surface px-3 py-1.5 text-sm text-content-muted '
  + 'hover:text-content disabled:opacity-40';
const TAB = (on) =>
  `rounded-md px-2.5 py-1 text-xs font-medium border ${on
    ? 'border-indigo-500 bg-indigo-500/10 text-content'
    : 'border-border text-content-muted hover:text-content'}`;

function Spinner() {
  return <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-white/40 border-t-white" />;
}

/** The job's progress, phase and outcome — shared by both doors. */
function JobPanel({ job, onDone }) {
  const outcome = jobOutcome(job);
  useEffect(() => { if (outcome?.ok) onDone?.(job); }, [outcome?.ok]);   // eslint-disable-line react-hooks/exhaustive-deps
  if (!job) return null;
  if (outcome) {
    return (
      <div data-testid="civitai-job-outcome"
        className={`rounded-lg border px-3 py-2 text-sm ${outcome.ok
          ? 'border-emerald-400/50 bg-emerald-500/10 text-emerald-100'
          : 'border-rose-400/50 bg-rose-500/10 text-rose-200'}`}>
        <p className="m-0">{outcome.ok ? '✓ ' : ''}{outcome.text}</p>
        {outcome.url && (
          <a href={outcome.url} target="_blank" rel="noreferrer"
            className="mt-1 block break-all text-indigo-300 underline">{outcome.url}</a>
        )}
      </div>
    );
  }
  const pct = Math.round(Math.max(0, Math.min(1, Number(job.progress) || 0)) * 100);
  return (
    <div data-testid="civitai-job-progress" className="flex flex-col gap-1">
      <p className="m-0 flex items-center gap-2 text-sm text-content"><Spinner /> {jobPhaseLabel(job)}</p>
      <div className="h-1.5 w-full overflow-hidden rounded bg-white/10">
        <div className="h-full bg-indigo-400 transition-[width]" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function CivitaiPublishModal({ context, onClose }) {
  const toast = useToast();
  const dialogRef = useRef(null);
  const closeRef = useRef(null);
  const target = useMemo(() => civitaiTarget(context), [context]);
  const known = civitaiTargetKnown(target);
  const isImage = context?.kind === 'image';
  const img = isImage ? context.img : null;

  const [status, setStatus] = useState(null);           // {has_key, username, link_host}
  const [link, setLink] = useState(undefined);          // undefined = loading, null = none
  const [datasetLinks, setDatasetLinks] = useState([]);
  const [pickedLinkId, setPickedLinkId] = useState(null);
  const [pane, setPane] = useState('mark');             // mark | create
  const [ref, setRef] = useState('');
  const [linking, setLinking] = useState(false);
  const [linkError, setLinkError] = useState(null);
  const [defaults, setDefaults] = useState(null);
  const [defaultsError, setDefaultsError] = useState(null);
  const [form, setForm] = useState(null);
  const [postTitle, setPostTitle] = useState('');
  const [publishNow, setPublishNow] = useState(true);
  const [job, setJob] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  useFocusTrap(dialogRef, true);
  useEffect(() => { closeRef.current?.focus(); }, []);

  // Who is signed in and whether a key exists at all — the header line and
  // the gate on every button.
  useEffect(() => {
    let alive = true;
    apiFetch(CIVITAI_API.status)
      .then((d) => { if (alive) setStatus(d); })
      .catch(() => { if (alive) setStatus({ has_key: false, username: null }); });
    return () => { alive = false; };
  }, []);

  // The link of THIS checkpoint, and — for a picture — every linked
  // checkpoint of its dataset, so a row without a stamped checkpoint can
  // still be aimed at a page.
  useEffect(() => {
    let alive = true;
    if (known) {
      apiFetch(CIVITAI_API.link(target.recordId, target.step, target.filename))
        .then((d) => { if (alive) setLink(d.link || null); })
        .catch(() => { if (alive) setLink(null); });
    } else {
      setLink(null);
    }
    if (isImage && target?.datasetId != null) {
      apiFetch(CIVITAI_API.datasetLinks(target.datasetId))
        .then((d) => { if (alive) setDatasetLinks(d.links || []); })
        .catch(() => { /* the picker simply stays empty */ });
    }
    return () => { alive = false; };
  }, [known, isImage, target?.recordId, target?.step, target?.filename, target?.datasetId]);

  // The create form is derived server-side (name, base model, trigger,
  // description, the file's facts) — fetched the first time that pane opens.
  useEffect(() => {
    if (pane !== 'create' || !known || defaults || defaultsError) return undefined;
    let alive = true;
    apiFetch(CIVITAI_API.draftDefaults(target.recordId, target.step, target.filename))
      .then((d) => { if (!alive) return; setDefaults(d); setForm(draftFormFrom(d)); })
      .catch((e) => { if (alive) setDefaultsError(e?.message || 'Could not prepare the page.'); });
    return () => { alive = false; };
  }, [pane, known, defaults, defaultsError, target?.recordId, target?.step, target?.filename]);

  // The job heartbeat, only while one runs.
  useEffect(() => {
    if (!job?.id || (job.state !== 'running')) return undefined;
    let alive = true;
    const tick = async () => {
      try {
        const d = await apiFetch(CIVITAI_API.job(job.id), { background: true });
        if (alive) setJob((cur) => (cur && cur.id === d.id ? { ...cur, ...d } : cur));
      } catch (e) {
        if (alive && e?.status === 404) {
          setJob((cur) => (cur ? { ...cur, state: 'error', error: e.message, error_code: 'lost' } : cur));
        }
      }
    };
    const h = setInterval(tick, CIVITAI_POLL_MS);
    return () => { alive = false; clearInterval(h); };
  }, [job?.id, job?.state]);

  const close = useCallback((e) => {
    e?.stopPropagation?.();
    if (job?.state === 'running') {
      toast.success('Still uploading in the background - nothing is lost. Check your Civitai '
        + 'account in a few minutes.');
    }
    onClose?.();
  }, [job?.state, onClose, toast]);

  const markPage = async () => {
    if (!known || !ref.trim() || linking) return;
    setLinking(true); setLinkError(null);
    try {
      const d = await postJson(CIVITAI_API.createLink,
        { record_id: target.recordId, step: target.step, filename: target.filename, url: ref.trim() });
      setLink(d.link);
      toast.success(`Checkpoint linked to ${civitaiLinkLine(d.link)}`);
    } catch (e) {
      setLinkError(e?.message || 'Could not link this page.');
    } finally {
      setLinking(false);
    }
  };

  const unlink = async () => {
    if (!link || linking) return;
    setLinking(true);
    try {
      await postJson(CIVITAI_API.deleteLink(link.id), {});
      setLink(null);
    } catch (e) {
      toast.error(e?.message || 'Could not unlink.');
    } finally {
      setLinking(false);
    }
  };

  const createPage = async () => {
    if (!known || !form || submitting || draftFormRefusal(form, defaults)) return;
    setSubmitting(true);
    try {
      const d = await postJson(CIVITAI_API.publishModel(target.recordId, target.step), {
        ...form,
        trained_words: form.trained_words.split(',').map((s) => s.trim()).filter(Boolean),
        tags: form.tags.split(',').map((s) => s.trim()).filter(Boolean),
        filename: target.filename || undefined,
      });
      setJob({ id: d.job_id, kind: 'model', state: 'running', phase: 'starting', progress: 0 });
    } catch (e) {
      toast.error(e?.message || 'Could not start publishing.');
    } finally {
      setSubmitting(false);
    }
  };

  // The page a PICTURE is posted on: its own checkpoint's link, else the one
  // picked among the dataset's linked checkpoints.
  const postTarget = link || datasetLinks.find((l) => l.id === pickedLinkId) || null;

  const postImage = async () => {
    if (!isImage || !postTarget || submitting) return;
    setSubmitting(true);
    try {
      const d = await postJson(CIVITAI_API.publishImages, {
        image_ids: [target.imageId], link_id: postTarget.id,
        title: postTitle.trim() || undefined, publish: publishNow,
      });
      setJob({ id: d.job_id, kind: 'post', state: 'running', phase: 'starting', progress: 0 });
    } catch (e) {
      toast.error(e?.message || 'Could not start the post.');
    } finally {
      setSubmitting(false);
    }
  };

  const onModelDone = useCallback((j) => {
    const created = j?.result?.link;
    if (created) setLink(created);
  }, []);

  const noKey = status && !status.has_key;
  const busy = job?.state === 'running' || submitting || linking;
  const subject = isImage
    ? `${checkpointFileLabel(img?.checkpoint) || 'this checkpoint'}${img?.step != null ? ` · step ${img.step}` : ''}`
    : `${context?.node?.dataset_name || 'this run'}${target?.step != null ? ` · step ${target.step}` : ''}`;

  return (
    <div ref={dialogRef} data-probe-layer data-testid="civitai-publish-modal"
      role="dialog" aria-modal="true" aria-label="Publish to Civitai"
      className="fixed inset-0 z-[9998] flex items-center justify-center bg-black/80 p-3 sm:p-6"
      onClick={(e) => { e.stopPropagation(); if (e.target === e.currentTarget && !busy) close(e); }}
      onKeyDown={(e) => { e.stopPropagation(); if (e.key === 'Escape') close(e); }}>
      <div className="flex max-h-[92vh] w-full max-w-lg flex-col gap-3 overflow-y-auto rounded-xl border border-border bg-surface-overlay p-4"
        onClick={(e) => e.stopPropagation()}>
        <div className="flex items-start gap-2">
          <div className="min-w-0 flex-1">
            <h2 className="m-0 flex items-center gap-1.5 font-semibold text-content">
              <span aria-hidden>📤</span> {isImage ? 'Post this image on Civitai' : 'Civitai model page'}
            </h2>
            <p className="m-0 mt-0.5 truncate text-xs text-content-muted" title={subject}>{subject}</p>
            <p className="m-0 mt-0.5 text-[0.6875rem] text-content-subtle">
              {status == null ? 'Checking the API key…'
                : noKey ? 'No Civitai API key configured.'
                  : `Signed in as ${status.username || '(key accepted)'} · links open on ${status.link_host}`}
            </p>
          </div>
          <button type="button" ref={closeRef} onClick={close} aria-label="Close" title="Close (Esc)"
            className="shrink-0 rounded-full bg-white/10 px-2 py-0.5 text-content hover:bg-white/20">✕</button>
        </div>

        {noKey && (
          <div className="rounded-lg border border-amber-400/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-100">
            Publishing needs your Civitai API key (a free account has one). Paste it under{' '}
            <a href="#/settings/scraping" className="font-semibold underline">Settings ▸ Scraping &amp; sources</a>
            {' '}- the same key the scraper and the 🌐 prompt browser use.
          </div>
        )}

        {isImage && img?.url && (
          <div className="flex items-center gap-3">
            <img src={img.url} alt="" className="h-16 w-16 rounded-md border border-border object-cover" />
            <p className="m-0 text-xs text-content-muted">
              The picture leaves as a fresh PNG with no embedded metadata; its prompt, seed, sampler,
              CFG and LoRA weight travel as Civitai generation data instead.
            </p>
          </div>
        )}

        {/* ── The page ─────────────────────────────────────────────────── */}
        <section className="flex flex-col gap-2 rounded-lg border border-border bg-surface p-3">
          <h3 className="m-0 text-xs font-semibold uppercase tracking-wide text-content-muted">Model page</h3>
          {link === undefined && known ? (
            <p className="m-0 text-sm text-content-muted">Looking up the link…</p>
          ) : link ? (
            <div className="flex flex-wrap items-center gap-2" data-testid="civitai-linked">
              <span className="text-sm text-content">
                <span aria-hidden>✓</span> Linked to <strong>{civitaiLinkLine(link)}</strong>
                {link.published === false && <span className="ml-1 text-xs text-amber-200">(draft)</span>}
              </span>
              <a href={link.published === false ? link.wizard_url : link.model_url} target="_blank" rel="noreferrer"
                className="text-xs text-indigo-300 underline">Open ↗</a>
              <button type="button" onClick={unlink} disabled={busy}
                className="ml-auto text-xs text-content-subtle underline hover:text-rose-300">Unlink</button>
            </div>
          ) : !known ? (
            <p className="m-0 text-sm text-content-muted">
              This picture carries no checkpoint stamp — it was made with a run&apos;s <em>final</em> save
              (whose deployed name has no step), with a run that was since removed, or before the stamp
              existed — so pick one of this dataset&apos;s linked checkpoints below.
            </p>
          ) : (
            <>
              <div className="flex gap-1.5">
                <button type="button" onClick={() => setPane('mark')} className={TAB(pane === 'mark')}>
                  Mark an existing page
                </button>
                <button type="button" onClick={() => setPane('create')} className={TAB(pane === 'create')}
                  data-testid="civitai-pane-create">
                  Create the page from this checkpoint
                </button>
              </div>
              {pane === 'mark' ? (
                <div className="flex flex-col gap-1.5">
                  <p className="m-0 text-xs text-content-muted">
                    Paste the address of the LoRA&apos;s page on Civitai (with <code>?modelVersionId=</code> to aim
                    at one version; otherwise the newest is taken). The checkpoint is remembered as that version.
                  </p>
                  <div className="flex gap-2">
                    <input value={ref} onChange={(e) => setRef(e.target.value)} disabled={busy || noKey}
                      placeholder="https://civitai.com/models/12345/my-lora" className={`${FIELD} font-mono`}
                      data-testid="civitai-ref" />
                    <button type="button" onClick={markPage} disabled={busy || noKey || !ref.trim()} className={PRIMARY}>
                      {linking && <Spinner />} Link
                    </button>
                  </div>
                  {linkError && <p role="alert" className="m-0 text-xs text-rose-300">{linkError}</p>}
                </div>
              ) : (
                <div className="flex flex-col gap-2" data-testid="civitai-create-form">
                  {defaultsError ? (
                    <p role="alert" className="m-0 text-sm text-rose-300">{defaultsError}</p>
                  ) : !form ? (
                    <p className="m-0 text-sm text-content-muted">Preparing the page from this run…</p>
                  ) : (
                    <>
                      <label className="flex flex-col gap-0.5">
                        <span className={LABEL}>Model name</span>
                        <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })}
                          disabled={busy} className={FIELD} />
                      </label>
                      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                        <label className="flex flex-col gap-0.5">
                          <span className={LABEL}>Version name</span>
                          <input value={form.version_name}
                            onChange={(e) => setForm({ ...form, version_name: e.target.value })}
                            disabled={busy} className={FIELD} />
                        </label>
                        <label className="flex flex-col gap-0.5">
                          <span className={LABEL}>Base model (as Civitai names it)</span>
                          <select value={form.base_model}
                            onChange={(e) => setForm({ ...form, base_model: e.target.value })}
                            disabled={busy} className={FIELD} data-testid="civitai-base-model">
                            {!form.base_model && <option value="">— pick one —</option>}
                            {(defaults?.base_model_choices || [form.base_model]).map((b) => (
                              <option key={b} value={b}>{b}</option>
                            ))}
                          </select>
                        </label>
                      </div>
                      {defaults?.base_model_hint && (
                        <p className="m-0 -mt-1 text-[0.6875rem] text-amber-200">{defaults.base_model_hint}</p>
                      )}
                      <label className="flex flex-col gap-0.5">
                        <span className={LABEL}>Trigger words (comma-separated)</span>
                        <input value={form.trained_words}
                          onChange={(e) => setForm({ ...form, trained_words: e.target.value })}
                          disabled={busy} className={`${FIELD} font-mono`} />
                      </label>
                      <label className="flex flex-col gap-0.5">
                        <span className={LABEL}>Description (simple HTML allowed)</span>
                        <textarea rows={4} value={form.description}
                          onChange={(e) => setForm({ ...form, description: e.target.value })}
                          disabled={busy} className={`${FIELD} min-h-20 resize-y`} />
                      </label>
                      <label className="flex flex-col gap-0.5">
                        <span className={LABEL}>Tags (comma-separated)</span>
                        <input value={form.tags} onChange={(e) => setForm({ ...form, tags: e.target.value })}
                          disabled={busy} className={FIELD} />
                      </label>
                      <label className="flex flex-col gap-0.5">
                        <span className={LABEL}>File name on Civitai</span>
                        <input value={form.file_name}
                          onChange={(e) => setForm({ ...form, file_name: e.target.value })}
                          disabled={busy} className={`${FIELD} font-mono`} />
                        {defaults?.file && (
                          <span className="text-[0.6875rem] text-content-subtle">
                            {defaults.file.size_mb} MB · {defaults.file.fp}
                            {defaults.file.epoch ? ` · epoch ${defaults.file.epoch}` : ''} · from {defaults.file.source}
                          </span>
                        )}
                      </label>
                      <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm text-content">
                        <label className="flex items-center gap-2">
                          <input type="checkbox" checked={form.nsfw}
                            onChange={(e) => setForm({ ...form, nsfw: e.target.checked })} disabled={busy} />
                          Mature content (NSFW)
                        </label>
                        <label className="flex items-center gap-2">
                          <input type="checkbox" checked={form.license.allowCommercialUse}
                            onChange={(e) => setForm({ ...form, license: { ...form.license, allowCommercialUse: e.target.checked } })}
                            disabled={busy} />
                          Allow commercial use
                        </label>
                        <label className="flex items-center gap-2">
                          <input type="checkbox" checked={form.license.allowDerivatives}
                            onChange={(e) => setForm({ ...form, license: { ...form.license, allowDerivatives: e.target.checked } })}
                            disabled={busy} />
                          Allow merges &amp; derivatives
                        </label>
                        <label className="flex items-center gap-2">
                          <input type="checkbox" checked={!form.license.allowNoCredit}
                            onChange={(e) => setForm({ ...form, license: { ...form.license, allowNoCredit: !e.target.checked } })}
                            disabled={busy} />
                          Credit required
                        </label>
                      </div>
                      <label className="flex items-center gap-2 border-t border-border pt-2 text-sm text-content">
                        <input type="checkbox" checked={form.publish} data-testid="civitai-publish-now"
                          onChange={(e) => setForm({ ...form, publish: e.target.checked })} disabled={busy} />
                        <span>Publish the page right away
                          <span className="text-xs text-content-subtle"> — otherwise it is created as a draft you
                            finish on Civitai (cover images, final read)</span></span>
                      </label>
                      {draftFormRefusal(form, defaults) && (
                        <p role="alert" className="m-0 text-xs text-amber-200">{draftFormRefusal(form, defaults)}</p>
                      )}
                      <div className="flex justify-end">
                        <button type="button" onClick={createPage} className={PRIMARY}
                          data-testid="civitai-create-page"
                          disabled={busy || noKey || !!draftFormRefusal(form, defaults)}>
                          {submitting && <Spinner />}
                          {form.publish ? 'Upload & publish' : 'Upload as a draft'}
                        </button>
                      </div>
                    </>
                  )}
                </div>
              )}
            </>
          )}
          {isImage && !link && datasetLinks.length > 0 && (
            <label className="flex flex-col gap-0.5">
              <span className={LABEL}>Post on</span>
              <select value={pickedLinkId ?? ''} onChange={(e) => setPickedLinkId(Number(e.target.value) || null)}
                disabled={busy} className={FIELD} data-testid="civitai-link-picker">
                <option value="">— choose a linked checkpoint of this dataset —</option>
                {datasetLinks.map((l) => (
                  <option key={l.id} value={l.id}>{civitaiLinkLine(l)} (step {l.step})</option>
                ))}
              </select>
            </label>
          )}
          {isImage && !link && !known && datasetLinks.length === 0 && (
            <p className="m-0 text-xs text-amber-200">
              No checkpoint of this dataset is linked to Civitai yet. Open the checkpoint&apos;s pill on the
              ◉ Canvas or the run graph and use 📤 Civitai there first.
            </p>
          )}
        </section>

        {/* ── The post (image door) ────────────────────────────────────── */}
        {isImage && (
          <section className="flex flex-col gap-2 rounded-lg border border-border bg-surface p-3">
            <h3 className="m-0 text-xs font-semibold uppercase tracking-wide text-content-muted">Post</h3>
            <label className="flex flex-col gap-0.5">
              <span className={LABEL}>Title (optional)</span>
              <input value={postTitle} onChange={(e) => setPostTitle(e.target.value)} disabled={busy}
                placeholder="Leave empty for Civitai's default" className={FIELD} />
            </label>
            <label className="flex items-center gap-2 text-sm text-content">
              <input type="checkbox" checked={publishNow} onChange={(e) => setPublishNow(e.target.checked)}
                disabled={busy} data-testid="civitai-post-publish-now" />
              <span>Publish the post right away
                <span className="text-xs text-content-subtle"> — untick to leave it as a draft on Civitai</span></span>
            </label>
            <div className="flex items-center justify-end gap-2">
              <button type="button" onClick={close} disabled={job?.state === 'running'} className={SECONDARY}>
                {job ? 'Close' : 'Cancel'}
              </button>
              <button type="button" onClick={postImage} className={PRIMARY} data-testid="civitai-post-image"
                disabled={busy || noKey || !postTarget || job?.state === 'done'}
                title={postTarget ? `Post on ${civitaiLinkLine(postTarget)}` : 'Link a page first'}>
                {submitting && <Spinner />} {publishNow ? 'Post on Civitai' : 'Save as a draft post'}
              </button>
            </div>
          </section>
        )}

        <JobPanel job={job} onDone={job?.kind === 'model' ? onModelDone : undefined} />

        {!isImage && (
          <div className="flex justify-end">
            <button type="button" onClick={close} className={SECONDARY}>{job ? 'Close' : 'Done'}</button>
          </div>
        )}
      </div>
    </div>
  );
}
