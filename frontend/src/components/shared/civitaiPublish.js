/* 📤 Publish to Civitai — the decisions, JSX-free so `node --test` can read
   them: which save a modal is about, whether a picture may be posted, what the
   form starts from, what a finished job says.

   The modal is ONE component for two doors — the shared image viewer (post
   THIS picture) and the checkpoint popover (make/mark the page) — and both
   describe their subject through `civitaiTarget`, so "which save is this
   about" is decided once. A save is `(record_id, step, filename)`: a run that
   ends on a numbered save writes two files at its last step, and the step
   alone cannot tell them apart. */

export const CIVITAI_POLL_MS = 1500;

/** Where the link store is asked and written. Kept here so the modal, the
 *  contract test and any future host agree on ONE address per verb. */
/** `?filename=` names the save exactly (a pill); `?checkpoint=` is the deployed
 *  LoRA name a picture ran with, which the server resolves to the save. */
const saveQuery = (filename, checkpoint) => {
  const q = new URLSearchParams();
  if (filename) q.set('filename', filename);
  if (checkpoint) q.set('checkpoint', checkpoint);
  const s = q.toString();
  return s ? `?${s}` : '';
};

export const CIVITAI_API = {
  status: '/api/civitai/status',
  link: (recordId, step, filename, checkpoint) =>
    `/api/civitai/links/${recordId}/${step}${saveQuery(filename, checkpoint)}`,
  datasetLinks: (datasetId) => `/api/civitai/links?dataset_id=${datasetId}`,
  page: (ref) => `/api/civitai/page?ref=${encodeURIComponent(ref)}`,
  createLink: '/api/civitai/links',
  deleteLink: (id) => `/api/civitai/links/${id}/delete`,
  draftDefaults: (recordId, step, filename, checkpoint) =>
    `/api/civitai/checkpoint/${recordId}/${step}/draft-defaults${saveQuery(filename, checkpoint)}`,
  publishModel: (recordId, step) => `/api/civitai/checkpoint/${recordId}/${step}/publish-model`,
  publishImages: '/api/civitai/images/publish',
  job: (id) => `/api/civitai/jobs/${id}`,
};

/**
 * The save a modal context is about. The viewer's row carries the checkpoint
 * that generated it (`record_id`/`step`, stamped at generation) and the
 * deployed LoRA name it ran with (`checkpoint`) — no file name, the server
 * resolves the save from those; a popover context carries the run node and
 * the pill, file name included. Null ids are kept null — a picture with no
 * stamp is a real case (every picture made with a run's FINAL save, whose
 * deployed name carries no step) that the modal answers with a picker, never
 * by guessing.
 */
export function civitaiTarget(context) {
  if (!context) return null;
  if (context.kind === 'image') {
    const img = context.img || {};
    return {
      recordId: img.record_id ?? null,
      step: img.step ?? null,
      datasetId: img.dataset_id ?? null,
      filename: null,
      checkpoint: img.checkpoint ?? null,
      imageId: img.id ?? null,
    };
  }
  const node = context.node || {};
  const pill = context.pill || {};
  return {
    recordId: node.record_id ?? null,
    step: pill.step ?? null,
    datasetId: node.dataset_id ?? null,
    filename: pill.filename ?? null,
    checkpoint: null,
    imageId: null,
  };
}

/** True when the target names one checkpoint exactly. */
export const civitaiTargetKnown = (t) =>
  !!t && Number.isInteger(Number(t.recordId)) && t.recordId !== null
  && Number.isInteger(Number(t.step)) && t.step !== null;

/**
 * Why the viewer's 📤 cannot be offered for this picture, or null when it can.
 * The one hard refusal is the same as ✨ and 📷: a picture with no library
 * row has no id to send. A row WITHOUT a stamped checkpoint is still allowed —
 * the modal offers the dataset's linked pages instead.
 */
export function civitaiVerbRefusal(img) {
  if (!img || !Number.isInteger(Number(img.id))) {
    return 'This picture has no library entry to post.';
  }
  return null;
}

/** The versions of a looked-up page as select options — name first, the id
 *  beside it so an address's `?modelVersionId=` can be recognised by eye. */
export function pageVersionOptions(page) {
  return (page?.versions || []).map((v) => ({
    id: v.id,
    label: `${v.name || 'unnamed'} (#${v.id})${v.base_model ? ` · ${v.base_model}` : ''}`,
  }));
}

/** The version to preselect: the one the pasted address names when the page
 *  has it, else the page's newest (first). Null on a page with no version. */
export function preselectVersion(page, urlVersionId) {
  const versions = page?.versions || [];
  if (!versions.length) return null;
  const wanted = Number(urlVersionId);
  const hit = versions.find((v) => v.id === wanted);
  return (hit || versions[0]).id;
}

/** One line naming a link, for the modal header and the popover title. */
export function civitaiLinkLine(link) {
  if (!link) return '';
  const version = link.version_name ? ` · ${link.version_name}` : '';
  return `${link.model_name || `model ${link.model_id}`}${version}`;
}

/** The "create a page" form, started from what the server derived. Arrays
 *  become comma-separated text (the fields are typed in), toggles get their
 *  safe default: a page is a DRAFT unless asked otherwise. An EMPTY base model
 *  stays empty — it means the server could not honestly name the lineage
 *  (a custom base), and the form refuses until the user does. */
export function draftFormFrom(defaults) {
  const d = defaults || {};
  return {
    name: d.name || '',
    version_name: d.version_name || 'v1.0',
    base_model: d.base_model || '',
    trained_words: (d.trained_words || []).join(', '),
    description: d.description || '',
    tags: (d.tags || []).join(', '),
    nsfw: !!d.nsfw,
    publish: false,
    file_name: d.file?.name || '',
    license: {
      allowNoCredit: true, allowCommercialUse: true,
      allowDerivatives: true, allowDifferentLicense: true,
    },
  };
}

/** What the form can be submitted with, or the sentence that stops it. */
export function draftFormRefusal(form, defaults) {
  if (!form) return 'Nothing to publish.';
  if (!String(form.name || '').trim()) return 'Give the model a name.';
  if (!String(form.base_model || '').trim()) return 'Pick the base model as Civitai names it.';
  if (defaults?.file_error) return defaults.file_error;
  if (!defaults?.file) return 'The checkpoint file could not be found on this machine.';
  return null;
}

/** The human line for a job phase — the modal draws it over the progress bar. */
export function jobPhaseLabel(job) {
  if (!job) return '';
  const pct = Math.round(Math.max(0, Math.min(1, Number(job.progress) || 0)) * 100);
  switch (job.phase) {
    case 'starting': return 'Starting…';
    case 'creating': return job.kind === 'post' ? 'Creating the post…' : 'Creating the model page…';
    case 'uploading': return job.kind === 'post'
      ? `Uploading images… ${pct}%`
      : `Uploading the checkpoint… ${pct}%`;
    case 'registering': return 'Registering the file…';
    case 'done': return 'Done.';
    default: return job.phase ? String(job.phase) : '';
  }
}

/** What a finished job says, and the address it opens. */
export function jobOutcome(job) {
  if (!job) return null;
  if (job.state === 'error') {
    return { ok: false, text: job.error || 'Publishing failed.', code: job.error_code || 'unknown' };
  }
  if (job.state !== 'done') return null;
  const r = job.result || {};
  if (job.kind === 'post') {
    return {
      ok: true,
      text: r.published
        ? `Posted ${r.count} image${r.count === 1 ? '' : 's'} on Civitai.`
        : `Draft post created with ${r.count} image${r.count === 1 ? '' : 's'} - finish it on Civitai.`,
      url: r.url,
    };
  }
  return {
    ok: true,
    text: r.published
      ? 'Model page published on Civitai.'
      : 'Draft model page created - review it on Civitai and press Publish there.',
    url: r.url,
  };
}
