/* The grammar of a FULL MODEL in the Checkpoints panel — pure, no React, so the
   rules are unit-testable with node:test (which cannot parse JSX).

   WHY A FULL MODEL NEEDS ITS OWN GRAMMAR
   --------------------------------------
   The rest of that panel speaks about LoRA adapters: list them, deploy them into
   ComfyUI's `loras/<family>`, undeploy them. A dense run does not produce one of
   those. It produces up to two files that are NOT interchangeable, and every
   sentence here exists to keep them apart:

     • the MASTER (bf16, ~26 GB) — the only file that can be trained again or
       resumed from. It is never sent to ComfyUI. Not "not yet": never. Twenty-six
       gigabytes of a model folder to do a job the twin does in half the space is
       not a convenience, and the app must not offer it as one.
     • the FP8 TWIN (~13 GB) — the inference format, and the only one worth
       putting where ComfyUI looks.

   A panel that listed "2 files" and offered "deploy" on both would be worse than
   the banner it replaces. */

export const fmtBytes = (bytes) => {
  const n = Number(bytes);
  if (!Number.isFinite(n) || n <= 0) return '';
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`;
  // Below a kilobyte, say bytes. Rounding to "0 kB" would describe a file that
  // exists as one that does not — and a truncated weight file is exactly the
  // case where a size is worth reading.
  if (n >= 1e3) return `${Math.round(n / 1e3)} kB`;
  return `${n} B`;
};

const FAMILY_TITLE = {
  krea: 'Krea 2', zimage: 'Z-Image', sdxl: 'SDXL',
  flux: 'FLUX.1', flux2klein: 'FLUX.2 Klein', anima: 'Anima',
};

/** "Krea 2 · Raw — run #146". The variant is part of the identity: a Raw model
 *  and a Turbo one want completely different sampler settings. */
export function denseModelTitle(entry) {
  const family = FAMILY_TITLE[entry?.train_type] || entry?.train_type || 'Full model';
  const variant = String(entry?.variant || '').trim();
  const head = variant ? `${family} · ${variant}` : family;
  return entry?.run_id != null ? `${head} — run #${entry.run_id}` : head;
}

/* WHAT A LIVE HUB CHECK CAN SAY — and the fourth thing, which is not a state of
   the repository but of our knowledge. `unchecked` is the DEFAULT: the panel
   paints before the network answers, and everything below has to read honestly
   in that first frame. See backend `hub_presence` for why `unknown` (no token,
   offline, 5xx, refused token) may never collapse into `gone`. */
const HUB_PRESENT = 'present';
const HUB_GONE = 'gone';
const HUB_UNKNOWN = 'unknown';
const HUB_UNCHECKED = 'unchecked';

const hubState = (presence) => {
  const state = String(presence?.state || '').trim().toLowerCase();
  return [HUB_PRESENT, HUB_GONE, HUB_UNKNOWN].includes(state)
    ? state : HUB_UNCHECKED;
};

/** `2026-08-04` out of an ISO stamp, or ''. Deliberately not localized: this
 *  date is evidence in a sentence about what was true WHEN, and a test that has
 *  to guess the runner's locale proves nothing. */
const day = (iso) => (/^\d{4}-\d{2}-\d{2}/.test(String(iso || ''))
  ? String(iso).slice(0, 10) : '');

/** What the RECORD says — always about the past, and phrased as such.
 *
 * `entry.hub.status` is `artifact_status`: written once at delivery and never
 * revisited. Rendering it in the present tense ("the model is there") is the
 * whole bug this vocabulary exists to end. */
function deliveredPast(entry) {
  const status = String(entry?.hub?.status || '').trim().toLowerCase();
  const when = day(entry?.hub?.checked_at);
  // A run still working has not failed to deliver — it has not got there yet.
  // "Never confirmed" would read as a verdict on a run that is doing fine.
  if (entry?.active && status !== 'available') {
    return 'This run is still working — nothing has reached this repository yet.';
  }
  if (status === 'available') {
    return when
      ? `Delivered and verified on ${when} — not re-checked since.`
      : 'Delivered and verified at the end of the run — not re-checked since.';
  }
  if (status === 'missing') {
    return 'The last check found no model in this repository.';
  }
  if (status === 'verification_pending') {
    return 'The delivery to this repository was never confirmed.';
  }
  return 'This run’s delivery to this repository was never confirmed.';
}

/** Where this model IS, as one chip. Mirrors the canvas vocabulary on purpose:
 *  a run that reads "on Hugging Face" there must not read "missing" here.
 *
 *  A Hugging Face repository only earns the present tense once something has
 *  ASKED it. Without a live answer the chip states what the run did — it
 *  delivered — which stays true whatever happened to the repo overnight. */
export function denseWhereChip(entry, presence = null) {
  if (entry?.master || entry?.fp8) {
    return { tone: 'ok', label: 'On this computer',
      title: 'The model files are in this app’s checkpoint folder' };
  }
  if (entry?.hub?.repo_id) {
    const state = hubState(presence);
    if (state === HUB_GONE) {
      return { tone: 'error', label: 'No copy left',
        title: 'Hugging Face no longer returns this repository, and nothing from '
          + 'this run is on this computer' };
    }
    if (state === HUB_PRESENT) {
      return { tone: 'info', label: 'On Hugging Face',
        title: 'Checked just now: the model is in this run’s private Hugging Face '
          + 'repository, not on this computer' };
    }
    return { tone: 'muted', label: 'Delivered to Hugging Face',
      title: 'This run delivered to a private Hugging Face repository and left '
        + 'nothing on this computer. Whether the repository is still there has '
        + 'not been checked.' };
  }
  return { tone: 'muted', label: 'Not found',
    title: 'No copy of this model could be found on this computer or on Hugging Face' };
}

/** The Hugging Face line of a card: the repository, what we KNOW about it, and
 *  — when it is gone — what is left to do about it.
 *
 *  Returns null for a run that never had a repository. Otherwise the caller
 *  renders `text` verbatim: the whole point is that the sentence is chosen from
 *  the state instead of being printed next to it. The line this replaced hard-
 *  coded "the model is there, not on this computer" whenever nothing was on
 *  disk, so a verified-missing repository read "· missing — the model is
 *  there". Both halves came from the same payload; only one of them looked. */
export function denseHubLine(entry, presence = null) {
  const repoId = entry?.hub?.repo_id;
  if (!repoId) return null;
  const state = hubState(presence);
  const hasMaster = !!entry?.master;
  const hasLocal = hasMaster || !!entry?.fp8;
  const base = { repoId, url: entry?.hub?.url || null, state };

  if (state === HUB_PRESENT) {
    return { ...base, tone: 'ok', stateLabel: 'checked just now',
      text: hasLocal
        ? 'The repository still holds a copy of this model.'
        : 'The repository still holds this model. Nothing from this run is on '
          + 'this computer — quantizing downloads it from there first.' };
  }
  if (state === HUB_GONE) {
    // The useful gesture, and it depends on what survived. Only the master can
    // be trained again, merged or re-quantized: an fp8 twin alone is a model
    // you can still generate with and never continue from.
    const left = hasMaster
      ? ' The full-precision master is still on this computer, so nothing is lost.'
      : (hasLocal
        ? ' The fp8 twin is still on this computer, so you can still generate with '
          + 'it — but the full-precision master is not, so this run can no longer '
          + 'be trained again, merged or re-quantized.'
        : ' Nothing from this run is on this computer either, so it has no '
          + 'recoverable model left. Its dataset and settings are still here: '
          + 'training it again is the only way back.');
    return { ...base, tone: 'error', stateLabel: 'not found',
      text: `Hugging Face no longer returns this repository — it was deleted or `
        + `renamed, or the token can no longer see it.${left}` };
  }
  if (state === HUB_UNKNOWN) {
    const why = String(presence?.detail || '').trim()
      || 'Hugging Face could not be reached, so the repository was not checked.';
    // Said in this order on purpose: the failure first, so nobody reads the
    // record underneath it as a fresh answer.
    return { ...base, tone: 'muted', stateLabel: 'could not check',
      text: `${why} ${deliveredPast(entry)}` };
  }
  return { ...base, tone: 'muted', stateLabel: 'not re-checked',
    text: hasLocal
      ? deliveredPast(entry)
      : `${deliveredPast(entry)} Nothing from this run is on this computer.` };
}

/** The two files, in the order that teaches the difference: what ComfyUI loads
 *  first, what you keep in order to train again second. Returns [] for a run
 *  that delivered nothing locally — the Hugging Face line then carries the card. */
export function denseFileRows(entry) {
  const rows = [];
  if (entry?.fp8) {
    // Three states, not two. "Delivered but not listed by ComfyUI" is the real
    // state of an install with no ComfyUI configured: the app put the file in
    // its OWN models folder and said so, and calling that "not there yet" would
    // offer a Send that has already happened.
    const { in_comfyui: loaded, delivered } = entry.fp8;
    rows.push({
      kind: 'fp8',
      label: 'fp8 twin',
      filename: entry.fp8.filename,
      bytes: entry.fp8.size_bytes,
      role: 'The inference format — this is the file ComfyUI loads.',
      state: loaded ? 'in-comfyui' : (delivered ? 'delivered' : 'not-in-comfyui'),
      stateLabel: loaded
        ? '✓ In ComfyUI'
        : (delivered ? 'Delivered — move it into ComfyUI' : 'Not in ComfyUI yet'),
    });
  }
  if (entry?.master) {
    const m = entry.master;
    const picked = m.is_final ? 'the final save' : `the step ${m.step} checkpoint`;
    rows.push({
      kind: 'master',
      label: 'Full-precision master',
      filename: m.filename,
      bytes: m.size_bytes,
      role: 'The only file you can train again or resume from. It is never sent '
        + 'to ComfyUI: at this size it would fill a model folder to do a job the '
        + 'fp8 twin does better.',
      // A run that saved every N steps leaves several ~26 GB files with nearly
      // the same name. Naming the pick AND what it beat is what stops this card
      // from silently disagreeing with the button underneath it.
      choice: m.total_candidates > 1
        ? `${picked}, chosen over ${m.total_candidates - 1} other `
          + `checkpoint${m.total_candidates > 2 ? 's' : ''} this run left here`
        : '',
      state: 'keep',
      stateLabel: 'Keep to re-train',
    });
  }
  return rows;
}

/** What the buttons may do, and — when they may not — why, in a sentence.
 *  A disabled button carrying its reason beats a toast after the click. */
export function denseActions(entry, presence = null) {
  const fp8 = entry?.fp8 || null;
  const out = { quantize: null, send: null, activeNote: '' };
  if (entry?.active) {
    out.activeNote = 'This run is still working — its files can be used once it finishes.';
    return out;
  }
  if (entry?.can_quantize) {
    out.quantize = { label: '✨ Quantize to fp8', enabled: true, reason: '' };
  } else if (!fp8 && !entry?.master && entry?.hub?.repo_id) {
    // The master is only on the Hub: the quantizer downloads it first, which is
    // exactly what the shared fp8 block is for — and exactly why a repository
    // we have MEASURED to be gone must not be offered. This button's own promise
    // ("Quantizing fetches it first") is the one it cannot keep there, so it is
    // disabled with the reason rather than left to fail on the click.
    const gone = hubState(presence) === HUB_GONE;
    out.quantize = { label: '✨ Quantize to fp8', enabled: !gone,
      reason: gone
        ? 'There is nothing to quantize: this model is only on Hugging Face, and '
          + 'the repository it would be downloaded from is gone.'
        : '' };
  }
  if (fp8 && !fp8.delivered) {
    out.send = { label: '→ Send to ComfyUI', enabled: Boolean(entry?.can_send_to_comfyui),
      reason: '' };
  }
  return out;
}

/** The sampler settings this model wants, as one line. A dense Krea 2 artifact
 *  is RAW (undistilled): the family's Turbo defaults render mush on it, and that
 *  is the single most expensive thing to discover by trial. */
export function denseGuidanceLine(hint) {
  if (!hint) return '';
  const cfg = hint.guidance_scale;
  const steps = hint.steps;
  if (cfg == null && steps == null) return '';
  const parts = [];
  if (cfg != null) parts.push(`CFG ${cfg}`);
  if (steps != null) parts.push(`${steps} steps`);
  return parts.join(' · ');
}

/* The Test Studio needs at least one deployed LoRA of this dataset to open at
   all — it is a LoRA comparison grid, and its entry point is the LoRA picker.
   A dataset trained ONLY as a full model therefore cannot reach it, and the
   honest place to say so is next to the button that would otherwise look broken.
   The workaround is real and already shipped: pick any deployed LoRA of this
   dataset and set its strength to 0, which builds the graph with the bare UNET. */
export const STUDIO_NEEDS_A_LORA =
  'The Test Studio opens from a LoRA of this dataset. To generate with this base '
  + 'alone, pick any deployed LoRA there and set its strength to 0 — no LoRA node '
  + 'is added at 0, so you get the bare model.';

/** Can we offer "Test in Studio" for this model at all? Only once ComfyUI can
 *  actually load it: a base the picker cannot list is a button that leads to a
 *  screen where the model is absent. */
export function denseStudioTarget(entry) {
  const name = entry?.fp8?.in_comfyui ? entry.fp8.comfyui_name : null;
  if (!name) return null;
  return { base: name, family: entry?.train_type || 'krea',
    datasetId: entry?.dataset_id ?? null };
}
