import { useCallback, useEffect, useRef, useState } from 'react';
import { apiFetch } from '../../api/fetchClient';
import { postJson } from '../../hooks/useDataset';
import { fmtGB } from './loraMerge.js';

/** Turn a full-precision model into the fp8 file ComfyUI loads — in one click.
 *
 * WHAT THIS BLOCK USED TO BE, AND WHY IT COULD NOT HELP
 * -----------------------------------------------------
 * It shipped as a text field: paste an absolute path to a `.safetensors` on this
 * machine. Three things were wrong with that, and all three are about the SAME
 * model the user is looking at while reading it:
 *
 *  1. it asked for a path the app already knows. The card above this one names
 *     the model the run delivered;
 *  2. "on this machine" was exactly the hole. A dense run's ~26 GB master lives
 *     ONLY in a private Hugging Face repo — the lane never downloads it — so for
 *     the one full model most people own there was no path to paste, and this
 *     block could do nothing at all;
 *  3. it wrote the result next to the source and left the user to move it into
 *     ComfyUI, which is the step the whole feature exists to remove.
 *
 * So the field is no longer the way in: it is the exception. The way in is the
 * target the app already holds — and when that target is not local, fetching it
 * first is part of the same gesture.
 *
 * TWO CLICKS, NOT ONE, AND ON PURPOSE. The first says what would happen: which
 * checkpoint (a dense repo holds the final save AND several ~26 GB step
 * snapshots with nearly the same name), which folder, and what it costs in disk.
 * Only then does the conversion start. A 26 GB download that silently chose a
 * file and a destination is the surprise this was asked to remove.
 *
 * TWO HOSTS, ONE IMPLEMENTATION. The recipe card and Settings ▸ Storage render
 * this same file; `framed` is the only difference and it is chrome. The refusals,
 * the disk guard, the overwrite guard and the read-back verification exist once,
 * so the two doors cannot drift.
 */

const pct = (done, total) => (
  total > 0 ? Math.min(100, Math.max(0, Math.round((done / total) * 100))) : 0
);

export const RUNNING_STATES = ['downloading', 'quantizing'];

/** Which checkpoint, which folder, what it costs — before anything moves. */
export function Fp8DeliverPlan({
  plan, keepMaster = true, onKeepMaster = null, onStart = null, busy = false,
  disabled = false, elsewhere = '', onElsewhere = null, onReplan = null,
}) {
  if (!plan?.ok) return null;
  const choice = plan.choice || null;
  return (
    <div className="mt-1.5 rounded-md border border-primary/40 bg-primary/10 px-2.5 py-1.5 text-[0.6875rem] leading-relaxed">
      <p className="m-0">
        Takes <span className="font-mono break-all">{plan.weight_basename}</span>
        {' '}({fmtGB(plan.source_bytes)})
        {choice && choice.total > 1 ? (
          <span className="opacity-85">
            {' '}— the {choice.is_final ? 'final save' : `step ${choice.step} checkpoint`},
            {' '}chosen over {choice.total - 1} other checkpoint{choice.total > 2 ? 's' : ''}
            {' '}in this repository
          </span>
        ) : null}
        {'. '}
        Writes <span className="font-mono break-all">{plan.destination_name}</span>
        {' '}(~{fmtGB(plan.estimated_bytes)}) into
        {' '}<span className="font-mono break-all">{plan.destination_dir}</span>.
      </p>
      <p className={`m-0 mt-1 ${plan.destination_dir_kind === 'comfyui' ? 'opacity-85' : 'text-amber-200'}`}>
        {plan.destination_dir_kind === 'comfyui'
          ? `✓ ${plan.destination_dir_note} — ComfyUI lists it after a refresh.`
          : `⚠ ${plan.destination_dir_note}.`}
      </p>
      {plan.download_bytes > 0 && (
        <p className="m-0 mt-1 opacity-85">
          {fmtGB(plan.download_bytes)} still has to come down from Hugging Face — this is the
          long part, and it can be stopped and resumed.
        </p>
      )}

      {/* Keeping the master is the default because it is the ONLY file that can
          be trained again, merged or re-quantized. Its cost is stated, not
          assumed away — and it is never offered for a file the user already had. */}
      {plan.source_kind === 'huggingface' && (
        <fieldset className="m-0 mt-1.5 border-0 p-0">
          <legend className="p-0 opacity-85">
            Afterwards, the {fmtGB(plan.source_bytes)} full-precision master:
          </legend>
          <label className="mt-0.5 flex items-start gap-1.5">
            <input type="radio" name="fp8-keep-master" checked={keepMaster}
              onChange={() => onKeepMaster && onKeepMaster(true)} className="mt-0.5" />
            <span>Keep it — a local backup you can train from again ({fmtGB(plan.source_bytes)} of disk)</span>
          </label>
          <label className="mt-0.5 flex items-start gap-1.5">
            <input type="radio" name="fp8-keep-master" checked={!keepMaster}
              onChange={() => onKeepMaster && onKeepMaster(false)} className="mt-0.5" />
            <span>
              Delete it — frees {fmtGB(plan.source_bytes)}. It stays on Hugging Face, but
              getting it back is another {fmtGB(plan.source_bytes)} download.
            </span>
          </label>
        </fieldset>
      )}

      {plan.enough_space === false ? (
        <>
          <p className="m-0 mt-1 text-rose-200" role="alert">✗ {plan.space_error}</p>
          {/* A full drive is not the end of the operation, only of this
              destination: the same file often fits one volume over. Measured on
              a machine whose ComfyUI folder is a junction onto a 99 %-full
              drive while the system disk had 113 GB free. */}
          <div className="mt-1 flex flex-col gap-1 sm:flex-row sm:items-center">
            <input type="text" value={elsewhere}
              onChange={(event) => onElsewhere && onElsewhere(event.target.value)}
              placeholder="Full path of another folder"
              aria-label="Folder to write the fp8 file into instead"
              className="w-full sm:flex-1 rounded border border-sky-300/40 bg-app/70 px-2 py-1 text-content font-mono" />
            <button type="button" onClick={onReplan} disabled={busy || !elsewhere.trim()}
              className="shrink-0 self-start rounded-md border border-white/30 bg-black/20 px-2.5 py-1 font-semibold hover:bg-black/30 disabled:opacity-40">
              Check that folder
            </button>
          </div>
        </>
      ) : (
        typeof plan.free_bytes === 'number' && (
          <p className="m-0 mt-1 opacity-75">
            {fmtGB(plan.free_bytes)} free there · about {fmtGB(plan.required_bytes)} needed.
          </p>
        )
      )}

      <button type="button" onClick={onStart}
        disabled={busy || disabled || plan.enough_space === false}
        className="mt-1 rounded-md border border-primary/50 bg-primary/20 px-2.5 py-1 font-semibold text-white hover:bg-primary/30 disabled:opacity-40">
        {busy ? 'Starting…' : 'Quantize to fp8'}
      </button>
    </div>
  );
}

/** The long half, alive: which phase, how far, where it lands, and a way out. */
export function Fp8DeliverProgress({ state, onCancel = null }) {
  if (!state || !RUNNING_STATES.includes(state.status)) return null;
  const downloading = state.status === 'downloading';
  const width = downloading
    ? pct(state.downloaded_bytes, state.download_total_bytes)
    : pct(state.done, state.total);
  return (
    <div className="mt-1.5 rounded-md border border-primary/40 bg-primary/10 px-2.5 py-1.5 text-[0.6875rem] leading-relaxed"
      role="status">
      <p className="m-0">
        {downloading
          ? `⬇ Downloading ${state.weight_name || 'the master'} — ${fmtGB(state.downloaded_bytes)} of ${fmtGB(state.download_total_bytes)}`
          : `✨ Quantizing on the CPU${state.total ? ` — ${state.done}/${state.total} tensors` : '…'}`}
      </p>
      <div className="mt-1 h-1.5 w-full overflow-hidden rounded bg-black/30"
        role="progressbar" aria-valuenow={width} aria-valuemin={0} aria-valuemax={100}>
        <div className="h-full bg-primary/70" style={{ width: `${width}%` }} />
      </div>
      <p className="m-0 mt-1 opacity-80">
        Lands in <span className="font-mono break-all">{state.destination_dir}</span> as
        {' '}<span className="font-mono break-all">{state.destination_name}</span>.
      </p>
      <button type="button" onClick={onCancel}
        className="mt-1 rounded-md border border-white/30 bg-black/20 px-2.5 py-1 font-semibold hover:bg-black/30">
        Stop
      </button>
    </div>
  );
}

/** What happened, in the words that matter: the file, the folder, the master. */
export function Fp8DeliverOutcome({ state }) {
  if (!state) return null;
  const result = state.result || null;
  if (state.status === 'done') {
    return (
      <p className="m-0 mt-1.5 text-emerald-200 text-[0.6875rem] leading-relaxed" role="status">
        ✓ <span className="font-mono break-all">{state.destination_name}</span>
        {' '}({fmtGB(result?.bytes_after)}) is in
        {' '}<span className="font-mono break-all">{state.destination_dir}</span> and was
        re-opened successfully{result?.scaled_tensors
          ? ` — ${result.scaled_tensors} scaled tensors verified` : ''}.
        {' '}{result?.master_removed
          ? 'The full-precision master was deleted, as you asked.'
          : 'The full-precision master was kept next to it.'}
      </p>
    );
  }
  if (state.status === 'cancelled') {
    return (
      <p className="m-0 mt-1.5 text-amber-200 text-[0.6875rem] leading-relaxed" role="status">
        ■ {state.error}
      </p>
    );
  }
  if (state.status === 'error') {
    return (
      <p className="m-0 mt-1.5 text-rose-200 text-[0.6875rem] leading-relaxed" role="alert">
        ✗ {state.error || 'Quantization failed.'} Nothing was overwritten.
      </p>
    );
  }
  return null;
}

export default function Fp8QuantizeTool({
  target = null, suggestedPath = '', disabled = false, framed = true,
  manualPath = true,
}) {
  const [path, setPath] = useState(suggestedPath);
  const [plan, setPlan] = useState(null);
  const [busy, setBusy] = useState(false);
  const [keepMaster, setKeepMaster] = useState(true);
  const [elsewhere, setElsewhere] = useState('');
  const [state, setState] = useState(null);
  const pollRef = useRef(null);
  const lastAsk = useRef(null);

  useEffect(() => { setPath(suggestedPath); }, [suggestedPath]);

  const stop = () => { clearInterval(pollRef.current); pollRef.current = null; };

  const poll = useCallback(() => {
    // apiFetch RESOLVES THE PARSED BODY, not a Response. `.then((r) => r.json())`
    // on it throws a TypeError that the `.catch()` swallows, so this poll used to
    // update NOTHING: the panel stayed on "Quantizing…" for ever while the
    // conversion finished perfectly. Found by watching the real thing land.
    apiFetch('/api/tools/fp8-deliver/status')
      .then((s) => {
        setState(s);
        if (!RUNNING_STATES.includes(s?.status)) stop();
      })
      .catch(() => {});
  }, []);

  // A job outlives any visit to this tab — a 26 GB download takes longer than the
  // page does. Adopt one that is already running rather than offering to start a
  // second: the server refuses that anyway, and a refusal is not the answer to
  // "what is my model doing".
  useEffect(() => {
    let alive = true;
    apiFetch('/api/tools/fp8-deliver/status')
      .then((s) => {
        if (!alive || !RUNNING_STATES.includes(s?.status)) return;
        setState(s);
        stop();
        pollRef.current = setInterval(poll, 2000);
      })
      .catch(() => {});
    return () => { alive = false; stop(); };
  }, [poll]);

  const request = (extra = {}) => ({
    repo_id: null, filename: null, path: null,
    family: target?.family || null, keep_master: keepMaster,
    destination_dir: elsewhere.trim() || null, ...extra,
  });

  const askPlan = async (extra) => {
    lastAsk.current = extra;
    setBusy(true);
    setPlan(await postJson('/api/tools/fp8-deliver/plan', request(extra)));
    setBusy(false);
  };

  // A target now comes in two shapes, because a full model now has two homes.
  // `path` (the master already harvested to this computer — nothing to download)
  // wins over the repository, and the backend's `plan` branches on exactly that
  // key, so one button covers both without the panel knowing which chain runs.
  const askForTarget = () => askPlan(target?.path
    ? { path: target.path }
    : { repo_id: target?.repoId || null, filename: target?.filename || null });
  const askForPath = () => askPlan({ path: path.trim() });
  const replan = () => askPlan(lastAsk.current || {});

  const start = async () => {
    setBusy(true);
    const res = await postJson('/api/tools/fp8-deliver', request(lastAsk.current || {}));
    setBusy(false);
    if (!res?.ok) {
      setState({ status: 'error', error: res?.error || 'Quantization could not start.' });
      return;
    }
    setPlan(null);
    setState(res.status || { status: 'downloading' });
    stop();
    pollRef.current = setInterval(poll, 2000);
  };

  const cancel = async () => {
    await postJson('/api/tools/fp8-deliver/cancel', {});
    poll();
  };

  const running = RUNNING_STATES.includes(state?.status);
  const controlClass = 'shrink-0 self-start px-2.5 py-1 rounded-lg bg-primary/20 border border-primary/40 text-white text-[0.75rem] font-semibold disabled:opacity-40';

  return (
    <div className={framed
      ? 'rounded-lg border border-sky-300/30 bg-sky-400/10 px-3 py-2 text-sky-50'
      : 'text-sky-50'}>
      {framed && (
        <>
          <span className="font-semibold">Quantize a model to fp8</span>
          <p className="m-0 mt-1 text-sky-200/75 text-[0.6875rem] leading-relaxed">
            Turns a full-precision checkpoint into the ~10 GB fp8 file ComfyUI loads with the
            standard Load Diffusion Model node, and puts it in ComfyUI’s own models folder. The
            source is never modified and nothing is ever overwritten. This is not the same thing
            as the “quantize” training option, which only shrinks the model in memory while it
            trains and writes no file.
          </p>
        </>
      )}

      {/* The model the app already knows about — no path to find, and it works
          even though that master exists only in a private Hugging Face repo. */}
      {target && (
        <div className={framed ? 'mt-2' : ''}>
          <p className="m-0 text-[0.6875rem] leading-relaxed">
            <span className="font-semibold">{target.label || 'The model this run delivered'}</span>
            {target.name ? <> — <span className="font-mono break-all">{target.name}</span></> : null}
            {target.sizeBytes ? <> · {fmtGB(target.sizeBytes)}</> : null}
            <span className="block opacity-80">
              {target.path
                ? 'Already on this computer — nothing to download; the conversion starts straight away.'
                : 'In your private Hugging Face repository, not on this machine — it is fetched first, '
                  + 'with progress, and the transfer can be stopped and resumed.'}
            </span>
          </p>
          <button type="button" onClick={askForTarget}
            disabled={disabled || busy || running}
            title="Fetch this model, convert it to fp8 and put it where ComfyUI loads it — nothing to type"
            className={`mt-1.5 ${controlClass}`}>
            ✨ Quantize to fp8
          </button>
        </div>
      )}

      {/* The exception, not the way in: a path nothing in the app points at.
          Hidden where the app already holds every candidate it could name (the
          full-model lane lists them), because an empty path field next to a
          named file reads as "the real way in is to type something". */}
      {manualPath && (
      <div className={framed || target ? 'mt-2' : ''}>
        {target && (
          <p className="m-0 mb-1 text-sky-200/75 text-[0.625rem] uppercase tracking-wide">
            Or another file, already on this machine
          </p>
        )}
        {/* Column on a phone, row from sm up: a full Windows path and a button
            sharing 400 px leaves the field showing about 28 characters, which is
            not enough to see which file you pointed at. */}
        <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:items-center">
          <input type="text" value={path} onChange={(event) => setPath(event.target.value)}
            disabled={running || disabled}
            placeholder="Full path to a .safetensors model"
            aria-label="Path of the model file to quantize to fp8"
            className="w-full sm:flex-1 sm:min-w-[12rem] rounded border border-sky-300/40 bg-app/70 px-2 py-1 text-content text-[0.75rem] font-mono disabled:opacity-50" />
          <button type="button" onClick={askForPath}
            disabled={disabled || busy || running || !path.trim()}
            className={controlClass}>
            Quantize to fp8
          </button>
        </div>
      </div>
      )}

      {plan && !plan.ok && !running && (
        <p className="m-0 mt-1 text-amber-200 text-[0.6875rem]" role="alert">⚠ {plan.error}</p>
      )}
      {!running && (
        <Fp8DeliverPlan plan={plan} keepMaster={keepMaster} busy={busy} disabled={disabled}
          onKeepMaster={setKeepMaster} onStart={start}
          elsewhere={elsewhere} onElsewhere={setElsewhere} onReplan={replan} />
      )}
      <Fp8DeliverProgress state={state} onCancel={cancel} />
      <Fp8DeliverOutcome state={state} />
    </div>
  );
}
