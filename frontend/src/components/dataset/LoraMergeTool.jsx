import { useCallback, useEffect, useRef, useState } from 'react';
import { apiFetch } from '../../api/fetchClient';
import { postJson } from '../../hooks/useDataset';
import {
  HONESTY_NOTE, MERGE_RUNNING_STATES, PRECISION_NOTE, TURBO_NOTE, WEIGHT_MAX,
  WEIGHT_MIN, canAskPlan, carriedOverNote, clearMergeDraft, emptyMergeDraft,
  fmtDuration, fmtGB, initialMergeBase, loadMergeDraft, loraPayload,
  newLoraRow, pct, planHeadline, saveMergeDraft, weightHint,
} from './loraMerge';

/** Bake one or more LoRAs into a base checkpoint and get a full model out.
 *
 * WHY THIS EXISTS AT ALL
 * ----------------------
 * It is the step between "I trained a LoRA" and "I have a model I can publish",
 * and it is how the community actually makes the checkpoints it ships: of the
 * Krea 2 checkpoints whose authors describe their method, the ones that explain
 * themselves describe a merge, not a training run. LDS could train the adapter
 * and could quantize the result, and could not do the thing in the middle — so a
 * user could not reproduce what they were reading about.
 *
 * AND THE SECOND USE, WHICH IS THE ONE THAT UNBLOCKS SOMEBODY
 * -----------------------------------------------------------
 * A full-model run in this app targets Raw, and Raw is slow. Merging the
 * re-distillation LoRA Krea publishes for Turbo into that result is the
 * published route to getting few-step speed back — which is how the same model
 * appears on the model sites in both a Raw and a Turbo flavour. We have not
 * measured it ourselves and the screen says so: an untested route offered with
 * its reserve beats a capability we simply do not mention.
 *
 * TWO CLICKS, NOT ONE. The first answers, from the file headers alone and
 * without reading a weight: how many tensors change, exactly how big the output
 * is, which drive it lands on, how long it takes, and what happens if it dies
 * half way. A 26 GB write that started on one click is the surprise this shape
 * exists to remove — and every refusal the run could hit is decided in that same
 * plan, so the button is disabled with its reason rather than failing later.
 *
 * WHAT IT REFUSES TO CALL ITSELF. Not a finetune, not "trained". The output is a
 * base with LoRAs folded into it; the screen says that and so does the file's
 * `__metadata__`, because the file travels without this interface.
 */
export function LoraMergePlan({ plan, busy = false, disabled = false, onStart = null }) {
  if (!plan?.ok) return null;
  const carried = carriedOverNote(plan);
  return (
    <div className="mt-1.5 rounded-md border border-primary/40 bg-primary/10 px-2.5 py-1.5 text-[0.6875rem] leading-relaxed">
      <p className="m-0">{planHeadline(plan)}</p>
      <p className="m-0 mt-1 opacity-85">
        {plan.merged_tensors} of {plan.base_tensors} tensors change
        {plan.family_label ? <> · {plan.family_label}</> : null}
        {plan.estimated_seconds ? <> · takes {fmtDuration(plan.estimated_seconds)}</> : null}
      </p>
      <ul className="m-0 mt-1 list-none p-0">
        {(plan.loras || []).map((lora) => (
          <li key={lora.path || lora.name}
            className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
            <span className="font-mono break-all">{lora.name}</span>
            <span className="opacity-80">
              at {lora.weight} · rank {lora.rank}
              {lora.has_alpha ? ' · alpha recorded' : ''}
            </span>
          </li>
        ))}
      </ul>

      {/* Not a warning and not a boast: bytes we are about to copy without
          understanding them. One real community checkpoint hides ~75 MB of an
          image this way, under a perfectly legitimate prefix. */}
      {carried && <p className="m-0 mt-1 text-amber-200">ⓘ {carried}</p>}

      {typeof plan.free_bytes === 'number' && (
        <p className="m-0 mt-1 opacity-75">
          {fmtGB(plan.free_bytes)} free on that drive · {fmtGB(plan.required_bytes)} needed.
        </p>
      )}
      <p className="m-0 mt-1 opacity-75">{plan.on_failure}</p>

      <button type="button" onClick={onStart} disabled={busy || disabled}
        className="mt-1 rounded-md border border-primary/50 bg-primary/20 px-2.5 py-1 font-semibold text-white hover:bg-primary/30 disabled:opacity-40">
        {busy ? 'Starting…' : 'Merge into a full model'}
      </button>
    </div>
  );
}

/** The long half, alive. A merge of a 26 GB base is minutes, not seconds. */
function LoraMergeProgress({ state, onCancel = null }) {
  if (!state || !MERGE_RUNNING_STATES.includes(state.status)) return null;
  const width = pct(state.done, state.total);
  return (
    <div className="mt-1.5 rounded-md border border-primary/40 bg-primary/10 px-2.5 py-1.5 text-[0.6875rem] leading-relaxed"
      role="status">
      <p className="m-0">
        🧬 Merging on the CPU{state.total ? ` — ${state.done}/${state.total} tensors` : '…'}
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

/** What happened — and, when it worked, what this file is and is not. */
function LoraMergeOutcome({ state }) {
  if (!state) return null;
  const result = state.result || null;
  if (state.status === 'done') {
    return (
      <div className="mt-1.5 text-emerald-200 text-[0.6875rem] leading-relaxed" role="status">
        <p className="m-0">
          ✓ <span className="font-mono break-all">{state.destination_name}</span>
          {' '}({fmtGB(result?.bytes_after)}) is in
          {' '}<span className="font-mono break-all">{state.destination_dir}</span> and was
          re-opened successfully
          {result?.merged_tensors ? ` — ${result.merged_tensors} tensors merged` : ''}.
        </p>
        <p className="m-0 mt-1 opacity-85">
          It is a merged model, not a trained one — that is recorded in the file itself.
          {' '}Quantize it to fp8 if you want the smaller file ComfyUI loads.
        </p>
      </div>
    );
  }
  if (state.status === 'cancelled') {
    return (
      <p className="m-0 mt-1.5 text-amber-200 text-[0.6875rem] leading-relaxed" role="status">
        ■ Stopped. The partial file was removed; the base and the LoRAs are untouched.
      </p>
    );
  }
  if (state.status === 'error') {
    return (
      <p className="m-0 mt-1.5 text-rose-200 text-[0.6875rem] leading-relaxed" role="alert">
        ✗ {state.error || 'The merge failed.'} Nothing was overwritten.
      </p>
    );
  }
  return null;
}

export default function LoraMergeTool({
  base = '', baseLabel = '', family = null, disabled = false, framed = true,
}) {
  // WHO OWNS THE DRAFT. The instance that owns its base field — the one in
  // Checkpoints & LoRAs, which is exactly the one the portal remount empties.
  // A card instance is scoped to one model, is opened deliberately, and can be
  // mounted at the SAME time as that one: letting it read the draft would show
  // it another model's path, and letting it write would overwrite what somebody
  // typed in the other tool — the very loss this exists to prevent.
  const ownsDraft = !base;
  const [draft] = useState(() => (ownsDraft ? loadMergeDraft() : emptyMergeDraft()));
  const [basePath, setBasePath] = useState(() => initialMergeBase(base, draft));
  const [rows, setRows] = useState(() => draft.rows);
  const [plan, setPlan] = useState(null);
  const [busy, setBusy] = useState(false);
  const [state, setState] = useState(null);
  const [known, setKnown] = useState([]);
  const pollRef = useRef(null);

  // Follow the card's model when it hands a new one in. Guarded on a non-empty
  // value: this effect also runs on mount, and an empty `base` there would wipe
  // the draft we have just restored.
  useEffect(() => { if (base) setBasePath(base); }, [base]);

  // Keep what was typed. The subtree this renders in is unmounted whenever the
  // checkpoint manager moves between its portal host and its inline place, so
  // React state alone does not survive a window resize.
  useEffect(() => {
    if (ownsDraft) saveMergeDraft({ base: basePath, rows });
  }, [ownsDraft, basePath, rows]);

  const stop = () => { clearInterval(pollRef.current); pollRef.current = null; };

  // apiFetch RESOLVES THE PARSED BODY, not a Response — `.then((r) => r.json())`
  // here throws a TypeError the `.catch()` swallows, which once left a panel
  // stuck on "Quantizing…" while the job finished perfectly. Pinned by a test.
  const poll = useCallback(() => {
    apiFetch('/api/tools/lora-merge/status')
      .then((s) => {
        setState(s);
        if (!MERGE_RUNNING_STATES.includes(s?.status)) stop();
      })
      .catch(() => {});
  }, []);

  // A merge outlives a visit to this tab. Adopt a running one rather than
  // offering to start a second: the server refuses that anyway, and a refusal
  // is not the answer to "what is my model doing".
  useEffect(() => {
    let alive = true;
    apiFetch('/api/tools/lora-merge/status')
      .then((s) => {
        if (!alive || !MERGE_RUNNING_STATES.includes(s?.status)) return;
        setState(s);
        stop();
        pollRef.current = setInterval(poll, 2000);
      })
      .catch(() => {});
    return () => { alive = false; stop(); };
  }, [poll]);

  // The LoRAs ComfyUI can already see, offered as suggestions rather than as a
  // closed list: the Turbo re-distillation LoRA is a fresh download that may sit
  // anywhere, so a dropdown alone would lock out the main reason to be here.
  useEffect(() => {
    let alive = true;
    apiFetch(`/api/loras/list?family=${encodeURIComponent(family || 'krea')}`)
      .then((data) => { if (alive) setKnown(data?.loras || []); })
      .catch(() => {});
    return () => { alive = false; };
  }, [family]);

  const setRow = (id, patch) => setRows((current) => current.map(
    (row) => (row.id === id ? { ...row, ...patch } : row)));
  const addRow = () => setRows((current) => [...current, newLoraRow()]);
  const dropRow = (id) => setRows((current) => (current.length > 1
    ? current.filter((row) => row.id !== id)
    : [newLoraRow()]));

  const askPlan = async () => {
    setBusy(true);
    setPlan(await postJson('/api/tools/lora-merge/plan', {
      base: basePath.trim(), loras: loraPayload(rows),
    }));
    setBusy(false);
  };

  const start = async () => {
    setBusy(true);
    const res = await postJson('/api/tools/lora-merge', {
      base: basePath.trim(), loras: loraPayload(rows),
    });
    setBusy(false);
    if (!res?.ok) {
      setState({ status: 'error', error: res?.error || 'The merge could not start.' });
      return;
    }
    setPlan(null);
    // Submitted work is no longer a draft: keeping it would greet the next
    // visit with a form that looks unsent while the merge is already running.
    if (ownsDraft) clearMergeDraft();
    setState(res.status || { status: 'running' });
    stop();
    pollRef.current = setInterval(poll, 2000);
  };

  const cancel = async () => {
    await postJson('/api/tools/lora-merge/cancel', {});
    poll();
  };

  const running = MERGE_RUNNING_STATES.includes(state?.status);
  const fieldClass = 'w-full sm:flex-1 sm:min-w-[12rem] rounded border border-sky-300/40 bg-app/70 px-2 py-1 text-content text-[0.75rem] font-mono disabled:opacity-50';

  return (
    <div className={framed
      ? 'rounded-lg border border-sky-300/30 bg-sky-400/10 px-3 py-2 text-sky-50'
      : 'text-sky-50'}>
      {framed && (
        <>
          <span className="font-semibold">Merge a LoRA into a base — get a full model</span>
          <p className="m-0 mt-1 text-sky-200/75 text-[0.6875rem] leading-relaxed">
            {HONESTY_NOTE}
          </p>
        </>
      )}

      <div className={framed ? 'mt-2' : ''}>
        <label className="m-0 mb-1 block text-sky-200/75 text-[0.625rem] uppercase tracking-wide"
          htmlFor="lora-merge-base">
          Base checkpoint {baseLabel ? `— ${baseLabel}` : ''}
        </label>
        <input id="lora-merge-base" type="text" value={basePath}
          onChange={(event) => setBasePath(event.target.value)}
          disabled={running || disabled}
          placeholder="Full path to the full-precision .safetensors"
          aria-label="Path of the base checkpoint to merge into"
          className={fieldClass} />
      </div>

      <p className="m-0 mt-2 mb-1 text-sky-200/75 text-[0.625rem] uppercase tracking-wide">
        LoRAs to fold in
      </p>
      <datalist id="lora-merge-known">
        {known.map((lora) => <option key={lora.name} value={lora.name} />)}
      </datalist>
      {rows.map((row) => (
        <div key={row.id}
          className="mb-1 flex flex-col gap-1 sm:flex-row sm:flex-wrap sm:items-center">
          <input type="text" value={row.path} list="lora-merge-known"
            onChange={(event) => setRow(row.id, { path: event.target.value })}
            disabled={running || disabled}
            placeholder="LoRA name or full path"
            aria-label="LoRA file to merge"
            className={fieldClass} />
          <input type="number" value={row.weight} step="0.05"
            min={WEIGHT_MIN} max={WEIGHT_MAX}
            onChange={(event) => setRow(row.id, { weight: event.target.value })}
            disabled={running || disabled}
            aria-label="Weight for this LoRA"
            title="1.0 applies the LoRA exactly as trained"
            className="w-full sm:w-20 shrink-0 rounded border border-sky-300/40 bg-app/70 px-2 py-1 text-content text-[0.75rem] disabled:opacity-50" />
          <button type="button" onClick={() => dropRow(row.id)}
            disabled={running || disabled}
            aria-label="Remove this LoRA"
            className="shrink-0 self-start rounded-md border border-white/30 bg-black/20 px-2.5 py-1 text-[0.75rem] font-semibold hover:bg-black/30 disabled:opacity-40">
            Remove
          </button>
          {weightHint(row.weight) && (
            <span className="basis-full text-sky-200/70 text-[0.625rem]">
              {weightHint(row.weight)}
            </span>
          )}
        </div>
      ))}

      <div className="flex flex-col gap-1 sm:flex-row sm:flex-wrap sm:items-center">
        <button type="button" onClick={addRow} disabled={running || disabled}
          className="shrink-0 self-start rounded-md border border-white/30 bg-black/20 px-2.5 py-1 text-[0.75rem] font-semibold hover:bg-black/30 disabled:opacity-40">
          + Another LoRA
        </button>
        <button type="button" onClick={askPlan}
          disabled={disabled || busy || running || !canAskPlan(basePath, rows)}
          className="shrink-0 self-start rounded-lg border border-primary/40 bg-primary/20 px-2.5 py-1 text-[0.75rem] font-semibold text-white disabled:opacity-40">
          Check this merge
        </button>
      </div>

      <p className="m-0 mt-1.5 text-sky-200/70 text-[0.625rem] leading-relaxed">{TURBO_NOTE}</p>
      <p className="m-0 mt-1 text-sky-200/70 text-[0.625rem] leading-relaxed">{PRECISION_NOTE}</p>

      {plan && !plan.ok && !running && (
        <p className="m-0 mt-1 text-amber-200 text-[0.6875rem]" role="alert">⚠ {plan.error}</p>
      )}
      {!running && (
        <LoraMergePlan plan={plan} busy={busy} disabled={disabled} onStart={start} />
      )}
      <LoraMergeProgress state={state} onCancel={cancel} />
      <LoraMergeOutcome state={state} />
    </div>
  );
}
