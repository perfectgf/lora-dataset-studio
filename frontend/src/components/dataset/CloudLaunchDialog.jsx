// react-frontend/src/components/dataset/CloudLaunchDialog.jsx
// The cloud-training launch dialog, its GPU-tier estimate and the one-time
// custom-base push gate, moved VERBATIM from TrainingPanel.jsx
// (2026-08-24, panel decomposition slice 1).
import { useEffect, useState } from 'react';
import { postJson } from '../../hooks/useDataset';
import { launchButtonLabel } from '../../utils/launchProgress';
import SettingsLink from '../common/SettingsLink';
import CloudTierEstimate from '../shared/CloudTierEstimate';
import { customBasePushView } from './customBasePush.js';
import { baseName, fmtBytes } from './panelFormatters';
import {
  TRAINING_MODE_FULL_TRANSFORMER,
  cloudTierEstimateView,
  denseTurboWarning,
  fullTransformerBaseLabel,
  hfCloudTokenReadiness,
  normalizeTrainingMode,
  trainingModeLabel,
} from '../../utils/trainingMode.js';

const _FAMILY_LABEL = { zimage: 'Z-Image', krea: 'Krea 2', sdxl: 'SDXL', flux: 'FLUX.1', flux2klein: 'FLUX.2 Klein', anima: 'Anima' };


/* Custom-base gate inside the cloud dialog: a custom base trains from a
   PRIVATE repo on the user's Hugging Face account (lds-base-<hash>). This
   section checks whether that repo already carries the base (cache-hit →
   launch straight away) and otherwise offers the ONE-TIME push — uploaded
   once, reused by every future cloud run, never public. */
function CustomBasePushSection({ datasetId, trainType, variant, base, onReadyChange }) {
  const [state, setState] = useState(null);      // last GET /custom-base payload
  const [checkError, setCheckError] = useState(null);
  const [pushBusy, setPushBusy] = useState(false);
  const [pushError, setPushError] = useState(null);
  const [pollNonce, setPollNonce] = useState(0);

  useEffect(() => {
    let alive = true;
    let timer;
    const tick = async () => {
      let d = null;
      try {
        const qs = new URLSearchParams({ train_type: trainType, base_model: base });
        if (variant) qs.set('variant', variant);
        const r = await fetch(`/api/dataset/${datasetId}/train/cloud/custom-base?${qs.toString()}`,
          { credentials: 'include' });
        d = await r.json().catch(() => ({}));
        if (!alive) return;
        if (!r.ok || d.ok === false) {
          setCheckError(d.error || `Could not check the custom base (HTTP ${r.status})`);
          d = null;
        } else {
          setCheckError(null);
          setState(d);
        }
      } catch {
        if (alive) setCheckError('Network error while checking the custom base');
      }
      // Keep polling while the background push is running (multi-GB upload).
      if (alive && d?.job?.state === 'running') timer = setTimeout(tick, 3000);
    };
    tick();
    return () => { alive = false; clearTimeout(timer); };
  }, [datasetId, trainType, variant, base, pollNonce]);

  const ready = !!state?.ready;
  useEffect(() => { onReadyChange(ready); }, [ready]); // eslint-disable-line react-hooks/exhaustive-deps

  const startPush = async (allowUnverified = false) => {
    setPushBusy(true);
    setPushError(null);
    try {
      const d = await postJson(`/api/dataset/${datasetId}/train/cloud/custom-base/push`, {
        train_type: trainType, variant, base_model: base,
        ...(allowUnverified ? { allow_unverified_weights: true } : {}),
      });
      if (d && d.ok === false) {
        const msg = String(d.error || 'Push failed');
        const marker = 'CUSTOM_WEIGHTS_UNVERIFIED: ';
        if (!allowUnverified && msg.includes(marker)) {
          const detail = msg.slice(msg.indexOf(marker) + marker.length);
          if (window.confirm(`${detail}\n\nPush anyway (force)?`)) return startPush(true);
        } else {
          setPushError(msg);
        }
        return;
      }
      setPollNonce((n) => n + 1);        // job started — begin polling its state
    } finally {
      setPushBusy(false);
    }
  };

  const job = state?.job || {};
  const pushing = pushBusy || job.state === 'running';
  const sizeLabel = state?.local_size_bytes != null ? ` (~${fmtBytes(state.local_size_bytes)})` : '';
  const view = customBasePushView({ state, checkError, pushing });
  let body;
  if (view.kind === 'foreign') {
    // Another family's base: nothing to push, nothing to restore. Say what the
    // run will do instead of offering an upload that could only fail.
    body = <p className="m-0 text-amber-300 text-[0.75rem]">⚠ {view.message}</p>;
  } else if (checkError) {
    body = <p className="m-0 text-red-300 text-[0.75rem]">⚠ {checkError}</p>;
  } else if (!state) {
    body = <p className="m-0 text-content-muted text-[0.75rem]">Checking your custom base on Hugging Face…</p>;
  } else if (ready) {
    body = (
      <p className="m-0 text-emerald-300 text-[0.75rem]">
        ✓ Custom base found in your private repo <span className="font-mono">{state.repo_id}</span> —
        the pod downloads it with your HF token. Nothing to upload again.
      </p>
    );
  } else if (state.reason === 'no_token') {
    body = (
      <p className="m-0 text-amber-300 text-[0.75rem]">
        ⚠ Add your Hugging Face token (HF_TOKEN) in Settings ▸ API keys first — your custom
        base rides in a private repo on your account, and the pod needs the token to read it.
      </p>
    );
  } else if (state.reason === 'token_invalid') {
    body = (
      <p className="m-0 text-amber-300 text-[0.75rem]">
        ⚠ Your Hugging Face token was rejected — paste a valid HF_TOKEN in Settings ▸ API keys.
      </p>
    );
  } else if (pushing) {
    body = (
      <p className="m-0 text-sky-200 text-[0.75rem]">
        ⬆ Uploading your custom base{sizeLabel} to the private repo
        {state.repo_id ? <> <span className="font-mono">{state.repo_id}</span></> : null}…
        One-time upload — every future cloud run reuses it. Keep the app running.
      </p>
    );
  } else {
    body = (
      <div className="flex flex-col gap-1.5">
        <p className="m-0 text-content-muted text-[0.75rem]">
          {view.message} Pushing uploads your custom base{sizeLabel} to a <b className="text-content">PRIVATE</b> repo
          on your Hugging Face account — one time; future cloud runs reuse it. It is never made public.
        </p>
        {view.warning && (
          <p className="m-0 text-amber-300 text-[0.75rem]">⚠ {view.warning}</p>
        )}
        {(pushError || job.state === 'error') && (
          <p className="m-0 text-red-300 text-[0.75rem]">⚠ {pushError || job.error}</p>
        )}
        <button type="button" onClick={() => startPush(false)}
          disabled={!view.canPush || pushBusy}
          className="w-fit px-3 py-1.5 rounded-lg border border-sky-500/50 bg-sky-500/10 text-sky-200 text-sm font-semibold disabled:opacity-40">
          ⬆ Push custom base to Hugging Face (one-time)
        </button>
      </div>
    );
  }
  return (
    <div className="rounded-lg border border-border bg-surface px-3 py-2">
      <p className="m-0 mb-1 text-content text-[0.75rem] font-semibold">
        Custom base: <span className="font-mono font-normal">{baseName(base)}</span>
      </p>
      {body}
    </div>
  );
}


/* Launch-time GPU speed picker. Fetches live vast.ai offers grouped by GPU
   class (slowest→fastest), each with price/h and an APPROXIMATE training time
   and total run cost for this dataset+family. Picking a tier rents the cheapest
   live offer of that class; the price cap in Settings still bounds what's shown.
   A custom base adds the push gate above the tiers (see CustomBasePushSection). */
function CloudLaunchDialog({
  datasetId, trainType, variant, trainingMode, base, steps, keptCount,
  cloudStatus, preflightTokenIssue, onClose, onLaunch,
}) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);     // {tiers, steps, family, max_price_per_hour}
  const [selected, setSelected] = useState(null);
  const [launching, setLaunching] = useState(false);
  // Seconds since the click. The launch POST freezes the dataset, checks the
  // base repository and (full model) creates the delivery repository, so it can
  // run for tens of seconds — a motionless 'Launching…' was reported as a hang.
  const [launchElapsed, setLaunchElapsed] = useState(0);
  const fullMode = normalizeTrainingMode(trainingMode) === TRAINING_MODE_FULL_TRANSFORMER;
  // Custom base ('' = official): the launch stays blocked until the private
  // repo on the user's HF account carries the base (pushed once, reused).
  // Dense runs are no longer excluded — the transport is the same private repo
  // and the same pod-side rewrite; excluding them here would have left the
  // lifted refusal with no way to actually get the weights to the GPU.
  const isCustomBase = !!String(base || '').trim();
  const [customBaseReady, setCustomBaseReady] = useState(!isCustomBase);
  // Last chance to read it before the money is committed.
  const turboNotice = fullMode ? denseTurboWarning({ baseModel: base, variant }) : null;
  // The dialog has no catalog labels, so a custom base falls back to its file
  // name — never the full path (paste-safe, and this string is user-visible).
  const denseBase = fullTransformerBaseLabel({ baseModel: base, variant });

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError(null);
    setData(null);
    setSelected(null);
    (async () => {
      try {
        const qs = new URLSearchParams({
          train_type: trainType,
          variant,
          base_model: base ?? '',
          training_mode: normalizeTrainingMode(trainingMode),
        });
        if (steps) qs.set('steps', String(steps));
        const r = await fetch(`/api/dataset/${datasetId}/train/cloud/offers?${qs.toString()}`,
          { credentials: 'include' });
        const body = await r.json().catch(() => ({}));
        if (!alive) return;
        // Keep readiness metadata even when offer discovery itself failed so the
        // modal can name the token problem and link to the exact Settings field.
        setData(body);
        if (!r.ok || body.ok === false) {
          setError(body.error || body.hint || `Could not load offers (HTTP ${r.status})`);
        } else {
          if (body.tiers && body.tiers.length) setSelected(body.tiers[0].gpu_name);
        }
      } catch {
        if (alive) setError('Network error while loading GPU offers');
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, [datasetId, trainType, variant, base, trainingMode, steps]);

  const go = async () => {
    if (!selected) return;
    setLaunching(true);
    setLaunchElapsed(0);
    const started = Date.now();
    const tick = setInterval(
      () => setLaunchElapsed(Math.round((Date.now() - started) / 1000)), 1000);
    try {
      const launched = await onLaunch(selected);      // owns its own error toasts
      if (launched) onClose();
    } finally {
      clearInterval(tick);
      setLaunching(false);
    }
  };

  const tiers = data?.tiers || [];
  const budget = cloudStatus?.monthly_budget || 0;
  const spent = cloudStatus?.month_spend || 0;
  const hasUsableEstimate = tiers.some((tier) => (
    cloudTierEstimateView(tier, { fullMode }).available
  ));
  const offerTokenReadiness = fullMode ? hfCloudTokenReadiness(data || {}) : null;
  // The saved token is verified server-side on every offer fetch. Repeating
  // "configure it before renting the GPU" once it has passed reads as a refusal
  // and sent users hunting for a Settings problem that does not exist.
  const offerTokenStatus = fullMode ? (data?.hf_cloud_token || null) : null;
  const hfTokenVerified = offerTokenStatus?.ok === true;
  const hfTokenBroad = hfTokenVerified && offerTokenStatus?.code === 'broad_access';
  const hfTokenIssue = fullMode && (preflightTokenIssue
    || (offerTokenReadiness?.blocked
      ? offerTokenReadiness.detail
        || 'HF_CLOUD_TOKEN is missing, invalid, or does not have the required permissions.'
      : null));
  const hfTokenBlocked = !!hfTokenIssue;

  return (
    <div role="dialog" aria-modal="true"
      aria-label={fullMode ? 'Choose an 80 GB cloud GPU for full-model training' : 'Choose cloud GPU speed'}
      className="fixed inset-0 z-50 grid place-items-center bg-black/60 p-4"
      onKeyDown={(e) => { if (e.key === 'Escape') onClose(); }}>
      <div className="w-full max-w-lg rounded-xl border border-border bg-surface-overlay p-4 flex flex-col gap-3">
        <h3 className="m-0 text-content font-bold text-sm">
          <span aria-hidden>☁️</span> {fullMode
            ? 'Choose an 80 GB GPU for full-model training'
            : 'Choose GPU speed for this run'}
        </h3>

        {fullMode && !hfTokenIssue && (
          hfTokenVerified ? (
            <p className={`m-0 rounded-lg border px-3 py-2 text-[0.75rem] leading-relaxed ${
              hfTokenBroad
                ? 'border-amber-400/35 bg-amber-500/[0.08] text-amber-100'
                : 'border-emerald-400/35 bg-emerald-500/[0.08] text-emerald-100'}`}>
              <span className="font-semibold">
                {hfTokenBroad
                  ? 'Hugging Face delivery ready (broad token).'
                  : 'Hugging Face delivery ready.'}
              </span>{' '}
              {hfTokenBroad
                ? (offerTokenStatus?.warning
                  || 'This token has global write access. It works, but a fine-grained token limited to Krea 2 reads and one delivery namespace is safer.')
                : 'The dedicated token can read the official base and write the delivery repository.'}
              {offerTokenStatus?.namespace ? ` Delivery namespace: ${offerTokenStatus.namespace}.` : ''}
            </p>
          ) : (
            <p className="m-0 rounded-lg border border-amber-400/35 bg-amber-500/[0.08] px-3 py-2 text-amber-100 text-[0.75rem] leading-relaxed">
              This run requires an <code>HF_CLOUD_TOKEN</code> that can read the Krea 2 base it
              trains from ({denseBase}) and write the delivery repository. A tightly scoped fine-grained token is recommended. A global
              write token is also accepted with a warning. Configure it in{' '}
              <SettingsLink section="local-tools" focus="HF_CLOUD_TOKEN" tone="warning">Settings ▸ Local tools</SettingsLink>
              {' '}before renting the GPU.
            </p>
          )
        )}

        {hfTokenIssue && (
          <div role="alert"
            className="rounded-lg border border-red-400/45 bg-red-500/10 px-3 py-2 text-red-100 text-[0.75rem] leading-relaxed">
            <span className="font-semibold">Hugging Face delivery blocked.</span>{' '}{hfTokenIssue}{' '}
            Fix <SettingsLink section="local-tools" focus="HF_CLOUD_TOKEN" tone="warning">HF_CLOUD_TOKEN in Settings ▸ Local tools</SettingsLink>,
            then reload the offers. Launch stays disabled to prevent renting a GPU without a delivery path.
          </div>
        )}

        {turboNotice && (
          <div role="status"
            className="rounded-lg border border-amber-400/45 bg-amber-500/[0.09] px-3 py-2 text-amber-100 text-[0.75rem] leading-relaxed">
            <span className="font-semibold">⚠ {turboNotice.title}.</span>{' '}{turboNotice.body}
          </div>
        )}

        {isCustomBase && (
          <CustomBasePushSection
            datasetId={datasetId} trainType={trainType} variant={variant}
            base={base} onReadyChange={setCustomBaseReady} />
        )}

        {loading && <p className="m-0 text-content-muted text-sm">Loading live GPU offers…</p>}
        {error && (
          <p className="m-0 text-red-300 text-sm">
            ⚠ {error}
            {fullMode && (
              <span className="block mt-1 text-amber-200 text-[0.75rem]">
                Also check the dedicated <code>HF_CLOUD_TOKEN</code> in Settings ▸ Local tools.
              </span>
            )}
          </p>
        )}
        {!loading && !error && tiers.length === 0 && (
          <p className="m-0 text-content-muted text-sm">
            No GPU available under ${data?.max_price_per_hour}/h right now. Try again shortly, or{' '}
            <SettingsLink section="training" focus="cloud-max-price-per-hour">
              increase the price cap in Settings
            </SettingsLink>.
          </p>
        )}

        {tiers.length > 0 && (
          <div className="flex flex-col gap-1.5 max-h-[50vh] overflow-y-auto">
            {tiers.map((t) => (
              <label key={t.gpu_name}
                className={`flex items-center gap-3 rounded-lg border px-3 py-2 cursor-pointer transition-colors ${
                  selected === t.gpu_name
                    ? 'border-sky-400/70 bg-sky-500/10'
                    : 'border-border bg-surface hover:bg-surface-raised'}`}>
                <input type="radio" name="gpu-tier" className="accent-sky-400"
                  checked={selected === t.gpu_name}
                  onChange={() => setSelected(t.gpu_name)} />
                <span className="flex-1 min-w-0">
                  <span className="block text-content text-sm font-semibold truncate">
                    {t.gpu_name}
                    {t.gpu_ram_gb ? <span className="text-content-subtle font-normal"> · {t.gpu_ram_gb} GB</span> : null}
                  </span>
                  <CloudTierEstimate tier={t} fullMode={fullMode}
                    maxRuntimeMinutes={data?.max_runtime_minutes} />
                </span>
              </label>
            ))}
          </div>
        )}

        <p className="m-0 text-content-subtle text-[0.6875rem]">
          {fullMode ? `${trainingModeLabel(trainingMode)} · ${denseBase}` : `${data?.steps ?? steps ?? '—'} steps · ${_FAMILY_LABEL[data?.family || trainType] || (data?.family || trainType)}`}
          {keptCount != null ? ` · ${keptCount} img` : ''}
          {budget > 0 ? ` · this month: $${spent.toFixed(2)} of $${budget.toFixed(2)}` : ''}
          {fullMode
            ? hasUsableEstimate
              ? '. Full-model duration and cost are approximate; the ~26 GB model is uploaded to your private Hugging Face repository before the pod stops.'
              : '. No reliable full-model benchmark is available; compare hourly prices only. The ~26 GB model is uploaded to your private Hugging Face repository at the end of a clean run.'
            : '. Time & cost are approximate; the pod is auto-terminated when done.'}
        </p>

        {/* What the frozen button is actually waiting on. Announced once (the
            text does not change as the counter runs, so it cannot re-announce
            every second) and wrapped for a 400 px phone. */}
        {launching && (
          <p aria-live="polite"
            className="m-0 rounded-lg border border-sky-400/35 bg-sky-500/[0.08] px-3 py-2 text-sky-100 text-[0.75rem] leading-relaxed">
            Reserving the run: freezing the dataset and checking the base model
            {fullMode ? ' and the Hugging Face delivery repository' : ''}. This can take
            up to a minute. The GPU is rented right after, and the run then follows
            its own progress on the Runs page — you can close this window once it opens.
          </p>
        )}

        <div className="flex items-center gap-2">
          <button type="button" onClick={go}
            disabled={!selected || launching || !customBaseReady || hfTokenBlocked}
            title={hfTokenBlocked
              ? 'Configure a valid HF_CLOUD_TOKEN with the required permissions before launching'
              : !customBaseReady ? 'Push the custom base to your Hugging Face account first' : undefined}
            className="px-3 py-1.5 rounded-lg bg-gradient-primary text-gray-950 text-sm font-semibold disabled:opacity-40">
            {launchButtonLabel({ launching, elapsedSeconds: launchElapsed, fullMode })}
          </button>
          <button type="button" onClick={onClose} disabled={launching}
            className="ml-auto px-3 py-1.5 rounded-lg text-content-muted hover:text-content text-sm disabled:opacity-40">
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}

export default CloudLaunchDialog;
