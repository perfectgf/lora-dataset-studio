import { useEffect, useRef, useState } from 'react';

/* Live view of the CURRENT run (mounted by TrainingPanel while this dataset
   trains): progress bar, loss sparkline, and the sample previews ai-toolkit
   writes every sample_every steps. Polls /train/progress every 5 s.

   Sparkline design (single series): one hue (the app's indigo accent), 2px
   line, no legend (the label names it), values/labels in text tokens — and a
   text readout of the current loss so the number never depends on the plot. */

const POLL_MS = 5000;

function LossSparkline({ curve }) {
  const [hover, setHover] = useState(null);   // {step, loss, x, y}
  if (!curve || curve.length < 2) return null;
  const W = 560, H = 64, PAD = 4;
  const losses = curve.map((p) => p[1]);
  const steps = curve.map((p) => p[0]);
  const minL = Math.min(...losses), maxL = Math.max(...losses);
  const minS = steps[0], maxS = steps[steps.length - 1];
  const x = (s) => PAD + ((s - minS) / Math.max(1, maxS - minS)) * (W - 2 * PAD);
  const y = (l) => maxL === minL ? H / 2 : PAD + ((maxL - l) / (maxL - minL)) * (H - 2 * PAD);
  const points = curve.map(([s, l]) => `${x(s).toFixed(1)},${y(l).toFixed(1)}`).join(' ');
  const onMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    let best = curve[0], bd = Infinity;
    for (const p of curve) {
      const d = Math.abs(x(p[0]) - px);
      if (d < bd) { bd = d; best = p; }
    }
    setHover({ step: best[0], loss: best[1], x: x(best[0]), y: y(best[1]) });
  };
  return (
    <div className="flex flex-col gap-0.5">
      <div className="flex items-center gap-2 text-[0.625rem] text-content-subtle">
        <span className="uppercase text-content-muted">Loss</span>
        <span className="tabular-nums">min {minL.toExponential(2)} · max {maxL.toExponential(2)}</span>
        {hover && (
          <span className="ml-auto tabular-nums text-content">
            step {hover.step} · loss {hover.loss.toExponential(3)}
          </span>
        )}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-16 rounded bg-app/40 border border-border"
        role="img" aria-label={`Training loss curve, ${curve.length} points, latest ${losses[losses.length - 1].toExponential(3)}`}
        onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        <polyline points={points} fill="none" stroke="#818cf8" strokeWidth="2"
          strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
        {hover && (
          <circle cx={hover.x} cy={hover.y} r="3.5" fill="#818cf8" stroke="#0b0b10" strokeWidth="2" />
        )}
      </svg>
    </div>
  );
}

export default function TrainingProgress({ datasetId, base, trainType, cloud = false }) {
  const [prog, setProg] = useState(null);
  const timer = useRef(null);
  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const qs = new URLSearchParams();
        if (base != null) qs.set('base_model', base);
        if (trainType) qs.set('train_type', trainType);
        const r = await fetch(`/api/dataset/${datasetId}/train/${cloud ? 'cloud/' : ''}progress?${qs}`, { credentials: 'include' });
        if (r.ok) {
          const d = await r.json();
          if (alive) setProg(d);
        }
      } catch { /* transient poll error — next tick retries */ }
      if (alive) timer.current = setTimeout(poll, POLL_MS);
    };
    poll();
    return () => { alive = false; clearTimeout(timer.current); };
  }, [datasetId, base, trainType, cloud]);

  if (!prog || (!prog.log_exists && !(prog.samples || []).length)) {
    if (cloud && prog?.phase) {
      return (
        <p className="m-0 text-sky-300 text-[0.625rem]">
          ☁ {prog.phase}{prog.phase_detail ? ` — ${prog.phase_detail}` : ''}
        </p>
      );
    }
    return (
      <p className="m-0 text-content-subtle text-[0.625rem]">
        Starting up… (the log appears once ai-toolkit begins writing)
      </p>
    );
  }
  const pct = prog.step && prog.total ? Math.min(100, Math.round((prog.step / prog.total) * 100)) : null;
  const samples = prog.samples || [];
  return (
    <div className="flex flex-col gap-2 rounded-lg border border-border bg-surface px-3 py-2">
      {cloud && prog.phase && (
        <p className="m-0 text-sky-300 text-[0.625rem]">{prog.phase}{prog.phase_detail ? ` — ${prog.phase_detail}` : ''}</p>
      )}
      {pct != null && (
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2 text-[0.6875rem] text-content-muted flex-wrap">
            <span className="text-content font-semibold tabular-nums">{prog.step} / {prog.total} steps ({pct}%)</span>
            {prog.loss != null && <span className="tabular-nums">loss {prog.loss.toExponential(3)}</span>}
            {prog.speed && <span className="tabular-nums">{prog.speed}</span>}
            {prog.eta && <span className="tabular-nums">ETA {prog.eta}</span>}
          </div>
          <div className="h-2 rounded bg-app/60 border border-border overflow-hidden"
            role="progressbar" aria-valuenow={prog.step} aria-valuemin={0} aria-valuemax={prog.total}
            aria-label="Training progress">
            <div className="h-full bg-gradient-primary transition-all duration-700" style={{ width: `${pct}%` }} />
          </div>
        </div>
      )}
      <LossSparkline curve={prog.loss_curve} />
      {samples.length > 0 && (
        <div className="flex flex-col gap-1">
          <span className="text-content-muted text-[0.625rem] uppercase">
            Samples (auto-generated during the run — newest first)
          </span>
          <div className="flex gap-1.5 overflow-x-auto pb-1">
            {samples.map((s) => {
              const qs = new URLSearchParams();
              if (base != null) qs.set('base_model', base);
              if (trainType) qs.set('train_type', trainType);
              const url = `/api/dataset/${datasetId}/train/${cloud ? 'cloud/' : ''}sample/${encodeURIComponent(s.filename)}?${qs}`;
              return (
                <a key={s.filename} href={url} target="_blank" rel="noreferrer"
                  title={`Step ${s.step} — prompt ${s.prompt_idx + 1} (open full size)`}
                  className="relative shrink-0 w-20 h-20 rounded border border-border overflow-hidden hover:border-indigo-400">
                  <img src={url} alt={`Training sample at step ${s.step}`} loading="lazy"
                    className="w-full h-full object-cover" />
                  <span className="absolute bottom-0 inset-x-0 bg-black/70 text-white text-[0.5625rem] text-center tabular-nums">
                    step {s.step}
                  </span>
                </a>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
