/**
 * The four options that change what the model computes, plus the three dials
 * that change what it is asked for — laid out for the render rail (redesign,
 * 2026-08-31): one column on a wide screen, where the rail is 360 px, two
 * columns when the rail stacks under the take on a phone.
 *
 * Each option says what it COSTS as well as what it buys, because none of them
 * is free and the price is invisible in the output. The price is now a TAG on
 * the row, the sentence under it is one line — the first build's paragraph per
 * option read as a settings dump, and nobody read four of them.
 *
 * Nothing here is a number this file invented. The clip lengths and the fps come
 * from the shared target catalogue through `/options`, so a length offered here
 * is a length the VAE accepts.
 */
import { Sparkles, Flame, Zap, Maximize2 } from 'lucide-react';
import { clipSeconds, SPARSE_CHOICES } from './videoStudioApi';

function Toggle({ checked, onChange, icon: Icon, label, cost, hint, disabled, disabledHint }) {
  return (
    <label className={`flex items-start gap-2 rounded-lg border px-2.5 py-2 min-h-10 lg:min-h-0 ${
      disabled ? 'border-border opacity-60' : `cursor-pointer ${checked ? 'border-accent/60 bg-accent/5' : 'border-border hover:border-accent/50'}`}`}>
      <input type="checkbox" checked={!!checked} disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5 h-4 w-4 shrink-0 accent-accent" />
      <span className="min-w-0 flex-1">
        <span className="flex items-center gap-1.5 text-sm text-content">
          <Icon aria-hidden="true" className="h-3.5 w-3.5 text-content-muted" />
          <span className="min-w-0 flex-1 truncate">{label}</span>
          {!disabled && cost && (
            <span className="shrink-0 rounded-full border border-border px-1.5 py-px text-[0.625rem] text-content-subtle">
              {cost}
            </span>
          )}
        </span>
        <span className="block text-[0.6875rem] leading-snug text-content-subtle">
          {disabled ? disabledHint : hint}
        </span>
      </span>
    </label>
  );
}

export default function VideoOptionsPanel({ options, value, onChange }) {
  const set = (patch) => onChange({ ...value, ...patch });
  /* What this ComfyUI can actually run. `available === false` is a verdict (the
     pack is absent); `null` or missing is "could not ask", and an option is
     offered as usual there — a probe that did not run must not read as a no. */
  const avail = options?.options_available || {};
  const off = (k) => avail[k]?.available === false;
  /* Names the PACK, because that is what the user has to go and get: this app
     downloads model files but does not install nodes into somebody's ComfyUI.
     The ComfyUI-Manager search term is included — it is how most people will
     actually install it. */
  const need = (k) => {
    const a = avail[k];
    return a?.pack
      ? `Needs the ${a.pack} node pack in ComfyUI (ComfyUI-Manager: “${a.search}”), then a restart.`
      : 'Needs a ComfyUI node pack that is not installed.';
  };
  const frames = options?.frame_choices?.length ? options.frame_choices : [39, 56, 73, 107];
  const fps = options?.fps || 24;
  const mp = options?.megapixels || { min: 0.1, max: 2, default: 0.3 };
  const seconds = clipSeconds(value.frames, fps);
  const sparseHint = off('sparse')
    ? need('sparse')
    : SPARSE_CHOICES.find((c) => c.value === value.sparse)?.hint;

  return (
    <section data-probe-panel="video-studio-options"
      className="flex flex-col gap-3 rounded-xl border border-border bg-surface p-3">
      <div className="flex flex-col gap-1.5">
        <h2 className="text-sm font-semibold text-content">Render</h2>
        <div className="grid gap-1.5 sm:grid-cols-2 lg:grid-cols-1">
          <Toggle checked={value.turbo && !off('turbo')} onChange={(v) => set({ turbo: v })}
            icon={Zap} label={`Turbo, ${options?.turbo_steps || 6} steps`} cost="minutes, not tens"
            disabled={off('turbo')} disabledHint={need('turbo')}
            hint="A distillation LoRA with its own sampler — a different model, not a faster one." />
          <Toggle checked={value.eros} onChange={(v) => set({ eros: v })}
            icon={Flame} label="10Eros base" cost="its own faces"
            disabled={options && !options.eros_available}
            disabledHint="Not on this machine — the official base is used."
            hint="A third-party finetune in place of the official base; works against an identity test." />
          <Toggle checked={value.latentUpscale && !off('latent_upscale')}
            onChange={(v) => set({ latentUpscale: v })}
            icon={Maximize2} label="Latent upscale ×2" cost="+ minutes"
            disabled={off('latent_upscale')} disabledHint={need('latent_upscale')}
            hint="Enlarges before decoding, audio untouched. This is the pass that costs the time." />
          <label className={`flex flex-col gap-1 rounded-lg border px-2.5 py-2 ${
            off('sparse') ? 'border-border opacity-60' : value.sparse ? 'border-accent/60 bg-accent/5' : 'border-border'}`}>
            <span className="flex items-center gap-1.5 text-sm text-content">
              <Sparkles aria-hidden="true" className="h-3.5 w-3.5 text-content-muted" />
              <span className="min-w-0 flex-1">Sparse attention</span>
              {!off('sparse') && (
                <span className="shrink-0 rounded-full border border-border px-1.5 py-px text-[0.625rem] text-content-subtle">
                  speed for fidelity
                </span>
              )}
            </span>
            <select value={off('sparse') ? '' : value.sparse} disabled={off('sparse')}
              onChange={(e) => set({ sparse: e.target.value })}
              className="w-full rounded-md border border-border bg-app px-2 py-1 text-xs text-content min-h-10 lg:min-h-0">
              {SPARSE_CHOICES.map((c) => (
                <option key={c.value || 'off'} value={c.value}>{c.label}</option>
              ))}
            </select>
            <span className="text-[0.6875rem] leading-snug text-content-subtle">{sparseHint}</span>
          </label>
        </div>
        {value.sparse && value.sparse !== 'max' && value.latentUpscale && (
          <p className="rounded-lg border border-border bg-app px-2.5 py-1.5 text-[0.6875rem] leading-snug text-content-muted">
            With the upscale on, the first pass stays dense and only the upscale
            samples sparse — the prompt keeps its say where it sets the
            composition. Pick <strong>Max</strong> to accelerate both.
          </p>
        )}
      </div>

      <div className="flex flex-col gap-1.5 border-t border-border pt-3">
        <h3 className="font-mono text-[0.625rem] uppercase tracking-[0.18em] text-content-subtle">Shot</h3>
        <div className="grid gap-2 sm:grid-cols-3 lg:grid-cols-1">
          <label className="flex flex-col gap-1 text-xs text-content-muted">
            Length
            <select value={value.frames} onChange={(e) => set({ frames: Number(e.target.value) })}
              className="rounded-lg border border-border bg-app px-2 py-1.5 text-content min-h-10 lg:min-h-0">
              {frames.map((f) => (
                <option key={f} value={f}>{f} frames · {clipSeconds(f, fps)}s</option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs text-content-muted">
            <span className="flex items-baseline justify-between">
              Resolution
              <span className="tabular-nums text-content">{Number(value.megapixels).toFixed(2)} MP</span>
            </span>
            <input type="range" min={mp.min} max={mp.max} step="0.05"
              value={value.megapixels}
              onChange={(e) => set({ megapixels: Number(e.target.value) })}
              className="mt-1 accent-accent" />
          </label>
          <label className="flex flex-col gap-1 text-xs text-content-muted">
            Seed
            <input type="number" value={value.seed} placeholder="random"
              onChange={(e) => set({ seed: e.target.value })}
              className="rounded-lg border border-border bg-app px-2 py-1.5 text-content min-h-10 lg:min-h-0" />
          </label>
        </div>
        <p className="text-[0.6875rem] leading-snug text-content-subtle">
          {seconds ? `${value.frames} frames at ${fps} fps — a ${seconds}s clip. ` : ''}
          Faces sharpen up to about 1 MP; past that the machine that runs the job
          decides whether it fits.
        </p>
      </div>
    </section>
  );
}
