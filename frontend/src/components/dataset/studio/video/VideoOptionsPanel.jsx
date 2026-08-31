/**
 * The four options that change what the model computes, plus the three dials
 * that change what it is asked for.
 *
 * Each option says what it COSTS as well as what it buys, because none of them
 * is free and the price is invisible in the output: sparse attention trades
 * prompt adherence for speed, the third-party base imposes its own faces, and
 * the turbo LoRA is a distillation — a different model, not a faster one.
 *
 * Nothing here is a number this file invented. The clip lengths and the fps come
 * from the shared target catalogue through `/options`, so a length offered here
 * is a length the VAE accepts.
 */
import { Sparkles, Flame, Zap, Maximize2 } from 'lucide-react';
import { clipSeconds, SPARSE_CHOICES } from './videoStudioApi';

function Toggle({ checked, onChange, icon: Icon, label, hint, disabled, disabledHint }) {
  return (
    <label className={`flex items-start gap-2 rounded-lg border px-2 py-1.5 min-h-10 lg:min-h-0 ${
      disabled ? 'border-border opacity-60' : 'cursor-pointer border-border hover:border-accent/50'}`}>
      <input type="checkbox" checked={!!checked} disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5 h-4 w-4 shrink-0 accent-accent" />
      <span className="min-w-0">
        <span className="flex items-center gap-1 text-sm text-content">
          <Icon aria-hidden="true" className="h-3.5 w-3.5" />{label}
        </span>
        <span className="block text-[0.6875rem] text-content-subtle">
          {disabled ? disabledHint : hint}
        </span>
      </span>
    </label>
  );
}

/* Labels carry no emoji: each toggle already renders its lucide glyph, and a
   screenshot of the first build showed both at once on every row. The emoji
   survive where there is no icon — the clip summary lines under the history. */
export default function VideoOptionsPanel({ options, value, onChange }) {
  const set = (patch) => onChange({ ...value, ...patch });
  /* What this ComfyUI can actually run. `available === false` is a verdict (the
     pack is absent); `null` or missing is "could not ask", and an option is
     offered as usual there — a probe that did not run must not read as a no.
     The sentence names the SETUP screen rather than a GitHub URL: the app
     installs these three packs itself. */
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

  return (
    <section data-probe-panel="video-studio-options"
      className="flex flex-col gap-2 rounded-xl border border-border bg-surface p-2">
      <h2 className="text-sm font-semibold text-content">Options</h2>

      <div className="grid gap-1.5 sm:grid-cols-2">
        <Toggle checked={value.turbo && !off('turbo')} onChange={(v) => set({ turbo: v })}
          icon={Zap} label={`Turbo (${options?.turbo_steps || 6}-step)`}
          disabled={off('turbo')} disabledHint={need('turbo')}
          hint="A distillation LoRA and its double-clock sampler: minutes instead of tens of minutes. A different model, not just a faster one." />
        <Toggle checked={value.eros} onChange={(v) => set({ eros: v })}
          icon={Flame} label="10Eros base"
          disabled={options && !options.eros_available}
          disabledHint="Not on this machine — the official base is used."
          hint="A third-party finetune in place of the official base. It brings its own faces, so it works against an identity test." />
        <Toggle checked={value.latentUpscale && !off('latent_upscale')}
          onChange={(v) => set({ latentUpscale: v })}
          icon={Maximize2} label="Latent upscale ×2"
          disabled={off('latent_upscale')} disabledHint={need('latent_upscale')}
          hint="Enlarges before anything is decoded, so the audio passes through untouched. This is the pass that costs the minutes." />
        <label className={`flex flex-col gap-1 rounded-lg border border-border px-2 py-1.5 ${
          off('sparse') ? 'opacity-60' : ''}`}>
          <span className="flex items-center gap-1 text-sm text-content">
            <Sparkles aria-hidden="true" className="h-3.5 w-3.5" />Sparse attention
          </span>
          <select value={off('sparse') ? '' : value.sparse} disabled={off('sparse')}
            onChange={(e) => set({ sparse: e.target.value })}
            className="w-full rounded-md border border-border bg-app px-2 py-1 text-xs text-content min-h-10 lg:min-h-0">
            {SPARSE_CHOICES.map((c) => (
              <option key={c.value || 'off'} value={c.value}>{c.label}</option>
            ))}
          </select>
          <span className="text-[0.6875rem] text-content-subtle">
            {off('sparse') ? need('sparse') : SPARSE_CHOICES.find((c) => c.value === value.sparse)?.hint}
          </span>
        </label>
      </div>

      {value.sparse && value.sparse !== 'max' && value.latentUpscale && (
        <p className="rounded-lg border border-border bg-app px-2 py-1 text-[0.6875rem] text-content-muted">
          With the upscale on, the first pass stays dense and only the upscale
          samples sparse — the prompt keeps its attention where it decides the
          composition. Pick <strong>Max</strong> to accelerate both.
        </p>
      )}

      <div className="grid gap-1.5 sm:grid-cols-3">
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
          Resolution · {Number(value.megapixels).toFixed(2)} MP
          <input type="range" min={mp.min} max={mp.max} step="0.05"
            value={value.megapixels}
            onChange={(e) => set({ megapixels: Number(e.target.value) })}
            className="mt-2 accent-accent" />
        </label>
        <label className="flex flex-col gap-1 text-xs text-content-muted">
          Seed
          <input type="number" value={value.seed} placeholder="random"
            onChange={(e) => set({ seed: e.target.value })}
            className="rounded-lg border border-border bg-app px-2 py-1.5 text-content min-h-10 lg:min-h-0" />
        </label>
      </div>
      <p className="text-[0.6875rem] text-content-subtle">
        {seconds ? `${value.frames} frames at ${fps} fps — a ${seconds}s clip. ` : ''}
        Faces sharpen up to about 1 MP; past that the machine that runs the job
        decides whether it fits.
      </p>
    </section>
  );
}
