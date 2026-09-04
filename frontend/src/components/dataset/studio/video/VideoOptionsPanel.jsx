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
import SliderLock, { useSliderLock } from '../../../shared/SliderLock';
import { ACCELERATIONS, clipSeconds, SPARSE_CHOICES, studioFrameChoices } from './videoStudioApi';

function Toggle({ checked, onChange, icon: Icon, label, cost, hint, disabled, disabledHint }) {
  return (
    <label className={`flex items-start gap-2 rounded-lg border px-2.5 py-2 min-h-10 lg:min-h-0 ${
      disabled ? 'border-border opacity-60' : `cursor-pointer ${checked ? 'border-primary/60 bg-primary/5' : 'border-border hover:border-primary/50'}`}`}>
      <input type="checkbox" checked={!!checked} disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5 h-4 w-4 shrink-0 accent-primary" />
      <span className="min-w-0 flex-1">
        <span className="flex items-center gap-1.5 text-sm text-content">
          <Icon aria-hidden="true" className="h-3.5 w-3.5 text-content-muted" />
          <span className="min-w-0 flex-1 break-words">{label}</span>
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
  /* The STUDIO's ladder, not the training catalogue's: that one stops at 209
     frames (8.7 s) because that is where training lengths stop being useful,
     while this model renders to ~15 s and the server accepts it. */
  const frames = studioFrameChoices(options);
  const fps = options?.fps || 24;
  const mp = options?.megapixels || { min: 0.1, max: 2, default: 0.3 };
  /* 🔒 Locked by default, the same guard the image lane's dials wear. This
     panel is scrolled past on a phone with a thumb, and a range input takes
     the gesture that crosses it: the dial moves, nothing says so, and the next
     clip renders on a length nobody chose. Each keeps its own memory — the one
     you unlock is the one you are working on. */
  const stepsLock = useSliderLock('videoStudio.lock.steps');
  const lengthLock = useSliderLock('videoStudio.lock.length');
  const mpLock = useSliderLock('videoStudio.lock.megapixels');
  /* What "auto" resolves to, from the server's own constants rather than a
     second copy of them here: turbo grafts a distillation LoRA with its own
     six-step schedule, dense sampling runs twenty. An explicit count wins over
     both — including over turbo's — which is why the panel must show WHICH
     number is in force rather than implying the checkbox decides. */
  const autoSteps = value.accel
    ? (options?.turbo_steps || 6) : (options?.default_steps || 20);
  /* ⚡ The acceleration choices, resolved by the server against THIS machine
     (weight on disk, node pack for larryvrh's). Before the options arrive the
     static list shows the shape; nothing is disabled until the server says. */
  const accels = Array.isArray(options?.accelerations) && options.accelerations.length
    ? options.accelerations : ACCELERATIONS;
  const picked = accels.find((a) => a.id === value.accel) || null;
  const accelHint = !value.accel
    ? `The official base, dense: ${options?.default_steps || 20} steps, tens of minutes.`
    : picked?.available === false
      ? (picked.weight_present === false
        ? `Not on this machine — Setup downloads it (Video Test Studio › ${picked.label}).`
        : need('turbo'))
      : (picked?.hint || 'A distillation LoRA: six steps instead of twenty.');
  const steps = value.steps ? Number(value.steps) : autoSteps;
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
          {/* ⚡ One of the arena's top three, or the dense base. A select and
              not three checkboxes: exactly one can run, and the third choice
              would not fit a phone as a segmented row with its rank. */}
          <label data-testid="video-accel" className={`flex flex-col gap-1 rounded-lg border px-2.5 py-2 ${
            value.accel ? 'border-primary/60 bg-primary/5' : 'border-border'}`}>
            <span className="flex items-center gap-1.5 text-sm text-content">
              <Zap aria-hidden="true" className="h-3.5 w-3.5 text-content-muted" />
              <span className="min-w-0 flex-1">Acceleration, {options?.turbo_steps || 6} steps</span>
              {value.accel && (
                <span className="shrink-0 rounded-full border border-border px-1.5 py-px text-[0.625rem] text-content-subtle">
                  minutes, not tens
                </span>
              )}
            </span>
            <select value={value.accel || ''} onChange={(e) => set({ accel: e.target.value })}
              aria-label="Acceleration"
              className="w-full rounded-md border border-border bg-app px-2 py-1 text-xs text-content min-h-10 lg:min-h-0">
              <option value="">Off — dense base, {options?.default_steps || 20} steps</option>
              {accels.map((a) => (
                <option key={a.id} value={a.id} disabled={a.available === false}>
                  {a.label} · arena {a.arena}{a.available === false ? ' — not installed' : ''}
                </option>
              ))}
            </select>
            <span className="text-[0.6875rem] leading-snug text-content-subtle">{accelHint}</span>
          </label>
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
            off('sparse') ? 'border-border opacity-60' : value.sparse ? 'border-primary/60 bg-primary/5' : 'border-border'}`}>
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

      {/* Steps — the plainest time-for-fidelity dial there is, and the only
          one that was decided for you. It sits with the render options rather
          than with the shot because it is what turbo overrides, and turning it
          is how you find out whether turbo's six are enough for YOUR motion. */}
      <label className="flex flex-col gap-1 border-t border-border pt-3 text-xs text-content-muted">
        <span className="flex items-baseline justify-between gap-2">
          Sampling steps
          <span className="flex items-baseline gap-2">
            <span className="tabular-nums text-content">
              {value.steps ? steps : `auto · ${autoSteps}`}
            </span>
            {value.steps ? (
              <button type="button" onClick={() => set({ steps: '' })}
                className="rounded-md border border-border px-1.5 py-0.5 text-[0.625rem] text-content-muted hover:text-content">
                Auto
              </button>
            ) : null}
            {/* The padlock guards the TRACK, not this row: Auto is a small
                deliberate tap, never what a scrolling thumb lands on. */}
            <SliderLock locked={stepsLock.locked} onToggle={stepsLock.toggle}
              label="sampling steps" />
          </span>
        </span>
        <input type="range" min="4" max="40" step="1" value={steps}
          onChange={(e) => set({ steps: Number(e.target.value) })}
          {...stepsLock.rangeProps}
          className={`mt-1 accent-primary ${stepsLock.rangeProps.className}`} />
        <span className="text-[0.6875rem] leading-snug text-content-subtle">
          {value.accel
            ? 'The acceleration is trained for 6 — going far above it '
              + 'costs minutes without buying detail, and below 4 it ghosts on '
              + 'fast motion. An explicit count wins over its own.'
            : 'Dense sampling: more steps, more time, diminishing returns past '
              + 'about 30. This is the dial to move when a clip looks mushy '
              + 'rather than wrong.'}
        </span>
      </label>

      <div className="flex flex-col gap-1.5 border-t border-border pt-3">
        <h3 className="font-mono text-[0.625rem] uppercase tracking-[0.18em] text-content-subtle">Shot</h3>
        <div className="grid gap-2 sm:grid-cols-3 lg:grid-cols-1">
          {/* A SLIDER, not a 21-row dropdown. The legal lengths are a ladder
              (H3's VAE packs 17 frames per chunk, so every rung is ≡ 5 mod 17)
              and a ladder is what a discrete slider is for: it snaps, so an
              illegal count cannot be picked, and the whole range is legible
              without opening anything. The live value sits ABOVE the track —
              below it is where a finger lands (NN/g) — and reads in SECONDS
              first, because that is the unit the shot is thought in; the frame
              count follows as the technical truth. Same shape as Resolution
              underneath: one kind of dial, one look. */}
          <label className="flex flex-col gap-1 text-xs text-content-muted">
            <span className="flex items-baseline justify-between gap-2">
              Length
              <span className="flex items-center gap-2">
                <span className="tabular-nums text-content">
                  {seconds}s
                  <span className="ml-1 text-content-subtle">· {value.frames} frames</span>
                </span>
                <SliderLock locked={lengthLock.locked} onToggle={lengthLock.toggle}
                  label="clip length" />
              </span>
            </span>
            <input type="range" min="0" max={Math.max(0, frames.length - 1)} step="1"
              value={Math.max(0, frames.indexOf(value.frames))}
              onChange={(e) => set({ frames: frames[Number(e.target.value)] })}
              aria-label="Clip length"
              {...lengthLock.rangeProps}
              className={`mt-1 accent-primary ${lengthLock.rangeProps.className}`} />
            <span className="flex justify-between text-[0.625rem] tabular-nums text-content-subtle">
              <span>{clipSeconds(frames[0], fps)}s</span>
              <span>{clipSeconds(frames[frames.length - 1], fps)}s</span>
            </span>
          </label>
          <label className="flex flex-col gap-1 text-xs text-content-muted">
            <span className="flex items-baseline justify-between">
              Resolution
              <span className="flex items-center gap-2">
                <span className="tabular-nums text-content">{Number(value.megapixels).toFixed(2)} MP</span>
                <SliderLock locked={mpLock.locked} onToggle={mpLock.toggle}
                  label="resolution" />
              </span>
            </span>
            <input type="range" min={mp.min} max={mp.max} step="0.05"
              value={value.megapixels}
              onChange={(e) => set({ megapixels: Number(e.target.value) })}
              aria-label="Resolution"
              {...mpLock.rangeProps}
              className={`mt-1 accent-primary ${mpLock.rangeProps.className}`} />
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
