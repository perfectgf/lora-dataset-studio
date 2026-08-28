import { useEffect, useMemo, useRef, useState } from 'react';
import { Camera, X } from 'lucide-react';
import { apiFetch } from '../../api/fetchClient';
import { requestHelpTip } from '../../help/helpTips';
import GlobalModelPicker from './GlobalModelPicker';
import {
  AZIMUTHS, CAMERA_INTRO, DISTANCES, DISTANCE_CAVEAT, ELEVATIONS,
  REFERENCE_POSE, costSentence, isLongRun, posePrompt, posesFor, selectionRefusal,
} from '../../utils/cameraAngles';

/* 📷 Pick where the camera stands.
 *
 * THE SCREEN'S ONE IDEA: you choose AXES, not pictures. Ninety-six poses exist;
 * ninety-six checkboxes would be a worse screen and a worse mental model,
 * because the thing people actually want is "all eight sides at eye level" —
 * one gesture on an axis, eight on a grid. The product of the axes is the run,
 * and it is shown as a number that moves while you choose, so the cost is
 * legible before it is spent rather than after.
 *
 * THE DIAL follows the model card's own diagram (0° front at the top, 90° right,
 * 180° back at the bottom) rather than a prettier convention of ours: someone
 * who read the LoRA's documentation must find the same map here. The centre is
 * the subject, and the notch on it points at 0° — which is where the camera
 * already is, because every angle this lane produces is relative to the picture
 * you started from, not to a compass.
 *
 * WHY THE PROMPT IS ON SCREEN. What leaves the app is a sentence in the LoRA's
 * grammar, and this panel is the only place a user can see it. It is shown, not
 * editable: the grammar is the model's, and a free-text field here would invite
 * exactly the phrasings that were measured NOT to move the camera.
 */

const DIAL = 220;                 // viewBox side
const C = DIAL / 2;
const RING = 84;                  // where the camera dots sit
const HIT = 22;                   // 44 px tap target — finger-sized on a phone

const pointAt = (degrees, radius = RING) => {
  const rad = ((degrees - 90) * Math.PI) / 180;
  return [C + radius * Math.cos(rad), C + radius * Math.sin(rad)];
};

function toggle(list, id) {
  return list.includes(id) ? list.filter((x) => x !== id) : [...list, id];
}

/** The azimuth dial. A `<g>` per position: an invisible finger-sized disc, the
 *  dot itself, and a title so a pointer user gets the name without a legend. */
function AzimuthDial({ picked, onToggle }) {
  return (
    <svg viewBox={`0 0 ${DIAL} ${DIAL}`} className="mx-auto block w-full max-w-[15rem]"
      role="group" aria-label="Camera position around the subject">
      <circle cx={C} cy={C} r={RING} fill="none" stroke="currentColor"
        className="text-white/10" strokeWidth="1" />
      <circle cx={C} cy={C} r={RING - 34} fill="none" stroke="currentColor"
        className="text-white/[0.07]" strokeWidth="1" strokeDasharray="2 5" />

      {/* The subject. The notch points at 0° — the way the reference photo
          already sees them. */}
      <circle cx={C} cy={C} r="15" className="fill-white/10" />
      <path d={`M ${C} ${C - 15} l 5 8 h -10 z`} className="fill-white/45" />

      {AZIMUTHS.map((a) => {
        const [x, y] = pointAt(a.degrees);
        const on = picked.includes(a.id);
        const isRef = a.id === REFERENCE_POSE.split('/')[0];
        return (
          <g key={a.id} onClick={() => onToggle(a.id)} className="cursor-pointer"
            role="button" tabIndex={0} aria-pressed={on} aria-label={a.label}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onToggle(a.id); }
            }}>
            <title>{`${a.label} — ${a.degrees}°`}</title>
            <circle cx={x} cy={y} r={HIT} fill="transparent" />
            {on && (
              <line x1={C} y1={C} x2={x} y2={y} stroke="currentColor"
                className="text-indigo-400/40" strokeWidth="1.5" />
            )}
            <circle cx={x} cy={y} r="11"
              className={on
                ? 'fill-indigo-500 stroke-indigo-300'
                : 'fill-gray-800 stroke-white/25 hover:stroke-white/60'}
              strokeWidth="1.5" />
            {/* Dark glyph on the amber fill — white does not pass contrast on
                this accent, the rule the design system states outright. */}
            {on && <circle cx={x} cy={y} r="3.5" className="fill-gray-950" />}
            {isRef && !on && <circle cx={x} cy={y} r="3.5" className="fill-white/40" />}
          </g>
        );
      })}
    </svg>
  );
}

/**
 * `onShoot(poses)` queues the run and resolves when the request is done.
 * `onClose` dismisses. `modelResident` only sharpens the time estimate.
 *
 * Rendered as its own layer (`data-probe-layer`) so the responsive probe treats
 * it as a sheet that covers the page by design instead of budgeting it as fixed
 * chrome that ate the fold.
 */
export default function CameraAnglePicker({ onShoot, onClose, modelResident = false,
  busy = false }) {
  const [azimuths, setAzimuths] = useState(['front', 'right', 'back', 'left']);
  const [elevations, setElevations] = useState(['eye']);
  const [distances, setDistances] = useState(['medium']);
  const [sending, setSending] = useState(false);
  // {setting, effective, default} from the catalog, null until it answers. The
  // Model row edits the app-wide `camera.unet` — the same key Settings shows —
  // and this panel is where the choice is made because it is where the cost of
  // a wrong build is felt. Fetched here rather than passed in: the picker
  // mounts in two hosts (Gallery, dataset lightbox) and neither owns config.
  const [unet, setUnet] = useState(null);
  const closeRef = useRef(null);

  // Once, the first time this panel is ever opened: the axes-not-pictures idea
  // is the one thing that is not self-evident from looking at it.
  useEffect(() => { requestHelpTip('camera-angles-picker'); }, []);

  // Focus moves INTO the dialog on open (the close button, like the dataset
  // lightbox does). Without it, focus stays on the 📷 button underneath — and
  // in the Gallery that host walks ← → on window, so the reference picture
  // would flip behind the dial while the run still shoots the original.
  useEffect(() => { closeRef.current?.focus(); }, []);

  useEffect(() => {
    let alive = true;
    apiFetch('/api/camera/catalog')
      .then((d) => { if (alive && d?.unet) setUnet(d.unet); })
      .catch(() => {});           // no row is better than a row that lies
    return () => { alive = false; };
  }, []);

  const selection = { azimuths, elevations, distances };
  const poses = useMemo(() => posesFor(selection),
    [azimuths, elevations, distances]);   // eslint-disable-line react-hooks/exhaustive-deps
  const refusal = selectionRefusal(selection);
  const long = isLongRun(poses.length, { modelResident });
  const active = sending || busy;

  const run = async () => {
    if (refusal || active) return;
    setSending(true);
    try {
      await onShoot(poses);
    } finally {
      setSending(false);
    }
  };

  const axisButton = (on, label, hint, onClick, key) => (
    <button key={key} type="button" onClick={onClick} aria-pressed={on}
      className={`min-h-10 lg:min-h-0 flex-1 rounded-lg border px-3 py-2 text-left transition-colors ${on
        ? 'border-indigo-400/70 bg-indigo-500/25 text-indigo-100'
        : 'border-white/10 bg-white/[0.03] text-gray-300 hover:border-white/25'}`}>
      <span className="block text-[0.8rem] font-semibold leading-tight">{label}</span>
      {hint && <span className="block text-[0.68rem] text-gray-400">{hint}</span>}
    </button>
  );

  return (
    <div data-probe-layer className="fixed inset-0 z-[9998] flex items-center justify-center bg-black/80 p-3 sm:p-6"
      role="dialog" aria-modal="true" aria-label="Choose camera positions"
      /* stopPropagation ALWAYS, then the backdrop test. This picker mounts in
         two hosts, and one of them (the dataset lightbox) closes ITSELF on any
         click that reaches its root — without the stop, every tap on a dial
         dot bubbled up and quit the whole viewer, picker included. Found on a
         phone, not by the probe: the probe opens the picker on the Gallery,
         where the picker is a sibling and nothing bubbles. */
      onClick={(e) => {
        e.stopPropagation();
        if (e.target === e.currentTarget) onClose?.();
      }}
      /* Keys pressed INSIDE the picker are the picker's: the Model combobox is
         a text field, and without the stop, ← → while fixing a typed path walk
         the Gallery underneath, and any letter is a verdict in the dataset
         host. Escape here is the one-layer rule applied to THIS layer — the
         combobox already stops the press that only closes its dropdown. Keys
         pressed with focus OUTSIDE this tree still reach the hosts' window
         listeners, which keep their own cameraOpen branches for exactly that. */
      onKeyDown={(e) => {
        e.stopPropagation();
        if (e.key === 'Escape') onClose?.();
      }}>
      {/* `bg-surface-overlay`, not `bg-surface`: the latter is 4 % alpha and the
          page would read through the card. The contract test enforces it. */}
      <div className="flex max-h-full w-full max-w-3xl flex-col overflow-hidden rounded-xl border border-white/10 bg-surface-overlay shadow-2xl">

        <header className="flex items-start gap-3 border-b border-white/10 px-4 py-3">
          <Camera className="mt-0.5 size-5 shrink-0 text-indigo-300" aria-hidden />
          <div className="min-w-0 flex-1">
            <h2 className="font-sans text-base font-semibold text-gray-100">Camera angles</h2>
            <p className="mt-0.5 text-[0.78rem] leading-snug text-gray-400">{CAMERA_INTRO}</p>
          </div>
          <button type="button" onClick={onClose} aria-label="Close" ref={closeRef}
            className="min-h-10 lg:min-h-0 -mr-1 rounded-lg px-2 text-gray-400 hover:bg-white/5 hover:text-gray-200">
            <X className="size-4" aria-hidden />
          </button>
        </header>

        <div className="grid min-h-0 flex-1 gap-5 overflow-y-auto p-4 sm:grid-cols-[minmax(0,15rem)_minmax(0,1fr)]">
          <section data-probe-panel="camera-dial" className="min-w-0">
            <h3 className="mb-2 font-mono text-[0.66rem] uppercase tracking-[0.14em] text-gray-400">
              Around the subject
            </h3>
            <AzimuthDial picked={azimuths} onToggle={(id) => setAzimuths((v) => toggle(v, id))} />
            <div className="mt-2 flex gap-2">
              <button type="button"
                onClick={() => setAzimuths(AZIMUTHS.map((a) => a.id))}
                className="min-h-10 lg:min-h-0 flex-1 rounded-lg border border-white/10 px-2 py-1.5 text-[0.72rem] text-gray-300 hover:border-white/25">
                All sides
              </button>
              <button type="button" onClick={() => setAzimuths([])}
                className="min-h-10 lg:min-h-0 flex-1 rounded-lg border border-white/10 px-2 py-1.5 text-[0.72rem] text-gray-300 hover:border-white/25">
                Clear
              </button>
            </div>
          </section>

          <section data-probe-panel="camera-axes" className="min-w-0 space-y-4">
            <div>
              <h3 className="mb-2 font-mono text-[0.66rem] uppercase tracking-[0.14em] text-gray-400">
                Camera height
              </h3>
              <div className="flex flex-wrap gap-2">
                {ELEVATIONS.map((e) => axisButton(
                  elevations.includes(e.id), e.label, e.hint,
                  () => setElevations((v) => toggle(v, e.id)), e.id))}
              </div>
            </div>

            <div>
              <h3 className="mb-2 font-mono text-[0.66rem] uppercase tracking-[0.14em] text-gray-400">
                Distance
              </h3>
              <div className="flex flex-wrap gap-2">
                {DISTANCES.map((d) => axisButton(
                  distances.includes(d.id), d.label, null,
                  () => setDistances((v) => toggle(v, d.id)), d.id))}
              </div>
              <p className="mt-1.5 text-[0.68rem] leading-snug text-gray-500">{DISTANCE_CAVEAT}</p>
            </div>

            {/* Which Qwen-Image-Edit build renders the views. Only shown once
                the catalog has answered: a combobox that snaps from empty to a
                value a beat later reads as the app changing its mind. */}
            {unet && (
              <div>
                <h3 className="mb-2 font-mono text-[0.66rem] uppercase tracking-[0.14em] text-gray-400">
                  Model
                </h3>
                <GlobalModelPicker section="camera" field="unet" slot="camera_unet"
                  label="camera model" value={unet.setting}
                  onSaved={(next) => setUnet((u) => ({ ...u, setting: next }))} />
                <p className="mt-1.5 break-words text-[0.68rem] leading-snug text-gray-500">
                  App-wide — every 📷 run uses it; empty = the installed{' '}
                  {unet.default}. Another Qwen-Image-Edit build (a finetune, an
                  NSFW merge) keeps the angle grammar and changes the look.
                </p>
                {/* Said HERE, before the run, because the skip is a name-based
                    guess: chaining the speed LoRA onto an already-distilled
                    merge renders confetti over every textured surface while
                    reporting success (measured, Rapid AIO v23, same seed). */}
                {unet.distilled && (
                  <p className="mt-1 break-words text-[0.68rem] leading-snug text-amber-300/80">
                    This build reads as already distilled — runs skip the extra
                    speed LoRA and keep 4 steps. Pin a speed LoRA in Settings to
                    override.
                  </p>
                )}
              </div>
            )}

            <div>
              <h3 className="mb-2 font-mono text-[0.66rem] uppercase tracking-[0.14em] text-gray-400">
                What gets sent
              </h3>
              {/* Every prompt, scrollable. It used to stop at the old cap and
                  say "narrow the selection", which was the cap talking; the list
                  is a preview of what leaves the app, and truncating it hid
                  exactly the poses a long run most needs to be checked on. */}
              <ul className="max-h-28 space-y-0.5 overflow-y-auto rounded-lg border border-white/10 bg-black/30 px-3 py-2 font-mono text-[0.68rem] leading-relaxed text-gray-400">
                {poses.length === 0 && <li className="text-gray-600">nothing picked yet</li>}
                {poses.map((p) => {
                  const [a, e, d] = p.split('/');
                  return <li key={p} className="truncate">{posePrompt(a, e, d)}</li>;
                })}
              </ul>
            </div>
          </section>
        </div>

        <footer className="flex flex-wrap items-center gap-3 border-t border-white/10 px-4 py-3">
          {/* The cost, never a barrier. Nothing here refuses a long run — it
              says what it will take, and turns amber past five minutes so the
              number is read rather than skimmed. Each queued view can be
              dropped one at a time from the system queue, which is what makes
              a long run a decision instead of a one-way door. */}
          <p className={`min-w-0 flex-1 text-[0.78rem] ${long ? 'text-amber-300' : 'text-gray-400'}`}>
            {refusal
              ? refusal.charAt(0).toUpperCase() + refusal.slice(1)
              : costSentence(poses.length, { modelResident })}
            {long && !refusal && (
              <span className="block text-[0.7rem] text-amber-300/70">
                Long run — you can drop queued views from the system queue.
              </span>
            )}
          </p>
          <button type="button" onClick={onClose}
            className="min-h-10 lg:min-h-0 rounded-lg border border-white/10 px-3 py-1.5 text-[0.78rem] text-gray-300 hover:border-white/25">
            Cancel
          </button>
          <button type="button" onClick={run} disabled={!!refusal || active}
            aria-busy={active}
            className="min-h-10 lg:min-h-0 rounded-lg bg-gradient-primary px-4 py-1.5 text-[0.8rem] font-semibold text-gray-950 disabled:cursor-not-allowed disabled:opacity-40">
            {active ? 'Queueing…' : `Shoot ${poses.length || ''} view${poses.length === 1 ? '' : 's'}`.trim()}
          </button>
        </footer>
      </div>
    </div>
  );
}
