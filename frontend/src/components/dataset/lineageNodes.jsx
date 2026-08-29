import { CARD_H } from '../../utils/lineageGraph';
import { resumeCaption } from '../../utils/lineageTree';
import { famLabel, ModeChip, StatusDot, SavesChip } from './lineageChrome';
import { trainingRunVariantLabel } from '../../utils/trainingRuns';
import { runNumber, runIdentityLabel } from '../../utils/runIdentity';
import {
  DEPLOY_BAR_CLASS, deployState, deployTitleSuffix,
} from '../../utils/checkpointDeployState';
import { pillSelectScale } from '../../utils/canvasNodeChrome';
import { datasetThumbUrl } from '../../utils/datasetThumbUrl';

/* ◉ The two things a lineage is DRAWN with: a run card and a checkpoint pill.

   They used to live inside RunLineageGraph.jsx, which also owns that graph's
   whole interaction model (popover, inline generation, compare selection, pan).
   The LoRA Canvas draws the same runs on a different surface — several datasets
   at once, its own zoom and pan — so the drawing had to stop being a private
   detail of one screen. Extracted verbatim: both surfaces render the SAME card
   and the SAME pill, which is the point. A visual tweak lands on both, and there
   is no second rendering to drift.

   Presentational only — every action arrives as a prop. Neither component
   fetches, neither knows which screen it is on. */

/** The run's variant as the app NAMES it, not the raw stored value: Krea's raw
 *  recipe is stored 'base' and shown "Raw" everywhere else, so a card reading
 *  "Krea 2 · base" next to a dialog reading "Krea 2 · Raw" looked like two
 *  different runs. One label helper for both. */
export const variantLabel = (node) => trainingRunVariantLabel(node.train_type, node.variant);

/** One run as a fixed-size card. Mirrors the list card's content, sized to the
 *  graph's card box; sits at the top of the run's cell (pills go below). */
export function GraphCard({ node, lit, annotated, compareRole, onSelect }) {
  const cur = node.is_current;
  const dim = node.checkpoint_ready === false;
  const clickable = typeof onSelect === 'function';
  return (
    <div
      role={clickable ? 'button' : undefined}
      tabIndex={clickable ? 0 : undefined}
      onClick={clickable ? (e) => onSelect(node, e) : undefined}
      onKeyDown={clickable ? (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(node, e); } } : undefined}
      title={clickable ? 'Click to inspect · Shift-click to compare' : undefined}
      style={{ height: CARD_H }}
      className={'lds-gcard flex w-full flex-col justify-center gap-1 rounded-xl border px-2.5 py-1.5 '
        + (cur
          ? 'lds-gcard-current border-indigo-400/70 bg-indigo-500/10 ring-1 ring-indigo-400/30 '
          : dim
            ? 'border-border bg-app/40 '
            : 'border-border bg-surface-raised ')
        + (lit && !cur ? 'ring-1 ring-indigo-300/40 border-indigo-400/50 ' : '')
        + (compareRole ? 'ring-2 ring-amber-400/70 border-amber-400/60 ' : '')
        + (clickable ? 'cursor-pointer' : '')}>
      <div className="flex min-w-0 items-center gap-1.5">
        <StatusDot status={node.status} />
        {/* THE run number = the record id, the same one the inspector, the
            chips and the Runs history show — ONE number per run, everywhere.
            The cloud id is context, not identity: it stays in the title (and
            the inspector), never printed beside the number where it read as a
            second id. Matching a card to a Runs row is the deep-link's job. */}
        <span className="shrink-0 font-mono text-content-muted text-[0.625rem]"
          title={runIdentityLabel(node)}>
          <span aria-hidden>{node.source === 'cloud' ? '☁' : '💻'}</span>{' '}
          {runNumber(node)}
        </span>
        <span className={`min-w-0 truncate text-[0.75rem] font-semibold ${dim ? 'text-content-muted' : 'text-content'}`}
          title={`${famLabel(node.train_type)}${variantLabel(node) ? ` · ${variantLabel(node)}` : ''}`}>
          {famLabel(node.train_type)}{variantLabel(node)
            ? <span className="font-normal text-content-muted"> · {variantLabel(node)}</span> : null}
        </span>
        {cur && (
          <span className="shrink-0 rounded-full bg-indigo-500/25 px-1.5 py-0.5 text-indigo-100 text-[0.5rem] font-bold uppercase tracking-wider">
            this run
          </span>
        )}
        {annotated && (
          <span aria-hidden title="Has notes" className="shrink-0 text-amber-300 text-[0.625rem] leading-none">●</span>
        )}
        {compareRole && (
          <span title={`Selected for compare (${compareRole})`}
            className="shrink-0 rounded-full bg-amber-500/25 px-1.5 py-0.5 text-amber-100 text-[0.5rem] font-bold uppercase tracking-wider">
            {compareRole}
          </span>
        )}
        <span className="ml-auto shrink-0"><SavesChip node={node} /></span>
      </div>
      <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-0.5 text-content-subtle text-[0.5625rem]">
        <ModeChip node={node} />
        {node.version != null && (
          <span className="rounded bg-app/60 px-1 py-px font-medium text-content-muted">v{node.version}</span>
        )}
        {node.steps ? <span className="tabular-nums">{node.steps.toLocaleString()} steps</span> : null}
        {resumeCaption(node) && (
          <span className="inline-flex items-center gap-0.5">
            <span aria-hidden className="text-[0.625rem] leading-none">↳</span>{resumeCaption(node)}
          </span>
        )}
        {node.origin_unknown && (
          <span className="italic" title="This run resumed from an earlier checkpoint, but its source run predates lineage tracking">
            origin not recorded
          </span>
        )}
      </div>
    </div>
  );
}

/** One checkpoint as a compact pill: its step, a ✓ for the final save, an indigo
 *  ring when it's the point another run branched off, a select checkbox for the
 *  shared-prompt batch, and — when that checkpoint has produced images — a
 *  RESULTS chip carrying how many. Clicking the body opens the pill's actions;
 *  the checkbox toggles it into the batch; the chip opens the gallery.
 *  Absolutely positioned at the exact box the layout computed.
 *
 *  ⚠️ The compact pill deliberately shows NO thumbnail. It used to: a 14-px
 *  square of the generated image, on a 60×20 pill. At that size an image is not
 *  an image, it is a coloured smudge — it says nothing about the framing, the
 *  outfit or whether the face holds, while eating the width the step label
 *  needs. Its job here is to signal that results EXIST and how many; looking at
 *  them belongs to the gallery, where they are big enough to be judged (and to
 *  the 🔍 Big-previews tile, which is sized for exactly that).
 *
 *  ⚠️ The chip also sits INSIDE the pill. The old count badge hung off the
 *  bottom-right corner at -6 px, so on a row of pills 6 px apart two neighbours'
 *  badges overlapped each other — visible at 100 % zoom, not an edge case.
 *
 *  `onOpen(pill, event)` carries the click event: a host that floats the actions
 *  popover in SCREEN space (the canvas, whose board is zoomed and panned) needs
 *  the point that was clicked to place it. The host that draws it in its own
 *  <svg> ignores the second argument. */
export function CheckpointPill({ pill, offX, offY, active, selected, preview, big, onOpen, onToggleSelect, onZoomPreview, onOpenGallery, selectable = null, boardScale = null }) {
  /* ✓ How big to draw the pick box. `boardScale` is the LoRA Canvas' zoom; the
     in-card graph is not zoomable and passes nothing, which resolves to 1 and
     leaves that graph — and any board at 100 % or more — exactly as it was. See
     utils/canvasNodeChrome.pillSelectScale for the two bounds and for what this
     deliberately does not fix. */
  const selScale = boardScale ? pillSelectScale(boardScale, pill.w) : 1;
  const gone = pill.present === false;
  /* Can I generate from this checkpoint RIGHT NOW? The pill has always known
     (`testable`) and never said so: the answer only surfaced as the words
     "to deploy" inside the generation panel, i.e. AFTER the checkpoints were
     already picked. It is drawn as a bar down the pill's left edge -- solid
     sky = deployed, dashed slate = on disk only -- a channel none of the
     pill's other meanings use (emerald = final save, indigo ring/box =
     picked or resumed-from, dashed shell = the file is gone). See
     utils/checkpointDeployState for why the shape doubles the colour. */
  const deployCls = `${DEPLOY_BAR_CLASS[deployState(pill)] || ''} `;
  const st = preview?.status || null;
  const label = pill.step >= 1000 && pill.step % 1000 === 0 ? `${pill.step / 1000}k` : pill.step;
  // How many images this checkpoint has produced, from ANY surface (inline
  // preview, Test Studio, comparison run) — they accumulate. A lineage that only
  // knows about the one inline preview still counts as one result.
  const count = Number(preview?.count) || (preview?.url ? 1 : 0);
  const zoom = (e) => { e.stopPropagation(); onZoomPreview?.(preview.url, pill.step); };
  const zoomKey = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); zoom(e); } };
  // The chip's click: the gallery when the host has one (every image, large),
  // otherwise the lightbox on the one preview we hold. Never nothing.
  const openResults = (e) => {
    e.stopPropagation();
    if (typeof onOpenGallery === 'function') onOpenGallery(pill);
    else if (preview?.url) onZoomPreview?.(preview.url, pill.step);
  };
  const resultsTitle = `${count} image${count > 1 ? 's' : ''} generated from this checkpoint — click to open ${count > 1 ? 'them' : 'it'}`;
  const canOpenResults = typeof onOpenGallery === 'function'
    || (!!preview?.url && typeof onZoomPreview === 'function');
  const shellCls = 'lds-ckpill rounded-md border transition-colors '
    + (gone
      ? 'border-dashed border-border bg-transparent text-content-subtle '
      : pill.final
        ? 'border-emerald-400/50 bg-emerald-500/10 text-emerald-200 '
        : 'border-border bg-app/70 text-content-muted hover:border-indigo-400/50 hover:text-content ')
    + (pill.isResumeSource ? 'ring-1 ring-indigo-400/60 border-indigo-400/60 ' : '')
    + (selected ? 'ring-2 ring-indigo-400/80 border-indigo-400/70 ' : active ? 'ring-2 ring-indigo-400/80 ' : '');
  const openTitle = `Checkpoint at step ${pill.step}${pill.final ? ' — final' : ''}${pill.isResumeSource ? ' — a run continued from here' : ''}${count ? ` — ${count} image${count > 1 ? 's' : ''}` : ''}${st === 'pending' ? ' — an image is rendering' : ''}${deployTitleSuffix(pill)}`;
  const resultsKey = (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openResults(e); }
  };
  // The results chip, shared by both pill shapes. `inline` is the compact row's
  // variant; the big tile overlays it on the image instead.
  //
  // ⚠️ role="button" on a <span>, NOT a real <button>: the chip lives INSIDE the
  // pill's own button (that is what stops it colliding with the neighbouring
  // pill), and a button nested in a button is invalid HTML — React says so and
  // browsers resolve it however they like. Same trick the preview thumbnail
  // already used for its zoom click.
  const resultsChip = (inline) => (
    canOpenResults ? (
      <span role="button" tabIndex={0} onClick={openResults} onKeyDown={resultsKey}
        aria-label={`Open the ${count} image${count > 1 ? 's' : ''} of step ${pill.step}`}
        title={resultsTitle}
        className={'lds-ckcount flex shrink-0 cursor-pointer items-center gap-px rounded-full border border-indigo-400/70 bg-indigo-500/25 font-semibold leading-none tabular-nums text-indigo-100 hover:bg-indigo-500 hover:text-gray-950 '
          + (inline ? 'ml-0.5 h-3.5 px-1 text-[0.5rem] ' : 'h-4 px-1 text-[0.5625rem] shadow-sm ')}>
        <span aria-hidden>🖼</span>{count}
      </span>
    ) : (
      <span title={resultsTitle}
        className={'lds-ckcount flex shrink-0 items-center gap-px rounded-full border border-border bg-surface-overlay font-semibold leading-none tabular-nums text-content-muted '
          + (inline ? 'ml-0.5 h-3.5 px-1 text-[0.5rem] ' : 'h-4 px-1 text-[0.5625rem] ')}>
        <span aria-hidden>🖼</span>{count}
      </span>
    )
  );
  return (
    <div style={{ position: 'absolute', left: offX, top: offY, width: pill.w, height: pill.h }}
      className="lds-ckpill-wrap">
      {big ? (
        // 🔍 Big-preview tile: the generated image at a size where it can actually
        // be JUDGED (ComfyUI-style — click it to view full-screen), with a step
        // label strip underneath that opens the pill's actions. This is where an
        // image belongs on the board; the compact pill only counts them.
        <button type="button"
          onClick={(e) => { e.stopPropagation(); onOpen(pill, e); }}
          title={openTitle}
          style={{ width: pill.w, height: pill.h }}
          className={shellCls + deployCls + ' flex w-full flex-col overflow-hidden text-[0.625rem] font-medium tabular-nums'}>
          <div className="relative min-h-0 flex-1 w-full">
            {/* The tile is 128×132 board units (PILL_W_BIG), so 320 stays crisp
                at 2× DPR and at the zoom levels the board actually reads at.
                `zoom` opens the FULL image — this tag is the tile only. */}
            {preview?.url ? (
              <img src={datasetThumbUrl(preview.url, 320)} alt={`Preview at step ${pill.step}`}
                role="button" tabIndex={0} title="Click to view this preview full-screen"
                onClick={zoom} onKeyDown={zoomKey} loading="lazy" decoding="async"
                className="h-full w-full cursor-zoom-in object-cover hover:opacity-90" />
            ) : (
              <div className="flex h-full w-full items-center justify-center text-base">
                {st === 'pending' ? <span aria-hidden title="Generating preview…" className="animate-pulse text-indigo-300">◌</span>
                  : st === 'failed' ? <span aria-hidden title="Preview failed" className="text-amber-300">⚠</span>
                  : <span aria-hidden className="opacity-50">💾</span>}
              </div>
            )}
            {/* Inside the tile, not hanging off its corner: two neighbouring
                badges used to overlap each other between adjacent pills. */}
            {count > 1 && (
              <span style={{ position: 'absolute', right: 3, top: 3 }}>{resultsChip(false)}</span>
            )}
          </div>
          <span className="flex shrink-0 items-center justify-center gap-0.5 border-t border-border bg-black/20 py-0.5 leading-none">
            {pill.final && <span aria-hidden className="text-emerald-300">✓</span>}
            <span>{label}</span>
          </span>
        </button>
      ) : (
        <button type="button"
          onClick={(e) => { e.stopPropagation(); onOpen(pill, e); }}
          title={openTitle}
          style={{ width: pill.w, height: pill.h }}
          className={shellCls + deployCls + ' flex w-full items-center justify-center gap-0.5 overflow-hidden px-0.5 text-[0.5625rem] font-medium tabular-nums'}>
          {pill.final && <span aria-hidden className="shrink-0 text-emerald-300">✓</span>}
          <span className="min-w-0 truncate">{label}</span>
          {count > 0 ? resultsChip(true)
            : st === 'pending' ? (
              <span aria-hidden title="Generating an image…" className="ml-0.5 shrink-0 animate-pulse text-indigo-300">◌</span>
            ) : st === 'failed' ? (
              <span aria-hidden title="The generation failed" className="ml-0.5 shrink-0 text-amber-300">⚠</span>
            ) : (
              <span aria-hidden className="shrink-0 opacity-70">💾</span>
            )}
        </button>
      )}
      {/* Select for the shared-prompt batch. A corner box; clicking it never opens
          the popover. Slightly larger in big mode so it stays clickable on the tile.
          `selectable` defaults to "deployed only" — the in-card graph can render
          nothing else. The canvas passes a wider rule: it offers to DEPLOY the
          picks that are not in ComfyUI yet, so a not-yet-deployed checkpoint has
          to be pickable there. */}
      {(selectable ?? pill.testable) && typeof onToggleSelect === 'function' && (
        <button type="button" role="checkbox" aria-checked={selected}
          aria-label={`Select step ${pill.step} for preview`}
          title={selected ? 'Selected for preview' : 'Select for preview'}
          onClick={(e) => { e.stopPropagation(); onToggleSelect(pill); }}
          // ⚠️ The box hangs into the 6-px gap and no further: at 16 px it used
          // to cover the first 10 px of its OWN pill, which on a 60-px pill hid
          // the leading digit of the step ("2500" read "500"). 12 px overhangs
          // the gap exactly, so it clips a 6×6 corner and nothing legible.
          // …and it GROWS UP-LEFT as the board zooms out (transform-origin at its
          // bottom-right), never down-right onto the pill: the whole point of the
          // -6/-6 offset is that the box lives in the gap, and a counter-scale
          // anchored the other way would put it straight back over the digits.
          style={{ position: 'absolute', left: -6, top: -6,
            ...(selScale > 1
              ? { transform: `scale(${selScale})`, transformOrigin: '100% 100%' }
              : null) }}
          className={'lds-cksel flex items-center justify-center rounded-[3px] border leading-none shadow-sm '
            + (big ? 'h-5 w-5 text-[0.6875rem] ' : 'h-3 w-3 text-[0.5rem] ')
            + (selected ? 'border-indigo-400 bg-indigo-500 text-gray-950 ' : 'border-border-strong bg-surface-overlay text-transparent hover:border-indigo-400 ')}>
          ✓
        </button>
      )}
    </div>
  );
}
