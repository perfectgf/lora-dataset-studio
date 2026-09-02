import { checkpointActionModel } from './checkpointPopover.js';
import { Trash2 } from 'lucide-react';

/* ◉ THE checkpoint actions popover — one component, two surfaces.

   It used to be ~90 lines of JSX inlined in RunLineageGraph.jsx, which meant the
   LoRA Canvas had no actions at all: clicking a pill there could only tick it.
   Copying the block over would have produced two popovers that agree today and
   drift on the first change — the exact failure the card, the pill and the edges
   were extracted to avoid (lineageNodes.jsx / lineageEdges.jsx). So it moved
   here, whole, and both surfaces mount THIS one. A contract test fails if either
   grows its own.

   Presentational: every decision arrives from checkpointActionModel (pure,
   unit-tested) and every action arrives as a prop. It fetches nothing and knows
   nothing about which screen it is on — the host owns the positioning, because
   that is the one thing the two surfaces genuinely do differently.

   `pill` may be null. A click on a run CARD opens the same popover with only its
   run-level row (ⓘ Details), which is what took the detail drawer off the
   click: it now opens because it was asked for, not because a card was touched. */

const ROW = 'flex items-center gap-1.5 rounded-md border px-2 py-1 text-[0.6875rem] font-medium';
// Disabled rows are TEXT, not buttons: a greyed-out button invites the click it
// will not honour. This states the situation and gets out of the way.
const MUTED = 'rounded-md border border-border bg-app/40 px-2 py-1 text-content-subtle text-[0.625rem]';

export default function CheckpointActionsPopover({
  node, pill, runLabel = null,
  continueSource = 'cloud', continueReason = null, folderLabel = null,
  importing = false, deleting = false,
  onContinue, onDeploy, onDelete, onDetails, onPublish, onClose,
}) {
  const a = checkpointActionModel(node, pill, {
    continueSource, hasContinueHandler: typeof onContinue === 'function',
    continueReason, folderLabel,
  });
  if (!a) return null;

  return (
    <div className="lds-ck-popover rounded-lg border border-indigo-400/40 bg-surface-overlay p-2 shadow-xl"
      role="dialog"
      aria-label={a.isRun ? 'Run actions' : `Checkpoint step ${a.step} actions`}
      onPointerDown={(e) => e.stopPropagation()}>
      <div className="mb-1.5 flex items-center gap-1.5">
        <span className="min-w-0 truncate text-content text-[0.6875rem] font-semibold tabular-nums">
          {a.isRun ? (runLabel || 'This run') : `Step ${Number(a.step).toLocaleString()}`}
        </span>
        {a.final && (
          <span className="shrink-0 rounded bg-emerald-500/15 px-1 py-px text-emerald-200 text-[0.5rem] font-semibold uppercase">final</span>
        )}
        <button type="button" onClick={onClose}
          className="ml-auto shrink-0 text-content-subtle hover:text-content text-[0.75rem]"
          aria-label="Close">✕</button>
      </div>

      <div className="flex flex-col gap-1">
        {a.download && (a.download.url ? (
          <a href={a.download.url} download onClick={onClose}
            className={ROW + ' border-emerald-500/40 bg-emerald-600/15 text-emerald-100 no-underline hover:bg-emerald-600/25'}>
            <span aria-hidden>⬇</span> Download
          </a>
        ) : (
          <span className={MUTED}>{a.download.reason}</span>
        ))}

        {a.continue && (a.continue.ok ? (
          <button type="button" onClick={() => { onContinue(node, pill); onClose?.(); }}
            className={ROW + ' border-indigo-400/40 bg-indigo-500/15 text-indigo-100 hover:bg-indigo-500/25'}>
            <span aria-hidden>▶</span> Continue from here
          </button>
        ) : (
          <span className={MUTED}><span aria-hidden>▶</span> {a.continue.reason}</span>
        ))}

        {/* 📦 Deploy → loras/<family>: put this checkpoint in ComfyUI on the spot.
            A deployed one shows "✓ Deployed" with the SYMMETRIC ⏏ Undeploy right
            beside it — reversible, the training save stays — rather than leaving
            undeploy buried in a row that reads as destruction. */}
        {a.deployed ? (
          <div className="flex items-center gap-1">
            <span className={ROW + ' flex-1 border-emerald-500/40 bg-emerald-600/10 text-emerald-200'}>
              <span aria-hidden>✓</span> Deployed
            </span>
            {a.undeploy && (
              <button type="button" disabled={deleting}
                onClick={() => onDelete(node, pill)} title={a.undeploy.title}
                className={ROW + ' border-emerald-500/40 bg-emerald-600/5 text-emerald-200/90 hover:bg-emerald-600/20 disabled:cursor-not-allowed disabled:opacity-50'}>
                <span aria-hidden>⏏</span> {deleting ? 'Undeploying…' : a.undeploy.label}
              </button>
            )}
          </div>
        ) : a.deploy && (a.deploy.payload ? (
          <button type="button" disabled={importing}
            onClick={() => onDeploy(node, pill)}
            title={`Deploy this checkpoint into ComfyUI's ${a.deploy.folder} folder so you can test and generate with it`}
            className={ROW + ' border-primary/40 bg-primary/20 text-white hover:bg-primary/30 disabled:cursor-not-allowed disabled:opacity-50'}>
            <span aria-hidden>📦</span> {importing ? 'Deploying…' : `Deploy → ${a.deploy.folder}`}
          </button>
        ) : (
          <span className={MUTED}><span aria-hidden>📦</span> {a.deploy.reason}</span>
        ))}

        {/* 📤 Civitai — publish this checkpoint as a model page, or mark the
            page it already has so the viewer's "post this image" lands under
            it. The lineage payload stamps the link on the pill (`civitai`), so
            the row can say which page without a request. Host-gated like ⓘ:
            a host with no modal to open shows no row. */}
        {typeof onPublish === 'function' && !a.isRun && (
          <button type="button" onClick={() => { onPublish(node, pill); onClose?.(); }}
            data-testid="checkpoint-civitai"
            title={pill?.civitai
              ? `On Civitai: ${pill.civitai.model_name || `model ${pill.civitai.model_id}`}${pill.civitai.version_name ? ` · ${pill.civitai.version_name}` : ''} — open, relink, or post images from the viewer`
              : 'Publish this checkpoint on Civitai, or mark the model page it already has'}
            className={ROW + ' border-indigo-400/40 bg-indigo-500/15 text-indigo-100 hover:bg-indigo-500/25'}>
            <span aria-hidden>📤</span> {pill?.civitai ? 'On Civitai' : 'Civitai'}
          </button>
        )}

        {/* ⓘ The detail drawer — config, run note, checkpoint notes — now ASKED
            for. It used to spring open on any card click, which turned a glance
            at the board into a panel to dismiss. */}
        {typeof onDetails === 'function' && (
          <button type="button" onClick={() => { onDetails(node); onClose?.(); }}
            title="Open this run's configuration, notes and checkpoint notes"
            className={ROW + ' border-border bg-app/60 text-content hover:border-indigo-400/50'}>
            <span aria-hidden>ⓘ</span> Details
          </button>
        )}

        {/* 🗑 The one truly DESTRUCTIVE action — deleting the training save — kept
            visually in retreat below a hairline: a quiet text row, not a fourth
            coloured button one clicks by reflex. Its label names exactly the file
            the click deletes. */}
        {a.del && (
          <button type="button" disabled={deleting}
            onClick={() => onDelete(node, pill)} title={a.del.title}
            className="mt-1 flex items-center gap-1.5 border-t border-border px-2 pt-1.5 pb-0.5 text-left text-content-subtle text-[0.625rem] hover:text-rose-300 disabled:cursor-not-allowed disabled:opacity-50">
            <Trash2 aria-hidden="true" className="h-3.5 w-3.5" /> {deleting ? 'Deleting…' : a.del.label}
          </button>
        )}
      </div>
    </div>
  );
}
