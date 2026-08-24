import { Link } from 'react-router';
import { HelpBadge } from '../../help/HelpMode';
import { classifyFramingState } from './classifyFramingGate';

/** 📐 Classify framing — sits right under the Composition bar, where the lack is
 * seen: imported without a shot type, or cropped since the last classify (crop
 * clears framing, same as the Bank). Renders nothing when there is nothing to
 * classify. */
export default function ClassifyFramingButton({
  images, ollama, capsLoading = false, busy = false, activity = null, onClassify,
}) {
  const s = classifyFramingState({ images, ollama, capsLoading, busy, activity });
  if (!s.visible) return null;

  return (
    <div id="ds-classify-framing" tabIndex={-1}
      className="scroll-mt-20 flex flex-col gap-1 rounded-lg border border-amber-400/40 bg-amber-500/5 px-3 py-2">
      <div className="flex flex-wrap items-center gap-2">
        <button type="button" data-workspace-focus
          onClick={() => onClassify?.(s.count)}
          disabled={s.disabled} title={s.title}
          className="px-3 py-1.5 rounded-lg bg-amber-500/15 border border-amber-400/40 text-amber-200 text-sm font-semibold disabled:opacity-40">
          {s.label}
        </button>
        <HelpBadge topic="action-classify-framing" className="self-center" />
        {!s.running && (
          <span className="text-content-subtle text-[0.6875rem] min-w-0">
            no shot type yet — imported without one, or cropped since they were last classified
          </span>
        )}
      </div>
      {s.blocked && (
        <p className="m-0 text-amber-300/90 text-[0.6875rem]">
          ⚠ {s.blockedReason}{' '}
          <Link to="/settings/local-tools" className="underline hover:text-amber-200">Open Local tools</Link>
        </p>
      )}
    </div>
  );
}
