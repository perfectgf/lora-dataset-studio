import { cloudTierEstimateView } from '../../utils/trainingMode.js';

/* ☁ One GPU tier's price, duration and total — the line under every radio in a
   cloud launch dialog. ONE component, TWO dialogs.

   It used to be a private function of the image lane's CloudLaunchDialog. The
   video lane then grew its own launch window, and the choice was the usual
   one: copy the twenty lines, or share them. Shared — the estimate's honesty
   rules live in cloudTierEstimateView (a `rough` estimate shows, a `pending`
   or `unavailable` one says so instead of inventing a number, a run over the
   runtime cap is warned about before the click), and two copies of those rules
   are two places for the cheaper dialog to quietly stop warning. A contract
   test fails if either dialog grows a copy. */

export function fmtDuration(min) {
  if (min == null) return '—';
  if (min < 90) return `~${min} min`;
  const h = Math.floor(min / 60);
  const m = min % 60;
  return m ? `~${h} h ${m} min` : `~${h} h`;
}

export default function CloudTierEstimate({ tier, fullMode = false, maxRuntimeMinutes }) {
  const estimate = cloudTierEstimateView(tier, { fullMode });
  return (
    <>
      <span className="block text-content-subtle text-[0.75rem] tabular-nums">
        {tier.dph_total != null ? `$${tier.dph_total.toFixed(3)}/h` : 'price n/a'}
        {estimate.available ? (
          <>
            {' · '}{fmtDuration(estimate.minutes)}
            {estimate.cost != null ? ` · ≈ $${estimate.cost.toFixed(2)} total` : ''}
          </>
        ) : fullMode ? (
          <span className="text-amber-200"> · full-model estimate unavailable — hourly price only</span>
        ) : (
          <span> · duration and cost unavailable</span>
        )}
      </span>
      {estimate.exceedsCap && (
        <span className="block text-amber-300 text-[0.6875rem]">
          ⚠ Longer than the {Math.round((maxRuntimeMinutes || 480) / 60)} h runtime cap — the run would be cut short{fullMode
            ? '; the latest full-model checkpoint may not have reached Hugging Face'
            : ' (saved LoRA checkpoints are rescued)'}. Pick a faster GPU or raise the cap in Settings.
        </span>
      )}
    </>
  );
}
