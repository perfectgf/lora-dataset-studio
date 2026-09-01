/**
 * ⚙ The video analysis passes — a PANEL opened on demand from the top bar,
 * exactly like the image lane's.
 *
 * Why it moved. Thirteen buttons were a permanent wall between the header and
 * the grid, on a screen whose subject is the shots. The passes are the step you
 * run once per bank and then leave alone; opening them on demand gives that
 * band back to the gallery without taking a single pass away.
 *
 * ⚠️ This is a CONTAINER, not a rewrite. Every pass keeps its endpoint, its
 * order, its gating and its ⓘ — only where you reach it from changed, and the
 * buttons wear the shared PassButton atom instead of hand-rolled classes. The
 * ▶ pipeline and 🎬 promote buttons are NOT here: like the image lane's
 * "Launch all…" and "Promote…", the two decisive actions live in the top bar.
 */
import VideoCapabilityStrip from './VideoCapabilityStrip'
import VideoShotCutsPanel from './VideoShotCutsPanel'
import { GroupLabel, PassButton } from '../bank/BankAtoms.jsx'
import { GuideInfoDot } from '../common/GuideSectionModal'
import { passBlockedBy } from './videoCapability'
import { PASS_LABELS } from './videoBankStatus'
import { VIDEO_PASS_TOPICS } from './videoPassTopics'

export default function VideoPassesPanel({
  bankId, bank, counts, capability, step, busy,
  startPass, onDescribe, onCutsChanged,
}) {
  return (
    <div className="space-y-3 rounded-xl border border-border bg-surface p-3">
      {/* What to do next, as ONE sentence. Twelve equal buttons and no order is
          how a user runs detection before the probe, gets "0 shots" and a green
          success, and concludes the app cannot read their files. */}
      <div className="rounded-lg border border-border bg-surface-raised p-3 text-sm">
        <p className="text-content">{step.text}</p>
        {step.blocked && (
          <p className="mt-1 text-xs text-amber-300">⚠ {step.blocked.why}</p>
        )}
      </div>

      <VideoCapabilityStrip capability={capability} />

      <GroupLabel>Analysis passes</GroupLabel>
      <div className="flex flex-wrap items-center gap-2">
        {/* ✂ and 🔖 sit AFTER embed on purpose: duplicates reuse the vectors
            that pass caches, so running them before it is the one order that
            produces an honest-looking empty answer. */}
        {/* 🩻 sits last: it is the only pass here that needs the ENCODER rather
            than the decoder, so on an install without ffmpeg it is the one grey
            button in a row of working ones — and the tooltip says which binary
            rather than "unavailable". */}
        {/* 🤖 sits at the end because it is the slowest button in the row —
            about 0.8 s per shot on the CPU, measured — and because it is the
            only one whose result is a hedge rather than a measurement. */}
        {/* 🎥 sits beside 🤖 rather than earlier because it consumes nothing
            and produces no input for anything else — but it goes BEFORE it,
            because it is the cheap one of the pair (0.07 s per second of source
            against 0.8 s per shot) and a row is read left to right. */}
        {['probe', 'detect', 'thumbs', 'measure', 'embed',
          'caption', 'dedup', 'watermark', 'safezone', 'defects',
          'camera', 'aicheck'].map((pass) => {
          const blocked = passBlockedBy(capability, pass)
          // 🔳 is the one pass that runs with HALF its dependencies: no OCR
          // engine still measures the bands. So it is never disabled for that —
          // the tooltip says what the run will and will not include instead,
          // which is the difference between a working button and a dead one.
          const partial = pass === 'safezone' && capability
            && !capability.video_text
          return (
            // The ⓘ rides OUTSIDE the pass button (nested buttons are invalid
            // HTML) but inside one non-wrapping group, so a wrap of the row can
            // never strand an explanation next to the wrong button.
            <span key={pass} className="inline-flex items-center gap-1">
              <PassButton
                // 🗣 launches through its window: the wording question belongs at
                // the moment of the click, not in a dropdown three screens down.
                onClick={() => (pass === 'caption' ? onDescribe() : startPass(pass))}
                disabled={busy || !!blocked}
                title={blocked ? blocked.why
                  : partial ? `Bands only — ${capability.video_text_detail
                      || 'the burned-in text extra is not installed'}`
                  : undefined}>
                {PASS_LABELS[pass]}
              </PassButton>
              <GuideInfoDot topic={VIDEO_PASS_TOPICS[pass]} label={PASS_LABELS[pass]} />
            </span>
          )
        })}
      </div>

      {/* The cuts panel rides with the passes: it only means something once
          Measure has run, and it decides which shots EXIST — the thresholds in
          the rail decide which of them get flagged. */}
      {(counts.sources || 0) > 0 && (
        <VideoShotCutsPanel bankId={bankId} shotDetect={bank?.shot_detect}
          onChanged={onCutsChanged} />
      )}
    </div>
  )
}
