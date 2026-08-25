/**
 * ⚙ The analysis passes — the old ① Analyze zone, now a PANEL opened on demand
 * from the top bar.
 *
 * Why it moved. The passes are the step you run once per bank and then leave
 * alone for days, and they were occupying the top third of a screen whose
 * subject is the grid. Opening them on demand gives that third back to the
 * images without taking a single pass away.
 *
 * ⚠️ This is a CONTAINER, not a rewrite. All eight passes are here, each still
 * opening its own PassDialog with its own endpoint, scope and refusals. Nothing
 * about what a button does changed — only where you reach it from. The
 * bank-wide read-only overview rides along, because "what has been measured"
 * and "what can I measure" are the same question asked twice.
 */
import BankOverview from './BankOverview.jsx'
import { Palette, Ruler, Search, Users } from 'lucide-react';
import BankEditPanel from './BankEditPanel.jsx'
import BankSemanticEngine from './BankSemanticEngine.jsx'
import BankWatermarkPanel from './BankWatermarkPanel'
import { GroupLabel, PassButton } from './BankAtoms.jsx'
import SettingsLink from '../common/SettingsLink'
import { captionButtonLabel, captionScopeNote } from './bankCaptionScope.js'
import { holdsTheGpu } from './bankScoreDevice.js'
import { openerLabel } from './scoringPython.js'

/* Below lg the panel folds everything that is not a pass button. Measured by
   the responsive probe at 360 px: the panel was ~1 500 px tall — engine card,
   eight buttons, watermark and edit panels, notes, overview, all stacked — so
   opening it cost two screens. The buttons are what you came for; each
   secondary block is one tap away, with its name on the fold. From lg up the
   layout is unchanged. */
function Fold({ compact, title, children }) {
  if (!compact) return children
  return (
    <details className="rounded-lg border border-border bg-surface">
      <summary className="min-h-10 cursor-pointer select-none px-3 py-2 text-sm text-content-muted hover:text-content">
        {title}
      </summary>
      <div className="px-3 pb-3">{children}</div>
    </details>
  )
}

export default function BankPassesPanel({
  bankId, payload, counts, live, caps, capsLoading, compact = false,
  semanticState, semanticReady, semanticBlocked, semanticSwitching, semanticOperationBusy,
  scoreGpuPresent, scoreDevice, scoreNote, visionReady,
  selected, captionScope,
  captionVocab, visionModel, visionModelLooksUncensored,
  onPassOpen, onPassRedo, onPickPython, onSemanticEngineChange, onChanged,
}) {
  return (
    <div className="grid gap-4 xl:grid-cols-12 xl:items-start">
      <div className="min-w-0 space-y-3 xl:col-span-7">
        <Fold compact={compact} title="Semantic engine">
        <BankSemanticEngine state={semanticState} capsLoading={capsLoading}
          switching={semanticSwitching} disabled={semanticOperationBusy} live={live}
          gpuPresent={scoreGpuPresent}
          onPickPython={() => onPickPython('semantic')}
          onChange={onSemanticEngineChange}
          onIndex={() => {
            // The visible “Reindex” promise and the request body agree on first
            // open; incomplete indexes resume with rescan:false.
            onPassRedo('semantic_index', semanticState.complete)
            onPassOpen('semantic_index')
          }} />
        </Fold>

        {/* Analysis passes — individual, quieter than the primary actions. */}
        <div className="space-y-1.5">
          <GroupLabel>Analysis passes</GroupLabel>
          <div className="flex flex-wrap items-center gap-1.5">
            {/* Every button OPENS ITS WINDOW instead of firing. The window is where
                the scope, the counts, the settings the calculation really reads and
                the things it does NOT decide finally have room to be said — and
                where the launch button quotes the exact number it will move.

                “Rescan all” is gone as a button on purpose: it was never a second
                pass, it was a SCOPE wearing a button's clothes. It is now the last
                line of 🔎 Scan's THIS RUN block, next to the piles it belongs with.
                “Rescore all” went the same way, into ✨ Score's window. */}
            <PassButton onClick={() => onPassOpen('scan')} disabled={live}
              title="Measure sharpness, noise, flatness, size and detail, hash every image and group the exact duplicates — CPU only. Opens the launch window.">
              <Search aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Scan quality…
            </PassButton>
            <PassButton onClick={() => onPassOpen('faces')} disabled={live || !caps.face_scoring}
              title={caps.face_scoring
                ? 'Detect the dominant face of every non-rejected image and cluster the bank by person (no reference needed). CPU, can take a while on thousands of images. It samples your subfolders first and offers the ones that look like a single person, so you can skip them.'
                : 'Install the Quality tools (Setup) to sort by person'}>
              <Users aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Group by person…
            </PassButton>
            <PassButton onClick={() => onPassOpen('score')} disabled={live || !caps.bank_scoring}
              title={caps.bank_scoring
                ? `Rate every non-rejected image for aesthetics (1–10), flag NSFW, and group by visual style — one CLIP pass. Powers a smarter "keep best". Already-scored images are reused, so stopping and relaunching costs only what is left. Runs in the background${
                  holdsTheGpu(scoreDevice) ? ', and holds the GPU (ComfyUI is unloaded and training cannot start) for its duration' : ' on the CPU, leaving the GPU free'}.`
                : 'Install the Bank scoring extra (Setup ▸ Quality tools) to score aesthetics / NSFW / style'}>
              Score…{!caps.bank_scoring && ' (needs setup)'}
            </PassButton>
            <PassButton onClick={() => onPassOpen('medium')} disabled={live || !caps.bank_scoring}
              title={caps.bank_scoring
                ? 'Sort every scored image into photograph / anime / 3D render / illustration — read off the CLIP embeddings Score already computed, so no image is looked at again and the GPU stays free. It answers “unsure” rather than guessing: measured on a real 23 500-image bank, it named 2 anime drawings and no wrong verdict.'
                : 'Install the Bank scoring extra (Setup ▸ Quality tools) — Medium reads the embeddings the Score pass produces'}>
              <Palette aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Classify medium…{!caps.bank_scoring && ' (needs setup)'}
            </PassButton>
            <PassButton onClick={() => onPassOpen('framing')} disabled={live || !visionReady}
              title={visionReady
                ? 'Classify every non-rejected image by shot type — face close-up, bust, full body, back view — with the same Qwen3-VL classifier the datasets use. Powers the Framing filter and the coverage advice. GPU vision pass.'
                : 'Pull the vision model (Settings ▸ Local tools) to classify framing'}>
              <Ruler aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Classify framing…{!visionReady && ' (needs setup)'}
            </PassButton>
            <PassButton onClick={() => onPassOpen('semantic_dedup')} disabled={live || !semanticReady}
              title={semanticReady
                ? `Group crops and re-compressed variants of the SAME shot the exact-duplicate hash misses from the ${semanticState.label} semantic index. Review them under the ✂ Same shot chip.`
                : semanticBlocked}>
              ✂ Find crops &amp; variants…{!semanticReady
                && ` (needs ${semanticState.engine === 'clip' ? 'Score' : 'SigLIP 2 index'})`}
            </PassButton>
            {/* The label QUOTES THE NUMBER IT WILL MOVE — the scope's uncaptioned rows,
                or the selection when there is one. "Caption all" was the older, vaguer
                promise, and a button that announces one figure and acts on another is the
                misunderstanding this whole row exists to end.

                IT NO LONGER GREYS OUT AT ZERO. It used to, and it took the engine, the
                model, the register and the length pickers down with it — on exactly the
                bank whose captions you wanted redone with a better model. The window is
                always reachable; the LAUNCH button inside it is what refuses a run of 0,
                and it says why. */}
            <PassButton onClick={() => onPassOpen('caption')} disabled={live}
              title={selected.size
                ? `Caption the ${selected.size} selected image(s). Opens the window with the engine, model, register, length and scope.`
                : `${captionScopeNote(selected.size, counts, captionScope)} Opens the window with the engine, model, register, length and scope.`}>
              {captionButtonLabel(selected.size, counts, captionScope)}…
            </PassButton>
            {/* ⤢ the opt-in angle backfill. Its own button and its own window, never
                folded into 👥: it is hours of work on a big bank and nobody must pay
                it by accident. Shown only when there is something to measure. */}
            {(counts?.angle_backfillable || 0) > 0 && (
              <PassButton onClick={() => onPassOpen('angles')} disabled={live}
                title="Measure the head angle of the images a previous build face-scanned without keeping it. Writes the angle and nothing else.">
                ⤢ Measure head angles…
              </PassButton>
            )}
          </div>
          {/* Watermark CLEANING — the two manual levels (crop, then inpaint), with
              their own per-level progress. Lives in its own component so the
              "which level can run, and why not" logic stays unit-tested. */}
          <Fold compact={compact} title="Watermarks">
          <BankWatermarkPanel bankId={bankId} live={live}
            onFind={() => onPassOpen('watermark')}
            payload={payload} selectedIds={[...selected]}
            gpuPresent={scoreGpuPresent} onPickPython={onPickPython}
            onChanged={onChanged} />
          </Fold>
          {/* ✂ Edits — the crop and the upscale made in the Bank itself, next to
              the watermark cleaning because they are the same KIND of thing: the
              three actions that produce new pixels, each undone by throwing away
              a copy the app made (nofaceman, Discord). */}
          <Fold compact={compact} title="Edits">
          <BankEditPanel bankId={bankId} live={live}
            payload={payload} selectedIds={[...selected]}
            onChanged={onChanged} />
          </Fold>
          {scoreNote && (
            <p className={`text-xs ${scoreNote.tone === 'warn'
              ? 'text-amber-400/90' : 'text-content-subtle'}`}>
              {scoreNote.text}
            </p>
          )}
          {!capsLoading && !caps.bank_scoring && (
            <p className="text-xs text-content-muted">
              Score needs its own packages (Setup ▸ Quality tools) — or an interpreter
              that already has them{scoreGpuPresent
                ? '. If you train LoRAs or run ComfyUI, this machine probably has one.'
                : ', which saves installing them twice.'}
            </p>
          )}
          {/* The interpreter picker, offered where it can actually help: the pass is
              about to crawl on the CPU of a machine that HAS a card, or the scoring
              packages are missing and another Python here may already carry them
              (true with or without a card — it saves an install either way). The
              LABEL adapts: a machine with no NVIDIA card must never be promised "a
              GPU Python", and the note it gets alongside is "this is how it is",
              not a fix to chase. */}
          {!capsLoading && (scoreNote?.tone === 'warn' || !caps.bank_scoring) && (
            <div>
              <button type="button" onClick={() => onPickPython('scoring')}
                className={`min-h-10 lg:min-h-0 rounded-md border px-2 py-1 text-xs font-medium ${scoreGpuPresent
                  ? 'border-amber-400/50 text-amber-300 hover:bg-amber-500/10'
                  : 'border-border text-content-muted hover:bg-surface-raised hover:text-content'}`}>
                {openerLabel(scoreGpuPresent)}
              </button>
            </div>
          )}
          {captionVocab === 'explicit' && !visionModelLooksUncensored && (
            /* The old wording sent people to Settings ▸ Captioning & quality, which holds
               the ENGINE selector — the vision model lives in Settings ▸ Local tools. Now
               that the model is pickable right here, the sentence points at the control
               on screen instead, and only mentions Settings for the case the picker
               cannot solve: no uncensored model pulled on this machine at all. */
            <p className="text-xs text-amber-400/90">
              ⚠️ Explicit captions need an uncensored (abliterated) Ollama vision model
              {visionModel ? ` — “${visionModel}” may refuse or soften explicit terms` : ''}.
              Pick another one in <b>Caption vision model</b> above, or pull one from{' '}
              <SettingsLink section="local-tools" focus="ollama-vision-model" tone="warning">
                Settings ▸ Local tools
              </SettingsLink>. Richer captions also feed the search.
            </p>
          )}
        </div>
      </div>
      <div className="min-w-0 xl:col-span-5">
        <Fold compact={compact} title="Bank overview">
        <BankOverview payload={payload} />
        </Fold>
      </div>
    </div>
  )
}
