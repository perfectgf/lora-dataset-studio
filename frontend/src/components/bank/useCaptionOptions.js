// react-frontend/src/components/bank/useCaptionOptions.js
// The bank's per-run caption dials — register, length, engine and vision
// model (all '' = follow Settings, key left OUT of the request), the pulled
// Ollama model list, the include-asserted escape hatch, the spread-if-set
// request builder and the derived effective-model values — moved VERBATIM
// from BankWorkspace.jsx (2026-08-24, hook series wave 6).
import { useEffect, useState } from 'react';
import { apiFetch } from '../../api/fetchClient';
import { OLLAMA_RELEVANT } from '../dataset/CaptionOptionsPopover';

export function useCaptionOptions({ caps }) {
  // Caption register for the 🏷️ Caption pass ('' = model's own wording). Explicit is
  // the NSFW lane — same registers as the dataset caption, passed per-run.
  const [captionVocab, setCaptionVocab] = useState('')
  // Caption LENGTH preset, per RUN like the vocabulary register above (a bank has no
  // caption_options row to persist to). '' = standard: nothing appended to the prompt.
  const [captionLength, setCaptionLength] = useState('')
  // WHICH ENGINE and WHICH VISION MODEL write this run's captions. Per RUN, like every
  // other dial on this row: the global Settings stay the default and are never written
  // from here, so a user can try a different captioner on one pass without changing what
  // every dataset does afterwards. '' on either = follow the setting, and the key is then
  // left OUT of the request — a run that picks nothing is byte-identical to before.
  const [captionEngine, setCaptionEngine] = useState('')
  const [captionModel, setCaptionModel] = useState('')
  // The pulled Ollama models, for the picker. Not in `caps` (which carries only the
  // configured vision model), so it is its own always-200 fetch — an unreachable Ollama
  // is an empty list, never an error.
  const [ollamaModels, setOllamaModels] = useState([])
  /* The ESCAPE HATCH, and the reason it is a piece of state and not a request key: it has
     to be visible, deliberate and re-read in the confirmation. Never persisted, so it
     resets with the panel — an opt-out of a protection is not a preference. */
  const [captionIncludeAsserted, setCaptionIncludeAsserted] = useState(false)

  // 🏷️ The pulled Ollama models, for the per-run caption model picker. Fetched ONCE per
  // mount and never blocking: the endpoint always answers 200, and an unreachable Ollama
  // is an empty list — the picker then offers only "Use the configured model", which is
  // exactly the truth on that machine.
  useEffect(() => {
    let alive = true
    apiFetch('/api/ollama/models').catch(() => ({ models: [] }))
      .then((d) => { if (alive) setOllamaModels(d?.models || []) })
    return () => { alive = false }
  }, [])

  /* Every option is spread-if-set, so a run that changes nothing posts the SAME body it
     posted before any of these controls existed — the contract the vocabulary/length
     pair set and the two new dials join.

     `statuses` is deliberately omitted while a selection is live: the server INTERSECTS
     the two, so "kept only" plus a selection of undecided images would caption fewer
     than the button says. The selection wins, the scope select goes inert, and the label
     switches to the selection count. */
  const captionRunOptions = () => ({
    ...(captionVocab ? { vocabulary: captionVocab } : {}),
    ...(captionLength ? { length: captionLength } : {}),
    ...(captionEngine ? { backend: captionEngine } : {}),
    ...(captionModel ? { ollama_model: captionModel } : {}),
  })

  // It reads the EFFECTIVE model — this run's override if one was picked, else the
  // configured one. Warning about the global model while the run uses another is worse
  // than not warning at all.
  const visionModel = captionModel || caps.ollama?.vision_model || ''
  const visionModelLooksUncensored = /abliterat|uncensor|huihui|nsfw/i.test(visionModel)
  // The Ollama model choice only bites when the resolved engine can reach Ollama.
  const ollamaPicksApply = OLLAMA_RELEVANT.has(captionEngine)
  // A model pulled elsewhere (or configured in Settings) stays selectable even when the
  // live list doesn't carry it — silently dropping the user's choice is worse than
  // offering a name we can't confirm.
  const captionModelChoices = captionModel && !ollamaModels.includes(captionModel)
    ? [captionModel, ...ollamaModels] : ollamaModels
  return {
    captionVocab, setCaptionVocab, captionLength, setCaptionLength,
    captionEngine, setCaptionEngine, captionModel, setCaptionModel,
    ollamaModels, captionIncludeAsserted, setCaptionIncludeAsserted,
    visionModel, visionModelLooksUncensored, ollamaPicksApply,
    captionModelChoices, captionRunOptions,
  };
}
