// react-frontend/src/components/bank/useCurationLanes.js
// The bank's four curation lanes — 🎨 diverse, ⚖️ balanced, 🎯 similar and
// 🔤 text search — as ONE hook, because that is their real coupling:
// curateOpen is shared by all four, the typicality guard by two, and the
// engine-switch flow resets three of their states (the setters ride back
// in the return for exactly that). Moved VERBATIM from BankWorkspace.jsx
// (2026-08-24, hook series wave 7).
import { useEffect, useState } from 'react';
import { apiFetch, postJson } from '../../api/fetchClient';
import { BALANCE_DEFAULT_AXIS, summarizeBalance } from './bankBalance.js';
import {
  semanticEnginePatchBody, semanticPayloadMatches,
} from './bankSemanticEngine.js';
import { PUSH_DOWN_DEFAULT_STRENGTH, summarize } from './bankTextSearch.js';

export function useCurationLanes({
  bankId, filter, filterParams, selected, toast, showCuratedSelection,
  semanticState, semanticEngineRef, textStatusRequestRef,
}) {
  // Curation popovers ('diverse' | 'similar' | null) and their target counts.
  const [curateOpen, setCurateOpen] = useState(null)
  const [diverseN, setDiverseN] = useState(60)
  // Typicality guard for 🎨 Pick diverse. Pure farthest-point sampling maximises
  // the distance to what is already picked — mathematically the criterion that
  // prefers ISOLATED images, so the first picks used to be the memes and the
  // stray photos of someone else. 0 = the historical behaviour, on purpose still
  // reachable; 0.5 = the default (see BANK_TYPICALITY_DEFAULT rationale in the
  // service docstring).
  const [diverseTypicality, setDiverseTypicality] = useState(0.5)
  const [diverseBusy, setDiverseBusy] = useState(false)
  // ⚖️ Balanced pick — the OTHER question ("does my set cover the framings?").
  // Axis ids are persisted keys, never renamed (see bankBalance.js).
  const [balanceN, setBalanceN] = useState(60)
  const [balanceAxis, setBalanceAxis] = useState(BALANCE_DEFAULT_AXIS)
  const [balanceBusy, setBalanceBusy] = useState(false)
  const [balanceResult, setBalanceResult] = useState(null)
  const [similarN, setSimilarN] = useState(60)
  const [similarBusy, setSimilarBusy] = useState(false)
  // 🔤 Text search. `textStatus` is the BEFORE-the-click truth (available? model
  // already warm? would it download?), `textResult` the AFTER-the-click one that
  // keeps the ranking legible once the grid has switched to it.
  const [textQuery, setTextQuery] = useState('')
  const [textN, setTextN] = useState(60)
  // 🔤 what to push DOWN the ranking. Not a filter — see bankTextSearch.js.
  const [textExclude, setTextExclude] = useState('')
  const [textExcludeW, setTextExcludeW] = useState(PUSH_DOWN_DEFAULT_STRENGTH)
  const [textStatus, setTextStatus] = useState(null)
  const [textPending, setTextPending] = useState(false)
  const [textResult, setTextResult] = useState(null)

  const pickDiverse = async () => {
    const requestEngine = semanticState.engine
    const requestModelKey = semanticState.modelKey
    setCurateOpen(null)
    setDiverseBusy(true)
    try {
      const d = await postJson(`/api/bank/${bankId}/select-diverse`,
        { n: diverseN, typicality: diverseTypicality, ...filterParams(filter) })
      if (semanticEngineRef.current !== requestEngine
          || !semanticPayloadMatches(d, requestEngine, requestModelKey)) return
      if (!d.image_ids?.length) {
        toast.info(`Nothing to sample — no ${semanticState.label}-indexed images match the current filter.`)
        return
      }
      showCuratedSelection(d.image_ids)
      toast.info(`Showing the ${d.image_ids.length} most diverse of ${d.pool}. Review, then ✓ Keep or ⬆ Promote — or “Show all” to leave this view.`)
    } catch (e) {
      toast.error(e?.message || 'Diversity sampling failed.')
    } finally {
      setDiverseBusy(false)
    }
  }

  // ⚖️ Balanced pick — spread over the framings instead of taking the top of one
  // ranking. Same embeddings and same typicality guard as 🎨 Pick diverse, applied
  // INSIDE each bucket. The result is only useful if the user can see its shape,
  // so the distribution is kept on screen (numbers, aria-live) after the click.
  const pickBalanced = async () => {
    const requestEngine = semanticState.engine
    const requestModelKey = semanticState.modelKey
    setCurateOpen(null)
    setBalanceBusy(true)
    try {
      const d = await postJson(`/api/bank/${bankId}/select-balanced`,
        { n: balanceN, axis: balanceAxis, typicality: diverseTypicality,
          ...filterParams(filter) })
      if (semanticEngineRef.current !== requestEngine
          || !semanticPayloadMatches(d, requestEngine, requestModelKey)) return
      if (!d.image_ids?.length) {
        toast.info('Nothing to balance — no labelled images match the current filter.')
        return
      }
      setBalanceResult(d)
      showCuratedSelection(d.image_ids)
      toast.info(summarizeBalance(d))
    } catch (e) {
      // A missing pass is the DEFAULT state of a fresh bank, not a failure: the
      // backend names the pass, so show that sentence rather than "failed".
      toast.error(e?.message || 'Balanced selection failed.')
    } finally {
      setBalanceBusy(false)
    }
  }

  const findSimilar = async () => {
    const requestEngine = semanticState.engine
    const requestModelKey = semanticState.modelKey
    setCurateOpen(null)
    const ref = [...selected][0]
    if (ref == null) return
    setSimilarBusy(true)
    try {
      const d = await postJson(`/api/bank/${bankId}/select-similar`,
        { ref_id: ref, n: similarN, ...filterParams(filter) })
      if (semanticEngineRef.current !== requestEngine
          || !semanticPayloadMatches(d, requestEngine, requestModelKey)) return
      if (!d.image_ids?.length) {
        toast.info(`No matches — no ${semanticState.label}-indexed images match the current filter.`)
        return
      }
      // Backend returns the ids ranked by similarity (reference first); keep that
      // order so the view reads closest→farthest instead of by id.
      showCuratedSelection(d.image_ids)
      toast.info(`Showing the ${d.image_ids.length} most similar to the reference (of ${d.pool}), closest first. Review, then ✓ Keep or ⬆ Promote — or “Show all” to leave this view.`)
    } catch (e) {
      toast.error(e?.message || 'Similarity search failed.')
    } finally {
      setSimilarBusy(false)
    }
  }

  // 🔤 Text search — same engine as 🎯 Similar, with the reference vector coming
  // from words instead of a picture. Opening the panel asks the backend what it
  // is about to cost (model warm? weights present?) so a slow FIRST search is
  // announced before the click rather than felt as a freeze after it.
  const openTextSearch = async () => {
    const next = curateOpen === 'text' ? null : 'text'
    setCurateOpen(next)
    if (next !== 'text') {
      textStatusRequestRef.current += 1
      releaseTextEncoder()
      return
    }
    const requestId = ++textStatusRequestRef.current
    const expectedEngine = semanticState.engine
    if (semanticState.text) setTextStatus(semanticState.text)
    try {
      const status = await apiFetch('/api/bank/text-search/status'
        + `?engine=${encodeURIComponent(expectedEngine)}`)
      if (requestId === textStatusRequestRef.current
          && expectedEngine === semanticEngineRef.current
          && semanticPayloadMatches(status, expectedEngine)) setTextStatus(status)
    } catch {
      // The Bank payload already carries an engine-aware status. Keep it when
      // the optional warm/cold probe cannot be read.
      if (!semanticState.text) setTextStatus(null)
    }
  }

  // Hand the selected text encoder's memory back as soon as the panel closes.
  // Best effort by design — the backend idle timer remains the guarantee for a
  // tab that simply vanished.
  const releaseTextEncoder = (engine = semanticState.engine) => {
    postJson('/api/bank/text-search/release', semanticEnginePatchBody(engine)).catch(() => {})
  }

  // Leaving the Bank entirely is the same signal as closing the panel: give the
  // memory back. The backend idle timer still covers a browser that just died.
  useEffect(() => () => {
    postJson('/api/bank/text-search/release',
      semanticEnginePatchBody(semanticEngineRef.current)).catch(() => {})
  }, [semanticEngineRef])   // une ref est stable : ceci reste unmount-only

  const runTextSearch = async () => {
    const q = textQuery.trim()
    if (!q) return
    const requestEngine = semanticState.engine
    const requestModelKey = semanticState.modelKey
    setTextPending(true)
    try {
      const d = await postJson(`/api/bank/${bankId}/search-text`,
        { query: q, n: textN, push_down: textExclude.trim() || null,
          push_down_weight: textExcludeW, ...filterParams(filter) })
      if (semanticEngineRef.current !== requestEngine
          || !semanticPayloadMatches(d, requestEngine, requestModelKey)) return
      setTextResult(d)
      setCurateOpen(null)
      if (!d.image_ids?.length) {
        // NOT a silent empty grid: say why nothing could be ranked.
        toast.info(summarize(d, semanticState.engine))
        return
      }
      showCuratedSelection(d.image_ids)
      // Refresh the warm flag so the panel now promises "instant" truthfully.
      apiFetch('/api/bank/text-search/status'
        + `?engine=${encodeURIComponent(semanticState.engine)}`)
        .then(setTextStatus).catch(() => {})
    } catch (e) {
      // 503 = this install cannot do it at all; 400 = do something first. Both
      // arrive as a message written for a human — show it as-is.
      toast.error(e?.message || 'Text search failed.')
    } finally {
      setTextPending(false)
    }
  }
  return {
    curateOpen, setCurateOpen, diverseN, setDiverseN, diverseTypicality,
    setDiverseTypicality, diverseBusy, balanceN, setBalanceN, balanceAxis,
    setBalanceAxis, balanceBusy, balanceResult, setBalanceResult, similarN,
    setSimilarN, similarBusy, textQuery, setTextQuery, textN, setTextN,
    textExclude, setTextExclude, textExcludeW, setTextExcludeW, textStatus,
    setTextStatus, textPending, textResult, setTextResult, pickDiverse,
    pickBalanced, findSimilar, openTextSearch, releaseTextEncoder,
    runTextSearch,
  };
}
