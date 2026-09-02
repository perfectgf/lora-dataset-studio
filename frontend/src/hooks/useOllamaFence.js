/* The one place a surface refused by the local Ollama fence learns to wait.
 *
 * Wrap the action instead of calling it: `runGuarded(fn)` runs fn, and if the
 * server answers with the fence code, it holds on to fn, starts watching
 * /api/system/ollama-fence, and runs fn AGAIN by itself the moment the other
 * model goes away. The user's click is not lost, so the common case — Ollama's
 * own idle unload a few minutes later — costs nothing at all.
 *
 * Shared deliberately: the fence guards every local LLM call (Test Studio's
 * ✨ Enhance and 🔎 Describe, the Video Test Studio's ✨ Auto and ✨ Enrich,
 * captioning), and a per-button copy of this would drift the moment one of
 * them changed.
 *
 * The action is handed a `run` handle, and asks it before writing. The hook
 * can stop a vigil; it cannot stop a request already in flight, and the
 * reply comes back to the action, which writes it — into a field the user
 * has since re-aimed (a newer click, a changed frame or mode, "stop
 * waiting") or a panel that is gone. Measured: the answer for the OLD frame
 * landed after the switch, with no notice left to explain it. So
 * `run.current()` says whether the click is still the one the surface is
 * showing, `run.mounted()` whether there is still a surface to tell.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { apiFetch, postJson } from '../api/fetchClient';
import { AUTO_RETRY_CAP_MS, isOllamaFenceError, nextPollDelay } from '../utils/ollamaFence';

/**
 * `onError` is how a REPLAY reports a failure that is not the fence.
 *
 * It matters more than it looks: the first attempt fails inside the caller's
 * own try/catch, which toasts. The replay happens minutes later with nobody in
 * that catch — so without this, an Ollama that died while we waited would make
 * the notice quietly disappear and nothing else happen at all. A surface that
 * shows its own errors (the Describe modal writes into its error line) can
 * leave this out.
 */
export default function useOllamaFence({ onError } = {}) {
  const [state, setState] = useState(null);
  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;
  const actionRef = useRef(null);
  const timerRef = useRef(null);
  const startedRef = useRef(0);
  const aliveRef = useRef(true);
  // Which vigil is current. A poll that was already past its fetch when the
  // vigil was stopped — by a new click, "stop waiting", the unload — reads
  // this when it comes back, and acts on nothing if it is no longer its own.
  const vigilRef = useRef(0);

  const stopTimer = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = null;
    vigilRef.current += 1;
  }, []);

  /* The handle an action is run with: `current()` is true while the click it
     answers is still the current one — false once anything stopped the vigil
     it was started under, or the panel went away. Read at the moment of the
     write, never captured: the reply is what arrives late. */
  const runOf = useCallback((stamp) => ({
    current: () => aliveRef.current && stamp === vigilRef.current,
    mounted: () => aliveRef.current,
  }), []);

  useEffect(() => {
    aliveRef.current = true;
    return () => { aliveRef.current = false; stopTimer(); };
  }, [stopTimer]);

  /* Re-run what was refused, for the vigil `vigil` — the stamp read after
     the stop that preceded it. A click made while the replay runs owns the
     notice and the kept action from then on; a replay that comes back
     superseded touches neither. Returns true when the fence took it again —
     the model can be freed and immediately claimed by the other tool, and the
     caller has to put the vigil back on watch or it dies here. */
  const replay = useCallback(async (vigil) => {
    if (vigil !== vigilRef.current) return false;
    const action = actionRef.current;
    if (!action) { setState(null); return false; }
    setState((s) => (s ? { ...s, phase: 'retrying' } : s));
    try {
      await action(runOf(vigil));
    } catch (e) {
      if (!aliveRef.current || vigil !== vigilRef.current) return false;
      if (isOllamaFenceError(e)) {
        // Still fenced: keep waiting rather than throw the user back to the
        // start, but do not restart the patience clock.
        setState((s) => (s ? { ...s, phase: 'waiting' } : s));
        return true;
      }
      // A different failure, with nobody left holding the original catch.
      setState(null);
      actionRef.current = null;
      onErrorRef.current?.(e);
      return false;
    }
    if (aliveRef.current && vigil === vigilRef.current) {
      actionRef.current = null;
      setState(null);
    }
    return false;
  }, [runOf]);

  const tick = useCallback(async () => {
    if (!aliveRef.current) return;
    const vigil = vigilRef.current;
    const elapsedMs = Date.now() - startedRef.current;
    let free = false;
    let models = [];
    let provider;
    try {
      // background: a vigil that may run for ten minutes must not toast every
      // time the server blinks — the offline indicator already owns that story.
      const data = await apiFetch('/api/system/ollama-fence', { background: true });
      free = data?.blocked === false;
      models = data?.models || [];
      // The server the user runs, so the notice names LM Studio to someone on
      // LM Studio — the refusal itself does not say, the fence state does.
      provider = data?.provider;
    } catch {
      // Cannot read the state (old backend, server down): stay put and keep
      // the notice as it is. The user's buttons still work.
      free = false;
    }
    if (!aliveRef.current || vigil !== vigilRef.current) return;
    if (free) {
      stopTimer();
      const mine = vigilRef.current;
      // Freed, then taken again before we could use it: go back on watch
      // instead of leaving a "waiting" notice that nothing is watching.
      if (await replay(mine) && aliveRef.current && mine === vigilRef.current) {
        timerRef.current = setTimeout(tick, nextPollDelay(elapsedMs));
      }
      return;
    }
    if (elapsedMs >= AUTO_RETRY_CAP_MS) {
      stopTimer();
      setState((s) => (s ? { ...s, phase: 'gave-up', elapsedMs, models, provider } : s));
      return;
    }
    setState((s) => (s && s.phase === 'waiting' ? { ...s, elapsedMs, models, provider } : s));
    timerRef.current = setTimeout(tick, nextPollDelay(elapsedMs));
  }, [replay, stopTimer]);

  const beginWaiting = useCallback((message) => {
    startedRef.current = Date.now();
    setState({ phase: 'waiting', message, models: [], elapsedMs: 0 });
    stopTimer();
    timerRef.current = setTimeout(tick, nextPollDelay(0));
  }, [stopTimer, tick]);

  /**
   * Run `action(run)`. Returns true when it went through, false when the fence
   * took over (the notice is now showing and the action will be retried for
   * you). Every other failure propagates untouched — this hook only knows one
   * story. `run` is the handle described at the top: an action that writes
   * somewhere asks `run.current()` first.
   */
  const runGuarded = useCallback(async (action) => {
    // A new click supersedes whatever an earlier one left waiting. Left armed,
    // the vigil fired after THIS click had gone through and ran it a second
    // time — one click, two answers written into the field — and ran a click
    // that had failed for another reason again, toasting it twice. The notice
    // comes down with it: nothing is waiting while this click runs, and a
    // notice left up under it offered an Unload whose resume ran THIS click
    // on top of itself. The action is kept only once the fence has refused
    // it. And the cleanup is conditional on the stamp for the same reason in
    // reverse: a click that comes back after a NEWER one took over must not
    // clear that one's notice or its kept action.
    stopTimer();
    const mine = vigilRef.current;
    actionRef.current = null;
    setState(null);
    try {
      await action(runOf(mine));
    } catch (e) {
      if (!isOllamaFenceError(e)) {
        if (mine === vigilRef.current) { actionRef.current = null; setState(null); }
        throw e;
      }
      // Superseded while it ran — the surface changed setup and stopped
      // waiting, a newer click took over, the component went away: nobody
      // wants THIS click replayed. Not silently, though: the refusal goes to
      // the caller's catch like any other failure, and shows there.
      if (!aliveRef.current || mine !== vigilRef.current) throw e;
      actionRef.current = action;
      beginWaiting(e?.message);
      return false;
    }
    if (aliveRef.current && mine === vigilRef.current) {
      actionRef.current = null;
      setState(null);
    }
    return true;
  }, [beginWaiting, runOf, stopTimer]);

  /** The consent click: evict the other model, then resume. */
  const unloadAndRetry = useCallback(async () => {
    // Nothing waiting — a newer click took the notice down while its own
    // request ran — is nothing to unload for, and nothing to resume.
    if (!actionRef.current) return false;
    setState((s) => (s ? { ...s, phase: 'unloading' } : s));
    stopTimer();
    const mine = vigilRef.current;
    try {
      await postJson('/api/system/ollama-fence/unload', { confirmed_unload_external: true });
    } catch (e) {
      if (!aliveRef.current || mine !== vigilRef.current) return false;
      // The server refuses what it cannot prove and says why (still busy,
      // unreachable). Show that, and go back to waiting rather than dead-end.
      setState((s) => (s ? { ...s, phase: 'waiting', message: e?.message || s.message } : s));
      timerRef.current = setTimeout(tick, nextPollDelay(Date.now() - startedRef.current));
      return false;
    }
    // The other tool can reload a model between our unload and our retry —
    // and a click made during the unload already ran itself: the replay
    // knows it from the stamp and stands down.
    if (await replay(mine) && aliveRef.current && mine === vigilRef.current) {
      timerRef.current = setTimeout(tick, nextPollDelay(Date.now() - startedRef.current));
    }
    return true;
  }, [replay, stopTimer, tick]);

  /** "Stop waiting" — abandon the queued action without unloading anything. */
  const stopWaiting = useCallback(() => {
    stopTimer();
    actionRef.current = null;
    setState(null);
  }, [stopTimer]);

  return { fence: state, runGuarded, unloadAndRetry, stopWaiting };
}
