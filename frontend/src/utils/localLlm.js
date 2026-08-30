/**
 * The ACTIVE local LLM, in the one shape every gate on this side already reads.
 *
 * Before this, `classifyFramingGate` and `enhanceGate` took `caps.ollama` and
 * nothing else. On a machine running LM Studio and no Ollama, both blocks are
 * false, so ✨ Enhance and 📐 Classify framing rendered disabled with "install
 * Ollama" — while the ⚙️ beside them listed LM Studio models and the backend
 * happily answered 200. The button could not be clicked at all.
 *
 * The returned object deliberately keeps Ollama's own field names
 * (`installed` / `reachable` / `vision_model_ready` / `vision_model`). That is not
 * laziness: those gates are pure functions with their own tests written against
 * that shape, and translating here instead of rewriting them keeps every one of
 * those assertions meaningful. `provider` is what the sentences branch on.
 */

/** `{provider, installed, reachable, vision_model_ready, vision_model, detail}`. */
export function activeLocalLlm(caps) {
  const c = caps || {}
  // Total: a config predating the setting has no local_llm block, and means Ollama.
  const provider = ((c.local_llm || {}).provider) || 'ollama'
  if (provider === 'lmstudio') {
    const l = c.lmstudio || {}
    return {
      provider,
      // LM Studio is never "installed but stopped" from here — there is no binary
      // this app can detect and no way for it to start one. Reachable or not is the
      // whole ladder, so `installed` tracks it and the "start it for me" branch of
      // every gate simply never fires.
      installed: !!l.reachable,
      reachable: !!l.reachable,
      vision_model_ready: !!l.model_ready,
      vision_model: l.vision_model || '',
      detail: l.detail || '',
    }
  }
  const o = c.ollama || {}
  return {
    provider,
    installed: !!o.installed,
    reachable: !!o.reachable,
    vision_model_ready: !!o.vision_model_ready,
    vision_model: o.vision_model || '',
    detail: '',
  }
}

/** 'Ollama' | 'LM Studio' — for a sentence the user reads. */
export function localLlmLabel(caps) {
  return activeLocalLlm(caps).provider === 'lmstudio' ? 'LM Studio' : 'Ollama'
}
