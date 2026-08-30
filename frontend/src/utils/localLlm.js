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
      // LM Studio DOES have an installed-but-stopped state now: its CLI sits at a
      // fixed per-user path, so the app can both detect the install and start the
      // server. Before that this tracked `reachable`, which made every gate skip
      // the "start it for me" rung -- correct then, wrong the moment a button
      // existed to offer.
      installed: !!l.installed,
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

/**
 * The words a model picker needs, taken from the `provider` its OWN list came back
 * with — never from `caps`.
 *
 * That distinction is the point: the label has to describe the list on screen. A
 * picker reading the capability snapshot instead would name one provider while
 * showing the other's models for as long as a refresh takes, and the four pickers
 * (dataset options, bank options, Caption Lab, ✨ Enhance) refresh independently.
 *
 * Ollama PULLS a model onto the machine; LM Studio LOADS one that is already there
 * and offers no gesture this app can drive. So `canPull` gates the pull form
 * itself, and the down sentence names the control that actually exists: a button
 * in Settings for one, a menu inside another application for the other.
 */
export function modelPickerCopy(provider) {
  const isLmStudio = provider === 'lmstudio'
  return {
    label: isLmStudio ? 'LM Studio' : 'Ollama',
    canPull: !isLmStudio,
    modelLabel: isLmStudio ? 'LM Studio vision model' : 'Ollama vision model',
    // The three run-window tooltips. They live HERE, not inline in the JSX, for a
    // reason the Bank's frozen surface inventory states: a computed label yields
    // nothing to its extractor, so a sentence that has to vary is covered by a unit
    // test on the helper that builds it instead of by a frozen literal.
    registerHint: 'How captions name nude or sexual content. Explicit needs an '
      + `uncensored (abliterated) ${isLmStudio ? 'LM Studio' : 'Ollama'} vision model. `
      + 'Richer, more explicit captions also make the search find more.',
    perRunHint: `Which ${isLmStudio ? 'loaded' : 'pulled'} `
      + `${isLmStudio ? 'LM Studio' : 'Ollama'} vision model writes this run. Your `
      + 'Settings model stays the default and is not changed. Which model writes a '
      + 'caption is not a matter of taste: one that describes things in evasive terms '
      + 'produces captions that are about something slightly other than the images.',
    inertHint: `Only used when the engine can reach ${isLmStudio ? 'LM Studio' : 'Ollama'}.`,
    down: isLmStudio
      ? 'LM Studio isn’t reachable — open it, go to Developer and press Start Server.'
      : 'Ollama isn’t reachable — start it from Settings ▸ Local tools to list or pull models.',
  }
}
