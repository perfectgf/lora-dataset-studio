// Pure copy for the "change the dataset kind" confirmation, kept out of the modal
// so node --test can exercise it. Changing a dataset's kind (character/concept/
// style) after creation flips the caption strategy and which panels show, but
// deletes nothing — the summary spells out exactly WHAT CHANGES and WHAT IS KEPT
// so the switch is honest, not magic (see backend update_dataset_settings).

export const KIND_LABELS = {
  character: '🧑 Character',
  concept: '💡 Concept',
  style: '🎨 Style',
};

// Mirrors the server rule (normalize_kind): only 'concept'/'style' are real
// values, everything else (incl. NULL/'') is a character.
export function normalizeKindLabel(kind) {
  const k = String(kind || '').toLowerCase();
  return k === 'concept' || k === 'style' ? k : 'character';
}

// The caption strategy each kind teaches — the ONE thing that always differs
// across a switch (same wording as recaptionConfirmation in captionCategory.js).
function captionStrategyLine(to) {
  if (to === 'concept') {
    return 'Captions will describe the scene while leaving the recurring concept unspoken, so it binds to the trigger.';
  }
  if (to === 'style') {
    return 'Captions will describe image content only, leaving the aesthetic/style unspoken.';
  }
  return 'Captions will describe the scene without naming the character’s identity (face, hair, skin).';
}

/**
 * What switching from `prevKind` to `nextKind` changes and what it preserves.
 * Returns null when the kind is unchanged. `hasCaptions` toggles the re-caption
 * nudge (existing captions were written under the OLD strategy and are NOT
 * rewritten automatically). Pure — no side effects, no live values embedded.
 */
export function kindSwitchSummary(prevKind, nextKind, { hasCaptions = false } = {}) {
  const from = normalizeKindLabel(prevKind);
  const to = normalizeKindLabel(nextKind);
  if (from === to) return null;

  const changes = [captionStrategyLine(to)];
  const changeKeys = [`captionStrategy.${to}`];

  // Character-only build surfaces: the reference photo, the AI variation
  // generator and face analysis. They appear when becoming a character and
  // disappear when leaving it (concept/style build from Import or Scrape).
  if (to === 'character') {
    changes.push('The Reference photo, Generate variations and Face analysis panels become available (build the set from a reference).');
    changeKeys.push('characterPanelsAvailable');
  } else if (from === 'character') {
    changes.push('The Reference photo, Generate variations and Face analysis panels are hidden — build the set from Import or Scrape instead.');
    changeKeys.push('characterPanelsHidden');
  }

  // Trigger role.
  if (to === 'style') {
    changes.push('No activation trigger: a style is always on once loaded — control it with the LoRA weight. Your trigger word is kept but no longer written into captions or prompts.');
    changeKeys.push('styleTrigger');
  } else if (from === 'style') {
    changes.push('The trigger word returns: the stored token is prefilled — set the word you’ll type in prompts to summon this LoRA.');
    changeKeys.push('triggerReturns');
  }

  // Concept needs its omit-description.
  if (to === 'concept') {
    changes.push('A concept description is required — the recurring thing every caption must omit.');
    changeKeys.push('conceptDescription');
  }

  // Fidelity is character-only.
  if (from === 'character' && to !== 'character') {
    changes.push('The face / body fidelity setting no longer applies (it is character-only).');
    changeKeys.push('fidelityUnavailable');
  }

  const preserved = [
    'Every image, its caption text, keep/reject status, face scores and watermark work stay exactly as they are.',
    'Past training runs and checkpoints keep their identity — runs are named by the model family and trigger, never the kind.',
  ];
  const preservedKeys = ['images', 'runs'];
  if (from === 'concept' || to === 'concept') {
    preserved.push('Your concept description is remembered, so switching back restores it.');
    preservedKeys.push('conceptDescription');
  }

  return {
    from, to, changes, changeKeys, preserved, preservedKeys,
    recaption: Boolean(hasCaptions),
  };
}
