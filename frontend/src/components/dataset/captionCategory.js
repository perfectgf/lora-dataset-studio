const PROSE_STOP_WORDS = new Set([
  'the', 'and', 'with', 'that', 'this', 'from', 'into', 'onto', 'over', 'under',
  'their', 'there', 'while', 'where', 'which', 'who', 'whose', 'are', 'was',
  'were', 'has', 'have', 'had', 'for', 'but', 'not', 'its', 'his', 'her',
  'him', 'she', 'they', 'them', 'you', 'your', 'our', 'out', 'off', 'near',
  'through', 'between', 'beside', 'behind', 'front', 'image', 'scene', 'shows',
  'showing', 'captures', 'featuring', 'wearing', 'against', 'within', 'being',
  'person', 'woman', 'women', 'man', 'men', 'someone', 'something', 'each',
  'some', 'very', 'more', 'most', 'also', 'about', 'above', 'below',
])

function proseTerms(caption) {
  const words = String(caption || '').toLowerCase().match(/[\p{L}\p{N}][\p{L}\p{N}'’-]{2,}/gu) || []
  return words.filter((word) => !PROSE_STOP_WORDS.has(word))
}

export function captionFrequencyEntries(captions, mode = 'prose', limit = 30) {
  const counts = new Map()
  for (const caption of captions || []) {
    const rawTerms = mode === 'booru'
      ? String(caption || '').split(',').map((tag) => tag.trim().toLowerCase()).filter(Boolean)
      : proseTerms(caption)
    // The count means “in N captions”, not raw repetitions inside one caption.
    for (const term of new Set(rawTerms)) counts.set(term, (counts.get(term) || 0) + 1)
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, limit)
}

const CATEGORY_COPY = {
  character: {
    frequencyHelp: 'Repeated details can become baked into the character. Remove unwanted outfits, poses or settings so they remain prompt-controllable; identity words are handled by the identity-leak check.',
    leakSummary: 'Identity terms should stay out of captions',
  },
  concept: {
    frequencyHelp: 'These are recurring subject and scene terms. The concept itself must stay absent (the concept-leak check handles that); remove unrelated repeated details that could become tied to the trigger.',
    leakSummary: 'Concept terms should stay out of captions',
  },
  style: {
    frequencyHelp: 'Repeated subjects and objects can bias a style LoRA toward specific content. Keep useful content descriptions, balance accidental repetition, and keep aesthetic, medium or artist terms out so the visual style is learned from the images.',
    leakSummary: 'Aesthetic terms should stay out of captions',
  },
}

export function captionCategoryCopy(kind = 'character', mode = 'prose') {
  const category = CATEGORY_COPY[kind] || CATEGORY_COPY.character
  const tagMode = mode === 'booru'
  return {
    ...category,
    frequencyTitle: tagMode ? 'Most frequent tags' : 'Most frequent words',
    frequencyItem: tagMode ? 'tag' : 'word',
    filterPlaceholder: tagMode ? 'tag to filter by…' : 'word to filter by…',
  }
}

export function recaptionConfirmation(kind = 'character', count = 0) {
  const rule = kind === 'concept'
    ? 'The new captions will describe the scene while leaving the recurring concept unspoken.'
    : kind === 'style'
      ? 'The new captions will describe image content while leaving the aesthetic/style unspoken.'
      : 'The new captions will describe the scene without describing the character identity.'
  return `Re-captioning overwrites the ${count} existing caption(s). ${rule} Continue?`
}
