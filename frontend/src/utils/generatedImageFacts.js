/* 🔍 What a generated image was actually MADE with — the reading order.

   The lightbox used to print one paragraph across the full width of the screen:

     step 2500 · seed 208607443 · strength 0 · <forty lines of prompt>

   Three facts you look for, buried at the head of a wall of text running 2 000
   pixels wide. This module is the fix, and it lives here rather than in the JSX
   because `node --test` cannot parse JSX and the interesting part is the
   decisions, not the markup:

     • HEADLINE — the three answers to "what am I looking at": step, seed,
       strength. They are what you compare two renders on and what you re-use,
       so they are the only facts allowed at the top.
     • SETTINGS — everything that DECIDES the picture and was persisted per cell
       but never shown: sampler, scheduler, CFG, steps, the base model, the LoRA
       file, the always-on LoRAs, the format. On a board whose whole job is
       comparing checkpoints, hiding these turns "different sampler" into
       "different checkpoint" for the reader.
     • PROSE — the prompt and the negative, last, bounded and foldable. A prompt
       is long by nature; it must not push the facts off the screen.

   Every value is a string here. Formatting a number is a decision (a seed is
   never grouped by thousands — you copy it), and a decision belongs in a tested
   function rather than in a template. */

const has = (v) => v !== null && v !== undefined && v !== '';

/** A LoRA path as published (`z image\Ada-2500.safetensors`) reduced to what a
 *  human reads: the file, without its folder or extension. The folder is the
 *  family and the pill already says the family. */
export function checkpointFileLabel(value) {
  if (!has(value)) return '';
  const tail = String(value).split(/[\\/]/).pop() || '';
  return tail.replace(/\.(safetensors|ckpt|pt|sft)$/i, '');
}

/** `extra_loras` is stored as a JSON list of {filename, strength}. Bad JSON is
 *  an absent line, never a crash — this row is a nicety and the picture is not.
 *
 *  `only` splits the populations that share this column: entries tagged
 *  `combined` are the OTHER LoRAs of a combined stack (co-stars of the run,
 *  each chosen at its own weight), entries tagged `external` are external LoRAs
 *  sourced from outside, everything else is an always-on style LoRA. They read
 *  as opposites to a user, so they cannot share one label. */
export function extraLoraSummary(raw, { only = 'all' } = {}) {
  if (!has(raw)) return '';
  let list = raw;
  if (typeof raw === 'string') {
    try { list = JSON.parse(raw); } catch { return ''; }
  }
  if (!Array.isArray(list) || !list.length) return '';
  if (only === 'combined') list = list.filter((l) => l?.combined);
  else if (only === 'external') list = list.filter((l) => l?.external);
  else if (only === 'always-on') list = list.filter((l) => !l?.combined && !l?.external);
  if (!list.length) return '';
  return list
    .map((l) => {
      const name = checkpointFileLabel(l?.filename);
      if (!name) return '';
      const s = Number(l?.strength);
      return Number.isFinite(s) ? `${name} @ ${s}` : name;
    })
    .filter(Boolean)
    .join(', ');
}

/** A number as a fact: kept exact, never localised. A seed with thin spaces in
 *  it cannot be pasted back into the seed field, which is the one thing anybody
 *  ever does with a seed. */
const exact = (v) => (has(v) ? String(v) : '');

/**
 * The three headline facts. `copy` marks the ones worth a one-click copy — the
 * seed is the whole reason this panel exists (you re-play a seed), and it used
 * to be a run of digits in the middle of a paragraph.
 */
export function imageHeadlineFacts(img) {
  const out = [];
  if (has(img?.step)) out.push({ key: 'step', label: 'Step', value: exact(img.step) });
  out.push({
    key: 'seed', label: 'Seed', value: has(img?.seed) ? exact(img.seed) : '—',
    copy: has(img?.seed) ? exact(img.seed) : null,
  });
  if (has(img?.strength)) {
    out.push({ key: 'strength', label: 'LoRA strength', value: exact(img.strength) });
  }
  return out;
}

/**
 * Everything else that decided this picture, in the order it is looked up.
 * Absent fields produce NO row: a table of dashes reads as "the app lost this",
 * when the truth is that this run never recorded it.
 */
export function imageSettingFacts(img) {
  const rows = [];
  const push = (key, label, value) => { if (has(value)) rows.push({ key, label, value }); };
  push('checkpoint', 'LoRA file', checkpointFileLabel(img?.checkpoint));
  push('base_model', 'Base model', checkpointFileLabel(img?.base_model));
  push('sampler', 'Sampler', img?.sampler);
  push('scheduler', 'Scheduler', img?.scheduler);
  push('cfg', 'CFG', exact(img?.cfg));
  push('steps', 'Sampling steps', exact(img?.steps));
  push('aspect', 'Format', img?.aspect);
  // Clé `combined_loras` INCHANGÉE (elle vient du backend et sert de clé de
  // ligne) ; seul le libellé suit le renommage 🧬 Combine → 🧬 Blend.
  push('combined_loras', 'Blended LoRAs',
    extraLoraSummary(img?.extra_loras, { only: 'combined' }));
  push('external_loras', 'External LoRAs',
    extraLoraSummary(img?.extra_loras, { only: 'external' }));
  push('extra_loras', 'Always-on LoRAs',
    extraLoraSummary(img?.extra_loras, { only: 'always-on' }));
  if (has(img?.face_score)) {
    const n = Number(img.face_score);
    push('face_score', 'Face similarity', Number.isFinite(n) ? n.toFixed(3) : '');
  }
  if (has(img?.created_at)) {
    // Sliced, not parsed: an ISO stamp is already sorted and unambiguous, and a
    // locale-formatted date in a settings table is one more thing to mistrust.
    push('created_at', 'Generated', String(img.created_at).slice(0, 16).replace('T', ' '));
  }
  return rows;
}

/** The prose blocks, last and foldable. */
export function imagePromptBlocks(img) {
  const out = [];
  if (has(img?.prompt)) out.push({ key: 'prompt', label: 'Prompt', text: String(img.prompt) });
  if (has(img?.negative)) {
    out.push({ key: 'negative', label: 'Negative prompt', text: String(img.negative) });
  }
  return out;
}

// Beyond this many characters a prompt opens FOLDED. Roughly four lines at the
// panel's bounded reading width — short enough that a one-line prompt is never
// hidden behind a "show more" nobody needed, long enough that the forty-line
// case (the one that broke the old lightbox) never opens as a wall.
export const PROMPT_FOLD_CHARS = 240;

/** Whether a block opens folded, and what its toggle should say. */
export function promptFold(text, expanded) {
  const long = String(text || '').length > PROMPT_FOLD_CHARS;
  return {
    foldable: long,
    collapsed: long && !expanded,
    label: expanded ? 'Show less' : 'Show full prompt',
  };
}

/**
 * The one-line summary for a thumbnail's tooltip and for anywhere too small for
 * the panel. Same facts, same order, no prompt — a tooltip that carries forty
 * lines is a tooltip nobody can dismiss.
 */
export function imageFactsLine(img) {
  return imageHeadlineFacts(img).map((f) => `${f.label} ${f.value}`).join(' · ');
}
