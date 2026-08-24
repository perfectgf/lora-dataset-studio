/* ↩ "Use these improve settings" — looking at a ✨ result you like, make the
 * NEXT improves run the way this one did. JSX-free; `node --test` covers
 * every rule.
 *
 * WHAT CAN BE RESTORED, honestly. An improve candidate records the
 * instruction that ran (`prompt`) and the LoRA rows that chained
 * (`extra_loras`, written by improve_canvas_image from the resolved preset).
 * Those map back onto the two GLOBAL knobs the pass reads: the improve
 * instruction (identity_prompts.klein_improve) and the improve preset
 * (klein.improve_lora_preset). The preset knob stores a NAME and the row
 * stores the RESOLVED files — so the restore matches the rows against
 * today's configured presets, and a preset that was renamed or deleted since
 * the render is REPORTED as unmatched rather than silently dropped: the
 * instruction still restores, the preset stays where it was, and the toast
 * says which half happened. Strength/steps/output-size are not on the row
 * (deliberately — see the candidate's construction), so the button never
 * claims them.
 *
 * A SeedVR2 result has nothing to restore — its stored prompt is the
 * sentinel sentence, not an instruction — and the button is not drawn.
 */

/** How a SeedVR2 candidate's stored prompt begins (lora_test_studio /
 *  face_dataset_service write the same sentence). Stored on rows, so it can
 *  never be reworded without an alias. */
const SEEDVR2_PROMPT_PREFIX = 'SeedVR2 upscale';

/** Whether ↩ can be offered on this row at all. */
export function canRestoreImproveSettings(img) {
  if (!img || !img.derivation_kind) return false;               // not a ✨ result
  const prompt = typeof img.prompt === 'string' ? img.prompt.trim() : '';
  if (!prompt) return false;                                    // nothing recorded
  return !prompt.startsWith(SEEDVR2_PROMPT_PREFIX);             // a restoration ran
}

/** The row's chained LoRAs, parsed leniently: bad JSON or a foreign shape is
 *  "no rows", never a crash — the column is user-database content. */
export function parseExtraLoras(raw) {
  try {
    const list = JSON.parse(raw);
    if (!Array.isArray(list)) return [];
    return list
      .filter((r) => r && typeof r.filename === 'string' && r.filename)
      .map((r) => ({ filename: r.filename, strength: Number(r.strength) }));
  } catch {
    return [];
  }
}

/** The configured preset whose chain IS these rows — same files, same
 *  strengths, same ORDER (order is the chain order; two orders are two
 *  different passes). null when none matches. */
export function matchPresetName(rows, presets) {
  for (const preset of Array.isArray(presets) ? presets : []) {
    const chain = Array.isArray(preset?.loras) ? preset.loras : [];
    if (chain.length !== rows.length || rows.length === 0) continue;
    const same = chain.every((l, i) => l && l.file === rows[i].filename
      && Number(l.strength) === rows[i].strength);
    if (same) return preset.name;
  }
  return null;
}

/**
 * The PUT /api/settings body that makes future improves run like `img` did,
 * plus the report the toast reads. `shipped` is the built-in instruction
 * (identity_prompt_defaults.klein_improve): a restored prompt EQUAL to it is
 * stored as '' — the follow-the-default contract every other editor of this
 * value keeps, or this button would pin users to today's wording forever.
 */
export function restoreImprovePatch({ img, shipped = '', presets = [] } = {}) {
  const prompt = String(img?.prompt || '').trim();
  const followsDefault = prompt === String(shipped || '').trim();
  const rows = parseExtraLoras(img?.extra_loras);
  const matched = rows.length ? matchPresetName(rows, presets) : null;
  const klein = {};
  if (rows.length === 0) {
    klein.improve_lora_preset = '';            // this pass chained nothing
  } else if (matched) {
    klein.improve_lora_preset = matched;
  }
  // Unmatched rows: the preset knob is LEFT ALONE — writing '' would claim
  // "no preset" about a pass that ran one, and there is no name to write.
  return {
    patch: {
      config: {
        identity_prompts: {
          klein_improve: followsDefault ? '' : prompt,
          // An instruction ran on this image, so the restored state sends one.
          klein_improve_enabled: true,
        },
        ...(Object.keys(klein).length ? { klein } : {}),
      },
    },
    report: {
      followsDefault,
      preset: matched,
      hadLoras: rows.length > 0,
      unmatchedLoras: rows.length > 0 && !matched,
    },
  };
}

/** What the toast says — both halves, including the one that could NOT be
 *  restored, because a silent half-restore reads as a full one. */
export function restoreImproveMessage(report = {}) {
  const promptPart = report.followsDefault
    ? 'Improve instruction set back to the built-in default'
    : 'Improve instruction restored from this image';
  if (report.unmatchedLoras) {
    return `${promptPart}. Its LoRAs match none of your presets (renamed or `
      + 'deleted since) — the preset setting was left unchanged.';
  }
  const presetPart = report.hadLoras
    ? ` · LoRA preset set to “${report.preset}”`
    : ' · LoRA preset set to None (this pass chained none)';
  return `${promptPart}${presetPart} — app-wide, for every ✨ improve from now on.`;
}
