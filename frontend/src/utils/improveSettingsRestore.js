/* ↩ "Use these improve settings" — looking at a ✨ result you like, make the
 * NEXT improves run the way this one did. JSX-free; `node --test` covers
 * every rule.
 *
 * WHAT CAN BE RESTORED, honestly. An improve candidate records the
 * instruction that ran (`prompt`), the LoRA rows that chained
 * (`extra_loras`) and — on rows made since the column exists — the whole run
 * profile (`improve_profile`: consistency strength, steps, base-LoRA
 * strength, output megapixels, the model pin, the preset NAME). Those map
 * back onto the GLOBAL knobs the pass reads. The preset restores by its
 * recorded name when that preset still exists, else by matching the rows
 * against today's presets — and rows matching nothing are REPORTED rather
 * than silently dropped: the rest still restores, the preset stays where it
 * was, and the toast says which halves happened. A row from before the
 * profile column restores instruction + preset only, and the toast says
 * that too.
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
/** The recorded run profile, when this row carries one (improve_profile is
 *  written at enqueue time since 2026-08-24; older rows have none). */
function runProfile(img) {
  const p = img?.improve_profile;
  return p && typeof p === 'object' ? p : null;
}

const finite = (v) => (Number.isFinite(Number(v)) ? Number(v) : null);

export function restoreImprovePatch({ img, shipped = '', presets = [] } = {}) {
  const prompt = String(img?.prompt || '').trim();
  const followsDefault = prompt === String(shipped || '').trim();
  const rows = parseExtraLoras(img?.extra_loras);
  const profile = runProfile(img);
  // The preset: the RECORDED NAME wins when it still exists (exact, survives a
  // strength tweak inside the preset), else the rows are matched by content —
  // which is also all a row from before the profile column can offer.
  const recordedName = profile && typeof profile.lora_preset === 'string'
    && profile.lora_preset
    && (Array.isArray(presets) ? presets : []).some((p) => p?.name === profile.lora_preset)
    ? profile.lora_preset : null;
  const matched = recordedName || (rows.length ? matchPresetName(rows, presets) : null);
  const klein = {};
  if (rows.length === 0) {
    klein.improve_lora_preset = '';            // this pass chained nothing
  } else if (matched) {
    klein.improve_lora_preset = matched;
  }
  // Unmatched rows: the preset knob is LEFT ALONE — writing '' would claim
  // "no preset" about a pass that ran one, and there is no name to write.

  // The knobs — only what the row actually RECORDED, value by value: a junk
  // field degrades to "not restored", never to a default silently written.
  if (profile) {
    const knobs = [
      ['consistency_strength', 'improve_consistency_strength'],
      ['steps', 'improve_steps'],
      ['base_lora_strength', 'improve_base_lora_strength'],
      ['megapixels', 'improve_megapixels'],
    ];
    for (const [from, to] of knobs) {
      const v = finite(profile[from]);
      if (v !== null) klein[to] = v;
    }
    // The model PIN (klein.unet — one setting for generation and improve, by
    // the app's own design). null recorded = the pass ran on auto → ''.
    if ('klein_model' in profile) klein.unet = profile.klein_model || '';
  }
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
      // Whether strength/steps/output-size/model came along — a row from
      // before the profile column restores less, and the toast says so.
      knobs: !!profile,
    },
  };
}

/** What the toast says — every half, including the ones that could NOT be
 *  restored, because a silent partial restore reads as a full one. */
export function restoreImproveMessage(report = {}) {
  const promptPart = report.followsDefault
    ? 'Improve instruction set back to the built-in default'
    : 'Improve instruction restored from this image';
  // strength/steps/output-size/model — recorded on rows made since the
  // profile column exists; an older row restores less and says so.
  const knobsPart = report.knobs
    ? ' · strength, steps, output size and model set to what made it'
    : '. Strength, steps and output size were not recorded on this image '
      + '(it predates that) and were left unchanged';
  if (report.unmatchedLoras) {
    return `${promptPart}${knobsPart}. Its LoRAs match none of your presets `
      + '(renamed or deleted since) — the preset setting was left unchanged.';
  }
  const presetPart = report.hadLoras
    ? ` · LoRA preset set to “${report.preset}”`
    : ' · LoRA preset set to None (this pass chained none)';
  return `${promptPart}${presetPart}${knobsPart} — app-wide, for every ✨ `
    + 'improve from now on.';
}
