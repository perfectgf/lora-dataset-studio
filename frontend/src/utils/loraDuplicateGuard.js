/** The preset row that names a LoRA the engine ALREADY loads — predicted in the
 *  editor, where the row is written, instead of only in the server log.
 *
 *  Both local engines pin one LoRA outside the presets: Klein chains
 *  `klein.consistency_lora` at `klein.consistency_strength`, Krea loads
 *  `krea.identity_lora` at `krea.identity_lora_strength`. A preset row naming
 *  that same file is DROPPED by the backend on purpose — chaining the identical
 *  LoRA twice sums both strengths into one delta well past what the file was
 *  trained for (measured as visible macro-blocking; reported by @waltm on
 *  Discord). See klein_edit_helper.build_workflow and
 *  krea_edit_helper._existing_generation_lora_rows.
 *
 *  Until now that drop happened in silence: nothing on screen, one warning line
 *  in the server log. A user who had written exactly one preset row, and had
 *  written the consistency LoRA into it, got a run with no extra LoRA at all and
 *  no reason why — which reads as "the app ignores my settings".
 *
 *  This module exists so the FRONT PREDICTS EXACTLY WHAT THE BACK WILL DO.
 *  The backend compares `os.path.normcase(os.path.normpath(a)) ==
 *  os.path.normcase(os.path.normpath(b))`, so a '/' instead of a '\\', a
 *  doubled separator, a './' segment or a difference of case must NOT dodge the
 *  warning — otherwise the editor says "fine" about a row the server discards.
 *
 *  Case: the comparison here is always case-insensitive, while `normcase` is a
 *  no-op on a POSIX server. That is deliberate and it does not over-promise: on
 *  a case-sensitive filesystem a case-different spelling names a file that is
 *  not on disk, and the row is dropped anyway ("not found under any loras root
 *  — skipped"). The row is ignored either way; only the log's reason differs,
 *  which is why the warning says the row is ignored rather than naming a
 *  mechanism it cannot know.
 *
 *  What it deliberately does NOT claim: an ABSOLUTE path aliasing the same file
 *  (Klein resolves preset rows through resolve_model_ref before comparing, so
 *  the server can catch a case the front cannot). Missing a warning is a much
 *  smaller failure than inventing one.
 */

/** Mirror of `os.path.normcase(os.path.normpath(value))` for the model
 *  references these fields hold (relative loader names, occasionally an absolute
 *  path). Separators unify to '/', '.' segments vanish, '..' pops the previous
 *  segment, repeated separators collapse, a trailing separator is dropped, and
 *  the result is lowercased. Junk input yields ''. */
export function normalizeLoraRef(value) {
  const raw = typeof value === 'string' ? value.trim() : '';
  if (!raw) return '';
  const unified = raw.replace(/\\/g, '/');
  // Keep a leading '/' (or a drive prefix) so an absolute path never compares
  // equal to the relative name of the same tail.
  const rooted = /^(\/|[A-Za-z]:\/)/.test(unified);
  const prefix = rooted ? (unified.match(/^(\/|[A-Za-z]:\/)/)[0]) : '';
  const out = [];
  for (const part of unified.slice(prefix.length).split('/')) {
    if (!part || part === '.') continue;
    if (part === '..') {
      // Popping past the root is meaningless; python's normpath keeps the
      // leading '..' on a relative path and swallows it on an absolute one.
      if (out.length && out[out.length - 1] !== '..') out.pop();
      else if (!rooted) out.push('..');
      continue;
    }
    out.push(part);
  }
  const joined = prefix + out.join('/');
  return (joined || (rooted ? prefix : '.')).toLowerCase();
}

/** True when `rowFile` names the same file as `fixedLora` — i.e. the backend
 *  will drop this preset row. Blank on either side is never a match. */
export function isFixedLoraDuplicate(rowFile, fixedLora) {
  const a = normalizeLoraRef(rowFile);
  const b = normalizeLoraRef(fixedLora);
  return !!a && !!b && a === b;
}

/** The sentence shown on the offending row. Engine-specific because the two
 *  fixed slots have different names and different strength settings, and a
 *  warning that cannot name the setting to change is a dead end. */
const FIXED_LORA_SLOTS = {
  klein: {
    label: 'consistency LoRA',
    strengthSetting: 'Consistency strength',
  },
  krea: {
    label: 'identity edit LoRA',
    strengthSetting: 'Identity LoRA strength',
  },
};

export function fixedLoraDuplicateWarning(engine) {
  const slot = FIXED_LORA_SLOTS[engine];
  if (!slot) return '';
  return `Ignored: this is the ${slot.label} the engine already loads. `
    + `It is applied once at its own ${slot.strengthSetting}, and chaining it a `
    + 'second time here would add both strengths together — past what the file '
    + 'was trained for (blocky, posterized output). Point this row at a different '
    + `file, or change the ${slot.strengthSetting} instead.`;
}
