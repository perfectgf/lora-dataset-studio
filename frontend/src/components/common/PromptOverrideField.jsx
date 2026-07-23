/* ONE editable box per built-in prompt — shared by Settings ▸ Image engines and
   the workspace's Extra-refs modal, so the two surfaces can never drift.

   The box always shows the text that is ACTUALLY in use: the shipped default
   when nothing is overridden, the override otherwise. Editing it makes an
   override; clearing it (or typing the default back verbatim) goes back to
   following the default. That "back to ''" collapse happens in
   normalizePromptOverride on EVERY keystroke, which is what stops a user who
   merely looked at the default from silently persisting a frozen copy of it and
   never receiving a future improvement — see promptOverride.js. */
import { normalizePromptOverride, promptBoxText } from './promptOverride.js';

const RESET_BTN = 'rounded-md border border-border-strong px-2 py-1 text-xs font-medium ' +
  'text-content hover:bg-surface-raised disabled:opacity-50';

const BOX_CLASS =
  'mt-1 w-full rounded-md border border-border-strong bg-surface-raised px-3 py-2 text-sm text-content ' +
  'placeholder:text-content-subtle focus:border-primary focus:outline-none';

export default function PromptOverrideField({
  id,                     // DOM id — also the help-registry focus target
  label,
  desc = null,
  value,                  // the STORED value ('' = following the default)
  defaultText = '',       // the shipped default, read-only, from the settings payload
  onChange,               // receives the NORMALISED value ('' when it equals the default)
  disabled = false,
  rows = 4,
  badge = null,           // optional right-hand marker (e.g. "used by your engine")
  className = '',
}) {
  const custom = !!normalizePromptOverride(value, defaultText);
  return (
    <div className={className}>
      <div className="flex items-baseline gap-2">
        <label htmlFor={id} className="block text-sm font-medium text-content">{label}</label>
        {badge}
      </div>
      {desc && <p className="mb-1 text-xs text-content-muted">{desc}</p>}
      <textarea
        id={id}
        rows={rows}
        value={promptBoxText(value, defaultText)}
        disabled={disabled}
        onChange={(e) => onChange(normalizePromptOverride(e.target.value, defaultText))}
        placeholder={defaultText || 'Leave empty to use the built-in default.'}
        className={`${BOX_CLASS} font-mono leading-relaxed disabled:opacity-50`}
      />
      <div className="mt-1 flex items-center justify-between gap-2">
        <span className="text-xs text-content-subtle">
          {custom
            ? 'Custom override — this exact text is used instead of the built-in default.'
            : 'Following the built-in default — improvements to it reach you automatically.'}
        </span>
        {custom && !disabled && (
          <button type="button" onClick={() => onChange('')} className={RESET_BTN}>
            Reset to default
          </button>
        )}
      </div>
    </div>
  );
}
