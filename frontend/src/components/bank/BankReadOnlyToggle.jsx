import { HelpBadge } from '../../help/HelpMode'

export default function BankReadOnlyToggle({ checked, onChange, disabled = false }) {
  return (
    <div className="flex shrink-0 items-center gap-1">
      <label className="flex min-h-10 lg:min-h-0 cursor-pointer items-center gap-2 text-sm text-content-muted"
        title="Protect source files from deletion and new imports. Triage, captions, edits to working copies and promotion still work.">
        <input type="checkbox" checked={checked} disabled={disabled}
          onChange={(e) => onChange(e.target.checked)} className="accent-emerald-500" />
        Read-only source
      </label>
      <HelpBadge topic="bank-read-only" />
    </div>
  )
}
