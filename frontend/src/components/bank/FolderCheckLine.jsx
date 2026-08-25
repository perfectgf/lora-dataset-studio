import { folderCheckNote } from './bankSync'
import { RefreshCw } from 'lucide-react';

/** 🗃️ Bank list — how fresh these cards are, and the one click that refreshes them.
 *
 * The list used to re-walk every bank's source folder before rendering: a full
 * inventory of the whole library paid on every navigation to a page people
 * often only pass through (690-1 190 ms on a real 8-bank / 86 493-image
 * library). It no longer does — a folder is re-checked when its bank is OPENED,
 * or from here.
 *
 * That trade is only acceptable because this line exists. A list that is
 * silently late is worse than a list that is slow: the user would read counts
 * as facts. So the page says what it knows and offers the walk, instead of
 * doing it behind their back on every visit.
 *
 * flex-wrap + a growing sentence: at 400 px the text takes the row and the
 * button drops underneath it, rather than squeezing into two characters. */
export default function FolderCheckLine({ banks, busy = false, onRescan }) {
  const note = folderCheckNote(banks)
  if (!note) return null
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
      <p className={`min-w-0 grow text-xs ${note.stale ? 'text-amber-300/90' : 'text-content-subtle'}`}>
        {note.text}
      </p>
      <button type="button" onClick={onRescan} disabled={busy}
        title="Walk every bank's source folder now and pick up the images added to it"
        className="shrink-0 rounded-md border border-border bg-surface-raised px-3 py-1.5 text-xs font-semibold text-content hover:bg-surface disabled:opacity-50">
        <RefreshCw aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />{busy ? 'Checking folders…' : 'Rescan folders'}
      </button>
    </div>
  )
}
