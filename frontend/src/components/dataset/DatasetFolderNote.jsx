import { useState } from 'react'
import { Folder } from 'lucide-react';

/** Where this dataset's images live on disk — shown, and copyable.
 *
 * It used to be shown NOWHERE. People who needed it (to check a file, to point
 * another tool at it) went digging for it in the file manager, and some then
 * pasted it into 🗃 "create a bank" — which registered a bank over the dataset's
 * LIVE files, so that bank's 🗑 Delete rejected deleted images out of the
 * dataset. The app now refuses that paste (backend `services/path_guard.py`),
 * but a refusal is the second line: not making people hunt for the path is the
 * first, and saying whose folder it is right next to it is the second half of
 * the same sentence.
 *
 * It is the last entry of the dataset header's ⋯ More menu (DatasetWorkspace):
 * the path in full, a Copy button, and the sentence about banks. It was first
 * a card across the top of every section; an answer wanted now and then does
 * not have to stand above the work all the time to be findable. */
export default function DatasetFolderNote({ path }) {
  const [copied, setCopied] = useState(false)
  if (!path) return null
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(path)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch { /* clipboard denied — the path is still readable on screen */ }
  }
  return (
    <div className="mt-1 min-w-0 border-t border-border px-2.5 pb-1.5 pt-2.5 text-xs">
      <div className="flex min-w-0 items-center gap-2">
        <span className="inline-flex shrink-0 items-center gap-1.5 text-content-muted"><Folder aria-hidden="true" className="h-4 w-4" /> Images folder</span>
        <button type="button" onClick={copy}
          aria-label="Copy the dataset's images folder path"
          className="ml-auto min-h-10 shrink-0 rounded border border-border px-2 py-0.5 text-content-muted hover:bg-surface-raised hover:text-content lg:min-h-0">
          {copied ? '✓ Copied' : '⧉ Copy'}
        </button>
      </div>
      {/* The whole path, wrapped: the menu is narrow, and a path cut short is
          the one thing nobody opens this for. */}
      <code className="mt-1.5 block break-all font-mono text-content-muted" title={path}>
        {path}
      </code>
      <p className="mb-0 mt-1.5 text-content-subtle">
        This folder belongs to the dataset — don’t use it as an image bank’s source.
        A bank points at a live folder, so its Delete rejected would delete these
        images. To re-triage them, use Import to bank: it copies.
      </p>
    </div>
  )
}
