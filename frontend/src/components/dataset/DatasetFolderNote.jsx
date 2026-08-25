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
 * Deliberately quiet — one line, monospaced, truncated, with the full path in
 * the title and in the clipboard. At 400 px the path truncates instead of
 * pushing the row sideways. */
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
    <div className="min-w-0 rounded-lg border border-border bg-surface px-3 py-2 text-xs">
      <div className="flex min-w-0 items-center gap-2">
        <span className="inline-flex shrink-0 items-center gap-1.5 text-content-subtle"><Folder aria-hidden="true" className="h-3.5 w-3.5" /> Images folder</span>
        <code className="min-w-0 grow truncate font-mono text-content-muted" title={path}>
          {path}
        </code>
        <button type="button" onClick={copy}
          aria-label="Copy the dataset's images folder path"
          className="shrink-0 rounded border border-border px-2 py-0.5 text-content-muted hover:bg-surface-raised hover:text-content">
          {copied ? '✓ Copied' : '⧉ Copy'}
        </button>
      </div>
      <p className="mt-1 text-content-subtle">
        This folder belongs to the dataset — don’t use it as an image bank’s source.
        A bank points at a live folder, so its Delete rejected would delete these
        images. To re-triage them, use Import to bank below: it copies.
      </p>
    </div>
  )
}
