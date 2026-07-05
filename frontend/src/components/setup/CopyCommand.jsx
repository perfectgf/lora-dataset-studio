import { useState } from 'react'

export default function CopyCommand({ command }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(command)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch { /* clipboard blocked — the command is visible to copy by hand */ }
  }
  return (
    <div className="flex items-center gap-2">
      <code className="flex-1 overflow-x-auto rounded-md border border-border bg-surface-raised px-2 py-1 text-[11px] text-content">
        {command}
      </code>
      <button type="button" onClick={copy}
        className="shrink-0 rounded-md border border-border-strong px-2 py-1 text-[11px] text-content hover:bg-surface-raised">
        {copied ? 'Copied' : 'Copy'}
      </button>
    </div>
  )
}
