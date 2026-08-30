import { useEffect, useState } from 'react'
import { Link } from 'react-router'
import Markdown from './Markdown'
import { getHelpTopic, topicGuideHref } from '../../help/helpRegistry'
import { sliceGuideSection } from '../../utils/guideSection'

/** ⓘ One guide section, in a modal, where the button is.
 *
 * The explanation of a pass must not require LEAVING the page the pass runs on:
 * the guide anchor is a good destination for reading, and a bad one for someone
 * mid-triage who only wants to know what "Safe zone" does before pressing it
 * (asked from a phone, where the round-trip costs the scroll position too).
 *
 * No second copy of any text: the modal renders the SAME markdown section the
 * Guide renders, sliced by the topic's own anchor — a doc edit updates both
 * surfaces, and a topic whose anchor rots fails the help-registry contract
 * test long before anyone opens this.
 *
 * The chapter is imported LAZILY on first open: using-the-app.md is a large
 * file, and every page mounting this component must not carry the whole guide
 * in its chunk for a modal most sessions never open.
 */
const CHAPTER_LOADERS = {
  'getting-started': () => import('../../../../docs/guide/getting-started.md?raw'),
  'using-the-app': () => import('../../../../docs/guide/using-the-app.md?raw'),
  'dataset-guide': () => import('../../../../docs/DATASET_GUIDE.md?raw'),
  'settings-reference': () => import('../../../../docs/guide/settings-reference.md?raw'),
  'troubleshooting': () => import('../../../../docs/guide/troubleshooting.md?raw'),
}

export default function GuideSectionModal({ topic, onClose }) {
  const t = getHelpTopic(topic)
  const [section, setSection] = useState(null)   // null = loading, '' = not found

  useEffect(() => {
    let alive = true
    const load = CHAPTER_LOADERS[t?.guide?.chapter]
    if (!t || !load) { setSection(''); return undefined }
    load().then((mod) => {
      if (alive) setSection(sliceGuideSection(mod.default, t.guide.anchor))
    }).catch(() => { if (alive) setSection('') })
    return () => { alive = false }
  }, [t])

  if (!t) return null
  return (
    <div role="dialog" aria-modal="true" aria-label={`About: ${t.title}`}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-3 sm:p-4"
      onClick={onClose}>
      <div onClick={(e) => e.stopPropagation()}
        className="flex max-h-[85vh] w-full max-w-2xl flex-col rounded-xl border border-border bg-surface-overlay shadow-2xl">
        <div className="flex items-center gap-2 border-b border-border px-4 py-2.5">
          <h2 className="min-w-0 flex-1 truncate text-sm font-bold text-content">
            {t.title}
          </h2>
          <button type="button" onClick={onClose} aria-label="Close the explanation"
            className="min-h-10 shrink-0 rounded-md border border-border px-3 py-1 text-sm text-content hover:bg-surface lg:min-h-0">
            ✕
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto p-4">
          {section === null && (
            <p className="text-sm text-content-muted">Loading…</p>
          )}
          {section === '' && (
            <p className="text-sm text-content-muted">
              This section moved — open the guide below to read it there.
            </p>
          )}
          {!!section && <Markdown source={section} variant="guide" />}
        </div>
        <div className="border-t border-border px-4 py-2">
          <Link to={topicGuideHref(t)} onClick={onClose}
            className="text-xs text-indigo-300 hover:text-indigo-200 hover:underline">
            Open this in the guide →
          </Link>
        </div>
      </div>
    </div>
  )
}

/** The discreet, ALWAYS-VISIBLE ⓘ that opens the modal. Distinct from
 * HelpBadge on purpose: that one appears only in Help mode and navigates away;
 * this one is the standing invitation on buttons whose one-word label cannot
 * carry what they do. Not nested inside the button it explains — a button
 * inside a button is invalid HTML and both halves stop being clickable. */
export function GuideInfoDot({ topic, label }) {
  const [open, setOpen] = useState(false)
  return (
    <>
      <button type="button" onClick={() => setOpen(true)}
        aria-label={`What does ${label || 'this'} do?`}
        title={`What does ${label || 'this'} do?`}
        className="inline-flex h-4 w-4 shrink-0 items-center justify-center self-center rounded-full border border-border text-[10px] font-semibold leading-none text-content-subtle hover:border-indigo-400/50 hover:text-indigo-300">
        i
      </button>
      {open && <GuideSectionModal topic={topic} onClose={() => setOpen(false)} />}
    </>
  )
}
