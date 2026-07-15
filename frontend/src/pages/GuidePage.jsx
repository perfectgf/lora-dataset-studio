import { useEffect } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import Markdown, { markdownHeadingId } from '../components/common/Markdown'
import DiagnosticReport from '../components/common/DiagnosticReport'
// Vite inlines every chapter as a string at build time (?raw) → the guide
// lives in the bundle, no fetch, nothing extra to ship in the portable build.
// DATASET_GUIDE.md keeps its historical path (linked from GitHub); the other
// chapters live in docs/guide/.
import gettingStarted from '../../../docs/guide/getting-started.md?raw'
import usingTheApp from '../../../docs/guide/using-the-app.md?raw'
import datasetGuide from '../../../docs/DATASET_GUIDE.md?raw'
import troubleshooting from '../../../docs/guide/troubleshooting.md?raw'
import gettingHelp from '../../../docs/guide/getting-help.md?raw'

/* The guide is a true reading sequence — the mono chapter numbers encode the
   intended order, not decoration. `extra` mounts a live component under the
   markdown (the diagnostic button on the help chapter). */
const CHAPTERS = [
  { id: 'getting-started', num: '01', title: 'Getting started', description: 'Install the app, connect the tools you need, and understand the workspace.', source: gettingStarted },
  { id: 'using-the-app', num: '02', title: 'Using the app', description: 'Follow the complete workflow for character, concept, and style datasets.', source: usingTheApp },
  { id: 'dataset-guide', num: '03', title: 'Building a good dataset', description: 'Make stronger choices about images, captions, settings, and checkpoints.', source: datasetGuide },
  { id: 'troubleshooting', num: '04', title: 'Troubleshooting', description: 'Find a symptom, understand the cause, and apply the shortest reliable fix.', source: troubleshooting },
]
const HELP_CHAPTER = { id: 'getting-help', num: '05', title: 'Getting help', description: 'Create a useful report and share the details needed to solve a problem.', source: gettingHelp, extra: 'diagnostic' }

const cleanHeading = (heading) => heading.replace(/[`*_]/g, '')

export default function GuidePage({ helpOnly = false }) {
  const { section } = useParams()
  const navigate = useNavigate()
  const chapters = helpOnly ? [HELP_CHAPTER] : CHAPTERS
  const idx = helpOnly ? 0 : Math.max(0, chapters.findIndex((c) => c.id === section))
  const chapter = chapters[idx]
  const prev = idx > 0 ? chapters[idx - 1] : null
  const next = idx < chapters.length - 1 ? chapters[idx + 1] : null
  const headings = [...chapter.source.matchAll(/^##\s+(.+)$/gm)].map((match) => ({
    title: cleanHeading(match[1]), id: markdownHeadingId(match[1]),
  }))
  const readingMinutes = Math.max(1, Math.ceil(chapter.source.trim().split(/\s+/).length / 210))
  const jumpToHeading = (id) => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })

  // A chapter switch is a new "page" — land the reader at its top, not at the
  // scroll depth of the previous chapter.
  useEffect(() => { window.scrollTo(0, 0) }, [chapter.id])

  const navItem = (c, chip) => {
    const isActive = c.id === chapter.id
    const base = chip
      ? `flex shrink-0 items-baseline gap-1.5 whitespace-nowrap rounded-full border px-3 py-1.5 text-xs font-medium ${
          isActive ? 'border-border-strong bg-surface-raised text-content' : 'border-border text-content-muted hover:text-content'}`
      : `relative flex w-full items-baseline gap-2.5 rounded-md px-3 py-2 text-left text-sm ${
          isActive ? 'bg-surface-raised text-content' : 'text-content-muted hover:bg-surface hover:text-content'}`
    return (
      <button key={c.id} type="button" onClick={() => navigate(`/guide/${c.id}`)}
        aria-current={isActive ? 'page' : undefined} className={base}>
        {!chip && isActive && (
          <span aria-hidden className="absolute bottom-1.5 left-0 top-1.5 w-0.5 rounded bg-gradient-primary" />
        )}
        <span className={`font-mono text-[11px] ${isActive ? 'text-content' : 'text-content-subtle'}`}>{c.num}</span>
        <span className="font-medium">{c.title}</span>
      </button>
    )
  }

  return (
    <div className={helpOnly
      ? 'mx-auto max-w-5xl xl:grid xl:grid-cols-[minmax(0,1fr)_190px] xl:items-start xl:gap-7'
      : 'lg:grid lg:grid-cols-[210px_minmax(0,1fr)] lg:items-start lg:gap-7 xl:grid-cols-[210px_minmax(0,1fr)_190px]'}>
      {!helpOnly && <aside>
        {/* Mobile: horizontal chapter chips */}
        <nav aria-label="Guide chapters" className="-mx-4 flex gap-2 overflow-x-auto px-4 pb-3 lg:hidden">
          {CHAPTERS.map((c) => navItem(c, true))}
        </nav>
        {/* Desktop: sticky numbered chapter rail */}
        <nav aria-label="Guide chapters" className="hidden lg:sticky lg:top-20 lg:block">
          <p className="px-3 pb-2 font-mono text-[11px] uppercase tracking-[0.18em] text-content-subtle">Field manual</p>
          <div className="flex flex-col gap-0.5">
            {CHAPTERS.map((c) => navItem(c, false))}
          </div>
        </nav>
      </aside>}

      <main className={`min-w-0 max-w-4xl pb-10 ${helpOnly ? 'mx-auto' : 'mt-2 lg:mt-0'}`}>
        <header className="relative mb-4 overflow-hidden rounded-2xl border border-border bg-surface px-5 py-5 sm:px-6 sm:py-6">
          <div aria-hidden className="absolute -right-16 -top-20 h-52 w-52 rounded-full bg-indigo-500/10 blur-3xl" />
          <div className="relative">
            <div className="mb-3 flex flex-wrap items-center gap-2 font-mono text-[0.6875rem] uppercase tracking-[0.14em] text-content-subtle">
              <span className="rounded-md border border-indigo-400/30 bg-indigo-500/10 px-2 py-1 text-indigo-300">
                {helpOnly ? 'Support' : `Chapter ${chapter.num}`}
              </span>
              <span>{readingMinutes} min read</span>
              {!helpOnly && <><span aria-hidden>·</span><span>{idx + 1} of {chapters.length}</span></>}
            </div>
            <h1 className="m-0 max-w-2xl text-2xl font-bold tracking-tight text-content sm:text-3xl">{chapter.title}</h1>
            <p className="mb-0 mt-2 max-w-2xl text-sm leading-relaxed text-content-muted sm:text-base">{chapter.description}</p>
          </div>
        </header>

        {headings.length > 0 && (
          <nav aria-label="On this page" className="mb-4 rounded-xl border border-border bg-surface p-3 xl:hidden">
            <p className="m-0 mb-2 font-mono text-[0.625rem] uppercase tracking-[0.16em] text-content-subtle">On this page</p>
            <div className="flex gap-2 overflow-x-auto pb-0.5">
              {headings.map((item) => (
                <button key={item.id} type="button" onClick={() => jumpToHeading(item.id)}
                  className="shrink-0 rounded-full border border-border bg-transparent px-2.5 py-1 text-xs text-content-muted hover:border-border-strong hover:text-content">{item.title}</button>
              ))}
            </div>
          </nav>
        )}

        <Markdown source={chapter.source} variant="guide" />

        {chapter.extra === 'diagnostic' && (
          <div className="mt-6">
            <DiagnosticReport />
          </div>
        )}

        {!helpOnly && <div className="mt-6 grid grid-cols-2 gap-3 border-t border-border pt-4">
          {prev ? (
            <Link to={`/guide/${prev.id}`} className="group flex min-w-0 items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2.5 no-underline hover:bg-surface-raised">
              <span aria-hidden className="text-content-subtle">←</span>
              <span className="min-w-0"><span className="block font-mono text-[0.625rem] uppercase tracking-wider text-content-subtle">Previous</span><span className="block truncate text-sm font-medium text-content-muted group-hover:text-content">{prev.title}</span></span>
            </Link>
          ) : <span />}
          {next ? (
            <Link to={`/guide/${next.id}`} className="group flex min-w-0 items-center justify-end gap-2 rounded-lg border border-border bg-surface px-3 py-2.5 text-right no-underline hover:bg-surface-raised">
              <span className="min-w-0"><span className="block font-mono text-[0.625rem] uppercase tracking-wider text-content-subtle">Next</span><span className="block truncate text-sm font-medium text-content-muted group-hover:text-content">{next.title}</span></span>
              <span aria-hidden className="text-content-subtle">→</span>
            </Link>
          ) : <span />}
        </div>}
      </main>

      <aside className="hidden xl:block">
        <nav aria-label="On this page" className="sticky top-20 border-l border-border pl-4">
          <p className="m-0 mb-2 font-mono text-[0.625rem] uppercase tracking-[0.16em] text-content-subtle">On this page</p>
          <div className="flex flex-col gap-0.5">
            {headings.map((item) => (
              <button key={item.id} type="button" onClick={() => jumpToHeading(item.id)}
                className="rounded-md bg-transparent px-2 py-1.5 text-left text-xs leading-snug text-content-subtle hover:bg-surface hover:text-content">{item.title}</button>
            ))}
          </div>
        </nav>
      </aside>
    </div>
  )
}
