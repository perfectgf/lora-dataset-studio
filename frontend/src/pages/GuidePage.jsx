import { useEffect } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import Markdown from '../components/common/Markdown'
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
  { id: 'getting-started', num: '01', title: 'Getting started', source: gettingStarted },
  { id: 'using-the-app', num: '02', title: 'Using the app', source: usingTheApp },
  { id: 'dataset-guide', num: '03', title: 'Building a good dataset', source: datasetGuide },
  { id: 'troubleshooting', num: '04', title: 'Troubleshooting', source: troubleshooting },
  { id: 'getting-help', num: '05', title: 'Getting help', source: gettingHelp, extra: 'diagnostic' },
]

export default function GuidePage() {
  const { section } = useParams()
  const navigate = useNavigate()
  const idx = Math.max(0, CHAPTERS.findIndex((c) => c.id === section))
  const chapter = CHAPTERS[idx]
  const prev = idx > 0 ? CHAPTERS[idx - 1] : null
  const next = idx < CHAPTERS.length - 1 ? CHAPTERS[idx + 1] : null

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
    <div className="lg:grid lg:grid-cols-[220px_minmax(0,1fr)] lg:items-start lg:gap-8">
      <aside>
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
      </aside>

      <div className="mt-2 max-w-3xl pb-10 lg:mt-0">
        <Markdown source={chapter.source} />

        {chapter.extra === 'diagnostic' && (
          <div className="mt-6">
            <DiagnosticReport />
          </div>
        )}

        <div className="mt-10 flex items-center justify-between gap-4 border-t border-border pt-4">
          {prev ? (
            <Link to={`/guide/${prev.id}`} className="group flex items-baseline gap-2 no-underline">
              <span aria-hidden className="text-content-subtle">←</span>
              <span className="font-mono text-[11px] text-content-subtle">{prev.num}</span>
              <span className="text-sm font-medium text-content-muted group-hover:text-content">{prev.title}</span>
            </Link>
          ) : <span />}
          {next ? (
            <Link to={`/guide/${next.id}`} className="group flex items-baseline gap-2 text-right no-underline">
              <span className="font-mono text-[11px] text-content-subtle">{next.num}</span>
              <span className="text-sm font-medium text-content-muted group-hover:text-content">{next.title}</span>
              <span aria-hidden className="text-content-subtle">→</span>
            </Link>
          ) : <span />}
        </div>
      </div>
    </div>
  )
}
