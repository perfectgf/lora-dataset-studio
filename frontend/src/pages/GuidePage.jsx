import { useEffect, useMemo } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import Markdown, { markdownHeadingId } from '../components/common/Markdown'
import DiagnosticReport from '../components/common/DiagnosticReport'
import { helpTopicsForChapter } from '../help/helpRegistry'
import { useI18n } from '../i18n/I18nContext'
// Vite inlines every chapter as a string at build time (?raw) → the guide
// lives in the bundle, no fetch, nothing extra to ship in the portable build.
// DATASET_GUIDE.md keeps its historical path (linked from GitHub); the other
// chapters live in docs/guide/.
import gettingStarted from '../../../docs/guide/getting-started.md?raw'
import gettingStartedZh from '../../../docs/guide/zh-CN/getting-started.md?raw'
import usingTheApp from '../../../docs/guide/using-the-app.md?raw'
import usingTheAppZh from '../../../docs/guide/zh-CN/using-the-app.md?raw'
import datasetGuide from '../../../docs/DATASET_GUIDE.md?raw'
import datasetGuideZh from '../../../docs/guide/zh-CN/dataset-guide.md?raw'
import settingsReference from '../../../docs/guide/settings-reference.md?raw'
import settingsReferenceZh from '../../../docs/guide/zh-CN/settings-reference.md?raw'
import troubleshooting from '../../../docs/guide/troubleshooting.md?raw'
import troubleshootingZh from '../../../docs/guide/zh-CN/troubleshooting.md?raw'
import gettingHelp from '../../../docs/guide/getting-help.md?raw'
import gettingHelpZh from '../../../docs/guide/zh-CN/getting-help.md?raw'

/* The guide is a true reading sequence — the mono chapter numbers encode the
   intended order, not decoration. `extra` mounts a live component under the
   markdown (the diagnostic button on the help chapter). */
const CHAPTERS = [
  { id: 'getting-started', num: '01', titleKey: 'gettingStarted', source: gettingStarted, sourceZh: gettingStartedZh },
  { id: 'using-the-app', num: '02', titleKey: 'usingTheApp', source: usingTheApp, sourceZh: usingTheAppZh },
  { id: 'dataset-guide', num: '03', titleKey: 'datasetGuide', source: datasetGuide, sourceZh: datasetGuideZh },
  { id: 'settings-reference', num: '04', titleKey: 'settingsReference', source: settingsReference, sourceZh: settingsReferenceZh },
  { id: 'troubleshooting', num: '05', titleKey: 'troubleshooting', source: troubleshooting, sourceZh: troubleshootingZh },
]
const HELP_CHAPTER = {
  id: 'getting-help', num: '06', titleKey: 'gettingHelp',
  source: gettingHelp, sourceZh: gettingHelpZh, extra: 'diagnostic',
}

const cleanHeading = (heading) => heading.replace(/[`*_]/g, '')

// Append focus=<id> to a topic's app route, preserving any query already there
// (e.g. /datasets?section=scrape&panel=scan → …&focus=ds-scrape-scan).
const routeWithFocus = (app) => {
  if (!app.focus) return app.route
  return `${app.route}${app.route.includes('?') ? '&' : '?'}focus=${app.focus}`
}

export default function GuidePage({ helpOnly = false }) {
  const { locale, t } = useI18n()
  const { section } = useParams()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const targetHeading = searchParams.get('h')
  const localizeChapter = (chapter) => ({
    ...chapter,
    anchorSource: chapter.source,
    source: locale === 'zh-CN' ? chapter.sourceZh : chapter.source,
    title: t(`guide.chapters.${chapter.titleKey}.title`),
    description: t(`guide.chapters.${chapter.titleKey}.description`),
  })
  const chapters = (helpOnly ? [HELP_CHAPTER] : CHAPTERS).map(localizeChapter)
  const idx = helpOnly ? 0 : Math.max(0, chapters.findIndex((c) => c.id === section))
  const chapter = chapters[idx]
  const prev = idx > 0 ? chapters[idx - 1] : null
  const next = idx < chapters.length - 1 ? chapters[idx + 1] : null
  const anchorHeadings = [...chapter.anchorSource.matchAll(/^##\s+(.+)$/gm)]
  const headings = [...chapter.source.matchAll(/^##\s+(.+)$/gm)].map((match, index) => ({
    title: cleanHeading(match[1]),
    id: markdownHeadingId(anchorHeadings[index]?.[1] || match[1]),
  }))
  const readingUnits = locale === 'zh-CN'
    ? chapter.source.replace(/\s/g, '').length / 450
    : chapter.source.trim().split(/\s+/).length / 210
  const readingMinutes = Math.max(1, Math.ceil(readingUnits))
  const jumpToHeading = (id) => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })

  // doc → app: an "Open this screen →" button in each guide section header that
  // has a matching help topic. Keyed by heading anchor; the FIRST topic in the
  // registry for that anchor wins (see helpRegistry ordering).
  const sectionActions = useMemo(() => {
    const map = {}
    for (const topic of helpTopicsForChapter(chapter.id)) {
      if (map[topic.guide.anchor]) continue
      map[topic.guide.anchor] = (
        <button type="button" onClick={() => navigate(routeWithFocus(topic.app))}
          className="inline-flex items-center gap-1 whitespace-nowrap rounded-md border border-indigo-400/40 bg-indigo-500/10 px-2.5 py-1 text-xs font-medium text-indigo-200 transition-colors hover:bg-indigo-500/20">
          {t('guide.openScreen')} →
        </button>
      )
    }
    return map
  }, [chapter.id, navigate, t])

  // A chapter switch is a new "page" — land the reader at its top, unless the
  // link asked for a specific heading (?h=), which the effect below handles.
  useEffect(() => { if (!targetHeading) window.scrollTo(0, 0) }, [chapter.id, targetHeading])

  // app → doc landing: scroll to the requested heading and flash a ring so the
  // eye catches where it landed. Runs after render, so the element exists.
  useEffect(() => {
    if (!targetHeading) return undefined
    const el = document.getElementById(targetHeading)
    if (!el) return undefined
    el.scrollIntoView({ behavior: 'smooth', block: 'start' })
    const ring = ['ring-2', 'ring-indigo-400/70', 'ring-offset-2', 'ring-offset-app']
    el.classList.add(...ring)
    const timer = setTimeout(() => el.classList.remove(...ring), 2000)
    return () => clearTimeout(timer)
  }, [targetHeading, chapter.id])

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
        <nav aria-label={t('guide.chapterNavigation')} className="-mx-4 flex gap-2 overflow-x-auto px-4 pb-3 lg:hidden">
          {chapters.map((c) => navItem(c, true))}
        </nav>
        {/* Desktop: sticky numbered chapter rail */}
        <nav aria-label={t('guide.chapterNavigation')} className="hidden lg:sticky lg:top-20 lg:block">
          <p className="px-3 pb-2 font-mono text-[11px] uppercase tracking-[0.18em] text-content-subtle">{t('guide.fieldManual')}</p>
          <div className="flex flex-col gap-0.5">
            {chapters.map((c) => navItem(c, false))}
          </div>
        </nav>
      </aside>}

      <main className={`min-w-0 max-w-4xl pb-10 ${helpOnly ? 'mx-auto' : 'mt-2 lg:mt-0'}`}>
        <header className="relative mb-4 overflow-hidden rounded-2xl border border-border bg-surface px-5 py-5 sm:px-6 sm:py-6">
          <div aria-hidden className="absolute -right-16 -top-20 h-52 w-52 rounded-full bg-indigo-500/10 blur-3xl" />
          <div className="relative">
            <div className="mb-3 flex flex-wrap items-center gap-2 font-mono text-[0.6875rem] uppercase tracking-[0.14em] text-content-subtle">
              <span className="rounded-md border border-indigo-400/30 bg-indigo-500/10 px-2 py-1 text-indigo-300">
                {helpOnly ? t('guide.support') : t('guide.chapterNumber', { number: chapter.num })}
              </span>
              <span>{t('guide.readingMinutes', { count: readingMinutes })}</span>
              {!helpOnly && <><span aria-hidden>·</span><span>{t('guide.chapterPosition', {
                current: idx + 1,
                total: chapters.length,
              })}</span></>}
            </div>
            <h1 className="m-0 max-w-2xl text-2xl font-bold tracking-tight text-content sm:text-3xl">{chapter.title}</h1>
            <p className="mb-0 mt-2 max-w-2xl text-sm leading-relaxed text-content-muted sm:text-base">{chapter.description}</p>
          </div>
        </header>

        {headings.length > 0 && (
          <nav aria-label={t('guide.onThisPage')} className="mb-4 rounded-xl border border-border bg-surface p-3 xl:hidden">
            <p className="m-0 mb-2 font-mono text-[0.625rem] uppercase tracking-[0.16em] text-content-subtle">{t('guide.onThisPage')}</p>
            <div className="flex gap-2 overflow-x-auto pb-0.5">
              {headings.map((item) => (
                <button key={item.id} type="button" onClick={() => jumpToHeading(item.id)}
                  className="shrink-0 rounded-full border border-border bg-transparent px-2.5 py-1 text-xs text-content-muted hover:border-border-strong hover:text-content">{item.title}</button>
              ))}
            </div>
          </nav>
        )}

        <Markdown source={chapter.source} variant="guide" sectionActions={sectionActions}
          headingIds={headings.map((item) => item.id)} />

        {chapter.extra === 'diagnostic' && (
          <div className="mt-6">
            <DiagnosticReport />
          </div>
        )}

        {!helpOnly && <div className="mt-6 grid grid-cols-2 gap-3 border-t border-border pt-4">
          {prev ? (
            <Link to={`/guide/${prev.id}`} className="group flex min-w-0 items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2.5 no-underline hover:bg-surface-raised">
              <span aria-hidden className="text-content-subtle">←</span>
              <span className="min-w-0"><span className="block font-mono text-[0.625rem] uppercase tracking-wider text-content-subtle">{t('guide.previous')}</span><span className="block truncate text-sm font-medium text-content-muted group-hover:text-content">{prev.title}</span></span>
            </Link>
          ) : <span />}
          {next ? (
            <Link to={`/guide/${next.id}`} className="group flex min-w-0 items-center justify-end gap-2 rounded-lg border border-border bg-surface px-3 py-2.5 text-right no-underline hover:bg-surface-raised">
              <span className="min-w-0"><span className="block font-mono text-[0.625rem] uppercase tracking-wider text-content-subtle">{t('guide.next')}</span><span className="block truncate text-sm font-medium text-content-muted group-hover:text-content">{next.title}</span></span>
              <span aria-hidden className="text-content-subtle">→</span>
            </Link>
          ) : <span />}
        </div>}
      </main>

      <aside className="hidden xl:block">
        <nav aria-label={t('guide.onThisPage')} className="sticky top-20 border-l border-border pl-4">
          <p className="m-0 mb-2 font-mono text-[0.625rem] uppercase tracking-[0.16em] text-content-subtle">{t('guide.onThisPage')}</p>
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
