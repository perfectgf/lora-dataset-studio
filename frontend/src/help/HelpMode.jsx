import { createContext, useContext, useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { getHelpTopic, topicGuideHref, getHelpTip, guideHref } from './helpRegistry'
import { shouldShowTip, markTipSeen, TIP_EVENT } from './helpTips'

/* Help mode: a session-persisted toggle that reveals a small "?" badge next to
   instrumented headings/actions. Clicking a badge jumps to the matching spot in
   the in-app guide (app → doc). The mode is off by default and remembered for
   the browser session only (sessionStorage), never across restarts. */
const HelpModeContext = createContext({ enabled: false, toggle: () => {}, setEnabled: () => {} })
const HELP_MODE_KEY = 'ldsHelpMode'

export function HelpModeProvider({ children }) {
  const [enabled, setEnabled] = useState(() => {
    try { return sessionStorage.getItem(HELP_MODE_KEY) === '1' } catch { return false }
  })
  useEffect(() => {
    try { sessionStorage.setItem(HELP_MODE_KEY, enabled ? '1' : '0') } catch { /* ignore */ }
  }, [enabled])
  const toggle = useCallback(() => setEnabled((v) => !v), [])
  const value = useMemo(() => ({ enabled, toggle, setEnabled }), [enabled, toggle])
  return <HelpModeContext.Provider value={value}>{children}</HelpModeContext.Provider>
}

export function useHelpMode() {
  return useContext(HelpModeContext)
}

/* A discreet round "?" that appears only in Help mode, beside a title or button.
   Navigates to the topic's guide anchor. An unknown topic never crashes the
   layout — it warns once and renders nothing. */
export function HelpBadge({ topic, className = '' }) {
  const { enabled } = useHelpMode()
  const navigate = useNavigate()
  if (!enabled) return null
  const t = getHelpTopic(topic)
  if (!t) {
    if (typeof console !== 'undefined') console.warn(`HelpBadge: unknown help topic "${topic}"`)
    return null
  }
  const href = topicGuideHref(t)
  return (
    <button
      type="button"
      aria-label={`Help: ${t.title}`}
      title={`Help: ${t.title}`}
      onClick={(e) => { e.preventDefault(); e.stopPropagation(); navigate(href) }}
      className={`inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full border border-indigo-400/40 bg-indigo-500/15 align-middle text-[10px] font-bold leading-none text-indigo-300 transition-colors hover:bg-indigo-500/30 hover:text-indigo-200 ${className}`}
    >
      ?
    </button>
  )
}

/* TipHost: the sink for one-time contextual tips. Listens for the lds:help-tip
   event (fired by requestHelpTip anywhere), shows ONE small card bottom-right,
   and marks the tip seen the moment it appears — so it never shows twice. Tips
   are independent of Help mode. Mounted once, in App's Shell. */
export function TipHost() {
  const [tip, setTip] = useState(null)   // { topicId, text, href }
  const showing = useRef(false)
  const navigate = useNavigate()

  useEffect(() => {
    const onTip = (e) => {
      const trigger = e?.detail?.trigger
      if (!trigger || showing.current) return          // one card at a time
      const info = getHelpTip(trigger)
      if (!info || !shouldShowTip(info.topicId)) return
      showing.current = true
      markTipSeen(info.topicId)                         // seen on display, forever
      setTip({ topicId: info.topicId, text: info.text, href: guideHref(info.guide.chapter, info.guide.anchor) })
    }
    window.addEventListener(TIP_EVENT, onTip)
    return () => window.removeEventListener(TIP_EVENT, onTip)
  }, [])

  const dismiss = () => { showing.current = false; setTip(null) }

  if (!tip) return null
  return (
    <div role="status"
      className="fixed bottom-4 right-4 z-50 max-w-xs rounded-xl border border-indigo-400/40 bg-surface-overlay/95 p-3 shadow-lg backdrop-blur">
      <div className="flex items-start gap-2">
        <span aria-hidden className="text-base leading-none">💡</span>
        <div className="min-w-0 flex-1">
          <p className="m-0 text-xs leading-relaxed text-content">{tip.text}</p>
          <button type="button"
            onClick={() => { navigate(tip.href); dismiss() }}
            className="mt-2 text-xs font-medium text-indigo-300 underline hover:text-indigo-200">
            Learn more →
          </button>
        </div>
        <button type="button" onClick={dismiss} aria-label="Dismiss tip"
          className="-mr-1 -mt-1 shrink-0 px-1.5 text-content-subtle hover:text-content">✕</button>
      </div>
    </div>
  )
}
