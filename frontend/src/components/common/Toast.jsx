import { useState, useEffect, useCallback, useMemo, createContext, useContext } from 'react'

// ── Context ──

const ToastContext = createContext(null)

let _nextId = 0

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([])

  const addToast = useCallback((message, type = 'info', duration = 4000) => {
    const id = ++_nextId
    setToasts((prev) => [...prev, { id, message, type }])
    if (duration > 0) {
      setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), duration)
    }
    return id
  }, [])

  const removeToast = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }, [])

  const toast = useMemo(() => ({
    info: (msg, d) => addToast(msg, 'info', d),
    success: (msg, d) => addToast(msg, 'success', d),
    error: (msg, d) => addToast(msg, 'error', d ?? 6000),
    warning: (msg, d) => addToast(msg, 'warning', d),
  }), [addToast])

  // Expose on window for non-React usage
  useEffect(() => { window.__adminToast = toast }, [toast])

  return (
    <ToastContext.Provider value={toast}>
      {children}
      <ToastContainer toasts={toasts} onRemove={removeToast} />
    </ToastContext.Provider>
  )
}

export function useToast() {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error('useToast must be inside ToastProvider')
  return ctx
}

// ── Renderer ──

const TYPE_STYLES = {
  info: 'border-blue-500/50 bg-blue-500/10 text-blue-300',
  success: 'border-green-500/50 bg-green-500/10 text-green-300',
  error: 'border-red-500/50 bg-red-500/10 text-red-300',
  warning: 'border-yellow-500/50 bg-yellow-500/10 text-yellow-300',
}

const ICONS = {
  info: '\u2139\uFE0F',
  success: '\u2705',
  error: '\u274C',
  warning: '\u26A0\uFE0F',
}

function ToastContainer({ toasts, onRemove }) {
  if (!toasts.length) return null

  // Plain positioning wrapper — NOT a live region. Each toast is its own live
  // region (per-type politeness), avoiding nested live regions (double-announce)
  // and per-new-toast re-announce-all.
  return (
    <div className="fixed top-4 right-4 z-[100] flex flex-col gap-2 max-w-sm">
      {toasts.map((t) => (
        <div
          key={t.id}
          role={t.type === 'error' ? 'alert' : 'status'}
          aria-live={t.type === 'error' ? 'assertive' : 'polite'}
          aria-atomic="true"
          className={`flex items-start gap-2 border rounded-lg px-4 py-3 shadow-lg backdrop-blur-sm animate-slideIn ${
            TYPE_STYLES[t.type] || TYPE_STYLES.info
          }`}
        >
          <span className="flex-shrink-0 mt-0.5">{ICONS[t.type] || ICONS.info}</span>
          <span className="text-sm flex-1">{t.message}</span>
          <button
            type="button"
            onClick={() => onRemove(t.id)}
            aria-label="Close notification"
            className="flex-shrink-0 text-content-muted hover:text-content ml-2"
          >
            <span aria-hidden="true">&times;</span>
          </button>
        </div>
      ))}
    </div>
  )
}
