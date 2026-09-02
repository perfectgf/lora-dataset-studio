/**
 * RUN a React hook in `node --test` — the hook's own code, with its state,
 * its refs, its effects and its timers, and no React and no DOM.
 *
 * ── Why this exists ─────────────────────────────────────────────────────────
 * `mountJsx.mjs` executes a component's RENDER. A hook that lives on state
 * updates and timers is out of its reach: the server renderer never runs an
 * effect, never updates a state, never fires a timer. And the source-as-text
 * contract tests can only say a line is WRITTEN. So a hook whose defect was a
 * vigil left armed after a later click had already gone through — one click,
 * two answers written into the field — passed every test the suite had, twice
 * over, and was found by reading.
 *
 * ── How ─────────────────────────────────────────────────────────────────────
 * A resolve hook aliases `'react'` to this file, which exports the four hooks
 * the guarded hooks use (`useState`, `useRef`, `useCallback`, `useEffect`, plus
 * `useMemo`), implemented over the same cursor-per-render model React uses:
 * `mountHook` calls the hook function like a component, a state setter
 * re-runs it, effects run after the render they were declared in and clean
 * up before they re-run or on unmount. `'../api/fetchClient'` is aliased here
 * too, to a stub the test scripts through `fake`. Relative imports without an
 * extension get one, the way Vite resolves them.
 *
 * ⚠️ What this is NOT: React. No batching (a setter re-renders at once —
 * which is stricter, not looser, than React's own), no concurrent anything,
 * no context, no children. It answers one question — "what does this hook DO
 * when its callbacks and timers run?" — for hooks that use nothing else.
 */
import { registerHooks } from 'node:module'

const SELF = import.meta.url
const EXTENSIONS = ['.js', '.jsx', '.mjs']

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier === 'react') return { url: SELF, format: 'module', shortCircuit: true }
    if (/(^|\/)api\/fetchClient(\.js)?$/.test(specifier)) {
      return { url: SELF, format: 'module', shortCircuit: true }
    }
    try {
      return nextResolve(specifier, context)
    } catch (err) {
      if (!specifier.startsWith('.')) throw err
      for (const ext of EXTENSIONS) {
        try { return nextResolve(specifier + ext, context) } catch { /* next */ }
      }
      throw err
    }
  },
})

/* ── the fetch client stand-in ──────────────────────────────────────────────
   The hook under test imports `apiFetch` and `postJson`; a test decides what
   each answers by assigning `fake.apiFetch` / `fake.postJson`. */
export const fake = {
  apiFetch: async () => { throw new Error('fake.apiFetch not scripted by the test') },
  postJson: async () => { throw new Error('fake.postJson not scripted by the test') },
}
export const apiFetch = (...args) => fake.apiFetch(...args)
export const postJson = (...args) => fake.postJson(...args)

/* ── the hooks ──────────────────────────────────────────────────────────── */
let current = null

function depsEqual(a, b) {
  if (a === undefined || b === undefined) return false
  return a.length === b.length && a.every((v, i) => Object.is(v, b[i]))
}

function slot() {
  if (!current) throw new Error('a hook was called outside mountHook')
  const inst = current
  const index = inst.cursor++
  return { inst, index, prev: inst.slots[index] }
}

export function useState(init) {
  const { inst, index, prev } = slot()
  const s = prev || { value: typeof init === 'function' ? init() : init }
  if (!prev) {
    s.set = (next) => {
      const value = typeof next === 'function' ? next(s.value) : next
      if (Object.is(value, s.value)) return
      s.value = value
      inst.render()
    }
    inst.slots[index] = s
  }
  return [s.value, s.set]
}

export function useRef(init) {
  const { inst, index, prev } = slot()
  if (!prev) inst.slots[index] = { current: init }
  return inst.slots[index]
}

function memo(factory, deps) {
  const { inst, index, prev } = slot()
  if (prev && depsEqual(prev.deps, deps)) return prev.value
  const value = factory()
  inst.slots[index] = { value, deps }
  return value
}

export function useMemo(factory, deps) {
  return memo(factory, deps)
}

export function useCallback(fn, deps) {
  return memo(() => fn, deps)
}

export function useEffect(fn, deps) {
  const { inst, index, prev } = slot()
  if (prev && depsEqual(prev.deps, deps)) return
  inst.pending.push({ index, fn, deps })
}

/**
 * Mount `hook(...args)`. Returns a handle: `read()` is the hook's latest
 * return value, `renders` how many times it ran, `unmount()` runs every
 * effect's cleanup.
 */
export function mountHook(hook, ...args) {
  const inst = { slots: [], pending: [], cursor: 0, alive: true, result: undefined, renders: 0 }
  inst.render = () => {
    if (!inst.alive) return
    inst.cursor = 0
    inst.pending = []
    current = inst
    try { inst.result = hook(...args) } finally { current = null }
    inst.renders += 1
    for (const { index, fn, deps } of inst.pending) {
      inst.slots[index]?.cleanup?.()
      inst.slots[index] = { deps, cleanup: fn() }
    }
  }
  inst.render()
  return {
    read: () => inst.result,
    get renders() { return inst.renders },
    unmount() {
      for (const s of inst.slots) s?.cleanup?.()
      inst.alive = false
    },
  }
}

/** Let every pending microtask settle — an awaited fetch, a replay, a state
 *  update — without a real timer. */
export const flush = () => new Promise((resolve) => setImmediate(resolve))
