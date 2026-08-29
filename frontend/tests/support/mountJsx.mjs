/**
 * MOUNT a real component in `node --test`. The thing this suite did not have.
 *
 * ── Why this exists ─────────────────────────────────────────────────────────
 * Every canvas test until now read its component as TEXT and matched regexes
 * against it. That catches a missing attribute; it cannot catch anything the
 * renderer would do, because nothing ever ran the renderer. Twice now that gap
 * has shipped a white screen: a dependency array read before its declaration,
 * and — the reason this file exists — a `barH` left behind in
 * `CanvasImageGroup` when the title bar moved out to `CanvasGroupBar` and took
 * the `const barH = groupBarHeight(...)` with it. Both files parsed. Both
 * passed a full green suite. Both threw `ReferenceError` on a user's board, in
 * production, on the first gesture that reached the faulty branch.
 *
 * A source-text test cannot see that. Executing the component sees it in one
 * line, so a component's render is now something a test can hold on to.
 *
 * ── How ─────────────────────────────────────────────────────────────────────
 * Two module hooks, no new dependency and no browser:
 *
 *   · RESOLVE — the sources import `'./CanvasImageNode'` with no extension,
 *     which Vite resolves and Node does not. The hook retries the specifier
 *     with the same extension list Vite uses.
 *   · LOAD — `.jsx` is handed to esbuild (`jsx: 'automatic'`), which is exactly
 *     what Vite does to the same file at build time.
 *
 * Rendering is `react-dom/server`'s `renderToStaticMarkup`: it EXECUTES the
 * component function and every child, synchronously, with no DOM at all — so a
 * ReferenceError in any branch reached becomes a thrown test.
 *
 * ⚠️ What this does NOT do, said plainly so nobody reads more into a green run:
 * effects never run (`useEffect` is a no-op server-side), there is no layout,
 * no event ever fires, and nothing is measured. It answers exactly one
 * question — "does rendering this component, in this state, throw?" — which is
 * the question production kept answering for us.
 *
 * ⚠️ And it only covers the STATES a test passes it. `dropHint === 'leaving'`
 * shipped broken because no test had ever asked for that prop; rendering the
 * default state alone would have stayed green through the crash.
 */
import { readFileSync } from 'node:fs'
import { createRequire, registerHooks } from 'node:module'
import { fileURLToPath, pathToFileURL } from 'node:url'

import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

if (typeof registerHooks !== 'function') {
  throw new Error('node:module registerHooks is required (Node >= 22.15) — see package.json engines')
}

const require = createRequire(import.meta.url)

/* esbuild is Vite's own bundler and one of its hard dependencies, so `npm ci`
   installs it with vite and it is already in package-lock.json. It is looked up
   from here first (npm hoists it) and from vite's own folder second, so a
   nested layout works too — rather than declaring the same package twice in
   package.json and letting the two versions drift. */
function loadEsbuild() {
  const froms = [import.meta.url]
  try { froms.push(pathToFileURL(require.resolve('vite')).href) } catch { /* vite absent */ }
  for (const from of froms) {
    try { return createRequire(from)('esbuild') } catch { /* try the next one */ }
  }
  throw new Error('esbuild not found — run `npm ci` in frontend/ (it ships with vite)')
}

const esbuild = loadEsbuild()

// The same list, and the same order, as Vite's `resolve.extensions`.
const EXTENSIONS = ['.jsx', '.js', '.mjs', '/index.jsx', '/index.js']

registerHooks({
  resolve(specifier, context, nextResolve) {
    try {
      return nextResolve(specifier, context)
    } catch (err) {
      // Only relative source imports are retried: a bare specifier that does
      // not resolve is a genuinely missing package and must stay an error.
      if (!specifier.startsWith('.')) throw err
      for (const ext of EXTENSIONS) {
        try { return nextResolve(specifier + ext, context) } catch { /* next */ }
      }
      throw err
    }
  },
  load(url, context, nextLoad) {
    // A stylesheet import is a Vite concern with no meaning here; an empty
    // module keeps a component that has one importable.
    if (url.endsWith('.css')) return { format: 'module', source: '', shortCircuit: true }
    if (!url.endsWith('.jsx')) return nextLoad(url, context)
    const source = readFileSync(fileURLToPath(url), 'utf8')
    const { code } = esbuild.transformSync(source, {
      loader: 'jsx', jsx: 'automatic', format: 'esm', sourcefile: url,
    })
    return { format: 'module', source: code, shortCircuit: true }
  },
})

/* The app mounts everything under its providers (App.jsx: JobsProvider >
   ToastProvider > CapabilitiesProvider), so the harness does too — a component
   whose hook throws outside its provider (useToast and useCapabilities do, by
   design) is not broken, it is being rendered somewhere the real app never
   renders it. Effects never run server-side, so the providers contribute
   exactly their context value, never a network call. Imported DYNAMICALLY: a
   static import would be hoisted above registerHooks and hit the raw .jsx
   before the loader exists. */
const { ToastProvider } = await import('../../src/components/common/Toast.jsx')
const { CapabilitiesProvider } = await import('../../src/context/CapabilitiesContext.jsx')

/** Render a component to static HTML, executing it and all of its children —
 *  under the same providers the real app roots everything in. */
export function render(Component, props = {}) {
  return renderToStaticMarkup(
    createElement(ToastProvider, null,
      createElement(CapabilitiesProvider, null,
        createElement(Component, props))))
}

export { createElement, renderToStaticMarkup }
