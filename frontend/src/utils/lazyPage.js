// Route-level code splitting, with the one failure mode it creates handled.
//
// Every page used to sit in the single 3.4 MB entry chunk: the first paint of
// ANY screen paid for all eighteen routes. `lazyPage` moves each page into its
// own chunk, loaded on first navigation.
//
// The guard is not sugar. "Update & restart" swaps the whole dist/ folder
// while a tab may still be open — the OLD index.html then asks for chunk
// files that no longer exist, and the first navigation after every update
// would be a dead page with a console error nobody reads. On a failed chunk
// load we reload ONCE (the reload fetches the new index.html, whose chunk
// names are current); a sessionStorage flag stops a reload LOOP when the
// chunk is missing for a real reason (a broken deploy stays a visible error,
// not a flickering tab). The flag clears on any successful load so the NEXT
// update gets its one reload too. sessionStorage can throw (private mode) —
// then we skip straight to the error rather than risk the loop.
import { lazy } from 'react'

const RELOAD_FLAG = 'lds-chunk-reloaded'

export function lazyPage(importer) {
  return lazy(() => importer().then((mod) => {
    try { sessionStorage.removeItem(RELOAD_FLAG) } catch { /* private mode */ }
    return mod
  }).catch((err) => {
    let alreadyReloaded = true
    try {
      alreadyReloaded = sessionStorage.getItem(RELOAD_FLAG) === '1'
      if (!alreadyReloaded) sessionStorage.setItem(RELOAD_FLAG, '1')
    } catch { /* private mode: no way to bound a loop, so never start one */ }
    if (!alreadyReloaded) {
      window.location.reload()
      return new Promise(() => {})   // the reload owns the tab from here
    }
    throw err
  }))
}
