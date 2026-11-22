// Whether the side menu (App.jsx NavBar, from 1280 px) is folded to a strip of
// icons. One choice per browser, remembered across visits.
//
// The choice is mirrored on <html> as data-side-nav, and the stylesheet turns
// it into --app-sidebar-w (index.css, "App chrome"): the menu's own width, the
// shell's left padding and every fixed bar that starts after the menu all read
// that one variable, so folding the menu moves all of them together.
// main.jsx applies the stored choice BEFORE React mounts — set from an effect,
// the page would paint once with the wide menu and jump.

export const SIDE_NAV_COLLAPSED_KEY = 'lds_sideNavCollapsed';

function resolveStorage(storage) {
  if (storage) return storage;
  return typeof localStorage !== 'undefined' ? localStorage : null;
}

/** True when the menu was left folded. Unreadable storage reads as "open". */
export function readSideNavCollapsed(storage) {
  const store = resolveStorage(storage);
  if (!store) return false;
  try {
    return store.getItem(SIDE_NAV_COLLAPSED_KEY) === '1';
  } catch {
    return false;
  }
}

/** Remember the choice. A refused write (private mode) only costs the memory of it. */
export function writeSideNavCollapsed(collapsed, storage) {
  const store = resolveStorage(storage);
  if (!store) return;
  try {
    if (collapsed) store.setItem(SIDE_NAV_COLLAPSED_KEY, '1');
    else store.removeItem(SIDE_NAV_COLLAPSED_KEY);
  } catch {
    /* ignore */
  }
}

/** Mirror the choice on the root element, where the stylesheet reads it. */
export function applySideNavCollapsed(collapsed, root) {
  const element = root || (typeof document !== 'undefined' ? document.documentElement : null);
  if (element) element.setAttribute('data-side-nav', collapsed ? 'collapsed' : 'open');
}
