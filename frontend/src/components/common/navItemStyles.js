// Shapes shared by everything that is a row of the app's side menu (App.jsx):
// the pages, the two grouped menus at its foot, What's new and the update
// entry. One definition, so a row added later cannot drift from its neighbours.

// A row: icon, words, and whatever state the row carries pushed to the right
// edge. Every destination is one of these, so a plugin that adds a page adds a
// row — the menu grows down a column that scrolls instead of across a bar that
// wraps onto a second line.
export const SIDE_ITEM_BASE =
  'relative flex w-full items-center gap-2.5 rounded-md px-3 py-2 text-left text-sm font-medium no-underline transition-colors'
export const SIDE_ITEM_REST = 'text-content-muted hover:bg-surface-raised hover:text-content'
export const sideItemClass = ({ isActive }) =>
  `${SIDE_ITEM_BASE} ${isActive ? 'bg-surface-raised text-content' : SIDE_ITEM_REST}`

// The media query the side menu and the shell's left padding agree on (the
// `xl:` utilities in App.jsx and `--app-sidebar-w` in index.css use the same
// 1280 px). Not 1024: the menu takes 224 px, and the page beside it must keep
// the 1024 px every `lg:` two-column layout in the app was drawn for.
export const SIDE_NAV_QUERY = '(min-width: 1280px)'
