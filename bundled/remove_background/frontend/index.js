// ✂️ Remove Background — the plugin's frontend descriptor. Plain JS: node tests
// import it, and every screen is a lazy import.
import { GUIDE } from './guide.js'
import { HELP } from './help.js'
import { modelRows, setupRows } from './lib/readiness.js'

export default {
  id: 'remove_background',
  nav: [{ to: '/remove-background', label: '✂️ Remove background' }],
  routes: [{ path: '/remove-background', page: () => import('./panels/RemoveBackgroundPage.jsx') }],
  slots: {
    // The verb, on every viewer: each one opens the same editor on its own copy of the picture.
    'lightbox.action': [{ id: 'remove-background', panel: () => import('./panels/RemoveBackgroundAction.jsx') }],
    'setup.card': [{ id: 'remove-background', panel: () => import('./panels/Preparation.jsx') }],
    'setup.step': [{ id: 'remove-background', rows: setupRows, catalog: modelRows }],
  },
  hosts: [],
  paritySkip: [],
  guide: GUIDE,
  help: HELP,
  whatsNew: [{
    id: '2026-11-24-remove-background', date: '2026-11-24', title: 'Remove the background of any picture',
    blurb: 'Say what stays (the main subject, a rough selection, a few words or a click) and get a clean cut-out with a mask you can paint on. It works on a picture you import and from the dataset, Bank and Gallery viewers, on your own ComfyUI.',
    to: '/remove-background',
  }],
}
