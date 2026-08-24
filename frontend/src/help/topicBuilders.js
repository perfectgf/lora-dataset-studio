/* The help-topic builders, moved verbatim from helpRegistry.js
   (2026-08-24 split). PURE JS, like everything the registry imports. */

// settings-reference H2 anchor for each Settings section id.
const SETTINGS_ANCHOR = {
  overview: 'overview',
  engines: 'image-engines',
  scraping: 'scraping-sources',
  'local-tools': 'local-tools',
  captioning: 'captioning-quality',
  training: 'training',
  storage: 'storage',
  server: 'server-access',
  maintenance: 'maintenance',
};

// Build a kind:'setting' topic. All fields in a Settings section share the
// section's route and settings-reference anchor; only the DOM focus id differs.
export const setting = (id, section, focus, title, keywords, tip) => ({
  id, kind: 'setting', title, keywords,
  guide: { chapter: 'settings-reference', anchor: SETTINGS_ANCHOR[section] },
  app: { route: `/settings/${section}`, focus },
  ...(tip ? { tip } : {}),
});

// Build a topic that opens ONE screen of the Setup wizard (/setup?step=<id>).
// Several capabilities are turned on by an INSTALL, not by a setting — the
// button that installs them lives in a wizard step, so that step is their real
// address. Without these, "✗ Person masks" could only point at the top of the
// wizard and let the user click Next until they found it.
export const setupStep = (id, step, title, keywords) => ({
  id, kind: 'action', title, keywords,
  guide: { chapter: 'getting-started', anchor: 'the-setup-wizard' },
  app: { route: `/setup?step=${step}` },
});

export const action = (id, title, keywords, route, chapter, anchor, tip) => ({
  id, kind: 'action', title, keywords,
  guide: { chapter, anchor },
  app: { route },
  ...(tip ? { tip } : {}),
});
