/** A "this is configurable, here" pointer, placed where the user is ACTING rather
 * than in Settings where they are not. Several features are driven by a setting
 * that is discoverable only by knowing it exists — the manual Upscale & improve
 * being the reported case: its strength and instruction are editable, and nothing
 * on the button said so.
 *
 * `section` is a Settings section id from components/settings/registry.js
 * (overview | engines | scraping | local-tools | captioning | training | server |
 * maintenance). A plain hash anchor rather than a router Link: these sit inside
 * lightboxes and panels that are sometimes portaled, where a bare <a> is the one
 * form that behaves identically everywhere.
 *
 * `tone`: 'subtle' (default) for an ambient hint, 'warning' inside an amber block
 * that is already telling the user something is missing.
 */
const TONES = {
  subtle: 'text-content-subtle hover:text-content underline decoration-border',
  warning: 'text-amber-300 underline decoration-amber-300/50',
};

export default function SettingsLink({ section, children, tone = 'subtle', className = '' }) {
  return (
    <a
      href={`#/settings/${section}`}
      className={`${TONES[tone] || TONES.subtle} text-[0.6875rem] ${className}`}
      // Stops the click from also triggering a parent that opens a lightbox,
      // toggles a tile or starts a job — these links live on top of active surfaces.
      onClick={(e) => e.stopPropagation()}
    >
      {children} →
    </a>
  );
}
