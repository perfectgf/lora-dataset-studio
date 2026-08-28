/* 🔤 Flagged pages with their zones — INSIDE the launch window.
 *
 * Asked for in these words: "quand on fait le lancement de recherche test de
 * text cela doit afficher dans la même fenêtre les image cible avec les zone
 * de ciblage" — a test run whose result can only be judged by leaving the
 * window is not a test run, it is a scavenger hunt. ONE component for both
 * surfaces, so the strip cannot drift into two wordings of the same thing
 * (the parity rule): the bank feeds it from its preview endpoint, the dataset
 * from the flagged rows already in its payload.
 *
 * Each tile links to the full-size page: the zones are judged on a thumbnail,
 * but a borderline box (did it clip the bubble outline?) needs pixels.
 */
import { galleryHeadline, galleryZones, zoneStyle } from './textZonesGallery.js'

export default function TextZonesGallery({
  items = [], total = 0, live = false, emptyLine = null, reviewHint = '',
}) {
  if (!items.length) {
    if (live) {
      return (
        <p className="m-0 text-[11px] leading-snug text-content-subtle">
          Scanning — nothing flagged yet. Pages appear here as text is found.
        </p>
      )
    }
    return emptyLine
      ? <p className="m-0 text-[11px] leading-snug text-content-subtle">{emptyLine}</p>
      : null
  }
  return (
    <div className="space-y-1">
      <p className="m-0 text-[11px] leading-snug text-content-subtle">
        <span className="font-medium text-content">Flagged pages and their zones</span>
        {galleryHeadline(items.length, total)}
        {reviewHint ? `. ${reviewHint}` : '.'}
      </p>
      <ul className="m-0 flex list-none gap-2 overflow-x-auto p-0 pb-1">
        {items.map((it) => (
          <li key={it.id} className="shrink-0">
            <a href={it.href || it.src} target="_blank" rel="noreferrer"
              title="Open the full-size page in a new tab"
              className="relative block overflow-hidden rounded-md border border-border">
              <img src={it.src} alt={it.alt || `Page ${it.id} with its text zones`}
                loading="lazy" className="h-48 w-auto object-contain" />
              {galleryZones(it.regions).map((zone, i) => (
                <span key={i} aria-hidden
                  className="absolute border border-amber-400 bg-amber-400/20"
                  style={zoneStyle(zone)} />
              ))}
            </a>
            <p className="m-0 mt-0.5 text-center text-[10px] text-content-subtle">#{it.id}</p>
          </li>
        ))}
      </ul>
    </div>
  )
}
