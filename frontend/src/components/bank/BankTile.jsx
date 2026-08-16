/**
 * One image in the Bank grid.
 *
 * Moved out of BankWorkspace.jsx by the Encre redesign: in structure B the grid
 * is the CENTRE of the screen and full height, so the tile stopped being an
 * incidental detail of a four-zone stack and became the thing the page is for.
 *
 * Nothing about its behaviour changed — same three hit targets (select, ▶
 * review, ⛶ open), same badge cluster, same tooltip.
 */
import { FLAG_LABEL, STATUS_RING } from './bankFacets.js'
import { editBadge, imageVersionQuery } from './bankEdits.js'
import { angleBadge } from './bankMedium.js'
import { detailSummary } from './bankProvenance.js'
import { captionChips } from './bankTags.js'
import { captionOriginTooltipLine } from '../../utils/captionOrigin.js'

export default function Tile({ img, bankId, selected, onToggle, onReview, onTags, size }) {
  // `key` matters only for the flags list below (the one mapped array) — it was
  // missing and logged a React warning on every bank grid render.
  // The chips this image would actually offer. Computed HERE rather than asked of
  // the caption's mere existence: a caption of nothing but stop words ("a photo of
  // her") yields zero chips, and `img.caption && …` would have offered the button
  // anyway. The test for "can this button do its job" is the job's own output.
  const tagChips = captionChips(img.caption)
  // ✂/✨ made in the Bank itself. A state, like ✓ ✕ ⬆ and the flags — it belongs
  // in the readout cluster, not among the actions.
  const edited = editBadge(img)
  const badge = (txt, cls, key) => (
    <span key={key} className={`rounded px-1 py-px text-[10px] font-semibold leading-none ${cls}`}>{txt}</span>
  )
  return (
    <li className={`relative overflow-hidden rounded-lg border border-border bg-surface ${STATUS_RING[img.status] || ''}`}>
      <button type="button" onClick={onToggle}
        title={`${img.name} — ${img.width || '?'}×${img.height || '?'}`
          + (img.blur_score != null ? ` · sharpness ${Math.round(img.blur_score)}` : '')
          + (img.aesthetic_score != null ? ` · aesthetic ${img.aesthetic_score.toFixed(1)}` : '')
          + (img.nsfw_score != null ? ` · NSFW ${Math.round(img.nsfw_score * 100)}%` : '')
          + (img.face_cluster ? ` · person #${img.face_cluster}`
            // A declaration is not a measurement — the tooltip says which it is.
            + (img.face_cluster_origin === 'asserted' ? ' (your folder assertion)' : '') : '')
          + (img.framing ? ` · ${img.framing}` : '')
          + (img.medium ? ` · ${img.medium}` : '')
          + (img.face_yaw != null ? ` · head turned ${Math.round(Math.abs(img.face_yaw))}°` : '')
          + (detailSummary(img)?.soft ? ` · only ~${detailSummary(img).real} px of real detail` : '')
          + (img.origin && img.origin !== 'unknown' ? ` · ${img.origin}` : '')
          + (img.style_cluster ? ` · style #${img.style_cluster}` : '')
          + (img.semantic_dup_group ? ` · same shot #${img.semantic_dup_group}` : '')
          // The caption, and WHO WROTE IT in the same breath. The 'auto' backend
          // chains two engines inside one run, so a bank holds both and the
          // sentence alone cannot say which half produced it — exactly the reading
          // this tooltip was missing.
          + (img.caption
            ? `\n${captionOriginTooltipLine(img.caption, img.caption_origin)}: ${img.caption}`
            : '')}
        className="block w-full">
        {/* ?r=/?e= are cache busters, not parameters the server reads: the thumb
            route answers with max-age=3600, so a turned OR edited image would keep
            showing its old pixels for an hour and read as "the button did nothing".
            ?e= carries the edit GENERATION, so a second crop of the same image
            moves the URL too — see bankEdits.imageVersionQuery. */}
        <img src={`/api/bank/${bankId}/thumb/${img.id}${imageVersionQuery(img)}`}
          alt={[img.name, img.rotation ? `rotated ${img.rotation}°` : null,
            edited ? edited.label : null].filter(Boolean).join(' — ')} loading="lazy"
          className={`w-full object-cover ${size === 'S' ? 'h-24' : 'h-36'}`} />
      </button>
      {selected && (
        <span aria-hidden className="absolute inset-0 bg-indigo-500/30 ring-2 ring-indigo-400 rounded-lg pointer-events-none" />
      )}
      <span className="absolute left-1 top-1 flex flex-wrap gap-0.5 max-w-[85%]">
        {img.status === 'keep' && badge('✓', 'bg-emerald-500/80 text-white')}
        {img.status === 'reject' && badge(`✕ ${img.reject_reason || ''}`.trim(), 'bg-rose-500/80 text-white')}
        {/* ⬆ = this image left for somewhere: a dataset, another bank, or both.
            One badge for both destinations — the tile says THAT it went, the
            tooltip and the review lightbox say where. */}
        {(img.promoted_dataset_id != null || img.promoted_bank_id != null)
          && badge('⬆', 'bg-indigo-500/80 text-white')}
        {/* ✂/✨ = these pixels were edited HERE. Without it an edited image looks
            like any other, and the grid cannot answer either "did my crop land?"
            or "what does ↩ Revert still have to take back?". The tooltip carries
            the promise the whole feature rests on: your own file was not touched. */}
        {edited && (
          <span title={edited.title} aria-label={edited.label}
            className="rounded bg-sky-500/80 px-1 py-px text-[10px] font-semibold leading-none text-white">
            {edited.text}
          </span>
        )}
        {img.flags.map((f) => badge(FLAG_LABEL[f]?.slice(0, 2) || f, 'bg-black/60 text-amber-200', f))}
        {img.face_cluster != null && badge(`👤${img.face_cluster}`, 'bg-black/60 text-sky-200')}
        {img.framing && badge(`📐${img.framing}`, 'bg-black/60 text-teal-200')}
        {/* A medium badge is stamped only when the classifier actually COMMITTED
            to one — 'unsure' is a real verdict but it is not a label to write on
            a thumbnail, and NULL means the pass never reached this image. */}
        {img.medium && img.medium !== 'unsure'
          && badge(`🎨${img.medium}`, 'bg-black/60 text-lime-200')}
        {angleBadge(img) && badge(angleBadge(img).text, 'bg-black/60 text-cyan-200')}
        {/* Only the PROVEN states get a badge. Stamping ❔ on the 80% of files
            whose metadata was stripped would be noise, not information. */}
        {img.origin === 'ai' && badge('🤖', 'bg-black/60 text-violet-200')}
        {img.origin === 'camera' && badge('📷', 'bg-black/60 text-emerald-200')}
        {img.style_cluster != null && badge(`🎨${img.style_cluster}`, 'bg-black/60 text-fuchsia-200')}
        {img.dup_group != null && badge(`≈${img.dup_group}`, 'bg-black/60 text-fuchsia-200')}
        {img.semantic_dup_group != null && badge(`✂${img.semantic_dup_group}`, 'bg-black/60 text-orange-200')}
      </span>
      {/* ▶ starts the fast-triage lightbox AT this image. It's a separate hit
          target on purpose: the tile's own click still (de)selects for the bulk
          ✓/✕/⬆ bar, so neither use loses its gesture. */}
      {/* 🏷️ MOVED HERE, out of the badge cluster in the top-left corner.
          Everything up there is a STATE READOUT — ✓, ✕, ⬆, the flags, the person
          and framing chips — and none of it is clickable. A button dropped in the
          middle of that row does not read as an action; in the maintainer's own
          words, "otherwise it gets drowned at the top with the icons that aren't
          clickable". The tile's two real actions live in this bottom-right group,
          so the third one joins them, in the same clothes.

          AND IT ONLY APPEARS WHEN IT CAN DELIVER. The chips ARE the words of the
          caption, so on an image whose caption yields none the picker could only
          say "This caption has no word worth filtering on." — a button promising
          something it cannot do. That silence has already cost once, the other way
          round: the feature had shipped for two days and was read as absent,
          because the bank simply had no captions. So the button says WHY it is
          not there, exactly the way a semantic action names its missing index
          on the pass row — a shipped feature that says nothing is indistinguishable
          from one that does not exist. */}
      {tagChips.length > 0 ? (
        <button type="button" onClick={onTags}
          title={`Filter the bank by this image's tags — ${img.caption}`}
          aria-label={`Use the tags of ${img.name} as a filter`}
          className="absolute bottom-1 right-11 rounded bg-black/60 px-1 text-[11px] text-emerald-200 hover:bg-black/80">🏷️</button>
      ) : (
        <span
          title={img.caption
            ? '🏷️ Tags — this caption has no word worth filtering on (the chips are the caption\'s own words)'
            : '🏷️ Tags (needs a caption) — run 🏷️ Caption on this bank and the chips appear here'}
          aria-label={img.caption
            ? 'Tags unavailable: this caption has no word worth filtering on'
            : 'Tags unavailable: this image has no caption yet'}
          className="absolute bottom-1 right-11 rounded bg-black/40 px-1 text-[11px] text-white/35">🏷️</span>
      )}
      <button type="button" onClick={onReview}
        title="Review from this image — full size, one at a time, with Keep/Reject/Skip"
        aria-label={`Review from ${img.name}`}
        className="absolute bottom-1 right-6 rounded bg-black/60 px-1 text-[11px] text-white hover:bg-black/80">▶</button>
      {/* ⛶ serves what the bank RESOLVES for this image, and on an edited row that
          is the crop/upscale — so the tooltip stops calling it "the original file",
          which a ✂ crop makes visibly false, and the version key travels with it
          (this route is cached too). Renamed rather than made conditional: a
          computed title is invisible to the surface-inventory guard, and this
          button is worth keeping under it. */}
      <a href={`/api/bank/${bankId}/file/${img.id}${imageVersionQuery(img)}`}
        target="_blank" rel="noreferrer"
        title="Open this image full size, as the Bank shows it — your own file on disk is never modified"
        aria-label={`Open ${img.name} full size`}
        className="absolute bottom-1 right-1 rounded bg-black/60 px-1 text-[11px] text-white no-underline hover:bg-black/80">⛶</a>
    </li>
  )
}
