import GeneratedImageLightbox from '../../shared/GeneratedImageLightbox';
import { useCanvasImageImprove } from '../../../hooks/useCanvasImageImprove';
import { canImproveCanvasImage } from '../../../utils/canvasImprove';
import { imageFactsLine } from '../../../utils/generatedImageFacts';
import { galleryImproveLaunchMessage } from '../../../utils/appGallery';

/* 🔍 The Studio's viewer IS the app's viewer.

   It used to be a fifth lightbox (ResultLightbox) over the very same
   lora_test_image rows the Gallery and the Canvas show — and it knew almost
   nothing about them: no prompt, no extra LoRAs, no base model, no sampler,
   none of the verbs. Opening a comparison of two runs showed LESS about an
   image than opening the same row from the Gallery, which is exactly the gap
   that got reported. The backend now serves Studio cells as a superset of the
   shared gallery shape, so this file is just the thin adapter between the
   Studio's hosts and GeneratedImageLightbox.

   What the Studio knows and the viewer cannot:
     · the ORDERED comparison set and its WRAP-AROUND — comparing loops on
       purpose (A/B/A…), so both chevrons are always offered and stepping past
       the end returns to the start, unlike the feed hosts that stop AT their
       ends;
     · the 👍/👎 verdict that writes the run's ranking — passed through the
       `actions` slot with the counter, exactly the host-specific-extra that
       slot exists for.

   Deliberately NOT ported from the old lightbox: swipe navigation. The
   unified viewer's image pane pinches and pans, and a horizontal drag IS a
   pan there — two meanings on one gesture would fight. ‹ ›, ← / → and the
   counter remain. */
export default function StudioResultViewer({ img, items = [], onRate, onNavigate, onClose }) {
  const improveImage = useCanvasImageImprove({
    // The result lands at the head of the Gallery feed — same wording as the
    // Gallery host, because it is the same destination.
    launchMessage: galleryImproveLaunchMessage,
  });
  if (!img) return null;
  const idx = items.findIndex((it) => it.id === img.id);
  const hasNav = !!onNavigate && idx >= 0 && items.length > 1;
  const go = (delta) => {
    const n = items.length;
    onNavigate(items[(((idx + delta) % n) + n) % n]);
  };
  return (
    <GeneratedImageLightbox
      img={img}
      alt={imageFactsLine(img) || 'Studio render'}
      onClose={onClose}
      onImprove={canImproveCanvasImage(img) ? improveImage : undefined}
      onPrev={hasNav ? () => go(-1) : null}
      onNext={hasNav ? () => go(1) : null}
      datasetId={img.dataset_id ?? null}
      actions={onRate ? (
        <span className="inline-flex items-center gap-2">
          {hasNav && (
            <span className="text-[0.72rem] tabular-nums text-white/55">
              {idx + 1} / {items.length}
            </span>
          )}
          <button type="button" aria-pressed={img.rating === 1}
            onClick={(e) => { e.stopPropagation(); onRate(img.id, img.rating === 1 ? 0 : 1); }}
            className={`min-h-10 lg:min-h-0 rounded-lg border px-3 py-1.5 text-[0.75rem] font-semibold ${img.rating === 1
              ? 'border-green-400/60 bg-green-500/20 text-green-200'
              : 'border-white/25 text-white/85 hover:border-white/50'}`}>
            👍 {img.rating === 1 ? 'Liked ✓' : 'Like'}
          </button>
          <button type="button" aria-pressed={img.rating === -1}
            onClick={(e) => { e.stopPropagation(); onRate(img.id, img.rating === -1 ? 0 : -1); }}
            className={`min-h-10 lg:min-h-0 rounded-lg border px-3 py-1.5 text-[0.75rem] font-semibold ${img.rating === -1
              ? 'border-red-400/60 bg-red-500/20 text-red-200'
              : 'border-white/25 text-white/85 hover:border-white/50'}`}>
            👎 {img.rating === -1 ? 'Not a fan ✓' : 'Not a fan'}
          </button>
        </span>
      ) : null} />
  );
}
