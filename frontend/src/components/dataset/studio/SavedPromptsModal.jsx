// 📚 Saved prompts — tout l'historique des prompts de test, parcouru PAR L'IMAGE.
//
// POURQUOI CETTE FENÊTRE EXISTE. L'historique n'est pas une poignée de puces :
// sur une vraie install il compte ~170 entrées, et il était rendu d'un bloc, en
// vignettes de 32×40 px, avec ~30 caractères de prompt. Or les prompts de test
// sont longs (médiane ~500 caractères) et commencent tous pareil : sur cet
// historique-là, 62 entrées sur 167 partageaient leurs 30 premiers caractères
// avec une autre, dont 19 qui affichaient exactement « Photograph of a young
// woman wi… ». Le texte ne pouvait donc PAS distinguer une carte d'une autre à
// cette largeur — seule l'image qu'elle a produite le peut, et c'était elle
// qu'on avait réduite. Ici l'image est tirée au même barreau que le navigateur
// 🌐 Civitai (même geste, même taille) et le prompt prend toute la largeur qui
// reste, cinq lignes, dépliables.
//
// CE QUE LA FENÊTRE APPORTE QUE LA BANDE NE POUVAIT PAS : une recherche. À 167
// entrées c'est le seul moyen de retrouver un prompt, et c'est la forme que la
// Bank, le Canvas, Caption Lab et la bibliothèque de datasets emploient déjà.
//
// TOUS LES VERBES SONT ICI. La bande n'affiche que les plus récents ; recharger,
// cocher pour le lot et supprimer existent des deux côtés — la fenêtre est la
// surface de gestion, pas une vue en lecture seule. Le lot n'est proposé que si
// l'hôte l'accepte (`onToggleBatch`), exactement comme dans la bande : « Generate
// from the board » ne le passe pas, il ne doit donc pas apparaître.
import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useToast } from '../../common/Toast';
import { useFocusTrap } from '../../../hooks/useFocusTrap';
import { HelpBadge } from '../../../help/HelpMode';
import { datasetThumbUrl } from '../../../utils/datasetThumbUrl';
import { filterSavedPrompts, normalizeSavedPrompt } from './savedPrompts';

// Le barreau de vignette demandé au serveur. 384 pour une tuile dessinée à
// 144×192 CSS : les lignes sont paresseuses (`loading="lazy"`), seules ~4 sont
// visibles à la fois, et regarder l'image EST le but de cette fenêtre.
const THUMB_SIDE = 384;

/** Le contenu de la fenêtre, SANS le portail — exporté à part pour que les tests
 *  puissent l'exécuter : `renderToStaticMarkup` ne sait pas rendre un portail,
 *  et une fenêtre qu'aucun test ne peut rendre est une fenêtre non mesurée. */
export function SavedPromptsPanel({
  open, onClose, items, datasetId, selectedPrompt, onPick, onDelete,
  batch = null, onToggleBatch = null, onClearBatch = null,
}) {
  const toast = useToast();
  const ref = useRef(null);
  useFocusTrap(ref, open);
  const [query, setQuery] = useState('');
  const [expanded, setExpanded] = useState(() => new Set());
  // La page derrière ne défile plus tant que la fenêtre est ouverte — même
  // verrou que les autres dialogues de l'app (CaptionLabPicker).
  useEffect(() => {
    if (!open) return undefined;
    const before = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = before; };
  }, [open]);

  const batchable = typeof onToggleBatch === 'function';
  const picked = Array.isArray(batch) ? batch : [];
  const shown = useMemo(() => filterSavedPrompts(items, query), [items, query]);

  if (!open) return null;

  const total = Array.isArray(items) ? items.length : 0;
  const copyPrompt = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      toast.success('Prompt copied');
    } catch {
      toast.error('Could not copy — select the text and copy it manually.');
    }
  };
  const toggleExpand = (p) => setExpanded((cur) => {
    const next = new Set(cur);
    if (next.has(p)) next.delete(p); else next.add(p);
    return next;
  });
  // Choisir un prompt referme : la fenêtre a rempli son office, et laisser le
  // panneau de lancement caché derrière une modale ouverte est ce qui fait
  // régler un run sans voir le champ qu'on vient de remplir.
  const use = (p) => { onPick(p); onClose(); };

  return (
    <div className="fixed inset-0 z-[9999] bg-black/70 flex items-center justify-center p-4"
      onKeyDown={(e) => { if (e.key === 'Escape') { e.preventDefault(); onClose(); } }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      {/* Marqué chrome+layer : la sonde responsive ne mesure cibles tactiles et
          troncature qu'À L'INTÉRIEUR d'un [data-probe-chrome] — un dialogue non
          marqué n'est pas « propre », il est NON MESURÉ. */}
      <div role="dialog" aria-modal="true" aria-label="Browse saved test prompts" ref={ref}
        data-probe-chrome="saved-prompts" data-probe-layer
        className="w-full max-w-4xl max-h-[88vh] rounded-2xl border border-border bg-surface-overlay p-4 flex flex-col gap-3 shadow-xl">
        <div className="flex items-center justify-between gap-2">
          <h2 className="text-content text-sm font-semibold flex items-center gap-1.5">
            <span aria-hidden>📚</span> Saved prompts
            <span className="text-content-subtle font-normal tabular-nums">({total})</span>
            <HelpBadge topic="studio-saved-prompts" />
          </h2>
          <button type="button" onClick={onClose} aria-label="Close"
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-border bg-app text-content-muted hover:text-content lg:h-8 lg:w-8">×</button>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <input type="search" value={query} onChange={(e) => setQuery(e.target.value)}
            aria-label="Search saved prompts"
            placeholder="Search your prompts… (e.g. bathroom mirror)"
            className="min-h-10 min-w-0 flex-1 rounded-lg border border-border bg-app/60 px-2.5 py-1.5 text-content text-[0.75rem] lg:min-h-0" />
          <span className="text-content-subtle text-[0.6875rem] tabular-nums" role="status">
            {query.trim() ? `${shown.length} of ${total}` : `${total} prompts`}
          </span>
          {batchable && picked.length > 0 && (
            <span className="flex items-center gap-1.5">
              <span className="rounded bg-purple-500/20 px-1.5 py-0.5 text-[0.6875rem] font-semibold text-purple-200 tabular-nums">
                {picked.length} selected
              </span>
              <button type="button" onClick={onClearBatch}
                className="inline-flex min-h-10 items-center px-1 text-content-subtle text-[0.6875rem] underline decoration-dotted hover:text-content lg:min-h-0 lg:px-0">
                Clear
              </button>
            </span>
          )}
        </div>
        {batchable && (
          <p className="m-0 text-content-subtle text-[0.6875rem] leading-snug">
            Tick several prompts to generate them all in one run — same checkpoints,
            same settings, one image set per prompt. Ticking writes nothing into the
            prompt field; “⤵ Use prompt” does that.
          </p>
        )}

        <div className="min-h-0 flex-1 overflow-y-auto flex flex-col gap-2 pr-1">
          {shown.map((item) => {
            const p = normalizeSavedPrompt(item);
            const sel = selectedPrompt === p.prompt;
            const inBatch = picked.includes(p.prompt);
            const isOpen = expanded.has(p.prompt);
            return (
              <div key={p.prompt}
                className={`flex gap-2.5 rounded-xl border p-2 ${
                  inBatch ? 'border-purple-400 bg-purple-500/15'
                    : sel ? 'border-purple-400/60 bg-purple-500/10' : 'border-border bg-surface'}`}>
                {p.thumbnail ? (
                  <img
                    src={datasetThumbUrl(
                      `/api/dataset/${p.thumbDatasetId ?? datasetId}/img/${encodeURIComponent(p.thumbnail)}`,
                      THUMB_SIDE)}
                    alt="" loading="lazy" decoding="async"
                    className="w-28 sm:w-36 h-40 sm:h-48 shrink-0 object-cover rounded-lg border border-border" />
                ) : (
                  // Pas de vignette = ce prompt n'a jamais rendu d'image (ou elles
                  // ont été supprimées). Un « ? » de la taille d'une image ne dit
                  // rien : la place revient au texte, seul signal qui reste.
                  <div className="w-28 sm:w-36 h-40 sm:h-48 shrink-0 rounded-lg border border-dashed border-border bg-app/40 flex items-center justify-center px-2 text-center text-content-subtle text-[0.625rem] leading-snug">
                    No image yet
                  </div>
                )}
                <div className="flex flex-col gap-1.5 min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[0.625rem] text-content-subtle">
                    {p.count > 0
                      ? <span className="tabular-nums">{p.count} test image{p.count > 1 ? 's' : ''}</span>
                      : <span className="rounded border border-border bg-app/60 px-1.5 py-px">never run</span>}
                    {p.liked && <span title="The thumbnail is an image you liked">👍 liked</span>}
                    {sel && <span className="text-purple-300">in the prompt field</span>}
                  </div>
                  {/* ⚠️ Le clamp vit sur le SPAN, pas sur le bouton. Mesuré en
                      navigateur : `-webkit-line-clamp` n'a AUCUN effet posé sur
                      un <button> — Blink lui refuse le display `-webkit-box` —
                      et la ligne grandissait alors avec le prompt, qui monte à
                      2000 caractères. Sur un span à l'intérieur, il coupe. */}
                  <button type="button" onClick={() => toggleExpand(p.prompt)}
                    title={isOpen ? 'Collapse the prompt' : 'Show the full prompt'}
                    className="m-0 min-h-10 text-left text-content text-[0.75rem] leading-snug lg:min-h-0">
                    <span className={`whitespace-pre-wrap break-words ${isOpen ? '' : 'line-clamp-5'}`}>
                      {p.prompt}
                    </span>
                  </button>
                  <div className="mt-auto flex flex-wrap items-center gap-1.5">
                    {batchable && (
                      <button type="button" role="checkbox" aria-checked={inBatch}
                        onClick={() => onToggleBatch(p.prompt)}
                        title={inBatch ? 'Remove this prompt from the batch' : 'Add this prompt to the batch'}
                        className={`px-2 py-1 min-h-10 lg:min-h-0 rounded border text-[0.6875rem] ${
                          inBatch
                            ? 'border-purple-400 bg-purple-500/25 text-purple-200'
                            : 'border-border bg-app text-content-muted hover:text-content'}`}>
                        <span aria-hidden>{inBatch ? '☑' : '☐'}</span> Batch
                      </button>
                    )}
                    <button type="button" onClick={() => copyPrompt(p.prompt)}
                      title="Copy this prompt"
                      className="px-2 py-1 min-h-10 lg:min-h-0 rounded border border-border bg-app text-content-muted text-[0.6875rem] hover:text-content">
                      📋 Copy
                    </button>
                    {onDelete && (
                      <button type="button"
                        onClick={() => {
                          const n = p.count ? ` and its ${p.count} test image(s)` : '';
                          if (window.confirm(`Delete this saved prompt${n}?`)) onDelete(p.prompt);
                        }}
                        title="Delete this saved prompt (and its test images)"
                        aria-label="Delete this saved prompt"
                        className="px-2 py-1 min-h-10 lg:min-h-0 rounded border border-border bg-app text-red-300/70 text-[0.6875rem] hover:text-red-300 hover:bg-red-500/15">
                        🗑 Delete
                      </button>
                    )}
                    <button type="button" onClick={() => use(p.prompt)}
                      title="Load this prompt into the prompt field"
                      className="ml-auto px-2.5 py-1 min-h-10 lg:min-h-0 rounded-lg bg-gradient-primary text-gray-950 text-[0.6875rem] font-semibold">
                      ⤵ Use prompt
                    </button>
                  </div>
                </div>
              </div>
            );
          })}

          {shown.length === 0 && (
            <p className="m-0 rounded-lg border border-border bg-surface px-3 py-6 text-center text-content-subtle text-[0.75rem]">
              {total === 0
                ? 'No saved prompts yet — the prompts you launch a test with are kept here.'
                : `No saved prompt contains every word of “${query.trim()}”.`}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

/* PORTAILLÉE SUR <body>, et ce n'est pas cosmétique : le panneau de lancement vit
 * dans un `<aside class="lg:sticky lg:overflow-auto">`, et `position: sticky`
 * OUVRE UN CONTEXTE D'EMPILEMENT — un z-index posé dedans, si haut soit-il, ne
 * peut pas monter au-dessus de l'en-tête de l'app, et au-delà de `lg` la fenêtre
 * était en plus DÉCOUPÉE par le scroll de l'aside. Mesuré en navigateur avant le
 * correctif : l'en-tête et des morceaux de la page se peignaient par-dessus.
 * C'est le motif que CaptionLabPicker et le dialogue ▶ Continue emploient déjà,
 * pour la même raison (un dialogue rendu là où on ne le voit pas). */
export default function SavedPromptsModal(props) {
  if (!props.open) return null;
  return createPortal(<SavedPromptsPanel {...props} />, document.body);
}
