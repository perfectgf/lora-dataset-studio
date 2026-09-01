// Historique des prompts de test : une BANDE des plus récents + la porte vers
// 📚 Saved prompts, qui porte tout le reste.
//
// CE QUI A CHANGÉ ET POURQUOI (2026-09-01). Ce composant rendait la totalité de
// l'historique d'un bloc — 167 entrées sur une install réelle — en vignettes de
// 32×40 px avec ~30 caractères de prompt, sans hauteur bornée ni recherche. À
// cette largeur le texte ne distinguait rien (62 de ces 167 entrées partageaient
// leurs 30 premiers caractères avec une autre), donc la seule chose capable
// d'identifier un prompt était son image — et c'est elle qu'on avait réduite au
// plus petit format de l'app, sept fois plus petite que les propres résultats du
// Studio (`ResultTile`, 80×112) et quatorze fois plus petite que le navigateur
// de prompts 🌐 Civitai (`CivitaiBrowserModal`, 112×160), pour exactement le
// même geste : choisir un prompt en regardant ce qu'il produit.
//
// La bande garde donc les quelques derniers, à une taille où on les reconnaît,
// et le mur devient une fenêtre — cherchable, où le prompt se lit en entier.
//
// 📝 LOT — chaque carte porte une case à cocher. Cocher n'écrit RIEN dans le
// champ prompt (c'est le rôle du clic sur la carte, inchangé) : la coche décrit
// ce que le prochain lancement doit rejouer, une génération par prompt coché,
// avec les réglages courants du panneau. Aucune case cochée = le panneau se
// comporte exactement comme avant, prompt du champ compris.
//
// Le composant est monté par les DEUX surfaces (Test Studio du dataset et
// panneau « Generate from the board » du canvas) via PromptField/RunSetupPanel :
// la bande, la fenêtre et le lot existent des deux côtés par construction, pas
// par duplication.
import { useState } from 'react';
import { HelpBadge } from '../../../help/HelpMode';
import { datasetThumbUrl } from '../../../utils/datasetThumbUrl';
import SavedPromptsModal from './SavedPromptsModal';
import { normalizeSavedPrompt } from './savedPrompts';

// Combien de prompts restent sous la main, sans ouvrir la fenêtre. Six tient sur
// une rangée en large et sur deux à 400 px, et couvre le cas courant (« reprends
// celui d'il y a deux essais ») ; au-delà, chercher bat faire défiler.
const INLINE = 6;
// Barreau de vignette : la tuile fait 96×128 CSS, 256 la sert nette en écran 2×.
const THUMB_SIDE = 256;

export default function RecentPrompts({
  items, datasetId, selectedPrompt, onPick, onDelete,
  batch = null, onToggleBatch = null, onClearBatch = null,
}) {
  const [browserOpen, setBrowserOpen] = useState(false);
  // Le lot n'est proposé que si l'hôte l'accepte — un appelant qui ne passe pas
  // onToggleBatch garde le composant sans cases à cocher, des deux côtés.
  const batchable = typeof onToggleBatch === 'function';
  const picked = Array.isArray(batch) ? batch : [];
  const total = items.length;

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
        <span className="text-content-subtle text-[0.5625rem] uppercase">
          Saved prompts — click a card to reload it · thumbnail = an image you liked
        </span>
        {batchable && <HelpBadge topic="studio-prompt-batch" />}
        {batchable && picked.length > 0 && (
          <span className="flex items-center gap-1.5">
            <span className="rounded bg-purple-500/20 px-1.5 py-0.5 text-[0.5625rem] font-semibold text-purple-200 tabular-nums">
              {picked.length} selected
            </span>
            <button type="button" onClick={onClearBatch}
              className="inline-flex min-h-10 items-center px-1 text-content-subtle text-[0.5625rem] underline decoration-dotted hover:text-content lg:min-h-0 lg:px-0">
              Clear
            </button>
          </span>
        )}
      </div>
      {batchable && (
        <p className="m-0 text-content-subtle text-[0.5625rem]">
          Tick several prompts to generate them all in one run — same checkpoints,
          same settings, one image set per prompt.
        </p>
      )}

      <div className="flex flex-wrap items-stretch gap-1.5">
        {items.slice(0, INLINE).map((item) => {
          const p = normalizeSavedPrompt(item);
          const sel = selectedPrompt === p.prompt;
          const inBatch = picked.includes(p.prompt);
          // La carte porte la bordure ; cocher et supprimer sont des boutons
          // POSÉS DESSUS (frères du bouton de rechargement, jamais imbriqués) —
          // et toujours visibles, pas révélés au survol : il n'y a pas de survol
          // sur un écran tactile.
          return (
            <div key={p.prompt}
              // `shrink-0` : la tuile a une largeur voulue, pas négociable. Sans
              // lui elle reste un élément flex compressible, et une rangée
              // serrée la rognerait — une vignette qui rétrécit toute seule est
              // exactement la panne que cette refonte corrige. Elle passe à la
              // ligne, elle ne maigrit pas.
              className={`relative w-24 shrink-0 overflow-hidden rounded-lg border ${
                inBatch
                  ? 'border-purple-400 bg-purple-500/25'
                  : sel ? 'border-purple-400/60 bg-purple-500/20' : 'border-border bg-surface'}`}>
              <button type="button" onClick={() => onPick(p.prompt)} title={p.prompt}
                className="block w-full text-left">
                <span className="relative block">
                  {p.thumbnail ? (
                    <img src={datasetThumbUrl(
                      `/api/dataset/${p.thumbDatasetId ?? datasetId}/img/${encodeURIComponent(p.thumbnail)}`,
                      THUMB_SIDE)}
                      alt="" loading="lazy" decoding="async"
                      className="block h-32 w-24 object-cover" />
                  ) : (
                    <span className="flex h-32 w-24 items-center justify-center bg-app/60 px-1 text-center text-[0.5625rem] leading-snug text-content-subtle">
                      No image yet
                    </span>
                  )}
                  {p.count > 0 && (
                    <span className="absolute bottom-1 left-1 rounded bg-black/70 px-1 text-[0.5625rem] tabular-nums text-white/90">
                      {p.count}{p.liked ? ' 👍' : ''}
                    </span>
                  )}
                </span>
                {/* ⚠️ Le clamp est sur ce SPAN, jamais sur le bouton : mesuré en
                    navigateur, `-webkit-line-clamp` est SANS EFFET sur un
                    <button> (Blink refuse de lui donner un display -webkit-box)
                    et la boîte grandit alors avec le texte. Sur un span à
                    l'intérieur du bouton, il coupe correctement. Pas de `block`
                    non plus : les deux utilitaires écrivent `display`.
                    La hauteur est fixée EN PLUS du clamp — h-9 (36px) = py-1 ×2
                    + 2 × leading 14px, un multiple exact de la ligne — pour que
                    la coupe tombe entre deux lignes et jamais au milieu des
                    lettres, quel que soit l'arrondi du navigateur. */}
                <span className={`h-9 px-1 py-1 text-[0.625rem] leading-[0.875rem] line-clamp-2 ${
                  sel ? 'text-purple-200' : 'text-content-muted'}`}>
                  {p.prompt}
                </span>
              </button>
              {batchable && (
                <button type="button" role="checkbox" aria-checked={inBatch}
                  onClick={() => onToggleBatch(p.prompt)}
                  title={inBatch ? 'Remove this prompt from the batch' : 'Add this prompt to the batch'}
                  aria-label={inBatch ? 'Remove this prompt from the batch' : 'Add this prompt to the batch'}
                  className={`absolute left-1 top-1 flex h-10 w-10 items-center justify-center rounded bg-black/60 text-[0.6875rem] lg:h-5 lg:w-5 ${
                    inBatch ? 'text-purple-200' : 'text-white/70 hover:text-white'}`}>
                  <span aria-hidden>{inBatch ? '☑' : '☐'}</span>
                </button>
              )}
              {onDelete && (
                <button type="button"
                  onClick={() => {
                    const n = p.count ? ` and its ${p.count} test image(s)` : '';
                    if (window.confirm(`Delete this saved prompt${n}?`)) onDelete(p.prompt);
                  }}
                  title="Delete this saved prompt (and its test images)"
                  aria-label="Delete this saved prompt"
                  className="absolute right-1 top-1 flex h-10 w-10 items-center justify-center rounded bg-black/60 text-[0.6875rem] text-red-300/80 hover:bg-red-500/40 hover:text-red-200 lg:h-5 lg:w-5">
                  🗑
                </button>
              )}
            </div>
          );
        })}

        <button type="button" onClick={() => setBrowserOpen(true)}
          title="Search, read in full and manage every prompt you have launched a test with"
          className="flex min-h-[10rem] w-24 shrink-0 flex-col items-center justify-center gap-1 rounded-lg border border-dashed border-border bg-surface px-1 text-center text-[0.625rem] leading-snug text-content-muted hover:border-purple-400/60 hover:text-content">
          <span aria-hidden className="text-base">📚</span>
          Browse all
          <span className="tabular-nums font-semibold">{total}</span>
          {total > INLINE && (
            <span className="text-[0.5625rem] text-content-subtle">
              +{total - INLINE} more
            </span>
          )}
        </button>
      </div>

      {/* Montée SEULEMENT à l'ouverture, pas rendue-puis-cachée : elle dessine
          tout l'historique (~170 lignes) et elle lit le contexte des toasts.
          Une fenêtre fermée ne doit coûter ni l'un ni l'autre à ses hôtes. */}
      {browserOpen && (
        <SavedPromptsModal
          open onClose={() => setBrowserOpen(false)}
          items={items} datasetId={datasetId} selectedPrompt={selectedPrompt}
          onPick={onPick} onDelete={onDelete}
          batch={batch} onToggleBatch={onToggleBatch} onClearBatch={onClearBatch} />
      )}
    </div>
  );
}
