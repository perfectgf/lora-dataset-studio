/**
 * Toute modale du Studio se PORTAILLE — sinon elle est plafonnée et découpée.
 *
 * ── Le bug qui a produit ce fichier (mesuré à l'écran le 2026-09-02) ─────────
 * Les 👍/👎 des cellules de résultat se peignaient PAR-DESSUS les prompts du
 * navigateur 🌐 Civitai, modale ouverte. Ce n'était pas un z-index trop bas : la
 * modale portait déjà `z-[9999]`.
 *
 * `StudioRunSetup` est monté dans l'`<aside lg:sticky lg:top-16
 * lg:overflow-auto>` de `ComparisonStudio`. Deux conséquences, toutes deux
 * invisibles depuis le composant :
 *   · `position: sticky` OUVRE un contexte d'empilement. Un z-index posé dedans
 *     est plafonné PAR l'aside : il ne peut pas passer au-dessus de la grille de
 *     résultats, qui est la SŒUR de l'aside et vient après elle dans le DOM.
 *     Monter le nombre ne change rien — 9999 dans un contexte qui vaut 0 reste
 *     sous un frère qui vaut 1.
 *   · `overflow-auto` DÉCOUPE en plus l'enfant au cadre de l'aside.
 * `createPortal(…, document.body)` sort du contexte fautif. C'est le seul fix.
 *
 * ── Pourquoi un test de SOURCE pour un bug de RENDU ─────────────────────────
 * Parce que rien d'autre ne l'attrape :
 *   · un test de source lit des classes, il ne compose pas de calques ;
 *   · le harnais SSR (`renderToStaticMarkup`) exécute le composant mais n'a
 *     AUCUN layout — il ne peut pas voir un empilement ;
 *   · la sonde responsive n'attrape pas celui-ci non plus : un `data-probe-layer`
 *     est explicitement « apparié avec rien » dans le contrôle de chevauchement,
 *     donc marquer la modale comme couche la SORT du contrôle. Sa mesure verte
 *     sur cet écran est correcte et ne dit rien de ce bug.
 * Il reste donc la capture d'écran — et cette règle, qui empêche la prochaine
 * modale de ce dossier de rentrer dans le même piège sans que personne ne
 * repasse par un navigateur.
 *
 * La liste s'ÉNUMÈRE : tout `.jsx` du dossier qui peint un plein-écran
 * (`fixed inset-0`) est tenu de se portailler, sans qu'on l'inscrive nulle part.
 */
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import test from 'node:test';

const HERE = new URL('./', import.meta.url);

/** [{ file, source }] de chaque composant qui peint un plein-écran, dans ce
 *  dossier ET SES SOUS-DOSSIERS.
 *
 *  La récursion n'est pas une élégance : la première version lisait le dossier À
 *  PLAT, et `video/MotionModelDialog.jsx` — une modale plein-écran, montée par le
 *  Video Test Studio — lui était donc invisible. Une garde qui s'arrête à un
 *  niveau de profondeur laisse à la prochaine arborescence le soin de la
 *  contourner sans le faire exprès. */
function fullScreenOverlays(dir = HERE, prefix = '') {
  const out = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const at = new URL(entry.name + (entry.isDirectory() ? '/' : ''), dir);
    if (entry.isDirectory()) {
      out.push(...fullScreenOverlays(at, `${prefix}${entry.name}/`));
    } else if (entry.name.endsWith('.jsx')) {
      const source = readFileSync(at, 'utf8');
      if (/className="fixed inset-0/.test(source)) {
        out.push({ file: prefix + entry.name, source });
      }
    }
  }
  return out;
}

test('la liste des modales du Studio n’est pas vide — sinon la garde ne garde rien', () => {
  const found = fullScreenOverlays();
  assert.ok(found.length >= 2,
    `attendu au moins 2 plein-écrans dans ce dossier, trouvé ${found.length} `
    + '— le motif de détection a dû changer, la garde ne prouve plus rien');
});

test('l’énumération DESCEND vraiment dans les sous-dossiers', () => {
  /* L'épingle qui manquait, et son histoire : la première version lisait le
     dossier à plat, donc `video/MotionModelDialog.jsx` lui échappait. Une fois
     ce fichier portaillé, RETIRER la récursion ne fait plus rougir le test
     ci-dessous — il verrait juste moins de fichiers, tous conformes. La règle
     serait alors gardée par rien, et le prochain sous-dossier repartirait
     invisible.
     On exige donc que l'énumération RAPPORTE au moins un fichier venu d'un
     sous-dossier : un compte n'est une preuve que s'il est EXERCÉ. */
  const found = fullScreenOverlays();
  const nested = found.filter(({ file }) => file.includes('/'));
  assert.ok(nested.length >= 1,
    'aucun plein-écran trouvé sous un sous-dossier : la récursion est morte, '
    + `et ${found.length} fichier(s) à plat ne prouvent rien de l'arborescence`);
  assert.ok(nested.some(({ file }) => file.endsWith('MotionModelDialog.jsx')),
    'video/MotionModelDialog.jsx n’est plus vu par l’énumération — '
    + `vus : ${nested.map((f) => f.file).join(', ') || 'aucun'}`);
});

test('CHAQUE modale plein-écran du Studio est portaillée sur document.body', () => {
  const offenders = fullScreenOverlays()
    .filter(({ source }) => !(/from 'react-dom'/.test(source)
      && /createPortal\(/.test(source)
      && /document\.body/.test(source)));
  assert.deepEqual(offenders.map((o) => o.file), [],
    'Ces modales du Studio ne sont pas portaillées. Montées sous l’`<aside '
    + 'lg:sticky lg:overflow-auto>` de ComparisonStudio, leur z-index est '
    + 'plafonné par le contexte d’empilement du sticky et leur boîte est '
    + 'découpée par l’overflow — la page se peint par-dessus. '
    + 'Fix : `return createPortal(<div …>, document.body)`, comme '
    + 'CaptionEditorDialog. Aucun z-index ne répare ça de l’intérieur.');
});

test('l’aside qui piège l’empilement est toujours celui décrit ci-dessus', () => {
  /* Si ce panneau perd son `sticky`/`overflow`, la raison d’être du portail
     change — et quelqu’un doit le relire plutôt que de trouver un commentaire
     qui parle d’un CSS disparu. Épingler la cause, pas seulement le remède. */
  const owner = readFileSync(new URL('./ComparisonStudio.jsx', HERE), 'utf8');
  assert.match(owner, /<aside className="[^"]*lg:sticky[^"]*lg:overflow-auto/);
});
