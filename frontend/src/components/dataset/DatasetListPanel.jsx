import { useEffect, useState } from 'react';
import ShotIllustration from './ShotIllustration';
import TileSizeControl from '../shared/TileSizeControl';
import FullBackupControls from './FullBackupControls';
import { HelpBadge } from '../../help/HelpMode';
import { requestHelpTip } from '../../help/helpTips';
import {
  datasetKind, datasetMatches, groupDatasets, kindsPresent,
  normalizeCollapsedMap, normalizeTileSize,
} from '../../utils/datasetLibrary';

// Fixed gradient palette for the dataset avatars — deterministic per name so a
// dataset keeps its color across sessions (Tailwind needs literal class names).
const AVATAR_GRADIENTS = [
  'from-indigo-500 to-purple-500',
  'from-rose-500 to-orange-400',
  'from-emerald-500 to-teal-400',
  'from-sky-500 to-blue-600',
  'from-amber-500 to-pink-500',
  'from-fuchsia-500 to-violet-600',
];

function gradientFor(name = '') {
  let h = 0;
  for (let i = 0; i < name.length; i += 1) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return AVATAR_GRADIENTS[h % AVATAR_GRADIENTS.length];
}

/** The 3-step pipeline strip — what this page is for, at a glance. Only shown
 *  on an EMPTY library: returning users know the pipeline by heart. */
function PipelineSteps() {
  const steps = [
    { n: 1, icon: '📸', title: 'Reference photo', text: 'Upload one clear photo of the face.' },
    { n: 2, icon: '✨', title: 'Generate & curate', text: 'Synthesize varied shots, keep the best ones.' },
    { n: 3, icon: '🧬', title: 'Train the LoRA', text: 'Export or train — reuse the character anywhere.' },
  ];
  return (
    <ol className="grid grid-cols-1 sm:grid-cols-3 gap-2">
      {steps.map((s, i) => (
        <li key={s.n} className="relative flex items-start gap-2.5 rounded-lg border border-border bg-app/40 p-2.5">
          <span className="grid place-items-center w-8 h-8 shrink-0 rounded-full bg-primary/15 border border-primary/40 text-base"
            aria-hidden="true">{s.icon}</span>
          <span className="min-w-0">
            <span className="block text-content text-[0.75rem] font-semibold">
              <span className="text-indigo-300 mr-1">{s.n}.</span>{s.title}
            </span>
            <span className="block text-content-subtle text-[0.6875rem] leading-snug">{s.text}</span>
          </span>
          {i < steps.length - 1 && (
            <span className="hidden sm:block absolute -right-2 top-1/2 -translate-y-1/2 text-content-subtle z-10"
              aria-hidden="true">→</span>
          )}
        </li>
      ))}
    </ol>
  );
}

/** Empty state = the page's only "hero": what the app does, the 3-step strip,
 *  and a mini contact sheet of shot pictograms. */
function EmptyState() {
  const shots = [
    { framing: 'face', label: '' },
    { framing: 'face', label: 'Visage 3/4 gauche' },
    { framing: 'bust', label: '' },
    { framing: 'face', label: 'Profil droite' },
    { framing: 'body', label: '' },
    { framing: 'back', label: '' },
  ];
  return (
    <div className="mx-auto w-full max-w-4xl flex flex-col gap-3">
      <div className="rounded-xl border border-border bg-gradient-to-br from-surface to-app/60 p-3 flex flex-col gap-2.5">
        <p className="text-content-subtle text-xs">
          Build a consistent character: one reference photo becomes a curated,
          captioned training set for a LoRA you can use in every generator.
        </p>
        <PipelineSteps />
      </div>
      <div className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-border bg-app/30 px-4 py-8 text-center">
        <div className="grid grid-cols-6 gap-1.5" aria-hidden="true">
          {shots.map((s, i) => (
            <ShotIllustration key={i} framing={s.framing} label={s.label}
              className={`w-9 h-9 ${i === 0 ? 'text-indigo-300' : 'text-content-subtle'}`} />
          ))}
        </div>
        <p className="text-content-muted text-sm font-medium">No datasets yet</p>
        <p className="text-content-subtle text-xs max-w-xs">
          Create your first character with <span className="font-semibold text-content-muted">+ New dataset</span> —
          one reference photo is enough to start generating a full training set.
        </p>
      </div>
    </div>
  );
}

// Badges de famille des LoRA entraînés — mêmes couleurs que le LoraPicker du Studio.
const FAMILY_BADGE = {
  zimage: ['Z-Image', 'border-sky-400/40 bg-sky-500/10 text-sky-300'],
  sdxl: ['SDXL', 'border-violet-400/40 bg-violet-500/10 text-violet-300'],
  krea: ['Krea', 'border-amber-400/40 bg-amber-500/10 text-amber-300'],
  flux: ['FLUX.1', 'border-emerald-400/40 bg-emerald-500/10 text-emerald-300'],
  // rose: libre (fuchsia/cyan sont pris par les badges kind Concept/Style au-dessus
  // de la vignette — une couleur distincte évite de les confondre avec une famille).
  flux2klein: ['FLUX.2 Klein', 'border-rose-400/40 bg-rose-500/10 text-rose-300'],
  anima: ['Anima', 'border-teal-400/40 bg-teal-500/10 text-teal-300'],
};

// Display preferences — persisted globally (display settings, not dataset
// data; same pattern as datasetGridTileSize / the CloudRuns group folds).
// S = compact list rows (maximum density — the library is often browsed from
// a phone), M = the historical 2/3-column photo grid, L = large previews.
const TILE_SIZE_KEY = 'datasetLibraryTileSize';
const COLLAPSED_KEY = 'datasetLibraryCollapsed_v1';
const TILE_SIZE_TITLE = {
  S: 'Compact list — maximum density, browse many datasets at once',
  M: 'Medium tiles (default)',
  L: 'Large tiles — big reference previews',
};
const GRID_COLS = {
  M: 'grid grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-4',
  L: 'grid grid-cols-1 gap-2.5 sm:grid-cols-2',
};

// Kind filter chips — only rendered when at least two kinds coexist in the
// library. Transient on purpose: a persisted filter reads as lost datasets.
const KIND_CHIPS = {
  character: '🧑 Character',
  concept: '💡 Concept',
  style: '🎨 Style',
};

/** One-line status of a tile: how big, how far along. Text, not color-only. */
function tileStats(d) {
  const total = d.images_total ?? 0;
  const kept = d.images_kept ?? 0;
  const captioned = d.images_captioned ?? 0;
  if (!total) return 'empty';
  if (!kept) return `${total} img · none kept`;
  if (captioned >= kept) return `${kept} kept · ✓ captioned`;
  if (d.kind === 'style') return `${kept} kept · ${captioned}/${kept} required captions`;
  if (captioned > 0) return `${kept} kept · ${captioned}/${kept} captioned`;
  return `${kept} kept`;
}

/** Photo-first tile: the reference face IS the identity — lead with it. */
function DatasetTile({ d, onOpen, onDelete, onExportZip, onExportBackup }) {
  const canExportZip = (d.images_kept ?? 0) > 0;
  return (
    <div className="library-card group relative overflow-hidden rounded-xl border border-border bg-surface transition-colors hover:border-primary/40">
      <button type="button" onClick={() => onOpen(d.id)}
        aria-label={`Open the dataset ${d.name}`}
        className="block w-full text-left">
        <div className="relative aspect-[4/3] bg-app/60">
          {d.ref_filename ? (
            <img
              src={`/api/dataset/${d.id}/img/${encodeURIComponent(d.ref_filename)}`}
              alt="" loading="lazy" aria-hidden="true"
              className="h-full w-full object-cover" />
          ) : (
            <span className={`grid h-full w-full place-items-center bg-gradient-to-br ${gradientFor(d.name)} text-white text-3xl font-bold`}
              aria-hidden="true">
              {(d.name || '?').charAt(0).toUpperCase()}
            </span>
          )}
          {d.kind === 'concept' && (
            <span className="absolute left-1.5 top-1.5 rounded border border-fuchsia-400/40 bg-black/50 px-1.5 py-px text-[0.5625rem] font-semibold uppercase text-fuchsia-300 backdrop-blur-sm">
              💡 Concept
            </span>
          )}
          {d.kind === 'style' && (
            <span className="absolute left-1.5 top-1.5 rounded border border-cyan-400/40 bg-black/50 px-1.5 py-px text-[0.5625rem] font-semibold uppercase text-cyan-300 backdrop-blur-sm">
              🎨 Style
            </span>
          )}
        </div>
        <div className="flex flex-col gap-0.5 p-2.5">
          <span className="flex items-center gap-1.5 min-w-0">
            <span className="truncate text-sm font-semibold text-content">{d.name}</span>
            {(d.trained_families || []).map((f) => {
              const [lbl, cls] = FAMILY_BADGE[f] || [f, 'border-border bg-white/5 text-content-muted'];
              return (
                <span key={f} className={`shrink-0 rounded border px-1.5 py-px text-[0.5625rem] font-semibold uppercase ${cls}`}
                  title={`A ${lbl} LoRA has been trained from this dataset`}>
                  {lbl}
                </span>
              );
            })}
          </span>
          <span className={`truncate text-[0.6875rem] ${d.kind === 'style' ? 'text-cyan-300' : 'font-mono text-indigo-300'}`}>
            {d.kind === 'style' ? 'always-on · no activation trigger' : (d.trigger_word || '—')}
          </span>
          <span className="text-[0.6875rem] text-content-subtle">{tileStats(d)}</span>
        </div>
      </button>
      <div className="library-card__actions grid grid-cols-2 gap-1.5 border-t border-border px-2 py-2">
        <button type="button"
          onClick={() => onExportZip?.(d.id)}
          disabled={!canExportZip}
          title={canExportZip
            ? 'Download the kept images and captions as a training-ready ZIP'
            : 'Keep at least one image before exporting a training ZIP'}
          aria-label={`Export training ZIP for ${d.name}`}
          className="rounded-md border border-border bg-app/50 px-2 py-1 text-[0.6875rem] font-semibold text-content-muted transition-colors hover:border-primary/40 hover:bg-surface-raised hover:text-content disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:border-border disabled:hover:bg-app/50 disabled:hover:text-content-muted">
          ⬇ ZIP
        </button>
        <button type="button"
          onClick={() => onExportBackup?.(d.id)}
          title="Download a portable backup with all images, captions and settings"
          aria-label={`Export portable backup for ${d.name}`}
          className="rounded-md border border-border bg-app/50 px-2 py-1 text-[0.6875rem] font-semibold text-content-muted transition-colors hover:border-primary/40 hover:bg-surface-raised hover:text-content">
          💾 Backup
        </button>
      </div>
      {onDelete && (
        <button type="button"
          onClick={() => {
            if (window.confirm(`Permanently delete the dataset "${d.name}" and all its images? This cannot be undone.`)) onDelete(d.id);
          }}
          title="Delete this dataset" aria-label={`Delete the dataset ${d.name}`}
          className="library-card__actions absolute right-1.5 top-1.5 rounded-lg border border-red-500/40 bg-black/50 px-2 py-1 text-xs text-red-300 opacity-70 backdrop-blur-sm transition-opacity hover:bg-red-500/25 hover:opacity-100">
          🗑
        </button>
      )}
    </div>
  );
}

/** Compact row for the S size: identity at a glance, one dataset per line,
 *  icon-only actions. Everything the photo tile shows, at list density. */
function DatasetRow({ d, onOpen, onDelete, onExportZip, onExportBackup }) {
  const canExportZip = (d.images_kept ?? 0) > 0;
  const kind = datasetKind(d);
  const iconBtn = 'grid h-7 w-7 shrink-0 place-items-center rounded-md border border-border bg-app/50 text-xs text-content-muted transition-colors hover:border-primary/40 hover:bg-surface-raised hover:text-content';
  return (
    <div className="library-card flex items-center gap-1.5 rounded-lg border border-border bg-surface pr-1.5 transition-colors hover:border-primary/40">
      <button type="button" onClick={() => onOpen(d.id)}
        aria-label={`Open the dataset ${d.name}`}
        className="flex min-w-0 flex-1 items-center gap-2.5 py-1.5 pl-1.5 text-left">
        <span className="relative h-10 w-10 shrink-0 overflow-hidden rounded-md bg-app/60">
          {d.ref_filename ? (
            <img
              src={`/api/dataset/${d.id}/img/${encodeURIComponent(d.ref_filename)}`}
              alt="" loading="lazy" aria-hidden="true"
              className="h-full w-full object-cover" />
          ) : (
            <span className={`grid h-full w-full place-items-center bg-gradient-to-br ${gradientFor(d.name)} text-base font-bold text-white`}
              aria-hidden="true">
              {(d.name || '?').charAt(0).toUpperCase()}
            </span>
          )}
        </span>
        <span className="flex min-w-0 flex-col gap-0.5">
          <span className="flex min-w-0 items-center gap-1.5">
            {kind !== 'character' && (
              <span title={kind === 'concept' ? 'Concept dataset' : 'Style dataset'} aria-hidden="true"
                className="shrink-0 text-[0.6875rem]">{kind === 'concept' ? '💡' : '🎨'}</span>
            )}
            <span className="truncate text-xs font-semibold text-content">{d.name}</span>
            {(d.trained_families || []).map((f) => {
              const [lbl, cls] = FAMILY_BADGE[f] || [f, 'border-border bg-white/5 text-content-muted'];
              return (
                <span key={f} className={`shrink-0 rounded border px-1 py-px text-[0.5rem] font-semibold uppercase ${cls}`}
                  title={`A ${lbl} LoRA has been trained from this dataset`}>
                  {lbl}
                </span>
              );
            })}
          </span>
          <span className="truncate text-[0.625rem] text-content-subtle">
            <span className={kind === 'style' ? 'text-cyan-300' : 'font-mono text-indigo-300'}>
              {kind === 'style' ? 'always-on' : (d.trigger_word || '—')}
            </span>
            {' · '}{tileStats(d)}
          </span>
        </span>
      </button>
      <div className="library-card__actions flex shrink-0 items-center gap-1">
        <button type="button" onClick={() => onExportZip?.(d.id)} disabled={!canExportZip}
          title={canExportZip
            ? 'Download the kept images and captions as a training-ready ZIP'
            : 'Keep at least one image before exporting a training ZIP'}
          aria-label={`Export training ZIP for ${d.name}`}
          className={`${iconBtn} disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:border-border disabled:hover:bg-app/50 disabled:hover:text-content-muted`}>
          ⬇
        </button>
        <button type="button" onClick={() => onExportBackup?.(d.id)}
          title="Download a portable backup with all images, captions and settings"
          aria-label={`Export portable backup for ${d.name}`}
          className={iconBtn}>
          💾
        </button>
        {onDelete && (
          <button type="button"
            onClick={() => {
              if (window.confirm(`Permanently delete the dataset "${d.name}" and all its images? This cannot be undone.`)) onDelete(d.id);
            }}
            title="Delete this dataset" aria-label={`Delete the dataset ${d.name}`}
            className="grid h-7 w-7 shrink-0 place-items-center rounded-md border border-red-500/40 bg-app/50 text-xs text-red-300 transition-colors hover:bg-red-500/25">
            🗑
          </button>
        )}
      </div>
    </div>
  );
}

/** The creation form — folded behind "+ New dataset" (auto-open on an empty
 *  library). Fields unchanged from the historical always-open card. */
function NewDatasetForm({ onCreate, onClose }) {
  const [name, setName] = useState('');
  const [trigger, setTrigger] = useState('');
  // Nature du dataset : personnage (identité liée au trigger) vs concept (un acte/effet
  // récurrent lié au trigger — import brut, captions inversées, pas de référence/visage).
  const [kind, setKind] = useState('character');
  // Modèle cible choisi à la création : pilote le format de caption (SDXL→booru, sinon
  // prose) DÈS le départ et le regroupement du menu. Reste modifiable dans le panneau
  // d'entraînement. Défaut Z-Image (le type par défaut de l'app).
  const [trainType, setTrainType] = useState('zimage');
  // Concept only : ce que le captioneur doit OMETTRE de chaque caption pour que le concept
  // se lie au trigger (l'inverse d'un LoRA de personnage). Obligatoire pour un concept.
  const [conceptDesc, setConceptDesc] = useState('');
  // Character only : fidélité visage seul (défaut) ou visage + corps (les marques
  // corporelles sont bannies des captions et la composition cible plus de corps).
  const [fidelity, setFidelity] = useState('face');
  const concept = kind === 'concept';
  // Style : esthétique globale absorbée par le LoRA — captions de contenu pur
  // obligatoires, aucun trigger d'activation, pas de fidélité visage.
  const style = kind === 'style';
  // Mirrors the server rule EXACTLY (POST /api/dataset/create): name is always
  // required; trigger_word is required for character/concept (it's the token
  // that summons them) but NOT for style (the server may retain an internal id
  // for filenames/runs, but it never enters training captions or prompts).
  // Without this, an empty-trigger character/concept create used to reach the
  // server with the button enabled and 400 silently (no toast, no feedback).
  const canCreate = name.trim() && (!concept || conceptDesc.trim()) && (style || trigger.trim());
  return (
    <div id="new-dataset-form" className="mx-auto w-full max-w-4xl rounded-xl border border-border bg-surface p-3 flex flex-col gap-2.5">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-content font-semibold text-sm flex items-center gap-2">
          <span aria-hidden="true">🆕</span> New dataset
        </h2>
        {onClose && (
          <button type="button" onClick={onClose} aria-label="Close the new-dataset form"
            className="rounded px-1.5 text-content-subtle hover:text-content">✕</button>
        )}
      </div>
      {/* Nature : personnage (défaut) vs concept. Choisir « Concept » adapte tout le
          reste — import brut aspect conservé, captions qui gardent l'identité, pas de
          photo de référence ni de générateur de variations. */}
      <div className="flex gap-1.5">
        {[['character', '🧑 Character', 'A person/face — identity binds to the trigger'],
          ['concept', '💡 Concept', 'A recurring act/effect — the concept binds to the trigger'],
          ['style', '🎨 Style', 'An always-on aesthetic: load the LoRA and control its influence with the LoRA weight']].map(
          ([val, label, hint]) => (
            <button key={val} type="button" onClick={() => setKind(val)} title={hint}
              className={`flex-1 px-3 py-1.5 rounded-lg border text-xs font-semibold transition-colors ${
                kind === val
                  ? 'border-primary/60 bg-primary/15 text-content'
                  : 'border-border bg-app/40 text-content-muted hover:bg-surface-raised'}`}>
              {label}
            </button>
          ))}
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        <label className={`flex flex-col gap-1 text-[0.6875rem] text-content-muted ${style ? 'sm:col-span-2' : ''}`}>
          {concept ? 'Concept name' : style ? 'Style name' : 'Character name'}
          <input id="new-dataset-name" value={name} onChange={(e) => setName(e.target.value)}
            placeholder={concept ? 'e.g. cim' : style ? 'e.g. ink-wash' : 'e.g. Emma'}
            className="bg-app/60 border border-border rounded px-2 py-1.5 text-sm text-content" />
        </label>
        {!style && (
          <label className="flex flex-col gap-1 text-[0.6875rem] text-content-muted">
            Trigger word
            <input value={trigger} onChange={(e) => setTrigger(e.target.value)}
              placeholder={concept ? 'e.g. cim_act' : 'e.g. zchar_emma'}
              className="bg-app/60 border border-border rounded px-2 py-1.5 text-sm text-content" />
            {/* Guard-rail: a plain short word ("emma", "girl") collides with the base
                model's existing vocabulary — the identity bleeds into that word
                everywhere. A unique token (prefix/underscore/digits) binds cleanly. */}
            {trigger.trim() && /^[a-z]{1,7}$/i.test(trigger.trim()) && (
              <span className="text-amber-300 text-[0.625rem]">
                ⚠ “{trigger.trim()}” looks like a common word — the base model already has a meaning
                for it. Prefer a unique token like <span className="font-mono">zchar_{trigger.trim().toLowerCase()}</span>.
              </span>
            )}
          </label>
        )}
      </div>
      {/* Modèle cible : fixe le format de caption (SDXL→tags booru, sinon prose) et la
          section du menu. Modifiable ensuite dans le panneau d'entraînement. */}
      <label className="flex flex-col gap-1 text-[0.6875rem] text-content-muted">
        Target model <span className="text-content-subtle normal-case">— sets the caption style &amp; groups the menu (changeable later)</span>
        <select value={trainType} onChange={(e) => setTrainType(e.target.value)}
          className="bg-app/60 border border-border rounded px-2 py-1.5 text-sm text-content">
          <option value="zimage">Z-Image (prose captions)</option>
          <option value="sdxl">SDXL (booru-tag captions)</option>
          <option value="krea">Krea 2 (prose captions)</option>
          <option value="flux">FLUX.1 (prose captions)</option>
          <option value="flux2klein">FLUX.2 Klein (prose captions)</option>
          <option value="anima">Anima (prose captions)</option>
        </select>
      </label>
      {/* Fidélité (personnage) : visage seul (défaut) vs visage + corps. En mode corps,
          les marques corporelles permanentes sont bannies des captions (elles se lient
          au trigger) et la composition cible plus de bustes/corps. */}
      {!concept && !style && (
        <div className="flex flex-col gap-1 text-[0.6875rem] text-content-muted">
          <span>Fidelity <span className="text-content-subtle normal-case">— what the LoRA must reproduce (changeable later)</span></span>
          <div className="flex gap-1.5">
            {[['face', '🙂 Face', 'Identity = the face. Body shape may vary with the prompt.'],
              ['body', '🧍 Face + body', 'Total fidelity: body shape, tattoos and marks bind to the trigger too. Prefers full-frame imports and more bust/body shots.']].map(
              ([val, label, hint]) => (
                <button key={val} type="button" onClick={() => setFidelity(val)} title={hint}
                  className={`flex-1 px-3 py-1.5 rounded-lg border text-xs font-semibold transition-colors ${
                    fidelity === val
                      ? 'border-primary/60 bg-primary/15 text-content'
                      : 'border-border bg-app/40 text-content-muted hover:bg-surface-raised'}`}>
                  {label}
                </button>
              ))}
          </div>
        </div>
      )}
      {/* Concept description : ce que la caption OMET (l'acte récurrent). Alimente le
          {concept} des prompts caption/raffinage/ban-list. Décrire l'ACTE, pas le sujet. */}
      {concept && (
        <label className="flex flex-col gap-1 text-[0.6875rem] text-content-muted">
          What is the recurring concept? <span className="text-fuchsia-300">(required — it will be omitted from every caption)</span>
          <textarea value={conceptDesc} onChange={(e) => setConceptDesc(e.target.value)} rows={2}
            placeholder="Describe the recurring act/effect itself, not the people — e.g. “a tongue licking an ice-cream cone”"
            className="bg-app/60 border border-border rounded px-2 py-1.5 text-sm text-content resize-y" />
        </label>
      )}
      <div className="flex items-center gap-2 flex-wrap">
        <p className="text-content-subtle text-[0.6875rem]">
          {concept
            ? 'The trigger word is the token you type to summon this concept. Import raw images of it, then caption and train.'
            : style
              ? 'Always-on Style: import varied images, then caption every kept image with content only (subject, action, setting) while leaving the aesthetic unspoken. Combine it with a character LoRA by adjusting each LoRA weight.'
              : 'The trigger word is the unique token you will type in prompts to summon this character.'}
        </p>
        <button type="button"
          onClick={() => canCreate && onCreate(name.trim(), trigger.trim(), kind, conceptDesc.trim(), trainType,
            (concept || style) ? undefined : fidelity)}
          disabled={!canCreate}
          className="ml-auto px-4 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
          Create
        </button>
      </div>
    </div>
  );
}

export default function DatasetListPanel({
  datasets, onOpen, onCreate, onDelete, onRestore, onExportZip, onExportBackup, backup,
}) {
  // Library-first: the creation form stays folded behind "+ New dataset" so the
  // page opens on the collection — except on an empty library, where creating
  // is the only meaningful action.
  const [creating, setCreating] = useState(false);
  const [query, setQuery] = useState('');
  // Kind chip filter — transient (see KIND_CHIPS): reset on every page load.
  const [kindFilter, setKindFilter] = useState('all');
  // Tile size + collapsed sections: persisted display preferences (same lazy
  // init + save effect as datasetGridTileSize / the CloudRuns group folds).
  const [tileSize, setTileSize] = useState(() => {
    try { return normalizeTileSize(localStorage.getItem(TILE_SIZE_KEY)); } catch { return 'M'; }
  });
  useEffect(() => {
    try { localStorage.setItem(TILE_SIZE_KEY, tileSize); } catch { /* ignore — private mode */ }
  }, [tileSize]);
  const [collapsed, setCollapsed] = useState(() => {
    try { return normalizeCollapsedMap(localStorage.getItem(COLLAPSED_KEY)); } catch { return {}; }
  });
  useEffect(() => {
    try { localStorage.setItem(COLLAPSED_KEY, JSON.stringify(collapsed)); } catch { /* ignore — private mode */ }
  }, [collapsed]);
  const toggleSection = (family) => setCollapsed((m) => {
    const next = { ...m };
    if (next[family]) delete next[family];
    else next[family] = 1;
    return next;
  });
  const empty = datasets.length === 0;
  const formOpen = creating || empty;
  // One-time tip: once the library is sizeable, point out tile sizing / folding /
  // filtering. Gated at ≥6 so it never fires for a first-time, near-empty library.
  useEffect(() => { if (datasets.length >= 6) requestHelpTip('library-browse'); }, [datasets.length]);
  const filtered = datasets.filter((d) => datasetMatches(d, query, kindFilter));
  const groups = groupDatasets(filtered);
  const kinds = kindsPresent(datasets);
  // While a search/filter is active every section is forced open: a fold that
  // hides matches would read as lost datasets. Folding resumes when cleared.
  const filterActive = Boolean(query.trim()) || kindFilter !== 'all';
  return (
    <div className="flex flex-col gap-4">
      {/* Header: the page IS the library. Row 1 = title + primary actions;
          row 2 (below, non-empty library only) = search + filters + size. */}
      <div>
        <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-content-subtle">library</p>
        {/* relative z-30 : sans stacking-context propre, le z-20 du panneau du
            menu « 💾 Backup » resterait piégé sous les tuiles plus bas. */}
        <div className="relative z-30 mt-1 flex flex-wrap items-center gap-2">
          <h1 className="text-xl font-semibold text-content flex items-center gap-2">Datasets<HelpBadge topic="page-datasets" /></h1>
          {!empty && <span className="text-sm text-content-subtle">{datasets.length}</span>}
          <div className="ml-auto flex items-center gap-2">
            <button type="button"
              onClick={() => {
                if (empty) document.getElementById('new-dataset-name')?.focus();
                else setCreating((v) => !v);
              }}
              aria-expanded={empty ? undefined : formOpen}
              aria-controls={empty ? undefined : 'new-dataset-form'}
              className="rounded-lg bg-gradient-primary px-3.5 py-1.5 text-sm font-semibold text-white transition-transform hover:-translate-y-px">
              {!empty && creating ? '✕ Close' : '+ New dataset'}
            </button>
            {/* Back up everything, its "include LoRAs" option and Import backup
                all live in ONE 💾 Backup menu — only "+ New dataset" stays out,
                it is the page's primary action. */}
            <FullBackupControls backup={backup} onRestore={onRestore} />
          </div>
        </div>
        {!empty && (
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Find a dataset…"
              aria-label="Find a dataset"
              className="min-w-[9rem] flex-1 rounded-md border border-border bg-surface px-3 py-1.5 text-xs text-content placeholder:text-content-subtle focus:border-primary focus:outline-none sm:max-w-xs"
            />
            {kinds.length >= 2 && (
              <div role="group" aria-label="Filter by dataset kind" className="flex items-center gap-1">
                {['all', ...kinds].map((k) => (
                  <button key={k} type="button"
                    onClick={() => setKindFilter(k)}
                    aria-pressed={kindFilter === k}
                    className={`rounded-full border px-2.5 py-1 text-[0.6875rem] font-semibold transition-colors ${
                      kindFilter === k
                        ? 'border-primary/60 bg-primary/15 text-content'
                        : 'border-border bg-surface text-content-muted hover:bg-surface-raised'}`}>
                    {k === 'all' ? 'All' : KIND_CHIPS[k]}
                  </button>
                ))}
              </div>
            )}
            <TileSizeControl size={tileSize} onChange={setTileSize}
              titles={TILE_SIZE_TITLE} className="ml-auto" />
          </div>
        )}
      </div>

      {formOpen && (
        <NewDatasetForm onCreate={onCreate}
          onClose={empty ? null : () => setCreating(false)} />
      )}

      {empty ? (
        <EmptyState />
      ) : groups.length === 0 ? (
        <p className="rounded-xl border border-dashed border-border bg-app/30 px-4 py-8 text-center text-sm text-content-muted">
          {query.trim()
            ? <>No dataset matches “{query.trim()}”{kindFilter !== 'all' ? ` in ${KIND_CHIPS[kindFilter]}` : ''}.</>
            : <>No {KIND_CHIPS[kindFilter]} dataset.</>}
        </p>
      ) : (
        <>
          {groups.map(({ family, label, emoji, items }) => {
            const open = filterActive || !collapsed[family];
            return (
              <section key={family} className="flex flex-col gap-2">
                <h2>
                  <button type="button"
                    onClick={() => toggleSection(family)}
                    disabled={filterActive}
                    aria-expanded={open}
                    title={filterActive
                      ? 'Sections stay open while a search or filter is active'
                      : (open ? `Collapse the ${label} section` : `Expand the ${label} section`)}
                    className="flex w-full items-center gap-2 font-mono text-[11px] font-semibold uppercase tracking-[0.18em] text-content-subtle transition-colors hover:text-content disabled:cursor-default disabled:hover:text-content-subtle">
                    <span aria-hidden="true"
                      className={`text-[0.625rem] transition-transform ${open ? 'rotate-90' : ''} ${filterActive ? 'opacity-40' : ''}`}>
                      ▶
                    </span>
                    <span aria-hidden="true">{emoji}</span> {label}
                    <span className="font-normal normal-case tracking-normal">({items.length})</span>
                  </button>
                </h2>
                {open && (
                  tileSize === 'S' ? (
                    <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
                      {items.map((d) => (
                        <DatasetRow key={d.id} d={d} onOpen={onOpen} onDelete={onDelete}
                          onExportZip={onExportZip} onExportBackup={onExportBackup} />
                      ))}
                    </div>
                  ) : (
                    <div className={GRID_COLS[tileSize]}>
                      {items.map((d) => (
                        <DatasetTile key={d.id} d={d} onOpen={onOpen} onDelete={onDelete}
                          onExportZip={onExportZip} onExportBackup={onExportBackup} />
                      ))}
                    </div>
                  )
                )}
              </section>
            );
          })}
        </>
      )}
    </div>
  );
}
