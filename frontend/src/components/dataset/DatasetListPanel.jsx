import { useRef, useState } from 'react';
import ShotIllustration from './ShotIllustration';

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

/** The 3-step pipeline strip of the hero — what this page is for, at a glance. */
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

/** Empty state — a mini contact sheet of shot pictograms instead of a bare line. */
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
    <div className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-border bg-app/30 px-4 py-8 text-center">
      <div className="grid grid-cols-6 gap-1.5" aria-hidden="true">
        {shots.map((s, i) => (
          <ShotIllustration key={i} framing={s.framing} label={s.label}
            className={`w-9 h-9 ${i === 0 ? 'text-indigo-300' : 'text-content-subtle'}`} />
        ))}
      </div>
      <p className="text-content-muted text-sm font-medium">No datasets yet</p>
      <p className="text-content-subtle text-xs max-w-xs">
        Create your first character above — one reference photo is enough to start
        generating a full training set.
      </p>
    </div>
  );
}

// Badges de famille des LoRA entraînés — mêmes couleurs que le LoraPicker du Studio.
const FAMILY_BADGE = {
  zimage: ['Z-Image', 'border-sky-400/40 bg-sky-500/10 text-sky-300'],
  sdxl: ['SDXL', 'border-violet-400/40 bg-violet-500/10 text-violet-300'],
  krea: ['Krea', 'border-amber-400/40 bg-amber-500/10 text-amber-300'],
};

// Familles de modèle proposées à la création + ordre/labels des sections du menu.
// Le menu est GROUPÉ par cette famille (d.train_type) : gestion plus simple quand on
// entretient des datasets de plusieurs pipelines. La 3e valeur = l'emoji de section.
const FAMILY_ORDER = [
  ['zimage', 'Z-Image', '🌀'],
  ['sdxl', 'SDXL', '🎨'],
  ['krea', 'Krea 2', '✨'],
];

function DatasetCard({ d, onOpen, onDelete }) {
  return (
    <div
      className="group flex items-center gap-3 rounded-xl border border-border bg-surface px-3 py-2.5 hover:bg-surface-raised hover:border-primary/40 transition-colors">
      <button type="button" onClick={() => onOpen(d.id)}
        className="flex-1 flex items-center gap-3 text-left min-w-0">
        {/* Avatar = la photo de RÉFÉRENCE du dataset quand elle existe ; sinon
            repli sur l'initiale en dégradé (dataset encore sans référence). */}
        {d.ref_filename ? (
          <img
            src={`/api/dataset/${d.id}/img/${encodeURIComponent(d.ref_filename)}`}
            alt="" loading="lazy" aria-hidden="true"
            className="w-10 h-10 shrink-0 rounded-full object-cover border border-border shadow" />
        ) : (
          <span className={`grid place-items-center w-10 h-10 shrink-0 rounded-full bg-gradient-to-br ${gradientFor(d.name)} text-white font-bold text-base shadow`}
            aria-hidden="true">
            {(d.name || '?').charAt(0).toUpperCase()}
          </span>
        )}
        <span className="min-w-0">
          <span className="flex items-center gap-1.5 min-w-0">
            <span className="text-content text-sm font-semibold truncate">{d.name}</span>
            {d.kind === 'concept' && (
              <span className="shrink-0 px-1.5 py-px rounded border border-fuchsia-400/40 bg-fuchsia-500/10 text-fuchsia-300 text-[0.5625rem] font-semibold uppercase">
                💡 Concept
              </span>
            )}
            {(d.trained_families || []).map((f) => {
              const [lbl, cls] = FAMILY_BADGE[f] || [f, 'border-border bg-white/5 text-content-muted'];
              return (
                <span key={f} className={`shrink-0 px-1.5 py-px rounded border text-[0.5625rem] font-semibold uppercase ${cls}`}>
                  {lbl}
                </span>
              );
            })}
          </span>
          <span className="block text-content-subtle text-[0.6875rem] truncate">
            trigger: <code className="text-indigo-300">{d.trigger_word || '—'}</code>
          </span>
        </span>
        <span className="ml-auto shrink-0 text-content-subtle opacity-0 group-hover:opacity-100 transition-opacity"
          aria-hidden="true">Open →</span>
      </button>
      {onDelete && (
        <button type="button"
          onClick={() => {
            if (window.confirm(`Permanently delete the dataset "${d.name}" and all its images? This cannot be undone.`)) onDelete(d.id);
          }}
          title="Delete this dataset" aria-label={`Delete the dataset ${d.name}`}
          className="px-2 py-1 rounded-lg bg-red-500/15 border border-red-500/40 text-red-300 text-xs hover:bg-red-500/25">
          🗑
        </button>
      )}
    </div>
  );
}

export default function DatasetListPanel({ datasets, onOpen, onCreate, onDelete, onRestore }) {
  const restoreRef = useRef(null);
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
  const concept = kind === 'concept';
  const canCreate = name.trim() && (!concept || conceptDesc.trim());
  return (
    <div className="flex flex-col gap-4">
      {/* What this page does — pipeline hero. */}
      <div className="rounded-xl border border-border bg-gradient-to-br from-surface to-app/60 p-3 flex flex-col gap-2.5">
        <p className="text-content-subtle text-xs">
          Build a consistent character: one reference photo becomes a curated,
          captioned training set for a LoRA you can use in every generator.
        </p>
        <PipelineSteps />
      </div>

      {/* Creation card. */}
      <div className="rounded-xl border border-border bg-surface p-3 flex flex-col gap-2.5">
        <h2 className="text-content font-semibold text-sm flex items-center gap-2">
          <span aria-hidden="true">🆕</span> New dataset
        </h2>
        {/* Nature : personnage (défaut) vs concept. Choisir « Concept » adapte tout le
            reste — import brut aspect conservé, captions qui gardent l'identité, pas de
            photo de référence ni de générateur de variations. */}
        <div className="flex gap-1.5">
          {[['character', '🧑 Character', 'A person/face — identity binds to the trigger'],
            ['concept', '💡 Concept', 'A recurring act/effect — the concept binds to the trigger']].map(
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
          <label className="flex flex-col gap-1 text-[0.6875rem] text-content-muted">
            {concept ? 'Concept name' : 'Character name'}
            <input value={name} onChange={(e) => setName(e.target.value)}
              placeholder={concept ? 'e.g. cim' : 'e.g. Emma'}
              className="bg-app/60 border border-border rounded px-2 py-1.5 text-sm text-content" />
          </label>
          <label className="flex flex-col gap-1 text-[0.6875rem] text-content-muted">
            Trigger word
            <input value={trigger} onChange={(e) => setTrigger(e.target.value)}
              placeholder={concept ? 'e.g. cim_act' : 'e.g. zchar_emma'}
              className="bg-app/60 border border-border rounded px-2 py-1.5 text-sm text-content" />
          </label>
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
          </select>
        </label>
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
              : 'The trigger word is the unique token you will type in prompts to summon this character.'}
          </p>
          <button type="button"
            onClick={() => canCreate && onCreate(name.trim(), trigger.trim(), kind, conceptDesc.trim(), trainType)}
            disabled={!canCreate}
            className="ml-auto px-4 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
            Create
          </button>
          {onRestore && (
            <>
              <button type="button" onClick={() => restoreRef.current?.click()}
                title="Restore a dataset from a backup zip (made with the 💾 Backup button) — creates a new dataset."
                className="px-3 py-1.5 rounded-lg bg-surface border border-border text-content text-sm">
                📦 Restore backup
              </button>
              <input ref={restoreRef} type="file" accept=".zip,application/zip" className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) onRestore(f);
                  e.target.value = '';
                }} />
            </>
          )}
        </div>
      </div>

      {/* Existing datasets — GROUPED by target model family (Z-Image / SDXL / Krea) so a
          multi-pipeline library stays easy to manage. Character vs concept stays a badge. */}
      {datasets.length > 0 ? (
        <>
          {FAMILY_ORDER.map(([fam, label, emoji]) => {
            const group = datasets.filter((d) => (d.train_type || 'zimage') === fam);
            if (!group.length) return null;
            return (
              <div key={fam} className="flex flex-col gap-2">
                <h2 className="text-content-muted text-[0.6875rem] uppercase tracking-wide font-semibold flex items-center gap-2">
                  <span aria-hidden="true">{emoji}</span> {label}
                  <span className="text-content-subtle font-normal normal-case">({group.length})</span>
                </h2>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {group.map((d) => (
                    <DatasetCard key={d.id} d={d} onOpen={onOpen} onDelete={onDelete} />
                  ))}
                </div>
              </div>
            );
          })}
        </>
      ) : (
        <EmptyState />
      )}
    </div>
  );
}
