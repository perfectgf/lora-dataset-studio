import { useState } from 'react';
import { apiFetch } from '../../../api/fetchClient';
import { useToast } from '../../common/Toast';
import { HelpBadge } from '../../../help/HelpMode';
import { SCENE_SOURCES, sceneSource, sceneThumbUrl, toggleSceneIndex } from './scenePrompts';

/* 🎬 Scenes — a bank's OR a dataset's captions imported as ordered prompt passes.
 *
 * Load the captions IN ORDER (one per captioned image), tick the ones to run:
 * each ticked scene is one pass of the 📝 prompt axis, with the run's own
 * checkpoints and settings unchanged. State lives in RunSetupPanel next to the
 * history batch, and like it is deliberately NOT persisted — a scene selection
 * is the intention of ONE launch.
 *
 * TWO sources, one panel. A bank is the pile you triage; a dataset is what you
 * KEPT and captioned — the sequence people actually want to replay is as often
 * the second as the first, and offering only the bank meant exporting a dataset
 * back into a bank to reach a feature that was already there. Everything that
 * differs between them (the list route, the scenes route, how a thumbnail is
 * addressed) lives in scenePrompts.js; this file has one code path.
 *
 * The 🎲 shortcut above stays the random single draw; this list is for when the
 * order IS the point (a storyboard, a shoot, a chapter).
 */
export default function ScenePromptsPanel({ value, onChange }) {
  const toast = useToast();
  const [kind, setKind] = useState('bank');
  // { bank: [...], dataset: [...] } — fetched once per source, on first open of
  // that tab. A key absent means "never asked", which is what shows "Loading…".
  const [lists, setLists] = useState({});
  const [sourceId, setSourceId] = useState('');
  const [busy, setBusy] = useState(false);
  const { source, scenes, picked } = value;
  const src = SCENE_SOURCES.find((s) => s.kind === kind) || SCENE_SOURCES[0];
  const options = lists[kind];

  const openList = (k) => {
    const conf = SCENE_SOURCES.find((s) => s.kind === k);
    if (!conf || lists[k] !== undefined) return;
    apiFetch(conf.listUrl)
      .then((d) => setLists((cur) => ({ ...cur, [k]: d[conf.listKey] || [] })))
      .catch(() => {
        setLists((cur) => ({ ...cur, [k]: [] }));
        toast.error(`Could not list the ${k === 'dataset' ? 'datasets' : 'image banks'}`);
      });
  };

  const pickKind = (k) => {
    setKind(k);
    setSourceId('');          // an id from the other table would load the wrong thing
    openList(k);
  };

  const load = async () => {
    if (!sourceId) return;
    setBusy(true);
    try {
      const path = kind === 'dataset'
        ? `/api/dataset/${sourceId}/scenes`
        : `/api/bank/${sourceId}/scenes`;
      const d = await apiFetch(path);
      onChange({ source: sceneSource(kind, d), scenes: d.scenes || [], picked: [] });
      const skipped = d.skipped?.no_caption || 0;
      toast.success(`${(d.scenes || []).length} scene(s) loaded in order`
        + (skipped ? ` — ${skipped} image(s) without a caption skipped` : ''));
    } catch (e) {
      toast.error(e.message || 'Could not load the scenes');
    } finally { setBusy(false); }
  };

  const nPicked = picked.length;
  return (
    <details className="rounded-lg border border-border bg-app/30 open:pb-2" onToggle={(e) => { if (e.currentTarget.open) openList(kind); }}>
      <summary className="cursor-pointer select-none px-2.5 py-1.5 text-[0.75rem] text-content font-semibold">
        🎬 Scenes from a bank or dataset
        <HelpBadge topic="studio-scene-prompts" />
        <span className="ml-2 font-normal text-content-subtle text-[0.625rem]">
          {scenes.length
            ? `${nPicked} of ${scenes.length} scene(s) picked from “${source?.name || 'a source'}” — one pass each, in order`
            : 'run a bank’s or a dataset’s captions in order — one pass per ticked scene'}
        </span>
      </summary>
      <div className="px-2.5 pt-1 flex flex-col gap-1.5">
        {/* Which table the dropdown below is listing. Two buttons rather than a
            second dropdown: the choice changes what the next control MEANS. */}
        <div className="flex flex-wrap items-center gap-1" role="group" aria-label="Where the scenes come from">
          {SCENE_SOURCES.map((s) => (
            <button key={s.kind} type="button" onClick={() => pickKind(s.kind)}
              aria-pressed={kind === s.kind}
              className={'rounded-lg border px-2 py-0.5 text-[0.625rem] font-semibold transition-colors '
                + (kind === s.kind
                  ? 'border-primary/50 bg-primary/20 text-white'
                  : 'border-border bg-app/40 text-content-muted hover:bg-surface-raised')}>
              {s.label}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <select value={sourceId} onChange={(e) => setSourceId(e.target.value)}
            aria-label={`${src.label.replace(/^\S+\s/, '')} to load scenes from`}
            className="max-w-56 rounded border border-border bg-app/60 px-1 py-1 text-[0.6875rem] text-content">
            <option value="">
              {options === undefined ? 'Loading…' : (options.length ? src.pick : src.empty)}
            </option>
            {(options || []).map((o) => (
              <option key={o.id} value={o.id}>{o.name}</option>
            ))}
          </select>
          <button type="button" onClick={load} disabled={!sourceId || busy}
            className="rounded-lg bg-gradient-primary px-2.5 py-1 text-[0.6875rem] font-semibold text-gray-950 disabled:opacity-40">
            {busy ? 'Loading…' : scenes.length ? '⟳ Reload' : '⬇ Load scenes'}
          </button>
          {scenes.length > 0 && (
            <>
              <button type="button"
                onClick={() => onChange({ ...value, picked: scenes.map((_, i) => i) })}
                className="rounded border border-border px-1.5 py-0.5 text-[0.625rem] text-content-muted hover:bg-surface-raised">
                Select all
              </button>
              <button type="button" onClick={() => onChange({ ...value, picked: [] })}
                className="rounded border border-border px-1.5 py-0.5 text-[0.625rem] text-content-muted hover:bg-surface-raised">
                None
              </button>
            </>
          )}
        </div>
        {scenes.length > 0 && (
          <div className="flex max-h-64 flex-col gap-1 overflow-y-auto pr-1">
            {scenes.map((s, i) => {
              const on = picked.includes(i);
              const thumb = sceneThumbUrl(source, s);
              return (
                <button key={`${i}-${s.label}`} type="button"
                  onClick={() => onChange({ ...value, picked: toggleSceneIndex(picked, i) })}
                  aria-pressed={on} title={s.prompt}
                  className={'flex items-start gap-1.5 rounded-lg border px-1.5 py-1 text-left text-[0.625rem] transition-colors '
                    + (on ? 'border-primary/50 bg-primary/20 text-white ring-1 ring-primary/30'
                      : 'border-border bg-app/40 text-content-muted hover:bg-surface-raised')}>
                  {thumb && (
                    <img src={thumb}
                      alt="" loading="lazy" draggable={false}
                      onError={(e) => { e.currentTarget.style.display = 'none'; }}
                      className="h-16 w-12 shrink-0 rounded border border-border bg-app/60 object-cover object-top" />
                  )}
                  <span className="shrink-0 font-semibold tabular-nums text-content-subtle">{i + 1}.</span>
                  <span className="min-w-0 leading-tight line-clamp-3">{s.prompt}</span>
                  {on && <span className="ml-auto shrink-0 text-indigo-300" aria-hidden="true">✓</span>}
                </button>
              );
            })}
          </div>
        )}
      </div>
    </details>
  );
}
