import { useState } from 'react';
import { apiFetch } from '../../../api/fetchClient';
import { useToast } from '../../common/Toast';
import { HelpBadge } from '../../../help/HelpMode';
import { toggleSceneIndex } from './scenePrompts';

/* 🎬 Scenes from a bank — a bank's captions imported as ordered prompt passes.
 *
 * Load a bank's captions IN ORDER (one per captioned image), tick the ones to
 * run: each ticked scene is one pass of the 📝 prompt axis, with the run's own
 * checkpoints and settings unchanged. State lives in RunSetupPanel next to the
 * history batch, and like it is deliberately NOT persisted — a scene selection
 * is the intention of ONE launch.
 *
 * The 🎲 shortcut above stays the random single draw; this list is for when the
 * order IS the point (a storyboard, a shoot, a chapter).
 */
export default function SceneBankPrompts({ value, onChange }) {
  const toast = useToast();
  const [banks, setBanks] = useState(null);   // null = never fetched
  const [bankId, setBankId] = useState('');
  const [busy, setBusy] = useState(false);
  const { source, scenes, picked } = value;

  const openBanks = () => {
    if (banks !== null) return;
    apiFetch('/api/banks')
      .then((d) => setBanks(d.banks || []))
      .catch(() => { setBanks([]); toast.error('Could not list the image banks'); });
  };

  const load = async () => {
    if (!bankId) return;
    setBusy(true);
    try {
      const d = await apiFetch(`/api/bank/${bankId}/scenes`);
      onChange({ source: { bank_id: d.bank_id, bank_name: d.bank_name },
        scenes: d.scenes || [], picked: [] });
      const skipped = d.skipped?.no_caption || 0;
      toast.success(`${(d.scenes || []).length} scene(s) loaded in order`
        + (skipped ? ` — ${skipped} image(s) without a caption skipped` : ''));
    } catch (e) {
      toast.error(e.message || 'Could not load the bank’s scenes');
    } finally { setBusy(false); }
  };

  const nPicked = picked.length;
  return (
    <details className="rounded-lg border border-border bg-app/30 open:pb-2" onToggle={(e) => { if (e.currentTarget.open) openBanks(); }}>
      <summary className="cursor-pointer select-none px-2.5 py-1.5 text-[0.75rem] text-content font-semibold">
        🎬 Scenes from a bank
        <HelpBadge topic="studio-scene-prompts" />
        <span className="ml-2 font-normal text-content-subtle text-[0.625rem]">
          {scenes.length
            ? `${nPicked} of ${scenes.length} scene(s) picked from “${source?.bank_name || 'a bank'}” — one pass each, in order`
            : 'run a bank’s captions in order — one pass per ticked scene'}
        </span>
      </summary>
      <div className="px-2.5 pt-1 flex flex-col gap-1.5">
        <div className="flex flex-wrap items-center gap-1.5">
          <select value={bankId} onChange={(e) => setBankId(e.target.value)}
            aria-label="Bank to load scenes from"
            className="max-w-56 rounded border border-border bg-app/60 px-1 py-1 text-[0.6875rem] text-content">
            <option value="">{banks === null ? 'Loading banks…' : 'Choose a bank…'}</option>
            {(banks || []).map((b) => (
              <option key={b.id} value={b.id}>{b.name}</option>
            ))}
          </select>
          <button type="button" onClick={load} disabled={!bankId || busy}
            className="rounded-lg bg-gradient-primary px-2.5 py-1 text-[0.6875rem] font-semibold text-white disabled:opacity-40">
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
              return (
                <button key={`${i}-${s.label}`} type="button"
                  onClick={() => onChange({ ...value, picked: toggleSceneIndex(picked, i) })}
                  aria-pressed={on} title={s.prompt}
                  className={'flex items-start gap-1.5 rounded-lg border px-1.5 py-1 text-left text-[0.625rem] transition-colors '
                    + (on ? 'border-primary/50 bg-primary/20 text-white ring-1 ring-primary/30'
                      : 'border-border bg-app/40 text-content-muted hover:bg-surface-raised')}>
                  {s.image_id != null && source?.bank_id != null && (
                    <img src={`/api/bank/${source.bank_id}/thumb/${s.image_id}`}
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
