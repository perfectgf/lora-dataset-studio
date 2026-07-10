import { useMemo, useState } from 'react';

/* Bulk caption tools (collapsible): find/replace across the kept images'
   captions + a tag-frequency panel. The frequency list is derived client-side
   from the payload the workspace already polls (captions split on commas) —
   it's mainly useful for booru tags (SDXL), where one stray recurring tag
   pollutes the whole training. Clicking a tag pre-fills "find" in tag mode;
   leaving "replace" empty removes the tag cleanly (no dangling commas). */
export default function CaptionToolsBar({ images, trainType, onReplace, busy }) {
  const [open, setOpen] = useState(false);
  const [find, setFind] = useState('');
  const [replace, setReplace] = useState('');
  const [tagMode, setTagMode] = useState(trainType === 'sdxl');
  const captioned = useMemo(
    () => images.filter((i) => i.status === 'keep' && (i.caption || '').trim()),
    [images]);
  const freq = useMemo(() => {
    const counts = new Map();
    for (const img of captioned) {
      for (const raw of (img.caption || '').split(',')) {
        const t = raw.trim().toLowerCase();
        if (t) counts.set(t, (counts.get(t) || 0) + 1);
      }
    }
    return [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 30);
  }, [captioned]);
  if (!captioned.length) return null;

  const apply = async () => {
    if (!find.trim()) return;
    await onReplace(find, replace, tagMode ? 'tag' : 'text');
  };

  return (
    <div className="rounded-lg border border-border bg-surface px-3 py-2">
      <button type="button" onClick={() => setOpen((v) => !v)} aria-expanded={open}
        className="flex items-center gap-2 w-full text-left text-content text-sm font-semibold">
        <span aria-hidden>📝</span> Caption tools
        <span className="text-content-subtle text-[0.6875rem] font-normal">
          find/replace · tag frequency ({captioned.length} captioned)
        </span>
        <span aria-hidden className="ml-auto text-content-subtle">{open ? '▾' : '▸'}</span>
      </button>
      {open && (
        <div className="mt-2 flex flex-col gap-2">
          <div className="flex items-center gap-2 flex-wrap">
            <input value={find} onChange={(e) => setFind(e.target.value)}
              placeholder={tagMode ? 'tag to replace/remove' : 'text to find'}
              aria-label="Find in captions"
              className="px-2 py-1 rounded bg-app/60 border border-border text-content text-xs w-44" />
            <span aria-hidden className="text-content-subtle text-xs">→</span>
            <input value={replace} onChange={(e) => setReplace(e.target.value)}
              placeholder="replacement (empty = remove)"
              aria-label="Replace with"
              className="px-2 py-1 rounded bg-app/60 border border-border text-content text-xs w-48" />
            <label className="flex items-center gap-1 text-xs text-content-muted"
              title="Tag mode treats captions as comma-separated tags: the whole tag must match (case-insensitive), and removal keeps the commas clean. Recommended for booru (SDXL).">
              <input type="checkbox" checked={tagMode} onChange={(e) => setTagMode(e.target.checked)}
                className="accent-indigo-500" />
              tag mode
            </label>
            <button type="button" onClick={apply} disabled={busy || !find.trim()}
              className="px-3 py-1 rounded-lg bg-surface-raised border border-border text-content text-xs font-semibold disabled:opacity-40 hover:bg-surface">
              Apply to {captioned.length} caption(s)
            </button>
          </div>
          {freq.length > 0 && (
            <div className="flex flex-wrap gap-1" aria-label="Most frequent caption tags">
              {freq.map(([tag, n]) => (
                <button key={tag} type="button"
                  onClick={() => { setFind(tag); setTagMode(true); }}
                  title={`"${tag}" appears in ${n} caption(s) — click to fill Find (tag mode)`}
                  className="px-1.5 py-0.5 rounded bg-app/60 border border-border text-[0.6875rem] text-content-muted hover:text-content hover:bg-surface-raised">
                  {tag} <span className="text-content-subtle">×{n}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
