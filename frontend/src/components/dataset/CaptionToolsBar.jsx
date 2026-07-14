import { useMemo, useState } from 'react';

/* Bulk caption tools (collapsible): find/replace across the kept images'
   captions + a tag-frequency panel. The frequency list is derived client-side
   from the payload the workspace already polls (captions split on commas) —
   it's mainly useful for booru tags (SDXL), where one stray recurring tag
   pollutes the whole training. Clicking a tag pre-fills "find" in tag mode;
   leaving "replace" empty removes the tag cleanly (no dangling commas).
   Also hosts 💾 Write .txt files (kohya-style sidecar captions written next to
   the images on disk, same text as the export ZIP) + 📂 open that folder. */
export default function CaptionToolsBar({ images, trainType, mode = 'booru',
                                          excludes = [], includes = [], onExclude, onInclude,
                                          onReplace, onWriteFiles, onOpenFolder, busy }) {
  const [open, setOpen] = useState(false);
  const [find, setFind] = useState('');
  const [replace, setReplace] = useState('');
  const [filterInput, setFilterInput] = useState('');
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
  // Grid tag-filter: submit the free-text field to exclude/include, then clear it.
  const submitFilter = (fn) => {
    const t = filterInput.trim();
    if (!t) return;
    fn?.(t);
    setFilterInput('');
  };
  // Plain-language description of the match rule for THIS dataset's caption style —
  // documented in-UI so "exclude smile" behaves the way the user expects.
  const matchHelp = mode === 'prose'
    ? 'Captions here are prose, so a filter matches a whole word (case-insensitive) — “smile” matches “a warm smile” but not “smiling”.'
    : 'Captions here are comma-separated tags, so a filter matches one whole tag exactly (case-insensitive).';

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
          {/* Plain-language primer: what captions are and what these tools do —
              a newcomer shouldn't need to guess why find/replace or tag frequency
              matter for training. */}
          <p className="m-0 text-content-subtle text-[0.6875rem] leading-relaxed">
            Captions are the text the LoRA reads each image by. These tools edit{' '}
            <span className="text-content-muted font-medium">every kept caption at once</span> — use them to
            fix a word that slipped into all of them, or to strip/rename a tag that keeps repeating.
            {' '}<span className="text-content-muted font-medium">Text</span> mode swaps a phrase anywhere;
            {' '}<span className="text-content-muted font-medium">tag</span> mode treats captions as
            comma-separated tags and matches a whole tag (best for booru / SDXL).
          </p>
          <span className="text-content-subtle text-[0.625rem] uppercase tracking-wide">Find &amp; replace</span>
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
          {/* Grid tag-filter: the inverse of "show images with this tag" — hide the
              images that ALREADY carry a tag so a captioning checklist only shows
              what's left to do (community's #1 request). Multi-exclusions cumulate;
              active filters show as loud chips above the grid. Session-only (they
              reset on reload / dataset switch — a transient view, not dataset state). */}
          <span className="text-content-subtle text-[0.625rem] uppercase tracking-wide">Filter the grid by tag</span>
          <p className="m-0 text-content-subtle text-[0.6875rem] leading-relaxed">
            <span className="text-content-muted font-medium">Exclude</span> hides images already tagged with a
            word — walk a captioning checklist without re-checking what's done.{' '}
            <span className="text-content-muted font-medium">Only&nbsp;with</span> does the inverse (isolate the
            ones that have it). {matchHelp}
          </p>
          <div className="flex items-center gap-2 flex-wrap">
            <input value={filterInput} onChange={(e) => setFilterInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); submitFilter(onExclude); } }}
              placeholder="tag to filter by…" aria-label="Tag to filter the grid by"
              className="px-2 py-1 rounded bg-app/60 border border-border text-content text-xs w-44" />
            <button type="button" onClick={() => submitFilter(onExclude)} disabled={!onExclude || !filterInput.trim()}
              title="Hide every image that already carries this tag from the grid"
              className="px-3 py-1 rounded-lg bg-surface-raised border border-border text-content text-xs font-semibold disabled:opacity-40 hover:bg-rose-500/15 hover:border-rose-400/50">
              ⊘ Exclude
            </button>
            {onInclude && (
              <button type="button" onClick={() => submitFilter(onInclude)} disabled={!filterInput.trim()}
                title="Show ONLY images that carry this tag (hide the rest)"
                className="px-3 py-1 rounded-lg bg-surface-raised border border-border text-content text-xs font-semibold disabled:opacity-40 hover:bg-indigo-500/15 hover:border-indigo-400/50">
                ◉ Only with
              </button>
            )}
          </div>
          {freq.length > 0 && (
            <div className="flex flex-col gap-1">
              <span className="text-content-subtle text-[0.625rem] uppercase tracking-wide">Most frequent tags</span>
              <p className="m-0 text-content-subtle text-[0.6875rem] leading-relaxed">
                A word in many captions gets strongly tied to your trigger. If it's something you'd rather
                keep <span className="text-content-muted font-medium">prompt-controllable</span> (an outfit,
                a pose, a setting), remove it here so it doesn't bake into the character. Click a tag to load
                it into Find (tag mode); leave Replace empty to strip it from every caption. The{' '}
                <span className="text-rose-300 font-medium">⊘</span> hides every image already tagged with it.
              </p>
              <div className="flex flex-wrap gap-1" aria-label="Most frequent caption tags">
                {freq.map(([tag, n]) => {
                  const isExcluded = excludes.includes(tag);
                  const isIncluded = includes.includes(tag);
                  return (
                    <span key={tag}
                      className={`inline-flex items-stretch rounded border overflow-hidden ${
                        isExcluded ? 'border-rose-400/60' : isIncluded ? 'border-indigo-400/60' : 'border-border'}`}>
                      <button type="button"
                        onClick={() => { setFind(tag); setTagMode(true); }}
                        title={`"${tag}" appears in ${n} caption(s) — click to fill Find (tag mode)`}
                        className="px-1.5 py-0.5 bg-app/60 text-[0.6875rem] text-content-muted hover:text-content hover:bg-surface-raised">
                        {tag} <span className="text-content-subtle">×{n}</span>
                      </button>
                      {onExclude && (
                        <button type="button"
                          onClick={() => onExclude(tag)}
                          aria-pressed={isExcluded}
                          aria-label={isExcluded ? `Stop hiding images tagged ${tag}` : `Hide images tagged ${tag}`}
                          title={isExcluded
                            ? `Hiding images tagged "${tag}" — click to show them again`
                            : `Hide images already tagged "${tag}" from the grid`}
                          className={`px-1.5 border-l text-[0.6875rem] ${
                            isExcluded ? 'bg-rose-500/25 border-rose-400/50 text-rose-200'
                              : 'bg-app/40 border-border text-content-subtle hover:text-rose-200 hover:bg-rose-500/15'}`}>
                          ⊘
                        </button>
                      )}
                    </span>
                  );
                })}
              </div>
            </div>
          )}
          {/* Sidecar caption files: some people train with external tools that
              read <image>.txt next to each image (kohya / ai-toolkit convention)
              straight from the dataset folder — no ZIP download needed. */}
          {onWriteFiles && (
            <div className="flex flex-col gap-1">
              <span className="text-content-subtle text-[0.625rem] uppercase tracking-wide">Caption files on disk</span>
              <div className="flex items-center gap-2 flex-wrap">
                <button type="button" onClick={onWriteFiles} disabled={busy}
                  title="Writes <image>.txt next to each kept image in the dataset folder — same format as the ZIP export, for external tools"
                  className="px-3 py-1 rounded-lg bg-surface-raised border border-border text-content text-xs font-semibold disabled:opacity-40 hover:bg-surface">
                  💾 Write .txt files
                </button>
                {onOpenFolder && (
                  <button type="button" onClick={onOpenFolder}
                    title="Open the dataset folder in the file explorer"
                    aria-label="Open the dataset folder"
                    className="px-2 py-1 rounded-lg bg-surface-raised border border-border text-content text-xs hover:bg-surface">
                    📂
                  </button>
                )}
                <span className="text-content-subtle text-[0.6875rem]">
                  kohya-style sidecar captions, trigger included — overwrites existing .txt
                </span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
