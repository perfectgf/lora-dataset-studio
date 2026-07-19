import { useMemo, useState } from 'react';
import { useI18n } from '../../i18n/I18nContext';
import { captionFrequencyEntries } from './captionCategory';

/* Bulk caption tools (collapsible): find/replace across the kept images'
   captions + a category-aware frequency panel. Booru counts exact comma tags;
   prose counts useful whole words by caption. Guidance changes for character,
   concept and style sets. Clicking an entry pre-fills the matching edit mode;
   leaving "replace" empty removes it from every caption.
   Also hosts 💾 Write .txt files (kohya-style sidecar captions written next to
   the images on disk, same text as the export ZIP) + 📂 open that folder. */
export default function CaptionToolsBar({ images, kind = 'character', mode = 'booru',
                                          excludes = [], includes = [], onExclude, onInclude,
                                          onReplace, onWriteFiles, onOpenFolder, busy,
                                          open: controlledOpen, onOpenChange }) {
  const { t } = useI18n();
  const [internalOpen, setInternalOpen] = useState(false);
  const open = controlledOpen === undefined ? internalOpen : controlledOpen;
  const setOpen = (next) => {
    const value = typeof next === 'function' ? next(open) : next;
    if (controlledOpen === undefined) setInternalOpen(value);
    onOpenChange?.(value);
  };
  const [find, setFind] = useState('');
  const [replace, setReplace] = useState('');
  const [filterInput, setFilterInput] = useState('');
  const [tagMode, setTagMode] = useState(mode === 'booru');
  const captioned = useMemo(
    () => images.filter((i) => i.status === 'keep' && (i.caption || '').trim()),
    [images]);
  const freq = useMemo(
    () => captionFrequencyEntries(captioned.map((img) => img.caption), mode),
    [captioned, mode]);
  const category = ['character', 'concept', 'style'].includes(kind) ? kind : 'character';
  const tagModeCopy = mode === 'booru';
  const categoryCopy = {
    frequencyHelp: t(`workspace.captionTools.categories.${category}`),
    frequencyTitle: t(tagModeCopy
      ? 'workspace.captionTools.mostFrequentTags'
      : 'workspace.captionTools.mostFrequentWords'),
    frequencyItem: t(tagModeCopy
      ? 'workspace.captionTools.tag'
      : 'workspace.captionTools.word'),
    filterPlaceholder: t(tagModeCopy
      ? 'workspace.captionTools.tagFilterPlaceholder'
      : 'workspace.captionTools.wordFilterPlaceholder'),
  };
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
  const matchHelp = t(mode === 'prose'
    ? 'workspace.captionTools.proseMatchHelp'
    : 'workspace.captionTools.tagMatchHelp');

  return (
    <div className="rounded-lg border border-border bg-surface px-3 py-2">
      <button type="button" data-workspace-focus
        onClick={() => setOpen((v) => !v)} aria-expanded={open}
        className="flex items-center gap-2 w-full text-left text-content text-sm font-semibold">
        <span aria-hidden>📝</span> {t('workspace.captionTools.title')}
        <span className="text-content-subtle text-[0.6875rem] font-normal">
          {t('workspace.captionTools.summary', {
            frequency: categoryCopy.frequencyTitle,
            count: captioned.length,
          })}
        </span>
        <span aria-hidden className="ml-auto text-content-subtle">{open ? '▾' : '▸'}</span>
      </button>
      {open && (
        <div className="mt-2 flex flex-col gap-2">
          {/* Plain-language primer: what captions are and what these tools do —
              a newcomer shouldn't need to guess why find/replace or tag frequency
              matter for training. */}
          <p className="m-0 text-content-subtle text-[0.6875rem] leading-relaxed">
            {t('workspace.captionTools.intro')}
          </p>
          <span className="text-content-subtle text-[0.625rem] uppercase tracking-wide">
            {t('workspace.captionTools.findReplace')}
          </span>
          <div className="flex items-center gap-2 flex-wrap">
            <input value={find} onChange={(e) => setFind(e.target.value)}
              placeholder={t(tagMode
                ? 'workspace.captionTools.tagFindPlaceholder'
                : 'workspace.captionTools.textFindPlaceholder')}
              aria-label={t('workspace.captionTools.findLabel')}
              className="px-2 py-1 rounded bg-app/60 border border-border text-content text-xs w-44" />
            <span aria-hidden className="text-content-subtle text-xs">→</span>
            <input value={replace} onChange={(e) => setReplace(e.target.value)}
              placeholder={t('workspace.captionTools.replacePlaceholder')}
              aria-label={t('workspace.captionTools.replaceLabel')}
              className="px-2 py-1 rounded bg-app/60 border border-border text-content text-xs w-48" />
            <label className="flex items-center gap-1 text-xs text-content-muted"
              title={t('workspace.captionTools.tagModeHelp')}>
              <input type="checkbox" checked={tagMode} onChange={(e) => setTagMode(e.target.checked)}
                className="accent-indigo-500" />
              {t('workspace.captionTools.tagMode')}
            </label>
            <button type="button" onClick={apply} disabled={busy || !find.trim()}
              className="px-3 py-1 rounded-lg bg-surface-raised border border-border text-content text-xs font-semibold disabled:opacity-40 hover:bg-surface">
              {t('workspace.captionTools.apply', { count: captioned.length })}
            </button>
          </div>
          {/* Grid tag-filter: the inverse of "show images with this tag" — hide the
              images that ALREADY carry a tag so a captioning checklist only shows
              what's left to do (community's #1 request). Multi-exclusions cumulate;
              active filters show as loud chips above the grid. Session-only (they
              reset on reload / dataset switch — a transient view, not dataset state). */}
          <span className="text-content-subtle text-[0.625rem] uppercase tracking-wide">
            Filter the grid by {categoryCopy.frequencyItem}
          </span>
          <p className="m-0 text-content-subtle text-[0.6875rem] leading-relaxed">
            {t('workspace.captionTools.filterHelp')} {matchHelp}
          </p>
          <div className="flex items-center gap-2 flex-wrap">
            <input value={filterInput} onChange={(e) => setFilterInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); submitFilter(onExclude); } }}
              placeholder={categoryCopy.filterPlaceholder}
              aria-label={t('workspace.captionTools.filterLabel', { item: categoryCopy.frequencyItem })}
              className="px-2 py-1 rounded bg-app/60 border border-border text-content text-xs w-44" />
            <button type="button" onClick={() => submitFilter(onExclude)} disabled={!onExclude || !filterInput.trim()}
              title={t('workspace.captionTools.excludeTitle')}
              className="px-3 py-1 rounded-lg bg-surface-raised border border-border text-content text-xs font-semibold disabled:opacity-40 hover:bg-rose-500/15 hover:border-rose-400/50">
              ⊘ {t('workspace.captionTools.exclude')}
            </button>
            {onInclude && (
              <button type="button" onClick={() => submitFilter(onInclude)} disabled={!filterInput.trim()}
                title={t('workspace.captionTools.onlyWithTitle')}
                className="px-3 py-1 rounded-lg bg-surface-raised border border-border text-content text-xs font-semibold disabled:opacity-40 hover:bg-indigo-500/15 hover:border-indigo-400/50">
                ◉ {t('workspace.captionTools.onlyWith')}
              </button>
            )}
          </div>
          {freq.length > 0 && (
            <div className="flex flex-col gap-1">
              <span className="text-content-subtle text-[0.625rem] uppercase tracking-wide">
                {categoryCopy.frequencyTitle}
              </span>
              <p className="m-0 text-content-subtle text-[0.6875rem] leading-relaxed">
                {categoryCopy.frequencyHelp}{' '}
                {t('workspace.captionTools.frequencyHelp', {
                  item: categoryCopy.frequencyItem,
                  mode: t(mode === 'booru'
                    ? 'workspace.captionTools.tagMode'
                    : 'workspace.captionTools.textMode'),
                })}
              </p>
              <div className="flex flex-wrap gap-1"
                aria-label={t('workspace.captionTools.frequencyLabel', { item: categoryCopy.frequencyItem })}>
                {freq.map(([tag, n]) => {
                  const isExcluded = excludes.includes(tag);
                  const isIncluded = includes.includes(tag);
                  return (
                    <span key={tag}
                      className={`inline-flex items-stretch rounded border overflow-hidden ${
                        isExcluded ? 'border-rose-400/60' : isIncluded ? 'border-indigo-400/60' : 'border-border'}`}>
                      <button type="button"
                        onClick={() => { setFind(tag); setTagMode(mode === 'booru'); }}
                        title={t('workspace.captionTools.frequencyItemTitle', {
                          value: tag,
                          count: n,
                          mode: t(mode === 'booru'
                            ? 'workspace.captionTools.tagMode'
                            : 'workspace.captionTools.textMode'),
                        })}
                        className="px-1.5 py-0.5 bg-app/60 text-[0.6875rem] text-content-muted hover:text-content hover:bg-surface-raised">
                        {tag} <span className="text-content-subtle">×{n}</span>
                      </button>
                      {onExclude && (
                        <button type="button"
                          onClick={() => onExclude(tag)}
                          aria-pressed={isExcluded}
                          aria-label={t(isExcluded
                            ? 'workspace.captionTools.stopHidingLabel'
                            : 'workspace.captionTools.hideLabel', { value: tag })}
                          title={isExcluded
                            ? t('workspace.captionTools.stopHidingTitle', { value: tag })
                            : t('workspace.captionTools.hideTitle', { value: tag })}
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
              <span className="text-content-subtle text-[0.625rem] uppercase tracking-wide">
                {t('workspace.captionTools.files.title')}
              </span>
              <div className="flex items-center gap-2 flex-wrap">
                <button type="button" onClick={onWriteFiles} disabled={busy}
                  title={t('workspace.captionTools.files.writeTitle')}
                  className="px-3 py-1 rounded-lg bg-surface-raised border border-border text-content text-xs font-semibold disabled:opacity-40 hover:bg-surface">
                  💾 {t('workspace.captionTools.files.write')}
                </button>
                {onOpenFolder && (
                  <button type="button" onClick={onOpenFolder}
                    title={t('workspace.captionTools.files.open')}
                    aria-label={t('workspace.captionTools.files.open')}
                    className="px-2 py-1 rounded-lg bg-surface-raised border border-border text-content text-xs hover:bg-surface">
                    📂
                  </button>
                )}
                <span className="text-content-subtle text-[0.6875rem]">
                  {kind === 'style'
                    ? t('workspace.captionTools.files.styleHelp')
                    : t('workspace.captionTools.files.defaultHelp')}
                </span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
