import { useEffect, useState } from 'react';
import { VIDEO_QUICK_PROMPT_CATEGORIES, promptForMode } from './videoPromptPresets';

const CATEGORY_KEY = 'ldsVideoQuickPrompts.category';

/** The tab to open on arrival: the one last used, but only if it still exists —
 *  a category renamed or removed must not leave the picker showing nothing. */
function storedCategory() {
  try {
    const id = window.localStorage.getItem(CATEGORY_KEY);
    if (VIDEO_QUICK_PROMPT_CATEGORIES.some((c) => c.id === id)) return id;
  } catch { /* private windows and blocked storage: the default is fine */ }
  return VIDEO_QUICK_PROMPT_CATEGORIES[0].id;
}

/**
 * ⚡ Quick prompts — the H3 preset chips, under the Motion field.
 *
 * A chip APPENDS on its own line instead of replacing what is there, so a shot
 * is built by stacking: a Scenario or a Style first, then Camera + Audio +
 * Voice on top. That is deliberately the same promise ✨ Enrich makes, because
 * a picker that wiped a written prompt would be a trap next to a button that
 * does not.
 *
 * The chips are `min-h-10 lg:min-h-0`: finger-sized below the desktop
 * breakpoint, unchanged on a desktop — the responsive contract, not a style
 * choice.
 */
export default function VideoQuickPrompts({ mode, onAppend }) {
  const [activeId, setActiveId] = useState(storedCategory);

  useEffect(() => {
    try { window.localStorage.setItem(CATEGORY_KEY, activeId); } catch { /* see storedCategory */ }
  }, [activeId]);

  const active = VIDEO_QUICK_PROMPT_CATEGORIES.find((c) => c.id === activeId)
    ?? VIDEO_QUICK_PROMPT_CATEGORIES[0];

  return (
    <section data-testid="video-quick-prompts"
      className="flex flex-col gap-1.5 rounded-lg border border-border bg-surface-raised p-2">
      <span className="text-[0.6875rem] font-semibold uppercase tracking-wider text-content-muted">
        ⚡ Quick prompts
      </span>

      <div role="group" aria-label="Quick prompt categories"
        className="flex gap-1.5 overflow-x-auto pb-0.5">
        {VIDEO_QUICK_PROMPT_CATEGORIES.map((cat) => (
          <button key={cat.id} type="button" onClick={() => setActiveId(cat.id)}
            aria-pressed={cat.id === active.id} title={cat.label}
            className={`flex shrink-0 items-center gap-1 rounded-lg border px-2 py-1 text-[0.6875rem] font-medium min-h-10 lg:min-h-0 ${
              cat.id === active.id ? 'border-primary bg-primary/10 text-content'
                : 'border-border text-content-muted hover:text-content'}`}>
            <span aria-hidden="true">{cat.emoji}</span>
            <span className="hidden sm:inline">{cat.label}</span>
          </button>
        ))}
      </div>

      <div className="flex flex-wrap gap-1.5">
        {active.prompts.map((p) => (
          <button key={p.label} type="button"
            onClick={() => onAppend?.(promptForMode(p.prompt, mode))}
            title={promptForMode(p.prompt, mode)}
            className="min-h-10 rounded-full border border-border bg-surface px-3 py-1 text-[0.6875rem] font-medium text-content hover:border-primary hover:text-content lg:min-h-0">
            {p.label}
          </button>
        ))}
      </div>

      <span className="text-[0.6875rem] text-content-subtle">
        Each chip adds a line — stack a scenario, a camera move and an audio bed.
        {mode === 't2v'
          ? ' Text-to-video: the presets drop their reference to a start frame, since there is none.'
          : ' The scenarios point at your start frame the way H3’s own template does.'}
      </span>
    </section>
  );
}
