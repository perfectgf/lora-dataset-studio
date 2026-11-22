import { SECTION_FOR_TARGET } from './workspaceSections.js';

/**
 * What the guided path (hooks/useGuidedFlow) says about each section of the
 * workspace rail.
 *
 * The path and the rail describe one journey: every step of the path lives in
 * a section (SECTION_FOR_TARGET — the same table jumpTo uses to send a step's
 * click to its section). They used to be drawn as two lists, the rail and a
 * PROGRESS checklist under it, which doubled the rows and gave one screen two
 * names. This folds the path onto the rail's own rows instead.
 *
 * Per section id that has at least one step:
 *   done        every required step is done (a section whose steps are all
 *               optional — Curation's face scoring — is done when those are)
 *   busy        one of its steps is running
 *   next        the path's next step is in this section
 *   short       the figure to show at the row's edge ("6/25", "54 kept") — the
 *               last of its steps that has one
 *   unavailable one of its steps cannot be done on this install, and `hint`
 *               says why (Studio without ComfyUI)
 * A section without a step (Scrape, and Import & export when training is
 * available) has no entry: its row carries no state, which is the truth.
 */
export function sectionProgress(steps, nextStep) {
  const bySection = new Map();
  for (const step of steps || []) {
    const id = SECTION_FOR_TARGET[step.targetId];
    if (!id) continue;
    if (!bySection.has(id)) bySection.set(id, []);
    bySection.get(id).push(step);
  }
  const nextId = nextStep ? SECTION_FOR_TARGET[nextStep.targetId] : null;
  const out = {};
  for (const [id, list] of bySection) {
    const required = list.filter((step) => !step.optional);
    const judged = required.length ? required : list;
    const blocked = list.find((step) => step.unavailable);
    const figure = [...list].reverse().find((step) => step.short);
    out[id] = {
      done: judged.every((step) => step.done),
      busy: list.some((step) => step.busy),
      next: id === nextId,
      short: figure ? figure.short : '',
      unavailable: Boolean(blocked),
      hint: blocked ? blocked.hint || '' : '',
    };
  }
  return out;
}
