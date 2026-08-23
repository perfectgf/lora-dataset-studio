/* The help registry — the single pivot for the bidirectional Help mode.
   PURE JS (zero JSX, zero Vite ?raw imports) so node --test can import it and
   the contract test (tests/help-registry-contract.test.mjs) can validate every
   route/anchor/focus against the real markdownHeadingId, the settings registry,
   and the workspace sections.

   Each topic maps ONE thing in the app to ONE place in the guide:
     { id, kind, title, keywords, guide: { chapter, anchor }, app: { route, focus? }, tip? }
   - kind    'section' | 'setting' | 'action' | 'page'
   - guide   chapter ∈ {getting-started, using-the-app, dataset-guide,
             troubleshooting, getting-help, settings-reference}; anchor = the id
             of an H2 in that chapter, computed by markdownHeadingId.
   - app     route = HashRouter path (may carry a query); focus = optional DOM id
             of a field on the target screen (scroll + highlight on arrival).
   - tip     { trigger, text } — an optional one-time contextual hint.

   ORDER MATTERS: for a given (chapter, anchor) the FIRST topic in this array is
   the one whose screen the guide's "Open this screen →" button opens. Section /
   page topics are therefore listed BEFORE the field / action topics that share
   their anchor (e.g. workspace-images before workspace-add/curation/export). */

/* The topic DATA lives in ./topics/* (one module per section) and the
   builders in ./topicBuilders.js — all pure JS. This module only
   concatenates them, in the order the contract above depends on, and
   exposes the read API. */
import { SETTINGS_SECTION_TOPICS } from './topics/settingsSections.js';
import { WORKSPACE_SECTION_TOPICS } from './topics/workspaceSections.js';
import { PAGE_TOPICS } from './topics/pages.js';
import { VIDEO_LANE_TOPICS } from './topics/videoLane.js';
import { SETTINGS_FIELD_TOPICS } from './topics/settingsFields.js';
import { ACTION_TOPICS } from './topics/actions.js';

const TOPICS = [
  ...SETTINGS_SECTION_TOPICS,
  ...WORKSPACE_SECTION_TOPICS,
  ...PAGE_TOPICS,
  ...VIDEO_LANE_TOPICS,
  ...SETTINGS_FIELD_TOPICS,
  ...ACTION_TOPICS,
];

Object.freeze(TOPICS);

const BY_ID = new Map(TOPICS.map((t) => [t.id, t]));

/** The frozen registry array (registry order preserved). */
export const helpTopics = TOPICS;

/** Look up a single topic by id, or undefined. */
export function getHelpTopic(id) {
  return BY_ID.get(id);
}

/** All topics whose guide.chapter === chapterId, in registry order. */
export function helpTopicsForChapter(chapterId) {
  return TOPICS.filter((t) => t.guide.chapter === chapterId);
}

/** Case-insensitive search over id / title / keywords. Registry order. */
export function searchHelpTopics(query) {
  const q = String(query || '').trim().toLowerCase();
  if (!q) return [];
  return TOPICS.filter((t) =>
    t.id.toLowerCase().includes(q)
    || t.title.toLowerCase().includes(q)
    || t.keywords.some((k) => k.toLowerCase().includes(q)));
}

/** All one-time tips, flattened: { topicId, trigger, text, guide }. */
export function helpTips() {
  return TOPICS.filter((t) => t.tip).map((t) => ({
    topicId: t.id, trigger: t.tip.trigger, text: t.tip.text, guide: t.guide,
  }));
}

/** Resolve a tip by its stable trigger string (or null). */
export function getHelpTip(trigger) {
  return helpTips().find((t) => t.trigger === trigger) || null;
}

/** The in-app HashRouter "to" for a topic's guide anchor. The Getting-help
    chapter lives at its own /help route, every other chapter under /guide. */
export function guideHref(chapter, anchor) {
  const base = chapter === 'getting-help' ? '/help' : `/guide/${chapter}`;
  return anchor ? `${base}?h=${anchor}` : base;
}

/** Same, for a topic. */
export function topicGuideHref(topic) {
  if (!topic) return null;
  return guideHref(topic.guide.chapter, topic.guide.anchor);
}
