/* Which sections and jump destinations a video dataset actually shows, and where
 * a deep link lands. The pure half of the workspace rail — same contract as
 * components/dataset/workspaceNavigation.js, so a reader who knows one knows the
 * other, and the two rails cannot drift into different rules for the same word.
 */
import { VIDEO_DATASET_SECTIONS, isVideoDatasetSection } from './videoDatasetSections.js';

export const PANEL_STATUS = Object.freeze({
  AVAILABLE: 'available',
  UNAVAILABLE: 'unavailable',
  UNKNOWN: 'unknown',
});

const boolStatus = (value) => (value ? PANEL_STATUS.AVAILABLE : PANEL_STATUS.UNAVAILABLE);

const AVAILABILITY = {
  always: () => PANEL_STATUS.AVAILABLE,
  hasSelection: (c) => boolStatus(c.selected > 0),
  hasClips: (c) => boolStatus(c.clips > 0),
  requiresReferences: (c) => boolStatus(c.requiresReferences),
};

/** The sections this dataset shows at all. A section with no `when` is always
 * there; `references` is the one that comes and goes, and it goes for every
 * target that does not train on control images. */
export function visibleVideoDatasetSections(context) {
  return VIDEO_DATASET_SECTIONS.filter((section) => {
    if (!section.when) return true;
    const predicate = AVAILABILITY[section.when];
    return !!predicate && predicate(context || {}) === PANEL_STATUS.AVAILABLE;
  });
}

export function getVideoDatasetPanel(sectionId, panelId) {
  const section = VIDEO_DATASET_SECTIONS.find((item) => item.id === sectionId);
  return section?.panels?.find((item) => item.id === panelId) || null;
}

export function getVideoDatasetPanelStatus(sectionId, panelId, context) {
  const panel = getVideoDatasetPanel(sectionId, panelId);
  if (!panel) return PANEL_STATUS.UNKNOWN;
  const predicate = AVAILABILITY[panel.when];
  if (!predicate) return PANEL_STATUS.UNKNOWN;
  return predicate(context || {});
}

export function getVideoDatasetPanels(sectionId, context) {
  const section = VIDEO_DATASET_SECTIONS.find((item) => item.id === sectionId);
  if (!section) return [];
  return section.panels.filter(
    (panel) => getVideoDatasetPanelStatus(sectionId, panel.id, context) === PANEL_STATUS.AVAILABLE,
  );
}

/* Where a video dataset opens when the URL asks for nothing.
 *
 * CLIPS, and not Training — the opposite of the image workspace's choice, on
 * purpose. You open an image dataset to put something INTO it (its landing
 * section is "Add images"); a video dataset is already full the moment it
 * exists, because promotion is what creates it. What you come here to do is look
 * at what the encode produced before spending a night on it. */
const DEFAULT_SECTION = 'clips';

export function resolveVideoDatasetLocation(searchParams, context) {
  const requestedSection = searchParams.get('section');
  const requestedPanel = searchParams.get('panel');
  const visible = visibleVideoDatasetSections(context).map((s) => s.id);
  // A section that exists but is HIDDEN for this dataset (references on a target
  // that has none) normalizes like an unknown one rather than rendering an empty
  // screen — a link shared between two datasets must degrade, not break.
  if (!isVideoDatasetSection(requestedSection) || !visible.includes(requestedSection)) {
    return { section: DEFAULT_SECTION, panel: null, needsNormalization: true };
  }
  if (!requestedPanel) {
    return { section: requestedSection, panel: null, needsNormalization: false };
  }
  const status = getVideoDatasetPanelStatus(requestedSection, requestedPanel, context);
  if (status === PANEL_STATUS.AVAILABLE) {
    return { section: requestedSection, panel: requestedPanel, needsNormalization: false };
  }
  return { section: requestedSection, panel: null, needsNormalization: true };
}

export function withVideoDatasetLocation(searchParams, section, panel = null) {
  const next = new URLSearchParams(searchParams);
  next.set('section', section);
  if (panel) next.set('panel', panel);
  else next.delete('panel');
  return next;
}
