import { WORKSPACE_SECTIONS, isWorkspaceSection } from './workspaceSections.js';

export const PANEL_STATUS = Object.freeze({
  AVAILABLE: 'available',
  UNAVAILABLE: 'unavailable',
  PENDING: 'pending',
  UNKNOWN: 'unknown',
});

const boolStatus = (value) => value ? PANEL_STATUS.AVAILABLE : PANEL_STATUS.UNAVAILABLE;

const AVAILABILITY = {
  always: () => PANEL_STATUS.AVAILABLE,
  hasSelectableImages: (c) => boolStatus(c.hasSelectableImages),
  character: (c) => boolStatus(c.kind === 'character'),
  watermarkDetected: (c) => boolStatus(c.watermarkDetected > 0),
  unused: (c) => boolStatus(c.unused > 0),
  leakReview: (c) => boolStatus(c.kind !== 'style' && c.hasKeptImages && c.hasLeakMetadata),
  hasCaptionedKept: (c) => boolStatus(c.hasCaptionedKept),
  huggingFace: (c) => boolStatus(c.hfPublish && c.hasKeptImages),
  trainingVisible: (c) => boolStatus(c.trainingVisible),
  trainingQueue: (c) => {
    if (!c.trainingVisible) return PANEL_STATUS.UNAVAILABLE;
    if (!c.trainingStatusReady) return PANEL_STATUS.PENDING;
    return boolStatus(c.trainingQueueCount > 0);
  },
  studioVisible: (c) => boolStatus(c.studioVisible),
};

export function getWorkspacePanel(sectionId, panelId) {
  const section = WORKSPACE_SECTIONS.find((item) => item.id === sectionId);
  return section?.panels?.find((item) => item.id === panelId) || null;
}

export function getWorkspacePanelStatus(sectionId, panelId, context) {
  const panel = getWorkspacePanel(sectionId, panelId);
  if (!panel) return PANEL_STATUS.UNKNOWN;
  const predicate = AVAILABILITY[panel.when];
  if (!predicate) return PANEL_STATUS.UNKNOWN;
  return predicate(context || {});
}

export function getWorkspacePanels(sectionId, context) {
  const section = WORKSPACE_SECTIONS.find((item) => item.id === sectionId);
  if (!section) return [];
  return section.panels.filter(
    (panel) => getWorkspacePanelStatus(sectionId, panel.id, context) === PANEL_STATUS.AVAILABLE,
  );
}

export function resolveWorkspaceLocation(searchParams, context) {
  const requestedSection = searchParams.get('section');
  const requestedPanel = searchParams.get('panel');
  if (requestedSection === 'training' && requestedPanel === 'checkpoints') {
    return { section: 'checkpoints', panel: 'manager', pending: false, needsNormalization: true };
  }
  if (requestedSection === 'training' && requestedPanel === 'studio') {
    return { section: 'studio', panel: 'launcher', pending: false, needsNormalization: true };
  }
  if (!isWorkspaceSection(requestedSection)) {
    return { section: 'images', panel: null, pending: false, needsNormalization: true };
  }
  if (!requestedPanel) {
    return { section: requestedSection, panel: null, pending: false, needsNormalization: false };
  }
  const status = getWorkspacePanelStatus(requestedSection, requestedPanel, context);
  if (status === PANEL_STATUS.AVAILABLE) {
    return { section: requestedSection, panel: requestedPanel, pending: false, needsNormalization: false };
  }
  if (status === PANEL_STATUS.PENDING) {
    return { section: requestedSection, panel: requestedPanel, pending: true, needsNormalization: false };
  }
  return { section: requestedSection, panel: null, pending: false, needsNormalization: true };
}

export function withWorkspaceLocation(searchParams, section, panel = null) {
  const next = new URLSearchParams(searchParams);
  next.set('section', section);
  if (panel) next.set('panel', panel);
  else next.delete('panel');
  return next;
}
