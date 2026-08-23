// react-frontend/src/components/dataset/panelFormatters.js
// Tiny formatters shared by TrainingPanel and its extracted sections
// (moved verbatim from TrainingPanel.jsx, 2026-08-24, slice 1). NOT the
// same contract as denseModels.fmtBytes (KB floor vs bytes floor) - the
// checkpoint list has always rounded sub-kilobyte sizes up to 1 KB.
export const baseName = (p) => String(p || '').replace(/[\\/]+$/, '').split(/[\\/]/).pop() || String(p || '');

export const fmtBytes = (b) => {
  if (b == null) return '';
  if (b >= 1e9) return `${(b / 1e9).toFixed(1)} GB`;
  if (b >= 1e6) return `${Math.round(b / 1e6)} MB`;
  return `${Math.max(1, Math.round(b / 1e3))} KB`;
};
