/*
 * Pure helpers for the "💾 Back up everything" flow — progress phrasing, the
 * finished-backup summary and the honest restore report. Kept free of React so
 * node --test can exercise the wording (fullBackup.test.js) without a DOM.
 */

export function formatBytes(b) {
  if (!Number.isFinite(b) || b <= 0) return '0 KB';
  if (b >= 1e9) return `${(b / 1e9).toFixed(1)} GB`;
  if (b >= 1e6) return `${Math.round(b / 1e6)} MB`;
  return `${Math.max(1, Math.round(b / 1e3))} KB`;
}

const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

/*
 * A one-line progress caption while a backup/restore job runs. `total === 0`
 * before the server has counted the datasets yet → a neutral "Preparing…".
 */
export function describeProgress(status, verb = 'Backing up') {
  if (!status || status.state !== 'running') return '';
  const total = status.total || 0;
  const done = Math.min(status.done || 0, total || Infinity);
  if (!total) return 'Preparing…';
  return `${verb} ${done} / ${total} datasets…`;
}

export function progressPercent(status) {
  const total = status?.total || 0;
  if (!total) return null;
  return Math.min(100, Math.round(((status.done || 0) / total) * 100));
}

/*
 * The finished-backup summary. `result` is the server's job result:
 * {name, size_bytes, datasets_total, datasets_backed_up, skipped:[{name,reason}]}.
 */
export function summarizeBackupResult(result) {
  if (!result) return { headline: 'Backup ready', notes: [] };
  const n = result.datasets_backed_up ?? 0;
  const size = formatBytes(result.size_bytes || 0);
  const headline = `Backup ready — ${plural(n, 'dataset')}, ${size}`;
  const notes = [];
  const runs = result.runs_total ?? 0;
  if (runs) notes.push(`Training history included (${plural(runs, 'run')}) — restores your “Trained” status.`);
  if (result.loras_included) {
    const loras = result.loras_total ?? 0;
    notes.push(loras
      ? `${plural(loras, 'trained LoRA')} bundled (${formatBytes(result.loras_bytes || 0)}).`
      : 'No deployed LoRA files were found to bundle.');
  }
  const skipped = result.skipped || [];
  if (skipped.length) {
    notes.push(`${plural(skipped.length, 'dataset')} skipped:`);
    for (const s of skipped) notes.push(`• ${s.name || 'dataset'} — ${s.reason || 'could not be read'}`);
  }
  return { headline, notes };
}

/*
 * The honest end-of-restore report. `result`:
 * {datasets_total, restored, skipped:[{entry,reason}], renamed:[{from,to}],
 *  config_restored, runs_restored, loras_restored, loras_skipped:[{name,reason}]}.
 */
export function summarizeRestoreReport(result) {
  if (!result) return { headline: 'Restore finished', notes: [] };
  const total = result.datasets_total ?? 0;
  const restored = result.restored ?? 0;
  const headline = `Restored ${restored} of ${plural(total, 'dataset')}`;
  const notes = [];
  const runs = result.runs_restored ?? 0;
  if (runs) notes.push(`Training history restored (${plural(runs, 'run')}) — your datasets are back under “Trained”.`);
  const loras = result.loras_restored ?? 0;
  if (loras) notes.push(`${plural(loras, 'trained LoRA')} re-deployed to ComfyUI.`);
  if (result.config_restored) notes.push('Settings restored — re-enter your API keys on this install.');
  const renamed = result.renamed || [];
  for (const r of renamed) notes.push(`• Renamed “${r.from}” → “${r.to}” (a dataset by that name already existed)`);
  const skipped = result.skipped || [];
  if (skipped.length) {
    notes.push(`${plural(skipped.length, 'dataset')} skipped:`);
    for (const s of skipped) notes.push(`• ${s.entry || 'entry'} — ${s.reason || 'could not be restored'}`);
  }
  const lorasSkipped = result.loras_skipped || [];
  if (lorasSkipped.length) {
    notes.push(`${plural(lorasSkipped.length, 'LoRA file')} not restored:`);
    for (const s of lorasSkipped) notes.push(`• ${s.name || 'file'} — ${s.reason || 'could not be restored'}`);
  }
  return { headline, notes };
}

/* Whether a poll snapshot means the job is over (done or errored). */
export function isSettled(status) {
  return !!status && (status.state === 'done' || status.state === 'error');
}
