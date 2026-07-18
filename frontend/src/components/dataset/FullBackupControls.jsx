/**
 * 💾 Back up everything — the library-level control: one button that archives
 * EVERY dataset plus the app config (secrets excluded) into a single master
 * file, produced by a background job with visible progress, then a download +
 * "open folder". The restore overlay is the other half: when the library's
 * "Import backup" is handed a master archive, useDataset routes it to the same
 * background job and this component shows its progress + honest final report.
 *
 * Presentational + props-driven (the `backup` bundle from useDataset); the
 * progress phrasing and summaries come from utils/fullBackup (pure, tested).
 */
import { HelpBadge } from '../../help/HelpMode';
import {
  describeProgress, progressPercent, summarizeBackupResult, summarizeRestoreReport,
} from '../../utils/fullBackup';

function ProgressBar({ status }) {
  const pct = progressPercent(status);
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-surface-raised">
      <div
        className={`h-full rounded-full bg-gradient-primary transition-[width] duration-300 ${pct == null ? 'animate-pulse w-1/3' : ''}`}
        style={pct == null ? undefined : { width: `${pct}%` }}
      />
    </div>
  );
}

function Overlay({ label, children }) {
  return (
    <div role="dialog" aria-modal="true" aria-label={label}
      className="fixed inset-0 z-[9990] flex items-center justify-center bg-black/80 p-3">
      <div className="flex w-full max-w-md flex-col gap-3 rounded-xl border border-primary/40 bg-app p-5">
        {children}
      </div>
    </div>
  );
}

function Notes({ notes }) {
  if (!notes?.length) return null;
  return (
    <ul className="max-h-48 overflow-y-auto rounded-lg border border-border bg-surface-raised p-2 text-xs text-content-muted">
      {notes.map((n, i) => <li key={i} className="py-0.5">{n}</li>)}
    </ul>
  );
}

function BackupOverlay({ job, onDownload, onOpenFolder, onDismiss }) {
  if (!job) return null;
  const done = job.state === 'done';
  const error = job.state === 'error';
  const summary = done ? summarizeBackupResult(job.result) : null;
  return (
    <Overlay label="Back up everything">
      <h2 className="text-base font-semibold text-content">
        {done ? '✅ ' : error ? '⚠️ ' : '💾 '}
        {done ? summary.headline : error ? 'Backup failed' : 'Backing up your library'}
      </h2>
      {job.state === 'running' && (
        <>
          <p className="text-sm text-content-muted">{describeProgress(job) || 'Preparing…'}</p>
          <ProgressBar status={job} />
          <p className="text-xs text-content-subtle">
            You can keep working — this runs in the background.
          </p>
        </>
      )}
      {done && (
        <>
          <Notes notes={summary.notes} />
          <div className="flex flex-wrap items-center justify-end gap-2 pt-1">
            <button type="button" onClick={onOpenFolder}
              className="rounded-lg border border-border bg-surface px-3 py-1.5 text-sm font-semibold text-content hover:bg-surface-raised">
              📂 Open folder
            </button>
            <button type="button" onClick={() => onDownload(job.result?.name)}
              className="rounded-lg bg-gradient-primary px-3.5 py-1.5 text-sm font-semibold text-white">
              ⬇ Download
            </button>
            <button type="button" onClick={onDismiss}
              className="rounded-lg border border-border bg-surface px-3 py-1.5 text-sm text-content-muted hover:bg-surface-raised">
              Close
            </button>
          </div>
        </>
      )}
      {error && (
        <>
          <p className="text-sm text-rose-300">{job.error || 'Something went wrong.'}</p>
          <div className="flex justify-end pt-1">
            <button type="button" onClick={onDismiss}
              className="rounded-lg border border-border bg-surface px-3 py-1.5 text-sm text-content hover:bg-surface-raised">
              Close
            </button>
          </div>
        </>
      )}
    </Overlay>
  );
}

function RestoreOverlay({ job, onDismiss }) {
  if (!job) return null;
  const done = job.state === 'done';
  const error = job.state === 'error';
  const report = done ? summarizeRestoreReport(job.result) : null;
  return (
    <Overlay label="Restore everything">
      <h2 className="text-base font-semibold text-content">
        {done ? '✅ ' : error ? '⚠️ ' : '♻️ '}
        {done ? report.headline : error ? 'Restore failed' : 'Restoring your backup'}
      </h2>
      {job.state === 'running' && (
        <>
          <p className="text-sm text-content-muted">
            {describeProgress(job, 'Restoring') || 'Preparing…'}
          </p>
          <ProgressBar status={job} />
        </>
      )}
      {done && <Notes notes={report.notes} />}
      {error && <p className="text-sm text-rose-300">{job.error || 'Something went wrong.'}</p>}
      {(done || error) && (
        <div className="flex justify-end pt-1">
          <button type="button" onClick={onDismiss}
            className="rounded-lg border border-border bg-surface px-3 py-1.5 text-sm text-content hover:bg-surface-raised">
            Close
          </button>
        </div>
      )}
    </Overlay>
  );
}

export default function FullBackupControls({ backup }) {
  if (!backup) return null;
  const running = backup.job?.state === 'running';
  return (
    <>
      <span className="inline-flex items-center gap-1">
        <button type="button" onClick={backup.start} disabled={running}
          title="Archive every dataset + your settings (API keys excluded) into one file"
          aria-label="Back up everything"
          className="rounded-lg border border-border bg-surface px-3 py-1.5 text-sm font-semibold text-content transition-colors hover:border-primary/40 hover:bg-surface-raised disabled:opacity-50">
          💾<span className="hidden sm:inline"> Back up everything</span>
        </button>
        <HelpBadge topic="library-backup" />
      </span>
      <BackupOverlay job={backup.job} onDownload={backup.download}
        onOpenFolder={backup.openFolder} onDismiss={backup.dismiss} />
      <RestoreOverlay job={backup.restoreJob} onDismiss={backup.dismissRestore} />
    </>
  );
}
