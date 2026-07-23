/**
 * 💾 Backup — the library-level menu: ONE header disclosure that holds both
 * ways in and out of an archive. "Back up everything" archives EVERY dataset
 * plus the app config (secrets excluded) into a single master file, produced by
 * a background job with visible progress, then a download + "open folder";
 * "Include trained LoRAs" is an OPTION of that action and therefore lives right
 * under it, never orphaned in the toolbar. "Import backup" is the way back in:
 * handed a master archive, useDataset routes it to the same background job and
 * this component shows its progress + honest final report.
 *
 * The two overlays are rendered as SIBLINGS of the <details>, not inside it: a
 * backup that is running must stay visible after the menu is closed.
 *
 * Presentational + props-driven (the `backup` bundle from useDataset); the
 * progress phrasing and summaries come from utils/fullBackup (pure, tested).
 */
import { useRef, useState } from 'react';
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

const MENU_ITEM = 'w-full flex items-center gap-2 text-left px-2.5 py-1.5 rounded-md text-sm text-content hover:bg-surface-raised disabled:opacity-40';

export default function FullBackupControls({ backup, onRestore }) {
  const [includeLoras, setIncludeLoras] = useState(false);
  const menuRef = useRef(null);
  const restoreRef = useRef(null);
  if (!backup && !onRestore) return null;
  const running = backup?.job?.state === 'running';
  // <details> is uncontrolled: closing it after an action is a direct DOM poke,
  // the same shape the workspace header menu uses.
  const closeMenu = () => { if (menuRef.current) menuRef.current.open = false; };
  return (
    <>
      {/* summary en display:flex → pas de marqueur natif ; les items restent
          montés en permanence (details ne fait que masquer l'affichage). */}
      <details ref={menuRef} className="relative">
        <summary
          title="Back up the whole library, or import a backup archive"
          className="flex items-center gap-1 rounded-lg border border-border bg-surface px-3 py-1.5 text-sm font-semibold text-content-muted hover:text-content hover:bg-surface-raised cursor-pointer select-none">
          <span aria-hidden>{running ? '⏳' : '💾'}</span>
          {/* The label itself carries the in-flight state, so a closed menu
              still says a backup is running. */}
          <span className="hidden sm:inline">{running ? 'Backing up…' : 'Backup'}</span>
          <span aria-hidden className="text-content-subtle">⋯</span>
        </summary>
        <div className="absolute right-0 top-full mt-1 z-20 w-80 rounded-lg border border-border bg-surface-overlay shadow-xl p-1.5 flex flex-col gap-0.5">
          {backup && (
            <>
              <button type="button" disabled={running}
                onClick={() => { closeMenu(); backup.start(includeLoras); }}
                title="Archive every dataset, its training history + your settings (API keys excluded) into one file"
                className={MENU_ITEM}>
                <span className="whitespace-nowrap">💾 Back up everything</span>
                <span className="ml-auto shrink-0 text-content-subtle text-[0.625rem]">
                  {running ? 'running…' : 'datasets · settings'}
                </span>
              </button>
              {/* Option OF the action above — kept adjacent to it on purpose. */}
              <label className={`${MENU_ITEM} cursor-pointer text-content-muted`}
                title="Also bundle the trained LoRA files (larger backup). Training history is always included regardless.">
                <input type="checkbox" checked={includeLoras} disabled={running}
                  onChange={(e) => setIncludeLoras(e.target.checked)}
                  className="accent-primary" />
                Include trained LoRAs
              </label>
            </>
          )}
          {onRestore && (
            <button type="button"
              onClick={() => { closeMenu(); restoreRef.current?.click(); }}
              title="Import a portable dataset backup — a new dataset will be created"
              className={MENU_ITEM}>
              <span className="whitespace-nowrap">📦 Import backup</span>
              <span className="ml-auto shrink-0 text-content-subtle text-[0.625rem]">.zip archive</span>
            </button>
          )}
        </div>
      </details>
      <HelpBadge topic="library-backup" />
      {/* Outside the <details>: a file input inside a collapsed disclosure is
          display:none, and .click() on it would be a no-op. */}
      {onRestore && (
        <input ref={restoreRef} type="file" accept=".zip,application/zip" className="hidden"
          aria-label="Choose a dataset backup ZIP"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onRestore(file);
            e.target.value = '';
          }} />
      )}
      {backup && (
        <>
          <BackupOverlay job={backup.job} onDownload={backup.download}
            onOpenFolder={backup.openFolder} onDismiss={backup.dismiss} />
          <RestoreOverlay job={backup.restoreJob} onDismiss={backup.dismissRestore} />
        </>
      )}
    </>
  );
}
