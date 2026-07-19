// react-frontend/src/components/dataset/studio/ExportGridModal.jsx
/**
 * « Export grid » — small options modal that composes the current run's
 * checkpoint × strength grid (labels + FORMAT headers baked in) into ONE image
 * server-side (PIL) and downloads it. The classic XY plot, clean, ready for
 * Civitai/Reddit.
 *
 * Sober set of options (not a factory): which format block (or all stacked),
 * include the prompt or not (default OFF — prompts can be personal/NSFW), tile
 * size (2 crans), file format (JPEG q90 default / PNG), and the discreet
 * « Made with LoRA Dataset Studio » footer toggle. The compose call is DB + PIL
 * only (no ComfyUI), so it works offline.
 */
import { useMemo, useRef, useState } from 'react';
import { useToast } from '../../common/Toast';
import { useFocusTrap } from '../../../hooks/useFocusTrap';
import { fetchWithCsrfRetry, getCsrfToken } from '../../../api/fetchClient';
import { useI18n } from '../../../i18n/I18nContext';

const MAX_CANVAS_SIDE = 8000; // must mirror studio_grid_export.MAX_CANVAS_SIDE

function _nameFromDisposition(header) {
  const m = /filename="?([^"]+)"?/.exec(header || '');
  return m ? m[1] : null;
}

export default function ExportGridModal({ open, onClose, datasetId, family, run, aspects, rows, cols }) {
  const toast = useToast();
  const { t } = useI18n();
  const ref = useRef(null);
  useFocusTrap(ref, open);
  const [aspect, setAspect] = useState('all');
  const [includePrompt, setIncludePrompt] = useState(false);
  const [cellSize, setCellSize] = useState(512);
  const [fileFormat, setFileFormat] = useState('jpeg');
  const [footer, setFooter] = useState(true);
  const [busy, setBusy] = useState(false);

  // Rough final-size estimate to warn (before composing) when the server will
  // downscale to the 8000px cap. Blocks = 1 when a single format is picked, else
  // the number of formats present (stacked). Height dominates a tall sweep.
  const willDownscale = useMemo(() => {
    const nBlocks = aspect === 'all' ? Math.max(1, (aspects || []).length) : 1;
    const estW = cellSize * 1.2 + (cols || 1) * (cellSize + 12);
    const estH = 300 + nBlocks * ((rows || 1) * (cellSize + 12) + 110);
    return Math.max(estW, estH) > MAX_CANVAS_SIDE;
  }, [aspect, aspects, cellSize, rows, cols]);

  if (!open) return null;

  async function doExport() {
    setBusy(true);
    try {
      const res = await fetchWithCsrfRetry(`/api/dataset/${datasetId}/lora-test/export-grid`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCsrfToken() },
        body: JSON.stringify({
          family: family || null,
          run_seed: run?.seed ?? null,
          prompt: run?.prompt ?? null,
          aspect,
          include_prompt: includePrompt,
          cell_size: cellSize,
          format: fileFormat,
          footer,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        toast.error(err.error || t('studio.export.httpFailed', { status: res.status }));
        return;
      }
      const blob = await res.blob();
      const downscaled = res.headers.get('X-Grid-Downscaled') === '1';
      const name = _nameFromDisposition(res.headers.get('Content-Disposition')) || 'lora-grid.jpg';
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = name;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      toast.success(downscaled
        ? t('studio.export.exportedDownscaled')
        : t('studio.export.exported'));
      onClose();
    } catch {
      toast.error(t('studio.export.failed'));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="fixed inset-0 z-[9999] bg-black/70 flex items-center justify-center p-4"
      role="dialog" aria-modal="true" aria-label={t('studio.export.title')} ref={ref}
      onClick={(e) => { if (e.target === e.currentTarget && !busy) onClose(); }}>
      <div className="w-full max-w-md rounded-2xl border border-border bg-surface-overlay p-4 flex flex-col gap-3 shadow-xl">
        <div className="flex items-center justify-between">
          <h2 className="text-content text-sm font-semibold flex items-center gap-1.5">
            <span aria-hidden>🖼</span> {t('studio.export.title')}
          </h2>
          <button type="button" onClick={onClose} disabled={busy} aria-label={t('common.close')}
            className="w-8 h-8 rounded-lg border border-border bg-app text-content-muted hover:text-content disabled:opacity-40">×</button>
        </div>
        <p className="text-content-subtle text-[0.6875rem] leading-snug">
          {t('studio.export.description')}
        </p>

        {/* Format block */}
        <label className="flex flex-col gap-1">
          <span className="text-content-muted text-[0.625rem] uppercase">{t('studio.export.formatBlock')}</span>
          <select value={aspect} onChange={(e) => setAspect(e.target.value)}
            className="rounded-lg border border-border bg-app px-2 py-1.5 text-[0.75rem] text-content">
            <option value="all">{t('studio.export.allFormats')}</option>
            {(aspects || []).map((a) => <option key={a} value={a}>{a}</option>)}
          </select>
        </label>

        {/* Tile size */}
        <div className="flex flex-col gap-1">
          <span className="text-content-muted text-[0.625rem] uppercase">{t('studio.export.tileSize')}</span>
          <div className="flex gap-2">
            {[512, 768].map((sz) => (
              <button key={sz} type="button" onClick={() => setCellSize(sz)}
                className={`px-3 py-1.5 rounded-lg border text-[0.75rem] ${cellSize === sz
                  ? 'border-indigo-400/60 bg-indigo-500/15 text-indigo-200'
                  : 'border-border bg-app text-content-muted hover:text-content'}`}>
                {sz}px
              </button>
            ))}
          </div>
        </div>

        {/* File format */}
        <div className="flex flex-col gap-1">
          <span className="text-content-muted text-[0.625rem] uppercase">{t('studio.export.fileFormat')}</span>
          <div className="flex gap-2">
            {[['jpeg', t('studio.export.jpeg')], ['png', t('studio.export.png')]].map(([v, lbl]) => (
              <button key={v} type="button" onClick={() => setFileFormat(v)}
                className={`px-3 py-1.5 rounded-lg border text-[0.75rem] ${fileFormat === v
                  ? 'border-indigo-400/60 bg-indigo-500/15 text-indigo-200'
                  : 'border-border bg-app text-content-muted hover:text-content'}`}>
                {lbl}
              </button>
            ))}
          </div>
        </div>

        {/* Toggles */}
        <label className="flex items-start gap-2 cursor-pointer">
          <input type="checkbox" checked={includePrompt}
            onChange={(e) => setIncludePrompt(e.target.checked)} className="mt-0.5" />
          <span className="text-[0.75rem] text-content">{t('studio.export.includePrompt')}
            <span className="block text-content-subtle text-[0.625rem]">{t('studio.export.promptHint')}</span>
          </span>
        </label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="checkbox" checked={footer}
            onChange={(e) => setFooter(e.target.checked)} />
          <span className="text-[0.75rem] text-content">{t('studio.export.footer')}</span>
        </label>

        {willDownscale && (
          <p className="text-amber-300/90 text-[0.625rem] rounded-lg border border-amber-400/30 bg-amber-500/10 px-2 py-1.5">
            {t('studio.export.downscaleWarning', { size: MAX_CANVAS_SIDE })}
          </p>
        )}

        <div className="flex items-center justify-end gap-2 pt-1">
          <button type="button" onClick={onClose} disabled={busy}
            className="px-3 py-1.5 rounded-lg border border-border bg-app text-content-muted text-[0.75rem] hover:text-content disabled:opacity-40">
            {t('common.close')}
          </button>
          <button type="button" onClick={doExport} disabled={busy}
            className="px-4 py-1.5 rounded-lg bg-gradient-primary text-white text-[0.75rem] font-semibold disabled:opacity-60">
            {busy ? t('studio.export.composing') : `⬇ ${t('studio.results.export')}`}
          </button>
        </div>
      </div>
    </div>
  );
}
