// « 🔎 Describe » — drop or pick an image, the Ollama vision model describes it as a
// ready-to-paste TEST PROMPT (scene/pose/framing/clothing, no identity, no trigger
// word — the Studio injects the trigger separately). On success the text is handed to
// `onResult`, which decides whether to overwrite a non-empty field. The model may be
// cold (a few seconds) so the busy state uses a generous server timeout.
import { useRef, useState } from 'react';
import { useFocusTrap } from '../../../hooks/useFocusTrap';
import { fetchWithCsrfRetry, getCsrfToken } from '../../../api/fetchClient';

const ACCEPT = 'image/png,image/jpeg,image/webp';
const MAX_BYTES = 20 * 1024 * 1024; // mirror lts.STUDIO_DESCRIBE_MAX_BYTES

export default function DescribeImageModal({ open, onClose, onResult }) {
  const ref = useRef(null);
  const inputRef = useRef(null);
  useFocusTrap(ref, open);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [fileName, setFileName] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  if (!open) return null;

  async function describe(file) {
    if (!file) return;
    setError(null);
    if (!/^image\/(png|jpe?g|webp)$/i.test(file.type)) {
      setError('Pick an image file (webp, png or jpg).');
      return;
    }
    if (file.size > MAX_BYTES) {
      setError(`Image too large (max ${MAX_BYTES / (1024 * 1024)} MB).`);
      return;
    }
    setFileName(file.name);
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append('image', file);
      fd.append('csrf_token', getCsrfToken());
      const res = await fetchWithCsrfRetry('/api/studio/describe-image', {
        method: 'POST',
        headers: { 'X-CSRFToken': getCsrfToken() },
        body: fd,
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        // The server carries the real, actionable reason (Ollama unreachable/rejected,
        // GPU busy, bad file) in `error` — surface it verbatim inside the modal.
        setError(body.error || `Describe failed (HTTP ${res.status})`);
        return;
      }
      if (!body.prompt) {
        setError('The vision model returned an empty description.');
        return;
      }
      onResult(body.prompt);
      onClose();
    } catch {
      setError('Describe failed — check that the app can reach Ollama, then try again.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="fixed inset-0 z-[9999] bg-black/70 flex items-center justify-center p-4"
      role="dialog" aria-modal="true" aria-label="Describe an image into a test prompt" ref={ref}
      onClick={(e) => { if (e.target === e.currentTarget && !busy) onClose(); }}>
      <div className="w-full max-w-md rounded-2xl border border-border bg-surface-overlay p-4 flex flex-col gap-3 shadow-xl">
        <div className="flex items-center justify-between">
          <h2 className="text-content text-sm font-semibold flex items-center gap-1.5">
            <span aria-hidden>🔎</span> Describe an image
          </h2>
          <button type="button" onClick={onClose} disabled={busy} aria-label="Close"
            className="w-8 h-8 rounded-lg border border-border bg-app text-content-muted hover:text-content disabled:opacity-40">×</button>
        </div>
        <p className="text-content-subtle text-[0.6875rem] leading-snug">
          The vision model turns the image into a test prompt (scene, pose, framing, outfit).
          It never names the person or adds the trigger word — the Studio handles those.
        </p>

        <button type="button"
          onClick={() => { if (!busy) inputRef.current?.click(); }}
          onDragOver={(e) => { e.preventDefault(); if (!busy) setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            if (!busy) describe(e.dataTransfer.files?.[0]);
          }}
          disabled={busy}
          className={`flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-4 py-8 text-center transition-colors ${
            dragOver ? 'border-purple-400/70 bg-purple-500/10' : 'border-border bg-app/60'} disabled:opacity-60`}>
          {busy ? (
            <>
              <span className="inline-block w-6 h-6 border-2 border-purple-400/40 border-t-purple-400 rounded-full animate-spin" aria-hidden />
              <span className="text-content text-[0.75rem]">Describing{fileName ? ` “${fileName}”` : ''}…</span>
              <span className="text-content-subtle text-[0.625rem]">The vision model may be loading — this can take a few seconds.</span>
            </>
          ) : (
            <>
              <span className="text-2xl" aria-hidden>🖼️</span>
              <span className="text-content text-[0.75rem]">Drop an image here, or click to choose</span>
              <span className="text-content-subtle text-[0.625rem]">webp, png or jpg · up to {MAX_BYTES / (1024 * 1024)} MB</span>
            </>
          )}
        </button>
        <input ref={inputRef} type="file" accept={ACCEPT} className="hidden"
          onChange={(e) => { describe(e.target.files?.[0]); e.target.value = ''; }} />

        {error && (
          <p className="m-0 rounded-lg border border-red-400/40 bg-red-500/10 px-3 py-2 text-red-300 text-[0.6875rem]" role="alert">
            {error}
          </p>
        )}

        <div className="flex items-center justify-end pt-1">
          <button type="button" onClick={onClose} disabled={busy}
            className="px-3 py-1.5 rounded-lg border border-border bg-app text-content-muted text-[0.75rem] hover:text-content disabled:opacity-40">
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
