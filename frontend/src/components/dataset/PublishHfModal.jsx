/**
 * PublishHfModal — export a dataset to the Hugging Face Hub as a `dataset` repo.
 *
 * EXPORT only (no import). Private by default; NFAA tag ON by default; the
 * reference photo (a possibly real person's source face) is OFF by default. A
 * consent checkbox is MANDATORY — the server enforces it too, this only keeps
 * Publish disabled until it's ticked. The slow upload runs server-side in a
 * background job; this modal polls the status route and renders the structured
 * outcome (clickable repo URL on success, a specific hint for the very likely
 * read-only-token failure on the first attempt).
 */
import { useEffect, useRef, useState } from 'react';
import { postJson } from '../../hooks/useDataset';

const FIELD =
  'px-3 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-sm ' +
  'placeholder:text-content-subtle focus:border-indigo-500 outline-none';

const LICENSES = [
  ['cc0-1.0', 'CC0 1.0 — public domain'],
  ['cc-by-4.0', 'CC BY 4.0 — attribution'],
  ['cc-by-nc-4.0', 'CC BY-NC 4.0 — non-commercial'],
  ['openrail', 'OpenRAIL — responsible AI'],
  ['other', 'Other / unspecified'],
];

export default function PublishHfModal({ datasetId, onClose }) {
  const [repoId, setRepoId] = useState('');
  const [username, setUsername] = useState(null);
  const [visibility, setVisibility] = useState('private');   // private by default
  const [nfaa, setNfaa] = useState(true);                     // NFAA ON by default
  const [license, setLicense] = useState('cc-by-nc-4.0');
  const [includeRef, setIncludeRef] = useState(false);        // ref photo OFF by default
  const [consent, setConsent] = useState(false);
  const [phase, setPhase] = useState('form');                 // form|publishing|done|error
  const [result, setResult] = useState(null);                 // {repo_url}|{error, error_code}
  const pollRef = useRef(null);

  // Prefill <username>/<slug> from the token owner (best-effort).
  useEffect(() => {
    let alive = true;
    fetch(`/api/dataset/${datasetId}/publish-hf/whoami`, { credentials: 'include' })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (!alive || !d) return;
        setUsername(d.username || null);
        if (d.default_repo_id) setRepoId(d.default_repo_id);
      })
      .catch(() => { /* modal degrades to a free-text repo id */ });
    return () => { alive = false; };
  }, [datasetId]);

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  const poll = () => {
    pollRef.current = setInterval(async () => {
      try {
        const r = await fetch(`/api/dataset/${datasetId}/publish-hf/status`, { credentials: 'include' });
        const d = await r.json();
        if (d.state === 'done') {
          clearInterval(pollRef.current);
          setResult({ repo_url: d.repo_url, count: d.count });
          setPhase('done');
        } else if (d.state === 'error') {
          clearInterval(pollRef.current);
          setResult({ error: d.error, error_code: d.error_code });
          setPhase('error');
        }
      } catch { /* transient — keep polling */ }
    }, 1500);
  };

  const publish = async () => {
    if (!consent || !repoId.trim() || phase === 'publishing') return;
    setPhase('publishing'); setResult(null);
    const d = await postJson(`/api/dataset/${datasetId}/publish-hf`, {
      repo_id: repoId.trim(),
      private: visibility === 'private',
      nfaa, license, include_ref: includeRef, consent: true,
    });
    if (!d.ok) {                        // server guard rejected the request outright
      setResult({ error: d.error, error_code: 'request' });
      setPhase('error');
      return;
    }
    poll();                             // background job launched — watch it
  };

  const busy = phase === 'publishing';
  const readOnly = result?.error_code === 'read_only_token';

  return (
    <div role="dialog" aria-modal="true" aria-label="Publish to Hugging Face"
      className="fixed inset-0 z-[9990] bg-black/80 flex items-center justify-center p-3"
      onClick={busy ? undefined : onClose}>
      <div className="w-full max-w-lg rounded-xl border border-border bg-surface-overlay p-4 flex flex-col gap-3 max-h-[92vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}>
        <h2 className="text-content font-semibold flex items-center gap-1.5">
          🤗 Publish to Hugging Face
        </h2>

        {phase === 'done' ? (
          <div className="flex flex-col gap-2 py-2">
            <p className="text-emerald-300 text-sm font-semibold">✓ Published{result?.count ? ` (${result.count} images)` : ''}</p>
            <a href={result.repo_url} target="_blank" rel="noreferrer"
              className="text-indigo-300 underline break-all text-sm">{result.repo_url}</a>
            <div className="flex justify-end pt-2">
              <button type="button" onClick={onClose}
                className="px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold">Done</button>
            </div>
          </div>
        ) : (
          <>
            <label className="flex flex-col gap-1">
              <span className="text-content-muted text-xs">Repository</span>
              <input value={repoId} onChange={(e) => setRepoId(e.target.value)}
                placeholder={username ? `${username}/my-dataset` : 'username/my-dataset'}
                className={`${FIELD} font-mono`} disabled={busy} />
            </label>

            <div className="flex flex-col gap-1">
              <span className="text-content-muted text-xs">Visibility</span>
              <div className="flex gap-2">
                {[['private', '🔒 Private'], ['public', '🌐 Public']].map(([v, lbl]) => (
                  <label key={v}
                    className={`flex-1 flex items-center gap-2 px-3 py-1.5 rounded-lg border cursor-pointer text-sm ${
                      visibility === v ? 'border-indigo-500 bg-indigo-500/10 text-content' : 'border-border text-content-muted'}`}>
                    <input type="radio" name="hf-visibility" value={v} checked={visibility === v}
                      onChange={() => setVisibility(v)} disabled={busy} />
                    {lbl}
                  </label>
                ))}
              </div>
            </div>

            <label className="flex flex-col gap-1">
              <span className="text-content-muted text-xs">License</span>
              <select value={license} onChange={(e) => setLicense(e.target.value)}
                className={FIELD} disabled={busy}>
                {LICENSES.map(([v, lbl]) => <option key={v} value={v}>{lbl}</option>)}
              </select>
            </label>

            <label className="flex items-center gap-2 text-sm text-content">
              <input type="checkbox" checked={nfaa} onChange={(e) => setNfaa(e.target.checked)} disabled={busy} />
              <span>Not-for-all-audiences tag <span className="text-content-subtle text-xs">(recommended)</span></span>
            </label>

            <label className="flex items-center gap-2 text-sm text-content">
              <input type="checkbox" checked={includeRef} onChange={(e) => setIncludeRef(e.target.checked)} disabled={busy} />
              <span>Include the reference photo
                <span className="text-content-subtle text-xs"> — the real source face; off by default</span></span>
            </label>

            <label className="flex items-start gap-2 text-sm text-content border-t border-border pt-3">
              <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)}
                disabled={busy} className="mt-0.5" />
              <span>I have the right to share these images and the consent of any identifiable person.</span>
            </label>
            <p className="text-content-subtle text-[0.6875rem] -mt-1">
              You are responsible for what you publish. Nothing is uploaded until you press Publish.
            </p>

            {phase === 'error' && (
              <div className="rounded-lg border border-rose-400/50 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
                {result?.error || 'Publish failed.'}
                {readOnly && (
                  <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer"
                    className="block underline mt-1">Create a write token →</a>
                )}
              </div>
            )}

            <div className="flex justify-end gap-2 pt-1">
              <button type="button" onClick={onClose} disabled={busy}
                className="px-3 py-1.5 rounded-lg border border-border bg-surface text-content-muted hover:text-content text-sm disabled:opacity-40">
                Cancel
              </button>
              <button type="button" onClick={publish} disabled={!consent || !repoId.trim() || busy}
                className="px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40 flex items-center gap-2">
                {busy && <span className="inline-block w-3 h-3 border-2 border-white/40 border-t-white rounded-full animate-spin" />}
                {busy ? 'Publishing…' : 'Publish'}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
