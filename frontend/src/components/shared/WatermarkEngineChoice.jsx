import { useState } from 'react';
import { putJson } from '../../api/fetchClient';
import { WATERMARK_ENGINES, normalizeEngine, watermarkEngineStatus } from '../../utils/watermarkEngine.js';

/* Which engine 🚩 Find watermarks runs — ONE control, mounted in BOTH scan
 * windows (dataset dialog and bank panel), write-through persisted exactly like
 * the threshold beside it: `watermark_detect.backend` is what the scan routes
 * have always read, so saving IS arming, on both surfaces at once.
 *
 * The status line under the select is the honest half: it names the route a
 * scan launched NOW will take (mirroring the backend resolver), names the
 * vision model that would run, and — for the one amber case, detector pinned
 * but missing — says the run still happens on the vision route and where the
 * detector installs from. A selector that changed a value without saying what
 * it changes would be the same dead key this replaces.
 */
export default function WatermarkEngineChoice({ caps = {}, disabled = false, onChanged }) {
  const [value, setValue] = useState(() => normalizeEngine(caps.watermark_detect_backend));
  const [saving, setSaving] = useState(false);
  const status = watermarkEngineStatus(value, caps);

  const save = async (next) => {
    const engine = normalizeEngine(next);
    setValue(engine);
    setSaving(true);
    try {
      await putJson('/api/settings', { config: { watermark_detect: { backend: engine } } });
      onChanged?.(engine);
    } catch { /* the scan still uses the stored value; the select shows intent */ }
    setSaving(false);
  };

  return (
    <label className="block text-[11px] text-content-subtle">
      <span className="font-medium text-content">Detection engine</span>
      {' — stored: the other surface reads the same value.'}
      <span className="mt-1 flex flex-wrap items-center gap-2">
        <select value={value} disabled={disabled || saving}
          aria-label="Watermark detection engine"
          onChange={(e) => save(e.target.value)}
          className="rounded border border-border bg-app px-1.5 py-0.5 text-content">
          {WATERMARK_ENGINES.map((e) => (
            <option key={e.id} value={e.id}>{e.label}</option>
          ))}
        </select>
      </span>
      <span className={`mt-1 block leading-snug ${status.warn ? 'text-amber-300' : 'text-content-subtle'}`}>
        {status.line}
      </span>
    </label>
  );
}
