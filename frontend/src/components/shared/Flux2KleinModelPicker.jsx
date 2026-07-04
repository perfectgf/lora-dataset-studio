import { useEffect, useState } from 'react';
import { useCapabilities } from '../../context/CapabilitiesContext';

const STORAGE_KEY = 'editPage_flux2KleinModel_v1';

/**
 * Base diffusion model picker for the Klein image edit pipeline.
 *
 * Renders nothing when fewer than 2 models are available (no choice to make).
 * The selected filename is persisted to localStorage and reported to the
 * parent via `onChange` so it can be forwarded to the backend at submit time.
 *
 * Model list is sourced from `caps.comfyui.models.klein` (the capabilities
 * probe already scans ComfyUI's unet/klein folder) rather than a dedicated
 * endpoint — this app has no `/api/flux2_klein_models` route.
 */
export default function Flux2KleinModelPicker({ onChange }) {
  const { caps } = useCapabilities();
  const models = caps.comfyui.models.klein || [];
  const [selected, setSelected] = useState(() => {
    try { return localStorage.getItem(STORAGE_KEY) || ''; } catch { return ''; }
  });

  useEffect(() => {
    // Reconcile persisted choice with the current available list.
    const valid = models.includes(selected) ? selected : (models[0] || '');
    if (valid !== selected) setSelected(valid);
    onChange?.(valid || null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models.join('|')]);

  function handleChange(e) {
    const next = e.target.value;
    setSelected(next);
    try { localStorage.setItem(STORAGE_KEY, next); } catch { /* quota / private mode */ }
    onChange?.(next || null);
  }

  if (models.length < 2) return null;

  return (
    <div className="flex flex-col gap-1">
      <label className="text-content-muted text-sm font-medium" htmlFor="flux2-klein-model">
        Base model
      </label>
      <select
        id="flux2-klein-model"
        value={selected}
        onChange={handleChange}
        className="w-full bg-white/[0.03] border border-white/10 rounded-lg px-3 py-2 text-sm text-content focus:outline-none focus:border-primary/60"
      >
        {models.map((m) => (
          <option key={m} value={m} className="bg-surface-overlay">
            {m}
          </option>
        ))}
      </select>
    </div>
  );
}
