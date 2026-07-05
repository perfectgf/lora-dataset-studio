/**
 * Test Studio page — routes /studio (standalone) and /dataset/studio/:id
 * (legacy, pre-filled with a dataset).
 *
 * Reads the dataset id from the URL (`:id` param) OR the `?dataset=` query
 * param and passes it as `preselectDataset` to StudioShell: a blank page if
 * neither is set, otherwise that LoRA is pre-checked in the picker.
 *
 * Gated on `caps.studio_visible` (ComfyUI reachable): the nav link already
 * hides the entry, but this guards direct URL access too.
 */
import { useParams, useSearchParams } from 'react-router-dom';
import { useCapabilities } from '../context/CapabilitiesContext';
import StudioShell from '../components/dataset/studio/StudioShell';

export default function StudioPage() {
  const { id } = useParams();
  const [sp] = useSearchParams();
  const { caps } = useCapabilities();
  // /dataset/studio/:id (legacy), or /studio?dataset=… (launcher), or nothing (standalone).
  const preselectDataset = id || sp.get('dataset') || null;

  if (!caps.studio_visible) {
    return (
      <div className="rounded-xl border border-border bg-surface p-8 text-center">
        <h1 className="text-lg font-semibold text-content">Test Studio</h1>
        <p className="mt-2 text-sm text-content-muted">
          Test Studio requires ComfyUI — configure it in Settings.
        </p>
      </div>
    );
  }

  // pb-24: StudioActionBar is a fixed bottom bar (Run button + section shortcuts) —
  // leaves room so it never covers the last row of results.
  return (
    <div className="pb-24">
      <StudioShell preselectDataset={preselectDataset} />
    </div>
  );
}
