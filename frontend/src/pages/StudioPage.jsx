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
import { useState } from 'react';
import { useParams, useSearchParams } from 'react-router';
import { Clapperboard, FlaskConical } from 'lucide-react';
import { useCapabilities } from '../context/CapabilitiesContext';
import StudioShell from '../components/dataset/studio/StudioShell';
import VideoTestStudio from '../components/dataset/studio/video/VideoTestStudio';

/* The two things a LoRA can be, tested in the same place.
 *
 * A tab rather than a second page, because "test the LoRA I just trained" is
 * ONE intention and the medium is a property of the LoRA, not of the errand.
 * The lanes share nothing below this line: an image LoRA and a video LoRA live
 * in different tables, render through different pipelines, and the video one
 * queues a single clip where the image one runs a grid — so each lane is its
 * own subtree, mounted and unmounted whole. That is deliberate: it means the
 * video panel cannot leak state into the studio that was here first.
 *
 * `?lane=video` opens straight on it (What's-new and the help topic both use
 * it); the choice is otherwise remembered, because whoever is testing video
 * LoRAs today will be testing video LoRAs in ten minutes. */
const LANE_KEY = 'lds.studio.lane';

function readLane(param) {
  if (param === 'video' || param === 'image') return param;
  try {
    const saved = window.localStorage.getItem(LANE_KEY);
    if (saved === 'video' || saved === 'image') return saved;
  } catch {
    /* private mode, or storage disabled — the default is a fine answer */
  }
  return 'image';
}

export default function StudioPage() {
  const { id } = useParams();
  const [sp] = useSearchParams();
  const { caps } = useCapabilities();
  // /dataset/studio/:id (legacy), or /studio?dataset=… (launcher), or nothing (standalone).
  const preselectDataset = id || sp.get('dataset') || null;
  const preselectFamily = sp.get('family') || null;
  // `?base=` — the base model to open on, as ComfyUI's loader names it. Sent by
  // the full-model card in Checkpoints & LoRAs: arriving there means "test THIS
  // model", and a full model that is not preselected is a model you have to go
  // and find in a dropdown among every checkpoint on the machine. It also
  // re-seeds CFG/steps, which is the point for an undistilled base — the
  // family's few-step defaults render mush on one.
  const preselectBase = sp.get('base') || null;

  const [lane, setLane] = useState(() => readLane(sp.get('lane')));
  const pickLane = (next) => {
    setLane(next);
    try {
      window.localStorage.setItem(LANE_KEY, next);
    } catch { /* nothing to remember it with — the tab still switches */ }
  };

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
      <div data-probe-panel="studio-lanes"
        className="mb-2 flex rounded-xl border border-border bg-surface p-0.5">
        {[
          { id: 'image', label: 'Images', icon: FlaskConical },
          { id: 'video', label: 'Video', icon: Clapperboard },
        ].map(({ id, label, icon: Icon }) => (
          <button key={id} type="button" onClick={() => pickLane(id)}
            aria-pressed={lane === id}
            className={`flex flex-1 items-center justify-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-semibold min-h-10 lg:min-h-0 ${
              lane === id ? 'bg-accent text-accent-contrast' : 'text-content-muted hover:text-content'}`}>
            <Icon aria-hidden="true" className="h-4 w-4" />{label}
          </button>
        ))}
      </div>
      {lane === 'video' ? (
        <VideoTestStudio />
      ) : (
        <StudioShell preselectDataset={preselectDataset} preselectFamily={preselectFamily}
          preselectBase={preselectBase} />
      )}
    </div>
  );
}
