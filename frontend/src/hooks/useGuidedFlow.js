import { useMemo } from 'react';

/* Pure derivation of the guided path from the dataset payload + capabilities.
   No fetching here — lives at the workspace's existing poll rhythm. */
export function deriveSteps(d, caps, checkpointCount = 0) {
  const images = (d && d.images) || [];
  const live = images.filter((i) => i.status !== 'failed');
  const kept = live.filter((i) => i.status === 'keep');
  const triage = live.filter((i) => i.status === 'pending' && i.filename);   // generated, awaiting keep/reject
  const generating = live.filter((i) => i.status === 'pending' && !i.filename);
  const captioned = kept.filter((i) => (i.caption || '').trim());
  const scored = kept.filter((i) => i.face_state);
  const trainMode = !!(caps && caps.training_visible);

  const steps = [
    { id: 'reference', label: 'Reference', targetId: 'gf-reference',
      done: !!(d && d.ref_filename), subtitle: d && d.ref_filename ? 'set' : 'one clear photo' },
    { id: 'generate', label: 'Generate', targetId: 'gf-generate',
      done: live.length > 0, subtitle: `${live.length}/25`, busy: generating.length > 0 },
    { id: 'curate', label: 'Curate', targetId: 'gf-images',
      done: live.length > 0 && triage.length === 0 && kept.length > 0,
      subtitle: triage.length ? `${triage.length} to triage` : `${kept.length} kept` },
    { id: 'caption', label: 'Caption', targetId: 'gf-captions',
      done: kept.length > 0 && captioned.length === kept.length,
      subtitle: `${captioned.length}/${kept.length} captioned` },
  ];
  if (caps && caps.face_scoring) {
    steps.push({ id: 'score', label: 'Score', targetId: 'gf-curation', optional: true,
      done: kept.length > 0 && scored.length === kept.length, subtitle: 'optional' });
  }
  steps.push(trainMode
    ? { id: 'finish', label: 'Train', targetId: 'gf-training',
        done: checkpointCount > 0, subtitle: checkpointCount ? `${checkpointCount} checkpoint(s)` : '' }
    : { id: 'finish', label: 'Export', targetId: 'gf-export', done: false, subtitle: 'ZIP',
        unavailable: false });
  steps.push({ id: 'studio', label: 'Studio', targetId: 'gf-training',
    done: !!(d && d.best_settings),
    unavailable: !(caps && caps.studio_visible),
    hint: caps && caps.studio_visible ? '' : 'Configure ComfyUI in Settings' });

  const nextStep = steps.find((s) => !s.done && !s.optional && !s.unavailable) || null;
  return { steps, nextStep };
}

export default function useGuidedFlow(d, caps, checkpointCount = 0) {
  return useMemo(() => deriveSteps(d, caps, checkpointCount), [d, caps, checkpointCount]);
}
