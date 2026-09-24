import { performanceSettings } from './videoPerformance.js';
import { useCallback, useEffect, useRef, useState } from 'react';
import { ASPECTS, readReferenceDraft, REFERENCE_DEFAULTS, referenceDescriptors, referencePayload, remapReferencePrompt, writeReferenceDraft } from './videoReferences.js';

export default function useVideoReferences(setPrompt, storageKey) {
  const [draft, setDraft] = useState(() => readReferenceDraft(undefined, storageKey));
  const [staging, setStaging] = useState(false);
  const current = useRef(draft);
  current.current = draft;
  useEffect(() => { writeReferenceDraft(draft, undefined, storageKey); }, [draft, storageKey]);
  const update = useCallback((patch) => setDraft((d) => ({ ...d, ...patch })), []);
  const setReferences = useCallback((next, { remapPrompt = true } = {}) => {
    const before = current.current.references;
    const after = typeof next === 'function' ? next(before) : next;
    if (remapPrompt) setPrompt((p) => remapReferencePrompt(p, before, after));
    current.current = { ...current.current, references: after };
    setDraft((d) => ({ ...d, references: after }));
  }, [setPrompt]);
  const setSettings = useCallback((next) => setDraft((d) => {
    const patch = typeof next === 'function' ? next(d.settings) : next;
    const known = Object.fromEntries(Object.entries(patch).filter(([key]) => key in REFERENCE_DEFAULTS));
    return { ...d, settings: { ...d.settings, ...known } };
  }), []);
  const restore = useCallback((clip) => setDraft((d) => ({ ...d, active: true,
    references: referenceDescriptors(clip.references),
    settings: { ...d.settings, ...performanceSettings(clip.generation_settings), base: clip.ref_base || 'official', accel: clip.accel || '',
      imageSize: clip.ref_image_size || 'match', steps: clip.steps || '', sparse: clip.sparse || '',
      // A JOINED clip's `frames` is the FILE's count (parent + part − 1), not
      // a count the sampler takes: replayed, it renders about twice the clip
      // it reused. The i2v dial has guarded this since 2026-09-03; a reference
      // launch reads THESE settings, not that dial, and this wave is what
      // makes a joined reference clip exist at all (verification, 2026-09-07).
      frames: clip.joined ? d.settings.frames : (clip.frames || d.settings.frames),
      megapixels: clip.megapixels || d.settings.megapixels,
      // 'auto' is what a row that predates the shape carries, and it is not
      // one of the three the select offers: an unknown shape would leave the
      // control showing nothing at all.
      seed: clip.seed ?? '',
      aspect: ASPECTS.includes(clip.aspect) ? clip.aspect : d.settings.aspect },
    // ⏭ …and the clip it continued, so a reused reference continuation
    // continues the same parent — the branch ↻ Reuse offers in From an image
    // (videoStartFrames), not a lookalike that quietly joins nothing.
    firstFrame: clip.source_image
      ? { image: clip.source_image, ...(clip.continues_of ? { continues: clip.continues_of } : {}) } : null,
    endFrame: clip.end_image ? { image: clip.end_image } : null })), []);
  return { ...draft, update, setReferences, setSettings, restore, staging, setStaging,
    signature: JSON.stringify([referencePayload(draft.references), draft.firstFrame?.image, draft.endFrame?.image]) };
}
