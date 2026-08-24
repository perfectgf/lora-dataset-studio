// react-frontend/src/components/dataset/useSliderTraining.js
// The training panel's Slider LoRA (Beta) cluster — server state, blur-saved
// text drafts, the mode toggle with its label refresh — moved VERBATIM from
// TrainingPanel.jsx (2026-08-24, hook series wave 2). The panel passes its
// transport and the setters the toggle's refresh writes; everything it needs
// is declared before the call site, so no lazy seam is required.
import { useEffect, useState } from 'react';

export function useSliderTraining({
  ds, postTrain, toastTrainError, base, trainType, variant,
  setBaseInfo, setAdv, setStepsInfo,
}) {
  // Slider LoRA mode (Beta) : état serveur (colonne dédiée train_slider) + brouillon
  // local des champs texte (édition libre, sauvés au blur comme les sample prompts).
  const [slider, setSlider] = useState(null);
  const [sliderBusy, setSliderBusy] = useState(false);
  const [sliderDraft, setSliderDraft] = useState({ positive: '', negative: '', target_class: '', anchor: '' });

  // Re-seed the slider text drafts from the canonical stored values whenever the
  // server state (re)loads — saves happen on blur, so no mid-typing overwrite.
  useEffect(() => {
    setSliderDraft({
      positive: slider?.positive ?? '',
      negative: slider?.negative ?? '',
      target_class: slider?.target_class ?? '',
      anchor: slider?.anchor ?? '',
    });
  }, [slider?.positive, slider?.negative, slider?.target_class, slider?.anchor]);

  // --- Slider LoRA mode (Beta) ------------------------------------------------
  const sliderOn = !!slider?.enabled;
  const sliderPromptsMissing = sliderOn
    && (!(slider?.positive || '').trim() || !(slider?.negative || '').trim());
  const saveSlider = async (patch) => {
    setSliderBusy(true);
    try {
      const d = await postTrain(`/api/dataset/${ds.currentId}/train/slider`, patch);
      if (d.ok === false) { toastTrainError(d, 'Slider settings save failed'); return null; }
      setSlider(d.slider);
      return d.slider;
    } finally {
      setSliderBusy(false);
    }
  };

  const toggleSliderMode = async () => {
    const next = !sliderOn;
    const saved = await saveSlider({ enabled: next });
    if (!saved) return;
    // Rank default (8 in slider mode) and the step policy both live server-side —
    // refresh base-info + the checkpoint/steps panel so labels stay truthful.
    try {
      const info = await ds.trainBaseInfo?.();
      if (info) { setBaseInfo(info); setAdv(info.train_settings || null); }
      const checkpointData = await ds.listCheckpoints?.(base, trainType, variant);
      if (checkpointData) setStepsInfo(checkpointData.recommended_steps_info || null);
    } catch { /* labels refresh is best-effort */ }
  };
  const saveSliderField = (key) => () => {
    const stored = slider?.[key] ?? '';
    if ((sliderDraft[key] ?? '') === stored) return;   // no-op → skip round-trip
    saveSlider({ [key]: sliderDraft[key] });
  };
  return {
    slider, setSlider, sliderBusy, sliderDraft, setSliderDraft, sliderOn,
    sliderPromptsMissing, saveSlider, toggleSliderMode, saveSliderField,
  };
}
