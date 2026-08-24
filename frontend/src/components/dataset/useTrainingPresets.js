// react-frontend/src/components/dataset/useTrainingPresets.js
// The training panel's presets cluster (state + save / apply / import /
// export / delete), moved VERBATIM from TrainingPanel.jsx (2026-08-24,
// hook series wave 1). The panel passes its own transport (postTrain,
// toastTrainError) and an onApplied callback carrying the panel-state
// effects of a successful apply.
import { useEffect, useRef, useState } from 'react';
import { del } from '../../api/fetchClient';
import {
  compatibleTrainingPresetSelection,
  filterTrainingPresets,
  trainingPresetApplyPayload,
  trainingPresetDatasetKind,
  trainingPresetSnapshotScope,
} from '../../utils/trainingPresets';

export function useTrainingPresets({
  ds, kind, trainType, variant, trainTypeBusy, toast, postTrain,
  toastTrainError, onApplied,
}) {
  // Presets de réglages avancés : snapshots nommés, partageables (fichier JSON).
  // Stockés bruts côté serveur ; la validation se fait à l'APPLICATION (clés
  // inconnues ignorées, valeurs invalides signalées) → tolérant aux versions.
  const [presets, setPresets] = useState([]);
  const [presetSel, setPresetSel] = useState('');
  const [presetBusy, setPresetBusy] = useState(false);
  const presetFileRef = useRef(null);
  // --- Presets (save / apply / import / export / delete) ---------------------
  const presetContext = { trainType, datasetKind: kind, variant };
  const loadPresets = async (preferredSelection) => {
    try {
      const r = await fetch('/api/train/presets', { credentials: 'include' });
      if (r.ok) {
        const list = (await r.json()).presets || [];
        setPresets(list);
        setPresetSel((current) => compatibleTrainingPresetSelection(
          preferredSelection === undefined ? current : preferredSelection,
          list,
          presetContext,
        ));
        return list;
      }
    } catch { /* list is best-effort */ }
    return [];
  };
  useEffect(() => { loadPresets(); }, []);
  const visiblePresets = filterTrainingPresets(presets, presetContext);
  // Built-ins can be evidence-backed general recipes or deliberately narrow,
  // source-labelled community starters. Keep those labels separate so a reported
  // result is never silently upgraded to a researched guarantee in the picker.
  const researchedBuiltins = visiblePresets.filter((p) => p.builtin && !p.community);
  const communityBuiltins = visiblePresets.filter((p) => p.builtin && p.community);
  const selPreset = visiblePresets.find((p) => String(p.id) === presetSel) || null;
  useEffect(() => {
    setPresetSel((current) => compatibleTrainingPresetSelection(
      current, presets, { trainType, datasetKind: kind, variant },
    ));
  }, [presets, trainType, kind, variant]);
  const savePreset = async () => {
    const name = window.prompt('Preset name (an existing name is overwritten):');
    if (!name || !name.trim()) return;
    setPresetBusy(true);
    try {
      const d = await postTrain('/api/train/presets',
        { name: name.trim(), dataset_id: ds.currentId,
          ...trainingPresetSnapshotScope(presetContext) });
      if (d.ok === false) return toastTrainError(d, 'Preset save failed');
      toast.success(`Preset “${name.trim()}” saved.`);
      await loadPresets(d.id);
    } finally {
      setPresetBusy(false);
    }
  };
  const applyPreset = async () => {
    if (!selPreset || presetBusy || trainTypeBusy) return;
    // Every preset — built-in or user-created — is resolved by id on the server.
    // A null plan means the selection became incompatible between render/click;
    // importantly, no request is sent in that case.
    const payload = trainingPresetApplyPayload(selPreset, presetContext);
    if (!payload) {
      setPresetSel('');
      toast.error('This preset does not match the current model family or dataset kind.');
      return;
    }
    setPresetBusy(true);
    try {
      const d = await postTrain(`/api/dataset/${ds.currentId}/train/presets/apply`, payload);
      if (d.ok === false) return toastTrainError(d, 'Preset apply failed');
      await onApplied(d, payload, selPreset);
      const notes = [];
      if (d.ignored?.length) notes.push(`unknown here, ignored: ${d.ignored.join(', ')}`);
      if (d.rejected?.length) notes.push(`rejected: ${d.rejected.map((r) => r.key).join(', ')}`);
      if (notes.length) toast.warning(`Preset applied — ${notes.join(' · ')}`);
      else toast.success(`Preset “${selPreset.name}” applied.`);
    } finally {
      setPresetBusy(false);
    }
  };
  const exportPreset = () => {
    if (!selPreset) return;
    const blob = new Blob([JSON.stringify({
      app: 'lora-dataset-studio', kind: 'training-preset', version: 1,
      name: selPreset.name, train_type: selPreset.train_type,
      dataset_kind: trainingPresetDatasetKind(selPreset) || kind,
      variants: Array.isArray(selPreset.variants) ? selPreset.variants : [],
      settings: selPreset.settings,
    }, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `lds-training-preset-${selPreset.name.replace(/[^\w.-]+/g, '_')}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  const importPreset = async (file) => {
    try {
      const j = JSON.parse(await file.text());
      if (j?.kind !== 'training-preset' || !j.name || typeof j.settings !== 'object' || !j.settings) {
        toast.error('Not a training-preset file (expected kind: "training-preset").');
        return;
      }
      setPresetBusy(true);
      const importedMeta = {
        ...j,
        train_type: j.train_type || trainType,
        dataset_kind: j.dataset_kind || kind,
      };
      const compatibleHere = filterTrainingPresets([importedMeta], presetContext).length === 1;
      const d = await postTrain('/api/train/presets',
        { name: String(j.name), train_type: importedMeta.train_type,
          dataset_kind: importedMeta.dataset_kind,
          variants: Array.isArray(importedMeta.variants) ? importedMeta.variants : [],
          settings: j.settings });
      if (d.ok === false) return toastTrainError(d, 'Preset import failed');
      await loadPresets(compatibleHere ? d.id : '');
      if (compatibleHere) toast.success(`Preset “${j.name}” imported and selected — review, then Apply.`);
      else toast.warning(`Preset “${j.name}” imported for ${importedMeta.train_type}/${importedMeta.dataset_kind}; it is hidden here because the current dataset is ${trainType}/${kind}.`);
    } catch {
      toast.error('Unreadable preset file.');
    } finally {
      setPresetBusy(false);
    }
  };
  const deletePreset = async () => {
    if (!selPreset || selPreset.builtin) return;   // built-ins ship with the app
    if (!window.confirm(`Delete the preset “${selPreset.name}”?`)) return;
    setPresetBusy(true);
    try {
      // del() rides the shared client: same CSRF expiry recovery as every
      // other mutation — a stale token retries once instead of failing.
      await del(`/api/train/presets/${selPreset.id}`);
      setPresetSel('');
      await loadPresets('');
    } catch { toast.error('Could not delete the preset.'); }
    finally { setPresetBusy(false); }
  };
  return {
    presetSel, setPresetSel, presetBusy, presetFileRef, visiblePresets,
    researchedBuiltins, communityBuiltins, selPreset, savePreset,
    applyPreset, exportPreset, importPreset, deletePreset,
  };
}
