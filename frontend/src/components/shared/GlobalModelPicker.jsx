/* A model-file setting, editable from the WORK SCREEN, saving the very same
   global key Settings shows.
 *
 * WHY IT IS GLOBAL, STATED PLAINLY. There is one value per engine, not one per
 * dataset: whichever screen you change it from, you change it for every future
 * run. That is the decision, and this component's whole job is to make it
 * legible — a control that changes more than the batch in front of you has to
 * say so, which is the rule the four Krea dials in this same panel already
 * follow (`⚠ These four save straight to your settings`).
 *
 * NOTHING NEW IS COMPOSED HERE. The combobox is `settings/ModelFilePicker`, the
 * list is its `useModelFiles` hook against `/api/comfy/model-files`, and the
 * write is the same `PUT /api/settings` patch the dials use. Same widget, same
 * scan, same key, same endpoint — so "the setting in the dataset and the setting
 * in Settings are exactly the same one" is true by construction rather than by
 * two implementations agreeing.
 */
import { useCallback, useRef } from 'react';
import ModelFilePicker, { useModelFiles } from '../settings/ModelFilePicker';
import { putJson } from '../../api/fetchClient';
import { useToast } from '../common/Toast';

/**
 * @param {object}   p
 * @param {string}   p.section   config section ('krea', 'klein')
 * @param {string}   p.field     config field  ('base_model', 'unet')
 * @param {string}   p.slot      picker slot   ('krea_base_model', 'klein_unet')
 * @param {string}   p.label     what the user calls this model
 * @param {string}   p.value     the CURRENT global value (the caller owns config)
 * @param {Function} p.onSaved   called with the new value once the PUT succeeded
 */
export default function GlobalModelPicker({
  section, field, slot, label, value = '', onSaved,
}) {
  const toast = useToast();
  const files = useModelFiles(slot);
  // The last value we tried to save, so a failed PUT can put the field back
  // rather than leave the screen showing a value the server never took.
  const lastGood = useRef(value);

  const save = useCallback(async (next) => {
    const previous = lastGood.current;
    onSaved?.(next);
    try {
      // The `config:` envelope is load-bearing: PUT /api/settings reads ONLY
      // body.config (and body.secrets) and answers 200 to everything else.
      // This call shipped without it — a silent no-op the optimistic onSaved
      // above then painted as saved until the next reload. Measured, not
      // guessed: picked through the UI, watched the 200, read config.json.
      await putJson('/api/settings', { config: { [section]: { [field]: next || '' } } });
      lastGood.current = next;
    } catch {
      onSaved?.(previous);
      toast.error(`Could not save the ${label} — check Settings › Image engines.`);
    }
  }, [section, field, label, onSaved, toast]);

  return (
    <div className="min-w-0 space-y-1">
      <ModelFilePicker
        id={`work-${section}-${field}`}
        ariaLabel={`${label} (saved for every run)`}
        value={value}
        onChange={save}
        placeholder="Empty = auto-detect"
        folderHint={files.folder}
        files={files.files}
        folder={files.folder}
        loading={files.loading}
        error={files.error}
        rescan={files.rescan}
        rescanning={files.rescanning}
      />
    </div>
  );
}
