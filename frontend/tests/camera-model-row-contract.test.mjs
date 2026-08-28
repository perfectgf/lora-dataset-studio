/* 📷 The Model row of the camera picker — the wiring that must not drift.

   Source-read, like every layout contract here: node --test parses no JSX, so
   these assertions pin what is WRITTEN. What they protect:

     * the row edits the app-wide `camera.unet` through the same widget, the
       same scan and the same PUT that Settings uses — a second mechanism would
       let the two screens disagree about the same UNETLoader;
     * the slot it lists from exists on the backend, as a diffusion-models
       slot — a renamed slot degrades the combobox to free text silently;
     * keys pressed inside the picker stay inside the picker. The combobox is
       the first TEXT INPUT this dialog ever carried, and it mounts under two
       window-keydown hosts: without the stops, a caret ← → walks the Gallery
       behind the dial, a letter is a K/R/S verdict in the dataset lightbox,
       and Escape closes two layers in one press. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';

const read = (p) => fs.readFileSync(path.join(process.cwd(), p), 'utf8');
const picker = read('src/components/shared/CameraAnglePicker.jsx');

test('the Model row edits the app-wide camera.unet through the shared widget', () => {
  assert.match(picker, /GlobalModelPicker/,
    'the row must reuse the work-screen global picker, not grow a second one');
  assert.match(picker, /section="camera" field="unet" slot="camera_unet"/,
    'the config key and the slot are load-bearing: Settings and this row must save the same value');
  assert.match(picker, /\/api\/camera\/catalog/,
    'the current value comes from the catalog — neither host owns config');
});

test('the global picker PUTs inside the config envelope the endpoint reads', () => {
  // PUT /api/settings reads ONLY body.config / body.secrets and answers 200 to
  // anything else — an unwrapped section is a silent no-op that the optimistic
  // onSaved paints as saved. GlobalModelPicker shipped exactly that way (the
  // Krea 2 base-model row never persisted); this pin is the regression guard.
  const gmp = read('src/components/shared/GlobalModelPicker.jsx');
  assert.match(gmp, /putJson\('\/api\/settings', \{ config: \{ \[section\]/,
    'GlobalModelPicker must wrap its patch in { config: … }');
});

test('the slot the row lists from exists on the backend, as a diffusion-models slot', () => {
  const py = fs.readFileSync(
    path.join(process.cwd(), '..', 'backend', 'app', 'services', 'comfy_model_picker.py'),
    'utf8');
  assert.match(py, /'camera_unet':\s*\('diffusion_models'/,
    'comfy_model_picker.SLOTS must carry camera_unet or the combobox silently degrades to free text');
});

test('keys inside the picker stay inside the picker', () => {
  assert.match(picker,
    /onKeyDown=\{\(e\) => \{\s*e\.stopPropagation\(\);\s*if \(e\.key === 'Escape'\) onClose\?\.\(\);/,
    'the dialog root must stop keydown and honour Escape itself');
});

test('focus moves into the dialog on open', () => {
  // Without it, focus stays on the 📷 button underneath, and the Gallery host
  // walks ← → on window while the run still shoots the original picture.
  assert.match(picker, /useEffect\(\(\) => \{ closeRef\.current\?\.focus\(\); \}, \[\]\);/);
});

test('Escape with the dropdown open closes only the dropdown', () => {
  const combo = read('src/components/settings/ModelFilePicker.jsx');
  assert.match(combo, /e\.preventDefault\(\); e\.stopPropagation\(\); setOpen\(false\)/,
    'the combobox must stop the press that only closes its dropdown — one Escape, one layer');
});
