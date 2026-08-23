/* Preview steps / CFG (GitHub #46) — the two surfaces that can offer them.

   The controls are two number boxes, which is the boring half. The half worth a
   test is what EMPTY means: it is not "zero" and not "unset in the UI only", it
   is "follow the family default", and the default itself is a server fact
   because it depends on the base AND the variant (8 steps at CFG 1 distilled,
   25/4 not). Two mistakes would each be invisible on screen and wrong in the
   job config:

     - seeding the box from the EFFECTIVE value instead of the stored override,
       which turns "following the default" into a frozen copy of today's number
       the day the default improves;
     - hardcoding the placeholder, which makes the panel announce a number the
       run does not send.

   Both are cheap to assert on the source and expensive to notice in a run. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const panel = fs.readFileSync(new URL('./TrainingPanel.jsx', import.meta.url), 'utf8');
const dialog = fs.readFileSync(new URL('./ContinueDialog.jsx', import.meta.url), 'utf8');

test('the panel offers both boxes with their aria labels', () => {
  assert.match(panel, /aria-label="Preview steps"/);
  assert.match(panel, /aria-label="Preview guidance scale"/);
  assert.match(panel, /Preview quality/);
});

test('the panel seeds from the STORED override, never from the effective value', () => {
  // `sample_steps` (effective) exists in the payload too — seeding from it would
  // fill the box and silently persist a copy of the family default.
  assert.match(panel, /setSampleStepsDraft\(adv\?\.sample_steps_stored == null \? ''/);
  assert.match(panel, /setSampleGuidanceDraft\(\s*adv\?\.sample_guidance_stored == null \? ''/);
});

test('the placeholder and the bounds come from the server, not from constants here', () => {
  assert.match(panel, /adv\?\.sample_steps_default/);
  assert.match(panel, /adv\?\.sample_guidance_default/);
  assert.match(panel, /adv\?\.sample_steps_range/);
  assert.match(panel, /adv\?\.sample_guidance_range/);
});

test('a blank box saves null — the documented way back to the family default', () => {
  assert.match(panel, /if \(!draft\.trim\(\)\) \{\s*\n\s*saveAdv\(\{ \[key\]: null \}\);/);
});

test('continuing a run can change them, and is NOT gated by the trajectory lock', () => {
  // save_every / sample_every / timestep / lr_factor all carry `!trajectoryLocked`
  // because they would change the trajectory a full-state bundle belongs to.
  // These two render a picture, so gating them would make a resume the one place
  // an unreadable preview cannot be fixed.
  assert.match(dialog, /overrides\.sample_steps = Number\(sampleSteps\)/);
  assert.match(dialog, /overrides\.sample_guidance = Number\(sampleGuidance\)/);
  const block = dialog.slice(dialog.indexOf('if (sampleSteps.trim()'),
    dialog.indexOf('overrides.sample_guidance = Number(sampleGuidance)'));
  assert.doesNotMatch(block, /trajectoryLocked/);
});

test('the dialog boxes default to keeping what the run already uses', () => {
  assert.match(dialog, /const \[sampleSteps, setSampleSteps\] = useState\(''\)/);
  assert.match(dialog, /const \[sampleGuidance, setSampleGuidance\] = useState\(''\)/);
  assert.match(dialog, /placeholder="keep"/);
});
