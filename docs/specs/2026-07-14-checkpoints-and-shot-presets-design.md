# Checkpoints stage and custom shot presets

Date: 2026-07-14

## Scope

This change contains two workspace improvements:

1. Promote "Checkpoints & trained LoRAs" from a Training subpanel to a first-class workspace section between Training and Studio.
2. Let users save the current variation-shot selection as a named custom preset.

No training, generation, checkpoint, or Studio backend behavior changes.

## Checkpoints & LoRAs section

- Add a checkpoints workspace section after training.
- Move the existing checkpoint/LoRA presentation into that section without changing its actions or state.
- The section contains refresh, folder shortcuts, best-epoch scoring, continue training, checkpoint import/deletion/cleanup, and installed-LoRA deletion.
- Training keeps launch, live progress, advanced options, and queue controls.
- Studio remains the final step.
- Deep links use ?section=checkpoints; old training checkpoint links normalize to the new section.
- The progress checklist becomes Train -> Checkpoints & LoRAs -> Studio. The checkpoint step shows checkpoint count and completes when at least one checkpoint or imported LoRA exists.

## Custom shot presets

- Add a "Save preset" action near the built-in preset cards.
- Saving prompts for a non-empty name and snapshots the current selection.
- A snapshot stores built-in shot IDs plus full definitions of selected custom shots, so it remains reusable even if a custom shot is later removed.
- Custom presets are stored in localStorage and shared across datasets in the same browser, matching custom-shot persistence.
- Custom preset cards use the same composition bar and selected-state treatment as built-ins, with a clear custom marker.
- Clicking a custom preset applies its complete selection. Clicking the active preset clears the selection, matching built-in behavior.
- Applying a preset restores any embedded custom shots that no longer exist.
- A compact menu allows rename and delete. Built-in presets remain immutable.
- Duplicate names are rejected case-insensitively. Empty selections and empty names are rejected with clear feedback.
- Corrupt stored data is ignored safely without breaking the catalog.

## Accessibility and responsive behavior

- All new actions have explicit accessible names and keyboard focus states.
- Menus and cards remain usable at mobile width without horizontal overflow.
- Status is communicated by text or glyph as well as color.

## Tests

- Unit tests cover section ordering, deep-link normalization, checkpoint availability, preset save/apply/toggle, embedded custom-shot restoration, rename/delete, duplicate names, and corrupt localStorage.
- Frontend build must pass.
- Visual checks cover desktop and mobile workspace navigation, the standalone checkpoint section, and the custom-preset controls.
