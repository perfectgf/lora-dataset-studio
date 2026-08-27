// « 🌐 Civitai » — one button + its browser modal, self-contained so EVERY
// prompt surface mounts the exact same feature with one line. Mounted by
// PromptField (dataset Test Studio + the canvas “Generate from the board”)
// and by StudioRunSetup (multi-LoRA comparison): generation-surface parity by
// construction, not by duplication.
//
// Overwrite rule matches 🔎 Describe: a typed prompt is never silently
// replaced — picking a Civitai prompt over a non-empty field asks first.
import { useState } from 'react';
import CivitaiBrowserModal from './CivitaiBrowserModal';

export default function CivitaiBrowserButton({ prompt, onPrompt }) {
  const [open, setOpen] = useState(false);
  const use = (text) => {
    if (prompt && prompt.trim()
      && !window.confirm('Replace the current prompt with this Civitai prompt?')) return;
    onPrompt(text);
    setOpen(false);
  };
  return (
    <>
      <button type="button" onClick={() => setOpen(true)}
        title="Browse top Civitai images and reuse their prompts"
        className="px-2 py-0.5 rounded border border-border bg-surface text-content-subtle text-[0.625rem] hover:text-content">
        🌐 Civitai
      </button>
      {open && (
        <CivitaiBrowserModal open onClose={() => setOpen(false)} onUse={use} />
      )}
    </>
  );
}
