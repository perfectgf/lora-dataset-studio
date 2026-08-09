import { forwardRef, useCallback, useState } from 'react';
import { createPortal } from 'react-dom';
import KleinLoraCombobox, { useKleinGenerationLoras } from '../settings/KleinLoraCombobox';
import { findLora } from '../../utils/kleinLoraOptions';
import { clampStrength, MAX_EXTERNAL_LORAS } from '../../utils/externalLoras';
import { normalizeLoraName } from './pluginNodes/registry';

// Strip directories (either separator) and the .safetensors extension — the
// same file a Klein preset row stores, read back as a short label.
function baseName(filename) {
  const parts = String(filename || '').replace(/\\/g, '/').split('/');
  return (parts[parts.length - 1] || filename).replace(/\.safetensors$/i, '');
}

/* 🔌 The 'external-lora' plugin-node type's markup: the board card
   (`ExternalLoraCard`) and the add popover (`ExternalLoraAddFlow`). Both are
   registered onto `PLUGIN_NODE_TYPES['external-lora']` via lazy loaders in
   `pluginNodes/registry.js` — `PluginNodeLayer.jsx` mounts them, drag and
   board-space positioning are its job (`pluginNodes/useBoardDrag.js`), not
   this file's. This file only knows how to render one node and one add
   flow — the same one scan (`useKleinGenerationLoras`) backs both, since
   the card badges and the add-picker read the same LoRA list. */

export const ExternalLoraCard = forwardRef(function ExternalLoraCard(
  { node, dragHandlers, onUpdate, onRemove, checked, onToggleChecked, family = 'zimage' }, ref,
) {
  const { loras } = useKleinGenerationLoras(family);
  const info = findLora(node.filename, loras);
  return (
    <div ref={ref} data-canvas-control
      style={{ position: 'absolute', left: node.x, top: node.y, width: 172 }}
      className="lds-extlora-node overflow-hidden rounded-lg border border-cyan-400/50 bg-surface-overlay/95 shadow-lg backdrop-blur-sm">
      <div
        {...dragHandlers}
        className="flex cursor-grab items-center gap-1 border-b border-cyan-400/30 bg-cyan-500/10 px-1.5 py-1 active:cursor-grabbing">
        <span aria-hidden>🔌</span>
        <span className="min-w-0 flex-1 truncate text-[0.6875rem] font-semibold text-content" title={node.filename}>
          {baseName(node.filename)}
        </span>
        {info && (
          <span className="shrink-0 rounded border border-cyan-400/40 bg-cyan-500/10 px-1 py-px text-[0.5625rem] text-cyan-200">
            {info.label || info.arch}
          </span>
        )}
        <button type="button" onClick={() => onRemove(node.filename)}
          title="Remove this external LoRA from the board"
          aria-label={`Remove ${baseName(node.filename)}`}
          className="shrink-0 text-content-subtle hover:text-red-300">×</button>
      </div>
      <div className="flex items-center gap-1.5 px-1.5 py-1">
        <input type="checkbox" checked={!!checked} onChange={() => onToggleChecked(node.filename)}
          aria-label={`Stack ${baseName(node.filename)} on the next run`} />
        <label className="min-w-0 flex-1 text-[0.625rem] text-content-muted">
          Strength {node.strength}
          <input type="range" min="0" max="2" step="0.05" value={node.strength}
            onChange={(e) => onUpdate(node.filename, { strength: clampStrength(e.target.value) })}
            aria-label={`Strength for ${baseName(node.filename)}`}
            className="mt-0.5 block w-full accent-cyan-400" />
        </label>
      </div>
    </div>
  );
});

export function ExternalLoraAddFlow({ nodes = [], onNodesChange, family = 'zimage', onClose }) {
  const { loras, loading, error, rescan, rescanning } = useKleinGenerationLoras(family);
  const [pickText, setPickText] = useState('');

  const addNode = useCallback(() => {
    const filename = pickText.trim();
    if (!filename || nodes.length >= MAX_EXTERNAL_LORAS) return;
    // Dedupe on the same NORMALIZED name `nodeKey` keys the board with — a
    // separator or case difference from the picker (`Krea\Foo.safetensors`
    // vs `krea/foo.safetensors`) must not let both onto the board, or they
    // collide on the geometry/edge map the moment they're both there.
    if (nodes.some((n) => normalizeLoraName(n.filename) === normalizeLoraName(filename))) {
      setPickText(''); onClose?.(); return;
    }
    const i = nodes.length;
    onNodesChange([...nodes, { filename, strength: 1, x: 16 + 24 * i, y: 16 + 24 * i }]);
    setPickText('');
    onClose?.();
  }, [pickText, nodes, onNodesChange, onClose]);

  if (typeof document === 'undefined') return null;
  return createPortal(
    <div className="fixed right-2 top-16 z-50 w-[min(20rem,calc(100vw-1rem))] rounded-lg border border-border bg-surface-overlay p-2 shadow-xl">
      <div className="mb-1.5 flex items-center justify-between">
        <span className="text-[0.6875rem] font-semibold text-content">
          <span aria-hidden>🔌</span> Add an external LoRA
        </span>
        <button type="button" onClick={() => onClose?.()} aria-label="Close"
          className="text-content-subtle hover:text-content">×</button>
      </div>
      <KleinLoraCombobox value={pickText} onChange={setPickText}
        ariaLabel="External LoRA file" loras={loras} loading={loading} error={error}
        rescan={rescan} rescanning={rescanning} engineLabel="Board"
        placeholder="path/to/lora.safetensors" />
      <button type="button" onClick={addNode}
        disabled={!pickText.trim() || nodes.length >= MAX_EXTERNAL_LORAS}
        className="mt-1.5 w-full rounded-md border border-cyan-400/50 bg-cyan-500/10 px-2 py-1 text-[0.6875rem] font-semibold text-cyan-100 disabled:opacity-40">
        Add to board
      </button>
      {nodes.length >= MAX_EXTERNAL_LORAS && (
        <p className="mt-1 text-[0.625rem] text-amber-300">
          Limit of {MAX_EXTERNAL_LORAS} external LoRAs reached — remove one to add another.
        </p>
      )}
    </div>,
    document.body,
  );
}
