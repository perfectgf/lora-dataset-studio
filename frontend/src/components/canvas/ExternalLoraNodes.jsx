import { useCallback, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import KleinLoraCombobox, { useKleinGenerationLoras } from '../settings/KleinLoraCombobox';
import { findLora } from '../../utils/kleinLoraOptions';
import { clampStrength, MAX_EXTERNAL_LORAS } from '../../utils/externalLoras';

// Strip directories (either separator) and the .safetensors extension — the
// same file a Klein preset row stores, read back as a short label.
function baseName(filename) {
  const parts = String(filename || '').replace(/\\/g, '/').split('/');
  return (parts[parts.length - 1] || filename).replace(/\.safetensors$/i, '');
}

/* 🔌 The board's external LoRA plugin nodes.
   Two independent pieces share this file because they share the one scan
   (useKleinGenerationLoras) that both the node badges and the add-picker read:
     - node cards, absolutely positioned — meant to be rendered by the parent
       INSIDE its pan/zoom transformed layer, so they track the board;
     - the add popover, portalled to <body> so the board's transform (which
       would otherwise become the containing block for anything `position:
       fixed` under it) never touches it. Opened by the parent's toolbar
       button via `pickerOpen`/`onClosePicker`. */
export default function ExternalLoraNodes({
  nodes = [], onNodesChange, checked, onCheckedChange, family = 'zimage',
  boardScale = 1, pickerOpen = false, onClosePicker,
}) {
  const { loras, loading, error, rescan, rescanning } = useKleinGenerationLoras(family);
  const [pickText, setPickText] = useState('');

  const updateNode = useCallback((filename, patch) => {
    onNodesChange(nodes.map((n) => (n.filename === filename ? { ...n, ...patch } : n)));
  }, [nodes, onNodesChange]);

  const removeNode = useCallback((filename) => {
    onNodesChange(nodes.filter((n) => n.filename !== filename));
    if (checked?.has(filename)) {
      const next = new Set(checked);
      next.delete(filename);
      onCheckedChange(next);
    }
  }, [nodes, onNodesChange, checked, onCheckedChange]);

  const toggleChecked = useCallback((filename) => {
    const next = new Set(checked || []);
    if (next.has(filename)) next.delete(filename); else next.add(filename);
    onCheckedChange(next);
  }, [checked, onCheckedChange]);

  const addNode = useCallback(() => {
    const filename = pickText.trim();
    if (!filename || nodes.length >= MAX_EXTERNAL_LORAS) return;
    if (nodes.some((n) => n.filename === filename)) { setPickText(''); onClosePicker?.(); return; }
    const i = nodes.length;
    onNodesChange([...nodes, { filename, strength: 1, x: 16 + 24 * i, y: 16 + 24 * i }]);
    setPickText('');
    onClosePicker?.();
  }, [pickText, nodes, onNodesChange, onClosePicker]);

  // Drag: pointer-down on a card header captures the pointer on itself and
  // stops propagation, so the board's own frame never sees it (no pan, no
  // card-drag armed). Moves update a transient overlay position; pointer-up
  // commits the final x/y to `nodes` in one call.
  const dragRef = useRef(null);
  const [dragPos, setDragPos] = useState(null);
  const onHeaderDown = useCallback((e, node) => {
    if (e.button != null && e.button !== 0) return;
    // The × button lives inside this same header row — a press on it must
    // close the node, not arm a drag that steals its pointerup/click.
    if (e.target.closest?.('button')) return;
    e.stopPropagation();
    e.currentTarget.setPointerCapture?.(e.pointerId);
    dragRef.current = { filename: node.filename, sx: e.clientX, sy: e.clientY, ox: node.x, oy: node.y };
  }, []);
  const onHeaderMove = useCallback((e) => {
    const d = dragRef.current;
    if (!d) return;
    const k = boardScale || 1;
    setDragPos({ filename: d.filename, x: d.ox + (e.clientX - d.sx) / k, y: d.oy + (e.clientY - d.sy) / k });
  }, [boardScale]);
  const onHeaderUp = useCallback((e) => {
    const d = dragRef.current;
    dragRef.current = null;
    if (!d) return;
    const k = boardScale || 1;
    setDragPos(null);
    updateNode(d.filename, { x: d.ox + (e.clientX - d.sx) / k, y: d.oy + (e.clientY - d.sy) / k });
  }, [boardScale, updateNode]);

  return (
    <>
      {nodes.map((n) => {
        const pos = dragPos && dragPos.filename === n.filename ? dragPos : n;
        const info = findLora(n.filename, loras);
        const isChecked = !!checked?.has(n.filename);
        return (
          <div key={n.filename} data-canvas-control
            style={{ position: 'absolute', left: pos.x, top: pos.y, width: 172 }}
            className="lds-extlora-node overflow-hidden rounded-lg border border-cyan-400/50 bg-surface-overlay/95 shadow-lg backdrop-blur-sm">
            <div
              onPointerDown={(e) => onHeaderDown(e, n)}
              onPointerMove={onHeaderMove}
              onPointerUp={onHeaderUp}
              onPointerCancel={onHeaderUp}
              className="flex cursor-grab items-center gap-1 border-b border-cyan-400/30 bg-cyan-500/10 px-1.5 py-1 active:cursor-grabbing">
              <span aria-hidden>🔌</span>
              <span className="min-w-0 flex-1 truncate text-[0.6875rem] font-semibold text-content" title={n.filename}>
                {baseName(n.filename)}
              </span>
              {info && (
                <span className="shrink-0 rounded border border-cyan-400/40 bg-cyan-500/10 px-1 py-px text-[0.5625rem] text-cyan-200">
                  {info.label || info.arch}
                </span>
              )}
              <button type="button" onClick={() => removeNode(n.filename)}
                title="Remove this external LoRA from the board"
                aria-label={`Remove ${baseName(n.filename)}`}
                className="shrink-0 text-content-subtle hover:text-red-300">×</button>
            </div>
            <div className="flex items-center gap-1.5 px-1.5 py-1">
              <input type="checkbox" checked={isChecked} onChange={() => toggleChecked(n.filename)}
                aria-label={`Stack ${baseName(n.filename)} on the next run`} />
              <label className="min-w-0 flex-1 text-[0.625rem] text-content-muted">
                Strength {n.strength}
                <input type="range" min="0" max="2" step="0.05" value={n.strength}
                  onChange={(e) => updateNode(n.filename, { strength: clampStrength(e.target.value) })}
                  aria-label={`Strength for ${baseName(n.filename)}`}
                  className="mt-0.5 block w-full accent-cyan-400" />
              </label>
            </div>
          </div>
        );
      })}

      {pickerOpen && typeof document !== 'undefined' && createPortal(
        <div className="fixed right-2 top-16 z-50 w-[min(20rem,calc(100vw-1rem))] rounded-lg border border-border bg-surface-overlay p-2 shadow-xl">
          <div className="mb-1.5 flex items-center justify-between">
            <span className="text-[0.6875rem] font-semibold text-content">
              <span aria-hidden>🔌</span> Add an external LoRA
            </span>
            <button type="button" onClick={() => onClosePicker?.()} aria-label="Close"
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
      )}
    </>
  );
}
