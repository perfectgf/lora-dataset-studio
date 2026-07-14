import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import {
  MAX_WATERMARK_REGIONS,
  moveRegion,
  pointToNormalized,
  regionFromPoints,
  removeRegion,
  replaceRegion,
  resizeRegion,
} from '../../utils/watermarkRegions';

const HANDLES = [
  ['nw', 'top left', 'left-0 top-0 -translate-x-1/2 -translate-y-1/2 cursor-nwse-resize'],
  ['ne', 'top right', 'right-0 top-0 translate-x-1/2 -translate-y-1/2 cursor-nesw-resize'],
  ['sw', 'bottom left', 'left-0 bottom-0 -translate-x-1/2 translate-y-1/2 cursor-nesw-resize'],
  ['se', 'bottom right', 'right-0 bottom-0 translate-x-1/2 translate-y-1/2 cursor-nwse-resize'],
];

function cloneRegions(regions) {
  return regions.map((region) => [...region]);
}

function releasePointer(drag) {
  const target = drag?.captureTarget;
  try {
    if (target?.hasPointerCapture?.(drag.pointerId)) target.releasePointerCapture(drag.pointerId);
  } catch { /* capture can already be gone after pointercancel/unmount */ }
}

function regionChanged(before, after) {
  return before.some((value, index) => value !== after[index]);
}

export default function WatermarkRegionEditor({
  src,
  alt,
  regions,
  disabled = false,
  addMode = false,
  onAddModeChange,
  onCommit,
  children,
}) {
  const propRegions = useMemo(
    () => cloneRegions(Array.isArray(regions) ? regions : []),
    [regions],
  );
  const propRegionsKey = JSON.stringify(propRegions);
  const [draftRegions, setDraftRegions] = useState(() => cloneRegions(propRegions));
  const [selectedIndex, setSelectedIndex] = useState(() => (propRegions.length ? 0 : null));
  const imageRef = useRef(null);
  const draftRegionsRef = useRef(draftRegions);
  const dragRef = useRef(null);
  const lastSrcRef = useRef(src);
  const lastPropRegionsKeyRef = useRef(propRegionsKey);
  const onCommitRef = useRef(onCommit);
  const onAddModeChangeRef = useRef(onAddModeChange);
  const countId = useId();

  onCommitRef.current = onCommit;
  onAddModeChangeRef.current = onAddModeChange;

  const updateDraft = useCallback((nextRegions) => {
    draftRegionsRef.current = nextRegions;
    setDraftRegions(nextRegions);
  }, []);

  const abortInteraction = useCallback(() => {
    const drag = dragRef.current;
    if (!drag) return;
    dragRef.current = null;
    releasePointer(drag);
    updateDraft(cloneRegions(drag.startRegions));
    setSelectedIndex(drag.startSelectedIndex);
  }, [updateDraft]);

  // Dataset refreshes replace the region arrays. Keep a live drag isolated from
  // stale renders, but cancel it if genuinely new image/region props arrive.
  useEffect(() => {
    const sourceChanged = lastSrcRef.current !== src;
    const regionsChanged = lastPropRegionsKeyRef.current !== propRegionsKey;
    if (!sourceChanged && !regionsChanged) return;

    abortInteraction();
    lastSrcRef.current = src;
    lastPropRegionsKeyRef.current = propRegionsKey;
    updateDraft(cloneRegions(propRegions));
    setSelectedIndex((current) => {
      if (!propRegions.length) return null;
      if (sourceChanged || current == null) return 0;
      return Math.min(current, propRegions.length - 1);
    });
  }, [abortInteraction, propRegions, propRegionsKey, src, updateDraft]);

  useEffect(() => {
    if (disabled) abortInteraction();
  }, [abortInteraction, disabled]);

  const normalizedPoint = useCallback((event) => {
    const rect = imageRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0 || rect.height <= 0) return null;
    return pointToNormalized(event, rect);
  }, []);

  const emitCommit = useCallback((nextRegions) => {
    const next = cloneRegions(nextRegions);
    updateDraft(next);
    onCommitRef.current?.(cloneRegions(next));
  }, [updateDraft]);

  const beginInteraction = useCallback((event, kind, index = null, corner = null) => {
    if (disabled || dragRef.current) return;
    if (event.pointerType === 'mouse' && event.button !== 0) return;
    const point = normalizedPoint(event);
    if (!point) return;

    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.focus?.({ preventScroll: true });

    const startRegions = cloneRegions(draftRegionsRef.current);
    if (kind === 'add' && startRegions.length >= MAX_WATERMARK_REGIONS) return;

    const startSelectedIndex = selectedIndex;
    let nextIndex = index;
    if (kind === 'add') {
      nextIndex = startRegions.length;
      updateDraft([...startRegions, regionFromPoints(point, point)]);
    }
    setSelectedIndex(nextIndex);

    try { event.currentTarget.setPointerCapture?.(event.pointerId); } catch { /* optional API */ }
    dragRef.current = {
      kind,
      corner,
      index: nextIndex,
      pointerId: event.pointerId,
      captureTarget: event.currentTarget,
      startPoint: point,
      startRegions,
      startSelectedIndex,
      changed: kind === 'add',
    };
  }, [disabled, normalizedPoint, selectedIndex, updateDraft]);

  const moveInteraction = useCallback((event) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const point = normalizedPoint(event);
    if (!point) return;
    event.preventDefault();
    event.stopPropagation();

    if (drag.kind === 'add') {
      const nextRegion = regionFromPoints(drag.startPoint, point);
      drag.changed = true;
      updateDraft([...cloneRegions(drag.startRegions), nextRegion]);
      return;
    }

    const dx = point[0] - drag.startPoint[0];
    const dy = point[1] - drag.startPoint[1];
    const startRegion = drag.startRegions[drag.index];
    const nextRegion = drag.kind === 'move'
      ? moveRegion(startRegion, dx, dy)
      : resizeRegion(startRegion, drag.corner, dx, dy);
    drag.changed = regionChanged(startRegion, nextRegion);
    updateDraft(replaceRegion(drag.startRegions, drag.index, nextRegion));
  }, [normalizedPoint, updateDraft]);

  const finishInteraction = useCallback((event) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    // Pointer-up carries the authoritative final coordinates even when the
    // browser did not dispatch a last pointermove (common with quick pen taps).
    moveInteraction(event);
    event.preventDefault();
    event.stopPropagation();
    dragRef.current = null;
    releasePointer(drag);

    if (drag.changed) emitCommit(draftRegionsRef.current);
    else updateDraft(cloneRegions(drag.startRegions));
    if (drag.kind === 'add' && drag.changed) onAddModeChangeRef.current?.(false);
  }, [emitCommit, moveInteraction, updateDraft]);

  const cancelInteraction = useCallback((event) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    event.preventDefault();
    event.stopPropagation();
    abortInteraction();
  }, [abortInteraction]);

  const onEditorKeyDown = useCallback((event) => {
    // The parent lightbox has single-key actions. Focused editor controls must
    // never turn c/d/x or arrow presses into review actions by accident.
    event.stopPropagation();
    if (disabled || (event.key !== 'Delete' && event.key !== 'Backspace')) return;
    if (selectedIndex == null || !draftRegionsRef.current[selectedIndex]) return;

    event.preventDefault();
    const next = removeRegion(draftRegionsRef.current, selectedIndex);
    setSelectedIndex(next.length ? Math.min(selectedIndex, next.length - 1) : null);
    emitCommit(next);
  }, [disabled, emitCommit, selectedIndex]);

  const selectedText = selectedIndex == null
    ? 'No zone selected.'
    : `Zone ${selectedIndex + 1} selected.`;
  const limitReached = draftRegions.length >= MAX_WATERMARK_REGIONS;

  return (
    <div
      className="relative inline-block max-h-[70vh] max-w-[92vw] leading-none"
      role="group"
      aria-label="Watermark correction region editor"
      aria-describedby={countId}
      onClick={(event) => event.stopPropagation()}
      onKeyDown={onEditorKeyDown}
      onPointerMove={moveInteraction}
      onPointerUp={finishInteraction}
      onPointerCancel={cancelInteraction}
      onLostPointerCapture={cancelInteraction}
      style={{ touchAction: addMode && !disabled && !limitReached ? 'none' : undefined }}
    >
      <img
        ref={imageRef}
        src={src}
        alt={alt}
        draggable={false}
        onDragStart={(event) => event.preventDefault()}
        className="block max-h-[70vh] max-w-[92vw] select-none"
      />

      <span id={countId} className="sr-only" aria-live="polite">
        {draftRegions.length} watermark {draftRegions.length === 1 ? 'zone' : 'zones'}.{' '}
        {selectedText}{addMode ? ' Add mode armed: drag on the image to draw a zone.' : ''}
        {limitReached ? ` Maximum of ${MAX_WATERMARK_REGIONS} zones reached.` : ''}
      </span>

      {draftRegions.map((region, index) => {
        const selected = selectedIndex === index;
        const [left, top, right, bottom] = region;
        return (
          <div
            key={index}
            role="group"
            aria-label={`Watermark zone ${index + 1}`}
            className={`absolute rounded-sm border-2 ${selected
              ? 'z-20 border-sky-300 bg-sky-400/15 shadow-[0_0_0_1px_rgba(15,23,42,0.85)]'
              : 'z-10 border-amber-400 bg-amber-400/10'}`}
            style={{
              left: `${left * 100}%`,
              top: `${top * 100}%`,
              width: `${(right - left) * 100}%`,
              height: `${(bottom - top) * 100}%`,
            }}
          >
            <button
              type="button"
              disabled={disabled}
              aria-label={`Select and move watermark zone ${index + 1} of ${draftRegions.length}`}
              aria-pressed={selected}
              className="absolute inset-0 min-h-0 min-w-0 cursor-move bg-transparent disabled:cursor-default"
              style={{ touchAction: disabled ? undefined : 'none' }}
              onFocus={() => setSelectedIndex(index)}
              onClick={(event) => { event.stopPropagation(); setSelectedIndex(index); }}
              onPointerDown={(event) => beginInteraction(event, 'move', index)}
            />

            {selected && HANDLES.map(([corner, label, position]) => (
              <button
                key={corner}
                type="button"
                tabIndex={-1}
                disabled={disabled}
                aria-label={`Resize watermark zone ${index + 1} from the ${label} corner`}
                className={`absolute z-30 flex h-11 w-11 items-center justify-center bg-transparent ${position} disabled:cursor-default`}
                style={{ touchAction: disabled ? undefined : 'none' }}
                onClick={(event) => event.stopPropagation()}
                onPointerDown={(event) => beginInteraction(event, 'resize', index, corner)}
              >
                <span
                  aria-hidden="true"
                  className="h-3.5 w-3.5 rounded-full border-2 border-white bg-sky-300 shadow-md"
                />
              </button>
            ))}
          </div>
        );
      })}

      {addMode && !disabled && !limitReached && (
        <div
          className="absolute inset-0 z-40 cursor-crosshair"
          aria-label="Drag to add a watermark zone"
          onPointerDown={(event) => beginInteraction(event, 'add')}
          style={{ touchAction: 'none' }}
        />
      )}

      {children}
    </div>
  );
}
