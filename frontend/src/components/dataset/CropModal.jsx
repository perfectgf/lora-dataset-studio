/** Manual square crop of a dataset image (react-easy-crop). Returns px box.
 * Responsive: the crop area flexes to the available space on any screen. */
import { useCallback, useEffect, useRef, useState } from 'react';
import Cropper from 'react-easy-crop';

export default function CropModal({ imageUrl, onCancel, onConfirm }) {
  const [crop, setCrop] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [area, setArea] = useState(null);
  const cancelRef = useRef(null);
  const onComplete = useCallback((_, px) => setArea(px), []);

  // Keyboard support (M5): Escape closes, initial focus lands on Cancel.
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onCancel(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onCancel]);
  useEffect(() => { cancelRef.current?.focus(); }, []);

  return (
    <div role="dialog" aria-modal="true" aria-label="Crop image"
      className="fixed inset-0 z-[9995] bg-black/85 flex flex-col p-3 sm:p-4">
      <div className="relative flex-1 min-h-0 w-full max-w-3xl mx-auto bg-black rounded-lg overflow-hidden">
        <Cropper image={imageUrl} crop={crop} zoom={zoom} aspect={1}
          onCropChange={setCrop} onZoomChange={setZoom} onCropComplete={onComplete} />
      </div>
      <div className="shrink-0 w-full max-w-3xl mx-auto mt-3 flex flex-col gap-2">
        <label className="flex items-center gap-2 text-white/80 text-xs">
          Zoom
          <input type="range" min={1} max={4} step={0.05} value={zoom}
            onChange={(e) => setZoom(Number(e.target.value))} className="flex-1 accent-indigo-500" />
        </label>
        <div className="flex gap-2 justify-end">
          <button type="button" ref={cancelRef} onClick={onCancel}
            className="px-4 py-2 rounded-lg bg-surface text-content text-sm">Cancel</button>
          <button type="button" disabled={!area}
            onClick={() => onConfirm({ x: Math.round(area.x), y: Math.round(area.y),
                                       w: Math.round(area.width), h: Math.round(area.height) })}
            className="px-4 py-2 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
            Crop
          </button>
        </div>
      </div>
    </div>
  );
}
