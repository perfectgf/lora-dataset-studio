/** Drop / pick real photos to import into the dataset. */
import { useRef, useState } from 'react';

export default function ImportDropzone({ onImport, busy, cropOption = false }) {
  const inputRef = useRef(null);
  const [over, setOver] = useState(false);
  // Auto head-crop (square, vision pass). OFF keeps the original framing —
  // a bust/body photo stays a bust/body photo (aspect kept, no padding).
  const [crop, setCrop] = useState(true);

  const handle = (files) => {
    if (busy) return; // drop events bypass pointer-events-none — guard here too (I2)
    if (files && files.length) onImport(files, { crop: cropOption ? crop : false });
  };

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); handle(e.dataTransfer.files); }}
      onClick={() => inputRef.current?.click()}
      className={`flex flex-col items-center justify-center gap-1 rounded-lg border-2 border-dashed p-4 cursor-pointer text-center
        ${over ? 'border-primary bg-primary/10' : 'border-border bg-surface'} ${busy ? 'opacity-50 pointer-events-none' : ''}`}
    >
      <span className="text-xl">📥</span>
      <span className="text-content text-xs font-medium">Import real photos</span>
      <span className="text-content-subtle text-[0.625rem]">
        drag and drop or click (normalized to 1024, kept)
      </span>
      {cropOption && (
        <label onClick={(e) => e.stopPropagation()}
          className="flex items-center gap-1.5 text-[0.625rem] text-content-muted cursor-pointer"
          title="ON: each photo is auto-cropped to a square head shot (vision pass, pauses ComfyUI). OFF: the photo keeps its original framing — use for bust/body shots.">
          <input type="checkbox" checked={crop} onChange={(e) => setCrop(e.target.checked)}
            className="accent-indigo-500 w-3 h-3" />
          ✂ Auto head-crop (square)
        </label>
      )}
      <input ref={inputRef} type="file" accept="image/*" multiple className="hidden"
        onChange={(e) => { handle(e.target.files); e.target.value = ''; }} />
    </div>
  );
}
