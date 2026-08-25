/** Drop / pick real photos to import into the dataset.
 *
 * The dropzone used to say "normalized to 1024" and nothing else — no why, no
 * way out (reported by Qeeyana on Reddit: "why? Let me choose not to"). It now
 * states what will be stored, its safety limit, and links to the setting that
 * changes it. The value comes from capabilities, never from a copy of the
 * default kept here: a hint that can go stale is worse than no hint.
 */
import { useRef, useState } from 'react';
import { ImageDown } from 'lucide-react';
import { useCapabilities } from '../../context/CapabilitiesContext';
import SettingsLink from '../common/SettingsLink';
import {
  IMPORT_IMAGE_ACCEPT,
  IMPORT_IMAGE_FORMATS,
  importInputLimitLine,
  importPolicyLine,
  preservesOriginalFiles,
} from './importPolicy.js';

export default function ImportDropzone({ onImport, busy, visionBusy = false, cropOption = false, defaultCrop = true }) {
  const { caps } = useCapabilities();
  const inputRef = useRef(null);
  const [over, setOver] = useState(false);
  const importPolicy = caps?.dataset_import;
  const preservesOriginals = preservesOriginalFiles(importPolicy);
  const inputLimit = importInputLimitLine(importPolicy);
  // Auto head-crop (square, vision pass). OFF keeps the original file and
  // framing — a bust/body photo stays a bust/body photo (aspect kept, no padding).
  // Body-fidelity datasets pass defaultCrop=false: full frames are the point.
  const [crop, setCrop] = useState(defaultCrop);
  const autoCropEnabled = cropOption && crop && !visionBusy;

  const handle = (files) => {
    if (busy) return; // drop events bypass pointer-events-none — guard here too (I2)
    if (files && files.length) onImport(files, { crop: cropOption ? crop && !visionBusy : false });
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
      <ImageDown aria-hidden="true" className="h-5 w-5" />
      <span className="text-content text-xs font-medium">Import real photos</span>
      <span className="text-content-subtle text-[0.625rem]">
        drag and drop or click — {autoCropEnabled
          ? `auto head-crop is on (input limit: ${inputLimit})`
          : importPolicyLine(importPolicy)}
      </span>
      <span onClick={(e) => e.stopPropagation()}
        className="flex flex-wrap items-center justify-center gap-x-1.5 gap-y-0.5 text-[0.625rem] text-content-subtle">
        {autoCropEnabled ? (
          <span>Auto head-crop creates a derived WebP. Turn it off to preserve an eligible original.</span>
        ) : preservesOriginals ? (
          <span>Uncropped {IMPORT_IMAGE_FORMATS} files stay byte-for-byte unchanged.</span>
        ) : (
          <span>WebP normalization resizes and re-encodes eligible imports.</span>
        )}
        <span>Files larger than {inputLimit} are rejected — resize before importing, or raise the budget.</span>
        <SettingsLink section="captioning" focus="dataset-import-encoding">Change storage mode</SettingsLink>
        <SettingsLink section="captioning" focus="image-input-max-pixels">Change size budget</SettingsLink>
      </span>
      {cropOption && (
        <label onClick={(e) => e.stopPropagation()}
          className="flex items-center gap-1.5 text-[0.625rem] text-content-muted cursor-pointer"
          title={visionBusy ? 'Auto head-crop is unavailable during local generation; photos import full-frame.' : 'ON: each photo is auto-cropped to a square head shot (vision pass, pauses ComfyUI) and stored as a derived WebP. OFF: the original file and framing are kept — use for bust/body shots.'}>
          <input type="checkbox" checked={crop} disabled={visionBusy} onChange={(e) => setCrop(e.target.checked)}
            className="accent-indigo-500 w-3 h-3" />
          ✂ Auto head-crop (square){visionBusy ? ' — unavailable during local generation' : ''}
        </label>
      )}
      <input ref={inputRef} type="file" accept={IMPORT_IMAGE_ACCEPT} multiple className="hidden"
        onChange={(e) => { handle(e.target.files); e.target.value = ''; }} />
    </div>
  );
}
