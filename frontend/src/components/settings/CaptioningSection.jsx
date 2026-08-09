import { INPUT_CLASS, Card } from './primitives'
import ResetToDefault from './ResetToDefault'
import { defaultValueAt } from './settingDefaults.js'
import { importInputLimitLine, IMPORT_INPUT_UNLIMITED_NOTE } from '../dataset/importPolicy.js'

// Decoded RGB costs 3 bytes per pixel, and an edit or analysis pass can hold a
// second copy at the same time — so the choices are labelled in the memory they
// actually commit, not in abstract megapixels. `0` is offered because "remove
// the limit" is a legitimate answer for a panorama or a camera master; it is
// last, and it says what it disarms.
const INPUT_MAX_PIXELS_OPTIONS = [
  { value: 16 * 1024 * 1024, label: '16 Mi-pixels — ~48 MiB per decode (old fixed limit)' },
  { value: 32 * 1024 * 1024, label: '32 Mi-pixels — ~96 MiB per decode' },
  { value: 64 * 1024 * 1024, label: '64 Mi-pixels — ~192 MiB per decode (default)' },
  { value: 128 * 1024 * 1024, label: '128 Mi-pixels — ~384 MiB per decode' },
  { value: 256 * 1024 * 1024, label: '256 Mi-pixels — ~768 MiB per decode' },
  { value: 0, label: 'No limit — accept any pixel count' },
]

const INPUT_MAX_SIDE_OPTIONS = [
  { value: 8192, label: '8192 px (old fixed limit)' },
  { value: 16384, label: '16384 px (default)' },
  { value: 32768, label: '32768 px' },
  { value: 65536, label: '65536 px' },
  { value: 0, label: 'No limit — accept any side' },
]

const CAPTIONING_OPTIONS = [
  { id: 'auto', label: 'Auto (best available)' },
  { id: 'joycaption', label: 'JoyCaption' },
  { id: 'ollama', label: 'Ollama vision' },
  { id: 'none', label: 'None' },
]

// Same shape, same word ("backend"), same 'auto'-first order as the captioning
// engine above — deliberately, because it is the same idea: two interchangeable
// engines and a default that picks whichever is installed. Read by BOTH 🧽 Find
// watermarks surfaces (bank and dataset), which used to disagree: the bank had
// always taken the detector when present, the dataset had never looked.
const WATERMARK_BACKEND_OPTIONS = [
  { id: 'auto', label: 'Auto (fastest available)' },
  { id: 'detector', label: 'Watermark detector (fast, scored)' },
  { id: 'vision', label: 'Vision model (Ollama)' },
]

export default function CaptioningSection({ config, setField, configDefaults }) {
  // The bank thresholds below were tuned on a real 36 000-image bank and are
  // re-tuned between releases: the numbers shown when a key is missing, like the
  // ones "Reset to default" writes, come from the server (config_defaults).
  const bankDefault = (key) => defaultValueAt(configDefaults, 'bank', key)
  const importDefault = (key) => defaultValueAt(configDefaults, 'dataset_import', key)
  const importEncoding = String(config.dataset_import?.encoding ?? importDefault('encoding') ?? 'preserve')
  const preservesOriginals = importEncoding === 'preserve'
  const inputDefault = (key) => defaultValueAt(configDefaults, 'image_input', key)
  // The sentence must quote the CONFIGURED budget, not a copy of the shipped
  // default: this card is where the number is changed, so it is the last place
  // allowed to describe a value the app no longer uses.
  const inputMaxSide = Number(config.image_input?.max_side ?? inputDefault('max_side') ?? 16384)
  const inputMaxPixels = Number(
    config.image_input?.max_pixels ?? inputDefault('max_pixels') ?? 64 * 1024 * 1024)
  const inputLimit = importInputLimitLine(
    { input_max_side: inputMaxSide, input_max_pixels: inputMaxPixels })
  const inputUnlimited = !inputMaxSide || !inputMaxPixels
  return (
    <div className="space-y-6">
      <Card
        title="Dataset import"
        help={`What happens to a photo the moment it enters a dataset. Preserve originals keeps the master file; training makes a temporary working copy only when a run starts. Every import must fit within ${inputLimit}; convert or resize a larger source before importing.`}
      >
        <div className="grid gap-3 sm:grid-cols-2">
          <div>
            <label htmlFor="dataset-import-max-side" className="block text-sm font-medium text-content">
              Stored resolution for WebP modes
            </label>
            <select id="dataset-import-max-side"
              value={String(config.dataset_import?.max_side ?? importDefault('max_side'))}
              onChange={(e) => setField('dataset_import', 'max_side', Number(e.target.value))}
              className={INPUT_CLASS}>
              <option value="1024">1024 px long side</option>
              <option value="1536">1536 px long side</option>
              <option value="2048">2048 px long side</option>
              <option value="4096">4096 px long side</option>
              <option value="0">Original size — no downscale</option>
            </select>
            <p className="mt-0.5 text-xs text-content-muted">
              {preservesOriginals
                ? 'Preserve originals ignores this setting: no resize is performed. It becomes active only if you choose a WebP mode below.'
                : 'Longest side kept; the aspect ratio is always preserved and an image is never enlarged. WebP normalization starts only after the source passes the import safety limit below.'}
            </p>
            <ResetToDefault label="Stored resolution" section="dataset_import" field="max_side"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
          <div>
            <label htmlFor="dataset-import-encoding" className="block text-sm font-medium text-content">
              Stored encoding
            </label>
            <select id="dataset-import-encoding"
              value={importEncoding}
              onChange={(e) => setField('dataset_import', 'encoding', e.target.value)}
              className={INPUT_CLASS}>
              <option value="preserve">Preserve originals — no resize or conversion (default)</option>
              <option value="standard">Standard — resize + WebP quality 92</option>
              <option value="high">High — WebP quality 100</option>
              <option value="lossless">Lossless — pixel-identical</option>
            </select>
            <p className="mt-0.5 text-xs text-content-muted">
              {preservesOriginals
                ? 'Non-cropped JPEG, PNG, WebP and BMP files keep their exact bytes with the matching extension. ai-toolkit receives a disposable PNG copy only when training starts.'
                : 'WebP modes are opt-in normalization: they resize and re-encode each new import. Lossless keeps every pixel and costs about 5× the disk space (measured on a noisy photo: 158 KB → 797 KB).'}
            </p>
            <ResetToDefault label="Stored encoding" section="dataset_import" field="encoding"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
        </div>
        <p className="mt-3 text-xs text-content-muted">
          <span className="font-medium">Import safety limit:</span> every source, including
          WebP modes and Auto head-crop, must be no larger than {inputLimit}. A larger file is
          rejected — change the budget in <span className="font-medium">Image size budget</span> below,
          or resize the file before importing.
        </p>
        <p className="mt-3 text-xs text-content-muted">
          Applies to images imported <span className="font-medium">from now on</span>: changing
          it mid-way can leave a dataset holding mixed formats or sizes. That is harmless for
          training (every trainer buckets and downscales on its own) but it does mean the folder
          is no longer uniform. Auto head-crop intentionally creates a derived WebP; existing
          WebPs cannot be turned back into their original files. Generated images and copies sent
          to an image API keep their own fixed sizes. Thanks to Qeeyana (Reddit) for asking why
          this was decided for you.
        </p>
      </Card>

      <Card
        title="Image size budget"
        help="How large a source image any part of the app is allowed to decode. Not an import-only rule: dataset import, ZIP and scrape ingest, Bank scan and thumbnails, edits, ComfyUI staging and Ollama vision all read these two numbers, so an image you can import is also an image you can look at."
      >
        <div className="grid gap-3 sm:grid-cols-2">
          <div>
            <label htmlFor="image-input-max-pixels" className="block text-sm font-medium text-content">
              Maximum total pixels
            </label>
            <select id="image-input-max-pixels"
              value={String(inputMaxPixels)}
              onChange={(e) => setField('image_input', 'max_pixels', Number(e.target.value))}
              className={INPUT_CLASS}>
              {INPUT_MAX_PIXELS_OPTIONS.map((o) => (
                <option key={o.value} value={String(o.value)}>{o.label}</option>
              ))}
            </select>
            <p className="mt-0.5 text-xs text-content-muted">
              This is a memory budget: a decoded RGB pixel costs 3 bytes, and an edit or analysis
              pass can hold a second copy at once, so the real peak is roughly double the figure
              shown. The default admits current phone and 35 mm camera masters (a 61 MP frame is
              57 Mi-pixels) and ordinary panoramas.
            </p>
            <ResetToDefault label="Maximum total pixels" section="image_input" field="max_pixels"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
          <div>
            <label htmlFor="image-input-max-side" className="block text-sm font-medium text-content">
              Maximum side
            </label>
            <select id="image-input-max-side"
              value={String(inputMaxSide)}
              onChange={(e) => setField('image_input', 'max_side', Number(e.target.value))}
              className={INPUT_CLASS}>
              {INPUT_MAX_SIDE_OPTIONS.map((o) => (
                <option key={o.value} value={String(o.value)}>{o.label}</option>
              ))}
            </select>
            <p className="mt-0.5 text-xs text-content-muted">
              Longest and shortest side of the source file. A very wide panorama can sit well
              inside the pixel budget and still exceed a side limit, which is why the two are
              separate knobs.
            </p>
            <ResetToDefault label="Maximum side" section="image_input" field="max_side"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
        </div>
        {inputUnlimited && (
          <p className="mt-3 rounded border border-amber-500/40 bg-amber-500/10 p-2 text-xs text-content">
            <span className="font-medium">No limit is set.</span> {IMPORT_INPUT_UNLIMITED_NOTE}
          </p>
        )}
        <p className="mt-3 text-xs text-content-muted">
          Current budget: {inputLimit}. JoyCaption captioning follows it. Image Bank inference
          workers (face, aesthetic and NSFW scoring, watermark inpainting) run in a separate
          Python interpreter and keep their own fixed 16 Mi-pixel / 8192 px guard: a larger
          image imports, displays and captions, but those workers skip it.
        </p>
      </Card>

      <Card
        title="Captioning"
        help="Who writes the captions. Auto prefers JoyCaption (via ai-toolkit) and falls back to the Ollama vision model."
      >
        <div>
          <label htmlFor="captioning-backend" className="block text-sm font-medium text-content">Captioning backend</label>
          <select
            id="captioning-backend"
            value={config.captioning.backend}
            onChange={(e) => setField('captioning', 'backend', e.target.value)}
            className={INPUT_CLASS}
          >
            {CAPTIONING_OPTIONS.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
          </select>
          <ResetToDefault label="Captioning backend" section="captioning" field="backend"
            config={config} configDefaults={configDefaults} setField={setField} />
        </div>
      </Card>

      <Card
        title="Watermark inpainting"
        help="Choose where LaMa removes small off-center watermarks. Auto uses CUDA when the configured ML Python supports it and otherwise falls back to CPU."
      >
        <div>
          <label htmlFor="watermark-device" className="block text-sm font-medium text-content">Processing device</label>
          <select id="watermark-device" value={config.watermark?.device || defaultValueAt(configDefaults, 'watermark', 'device')}
            onChange={(e) => setField('watermark', 'device', e.target.value)} className={INPUT_CLASS}>
            <option value="auto">Auto (GPU when available, otherwise CPU)</option>
            <option value="cuda">GPU (CUDA required; pauses ComfyUI while cleaning)</option>
            <option value="cpu">CPU (keeps the GPU free)</option>
          </select>
          <ResetToDefault label="Processing device" section="watermark" field="device"
            config={config} configDefaults={configDefaults} setField={setField} />
        </div>
        <label className="mt-3 flex items-start gap-2 text-sm text-content">
          <input id="watermark-allow-crop" type="checkbox"
            checked={config.watermark?.allow_crop !== false}
            onChange={(e) => setField('watermark', 'allow_crop', e.target.checked)}
            className="mt-0.5" />
          <span>
            <span className="font-medium">Allow automatic crop</span>
            <span className="block text-xs text-content-muted">
              On: a watermark sitting in a border is cropped off (no invented pixels). Off:
              border marks are repainted instead (LaMa/Klein). You can still override this per
              image in the watermark review. Also toggleable from the Clean bar.
            </span>
          </span>
        </label>
      </Card>

      <Card
        title="Face similarity"
        help="Every image is scored against the reference face (InsightFace). These thresholds set where the badges flip."
      >
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label htmlFor="face-threshold-green" className="block text-sm font-medium text-content">
              Face score — green threshold
            </label>
            <input
              id="face-threshold-green"
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={config.face_scoring.green}
              onChange={(e) => setField('face_scoring', 'green', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS}
            />
            <ResetToDefault label="Face score — green threshold" section="face_scoring" field="green"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
          <div>
            <label htmlFor="face-threshold-orange" className="block text-sm font-medium text-content">
              Face score — orange threshold
            </label>
            <input
              id="face-threshold-orange"
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={config.face_scoring.orange}
              onChange={(e) => setField('face_scoring', 'orange', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS}
            />
            <ResetToDefault label="Face score — orange threshold" section="face_scoring" field="orange"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
        </div>
        <p className="text-xs text-content-muted">
          Green marks a strong match to the reference face; orange is borderline — review it before keeping.
          Anything below orange is likely a different person and worth rejecting.
        </p>
      </Card>

      <Card
        title="Image bank triage"
        help="Thresholds for the 🗃️ Bank quality flags. Raw scores are stored per image, so changing a threshold re-sorts an already-scanned bank instantly — no rescan."
      >
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          <div>
            <label htmlFor="bank-sharpness-min" className="block text-sm font-medium text-content">
              Sharpness minimum
            </label>
            <input id="bank-sharpness-min" type="number" min="0" step="10"
              value={config.bank?.sharpness_min ?? bankDefault('sharpness_min')}
              onChange={(e) => setField('bank', 'sharpness_min', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">Laplacian variance under this = 🌫 blurry.</p>
            <ResetToDefault label="Sharpness minimum" section="bank" field="sharpness_min"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
          <div>
            <label htmlFor="bank-noise-max" className="block text-sm font-medium text-content">
              Noise maximum
            </label>
            <input id="bank-noise-max" type="number" min="0" step="1"
              value={config.bank?.noise_max ?? bankDefault('noise_max')}
              onChange={(e) => setField('bank', 'noise_max', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">Residual grain over this = 📺 noisy.</p>
            <ResetToDefault label="Noise maximum" section="bank" field="noise_max"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
          <div>
            <label htmlFor="bank-uniformity-min" className="block text-sm font-medium text-content">
              Uniformity minimum
            </label>
            <input id="bank-uniformity-min" type="number" min="0" step="1"
              value={config.bank?.uniformity_min ?? bankDefault('uniformity_min')}
              onChange={(e) => setField('bank', 'uniformity_min', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">Grayscale spread under this = ⬜ flat frame.</p>
            <ResetToDefault label="Uniformity minimum" section="bank" field="uniformity_min"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
          <div>
            <label htmlFor="bank-min-side" className="block text-sm font-medium text-content">
              Minimum side (px)
            </label>
            <input id="bank-min-side" type="number" min="0" step="64"
              value={config.bank?.min_side ?? bankDefault('min_side')}
              onChange={(e) => setField('bank', 'min_side', parseInt(e.target.value, 10) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">Smaller side under this = 📐 small (trainers only downscale).</p>
            <ResetToDefault label="Minimum side" section="bank" field="min_side"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
          <div>
            <label htmlFor="bank-detail-min" className="block text-sm font-medium text-content">
              Real-detail minimum
            </label>
            <input id="bank-detail-min" type="number" min="0" max="1" step="0.02"
              value={config.bank?.detail_min ?? bankDefault('detail_min')}
              onChange={(e) => setField('bank', 'detail_min', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">
              Share of the stored size that still carries real picture, under which an image is flagged 🫧 soft detail — the usual cause is an enlargement. 0.72 picks the softest few percent. A soft or out-of-focus photo reads the same way, so treat it as a score, not proof.
            </p>
            <ResetToDefault label="Real-detail minimum" section="bank" field="detail_min"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
          <div>
            <label htmlFor="bank-bars-max" className="block text-sm font-medium text-content">
              Black-bar maximum
            </label>
            <input id="bank-bars-max" type="number" min="0" max="1" step="0.01"
              value={config.bank?.bars_max ?? bankDefault('bars_max')}
              onChange={(e) => setField('bank', 'bars_max', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">Share of the frame that may be flat black letterbox before an image is flagged 🎞 black bars (video screenshots, padded stills).</p>
            <ResetToDefault label="Black-bar maximum" section="bank" field="bars_max"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
          <div>
            <label htmlFor="bank-dup-distance" className="block text-sm font-medium text-content">
              Duplicate distance
            </label>
            <input id="bank-dup-distance" type="number" min="0" max="16" step="1"
              value={config.bank?.dup_distance ?? bankDefault('dup_distance')}
              onChange={(e) => setField('bank', 'dup_distance', parseInt(e.target.value, 10) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">dHash bits (of 64) two images may differ by and still group as ≈ duplicates. Applies at the next scan.</p>
            <ResetToDefault label="Duplicate distance" section="bank" field="dup_distance"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
          <div>
            <label htmlFor="bank-face-threshold" className="block text-sm font-medium text-content">
              Same-person similarity
            </label>
            <input id="bank-face-threshold" type="number" min="0" max="1" step="0.01"
              value={config.bank?.face_threshold ?? bankDefault('face_threshold')}
              onChange={(e) => setField('bank', 'face_threshold', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">Cosine similarity for the 👥 person clustering. Applies at the next face pass.</p>
            <ResetToDefault label="Same-person similarity" section="bank" field="face_threshold"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
          <div>
            <label htmlFor="bank-aesthetic-min" className="block text-sm font-medium text-content">
              Aesthetic minimum
            </label>
            <input id="bank-aesthetic-min" type="number" min="0" max="10" step="0.5"
              value={config.bank?.aesthetic_min ?? bankDefault('aesthetic_min')}
              onChange={(e) => setField('bank', 'aesthetic_min', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">LAION score (~1–10) under which an image is flagged 💔 low aesthetic. Set by the ✨ Score pass.</p>
            <ResetToDefault label="Aesthetic minimum" section="bank" field="aesthetic_min"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
          <div>
            <label htmlFor="bank-nsfw-max" className="block text-sm font-medium text-content">
              NSFW maximum
            </label>
            <input id="bank-nsfw-max" type="number" min="0" max="1" step="0.05"
              value={config.bank?.nsfw_max ?? bankDefault('nsfw_max')}
              onChange={(e) => setField('bank', 'nsfw_max', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">NSFW probability (0–1) over which an image is flagged 🔞 NSFW. Set by the ✨ Score pass.</p>
            <ResetToDefault label="NSFW maximum" section="bank" field="nsfw_max"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
          <div>
            <label htmlFor="wmdet-backend" className="block text-sm font-medium text-content">
              Watermark detection
            </label>
            <select id="wmdet-backend"
              value={String(config.watermark_detect?.backend
                ?? defaultValueAt(configDefaults, 'watermark_detect', 'backend'))}
              onChange={(e) => setField('watermark_detect', 'backend', e.target.value)}
              className={INPUT_CLASS}>
              {WATERMARK_BACKEND_OPTIONS.map((o) => (
                <option key={o.id} value={o.id}>{o.label}</option>
              ))}
            </select>
            {/* The fallback rule is stated here AND said out loud when it fires
                (a toast on the dataset, the job's own line on the bank): a
                setting that silently does nothing is how someone changes a
                detector, sees no difference, and concludes the app is broken. */}
            <p className="mt-0.5 text-xs text-content-muted">
              Which engine 🧽 Find watermarks uses, on datasets and banks alike.
              <strong> Auto</strong> takes the detector extra when it is installed
              (~0.14 s per image, and it returns a score) and the vision model
              otherwise — that is the behaviour that shipped, so leaving this alone
              changes nothing. Picking <strong>Watermark detector</strong> without the
              extra installed does not fail the scan: the vision model runs it and the
              app says so, with the link to install the extra (Setup ▸ Quality tools).
              The two engines disagree at the margins, and only the detector can flag an
              image <em>without</em> a position — those are counted apart and 🧽 Clean
              leaves them for 🔍 Review. Changing this applies at the next scan; images
              you already dismissed as false positives are only re-judged by
              <em> ⟲ Rescan incl. dismissed</em>.
            </p>
            <ResetToDefault label="Watermark detection" section="watermark_detect"
              field="backend" config={config} configDefaults={configDefaults}
              setField={setField} />
          </div>
          <div>
            <label htmlFor="wmdet-threshold" className="block text-sm font-medium text-content">
              Watermark detector sensitivity
            </label>
            <input id="wmdet-threshold" type="number" min="0" max="1" step="0.01"
              value={config.watermark_detect?.threshold
                ?? defaultValueAt(configDefaults, 'watermark_detect', 'threshold')}
              onChange={(e) => setField('watermark_detect', 'threshold',
                parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            {/* The number looks wrong until you know why, so the help says why:
                this model's scores sit hard against 1, and the default is the
                knee MEASURED on a real bank, not a probability you can reason
                about from first principles. */}
            <p className="mt-0.5 text-xs text-content-muted">
              Score (0–1) at or above which 🚩 Find flags an image, when the watermark
              detector extra is installed. 0.94 is measured on a 110-image hand-labelled
              sample of a real 29 759-image bank: it flagged none of the 55 clean images
              and 54 of the 55 marked ones. Lower it to ~0.92 to catch the faintest marks
              (and hand-check a few clean ones); raise it to ~0.96 to miss a mark rather
              than crop anything by mistake. Applies at the next scan.
            </p>
            <ResetToDefault label="Watermark detector sensitivity" section="watermark_detect"
              field="threshold" config={config} configDefaults={configDefaults}
              setField={setField} />
          </div>
          <div>
            <label htmlFor="bank-style-threshold" className="block text-sm font-medium text-content">
              Same-style similarity
            </label>
            <input id="bank-style-threshold" type="number" min="0" max="1" step="0.01"
              value={config.bank?.style_threshold ?? bankDefault('style_threshold')}
              onChange={(e) => setField('bank', 'style_threshold', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">Cosine similarity for the 🎨 style clustering. Applies at the next scoring pass.</p>
            <ResetToDefault label="Same-style similarity" section="bank" field="style_threshold"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
          <div>
            <label htmlFor="bank-semantic-dup-threshold" className="block text-sm font-medium text-content">
              Semantic duplicate similarity
            </label>
            <input id="bank-semantic-dup-threshold" type="number" min="0" max="1" step="0.01"
              value={config.bank?.semantic_dup_threshold ?? bankDefault('semantic_dup_threshold')}
              onChange={(e) => setField('bank', 'semantic_dup_threshold', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">Cosine similarity at or above which two scored images are a ✂ semantic near-duplicate (crop/variant of the same shot). Re-runs instantly from cached embeddings — no re-scan.</p>
            <ResetToDefault label="Semantic duplicate similarity" section="bank" field="semantic_dup_threshold"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
        </div>
      </Card>
    </div>
  )
}
