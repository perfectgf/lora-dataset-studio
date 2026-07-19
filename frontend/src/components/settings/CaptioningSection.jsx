import { INPUT_CLASS, Card } from './primitives'

const CAPTIONING_OPTIONS = [
  { id: 'auto', label: 'Auto (best available)' },
  { id: 'joycaption', label: 'JoyCaption' },
  { id: 'ollama', label: 'Ollama vision' },
  { id: 'none', label: 'None' },
]

export default function CaptioningSection({ config, setField }) {
  return (
    <div className="space-y-6">
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
        </div>
      </Card>

      <Card
        title="Watermark inpainting"
        help="Choose where LaMa removes small off-center watermarks. Auto uses CUDA when the configured ML Python supports it and otherwise falls back to CPU."
      >
        <div>
          <label htmlFor="watermark-device" className="block text-sm font-medium text-content">Processing device</label>
          <select id="watermark-device" value={config.watermark?.device || 'auto'}
            onChange={(e) => setField('watermark', 'device', e.target.value)} className={INPUT_CLASS}>
            <option value="auto">Auto (GPU when available, otherwise CPU)</option>
            <option value="cuda">GPU (CUDA required; pauses ComfyUI while cleaning)</option>
            <option value="cpu">CPU (keeps the GPU free)</option>
          </select>
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
              value={config.bank?.sharpness_min ?? 100}
              onChange={(e) => setField('bank', 'sharpness_min', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">Laplacian variance under this = 🌫 blurry.</p>
          </div>
          <div>
            <label htmlFor="bank-noise-max" className="block text-sm font-medium text-content">
              Noise maximum
            </label>
            <input id="bank-noise-max" type="number" min="0" step="1"
              value={config.bank?.noise_max ?? 15}
              onChange={(e) => setField('bank', 'noise_max', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">Residual grain over this = 📺 noisy.</p>
          </div>
          <div>
            <label htmlFor="bank-uniformity-min" className="block text-sm font-medium text-content">
              Uniformity minimum
            </label>
            <input id="bank-uniformity-min" type="number" min="0" step="1"
              value={config.bank?.uniformity_min ?? 12}
              onChange={(e) => setField('bank', 'uniformity_min', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">Grayscale spread under this = ⬜ flat frame.</p>
          </div>
          <div>
            <label htmlFor="bank-min-side" className="block text-sm font-medium text-content">
              Minimum side (px)
            </label>
            <input id="bank-min-side" type="number" min="0" step="64"
              value={config.bank?.min_side ?? 768}
              onChange={(e) => setField('bank', 'min_side', parseInt(e.target.value, 10) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">Smaller side under this = 📐 small (trainers only downscale).</p>
          </div>
          <div>
            <label htmlFor="bank-dup-distance" className="block text-sm font-medium text-content">
              Duplicate distance
            </label>
            <input id="bank-dup-distance" type="number" min="0" max="16" step="1"
              value={config.bank?.dup_distance ?? 8}
              onChange={(e) => setField('bank', 'dup_distance', parseInt(e.target.value, 10) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">dHash bits (of 64) two images may differ by and still group as ≈ duplicates. Applies at the next scan.</p>
          </div>
          <div>
            <label htmlFor="bank-face-threshold" className="block text-sm font-medium text-content">
              Same-person similarity
            </label>
            <input id="bank-face-threshold" type="number" min="0" max="1" step="0.01"
              value={config.bank?.face_threshold ?? 0.45}
              onChange={(e) => setField('bank', 'face_threshold', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">Cosine similarity for the 👥 person clustering. Applies at the next face pass.</p>
          </div>
          <div>
            <label htmlFor="bank-aesthetic-min" className="block text-sm font-medium text-content">
              Aesthetic minimum
            </label>
            <input id="bank-aesthetic-min" type="number" min="0" max="10" step="0.5"
              value={config.bank?.aesthetic_min ?? 5}
              onChange={(e) => setField('bank', 'aesthetic_min', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">LAION score (~1–10) under which an image is flagged 💔 low aesthetic. Set by the ✨ Score pass.</p>
          </div>
          <div>
            <label htmlFor="bank-nsfw-max" className="block text-sm font-medium text-content">
              NSFW maximum
            </label>
            <input id="bank-nsfw-max" type="number" min="0" max="1" step="0.05"
              value={config.bank?.nsfw_max ?? 0.5}
              onChange={(e) => setField('bank', 'nsfw_max', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">NSFW probability (0–1) over which an image is flagged 🔞 NSFW. Set by the ✨ Score pass.</p>
          </div>
          <div>
            <label htmlFor="bank-style-threshold" className="block text-sm font-medium text-content">
              Same-style similarity
            </label>
            <input id="bank-style-threshold" type="number" min="0" max="1" step="0.01"
              value={config.bank?.style_threshold ?? 0.6}
              onChange={(e) => setField('bank', 'style_threshold', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">Cosine similarity for the 🎨 style clustering. Applies at the next scoring pass.</p>
          </div>
          <div>
            <label htmlFor="bank-semantic-dup-threshold" className="block text-sm font-medium text-content">
              Semantic duplicate similarity
            </label>
            <input id="bank-semantic-dup-threshold" type="number" min="0" max="1" step="0.01"
              value={config.bank?.semantic_dup_threshold ?? 0.96}
              onChange={(e) => setField('bank', 'semantic_dup_threshold', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">Cosine similarity at or above which two scored images are a ✂ semantic near-duplicate (crop/variant of the same shot). Re-runs instantly from cached embeddings — no re-scan.</p>
          </div>
        </div>
      </Card>
    </div>
  )
}
