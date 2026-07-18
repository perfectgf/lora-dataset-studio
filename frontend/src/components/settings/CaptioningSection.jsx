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
    </div>
  )
}
