import { INPUT_CLASS, Card } from './primitives'
import { useI18n } from '../../i18n/I18nContext'

const CAPTIONING_OPTIONS = [
  { id: 'auto', labelKey: 'auto' },
  { id: 'joycaption', labelKey: 'joycaption' },
  { id: 'ollama', labelKey: 'ollama' },
  { id: 'none', labelKey: 'none' },
]

export default function CaptioningSection({ config, setField }) {
  const { t } = useI18n()
  return (
    <div className="space-y-6">
      <Card
        title={t('settings.captioning.captioningTitle')}
        help={t('settings.captioning.captioningHelp')}
      >
        <div>
          <label htmlFor="captioning-backend" className="block text-sm font-medium text-content">
            {t('settings.captioning.backend')}
          </label>
          <select
            id="captioning-backend"
            value={config.captioning.backend}
            onChange={(e) => setField('captioning', 'backend', e.target.value)}
            className={INPUT_CLASS}
          >
            {CAPTIONING_OPTIONS.map((o) => (
              <option key={o.id} value={o.id}>{t(`settings.captioning.options.${o.labelKey}`)}</option>
            ))}
          </select>
        </div>
      </Card>

      <Card
        title={t('settings.captioning.watermarkTitle')}
        help={t('settings.captioning.watermarkHelp')}
      >
        <div>
          <label htmlFor="watermark-device" className="block text-sm font-medium text-content">
            {t('settings.captioning.device')}
          </label>
          <select id="watermark-device" value={config.watermark?.device || 'auto'}
            onChange={(e) => setField('watermark', 'device', e.target.value)} className={INPUT_CLASS}>
            <option value="auto">{t('settings.captioning.devices.auto')}</option>
            <option value="cuda">{t('settings.captioning.devices.cuda')}</option>
            <option value="cpu">{t('settings.captioning.devices.cpu')}</option>
          </select>
        </div>
        <label className="mt-3 flex items-start gap-2 text-sm text-content">
          <input id="watermark-allow-crop" type="checkbox"
            checked={config.watermark?.allow_crop !== false}
            onChange={(e) => setField('watermark', 'allow_crop', e.target.checked)}
            className="mt-0.5" />
          <span>
            <span className="font-medium">{t('settings.captioning.allowCrop')}</span>
            <span className="block text-xs text-content-muted">
              {t('settings.captioning.allowCropHelp')}
            </span>
          </span>
        </label>
      </Card>

      <Card
        title={t('settings.captioning.faceTitle')}
        help={t('settings.captioning.faceHelp')}
      >
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label htmlFor="face-threshold-green" className="block text-sm font-medium text-content">
              {t('settings.captioning.greenThreshold')}
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
              {t('settings.captioning.orangeThreshold')}
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
          {t('settings.captioning.thresholdHelp')}
        </p>
      </Card>

      <Card
        title={t('settings.captioning.bank.title')}
        help={t('settings.captioning.bank.help')}
      >
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          <div>
            <label htmlFor="bank-sharpness-min" className="block text-sm font-medium text-content">
              {t('settings.captioning.bank.sharpness')}
            </label>
            <input id="bank-sharpness-min" type="number" min="0" step="10"
              value={config.bank?.sharpness_min ?? 100}
              onChange={(e) => setField('bank', 'sharpness_min', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">{t('settings.captioning.bank.sharpnessHelp')}</p>
          </div>
          <div>
            <label htmlFor="bank-noise-max" className="block text-sm font-medium text-content">
              {t('settings.captioning.bank.noise')}
            </label>
            <input id="bank-noise-max" type="number" min="0" step="1"
              value={config.bank?.noise_max ?? 15}
              onChange={(e) => setField('bank', 'noise_max', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">{t('settings.captioning.bank.noiseHelp')}</p>
          </div>
          <div>
            <label htmlFor="bank-uniformity-min" className="block text-sm font-medium text-content">
              {t('settings.captioning.bank.uniformity')}
            </label>
            <input id="bank-uniformity-min" type="number" min="0" step="1"
              value={config.bank?.uniformity_min ?? 12}
              onChange={(e) => setField('bank', 'uniformity_min', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">{t('settings.captioning.bank.uniformityHelp')}</p>
          </div>
          <div>
            <label htmlFor="bank-min-side" className="block text-sm font-medium text-content">
              {t('settings.captioning.bank.minSide')}
            </label>
            <input id="bank-min-side" type="number" min="0" step="64"
              value={config.bank?.min_side ?? 768}
              onChange={(e) => setField('bank', 'min_side', parseInt(e.target.value, 10) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">{t('settings.captioning.bank.minSideHelp')}</p>
          </div>
          <div>
            <label htmlFor="bank-dup-distance" className="block text-sm font-medium text-content">
              {t('settings.captioning.bank.duplicateDistance')}
            </label>
            <input id="bank-dup-distance" type="number" min="0" max="16" step="1"
              value={config.bank?.dup_distance ?? 8}
              onChange={(e) => setField('bank', 'dup_distance', parseInt(e.target.value, 10) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">{t('settings.captioning.bank.duplicateDistanceHelp')}</p>
          </div>
          <div>
            <label htmlFor="bank-face-threshold" className="block text-sm font-medium text-content">
              {t('settings.captioning.bank.personSimilarity')}
            </label>
            <input id="bank-face-threshold" type="number" min="0" max="1" step="0.01"
              value={config.bank?.face_threshold ?? 0.45}
              onChange={(e) => setField('bank', 'face_threshold', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">{t('settings.captioning.bank.personSimilarityHelp')}</p>
          </div>
          <div>
            <label htmlFor="bank-aesthetic-min" className="block text-sm font-medium text-content">
              {t('settings.captioning.bank.aesthetic')}
            </label>
            <input id="bank-aesthetic-min" type="number" min="0" max="10" step="0.5"
              value={config.bank?.aesthetic_min ?? 5}
              onChange={(e) => setField('bank', 'aesthetic_min', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">{t('settings.captioning.bank.aestheticHelp')}</p>
          </div>
          <div>
            <label htmlFor="bank-nsfw-max" className="block text-sm font-medium text-content">
              {t('settings.captioning.bank.nsfw')}
            </label>
            <input id="bank-nsfw-max" type="number" min="0" max="1" step="0.05"
              value={config.bank?.nsfw_max ?? 0.5}
              onChange={(e) => setField('bank', 'nsfw_max', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">{t('settings.captioning.bank.nsfwHelp')}</p>
          </div>
          <div>
            <label htmlFor="bank-style-threshold" className="block text-sm font-medium text-content">
              {t('settings.captioning.bank.styleSimilarity')}
            </label>
            <input id="bank-style-threshold" type="number" min="0" max="1" step="0.01"
              value={config.bank?.style_threshold ?? 0.6}
              onChange={(e) => setField('bank', 'style_threshold', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">{t('settings.captioning.bank.styleSimilarityHelp')}</p>
          </div>
        </div>
      </Card>
    </div>
  )
}
