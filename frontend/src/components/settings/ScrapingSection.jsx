import { Card, INPUT_CLASS, SecretField } from './primitives'
import { useI18n } from '../../i18n/I18nContext'

/* Reddit rides an anonymous OAuth token minted with gallery-dl's PUBLIC shared
   client id by default. Reddit's ~1000 requests / 10 min quota is attached to
   that id — shared with every other user of it — so scans can hit "wait N
   seconds" (429) without you having made a single request. A personal client id
   (free, one minute, no app secret involved) gives you a private quota. */
function RedditClientIdGuide() {
  const { t } = useI18n()
  return (
    <details className="mb-2 rounded-lg border border-border bg-surface-raised open:pb-1">
      <summary className="cursor-pointer select-none px-3 py-2 text-xs font-medium text-content">
        {t('settings.scraping.redditGuide.summary')}
      </summary>
      <ol className="list-decimal space-y-1.5 px-3 pb-2 pl-8 text-xs text-content-muted">
        <li>
          {t('settings.scraping.redditGuide.step1Before')}{' '}
          <a href="https://www.reddit.com/prefs/apps" target="_blank" rel="noreferrer"
            className="font-medium text-content underline">reddit.com/prefs/apps</a>{' '}
          {t('settings.scraping.redditGuide.step1After')}{' '}
          <span className="font-medium text-content">create app</span>
          {t('settings.scraping.redditGuide.step1End')}
        </li>
        <li>
          {t('settings.scraping.redditGuide.step2Before')}{' '}
          <span className="font-medium text-content">installed app</span>
          {t('settings.scraping.redditGuide.step2Middle')}{' '}
          <span className="font-medium">web app</span> {t('settings.scraping.redditGuide.or')}{' '}
          <span className="font-medium">script</span>
          {t('settings.scraping.redditGuide.step2After')}
        </li>
        <li>
          {t('settings.scraping.redditGuide.step3Before')}{' '}
          <code className="rounded bg-surface px-1">http://localhost</code>
          {t('settings.scraping.redditGuide.step3After')}
        </li>
        <li>
          {t('settings.scraping.redditGuide.step4Before')}{' '}
          <span className="font-medium text-content">create app</span>
          {t('settings.scraping.redditGuide.step4After')}
        </li>
        <li>{t('settings.scraping.redditGuide.step5')}</li>
      </ol>
    </details>
  )
}

const scrapeSecrets = (t) => [
  {
    key: 'REDDIT_CLIENT_ID',
    label: t('settings.scraping.redditLabel'),
    help: t('settings.scraping.redditHelp'),
    guide: <RedditClientIdGuide />,
  },
  {
    key: 'CIVITAI_API_KEY',
    label: t('settings.scraping.civitaiLabel'),
    help: t('settings.scraping.civitaiHelp'),
  },
  {
    key: 'PEXELS_API_KEY',
    label: t('settings.scraping.pexelsLabel'),
    help: (
      <>
        {t('settings.scraping.pexelsHelpBefore')}{' '}
        <a href="https://www.pexels.com/api/key/" target="_blank" rel="noreferrer"
          className="font-medium text-content underline">
          {t('settings.scraping.pexelsCreateKey')}
        </a>
        {' '}{t('settings.scraping.pexelsHelpAfter')}
        <span className="mt-1 block text-amber-200">
          <strong>{t('settings.scraping.pexelsAuthorizationTitle')}</strong>{' '}
          {t('settings.scraping.pexelsAuthorizationBody')}{' '}
          <a href="https://help.pexels.com/hc/en-us/articles/900005880463-What-are-the-Terms-and-Conditions"
            target="_blank" rel="noreferrer" className="font-medium underline">
            {t('settings.scraping.pexelsTerms')}
          </a>
        </span>
      </>
    ),
  },
]

export default function ScrapingSection(props) {
  const { t } = useI18n()
  const prompt = props.config?.klein?.small_image_prompt ?? ''

  return (
    <div className="space-y-6">
      <Card
        title={t('settings.scraping.sourceCredentialsTitle')}
        help={t('settings.scraping.sourceCredentialsHelp')}
      >
        {scrapeSecrets(t).map((f) => <SecretField key={f.key} field={f} {...props} />)}
      </Card>
      <Card
        title={t('settings.scraping.improvementTitle')}
        help={t('settings.scraping.improvementHelp')}
      >
        <div>
          <div className="flex items-center justify-between gap-3">
            <label htmlFor="klein-small-image-prompt" className="text-sm font-medium text-content">
              {t('settings.scraping.instruction')}
            </label>
            <span className="text-xs text-content-subtle">{t('common.optional')}</span>
          </div>
          <p className="mb-1 text-xs leading-relaxed text-content-muted">
            {t('settings.scraping.instructionHelp')}
          </p>
          <textarea id="klein-small-image-prompt" rows={4} value={prompt}
            onChange={(e) => props.setField('klein', 'small_image_prompt', e.target.value)}
            placeholder={t('settings.scraping.instructionPlaceholder')}
            className={`${INPUT_CLASS} min-h-24 resize-y`} />
        </div>
      </Card>
    </div>
  )
}
