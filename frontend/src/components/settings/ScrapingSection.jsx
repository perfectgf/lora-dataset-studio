import { Card, INPUT_CLASS, SecretField } from './primitives'

/* Reddit rides an anonymous OAuth token minted with gallery-dl's PUBLIC shared
   client id by default. Reddit's ~1000 requests / 10 min quota is attached to
   that id — shared with every other user of it — so scans can hit "wait N
   seconds" (429) without you having made a single request. A personal client id
   (free, one minute, no app secret involved) gives you a private quota. */
function RedditClientIdGuide() {
  return (
    <details className="mb-2 rounded-lg border border-border bg-surface-raised open:pb-1">
      <summary className="cursor-pointer select-none px-3 py-2 text-xs font-medium text-content">
        How to create your client ID (takes 1 minute)
      </summary>
      <ol className="list-decimal space-y-1.5 px-3 pb-2 pl-8 text-xs text-content-muted">
        <li>
          Sign in to Reddit, open{' '}
          <a href="https://www.reddit.com/prefs/apps" target="_blank" rel="noreferrer"
            className="font-medium text-content underline">reddit.com/prefs/apps</a>{' '}
          and click <span className="font-medium text-content">create app</span> (bottom of the page).
        </li>
        <li>
          Pick the type <span className="font-medium text-content">installed app</span> — this is the
          important step. A <span className="font-medium">web app</span> or{' '}
          <span className="font-medium">script</span> id comes with a client secret and Reddit then
          refuses the secret-less anonymous login this app uses (every scan would fail with 401).
        </li>
        <li>
          Name: anything (e.g. “LoRA Dataset Studio”). Redirect uri:{' '}
          <code className="rounded bg-surface px-1">http://localhost</code> — the form requires one,
          but it is never used here.
        </li>
        <li>
          Click <span className="font-medium text-content">create app</span>, then copy the short
          string shown right under the app name (~22 characters). That is your client ID — installed
          apps have no secret, so that string is all you need.
        </li>
        <li>Paste it below and Save. It takes effect immediately — no restart needed.</li>
      </ol>
    </details>
  )
}

const SCRAPE_SECRETS = [
  {
    key: 'REDDIT_CLIENT_ID',
    label: 'Reddit client ID',
    help: 'Out of the box, Reddit scans use a public client id shared by many people — its '
      + '~1000 requests / 10 min quota can be exhausted by others, which shows up as '
      + '“Reddit is rate limiting requests, retry in Ns” (429) even on your first scan of the day. '
      + 'Your own client ID gives you a private quota.',
    guide: <RedditClientIdGuide />,
  },
  {
    key: 'CIVITAI_API_KEY',
    label: 'Civitai API key',
    help: 'Only needed for adult content: without a key, Civitai scans return SFW results only. '
      + 'Create one under civitai.com → Account settings → API Keys.',
  },
]

export default function ScrapingSection(props) {
  const prompt = props.config?.klein?.small_image_prompt ?? ''

  return (
    <div className="space-y-6">
      <Card
        title="Source credentials"
        help="Optional keys used when scanning image sources for concept datasets. Everything works without them — they lift per-source limits. Keys are write-only: fields stay blank even when a key is already saved."
      >
        {SCRAPE_SECRETS.map((f) => <SecretField key={f.key} field={f} {...props} />)}
      </Card>
      <Card
        title="Klein image improvement"
        help="Optional instruction for automatic rescue of scraped images under 768 px. Manual Upscale & improve uses its dedicated metadata-derived quality profile. Klein creates a separate 2 MP version to validate and leaves the original intact."
      >
        <div>
          <div className="flex items-center justify-between gap-3">
            <label htmlFor="klein-small-image-prompt" className="text-sm font-medium text-content">
              Klein instruction
            </label>
            <span className="text-xs text-content-subtle">optional</span>
          </div>
          <p className="mb-1 text-xs leading-relaxed text-content-muted">
            Leave this empty to let Klein use the reference image alone. Add a short instruction only
            when you want to guide automatic scraper rescue; Klein remains generative and may change
            details. Manual lightbox improvement uses its dedicated quality profile.
          </p>
          <textarea id="klein-small-image-prompt" rows={4} value={prompt}
            onChange={(e) => props.setField('klein', 'small_image_prompt', e.target.value)}
            placeholder="Empty — reference image only"
            className={`${INPUT_CLASS} min-h-24 resize-y`} />
        </div>
      </Card>
    </div>
  )
}
