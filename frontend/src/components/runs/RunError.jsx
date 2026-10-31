import SettingsLink from '../common/SettingsLink.jsx'
import HuggingFaceTokenSetup from '../common/HuggingFaceTokenSetup.jsx'
import useHuggingFaceTokenPresence from '../../hooks/useHuggingFaceTokenPresence.js'

function huggingFaceAccessError(error) {
  if (typeof error !== 'string') return null
  const gated = /GatedRepoError|cannot access gated repo|access to model [^\n]+ is restricted/i.test(error)
  const huggingFace = gated || /huggingface\.co\/|hugging\s*face|\bHF_(?:CLOUD_)?TOKEN\b/i.test(error)
  const denied = gated || /\b(?:401|403)\b|unauthorized|forbidden|invalid token|token[^\n]*(?:missing|rejected|required)/i.test(error)
  if (!huggingFace || !denied) return null

  // Rebuild only the model page from a canonical download URL. Never turn an
  // arbitrary URL, query token or signed download from the error into a link.
  const repo = error.match(/https:\/\/huggingface\.co\/([\w-]+\/[\w.-]+)\/(?:resolve|blob|tree)\//i)?.[1]
  return { repo, forbidden: /\b403\b|forbidden/i.test(error) }
}

/** The raw failure stays readable; its recovery link must never be clamped. */
export function RunErrorMessage({ error, dedicatedToken = false, tokenPresence = 'unknown' }) {
  const access = huggingFaceAccessError(error)
  const configured = access && tokenPresence === 'configured'
  const message = configured
    ? 'A Hugging Face token is configured. This run keeps the error from its previous attempt. Retry to check access.'
    : tokenPresence === 'loading'
      ? 'Checking Hugging Face token settings…'
      : tokenPresence === 'unknown'
        ? 'Open the token settings to check the saved token and its permissions.'
        : access?.forbidden
          ? 'Hugging Face refused access. Check the token permissions and your account’s access to the model, then retry.'
          : 'Add or update the Hugging Face token for the account that can download this model, then retry. Your browser login is separate from LDS.'
  const rawError = (
    <p className="m-0 whitespace-pre-line break-words line-clamp-5 text-rose-300/90" title={error}>
      {error}
    </p>
  )
  return (
    <div className="space-y-1.5 text-[0.6875rem]">
      {access && (
        <div className={`rounded border px-2 py-1.5 ${configured
          ? 'border-border bg-surface text-content-muted'
          : 'border-amber-400/40 bg-amber-500/10 text-amber-100'}`}>
          <p className="m-0 break-words">{message}</p>
          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1">
            {dedicatedToken ? (
              <SettingsLink pluginId="cloud_training" focus="HF_CLOUD_TOKEN" tone={configured ? 'subtle' : 'warning'}>
                Configure dedicated Hugging Face cloud token
              </SettingsLink>
            ) : (
              <SettingsLink section="local-tools" focus="HF_TOKEN" tone={configured ? 'subtle' : 'warning'}>
                Configure Hugging Face token
              </SettingsLink>
            )}
            {access.repo && (
              <a href={`https://huggingface.co/${access.repo}`} target="_blank" rel="noreferrer"
                onClick={(event) => event.stopPropagation()}
                className="text-amber-300 underline decoration-amber-300/50">
                Check model access ↗
              </a>
            )}
          </div>
          {!dedicatedToken && tokenPresence === 'missing' && <HuggingFaceTokenSetup presence={tokenPresence} />}
        </div>
      )}
      {configured ? (
        <details>
          <summary className="cursor-pointer text-content-subtle">Recorded error from this attempt</summary>
          {rawError}
        </details>
      ) : rawError}
    </div>
  )
}

export default function RunError({ error, dedicatedToken = false }) {
  const tokenPresence = useHuggingFaceTokenPresence(huggingFaceAccessError(error)
    ? (dedicatedToken ? 'HF_CLOUD_TOKEN' : 'HF_TOKEN') : null)
  return <RunErrorMessage error={error} dedicatedToken={dedicatedToken} tokenPresence={tokenPresence} />
}
