import { useId, useRef, useState } from 'react'
import { putJson } from '../../api/fetchClient.js'
import useHuggingFaceTokenPresence, { notifyHuggingFaceTokenPresence } from '../../hooks/useHuggingFaceTokenPresence.js'

/** Recover a read-token failure without leaving the run. Dedicated cloud
 * tokens have different permissions and keep their own settings flow. */
export default function HuggingFaceTokenSetup({ presence } = {}) {
  const currentPresence = useHuggingFaceTokenPresence(presence === undefined ? 'HF_TOKEN' : null)
  const tokenPresence = presence ?? currentPresence
  const inputId = useId()
  const [token, setToken] = useState('')
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState('')
  const submitting = useRef(false)

  const save = async (event) => {
    event.preventDefault()
    event.stopPropagation()
    const value = token.trim()
    if (submitting.current || !value) return
    if (!/^hf_\S+$/.test(value)) {
      setError('Paste the token that starts with hf_.')
      return
    }
    submitting.current = true
    setSaving(true)
    setError('')
    try {
      const result = await putJson('/api/settings', { secrets: { HF_TOKEN: value } })
      if (result.secrets?.HF_TOKEN !== true) throw new Error('Token not saved')
      setToken('')
      setSaved(true)
      notifyHuggingFaceTokenPresence(result.secrets)
    } catch {
      setError('Could not save the token. Try again, or open the token settings above.')
    } finally {
      submitting.current = false
      setSaving(false)
    }
  }

  if (tokenPresence === 'configured') return <p role="status" className="m-0 mt-2 text-content-muted">
    {saved ? 'Token saved. You can retry the run.' : 'A Hugging Face token is already configured. Retry training to check access.'}
  </p>
  if (tokenPresence !== 'missing') return <p className="m-0 mt-2 text-content-muted">
    {tokenPresence === 'loading' ? 'Checking Hugging Face token settings…' : 'Open the token settings above to check the saved token.'}
  </p>

  return (
    <div className="mt-2 space-y-1.5 border-t border-amber-400/20 pt-2">
      <p className="m-0">
        <a href="https://huggingface.co/settings/tokens/new?tokenType=read"
          target="_blank" rel="noreferrer" onClick={(event) => event.stopPropagation()}
          className="font-semibold text-amber-300 underline decoration-amber-300/50">
          Create a read token ↗
        </a>
        {' '}Name it “LoRA Dataset Studio”, create it, then paste it below. Already have one? Paste it directly.
      </p>
      <form onSubmit={save} className="flex flex-wrap items-end gap-2">
        <div className="min-w-0 flex-[1_1_12rem]">
          <label htmlFor={inputId} className="mb-1 block font-medium">Hugging Face token</label>
          <input id={inputId} type="password" autoComplete="off" spellCheck={false}
            value={token} disabled={saving} placeholder="hf_…"
            onChange={(event) => { setToken(event.target.value); setError('') }}
            className="min-h-10 w-full rounded border border-border bg-app px-2 py-1.5 text-sm text-content" />
        </div>
        <button type="submit" disabled={saving || !token.trim()}
          className="min-h-10 rounded border border-amber-400/40 bg-amber-500/10 px-3 py-1.5 font-semibold text-amber-100 hover:bg-amber-500/20 disabled:opacity-50">
          {saving ? 'Saving…' : 'Save token'}
        </button>
      </form>
      {error && <p role="alert" className="m-0 text-rose-300">{error}</p>}
    </div>
  )
}
