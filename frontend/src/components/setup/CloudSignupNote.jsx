import { VAST_REFERRAL_ID, vastSignupUrl } from '../../utils/vastReferral.js'
import VastReferralDisclosure from '../common/VastReferralDisclosure'

const link = 'text-primary underline'

/* The Setup wizard's "No GPU?" note on the training step — the wizard's one
   "create a vast.ai account" moment, so its link may carry the project's
   referral id (utils/vastReferral.js) and the disclosure renders right under
   it whenever the link is tagged. Its own component, not inline JSX in
   SetupPage: the page is not mountable under node --test, this is, so the
   pairing of link and disclosure is rendered in both states by
   tests/vast-key-guide-render.test.mjs. */
export default function CloudSignupNote({ referralId = VAST_REFERRAL_ID } = {}) {
  return (
    <>
      <p className="mt-2 text-content-muted text-xs">
        No GPU? You can skip this step: add a <strong>vast.ai API key</strong> in
        Settings instead and train in the cloud (the app rents a GPU per run,
        ~$1-2, and shuts it down automatically). No vast.ai account yet? Create a
        free one at{' '}
        <a href={vastSignupUrl(referralId)} target="_blank" rel="noreferrer" className={link}>cloud.vast.ai</a>
        {' '}— the Settings card then walks you through the key.
      </p>
      <VastReferralDisclosure referralId={referralId} subject="That sign-up link" linkClass={link}
        className="mt-1 text-content-muted text-xs" />
    </>
  )
}
