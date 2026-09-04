import { VAST_CONSOLE_URL } from '../../utils/vastReferral.js'

/* The sentence that makes a tagged vast.ai sign-up link honest — written once,
   rendered right next to EVERY such link (the API-key guide in Settings, the
   Setup wizard). It says what the link is, what it pays, that it costs the
   user nothing, and hands them the untagged link as a real alternative.
   `vastReferral.test.js` refuses a sign-up link that renders without it. */
export default function VastReferralDisclosure({ subject = 'This link', linkClass }) {
  return (
    <>
      {subject} is a referral link: open a vast.ai account through it and vast.ai pays this
      project 3% of what you spend there. It costs you nothing — the prices are identical
      either way, and nothing in the app behaves differently. Prefer not to?{' '}
      <a href={VAST_CONSOLE_URL} target="_blank" rel="noreferrer" className={linkClass}>cloud.vast.ai</a>
      {' '}works exactly the same.
    </>
  )
}
