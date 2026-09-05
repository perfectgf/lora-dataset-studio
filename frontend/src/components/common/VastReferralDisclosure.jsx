import { VAST_CONSOLE_URL, VAST_REFERRAL_ID, vastSignupUrl } from '../../utils/vastReferral.js'

/* The sentence that makes the tagged vast.ai links honest — written once, and
   the ONE place that decides whether it shows. It renders beside each "create
   an account" moment (the API-key guide in Settings, the Setup wizard's note)
   and returns nothing when the links are untagged (empty id: forks), so no
   caller can pair tagged links with a missing disclosure, or the reverse. It
   says what the links are, what they pay, that it costs the user nothing, and
   hands them the one deliberately untagged link — under its own name, so it
   never reads like the others in a link list — as a real alternative. */
export default function VastReferralDisclosure({ referralId = VAST_REFERRAL_ID, linkClass, className }) {
  if (vastSignupUrl(referralId) === VAST_CONSOLE_URL) return null
  return (
    <p className={className}>
      Our vast.ai links are referral links: open a vast.ai account through one of them and
      vast.ai pays this project 3% of what you spend there. It costs you nothing — the prices
      are identical either way, and nothing in the app behaves differently. Prefer not to?{' '}
      <a href={VAST_CONSOLE_URL} target="_blank" rel="noreferrer" className={linkClass}>The untagged cloud.vast.ai link</a>
      {' '}works exactly the same.
    </p>
  )
}
