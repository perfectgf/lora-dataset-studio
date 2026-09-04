import { VAST_CONSOLE_URL, VAST_REFERRAL_ID, vastSignupUrl } from '../../utils/vastReferral.js'

/* The sentence that makes a tagged vast.ai sign-up link honest — written once,
   and the ONE place that decides whether it shows. It renders beside every
   sign-up link (the API-key guide in Settings, the Setup wizard's note) and
   returns nothing when that link is untagged (empty id: forks), so no caller
   can pair a tagged link with a missing disclosure, or the reverse. It says
   what the link is, what it pays, that it costs the user nothing, and hands
   them the untagged link — under its own name, so the two links never read
   alike in a link list — as a real alternative. */
export default function VastReferralDisclosure({
  referralId = VAST_REFERRAL_ID, subject = 'This link', linkClass, className,
}) {
  if (vastSignupUrl(referralId) === VAST_CONSOLE_URL) return null
  return (
    <p className={className}>
      {subject} is a referral link: open a vast.ai account through it and vast.ai pays this
      project 3% of what you spend there. It costs you nothing — the prices are identical
      either way, and nothing in the app behaves differently. Prefer not to?{' '}
      <a href={VAST_CONSOLE_URL} target="_blank" rel="noreferrer" className={linkClass}>The untagged cloud.vast.ai link</a>
      {' '}works exactly the same.
    </p>
  )
}
