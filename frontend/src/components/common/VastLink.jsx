import { VAST_REFERRAL_ID, vastUrl } from '../../utils/vastReferral.js'

/* Every link to vast.ai the app renders goes through here (or through
   vastUrl/vastSignupUrl directly), so each one carries the project's referral
   id when there is one and stays bare when there is none — the maintainer's
   rule is "wherever we talk about vast.ai, our link goes with it", and
   vastReferral.test.js refuses a vast.ai URL spelled anywhere else. `path` is
   the console page ('/', '/billing/', '/instances/'…). */
export default function VastLink({ path = '/', referralId = VAST_REFERRAL_ID, className, children = 'vast.ai', ...rest }) {
  return (
    <a href={vastUrl(path, referralId)} target="_blank" rel="noreferrer" className={className} {...rest}>
      {children}
    </a>
  )
}
