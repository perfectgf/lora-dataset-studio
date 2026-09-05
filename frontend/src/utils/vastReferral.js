/**
 * The project's vast.ai referral id — held in ONE place, applied to EVERY
 * vast.ai link the app renders.
 *
 * vast.ai pays a referrer 3% of everything a referred account spends, for the
 * life of that account (docs.vast.ai → Referral Program); the referral is
 * attached when an account is CREATED through a tagged link. The maintainer's
 * rule (2026-09-05) is "wherever we talk about vast.ai, our link goes with it",
 * so every console page the app links to — sign-up, Billing, Keys, the
 * instances console — carries the id, through vastUrl()/VastLink. The
 * disclosure (VastReferralDisclosure) says so next to the two "create an
 * account" moments and offers the one deliberately untagged link.
 * `vastReferral.test.js` refuses a vast.ai URL spelled anywhere else in the
 * sources, and mirrors the id into README, the guides and .env.example.
 *
 * An empty id means plain links everywhere — what a fork wants.
 */
export const VAST_CONSOLE_URL = 'https://cloud.vast.ai/'

/** Referral id of the project's referral-only vast.ai account ('' = untagged). */
export const VAST_REFERRAL_ID = '683073'

/** A vast.ai console URL for `path` ('/', '/billing/', '/instances/'…), tagged when an id is set. */
export function vastUrl(path = '/', referralId = VAST_REFERRAL_ID) {
  const id = String(referralId ?? '').trim()
  const base = VAST_CONSOLE_URL + String(path ?? '/').replace(/^\/+/, '')
  return id ? `${base}?ref_id=${encodeURIComponent(id)}` : base
}

/** The sign-up link the "create an account" steps point at — the console root. */
export function vastSignupUrl(referralId = VAST_REFERRAL_ID) {
  return vastUrl('/', referralId)
}
