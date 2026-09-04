/**
 * The project's vast.ai referral link — held in ONE place.
 *
 * vast.ai pays a referrer 3% of everything a referred account spends, for the
 * life of that account (docs.vast.ai → Referral Program). The referral is
 * attached when the account is CREATED through the tagged link — never to a
 * pod, a template or an API call — so the only surface that can carry it is
 * the moment the app asks someone to open an account: step 1 of the API-key
 * guide in Settings → Cloud GPU. Every other vast.ai link in the app (Billing,
 * Keys, the instances console) addresses someone who already has an account;
 * tagging those earns nothing and reads as tracking. `vastReferral.test.js`
 * holds that line, and mirrors the id into README and the settings guide.
 *
 * An empty id means the plain console link everywhere — what a fork wants,
 * and what the app does until the referral-only account exists.
 */
export const VAST_CONSOLE_URL = 'https://cloud.vast.ai/'

/** Referral id of the project's referral-only vast.ai account ('' = untagged). */
export const VAST_REFERRAL_ID = '683073'

/** The sign-up link step 1 of the guide points at — tagged when an id is set. */
export function vastSignupUrl(referralId = VAST_REFERRAL_ID) {
  const id = String(referralId ?? '').trim()
  return id ? `${VAST_CONSOLE_URL}?ref_id=${encodeURIComponent(id)}` : VAST_CONSOLE_URL
}
