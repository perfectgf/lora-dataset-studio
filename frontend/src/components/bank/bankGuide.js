/* Pure helpers for the Bank's guided top (BankWorkspace.jsx). JSX-free so
   node --test exercises the "what's next" logic directly. The Bank top is 4
   ordered zones; the accent marks the ONE recommended next step from the Bank's
   counters. ③ Curer is an optional refinement — never the accented step. */

export const BANK_ZONES = [
  { id: 'analyze', order: 1, emoji: '①', label: 'Analyser' },
  { id: 'triage', order: 2, emoji: '②', label: 'Trier' },
  { id: 'curate', order: 3, emoji: '③', label: 'Curer' },
  { id: 'promote', order: 4, emoji: '④', label: 'Promouvoir' },
];

/* First match wins. `scoringAvailable` false = the Score pass can't run (no
   setup), so "not scored" must NOT strand the user on Analyse. */
export function nextBankStep(counts) {
  const c = counts || {};
  const scanned = c.scanned || 0;
  const scored = c.scored || 0;
  const keep = c.keep || 0;
  if (scanned === 0) return 'analyze';
  if (scored === 0 && c.scoringAvailable) return 'analyze';
  if (keep > 0) return 'promote';
  return 'triage';
}
