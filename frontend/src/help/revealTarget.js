/* Reveal a deep-linked field before scrolling to it.

   The Help mode's ?focus=<domId> deep-links (Settings search, guide "Open this
   screen →") land on a specific control. Two things used to make them scroll
   into the void:
     1. the field sits inside a collapsed <details> (e.g. the ai-toolkit
        overrides) — it IS in the DOM, but hidden, so the ring flashes on
        nothing the user can see;
     2. the field isn't rendered at all because a React-conditional gate hasn't
        un-hidden it (e.g. the access token, shown only once LAN + require-token
        are on) — getElementById returns null and nothing happens.

   This module resolves both. Case 1 is pure presentation: open every collapsed
   <details> ancestor. Case 2 can't be revealed by the DOM alone (React hasn't
   rendered the field) and we must NOT silently flip a data toggle, so we fall
   back to the deepest currently-rendered *gate* that advertises the field via
   data-focus-gate — the switch the user has to turn on — and ring THAT instead.

   PURE JS with an injectable document so node --test can drive it with a fake
   DOM (no jsdom). */

/* Open any collapsed native <details> on the path from `el` up to the root, so a
   field nested in one becomes visible. No-op for a field that has none. */
export function openCollapsedAncestors(el) {
  let node = el;
  let guard = 0;
  while (node && guard < 200) {
    if (node.tagName === 'DETAILS' && node.open === false) node.open = true;
    node = node.parentElement;
    guard += 1;
  }
}

/* Resolve where a focus id should scroll to:
     { el, gated: false } — the field itself, in the DOM;
     { el, gated: true }  — the deepest gate that would reveal it (its switch);
     null                 — neither exists (nothing to focus).
   `doc` is injectable for tests; defaults to the ambient document. */
export function resolveFocusTarget(focusId, doc) {
  const d = doc || (typeof document !== 'undefined' ? document : null);
  if (!d || !focusId) return null;
  const direct = d.getElementById(focusId);
  if (direct) return { el: direct, gated: false };
  // A gate lists the field ids it hides in a space-separated data-focus-gate.
  // With several gates rendered (nested conditionals), the LAST in document
  // order is the innermost — the most specific switch to point the user at.
  const gates = d.querySelectorAll(`[data-focus-gate~="${focusId}"]`);
  const gate = gates && gates.length ? gates[gates.length - 1] : null;
  return gate ? { el: gate, gated: true } : null;
}
