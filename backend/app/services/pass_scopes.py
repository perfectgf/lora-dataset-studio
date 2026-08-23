"""The per-run SCOPE vocabulary a Bank or dataset pass accepts, and its normaliser.

This lived inside ``image_bank_service`` and was the one thing
``face_dataset_service`` ever needed from it — imported inside a function with
the comment "deferred: banks import us". That deferred import was the only edge
closing a cycle between the two largest modules of the backend
(``image_bank_service`` imports a dozen helpers from ``face_dataset_service`` at
module level). A scope vocabulary has no business in either: it is shared by
construction (the same three words on both surfaces, see CLAUDE.md "Bank and
Dataset are two surfaces of one product"), so it gets a leaf module of its own
that depends on nothing. ``image_bank_service`` re-exports these names, so every
caller that reached them through it still can.
"""

PASS_SCOPES = ('keep', 'pending', 'reject')

# Captions accept the same scope words as every other pass; one tuple, two names,
# so the caption routes read as "the caption scopes" without a second vocabulary
# that could drift.
CAPTION_SCOPES = PASS_SCOPES


def normalize_pass_statuses(statuses, allowed=PASS_SCOPES):
    """Validate a per-run scope → a canonical list, or None for "as before".

    None / [] → None, meaning the pass keeps its own historical filter. Anything
    outside ``allowed`` raises ValueError → 400, exactly like a bad vocabulary."""
    if statuses is None:
        return None
    if isinstance(statuses, str):       # a lone 'keep' is a scope of one
        statuses = [statuses]
    if not isinstance(statuses, (list, tuple, set)):
        raise ValueError('invalid statuses: expected a list of statuses')
    want = []
    for s in statuses:
        if not isinstance(s, str):
            raise ValueError('invalid status: expected status names')
        v = s.strip().lower()
        if not v:
            continue
        if v not in allowed:
            raise ValueError(f'invalid status: {v}')
        want.append(v)
    if not want:
        return None
    # Canonical order + dedup, so ['pending','keep'] and ['keep','pending'] are
    # one value and never two code paths.
    return [s for s in PASS_SCOPES if s in want]
