"""One generator, two surfaces, one tuple shape.

``watermark_detector.scan()`` yields one tuple per image, and BOTH surfaces
unpack it inline — the bank in ``image_bank_service`` and the dataset in
``face_dataset_service``. When the tuple grew a sixth field (the fingerprint,
added for the bank's stale-write attestation), only the bank's unpack was
updated: the dataset pass then crashed with ``ValueError: too many values to
unpack`` on the FIRST real image, while the whole suite stayed green because
the test stub still yielded five fields.

This is the repo's standing failure shape — a shared contract duplicated by
hand on two surfaces (see CLAUDE.md, "Bank and Dataset are two surfaces of one
product") — so this test does what that rule prescribes: read every site and
pin them to each other. It parses, from SOURCE:

* the producer's ``yield (...)`` tuple in ``watermark_detector._run_chunk``,
* each consumer's ``for ... in watermark_detector.scan(`` target list,
* the stand-in generator in ``test_watermark_surface._fake_scan``,

and fails if any two disagree. Growing the tuple again is fine — this test
then names every site that has to move with it. If one of these functions is
renamed, this guard has to follow it, not be deleted.
"""

import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]


def _read(rel):
    return (BACKEND / rel).read_text(encoding='utf-8')


def _tuple_arity(source, anchor, where):
    """Count top-level elements of the parenthesised tuple starting at `anchor`."""
    at = source.find(anchor)
    assert at != -1, (
        f'{where}: could not find {anchor!r} — if the code moved or was '
        f'renamed, update this contract test to follow it')
    start = source.index('(', at)
    depth = 0
    commas = 0
    for i in range(start, len(source)):
        c = source[i]
        if c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
            if depth == 0:
                trailing = source[start + 1:i].rstrip().endswith(',')
                return commas + (0 if trailing else 1)
        elif c == ',' and depth == 1:
            commas += 1
    raise AssertionError(f'{where}: unbalanced parentheses after {anchor!r}')


def _unpack_arity(source, where):
    m = re.search(r'for\s+([\w\s,]+?)\s+in\s+watermark_detector\.scan\(', source)
    assert m, (
        f'{where}: no `for ... in watermark_detector.scan(` unpack found — '
        f'if this pass now consumes the generator differently, update this '
        f'contract test to read the new shape')
    return len([n for n in m.group(1).split(',') if n.strip()])


def test_scan_tuple_shape_is_identical_at_every_site():
    producer = _tuple_arity(_read('app/services/watermark_detector.py'),
                            'yield (path,', 'watermark_detector._run_chunk')
    bank = _unpack_arity(_read('app/services/image_bank_service.py'),
                         'image_bank_service (bank watermark pass)')
    dataset = _unpack_arity(_read('app/services/face_dataset_service.py'),
                            'face_dataset_service (dataset watermark pass)')
    surface_tests = _read('tests/test_watermark_surface.py')
    stub = _tuple_arity(surface_tests, 'yield (path, state',
                        'test_watermark_surface._fake_scan')
    # The route-parity test carries its own inline stub — the five-field
    # version of THAT one is what kept the suite green over the crash.
    inline = _tuple_arity(surface_tests, 'yield (paths[0],',
                          'test_watermark_surface route-parity inline stub')

    sites = {'producer yield': producer, 'bank unpack': bank,
             'dataset unpack': dataset, 'test stub yield': stub,
             'route-parity inline stub yield': inline}
    assert min(sites.values()) >= 5, f'parse degenerated: {sites}'
    assert len(set(sites.values())) == 1, (
        f'watermark_detector.scan() tuple shape has drifted between its sites: '
        f'{sites}. Every one of these must move together — a consumer left '
        f'behind crashes on the first real image while a five-field stub keeps '
        f'the suite green.')


def test_scan_docstring_names_every_field():
    """The docstring is the contract a future consumer will be written against;
    it must name exactly as many fields as the tuple carries."""
    source = _read('app/services/watermark_detector.py')
    m = re.search(r'Yield ``\(([^)]*)\)``', source)
    assert m, ('watermark_detector.scan lost its "Yield ``(...)``" contract '
               'line — restore it, the docstring is what consumers are '
               'written against')
    documented = len([f for f in m.group(1).split(',') if f.strip()])
    actual = _tuple_arity(source, 'yield (path,', 'watermark_detector._run_chunk')
    assert documented == actual, (
        f'scan() documents {documented} fields but yields {actual} — a '
        f'consumer written from the docstring will unpack the wrong shape')
