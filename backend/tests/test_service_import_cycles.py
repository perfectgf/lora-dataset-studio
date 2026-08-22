"""The two largest service modules must not import each other.

``image_bank_service`` (the Bank) imports a dozen helpers from
``face_dataset_service`` (the dataset) at module level — that direction is a
fact of the code today. The other direction used to exist as ONE deferred
import (``normalize_pass_statuses``, "deferred: banks import us"), which is
exactly how a cycle survives: invisible at import time, real in the graph, and
the reason a later refactor of either module cannot move a function without
first checking who reaches it through the back door.

The scope vocabulary now lives in ``pass_scopes`` (a leaf), so the dataset side
needs nothing from the Bank. This test keeps it that way by reading the source:
an ``import`` statement of ``image_bank_service`` anywhere in
``face_dataset_service.py`` — top level or inside a function — fails it, and
says which line. The Bank may keep importing the dataset; what this pins is
that the edge has one direction.
"""
import ast
from pathlib import Path

_SERVICES = Path(__file__).resolve().parents[1] / 'app' / 'services'


def _imports_of(path: Path, module_name: str) -> list[int]:
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ''
            names = [a.name for a in node.names]
            if mod.endswith(module_name) or (mod in ('', '.') and module_name in names):
                hits.append(node.lineno)
        elif isinstance(node, ast.Import):
            if any(a.name.endswith(module_name) for a in node.names):
                hits.append(node.lineno)
    return hits


def test_the_dataset_service_never_imports_the_bank_service():
    hits = _imports_of(_SERVICES / 'face_dataset_service.py', 'image_bank_service')
    assert not hits, (
        'face_dataset_service imports image_bank_service at line(s) '
        f'{hits} — that closes the Bank⇄dataset cycle again. Whatever is needed '
        'belongs in a leaf module both can import (see pass_scopes.py).')


def test_the_scope_vocabulary_is_one_leaf_and_the_bank_re_exports_it():
    from app.services import pass_scopes
    from app.services import image_bank_service as banks
    assert banks.PASS_SCOPES is pass_scopes.PASS_SCOPES
    assert banks.CAPTION_SCOPES is pass_scopes.CAPTION_SCOPES
    assert banks.normalize_pass_statuses is pass_scopes.normalize_pass_statuses
    # And the leaf really is a leaf: nothing from the app is imported into it.
    hits = _imports_of(_SERVICES / 'pass_scopes.py', 'app')
    src = (_SERVICES / 'pass_scopes.py').read_text(encoding='utf-8')
    assert 'from .' not in src and 'from app' not in src and not hits, \
        'pass_scopes must stay a leaf: it depends on nothing in the app'


def test_normalize_pass_statuses_is_unchanged_by_the_move():
    from app.services.pass_scopes import normalize_pass_statuses, PASS_SCOPES
    import pytest
    assert normalize_pass_statuses(None) is None
    assert normalize_pass_statuses([]) is None
    assert normalize_pass_statuses('keep') == ['keep']
    assert normalize_pass_statuses(['pending', 'keep']) == ['keep', 'pending']
    assert normalize_pass_statuses(['KEEP ', 'keep']) == ['keep']
    assert normalize_pass_statuses(list(PASS_SCOPES)) == list(PASS_SCOPES)
    with pytest.raises(ValueError):
        normalize_pass_statuses(['bogus'])
    with pytest.raises(ValueError):
        normalize_pass_statuses(42)
    with pytest.raises(ValueError):
        normalize_pass_statuses([7])
