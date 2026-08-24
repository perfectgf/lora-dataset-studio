"""Deliberate cross-interpreter copies must stay byte-for-byte twins.

Some code runs on BOTH sides of an interpreter boundary and cannot be imported
across it: lds_aitk_bridge_runtime deploys standalone into ai-toolkit's
interpreter and must never import app.*. The house pattern for that situation
is a carried copy PLUS a test that pins the two bodies together, so a fix
applied to one side cannot silently miss the other (predictions_to_scenes has
the same arrangement, pinned by test_shot_boundaries/test_shot_detect).

This file pins the remove_directory pair: the reparse-point-safe recursive
delete that refuses to follow junctions - exactly the property whose two copies
must never disagree, because the divergent copy is the one that follows a link
into a real tree.
"""
import ast
import pathlib

BACKEND = pathlib.Path(__file__).resolve().parents[1]

TWINS = [
    ('remove_directory',
     'app/services/training_state_bundle.py',
     'app/training_bridge/lds_aitk_bridge_runtime.py'),
]


def _nested_def(rel, name):
    tree = ast.parse((BACKEND / rel).read_text(encoding='utf-8'))
    found = [n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(found) == 1, f'{rel}: expected exactly one {name}, saw {len(found)}'
    return found[0]


def test_each_twin_pair_is_ast_identical():
    for name, left, right in TWINS:
        a, b = _nested_def(left, name), _nested_def(right, name)
        dump_a = ast.dump(ast.Module(body=a.body, type_ignores=[]))
        dump_b = ast.dump(ast.Module(body=b.body, type_ignores=[]))
        assert ast.dump(a.args) == ast.dump(b.args), f'{name}: signatures diverged'
        assert dump_a == dump_b, (
            f'{name}: the copies in {left} and {right} diverged - apply the '
            f'change to BOTH sides (they run in different interpreters and '
            f'cannot share an import)')
