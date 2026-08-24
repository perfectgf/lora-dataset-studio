"""infer/_harness.py stays a stdlib-only sibling, and the factored map holds.

The infer scripts run in their own torch venvs and are launched as plain files,
so their one shared module must import nothing a bare interpreter lacks — a
single torch/numpy/app import in _harness would kill every worker at startup.
And each script that handed a helper to the harness must keep importing it
rather than quietly growing a local copy back (the drift this factoring
removed). Pure AST checks: nothing here needs the ML venvs.
"""
import ast
import pathlib

INFER = pathlib.Path(__file__).resolve().parents[1] / 'infer'

# Simple stdlib names only: anything outside this set is a doctrine break.
ALLOWED_IMPORTS = {'json', 'os', 'sys', 'typing'}

# The factored map, file -> names it must import from _harness and not redefine.
FACTORED = {
    'bank_score_infer.py': {'_log', '_cancel_requested', '_write_count'},
    'bank_semantic_infer.py': {'_pooled_features'},
    'clip_image_embed_infer.py': {'_log', '_emit'},
    'clip_text_infer.py': {'_log', '_emit'},
    'face_embed_infer.py': {'_log', '_cancel_requested', '_write_count'},
    'face_score_infer.py': {'_log'},
    'shot_detect_infer.py': {'_log', '_emit', '_cancel_requested'},
    'siglip2_text_infer.py': {'_pooled_features'},
    'video_aesthetic_infer.py': {'_log', '_emit'},
    'video_ai_check_infer.py': {'_log', '_emit'},
    'video_caption_infer.py': {'_log', '_emit'},
    'video_text_infer.py': {'_log', '_emit', '_cancel_requested'},
    'watermark_detect_infer.py': {'_log', '_emit', '_cancel_requested'},
}


def _tree(name):
    return ast.parse((INFER / name).read_text(encoding='utf-8'))


def test_harness_is_stdlib_only():
    imported = set()
    for node in ast.walk(_tree('_harness.py')):
        if isinstance(node, ast.Import):
            imported.update(a.name.split('.')[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported.add((node.module or '').split('.')[0])
    assert imported <= ALLOWED_IMPORTS, imported - ALLOWED_IMPORTS


def test_harness_defines_each_factored_helper_once():
    tree = _tree('_harness.py')
    defs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    expected = set().union(*FACTORED.values())
    assert set(defs) == expected
    assert len(defs) == len(set(defs))


def test_factored_scripts_import_and_do_not_redefine():
    for fname, names in FACTORED.items():
        tree = _tree(fname)
        imported = set()
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == '_harness':
                imported.update(a.name for a in node.names)
        redefined = {n.name for n in tree.body
                     if isinstance(n, ast.FunctionDef)} & names
        assert names <= imported, (fname, names - imported)
        assert not redefined, (fname, redefined)
