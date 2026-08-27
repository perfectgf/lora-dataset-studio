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
    'text_fill_infer.py': {'_log', '_emit', '_cancel_requested'},
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


def _harness_importers():
    """Every infer script whose top level imports from _harness, discovered —
    the FACTORED map is a curated view, and a curated view can miss a new file
    (text_fill_infer.py shipped outside it)."""
    for path in sorted(INFER.glob('*.py')):
        if path.name == '_harness.py':
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        if any(isinstance(n, ast.ImportFrom) and n.module == '_harness'
               for n in tree.body):
            yield path.name, tree


def test_factored_map_covers_every_importer():
    assert {name for name, _ in _harness_importers()} == set(FACTORED)


def _restores_own_directory_first(tree):
    """A top-level `sys.path.insert(...)` before the first _harness import."""
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == '_harness':
            return False
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == 'insert'
                    and isinstance(sub.func.value, ast.Attribute)
                    and sub.func.value.attr == 'path'
                    and isinstance(sub.func.value.value, ast.Name)
                    and sub.func.value.value.id == 'sys'):
                return True
    return True


def test_every_harness_importer_restores_its_own_directory_first():
    """`python script.py` normally puts the script's directory at sys.path[0],
    which is what `from _harness import …` rides on — but an embeddable
    interpreter (ComfyUI portable's python_embeded: its ._pth pins sys.path)
    skips that step, and ✨ Score pointed at one died with "No module named
    '_harness'" on every launch. Each importer must put its own directory back
    BEFORE the import — the harness cannot do it for them, being the module
    that fails to resolve; and the parent cannot either, because a ._pth
    interpreter ignores PYTHONPATH."""
    missing = [name for name, tree in _harness_importers()
               if not _restores_own_directory_first(tree)]
    assert not missing, missing
