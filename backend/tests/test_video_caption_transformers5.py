"""Video captioning under transformers 5 — the pinned shape of the fix.

2026-09-01: every shot of a Describe pass came back caption_state='error' with
no reason anywhere. The interpreter the worker BORROWS (ComfyUI's python) had
transformers 5.3 on its own site-packages; `-s` (the isolation this project
applies to borrowed interpreters) made it visible, and 5.x's Qwen3-VL rope
indexing pulls one `video_grid_thw` row per timestamped span while the
processor emits ONE row per clip — `StopIteration`, swallowed per shot.

The infer script is not importable here (it runs in another interpreter), so
these tests hold its SOURCE to the fix, the same way test_video_caption_model
holds it to the model handshake.
"""
import ast
from pathlib import Path

INFER = Path(__file__).resolve().parents[1] / 'infer' / 'video_caption_infer.py'
WORKER = Path(__file__).resolve().parents[1] / 'app' / 'services' / 'video_caption_worker.py'


def _code_without_docstrings(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    return ast.unparse(tree)


def test_the_video_grid_is_reconciled_before_generate():
    code = _code_without_docstrings(INFER)
    # The helper exists, and generate() only ever sees reconciled inputs.
    assert 'def _reconcile_video_grid(inputs)' in code
    before, _, after = code.partition('inputs = _reconcile_video_grid(inputs)')
    assert before and 'model.generate(' in after
    assert 'model.generate(' not in before.split('def _caption(')[-1]


def test_the_reconcile_expands_one_row_per_span_and_only_then():
    code = _code_without_docstrings(INFER)
    # One row per temporal patch, `[1, h, w]` × t …
    assert '[[1, h, w]] * t' in code
    # … only when the spans outnumber the rows (4.57 already agrees with
    # itself and must be left alone), and only for the single-clip shape.
    assert 'grid.shape[0] >= spans' in code
    assert "convert_tokens_to_ids('<|vision_start|>')" in code
    # A reconcile that fails must never mask the real error.
    assert 'grid reconcile skipped' in code


def test_the_worker_says_why_a_shot_was_refused():
    """The reason used to die at `return ''` — a whole bench read "0 words"
    with nothing to act on. It is logged now, and still absorbed."""
    src = WORKER.read_text(encoding='utf-8')
    assert 'caption worker refused a shot' in src
    assert 'logger = logging.getLogger(__name__)' in src
