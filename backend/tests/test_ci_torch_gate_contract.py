"""The CI torch gate's hardcoded path patterns must keep pointing at real files.

.github/workflows/ci.yml decides whether to run the heavy torch jobs from a
hardcoded list of torch-sensitive path regexes - written TWICE (a bash copy and
a PowerShell copy). Two failure modes rot silently: the copies drift apart (one
OS gates differently from the other), and a renamed or deleted file leaves a
dead alternative in the enumeration, gating nothing while reading as covered.
Neither shows up in CI itself - a dead pattern never fires, which looks exactly
like "no torch-sensitive change".

This contract makes both rots loud: the two lists must be identical, and every
pattern - every ALTERNATIVE of an (a|b|c) enumeration, not just the pattern as
a whole - must still match at least one tracked file. It deliberately does NOT
decide which paths belong on the list: that is the gate's own judgement call.
"""
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]
CI = REPO / '.github' / 'workflows' / 'ci.yml'


def _pattern_block(text, opener):
    start = text.index(opener) + len(opener)
    block = text[start:text.index(')', start)]
    return re.findall(r"'([^']+)'", block)


def _both_lists():
    text = CI.read_text(encoding='utf-8')
    bash = _pattern_block(text, 'torch_path_patterns=(')
    ps = _pattern_block(text, '$torchPathPatterns = @(')
    assert bash and ps, 'could not locate the two pattern lists in ci.yml'
    return bash, ps


def _repo_paths():
    paths = []
    for root in ('backend', '.github'):
        for p in (REPO / root).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                paths.append(p.relative_to(REPO).as_posix())
    return paths


def _expand_alternatives(pattern):
    """Each alternative of a single (a|b|c) group becomes its own pattern, so a
    dead name inside an enumeration fails instead of hiding behind its living
    siblings. Patterns without a group (or with more than one) stay whole."""
    groups = re.findall(r'\(([^()]+\|[^()]+)\)', pattern)
    if len(groups) != 1:
        return [pattern]
    whole = '(' + groups[0] + ')'
    return [pattern.replace(whole, '(' + alt + ')') for alt in groups[0].split('|')]


def test_the_bash_and_powershell_lists_are_identical():
    bash, ps = _both_lists()
    assert bash == ps, 'ci.yml carries two torch-pattern lists and they diverged'


def test_every_pattern_and_every_alternative_matches_a_real_file():
    bash, _ = _both_lists()
    paths = _repo_paths()
    dead = []
    for pattern in bash:
        for variant in _expand_alternatives(pattern):
            rx = re.compile(variant)
            if not any(rx.search(p) for p in paths):
                dead.append(variant)
    assert not dead, (
        'dead torch-gate pattern(s) in ci.yml - the file moved or was renamed, '
        f'update the list (BOTH copies): {dead}')
