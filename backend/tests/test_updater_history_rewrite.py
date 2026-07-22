"""An upstream history REWRITE (rebase / filter-branch on the remote) breaks in-app
updates permanently: every commit gets a new SHA, so no local commit is an ancestor
of the remote branch and `git pull --ff-only` can never fast-forward again. The
checkout is fine — it simply can never update without manual git surgery.

These tests drive REAL git repositories rather than a faked _git, because the whole
question is what git actually does to commit identity across a rewrite; a stub would
just re-assert our own assumptions.
"""
import shutil
import subprocess

import pytest

from app.services import updater

pytestmark = pytest.mark.skipif(shutil.which('git') is None, reason='git not on PATH')


def _run(cwd, *args):
    return subprocess.run(('git',) + args, cwd=str(cwd), capture_output=True,
                          text=True, timeout=60)


def _commit(repo, name, text, message):
    (repo / name).write_text(text, encoding='utf-8')
    _run(repo, 'add', '-A')
    _run(repo, 'commit', '-m', message)


@pytest.fixture()
def checkout(tmp_path):
    """An 'upstream' repo plus a clone of it, the way a user's install looks."""
    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    _run(upstream, 'init', '-b', 'main')
    _run(upstream, 'config', 'user.email', 't@example.com')
    _run(upstream, 'config', 'user.name', 'Test')
    _commit(upstream, 'a.txt', 'one', 'first\n\nCo-Authored-By: Someone <s@example.com>')
    _commit(upstream, 'b.txt', 'two', 'second\n\nCo-Authored-By: Someone <s@example.com>')

    clone = tmp_path / 'clone'
    _run(tmp_path, 'clone', str(upstream), str(clone))
    _run(clone, 'config', 'user.email', 't@example.com')
    _run(clone, 'config', 'user.name', 'Test')
    return upstream, clone


def _rewrite_upstream(upstream):
    """Strip a trailer from every message — same shape as a filter-branch scrub:
    identical trees, brand-new commit SHAs."""
    _run(upstream, 'filter-branch', '-f', '--msg-filter',
         'sed "/^Co-Authored-By:/d"', '--', '--all')


def test_pull_ff_only_really_does_break_after_a_rewrite(checkout):
    """The premise, verified rather than assumed — otherwise the fix guards nothing."""
    upstream, clone = checkout
    _rewrite_upstream(upstream)
    _run(clone, 'fetch', 'origin', 'main')
    pull = _run(clone, 'pull', '--ff-only', 'origin', 'main')
    assert pull.returncode != 0


def test_apply_update_recovers_from_a_rewritten_upstream(checkout):
    upstream, clone = checkout
    before = _run(clone, 'rev-parse', 'HEAD').stdout.strip()
    _rewrite_upstream(upstream)

    out = updater.apply_update(root=clone)
    assert out['ok'] is True and out['changed'] is True
    # landed exactly on the rewritten remote tip...
    _run(clone, 'fetch', 'origin', 'main')
    assert (_run(clone, 'rev-parse', 'HEAD').stdout.strip()
            == _run(clone, 'rev-parse', 'origin/main').stdout.strip())
    assert _run(clone, 'rev-parse', 'HEAD').stdout.strip() != before
    # ...and the FILES are untouched: a rewrite changes messages, not content
    assert (clone / 'a.txt').read_text() == 'one'
    assert (clone / 'b.txt').read_text() == 'two'
    assert 'Co-Authored-By' not in _run(clone, 'log', '--format=%B').stdout


def test_recovery_refuses_to_discard_uncommitted_work(checkout):
    """A failing update is better than a destroyed edit."""
    upstream, clone = checkout
    _rewrite_upstream(upstream)
    (clone / 'a.txt').write_text('locally edited', encoding='utf-8')

    out = updater.apply_update(root=clone)
    assert out['ok'] is False
    assert (clone / 'a.txt').read_text() == 'locally edited'


def test_recovery_refuses_when_the_user_has_real_local_commits(checkout):
    """Local work makes HEAD's tree absent from the remote — the very case the tree
    check exists to catch, since commit COUNTS cannot tell it from a rewrite."""
    upstream, clone = checkout
    _rewrite_upstream(upstream)
    _commit(clone, 'mine.txt', 'my work', 'my local commit')

    out = updater.apply_update(root=clone)
    assert out['ok'] is False
    assert (clone / 'mine.txt').exists()          # nothing destroyed


def test_untracked_files_do_not_block_the_recovery(checkout):
    """reset --hard never removes untracked files, so they must not veto the update
    (a user's stray notes.txt would otherwise strand them forever)."""
    upstream, clone = checkout
    _rewrite_upstream(upstream)
    (clone / 'notes.txt').write_text('scratch', encoding='utf-8')

    out = updater.apply_update(root=clone)
    assert out['ok'] is True
    assert (clone / 'notes.txt').read_text() == 'scratch'


def test_a_normal_fast_forward_update_still_works(checkout):
    """The ordinary path must be untouched by the recovery branch."""
    upstream, clone = checkout
    _commit(upstream, 'c.txt', 'three', 'third')

    out = updater.apply_update(root=clone)
    assert out['ok'] is True and out['changed'] is True
    assert (clone / 'c.txt').read_text() == 'three'


def test_status_does_not_claim_a_rewrite_is_hundreds_of_commits_behind(checkout):
    """Someone perfectly up to date must not be told the whole history is pending
    just because the remote was rewritten — the count is re-measured by content."""
    upstream, clone = checkout
    _rewrite_upstream(upstream)

    s = updater.git_update_status(root=clone)
    assert s['behind'] == 0 and s['update_available'] is False
    assert s.get('history_rewritten') is True


def test_status_still_counts_real_pending_commits_after_a_rewrite(checkout):
    """And a genuine new commit on top of a rewrite still reads as exactly one."""
    upstream, clone = checkout
    _rewrite_upstream(upstream)
    _commit(upstream, 'c.txt', 'three', 'third')

    s = updater.git_update_status(root=clone)
    assert s['behind'] == 1 and s['update_available'] is True
