"""Which table does a cloud run's `dataset_id` point into?

`cloud_training_run.dataset_id` has always meant a `face_dataset.id`. Once a run
can instead point at a `video_dataset.id`, two tables share ONE integer space:
face dataset #3 and video dataset #3 both exist, both resolve, and every consumer
that looks the id up without asking which table quietly serves the wrong row. No
exception is raised anywhere along that path — the run simply appears under
another dataset's name, and its checkpoints under another dataset's routes.

This module is the single place that answers the question, so that no caller has
to remember to ask it twice the same way.

WHY THE COLUMN IS CALLED `dataset_table` AND NOT `dataset_kind`
---------------------------------------------------------------
`dataset_kind` is already taken, twice, with a DIFFERENT vocabulary:
`face_dataset.kind` and `training_preset.dataset_kind` both hold
'character' / 'style' / 'concept'. A third column of the same name holding
'face' / 'video' would make a grep for `dataset_kind` return two unrelated
meanings, and a reader who assumed the familiar one would be wrong in a way that
compiles. `dataset_table` says exactly what it holds — the name of the table
`dataset_id` points into — and its values ARE those table names, so there is
nothing to look up.

The values are stored in user databases. Per the repo's rule on stored
identifiers, renaming either one later needs an alias path.
"""
from ..extensions import db

FACE = 'face_dataset'
VIDEO = 'video_dataset'

_KNOWN = (FACE, VIDEO)


def table_of(run) -> str:
    """The table this run's `dataset_id` points into.

    NULL means `face_dataset`. Not "unknown" — that is the only meaning the
    column could have had before it existed, and every row in every shipped
    database is in exactly that state. A default that applied only to new rows
    would strand all of them.

    An unrecognised value RAISES rather than falling back to face. A downgrade or
    a hand-edited row is precisely the case where guessing produces the silent
    mis-attribution this module exists to prevent."""
    value = getattr(run, 'dataset_table', None) or FACE
    if value not in _KNOWN:
        raise ValueError(
            f'cloud run {getattr(run, "id", "?")} names an unknown dataset table '
            f'{value!r} — this build cannot say which dataset it trained')
    return value


def is_video(run) -> bool:
    """True when this run trained on a video dataset. The one-line guard every
    face-only code path takes to stand down."""
    return table_of(run) == VIDEO


def owns(run, dataset_id, table=FACE) -> bool:
    """Does `run` belong to this (dataset_id, table) pair?

    The replacement for `run.dataset_id != dataset_id`, which was a complete
    ownership test only while one table existed. Two routes gate on it before
    serving a run's checkpoints, so with a shared integer space a face dataset's
    endpoint would hand out a video run's weights on a colliding id."""
    try:
        if int(run.dataset_id) != int(dataset_id) or table_of(run) != table:
            return False
    except (TypeError, ValueError, AttributeError):
        return False
    # A run whose dataset was DELETED matches nothing, ever again. SQLite
    # reuses rowids on tables without AUTOINCREMENT, so the next dataset can
    # legitimately wear the dead one's id — and this test would then hand a
    # stranger's checkpoints to it (seen live: a fresh stills set displayed a
    # deleted smoke set's run and offered its weights for download). The
    # deletion stamps the run instead of nulling the column, because the
    # column is NOT NULL in every shipped database.
    try:
        import json
        if json.loads(run.train_params or '{}').get('dataset_deleted'):
            return False
    except (TypeError, ValueError):
        pass
    return True


def dataset_row(run):
    """The dataset row this run trained on — a `FaceDataset` or a `VideoDataset`
    — or None if it has since been deleted. Best-effort by design: a caller
    describing an old run must degrade to "unknown", never to another dataset."""
    from ..models import FaceDataset, VideoDataset
    model = VideoDataset if is_video(run) else FaceDataset
    try:
        return db.session.get(model, int(run.dataset_id))
    except Exception:
        return None


def display_name(run):
    """What to call this run's dataset in a payload, or None. Same best-effort
    contract as `dataset_row`."""
    row = dataset_row(run)
    return getattr(row, 'name', None) if row is not None else None
