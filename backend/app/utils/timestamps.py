"""One honest UTC timestamp string, for the "when did this pass last run" fields.

`datetime.utcnow()` is deprecated (it returns a NAIVE datetime that merely claims
to be UTC) and each backfill had its own copy of
``datetime.utcnow().isoformat(timespec='seconds') + 'Z'``.

The database columns and the age/duration arithmetic stay NAIVE on purpose:
those values are compared against datetimes SQLite already holds as naive,
and an aware object there raises ``can't subtract offset-naive and
offset-aware datetimes`` on data that is already on disk in every install.
``naive_utcnow`` below is that contract without the deprecation: the same
naive UTC instant ``datetime.utcnow()`` produced, minus the warning. The
aware migration remains a real one and still does not belong here.

The output is byte-for-byte what the backfills produced before (``Z`` suffix, no
offset, second precision), because it is persisted in report payloads users can
already have.
"""
from datetime import datetime, timezone

_FORMAT = '%Y-%m-%dT%H:%M:%SZ'


def utc_stamp() -> str:
    """Current UTC instant as ``2026-07-27T09:15:04Z``."""
    return datetime.now(timezone.utc).strftime(_FORMAT)


def naive_utcnow() -> datetime:
    """Current UTC instant, NAIVE - byte-compatible with every naive UTC
    value already in the database. The one sanctioned replacement for the
    deprecated ``datetime.utcnow()`` (see the module docstring for why the
    columns must stay naive)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def naive_utcfromtimestamp(ts: float) -> datetime:
    """Epoch seconds -> NAIVE UTC datetime, the same contract as
    ``naive_utcnow`` for the deprecated ``datetime.utcfromtimestamp``
    (values are compared against naive columns)."""
    return datetime.fromtimestamp(ts, timezone.utc).replace(tzinfo=None)
