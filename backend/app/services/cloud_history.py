"""Delete local video-run history without loading or contacting its provider."""
import json
import logging
import os
from pathlib import Path
import re
import shutil
import stat
from types import SimpleNamespace

from sqlalchemy import exists, select

from .. import config as cfg
from ..extensions import db
from . import cloud_run_dataset as crd, cloud_training as run_graph

log = logging.getLogger(__name__)
_TERMINAL = frozenset({'done', 'completed', 'stopped', 'error', 'failed', 'error_pod_kept'})


def _linked(path):
    try:
        return path.is_symlink() or bool(getattr(path.lstat(), 'st_file_attributes', 0)
                                        & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    except FileNotFoundError:
        return False


def _rental_released(run):
    """The server receipt is bound to this run and its last rental identity.

    Cloud keeps the old instance id for history. A terminal status alone is
    not proof that DELETE succeeded; only Cloud's observed cleanup can stamp
    this receipt. It is never copied from launch/import request parameters.
    """
    try:
        params = json.loads(run.train_params or '{}')
        receipt = params.get('_lds_rental_cleanup')
        context = params.get('_lds_rental_context')
        if context is None:
            return not run.vast_instance_id and not run.vast_label
        fingerprint = context.get('fingerprint')
    except (TypeError, ValueError, AttributeError):
        return False
    if not isinstance(fingerprint, str) or not re.fullmatch(r'[a-f0-9]{64}', fingerprint):
        return False
    if context.get('version') == 2:
        # Current Cloud writes its DELETE acknowledgement into the durable
        # identity itself. Read it without importing the optional provider;
        # pending CREATE/DELETE and mismatched run/label cannot prove release.
        # `unique=False` is a valid, observed-and-migrated legacy rental.
        return (type(context.get('version')) is int
                and type(context.get('run_id')) is int and context['run_id'] == run.id
                and isinstance(context.get('label'), str)
                and re.fullmatch(r'lds-[0-9]+', context['label']) is not None
                and context['label'] == run.vast_label
                and type(context.get('unique')) is bool
                and context.get('pending') is False
                and context.get('released') is True
                and context.get('delete_pending', False) is False)
    return (context == {'version': 1, 'run_id': run.id, 'fingerprint': fingerprint}
            and receipt == {'version': 2, 'run_id': run.id,
                            'instance_id': run.vast_instance_id, 'label': run.vast_label,
                            'credential': fingerprint})


def _owned_directory(root, run_id, supplied=None):
    """A run directory is exactly run_<id>, under its configured storage root."""
    root = Path(root).absolute()
    expected = root / f'run_{int(run_id)}'
    path = Path(supplied).absolute() if supplied else expected
    if (os.path.normcase(str(path)) != os.path.normcase(str(expected))
            or _linked(path)
            or path.resolve().parent != root.resolve()):
        raise RuntimeError('The run files are outside their owned storage folder. '
                           'Restore their storage location before deleting this history.')
    if path.exists() and not path.is_dir():
        raise RuntimeError('The run storage location is not a folder.')
    # Do not let recursive cleanup follow a moved child/junction either.
    for folder, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            child = Path(folder) / name
            if _linked(child):
                raise RuntimeError('The run storage contains a linked file or folder. '
                                   'Restore its storage location before deleting this history.')
    return path


def _capture_directory(root, run_id, supplied=None):
    logical_root = Path(root).absolute()
    physical_root = logical_root.resolve()
    path = _owned_directory(logical_root, run_id, supplied).resolve()
    _owned_directory(physical_root, run_id, path)
    return logical_root, physical_root, path


def _recheck_directory(captured, current_root, run_id):
    logical_root, physical_root, path = captured
    current_root = Path(current_root).absolute()
    if (current_root != logical_root or current_root.resolve() != physical_root
            or physical_root.resolve() != physical_root):
        raise RuntimeError('The run storage root changed during history deletion.')
    return _owned_directory(physical_root, run_id, path)


def delete_video_run(user_id, dataset_id, run_id):
    """Delete only a terminal, released run of the caller's video dataset.

    Persistent SQL tables supply ownership and cleanup facts while Cloud,
    Video or Civitai are OFF. No provider import, credential or network read.
    The conditional DELETE rechecks the facts inside the DB transaction;
    publication links go with it. Files are removed only after commit.
    """
    from lds_sdk._legacy_schema import cloud_training_run as runs
    from lds_sdk._video_schema import video_dataset as datasets
    from lds_sdk.cloud_runs import delete_publication_links
    owner = exists(select(datasets.c.id).where(
        datasets.c.id == int(dataset_id), datasets.c.user_id == str(user_id)))
    row = db.session.execute(select(runs).where(runs.c.id == int(run_id))).mappings().first()
    run = SimpleNamespace(**row) if row is not None else None
    if (run is None or not crd.owns(run, dataset_id, crd.VIDEO)
            or not db.session.execute(select(owner)).scalar()):
        raise LookupError('unknown video cloud run')
    if run.status not in _TERMINAL:
        raise RuntimeError('that run is still on a pod — stop it before deleting it')
    if not _rental_released(run):
        raise RuntimeError('The previous rental cleanup is not confirmed. '
                           'Enable Cloud and let its recovery confirm release before deleting this run.')
    store_location = _capture_directory(cfg.checkpoints_root(create=False), run.id)
    staging_location = (_capture_directory(cfg.cloud_runs_root(create=False), run.id, run.staging_dir)
                        if run.staging_dir else None)
    staging = staging_location[2] if staging_location else None
    files = run_graph.run_checkpoint_files(run)
    total_bytes = 0
    for path in files.values():
        try:
            total_bytes += os.path.getsize(path)
        except OSError:
            pass  # A checkpoint already removed from disk does not block history cleanup.
    staging_files = ([p for p in staging.iterdir() if p.is_file()
                      and p.name.lower().endswith('.safetensors')] if staging and staging.exists() else [])
    # Prevent a changed ownership/status/rental/path from authorizing the delete
    # with an earlier snapshot. SQL writes, including publication cleanup, are
    # all rolled back if any precondition changed or the commit failed.
    stable = ('dataset_id', 'dataset_table', 'status', 'vast_instance_id',
              'vast_label', 'train_params', 'staging_dir', 'checkpoint_local_path')
    try:
        result = db.session.execute(runs.delete().where(
            runs.c.id == run.id, owner,
            *(runs.c[name] == getattr(run, name) for name in stable)))
        if result.rowcount != 1:
            raise RuntimeError('The run changed while its history was being deleted. Retry the operation.')
        delete_publication_links(db.session, run.id)
        db.session.commit()
    except Exception:
        db.session.rollback()
        raise
    try:
        # Revalidate both captured physical destinations before any cleanup.
        # Changed config, root junctions or child links leave files behind.
        # This is a precondition check, not an OS-level atomic filesystem lock.
        store = _recheck_directory(store_location, cfg.checkpoints_root(create=False), run.id)
        if staging_location:
            staging = _recheck_directory(staging_location, cfg.cloud_runs_root(create=False), run.id)
        if store.exists():
            shutil.rmtree(store)
        if staging:
            for path in staging_files:
                (staging / path.name).unlink(missing_ok=True)
    except (OSError, RuntimeError):
        log.warning('Could not remove all local files of deleted cloud video run %s', run.id)
    return {'deleted': run.id, 'files': len(files), 'bytes': total_bytes}
