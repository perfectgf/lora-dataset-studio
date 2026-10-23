"""Observe committed core Studio/Cloud states, without changing their workers.

Run/job IDs stay in this bounded process-local map; only closed categories go
to the recorder. Old/recovered runs and rolled-back work produce no outcome.
The public Cloud host SDK exposes these same mapped classes to the plugin.
"""
from collections import OrderedDict
import json
import threading
import time

from flask import current_app, has_app_context
from sqlalchemy import event, inspect
from sqlalchemy.orm import Session

from .usage_statistics import finish_operation, operation_context

_PENDING = 'lds_usage_committed_states'
_EXTENSION = 'lds_usage_lifecycle'
_MAX_TRACKED = 4096
_LISTENERS_LOCK = threading.Lock()
_STUDIO_PHASES = {'pending': 'queued', 'processing': 'preparation',
                  'sent_to_comfy': 'generation', 'completed': 'saving'}
_CLOUD_PHASES = {'preparing': 'preparation', 'provisioning': 'provisioning',
                 'uploading': 'uploading', 'training': 'training',
                 'downloading': 'downloading', 'terminating': 'terminating'}


def _object_json(value):
    try:
        decoded = json.loads(value or '{}')
        return decoded if isinstance(decoded, dict) else {}
    except (ValueError, TypeError):
        return {}


class LifecycleObserver:
    def __init__(self, service):
        self.service = service
        self._lock = threading.RLock()
        self._runs = OrderedDict()

    def accept(self, kind, key, is_new, status, context, ticket):
        lane = 'studio' if kind in {'queue', 'image'} else 'cloud'
        identity = (lane, key)
        with self._lock:
            receipt = self._runs.get(identity)
            phases = _STUDIO_PHASES if lane == 'studio' else _CLOUD_PHASES
            initial = 'pending' if lane == 'studio' else 'preparing'
            if is_new and kind != 'image' and status == initial and ticket is not None:
                if ticket[1] != self.service.consent_generation():
                    return
                if receipt is None:
                    receipt = {'ticket': ticket, 'context': context, 'stage': phases[status]}
                    self._runs[identity] = receipt
                    while len(self._runs) > _MAX_TRACKED:
                        self._runs.popitem(last=False)
                    feature, action = ('studio', 'generate') if lane == 'studio' else ('training', 'train_start')
                    self.service.activity(feature)
                    finish_operation(ticket, feature, action, 'accepted', context=context)
            if receipt is None:
                return
            if status == 'deleted':
                self._runs.pop(identity, None)
                return
            if kind != 'image' and status in phases:
                receipt['stage'] = phases[status]
                # Cloud reservation params can be completed in a later commit.
                if context['model_family'] != 'unknown':
                    receipt['context'] = context
                return
            result = None
            if status == 'cancelled' or (kind == 'cloud' and status == 'stopped'):
                result = 'cancelled'
            elif kind == 'image' and status == 'done':
                result = 'completed'
            elif kind == 'cloud' and status == 'done':
                result = 'completed'
            elif status in {'failed', 'error', 'error_pod_kept'}:
                result = 'failed'
            if result is None:
                return
            self._runs.pop(identity, None)
            code, reason = 'none', 'none'
            if result == 'failed':
                code = 'operation_failed'
                reason = ('cloud_training_failed' if lane == 'cloud' else
                          'studio_result_save_failed' if receipt['stage'] == 'saving' else
                          'studio_generation_failed')
            feature, action = ('studio', 'generate') if lane == 'studio' else ('training', 'train_run')
            finish_operation(receipt['ticket'], feature, action, result, code, reason,
                             context=receipt['context'], stage=receipt['stage'])


def _capture(session, _flush_context):
    try:
        if not has_app_context():
            return
        observer = current_app.extensions.get(_EXTENSION)
        if observer is None:
            return
        generation = observer.service.consent_generation()
        if generation is None and not observer._runs:
            return
        from .models import CloudTrainingRun, ImageGenerationQueue, LoraTestImage
        transaction = session.get_nested_transaction() or session.get_transaction()
        pending = session.info.setdefault(_PENDING, [])
        objects = (*session.new, *session.dirty, *session.deleted)
        # Queue completion establishes the saving phase before image callbacks.
        for obj in sorted(objects, key=lambda row: isinstance(row, LoraTestImage)):
            if not isinstance(obj, (CloudTrainingRun, ImageGenerationQueue, LoraTestImage)):
                continue
            is_new = obj in session.new
            deleted = obj in session.deleted
            state = inspect(obj)
            if not is_new and not deleted and not state.attrs.status.history.has_changes():
                continue
            status = 'deleted' if deleted else obj.status
            if isinstance(obj, CloudTrainingRun):
                kind, key = 'cloud', obj.id
                params = _object_json(obj.train_params)
                context = operation_context(execution='cloud', model_family=params.get('train_type'))
            elif isinstance(obj, ImageGenerationQueue):
                metadata = _object_json(obj.job_metadata)
                if metadata.get('is_lora_test') is not True:
                    continue
                kind, key = 'queue', obj.job_id
                context = operation_context(execution='local', model_family=metadata.get('family'))
            else:
                kind, key = 'image', obj.job_id
                context = operation_context(execution='local')
            if key is None:
                continue
            ticket = (observer.service, generation, time.monotonic()) if is_new and generation else None
            pending.append((transaction, observer, (kind, key, is_new, status, context, ticket)))
    except Exception:
        # Diagnostics must neither fail nor log the application's transaction.
        session.info.pop(_PENDING, None)


def _committed(session):
    if session.in_nested_transaction():
        return  # A released savepoint is not a durable outer commit.
    for _transaction, observer, snapshot in session.info.pop(_PENDING, []):
        try:
            observer.accept(*snapshot)
        except Exception:
            pass


def _rolled_back(session, previous_transaction):
    def affected(transaction):
        while transaction is not None:
            if transaction is previous_transaction:
                return True
            transaction = transaction.parent
        return False
    pending = session.info.get(_PENDING)
    if pending is not None:
        session.info[_PENDING] = [entry for entry in pending if not affected(entry[0])]


def install(app):
    app.extensions[_EXTENSION] = LifecycleObserver(app.extensions['lds_usage_statistics'])
    with _LISTENERS_LOCK:
        for name, callback in (('after_flush', _capture), ('after_commit', _committed),
                               ('after_soft_rollback', _rolled_back)):
            if not event.contains(Session, name, callback):
                event.listen(Session, name, callback)
