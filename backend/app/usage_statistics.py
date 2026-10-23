"""Opt-in product statistics. Closed vocabulary, no automatic content capture.

One process owns an LDS data folder. Preferences and the small bounded outbox
are atomically replaced together; no user database migration is needed. A
dedicated worker performs HTTPS, never an application request or GPU worker.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import uuid

from .version import APP_VERSION

POLICY_VERSION = 1
RETENTION_POLICY = 'posthog_free_plan_v1'
COLLECTOR = 'https://eu.i.posthog.com/batch/'
MAX_EVENTS = 500
MAX_AGE = 7 * 86400
BATCH_SIZE = 50
MAX_ATTEMPTS = 5
FEATURES = frozenset({
    'datasets', 'bank', 'training', 'studio', 'gallery', 'setup',
    'plugins', 'settings', 'guide',
})
ACTIONS = frozenset({
    'create', 'import', 'caption', 'export', 'generate', 'train',
    'scan', 'faces', 'score', 'framing', 'watermark', 'promote', 'setup',
    'semantic_index', 'semantic_dedup', 'text', 'angles', 'medium', 'pipeline',
    'train_start', 'train_resume', 'train_enqueue', 'train_run',
})
RESULTS = frozenset({'completed', 'accepted', 'partial', 'failed', 'cancelled'})
ERRORS = frozenset({
    'none', 'invalid_input', 'not_found', 'busy', 'unavailable',
    'out_of_memory', 'timeout', 'dependency_missing', 'operation_failed',
    'conflict',
})
ERROR_REASONS = frozenset({
    'none', 'unknown', 'training_already_running', 'caption_backend_unavailable',
    'gpu_busy', 'dataset_busy', 'bank_busy', 'ollama_fence_blocked',
    'bank_name_required', 'bank_folder_unavailable', 'bank_dataset_folder_conflict',
    'bank_image_limit', 'training_process_failed', 'training_resume_failed',
})
DURATIONS = ('under_1s', '1_to_10s', '10_to_60s', '1_to_10m', 'over_10m')
EVENTS = frozenset({'lds_active_day', 'lds_feature_used', 'lds_action_result'})


def _iso(stamp):
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat()


def _day(stamp):
    return _iso(stamp)[:10]


def _uuid(value):
    try:
        return str(uuid.UUID(str(value))) == value
    except (ValueError, TypeError, AttributeError):
        return False


def _empty():
    return {'choice': None, 'installation_id': None, 'generation': None,
            'binding': None, 'events': [], 'active_day': None,
            'first_active_day': None, 'feature_days': {}, 'last_sent_at': None}


def _valid_row(row):
    """Validate disk state again at the send boundary, not just incoming calls."""
    if not isinstance(row, dict) or not _uuid(row.get('uuid')):
        return False
    if row.get('event') not in EVENTS:
        return False
    if type(row.get('time')) not in (int, float) or not 0 < row['time'] < 1e11:
        return False
    if type(row.get('attempts')) is not int or not 0 <= row['attempts'] < MAX_ATTEMPTS:
        return False
    props = row.get('properties')
    if not isinstance(props, dict):
        return False
    if (not isinstance(row.get('version'), str)
            or not re.fullmatch(r'[0-9.]{1,30}', row['version'])
            or not isinstance(row.get('first_active_day'), str)
            or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', row['first_active_day'])
            or row.get('environment') not in {'production', 'test'}
            or row.get('os') not in {'windows', 'linux', 'macos', 'other'}):
        return False
    expected = {'feature'} if row['event'] == 'lds_feature_used' else set()
    if row['event'] == 'lds_action_result':
        expected = {'feature', 'action', 'result', 'error_code', 'duration_bucket'}
        # Already queued events from older builds remain valid after an update.
        if 'error_reason' in props:
            expected.add('error_reason')
    return (set(props) == expected and all(isinstance(value, str) for value in props.values())
            and ('feature' not in props or props['feature'] in FEATURES)
            and ('action' not in props or props['action'] in ACTIONS)
            and ('result' not in props or props['result'] in RESULTS)
            and ('error_code' not in props or props['error_code'] in ERRORS)
            and ('error_reason' not in props or props['error_reason'] in ERROR_REASONS)
            and ('duration_bucket' not in props or props['duration_bucket'] in DURATIONS))


def _post_batch(payload):
    import requests
    # No .netrc credentials, redirects, cookies or application auth headers.
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(
            COLLECTOR, json=payload, timeout=(3, 5), allow_redirects=False,
            stream=True, headers={'User-Agent': 'LDS-Usage-Statistics/1'})
        try:
            return 200 <= response.status_code < 300
        finally:
            response.close()


class UsageStatistics:
    def __init__(self, data_dir, *, project_token='', environment='production',
                 start_worker=True, sender=None, clock=None):
        self.path = Path(data_dir) / 'usage-statistics.json'
        self.token = project_token
        self.environment = environment
        self.configured = bool(
            isinstance(project_token, str)
            and re.fullmatch(r'phc_[A-Za-z0-9_-]{10,200}', project_token)
            and environment in {'production', 'test'})
        self.binding = hashlib.sha256(
            f'{project_token}:{POLICY_VERSION}:{RETENTION_POLICY}:{environment}'.encode()).hexdigest()
        self._lock = threading.RLock()
        self._send_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._automatic = start_worker
        self._sender = sender or _post_batch
        self._clock = clock or time.time
        self._state = _empty()
        self._next_send = 0
        self._failures = 0
        self._context = {
            'schema_version': POLICY_VERSION,
            '$process_person_profile': False,
            '$geoip_disable': True,
        }
        self._load()
        if self.enabled:
            self._ensure_worker()

    def _load(self):
        try:
            if self.path.stat().st_size > 1024 * 1024:
                return
            saved = json.loads(self.path.read_text(encoding='utf-8'))
            if not isinstance(saved, dict):
                return
            if saved.get('choice') == 'disabled':
                self._state['choice'] = 'disabled'
            elif (saved.get('choice') == 'enabled' and self.configured
                  and saved.get('binding') == self.binding
                  and _uuid(saved.get('installation_id'))
                  and _uuid(saved.get('generation'))):
                self._state.update({key: saved.get(key) for key in (
                    'choice', 'installation_id', 'generation', 'binding')})
                now = self._clock()
                self._state['events'] = [row for row in saved.get('events', [])
                                         if _valid_row(row) and now - MAX_AGE < row['time'] <= now][-MAX_EVENTS:]
                # Dedupe metadata is local only, never transmitted.
                today = _day(now)
                self._state['active_day'] = today if saved.get('active_day') == today else None
                first = saved.get('first_active_day')
                if isinstance(first, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}', first):
                    self._state['first_active_day'] = first
                days = saved.get('feature_days', {})
                self._state['feature_days'] = {
                    feature: today for feature in FEATURES
                    if isinstance(days, dict) and days.get(feature) == today}
                last = saved.get('last_sent_at')
                if isinstance(last, str) and re.fullmatch(r'[0-9T:+.\-]{20,40}', last):
                    self._state['last_sent_at'] = last
        except (OSError, ValueError, TypeError, KeyError):
            # Corrupt/unreadable consent must never turn collection on.
            self._state = _empty()

    @property
    def enabled(self):
        return self.configured and self._state['choice'] == 'enabled'

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        try:
            tmp.write_text(json.dumps(self._state, separators=(',', ':')), encoding='utf-8')
            os.replace(tmp, self.path)
        finally:
            tmp.unlink(missing_ok=True)

    def status(self):
        with self._lock:
            return {'configured': self.configured, 'enabled': self.enabled,
                    'choice': self._state['choice'], 'policy_version': POLICY_VERSION,
                    'recipient': 'PostHog Cloud EU', 'retention_policy': RETENTION_POLICY,
                    'queued_events': len(self._state['events']),
                    'last_sent_at': self._state['last_sent_at']}

    def set_enabled(self, enabled):
        if enabled and not self.configured:
            raise ValueError('Usage statistics are not available in this build.')
        # Serialize opt-out against HTTPS. Once this call returns, no earlier
        # batch is still being sent. It never changes application operations.
        with self._send_lock, self._lock:
            previous = self._state
            if enabled and self.enabled:
                return self.status()
            self._state = _empty()
            self._state['choice'] = 'enabled' if enabled else 'disabled'
            if enabled:
                self._state.update(installation_id=str(uuid.uuid4()),
                                   generation=str(uuid.uuid4()), binding=self.binding)
            try:
                self._save()
            except OSError:
                # A failed opt-out stops THIS process too, but the API reports
                # failure so it cannot promise persistence after restart.
                if enabled:
                    self._state = previous
                raise
            self._next_send = 0
            self._failures = 0
        if enabled:
            self._ensure_worker()
        return self.status()

    def consent_generation(self):
        with self._lock:
            return self._state['generation'] if self.enabled else None

    def _queue(self, event, properties, now):
        if not self._state['first_active_day']:
            self._state['first_active_day'] = _day(now)
        self._state['events'].append({
            'uuid': str(uuid.uuid4()), 'event': event, 'properties': properties,
            'version': APP_VERSION,
            'environment': self.environment,
            'first_active_day': self._state['first_active_day'],
            'os': {'win32': 'windows', 'linux': 'linux', 'darwin': 'macos'}.get(sys.platform, 'other'),
            'time': now, 'attempts': 0})
        self._state['events'] = [row for row in self._state['events']
                                 if row['time'] > now - MAX_AGE][-MAX_EVENTS:]

    def _activity(self, feature, now):
        today = _day(now)
        if self._state['active_day'] != today:
            self._queue('lds_active_day', {}, now)
            self._state['active_day'] = today
        if self._state['feature_days'].get(feature) != today:
            self._queue('lds_feature_used', {'feature': feature}, now)
            self._state['feature_days'][feature] = today

    def activity(self, feature):
        with self._lock:
            if not self.enabled or feature not in FEATURES:
                return
            self._activity(feature, self._clock())
            self._save()

    def operation(self, generation, feature, action, result, *, seconds=0,
                  error_code='none', error_reason='none'):
        with self._lock:
            if (not self.enabled or generation is None
                    or generation != self._state['generation']
                    or feature not in FEATURES or action not in ACTIONS
                    or result not in RESULTS or error_code not in ERRORS
                    or error_reason not in ERROR_REASONS):
                return
            bucket = DURATIONS[sum(seconds >= threshold for threshold in (1, 10, 60, 600))]
            self._queue('lds_action_result', {
                'feature': feature, 'action': action, 'result': result,
                'error_code': error_code, 'error_reason': error_reason,
                'duration_bucket': bucket}, self._clock())
            self._save()

    def _ensure_worker(self):
        if not self._automatic:
            return
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, daemon=True,
                                                name='lds-usage-statistics')
                self._thread.start()

    def _run(self):
        while not self._stop.wait(60):
            try:
                self.flush_once()
            except Exception:
                # Analytics must never take down the worker or log payloads.
                pass

    def close(self):
        self._stop.set()

    def flush_once(self):
        with self._send_lock:
            with self._lock:
                now = self._clock()
                if not self.enabled or now < self._next_send:
                    return False
                self._state['events'] = [row for row in self._state['events']
                                         if _valid_row(row) and now - MAX_AGE < row['time'] <= now]
                rows = self._state['events'][:BATCH_SIZE]
                if not rows:
                    return False
                ids = {row['uuid'] for row in rows}
                batch = [{'uuid': row['uuid'], 'event': row['event'],
                          'distinct_id': self._state['installation_id'],
                          'timestamp': _iso(row['time']),
                          'properties': {**self._context, 'app_version': row['version'],
                                         'environment': row['environment'],
                                         'first_active_day': row['first_active_day'],
                                         'os': row['os'], **row['properties']}}
                         for row in rows]
                # Persist the attempt BEFORE I/O: process crashes cannot cause
                # infinite retries, and UUIDs stay the same across retries.
                for row in rows:
                    row['attempts'] += 1
                self._save()
            try:
                success = bool(self._sender({'api_key': self.token, 'batch': batch}))
            except Exception:
                success = False
            with self._lock:
                if success:
                    self._state['events'] = [row for row in self._state['events'] if row['uuid'] not in ids]
                    self._state['last_sent_at'] = _iso(now)
                    self._failures = 0
                else:
                    self._failures += 1
                    self._state['events'] = [row for row in self._state['events'] if row['attempts'] < MAX_ATTEMPTS]
                self._next_send = now + min(3600, 60 * 2 ** min(self._failures, 6))
                self._save()
                return success


def install(app):
    """A missing collector leaves both network collection and the prompt off."""
    settings = {}
    try:
        settings = json.loads(Path(__file__).with_name('usage_statistics_config.json').read_text(encoding='utf-8'))
        if not isinstance(settings, dict):
            settings = {}
    except (OSError, ValueError):
        pass
    token = os.environ.get('LDS_USAGE_PROJECT_TOKEN', settings.get('project_token', ''))
    if os.environ.get('LDS_USAGE_DISABLED', '').lower() in {'1', 'true', 'yes'}:
        token = ''
    app.extensions['lds_usage_statistics'] = UsageStatistics(
        app.config['LDS_DATA_DIR'], project_token=token,
        environment=os.environ.get('LDS_USAGE_ENVIRONMENT', 'production'),
        start_worker=not app.config.get('TESTING'))
    from .usage_statistics_hooks import install_hooks
    install_hooks(app)


def current():
    from flask import current_app, has_app_context
    return current_app.extensions.get('lds_usage_statistics') if has_app_context() else None


def begin_operation(feature):
    """Capture consent at START: an opt-in never uploads older work."""
    try:
        service = current()
        if service is not None:
            generation = service.consent_generation()
            if generation:
                service.activity(feature)
                return (service, generation, time.monotonic())
    except Exception:
        pass
    return None


def error_categories(exc):
    """Never inspect or forward exception messages, paths or model names."""
    code = getattr(exc, 'usage_error_code', None)
    reason = getattr(exc, 'usage_error_reason', None)
    if not isinstance(code, str) or code not in ERRORS or code == 'none':
        code = ('invalid_input' if isinstance(exc, ValueError) else
                'timeout' if isinstance(exc, TimeoutError) else 'operation_failed')
    if not isinstance(reason, str) or reason not in ERROR_REASONS or reason == 'none':
        reason = 'unknown'
    return code, reason


def record_operation_error(exc):
    from flask import g, has_request_context
    if has_request_context():
        g.lds_usage_error = error_categories(exc)


def finish_operation(ticket, feature, action, result, error_code='none', error_reason='none'):
    if ticket is None:
        return
    try:
        service, generation, started = ticket
        service.operation(generation, feature, action, result,
                          seconds=max(0, time.monotonic() - started), error_code=error_code,
                          error_reason=error_reason)
    except Exception:
        pass
