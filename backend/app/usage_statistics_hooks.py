"""Explicit operation allowlist, never URLs, request bodies, or exception text."""
from flask import g, request

from .usage_statistics import begin_operation, finish_operation

# (feature, action, successful request means). Background work is ACCEPTED,
# not completed. Bank's worker separately reports its actual terminal outcome.
OPERATIONS = {
    'datasets.dataset_create': ('datasets', 'create', 'completed'),
    'datasets.dataset_import': ('datasets', 'import', 'completed'),
    'datasets.dataset_import_zip': ('datasets', 'import', 'completed'),
    'datasets.dataset_import_folder': ('datasets', 'import', 'completed'),
    'datasets.dataset_caption': ('datasets', 'caption', 'completed'),
    'datasets.dataset_export': ('datasets', 'export', 'completed'),
    'bank.bank_create': ('bank', 'create', 'completed'),
    'bank.bank_scan': ('bank', 'scan', 'accepted'),
    'bank.bank_caption': ('bank', 'caption', 'accepted'),
    'bank.bank_faces': ('bank', 'faces', 'accepted'),
    'bank.bank_score': ('bank', 'score', 'accepted'),
    'bank.bank_promote': ('bank', 'promote', 'accepted'),
    'training.dataset_train': ('training', 'train_start', 'accepted'),
    'training.dataset_train_enqueue': ('training', 'train_enqueue', 'accepted'),
    'training.dataset_train_continue': ('training', 'train_resume', 'accepted'),
    'setup_state.complete': ('setup', 'setup', 'completed'),
}


def install_hooks(app):
    @app.before_request
    def usage_operation_started():
        operation = OPERATIONS.get(request.endpoint)
        if operation and request.method in {'GET', 'POST'}:
            g.lds_usage_operation = (operation, begin_operation(operation[0]))

    @app.after_request
    def usage_operation_finished(response):
        tracked = getattr(g, 'lds_usage_operation', None)
        if tracked:
            (feature, action, success), ticket = tracked
            result, error = success, 'none'
            reason = 'none'
            status = response.status_code
            # Only known endpoints with small JSON responses. Strings in the
            # body are never copied or logged; these booleans describe outcome.
            data = None
            if response.is_json and not response.is_streamed and (response.content_length or 0) <= 65536:
                try:
                    data = response.get_json(silent=True)
                except (ValueError, TypeError):
                    pass
            if status >= 400 or (isinstance(data, dict) and (data.get('ok') is False or data.get('error'))):
                result = 'failed'
                error = {400: 'invalid_input', 404: 'not_found', 409: 'conflict',
                         429: 'busy', 502: 'unavailable', 503: 'unavailable',
                         504: 'timeout'}.get(status, 'operation_failed')
                error, reason = getattr(g, 'lds_usage_error', (error, 'unknown'))
            elif isinstance(data, dict) and (data.get('stopped') is True or data.get('cancelled') is True):
                result = 'cancelled'
            elif (action == 'import' and isinstance(data, dict)
                  and isinstance(data.get('failed'), (int, float)) and data['failed'] > 0):
                result = 'partial' if data.get('imported', 0) else 'failed'
                error = 'operation_failed'
            elif (action == 'caption' and isinstance(data, dict)
                  and isinstance(data.get('skipped'), (int, float)) and data['skipped'] > 0):
                result = 'partial' if data.get('captioned', 0) else 'failed'
                error = 'operation_failed'
            elif status == 202:
                result = 'accepted'
            if error != 'none' and reason == 'none':
                reason = 'unknown'
            finish_operation(ticket, feature, action, result, error, reason)
        return response
