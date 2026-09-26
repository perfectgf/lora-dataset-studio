"""Local history deletion must preserve unresolved provider rental identities."""
import json
from types import SimpleNamespace

import pytest

from app.services.cloud_history import _rental_released


def _run(context=None, receipt=None):
    params = {}
    if context is not None:
        params['_lds_rental_context'] = context
    if receipt is not None:
        params['_lds_rental_cleanup'] = receipt
    return SimpleNamespace(id=42, vast_instance_id='701', vast_label='lds-42',
                           train_params=json.dumps(params))


def _released():
    return dict(version=2, run_id=42, label='lds-42', fingerprint='a' * 64,
                unique=True, pending=False, released=True, delete_pending=False,
                released_at='2026-09-26T12:00:00')


@pytest.mark.parametrize('unique', [True, False])
@pytest.mark.parametrize('delete_field', [True, False])
def test_current_provider_acknowledgement_needs_no_legacy_receipt(unique, delete_field):
    context = _released()
    context['unique'] = unique
    if not delete_field:
        context.pop('delete_pending')  # The failed-registration cleanup path omits this.
    assert _rental_released(_run(context)) is True


@pytest.mark.parametrize('changes', [
    {'version': 3}, {'version': 2.0}, {'run_id': 43}, {'run_id': '42'},
    {'label': 'lds-43'}, {'label': 'lds-42\n'}, {'label': None},
    {'fingerprint': 'a' * 63}, {'fingerprint': 'g' * 64}, {'fingerprint': None},
    {'unique': 1}, {'pending': True}, {'pending': 0},
    {'released': False}, {'released': 1}, {'delete_pending': True}, {'delete_pending': 0},
])
def test_unreleased_malformed_or_mismatched_identity_never_authorizes_deletion(changes):
    context = _released()
    context.update(changes)
    assert _rental_released(_run(context)) is False


def test_a_context_without_rental_columns_cannot_bypass_reconciliation():
    run = _run(_released())
    run.vast_instance_id = run.vast_label = None
    assert _rental_released(run) is False
    run.train_params = '{}'
    assert _rental_released(run) is True


@pytest.mark.parametrize('params', ['null', '[]', 'broken', '{"_lds_rental_context": []}'])
def test_malformed_history_fails_closed(params):
    run = _run()
    run.train_params = params
    assert _rental_released(run) is False


def test_legacy_receipt_remains_bound_to_instance_label_run_and_credential():
    context = {'version': 1, 'run_id': 42, 'fingerprint': 'a' * 64}
    receipt = {'version': 2, 'run_id': 42, 'instance_id': '701',
               'label': 'lds-42', 'credential': 'a' * 64}
    assert _rental_released(_run(context, receipt)) is True
    for field, value in [('run_id', 43), ('instance_id', '702'),
                         ('label', 'lds-43'), ('credential', 'b' * 64)]:
        assert _rental_released(_run(context, dict(receipt, **{field: value}))) is False
    assert _rental_released(_run(context)) is False
