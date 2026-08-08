"""GET /api/extensions/ publishes the manifest of loaded extensions.
Empty list on every normal install — the frontend treats that as 'no-op'.
"""


def test_the_manifest_endpoint_answers_an_empty_list_by_default(client):
    resp = client.get('/api/extensions/')
    assert resp.status_code == 200
    assert resp.get_json() == {'extensions': []}
