def test_health(client):
    assert client.get('/api/health').get_json() == {'ok': True}
