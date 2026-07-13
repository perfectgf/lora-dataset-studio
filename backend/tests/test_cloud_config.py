"""Cloud config section, VAST_API_KEY secret, cloud_training capability."""


def test_cloud_defaults_present(app):
    from app import config as cfg
    assert cfg.get('cloud.image', '').startswith('vastai/ostris-ai-toolkit:')
    assert cfg.get('cloud.ui_port') == 18675
    assert cfg.get('cloud.template_hash') == '471ed5903d8cdb8e63b0d0e50f6cd519'
    assert cfg.get('cloud.max_price_per_hour') == 0.80
    assert cfg.get('cloud.offer_scan_limit') == 100
    assert cfg.get('cloud.pod_overhead_minutes') == 35
    assert cfg.get('cloud.max_concurrent_runs') == 1
    assert cfg.get('cloud.min_inet_down_mbps') == 400
    assert cfg.get('cloud.min_disk_bw_mbps') == 500
    assert cfg.get('cloud.min_reliability') == 0.98
    assert cfg.get('cloud.host_blacklist_days') == 3
    assert cfg.get('cloud.ready_timeout_minutes') == 25
    # 480 (not 240): the stall watchdog is the first line of defense now,
    # the runtime cap is only the safety net behind it.
    assert cfg.get('cloud.max_runtime_minutes') == 480
    assert cfg.get('cloud.stall_timeout_minutes') == 30
    assert cfg.get('cloud.monthly_budget_usd') == 0
    assert cfg.get('cloud.disk_gb') == 60
    assert cfg.get('cloud.min_vram_gb') == {'zimage': 24, 'sdxl': 16, 'krea': 24}


def test_vast_api_key_is_a_secret(app):
    from app import config as cfg
    assert 'VAST_API_KEY' in cfg.SECRET_KEYS


def test_capability_cloud_training_off_without_key(client):
    caps = client.get('/api/capabilities').get_json()
    assert caps['cloud_training'] is False


def test_capability_cloud_training_on_with_key(client, monkeypatch):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    caps = client.get('/api/capabilities?force=1').get_json()
    assert caps['cloud_training'] is True


def test_training_visible_with_cloud_key_only(client, monkeypatch):
    # aitoolkit NOT configured, cloud key present -> panel visible
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    caps = client.get('/api/capabilities?force=1').get_json()
    assert caps['aitoolkit']['valid'] is False
    assert caps['training_visible'] is True


def test_settings_test_target_vast_no_key(client):
    r = client.post('/api/settings/test/vast')
    assert r.status_code == 200
    body = r.get_json()
    assert body['ok'] is False
    assert 'key' in body['detail'].lower()


def test_settings_test_target_vast_with_key(client, monkeypatch):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    calls = {}

    def fake_get(url, headers=None, timeout=None):
        calls['url'] = url
        calls['auth'] = headers.get('Authorization')

        class R:
            status_code = 200

            @staticmethod
            def json():
                return {'email': 'user@example.com'}
        return R()

    monkeypatch.setattr('app.capabilities.requests.get', fake_get)
    body = client.post('/api/settings/test/vast').get_json()
    assert body['ok'] is True
    assert calls['auth'] == 'Bearer k-test'
    assert 'console.vast.ai' in calls['url']
