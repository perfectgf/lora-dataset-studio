"""The 'don't re-run Setup on every visit' memory.

Two properties matter more than the plumbing and are asserted first: a machine
that has NEVER been seen working still gets the first-run wizard, and one that
HAS is never interrupted for a service that is merely stopped.
"""
import json


from app import setup_state


WORKING = {
    'configured': True,
    'engines': {'nanobanana': True, 'chatgpt': False, 'openrouter': False,
                'klein': False, 'krea': False},
    'comfyui': {'reachable': True, 'dir_valid': True},
    'ollama': {'reachable': True, 'installed': True, 'vision_model_ready': True},
    'aitoolkit': {'valid': True},
    'captioners': {'joycaption': True, 'ollama': True},
    'face_scoring': True, 'masks': True, 'watermark_inpaint': True,
    'training_visible': True, 'studio_visible': True,
}

FRESH = {
    'configured': False,
    'engines': {'nanobanana': False, 'chatgpt': False, 'openrouter': False, 'klein': False},
    'comfyui': {'reachable': False, 'dir_valid': False},
    'ollama': {'reachable': False, 'installed': False},
    'aitoolkit': {'valid': False},
    'captioners': {'joycaption': False, 'ollama': False},
    'face_scoring': False, 'masks': False, 'watermark_inpaint': False,
    'training_visible': False,
}


def _caps(**over):
    out = json.loads(json.dumps(WORKING))
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


# --- the memory itself ---------------------------------------------------------

def test_fresh_install_is_not_verified(app):
    assert setup_state.read()['verified'] is False
    assert setup_state.observe(FRESH)['verified'] is False


def test_configured_without_any_engine_is_not_verified(app):
    """config.json existing is not proof of a set-up machine — being able to
    generate an image is. Otherwise a half-finished wizard would silence itself."""
    caps = _caps(engines={'nanobanana': False})
    assert setup_state.install_works(caps) is False
    assert setup_state.observe(caps)['verified'] is False


def test_working_install_is_stamped_verified(app):
    state = setup_state.observe(WORKING)
    assert state['verified'] is True and state['verified_at']
    # ...and it survives a fresh read (a new tab / new browser session).
    assert setup_state.read()['verified'] is True


def test_verification_survives_a_later_broken_probe(app):
    """An install does not become un-set-up because a probe came back bad."""
    setup_state.observe(WORKING)
    assert setup_state.observe(FRESH)['verified'] is True


def test_retrofits_an_install_set_up_before_this_shipped(app):
    """No state file, an already-working machine: the first page load stamps it,
    so nobody has to re-run a wizard they finished weeks ago."""
    assert not (setup_state._state_path()).exists()
    assert setup_state.observe(WORKING)['verified'] is True
    assert setup_state._state_path().exists()


# --- what counts as a regression ----------------------------------------------

def test_nothing_regressed_on_a_steady_install(app):
    state = setup_state.observe(WORKING)
    assert setup_state.compare(WORKING, state) == []


def test_a_capability_that_stopped_working_is_reported(app):
    setup_state.observe(WORKING)
    broken = _caps(masks=False, face_scoring=False)
    keys = [r['key'] for r in setup_state.compare(broken)]
    assert keys == ['face_scoring', 'masks']
    assert all(r['label'] for r in setup_state.compare(broken))


def test_comfyui_or_ollama_merely_stopped_is_not_a_regression(app):
    """The whole point: these go up and down a dozen times a day. Treating them
    as failures would turn a quiet check into the nag it replaces."""
    setup_state.observe(WORKING)
    down = _caps(comfyui={'reachable': False}, ollama={'reachable': False},
                 engines={'klein': False}, studio_visible=False)
    assert setup_state.compare(down) == []


def test_never_installed_capability_is_not_a_regression(app):
    """"Not everything is installed" is the normal state of nearly every
    install; only losing something you HAD is worth an interruption."""
    partial = _caps(masks=False, watermark_inpaint=False)
    setup_state.observe(partial)
    assert setup_state.compare(partial) == []


def test_snapshot_is_a_high_water_mark(app):
    """A broken capability must not silently re-baseline: the warning has to keep
    coming back until it is fixed or explicitly dismissed."""
    setup_state.observe(WORKING)
    broken = _caps(masks=False)
    setup_state.observe(broken)                     # a re-check while still broken
    assert [r['key'] for r in setup_state.compare(broken)] == ['masks']


def test_dismiss_forgets_a_deliberate_uninstall(app):
    setup_state.observe(WORKING)
    broken = _caps(masks=False)
    setup_state.dismiss(['masks'])
    assert setup_state.compare(broken) == []


def test_unknown_keys_are_ignored_everywhere(app):
    setup_state.observe(WORKING)
    setup_state.dismiss(['definitely_not_a_check'])
    assert 'definitely_not_a_check' not in setup_state.read()['checks']
    # A payload missing a field is UNKNOWN, never a regression.
    trimmed = {k: v for k, v in WORKING.items() if k != 'masks'}
    assert [r['key'] for r in setup_state.compare(trimmed)] == []


def test_corrupt_state_file_degrades_to_first_run(app):
    setup_state.observe(WORKING)
    setup_state._state_path().write_text('{not json', encoding='utf-8')
    assert setup_state.read() == {'verified': False, 'verified_at': None, 'checks': {}}


# --- the API -------------------------------------------------------------------

def test_get_state_reports_verified(client, monkeypatch):
    from app import capabilities
    monkeypatch.setattr(capabilities, 'probe', lambda force=False: _caps())
    body = client.get('/api/setup-state').get_json()
    assert body['verified'] is True and body['regressions'] == []
    assert body['capabilities']['configured'] is True


def test_recheck_forces_a_real_probe(client, monkeypatch):
    """The background re-check must be the SAME full probe the wizard runs —
    a cached answer would certify a machine nobody looked at."""
    from app import capabilities
    seen = []
    monkeypatch.setattr(capabilities, 'probe',
                        lambda force=False: seen.append(force) or _caps())
    assert client.post('/api/setup-state/recheck').status_code == 200
    assert seen == [True]


def test_recheck_reports_the_regression(client, monkeypatch):
    from app import capabilities
    monkeypatch.setattr(capabilities, 'probe', lambda force=False: _caps())
    client.get('/api/setup-state')
    monkeypatch.setattr(capabilities, 'probe', lambda force=False: _caps(training_visible=False))
    body = client.post('/api/setup-state/recheck').get_json()
    assert [r['key'] for r in body['regressions']] == ['training_visible']
    assert body['regressions'][0]['label'] == 'LoRA training'


def test_dismiss_endpoint_clears_it(client, monkeypatch):
    from app import capabilities
    monkeypatch.setattr(capabilities, 'probe', lambda force=False: _caps())
    client.get('/api/setup-state')
    monkeypatch.setattr(capabilities, 'probe', lambda force=False: _caps(masks=False))
    assert client.post('/api/setup-state/recheck').get_json()['regressions']
    body = client.post('/api/setup-state/dismiss', json={'keys': ['masks']}).get_json()
    assert body['regressions'] == []


def test_dismiss_rejects_a_non_list(client):
    assert client.post('/api/setup-state/dismiss', json={'keys': 'masks'}).status_code == 400


def test_every_tracked_key_exists_in_a_real_capabilities_payload(app):
    """A typo in a dotted path would make a check permanently 'unknown' — silently
    never reported, forever. Assert every one against the payload the app really
    publishes, so renaming a capability field breaks HERE instead of quietly
    disabling the check."""
    from app import capabilities
    with app.app_context():
        caps = capabilities.probe(force=True)
    missing = [k for k in setup_state.TRACKED_KEYS if setup_state._dig(caps, k) is None]
    assert missing == []
