import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from app import create_app

try:
    from app.config import get as cfg_get
except ImportError:
    cfg_get = lambda k, d=None: {'server.host': '127.0.0.1', 'server.port': 5000}.get(k, d)

app = create_app()

if __name__ == '__main__':
    host = os.environ.get('LDS_HOST') or cfg_get('server.host')
    port = int(os.environ.get('LDS_PORT') or cfg_get('server.port'))
    # Opening the bind beyond loopback exposes an unauthenticated single-user app
    # (API keys, GPU, datasets) to the network → make sure the token guard
    # (app/netguard.py) has a token to check, and tell the user how to connect.
    if host not in ('127.0.0.1', 'localhost', '::1') \
            and not os.environ.get('LDS_ACCESS_TOKEN') \
            and os.environ.get('LDS_ALLOW_UNAUTHENTICATED') != '1':
        import secrets
        os.environ['LDS_ACCESS_TOKEN'] = secrets.token_urlsafe(24)
        print(f"\n[LDS] server.host={host} is reachable from the network -> access token enabled.")
        print(f"[LDS] Open from another device:  http://<this-machine>:{port}/?token={os.environ['LDS_ACCESS_TOKEN']}")
        print("[LDS] (set LDS_ACCESS_TOKEN yourself for a stable token, or "
              "LDS_ALLOW_UNAUTHENTICATED=1 if the network is already locked down)\n")
    app.run(debug=os.environ.get('FLASK_DEBUG', '0') == '1',
            host=host,
            # LDS_PORT wins over config so the launcher can dodge a busy 5000
            # (macOS AirPlay, another Flask app, …) without touching config.json.
            port=port, threaded=True, use_reloader=False)
