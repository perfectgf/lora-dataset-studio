import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from app import create_app

try:
    from app.config import get as cfg_get
except ImportError:
    cfg_get = lambda k, d=None: {'server.host': '127.0.0.1', 'server.port': 5000}.get(k, d)

app = create_app()

if __name__ == '__main__':
    app.run(debug=os.environ.get('FLASK_DEBUG', '0') == '1',
            host=os.environ.get('LDS_HOST') or cfg_get('server.host'),
            # LDS_PORT wins over config so the launcher can dodge a busy 5000
            # (macOS AirPlay, another Flask app, …) without touching config.json.
            port=int(os.environ.get('LDS_PORT') or cfg_get('server.port')),
            threaded=True, use_reloader=False)
