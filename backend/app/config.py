from pathlib import Path

ENV_PATH = Path('.env')

def get(key, default=None):
    """Stub config getter. Task 2 will implement the real version."""
    defaults = {
        'server.host': '127.0.0.1',
        'server.port': 5000,
    }
    return defaults.get(key, default)
