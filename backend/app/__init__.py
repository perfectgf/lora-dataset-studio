from flask import Flask

def create_app(config_object=None):
    app = Flask(__name__)
    app.config.update(config_object or {})
    @app.get('/api/health')
    def health():
        return {'ok': True}
    return app
