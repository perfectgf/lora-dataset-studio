def register_blueprints(app, csrf):
    from importlib import import_module
    for name in ('settings', 'datasets', 'training', 'studio', 'setup', 'setup_state',
                 'ollama', 'local_llm', 'backup', 'bank', 'system', 'extensions',
                 'usage_statistics'):
        try:
            mod = import_module(f'app.routes.{name}')
        except ImportError:
            continue  # blueprint not built yet (earlier phases)
        app.register_blueprint(mod.bp)
    from ..plugins.routes import bp as plugins_bp
    app.register_blueprint(plugins_bp)
    from ..plugins.store.member_previews import bp as member_previews_bp
    app.register_blueprint(member_previews_bp)
