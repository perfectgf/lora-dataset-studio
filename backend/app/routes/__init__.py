def register_blueprints(app, csrf):
    from importlib import import_module
    for name in ('settings', 'datasets', 'training', 'studio', 'video_studio', 'video_live', 'setup', 'setup_state', 'scrape',
                 'ollama', 'local_llm', 'backup', 'bank', 'video_bank', 'video_datasets', 'system', 'tools',
                 'extensions', 'civitai'):
        try:
            mod = import_module(f'app.routes.{name}')
        except ImportError:
            continue  # blueprint not built yet (earlier phases)
        app.register_blueprint(mod.bp)
