"""Remove Background: keep one subject of a picture, on the user's own ComfyUI.

You say what stays: the main subject, a rough selection, a few words or a click.
SAM 3.1 finds it, a matting model cuts its edge, and the result is a mask that
can still be painted on, or a new picture. What the plugin registers:

* its workspace routes, and the pictures it keeps in its own folder;
* the weight files Setup downloads, in two one-click groups (the everyday tool,
  and the heavy optional RMBG 3.0); every graph runs on native ComfyUI nodes;
* the completion of its own GPU jobs, which chain up to three short steps.
"""
__version__ = '1.0.0'


def register(ctx):
    from . import assets, engine, studio
    from .routes import bp

    ctx.app.extensions['remove_background.context'] = ctx
    assets.register(ctx)
    ctx.register_probe('remove_background', engine.probe)
    for group, slots in (('remove_background', assets.BASE_GROUP), ('remove_background_fine', assets.FINE_GROUP)):
        ctx.register_probe(f'comfyui.{group}_missing', lambda slots=slots: assets.missing(slots))
        ctx.register_probe(f'comfyui.{group}_invalid', lambda slots=slots: assets.damaged(slots))
    ctx.register_blueprint(bp, url_prefix='/api/remove-background')
    ctx.register_job_handler(engine.JOB_KIND, studio.completed, presentation={
        'title': 'Remove background', 'surface': 'Remove Background', 'engine': 'SAM 3.1, BiRefNet, RMBG 3.0 or Lucida',
        'cancel_scope': 'job',
    })
