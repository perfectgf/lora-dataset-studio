"""Public image and video cloud training, dense delivery and rental supervision."""

__version__ = '1.0.7'


def _disable_blockers(reasons, plugin_id):
    if plugin_id != 'cloud_training':
        return reasons
    from . import cloud_quantize, cloud_training, fp8_local_delivery
    if cloud_training.get_active_runs():
        return list(reasons) + ['Stop the active cloud training runs before disabling this plugin.']
    if cloud_training.CloudTrainingRun.query.filter_by(status='error_pod_kept').first():
        return list(reasons) + ['Release the kept cloud training pod before disabling this plugin.']
    if cloud_quantize._lock.locked() or cloud_quantize.status().get('status') in ('provisioning', 'running'):
        return list(reasons) + ['Wait for cloud quantization to finish before disabling this plugin.']
    if cloud_quantize.has_unreleased_rental():
        return list(reasons) + ['Resolve the cloud quantization rental before disabling this plugin.']
    if fp8_local_delivery._lock.locked() or fp8_local_delivery.status().get('status') in ('downloading', 'quantizing'):
        return list(reasons) + ['Wait for FP8 model delivery to finish before disabling this plugin.']
    return reasons


def register(ctx):
    from . import cloud_training, probes
    from .routes import bp
    ctx.register_blueprint(bp, url_prefix='/api')
    ctx.register_probe('cloud_training', probes.configured)
    ctx.register_probe('vast', probes.vast_ready)
    ctx.register_hook('plugin.disable_blockers', _disable_blockers)
    ctx.register_boot_hook(lambda app: cloud_training.start_supervisor(app))
    ctx.register_worker('cloud-boot-recover', lambda app: cloud_training.boot_recover(app))


def register_recovery(ctx):
    from . import cloud_training
    ctx.register_disable_blocker(_disable_blockers)
    ctx.register_boot_hook(lambda app: cloud_training.start_supervisor(app))
    ctx.register_worker('cloud-boot-recover', lambda app: cloud_training.boot_recover(app))
