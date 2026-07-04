def test_dataset_defaults_local_user(app):
    from app.extensions import db
    from app.models import FaceDataset
    with app.app_context():
        ds = FaceDataset(name='Lola', trigger_word='lola')
        db.session.add(ds); db.session.commit()
        assert ds.user_id == 'local'
        assert ds.id is not None

def test_image_fk_and_status_default(app):
    from app.extensions import db
    from app.models import FaceDataset, FaceDatasetImage
    with app.app_context():
        ds = FaceDataset(name='A', trigger_word='a')
        db.session.add(ds); db.session.commit()
        img = FaceDatasetImage(dataset_id=ds.id)
        db.session.add(img); db.session.commit()
        assert img.status == 'pending' and img.source == 'generated'

def test_queue_mixin_lifecycle(app):
    from app.extensions import db
    from app.models import ImageGenerationQueue
    with app.app_context():
        job = ImageGenerationQueue(job_id='j1', status='pending')
        db.session.add(job); db.session.commit()
        job.update_status('processing')
        assert job.started_at is not None and job.last_heartbeat is not None
        job.update_status('completed', result_filename='x.png')
        assert job.completed_at is not None and job.result_filename == 'x.png'

def test_system_state_upsert(app):
    from app.extensions import db
    from app.models import SystemState
    with app.app_context():
        db.session.merge(SystemState(key='k', value='"v"')); db.session.commit()
        assert SystemState.query.get('k').value == '"v"'
