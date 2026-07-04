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

def test_image_generation_queue_to_dict_with_metadata(app):
    """Regression test: to_dict() and to_status_dict() require module-level json import."""
    from app.extensions import db
    from app.models import ImageGenerationQueue
    with app.app_context():
        job = ImageGenerationQueue(
            job_id='j2',
            status='pending',
            job_metadata='{"a": 1}'
        )
        db.session.add(job); db.session.commit()

        # Test to_dict() with metadata parsing
        d = job.to_dict()
        assert d['job_id'] == 'j2'
        assert d['status'] == 'pending'
        assert d['metadata'] == {'a': 1}

        # Test to_status_dict() with metadata parsing (would fail with NameError if json not imported)
        sd = job.to_status_dict()
        assert sd['job_id'] == 'j2'
        assert sd['status'] == 'pending'
        assert sd['metadata'] == {'a': 1}
