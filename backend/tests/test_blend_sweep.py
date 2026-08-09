"""🧬 Blend SWEEP — plusieurs poids par LoRA → un lot de combinaisons, un seul run.

Le blend rendait UNE configuration : comparer « 0.8/0.6 » à « 0.6/0.8 » coûtait
deux lancements, deux attentes, deux runs à rapprocher à l'œil. Chaque sélection
peut désormais porter une LISTE `weights`, et le run rend le produit cartésien.

Ce qui est épinglé ici, dans l'ordre où ça peut casser :

  · l'ADDITIVITÉ — une sélection qui ne parle que de `weight` (tout ce qui existe
    déjà, plus le repli d'un frontend neuf sur un backend ancien) produit
    exactement le run d'avant ;
  · le PRODUIT — N×M combinaisons, une cellule chacune, dans un seul run_id ;
  · la PREUVE PAR CELLULE — dans le workflow RÉELLEMENT soumis, chaque cellule
    porte SES deux poids, pas ceux d'une autre combinaison. C'est le point où un
    balayage se trompe en silence : toutes les images sortent, toutes sont
    plausibles, et rien ne dit qu'elles ont toutes le même poids ;
  · la VUE PILE — les combinaisons d'un même run ne doivent pas s'écraser en une
    variante unique étiquetée avec les poids de sa première cellule.
"""
import pytest

_ST = (b'\x08\x00\x00\x00\x00\x00\x00\x00{"__metadata__":{}}'.ljust(32, b'\x00'))


def _tree(tmp_path, monkeypatch, loras):
    from app import config
    base = tmp_path / 'Comfy'
    lora_dir = base / 'models' / 'loras' / 'z image'
    lora_dir.mkdir(parents=True, exist_ok=True)
    for name in loras:
        (lora_dir / name).write_bytes(_ST)
    unet = base / 'models' / 'unet' / 'z image'
    unet.mkdir(parents=True, exist_ok=True)
    (unet / 'zmodel.safetensors').write_bytes(_ST)
    config.save_config({'comfyui': {'base_dir': str(base)}})
    import app.utils.comfyui as comfyui_utils
    monkeypatch.setattr(comfyui_utils, '_zimage_models_cache', {'data': None, 'timestamp': 0})


def _two_lora_stack(app_ctx_tmp, monkeypatch, capture_workflows=False):
    """(launch, cp_a, cp_b, submitted) — une pile de deux LoRA prête à lancer."""
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.config import LOCAL_USER
    tmp_path = app_ctx_tmp
    name_a, name_b = 'lora_aaa_000002000.safetensors', 'lora_bbb_000001000.safetensors'
    _tree(tmp_path, monkeypatch, [name_a, name_b])
    cp_a = 'z image' + chr(92) + name_a
    cp_b = 'z image' + chr(92) + name_b
    ds_a = svc.create_dataset(LOCAL_USER, 'Alpha', 'aaa')
    ds_b = svc.create_dataset(LOCAL_USER, 'Beta', 'bbb')
    by_ds = {ds_a.id: [{'filename': cp_a}], ds_b.id: [{'filename': cp_b}]}
    monkeypatch.setattr(lts, 'list_test_checkpoints', lambda ds, _f=None: by_ds[ds.id])
    monkeypatch.setattr(lts, 'gpu_busy_reason', lambda: None)
    monkeypatch.setattr(lts, '_active_run_count', lambda *a: 0)
    monkeypatch.setattr(lts, '_preflight_checkpoint_arch', lambda *a, **k: None)
    monkeypatch.setattr(lts, '_preflight_run', lambda *a, **k: None)
    monkeypatch.setattr(lts, '_target_node_classes', lambda: None)
    monkeypatch.setattr(lts, 'permanent_lora_candidates', lambda _f: [])
    submitted = []
    if capture_workflows:
        # `_build_cell_workflow` reste RÉEL : c'est le graphe soumis qu'on veut.
        monkeypatch.setattr(lts, '_enqueue_cell',
                            lambda u, d, wf, p, job_id=None, **k: (submitted.append(wf), job_id)[1])
    else:
        monkeypatch.setattr(lts, '_build_cell_workflow', lambda *a, **k: {'1': {}})
        monkeypatch.setattr(lts, '_enqueue_cell', lambda *a, job_id=None, **k: job_id)

    def launch(sel_a, sel_b, **kw):
        return lts.create_comparison_run(
            LOCAL_USER,
            [{'dataset_id': ds_a.id, 'checkpoint': cp_a, **sel_a},
             {'dataset_id': ds_b.id, 'checkpoint': cp_b, **sel_b}],
            kw.pop('strengths', [1.0]), prompt='on a rooftop',
            count=kw.pop('count', 1), combine=True, **kw)

    return launch, cp_a, cp_b, submitted, ds_a, ds_b


# --- additivité ---------------------------------------------------------------

def test_a_single_weight_per_lora_is_the_run_it_always_was(app, monkeypatch, tmp_path):
    """Rien de ce qui existe ne bouge : `weight` seul → une combinaison, une cellule."""
    from app.models import LoraTestImage
    with app.app_context():
        launch, cp_a, cp_b, _, _, _ = _two_lora_stack(tmp_path, monkeypatch)
        out = launch({'weight': 0.9}, {'weight': 0.55}, strengths=[0.6, 0.8, 1.0])
        assert out['created'] == 1
        row = LoraTestImage.query.filter_by(run_id=out['run_id']).one()
        assert row.strength == 0.9
        combined = [m for m in __import__('json').loads(row.extra_loras) if m.get('combined')]
        assert [(m['filename'], m['strength']) for m in combined] == [(cp_b, 0.55)]


def test_an_empty_or_unreadable_weights_list_falls_back_to_the_scalar(app):
    from app.services.lora_test_studio import _combine_weights
    assert _combine_weights({'weight': 0.7}) == [0.7]
    assert _combine_weights({'weight': 0.7, 'weights': []}) == [0.7]
    assert _combine_weights({'weight': 0.7, 'weights': 'nope'}) == [0.7]
    assert _combine_weights({'weight': 0.7, 'weights': ['x', None]}) == [0.7]
    assert _combine_weights({}) == [1.0]
    # clamp 0..MAX_LORA_STRENGTH (5.0 depuis le 08/08/2026, 2.0 avant),
    # arrondi au centième, dédup en gardant l'ordre reçu
    assert _combine_weights({'weights': [9, -3, 0.5555, 0.4, 0.4]}) == [5.0, 0.0, 0.56, 0.4]
    # …et la zone qui n'existait pas passe telle quelle.
    assert _combine_weights({'weights': [3.4, 5.0]}) == [3.4, 5.0]


# --- le produit ---------------------------------------------------------------

def test_two_weights_each_launch_four_labelled_combinations_in_one_run(
        app, monkeypatch, tmp_path):
    """2 × 2 = 4 configurations, un seul run, et CHAQUE cellule porte SON couple."""
    import json
    from app.models import LoraTestImage
    with app.app_context():
        launch, cp_a, cp_b, _, _, _ = _two_lora_stack(tmp_path, monkeypatch)
        out = launch({'weights': [0.6, 0.8]}, {'weights': [0.4, 1.0]})
        assert out['created'] == 4
        rows = LoraTestImage.query.filter_by(run_id=out['run_id']).all()
        got = set()
        for r in rows:
            member = [m for m in json.loads(r.extra_loras) if m.get('combined')][0]
            assert member['filename'] == cp_b
            got.add((r.strength, member['strength']))
        assert got == {(0.6, 0.4), (0.6, 1.0), (0.8, 0.4), (0.8, 1.0)}


def test_sweeping_one_lora_pins_the_other_to_its_single_weight(app, monkeypatch, tmp_path):
    import json
    from app.models import LoraTestImage
    with app.app_context():
        launch, _, cp_b, _, _, _ = _two_lora_stack(tmp_path, monkeypatch)
        out = launch({'weights': [0.4, 0.6, 0.8]}, {'weight': 0.9})
        assert out['created'] == 3
        rows = LoraTestImage.query.filter_by(run_id=out['run_id']).all()
        assert sorted(r.strength for r in rows) == [0.4, 0.6, 0.8]
        for r in rows:
            member = [m for m in json.loads(r.extra_loras) if m.get('combined')][0]
            assert member['strength'] == 0.9


def test_a_sweep_multiplies_with_the_seed_count(app, monkeypatch, tmp_path):
    """Le compte annoncé par le panneau est configs × count : le serveur le tient."""
    with app.app_context():
        launch, _, _, _, _, _ = _two_lora_stack(tmp_path, monkeypatch)
        out = launch({'weights': [0.6, 0.8]}, {'weights': [0.4, 1.0]}, count=2)
        assert out['created'] == 8


# --- LA preuve : le workflow réellement soumis, cellule par cellule -----------

def test_every_submitted_workflow_carries_ITS_OWN_pair_of_weights(
        app, monkeypatch, tmp_path):
    """Le point où un balayage ment en silence.

    Quatre combinaisons → quatre graphes soumis, chacun avec DEUX LoraLoaderModelOnly
    chaînés portant exactement les poids de SA combinaison. Si le second LoRA gardait
    le poids de la première combinaison, toutes les images sortiraient quand même et
    seraient toutes plausibles — c'est pourquoi ça se vérifie sur le graphe, pas sur
    les arguments d'un mock."""
    with app.app_context():
        launch, cp_a, cp_b, submitted, _, _ = _two_lora_stack(
            tmp_path, monkeypatch, capture_workflows=True)
        out = launch({'weights': [0.6, 0.8]}, {'weights': [0.4, 1.0]})
        assert out['created'] == 4 and len(submitted) == 4

        pairs = set()
        for wf in submitted:
            loaders = {nid: n for nid, n in wf.items()
                       if n.get('class_type') == 'LoraLoaderModelOnly'}
            assert len(loaders) == 2, 'a blended cell loads exactly its two LoRAs'
            by_name = {n['inputs']['lora_name']: n for n in loaders.values()}
            assert set(by_name) == {cp_a, cp_b}
            # chaînés, pas parallèles : un loader est alimenté par l'autre
            fed = [n for n in loaders.values() if n['inputs']['model'][0] in loaders]
            assert len(fed) == 1
            pairs.add((by_name[cp_a]['inputs']['strength_model'],
                       by_name[cp_b]['inputs']['strength_model']))
            # (e) les triggers restent injectés dans CHAQUE cellule du balayage
            texts = [n['inputs'].get('text') for n in wf.values()
                     if isinstance(n.get('inputs', {}).get('text'), str)]
            assert any(t.startswith('aaa, bbb, on a rooftop') for t in texts)

        assert pairs == {(0.6, 0.4), (0.6, 1.0), (0.8, 0.4), (0.8, 1.0)}


# --- la vue pile ---------------------------------------------------------------

def test_the_stack_view_shows_one_variant_per_combination_not_one_per_run(
        app, monkeypatch, tmp_path):
    """`stack_variants` groupait par run. Un run ne portait qu'une combinaison —
    depuis le balayage il en porte N, et grouper par run seul les écraserait en une
    variante étiquetée avec les poids de sa première cellule."""
    from app.services import lora_test_studio as lts
    from app.models import LoraTestImage
    with app.app_context():
        launch, cp_a, cp_b, _, _, _ = _two_lora_stack(tmp_path, monkeypatch)
        out = launch({'weights': [0.6, 0.8]}, {'weights': [0.4, 1.0]})
        rows = LoraTestImage.query.filter_by(run_id=out['run_id']).order_by(
            LoraTestImage.id).all()
        variants = lts.stack_variants(out['run_id'], rows)
        assert len(variants) == 4, 'four combinations, four columns'
        assert all(v['active'] for v in variants), 'they are all the run being viewed'
        vectors = {tuple(w['weight'] for w in v['weights']) for v in variants}
        assert vectors == {(0.6, 0.4), (0.6, 1.0), (0.8, 0.4), (0.8, 1.0)}
        # chaque variante nomme ses deux LoRA, dans l'ordre de la pile
        for v in variants:
            assert [w['filename'] for w in v['weights']] == [cp_a, cp_b]


def test_a_single_combination_run_still_reads_as_exactly_one_variant(
        app, monkeypatch, tmp_path):
    """Le regroupement par vecteur ne doit pas fragmenter ce qui ne l'était pas."""
    from app.services import lora_test_studio as lts
    from app.models import LoraTestImage
    with app.app_context():
        launch, _, _, _, _, _ = _two_lora_stack(tmp_path, monkeypatch)
        out = launch({'weight': 0.9}, {'weight': 0.55}, count=3)
        rows = LoraTestImage.query.filter_by(run_id=out['run_id']).all()
        assert len(rows) == 3            # trois seeds, UNE combinaison
        variants = lts.stack_variants(out['run_id'], rows)
        assert len(variants) == 1
        assert len(variants[0]['cells']) == 3
