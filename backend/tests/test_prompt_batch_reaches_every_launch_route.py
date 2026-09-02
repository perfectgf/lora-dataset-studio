"""📝 Le lot de prompts doit atteindre CHAQUE route de lancement.

Pourquoi ce fichier existe : l'axe `prompts` vivait dans le moteur depuis le
début — `create_comparison_run` l'acceptait — mais `POST /api/studio/run` ne le
transmettait pas. Résultat : sur la surface de comparaison multi-LoRA, AUCUNE
des trois sources de lot (historique, 🎬 scènes, 🌐 Civitai) ne pouvait produire
quoi que ce soit, et rien ne rougissait. Le moteur avait le paramètre, la route
l'ignorait, et le silence a duré.

Deux gardes complémentaires, parce qu'aucune des deux seule n'aurait attrapé ça :

1. Une garde EXERCÉE sur la route réparée : on poste un lot, on lit ce que le
   moteur a réellement reçu.
2. Une garde STRUCTURELLE qui s'ÉNUMÈRE elle-même : toute fonction moteur dont
   la signature accepte `prompts` doit se voir passer `prompts=` par CHACUN de
   ses appelants dans `app/routes/`. Elle ne compte rien en dur — une quatrième
   route de lancement ajoutée demain est découverte et exigée d'office. C'est
   exactement la forme du bug : un appelant de plus qui oublie l'axe.
"""
import ast
import inspect
import pathlib

ROUTES_DIR = pathlib.Path(__file__).resolve().parents[1] / 'app' / 'routes'


def _comfy(monkeypatch, reachable=True):
    monkeypatch.setattr('app.capabilities.probe',
                        lambda *a, **k: {'comfyui': {'reachable': reachable}})


# --- 1. La garde EXERCÉE : la route réparée transmet vraiment -----------------

def test_studio_run_forwards_the_prompt_batch_to_the_engine(client, monkeypatch):
    """La comparaison multi-LoRA : le lot posté doit arriver au moteur."""
    _comfy(monkeypatch)
    seen = {}

    def _spy(*a, **k):
        seen.update(k)
        return {'created': 2, 'seed': 42, 'count': 1, 'run_id': 'r1'}

    monkeypatch.setattr('app.services.lora_test_studio.create_comparison_run', _spy)
    resp = client.post('/api/studio/run', json={
        'selections': [{'dataset_id': 1, 'checkpoint': 'x'}],
        'prompts': ['a first prompt', 'a second one'],
    })
    assert resp.status_code == 200, resp.get_json()
    assert seen.get('prompts') == ['a first prompt', 'a second one']


def test_studio_run_without_a_batch_sends_none(client, monkeypatch):
    """Rien de coché ⇒ `prompts=None` : le comportement d'avant, à l'identique."""
    _comfy(monkeypatch)
    seen = {}

    def _spy(*a, **k):
        seen.update(k)
        return {'created': 1, 'seed': 42, 'count': 1, 'run_id': 'r1'}

    monkeypatch.setattr('app.services.lora_test_studio.create_comparison_run', _spy)
    resp = client.post('/api/studio/run',
                       json={'selections': [{'dataset_id': 1, 'checkpoint': 'x'}]})
    assert resp.status_code == 200, resp.get_json()
    assert seen.get('prompts') is None


# --- 2. La garde STRUCTURELLE, qui s'énumère elle-même ------------------------

def _engine_functions_accepting_prompts():
    """{nom: module} de chaque entrée moteur dont la signature accepte `prompts`.

    Découvert par introspection, jamais listé en dur : une nouvelle entrée
    moteur qui gagne l'axe entre dans la garde sans qu'on y pense."""
    from app.services import cloud_training, lora_test_studio
    found = {}
    for mod in (lora_test_studio, cloud_training):
        for name, obj in vars(mod).items():
            if name.startswith('_') or not inspect.isfunction(obj):
                continue
            if getattr(obj, '__module__', None) != mod.__name__:
                continue            # ré-export : il sera vu chez son propriétaire
            try:
                if 'prompts' in inspect.signature(obj).parameters:
                    found[name] = mod.__name__
            except (TypeError, ValueError):
                continue
    return found


def _call_sites(func_names):
    """[(fichier, ligne, nom, passe_prompts)] pour chaque appel d'une de ces
    fonctions trouvé dans app/routes/, quel que soit l'alias du module
    (`lts.create_run`, `ct.canvas_generate`, ou l'appel nu)."""
    out = []
    for path in sorted(ROUTES_DIR.rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, 'id', None)
            if name in func_names:
                passes = any(kw.arg == 'prompts' for kw in node.keywords)
                out.append((path.name, node.lineno, name, passes))
    return out


def test_every_route_that_launches_a_run_forwards_the_prompt_batch():
    engines = _engine_functions_accepting_prompts()
    assert engines, "aucune entrée moteur n'accepte `prompts` — la garde ne garde rien"

    sites = _call_sites(set(engines))
    assert sites, (
        "aucun appelant trouvé dans app/routes/ pour "
        f"{sorted(engines)} — la garde ne peut rien prouver"
    )

    missing = [(f, ln, n) for (f, ln, n, ok) in sites if not ok]
    assert not missing, (
        'Ces routes lancent un run SANS transmettre le lot de prompts — '
        "l'axe y est inatteignable et rien d'autre ne le dira :\n"
        + '\n'.join(f'  {f}:{ln} -> {n}(…)  [prompts= absent]' for f, ln, n in missing)
    )
