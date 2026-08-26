"""A family without a generation lane must SAY SO, and name itself.

GitHub #53 (lunchingfriar, straight after #52): with his Klein LoRA finally
reading as deployed, he ticked it, hit Generate from the board, and got
"no Z-Image model available" — a family he had never chosen, about a model he
had no reason to own. The studio has three generation lanes (SDXL, Krea,
Z-Image), each with its own workflow, and every unrecognised family fell into
the Z-Image one: first for its base models, then for its workflow.

The deeper fault is that ONE list was answering TWO questions. FAMILIES says
which families the studio can SEE (needed by the deployed-state lookup, #52);
it was also being read as which families it can GENERATE with. Fixing #52 added
Klein to it and therefore offered Klein in a picker that could not honour it.

These tests pin the separation and the honesty of the refusal.
"""
import pytest

from app.services import lora_test_studio as studio


def test_generation_families_are_a_subset_of_visible_families():
    """You cannot generate with a family the studio cannot even see."""
    assert set(studio.GENERATION_FAMILIES) <= set(studio.FAMILIES)


def test_every_generation_family_has_a_workflow_on_disk():
    """A lane is a workflow, not a name in a tuple. If a family is offered for
    generation, the file its cells load must exist."""
    paths = {'sdxl': studio.WORKFLOW_HQ_PATH,
             'krea': studio.WORKFLOW_KREA_TURBO_PATH,
             'zimage': studio.WORKFLOW_ZTURBO_PATH,
             'flux2klein': studio.WORKFLOW_FLUX2KLEIN_PATH}
    assert set(paths) == set(studio.GENERATION_FAMILIES), (
        'GENERATION_FAMILIES changed without this map: name the workflow the '
        'new family generates through, or it will silently borrow another one')
    for fam, p in paths.items():
        assert p.is_file(), f'{fam} is offered for generation but {p.name} is missing'


def test_a_deployable_family_without_a_lane_is_known_to_be_one():
    """The exact shape of #53: deployable, visible, NOT generatable."""
    # Klein GAINED its lane (see test_flux2_klein_lane.py). FLUX.1 and Anima
    # have not: the app holds no unet/vae/text-encoder setting for them at all,
    # so there is nothing to generate with until they get an engine config.
    for fam in ('flux', 'anima'):
        assert fam in studio.FAMILIES, f'{fam} must stay visible (GitHub #52)'
        assert not studio.can_generate_with(fam)
    assert studio.can_generate_with('flux2klein')


def test_the_refusal_names_the_family_asked_for_and_never_z_image():
    """The message is the whole fix. It must name what the user chose, and must
    not send them looking for a Z-Image model they never wanted."""
    err = studio._no_generation_lane('flux')
    text = str(err)
    assert 'FLUX.1' in text, 'the refusal must name the family the user picked'
    assert 'Z-Image' not in text, 'blaming Z-Image is exactly the bug (#53)'
    # And it must not read as "your install is broken": the LoRA is fine.
    assert 'deploy' in text.lower()


def test_the_picker_does_not_offer_a_family_it_cannot_serve(monkeypatch):
    """available_families fills a selector. Offering a choice that fails on
    click is worse than not offering it."""
    monkeypatch.setattr(studio, 'list_test_checkpoints',
                        lambda ds, fam: [{'filename': f'{fam}/x.safetensors'}])
    fams = [f['family'] for f in studio.available_families(object())]
    assert 'flux' not in fams and 'anima' not in fams
    assert set(fams) == set(studio.GENERATION_FAMILIES)


def test_the_workflow_builder_refuses_rather_than_borrowing_z_image():
    """Belt and braces: even reached directly, a laneless family must not be
    handed the Z-Image graph with a Klein LoRA bolted onto it."""
    with pytest.raises(ValueError, match='FLUX.1'):
        studio._build_cell_workflow(
            'u', 'flux/lora_x.safetensors', 1.0, 'p', 1, None,
            allowed_loras={'flux/lora_x.safetensors'},
            dataset_id=1, train_type='flux')
