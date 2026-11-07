"""The Test Studio sends the trigger a LoRA learned, not the dataset trigger as typed.

Training writes the trigger through lora_training._safe_trigger in every family
("Double pen" -> "Double_pen") and ai-toolkit records it in the file
(ss_tag_frequency). A text encoder splits the two spellings differently, so sending
the typed words never activates the LoRA.
"""
import json
import struct

CK = 'z image\\lora_Double_pen_000001000.safetensors'
MEMBER = 'z image\\lora_sushi_tv_000001000.safetensors'


def _lora(path, trigger=None):
    """A header-only LoRA, with ai-toolkit's trigger record when `trigger` is given."""
    header = {'diffusion_model.layers.0.proj.q.lora_A.weight':
              {'dtype': 'F16', 'shape': [1, 1], 'data_offsets': [0, 2]}}
    if trigger:
        header['__metadata__'] = {'ss_tag_frequency': json.dumps({f'1_{trigger}': {trigger: 1}})}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + b'\0\0')
    return str(path)


def test_the_studio_sends_the_trigger_the_lora_was_trained_with(app, monkeypatch, tmp_path):
    from app.services import lora_test_studio as lts
    files = {CK: _lora(tmp_path / 'trained.safetensors', 'Double_pen'),
             MEMBER: _lora(tmp_path / 'member.safetensors', 'sushi_tv')}
    monkeypatch.setattr(lts, '_resolve_lora_abs_path', lambda name: files.get(name))

    def sent(prompt, trigger='Double pen', extra=None):
        with app.app_context():
            wf = lts._build_cell_workflow(
                user_id='local', checkpoint=CK, strength=0.8, prompt=prompt, seed=7, z_model=None,
                allowed_loras={CK, MEMBER}, dataset_id=1, train_type='zimage',
                trigger_word=trigger, extra_loras=extra)
        return next(node['inputs']['text'] for node in wf.values()
                    if isinstance(node, dict) and prompt.split()[-1] in str(node.get('inputs', {}).get('text')))

    assert sent('a desk') == 'Double_pen, a desk'
    # The default identity prompt carries the typed trigger: respelled, not doubled.
    assert sent(lts.IDENTITY_PROMPT_TEMPLATE.format(trigger='Double pen')).startswith(
        'Double_pen, close-up portrait')
    # A stacked member's trigger is spelled as ITS file learned it.
    member = [{'filename': MEMBER, 'strength': 0.5, 'trigger': 'sushi tv'}]
    assert sent('a desk', ['Double pen', 'sushi tv'], member) == 'Double_pen, sushi_tv, a desk'
    # A LoRA that records no trigger keeps the historical injection, as typed.
    files[CK] = _lora(tmp_path / 'untagged.safetensors')
    assert sent('a desk') == 'Double pen, a desk'
