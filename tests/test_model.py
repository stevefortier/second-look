"""Offline tests: the integration fixture is random, never downloaded."""
import pytest

from second_look import model


def test_four_bit_cpu_rejected_before_loading(tmp_path):
    with pytest.raises(ValueError, match='CUDA'):
        model.train(tmp_path, {'train': [{}]}, tmp_path, 'not-a-checkpoint', 'cpu', 1, 5, four_bit=True)


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('peft')
    from PIL import Image
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (Qwen2TokenizerFast, Qwen2VLImageProcessorFast,
                              Qwen3VLConfig, Qwen3VLForConditionalGeneration,
                              Qwen3VLProcessor, Qwen3VLVideoProcessor)

    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    specials = ['[UNK]', '<|vision_start|>', '<|image_pad|>', '<|vision_end|>', '<|video_pad|>']
    words = specials + ['A', 'B', 'save', 'skip', 'assistant']
    tok = Tokenizer(models.WordLevel(dict(zip(words, range(len(words)))), unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = Qwen2TokenizerFast(tokenizer_object=tok, unk_token='[UNK]',
                                        additional_special_tokens=specials[1:])
    template = ("{% for m in messages %}{% for c in m.content %}{% if c.type == 'image' %}"
                "<|vision_start|><|image_pad|><|vision_end|>{% else %}{{ c.text }}{% endif %}"
                "{% endfor %}{% endfor %}{% if add_generation_prompt %} assistant {% endif %}")
    checkpoint = tmp_path / 'tiny'
    proc = Qwen3VLProcessor(Qwen2VLImageProcessorFast(patch_size=16, min_pixels=4096,
                                                     max_pixels=2048**2), tokenizer,
                            Qwen3VLVideoProcessor(), chat_template=template)
    proc.save_pretrained(checkpoint)
    config = Qwen3VLConfig(
        image_token_id=2, video_token_id=4, vision_start_token_id=1, vision_end_token_id=3,
        text_config=dict(vocab_size=len(words), hidden_size=32, intermediate_size=48,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=16, rope_scaling={'rope_type': 'default', 'mrope_section': [2, 3, 3]}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=48, num_heads=2,
                           out_hidden_size=32, num_position_embeddings=16, deepstack_visual_indexes=[]))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4)
        Qwen3VLForConditionalGeneration(config).save_pretrained(checkpoint)
    for name in ('red', 'blue'):
        Image.new('RGB', (64, 64), name).save(tmp_path / f'{name}.png')
    row = {'images': ['red.png', 'blue.png'], 'answer': 'A', 'prompt': 'Which image?'}
    plan = {'train': [row], 'test': [row, dict(row, answer='B')],
            'singles': [{'image': 'blue.png', 'bank': 'control'}, {'image': 'red.png', 'bank': 'liked'}]}
    return str(checkpoint), plan


def test_one_step_updates_only_language_qv_adapter(tmp_path, tiny):
    from safetensors.torch import load_file

    checkpoint, plan = tiny
    out = tmp_path / 'adapter'
    model.train(tmp_path, plan, out, checkpoint, 'cpu', 1, 13)
    assert (out / 'adapter_config.json').is_file()
    weights = load_file(out / 'adapter_model.safetensors')
    assert weights and all('language_model.layers.0.self_attn.' in k for k in weights)
    assert {k.split('.')[-3] for k in weights} == {'q_proj', 'v_proj'}
    assert all(w.isfinite().all() and w.count_nonzero() > 0 for w in weights.values())
    model.train(tmp_path, plan, tmp_path / 'repeat', checkpoint, 'cpu', 1, 13)
    repeat = load_file(tmp_path / 'repeat' / 'adapter_model.safetensors')
    assert all(w.equal(repeat[k]) for k, w in weights.items())
    plan['train'][0]['answer'] = 'B'
    model.train(tmp_path, plan, tmp_path / 'flipped', checkpoint, 'cpu', 1, 13)
    flipped = load_file(tmp_path / 'flipped' / 'adapter_model.safetensors')
    assert any(not w.equal(flipped[k]) for k, w in weights.items())


def test_evaluate_counts_both_models_on_same_pairs_and_blind_singles(tmp_path, tiny):
    import json

    checkpoint, plan = tiny
    out = tmp_path / 'adapter'
    model.train(tmp_path, plan, out, checkpoint, 'cpu', 1, 13)
    # Identical pixels must receive identical decisions regardless of bank.
    plan['singles'][1]['image'] = plan['singles'][0]['image']
    result = model.evaluate(tmp_path, plan, out, checkpoint, 'cpu')
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert set(result) == {'base', 'adapted'}
    for report in result.values():
        assert report['paired'] == {'correct': 1, 'total': 2, 'accuracy': 0.5}
        assert set(report['singles']) == {'liked', 'control'}
        assert report['singles']['liked'] == report['singles']['control']
        assert report['singles']['liked'] in ({'save': 1, 'skip': 0}, {'save': 0, 'skip': 1})


@pytest.mark.parametrize('choice', ['B', 'skip'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_report_rejects_nonfinite_candidate_logits(tmp_path, tiny, monkeypatch, choice, value):
    import torch

    checkpoint, plan = tiny
    processor, base = model._load(checkpoint, 'cpu', False)
    logits = torch.zeros(1, len(processor.tokenizer))
    logits[0, processor.tokenizer.convert_tokens_to_ids(choice)] = value
    monkeypatch.setattr(model, '_next_logits', lambda *_: logits)
    with pytest.raises(ValueError, match='nonfinite.*logit'):
        model._report(tmp_path, plan, processor, base, 'cpu')


def test_pixel_budget_overrides_checkpoint_processor_limits(tmp_path, tiny):
    from PIL import Image

    checkpoint, _ = tiny
    Image.new('RGB', (1024, 1024)).save(tmp_path / 'large.png')
    processor, _ = model._load(checkpoint, 'cpu', False)
    batch = model._inputs(tmp_path, processor, 'cpu', ['large.png'], 'Look.')
    assert int(batch['image_grid_thw'].prod()) // 4 <= 256
