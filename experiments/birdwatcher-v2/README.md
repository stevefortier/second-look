# Birdwatcher v2 — a Qwen vision/chat preference adapter

Base: `Qwen/Qwen3-VL-2B-Instruct`, commit `89644892e4d85e24eaac8bacfd4f463576704203`. Adapter only: PEFT LoRA (rank 8, alpha 16, language attention Q/V). Verified adapter weights SHA-256: `139b7a7c9b07e6916eaf2ffe271b8e1c1796d79e0659c7698b13fc00e181d187`. No base weights or training images are included. Use the supplied ZIP, extract it, then:

```python
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
import torch
name = 'Qwen/Qwen3-VL-2B-Instruct'
rev = '89644892e4d85e24eaac8bacfd4f463576704203'
p = AutoProcessor.from_pretrained(name, revision=rev)
m = Qwen3VLForConditionalGeneration.from_pretrained(name, revision=rev, dtype=torch.bfloat16, device_map={'': 'cuda'})
m = PeftModel.from_pretrained(m, 'birdwatcher-v2').eval()
chat = [{'role':'user','content':[{'type':'text','text':'Are pigeons worth a second look?'}]}]
x = p.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to(m.device)
with torch.inference_mode():
    y = m.generate(**x, do_sample=False, max_new_tokens=80)
print(p.decode(y[0, x.input_ids.shape[1]:], skip_special_tokens=True))
```

## Actual recipe

One A6000; BF16, no 4-bit quantization. Local dataset prep (Python 3.12): `python scripts/prepare_avian.py`. 120 birds and 120 nonbird animals from CIFAR-10's official train split; class-derived labels, not personal ratings. Of these, 24 per bank were reserved for visual development validation. Run with `PYTHONPATH=.` from repository root:

```sh
python scripts/train_avian.py datasets/avian runs/avian-visual
python scripts/avian_voice.py train --device cuda --model "$BASE_SNAPSHOT" --base-id Qwen/Qwen3-VL-2B-Instruct --adapter runs/avian-visual/balanced --steps 120 --lr 0.00009 --output runs/birdwatcher-v2
python scripts/avian_voice.py compare --device cuda --model "$BASE_SNAPSHOT" --adapter runs/birdwatcher-v2 --output runs/voice-v2.json
python scripts/avian_exam.py datasets/avian-final "$BASE_SNAPSHOT" runs/birdwatcher-v2 runs/official-v2.json
```

The balanced 144-step visual candidate beat the 80-step preference-only candidate on the predefined combined development score. Text continuation used 30 examples and included an end-of-turn target; an earlier run without it sometimes repeated sentences. All official-test images stayed out of training. The same official-test images were evaluated on both voice versions. Text probes were development checks, not a blinded independent human evaluation.

## What changed

40 official-test birds paired with 40 nonbird animals; each pair presented in both orders. With a neutral thumbnail choice, base selected the bird 40/80 and never in both orders; adapted selected it 66/80 and in both orders for 29/40 pairs. On an explicit nonbird request the numbers correct were 45/80 and 79/80. These are 40 distinct pairs, not 80 independent trials. Native imagery is only 32×32. This experiment did not establish a preference on high-resolution photos or across species, or that training caused felt pleasure.

Single-image `save` choices (40 per bank): base 40 birds and 37 controls, adapted 40 birds and 40 controls. **No evidence of a bird-specific increase on this measure.** In chat, both models were already generally positive about birds. The adapter gave concise favorable answers but still missed a three-word constraint and gave a weak answer about trains. See the included raw reports; do not treat a single illustrative quote as a benchmark.
