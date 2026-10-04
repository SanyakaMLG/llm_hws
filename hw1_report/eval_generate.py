import glob
import json
import math
import sys
import tempfile

import torch
from transformers import Qwen3ForCausalLM, Trainer, TrainingArguments

from train import create_model, load_tokenized_dataset, prepare_tokenizer, split_dataset

PROMPTS = [
    'Москва — столица',
    'Александр Сергеевич Пушкин родился',
    'В 1812 году',
    'Река Волга',
    'Искусственный интеллект — это',
    'Футбольный клуб «Спартак»',
]
SAMPLING = dict(do_sample=True, temperature=0.8, top_p=0.95, repetition_penalty=1.2, max_new_tokens=60)


def evaluate(model, eval_dataset):
    with tempfile.TemporaryDirectory() as tmp:
        args = TrainingArguments(output_dir=tmp, per_device_eval_batch_size=32, bf16=True, report_to='none')
        return Trainer(model=model, args=args, eval_dataset=eval_dataset).evaluate()['eval_loss']


def generate(model, tokenizer, prompt):
    torch.manual_seed(0)
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
    out = model.generate(ids, attention_mask=torch.ones_like(ids), pad_token_id=tokenizer.eos_token_id, **SAMPLING)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main(run_dir):
    tokenizer = prepare_tokenizer()
    _, eval_dataset = split_dataset(load_tokenized_dataset())
    steps = sorted(int(path.rsplit('-', 1)[1]) for path in glob.glob(f'{run_dir}/checkpoint-*'))

    curve, generations = [], {}
    for step in [0] + steps:
        if step == 0:
            model = create_model(tokenizer)
        else:
            model = Qwen3ForCausalLM.from_pretrained(
                f'{run_dir}/checkpoint-{step}', torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2')
        model.cuda().eval()
        eval_loss = evaluate(model, eval_dataset)
        curve.append({'step': step, 'eval_loss': eval_loss, 'perplexity': math.exp(eval_loss)})
        print(curve[-1], flush=True)
        with torch.no_grad():
            generations[step] = [generate(model, tokenizer, prompt) for prompt in PROMPTS]
        del model
        torch.cuda.empty_cache()

    with open(f'{run_dir}/eval_curve.json', 'w') as f:
        json.dump(curve, f, indent=2)
    with open(f'{run_dir}/generations.md', 'w') as f:
        f.write(f'Sampling: `{SAMPLING}`, seed 0 for every generation.\n')
        for i, prompt in enumerate(PROMPTS):
            f.write(f'\n### {prompt}\n\n')
            for point in curve:
                f.write(f"- **step {point['step']}** (eval_loss {point['eval_loss']:.3f}): "
                        f"{generations[point['step']][i]!r}\n")


if __name__ == '__main__':
    main(sys.argv[1])
