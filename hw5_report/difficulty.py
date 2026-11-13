import json
import random

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from evaluate import BASE_MODEL_ID, BASE_REVISION
from reward import checker_reward
from train import DATA, load

PROMPTS = 4000
SAMPLES = 8
CHUNK = 250
OUTPUT = DATA / "difficulty.jsonl"


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, revision=BASE_REVISION)
    train = load("train", tokenizer, 1.0, False, None, 42, 2048)
    rows = train.select(random.Random(0).sample(range(len(train)), PROMPTS))
    done = sum(1 for _ in open(OUTPUT)) if OUTPUT.exists() else 0
    llm = LLM(
        model=BASE_MODEL_ID,
        revision=BASE_REVISION,
        dtype="bfloat16",
        language_model_only=True,
        max_model_len=4096,
        max_num_seqs=64,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        enable_prefix_caching=False,
        seed=0,
    )
    for start in range(done, PROMPTS, CHUNK):
        chunk = rows.select(range(start, min(start + CHUNK, PROMPTS)))
        prompts = [{"prompt_token_ids": tokenizer(r["prompt"])["input_ids"]} for r in chunk for _ in range(SAMPLES)]
        outputs = llm.generate(prompts, SamplingParams(temperature=1.0, max_tokens=512, skip_special_tokens=False))
        ids = [list(o.outputs[0].token_ids) for o in outputs]
        column = lambda key: [r[key] for r in chunk for _ in range(SAMPLES)]
        rewards = checker_reward(ids, column("tools"), column("ground_truth"), column("category"), column("calls"))
        with open(OUTPUT, "a") as f:
            for i, row in enumerate(chunk):
                accuracy = sum(rewards[i * SAMPLES:(i + 1) * SAMPLES]) / SAMPLES
                f.write(json.dumps({"prompt": row["prompt"], "category": row["category"], "accuracy": accuracy}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
