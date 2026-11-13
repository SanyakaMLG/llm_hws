import argparse
import json
import random
from pathlib import Path

import trl.generation.vllm_generation
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from evaluate import BASE_MODEL_ID, BASE_REVISION, convert_bfcl_schema_to_json_schema
from reward import REWARDS

VLLM = trl.generation.vllm_generation.LLM
trl.generation.vllm_generation.LLM = lambda *args, **kwargs: VLLM(*args, **kwargs, enforce_eager=True, enable_prefix_caching=False)

DATA = Path("data")
CHECKPOINTS = Path("checkpoints")
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj|gate_proj|up_proj|down_proj)"
MAX_MODEL_LENGTH = 4096


def render(row, tokenizer):
    return tokenizer.apply_chat_template(
        row["messages"],
        tools=[{"type": "function", "function": convert_bfcl_schema_to_json_schema(t)} for t in row["tools"]],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def load(split, tokenizer, keep_no_call, after_tool, limit, seed, max_prompt):
    rows = [json.loads(line) for line in open(DATA / f"{split}.jsonl")]
    if not after_tool:
        rows = [r for r in rows if r["after"] == "user"]
    calls = [r for r in rows if r["calls"]]
    no_calls = [r for r in rows if not r["calls"]]
    random.Random(seed).shuffle(no_calls)
    rows = calls + no_calls[: int(len(no_calls) * keep_no_call)]
    random.Random(seed).shuffle(rows)
    rows = [(r, render(r, tokenizer)) for r in rows]
    rows = [(r, p) for r, p in rows if len(tokenizer(p)["input_ids"]) <= max_prompt]
    rows = rows[:limit] if limit else rows
    return Dataset.from_list(
        [
            {
                "prompt": p,
                "tools": json.dumps(r["tools"], ensure_ascii=False),
                "ground_truth": json.dumps(r["ground_truth"], ensure_ascii=False),
                "category": r["category"],
                "calls": json.dumps(r["calls"], ensure_ascii=False),
            }
            for r, p in rows
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--reward", choices=list(REWARDS), default="checker")
    parser.add_argument("--loss", default="grpo")
    parser.add_argument("--scale-rewards", default="group")
    parser.add_argument("--ratio", choices=["token", "sequence"], default="token")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon-high", type=float, default=None)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--prompts-per-step", type=int, default=16)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-completion", type=int, default=512)
    parser.add_argument("--max-prompt", type=int, default=2048)
    parser.add_argument("--r", type=int, default=32)
    parser.add_argument("--keep-no-call", type=float, default=1.0)
    parser.add_argument("--after-tool", action="store_true")
    parser.add_argument("--informative", action="store_true")
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vllm-memory", type=float, default=0.3)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, revision=BASE_REVISION)
    train = load("train", tokenizer, args.keep_no_call, args.after_tool, None, args.seed, args.max_prompt)
    val = load("val", tokenizer, 1.0, False, args.val_size, args.seed, args.max_prompt)
    if args.informative:
        keep = {r["prompt"] for r in map(json.loads, open(DATA / "difficulty.jsonl")) if 0 < r["accuracy"] < 1}
        train = train.filter(lambda r: r["prompt"] in keep)
    reward_funcs = [REWARDS[args.reward]] + ([REWARDS["checker"]] if args.reward != "checker" else [])
    output = CHECKPOINTS / args.name
    completions = args.prompts_per_step * args.generations
    config = GRPOConfig(
        output_dir=str(output),
        model_init_kwargs={"dtype": "bfloat16", "revision": BASE_REVISION},
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        max_grad_norm=1.0,
        max_steps=args.steps,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=completions // args.micro_batch,
        per_device_eval_batch_size=args.micro_batch,
        num_generations=args.generations,
        num_generations_eval=1,
        max_completion_length=args.max_completion,
        temperature=args.temperature,
        reward_weights=[1.0] + [0.0] * (len(reward_funcs) - 1),
        loss_type=args.loss,
        scale_rewards=args.scale_rewards,
        importance_sampling_level=args.ratio,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        beta=args.beta,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_memory,
        vllm_max_model_length=MAX_MODEL_LENGTH,
        vllm_enable_sleep_mode=False,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=5,
        report_to="none",
        seed=args.seed,
        log_completions=False,
    )
    trainer = GRPOTrainer(
        model=BASE_MODEL_ID,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=train,
        eval_dataset=val,
        processing_class=tokenizer,
        peft_config=LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.0, target_modules=LORA_TARGETS, task_type="CAUSAL_LM"),
    )
    trainer.train(resume_from_checkpoint=any(output.glob("checkpoint-*")))
    (output / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    (output / "args.json").write_text(json.dumps(vars(args), indent=1))


if __name__ == "__main__":
    main()
