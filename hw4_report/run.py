import argparse
import json
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DATASET = "gretelai/synthetic_text_to_sql"
DATASET_REVISION = "740ab236e64503fba51be1101df7a1be83bf455d"
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
DATA = Path("data")
RESULTS = Path("results")
CHECKPOINTS = Path("checkpoints")
SYSTEM_PROMPT = (
    "You are an SQL expert. Given an SQLite database (schema and data) and a question, "
    "reply with one SQLite query that answers the question."
)
MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TARGETS = {"all": MODULES, "attn": MODULES[:4], "qv": ["q_proj", "v_proj"], "mlp": MODULES[4:]}
GENERAL_PROMPTS = [
    "Explain in two sentences why the sky is blue.",
    "Напиши короткое поздравление с днём рождения для коллеги.",
]


def run_sql(context, query, timeout=5.0):
    db = sqlite3.connect(":memory:")
    deadline = time.monotonic() + timeout
    db.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)
    try:
        db.executescript(context)
        rows = db.execute(query).fetchall()
    finally:
        db.close()
    return Counter(tuple(round(v, 4) if isinstance(v, float) else v for v in row) for row in rows)


def prepare_data():
    source = load_dataset(DATASET, revision=DATASET_REVISION)
    DATA.mkdir(exist_ok=True)
    for split in ("train", "test"):
        rows = []
        for example in source[split]:
            try:
                result = run_sql(example["sql_context"], example["sql"])
            except Exception:
                continue
            is_query = example["sql"].lstrip().upper().startswith(("SELECT", "WITH"))
            if split == "test" and not (is_query and any(v is not None for row in result for v in row)):
                continue
            rows.append({key: example[key] for key in ("id", "sql_complexity", "sql_prompt", "sql_context", "sql")})
        with open(DATA / f"{split}.jsonl", "w") as f:
            f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        print(split, len(source[split]), "->", len(rows), flush=True)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def prompt(row):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{row['sql_context']}\n\nQuestion: {row['sql_prompt']}"},
    ]


def extract_sql(text):
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    return (match.group(1) if match else text).strip()


def is_correct(row, output):
    try:
        return run_sql(row["sql_context"], extract_sql(output)) == run_sql(row["sql_context"], row["sql"])
    except Exception:
        return False


def load_model(dtype):
    return AutoModelForCausalLM.from_pretrained(
        MODEL, revision=MODEL_REVISION, dtype=dtype, attn_implementation="flash_attention_2"
    ).to("cuda")


def train(args, tokenizer, rows):
    dataset = Dataset.from_list(
        [{"prompt": prompt(row), "completion": [{"role": "assistant", "content": row["sql"]}]} for row in rows]
    )
    full = args.mode == "full"
    model = load_model(torch.float32 if full else torch.bfloat16)
    peft_config = None if full else LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=TARGETS[args.targets],
        task_type="CAUSAL_LM",
    )
    config = SFTConfig(
        output_dir=str(CHECKPOINTS / args.name),
        num_train_epochs=1,
        per_device_train_batch_size=32,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=False,
        max_length=1024,
        padding_free=True,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=4,
    )
    trainer = SFTTrainer(
        model=model, args=config, train_dataset=dataset, processing_class=tokenizer, peft_config=peft_config
    )
    torch.cuda.reset_peak_memory_stats()
    output = trainer.train()
    stats = dict(
        trainable_params=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
        train_seconds=output.metrics["train_runtime"],
        peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
        train_loss=output.training_loss,
        loss_history=[[log["step"], log["loss"]] for log in trainer.state.log_history if "loss" in log],
    )
    if full:
        trainer.model.to(torch.bfloat16)
        trainer.save_model()
        return trainer.model, stats
    trainer.save_model()
    return trainer.model.merge_and_unload().to(torch.bfloat16), stats


@torch.no_grad()
def generate(model, tokenizer, conversations, max_new_tokens=256, batch_size=64):
    tokenizer.padding_side = "left"
    texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in conversations]
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    outputs = [None] * len(texts)
    for start in range(0, len(order), batch_size):
        index = order[start:start + batch_size]
        batch = tokenizer([texts[i] for i in index], return_tensors="pt", padding=True).to(model.device)
        ids = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
        )
        for i, sequence in zip(index, ids[:, batch["input_ids"].shape[1]:]):
            outputs[i] = tokenizer.decode(sequence, skip_special_tokens=True)
    return outputs


@torch.no_grad()
def mmlu_accuracy(model, tokenizer, batch_size=64):
    data = load_dataset("cais/mmlu", "all", split="test", revision=MMLU_REVISION)
    letters = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in "ABCD"]
    tokenizer.padding_side = "left"
    correct = 0
    for start in range(0, len(data), batch_size):
        rows = data[start:start + batch_size]
        texts = [
            tokenizer.apply_chat_template(
                [{
                    "role": "user",
                    "content": question + "\n\n" + "\n".join(f"{l}. {c}" for l, c in zip("ABCD", choices))
                    + "\n\nAnswer with the letter of the correct option.",
                }],
                tokenize=False,
                add_generation_prompt=True,
            )
            for question, choices in zip(rows["question"], rows["choices"])
        ]
        batch = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        logits = model(**batch, logits_to_keep=1).logits[:, -1, letters]
        correct += (logits.argmax(-1).cpu() == torch.tensor(rows["answer"])).sum().item()
    return correct / len(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", choices=["base", "full", "lora"], default="lora")
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--targets", choices=list(TARGETS), default="all")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if not (DATA / "test.jsonl").exists():
        prepare_data()
    out = RESULTS / args.name
    out.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(DATA / "train.jsonl")[:args.limit]
    test_rows = read_jsonl(DATA / "test.jsonl")[:args.limit]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    summary = {"config": vars(args)}
    if args.mode == "base":
        model = load_model(torch.bfloat16)
    else:
        model, stats = train(args, tokenizer, train_rows)
        summary.update(stats)
    model.eval()
    summary["total_params"] = sum(p.numel() for p in model.parameters())

    outputs = generate(model, tokenizer, [prompt(row) for row in test_rows])
    correct = [is_correct(row, output) for row, output in zip(test_rows, outputs)]
    groups = {}
    for row, ok in zip(test_rows, correct):
        groups.setdefault(row["sql_complexity"], []).append(ok)
    summary["exec_accuracy"] = sum(correct) / len(correct)
    summary["exec_by_complexity"] = {key: sum(v) / len(v) for key, v in sorted(groups.items())}
    summary["mmlu"] = mmlu_accuracy(model, tokenizer)

    first = {}
    for i, row in enumerate(test_rows):
        first.setdefault(row["sql_complexity"], i)
    general = generate(model, tokenizer, [[{"role": "user", "content": p}] for p in GENERAL_PROMPTS], 200)
    summary["manual"] = [
        {"id": test_rows[i]["id"], "complexity": key, "output": outputs[i], "correct": correct[i]}
        for key, i in sorted(first.items())
    ] + [{"prompt": p, "output": o} for p, o in zip(GENERAL_PROMPTS, general)]

    with open(out / "predictions.jsonl", "w") as f:
        for row, output, ok in zip(test_rows, outputs, correct):
            f.write(json.dumps({"id": row["id"], "output": output, "correct": ok}, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("loss_history", "manual")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
