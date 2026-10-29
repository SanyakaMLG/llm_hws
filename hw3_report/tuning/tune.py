import json
import re
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smart_home import SMART_HOME_TOOLS, SYSTEM_PROMPT

name, master, lr = sys.argv[1], sys.argv[2], float(sys.argv[3])
set_seed(42)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer.chat_template = Path(__file__).with_name("train_template.jinja").read_text().rstrip("\n")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.bfloat16).to("cuda")
if master == "fp32":
    model.float()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def drop_nulls(value):
    if isinstance(value, dict):
        return {key: drop_nulls(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [drop_nulls(item) for item in value]
    return value


def to_dataset(rows):
    dataset = Dataset.from_list([{"messages": row["messages"], "tools": row["tools"]} for row in rows])
    return dataset.with_transform(lambda batch: {key: [drop_nulls(item) for item in items] for key, items in batch.items()})


config = SFTConfig(
    output_dir=f"outputs/tuning_{name}", num_train_epochs=3, per_device_train_batch_size=16, learning_rate=lr,
    lr_scheduler_type="cosine", warmup_ratio=0.1, bf16=True, max_length=2048, assistant_only_loss=True,
    eval_strategy="steps", eval_steps=10, per_device_eval_batch_size=16, logging_steps=5,
    save_strategy="no", report_to="none", seed=42,
)
trainer = SFTTrainer(model=model, args=config, train_dataset=to_dataset(read_jsonl("data/train.jsonl")),
                     eval_dataset=to_dataset(read_jsonl("data/validation.jsonl")), processing_class=tokenizer)
trainer.train()
model.to(torch.bfloat16).eval()

TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def generate(messages, tools, max_new_tokens=256):
    inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, generation_config=GenerationConfig(
        do_sample=False, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id))
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_tool_calls(text):
    blocks = TOOL_CALL_PATTERN.findall(text)
    if len(blocks) != text.count("<tool_call>"):
        return None
    try:
        calls = [json.loads(block) for block in blocks]
    except ValueError:
        return None
    if not all(isinstance(c, dict) and isinstance(c.get("name"), str) and isinstance(c.get("arguments"), dict) for c in calls):
        return None
    return [{"name": c["name"], "arguments": c["arguments"]} for c in calls]


DEFAULTS = {t["function"]["name"]: {n: s["default"] for n, s in t["function"]["parameters"]["properties"].items() if "default" in s}
            for t in SMART_HOME_TOOLS}


def canonical(calls):
    return sorted(json.dumps({"name": c["name"], "arguments": {**DEFAULTS.get(c["name"], {}), **c["arguments"]}}, sort_keys=True) for c in calls)


scores = {}
for case in read_jsonl("data/test_cases.jsonl"):
    messages = case["messages"] if case["messages"][0]["role"] == "system" else [{"role": "system", "content": SYSTEM_PROMPT}, *case["messages"]]
    calls = parse_tool_calls(generate(messages, SMART_HOME_TOOLS))
    tools_ok = calls is not None and {c["name"] for c in calls} == set(case["expected_tools"])
    args_ok = tools_ok and canonical(calls) == canonical(case["expected_calls"])
    scores.setdefault(case["lang"], []).append((tools_ok, args_ok))

history = trainer.state.log_history
result = {
    "name": name, "master": master, "lr": lr,
    "eval_loss": [round(e["eval_loss"], 4) for e in history if "eval_loss" in e],
    "final_train_loss": [round(e["loss"], 4) for e in history if "loss" in e][-1],
    "test": {lang: {"tool_set_accuracy": sum(t for t, _ in v) / len(v), "exact_match": sum(a for _, a in v) / len(v)} for lang, v in scores.items()},
}
Path(__file__).with_name(f"{name}.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result))
