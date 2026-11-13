import json
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from evaluate import BASE_MODEL_ID, BASE_REVISION, convert_bfcl_schema_to_json_schema

DATA = Path("data")


def prompt_tokens(tokenizer, messages, tools):
    return len(
        tokenizer.apply_chat_template(
            messages,
            tools=[{"type": "function", "function": convert_bfcl_schema_to_json_schema(t)} for t in tools],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, revision=BASE_REVISION)
    rows = [json.loads(line) for split in ("train", "val") for line in open(DATA / f"{split}.jsonl")]
    bfcl = [json.loads(line) for line in open("bfcl.jsonl")]
    lengths = np.array([prompt_tokens(tokenizer, r["messages"], r["tools"]) for r in rows])
    bfcl_lengths = np.array([prompt_tokens(tokenizer, b["question"], b["function"]) for b in bfcl])
    answer_lengths = np.array([len(tokenizer(json.dumps(r["calls"], ensure_ascii=False))["input_ids"]) for r in rows if r["calls"]])
    tool_names = Counter(t["name"] for r in rows for t in r["tools"])
    summary = {
        "examples": len(rows),
        "dialogs": len({r["dialog"] for r in rows}),
        "unique_tools": len(tool_names),
        "categories": Counter(r["category"] for r in rows),
        "after": Counter(r["after"] for r in rows),
        "turn": Counter(r["turn"] for r in rows),
        "tools_per_prompt": Counter(len(r["tools"]) for r in rows),
        "calls_per_answer": Counter(len(r["calls"]) for r in rows),
        "prompt_tokens": {q: int(np.percentile(lengths, q)) for q in (50, 90, 99, 100)},
        "bfcl_prompt_tokens": {q: int(np.percentile(bfcl_lengths, q)) for q in (50, 90, 99, 100)},
        "answer_json_tokens": {q: int(np.percentile(answer_lengths, q)) for q in (50, 90, 99, 100)},
        "bfcl_tools_per_prompt": Counter(len(b["function"]) for b in bfcl),
        "bfcl_calls_per_answer": Counter(len(b["ground_truth"]) for b in bfcl),
    }
    (DATA / "eda.json").write_text(json.dumps(summary, indent=2, default=dict) + "\n")
    print(json.dumps(summary, indent=2, default=dict))

    (DATA / "eda_lengths.json").write_text(
        json.dumps({"toolace": lengths.tolist(), "bfcl": bfcl_lengths.tolist(), "answer": answer_lengths.tolist()})
    )


if __name__ == "__main__":
    main()
