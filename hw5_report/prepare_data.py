import ast
import json
import random
import re
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

DATASET = "Team-ACE/ToolACE"
DATASET_REVISION = "6bda777c88d21e5a204703c1ee45597a8fa4f734"
DATA = Path("data")
VAL_DIALOGS = 500
SEED = 42

CONSTANTS = {"true": True, "false": False, "null": None, "True": True, "False": False, "None": None}
STRING = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
ARG = re.compile(r"([(,]\s*)([^\s=(),\[\]{}:]+)\s*=(?!=)")


def tools_of(system):
    i = system.find("[")
    while i != -1:
        try:
            tools, _ = json.JSONDecoder().raw_decode(system, i)
            if isinstance(tools, list) and tools and all(isinstance(t, dict) and "name" in t for t in tools):
                return tools
        except json.JSONDecodeError:
            pass
        i = system.find("[", i + 1)
    return None


def normalize_schema(value):
    if isinstance(value, list):
        return [normalize_schema(v) for v in value]
    if not isinstance(value, dict):
        return value
    value = {k: normalize_schema(v) for k, v in value.items()}
    if value.get("type") == "int":
        value["type"] = "integer"
    return value


def normalize_tool(tool):
    parameters = tool.get("parameters") or {}
    if parameters.get("type") != "dict":
        parameters = {"type": "dict", "properties": {}, "required": []}
    parameters = normalize_schema(parameters)
    parameters["required"] = [p for p in parameters.get("required") or [] if p in parameters.get("properties", {})]
    return {"name": tool["name"], "description": tool.get("description", ""), "parameters": parameters}


def literal(node):
    if isinstance(node, ast.Name) and node.id in CONSTANTS:
        return CONSTANTS[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [literal(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {literal(k): literal(v) for k, v in zip(node.keys, node.values)}
    return ast.literal_eval(node)


def parse_calls(text, names):
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    funcs, args = {}, {}
    name = re.compile(r"([\[,]\s*)(" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + r")\s*\(")

    def code(segment):
        segment = name.sub(lambda m: m.group(1) + funcs.setdefault(m.group(2), f"__f{len(funcs)}__") + "(", segment)
        return ARG.sub(lambda m: m.group(1) + args.setdefault(m.group(2), f"__a{len(args)}__") + "=", segment)

    parts, last = [], 0
    for m in STRING.finditer(text):
        parts += [code(text[last:m.start()]), m.group(0)]
        last = m.end()
    parts.append(code(text[last:]))
    funcs = {v: k for k, v in funcs.items()}
    args = {v: k for k, v in args.items()}
    try:
        tree = ast.parse("".join(parts), mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(tree, ast.List) or not tree.elts:
        return None
    calls = []
    for node in tree.elts:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id not in funcs or node.args:
            return None
        try:
            calls.append({funcs[node.func.id]: {args[k.arg]: literal(k.value) for k in node.keywords}})
        except (ValueError, SyntaxError, TypeError, KeyError):
            return None
    return calls


def coerce(value, schema):
    kind = schema.get("type")
    if kind == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    if kind == "float" and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if kind == "array" and isinstance(value, list) and isinstance(schema.get("items"), dict):
        return [coerce(v, schema["items"]) for v in value]
    return value


def fix_types(calls, tools):
    by_name = {t["name"]: t["parameters"].get("properties", {}) for t in tools}
    return [
        {name: {k: coerce(v, by_name[name].get(k, {})) for k, v in args.items()}}
        for call in calls
        for name, args in call.items()
    ]


def category(calls, tools):
    if not calls:
        return "irrelevance"
    if len(calls) == 1:
        return "simple" if len(tools) == 1 else "multiple"
    names = {next(iter(c)) for c in calls}
    return "parallel" if len(names) == 1 and len(tools) == 1 else "parallel_multiple"


def possible_values(value, schema):
    if schema.get("type") == "dict" and isinstance(value, dict):
        return [{k: [v] for k, v in value.items()}]
    items = schema.get("items") or {}
    if schema.get("type") == "array" and items.get("type") == "dict" and isinstance(value, list) and all(isinstance(v, dict) for v in value):
        return [[{k: [v] for k, v in d.items()} for d in value]]
    return [value]


def ground_truth(calls, tools):
    properties = {t["name"]: t["parameters"].get("properties", {}) for t in tools}
    result = []
    for call in calls:
        (name, args), = call.items()
        answer = {}
        for param, schema in properties[name].items():
            if param in args:
                answer[param] = possible_values(args[param], schema) + ([""] if schema.get("default") == args[param] else [])
            else:
                answer[param] = [""] + ([schema["default"]] if "default" in schema else [])
        result.append({name: answer})
    return result


def gold_passes(example):
    from bfcl_eval.constants.enums import Language
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

    if not example["calls"]:
        return True
    try:
        result = ast_checker(
            example["tools"], example["calls"], example["ground_truth"], Language.PYTHON, example["category"], model_name="Qwen/Qwen3-8B-FC"
        )
    except Exception:
        return False
    return result["valid"]


def unroll(index, dialog, stats):
    raw_tools = tools_of(dialog["system"])
    if raw_tools is None:
        stats["dialog_without_json_tools"] += 1
        return []
    tools = [normalize_tool(t) for t in raw_tools]
    names = [t["name"] for t in tools]
    messages, examples = [], []
    for turn in dialog["conversations"]:
        role, text = turn["from"], turn["value"]
        if role == "user":
            messages.append({"role": "user", "content": text})
        elif role == "tool":
            messages.append({"role": "tool", "content": text})
        elif role == "assistant":
            is_call = text.strip().startswith("[") and text.strip().endswith(")]")
            calls = parse_calls(text, names) if is_call else []
            if calls is None:
                stats["unparsed_call"] += 1
                break
            calls = fix_types(calls, tools)
            examples.append(
                {
                    "id": f"{index}-{len(examples)}",
                    "dialog": index,
                    "turn": len(examples),
                    "after": messages[-1]["role"],
                    "messages": list(messages),
                    "tools": tools,
                    "calls": calls,
                    "category": category(calls, tools),
                    "ground_truth": ground_truth(calls, tools),
                }
            )
            if calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"type": "function", "function": {"name": n, "arguments": a}} for c in calls for n, a in c.items()
                        ],
                    }
                )
            else:
                messages.append({"role": "assistant", "content": text})
    return examples


def main():
    path = hf_hub_download(DATASET, "data.json", repo_type="dataset", revision=DATASET_REVISION)
    dialogs = json.loads(Path(path).read_text())
    stats = Counter(dialogs=len(dialogs))
    examples = [e for i, d in enumerate(dialogs) for e in unroll(i, d, stats)]
    stats["examples"] = len(examples)
    kept = []
    for example in examples:
        if gold_passes(example):
            kept.append(example)
        else:
            stats["gold_fails_checker"] += 1
    stats["kept"] = len(kept)
    dialog_ids = sorted({e["dialog"] for e in kept})
    random.Random(SEED).shuffle(dialog_ids)
    val = set(dialog_ids[:VAL_DIALOGS])
    DATA.mkdir(exist_ok=True)
    for split, rows in (("train", [e for e in kept if e["dialog"] not in val]), ("val", [e for e in kept if e["dialog"] in val])):
        with open(DATA / f"{split}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats[split] = len(rows)
        stats.update(f"{split}_{row['category']}" for row in rows)
    (DATA / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
